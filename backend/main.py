import asyncio
import json
import os
import queue
import re
import sys
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Allow importing the transformer package from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformer.generate import load_model_from_checkpoint, sample_next_token
from transformer.tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Config — override via environment variables
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", os.path.join(REPO_ROOT, "checkpoints", "best_model.pt")
)
VOCAB_PATH = os.environ.get(
    "VOCAB_PATH", os.path.join(REPO_ROOT, "transformer", "data", "vocab.json")
)
MERGES_PATH = os.environ.get(
    "MERGES_PATH", os.path.join(REPO_ROOT, "transformer", "data", "merges.json")
)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Load model + tokenizer once at startup
# ---------------------------------------------------------------------------
print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}…")
model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
model.eval()
print("Model loaded.")

tokenizer = Tokenizer.from_files(
    vocab_filepath=VOCAB_PATH,
    merges_filepath=MERGES_PATH,
    special_tokens=["<|endoftext|>"],
)
EOS_ID: int = tokenizer.encode("<|endoftext|>")[0]
print("Tokenizer loaded.")

SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]?\s*$')
MAX_OVERFLOW_TOKENS = 48
MAX_OVERFLOW_RATIO = 0.2

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TinyLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="Once upon a time", min_length=1)
    max_new_tokens: int = Field(default=200, ge=10, le=600)
    temperature: float = Field(default=0.8, ge=0.01, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/generate")
async def generate_stream(req: GenerateRequest):
    prompt_ids = tokenizer.encode(req.prompt)
    if not prompt_ids:
        raise HTTPException(status_code=400, detail="Prompt encodes to zero tokens.")

    token_queue: queue.Queue[str | None] = queue.Queue()

    def _run_inference():
        """Blocking inference loop — runs in a background thread."""
        try:
            generated = list(prompt_ids)
            context_length = model.context_length
            generated_text = ""
            overflow_tokens = min(
                MAX_OVERFLOW_TOKENS,
                max(8, int(req.max_new_tokens * MAX_OVERFLOW_RATIO)),
            )
            hard_limit = req.max_new_tokens + overflow_tokens

            with torch.no_grad():
                for step in range(hard_limit):
                    input_ids = generated[-context_length:]
                    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
                    logits = model(x)
                    next_logits = logits[0, -1, :]

                    next_id = sample_next_token(
                        next_logits,
                        temperature=req.temperature,
                        top_p=req.top_p,
                    )
                    generated.append(next_id)

                    # Decode only the newly added token
                    token_text = tokenizer.decode([next_id])
                    generated_text += token_text
                    token_queue.put(token_text)

                    if next_id == EOS_ID:
                        break

                    hit_target = (step + 1) >= req.max_new_tokens
                    sentence_complete = SENTENCE_END_RE.search(generated_text) is not None
                    if hit_target and sentence_complete:
                        break
        except Exception as exc:
            token_queue.put(f"\n\n[Error during generation: {exc}]")
        finally:
            token_queue.put(None)  # sentinel

    thread = threading.Thread(target=_run_inference, daemon=True)
    thread.start()

    async def _event_stream():
        loop = asyncio.get_event_loop()
        while True:
            # Pull from the queue without blocking the event loop
            token_text = await loop.run_in_executor(None, token_queue.get)
            if token_text is None:
                yield "data: [DONE]\n\n"
                break
            payload = json.dumps({"token": token_text})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
