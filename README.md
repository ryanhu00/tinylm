# TinyLM

A transformer language model built from scratch and trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

This project includes:
- A standalone transformer training/generation pipeline
- A FastAPI backend for serving generation requests
- A React (Vite) frontend chatbot UI

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Train the Transformer (from scratch)](#train-the-transformer-from-scratch)
- [Run Backend + Frontend](#run-backend--frontend)
- [Run Transformer Standalone](#run-transformer-standalone)
- [Model Architecture](#model-architecture)

## Project Structure

Repo structure:

```text
tinylm/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── checkpoints/          # Training output
│   └── best_model.pt
├── transformer/          # Standalone transformer implementation
│   ├── data/
│   ├── generate.py
│   ├── loss.py
│   ├── tokenizer.py
│   ├── train.py
│   └── transformer.py
├── uv.lock
├── backend/
│   ├── main.py
│   └── requirements.txt
└── frontend/             # React (Vite) chatbot UI
    ├── public/
    ├── src/              # React source
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Setup & Running

### Prerequisites

1. Python 3.11 or higher
2. Node.js 18+ and npm
3. (Recommended) `uv` for Python dependency management

Clone the repository:

```bash
git clone <your-repo-url>
cd tinylm
```

Install root Python dependencies (for standalone transformer usage):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

Dependency note:
- `pyproject.toml` contains root package dependencies used by the standalone `transformer` module.
- `backend/requirements.txt` contains backend API runtime dependencies.
- These are intentionally separate so backend/frontend app setup stays lightweight.


## Train the Transformer (from scratch)

The repository does **not** include large training artifacts (`.txt`, `.npy`, `.pt`).  
After cloning, run this pipeline first:

### 1. Download TinyStories raw text

```bash
mkdir -p transformer/data
cd transformer/data

curl -L -o TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o TinyStoriesV2-GPT4-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ../..
```

### 2. Build tokenizer + tokenized datasets

Run from `transformer/`:

```bash
cd transformer

python - <<'PY'
from pathlib import Path
import numpy as np
from tokenizer import train_bpe_tinystories, Tokenizer

data_dir = Path("data")
special_tokens = ["<|endoftext|>"]

train_bpe_tinystories()

tokenizer = Tokenizer.from_files(
    vocab_filepath=str(data_dir / "vocab.json"),
    merges_filepath=str(data_dir / "merges.json"),
    special_tokens=special_tokens,
)

train_text = (data_dir / "TinyStoriesV2-GPT4-train.txt").read_text(encoding="utf-8")
valid_text = (data_dir / "TinyStoriesV2-GPT4-valid.txt").read_text(encoding="utf-8")

train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
valid_ids = np.array(tokenizer.encode(valid_text), dtype=np.uint16)

np.save(data_dir / "tinystories_train_ids.npy", train_ids)
np.save(data_dir / "tinystories_dev_ids.npy", valid_ids)

PY

cd ..
```

### 3. Train

```bash
python -m transformer.train \
  --train_data transformer/data/tinystories_train_ids.npy \
  --val_data transformer/data/tinystories_dev_ids.npy \
  --checkpoint_dir checkpoints
```

This writes the best checkpoint to `checkpoints/best_model.pt` (at repo root).

## Run Backend + Frontend

After training finishes:

### 1. Backend API

Start the FastAPI backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access the API at:

```text
http://127.0.0.1:8000
```

Access the API docs at:

```text
http://127.0.0.1:8000/docs
```

### 2. Frontend

Run the React frontend:

```bash
cd frontend
npm install
npm start
```

Access the chatbot UI at:

```text
http://127.0.0.1:5173
```

## Run Transformer Standalone

If you want to sample from the model without running the backend/frontend stack, run the transformer modules directly from the repo root.

### Generate text from trained checkpoint

```bash
python -m transformer.generate \
  --checkpoint checkpoints/best_model.pt \
  --vocab transformer/data/vocab.json \
  --merges transformer/data/merges.json \
  --prompt "Once upon a time" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9
```

### Model Architecture

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Context length | 256 tokens |
| d_model | 512 |
| Attention heads | 8 |
| FFN size | 2,048 |
| Layers | 4 |
| Parameters | ~22M |

Custom BPE tokenizer trained on TinyStories. Sinusoidal positional encoding. Pre-norm transformer blocks with ReLU FFN.
