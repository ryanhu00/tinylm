import argparse
import torch
from transformer.tokenizer import Tokenizer
from transformer.transformer import TransformerLM, softmax


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    if temperature == 0.0:
        return logits.argmax(dim=-1).item()

    probs = softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff_mask = cumulative_probs - sorted_probs >= top_p
        sorted_probs[cutoff_mask] = 0.0

        sorted_probs /= sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        return sorted_indices[idx].item()

    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
):

    model.eval()
    device = next(model.parameters()).device
    context_length = model.context_length
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        input_ids = generated[-context_length:]
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1, :]

        next_token = sample_next_token(next_logits, temperature=temperature, top_p=top_p)
        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> TransformerLM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--vocab", type=str, default="data/vocab.json")
    parser.add_argument("--merges", type=str, default="data/merges.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of independent completions to generate")

    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = load_model_from_checkpoint(args.checkpoint, device)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}")
    print(f"Temperature: {args.temperature}  Top-p: {args.top_p}")

    for i in range(args.num_samples):

        output_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_id,
        )

        text = tokenizer.decode(output_ids)
        print(text)

    print(f"Generated {len(output_ids) - len(prompt_ids)} new tokens")


if __name__ == "__main__":
    main()
