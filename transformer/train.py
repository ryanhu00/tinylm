import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer.loss import cross_entropy
from transformer.transformer import TransformerLM


def run_get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + context_length + 1] for s in starts])

    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_t, targets_t


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_batches=50):
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        inputs, targets = run_get_batch(data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def save_loss_plots(train_losses, val_losses, val_steps, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(1, len(train_losses) + 1)
    ax1.plot(steps, train_losses, alpha=0.3, color="steelblue", label="Per-step")
    window = min(50, len(train_losses) // 2)
    if window > 1:
        smoothed = np.convolve(train_losses, np.ones(window) / window, mode="valid")
        ax1.plot(range(window, len(train_losses) + 1), smoothed, color="darkblue",
                 label=f"Smoothed (w={window})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if val_losses:
        ax2.plot(val_steps, val_losses, "o-", color="darkorange", markersize=3)
        def _safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                return float("inf")
        val_ppl = [_safe_exp(l) for l in val_losses]
        ax2_twin = ax2.twinx()
        ax2_twin.plot(val_steps, val_ppl, "o-", color="green", markersize=3, alpha=0.5)
        ax2_twin.set_ylabel("Perplexity", color="green")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss", color="darkorange")
    ax2.set_title("Validation Loss & Perplexity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss plots saved to {path}")


def train(args):
    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Data

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")
    print(f"  Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_steps: list[int] = []
    start_time = time.time()

    model.train()

    for step in range(1, args.max_steps + 1):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        inputs, targets = run_get_batch(
            train_data, args.batch_size, args.context_length, device,
        )

        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step * args.batch_size * args.context_length) / elapsed
            print(
                f"Step {step:>6d}/{args.max_steps} | "
                f"Train Loss {train_loss:.4f} | "
                f"LR {lr:.2e} | "
                f"Tok/s {tokens_per_sec:.0f} | "
                f"Elapsed {elapsed:.0f}s"
            )

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length,
                device, args.eval_batches,
            )
            try:
                val_ppl = math.exp(val_loss)
            except OverflowError:
                val_ppl = float("inf")
            val_losses.append(val_loss)
            val_steps.append(step)

            elapsed = time.time() - start_time
            improved = val_loss < best_val_loss
            marker = " *" if improved else ""
            print(
                f"Validation Loss {val_loss:.4f} | "
                f"Validation PPL {val_ppl:.2f} | "
                f"Elapsed {elapsed:.0f}s{marker}"
            )

            if math.isinf(val_ppl) or math.isnan(val_loss):
                break
            
            if improved:
                best_val_loss = val_loss
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "config": {
                        "vocab_size": args.vocab_size,
                        "context_length": args.context_length,
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "d_ff": args.d_ff,
                        "num_layers": args.num_layers,
                    },
                }
                ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint, ckpt_path)
                print(f"Best model saved to {ckpt_path}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s")
    best_ppl = math.exp(best_val_loss)
    print(f"Best val loss: {best_val_loss:.4f} | Best val PPL: {best_ppl:.2f}")
    print(f"Best val loss: {best_val_loss:.4f} | Best val PPL: {math.exp(best_val_loss):.2f}")

    save_loss_plots(train_losses, val_losses, val_steps, args.checkpoint_dir)

    log = {
        "args": vars(args),
        "num_params": num_params,
        "total_time_s": total_time,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_steps": val_steps,
    }
    log_path = os.path.join(args.checkpoint_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM on TinyStories")

    # Data
    parser.add_argument("--train_data", type=str, default="data/tinystories_train_ids.npy")
    parser.add_argument("--val_data", type=str, default="data/tinystories_dev_ids.npy")

    # Model Architecture
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=4)

    # Optimizer + Schedule
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging + Checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
