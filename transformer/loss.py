import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits

    sum_exp = torch.sum(torch.exp(shifted), dim=-1)
    log_denom = torch.log(sum_exp)

    target_logits = torch.gather(
        shifted, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = log_denom - target_logits

    return loss.mean()
    