from __future__ import annotations

import torch


def apply_action_mask_to_logits(
    logits: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    invalid_fill_value: float = -1e9,
) -> torch.Tensor:
    """Apply a hard action mask to logits.

    Args:
        logits: (B, A) float tensor.
        action_mask: (B, A) bool/0-1 tensor where 1/True means allowed.
        invalid_fill_value: value to place on invalid actions.

    Returns:
        Masked logits (B, A).

    Notes:
        - We don't use -inf to avoid NaNs in some mixed-precision / softmax paths.
        - If action_mask is None, logits are returned unchanged.
    """
    if action_mask is None:
        return logits

    if action_mask.dtype != torch.bool:
        action_mask = action_mask.to(dtype=torch.bool)

    # Ensure broadcastable + on same device
    action_mask = action_mask.to(device=logits.device)

    # Replace invalid actions with very negative value
    return logits.masked_fill(~action_mask, invalid_fill_value)


def safe_categorical_from_logits(
    logits: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.distributions.Categorical:
    """Create a categorical distribution, guaranteeing at least one valid action.

    If all actions are masked out for a sample (shouldn't happen), we fall back to
    unmasked logits for that row.
    """
    if action_mask is None:
        return torch.distributions.Categorical(logits=logits)

    if action_mask.dtype != torch.bool:
        action_mask = action_mask.to(dtype=torch.bool)

    # Identify bad rows (all invalid)
    valid_any = action_mask.any(dim=-1)
    if valid_any.all():
        masked_logits = apply_action_mask_to_logits(logits, action_mask)
        return torch.distributions.Categorical(logits=masked_logits)

    # Mixed batch: fix rows that have no valid actions by ignoring mask
    masked_logits = apply_action_mask_to_logits(logits, action_mask)
    # For invalid rows, copy original logits back in
    masked_logits = masked_logits.clone()
    masked_logits[~valid_any] = logits[~valid_any]
    return torch.distributions.Categorical(logits=masked_logits)
