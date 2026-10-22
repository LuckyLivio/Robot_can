"""Explicit ranking diagnostics; historical metrics remain unchanged."""
import numpy as np


def ranking_metrics(pred, target, groups, k, *, target_margin=.025, tie_atol=1e-8):
    """Measure score ordering with higher scores preferred.

    Only pairs whose target gap exceeds ``target_margin`` are informative.
    Strict reversal and tie rates use all informative pairs as denominator;
    ``pair_error_half_ties`` counts each predicted tie as half an error.
    Argmax uses the first candidate on ties, matching NumPy/PyTorch selection.
    """
    if not isinstance(groups, (int, np.integer)) or groups < 1:
        raise ValueError('groups must be a positive integer')
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    if not np.isfinite(target_margin) or target_margin < 0:
        raise ValueError('target_margin must be finite and nonnegative')
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError('tie_atol must be finite and nonnegative')
    pred = np.asarray(pred, dtype=np.float64).reshape(groups, k)
    target = np.asarray(target, dtype=np.float64).reshape(groups, k)
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('ranking inputs must be finite')
    choice = pred.argmax(-1)
    regret = target.max(-1) - target[np.arange(groups), choice]
    pd = pred[:, :, None] - pred[:, None, :]
    td = target[:, :, None] - target[:, None, :]
    mask = (np.abs(td) > target_margin) & np.triu(np.ones((k, k), bool), 1)[None]
    pair_pred, pair_target = pd[mask], td[mask]
    count = int(mask.sum())
    tied = np.abs(pair_pred) <= tie_atol
    reversed_pairs = (~tied) & (pair_pred * pair_target < 0)
    return {
        'mean_regret': float(regret.mean()),
        'optimal_within_025': float(np.mean(regret <= .025)),
        'strict_reversal_rate': float(reversed_pairs.mean()) if count else None,
        'tie_rate': float(tied.mean()) if count else None,
        'pair_error_half_ties': float((reversed_pairs.sum() + .5 * tied.sum()) / count) if count else None,
        'effective_pairs': count,
        'decided_pairs': int((~tied).sum()),
        'tied_pairs': int(tied.sum()),
        'groups': int(groups),
        'target_margin': float(target_margin),
        'prediction_tie_atol': float(tie_atol),
        # Exact old formula, deliberately named separately from the new rates.
        'legacy_wrong_pair_rate': float(np.mean(pair_pred * pair_target < 0)) if count else None,
    }
