"""Per-sample metrics: success_rate (binary) and progress_score (longest passing prefix)."""
from __future__ import annotations

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

TEST_CATEGORY = "multi_turn_ours"


def compute_success_rate(
    test_entry: dict,
    decoded_responses: list[list[list[str]]],
    ground_truth: list[list[str]],
    model_name: str,
) -> tuple[int, dict]:
    """Returns (1 if all turns pass else 0, raw checker result dict)."""
    res = multi_turn_checker(
        decoded_responses, ground_truth, test_entry, TEST_CATEGORY, model_name
    )
    return int(bool(res.get("valid", False))), res


def compute_progress_score(
    test_entry: dict,
    decoded_responses: list[list[list[str]]],
    ground_truth: list[list[str]],
    model_name: str,
) -> tuple[float, int]:
    """Find the largest k for which the official checker passes on prefix [0..k).

    The checker caches stateful backend instances in globals() keyed by
    (model_name, test_entry_id). To get a fresh execution per prefix we override
    the test_entry id with a synthetic suffix, which guarantees each prefix
    replay starts from initial_config.
    """
    n = len(ground_truth)
    if n == 0:
        return 1.0, 0

    largest_k = 0
    for k in range(1, n + 1):
        synthetic_entry = {**test_entry, "id": f"{test_entry['id']}_prefix_{k}"}
        # Pad model output if the sample errored mid-run; the checker treats empty
        # turns as failures when ground truth is non-empty.
        prefix_decoded = list(decoded_responses[:k])
        while len(prefix_decoded) < k:
            prefix_decoded.append([])
        res = multi_turn_checker(
            prefix_decoded,
            ground_truth[:k],
            synthetic_entry,
            TEST_CATEGORY,
            model_name,
        )
        if res.get("valid"):
            largest_k = k
        else:
            break  # prefix-monotone: once a turn fails, no superset can pass.

    return largest_k / n, largest_k
