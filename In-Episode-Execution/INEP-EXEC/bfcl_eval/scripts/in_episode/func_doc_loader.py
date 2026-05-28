"""Load multi-turn function documentation for a sample's involved classes."""
from __future__ import annotations

from functools import lru_cache

from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)
from bfcl_eval.utils import load_file


@lru_cache(maxsize=None)
def _load_class_docs(class_name: str) -> tuple[dict, ...]:
    if class_name not in MULTI_TURN_FUNC_DOC_FILE_MAPPING:
        raise KeyError(f"Unknown involved_class: {class_name}")
    path = MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]
    return tuple(load_file(path, use_lock=False))


def load_function_docs(
    involved_classes: list[str],
    excluded_function: list[str] | None = None,
) -> list[dict]:
    """Merge per-class function docs and drop names listed in excluded_function."""
    excluded = set(excluded_function or [])
    docs: list[dict] = []
    for cls in involved_classes:
        for doc in _load_class_docs(cls):
            if doc["name"] in excluded:
                continue
            docs.append(dict(doc))
    return docs
