"""Central configuration.  [T3]

Every H-Nav module reads its settings from here; nothing is hardcoded at a call
site. Values come from the repo-root ``.env`` (template: ``hnav/deploy/.env.template``)
with the real process environment taking precedence, matching the loader already
used by ``hnav/stage0/m1_geometry_calibration.py`` and ``hnav/deploy/check_env.py``.

Hard rule 4 of the brief: ``HNAV_MODE`` defaults to ``off``. A stray import must
never change a benchmark number, so nothing in this module has side effects
beyond reading files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_LIVE = "live"
MODES = (MODE_OFF, MODE_SHADOW, MODE_LIVE)


def load_env(repo: Path | None = None) -> dict[str, str]:
    """Process environment overlaid on the repo-root ``.env``.

    ``os.environ`` wins: ``setdefault`` only fills keys the process does not
    already define. Inline ``# comments`` are stripped, as in the T1 loader.
    """
    env = dict(os.environ)
    dotenv = (repo or REPO) / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.split("#")[0].strip())
    return env


@dataclass(frozen=True)
class HNavConfig:
    """Frozen snapshot of the environment, resolved once per process."""

    mode: str = MODE_OFF

    # embedder — GPU1. GPU0 hosts the LLM server and is never touched.
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    embed_device: int = 1
    embed_dtype: str = "float32"
    embed_batch: int = 32
    # Tokenizer truncation length. 8192 covers the benchmark's 4096-tiktoken
    # chunks (measured max 4,333 tokens / 17,675 chars) with headroom, under
    # the model's 32768 context and the embed endpoint's 16384 window. It is
    # part of the embedding cache namespace: changing it invalidates cached
    # vectors by design rather than silently reusing them (T12).
    embed_max_length: int = 8192
    # Memory bound for one padded forward (count x longest, in tokens). Purely
    # a memory knob - it never changes a text's vector (T13b).
    embed_max_batch_tokens: int = 8192
    # Served /v1/embeddings endpoint (vLLM :8001). Used by the LIVE/shadow
    # read stack, which runs inside the benchmark process with no GPU of its
    # own — fact vectors come from the shared disk cache first and fall back
    # to this endpoint WITHOUT persisting (see mab_adapter._build_components).
    embed_base_url: str = "http://localhost:8001/v1"

    # answer model — local vLLM, OpenAI-compatible, no API key, temperature 0.
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 128
    llm_api_key: str = "EMPTY"

    # NLI cross-encoder — read-gate stage 2 (T9). ``nli_device`` accepts an
    # int (CUDA ordinal) or a string ("cpu", "cuda:0"); the Stage-1 campaign
    # arms run with CUDA_VISIBLE_DEVICES unset and HNAV_NLI_DEVICE=cpu.
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    nli_device: int | str = 1
    nli_dtype: str = "float32"
    nli_batch: int = 16
    # Premise and hypothesis SHARE this budget. 512 is DeBERTa-v3's own
    # position limit, not headroom — CrossEp chunk pairs (p50 585 tokens)
    # still exceed it, and that residue is reported via the engine's
    # truncation counters rather than configured away (T12 note 5).
    nli_max_length: int = 512

    # Which read-path mechanism a LIVE run arms (T13). "rerank" is the T11
    # chunk-order permutation, kept as evidence and measured HARMFUL
    # (STAGE1_NULL_ANALIZI.md); "suppress" drops the stale fact and
    # "demote_late" moves the newest fact to the end of the page — both act at
    # FACT granularity inside the retrieved chunks (hnav/core/read_policy.py).
    # The default is deliberately the pre-T13 behaviour so that adding the new
    # mechanisms cannot change what an existing live command does; the campaign
    # sets HNAV_READ_MECHANISM explicitly and records it.
    read_mechanism: str = "rerank"

    # paths (both gitignored)
    cache_dir: Path = field(default_factory=lambda: REPO / "hnav/_cache")
    out_dir: Path = field(default_factory=lambda: REPO / "hnav/_out")

    # signal parameters — frozen here so calibration and confirmation share them
    top_m: int = 50          # neighbourhood depth for H_z / H_vn / dispersion
    churn_k: int = 10        # k for churn@k
    whiten_min_fit_n: int = 200   # refuse ABTT below this store size (plan §3)
    whiten_components: int = 3    # ABTT "all but the top" D

    run_id: str = "dev"

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    @property
    def emb_cache_dir(self) -> Path:
        return self.cache_dir / "emb"

    def require_not_live(self) -> None:
        """Stage-0 MEASUREMENT scripts must never run in live mode.

        Post-T8 status (deliberate protocol transition, STAGE1_PLAN.md §2
        Faz B step 2): ``HNAV_MODE=live`` is now a legitimate mode for the
        READ-PATH rerank only — the MAB adapter no longer calls this. Every
        ``hnav/stage0/`` measurement still does, because a Stage-0 number
        taken under intervention is not a Stage-0 number.
        """
        if self.mode == MODE_LIVE:
            raise RuntimeError(
                "HNAV_MODE=live but this code path is a Stage-0 measurement, "
                "which permits off/shadow only. Live is reserved for the "
                "Stage-1 read-path campaign (STAGE1_PLAN.md)."
            )


def _as_path(repo: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else repo / p


def from_env(env: dict[str, str] | None = None, repo: Path | None = None) -> HNavConfig:
    """Build an :class:`HNavConfig`; unknown ``HNAV_MODE`` values are a hard error."""
    repo = repo or REPO
    env = env if env is not None else load_env(repo)

    mode = env.get("HNAV_MODE", MODE_OFF).strip().lower()
    if mode not in MODES:
        raise ValueError(f"HNAV_MODE={mode!r} not in {MODES}")

    nli_dev_raw = env.get("HNAV_NLI_DEVICE", "1").strip()
    nli_device: int | str = int(nli_dev_raw) if nli_dev_raw.lstrip("-").isdigit() \
        else nli_dev_raw

    return HNavConfig(
        mode=mode,
        embed_model=env.get("HNAV_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"),
        embed_device=int(env.get("HNAV_EMBED_DEVICE", "1")),
        embed_dtype=env.get("HNAV_EMBED_DTYPE", "float32"),
        embed_batch=int(env.get("HNAV_EMBED_BATCH", "32")),
        embed_max_length=int(env.get("HNAV_EMBED_MAX_LENGTH", "8192")),
        embed_max_batch_tokens=int(env.get("HNAV_EMBED_MAX_BATCH_TOKENS", "8192")),
        embed_base_url=env.get("QWEN3_EMBED_BASE_URL", "http://localhost:8001/v1"),
        llm_base_url=env.get("HNAV_LLM_BASE_URL", "http://localhost:8000/v1"),
        llm_model=env.get("HNAV_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        llm_temperature=float(env.get("HNAV_LLM_TEMPERATURE", "0")),
        llm_max_tokens=int(env.get("HNAV_LLM_MAX_TOKENS", "128")),
        llm_api_key=env.get("HNAV_LLM_API_KEY", "EMPTY"),
        nli_model=env.get("HNAV_NLI_MODEL", "cross-encoder/nli-deberta-v3-large"),
        nli_device=nli_device,
        nli_dtype=env.get("HNAV_NLI_DTYPE", "float32"),
        nli_batch=int(env.get("HNAV_NLI_BATCH", "16")),
        nli_max_length=int(env.get("HNAV_NLI_MAX_LENGTH", "512")),
        read_mechanism=env.get("HNAV_READ_MECHANISM", "rerank").strip().lower(),
        cache_dir=_as_path(repo, env.get("HNAV_CACHE_DIR", "hnav/_cache")),
        out_dir=_as_path(repo, env.get("HNAV_OUT_DIR", "hnav/_out")),
        top_m=int(env.get("HNAV_TOP_M", "50")),
        churn_k=int(env.get("HNAV_CHURN_K", "10")),
        whiten_min_fit_n=int(env.get("HNAV_WHITEN_MIN_FIT_N", "200")),
        whiten_components=int(env.get("HNAV_WHITEN_COMPONENTS", "3")),
        run_id=env.get("HNAV_RUN_ID", "dev"),
    )


_CACHED: HNavConfig | None = None


def get_config(refresh: bool = False) -> HNavConfig:
    """Process-wide config. ``refresh=True`` re-reads the environment (tests)."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = from_env()
    return _CACHED


def get_mode() -> str:
    """The one function the benchmark hooks call. Cheap, never raises."""
    try:
        return get_config().mode
    except Exception:  # noqa: BLE001 — a misconfigured env must not break the benchmark
        return MODE_OFF
