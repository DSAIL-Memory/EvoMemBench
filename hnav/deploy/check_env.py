#!/usr/bin/env python3
"""Pre-flight check. Run on the GPU machine BEFORE T1 burns any compute.

    python hnav/deploy/check_env.py

Verifies, in order and without loading the 8GB embedder:
  1. repo layout + benchmark data files present
  2. the committed stdlib measurement scripts still reproduce their numbers
  3. torch sees CUDA, and the chosen embed device has enough free VRAM
  4. the embedding model is cached locally
  4b. the NLI cross-encoder is cached AND labels a 3-pair smoke correctly
      (known contradiction / entailment / neutral, on CPU) — Stage-1 read gate
  5. the LLM endpoint on :8000 answers, and its model id is recorded
  6. (optional) the embedding endpoint on :8001, if you have served it

Exit code 0 = clear to run T1. Non-zero = fix before proceeding.
Nothing here mutates repo state.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OK, WARN, FAIL = "  ok ", " warn", " FAIL"
_failures: list[str] = []
_warnings: list[str] = []


def report(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  —  {detail}" if detail else ""))
    if status == FAIL:
        _failures.append(label)
    elif status == WARN:
        _warnings.append(label)


def load_env() -> dict:
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.split("#")[0].strip())
    return env


# ── 1. data files ────────────────────────────────────────────────────────────
def check_data() -> None:
    required = {
        "Conflict_Resolution": REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json",
        "Accurate_Retrieval": REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Accurate_Retrieval.json",
        "CL-bench": REPO / "Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl",
        "CL-bench baseline": REPO / "Cross-Episode-Knowledge/CROSSEP-KNOW/outputs/context_nomemory/"
                                    "CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory_graded.jsonl",
    }
    for name, path in required.items():
        if path.exists():
            report(OK, f"data: {name}", f"{path.stat().st_size / 1e6:.1f} MB")
        else:
            report(FAIL, f"data: {name}", f"missing at {path.relative_to(REPO)}")


# ── 2. reproduce committed measurements ──────────────────────────────────────
def check_repro() -> None:
    script = REPO / "hnav/labeling/conflict_analysis.py"
    if not script.exists():
        report(FAIL, "repro: conflict_analysis.py", "missing — wrong branch checked out?")
        return
    try:
        out = subprocess.run([sys.executable, str(script)], capture_output=True,
                             text=True, timeout=600, cwd=REPO).stdout
    except Exception as e:  # noqa: BLE001
        report(FAIL, "repro: conflict_analysis.py", str(e)[:120])
        return

    # sh_262k must report 11037 keys / 7197 conflicted
    keys = re.search(r"distinct \(relation,subject\) keys : (\d+)", out.split("factconsolidation_sh_262k")[-1]) \
        if "factconsolidation_sh_262k" in out else None
    conf = re.search(r"CONFLICTED keys \(>1 distinct obj\): (\d+)", out.split("factconsolidation_sh_262k")[-1]) \
        if "factconsolidation_sh_262k" in out else None
    if keys and conf and keys.group(1) == "11037" and conf.group(1) == "7197":
        report(OK, "repro: conflict census", "sh_262k = 11037 keys / 7197 conflicted (65.2%)")
    else:
        got = f"keys={keys.group(1) if keys else '?'} conflicted={conf.group(1) if conf else '?'}"
        report(FAIL, "repro: conflict census", f"expected 11037/7197, got {got}")


# ── 3. torch / CUDA / VRAM ───────────────────────────────────────────────────
def check_gpu(env: dict) -> None:
    try:
        import torch
    except ImportError:
        report(FAIL, "torch", "not installed — run hnav/deploy/setup_remote.sh")
        return
    if not torch.cuda.is_available():
        report(FAIL, "torch.cuda", "no CUDA device visible")
        return

    n = torch.cuda.device_count()
    report(OK, "torch", f"{torch.__version__}, {n} CUDA device(s)")

    dev = int(env.get("HNAV_EMBED_DEVICE", "1"))
    if dev >= n:
        report(FAIL, "embed device", f"HNAV_EMBED_DEVICE={dev} but only {n} device(s) visible")
        return

    free, total = torch.cuda.mem_get_info(dev)
    free_gb, total_gb = free / 1e9, total / 1e9
    name = torch.cuda.get_device_name(dev)
    dtype = env.get("HNAV_EMBED_DTYPE", "float32")
    need = 17.0 if dtype == "float32" else 9.0   # 4B params + activations, with headroom

    if free_gb >= need:
        report(OK, f"GPU{dev} ({name})", f"{free_gb:.1f}/{total_gb:.1f} GB free, need ~{need:.0f} for {dtype}")
    elif dtype == "float32" and free_gb >= 9.0:
        report(WARN, f"GPU{dev} ({name})",
               f"{free_gb:.1f} GB free — not enough for float32. Set HNAV_EMBED_DTYPE=float16 "
               f"and keep it pinned for ALL later tasks.")
    else:
        report(FAIL, f"GPU{dev} ({name})", f"only {free_gb:.1f} GB free, need ~{need:.0f}")

    for d in range(n):
        if d != dev:
            f, t = torch.cuda.mem_get_info(d)
            report(OK, f"GPU{d} (other)", f"{f/1e9:.1f}/{t/1e9:.1f} GB free — left alone")


# ── 4. embedding weights cached ──────────────────────────────────────────────
def check_embed_weights(env: dict) -> None:
    model = env.get("HNAV_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
    try:
        from huggingface_hub import snapshot_download
        # Probe with allow_patterns: setup downloads a FILTERED snapshot
        # (*.json / *.safetensors / tokenizer*), so demanding every repo file
        # (README, .gitattributes, ...) via a bare local_files_only call would
        # false-negative on a perfectly good cache. config.json + weights are
        # what the embedder actually needs.
        path = snapshot_download(model, local_files_only=True,
                                 allow_patterns=["config.json", "*.safetensors"])
        report(OK, "embed weights", f"{model} cached at {path}")
    except ImportError:
        report(WARN, "embed weights", "huggingface_hub missing; cannot verify")
    except Exception:  # noqa: BLE001
        hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface (default)")
        report(FAIL, "embed weights",
               f"{model} not found under HF_HOME={hf_home} — run setup, and note "
               f"HF_HOME must match where setup cached it")


# ── 4b. NLI cross-encoder (read gate stage 2, T9) ────────────────────────────
# Three pairs with KNOWN labels: an arena-shaped supersession pair
# (contradiction), a canonical SNLI entailment and a canonical SNLI neutral.
# The smoke runs on CPU deliberately — check_env must not claim VRAM before a
# run — and takes a few seconds for 3 pairs.
NLI_SMOKE = [
    ("Thomas Kyd was born in the city of London.",
     "Thomas Kyd was born in the city of Madrid.", "contradiction"),
    ("A soccer game with multiple males playing.",
     "Some men are playing a sport.", "entailment"),
    ("An older and younger man are smiling.",
     "Two men are laughing at cats playing on the floor.", "neutral"),
]


def check_nli(env: dict) -> None:
    model = env.get("HNAV_NLI_MODEL", "cross-encoder/nli-deberta-v3-large")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model, local_files_only=True,
                                 allow_patterns=["config.json", "*.safetensors"])
        report(OK, "nli weights", f"{model} cached at {path}")
    except ImportError:
        report(WARN, "nli weights", "huggingface_hub missing; cannot verify")
        return
    except Exception:  # noqa: BLE001
        hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface (default)")
        report(FAIL, "nli weights",
               f"{model} not found under HF_HOME={hf_home} — fetch with: python -c "
               f"\"from huggingface_hub import snapshot_download; snapshot_download('{model}')\"")
        return

    try:
        import sys as _sys
        if str(REPO) not in _sys.path:
            _sys.path.insert(0, str(REPO))
        from hnav.core.read_gate import CrossEncoderNLI
        nli = CrossEncoderNLI(model_name=model, device="cpu", batch=4, max_length=128)
        scores = nli.score_pairs([(p, h) for p, h, _ in NLI_SMOKE])
        bad = [f"'{p[:36]}…'→'{h[:36]}…' got {s.label}, want {want}"
               for (p, h, want), s in zip(NLI_SMOKE, scores) if s.label != want]
        if bad:
            report(FAIL, "nli smoke", "; ".join(bad))
        else:
            # The gate's criterion is BIDIRECTIONAL contradiction: assert the
            # arena pair also contradicts in the reverse direction.
            p, h, _ = NLI_SMOKE[0]
            rev = nli.score(h, p)
            if rev.label == "contradiction":
                report(OK, "nli smoke", "3/3 labels as expected; arena pair "
                                        "contradicts in BOTH directions (cpu)")
            else:
                report(FAIL, "nli smoke",
                       f"reverse direction of the arena pair got {rev.label}, "
                       f"want contradiction — bidirectional criterion unusable")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "nli smoke", f"inference failed: {str(e)[:120]}")


# ── 5/6. endpoints ───────────────────────────────────────────────────────────
def check_endpoint(env: dict, base_key: str, label: str, required: bool) -> str | None:
    base = env.get(base_key, "").rstrip("/")
    if not base:
        report(WARN, label, f"{base_key} unset")
        return None
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/models", timeout=10) as r:
            data = json.loads(r.read())
        ids = [m.get("id") for m in data.get("data", [])]
        report(OK, label, f"{base} serving: {', '.join(str(i) for i in ids)}")
        return ids[0] if ids else None
    except Exception as e:  # noqa: BLE001
        report(FAIL if required else WARN, label, f"{base} unreachable ({str(e)[:80]})")
        return None


def main() -> int:
    env = load_env()
    if not (REPO / ".env").exists():
        report(WARN, ".env", "not found — copy hnav/deploy/.env.template to .env")

    print("\n── data ─────────────────────────────────────────────────────────")
    check_data()
    print("\n── reproduce committed measurements ─────────────────────────────")
    check_repro()
    print("\n── gpu ──────────────────────────────────────────────────────────")
    check_gpu(env)
    print("\n── embedding model ──────────────────────────────────────────────")
    check_embed_weights(env)
    print("\n── nli cross-encoder ────────────────────────────────────────────")
    check_nli(env)
    print("\n── endpoints ────────────────────────────────────────────────────")
    llm_id = check_endpoint(env, "HNAV_LLM_BASE_URL", "LLM  :8000", required=False)
    check_endpoint(env, "QWEN3_EMBED_BASE_URL", "embed :8001 (optional until T4)", required=False)

    declared = env.get("HNAV_LLM_MODEL")
    if llm_id and declared and llm_id != declared:
        report(WARN, "LLM model id",
               f"endpoint serves {llm_id!r} but HNAV_LLM_MODEL={declared!r} — record the served id")

    mode = env.get("HNAV_MODE", "off")
    if mode == "live":
        # Post-T8 transition (STAGE1_PLAN.md §2 Faz B step 2): live is legal
        # for the Stage-1 READ-PATH campaign only. Stage-0 measurement scripts
        # still refuse it themselves (config.require_not_live). A repo-root
        # .env pinning live would leak into everything — campaign arms should
        # set HNAV_MODE per process instead, hence the warning.
        report(WARN, "HNAV_MODE", "'live' — legal only for the Stage-1 read-path "
                                  "campaign; keep .env at off and set live per-run")
    else:
        report(OK, "HNAV_MODE", mode)

    print("\n" + "=" * 66)
    if _failures:
        print(f" NOT READY — {len(_failures)} blocking issue(s):")
        for f in _failures:
            print(f"   - {f}")
        return 1
    print(" READY for T1." + (f"  ({len(_warnings)} warning(s))" if _warnings else ""))
    print("   bash hnav/deploy/run_t1.sh")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
