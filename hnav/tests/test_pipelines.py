"""Tests for pipelines/_shared/runner.py — the frozen-pipeline driver.

The driver holds no science, so the tests pin its guarantees: the frozen
artifacts are verified byte-for-byte, the analysis reproduces the committed
campaign numbers from the committed artifact, and a tampered pin refuses.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from pipelines._shared import runner

REPO = pathlib.Path(__file__).resolve().parents[2]


def _spec(name):
    return runner.load_pipeline(REPO / "pipelines" / name)


ALL_ARMS = sorted(d.name for d in (REPO / "pipelines").iterdir()
                  if (d / "pipeline.json").exists())


def test_pinned_hashes_match_the_committed_operating_points():
    # [E2E-4] every arm, not just the two originals: five arms (including both
    # new ones) shipped with unverified pins while this test passed.
    assert len(ALL_ARMS) >= 7, ALL_ARMS
    for name in ALL_ARMS:
        spec = _spec(name)
        assert runner.sha256(REPO / spec["operating_point"]) == \
            spec["operating_point_sha256"], name


def test_verify_frozen_passes_on_the_real_artifacts(monkeypatch):
    monkeypatch.delenv("HNAV_MODE", raising=False)
    monkeypatch.delenv("HNAV_EMBED_MODEL", raising=False)
    for name in ALL_ARMS:
        assert runner.verify_frozen(_spec(name)) == [], name


def test_verify_frozen_refuses_a_tampered_pin_and_a_foreign_embedder(monkeypatch):
    spec = _spec("hnav_raw")
    spec["operating_point_sha256"] = "0" * 64
    problems = runner.verify_frozen(spec)
    assert any("MODIFIED" in p for p in problems)

    spec = _spec("hnav_abtt")
    monkeypatch.setenv("HNAV_EMBED_MODEL", "some/other-encoder")
    from hnav.config import get_config
    get_config(refresh=True)
    try:
        problems = runner.verify_frozen(spec)
        assert any("new calibration campaign" in p for p in problems)
    finally:
        monkeypatch.delenv("HNAV_EMBED_MODEL")
        get_config(refresh=True)


def test_abtt_whitening_fingerprint_is_cross_checked():
    spec = _spec("hnav_abtt")
    art = json.loads((REPO / spec["operating_point"]).read_text(encoding="utf-8"))
    assert art["provenance"]["whitening_fingerprint"] == \
        spec["whitening_fingerprint"]


def test_analysis_reproduces_the_committed_campaign_numbers():
    """The committed A1 artifact IS the shipped result; the driver's analysis
    must read 17->37/66 conflicted and 64/100 overall out of it, and find no
    void condition."""
    a = runner.analyse_artifact(
        REPO / "stage0_results/abtt/abtt_arm_A1_raw_sh64k.json")
    assert a["rows"]["conflicted"] == pytest.approx(
        {"n": 66, "native": 17, "hnav": 37, "net": 20,
         "mcnemar_p": a["rows"]["conflicted"]["mcnemar_p"]})
    assert a["rows"]["conflicted"]["mcnemar_p"] < 1e-4
    assert a["rows"]["all"]["hnav"] == 64
    assert a["void"] == [] and a["aa_discordant"] == 0
    assert a["token_delta_pct"] < 0


def test_confirmatory_flag_is_present_exactly_for_sh64k():
    spec = _spec("hnav_raw")
    out = pathlib.Path("x.json")
    assert "--confirmatory" in runner.detector_gap_cmd(spec, "sh_64k", out, False, False)
    assert "--confirmatory" not in runner.detector_gap_cmd(spec, "sh_6k", out, False, False)
    abtt = runner.detector_gap_cmd(_spec("hnav_abtt"), "sh_6k", out, False, False)
    assert "--geometry-space" in abtt and "--whitening-artifact" in abtt
