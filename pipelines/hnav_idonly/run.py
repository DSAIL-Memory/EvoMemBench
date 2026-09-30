#!/usr/bin/env python3
"""Run this frozen pipeline against a new answering model. See README.md."""
import pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pipelines._shared.runner import main   # noqa: E402
raise SystemExit(main(HERE))
