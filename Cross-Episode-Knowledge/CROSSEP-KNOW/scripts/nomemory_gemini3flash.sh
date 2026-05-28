#!/usr/bin/env bash
bash run_context_memory.sh \
    --no-memory \
    --infer-model gemini-3-flash-preview \
    --workers 25 \
    --skip-compare
