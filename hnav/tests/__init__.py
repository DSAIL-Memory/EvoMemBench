"""H-Nav test suite.

Every test in here runs with **no torch, no GPU and no network**. That is not a
convenience — it is the acceptance bar for the local build phase, because the
code is written on a machine that cannot run the real embedder and must work on
the GPU box first try. Anything that cannot be verified locally is listed in
``hnav/BUILD_NOTES.md`` under "necessarily untested until it reaches the GPU".
"""
