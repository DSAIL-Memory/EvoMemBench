"""QDA-style conflict scorer on embedding differences.  [QDA campaign]

Offline tier: everything here except :mod:`score` may read gold labels.
Nothing under ``hnav/core/`` or ``hnav/adapters/`` may import from this
package. :mod:`score` is the artifact-only consumer (loads ``weights.npz``,
never touches the dataset) so a later shadow/live arm can call it without
crossing the leakage boundary.

The model (PREREG.md in ``stage0_results/qda_filter/`` is the authority):
unit fact embeddings decompose locally as ``v = mu + phi_subject +
psi_relation + chi_(relation,object) + eta``, so a conflict pair's difference
``d = v_new - v_old`` is a low-rank relation-conditioned object edit while a
subject-swap negative's ``d`` is diffuse subject variation. The score is the
Gaussian log-likelihood ratio of the two difference distributions (QDA):
CES (`hnav.geometry_filter`) is this score with eigen-weights quantized to
{+1, -1, 0} and the norm discarded; the deliverable is the unquantized
version plus the ordered (old->new) and norm terms.

Upstream frozen inputs, applied and never re-fit:
    stage0_results/abtt/abtt_whitening_D128.json      (P_abtt, D=128)
    stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz
    hnav/_cache/emb/                                   (campaign embeddings)

Outputs land in ``stage0_results/qda_filter/``.
"""
