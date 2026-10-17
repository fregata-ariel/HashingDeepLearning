# Scatterbrain Appendix C/D review

The primary appendices became accessible on 2026-10-17. Ordinary HTTPS with
default TLS verification returned HTTP200 for each URL on the first attempt.
The earlier CONNECT403 observation is historical. The user-supplied conference
main PDF remains unchanged; the following are separately identified artifacts.

| Version | Local PDF | SHA-256 | Appendix locators |
| --- | --- | --- | --- |
| NeurIPS 2021 publisher supplement | `papers/pdf/scatterbrain-2021-neurips-supplement.pdf` | `0b4adeba7c339a3f997ff35c39e99232fc44ca5e78457c719614b5bf23773ed9` | C: PDF7/printed21; D.3: PDF15/printed29 |
| arXiv:2110.15343v1 | `papers/pdf/scatterbrain-2021-arxiv-v1.pdf` | `24d255fb7bd129ae6a2fb25cbafe80709d98430e5e4ee52d86c484987124ca07` | C: PDF22; D.3: PDF30 |

The publisher supplement has 20 pages according to pdfinfo. Its final-page
object/annotation produces a pdftotext warning; the cited C/D pages extract
and render successfully and were visually reviewed. The 34-page arXiv PDF
extracts successfully. Pagination and reference numbering differ between these
versions; byte/text identity is not asserted. Exact URLs, sizes and acquisition
observations are in `papers/scatterbrain-paper-provenance.json`.

## Algorithm 1 and shared normalization

Appendix C Algorithm 1 explicitly samples IID Gaussian W and defines the
positive feature map `phi(x) = exp(Wx - ||x||²/2) / sqrt(m)`. It separately forms
low-rank numerator/row sums, and sparse exact-minus-low-rank residuals on the
collision support. Lines21–23 return

`diag(D_lr + D_s)^(-1) (O_lr + O_s)`.

The shared denominator is therefore directly described by the primary
algorithm. A selected pair contributes once through set support. The later
implementation's occurrence-count correction is checked independently against
that union-support interpretation; Algorithm 1 does not specify its exact
multi-round implementation.

This correspondence does not equate the released orthogonal/cosh features,
epsilon additions, denominator floors, residual dropout or rank-sorted
XBOX+/SMYRF support with the displayed IID-Gaussian/hash construction. Nor does
a ratio of numerator and denominator inherit unnormalized-kernel unbiasedness.

## Printed causal inconsistencies

Both versions define a lower-triangular binary mask M but print
`exp((QK^T) ⊙ M)`. That expression gives exp(0)=1 on forbidden positions;
masking the exponential kernel instead would be `exp(QK^T) ⊙ M`. The same
paragraph says to exclude sparse locations `i>j`, inconsistent with keeping a
lower triangle, while the later `(S⊙M)V` expression uses that triangle.
These discrepancies were confirmed in rendered pages, not inferred from
pdftotext. They are recorded as printed inconsistencies. Causal execution
remains outside the current CPU contract; no silent transcription fix is
presented as a verified primary algorithm.

## Theorem 2, Appendix D.3 and the cited cross-polytope result

The statement concerns one unnormalized entry `A = exp(q^T k)`, with unit-sphere
q/k and distance less than tau. The proof conditions on selected support, then
substitutes unconditional random-feature moments in the off-support branch.
It does not explicitly state the required conditional-moment justification.
Independence of hash/support randomness and feature randomness is sufficient;
appropriate conditional first/second moment equalities can also suffice.
The between-conditional-means term in total variance vanishes only when those
conditional means agree with A.

Under independent support M and unbiased finite-variance feature estimator Z,
our separate algebra gives

`Ahat = M*A + (1-M)*Z`,
`E[Ahat] = A`,
`Var(Ahat) = (1-P(M=1))*Var(Z)`.

With feature bias b, the independent mixture instead has bias `(1-p)*b` and
variance `(1-p)*Var(Z) + p*(1-p)*b²`. Unconditional unbiasedness alone does not
justify replacing conditional moments. These equations are our explicit
derivation, not additional hypotheses quoted from the printed theorem.
Neither D.3 nor these identities prove unbiased normalized attention.

The cited *Practical and Optimal LSH for Angular Distance* was also obtained
from its primary arXiv source (visible `1509.02897v1` stamp), preserved as
`papers/pdf/angular-lsh-2015-arxiv-v1.pdf`, SHA256
`a771ebf1f0982b102ee1e47bd0cf886eefd9af04588248c3f4956fa105dee466`.
Theorem1, PDF5, explicitly assumes unit-sphere inputs at actual distance
`||p-q|| = tau`, `0 < tau < 2`. For a single full Gaussian cross-polytope hash,
it states `ln(1/p_base) = tau²/(4-tau²)*ln(d) + O_tau(ln ln(d))`.
AppendixB.2, PDF17, makes the two-sided remainder explicit.

Scatterbrain instead uses a distance threshold and does not explicitly tie
its displayed p to Algorithm1's K/L amplification. Replacing actual distance
by an upper threshold does not establish equality of the pair probability.
Under independent repeated hashes, the familiar AND-K/OR-L event has probability
`1-(1-p_base^K)^L`; this is our event-algebra derivation, not an extra statement
quoted from Scatterbrain's proof. The asymptotic single-hash result is not a
finite-dimensional probability for later rank-sorted XBOX+/SMYRF clusters.
Reading these primary hypotheses is not independent theorem reproduction or
validation of all proofs in D.1/D.2.

The source-mapping review is complete at these explicit boundaries. Actual
CPU results remain the independently checked implementation contracts; paper
model metrics, causal execution and statistical sampling quality are separate.
