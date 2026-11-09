# Ergodic control via score matching on the Fleming–Viot QSD — example report

**Status: draft for review.** Numbers below are from `N=256` FD and the runs in
`figures/`. Please check the math statements and framing.

## 1. Problem

A 2D array-of-Gaussians on the torus `[0,4]^2` turned into an ergodic-control /
quasi-stationary problem by adding a killing rate.

- Base (uncontrolled) dynamics: `dX = -∇V(X) dt + √a dW`, `a = 2`, with
  `V = -log m` where `m` is a 4×4 mixture of Gaussians (`σ = 0.2`). So the base
  invariant measure is exactly the mixture `m`.
- Killing rate `c(x) = κ (2 - cos(2πx₁/4) - cos(2πx₂/4))`, `κ = 2` ("corner"
  pattern): low killing at the 4 corner wells, high at the center.

Chosen by a sweep (`figures/explore_*.png`): κ=2 reshapes the uniform 16-well base
into a QSD/optimal measure concentrated in the 4 low-killing corner wells — a
clean, visual control task that doesn't fully collapse.

## 2. Method and the relations being tested

Sub-generator `L - c` with `L f = -∇V·∇f + (a/2)Δf`. Principal eigenpair
`(λ, φ)`: `(L - c)φ = -λφ`, `φ > 0`. Then

- **QSD** (survival-conditioned law): `ν ∝ φ · m`.
- **Ergodic control value** = the eigenvalue:
  `λ = inf_u  limsup_T (1/T) E ∫₀ᵀ [ c(X_t) + |u_t|²/(2a) ] dt`,
  over `dX = (-∇V + u) dt + √a dW`.
- **Optimal control** = score of the eigenfunction: `u*(x) = a ∇log φ(x)`.
  Using `ν ∝ φ m` and `∇log m = -(2/a)∇V`, this is
  `u* = a ∇log ν + 2 ∇V`   (the `2∇V` coefficient is independent of `a`).
- **Optimally-controlled invariant**: `ρ* ∝ φ² m`.

Pipeline: (i) sample the QSD `ν` with forward–backward INS Fleming–Viot
(`sample_qsd.py`); (ii) learn `s_θ ≈ ∇log ν` by denoising score matching
(torch MLP, periodic Fourier features, `score_match.py`); (iii) form the control
`u* = a s_θ + 2∇V` and check it (`control_test.py`). A "clean" variant
(`clean_demo.py`) trains `s_θ` on samples drawn directly from the FD `ν`, to
separate the score→control method from the sampler.

## 3. Results

**Finite-difference reference** (`fd_reference.py`, `figures/fd_reference.png`):
`λ = 2.141`. The ergodic-cost identity is verified numerically with the *exact*
control: `E_{ρ*}[c + |u*|²/(2a)] = 2.139 ≈ λ` (split: `E_{ρ*}[c] = 1.42`,
control effort `0.72`).

**Score → control recovers the target measures.** With `s_θ` from FD-drawn `ν`
samples (`figures/clean_demo.png`): the controlled processes reproduce **both**
the QSD `ν` (drift `s_θ`) and the optimal `ρ*` (drift `-∇V + u*`) — 4-corner-
dominant, matching FD. Score error (ν-weighted) `‖s_θ - ∇log ν‖ ≈ 0.14`.

**Sign-duality check** (`figures/sign_check.png`, `sign_check.py`). The base/QSD
sampler runs drift **−∇V**; the optimally controlled process runs drift
**+∇V + a∇log ν** (the ∇V term flips sign). Using the *exact* FD score to isolate
the sign: `+∇V` recovers `ρ*` (L1 = 0.036), while the flipped `−∇V` does **not**
(L1 = 0.43 — the strong well-confinement traps mass across all 16 wells instead of
concentrating at the 4 corners). So the correct construction is `u* = a∇log φ`,
i.e. `-∇V + u* = +∇V + a∇log ν`.

**Ergodic cost is above optimal — by a quantifiable, non-fundamental margin.**

| pipeline | QSD source | score err | ergodic cost | vs λ=2.14 |
|---|---|---|---|---|
| exact FD control | — | 0 | 2.139 | optimal ✓ |
| clean demo | FD-drawn ν | 0.14 | 2.95 | +38% |
| FV/INS demo | Fleming–Viot | 0.20 | 3.88 | +81% |

The excess cost is `≈ E_{ρ*}[|s_θ - ∇log ν|²]` — the `|u|²` term amplifies score
error — so it is bounded by score accuracy, not a modeling error. Shapes (ν, ρ*)
are recovered even where the cost is elevated.

## 4. Two honest caveats (both about *inputs*, not the control method)

1. **FV/INS QSD sampling is discretization-biased here.** The pure-jump lattice
   under-concentrates the QSD on these sharp σ=0.2 wells with strong killing:
   `E_ν[c] = 2.68` at `h0=0.1`, `2.57` at `h0=0.05`, vs the true `2.14`
   (corner mass 53% vs 69%). Convergence in `h0` is slow (~`O(h^0.3)`), so
   matching FD through the sampler alone would need impractically fine `h0`.
   (`figures/qsd_samples_vs_fd.png`.)
2. **Single-scale DSM has an accuracy floor.** `σ_dsm ≈ 0.04` is a sweet spot;
   smaller (`0.025`) is *worse* (target variance `2/σ²` swamps the signal →
   score err 0.20, cost 3.86). ~0.14 score error is about the floor for this
   setup; a multi-scale / higher-fidelity score objective would be needed to
   push the cost toward λ.

## 5. Takeaway

The score-matching → control chain is **theoretically correct** (exact control
attains cost `= λ`) and **works in practice** (it recovers the QSD `ν` and the
optimal `ρ*`). The gap to optimal cost is attributable to two identifiable input
limitations — QSD-sampler discretization and DSM score accuracy — rather than to
the control construction.

Files: `problem.py`, `fd_reference.py`, `sample_qsd.py`, `score_match.py`,
`control_test.py`, `clean_demo.py`; figures in `figures/`.
