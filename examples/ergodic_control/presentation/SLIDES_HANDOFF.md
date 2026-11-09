# Slide brief — "Sampling-based control for ergodic-control / QSD problems"

**For:** a slide-building agent. **Goal of the talk:** show **how the method works** (a
sampling + score-matching pipeline for computing an optimal control), NOT to benchmark it
against other samplers. Keep it expository.

**Status note:** Some figures are still rendering on the cluster (marked [RENDERING]).
Build the deck skeleton, titles, bullets, captions, and speaker notes now; leave image
placeholders with the exact filenames below and drop the PNGs in when they land. All figure
files are in `examples/ergodic_control/figures/` (and mirrored into `presentation/`).

---

## The one-paragraph summary (for your own grounding)
We solve an **ergodic-control problem**: a diffusion `dX = −∇V dt + √a dW` on a periodic
2D box, *killed* at rate `c(x)`, and we want the control minimizing the long-run average
cost `c + |u|²/(2a)`. The optimum is the principal eigenpair of the sub-generator `L − c`:
eigenvalue **λ = optimal cost**, eigenfunction **φ**, with **QSD ν ∝ φ·m**, **optimal
controlled density ρ\* ∝ φ²·m**, and **optimal control u\* = a ∇log φ**. The method:
(1) sample ν with a **forward/backward Infinite-Swap Fleming–Viot** particle system;
(2) learn the control from those samples by **denoising score matching** (a score-based
generative model); (3) run the controlled process, which lands on ρ\*.

System specifics: torus [0,4]², diffusion `a = 2`, `V = −log(16-Gaussian mixture)`,
killing `c(x) = κ(2 − cos(2πx₁/L) − cos(2πx₂/L))`, κ=2 (high in the center, low in the four
corners → mass is driven to the low-cost corners). Exact **λ = 2.141** for the structured
(σ=0.2) mixture used in the mechanism figure.

---

## Deck structure (7 slides)

### 1 — Title / motivation
- Title: "Sampling-based optimal control via forward/backward Fleming–Viot + score matching"
- One line: "Compute an ergodic-control law from particle samples — no grid, no PDE solve."
- No figure.

### 2 — The problem  [figure: `fd_reference.png` — READY]
- Bullets: killed diffusion + running cost `c(x)`; the answer is the principal eigenpair of
  `L − c`: cost λ, QSD ν ∝ φm, optimal density ρ* ∝ φ²m, control u* = a∇log φ.
- Figure shows four fields: base mixture m, killing c(x) (high center), QSD ν, optimal ρ*
  (mass in the low-cost corners). Caption: "The target objects; λ = 2.141 is the optimal cost."
- Speaker note: "We know the exact answer from a grid solver, purely to check the method."

### 3 — How it works (the pipeline)  [figure: `howitworks_sharp.png` — RENDERING]  ← CENTERPIECE
- Three panels: (1) **Sample the QSD ν** with fwd/bwd Fleming–Viot (particle cloud in the
  wells); (2) **Learn the control** u* = a∇log φ by score-matching the samples (arrow field
  pushing mass out of the high-cost center toward the corners); (3) **Apply the control** —
  the controlled process's samples collapse onto ρ* (low-cost corners).
- Caption: "Sample the killed system's QSD → learn the control from the samples → the
  controlled process reaches ρ*."
- Speaker note: this is the whole method in one picture; walk left → right.

### 4 — Stage 1: sampling the QSD (Fleming–Viot + infinite swap)
- Bullets: N particles diffuse and are *killed* at rate c; a killed particle *respawns* onto
  a survivor → the ensemble concentrates onto ν. Two coupled ensembles (forward + backward)
  with an instantaneous **swap** symmetrize the estimate. Output = a **weighted empirical
  measure** (weights dt·ρ forward, dt·(1−ρ) backward).
- Optional inset figure: `supporting/ins_nu_samples.png` [READY] (ν sample cloud).

### 5 — Stage 2: learning the control (score matching)
- Bullets: the control is the score of the eigenfunction, u* = a∇log φ = a∇log ν + 2∇V, so
  learning ∇log ν gives the control. We use **denoising score matching** directly on the
  **weighted fwd/bwd measure** (draw training batches ∝ the particle weights — no
  resampling). Its off-basin particles supply score signal in the transition regions.
- No new figure required (arrow field already appears in slide 3, panel 2).

### 6 — It reaches the optimum  [figure: `control_weighted_smooth.png` — READY]
- Bullets: run the controlled process under the learned u* → its stationary law matches the
  exact ρ* (correlation ≈ 0.99) and it achieves cost within a few % of the optimal λ.
- Caption: "Exact ρ* (left) vs the controlled process under the learned control (right)."
- NOTE for the presenter: this validation figure is from the SMOOTH (σ=0.4) variant, where
  the cost gap is tightest (~5%). The mechanism figure (slide 3) uses the STRUCTURED (σ=0.2)
  mixture so the wells are visible. If you prefer one mixture throughout, tell the authors —
  a σ=0.2 validation figure can be produced.

### 7 — Outlook (optional)
- Bullets: sample-based + neural score ⇒ grid-free, so it extends to higher dimension where
  PDE/grid solvers fail. (State, don't show — this is 2D.)

---

## Figure inventory (filenames + status)
| slide | file (in figures/) | shows | status |
|---|---|---|---|
| 2 | `fd_reference.png` | problem setup (m, c, ν, ρ*), λ=2.141 | READY |
| 3 | `howitworks_sharp.png` | 3-stage pipeline on σ=0.2 (real wells) | **RENDERING** |
| 3-alt | `howitworks.png` | same pipeline on σ=0.4 (flatter, cleaner control) | READY |
| 4 | `supporting/ins_nu_samples.png` | ν sample cloud | READY |
| 6 | `control_weighted_smooth.png` | ρ* match (learned control) | READY |

## Framing guardrails (please respect)
- Frame everything as **"here is how the method works"** — mechanism, not comparison. Do NOT
  add "beats method X" / MSE bars; that's out of scope for this deck.
- The control is validated **functionally** (reaches ρ*, near-optimal cost), not by a
  pointwise control-field match — the field is approximate. Say "reaches the optimal
  distribution at near-optimal cost," not "recovers the exact control."
- Numbers to quote: optimal cost λ = 2.141 (σ=0.2) / 2.671 (σ=0.4); ρ*-match corr ≈ 0.99;
  cost within ~5% of optimal (σ=0.4). One demonstrative 2D example — don't over-generalize.

## Source
`examples/ergodic_control/`: problem `problem.py` (σ=0.2) / `problem_smooth.py` (σ=0.4);
exact solve `fd_reference.py` / `fd_smooth.py`; pipeline `control_weighted_smooth.py`
(+ `_sharp` variant); figures `plot_howitworks.py` (+ `_sharp`).
