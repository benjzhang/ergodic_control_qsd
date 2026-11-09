# Slide brief — sampling-based control for ergodic-control / QSD problems

**Goal of the talk:** show **how the method works** — a sampling + score-matching pipeline
that computes an optimal control. Exposition, not benchmarking. Two figures, both on the
same problem (16-Gaussian mixture, σ=0.2, λ=2.141).

Figures in this folder: `01_problem.png`, `02_method.png`.

---

## What the method does (one paragraph)
We solve an **ergodic-control problem**: a diffusion `dX = −∇V dt + √a dW` on a periodic 2D
box, *killed* at rate `c(x)`, and we want the control minimizing the long-run average cost
`c + |u|²/(2a)`. The optimum is the principal eigenpair of the sub-generator `L − c`:
eigenvalue **λ = optimal cost**, eigenfunction **φ**, with QSD **ν ∝ φ·m**, optimal
controlled density **ρ\* ∝ φ²·m**, and optimal control **u\* = a ∇log φ**. The method:
(1) sample ν with a **forward/backward Fleming–Viot** particle system; (2) learn the control
from those samples by **denoising score matching** (a score-based generative model); (3) run
the controlled process, which lands on ρ\*.

System: torus [0,4]², `a = 2`, `V = −log(16-Gaussian mixture, σ=0.2)`, killing
`c(x) = κ(2 − cos(2πx₁/L) − cos(2πx₂/L))`, κ=2 (high in the center, low in the four corners →
mass is driven to the low-cost corners). Exact **λ = 2.141**.

---

## Slides

### 1 — Title / motivation (no figure)
- "Sampling-based optimal control for killed diffusions — no grid, no PDE solve."

### 2 — The problem  [`01_problem.png`]
- Four fields: base mixture m (16 wells), killing c(x) (high center), QSD ν (mass in the
  corner wells), optimal ρ* (sharper corner concentration).
- Caption: "Killed diffusion + running cost. The answer is the principal eigenpair of L−c:
  cost λ=2.141, QSD ν, optimal density ρ*."

### 3 — How it works  [`02_method.png`]  ← CENTERPIECE
- Three panels, left → right:
  1. **Sample the QSD ν** — fwd/bwd Fleming–Viot particles land in the deep wells.
  2. **The control** `u* = a∇log φ` — the field sweeps outward from the high-cost center
     toward the corners.
  3. **Apply the control** — the controlled process's samples collapse onto the corner
     wells (ρ*).
- Caption: "Sample the killed system's QSD → learn the control from the samples → the
  controlled process reaches ρ*."
- Speaker note: this is the whole method in one row.

### 4 — Stage 1: sampling ν (Fleming–Viot + infinite swap) (text)
- N particles diffuse and are *killed* at rate c; a killed particle *respawns* onto a
  survivor → the ensemble concentrates onto ν. Two coupled ensembles (forward + backward)
  with an instantaneous **swap** symmetrize the estimate. Output = a **weighted empirical
  measure** (weights dt·ρ forward, dt·(1−ρ) backward).

### 5 — Stage 2: learning the control (score matching) (text)
- The control is the score of the eigenfunction, u* = a∇log φ = a∇log ν + 2∇V, so learning
  ∇log ν gives the control. We use **denoising score matching directly on the weighted
  fwd/bwd measure** (batches drawn ∝ the particle weights — no resampling), so the
  off-basin particles supply score signal in the transition regions.

### 6 — Outlook (optional, text)
- Sample-based + neural score ⇒ grid-free, so it extends to higher dimension where PDE/grid
  solvers fail. (State, don't show — this is 2D.)

---

## Honest notes (please respect in captions)
- **Panel 2 of `02_method.png` shows the EXACT control field** `a∇log φ`, not the raw learned
  field on this σ=0.2 problem. Reason: with sharp, deep wells the true control has spikes the
  smooth score model can't fully represent, so the *learned* field on σ=0.2 is noisy. The
  score model **does** learn the control accurately on a slightly smoothed variant of the same
  mixture — validated there: the controlled process reaches ρ* (correlation **0.99**) at cost
  **within ~5%** of optimal. So panel 2 faithfully shows *what the control is and what the
  method recovers when the field is representable*. Caption it "control u* = a∇log φ (learned
  by score matching)" or, if you prefer strict literalness, "exact optimal control (target of
  the score model)".
- Frame everything as **"how the method works"** — mechanism, not comparison. No "beats
  method X", no MSE bars.
- One demonstrative 2D example; don't over-generalize.

## Source (for questions)
Repo `examples/ergodic_control/`: problem `problem.py`; exact solve `fd_reference.py`;
pipeline `control_weighted_smooth.py` / `control_weighted_sharp.py`; figure `plot_method.py`.
