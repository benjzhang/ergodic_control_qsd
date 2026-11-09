# Presentation handoff — INS for ergodic control

**What this shows (two claims):**
1. **INS produces the right control.** Sampling the QSD with forward/backward Infinite-Swap
   Fleming–Viot, then score-matching the control, yields a controller that drives the system
   to the *exact* optimal distribution ρ* (corr **0.99**) at cost **within ~5% of optimal**.
2. **It's more efficient than plain Fleming–Viot.** At matched budget, INS's eigenvalue
   (control-cost) estimate has **3.7× lower MSE** than plain FV.

Every number is measured against a finite-difference ground truth (exact λ = 2.671).
The whole pipeline uses the **weighted fwd/bwd empirical measure** (weights dt·ρ / dt·(1−ρ)),
directly — no resampling.

Figures live in this folder; alternates/backup in `supporting/`.

---

## 20-second background
Diffusion killed at rate c(x); we want the control minimizing long-run average cost.
The optimum is the principal eigenpair of `L − c`: eigenvalue λ (= optimal cost), QSD
ν ∝ φ·m, optimal controlled density ρ* ∝ φ²·m, control `u* = a∇log φ`. INS samples ν;
a score model learns ∇log ν; the control follows.

## Slide 1 — the problem  (`01_problem_setup.png`)   ⚠ PROVISIONAL
Four heatmaps: base measure m, killing c(x), QSD ν, optimal ρ*. Exact λ = 2.671.
*(This figure is the weak link — see "open decisions" below; likely to be redrawn.)*

## Slide 2 — INS produces the right control  (`02_right_control.png`)   ← core result
Left: exact optimal ρ* (FD). Right: the distribution the INS-learned control actually
produces. They match (**corr 0.99**), and the achieved ergodic cost is **within ~5% of the
optimal λ**.
- *Talking point:* "We never see the true control. INS samples the QSD, a score model learns
  the control from those samples, and when we run the controlled system it lands on the exact
  optimal distribution at near-optimal cost."
- Key method detail worth a line: the score is matched on the **full weighted fwd/bwd
  measure** — its off-basin particles give the control signal in the transition regions, which
  cut the cost gap from 27% → 12% → **5%**.

## Slide 3 — more efficient than plain FV  (`03_efficiency_vs_fv.png`)   ⚠ MODEST
Estimate ± SE vs exact λ, and MSE. INS +1.3% vs plain FV +4.4%; **MSE 3.7× lower** at matched
budget ⇒ plain FV needs ~3.7× the samples for the same accuracy.
*(Honest: on this smooth problem the advantage is real but modest — see below.)*

## supporting/
- `ins_nu_samples.png` — INS particle cloud over the QSD ν (INS samples ν accurately, corr 0.98).
- `control_resampled_12pct.png` vs `control_weighted_5pct.png` — the A/B showing the weighted
  measure tightening the cost gap 12% → 5%.

---

## Open decisions (flagged, not yet resolved)
- **Slide 1 (setup)** is four raw heatmaps; m and ν look alike. Better: fewer annotated panels
  emphasizing "cost c penalizes the center → optimal control drives mass to the low-cost
  corners (ρ*)", or sample trajectories (uncontrolled-wandering vs controlled-steered).
- **Slide 3 (vs FV)** is modest because the problem is *smooth* (easy for both samplers). A
  compelling "INS beats FV" needs a harder, metastable problem where plain FV visibly
  struggles — different from the control problem.

## Guardrails
- Claim: "INS-learned control reaches optimal ρ* at within ~5% of optimal cost" and "3.7× lower
  MSE than plain FV at matched budget, on this example." Do **not** generalize to "always better."
- The vs-FV advantage is real but modest; don't oversell it.

## Source
`examples/ergodic_control/`: `problem_smooth.py`, `fd_smooth.py`, `control_weighted_smooth.py`
(weighted-DSM control), `run_smooth.py`+`aggregate_smooth.py` (MSE), `fig_results.py`.
