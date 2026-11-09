# Fleming–Viot Math-Consistency Audit

**Scope.** This audit checks whether the code faithfully implements the specified *mathematics* (the pseudocode + the 4-case forward/backward kill/clone rule). It is **not** an assessment of numerical accuracy, convergence, or statistical outcomes.

**Files audited.**
- `/work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/utils/fleming_viot.py`
- `/work/pi_markos_umass_edu/bjzhang_umass_edu/ergodic_control_qsd/utils/sde_simulation.py`

---

## 1. Top-line verdict

**YES — the code is consistent with the specified math.** Every one of the nine math items (1)–(9) and both marginals of the 4-case rule (x-events and y-events) map correctly onto the implementation. No genuine sign error, off-by-one, misplaced absolute value, or relabeling defect was found. All deviations observed are confined to the pre-approved intentional-differences list.

The only *literal-reading* subtlety — self-selection (`newindex == index`) leaving the active particle unmoved even for a kill, rather than strictly resampling "the forward particle at donor `i'=i`" per a word-for-word reading of case (8a) — is standard Fleming–Viot convention and was explicitly pre-endorsed in the component spec. It is **not** counted as a discrepancy.

---

## 2. Per-component table

| # | Component | Spec | Verdict | Key evidence (file:line) |
|---|-----------|------|---------|--------------------------|
| 1 | `sde_transition_rates` | (1) Rates | MATCHES | `sde_simulation.py:18-19` primary `r±=(±h·b+a)/(2h²)`; `:20` strict `<0` OR-trigger recomputes both vectors; `:21-22` `(h·max(±b,0)+a/2)/h²`; `:24` `append(+,−)` gives length-2d, +jumps first |
| 2 | `one_step_sde` | (2) OneStep | MATCHES | `sde_simulation.py:42` `p=rates/Σrates`; `:43` `choice` over `{0..2d−1}`; `:45-46` `idx<d ⇒ +h` on coord `idx`; `:47-48` `idx≥d ⇒ −h` on coord `idx−d` (0-indexed relabel of 1-indexed spec) |
| 3 | `inf_swap_rate` | (3) InfSwap; ρ=P(x forward) | MATCHES | `fleming_viot.py:19` `1/(exp(2(Vx−Vy)/ε)+1)` character-exact; interpretation confirmed by drift `:34`, kill split `:57-59`, loop weights `:189-190/239`, estimator `:306` |
| 4 | `symmetrized_dynamics`, `symmetrized_kill_clone_rate` | (4)(5) | MATCHES | `fleming_viot.py:34` `bsymm=(1−2ρ)·DV(x)`; `:58` `r_kf=ρ·c(x)`; `:59` `r_kb=(1−ρ)(c(x)−ΔV(x))`; sum identity `= c−(1−ρ)ΔV`; no abs applied here (deferred to `:98`) |
| 5 | `event_rates` | (6) net & tuple | MATCHES | `fleming_viot.py:92` `r_dyn=Rates(x,b_symm,a(x),h)`; `:98` `net=Σr_dyn+|r_kf|+|r_kb|` (abs **per channel**, not on signed sum); `:100` returns `(net,r_dyn,r_kf,r_kb,h_step)` with signed channels |
| 6 | `killing_cloning` internals | (8) mechanics | MATCHES | `fleming_viot.py:131` `rate>0 ⇒ kill`, `:138` else `clone`; kill `:132-135` sets active←donor(x wp ρ_all[i'], else y); clone `:139-142` overwrites donor with `xnow` (copied `:125`); self-select no-op `:128-129`; recompute sets `[i]` (kill) / `[i,i']` (clone) |
| 7 | Channel routing | (7) selection | MATCHES | `fleming_viot.py:244` `U~Unif(0,1)`; `:245-246` DYN if `U<Σr_dyn/net`; `:249` FWD if `U<(Σr_dyn+|r_kf|)/net` (passes ρ-array + `rkf`); `:251-252` else BWD (passes `1−ρ` + `rkb`); three half-open intervals partition [0,1) |
| 8 | `fleming_viot` main loop | (9) | MATCHES | `fleming_viot.py:191-192` 2N clocks rate=net; `:212` argmin over `append(clocksx,clocksy)`; `:222/231` x/y branch + role swap; `:241` `i=index mod N`; `:255-260` correct write-back; `:263-272` recompute only changed pairs |

---

## 3. The 4-case rule verification

The core math (8) is the forward/backward kill/clone rule. Because `killing_cloning` is generic, correctness hinges on the wiring at the call sites. Confirmed for **both** marginals.

### 3a. X-event (`index < N`, `fleming_viot.py:222-230`)

Setup: `particles1 = x.copy()`, `particles2 = y.copy()`, `rho_particles1 = ρ` (the forward-weight array), `transition_index = index`, active = `x^i`. `xnow = x^i` is copied at `:125` before any mutation.

| Case | Channel/sign | Code path | Result | Spec |
|------|--------------|-----------|--------|------|
| (a) killed as forward | FWD, `rkf>0` | `:250` call w/ `ρ`; kill `:132-135` | `x^i ← x^{i'}` wp `ρ^{i'}`, else `y^{i'}` = forward particle at donor | 8a ✓ |
| (b) cloned as forward | FWD, `rkf<0` | `:250` call w/ `ρ`; clone `:139-142` | forward particle at donor `← x^i` | 8b ✓ |
| (c) killed as backward | BWD, `rkb>0` | `:252` call w/ `1−ρ`; kill `:132-135` | `x^i ← x^{i'}` wp `(1−ρ^{i'})`, else `y^{i'}` wp `ρ^{i'}` = backward particle | 8c ✓ |
| (d) cloned as backward | BWD, `rkb<0` | `:252` call w/ `1−ρ`; clone `:139-142` | backward particle at donor `← x^i` | 8d ✓ |

Write-back `:255-257` stores `particles1→x`, `particles2→y`. Donor weight is read at the **donor** index `newindex` (`rho_all[newindex]` = `ρ^{i'}`), matching the spec's `ρ^{i'}`.

### 3b. Y-event (`index ≥ N`, `fleming_viot.py:231-239`)

Setup: `particles1 = y.copy()`, `particles2 = x.copy()`, `rho_particles1 = 1−ρ`, active = `y^i`. Rates `rkf_y=(1−ρ^i)c(y^i)`, `rkb_y=ρ^i(c(y^i)−ΔV(y^i))` come from `event_rates(y, 1−ρ, …)` (`:190/:268`) — i.e. spec (5) with effective forward-weight `1−ρ^i`, exactly matching "y is forward wp `1−ρ`". Inside `killing_cloning`, `xnow = particles1[i] = y^i` (`:125`).

**Why it works:** for a y-event the code applies **two** swaps that cancel — the particle arrays are swapped (`particles1↔particles2`, `:232-233`) *and* the weight is complemented (`ρ→1−ρ`, `:239`). Hence `x_allparticles[i'] wp rho_all[i']` inside `killing_cloning` reproduces the **global** forward-particle definition `x^{i'} wp ρ^{i'}` regardless of marginal:

| Case | Channel | `rho_all` passed | Result (global) | Spec |
|------|---------|------------------|-----------------|------|
| (a) | FWD (`:250`) | `1−ρ` | active `y^i ←` forward particle at `i'` (`x^{i'}` wp `ρ^{i'}`, else `y^{i'}`) | 8a ✓ |
| (b) | FWD (`:250`) | `1−ρ` | forward particle at `i' ← y^i` | 8b ✓ |
| (c) | BWD (`:252`) | `1−(1−ρ)=ρ` | active `y^i ←` backward particle at `i'` | 8c ✓ |
| (d) | BWD (`:252`) | `ρ` | backward particle at `i' ← y^i` | 8d ✓ |

Write-back `:258-260` stores `particles1→y`, `particles2→x`; a y-event clone can legitimately overwrite the x-member of a donor pair, and this propagates both populations correctly.

---

## 4. Adversarial refutation results

Six independent refutation attempts were run against the strongest correctness claims. **None was refuted** — every claim survived.

| Claim | Refuted? | Outcome |
|-------|----------|---------|
| X-event 4-case rule holds end-to-end (a–d) | **No** | All four cases trace correctly; donor weight read at `i'`; `xnow` not stale (copied `:125`); only edge is benign self-select no-op |
| Y-event 4-case rule holds under relabeling (`particles1=y`, `ρ→1−ρ`) | **No** | Double swap (arrays + weight) cancels; every channel resamples/overwrites the correct **global** forward/backward particle |
| Sign/channel/normalization exactly consistent; 3 event-type probs sum to exactly 1 | **No** | Same `net` and `dyn` array feed both `:98` and `:245`, so partition is definitional, not approximate; forward KILLs iff `rkf>0` |
| `c(x)=0 ⇒ r_kf=0`, forward channel never fires; all kill/clone via backward | **No** | `:58` gives `r_kf=0` exactly; `:249` elif reduces to `U<dyn_frac`, unreachable after `:246` already false — dead branch when `c=0` |
| Donor uniform over N pairs, correct `ρ_all[i']`, no off-by-one across argmin/`mod N`/recompute set | **No** | `newindex` uniform on `{0..N−1}`; all 8 (event×channel×op) combos verified; recompute indices are in-pair `{0..N−1}` |
| `net=|r_kf|+|r_kb|` essential: forward-KILL + backward-CLONE fire as 2 independent events, no signed collapse | **No** | Constructed `c=1,ΔV=5,ρ=0.5 ⇒ r_kf=+0.5, r_kb=−2`: net uses `|0.5|+|−2|=2.5 ≠ |r_kf+r_kb|=1.5`; signed sum appears only in docstring `:47`, never computed |

**Refuted claims: none. Counterexamples that broke the code: none.**

---

## 5. Genuine discrepancies

**NONE.**

No genuine math-vs-code discrepancy was found. Every sign, index partition, absolute-value placement, threshold, and donor/weight wiring matches the spec.

### Deviations observed but excluded (all on the pre-approved intentional-differences list)
- Optional `wrap_period` periodic wrapping (`sde_simulation.py:51-52`, `:103-104`).
- O(n) list accumulation of trajectories (`fleming_viot.py:177-181, 290-295`).
- Extra `all_rho` return value (`:185, :295`).
- Reused randomized `h_step` drawn in `event_rates` and returned in the tuple (`:87, :100`).
- Trailing singleton dims from `V` returning shape `(1,)` in 1D.
- Recompute-changed at loop bottom (`:263-272`) rather than top — functionally equivalent.

### Explicitly-endorsed literal subtlety (not a discrepancy)
- Self-selection `newindex == index` (`fleming_viot.py:128-129`) leaves the active particle unmoved for both kill and clone. A strict word-for-word reading of (8a) would set the active to "the forward particle at donor `i'=i`"; the no-op is standard FV convention and was pre-endorsed in the component spec.
