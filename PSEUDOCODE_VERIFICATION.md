# Pseudocode Verification: Infinite-Swapping Fleming-Viot Implementation

**Files audited**
- `utils/sde_simulation.py` (`sde_transition_rates`, `one_step_sde`)
- `utils/fleming_viot.py` (`inf_swap_rate`, `symmetrized_dynamics`, `symmetrized_kill_clone_rate`, `event_rates`, `killing_cloning`, `fleming_viot`)

## Summary verdict

**The code faithfully implements the pseudocode. No genuine discrepancies were found.**

Every algorithm matches the specification line for line. All deviations encountered are the pre-approved intentional differences (periodic wrapping, list accumulation of trajectories, the extra `all_rho` return value, `h_step` carried through the rate tuple, and trailing singleton dims from `V`). One important cross-cutting detail — the pseudocode recomputes rho/rates for `I` at the *top* of the loop while the code recomputes at the *bottom* — was checked and is functionally equivalent: the same set of clocks/rates is freshly drawn for the same index set each iteration.

Per-algorithm verdicts:

| Algorithm | Code symbol | Verdict |
|---|---|---|
| Rates | `sde_transition_rates` | MATCHES |
| OneStep | `one_step_sde` | MATCHES WITH NOTE (optional `wrap_period`) |
| InfSwap | `inf_swap_rate` | MATCHES |
| EventRates | `event_rates` (+ helpers) | MATCHES WITH NOTE (extra `h_step` in tuple) |
| KillClone | `killing_cloning` | MATCHES |
| InfSwapFlemingViot | `fleming_viot` | MATCHES WITH NOTE (top-vs-bottom recompute; extra `all_rho`) |

---

## 1. Rates → `sde_transition_rates` (sde_simulation.py:6–25)

**Verdict: MATCHES**

Standard branch:
- Line 18: `rates_plus = (h * beval + aeval) / (2 * h**2)` ↔ pseudocode `r_plus[i] = (h*b_i + a_ii)/(2 h^2)`. Match (vectorized over `i`).
- Line 19: `rates_minus = (-h * beval + aeval) / (2 * h**2)` ↔ `r_minus[i] = (-h*b_i + a_ii)/(2 h^2)`. Match.

Negative-rate fallback:
- Line 20: `if np.any(rates_plus < 0) or np.any(rates_minus < 0):` ↔ `if any r_plus[i] < 0 or r_minus[i] < 0`. Match.
- Line 21: `rates_plus = (h * np.maximum(beval,0) + aeval/2) / (h**2)` ↔ `(h*max(b_i,0) + a_ii/2)/h^2`. Match — note the `1/h^2` denominator (no factor 2) and the `a_ii/2` numerator, both correct.
- Line 22: `rates_minus = (h * np.maximum(-beval,0) + aeval/2) / (h**2)` ↔ `(h*max(-b_i,0) + a_ii/2)/h^2`. Match.

Concatenation order:
- Line 24: `rates = np.append(rates_plus, rates_minus)` ↔ `concatenate(r_plus, r_minus)`, first `d` are +jumps. Match. The docstring (line 15) also documents this ordering.

---

## 2. OneStep → `one_step_sde` (sde_simulation.py:27–54)

**Verdict: MATCHES WITH NOTE**

- Lines 41–42: `total_rate = np.sum(rates); p = rates / total_rate` ↔ `r_total = sum(r); p = r/r_total`. Match.
- Line 43: `jump_direction = np.random.choice(np.arange(np.size(p)), p=p)` ↔ sample `j` from `{1..2d}` w.p. `p`. Match (0-indexed: `j ∈ {0..2d-1}`).
- Line 40: `d = len(x)`. Index split:
  - Lines 45–46: `if jump_direction < d: new_position[jump_direction] += h` ↔ pseudocode `if j <= d: x_new[j] = x[j] + h`. Match. In 1-indexed pseudocode `j <= d` ⇔ in 0-indexed `j < d`; `+h` applied to coordinate `j`. Correct.
  - Lines 47–48: `else: new_position[jump_direction - d] -= h` ↔ `else: x_new[j-d] = x[j-d] - h`. Match (`-h`, coordinate `j-d`).
- Line 44: `new_position = x.copy()` ensures `x` is not mutated. Consistent with returning a new state.

**Note (intentional, not a bug):** Lines 51–52 apply optional periodic wrapping `np.mod(new_position + wrap_period/2, wrap_period) - wrap_period/2` when `wrap_period is not None`. This is the pre-approved periodic-boundary extension; the pseudocode has no such argument. With `wrap_period=None` (default) the branch is skipped and behavior is identical to the spec.

---

## 3. InfSwap → `inf_swap_rate` (fleming_viot.py:7–20)

**Verdict: MATCHES**

- Lines 16–17: `Vx = V(x); Vy = V(y)`.
- Line 19: `rho_eps = 1 / (np.exp(2 *(Vx - Vy)/epsilon) + 1)` ↔ pseudocode `rho = 1 / (exp(2(V(x)-V(y))/eps) + 1)`. Exact match.

**Note (intentional):** If `V` returns shape `(1,)` in 1D, `rho_eps` carries a trailing singleton dim. This is the pre-approved shape note; arithmetic is unaffected.

---

## 4. EventRates → `event_rates` (fleming_viot.py:68–100) + helpers

**Verdict: MATCHES WITH NOTE**

`symmetrized_dynamics` (lines 23–36):
- Line 34: `bsymm = (1 - 2 * rho_eps) * DV(x)` ↔ pseudocode `b_symm = (1 - 2 rho) * DV(x)`. Match.

`symmetrized_kill_clone_rate` (lines 39–61):
- Line 57: `cx = c(x)`.
- Line 58: `r_kf = rhoeps * cx` ↔ `r_kf = rho * c(x)`. Match.
- Line 59: `r_kb = (1 - rhoeps) * (cx - D2V(x))` ↔ `r_kb = (1 - rho) * (c(x) - D2V(x))`. Match. (`D2V` is the Laplacian / "Delta V" as specified.)

`event_rates` (lines 68–100):
- Line 90: `bsymm = symmetrized_dynamics(x, rhoeps, DV)`.
- Line 91: `aeval = a(x)`.
- Line 92: `dynamics_rates = sde_transition_rates(x, bsymm, aeval, h_step)` ↔ `r_dyn = Rates(x, b_symm, a(x), h)`. Match.
- Line 95: `r_kf, r_kb = symmetrized_kill_clone_rate(x, rhoeps, D2V, c)`. Match.
- Line 98: `net_event_rate = np.sum(dynamics_rates) + abs(r_kf) + abs(r_kb)` ↔ `r_net = sum(r_dyn) + |r_kf| + |r_kb|`. Match.
- Line 100: `return net_event_rate, dynamics_rates, r_kf, r_kb, h_step`. Tuple order is `(r_net, r_dyn, r_kf, r_kb)` exactly as the pseudocode, with a trailing `h_step`.

**Note (intentional):** Line 87 draws a per-event randomized step `h_step = np.random.uniform(h/2, 3*h/2)` which is used both for the rates and returned so `OneStep` reuses the same `h`. The pseudocode treats `h` as a fixed argument. This randomized-lattice step is consistent with the rest of the codebase (`pure_jump_approx_diffusion` line 81 does the same) and only affects the step magnitude, not the algorithm structure. Returning `h_step` as a 5th tuple element is the mechanism by which `fleming_viot` feeds the *same* `h` into `OneStep` — necessary and correct.

---

## 5. KillClone → `killing_cloning` (fleming_viot.py:108–145)

**Verdict: MATCHES**

Signature note: `rate` is the SCALAR channel rate (`r_kf` or `r_kb`), and `rho_all` is the swap-weight array of the channel's marginal — exactly as the pseudocode's scalar `c` and `rho` array.

- Line 124: `Nparticles = x_allparticles.shape[0]`.
- Line 125: `xnow = x_allparticles[index,:].copy()` ↔ `x^i` snapshot.
- Line 126: `newindex = np.random.choice(Nparticles)` ↔ `i' = random choice from {1..N}`. Match.

Self-selection no-op:
- Lines 128–129: `if newindex == index: index_to_recompute = np.array([index])` ↔ pseudocode `if i' == i: x^i = x^{i'} (no-op); I = {i'}`. Match. The assignment is a true no-op (would copy `x^i` onto itself), so it is correctly omitted; `I = {i}` here equals `{i'}` since `i'==i`. Match.

Kill (`rate > 0`):
- Lines 131–136:
  - `if np.random.rand() < rho_all[newindex]: x_allparticles[index,:] = x_allparticles[newindex,:].copy()` ↔ `x^i = x^{i'}` w.p. `rho^{i'}`.
  - `else: x_allparticles[index,:] = y_allparticles[newindex,:].copy()` ↔ `x^i = y^{i'}`.
  - `index_to_recompute = np.array([index])` ↔ `I = {i}`. Match. Respawns active particle from the donor's marginal.

Clone (`rate <= 0`, i.e. `c < 0` branch; `c == 0` cannot reach here because the channel only fires when `|rate| > 0`):
- Lines 138–143:
  - `if np.random.rand() < rho_all[newindex]: x_allparticles[newindex,:] = xnow` ↔ `x^{i'} = x^i` w.p. `rho^{i'}`.
  - `else: y_allparticles[newindex,:] = xnow` ↔ `y^{i'} = x^i`.
  - `index_to_recompute = np.array([index, newindex])` ↔ `I = {i, i'}`. Match. Writes active particle onto the donor index.

- Line 145: returns `(x_allparticles, y_allparticles, index_to_recompute)` ↔ `({x^j}, {y^j}, I)`. Match.

---

## 6. InfSwapFlemingViot → `fleming_viot` (fleming_viot.py:151–295)

**Verdict: MATCHES WITH NOTE**

### Initialization (lines 173–207)
- Line 184: `rho_particles = [inf_swap_rate(x^i, y^i, V, eps) for i]` ↔ initial `rho^i`. Match.
- Line 189: `ratesx[i] = event_rates(x^i, rho^i, ...)` ↔ EventRates for x-marginal with `rho`. Match.
- Line 190: `ratesy[i] = event_rates(y^i, 1-rho^i, ...)` ↔ EventRates for y-marginal with `1-rho`. Match (the y-marginal uses `1-rho` exactly as the spec).
- Lines 191–192: `clocksx[i] = Exponential(1/ratesx[i,0])`, `clocksy` similarly ↔ `tau^{x,i} = Exponential(1/r_net^{x,i})`. Match (np.random.exponential takes the scale = mean = `1/r_net`).
- Lines 194–207 unpack the rate tuples into per-component arrays.

### Argmin / clock handling (lines 212–219)
- Line 212: `index = np.argmin(np.append(clocksx, clocksy))` ↔ `i' = argmin over all tau^{x,j}, tau^{y,j}` on the concatenated array (x first, then y). Match.
- Line 213: `elapsed_time = np.min(np.append(clocksx, clocksy))` ↔ `t_delta`. Match.
- Line 214: `current_time += elapsed_time` ↔ `t_now += t_delta`. Match.
- Lines 218–219: `clocksx -= elapsed_time; clocksy -= elapsed_time` ↔ subtract `t_delta` from all clocks. Match.

### x-event vs y-event selection (lines 222–239)
- x-event (`index < Nparticles`, lines 222–230): `particles1 = x`, `particles2 = y`, rates taken from `*_x[index]`, `rho_particles1 = rho_particles`. ↔ pseudocode `p1={x}, p2={y}, rho_1={rho^j}`. Match.
- y-event (`else`, lines 231–239): `particles1 = y`, `particles2 = x`, rates from `*_y[index - Nparticles]`, `rho_particles1 = 1 - rho_particles`. ↔ `p1={y}, p2={x}, rho_1={1-rho^j}`. Match. The `index - Nparticles` offset correctly maps the concatenated clock index back to the particle index.

### Active index (line 241)
- `transition_index = np.mod(index, Nparticles)` ↔ pseudocode `i = i' % N` (0-indexed). Match.

### 3-way event-type branch (lines 244–252)
- Line 244: `a_draw = np.random.rand()` ↔ `a_draw = U(0,1)`.
- Line 245: `dyn_frac = np.sum(dynamics_rate) / net_event_rate` ↔ `sum(r_dyn)/r_net`.
- Line 246–248 (dynamics): `if a_draw < dyn_frac: particles1[transition_index] = one_step_sde(particles1[transition_index], dynamics_rate, h_step, wrap_period); index_to_recompute = [transition_index]` ↔ `p1^i = OneStep(p1^i, r_dyn, h); I = {i}`. Match. (`h_step` here is the same draw used to compute `dynamics_rate`, see §4 note.)
- Line 249–250 (forward channel): `elif a_draw < dyn_frac + abs(rkf)/net_event_rate: killing_cloning(particles1, particles2, rho_particles1, transition_index, rkf)` ↔ threshold `(sum(r_dyn)+|r_kf|)/r_net`, passes `rho_1` and scalar `r_kf`. Match — forward channel passes `rho_1` (not `1-rho_1`) and `r_kf`.
- Line 251–252 (backward channel): `else: killing_cloning(particles1, particles2, 1-rho_particles1, transition_index, rkb)` ↔ passes `1 - rho_1` and scalar `r_kb`. Match — backward channel passes `1-rho_1` and `r_kb`.

This is the subtlest part of the spec and it is implemented correctly: forward uses `rho_1`, backward uses `1-rho_1`; forward uses `r_kf`, backward uses `r_kb`.

### Post-event array reassignment (lines 255–260)
- x-event: `x_allparticles = particles1; y_allparticles = particles2` ↔ `{x_m} = p1; {y_m} = p2`. Match.
- y-event: `y_allparticles = particles1; x_allparticles = particles2` ↔ `{y_m} = p1; {x_m} = p2`. Match. The p1/p2 swap on the y-branch is handled correctly (p1 is the y-array in that branch, so it is written back to y).

### Recompute only for changed indices (lines 263–272)
- `for i in index_to_recompute:` recomputes `rho_particles[i]`, `ratesx[i]`, `ratesy[i]`, and draws fresh `clocksx[i]`, `clocksy[i]`. ↔ pseudocode recomputes `rho^i`, EventRates, and `tau` for `i in I`. Match.

**NOTE — top-of-loop vs bottom-of-loop recompute (confirmed equivalent):** The pseudocode recomputes rho/rates/clocks for `I` at the *top* of each iteration (the `for i in I` block before selecting the event). The code instead computes the initial rates once before the loop (lines 184–192) and then recomputes only the changed indices at the *bottom* of the loop (lines 263–272). These are functionally equivalent: in both schemes, at the start of any given event selection, every index has rates/clocks that were last refreshed exactly when that index was in the change set `I` of the previous event (or at initialization). The fresh-clock set, fresh-rate set, and the residual (subtracted) clocks for unchanged indices are identical between the two orderings. No behavioral difference.

Also note the clock semantics are consistent: changed indices get brand-new `Exponential` draws (lines 271–272), unchanged indices keep their residual clocks after the global subtraction (lines 218–219) — matching the pseudocode's "only indices in I get fresh draws; all clocks lose `t_delta`".

### Bottom-of-loop bookkeeping (lines 275–293)
- Lines 275–288 re-unpack the (now partially updated) rate arrays for the next iteration. Correct.
- Lines 290–293 append snapshots of `x`, `y`, `time`, `rho`. (List accumulation, intentional.)

### Return (line 295)
- `return np.array(all_positionsx), np.array(all_positionsy), np.array(all_time), np.array(all_rho)` ↔ pseudocode returns `{x_m}, {y_m}, t_all`.

**Notes (all intentional, not bugs):**
- The function additionally returns `all_rho` (4th element); the pseudocode omits it. Pre-approved.
- Trajectories are accumulated in Python lists and stacked once at the end (lines 179–181, 290–293, 295) rather than vstack-per-event. Pre-approved, behavior-preserving (and an O(n) vs O(n²) improvement).
- Optional `wrap_period` is threaded into `one_step_sde` at line 247. Pre-approved periodic-boundary extension; inert when `None`.
- Shapes may carry a trailing singleton dim because `V` returns shape `(1,)` in 1D. Pre-approved.

---

## 7. Final list of discrepancies

**None.**

All algorithms match the pseudocode. The only deviations are the pre-declared intentional ones:
1. Optional `wrap_period` periodic-boundary wrapping (`one_step_sde`, `fleming_viot`).
2. List accumulation of trajectories instead of `vstack` per event.
3. `fleming_viot` also returns `all_rho`.
4. `event_rates` draws a randomized lattice step `h_step ∈ U(h/2, 3h/2)` and returns it as a 5th tuple element so `one_step_sde` reuses the same `h` (consistent with `pure_jump_approx_diffusion`).
5. Trailing singleton dimensions from `V` returning shape `(1,)` in 1D.
6. Rates/clocks are recomputed at the bottom of the loop for the changed index set rather than at the top for `I` — verified functionally equivalent.
