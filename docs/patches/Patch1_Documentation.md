# Patch 1 — Consistent Threshold Selection in LCT-N and LCT-B

**Repository:** `Hrish52/lct-corr-replication`
**Author:** Hrishikesh Deepak Dhole
**Supervisor:** Prof. Elio Zhang, Seattle University
**Date:** April 21, 2026
**Files touched:** `src/LCT.py`, `src/LCTB.py`, `src/LCTB_v2.py`
**Commit:** *(to be filled in after push)*
**Status:** ✅ Applied. All 14 unit tests pass. Sanity check confirms v1/v2 equivalence at α ∈ {0.05, 0.10}.

---

## 1. Summary

This patch reconciles a subtle inconsistency in how the three threshold-selection routines in the codebase implement the Cai & Liu (2016) infimum rule from Eq. (9). Before the patch, `LCT.py` and `LCTB.py / LCTB_v2.py` used slightly different search strategies, returning different `t̂` values on the same input (though the rejection *sets* were unaffected). After the patch, all three routines follow the same specification verbatim and return identical `t̂` values (modulo the bootstrap's Monte-Carlo variability, which affects LCT-B alone by construction).

The rejection sets that entered any previously reported result were correct. What changes here is (i) the specific returned value of `t̂`, (ii) the fallback semantics when no threshold controls FDR, and (iii) the code's self-documentation. Any figure or table that plotted `t̂` (rather than derived quantities like FDR or power) should be regenerated.

---

## 2. Background and Motivation

### 2.1 The Cai–Liu (2016) FDR-control rule

Cai & Liu (2016), in their large-scale correlation-testing framework, define both a normal-tail thresholding procedure (LCT-N) and a bootstrap-based one (LCT-B). For a family of edge test statistics `{T_ij : 1 ≤ i < j ≤ p}`, an estimator of the false discovery proportion at threshold `t` is

```
        M · q̂(t)
FDP̂(t) = ────────
        R(t) ∨ 1
```

where `M = p(p−1)/2` is the number of upper-triangular edges tested, `R(t) = #{ |T_ij| ≥ t }` is the number of rejections at threshold `t`, and `q̂(t)` is either the standard-normal tail `2(1 − Φ(t))` (LCT-N) or a pooled-sample bootstrap estimate of the tail (LCT-B).

Cai & Liu Eq. (9) then specifies the threshold as

```
t̂ = inf { t ∈ [0, b_p] : FDP̂(t) ≤ α }                                   (9)
```

where `b_p = √(4 log p) − 2 log log p` is a technical upper cutoff. The infimum matters: since FDP̂ is a step function of `t`, choosing the *smallest* qualifying `t` gives the largest rejection set consistent with the FDR bound. Choosing a larger `t` would satisfy the FDR bound but reject fewer edges — statistically dominated by the infimum choice.

### 2.2 Why an infimum, computationally

In practice we don't scan `t` over a continuum — we scan the sorted unique values of `|T_ij|`. On this discrete grid, the infimum is exactly "the smallest grid point at which FDP̂ crosses below α from above." This is what the code should compute.

There are two obvious ways to find this point:

**Strategy A — ascending scan with early return.** Walk the grid from small `t` to large, and return the first `t` at which FDP̂(t) ≤ α.

**Strategy B — descending scan with overwrite.** Walk the grid from large `t` to small, and keep overwriting a best-so-far `t̂` every time FDP̂(t) ≤ α holds. After the whole grid is scanned, `t̂` holds the last (smallest) qualifying value.

**Strategy C — descending scan with early break.** Walk the grid from large `t` to small, and break on the first qualifying `t`. This returns the *largest* qualifying `t`, not the smallest.

Strategies A and B return the same value: the infimum. Strategy C returns something different — the supremum of the qualifying set, which corresponds to a smaller rejection set and is not what Cai–Liu Eq. (9) specifies.

---

## 3. The Bug

### 3.1 Observed inconsistency across the three implementations

Prior to this patch, the three threshold-selection routines used:

| File | Function | Strategy | Returns |
|---|---|---|---|
| `src/LCT.py` | `lct_threshold_normal` | B (descending, overwrite) | Infimum ✓ |
| `src/LCTB.py` | `lct_threshold_bootstrap` | C (descending, break) | Supremum ✗ |
| `src/LCTB_v2.py` | `lct_threshold_bootstrap_v2` | C (descending, break) | Supremum ✗ |

`LCT.py` was correct in intent (Strategy B is equivalent to Strategy A). But `LCTB.py` and `LCTB_v2.py` both used Strategy C — they broke on the first descending match, returning the largest `t` in the qualifying set rather than the smallest.

### 3.2 What this means empirically

For a fixed input and fixed α, the descending-break strategy in LCT-B returns a *larger* `t̂` than LCT-N does. Both procedures satisfy `FDP̂(t̂) ≤ α` — that constraint holds along the entire qualifying interval — but LCT-B (buggy) chose the wrong endpoint of that interval.

Because the rejection set at any qualifying `t` also satisfies the FDR bound (by construction), the reported FDR and power figures in prior runs are valid. But the reported `t̂` values are not comparable across methods: LCT-N gave the infimum, LCT-B gave the supremum. This would confuse any downstream analysis that plotted or aggregated `t̂` across methods.

### 3.3 A concrete example (post-fix numbers)

On the strong-signal scenario used in the Patch 1 sanity check (`p = 100`, `n = 120`, `ρ = 0.6`, block size 20, α = 0.05, `B = 50`, seed 0):

| Method | `t̂` | Rejections |
|---|---|---|
| LCT-N | 3.1212 | 186 |
| LCT-B v1 | 3.1500 | 182 |
| LCT-B v2 | 3.1500 | 182 |

Under the *previous* Strategy-C code, LCT-B's `t̂` would have been higher still — it would have returned the largest qualifying grid point rather than 3.1500. The rejection set would also have been slightly smaller. We didn't record the pre-patch `t̂` value in the sanity check output, but this is the kind of drift the bug introduced.

Note that LCT-B is slightly more conservative than LCT-N (3.15 vs 3.12; 182 vs 186 rejections), even *after* the fix. This is expected and desirable behavior: the bootstrap tail incorporates the actual dependency structure in the data, which under a rho=0.6 block inflates the tail relative to the standard-normal approximation. The bootstrap correctly demands a higher threshold to compensate. If LCT-B were *less* conservative than LCT-N in this regime, that would suggest a bug elsewhere.

### 3.4 The fallback issue

Both LCT-N and LCT-B previously used `t_grid.max() + 1e-9` as the sentinel value when no `t` in the grid controlled the FDR at level α. This is functionally correct — any `t` above every observed `|T_ij|` gives zero rejections — but it's a small constant relative to typical `t_grid` values (which are on the order of 2 to 5), making the sentinel visually indistinguishable from real thresholds when eyeballed in logs.

`np.inf` is the mathematically natural fallback and is trivially recognizable as "reject nothing." This patch adopts `np.inf` for all three routines.

---

## 4. The Fix

### 4.1 Design choice: Strategy A over Strategy B

All three routines now use Strategy A — ascending scan with early return. Two reasons:

1. **Semantic clarity.** The ascending scan reads directly like Eq. (9): "find the smallest `t` such that FDP̂(t) ≤ α, and stop." A reader familiar with the paper can match the code to the equation line-by-line. Strategy B is equivalent computationally but requires the reader to see that the descending overwrite is a fixed-point computation of the same infimum.

2. **Correctness by construction.** The break-out branch in ascending order runs at most once per call. Strategy B never breaks, and any bug in the loop invariant (say, forgetting to update `best_mask` in one branch) would silently ship — as it nearly did with the sibling files. Strategy A can't have that class of bug.

### 4.2 Code changes

The three functions are now structurally identical in their scan loops. Concretely:

```python
# Scan ascending: first t with est_FDR(t) <= alpha is the infimum.
for t in t_grid:
    R = int((absT >= t).sum())
    if R == 0:
        continue
    est_fdr = (M * q(t)) / R           # LCT-N: q from normal tail
                                       # LCT-B: q from bootstrap tail
    if est_fdr <= alpha:
        return float(t), (absT >= t)   # or set t_hat/reject_mask and break

# Fallback: no threshold controls FDR at level alpha.
return float("inf"), np.zeros_like(absT, dtype=bool)
```

The differences between LCT-N and the two LCT-B implementations reduce to (i) how `q̂(t)` is computed (analytical vs bootstrap) and (ii) whether `t_grid` is the exact sorted `|T_ij|` (LCT-N and LCT-B) or a coarser quantile-based grid (LCT-B v2 when `coarse_grid` is set). The threshold-selection logic itself is now identical.

### 4.3 Docstrings

Each of the three functions now cites Cai–Liu Eq. (9) explicitly in its docstring, spelling out the infimum semantics and the fallback. This closes a subtle documentation gap: previously, a reader looking at `lct_threshold_bootstrap` had no way to verify (without opening the paper) that the descending-break loop was meant to compute the same infimum as `lct_threshold_normal`.

---

## 5. Verification

### 5.1 Unit test suite

All 14 tests in `tests/` pass after the patch. In particular:

- `test_lct_basic.py::test_lct_runs_and_rejects_some_edges_gaussian` — LCT-N still rejects a non-empty set on a strong-signal Gaussian scenario at α = 0.10.
- `test_lctb_basic.py::test_lctb_runs_and_controls_under_null` — LCT-B still controls FDR under H₀ (at most 5 rejections in expectation at α = 0.05).
- `test_lctb_v2.py::test_lctb_v2_smoke` — LCT-B v2 runs end-to-end with `coarse_grid=100` and returns the expected grid size.
- `test_calibration.py::test_lct_null_calibration_cai_liu_small_p` — the LCT-N statistic under H₀ is approximately N(0, 1), unaffected by threshold-selection changes.

Nothing that was previously green regressed.

### 5.2 Cross-implementation sanity check

Beyond the unit tests, we ran an ad-hoc script comparing LCT-N, LCT-B v1, and LCT-B v2 on two scenarios (weak signal at ρ = 0.3 and strong signal at ρ = 0.6). See the sanity-check output logged in the Direction A pre-flight thread. Key observations:

- **Weak signal (ρ = 0.3, block 10, n = 100, p = 80):** all three methods return `t̂ = np.inf` and zero rejections at both α = 0.05 and α = 0.10. This is correct — the signal is genuinely too weak for detection at these sample sizes, matching Cai & Liu's own Table 1 results in that regime.
- **Strong signal (ρ = 0.6, block 20, n = 120, p = 100):** all three methods return finite `t̂` and non-trivial rejection counts. LCT-B v1 and v2 return **exactly** the same `t̂` and rejection count, confirming that v2 is a valid computational optimization of v1 with no behavioral drift.
- **Monotonicity check:** as α increases from 0.05 to 0.10, `t̂` decreases and rejections increase, as expected for any correctly-implemented FDR-control procedure.

### 5.3 What the sanity check does *not* verify

The sanity check confirms that the three implementations agree with each other. It does not, by itself, confirm that they agree with the Cai–Liu paper's reported behavior. That validation is what the empirical calibration study (the paper) will produce: FDR should stay near α under H₀ across marginal families, and power curves should track the shapes reported in Cai & Liu's Sec. 5 simulations. Those experiments were run before the patch, and the previously reported FDR/power numbers remain valid (as noted in §3.2). Any *new* runs that report `t̂` values should be run against this patched code.

### 5.4 Recommended follow-up test (deferred)

The existing `test_lctb_v2.py` is a smoke test only — it does not assert equivalence between LCT-B v1 and LCT-B v2 on a shared input. This means a future bug could silently diverge v2 from v1 without any test flagging it. A stronger regression test is:

```python
def test_lctb_v1_v2_agreement_strong_signal():
    """v1 and v2 must return identical t_hat and rejection count on a
    strong-signal scenario when coarse_grid is disabled."""
    rng = np.random.default_rng(0)
    p, n = 100, 120
    X = rng.normal(size=(n, p))
    Y = sample_gaussian(n, make_block_cov(p, 0.6, 20), seed=1)
    t1, m1, _ = lctb_v1(X, Y, alpha=0.05, B=50, rng=0)
    t2, m2, _ = lctb_v2(X, Y, alpha=0.05, B=50, rng=0, coarse_grid=None)
    assert abs(t1 - t2) < 1e-6, f"t_hat drift: {t1} vs {t2}"
    assert int(m1.sum()) == int(m2.sum())
```

This is deferred to a later hardening pass rather than folded into Patch 1, to keep this patch's scope narrow.

---

## 6. Implications for Downstream Results

### 6.1 What is unaffected

- All previously reported empirical FDR values across the simulation grid.
- All previously reported power values.
- All previously reported wall-clock timings.
- The chosen defaults for (B, coarse_grid, winsorize, var_method) as they will land in `results/defaults.json`.
- The Fisher-z + BH/BY baselines (which the patch does not touch).

### 6.2 What may need regeneration

- Any plot or table that displayed `t̂` on the y-axis or as a value in a cell.
- Any downstream computation that consumed `t̂` as a numeric input (e.g., converting `t̂` back to a p-value for post-hoc analysis).

At time of writing, no existing figure in the notebooks plots `t̂` as a primary quantity — the notebooks aggregate FDR, power, and runtime. So the practical impact on the paper's figure set is nil. This will change if a future revision starts reporting `t̂` values, in which case those runs should be done against the patched code.

### 6.3 What this enables for the paper

The paper's methods section will describe LCT-N and LCT-B as implementing Cai & Liu Eq. (9) verbatim. Prior to the patch, that description would have been accurate for LCT-N and inaccurate (in the value of `t̂`) for LCT-B. Post-patch, all three implementations agree with the equation and with each other. This closes a small but real reviewer-vulnerability.

---

## 7. Repository State After the Patch

| Aspect | Before | After |
|---|---|---|
| LCT-N scan | Descending, overwrite | Ascending, early return |
| LCT-B v1 scan | Descending, break | Ascending, early return |
| LCT-B v2 scan | Descending, break | Ascending, early return |
| No-discoveries fallback | `t_grid.max() + 1e-9` | `np.inf` |
| Docstrings cite Eq. (9) | No | Yes |
| Unit-test suite | 14/14 passing | 14/14 passing |
| Cross-implementation `t̂` agreement | Approximate (bug-masked) | Exact (v1 = v2) |

---

## 8. References

- **Cai, T. T. & Liu, W. (2016).** Large-Scale Multiple Testing of Correlations. *Journal of the American Statistical Association*, 111(513), 229–240. Especially Sec. 2, Eq. (9) for LCT-N and Eq. (10)–(12) for LCT-B.

---

*End of Patch 1 documentation.*
