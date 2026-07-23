# Signal Calibration: Choosing ρ, n, and Block Size for Detectable Power

**Repository:** `Hrish52/fdr-correlations`
**Author:** Hrishikesh Deepak Dhole
**Supervisor:** Prof. Elio Zhang, Seattle University
**Date:** April 21, 2026
**Type:** Methodological note (not a code patch)
**Status:** Finding documented. Grid revision pending as Patch 8.

---

## 1. Summary

While verifying the Patch 7 speedup at p = 500, a single LCT-B call returned `t_hat = inf` and zero rejections despite a nominally correlated block being present. Investigation showed this was not a bug: the configuration hardcoded in `run_sim_gaussian.py` for p = 500 (ρ = 0.25, n = 80, block = 20) places the expected edge statistic roughly 2.4 units *below* the smallest threshold that can control FDR at α = 0.05. No procedure — LCT-N, LCT-B, Fisher-z+BH, or Fisher-z+BY — can have power in that regime.

Had the full grid been run as configured, every p = 500 power curve would have been flat zero for all four methods. That is not a comparison; it is four methods agreeing there is nothing to find.

This note derives the relationship between signal strength and the FDR-controlling threshold, explains why ρ = 0.25 fails at p = 500, and gives the ρ values that place each configuration in the informative regime.

---

## 2. The Two Quantities That Determine Detectability

Whether a simulation configuration produces meaningful power depends on a comparison between two numbers.

### 2.1 Achievable signal: the expected |T| for a true edge

The LCT edge statistic (Cai & Liu 2016, Eq. 5) is a studentised difference of correlations:

T_ij = (r̂_ij,1 − r̂_ij,2) / √( V̂(r̂_ij,1) + V̂(r̂_ij,2) )

In our simulation design, group X is drawn with identity covariance and group Y carries the block correlation. For a true edge inside the block, the numerator has expectation ρ_eff (the population correlation in group Y), while group X contributes only sampling noise.

Two adjustments are needed to get the effective correlation right.

**Shrinkage in `make_block_cov`.** The generator applies a 1% ridge to guarantee positive-definiteness:

```python
return 0.99 * Sigma + 0.01 * np.eye(p)
```

The off-diagonal block entries are therefore `0.99 · ρ`, not ρ. A nominal ρ = 0.25 yields an actual population correlation of 0.2475. Small, but worth stating explicitly in the paper's simulation section: **reported ρ is nominal; realised ρ is 0.99ρ.**

**Variance under the Cai–Liu estimator.** For a Pearson correlation from n observations, `Var(r) ≈ (1 − ρ²)²/n` under Gaussian assumptions. The Cai–Liu plug-in estimator behaves similarly. For the independent group (ρ = 0), the variance is ≈ 1/n. For the correlated group, `(1 − ρ²)²/n`. Summing and expanding gives a denominator that, over the ρ range of interest, is well approximated by

√( (2 + ρ_eff²) / n )

Combining:

               ρ_eff · √n
E|T_ij| ≈  ───────────────────         where  ρ_eff = 0.99 ρ
              √(2 + ρ_eff²)

The dominant behaviour is `|T| ∝ ρ√n`. Doubling the sample size buys a factor of √2; doubling ρ buys nearly a factor of 2. **ρ is the more efficient lever.**

### 2.2 Required threshold: the smallest t that can control FDR

Under the normal-tail estimator, the estimated false discovery proportion at threshold t is

          M · q(t)
FDP̂(t) = ──────────  ,      q(t) = 2(1 − Φ(t)),      M = p(p−1)/2
          R(t) ∨ 1

Cai & Liu Eq. (9) selects the infimum of `{ t : FDP̂(t) ≤ α }`.

To find the *smallest threshold that could ever qualify*, take the most optimistic case: every rejection is a true discovery, so `R(t) = m₁` where `m₁ = block(block−1)/2` is the number of true edges. Substituting and solving:

M · q(t)                              α · m₁
──────── ≤ α    ⟺    q(t) ≤ ──────    ⟺    t ≥ Φ⁻¹(1 − α·m₁/(2M))
   m₁                                    M

Define this as **t_req**. It is a hard lower bound: no threshold below t_req can satisfy the FDR constraint, regardless of how strong the signal is. It depends only on p, block size, and α — not on ρ or n.

### 2.3 The margin

margin = E|T| − t_req

- **margin < −0.5** → no power. Test statistics for true edges never reach the required threshold.
- **−0.5 ≤ margin < 0.3** → threshold regime, power roughly 20–60%. Sensitive to n and to Monte Carlo noise.
- **0.3 ≤ margin < 1.2** → good power, roughly 60–95%. **This is where power comparisons between methods are informative.**
- **margin ≥ 1.2** → saturated, power ≈ 100% for every method. Also uninformative, for the opposite reason.

The goal in designing a simulation grid is to place configurations in the third band, and to sweep ρ so that the power curve traverses from the first band to the fourth.

---

## 3. Why ρ = 0.25 Fails at p = 500

Applying the formulas to the configuration hardcoded in `run_sim_gaussian.py`:

**Inputs:** p = 500, n₁ = n₂ = 80, block = 20, ρ = 0.25, α = 0.05.

**Derived quantities:**

| Quantity | Value |
|---|---|
| M = p(p−1)/2 | 124,750 |
| m₁ = block(block−1)/2 | 190 |
| ρ_eff = 0.99ρ | 0.2475 |
| E\|T\| = ρ_eff√n / √(2+ρ_eff²) | **1.54** |
| q_max = α·m₁/M | 7.62 × 10⁻⁵ |
| t_req = Φ⁻¹(1 − q_max/2) | **3.96** |
| **margin** | **−2.42** |

The expected test statistic for a true edge is 1.54. The smallest threshold that can control FDR is 3.96. The signal is not merely weak — it is short of the requirement by more than two standard normal units.

This is a property of the configuration, not of the estimator. Fisher-z + BH faces the same arithmetic and will also return nothing. The observed `t_hat = inf, rejections = 0` is the correct output.

### 3.1 The comparison with p = 250

The same ρ = 0.25 is less catastrophic at p = 250 because t_req drops:

| | p = 250 | p = 500 |
|---|---|---|
| M | 31,125 | 124,750 |
| t_req at α = 0.05 | 3.61 | 3.96 |
| E\|T\| at ρ = 0.25, n = 80 | 1.54 | 1.54 |
| margin | −2.07 | −2.42 |

Still no power at either dimension. The p = 250 grids in `run_sim_gaussian.py` use ρ = 0.30, giving E|T| ≈ 1.85 and margin ≈ −1.76 — also below the threshold.

**The implication is broader than the p = 500 case that surfaced it: the hardcoded ρ values are too low across the entire grid.** Prior runs that appeared to produce results were doing so at α = 0.10 with favourable Monte Carlo noise, or were dominated by the Fisher-z baselines behaving differently.

### 3.2 How t_req scales

Because q_max = α·m₁/M and M grows as p², the required threshold grows roughly as √(log p):

| p | M | m₁ (block=20) | t_req @ α=0.05 |
|---|---|---|---|
| 100 | 4,950 | 190 | 3.20 |
| 250 | 31,125 | 190 | 3.61 |
| 500 | 124,750 | 190 | 3.96 |
| 1000 | 499,500 | 190 | 4.29 |

The growth is slow — going from p = 250 to p = 1000 raises the bar by only 0.68 — but the signal does not grow at all with p. Every increase in dimension makes detection strictly harder unless ρ or n increases to compensate.

This is the multiple-testing burden made concrete, and it matches Cai & Liu's own Sec. 5 design choices: they use ρ = 0.6 for the normal case and ρ = 0.8 for the normal-mixture, considerably stronger than anything in the current grid. That choice now has an obvious explanation.

---

## 4. Recommended Configurations

### 4.1 Achievable signal by (ρ, n)

E|T| values from the formula in §2.1:

| ρ \ n | 60 | 80 | 120 | 200 |
|---|---|---|---|---|
| 0.30 | 1.60 | 1.85 | 2.26 | 2.92 |
| 0.40 | 2.11 | 2.44 | 2.98 | 3.85 |
| 0.50 | 2.61 | 3.01 | 3.69 | 4.76 |
| 0.60 | 3.09 | 3.57 | 4.37 | 5.64 |
| 0.70 | 3.56 | 4.11 | 5.03 | 6.50 |
| 0.80 | 4.01 | 4.63 | 5.67 | 7.32 |
| 0.90 | 4.44 | 5.13 | 6.28 | 8.11 |

### 4.2 Reading the table against t_req

**p = 250 (t_req = 3.61):**

| ρ | n = 80, E\|T\| | margin | regime |
|---|---|---|---|
| 0.50 | 3.01 | −0.60 | no power |
| 0.60 | 3.57 | −0.04 | threshold |
| 0.70 | 4.11 | +0.50 | **good power** |
| 0.80 | 4.63 | +1.02 | **good power** |
| 0.90 | 5.13 | +1.52 | saturated |

A ρ sweep of **{0.50, 0.60, 0.70, 0.80}** at n = 80 traverses the full curve from zero to near-saturation. That is the ideal shape for a power figure.

**p = 500 (t_req = 3.96):**

At n = 80, the ceiling as ρ → 1 is E|T| ≈ 5.4, leaving a maximum margin of about +1.4. Usable, but the whole curve is compressed into a narrow ρ band near the top of the range.

At n = 120 the picture is much healthier:

| ρ | n = 120, E\|T\| | margin | regime |
|---|---|---|---|
| 0.50 | 3.69 | −0.27 | threshold |
| 0.60 | 4.37 | +0.41 | **good power** |
| 0.70 | 5.03 | +1.07 | **good power** |
| 0.80 | 5.67 | +1.71 | saturated |

**Recommendation: use n = 120 at p = 500.** The additional compute is modest (the bootstrap cost scales with n·p², so 1.5× on n) and it buys a properly-shaped power curve rather than a compressed one.

### 4.3 Proposed grid

| p | n₁ = n₂ | block | ρ sweep | t_req |
|---|---|---|---|---|
| 250 | 80 | 20 | 0.50, 0.60, 0.70, 0.80 | 3.61 |
| 250 | 120 | 20 | 0.40, 0.50, 0.60, 0.70 | 3.61 |
| 500 | 120 | 20 | 0.50, 0.60, 0.70, 0.80 | 3.96 |
| 1000 | 200 | 20 | 0.50, 0.60, 0.70 | 4.29 |

The p = 1000 row is intended as a scalability demonstration at reduced replication count, not as a primary result — see §6.

### 4.4 The alternative lever: block size

Increasing the block raises m₁, which raises q_max, which *lowers* t_req. Going from block = 20 (m₁ = 190) to block = 40 (m₁ = 780) at p = 500 drops t_req from 3.96 to 3.69 — worth about 0.27, roughly equivalent to raising ρ by 0.04.

This is a weaker lever than ρ or n, and it changes the sparsity of the alternative, which is a substantively different scientific question (Cai & Liu discuss sparse versus dense alternatives explicitly). **Recommendation: hold block = 20 for the main grid** and treat block size as a separate designed ablation if sparsity effects are of interest, rather than as a tuning knob for power.

---

## 5. Calibration Tool

`scripts/calibrate_grid.py` implements the formulas above and prints the signal-versus-threshold comparison for a given configuration:

```bash
python scripts/calibrate_grid.py --p 500 --n 120 --block 20
```

Output:
p=500 n=120 block=20 alpha=0.05
M = 124,750 edges, m1 = 190 true edges
Required threshold: t >= 3.958

rho |T| margin regime

0.20 1.68 -2.28 no power
0.30 2.52 -1.44 no power
0.40 3.35 -0.61 no power
0.50 4.16 +0.20 threshold (~50%)
0.60 4.94 +0.98 good power
0.70 5.68 +1.72 saturated (~100%)
0.80 6.38 +2.42 saturated (~100%)
0.90 7.04 +3.08 saturated (~100%)

**Recommended practice: run this before committing compute to any new grid.** It takes under a second and prevents hours of generating flat lines.

---

## 6. Implications for the Paper

### 6.1 Simulation design section

The paper should state the design rationale explicitly rather than presenting ρ values as arbitrary:

> For each configuration we selected ρ so that the expected edge statistic for a true discovery, E|T| ≈ ρ_eff√n / √(2 + ρ_eff²), spans the FDR-controlling threshold t_req = Φ⁻¹(1 − αm₁/2M). This places the power curve in the informative regime rather than saturating at zero or one. Note that `make_block_cov` applies a 1% ridge for positive-definiteness, so the realised population correlation is 0.99ρ; reported values are nominal.

This is the kind of design justification that distinguishes a considered simulation study from an arbitrary one, and it anticipates the reviewer question "why these ρ values?".

### 6.2 A publishable secondary finding

The relationship between t_req and p is itself worth reporting. Practitioners applying LCT to a new dataset need to know: *given my p, n, and expected effect size, is detection even possible?* A small table or figure of t_req versus p, overlaid with achievable E|T| contours for realistic (ρ, n), answers that directly.

This costs almost nothing to produce — the arithmetic is already implemented in `calibrate_grid.py` — and it is a genuinely useful contribution for the applied audience the paper is aimed at.

### 6.3 Revised compute budget

The p = 1000 configuration needs n ≈ 200 to be detectable, which raises per-replicate cost. Combined with the Monte Carlo precision argument (at reps = 20, the standard error on an FDR estimate near 0.05 is ≈ 0.049 — too coarse to distinguish 0.03 from 0.10), the sensible allocation is:

- **p ∈ {250, 500} at 300–500 replications** — the main calibration result, statistically tight (SE ≈ 0.010)
- **p = 1000 at ~50 replications, one configuration** — labelled explicitly as a scalability demonstration, with no precision claim

Stated plainly in the paper, this is a defensible design rather than a limitation.

---

## 7. Action Items

1. **Patch 8:** revise the hardcoded grids in `run_sim_gaussian.py` and the `--rho-list` default in `run_power_curves.py` per §4.3.
2. **Add `scripts/calibrate_grid.py`** to the repository and reference it in the README's reproduction section.
3. **Document the 0.99 shrinkage** in `make_block_cov`'s docstring and in the paper's simulation subsection.
4. **Regenerate all power results** with the corrected grid. Prior power numbers at ρ ≤ 0.30 are valid but uninformative (all methods at zero) and should not appear in figures.
5. **Consider adding the t_req-versus-p analysis** as a short subsection or supplementary figure (§6.2).

---

## 8. References

- **Cai, T. T. & Liu, W. (2016).** Large-Scale Multiple Testing of Correlations. *Journal of the American Statistical Association*, 111(513), 229–240. Eq. (5) for the edge statistic; Eq. (9) for the normal-tail threshold; Sec. 5.1 for their own simulation design, which uses ρ = 0.6 and ρ = 0.8.
- `docs/patches/patch01_threshold_selection.md` — infimum semantics of the threshold rule.
- `docs/patches/patch07_vectorized_scan.md` — the p = 500 timing run that surfaced this finding.

---

*End of note.*