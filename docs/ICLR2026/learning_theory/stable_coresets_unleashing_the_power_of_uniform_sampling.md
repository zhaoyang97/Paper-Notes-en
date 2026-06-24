---
title: >-
  [Paper Note] Stable Coresets: Unleashing the Power of Uniform Sampling
description: >-
  [ICLR 2026][Learning Theory][coreset] This paper introduces the new concept of "stable coresets" positioned between weak and strong coresets. It proves that **uniform sampling alone** (a sample of size $O(\epsilon^{-2}\log d)$) can construct a stable coreset for the 1-median problem under the $\ell_1$ metric. This elevates uniform sampling from a "cheap, data-oblivious, and streamable/distributed" heuristic to a tool with rigorous guarantees that transfer to all sub-metrics e…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Coreset Theory"
  - "coreset"
  - "uniform sampling"
  - "1-median"
  - "$\\ell_1$ metric"
  - "VC dimension"
date: 2026-05-08
content_hash: cb5652241ef172bd
---

# Stable Coresets: Unleashing the Power of Uniform Sampling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=sOpAa8iR0A](https://openreview.net/forum?id=sOpAa8iR0A)  
**Code**: https://github.com/amircarmel-lab/StableCoresets  
**Area**: Learning Theory / Coreset Theory  
**Keywords**: coreset, uniform sampling, 1-median, $\ell_1$ metric, VC dimension

## TL;DR
This paper introduces the new concept of "stable coresets" positioned between weak and strong coresets. It proves that **uniform sampling alone** (a sample of size $O(\epsilon^{-2}\log d)$) can construct a stable coreset for the 1-median problem under the $\ell_1$ metric. This elevates uniform sampling from a "cheap, data-oblivious, and streamable/distributed" heuristic to a tool with rigorous guarantees that transfer to all sub-metrics embeddable in $\ell_1$ (e.g., Kendall-tau, Jaccard).

## Background & Motivation

**Background**: Large-scale clustering typically adopts the "sketch-and-solve" paradigm—compressing data into a small summary (coreset) independent of the original data size, then running expensive optimization/learning algorithms on it. Literature defines two main types. **Weak coresets** only guarantee that a (near) optimal solution on the coreset is also near-optimal for the original data, without guaranteeing the objective value itself; they align with the sketch-and-solve paradigm and can **sometimes be constructed via uniform sampling**, making them fast and easy for streaming/distributed implementation. **Strong coresets** are more powerful: they require that for **every** possible center $c$ in the metric space, the cost on the coreset approximates the cost on the original data (i.e., $\mathrm{cost}(c,Q)\in(1\pm\epsilon)\,\mathrm{cost}(c,P)$). This broadens their applicability and grants the useful property of **sub-metric transferability**: if a metric space $X$ has a strong coreset, any sub-metric isometrically embedded in $X$ (like Kendall-tau in $\ell_1$) automatically has one as well.

**Limitations of Prior Work**: The strong guarantees of strong coresets come at a heavy price—construction algorithms **must read the entire dataset**. A classic counterexample involves dense clusters with a single distant outlier; while the outlier barely affects the optimal center, it contributes significantly to the **objective value**. A strong coreset must identify and include such outliers, which is impossible in sublinear time or via uniform sampling (which rare clusters/outliers almost never trigger). Conversely, while uniform sampling is cheap and data-oblivious, it only yields weak coresets, which **lack sub-metric transferability**, requiring a new analysis for every metric.

**Key Challenge**: There is a conflict between broad applicability (sub-metric transfer of strong coresets) and efficient construction (sublinear, data-oblivious nature of uniform sampling). Strong coresets require full data passes, while weak coresets cannot be transferred to sub-metrics.

**Goal**: The paper aims to find a "sweet spot" concept that retains the sub-metric transferability of strong coresets but can be constructed via the cheapest possible method: uniform sampling. The focus is initially on the cleanest $k=1$ (1-median) case, as uniform sampling succeeds there without additional data assumptions and serves as a building block for general $k$-median.

**Key Insight**: The authors observe that strong coresets are expensive because they must **preserve the absolute cost** of every center. However, many downstream applications (finding optimal solutions, approximation algorithms, adding fairness constraints) only **need to preserve the relative order of costs** between different centers, not the absolute values. This observation is key to why uniform sampling works—outliers shift absolute costs but do not necessarily change the ranking of centers.

**Core Idea**: The authors define **stable coresets**, which guarantee that the relative order of costs between centers is approximately consistent between the coreset and the original data (rather than preserving absolute costs). This concept sits strictly between weak and strong coresets, inheriting sub-metric transferability while being weak enough to be satisfied by uniform sampling.

## Method

### Overall Architecture

This is a **theoretical paper** without a trainable pipeline. Its core consists of "defining a new concept + a two-stage proof." The logic is as follows: first, define the three types of coresets in a unified language and establish their hierarchy (strong ⊂ stable ⊂ weak). Then, prove the main theorem (Theorem 1.4: uniform sampling → stable coreset for $\ell_1$) in two steps:

-   **Step 1 (General Framework for any metric, Section 3)**: Abstract a sufficient condition called "relative cost-difference approximation" (RCDA). Prove that if a subset $Q$ satisfies RCDA (plus a simple condition naturally met by uniform samples), $Q$ is a stable coreset. This step is metric-independent.
-   **Step 2 (Instantiation for $\ell_1$, Section 4)**: Use $\epsilon$-approximation techniques from PAC learning. Prove the VC dimension of "axis-aligned threshold functions" for $\ell_1$ is only $O(\log d)$. Thus, a uniform sample of size $O(\epsilon^{-2}\log d)$ is an $\epsilon$-approximation, which is then shown (via a key technical lemma) to be an $O(\epsilon)$-RCDA, completing the proof using the Step 1 framework.

Subsequently (Section 5), the conclusion is extended to metrics like Kendall-tau, Jaccard, Hamming, and $\ell_2$ via "embedding implies transferability," along with $k$-median approximation algorithms and an enhanced version for "C-dispersed" inputs.

### Key Designs

**1. Definition of Stable Coreset: Preserving Order, Not Absolute Cost**

To address the limitations where strong coresets require catching outliers and weak coresets don't transfer, the authors relax "preserving absolute cost" to "preserving the relative order of costs." Formally, a subset $Q\subseteq P$ is a stable $(\epsilon, \eta)$-coreset of $P$ iff:

$$\forall c_1,c_2\in X,\quad \mathrm{cost}(c_1,Q)\le(1+\epsilon)\,\mathrm{cost}(c_2,Q)\;\Rightarrow\;\mathrm{cost}(c_1,P)\le(1+\eta)\,\mathrm{cost}(c_2,P).$$

The hierarchy is clear: weak coresets (Def 1.1) only compare center $c$ to the optimal value $\mathrm{opt}(Q)$ on $Q$; strong coresets (Def 1.2) require $\mathrm{cost}(c,Q)\in(1\pm\epsilon)\,\mathrm{cost}(c,P)$ for **every** $c$ (preserves absolute values); stable coresets, like strong ones, impose geometric constraints on **all pairs of centers** $c_1, c_2$ (enabling sub-metric transfer) but use a "comparative" structure like weak ones (requiring only order preservation). Thus, stable coresets do not need to capture outliers that shift absolute costs uniformly across all centers.

The authors establish a strict hierarchy (Prop 2.1): every strong $\epsilon$-coreset is a stable $(\epsilon, 4\epsilon)$-coreset, and every stable $(\epsilon, \eta)$-coreset is a weak $(\epsilon, \eta)$-coreset. They also prove stable guarantees transfer via isometric embedding (Prop 2.3).

**2. RCDA Sufficient Condition: Reducing Stability to Verifiable Approximation**

Proving stability directly for a uniform sample is difficult. The authors introduce **relative cost-difference approximation ($\epsilon$-RCDA)** as a bridge. Let the normalized cost be $\overline{\mathrm{cost}}(x,P):=\tfrac1{|P|}\mathrm{cost}(x,P)$, and $\mu$ be the optimal median of $P$. $Q$ is an $\epsilon$-RCDA of $P$ if:

$$\forall x\in X,\quad \big|\,(\overline{\mathrm{cost}}(x,P)-\overline{\mathrm{cost}}(\mu,P))-(\overline{\mathrm{cost}}(x,Q)-\overline{\mathrm{cost}}(\mu,Q))\,\big|\le \epsilon\cdot\overline{\mathrm{cost}}(x,P).$$

This requires that the "cost difference" of any center $x$ relative to a reference $\mu$ is nearly the same on $P$ and $Q$. The framework's main result (Theorem 3.1) states: if $Q$ is an $\epsilon$-RCDA of $P$ and satisfies $\overline{\mathrm{cost}}(\mu,Q)\le c\cdot\overline{\mathrm{cost}}(\mu,P)$, then $Q$ is a stable $(\epsilon/c, 4\epsilon)$-coreset of $P$.

**3. $\ell_1$ Instantiation: Realizing RCDA via $\epsilon$-approximation and $O(\log d)$ VC Dimension**

To prove a uniform sample satisfies RCDA in $\ell_1$, the authors utilize **$\epsilon$-approximation** from PAC learning. Let $T=\{\tau_{i,r}:\tau_{i,r}(x)=\mathbf 1\{x[i]\le r\}\}$ be the family of axis-aligned threshold functions. $Q$ is an $\epsilon$-approximation of $P$ if the empirical distribution function (edf) is preserved for every dimension $i$ and threshold $r$: $|\mathrm{edf}_Q(i,r)-\mathrm{edf}_P(i,r)|\le\epsilon$.

-   **Logarithmic VC Dimension (Prop 4.2)**: $\lfloor\log d\rfloor\le \mathrm{VCdim}(T)\le 2\log d$. This ensures the sample size depends only on $\log d$ rather than $d$.
-   **Technical Lemma (Lemma 4.6)**: In $\ell_1^d$, if $Q$ is an $\epsilon$-approximation of $P$, it is a $20\epsilon$-RCDA. This follows because $\ell_1$ costs can be decomposed by coordinate as integrals of the edf.

Combining these: a uniform sample of size $O(\epsilon^{-2}\log\tfrac{d}{\delta})$ is an $\epsilon$-approximation with probability $1-\delta$, which makes it an $O(\epsilon)$-RCDA. Applying Theorem 3.1 yields the main theorem: **a uniform sample of size $O(\epsilon^{-2}\log d)$ is, with probability $\ge\tfrac{4}{5}$, a stable $(\epsilon/6, 4\epsilon)$-coreset for 1-median in $\ell_1^d$.**

**4. Transfer and Strengthening: Extensions of the 1-median Result**

Since stable coresets inherit sub-metric transferability, once established for $\ell_1$, metrics that embed into $\ell_1$ (Hamming, Kendall-tau, Jaccard, tree metrics) automatically gain uniform-sampling-based coresets. For $\ell_2$, Dvoretzky's theorem implies a stable coreset of size $O(\epsilon^{-2}\log(d/\epsilon))$. Furthermore: (i) the results enable **approximation algorithms for general $k$-median** (e.g., for Kendall-tau/Jaccard); (ii) for **C-dispersed** inputs (where diameter $\le C \times$ average distance), uniform sampling yields a **strong coreset** (Theorem 5.5).

## Key Experimental Results

Experiments validate stable coresets on real datasets. The metric is relative error $\widehat E=\dfrac{\mathrm{cost}(\hat c_Q,P)-\mathrm{cost}(\hat c_P,P)}{\mathrm{cost}(\hat c_P,P)}$, measuring the gap between the coreset's optimal center $\hat c_Q$ and the true optimal center $\hat c_P$ when evaluated on the full data $P$.

| Dataset | Size $n$ | Dim $d$ | Metric / Use Case |
| :--- | :--- | :--- | :--- |
| Yellow Taxi NYC (YT) | 2.8M | 11 | $\ell_1$ 1-median |
| Twitter | 1.3M | 3 | $\ell_1$ 1-median |
| Single-Cell Gene Expr. (SCGE) | 7,865 | 33,586 | $\ell_1$ Dimensionality Test |
| My Anime List (MAL) | 234K | 50 | Kendall-tau Rank Aggregation |

### Main Results: Uniform Sampling vs. Importance Sampling

| Dimension | Uniform Sampling (Ours) | Importance Sampling (Jiang et al. 2024) |
| :--- | :--- | :--- |
| Error $\widehat E$ | **Comparable** to Importance Sampling | Baseline |
| Construction Time (YT / Twitter / SCGE) | 500 samples ≈ **0.0001s** | ≈ 82 / 114 / 512s |
| Requires Full Data Pass | No (data-oblivious, constant time/sample) | Yes (linear time per point sensitivity) |

**Conclusion**: Uniform sampling achieves **similar approximation quality** to expensive importance sampling but is **5-6 orders of magnitude faster** to construct.

### Key Findings
- **Application 1: Rank Aggregation**: On MAL, small coresets approximate full data results using Kendall-tau heuristics. Occasionally, coreset results yield negative relative error (outperforming full data results on heuristics).
- **Application 2: Fairness Constraints**: On rank data with fairness-constrained ILP, small coresets support constrained optimization even when constraints were unknown during sampling.
- **Application 3: Dimensionality Dependence**: On SCGE, relative error remains largely stable as $d$ increases, suggesting the $\log d$ theoretical bound may be improvable to be dimension-independent.

## Highlights & Insights
- **The "Order Preservation" insight**: Relaxing the requirement from absolute values to relative cost ordering bypasses the need to capture outliers, which is the primary failure mode of uniform sampling for strong coresets.
- **RCDA as a Bridge**: Reducing stable coreset verification to a classic $\epsilon$-approximation / VC dimension problem provides a clean proof structure that can be applied to other metrics.
- **Free Transfer via Embedding**: By proving the result for $\ell_1$, the paper provides the first uniform-sampling-based coreset guarantees for Hamming, Kendall-tau, and Jaccard metrics.
- **Strong Coresets on C-dispersed Data**: The paper unifies the spectrum by showing uniform sampling yields strong coresets on sufficiently dispersed data, providing a practical acceleration for NP-hard discrete median problems.

## Limitations & Future Work
- **Dimension Dependence**: The theoretical bound includes $\log d$, though empirical evidence suggests it might be possible to eliminate this.
- **Limited to $k=1$**: The core results cover 1-median. Extending to general $k$-median requires structural assumptions, as uniform sampling cannot reliably capture small, distant clusters for $k > 1$.
- **Reliance on $\ell_1$ Embedding**: Practical applicability is limited to metrics that can be embedded into $\ell_1$ with low distortion.
- **Theoretical Constants**: The constants derived through the various approximation stages are relatively loose, leaving a gap between theory and practice.

## Related Work & Insights
- **vs. Strong Coresets (e.g., Braverman et al. 2022)**: Strong coresets preserve absolute costs and transfer to sub-metrics but require full data passes. Ours achieves sub-metric transfer and uniform sampling by only preserving order.
- **vs. Weak Coresets (e.g., Huang et al. 2023a)**: Weak coresets can use uniform sampling but **do not transfer to sub-metrics**. Ours adds the critical transferability property.
- **vs. Importance Sampling (Jiang et al. 2024)**: Sensitivity-based sampling provides strong coresets but is computationally expensive. Ours provides similar precision at negligible construction cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the new "stable coreset" concept and provides the first rigorous uniform sampling guarantees for $\ell_1$ and its embedded metrics.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validates results across various datasets and metrics, though scale is tailored to theoretical verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear hierarchy of concepts and well-structured two-stage proof framework.
- Value: ⭐⭐⭐⭐⭐ Elevates uniform sampling from a heuristic to a rigorously supported and widely applicable tool for large-scale clustering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] On Coreset for LASSO Regression Problem with Sensitivity Sampling](on_coreset_for_lasso_regression_problem_with_sensitivity_sampling.md)
- [\[ICLR 2026\] Random Spiking Neural Networks are Stable and Spectrally Simple](random_spiking_neural_networks_are_stable_and_spectrally_simple.md)
- [\[ICLR 2026\] Smooth Calibration Error: Uniform Convergence and Functional Gradient Analysis](smooth_calibration_error_uniform_convergence_and_functional_gradient_analysis.md)
- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)

</div>

<!-- RELATED:END -->
