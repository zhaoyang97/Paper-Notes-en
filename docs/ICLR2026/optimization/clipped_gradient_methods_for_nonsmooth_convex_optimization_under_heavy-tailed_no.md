---
title: >-
  [Paper Note] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis
description: >-
  [ICLR 2026][Optimization][heavy-tailed noise] This paper provides a more refined convergence analysis for Clipped SGD under heavy-tailed noise. By utilizing Freedman’s inequality more effectively and providing tighter clipping error bounds, the authors accelerate the known optimal high-probability convergence rates by a factor of $\mathrm{poly}(1/d_{\mathrm{eff}})$ ($d_{\mathrm{eff}}$ is the "generalized effective dimension" defined by the authors). Furthermore…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "heavy-tailed noise"
  - "gradient clipping"
  - "high-probability convergence"
  - "nonsmooth convex optimization"
  - "effective dimension"
date: 2026-05-08
content_hash: 1d26619322070d62
---

# Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2dbLeXDe39](https://openreview.net/forum?id=2dbLeXDe39)  
**Code**: None (Pure theoretical paper)  
**Area**: optimization  
**Keywords**: heavy-tailed noise, gradient clipping, high-probability convergence, nonsmooth convex optimization, effective dimension

## TL;DR
This paper provides a more refined convergence analysis for Clipped SGD under heavy-tailed noise. By utilizing Freedman’s inequality more effectively and providing tighter clipping error bounds, the authors accelerate the known optimal high-probability convergence rates by a factor of $\mathrm{poly}(1/d_{\mathrm{eff}})$ ($d_{\mathrm{eff}}$ is the "generalized effective dimension" defined by the authors). Furthermore, the results for expected convergence **break existing lower bounds** and perfectly match (i.e., are optimal) the newly proven lower bounds.

## Background & Motivation
**Background**: In stochastic first-order optimization, SGD has been extensively studied under the classical assumption of "finite second moment of gradient noise (finite variance)." However, recent empirical observations show that gradient noise in modern machine learning tasks is often **heavy-tailed**—the second moment may not even exist. A more realistic assumption is that there exists some $p \in (1,2]$ such that the $p$-th moment is bounded, i.e., $\mathbb{E}\|g-\nabla f\|^p \le \sigma_l^p$. Under such heavy-tailed noise, standard SGD fails, and **gradient clipping** is a repeatedly validated remedy with theoretical guarantees, leading to the Clipped SGD method.

**Limitations of Prior Work**: For nonsmooth (strongly) convex problems, the best prior high-probability convergence rates were established by Liu & Zhou (2023): $O\!\big(\sigma_l\ln(1/\delta)\,T^{\frac1p-1}\big)$ for the convex case and $O\!\big(\sigma_l^2\ln^2(1/\delta)\,T^{\frac2p-2}\big)$ for the strongly convex case. These rates were long considered "optimal" because they match existing lower bounds in expectation if $\mathrm{poly}(\ln(1/\delta))$ is treated as a constant. However, Das et al. (2024) achieved a faster rate for the specific case of $p=2$ by introducing the **effective dimension** $d_{\mathrm{eff}} \in [1,d]$, suggesting that the "optimal" consensus regarding terms involving $\mathrm{poly}(\ln(1/\delta))$ might be incorrect—there could be room for improvement across all $p \in (1,2]$.

**Key Challenge**: The problem stems from two "looser" aspects of previous analysis. First, when proving the concentration of Term I, literature has consistently used $\mathbb{E}\|d_t^u\|^2$ to bound conditional variance, but only the variance along the direction of a predictor vector needs to be controlled; using the entire second moment is overly conservative. Second, the clipping error (especially the bias term $\|d_t^b\|$) has historically used the bound $O(\sigma_l^p\tau^{1-p})$, which the authors find is not tight. While Das et al. (2024) implicitly addressed the first point, they relied on a complex "iterative refinement" process, resulting in an extra $\ln(\ln T/\delta)$ factor and the restriction $T \ge \Omega(\ln \ln d)$.

**Goal**: (1) Prove that a universal improvement exists for all $p \in (1,2]$, providing faster high-probability rates; (2) extend the same insights to expected convergence to see if known lower bounds can be surpassed; (3) fill the gap in missing lower bounds for the heavy-tailed setting to determine if the new rates are optimal.

**Key Insight / Core Idea**: The authors characterize "how heavy-tailed the noise is" more precisely. In addition to the standard $\mathbb{E}\|d\|^p \le \sigma_l^p$, they require the projected noise in any unit direction to satisfy $\mathbb{E}|\langle e,d\rangle|^p \le \sigma_s^p$ ($\sigma_s \le \sigma_l$). This leads to the definition of the **generalized effective dimension** $d_{\mathrm{eff}} \triangleq \sigma_l^2/\sigma_s^2$, which is used to recover quantities previously wasted by "worst-case directional estimation." In short: **by using fine-grained noise assumptions + two tighter inequalities, the overestimated terms in the analysis are reduced to their true magnitudes.**

## Method

### Overall Architecture
The algorithm studied is the classical Clipped (Proximal) SGD (Algorithm 1). The structural algorithm remains unchanged; the contribution lies entirely in the **analysis**. The composite optimization problem is written as $\inf_{x \in \mathcal{X}} F(x) \triangleq f(x) + r(x)$, where $f$ is $G$-Lipschitz convex and $r$ is $\mu$-strongly convex ($\mu \ge 0$, with $\mu=0$ being general convex). At each step, an unbiased noisy gradient $g_t = g(x_t, \xi_t)$ is sampled, clipped via $g_t^c = \mathrm{clip}_{\tau_t}(g_t)$, followed by a proximal update:

$$g_t^c = \min\!\Big\{1, \frac{\tau_t}{\|g_t\|}\Big\}g_t, \qquad x_{t+1} = \arg\min_{x \in \mathcal{X}}\; r(x) + \langle g_t^c, x\rangle + \frac{\|x-x_t\|^2}{2\eta_t}.$$

The logic of the paper is: first, reduce convergence to the high-probability control of three stochastic terms (denoted I, II, III) in an "almost surely" valid residual inequality. Then, two new tools—**better application of Freedman’s inequality** and **tighter clipping error bounds (Lemma 1)**—are used to shrink I and III, yielding faster rates than Liu & Zhou (2023). These same tools simplify expected convergence and match a newly proven lower bound using information-theoretic methods. This is a theoretical paper; no pipeline diagram is provided. The core lies in the "Key Designs" below.

The clipping threshold is chosen as $\tau_t = \max\{2G, \ \tau_\star T^{1/p}\}$, with the key term:
$$\tau_\star = \sigma_l/\varphi_\star^{1/p}, \qquad \varphi_\star \triangleq \max\Big\{\sqrt{d_{\mathrm{eff}}}\,\ln\tfrac3\delta, \ d_{\mathrm{eff}}\,\mathbb{1}[p<2]\Big\}.$$

### Key Designs

**1. Fine-grained heavy-tailed noise assumption and generalized effective dimension $d_{\mathrm{eff}}$**: 
Literature typically uses only a single scalar $\sigma_l$ for the heavy-tailed assumption: $\mathbb{E}\|d(x,\xi)\|^p \le \sigma_l^p$ (where $d = g - \nabla f$). The authors introduce an additional moment bound for projected directions $\mathbb{E}|\langle e,d(x,\xi)\rangle|^p \le \sigma_s^p$ (for all unit vectors $e$). They prove this is not an extra assumption: by Cauchy-Schwarz, if $\mathbb{E}\|d\|^p \le \sigma_l^p$ holds, there must exist $0 \le \sigma_s \le \sigma_l$ satisfying the projection bound; conversely, $\sigma_l \le \sqrt{\pi d/2}\,\sigma_s$. Thus:

$$d_{\mathrm{eff}} \triangleq \sigma_l^2/\sigma_s^2 \in \{0\} \cup [1, \pi d/2] = O(d).$$

When $p=2$, by setting $\sigma_l^2 = \sup_x \mathrm{Tr}(\Sigma(x))$ and $\sigma_s^2 = \sup_x \|\Sigma(x)\|$ (where $\Sigma$ is the noise covariance), $d_{\mathrm{eff}} = \mathrm{Tr}(\Sigma)/\|\Sigma\|$ recovers the effective dimension of Das et al. (2024). This quantity implies that when noise is "isotropic," $\sigma_s \approx \sigma_l$ and $d_{\mathrm{eff}} \approx 1$ (no gain); but when noise is concentrated in a few directions (e.g., low-rank covariance), $\sigma_s \ll \sigma_l$ and $d_{\mathrm{eff}}$ can be as large as $\Omega(d)$. All $1/\mathrm{poly}(d_{\mathrm{eff}})$ speedups originate here.

**2. Better use of Freedman’s inequality**: 
Term I in the residual is of the form $\big(\max_{t} \sum_{s \le t} \langle d_s^u, y_s\rangle\big)^2$, where $d_t^u = g_t^c - \mathbb{E}_{t-1}[g_t^c]$ is the unbiased part of the clipping error and $y_t$ is a predictable vector with $\|y_t\| \le 1$. Let $X_t = \langle d_t^u, y_t\rangle$ be a martingale difference sequence. Freedman’s inequality gives:
$$\sqrt{\mathrm{I}} \le O\!\Big(\max_t|X_t|\ln\tfrac1\delta + \sqrt{\textstyle\sum_t\mathbb{E}_{t-1}[X_t^2]\,\ln\tfrac1\delta}\Big).$$
Prior literature bounded the conditional variance via $\mathbb{E}_{t-1}[X_t^2] \le \mathbb{E}_{t-1}\|d_t^u\|^2 \le O(\sigma_l^p\tau^{2-p})$. The key observation is that since $\|y_t\| \le 1$:
$$\mathbb{E}_{t-1}[X_t^2] = y_t^\top\,\mathbb{E}_{t-1}\!\big[d_t^u(d_t^u)^\top\big]\,y_t \le \big\|\mathbb{E}_{t-1}[d_t^u(d_t^u)^\top]\big\|,$$
where the **operator norm** of the covariance matrix is at most $\mathbb{E}\|d_t^u\|^2$ (the trace) but can be much smaller. Combined with the operator norm bound $O(\sigma_s^p\tau^{2-p} + \sigma_l^p G^2\tau^{-p})$ provided in Lemma 1 (where the smaller $\sigma_s$ dominates), the high-probability bound for Term I is significantly tightened. While this was implicitly used for $p=2$ in Das et al. (2024), their "iterative refinement" added complexity; the authors show this complexity is unnecessary and that the simple approach works for all $p \in (1,2]$.

**3. Refined clipping error bounds (Lemma 1)**: 
The clipping error is decomposed into an unbiased part $d_t^u$ and a bias part $d_t^b = \mathbb{E}_{t-1}[g_t^c] - \nabla f(x_t)$. For $\tau \ge 2G$, Lemma 1 provides:
$$\big\|\mathbb{E}_{t-1}[d_t^u(d_t^u)^\top]\big\| \le O(\sigma_s^p\tau^{2-p} + \sigma_l^p G^2\tau^{-p}), \qquad \|d_t^b\| \le O(\sigma_s\sigma_l^{p-1}\tau^{1-p} + \sigma_l^p G\tau^{-p}).$$
The authors also remove the $\tau \ge 2G$ requirement for the second moment bound $\mathbb{E}_{t-1}\|d_t^u\|^2$ and improve constants. Compared to standard bounds, the bias term is tightened from $O(\sigma_l^p\tau^{1-p})$ to $O(\sigma_s\sigma_l^{p-1}\tau^{1-p} + \cdots)$—which is strictly smaller since $\sigma_s \le \sigma_l$ and $\tau \ge 2G$. Term III ($\big(\sum_t \|d_t^b\|\big)^2$) is reduced via this tighter bias bound.

**4. Downstream results: Anytime variants and breakthrough in expected convergence**: 
The same tools are pushed in several directions. First, for the general convex case which usually requires knowing $T$ to remove the $\mathrm{poly}(\ln T)$ factor, the authors propose **Stabilized Clipped SGD** (using the stabilization technique from Fang et al. 2022) to achieve anytime results. Second, the tighter error bounds improve **expected convergence** to $O(\sigma_l d_{\mathrm{eff}}^{-(2-p)/(2p)}T^{\frac1p-1})$ for convex and $O(\sigma_l^2 d_{\mathrm{eff}}^{-(2-p)/p}T^{\frac2p-2})$ for strongly convex problems. For $p < 2$, these rates are **strictly faster** than the previously known lower bounds $\Omega(\sigma_l T^{\frac1p-1})$ and $\Omega(\sigma_l^2 T^{\frac2p-2})$. Third, the authors use information theory to prove new lower bounds for the **fine-grained oracle class** $\mathcal{G}^p_{\sigma_s, \sigma_l} \subseteq \mathcal{G}^p_{\sigma_l}$. Since old lower bounds were proved for the larger class $\mathcal{G}^p_{\sigma_l}$, they were loose for the subclass; the new matching lower bounds prove the optimality of the upper bounds in the fine-grained setting.

### Loss & Training
Not applicable: This paper does not introduce a new loss function. The algorithm is standard Clipped (Proximal) SGD. The novelty lies in the convergence analysis and the setting of hyperparameters ($\eta_t, \tau_t$).

## Key Experimental Results
This is a pure theoretical work without numerical experiments. The following tables summarize the quantitative contributions: convergence rate acceleration and refinement of clipping error bounds.

### Main Results: Rate Comparison (Leading terms, $T \to \infty, \delta \to 0$)

| Setting | Prev. SOTA | Ours | Gain |
|---------|------------|------|------|
| High-prob, Convex | $O(\sigma_l\ln\frac1\delta\,T^{\frac1p-1})$ (Liu & Zhou 23) | $O(\sigma_l\,d_{\mathrm{eff}}^{-\frac1{2p}}\ln^{1-\frac1p}\!\frac1\delta\,T^{\frac1p-1})$ | $\mathrm{poly}(1/d_{\mathrm{eff}}, 1/\ln\frac1\delta)$ |
| High-prob, S. Convex | $O(\sigma_l^2\ln^2\!\frac1\delta\,T^{\frac2p-2})$ (Liu & Zhou 23) | $O(\sigma_l^2\,d_{\mathrm{eff}}^{-\frac1p}\ln^{2-\frac2p}\!\frac1\delta\,T^{\frac2p-2})$ | $\mathrm{poly}(1/d_{\mathrm{eff}}, 1/\ln\frac1\delta)$ |
| Expected, Convex | Bound $\Omega(\sigma_l\,T^{\frac1p-1})$ | $O(\sigma_l\,d_{\mathrm{eff}}^{-\frac{2-p}{2p}}\,T^{\frac1p-1})$ | Breaks lower bound by $\Theta(1/d_{\mathrm{eff}}^{\frac{2-p}{2p}})$ (Optimal) |
| Expected, S. Convex | Bound $\Omega(\sigma_l^2\,T^{\frac2p-2})$ | $O(\sigma_l^2\,d_{\mathrm{eff}}^{-(2-p)/p}\,T^{\frac2p-2})$ | Breaks lower bound by $\Theta(1/d_{\mathrm{eff}}^{\frac{2-p}{p}})$ (Optimal) |

When $d_{\mathrm{eff}} = \Omega(d)$, the acceleration factor can reach $\mathrm{poly}(1/d)$. For $p=2$, the high-probability convex result is strictly superior to Das et al. (2024), removing extra terms and improving $\delta$ dependence from $\ln(\ln T/\delta)$ to $\ln(1/\delta)$.

### Mechanisms: Clipping Error Bound Comparison ($\tau \ge 2G$)

| Quantity | Standard Bound (6) | Lemma 1 (Ours) |
|----------|--------------------|----------------|
| $\|d_t^u\|$ | $O(\tau)$ | $O(\tau)$ (Same) |
| $\mathbb{E}_{t-1}\|d_t^u\|^2$ | $O(\sigma_l^p\tau^{2-p})$ (needs $\tau \ge 2G$) | $O(\sigma_l^p\tau^{2-p})$ (no $\tau \ge 2G$, better constants) |
| $\|\mathbb{E}_{t-1}[d_t^u(d_t^u)^\top]\|$ | — (N/A) | $O(\sigma_s^p\tau^{2-p} + \sigma_l^p G^2\tau^{-p})$ (New, smaller $\sigma_s$ dominates) |
| $\|d_t^b\|$ | $O(\sigma_l^p\tau^{1-p})$ | $O(\sigma_s\sigma_l^{p-1}\tau^{1-p} + \sigma_l^p G\tau^{-p})$ (Strictly tighter) |

### Key Findings
- Two sources of speedup perform different duties: operator norm bounds (with Freedman) shrink the variance-type Term I, while bias bounds $\|d_t^b\|$ shrink Term III. Both replace $\sigma_l$ with the smaller $\sigma_s$, which is the physical source of the $d_{\mathrm{eff}}$ factor.
- High-probability upper bounds still contain a high-order residual term (e.g., $O(\varphi GD/T)$ for convex) that becomes negligible after a critical time $T_\star = \Theta(d_{\mathrm{eff}}\ln^2(1/\delta) + d_{\mathrm{eff}}^2\mathbb{1}[p<2])$.
- Expected upper and lower bounds match perfectly, meaning expected convergence is already optimal. However, a gap remains for high-probability bounds in $\mathrm{poly}(\ln(1/\delta))$ terms.

## Highlights & Insights
- **"Using the right direction" is more valuable than "using stronger tools"**: The speedup for Term I does not come from a stronger concentration inequality, but from identifying that only variance in the direction of $y_t$ matters, thus replacing trace (second moment) with operator norm—a simple, often overlooked refinement.
- **Simplifying complex proofs**: Das et al. (2024) relied on "iterative refinement" for $p=2$, which introduced logarithmic side effects. This paper proves such complexity is unnecessary, achieving better results with simpler arguments that generalize to all $p \in (1,2]$.
- **Dual progress in upper and lower bounds**: The author clarifies that "old lower bounds were proved for a larger oracle class $\mathcal{G}^p_{\sigma_l}$ and are thus loose for the subgroup $\mathcal{G}^p_{\sigma_s, \sigma_l}$." This explains how upper bounds can break old lower bounds and directs the proof of new matching bounds.
- **Transferability**: The fine-grained noise assumption and the two clipping error insights are not limited to nonsmooth convex settings; they are general results likely applicable to smooth or non-convex settings.

## Limitations & Future Work
- The high-probability rates still have a $d_{\mathrm{eff}}$-dependent high-order residual term; whether this can be removed for all $T$ remains unknown.
- Gap in $\mathrm{poly}(\ln(1/\delta))$ terms for high-probability upper and lower bounds.
- Acceleration depends on $d_{\mathrm{eff}}$ being large (noise anisotropy). If noise is isotropic ($d_{\mathrm{eff}} \approx 1$), there is little improvement over Liu & Zhou (2023).
- Pure theoretical: Thresholds depend on $\sigma_s, \sigma_l$, and $d_{\mathrm{eff}}$. Practical application would require adaptive clipping strategies.

## Related Work & Insights
- **vs. Liu & Zhou (2023)**: They provided previous "optimal" high-probability rates. This work speeds them up by $\mathrm{poly}(1/d_{\mathrm{eff}})$ with tighter analysis, showing that "matching old lower bounds" does not imply true optimality.
- **vs. Das et al. (2024)**: They first introduced effective dimension for $p=2$. This work removes their $\ln(\ln T/\delta)$ artifacts and generalizes the concept to the entire $p \in (1,2]$ range.
- **vs. Gorbunov et al. (2024a/b)**: Part of the lineage of high-probability analysis for clipping. Gorbunov (2024b) handles composite optimization but uses a different algorithm (Prox-Clipped-SGD-Shift).
- **Insight**: The combination of operator norm, fine-grained noise, and identifying the correct oracle class constitutes a reusable methodology for tightening heavy-tailed analysis in other settings like online optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Refutes the "optimal rate" consensus and proves breakthrough expected rates.
- Experimental Thoroughness: ⭐⭐⭐⭐ Pure theory with complete proofs (upper/lower bounds, anytime variants), though lacks numerical validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain from motivation to insight to result.
- Value: ⭐⭐⭐⭐⭐ Methodologies are general and transferable to many heavy-tailed scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](decentralized_nonconvex_optimization_under_heavy-tailed_noise_normalization_and_.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](../../ICML2026/optimization/can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICLR 2026\] High-Probability Bounds for the Last Iterate of Clipped SGD](high-probability_bounds_for_the_last_iterate_of_clipped_sgd.md)

</div>

<!-- RELATED:END -->
