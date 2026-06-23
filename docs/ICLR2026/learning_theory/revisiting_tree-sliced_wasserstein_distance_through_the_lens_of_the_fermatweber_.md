---
title: >-
  [Paper Note] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem
description: >-
  [ICLR 2026][learning_theory][Tree-Sliced Wasserstein] This paper identifies that the true advantage of Tree-Sliced Wasserstein (TSW) over Sliced Wasserstein (SW) lies in its sampling encoding "positional" information, which existing TSW variants (centering Gaussian on data mean) fail to exploit. Leveraging the classic Fermat–Weber problem, the authors use the **geometric
tags:
  - ICLR 2026
  - learning_theory
  - Tree-Sliced Wasserstein
date: 2026-05-08
content_hash: 76aebcd17645f97c
---
# Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=kDqG03v05B](https://openreview.net/forum?id=kDqG03v05B)  
**Code**: https://github.com/thanhquangtran/FW-TSW  
**Area**: Learning Theory / Optimal Transport  
**Keywords**: Optimal Transport, Tree-Sliced Wasserstein, Fermat–Weber Problem, Geometric Median, Sampling Strategies

## TL;DR
This paper identifies that the true advantage of Tree-Sliced Wasserstein (TSW) over Sliced Wasserstein (SW) lies in its sampling encoding "positional" information, which existing TSW variants (centering Gaussian on data mean) fail to exploit. Leveraging the classic Fermat–Weber problem, the authors use the **geometric median** as the sampling center for the tree system's intersection, proposing FW-TSW / FW-TSW\*. This approach improves gradient flow, topic modeling, and diffusion model training with negligible computational overhead.

## Background & Motivation

**Background**: Optimal Transport (OT) compares probability measures while respecting data geometry, but exact OT complexity grows super-cubically with support points. Sliced Wasserstein (SW) projects high-dimensional measures onto 1D lines, using closed-form 1D OT solutions to achieve near-linear complexity. Recently, Tree-Sliced Wasserstein (TSW) replaced single projection lines with "tree systems" (a set of $k$ concurrent lines $T=(x,\theta_1,\dots,\theta_k)$ in $\mathbb{R}^d$), using a splitting map to distribute mass across lines, thereby capturing more complex topologies at low cost.

**Limitations of Prior Work**: An overlooked essential difference between SW and TSW is the **sampling strategy**. SW only samples directions $\theta\in S^{d-1}$, whereas TSW tree systems require sampling an **intersection point (position)** $x\in\mathbb{R}^d$. TSW naturally carries both "direction + position" information. However, existing TSW variants (TSW-SL, Db-TSW) handle this intersection point heuristically: since $\mathbb{R}^d$ is non-compact, they simply sample intersections from a Gaussian centered at the data mean.

**Key Challenge**: The mean is sensitive to outliers, and heuristic sampling fails to truly "align" the geometric centers of source and target distributions, wasting the positional degree of freedom. Furthermore, the commonly used splitting map ($\alpha(y,T)=\mathrm{softmax}(\{\xi\cdot d(y,T)_i\})$) is inherently **position-dependent**; the choice of intersection directly affects the rationality of mass allocation.

**Goal**: Design a **principled, data-dependent** intersection sampling distribution for TSW that places the tree root near both source and target data while making direction sampling more informative.

**Key Insight**: Since OT aims to align source and target distributions, the intersection point $x$ should **minimize its average distance to the data points**. This corresponds to the classic Fermat–Weber problem, whose solution is the **geometric median**. The geometric median is more robust to outliers than the mean and can be efficiently approximated via the Weiszfeld algorithm.

**Core Idea**: Use the geometric median (instead of the mean) as the center for sampling tree system intersections, injecting the Fermat–Weber principle into TSW construction to obtain FW-TSW. Further bias direction sampling toward the "source-to-target" directions to obtain the enhanced FW-TSW\*.

## Method

### Overall Architecture
FW-TSW follows this logic: given source measure $\mu$ and target measure $\nu$, TSW-based methods sample tree systems $T_1,\dots,T_L$, calculate closed-form OT distances $W_1$ on each tree, and average them. This work **does not change** the Monte Carlo estimation framework (i.e., $\widehat{\mathrm{TSW}}=\frac1L\sum_l W_1(R^\alpha_{T_l}f_\mu, R^\alpha_{T_l}f_\nu)$) but rather **redesigns the tree system sampling distribution $\sigma_T$**.

Specifically, a tree system consists of "an intersection point $x\in\mathbb{R}^d$ + $k$ directions $\theta_i\in S^{d-1}$". The sampling distribution is modeled as the product of these $k+1$ components. This paper modifies two components:

- **Intersection Component**: Samples $m$ points from both source and target, computes the geometric median $x^*$ via the Weiszfeld algorithm, and samples the intersection from a Gaussian centered at $x^*$ → resulting in $\sigma_{\mathrm{FW},\mu,\nu}$ and FW-TSW.
- **Direction Component (Optional Enhancement)**: Replaces uniform direction sampling with sampling biased toward "source point $x_i$ minus target point $y_j$" → resulting in $\sigma^*_{\mathrm{FW},\mu,\nu}$ and FW-TSW\*.

By inserting these modifications into the averaging framework and distributing mass via a splitting map, the final distance is FW-TSW($\mu,\nu$) $=\int_{\mathbb{T}} W_1(\mu_T,\nu_T)\,d\sigma_{\mathrm{FW},\mu,\nu}(T)$.

### Key Designs

**1. Geometric Median Sampling: Replacing Data Mean with the Fermat–Weber Solution**

This addresses the "unprincipled intersection sampling" issue. Given a point set $\{x_1,\dots,x_m,y_1,\dots,y_m\}$ from the combined source and target, the Fermat–Weber problem seeks the location minimizing the sum of distances:

$$x^* = \arg\min_{x\in\mathbb{R}^d}\ \frac1n\sum_{i=1}^n \lVert x - x_i\rVert_2,$$

where $x^*$ is the geometric median. The intersection is then sampled as $x\sim\mathcal{N}(x^*, cI_d)$. The tree system distribution becomes $\mathcal{N}(x^*,cI_d)\otimes U(S^{d-1})^{\otimes k}$. The constant $c$ controls concentration around $x^*$. The symmetry $\sigma_{\mathrm{FW},\mu,\nu}=\sigma_{\mathrm{FW},\nu,\mu}$ is maintained. The **Weiszfeld algorithm** approximates the solution:

$$x^{(t+1)} = \left(\sum_i \frac{x_i}{\lVert x^{(t)}-x_i\rVert_2}\right)\Big/\left(\sum_i \frac{1}{\lVert x^{(t)}-x_i\rVert_2}\right),$$

using iterative weighted averages. The overhead is $O(Tnd)$, which is negligible.

**2. Data-dependent Direction Sampling: Aligning Projections with Source-Target Displacement (FW-TSW\*)**

Uniform direction sampling fails to distinguish informative directions. This work introduces an enhancement: randomly select a source point $x_i$ and target point $y_j$ to construct direction:

$$\theta = \frac{\psi + \zeta\cdot s\cdot(x_i - y_j)}{\lVert \psi + \zeta\cdot s\cdot(x_i - y_j)\rVert_2}\in S^{d-1},$$

where $\psi\sim U(S^{d-1})$ is a base uniform direction, $s\sim U(\{\pm1\})$ is a random sign, and $\zeta>0$ controls the bias toward the displacement vector $(x_i-y_j)$. These directions tend to align with the transport path.

**3. Provable Metric Properties and Controlled Upper Bounds**

The authors prove that FW-TSW and FW-TSW\* are **semi-metrics** on $\mathcal{P}(\mathbb{R}^d)$, satisfying non-negativity, symmetry, and identity of indiscernibles, along with a **quasi-triangle inequality**:

$$\mathrm{FW\text{-}TSW}(\mu_1,\mu_2)\le \mathrm{FW\text{-}TSW}_{\mu_1,\mu_2}(\mu_1,\mu_3)+\mathrm{FW\text{-}TSW}_{\mu_1,\mu_2}(\mu_2,\mu_3).$$

 Crucially, anchoring the intersection at the geometric median allows for an **upper bound** (Theorem 4.5): when the center is the joint Fermat–Weber solution $v^*$:

$$\int_{\mathbb{T}} W_1(\mu_T,\nu_T)\,d\bar\sigma_{\mathrm{FW},\mu,\nu}(T)\le k\,W_2(\mu,\nu) + k(k-1)\cdot\frac{2\pi^{d/2}}{\Gamma\!\big(\tfrac{d+1}{2}\big)\Gamma\!\big(\tfrac12\big)}f(v^*),$$

where $f(v)=\int\lVert x-v\rVert_2\,d\mu+\int\lVert x-v\rVert_2\,d\nu$.

### Loss & Training
FW-TSW is a **distance metric**, not an additional training loss. It is inserted as a differentiable transport cost into three tasks: gradient flow ($D(\mu_t,\nu)$); topic modeling (replacing the KL term in VAE with $\lambda\,\mathrm{FW\text{-}TSW}$); and diffusion models (AGME loss in DDGAN). Computational complexity is $O(Lkn\log n + Lkdn + Tnd)$.

## Key Experimental Results

### Main Results

**Gradient Flow (25 Gaussians, Average Wasserstein distance over 5 runs, lower is better)**:

| Method | step 1000 | step 2000 | step 2500 |
|------|-----------|-----------|-----------|
| SW | 2.42e-03 | 1.69e-03 | 1.01e-03 |
| TSW-SL | 1.37e-06 | 9.13e-07 | 8.76e-07 |
| Db-TSW | 1.55e-06 | 9.50e-07 | 8.55e-07 |
| Db-TSW⊥ | 1.79e-06 | 1.14e-06 | 1.03e-06 |
| **FW-TSW (Ours)** | 1.51e-06 | 9.18e-07 | 8.40e-07 |
| **FW-TSW\* (Ours)** | 1.50e-06 | **9.04e-07** | **8.29e-07** |

FW-TSW\* achieves the overall best results by step 2500.

**Diffusion Models (DDGAN, CIFAR-10 unconditional generation, FID, lower is better)**:

| Model | FID ↓ | Time/Epoch(s) ↓ |
|------|-------|------------------|
| DDGAN | 3.64 | 72 |
| RPSW-DD | 2.82 | 76 |
| TSW-SL-DD | 2.83 | 80 |
| Db-TSW-DD | 2.60 | 84 |
| Db-TSW-DD⊥ | 2.53 | 85 |
| **FW-TSW-DD (Ours)** | 2.336 ± 0.003 | 85 |
| **FW-TSW\*-DD (Ours)** | **2.315 ± 0.002** | 87 |

Ours reduces FID by up to 0.215 compared to Prev. SOTA (Db-TSW-DD⊥) with nearly identical training time (85s vs 87s).

### Ablation Study (Topic Modeling, Topic Coherence CV, higher is better)

| Method | DBLP | M10 | BBC |
|------|------|-----|-----|
| SW-TM | 0.432 | 0.484 | 0.760 |
| TSW-SL-TM | 0.453 | 0.456 | 0.796 |
| Db-TSW-TM | 0.441 | 0.458 | 0.787 |
| **FW-TSW-TM (Ours)** | 0.505 | 0.498 | 0.792 |
| **FW\*-TSW-TM (Ours)** | **0.511** | **0.502** | **0.801** |

### Key Findings
- **Direction Enhancement (FW-TSW\*) consistently helps**: Across three tasks, FW-TSW\* generally outperforms FW-TSW, indicating that data-dependent directions and geometric median intersections provide complementary gains.
- **Negligible Overhead**: Significant FID improvements occur while single-epoch training time only increases from 85s to 87s.
- **Position information is crucial**: Simply switching from mean to geometric median allows FW-TSW to outperform Db-TSW, validating the core argument.

## Highlights & Insights
- **Turning an overlooked detail into a core contribution**: The author identifies that the difference between SW and TSW is not just "line vs. tree" but "direction-only vs. direction+position."
- **Modern reuse of classic tools**: The Fermat–Weber problem and Weiszfeld algorithm are classic results cleanly applied to TSW sampling.
- **Robust centers yield provable bounds**: The geometric median provides a stable "anchor," solving the issue of uncontrolled position info that hindered upper bound derivations in prior work.
- **Transferability**: Any method requiring a data-dependent sampling center in $\mathbb{R}^d$ can likely benefit from replacing the mean with the geometric median.

## Limitations & Future Work
- **Lack of explicit transport map**: Like all TSW variants, FW-TSW provides a distance value rather than a transport plan/map, making it less suitable for push-forward tasks.
- **Semi-metric properties**: It satisfies a quasi-triangle inequality rather than a strict one, technically weaker than a true Wasserstein metric.
- **Hyperparameter dependence**: Concentration $c$ and bias strength $\zeta$ still require tuning.

## Related Work & Insights
- **vs SW (Sliced Wasserstein)**: SW uses uniform spherical distributions and cannot distinguish informative directions; FW-TSW utilizes both direction and position with data-dependent sampling.
- **vs TSW-SL / Db-TSW**: These variants use data means and uniform directions; FW-TSW uses robust geometric medians and biased directions, achieving a controlled bound.
- **vs Trainable slicing (e.g., max-SW)**: Those methods require expensive iterative optimization; FW-TSW uses closed-form geometric medians and analytical direction construction, which is more stable and efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)
- [\[ICLR 2026\] Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis](best-of-n_through_the_smoothing_lens_kl_divergence_and_regret_analysis.md)
- [\[ICLR 2026\] On Coreset for LASSO Regression Problem with Sensitivity Sampling](on_coreset_for_lasso_regression_problem_with_sensitivity_sampling.md)
- [\[ICLR 2026\] Revisiting Active Sequential Prediction-Powered Mean Estimation](revisiting_active_sequential_prediction-powered_mean_estimation.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](better_bounds_for_the_distributed_experts_problem.md)

</div>

<!-- RELATED:END -->
