---
title: >-
  [Paper Note] Model Merging Scaling Laws in Large Language Models
description: >-
  [ICML 2026][Model Compression][scaling law] The authors empirically derived a dual-axis power law of the form $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ using 10,866 merged models. The base scale $N$ determines the performance floor, while the number of experts $k$ determines the tail. Four mainstream merging methods (Average, TA, TIES, DARE) share the same curve
tags:
  - ICML 2026
  - Model Compression
  - scaling law
  - power law
  - task arithmetic
  - TIES/DARE
date: 2026-05-08
content_hash: 38d75ac116e03960
---
# Model Merging Scaling Laws in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2509.24244](https://arxiv.org/abs/2509.24244)  
**Code**: https://github.com/InfiXAI/Merging-Scaling-Law (Available)  
**Area**: LLM Pre-training / Model Merging / Scaling Law  
**Keywords**: Model merging, scaling law, power law, task arithmetic, TIES/DARE

## TL;DR
The authors empirically derived a dual-axis power law of the form $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ using 10,866 merged models. The base scale $N$ determines the performance floor, while the number of experts $k$ determines the tail. Four mainstream merging methods (Average, TA, TIES, DARE) share the same curve, transforming the engineering questions of "how many experts to merge" and "when to stop" into a predictable and budgetable problem.

## Background & Motivation
**Background**: Model merging has become a low-cost "expert integration" paradigm following multi-task SFT. Linear weighting (Model Soups, Task Arithmetic) and versions with preprocessing (TIES, DARE) are widely used in scenarios involving LLMs and LoRA adapters.

**Limitations of Prior Work**: Merging remains largely heuristic—iterating through different subsets, sequences, and normalization coefficients is costly and lacks guidance from scaling laws similar to those in pre-training. Given a target loss, it is currently impossible to pre-determine the required number of experts or whether scaling the base model versus merging an additional expert is more cost-effective.

**Key Challenge**: The gain curve of merging is clearly non-linear, yet it exhibits regularity (steep early gains followed by saturation). Without an analytical form to describe this curve, engineering practice relies on exhaustive search, wasting GPU resources.

**Goal**: (1) Find a compact formula that simultaneously characterizes the impact of $N$ (base parameters) and $k$ (number of merged experts); (2) Prove its validity across different merging algorithms, backbones, in-domain, and cross-domain settings; (3) Provide a practical method to extrapolate the entire curve by measuring only three points.

**Key Insight**: Merging is viewed as "equal-weight averaging of multiple task vectors." Under second-order Taylor expansion, the variance of equal-weight averaging shrinks at a rate of $1/k$, and the variance enters the loss via the Hessian as the $A(N)/k$ term. This led the authors to expect a "floor + 1/k tail" structure, which was then verified through large-scale empirical evidence.

**Core Idea**: A unified power law of "floor + 1/(k+b) tail" is used to describe the CE curves of all merging methods. This integrates base scale and expert count into a single formula, making merging a budget-aware and predictable process.

## Method

### Overall Architecture
The paper addresses the engineering question of how many experts are needed and what base model size should be used for a given target loss. Using the Qwen2.5 series (0.5B/1.5B/3B/7B/14B/32B/72B), nine domain experts (algebra, analysis, geometry, discrete, number\_theory, code, chemistry, physics, biology) were fine-tuned from the same base. For each $(N,k)$ combination, all $\binom{9}{k}$ expert subsets were traversed or uniformly sampled. Four merging algorithms (Average, TA, TIES, DARE) were used to synthesize models and measure token-level CE, accumulating grid data from 10,866 merged models across in-domain and cross-domain evaluations. Weighted non-linear least squares were then used to fit a decoupled curve, validated by $R^2$ and residual structures.

### Key Designs

**1. Unified floor+tail power law: Integrating base scale and expert count**

As merging gains are non-linear and saturate quickly, the authors unified the CE of all merging methods into $\mathbb{E}[L\mid N,k]=L_*+BN^{-\beta}+\frac{A_0 N^{-\gamma}}{k+b}$. The first half $L_\infty(N)=L_*+BN^{-\beta}$ is the **floor**, which monotonically decreases with base scale $N$. The second half $A(N)/(k+b)$ (where $A(N)=A_0 N^{-\gamma}$) is the **tail**, which decays at a reciprocal rate with the number of experts $k$. During fitting, points were weighted $\propto k$ to suppress high variance at small $k$, resulting in $R^2>0.98$ for all four methods. This decoupling allows one to determine the ROI of "adding an expert vs. upgrading the base model" by comparing the relative magnitudes of the floor and tail terms.

**2. Deriving $1/k$ tail from second-order Taylor expansion**

To explain why the tail is exactly $1/k$ and why disparate methods like TIES and DARE share the same curve, the authors denoted each task vector as $v_i$. After equal-weight merging, the mean of the perturbation is $c\mu$ and the covariance shrinks to $\Sigma/k$. Second-order Taylor expansion of the loss yields:

$$\mathbb{E}[L]=L(\theta_0)+cg^\top\mu+\tfrac{1}{2}c^2\mu^\top H\mu+\tfrac{c^2}{2k}\mathrm{Tr}(H\Sigma)+\mathcal{O}(k^{-3/2})$$

The first three terms are independent of $k$ and constitute $L_\infty(N)$, while the term $\frac{c^2}{2k}\mathrm{Tr}(H\Sigma)$ represents the $A(N)/k$ tail. A supporting corollary explains that the standard deviation between subsets shrinks at $1/\sqrt{k}$. Algorithms like TIES and DARE merely modify the task vectors to some $\Psi(v)$, changing constants like the mean/covariance without altering the leading-order structure.

**3. Three-point fitting + Recommended expert count $k^*$**

Since running a full $k$-grid is expensive, and the formula has only three degrees of freedom ($L_\infty$, $A$, $b$), three points are theoretically enough. The authors demonstrated that fitting with $k\in\{1,2,4\}$ can recover the full 9-point $k$-curve with minimal error. This allows for identifying the "cost-effectiveness elbow" $k^*$: the marginal gain $\Delta_k\approx A/[(k+b)(k+1+b)]\sim k^{-2}$ collapses rapidly, with the elbow consistently falling at $k\approx5\sim6$ (85% of gains are achieved with 5 experts, 90% with 6).

### Loss & Training
No new training losses are introduced. All data points come from frozen bases and 9 independently fine-tuned domain experts. Evaluation uses token-level cross-entropy on 30M held-out tokens. Merging coefficients use equal-weight normalization $\alpha_{i,k}=c/k$. Curve fitting is performed via weighted non-linear least squares with weights $\propto k$.

## Key Experimental Results

### Main Results

| Setting | Model Scale $N$ | Domain Mean CE at $k=9$ | Decrease vs. 0.5B |
|------|------------|-----------------|---------------|
| In-domain | 0.5B | 0.739 | — |
| In-domain | 7B | ~0.52 | ~30% |
| In-domain | 32B | 0.430 | 41.9% |
| Cross-domain | 0.5B→32B | Parallel Shift | Floor and tail both shrink |
| Fitting Quality | All points | $R^2>0.98$ | Uniform floor/tail residuals |

### Ablation Study

| Configuration | Key Observation | Explanation |
|------|---------|------|
| Average / TA / TIES / DARE | Same formula $R^2>0.98$ | Method differences absorbed into $L_\infty$, $A$, and $b$ |
| Candidate Pool $M=9\to 8\to 7$ | Floor unchanged, tail reduction slows | Diversity primarily lowers the tail rather than the floor |
| Three-point $k\in\{1,2,4\}$ fitting | Inference error < few times full fit | Sufficient for budget decision-making |
| Donor Order (DARE) | Whisker length at $k=8$ shrinks ~83% | Order sensitivity shrinks at $1/(k+b)$ |
| Across Backbones (LLaMA-3.2 3B / LLaMA-3 8B) | Identical 1/k tail | Formula form is transferable |

### Key Findings
- The "larger base scales better" intuition is quantified: 32B reduces CE by 41.9% at $k=9$ compared to 0.5B, with both floor and tail shrinking simultaneously.
- The elbow point generally appears at $k\approx 5\sim 6$: reaching 85% of gains requires only 5 experts. Beyond this, additional experts primarily "refresh" the data.
- Methodological differences are flattened at large scales: at $N=32B$ and $k\approx 8$, the mean CE gap between Avg/TA/TIES/DARE is $\lesssim 2\%$.
- Order sensitivity also decays at $1/(k+b)$; meticulously selecting the sequence is essentially meaningless for $k\geq 6$.

## Highlights & Insights
- Transformed "folk wisdom" into a rigorous curve with $R^2>0.98$ using 10,866 models, providing the most authoritative empirical evidence in the merging field to date.
- The decoupling of floor and tail is highly practical: the relative magnitude of $A/L$ allows for immediate judgment on the ROI of "merging one more expert vs. upgrading the base model."
- The three-point fitting method upgrades scaling laws from "post-hoc summaries" to "predictive tools," allowing budget decisions to be made without running all $k$ values.

## Limitations & Future Work
- The formula only covers equal-weight normalization; non-equal weights or learned weights (e.g., routing-based merging) can only be explained at the leading order.
- Expert capacity is treated as an implicit variable within $A(N)$, without explicitly modeling "expert strength" dimensions such as LoRA rank or fine-tuning token counts.
- Evaluation is limited to cross-entropy; the consistency of the elbow point for long-tail tasks (e.g., code/math) using downstream accuracy remains to be verified.
- While diverse, the 9 domains are within the Mixture-of-Thoughts/OpenScience datasets; extrapolation to truly heterogeneous scenarios (multilingual, multimodal, safety alignment) requires further study.

## Related Work & Insights
- **vs. Kaplan/Chinchilla Scaling Laws**: While they characterize the relationship between $(N, D, C)$ and loss, this work adds the compositional dimension of "expert count $k$."
- **vs. Yadav et al. (2024) Empirical Study**: While the latter noted that method differences vanish as the number of experts increases, this work clarifies that observation through a unified formula dominated by $L_\infty(N)$ at large $k$.
- **vs. TIES/DARE Algorithms**: This work does not compete with them but rather places them in a unified framework, showing that their preprocessing only modifies constants without changing the power-law backbone.

## Rating
- Novelty: ⭐⭐⭐⭐ First to provide a dual-axis $(N,k)$ merging scaling law with first-order provable theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10,866 models across 7 scales and 4 methods is unparalleled in merging literature.
- Writing Quality: ⭐⭐⭐⭐ Clear alignment between formulas and figures; explains the physical meaning of floor/tail effectively.
- Value: ⭐⭐⭐⭐⭐ Provides a ready-to-use "three-point fitting → budget decision" workflow for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ACL 2025\] Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](../../ACL2025/model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)

</div>

<!-- RELATED:END -->
