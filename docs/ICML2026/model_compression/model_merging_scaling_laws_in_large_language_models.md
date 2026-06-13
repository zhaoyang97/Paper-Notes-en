---
title: >-
  [Paper Note] Model Merging Scaling Laws in Large Language Models
description: >-
  [ICML 2026][Model Compression][model merging] The authors empirically establish, using 10,866 merged models, a dual-axis power law of the form $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$: the base model size $N$ determines…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "model merging"
  - "scaling law"
  - "power law"
  - "task arithmetic"
  - "TIES/DARE"
date: 2026-05-08
content_hash: bbc9fef6268ee2d8
---

# Model Merging Scaling Laws in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2509.24244](https://arxiv.org/abs/2509.24244)  
**Code**: https://github.com/InfiXAI/Merging-Scaling-Law (available)  
**Area**: LLM Pretraining / Model Merging / Scaling Law  
**Keywords**: model merging, scaling law, power law, task arithmetic, TIES/DARE

## TL;DR
The authors empirically establish, using 10,866 merged models, a dual-axis power law of the form $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$: the base model size $N$ determines the floor, the number of experts $k$ determines the tail, and four mainstream merging methods (Average, TA, TIES, DARE) all share the same curve. This transforms the questions of "how many experts to merge" and "when to stop merging" into predictable, budgetable engineering problems.

## Background & Motivation
**Background**: Model merging has become a low-cost "expert integration" paradigm following multi-task SFT. Linear weighting (Model Soups, Task Arithmetic) and preprocessed variants (TIES, DARE) are widely used in LLMs, LoRA adapters, and related scenarios.

**Limitations of Prior Work**: Merging is still largely heuristic—trying different subsets, orders, and normalization coefficients, which is costly and lacks the guidance of a scaling law as in pretraining. Given a target loss, no one can answer in advance "how many experts are needed" or "is it more cost-effective to double the base or add another expert".

**Key Challenge**: The benefit curve of merging is clearly nonlinear, but exhibits regularity (steep early gains, later saturation). Without an analytic form to describe this curve, engineering practice relies on exhaustive search, wasting GPU resources.

**Goal**: (1) Find a compact formula that simultaneously characterizes the effects of $N$ (base parameter count) and $k$ (number of merged experts); (2) demonstrate its validity across different merging algorithms, backbones, and both in-domain and cross-domain settings; (3) provide a practical method to extrapolate the entire curve from just three measurements.

**Key Insight**: Treat merging as "equal-weight averaging of several task vectors". Under a second-order Taylor expansion, the variance of equal-weight averaging shrinks at a rate of $1/k$, and this variance enters the loss via the Hessian as the $A(N)/k$ term. The authors thus anticipate a "floor + 1/k tail" structure, which is validated at scale.

**Core Idea**: Use a unified "floor + 1/(k+b) tail" power law to describe the CE curves of all merging methods, integrating both base size and expert count into a single formula, making merging a budget-aware, predictable process.

## Method

### Overall Architecture
The authors fine-tune nine domain experts (algebra, analysis, geometry, discrete, number theory, code, chemistry, physics, biology) from the same base across the Qwen2.5 series (0.5B/1.5B/3B/7B/14B/32B/72B), covering both in-domain and cross-domain evaluations. For each $(N,k)$ combination, all or a uniform sample of $\binom{9}{k}$ expert subsets are traversed, and four merging algorithms (Average, TA, TIES, DARE) are used to synthesize models and measure token-level CE, resulting in a grid of 10,866 merged models. A weighted nonlinear least squares fit is then performed to a curve of the form $\mathbb{E}[L\mid N,k]=L_\infty(N)+A(N)/(k+b)$, where $L_\infty(N)=L_*+BN^{-\beta}$ and $A(N)=A_0 N^{-\gamma}$, with $R^2$ and residual structure used for validation.

### Key Designs

1. **Unified floor+tail power law**:

    - **Function**: Uses a single formula to simultaneously capture the effects of base size and expert count on merging loss.
    - **Mechanism**: $\mathbb{E}[L\mid N,k]=L_*+BN^{-\beta}+\frac{A_0 N^{-\gamma}}{k+b}$, where the floor term $L_*+BN^{-\beta}$ decreases monotonically with $N$, and the tail term $A_0 N^{-\gamma}/(k+b)$ decays reciprocally with $k$; fitting uses weights $\propto k$ to stabilize early $k$ noise, and all methods achieve $R^2>0.98$ across all slices.
    - **Design Motivation**: Integrates the observations "larger bases merge better" and "diminishing returns with more experts" into a single expression, enabling direct comparison of the two terms for budget decisions ("add another expert vs. increase base size").

2. **Derivation of 1/k tail from second-order Taylor expansion**:

    - **Function**: Explains why all merging algorithms exhibit a $1/k$ tail under equal-weight normalization.
    - **Mechanism**: Each task vector is denoted $v_i$; after equal-weight merging, the perturbation mean is $c\mu$, covariance is $\Sigma/k$; a second-order Taylor expansion of the loss yields $\mathbb{E}[L]=L(\theta_0)+cg^\top\mu+\frac{1}{2}c^2\mu^\top H\mu+\frac{c^2}{2k}\mathrm{Tr}(H\Sigma)+\mathcal{O}(k^{-3/2})$, where the first three terms condense to $L_\infty(N)$ and the last term is $A(N)/k$; a corollary further shows that the std among subsets shrinks as $1/\sqrt{k}$. Preprocessing algorithms like TIES/DARE are absorbed as modifications to $\Psi(v)$, not altering the leading-order form.
    - **Design Motivation**: Provides a theoretical explanation for the $1/k$ behavior, not just empirical fitting; also explains why diverse implementations like TIES and DARE ultimately fall on the same curve.

3. **Three-point fitting + recommended $k^*$ budget algorithm**:

    - **Function**: Extrapolates the entire $k$-curve using only $k\in\{1,2,4\}$, and provides the "most cost-effective number of experts" $k^*$.
    - **Mechanism**: The formula has only three degrees of freedom ($L_\infty$, $A$, $b$), so three points suffice in theory; empirical results show three-point fitting recovers the full 9-point curve and stably estimates $k^*$ at $5\sim 6$, corresponding to the elbow position $\Delta_k\approx A/[(k+b)(k+1+b)]\sim k^{-2}$.
    - **Design Motivation**: In real scenarios, a full $k$-grid is costly; the three-point method enables a "measure a small batch, then decide budget" workflow, turning merging from trial-and-error into measurement plus extrapolation.

### Loss & Training
No new training losses are introduced; all data points are from frozen bases plus independently fine-tuned nine domain experts, evaluated on 30M held-out tokens using token-level cross-entropy. Merging coefficients use equal-weight normalization $\alpha_{i,k}=c/k$. Fitting uses weighted nonlinear least squares with weights $\propto k$ to suppress high variance at small $k$.

## Key Experimental Results

### Main Results

| Setting | Model Size $N$ | Domain Mean CE at $k=9$ | Reduction vs 0.5B |
|---------|----------------|-------------------------|-------------------|
| In-domain | 0.5B | 0.739 | — |
| In-domain | 7B | ~0.52 | ~30% |
| In-domain | 32B | 0.430 | 41.9% |
| Cross-domain | 0.5B→32B | Synchronously shifted down | Both floor and tail shrink |
| Fit Quality | All points | $R^2>0.98$ | Uniform residuals for floor/tail |

### Ablation Study

| Configuration | Key Observation | Notes |
|---------------|----------------|-------|
| Average / TA / TIES / DARE | Same formula $R^2>0.98$ | Method differences absorbed into $L_\infty$, $A$, $b$ constants |
| Candidate pool $M=9\to 8\to 7$ | Floor nearly unchanged, tail reduction lessens | Diversity mainly lowers tail, not floor |
| Three-point $k\in\{1,2,4\}$ fit | 9-point curve inference error < several times full fit | Three-point method sufficient for budget decisions |
| Different donor order (DARE) | Whisker length at $k=8$ shrinks by ~83% | Order sensitivity decays as $1/(k+b)$ |
| Cross-backbone (LLaMA-3.2 3B / LLaMA-3 8B) | Same $1/k$ tail | Formula is transferable |

### Key Findings
- "Larger bases merge better" is quantitatively established: 32B vs 0.5B at $k=9$ yields a 41.9% drop in CE, with both floor and tail shrinking—providing both lower asymptotic performance and reducing required expert count.
- The elbow typically appears at $k\approx 5\sim 6$: 85% of the gain is achieved with just 5 experts, 90% with 6; beyond this, adding experts mostly just "adds data".
- Method differences are flattened at large scales: At $N=32B$, $k\approx 8$, the mean CE gap among Avg/TA/TIES/DARE is $\lesssim 2\%$, and merge-to-merge variance shrinks as $\sim 1/k$ to a common floor.
- Order sensitivity also decays as $1/(k+b)$; for $k\geq 6$, careful ordering is essentially meaningless.

## Highlights & Insights
- By empirically validating "folk wisdom" with 10,866 real merged models and achieving $R^2>0.98$, this work far surpasses previous merging papers in scale and systematics, making it the most authoritative empirical evidence in the field.
- The decoupled floor and tail perspective is highly practical: the relative magnitude $A/L$ instantly indicates whether "adding another expert" or "increasing base size" yields higher ROI, directly benefiting compute allocation in industry.
- The three-point fitting method upgrades scaling law from a "post hoc summary" to a "predictive tool", allowing the elbow to be locked in without running all $k$; this "measurement-extrapolation" approach can transfer to other compositional studies (e.g., number of RAG retrieval sources, ensemble model count).

## Limitations & Future Work
- The formula only covers equal-weight normalization; for non-equal or learned weights (e.g., routing/optimization-based merges), it only explains the leading order, with differences absorbed as finite-$k$ bias.
- Expert capacity is treated as a latent variable within $A(N)$, without explicit modeling of LoRA rank, fine-tuning token count, or other "expert strength" dimensions; the paper acknowledges this as a natural extension.
- Evaluation uses only cross-entropy, which is still some distance from downstream task accuracy; whether the elbow for long-tail tasks like code/math is consistent remains to be verified.
- Although the nine domains are diverse, they all come from the Mixture-of-Thoughts/OpenScience data series; extrapolation to truly heterogeneous scenarios (e.g., multilingual, multimodal, safety alignment) remains to be tested.

## Related Work & Insights
- **vs Kaplan/Chinchilla pretraining scaling laws**: Those characterize the relationship between $(N, D, C)$ and loss; this work adds the "number of experts $k$" as a compositional dimension, showing it and $N$ are decoupled axes.
- **vs Yadav et al. (2024) empirical study**: The latter empirically noted "method differences shrink with more experts"; this work explains it with a unified formula: "common $L_\infty(N)$ dominates at large $k$, $A(N)/(k+b)$ tail dominates at small $k$".
- **vs TIES/DARE and other merging algorithms**: This work does not compete with them but "places them in the same framework", showing that such preprocessing only slightly modifies the mean/covariance of task vectors, without changing the power law skeleton.

## Rating
- Novelty: ⭐⭐⭐⭐ First to present a dual-axis $(N,k)$ merging scaling law with provable first-order theory; the formula itself is concise, but the approach is a natural extension in the scaling law lineage.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10,866 merged models, 9 domains, 7 scales, 4 methods, cross-backbone validation—scale is nearly unmatched in merging literature.
- Writing Quality: ⭐⭐⭐⭐ Clear integration of formulas and figures, with thorough explanation of the physical meaning of floor/tail; minor repetition between in-domain/cross-domain sections.
- Value: ⭐⭐⭐⭐⭐ Directly provides a "three-point fit → budget decision" actionable workflow, with immediate engineering significance for industrial merging, LoRA repository management, and expert routing.

## Related Papers

- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[CVPR 2026\] Model Merging in the Essential Subspace](../../CVPR2026/llm_pretraining/model_merging_in_the_essential_subspace.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
