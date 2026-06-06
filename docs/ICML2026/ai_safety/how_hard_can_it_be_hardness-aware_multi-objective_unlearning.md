---
title: >-
  [Paper Note] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning
description: >-
  [ICML 2026][AI Safety][machine unlearning] The trade-off between "forgetting vs. retaining" is formulated as a "first-order convex optimization problem with per-step constraints." The dot product of retain/forget gradien…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "machine unlearning"
  - "multi-objective optimization"
  - "constrained optimization"
  - "gradient dot product"
  - "collateral forgetting"
date: 2026-05-08
content_hash: f190ae9257945b5d
---

# How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning

**Conference**: ICML 2026  
**arXiv**: [2606.02119](https://arxiv.org/abs/2606.02119)  
**Code**: https://github.com/aoi3142/HAMU  
**Area**: AI Safety / Machine Unlearning  
**Keywords**: machine unlearning, multi-objective optimization, constrained optimization, gradient dot product, collateral forgetting

## TL;DR
The trade-off between "forgetting vs. retaining" is formulated as a "first-order convex optimization problem with per-step constraints." The dot product of retain/forget gradients, $\kappa = \bm{g_r}\cdot\bm{g_f}$, simultaneously serves as a hardness metric, an update direction switch, and an early stopping condition. This approach demonstrates superior stability over baselines such as GA, GDiff, SCRUB, and KL on CIFAR-10/ResNet-20 and Llama-2-7B/WaterDrum-TOFU.

## Background & Motivation

**Background**: Machine unlearning aims to erase the influence of a specific forget dataset $D_f$ from a trained model while preserving performance on the retain dataset $D_r$. Dominant approaches involve gradient ascent (GA, NPO) on the forget loss, fine-tuning (FT) on the retain loss, or a weighted combination of both (GDiff, KL, SCRUB).

**Limitations of Prior Work**: Weighted combination methods neither guarantee that the forget data is erased to a "specified degree" nor ensure that the retain capability is not inadvertently damaged (a phenomenon termed "collateral forgetting"). Consequently, users cannot pre-specify an instruction like "erase to quality $Q$ while minimizing retain loss."

**Key Challenge**: Whether the two objectives conflict depends on the similarity between $D_f$ and $D_r$. In the extreme case of $D_f = D_r$, it is impossible to unlearn one without damaging the other. Existing works neither quantify this conflict nor explicitly utilize such a metric within their algorithms.

**Goal**: (1) Provide a computable scalar to measure "how hard the unlearning task is"; (2) Propose an algorithm that guarantees a forget improvement $\geq Q$ while minimizing retain degradation; (3) Actively stop when conflicts become irreconcilable.

**Key Insight**: Starting from a first-order analysis of a single gradient descent step, the change in forget loss caused by a small step on $D_r$ is determined by the sign of the dot product $\nabla L(D_f)\cdot\nabla L(D_r)$. A more positive dot product indicates more coupled objectives, while a negative dot product facilitates easier unlearning.

**Core Idea**: Each unlearning step is formulated as a constrained convex problem: minimize retain degradation within a local neighborhood of radius $R$, subject to forget improvement $\geq Q$. The closed-form solution naturally identifies $\kappa = \bm{g_r}\cdot\bm{g_f}$ as the "hardness," and uses the threshold of $\kappa$ to switch between "standard gradient descent" and a "projection-based correction."

## Method

### Overall Architecture
HAMU (Hardness-Aware Multi-objective Unlearning) decomposes the unlearning process into $T$ iterative constrained subproblems. Each step takes the current weights $\bm{w}_t$ and batches from retain/forget data as input, performing three operations: (1) Estimating batch gradients $\bar{\bm{g}}_{\bm{r}}, \bar{\bm{g}}_{\bm{f}}$ and hardness $\bar\kappa$; (2) Comparing $\bar\kappa$ against two thresholds $\bar\kappa_1, \bar\kappa_2$ to decide whether to stop, apply a direct update, or apply a corrected update; (3) Updating $\bm{w}_{t+1} = \bm{w}_t + \Delta\bm{w}$. The process outputs the unlearned model after $T$ steps or upon early stopping due to $\bar\kappa > \bar\kappa_2$.

The methodology comprises a first-order convex subproblem, two dual variants (HAMU-Q / HAMU-U), and a layer-wise parallel implementation.

### Key Designs

1. **Hardness Metric $\kappa$ and First-order Constrained Subproblem**:
    - **Function**: Uses a scalar to simultaneously characterize "objective conflict," the "optimal lower bound of retain degradation," and "feasibility."
    - **Mechanism**: Applying first-order expansion within a local region $\|\Delta\bm{w}\|\leq R$, where $\Delta L(D_r)\approx \bm{g_r}\cdot\Delta\bm{w}$ and $\Delta L(D_f)\approx \bm{g_f}\cdot\Delta\bm{w}$. The subproblem is defined as:
      $$\min\ \bm{g_r}\cdot\Delta\bm{w}\ \text{s.t.}\ \bm{g_f}\cdot\Delta\bm{w}\geq Q,\ \|\Delta\bm{w}\|\leq R$$
      Under the feasibility condition $Q\leq R\|\bm{g_f}\|$, the optimal cost $F_r^*$ is monotonically non-decreasing with respect to $\kappa = \bm{g_r}\cdot\bm{g_f}$. Thus, $\kappa$ serves as a natural measure of hardness.
    - **Design Motivation**: Previous hardness metrics were post-hoc heuristics (training curves, influence functions) unusable during optimization. Here, $\kappa$ is theoretically equivalent to the "optimal lower bound of retain degradation" and costs only one gradient dot product to compute.

2. **Switching between Direct vs. Correction Updates**:
    - **Function**: Automatically selects between "following $-\bm{g_r}$" or "incorporating $\bm{g_f}$ to satisfy constraints" based on current difficulty.
    - **Mechanism**: Defines threshold $\kappa_1 = -Q\|\bm{g_r}\|/R$. If $\kappa \leq \kappa_1$ (easy), the direct update $\Delta\bm{w} = -\tfrac{R}{\|\bm{g_r}\|}\bm{g_r}$ is sufficient, equivalent to SGD on the retain set while naturally satisfying forgetting constraints. If $\kappa > \kappa_1$ (hard), direct update violates the constraint, requiring a correction:
      $$\Delta\bm{w}^* = \frac{Q}{\|\bm{g_f}\|^2}\bm{g_f} - \sqrt{R^2 - Q^2/\|\bm{g_f}\|^2}\,\frac{\bm{g_r}_\perp}{\|\bm{g_r}_\perp\|}$$
      where $\bm{g_r}_\perp$ is the orthogonal component of $\bm{g_r}$ relative to $\bm{g_f}$.
    - **Design Motivation**: Existing methods either strictly follow weighted gradients (failing in hard regions) or strictly apply gradient ascent (damaging retain unnecessarily in easy regions). HAMU switches automatically between regimes with a threshold $\kappa_1$ determined solely by $Q, R, \|\bm{g_r}\|$.

3. **Early Stopping $\kappa_2$ and Layer-wise Parallelization**:
    - **Function**: Stops when improvement is impossible without retain degradation and adapts the algorithm for large models via layer-wise constraints.
    - **Mechanism**: Adding $\bm{g_r}\cdot\Delta\bm{w}\leq 0$ (no retain degradation) yields the feasibility condition $\kappa \leq \kappa_2 \triangleq \sqrt{(\|\bm{g_r}\|\|\bm{g_f}\|)^2 - Q^2\|\bm{g_r}\|^2/R^2}$. If $\kappa > \kappa_2$, collateral forgetting is unavoidable, and the process breaks. Parallelization splits $\bm{w}$ into $\ell$ layers and assigns layer-wise quotas $Q_i = \tfrac{\|\bm{g_r}^{(i)}\|\|\bm{g_f}^{(i)}\|}{\sum_j\|\bm{g_r}^{(j)}\|\|\bm{g_f}^{(j)}\|}\cdot Q$.
    - **Design Motivation**: (a) $\kappa_2$ provides a theoretically grounded criterion to stop unlearning before excessive damage occurs; (b) Layer-wise independence aligns with empirical sensitivity differences and enables GPU parallelization.

### Loss & Training
The model maintains its original cross-entropy loss without introducing new learnable parameters. The primary hyperparameter is the learning rate $\eta$, where $R = \eta\|\bar{\bm{g}}_{\bm{r}}\|$ is implicitly set. Users select $Q$ (HAMU-Q) or $U$ (HAMU-U) based on requirements. To ensure first-order approximation validity, gradients are clipped to $\|\bm{g}\|_{\max}=1$ with $Q < \eta$.

## Key Experimental Results

### Main Results
CV tasks utilize ResNet-20 pre-trained on CIFAR-10; LLM tasks use Llama-2-7B-chat fine-tuned on WaterDrum-TOFU. Baselines include FT, GA, GDiff, KL, and SCRUB. Metrics track $\Delta L_f$ (forget improvement, higher is better) and $-\Delta L_r$ (retain improvement, higher is better).

| Scenario | Key Observation | Conclusion |
|----------|-----------------|------------|
| CIFAR-10, $\rho=0$ (easy) | HAMU/GDiff improve both objectives; GA/KL degrade retain; FT/SCRUB fail to forget | Both HAMU and GDiff perform well in easy regions |
| CIFAR-10, $\rho=0.75$ (hard) | Only HAMU variants achieve visible improvement without destroying the other objective | HAMU dominates in hard regions |
| Llama-2-7B / Semantic TOFU | Mean $\bar\kappa=6.1\times10^{-4}$ (semantic) vs $4.0\times10^{-4}$ (random); HAMU-Q improves both | Results are consistent in LLM scenarios |

### Ablation Study

| Configuration | Key Finding | Note |
|---------------|-------------|------|
| Full HAMU-Q | Both $\Delta L_f, -\Delta L_r$ are significantly positive | Standard |
| Global vs. Layer-wise Constraints | Performance is worse or negative with global constraints | Layer-wise is critical for LLM scalability |
| Disabling Stopping Criterion | $-\Delta L_r$ begins to decline after a certain epoch at $\rho=0.5$ | $\kappa_2$ triggers near the inflection point |
| Varying $Q/\eta$ | Relationship between $\Delta L_f$ and $Q/\eta$ is perfectly linear ($R^2=0.999$) | The first-order layer-wise approximation is high-fidelity |

### Key Findings
- **$\kappa$ correlates strongly with human-defined hardness $\rho$** (Pearson 0.994 for HAMU-Q), proving it accurately captures data similarity. Even on non-HAMU baselines, the trend of performance degradation as $\rho$ increases holds.
- **Pareto-front control**: Increasing $Q/U$ accelerates unlearning but eventually impacts retain performance, allowing users to map the Pareto-front using a single algorithm.
- **Unlearning becomes harder over time**: $\bar\kappa$ increases monotonically with epochs, reflecting shrinking feasible directions and justifying the $\kappa_2$ stopping condition.
- **Baselines struggle in hard regions**: GA/KL tend to destroy the retain set, while FT/SCRUB often fail to unlearn; only HAMU with explicit constraints achieves gains in both.

## Highlights & Insights
- **Constraint Formulation**: Shifting from "weighted multi-objective tuning" to a "constrained subproblem" turns implicit weights into interpretable, explicit quotas ($Q$).
- **Triple Role of $\kappa$**: Hardness metric, update switch, and stop condition—all derived from a simple gradient dot product.
- **Adaptive Allocation**: The layer-wise quota $Q_i \propto \|\bm{g_r}^{(i)}\|\|\bm{g_f}^{(i)}\|$ can be generalized to any scenario requiring budget allocation across layers (e.g., quantization or pruning).
- **Geometric Intuition**: The correction update $\Delta\bm{w}^*$ ensures a minimum projection on the forget direction before utilizing the remaining "budget" for retain optimization.

## Limitations & Future Work
- **First-order Approximation**: Approximation errors may occur with large learning rates or large Hessian eigenvalues (requiring smaller $\eta$ in LLM experiments).
- **Local Stopping**: As a per-iteration criterion, $\kappa_2$ may trigger slightly later than the global optimum; soft stopping ($\bar\kappa > \bar\kappa_2 - \varepsilon$) is suggested.
- **Inferred Target Quality**: Automatically mapping high-level metrics (e.g., MIA success rate) back to $Q$ remains an open problem.
- **Variance Analysis**: While empirically robust to batch size, a rigorous analysis of the variance in $\bar\kappa$ estimation is lacking.

## Related Work & Insights
- **vs. SCRUB / KL**: These use weighted or distillation targets that fail to balance objectives in hard scenarios.
- **vs. GDiff**: GDiff lacks an awareness of infeasibility, whereas HAMU recognizes conflict via $\kappa_2$.
- **vs. Newton-style Methods**: Second-order certified unlearning requires Hessian inversion, which is infeasible for LLMs. HAMU’s first-order approach with local trust regions is practical for large-scale models.

## Rating
- Novelty: ⭐⭐⭐⭐ Refreshing perspective on constrained convex unlearning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive CV/LLM testing with 5 baselines and sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from theory to engineering.
- Value: ⭐⭐⭐⭐ Provides a deployable algorithm for quota-driven unlearning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](../../AAAI2026/ai_safety/easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[ICML 2026\] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks](unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)
- [\[ICML 2026\] Flatness-Aware Stochastic Gradient Langevin Dynamics](flatness-aware_stochastic_gradient_langevin_dynamics.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Training-Free Coverless Multi-Image Steganography with Access Control](training-free_coverless_multi-image_steganography_with_access_control.md)

</div>

<!-- RELATED:END -->
