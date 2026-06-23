---
title: >-
  [Paper Note] On Optimal Hyperparameters for Differentially Private Deep Transfer Learning
description: >-
  [ICLR 2026][AI Safety][Paper Note] This paper systematically investigates the critical hyperparameters of clipping bound $C$ and batch size $B$ in Differentially Private (DP) transfer learning. It demonstrates that prevalent heuristics—such as "use small $C$ for strong privacy" and "use large batch sizes for a fixed number of steps"—are erroneous. Based
tags:
  - ICLR 2026
  - AI Safety
date: 2026-05-08
content_hash: 7c6e1c60e745eaeb
---
# On Optimal Hyperparameters for Differentially Private Deep Transfer Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=V3fEo612nE](https://openreview.net/forum?id=V3fEo612nE)  
**Code**: To be confirmed  
**Area**: AI Safety / Differential Privacy  
**Keywords**: Differential Privacy, Transfer Learning, Gradient Clipping, Batch Size, Hyperparameter Tuning

## TL;DR
This paper systematically investigates the critical hyperparameters of clipping bound $C$ and batch size $B$ in Differentially Private (DP) transfer learning. It demonstrates that prevalent heuristics—such as "use small $C$ for strong privacy" and "use large batch sizes for a fixed number of steps"—are erroneous. Based on a theory of optimal clipping via MSE decomposition and an analysis of cumulative DP noise, the authors explain why $(C, B, \eta)$ should be jointly tuned according to the "learning problem difficulty."

## Background & Motivation
**Background**: Training large models on sensitive data currently relies on DP transfer learning as the SOTA approach—pre-training a backbone on public data and fine-tuning it on private tasks using DP-SGD / DP-Adam. Due to the high computational overhead of DP optimization, practitioners often **only tune the learning rate for each task**, while treating batch size $B$ and clipping bound $C$ as constants (e.g., $C{=}1$, large batch size) that remain stable across privacy levels, backbones, and compute budgets.

**Limitations of Prior Work**: Through heatmaps (Figure 1), the authors find that fixing $C$ and $B$ at static values systematically degrades performance. Settings that perform best on "easy tasks" significantly deteriorate on "difficult tasks," and vice versa. Worse, this damage is concentrated on difficult samples and their dominant classes, hurting exactly where DP training struggles most.

**Key Challenge**: There is a **clear misalignment** between existing theory and empirical observations. Theory has long suggested that stronger privacy (smaller $\epsilon$) requires a smaller $C$; however, experiments show the opposite—**larger $C$ is often better under strong privacy**. Similarly, batch size tuning rules from a "fixed steps" perspective (minimizing batch size to optimize signal-to-noise ratio per step) fail in "fixed epoch (compute-constrained)" scenarios.

**Goal**: The problem is split into two sub-questions: (1) what determines the optimal clipping bound $C^*$ and why does strong privacy prefer a large $C$; (2) what constitutes the optimal batch size under a fixed epoch budget.

**Key Insight**: The authors introduce the unifying concept of "learning problem difficulty," determined by the privacy budget $\epsilon$, available data and compute, dataset difficulty, transfer complexity, and backbone capacity. All these factors ultimately influence optimal hyperparameters by **altering the gradient norm distribution**.

**Core Idea**: Clipping is viewed as "gradient reweighting," and an MSE decomposition is used to derive an optimal $C^*$ dependent on the true gradient distribution. Batch selection is established based on "cumulative DP noise + a minimum step lower bound," anchoring the tuning of $(C, B, \eta)$ to the problem difficulty rather than fixed default values.

## Method

### Overall Architecture
Rather than proposing a new algorithm, this paper provides a systematic theoretical and empirical analysis of $(C, B)$ behavior in DP transfer learning, culminating in actionable tuning guidelines (Table 1). The logic follows two main tracks: For the **clipping bound $C$**, the authors derive the optimal $C^*$ that minimizes the MSE of the clipped gradient at each step (Theorem 5.2) and link this MSE to optimization progress (Theorem 5.4, Corollary 5.5), proving that reducing MSE tightens the bound on per-step loss reduction. For **batch size $B$**, the authors address the failure of old rules in fixed-epoch scenarios by proposing the use of "cumulative DP noise $\sigma\sqrt{T}$ + minimum step lower bound" to predict the optimal batch size.

The experimental setup is standardized: ViT-Base / ViT-Tiny (high/low capacity backbones, ImageNet-21k pre-trained) with FiLM parameter-efficient fine-tuning (tuning only normalization layer scale/bias and the head, ~0.5–1.5% trainable parameters) for image tasks; DistilBERT + LoRA for text; and WideResNet-16-4 for training from scratch. The PRV accountant is used to calibrate the noise multiplier $\sigma$, with an exhaustive grid search for learning rate, batch size, and clipping bound.

### Key Designs

**1. Optimal clipping bound derived from MSE decomposition depends on true gradient distribution, not just $\epsilon$**

To explain the counter-intuitive phenomenon where "strong privacy prefers larger $C$," the authors decompose the Mean Squared Error between the clipped gradient $\tilde g$ and the true gradient $g$ under standard (non-normalized) DP-SGD into **clipping bias** and **DP noise variance**. The per-coordinate noise variance is $\sigma^2 C^2$, where a larger $C$ injects more noise but reduces clipping bias. Theorem 5.2 provides the optimal clipping constant $C^*$ that minimizes this MSE (under Assumption 5.1, assuming no mini-batch sampling noise):

$$C^* = \frac{N_{C^*}^{\top} G_{C^*}}{N_{C^*}^{\top} N_{C^*} + \sigma^2 d},$$

where $d$ is the gradient dimension, $\mathcal{I}_C = \{i : \|g_i\| > C\}$ is the set of indices for clipped samples, $G_C := \sum_{i\in\mathcal{I}_C} g_i$, and $N_C := \sum_{i\in\mathcal{I}_C} \frac{g_i}{\|g_i\|}$. Crucially, $C^*$ depends not only on $\sigma$ but also on the **direction and norm distribution of true gradients via $G_C$ and $N_C$**. While increasing $\sigma$ (fixing other variables) lowers $C^*$, larger gradient norms push $C^*$ higher. Experiments (Figure 3) show that as privacy tightens, the gradient norm distribution shifts right—easy samples become harder and their norms increase. This shift outweighs the direct effect of $\sigma$, causing $C^*$ to increase.

**2. Linking per-step MSE to optimization progress**

The authors argue that "optimal clipping" corresponds to "better optimization." Under $L$-smooth loss and step size $\eta \le 1/L$ (Assumption 5.3), Theorem 5.4 provides an upper bound on per-step loss improvement:

$$\mathbb{E}[L(\theta_{t+1})\mid\theta_t] \le L(\theta_t) - \frac{\eta}{2}\|\nabla L(\theta_t)\|_2^2 + \frac{\eta}{2}\,\mathrm{MSE}_t(C).$$

Since $\mathrm{MSE}(C) \ge 0$, Corollary 5.5 follows: minimizing $\mathrm{MSE}(C)$ minimizes the upper bound on per-step loss improvement. This connects "optimal clipping" to convergence—choosing $C^*$ directly tightens the bound on the loss reduction achievable at each step.

**3. Clipping as gradient reweighting: explaining task difficulty focus**

To explain what $C$ affects at a granular level, the authors interpret clipping as **reweighting across samples/classes**. The retained weight for class $y$ under clipping bound $C$ is defined as:

$$w_y(C) = \frac{1}{n_y}\sum_{i:\,y_i=y}\min\!\Big(1, \frac{C}{\|g_i\|_2}\Big),$$

where $n_y$ is the number of samples with label $y$. A small $C$ prioritizes easy samples/classes and suppresses difficult ones; a large $C$ makes weighting more uniform. Figure 4 shows that as problem difficulty increases, the gap between different $C$ values widens—small $C$ severely downweights difficult class signals, while large $C$ preserves them.

**4. Optimal batch size under fixed epoch: minimum steps + minimized cumulative DP noise**

Addressing the failure of the fixed-step rule in fixed-epoch scenarios, the authors use **cumulative DP noise standard deviation** $\sigma\sqrt{T}$ (where $T = E\cdot N/B$). The strategy is two-fold: (a) ensure a **minimum step lower bound** (e.g., at least 20 steps for a specific dataset); (b) satisfying this, **select the smallest batch size that keeps the cumulative noise near-optimal (within the $\sigma\sqrt{T}$ plateau)**. Under strong privacy (small $\epsilon$), $\sigma\sqrt{T}$ remains nearly constant over a wide range of batch sizes, allowing moderate batch sizes to outperform large ones.

### Loss & Training
No new training objectives are introduced. Experiments primarily use DP-Adam (with decoupled learning rate and clipping, Algorithm 1) and DP-SGD; $\delta = 10^{-5}$ is fixed. The noise multiplier is calibrated by the PRV accountant under add-remove adjacency. An empirical observation: the optimal learning rate under DP-Adam often scales with $\sqrt{B}$, suggesting that $(C, B, \eta)$ must be jointly tuned.

## Key Experimental Results

### Main Results
Testing across 4 datasets (SUN397, Cassava, CIFAR-100, 20 Newsgroups), privacy levels, model sizes, and compute budgets confirms the main conclusions:

| Condition Change | Shift in Opt. Clipping/Batch | Evidence |
|:---|:---|:---|
| Lower $\epsilon$ (Tight Privacy) | Increase $C$, Decrease $B$ | Figure 2 Left: Opt. $C$ is larger for small $\epsilon$ |
| Stronger Backbone / Easier Dataset | Use smaller $C$ | Figure 2 Right: ViT-Base prefers smaller $C$ |
| Weaker Backbone / Harder Dataset | Try larger $C$ | Figure 2 Right: ViT-Tiny prefers larger $C$ |
| Fewer Epochs (Compute Bound) | Avoid large $B$ (to keep steps) | Figure 7: Must meet min. step threshold |
| Auto-clipping AUTO-S (Tiny $C$) | Matches tuned $C$ only on easy tasks | Figure 5: Significantly worse on hard tasks |

### Ablation Study
Comparing fixed $(C, B)$ vs. difficulty-aware joint tuning:

| Configuration | Key Observation | Detail |
|:---|:---|:---|
| Fixed $C$, Fixed $B$ across tasks | Optimal settings for easy tasks fail on hard ones | Figure 1 heatmap |
| Old Rule (Per-step noise for $B$) | Does not saturate; always selects full batch | Figure 6: Ineffective under fixed epochs |
| Cumulative Noise $\sigma\sqrt{T}$ + Min Steps | Moderate batch sizes win under strong privacy | Figure 7: Explained by the plateau region |

### Key Findings
- **The root cause for larger $C$ under strong privacy is the right-shift in gradient distribution**: As difficulty increases, the gradient norm distribution shifts right and becomes more dispersed (Figure 3), pushing $C^*$ higher even as per-step noise increases.
- **Clipping damage scales asymmetrically with difficulty**: Small $C$ is relatively harmless for easy tasks but severely penalizes difficult classes in hard tasks.
- **Batch size should not be blindly maximized**: Under fixed compute, moderate batch sizes are often superior to large ones.
- **Joint tuning is essential**: Learning rate often masks the effects of $C$ and $B$; only exhaustive joint search reveals the nuanced interactions of DP-specific hyperparameters.

## Highlights & Insights
- **Unifying concept of "Problem Difficulty"**: Consolidates privacy budget, data/compute, dataset difficulty, and backbone capacity into their effect on gradient norm distributions.
- **MSE Decomposition to Optimization Progress**: Provides a theoretical bridge from per-step statistical quality to convergence bounds.
- **Cumulative Noise $\sigma\sqrt{T}$ over Per-step Noise**: A simple change in metric fixes systematic biases in batch tuning rules for compute-constrained settings.
- **Fairness perspective via "Clipping = Reweighting"**: Interpreting $C$ as a knob for class-wise gradient weights explains why DP training is inherently unfair to difficult or minority classes.

## Limitations & Future Work
- The main experiments focus on parameter-efficient (FiLM) fine-tuning for image classification. While DP-LoRA and text classification are tested, broader generalizability depends on those specific contexts.
- Theorem 5.2 relies on strong assumptions (no sampling noise, fixed direction) and $C^*$ depends on the true gradient, making it an explanatory tool rather than a practical automated algorithm.
- The minimum step threshold varies by dataset, and a general method to predict this limit is not provided.
- Full-parameter tuning is an outlier: large models have larger gradient norms, leading to preferences for exceptionally large $C$ regardless of task difficulty.

## Related Work & Insights
- **vs. Koloskova et al. (2023)**: They provided convergence guarantees for DP-SGD clipping, but their $C^*$ relies on unknowns (e.g., loss at optimum) and cannot explain the "large $C$ for strong privacy" trend. This paper uses MSE decomposition to offer an explanation based on observable gradient distributions.
- **vs. Ponomareva et al. (2023)**: They suggested finding the smallest $C$ that minimally impacts utility without noise; this paper shows this only works when noise doesn't shift the distribution significantly.
- **vs. Bu et al. (2023) AUTO-S**: They use tiny $C$ to avoid tuning; this paper proves this fails under high difficulty/strong privacy.
- **vs. De et al. (2022) / Panda et al. (2024)**: They recommend large batch sizes under fixed steps; this paper argues for moderate batch sizes under fixed epochs.

## Rating
- Novelty: ⭐⭐⭐⭐ Uses MSE decomposition and cumulative noise to provide a unifying and counter-intuitive explanation for old problems.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale joint grid searches across datasets, models, and budgets.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments, though some key detailed results are located in the appendix.
- Value: ⭐⭐⭐⭐⭐ Directly challenges default practices of fixing $C$ and $B$, providing actionable joint tuning guidelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text](pe-sgd_differentially_private_deep_learning_via_evolution_of_gradient_subspace_f.md)
- [\[ICLR 2026\] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)
- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](differentially_private_domain_discovery.md)
- [\[ICLR 2026\] Private Rate-Constrained Optimization with Applications to Fair Learning](private_rate-constrained_optimization_with_applications_to_fair_learning.md)
- [\[ICLR 2026\] PateGAIL++: Utility Optimized Private Trajectory Generation with Imitation Learning](pategail_utility_optimized_private_trajectory_generation_with_imitation_learning.md)

</div>

<!-- RELATED:END -->
