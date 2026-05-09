---
title: >-
  [Paper Note] Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning
description: >-
  [NeurIPS 2025][Model Compression][Continual Learning] This paper proposes the Perturb-and-Merge (P&M) framework, which introduces model merging mechanisms into the continual learning paradigm. During training, random perturbations are added along the task vector direction to smooth the loss landscape; during inference, a closed-form optimal coefficient is used to compute a convex combination of the historical model and the current task model. Combined with LoRA, the framework achieves memory-efficient state-of-the-art continual learning performance.
tags:
  - NeurIPS 2025
  - Model Compression
  - Continual Learning
  - Model Merging
  - Task Vector
  - Parameter Perturbation
  - LoRA
date: 2026-05-08
content_hash: b0593b2b5e6f43fb
---

# Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.22389](https://arxiv.org/abs/2505.22389)
**Code**: [github.com/qhmiao/P-M-for-Continual-Learning](https://github.com/qhmiao/P-M-for-Continual-Learning)
**Area**: Model Compression
**Keywords**: Continual Learning, Model Merging, Task Vector, Parameter Perturbation, LoRA

## TL;DR
This paper proposes the Perturb-and-Merge (P&M) framework, which introduces model merging mechanisms into the continual learning paradigm. During training, random perturbations are added along the task vector direction to smooth the loss landscape; during inference, a closed-form optimal coefficient is used to compute a convex combination of the historical model and the current task model. Combined with LoRA, the framework achieves memory-efficient state-of-the-art continual learning performance.

## Background & Motivation

Continual learning (CL) aims to enable models to continuously absorb new knowledge from a sequence of tasks while retaining previously acquired knowledge. Existing approaches—regularization, replay memory, architecture expansion, etc.—have made notable progress, yet nearly all of them use $\theta_t^*$ directly as the inference parameters for all tasks $1$ through $t$ after completing training on task $t$. Since $\theta_t^*$ is primarily optimized for task $t$, it provides no explicit guarantee of preserving performance on historical tasks, making catastrophic forgetting likely.

Meanwhile, the model merging literature has demonstrated an intriguing capability: multiple models independently trained on different tasks from the same pretrained initialization can be merged into a single unified model via parameter interpolation or more sophisticated methods, while retaining good performance across all tasks.

Although the two paradigms operate under different protocols, they share the same core objective—learning a single model that performs well on multiple tasks. CL has the advantage that all tasks are trained along a shared optimization trajectory, so parameters are more likely to reside near a joint optimum; model merging has the advantage of providing a stable post-training ensemble mechanism. Existing CL methods have not exploited this advantage of model merging.

The core idea of this paper is to unify the two: the **training phase** follows the sequential training of CL to obtain $\theta_t^*$, while the **inference phase** does not use $\theta_t^*$ directly but instead computes an optimal convex combination with the historical inference parameters $\hat{\theta}_{t-1}$ as $\hat{\theta}_t = \hat{\theta}_{t-1} + \alpha_t \Delta\theta_t^*$. The paper further shows that performance degradation caused by merging can be mitigated by perturbing parameters along the task vector direction during training—this perturbation serves as an unbiased stochastic approximation of a regularization term and introduces no additional forward or backward passes.

## Method

### Overall Architecture
P&M consists of two stages: (1) **Train with Perturbation**: during training on each task, perturbations along the task vector direction are randomly added to the parameters with a certain probability, serving as a stochastic approximation of a Hessian quadratic regularization term; (2) **Infer after Merging**: after training, the Fisher information matrix is computed, a closed-form optimal merging coefficient $\alpha_t^*$ is solved, and a convex combination of $\hat{\theta}_{t-1}$ and $\theta_t^*$ is used as the inference parameters. LoRA is incorporated to reduce memory overhead.

### Key Designs

1. **Infer after Merging**:

    - **Function**: After completing training on each task, instead of using the trained parameters directly for inference, a weighted combination with the historical inference parameters is performed.
    - **Mechanism**: The convex combination $\hat{\theta}_t = (1 - \alpha_t)\hat{\theta}_{t-1} + \alpha_t\theta_t^*$ is equivalent to scaling the task vector: $\hat{\theta}_t = \hat{\theta}_{t-1} + \alpha_t \Delta\theta_t^*$. To find the optimal $\alpha_t$, the total loss increase of the merged model across all tasks is analyzed via a second-order Taylor expansion, yielding:
    $$\alpha_t^* = -\frac{\sum_{i=1}^{t}(\hat{\theta}_{t-1} - \theta_i^*)^\top \mathbf{H}_i(\theta_i^*) \Delta\theta_t^*}{\sum_{i=1}^{t}\Delta\theta_t^{*\top} \mathbf{H}_i(\theta_i^*) \Delta\theta_t^*}$$
    The Hessian matrix is approximated by the diagonal empirical Fisher information matrix.
    - **Design Motivation**: Scaling the task vector leaves the historical parameters $\hat{\theta}_{t-1}$ intact, adjusting only the contribution of the new task to reduce forgetting. The closed-form solution eliminates the need for hyperparameter search.

2. **Train with Perturbation**:

    - **Function**: Parameter perturbation during training reduces performance degradation caused by merging.
    - **Mechanism**: The upper bound on merging degradation contains the term $\Delta\theta_t^{*\top} \mathbf{H}_t \Delta\theta_t^*$, which can be used as a training regularizer. A second-order symmetric finite difference approximation gives:
    $$\Delta\theta_t^\top \mathbf{H}_t \Delta\theta_t \approx \frac{1}{\epsilon^2}(\mathcal{L}_t(\theta_t + \epsilon\Delta\theta_t) + \mathcal{L}_t(\theta_t - \epsilon\Delta\theta_t) - 2\mathcal{L}_t(\theta_t))$$
    Direct computation requires three forward passes. A stochastic approximation is therefore proposed: at each training step, one of the original parameters, positively perturbed parameters, or negatively perturbed parameters is sampled with probabilities $p_0, p_+, p_-$ to compute the loss, such that the expectation matches the full regularized loss: $\mathbb{E}[\tilde{\mathcal{L}}_t(\theta)] = \mathcal{L}_t(\theta)$.
    - **Design Motivation**: Perturbing in the task vector direction effectively pre-adapts the model during training to the parameter shifts that merging may induce, enlarging the flat region of the loss landscape and reducing parameter conflicts. The key advantage is zero additional computational cost—only one forward pass per step is required.

3. **LoRA-P&M (efficient implementation with LoRA)**:

    - **Function**: LoRA low-rank decomposition reduces per-task parameter storage and Fisher matrix computation costs.
    - Linear layer updates take the form $\mathbf{W}_t = \mathbf{W}_{t-1} + \mathbf{A}_t\mathbf{B}_t$, with only rank-10 LoRA modules (applied to key and value projections) being updated.
    - The perturbation operation becomes: $\theta_t + \epsilon \cdot \text{LoRA}_t$ (i.e., scaling the LoRA parameters by $1+\epsilon$).
    - **Design Motivation**: Computing $\alpha_t^*$ requires storing $\theta_i^*$ and the Fisher matrix for all historical tasks; LoRA reduces storage costs from full-parameter scale to low-rank parameter scale.

### Loss & Training
The base loss is cross-entropy. The perturbation magnitude is $\epsilon = 0.5$, and the sampling probabilities are $p_0 = p_+ = p_- = 1/3$. AdamW is used as the optimizer, with a LoRA learning rate of 1e-3, a classification head learning rate of 1e-2, and a batch size of 256. Each task is trained for 10 epochs (5 epochs for DomainNet). ViT-B/16 (pretrained on ImageNet-21K and fine-tuned on ImageNet-1K) is used as the backbone.

## Key Experimental Results

### Main Results

**Comparison with CL methods on ImageNet-R (10 tasks)**

| Method | Acc↑ | AAA↑ |
|------|------|------|
| Full Fine-Tuning | 60.57 | 72.31 |
| L2P | 71.26 | 76.13 |
| CODA-Prompt | 74.05 | 78.14 |
| InfLoRA | 74.75 | 80.67 |
| SD-LoRA | 77.34 | 82.04 |
| LoRA (baseline) | 65.72 | 76.14 |
| **LoRA-P&M** | **79.95** | **85.29** |

**Comparison with model merging methods across six benchmarks (Acc)**

| Method | INR-10 | INR-20 | INA-10 | DN*-5 | C100-10 | CUB-10 |
|------|--------|--------|--------|-------|---------|--------|
| LoRA | 65.72 | 56.35 | 44.41 | 71.81 | 72.58 | 64.82 |
| Model Averaging | 76.90 | 74.64 | 54.54 | 81.84 | 87.52 | 74.87 |
| DARE | 75.09 | 66.03 | 55.87 | 80.58 | 87.28 | 76.57 |
| CoMA | 79.34 | 75.60 | 53.24 | 83.98 | 86.95 | 74.65 |
| **P&M** | **79.95** | **76.37** | **56.57** | **84.71** | **88.45** | **78.29** |

### Ablation Study

**Ablation of P&M components across six benchmarks (Acc)**

| Configuration | INR-10 | INR-20 | INA-10 | DN*-5 | C100-10 | CUB-10 |
|------|--------|--------|--------|-------|---------|--------|
| LoRA | 65.72 | 56.35 | 44.41 | 71.81 | 72.58 | 64.82 |
| LoRA-M (merging only) | 78.35 | 74.26 | 56.16 | 81.28 | 86.57 | 74.98 |
| LoRA-M + Gaussian noise | 78.48 | 74.13 | 49.51 | 83.00 | 85.83 | 74.09 |
| **LoRA-P&M (full)** | **79.95** | **76.37** | **56.57** | **84.71** | **88.45** | **78.29** |

**Global vs. per-module merging coefficients (ImageNet-R)**

| Strategy | 5 tasks | 10 tasks | 20 tasks |
|------|---------|----------|----------|
| Per-module $\alpha$ | 80.53 | 76.82 | 74.68 |
| Global $\alpha$ (ours) | 80.88 | 78.48 | 74.13 |

### Key Findings
- Adding "Infer after Merging" alone improves LoRA from 65.72 to 78.35 on INR-10, demonstrating the substantial value of post-training merging at inference time.
- Task-vector-direction perturbation substantially outperforms random Gaussian noise (CUB-10: 78.29 vs. 74.09), confirming that the choice of perturbation direction is critical.
- P&M primarily improves performance by reducing forgetting with negligible impact on plasticity—an effect akin to "gains without costs."
- The global merging coefficient performs on par with per-module strategies, and the learned per-module coefficient values are highly similar to one another, supporting the use of a single global coefficient.
- Loss landscape visualizations show that the convex combination path consistently lies in low-loss regions, and that perturbation training enlarges the flatness and width of the low-loss basin.

## Highlights & Insights
- The chain from "theoretical derivation → practical approximation → efficient implementation" is remarkably complete: from analyzing merging degradation, to deriving the Hessian regularization term, to finite-difference approximation, to stochastic perturbation implementation—each step is supported by clear theoretical and practical justification. The final implementation achieves regularization with zero additional computational cost, a rare instance of a genuine "free lunch."
- The perspective of unifying CL and model merging—two independently developed research directions—is highly inspiring: CL provides a shared optimization trajectory that makes parameters more amenable to merging, while model merging provides an explicit knowledge retention mechanism; the two are complementary rather than substitutive.

## Limitations & Future Work
- The Fisher information matrix is approximated diagonally, which cannot fully capture the true curvature structure of the loss landscape; more accurate yet efficient curvature approximations are worth exploring.
- Per-task LoRA parameters and Fisher matrices must be stored for all historical tasks, which can still impose memory pressure when the number of tasks is large.
- Validation is limited to classification tasks; other task types such as generation and detection remain to be explored.
- The convex combination assumption limits merging flexibility; non-convex combinations (e.g., task arithmetic) may be superior in certain scenarios.

## Related Work & Insights
- **vs. SD-LoRA**: SD-LoRA is the previous state-of-the-art CL method; P&M surpasses it by 2.61% on INR-10 and 1.89% on DomainNet, with a simpler approach that requires no specialized LoRA structural design.
- **vs. CoMA/CoFIMA**: Both model merging methods are comprehensively outperformed by P&M in the CL setting, suggesting that CL's shared optimization trajectory indeed improves merging quality.
- **vs. EWC**: EWC also employs the Fisher information matrix, but uses it to regularize the training loss. P&M uses it more elegantly to derive a closed-form solution for the merging coefficient.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The idea of introducing model merging into the CL inference phase is both novel and natural; the perturbation training as a stochastic approximation of Hessian regularization is theoretically elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, varying numbers of tasks, comprehensive comparisons against both CL and model merging methods, and rich ablation and analysis experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from theory to method to experiment is rigorous and clear; the four observations supported by loss landscape visualizations are highly convincing.
- **Value**: ⭐⭐⭐⭐⭐ The method is simple, effective, theoretically grounded, and computationally inexpensive, offering a new paradigm for the CL field.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[AAAI 2026\] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](../../AAAI2026/model_compression/beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)

<!-- RELATED:END -->
