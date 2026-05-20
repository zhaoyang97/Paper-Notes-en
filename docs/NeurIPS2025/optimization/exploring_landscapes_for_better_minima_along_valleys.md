---
title: >-
  [Paper Note] Exploring Landscapes for Better Minima along Valleys
description: >-
  [NeurIPS 2025][Optimization][Loss landscape exploration] This paper proposes an optimizer adapter "E" that incorporates an exponential moving average of gradient differences $\mathbf{a}_k = \text{EMA}(\mathbf{g}_k - \mat…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Loss landscape exploration"
  - "valley tracking"
  - "large-batch training"
  - "ALTO optimizer"
  - "gradient difference"
date: 2026-05-08
content_hash: fa7ed389ce94b025
---

# Exploring Landscapes for Better Minima along Valleys

**Conference**: NeurIPS 2025
**arXiv**: [2510.27153](https://arxiv.org/abs/2510.27153)  
**Code**: [PyPI](https://pypi.org/project/alto-optimizer/)  
**Area**: Optimization / Large-Batch Training
**Keywords**: Loss landscape exploration, valley tracking, large-batch training, ALTO optimizer, gradient difference

## TL;DR
This paper proposes an optimizer adapter "E" that incorporates an exponential moving average of gradient differences $\mathbf{a}_k = \text{EMA}(\mathbf{g}_k - \mathbf{g}_{k-1})$ into the gradient update, enabling optimizers to continue exploring "valleys" in the loss landscape for lower and flatter minima after reaching a local minimum. The resulting ALTO optimizer achieves an average improvement of 2.5% in test accuracy under large-batch training.

## Background & Motivation

**Background**: Nearly all gradient-based optimizers (SGD, Adam, etc.) cease searching once a local minimum is reached. Relying solely on local information provides no guarantee that the found minimum is the lowest or the most generalizable.

**Limitations of Prior Work**: (a) Traditional optimizers become "trapped" at local minima and cannot continue exploring the valley structure of the loss landscape. (b) Large-batch training is the most direct approach to fully utilizing GPU parallelism, but suffers from the challenge of "fewer update steps"—achieving comparable test accuracy to small-batch training with far fewer parameter updates. (c) Existing learning rate scaling rules (linear scaling $\eta_k \propto |\mathcal{Z}_k|$, square-root scaling) are constrained by task-dependent critical thresholds at very large batch sizes.

**Key Challenge**: In the loss landscape, optimizers must macroscopically be "captured" by large-scale valleys (walking along them) while microscopically escaping small-scale sharp minima—a duality that traditional optimizers address only partially.

**Goal**: To design an optimizer that continues exploring along valleys after reaching a local minimum, thereby finding lower and flatter minima.

**Key Insight**: The observation that $-\nabla\|\nabla f(\theta_k)\|^2 = -2\mathbf{H}_k \bar{\mathbf{g}}_k \approx \bar{\mathbf{g}}_k - \bar{\mathbf{g}}_{k-1}$ reveals that the gradient difference direction naturally possesses the property of repelling sharp minima and being attracted to flat ones.

**Core Idea**: Augmenting the optimizer's gradient with the term $\alpha \cdot \text{EMA}(\mathbf{g}_k - \mathbf{g}_{k-1})$ to microscopically repel sharp minima and macroscopically track valleys.

## Method

### Overall Architecture
The E-adaptor is a plug-in module compatible with any gradient-based optimizer. The core modification replaces the gradient $\mathbf{g}_k$ with $\mathbf{g}_k + \alpha \mathbf{a}_k$, where $\mathbf{a}_k = \beta_1 \mathbf{a}_{k-1} + (1-\beta_1)(\mathbf{g}_k - \mathbf{g}_{k-1})$ is the EMA of gradient differences. The standard Adam/Lamb update then proceeds as usual.

### Key Designs

1. **Directional Analysis of Gradient Differences**:

    - Function: Proves that $\mathbf{g}_k - \mathbf{g}_{k-1}$ is an ideal direction for escaping sharp minima.
    - Mechanism: Four stages of optimizer behavior near a minimum (①② descending → ②③ crossing → ③④ ascending → ④⑤ decelerating) are considered. The sign of the inner product $\langle \theta_k - \theta_{k-1}, \mathbf{g}_k - \mathbf{g}_{k-1} \rangle$ is analyzed:
        - In stages ②③ and ③④ (crossing the minimum): positive inner product → **accelerates** escape
        - In stages ①② and ④⑤: negative inner product → **decelerates** (captured by valley)
    - Comparison with $-\nabla\|\nabla f\|^2$: the latter yields a negative inner product in stage ②③ (counterproductively decelerating), making it inferior to gradient differences.
    - Design Motivation: $-\nabla\|\nabla f\|^2 = -2\mathbf{H}_k \bar{\mathbf{g}}_k$; a large Hessian $\mathbf{H}_k$ indicates a sharp minimum → large steps facilitate escape; a small $\mathbf{H}_k$ indicates a flat minimum → small steps lead to capture.

2. **ALTO Algorithm (Adapted Lamb with Exploration)**:

    - Function: Embeds the E-adaptor into the Lamb optimizer.
    - Core iterations:
        - $\mathbf{a}_k = \beta_1 \mathbf{a}_{k-1} + (1-\beta_1)(\mathbf{g}_k - \mathbf{g}_{k-1})$ (EMA of acceleration term)
        - $\mathbf{m}_k = \beta_2 \mathbf{m}_{k-1} + (1-\beta_2)(\mathbf{g}_k + \alpha \mathbf{a}_k)$ (first moment)
        - $\mathbf{v}_k = \beta_3 \mathbf{v}_{k-1} + (1-\beta_3)[\mathbf{g}_k + \alpha \mathbf{a}_k]^2$ (second moment)
        - Bias correction → $\hat{\mathbf{m}}_k, \hat{\mathbf{v}}_k$
        - $\mathbf{r}_k = \hat{\mathbf{m}}_k / (\sqrt{\hat{\mathbf{v}}_k} + \varepsilon_1) + \lambda_k \theta_k$
        - Layer-wise regularized update: $\theta_{k+1}^{(i)} = \theta_k^{(i)} - \eta_k \mathbf{r}_k^{(i)} \phi(\|\theta_k^{(i)}\|) / (\|\mathbf{r}_k^{(i)}\| + \varepsilon_2 \phi(\|\theta_k^{(i)}\|))$
    - Key constraint: $|\alpha| < 1/(1-\beta_1)$, derived from the stability condition of the continuous-time ODE.

3. **Why $\mathbf{a}_k$ is Added to $\mathbf{g}_k$ Rather Than $\mathbf{m}_k$**:

    - Adding to $\mathbf{m}_k$ implies that gradient differences are on the same order of magnitude as gradients → severe oscillation or ineffectiveness.
    - Adding to $\mathbf{g}_k$ means the resulting momentum contains EMA₂ (double exponential moving average) of gradient differences with weights $(1-\beta)^2 \beta^{k-i} \binom{k-i+1}{1}$ → smoother and more stable.
    - EMA₂ accumulates the most informative directions during the early stages of training (when gradients decay rapidly) and serves as a "navigation signal" in later stages.

4. **Effect of the Sign of $\alpha$**:

    - $\alpha > 0$: faster convergence but less exploration; suitable for small-batch training.
    - $\alpha < 0$: more exploration, finds flatter minima, but slower convergence; suitable for large-batch training.
    - Recommendation: small-batch $\alpha = 0.5, \beta_1 = 0.01$; large-batch $\alpha = -5, \beta_1 = 0.99$.

### Convergence Analysis
- **Theorem 1 (Non-convex)**: Under assumptions of $L$-smoothness and unbiased gradient with bounded variance, for $T \geq O(G_\infty^{1.5} \epsilon^{-2})$, $\frac{1}{T+1}\sum_{k=0}^T \mathbb{E}\|\nabla f_k(\theta_k)\|^2 \leq 4\epsilon^2$. This improves upon Lamb's $O(\epsilon^{-4})$ rate.
- **Theorem 2 (Convex)**: $R(T) \leq O(\sqrt{T})$, on par with Adam, but under more relaxed conditions ($\beta_2^2/\beta_3 < 1$ rather than $\beta_{1,k} = \beta_k \lambda^k$).

## Key Experimental Results

### Large-Batch ImageNet Training (ResNet-50, 90 epochs)

| Batch Size | Adam | AdamW | AdaBelief | Lamb | **ALTO** |
|---------|------|-------|-----------|------|----------|
| 1K | 73.08 | 75.65 | 73.32 | 77.06 | **77.22** |
| 2K | 73.08 | 74.93 | 73.48 | 77.11 | **77.25** |
| 4K | 73.32 | 74.65 | 73.41 | 76.92 | **77.35** |
| 8K | 73.11 | 74.40 | 73.14 | 76.89 | **77.10** |
| 16K | 73.09 | 74.10 | 73.00 | 76.66 | **76.87** |
| 32K | 72.50 | 73.57 | 72.89 | 76.42 | **76.70** |

### CIFAR-10/100 + ImageNet (ResNet-20/34)

| Dataset | Batch | SGD | Adam | Lamb | **ALTO** |
|--------|-----|-----|------|------|----------|
| CIFAR-10 | 128 | 91.85 | 89.88 | 90.89 | 91.24 |
| CIFAR-10 | 16384 | 80.86 | 87.34 | 83.56 | **88.83** |
| CIFAR-100 | 128 | 64.93 | 64.35 | 61.29 | **65.74** |
| CIFAR-100 | 16384 | 44.20 | 54.91 | 56.06 | **57.78** |
| ImageNet | 256 | 70.64 | 65.06 | 69.17 | 69.95 |
| ImageNet | 4086 | 49.35 | 54.96 | 70.34 | **70.83** |

### Training Time Comparison (VGG-16, CIFAR-100, batch=16384)

| Target Accuracy | ALTO (s) | Lamb (s) | Speedup |
|---------|---------|---------|--------|
| 20% | 137 | 196 | 1.43× |
| 40% | 334 | 409 | 1.23× |
| 60% | 608 | 865 | 1.42× |

### Key Findings
- ALTO outperforms the SOTA baseline (Lamb) across **all 17 CV+NLP experiments**.
- The advantage is more pronounced at large batch sizes: at batch=16384, ALTO surpasses Lamb by 5.27% on CIFAR-10.
- ALTO at large-batch ImageNet (batch=4086, 70.83%) **exceeds** SGD at small-batch (batch=256, 70.64%).
- In GPT-2 training, ALTO achieves a test perplexity of 78.37, substantially better than Lamb's 83.13.
- To reach the same accuracy, ALTO can **save 29.68%** of computation time.

## Highlights & Insights
- **Constraint derivation from ODE perspective**: The discrete optimizer is cast as a continuous-time ODE, and the stability condition on eigenvalue real parts yields the constraint $|\alpha| < 1/(1-\beta_1)$, bridging theoretical stability and practical hyperparameter selection.
- **EMA₂ = memory of early directions**: The EMA of EMA allows the optimizer to leverage informative directions accumulated during early training—when gradients decay rapidly—as a "navigation guide" in later stages, which is particularly beneficial for large-batch training.
- **Large batch = more reliable gradient differences**: Larger batches produce more accurate gradient estimates → more reliable gradient differences $\mathbf{g}_k - \mathbf{g}_{k-1}$ → greater advantage for ALTO, which explains why improvements are most pronounced in the large-batch regime.

## Limitations & Future Work
- Five additional hyperparameters are introduced ($\alpha, \beta_1, \beta_2, \beta_3, \varepsilon$); although the authors claim that only $\beta_1$ and $\eta$ typically require tuning, the overall tuning burden is still increased.
- Each step incurs a minor additional cost for maintaining the EMA of gradient differences, resulting in slightly longer epoch times compared to Lamb.
- The non-convex convergence analysis relies on relatively strong assumptions (Assumptions 3.3–3.5), particularly the monotonicity assumption, which may not hold in practice.
- Experiments are conducted on a single node with 4×A100 GPUs; communication bottlenecks in multi-node distributed settings remain unevaluated.
- The relationship between the sign of $\alpha$ and batch size is empirically motivated, lacking theoretical justification.

## Related Work & Insights
- **vs. Lamb [You et al., 2020]**: Lamb serves as the foundation of ALTO, with layer-wise regularization inherited from it. ALTO with the exploration term $\mathbf{a}_k$ outperforms Lamb at all batch sizes.
- **vs. AdaBelief [Zhuang et al., 2020]**: AdaBelief replaces $g_k^2$ with $(g_k - m_{k-1})^2$ in the adaptive learning rate, focusing on gradient prediction accuracy. ALTO instead focuses on the directional information contained in gradient differences.
- **vs. SAM/Sharpness-Aware methods**: SAM explicitly minimizes loss landscape sharpness. ALTO implicitly favors flat minima via gradient differences, without requiring additional forward passes.
- **vs. LR warmup/cosine schedules**: Scheduling strategies operate in the temporal dimension. ALTO's exploration operates in the geometric/directional dimension; the two are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of using gradient differences as a valley-tracking direction is original, and the EMA₂ design is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across CV/NLP/RL tasks, multiple models, and multiple batch sizes, with 17 consistently positive results.
- Writing Quality: ⭐⭐⭐⭐ The directional analysis diagrams and tables (Table 1, Fig. 2) are highly intuitive.
- Value: ⭐⭐⭐⭐ Directly applicable to large-batch training; ALTO can serve as a drop-in replacement for Lamb.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[NeurIPS 2025\] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](../../ICLR2026/optimization/when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](../../ICLR2026/optimization/exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)

</div>

<!-- RELATED:END -->
