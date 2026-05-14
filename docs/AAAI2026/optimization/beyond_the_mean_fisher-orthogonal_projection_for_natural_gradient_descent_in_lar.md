---
title: >-
  [Paper Note] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training
description: >-
  [AAAI2026][Optimization][Natural Gradient Descent] This paper proposes Fisher-Orthogonal Projection (FOP), which supplements variance information by orthogonally projecting sub-batch gradient differences under the Fisher…
tags:
  - "AAAI2026"
  - "Optimization"
  - "Natural Gradient Descent"
  - "Fisher Information Matrix"
  - "KFAC"
  - "Large Batch Training"
  - "Second-Order Optimization"
date: 2026-05-08
content_hash: 944a52eeb0fbbb0c
---

# Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training

**Conference**: AAAI2026
**arXiv**: [2508.13898](https://arxiv.org/abs/2508.13898)
**Code**: [yishunlu-222/fop](https://github.com/yishunlu-222/fop)
**Area**: Optimization
**Keywords**: Natural Gradient Descent, Fisher Information Matrix, KFAC, Large Batch Training, Second-Order Optimization

## TL;DR
This paper proposes Fisher-Orthogonal Projection (FOP), which supplements variance information by orthogonally projecting sub-batch gradient differences under the Fisher metric, enabling the second-order optimizer KFAC to remain effective in ultra-large batch training and achieving up to ×7.5 speedup.

## Background & Motivation
- Modern GPUs feature massive memory (e.g., 192GB on AMD MI300X), supporting batch sizes in the tens of thousands, yet most optimizers degrade significantly at such scales.
- **First-order methods**: As batch size increases, gradient noise diminishes, depriving SGD/Adam/AdamW of the stochastic noise that aids escape from sharp minima, thereby hurting generalization.
- **Second-order methods**: The Fisher matrix of KFAC becomes severely ill-conditioned at large batch sizes, requiring very high damping for stability; however, excessive damping erases curvature information, causing the optimizer to degenerate into standard gradient descent.
- Existing improvements (SENG, SP-NGD, etc.) introduce additional hyperparameters, rely on stale statistics, or require task-specific tuning, limiting their generality.

## Core Problem
How can the curvature advantage of natural gradient methods be recovered in ultra-large batch training without introducing additional hyperparameters, while maintaining the generalization accuracy of small-batch training?

## Method

### Core Idea
Naively averaged gradients discard valuable directional information across sub-batches. FOP splits a mini-batch into two sub-batches, computes their respective gradients $g_1, g_2$, retains the mean gradient $g_{\text{avg}}$ as the primary descent direction, and extracts the component of the gradient difference $g_{\text{diff}}$ that is orthogonal to the mean under the Fisher metric, using it as a supplementary update direction.

### Procedure

**1. Dual sub-batch gradient computation**: The mini-batch is split in half, and gradients are computed separately:

$$g_{\text{avg}} = \frac{1}{2}(g_1 + g_2), \quad g_{\text{diff}} = g_1 - g_2$$

**2. Fisher-orthogonal projection**: $g_{\text{diff}}$ is orthogonalized with respect to $g_{\text{avg}}$ under the Fisher inner product to remove redundant information:

$$s_{\text{proj}} = \frac{g_{\text{diff}}^\top F g_{\text{avg}}}{g_{\text{avg}}^\top F g_{\text{avg}} + \epsilon}, \quad g_{\text{diff}}^{\perp} = g_{\text{diff}} - s_{\text{proj}} \cdot g_{\text{avg}}$$

ensuring $\langle g_{\text{avg}}, g_{\text{diff}}^{\perp} \rangle_F = 0$.

**3. Adaptive mixing coefficient $\beta^*$**: A second-order Taylor expansion of the total loss is performed, and the mixing ratio minimizing the surrogate objective is solved per layer:

$$\beta^* = \frac{g_{\text{avg}}^\top F^{-1} g_{\text{diff}}^{\perp}}{(g_{\text{diff}}^{\perp})^\top F^{-1} g_{\text{diff}}^{\perp}}$$

When the orthogonal component points in an unfavorable direction, $\beta^* \to 0$ and FOP automatically reduces to standard KFAC, guaranteeing safety.

**4. Per-layer adaptive step size $\eta_\ell^*$**: An optimal step size is computed independently for each layer, adjusted automatically according to curvature and gradient alignment:

$$\eta_\ell^* = \frac{g_{\ell,\text{tot}}^\top F_\ell^{-1} g_{\ell,\text{comb}}}{g_{\ell,\text{comb}}^\top F_\ell^{-1} g_{\ell,\text{comb}}}$$

**5. Final update**: $d_\ell = \eta_0 \eta_\ell^* F_\ell^{-1} g_{\ell,\text{comb}}$

### KL Norm Analysis
- The KL norm of the FOP update decomposes into: a base term ($\mathcal{O}(1/\lambda^2)$) and cross/orthogonal terms ($\mathcal{O}(1/\lambda)$).
- During early training when $\beta < 0$, the cross term is negative and partially cancels the base term, creating a safety margin that permits lower damping $\lambda$.
- This provides the theoretical explanation for why FOP remains stable at large batch sizes without requiring high damping.

### Distributed FOP
- GPUs are partitioned into primary and secondary groups; each group performs AllReduce independently to obtain $g_1$ and $g_2$.
- One GPU per layer is designated as the "curvature expert," responsible for updating and inverting that layer's Fisher matrix.
- After computing the FOP update, the expert GPU broadcasts the result to all processes.
- Communication overhead is comparable to standard data parallelism, requiring only one additional AllReduce.

## Key Experimental Results

### CIFAR-10 + ResNet-18 (5 seeds)

| Batch Size | SGD | AdamW | KFAC | FOP |
|---|---|---|---|---|
| 2048 | 58ep/743s | 61ep/768s | 37ep/589s | **29ep/475s** |
| 4096 | 73ep/458s | 73ep/454s | 34ep/271s (×1.69) | **22ep/182s (×2.52)** |
| 8192 | — | — | 71ep/296s (×1.54) | **35ep/158s (×2.91)** |
| 16384 | — | — | 99ep/186s (×2.46) | **58ep/121s (×3.78)** |
| 32768 | — | — | — | **60ep/91s (×5.05)** |
| 50000 | — | — | — | **82ep/84s (×5.43)** |

Time to reach 91% accuracy; "—" indicates the threshold was not reached. FOP is the only method that converges at BS ≥ 32768.

### ImageNet-100 + T2T-ViT (3 seeds)
- To reach 80.6% Top-1: FOP at BS=4096 requires only 49ep/27.8min, achieving **×10.48** speedup over AdamW (BS=512).
- KFAC achieves ×6.45 speedup under the same configuration; FOP consistently outperforms.

### ImageNet-1K + ResNet-50 (3 seeds)
- To reach 75.9% Top-1: FOP at BS=8192 requires 40ep/335min, achieving **×7.50** speedup over SGD at BS=1024.
- SGD, Shampoo, LAMB, and KFAC all fail to reach the threshold at BS > 1024.

### CIFAR-LT + ResNet-32 (long-tail, 5 seeds)
- CIFAR-10-LT (IF=100): FOP achieves an error rate of 26.65%, 1.4% lower than the baseline and 1.94% lower than KFAC.
- CIFAR-100-LT (IF=100): FOP achieves an error rate of 58.97%, **3.3%** lower than the baseline.
- KFAC performs worse than the baseline at IF=100, while FOP outperforms all methods across the board.

## Highlights & Insights
1. **Theoretical elegance**: The Fisher-orthogonal projection design has clear geometric intuition, and the KL norm analysis provides a rigorous theoretical explanation.
2. **Adaptive safety**: The automatic degeneracy of $\beta^*$ ensures fallback to standard KFAC when the orthogonal component is unhelpful, preventing any degradation.
3. **Extreme scalability**: FOP is the only optimizer that converges at batch size 50,000 (the full CIFAR-10 dataset).
4. **No additional hyperparameters**: No task-specific tuning is required; learning rate scaling follows the linear rule.
5. **Plug-and-play**: Available as an open-source pip package, integrable into existing training pipelines with a single line of code.

## Limitations & Future Work
- Validation is limited to vision tasks (CNNs and ViTs); large-scale language model experiments are absent.
- The approximation of Hessian ≈ Fisher in the second-order Taylor expansion may be inaccurate during early training or for highly nonlinear models.
- The distributed implementation requires an even number of GPUs for primary/secondary partitioning.
- Additional Fisher matrix computation and inversion overhead remain compared to first-order methods.
- FOP is only combined with the KFAC framework; integration with other second-order methods (e.g., Shampoo's Kronecker structure) is unexplored.

## Related Work & Insights

| Method | Order | Large Batch Capability | Extra Hyperparams | ImageNet-1K 75.9% |
|---|---|---|---|---|
| SGD/AdamW | First | Poor (fails >4096) | None | 71ep/2511min (BS=1024) |
| LAMB | First | Moderate | Layer-wise LR | 67ep/2493min (BS=1024) |
| KFAC | Second | Moderate (fails >1024) | Damping | 35ep/1337min (BS=1024) |
| SENG | Second | Moderate | Low-rank params | 41ep (BS=4096) |
| SP-NGD | Second | Moderate | Multiple heuristics | 74.8–75.3% |
| **FOP** | **Second** | **Strong** | **None** | **40ep/335min (BS=8192)** |

**Broader implications**:
- The paradigm of using sub-batch variance information for orthogonal correction may generalize to other optimizers (e.g., Fisher variants of Adam).
- FOP's advantages on long-tailed distributions suggest that its curvature-aware updates are naturally suited to class-imbalanced settings, warranting exploration in long-tailed tasks such as medical imaging.
- The primary/secondary grouping strategy in the distributed design may offer insights for gradient aggregation in federated learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The Fisher-orthogonal projection idea is original; the adaptive $\beta$ and per-layer step size designs are well-developed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers CNNs/ViTs, multiple datasets, long-tailed scenarios, and multi-seed repetition; NLP experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and experiments are well-organized.
- **Value**: ⭐⭐⭐⭐ — Practically significant for large-batch training; plug-and-play design lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](../../NeurIPS2025/optimization/natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](../../ICLR2026/optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](../../CVPR2026/optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](../../ICLR2026/optimization/pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)

</div>

<!-- RELATED:END -->
