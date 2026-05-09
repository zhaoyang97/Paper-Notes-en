---
title: >-
  [Paper Note] LaCoOT: Layer Collapse through Optimal Transport
description: >-
  [ICCV2025][optimal transport] This paper proposes LaCoOT, an optimal transport-based regularization strategy that minimizes the Max-Sliced Wasserstein distance between intermediate feature distributions within a network during training, enabling the removal of entire layers post-training while maintaining performance and significantly reducing model depth and inference time.
tags:
  - ICCV2025
  - optimal transport
  - layer removal
  - depth reduction
  - Max-Sliced Wasserstein distance
  - model compression
date: 2026-05-08
content_hash: d30b02da3cdec1a3
---

# LaCoOT: Layer Collapse through Optimal Transport

**Conference**: ICCV2025
**arXiv**: [2406.08933](https://arxiv.org/abs/2406.08933)
**Code**: [VGCQ/LaCoOT](https://github.com/VGCQ/LaCoOT)
**Area**: Other
**Keywords**: optimal transport, layer removal, depth reduction, Max-Sliced Wasserstein distance, model compression

## TL;DR

This paper proposes LaCoOT, an optimal transport-based regularization strategy that minimizes the Max-Sliced Wasserstein distance between intermediate feature distributions within a network during training, enabling the removal of entire layers post-training while maintaining performance and significantly reducing model depth and inference time.

## Background & Motivation

### Computational Challenges of Foundation Models

The rise of foundation models (CLIP, Stable Diffusion, DiT, etc.) has introduced substantial computational overhead. Training a generative model incurs carbon emissions equivalent to driving 10 km, while generating 10k samples corresponds to 160 km. Although training costs are high, inference costs across countless end users multiply further once models are released publicly.

### Limitations of Prior Work

Mainstream complexity reduction methods include:
- **Unstructured pruning**: Yields little practical speedup on general-purpose hardware.
- **Structured pruning** (channel/filter pruning): Marginal benefits on modern parallel computing architectures; the true bottleneck is the **critical path length**.
- **Knowledge distillation to shallow student models**: Target architecture is unknown and may incur performance degradation.

### Challenges in Network Depth Reduction

Existing depth reduction methods primarily merge adjacent linear layers by removing nonlinear activations:
- **Layer Folding**: Replaces ReLU with PReLU to assess whether nonlinearities can be discarded.
- **EGP**: Prioritizes pruning layers with low nonlinearity utilization based on entropy metrics.
- **NEPENTHE**: Improves the entropy estimator used in EGP.
- **EASIER**: Evaluates the impact of removing nonlinearities using a validation set.

However, these methods share fundamental problems:
1. In ResNet architectures, when the second convolutional layer uses padding, no closed-form solution exists for merging two adjacent convolutional layers.
2. Layer fusion after removing activations at residual connections is infeasible.
3. They rely on iterative search, incurring large computational overhead (EASIER requires 34 training runs per target).

## Method

### Mechanism

Rather than relying on activation linearization and layer merging, LaCoOT directly minimizes the discrepancy between input and output feature distributions of each network block during training. After training, blocks with the smallest discrepancy can be removed entirely, as they approximate identity mappings.

### Regularization Strategy

The DNN is represented as a cascade of blocks, each with an input feature distribution and an output feature distribution.

**OT Regularization Term**: The average Max-Sliced Wasserstein distance across all blocks is used as a training regularizer. The Max-Sliced Wasserstein distance is the maximum-projection variant of the 2-Wasserstein distance, admits a closed-form solution, and is free from the curse of dimensionality.

**Overall Training Objective**: Task loss + regularization coefficient × OT regularization term. The regularization coefficient controls the strength: larger values allow more layers to be removed but may affect performance.

### Layer Removal Procedure

1. Train the network with regularization and evaluate performance.
2. Compute the Max-Sliced Wasserstein distance for each block.
3. Identify the block with the smallest distance and replace it with an Identity mapping.
4. Re-evaluate; if the performance drop does not exceed the threshold, continue removing the next block.
5. Stop when the performance drop exceeds the threshold.

### Theoretical Analysis

**Soft 1-Lipschitz Constraint**: The regularization is equivalent to imposing an orthogonality constraint on the block Jacobians, realizing a per-block soft 1-Lipschitz constraint. Prior work shows that 1-Lipschitz constraints do not limit the classification expressiveness of a network and can improve generalization.

**Stationary Point Analysis**: The loss and the regularization term act in opposition — unconstrained DNNs tend to introduce unnecessary distributional changes in intermediate layers. Regularization guides the network toward the "shortest path," eliminating redundant distributional shifts.

**Triangle Inequality Lower Bound**: The distance between the input distribution and the ground-truth distribution defines a tight lower bound on the regularization value, ensuring the network retains the minimum necessary distributional transformation capacity without underfitting.

## Key Experimental Results

### Experimental Setup

- **Classification models**: ResNet-18, MobileNetV2, Swin-T
- **Classification datasets**: CIFAR-10, Tiny-ImageNet-200, PACS, VLCS, Flowers-102, DTD, Aircraft
- **Generative model**: DiT-XL/2 (fine-tuned on ImageNet)
- **Baselines**: Layer Folding, EGP, NEPENTHE, EASIER

### Main Results

Critical path length and performance on CIFAR-10 + ResNet-18:

| Method | Top-1 Acc. | MACs (M) | Inference Time (ms) | Training Time |
|--------|-----------|----------|---------------------|---------------|
| Original | 91.77% | 140.19 | 7.90 | 30 min |
| Layer Folding | 88.76% | 147.53 | 9.89 | 160 min |
| EGP | 90.64% | 140.19 | 7.62 | 376 min |
| NEPENTHE | 89.26% | 140.19 | 7.71 | 288 min |
| EASIER | 90.35% | 140.19 | 7.07 | 533 min |
| **LaCoOT** | **90.99%** | **64.69** | **4.78** | **40 min** |

Key findings:

1. **MACs reduced by half**: LaCoOT reduces MACs from 140M to 65M (−54%), while baseline methods show virtually no reduction due to the inability to merge layers.
2. **40% inference speedup**: 4.78 ms vs. 7.90 ms.
3. **Highest training efficiency**: 40 min vs. 533 min for EASIER (13× faster).
4. **Minimal performance loss**: 90.99% vs. the original 91.77%, a drop of only 0.78%.

### Generative Model: DiT-XL/2

Fine-tuning DiT-XL/2 on ImageNet for only 5k steps:

- Upon removing 2 DiT blocks, LaCoOT achieves an FID-50k of 56.2, compared to 118.6 without regularization (LaCoOT halves the FID).
- Generated image quality is substantially better preserved — without regularization, content collapses entirely.

### Key Findings (Cross-Architecture)

- Swin-T on Tiny-ImageNet-200: LaCoOT outperforms baselines by 10% at the same critical path length.
- MobileNetV2: On this already highly optimized architecture, EASIER performs marginally better, but requires 20× the training time.
- EGP fails completely on MobileNetV2, pruning the final layer before the classification head in the first iteration and severing the information flow.

### Ablation Study

- **Without regularization**: Max-Sliced Wasserstein distance is not a reliable block importance metric and performs no better than random removal.
- **Regularization coefficient = 5**: Nearly half of the blocks can be removed with negligible performance loss.
- LaCoOT's metric outperforms both Block Influence (individual removal trials) and random baselines.

## Highlights & Insights

### Highlights

- **Entire layers are removed** rather than only nonlinearities, genuinely reducing critical path length.
- **Architecture-agnostic**: Applicable to ResNet, Swin, MobileNet, DiT, and other architectures.
- **Single training run**: An order of magnitude more efficient than iterative methods such as EASIER.
- **Scalable to foundation models**: Applicable to DiT with only a small number of fine-tuning steps.

### Limitations & Future Work

- Limited effectiveness on already under-fitted efficient architectures (e.g., MobileNetV2 on Tiny-ImageNet).
- Applicable only to blocks with matching input and output dimensions; cross-dimension blocks require future exploration via Gromov-Wasserstein distance.
- No retraining is performed after layer removal; incorporating a healing phase could further recover performance.

## Personal Reflections

1. Using optimal transport to quantify layer "redundancy" is conceptually natural — if the input and output distributions of a layer are nearly identical, it is effectively an identity mapping.
2. LaCoOT is complementary to structured pruning (width reduction): combining depth reduction with width reduction could yield even more efficient compression.
3. The side effects of the regularization merit further investigation — does the 1-Lipschitz constraint affect model robustness on out-of-distribution data?
4. The preliminary results on generative models (DiT) are highly promising and warrant validation at larger scales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](../../NeurIPS2025/others/variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)
- [\[AAAI 2026\] CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](../../AAAI2026/others/cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)

</div>

<!-- RELATED:END -->
