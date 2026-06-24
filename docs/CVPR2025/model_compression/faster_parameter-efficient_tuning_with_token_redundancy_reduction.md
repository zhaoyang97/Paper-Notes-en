---
title: >-
  [Paper Note] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)
description: >-
  [CVPR 2025][Model Compression][Parameter-Efficient Fine-Tuning] FPET (Faster Parameter-Efficient Tuning) is proposed to introduce a plug-and-play token redundancy reduction module in parameter-efficient tuning (PET). By merging approximately half of the tokens in the middle layer of Vision Transformers (ViTs) using a differentiable bipartite matching strategy, FPET achieves 20% faster inference than the original backbone and reduces GPU memory by around 40% while maintaining…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Parameter-Efficient Fine-Tuning"
  - "Token Merging"
  - "Inference Acceleration"
  - "Differentiable Matching"
  - "Straight-Through Estimator"
date: 2026-05-08
content_hash: 6873800f30090b30
---

# Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)

**Conference**: CVPR 2025  
**arXiv**: [2503.20282](https://arxiv.org/abs/2503.20282)  
**Code**: [github.com/kyk120/fpet](https://github.com/kyk120/fpet)  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning  
**Keywords**: Parameter-Efficient Fine-Tuning, Token Merging, Inference Acceleration, Differentiable Matching, Straight-Through Estimator

## TL;DR

FPET (Faster Parameter-Efficient Tuning) is proposed to introduce a plug-and-play token redundancy reduction module in parameter-efficient tuning (PET). By merging approximately half of the tokens in the middle layer of Vision Transformers (ViTs) using a differentiable bipartite matching strategy, FPET achieves 20% faster inference than the original backbone and reduces GPU memory by around 40% while maintaining comparable accuracy with state-of-the-art (SOTA) PET methods.

## Background & Motivation

**Background**: Parameter-efficient fine-tuning (PET) methods (e.g., LoRA, AdaptFormer, VPT) achieve storage efficiency by training only a small number of parameters. Some methods (e.g., RepAdapter, SSF) can avoid increasing inference latency. However, all of these methods inherit the inference latency and computational overhead of the pre-trained foundation model itself.

**Limitations of Prior Work**: Existing PET methods suffer from inference efficiency bottlenecks: they either introduce extra modules that increase inference overhead (e.g., Adapter) or can only maintain the original inference speed (e.g., parameter fusion in LoRA). This is suboptimal for scenarios requiring low latency, such as web platforms and edge devices.

**Key Challenge**: Parameter efficiency $\neq$ computational efficiency. PET solves the storage issue but not the speed issue. Existing token pruning/merging methods (such as ToMe) are designed for full fine-tuning scenarios; applying them directly to PET leads to sub-optimal merging due to their non-differentiability and the insufficient influence of Adapters.

**Key Insight**: Combine token redundancy reduction with PET by merging ~50% of the tokens at once in the middle layer of ViT, and design a differentiable matching strategy to make the merging process end-to-end optimizable.

**Core Idea**: PET + single-layer differentiable token merging = breaking the inference speed barrier of the original backbone.

## Method

### Overall Architecture

Based on the standard ViT-B/16 (12 layers), a token merging module is inserted at the 6th layer. The first 6 layers execute full computation on the original 196 tokens (ensuring that the influence of Adapters is fully manifested). At the 6th layer, 98 tokens are merged, and the subsequent 6 layers only process about 98 tokens, thereby significantly reducing the computational workload of the latter half.

### Key Designs

1. **Checkerboard Partitioning**: Unlike the alternating even-odd stripe partitioning in ToMe, a checkerboard pattern divides tokens into groups A and B. This allows each token in group B to merge with its four adjacent neighbors (up, down, left, right), whereas stripe partitioning only permits horizontal merging, thereby improving matching coverage.

2. **Differentiable Bipartite Matching**: An Adapter is first utilized to refine the key features of self-attention as $\mathbf{K'} = \mathbf{K} + s \cdot \text{ReLU}(\mathbf{K}\mathbf{W}_{down})\mathbf{W}_{up}$, followed by calculating the similarity matrix of the refined features. A crucial point is using a sigmoid function + threshold of 0.5 instead of max/top-k operations to construct a hard matching matrix, passing gradients via a Straight-Through Estimator (STE): $\tilde{\mathbf{C}}_{AB} = \hat{\mathbf{C}}_{AB} + \text{const}(\mathbf{C}_{AB} - \hat{\mathbf{C}}_{AB})$

3. **Single-Layer Merging Strategy**: Merging is performed only once at the 6th layer (instead of merging a small amount at every layer as in ToMe). This is because the Adapter's influence is not fully expressed in earlier layers, making similarities unreliable. A single merge also avoids the repeated overhead of computing similarity matrices, saving GPU memory.

### Loss & Training

The classification cross-entropy loss is identical to standard PET, and the token merging process is naturally optimized end-to-end via STE. Gradients propagate to the key-refinement Adapter but **do not propagate further** to the backbone (preventing push-pull effects between tokens from interfering with feature learning). Training runs for 100 epochs using the AdamW optimizer.

## Key Experimental Results

| Method | Acc (%) | Inference Time (ms) | FLOPs (G) | GPU Memory (GB) |
|------|---------|-------------|-----------|----------|
| Full Fine-tuning | 68.9 | 2.62 | 17.6 | 11.9 |
| VPT-Deep | 72.0 | 2.79 (+6.5%) | 18.5 | 9.8 |
| LoRA | 75.7 | 2.62 (+0%) | 17.6 | 8.4 |
| AdaptFormer | 76.2 | 2.68 (+1.5%) | 17.6 | 7.6 |
| Bi-AdaptFormer | 77.0 | 2.77 (+5.7%) | 17.7 | 7.6 |
| **FPET+LoRA** | **75.6** | **2.10 (-19.8%)** | **13.3** | **7.1** |
| **FPET+AdaptFormer** | **76.2** | **2.12 (-19.1%)** | **13.5** | **6.2** |
| **FPET+Bi-AdaptFormer** | **77.0** | **2.17 (-17.2%)** | **13.5** | **6.2** |

### Key Findings

- FPET is the only PET method that can make the inference speed **faster** than the original backbone (-19.8%).
- When combined with 5 SOTA PET methods (plug-and-play), accuracy is virtually preserved (77.0% vs 77.0%).
- FLOPs are reduced by approximately 24%, and training GPU memory is reduced by around 40-48%.
- Checkerboard vs Stripe Partitioning: Checkerboard partitioning provides better merging coverage and yields higher accuracy.
- The 6th layer is the optimal location for merging—merging too early drops accuracy, whereas merging too late provides insufficient efficiency gains.

### Detailed Performance on Three Groups of Classification Tasks

| Category | FPET+Bi-AdaptFormer | Bi-AdaptFormer | Difference |
|------|-------------------|----------------|------|
| Natural (7 tasks) | ~77% | ~77% | ≈0% |
| Specialized (4 tasks) | ~82% | ~82% | ≈0% |
| Structured (8 tasks) | ~72% | ~72% | ≈0% |
| **Inference Speed** | **2.17ms** | **2.77ms** | **-21.7%** |

Accuracy remains comparable across all three groups (Natural/Specialized/Structured), while inference is comprehensively accelerated.

## Highlights & Insights

- **Breaking the PET Inference Speed Barrier**—achieving faster inference than the original model after PET for the first time.
- **Plug-and-play Design**—can be directly stacked onto any existing PET methods (such as LoRA, AdaptFormer, Bi-LoRA, etc.).
- **Engineering Ingenuity of Differentiable Token Merging**—the combination of STE and gradient truncation ensures learnability while avoiding interference with features during backpropagation.

## Limitations & Future Work

- Evaluated only on VTAB-1K (1,000 samples per task), without testing on large-scale datasets like ImageNet.
- Fixed at merging 50% of the tokens at the 6th layer—different tasks may require adaptive merging positions and ratios.
- Applicable only to ViT architectures, and not directly suitable for CNNs or hybrid architectures.
- The impact of merging operations on fine-grained spatial tasks (such as detection and segmentation) has not been explored.

## Related Work & Insights

- **PET Methods**: LoRA, AdaptFormer, VPT, SSF, RepAdapter, Bi-LoRA, etc.
- **Token Compression**: ToMe (bipartite soft matching), DiffRate (learnt thresholds)—limited by non-differentiability.
- **Efficiency-Enhanced PET**: SynQT (decoupled learning), Pruned RepAdapter (structural pruning)—suffer from accuracy degradation.

## Rating

- Novelty: ⭐⭐⭐⭐ First effective combination of differentiable token merging and PET.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across 5 PET methods $\times$ 19 datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with a high density of information in figures and tables.
- Value: ⭐⭐⭐⭐⭐ Exceptionally high practical value, enabling plug-and-play acceleration for PET inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[CVPR 2025\] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation](parameter_efficient_mamba_tuning_via_projector-targeted_diagonal-centric_linear_.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)

</div>

<!-- RELATED:END -->
