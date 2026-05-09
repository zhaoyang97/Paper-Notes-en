---
title: >-
  [Paper Note] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression
description: >-
  [ICLR2026][Model Compression][dataset compression] This paper proposes the Dataset Color Quantization (DCQ) framework, which reduces color redundancy at the dataset level through three mechanisms — chromaticity-aware clustering, attention-guided palette allocation, and texture-preserved palette optimization — achieving storage compression while maintaining training performance.
tags:
  - ICLR2026
  - Model Compression
  - dataset compression
  - color quantization
  - palette sharing
  - attention guidance
  - texture preservation
date: 2026-05-08
content_hash: 5933de5560c695b0
---

# Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression

**Conference**: ICLR2026
**arXiv**: [2602.20650](https://arxiv.org/abs/2602.20650)
**Code**: N/A
**Area**: Model Compression
**Keywords**: dataset compression, color quantization, palette sharing, attention guidance, texture preservation

## TL;DR
This paper proposes the Dataset Color Quantization (DCQ) framework, which reduces color redundancy at the dataset level through three mechanisms — chromaticity-aware clustering, attention-guided palette allocation, and texture-preserved palette optimization — achieving storage compression while maintaining training performance.

## Background & Motivation

### State of the Field

**State of the Field**: The storage demands of large-scale image datasets pose significant challenges in resource-constrained environments. Existing dataset compression methods (dataset pruning, distillation) reduce data volume by discarding samples, yet overlook **intra-image color redundancy** — many pixels share nearly identical colors (e.g., smooth regions such as sky and walls). Existing color quantization (CQ) methods suffer from two major issues:

**Image-property-based CQ** (e.g., K-Means): lacks semantic guidance, leading to blurred semantic boundaries and uniform bit allocation between background and foreground.

**Model-aware CQ** (e.g., ColorCNN): preserves recognition accuracy but introduces abrupt texture/edge discontinuities — when ColorCNN quantizes CIFAR-10 to 4 colors, inference accuracy of a pretrained model reaches 77%, yet training on quantized data yields only 58%.

Key insight: existing methods are **inference-oriented** (optimizing pretrained model recognition on quantized images), whereas this paper is the first to propose **training-oriented** dataset color quantization.

## Method

### Overall Architecture
DCQ consists of three core modules forming a unified dataset-level color compression pipeline.

### Key Designs
**1. Chromaticity-Aware Clustering (CAC)**:
- Clusters dataset images via K-Means ($k=20$ clusters) using shallow features $\psi_{\text{shallow}}(x)$ from a pretrained model.
- Images within the same cluster share a palette, balancing cross-image consistency with quantization fidelity.
- Shallow features capture color distribution patterns (vs. deep features capturing semantics), motivated by the intuition: $i \uparrow \Rightarrow \text{Sem}(\psi_i) \uparrow, \text{Vis}(\psi_i) \downarrow$

**2. Attention-Guided Palette Allocation**:
- Obtains attention maps via Grad-CAM++, retaining the top $k_{Gra}\%$ pixels by attention value.
- Palette aggregation is restricted to high-attention regions (foreground/key objects), ensuring semantically critical areas receive richer color representation.
- K-Means is performed in LAB color space rather than RGB to better preserve perceptual similarity.

**3. Texture-Preserved Palette Optimization**:
- Inspired by style transfer, optimizes the palette via differentiable color quantization with a straight-through estimator (STE).
- Minimizes the edge distribution discrepancy $EL$ between the original and quantized images (edge information extracted via Sobel operator).
- Formula: $EL = \sum_{i=1}^{3} w_i \cdot \text{MSE}(G(I_{\text{orig}}^i), G(I_{\text{quant}}^i))$

Storage scheme: only indices and the shared palette are stored; quantized images are reconstructed on-the-fly during training.

Compression ratio formula: starting from standard 24-bit RGB (8 bits per channel), quantizing to $q$ bits reduces the palette to $2^q$ colors, yielding a compression ratio $q_r = 1 - q/24$. For example, 2-bit (only 4 colors) achieves a compression ratio of 91.7%.

## Key Experimental Results

### Main Results

| Dataset | Method | 2-bit (4 colors) | 1-bit (2 colors) |
|--------|------|-----------|-----------|
| CIFAR-10 | Random (pruning) | 77.04% | 70.08% |
| CIFAR-10 | TDDS | 77.32% | 72.46% |
| CIFAR-10 | **DCQ (Ours)** | **89.15%** | **79.90%** |
| CIFAR-100 | Random | 39.71% | 36.68% |
| CIFAR-100 | **DCQ (Ours)** | **57.69%** | **38.44%** |
| ImageNet-1K | **DCQ (Ours)** | **49.69%** | **35.95%** |

- CIFAR-10 full-precision accuracy is 95.45%; DCQ at 2-bit achieves 89.15% (only 6.3 points drop), whereas training on ColorCNN-quantized data yields only ~58%.
- Ablation: shallow-feature clustering (79.90% @ 1-bit) substantially outperforms label-based clustering (40.10%), random clustering (28.44%), and deep-feature clustering (42.10%).
- $k=20$ clusters is the optimal balance point (verified by ablation).
- On CIFAR-100 at 2-bit, DCQ (57.69%) surpasses the strongest pruning baseline TDDS (32.15%) by **25.5 percentage points**.
- On ImageNet-1K at 5-bit, DCQ (66.99%) approaches full-precision accuracy (73.54%), dropping only 6.5 points.
- Inference-oriented CQ baselines (MedianCut, OCTree) perform substantially worse than DCQ in training scenarios.
- Texture-preserved palette optimization contributes approximately 1–3 percentage points of improvement (see Appendix C.1 for ablation results).

## Highlights & Insights
1. **Novel problem formulation**: This is the first work to explicitly define training-oriented dataset-level color quantization, distinguishing it from inference-oriented conventional CQ.
2. **Orthogonal to dataset pruning**: DCQ reduces storage per image while pruning reduces the number of images; the two approaches can be applied jointly.
3. **Elegant shared palette design**: Images within the same cluster share a palette, reducing storage (only indices need to be stored) while improving cross-image consistency.
4. **Significant advantage under aggressive compression**: DCQ substantially outperforms dataset pruning methods, particularly at extreme low-bit settings (1–2 bits).

## Limitations & Future Work
- Relies on a pretrained model for feature extraction and attention maps (Grad-CAM++), introducing additional preprocessing overhead.
- Training effectiveness is validated only on ResNet-18/34; compatibility with modern architectures such as ViT has not been assessed.
- Accuracy degradation remains notable at extreme low bits (1-bit: CIFAR-10 79.90% vs. 95.45% full precision).
- Shared palettes may degrade quantization quality on datasets with large inter-class color variation.
- A fair storage-vs-accuracy comparison with recent dataset distillation methods (e.g., D4M, RDED) is absent.
- LAB color space conversion adds computational steps; scalability to very large datasets remains to be validated.
- The selection of $k_{Gra}\%$ in attention guidance requires ablation tuning and may vary across datasets.

## Related Work & Insights
- Compared to model-aware CQ methods such as ColorCNN and CQFormer, DCQ is the first to target training rather than inference optimization.
- Direct comparisons with dataset pruning methods (EL2N, Forgetting, CCS, TDDS) at matched compression ratios demonstrate DCQ's clear advantage at high compression rates.
- The chromaticity-aware clustering approach is extensible to quantization of other data characteristics (e.g., spectral features, texture complexity).
- DCQ is orthogonal to dataset pruning/distillation and can be combined: pruning first reduces sample count, then DCQ reduces per-sample storage.
- The effectiveness of shallow features for image clustering provides a useful reference for other data preprocessing tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (Training-oriented dataset color quantization is a new direction)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 datasets, multiple baselines, extensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, detailed method description)
- Value: ⭐⭐⭐⭐ (Opens a new dimension for dataset compression)
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](../../AAAI2026/model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)

<!-- RELATED:END -->
