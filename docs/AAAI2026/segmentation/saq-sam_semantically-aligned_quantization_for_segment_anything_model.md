---
title: >-
  [Paper Note] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model
description: >-
  [AAAI 2026][Segmentation][SAM] This paper proposes SAQ-SAM, which improves post-training quantization (PTQ) of SAM from a semantic alignment perspective. It introduces Perceptual Consistency Clipping (PCC) to handle extreme outliers in the mask decoder, and Prompt-Aware Reconstruction (PAR) to preserve semantic alignment between image and prompt interactions.
tags:
  - AAAI 2026
  - Segmentation
  - SAM
  - post-training quantization
  - semantic alignment
  - attention-aware
  - model compression
date: 2026-05-08
content_hash: 375ef791504a6b61
---

# SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model

**Conference**: AAAI 2026
**arXiv**: [2503.06515](https://arxiv.org/abs/2503.06515)
**Code**: [https://github.com/jingjing0419/SAQ-SAM](https://github.com/jingjing0419/SAQ-SAM)
**Area**: Segmentation
**Keywords**: SAM, post-training quantization, semantic alignment, attention-aware, model compression

## TL;DR

This paper proposes SAQ-SAM, which improves post-training quantization (PTQ) of SAM from a semantic alignment perspective. It introduces Perceptual Consistency Clipping (PCC) to handle extreme outliers in the mask decoder, and Prompt-Aware Reconstruction (PAR) to preserve semantic alignment between image and prompt interactions.

## Background & Motivation

SAM exhibits strong zero-shot segmentation capability, but its large parameter count and high computational cost hinder deployment on edge devices. PTQ is an efficient compression approach that requires only a small amount of unlabeled data to calibrate quantization parameters. However, directly applying existing PTQ methods to SAM presents two unique challenges:

### Challenge 1: Extreme Outliers in the Mask Decoder

The QK activations in SAM's mask decoder exhibit an **extreme outlier distribution**: most values are concentrated in a narrow range (e.g., $[-1, 1]$), while outliers can exceed the normal range by 180× (e.g., $[-167, 177]$). The authors identify a critical phenomenon: **aggressively clipping these outliers has almost no impact on segmentation performance**. For instance, clipping $[-167, 177]$ to $[-1, 1]$ does not degrade performance—it actually improves it.

However, conventional distribution-based metrics (e.g., MSE) cannot support such aggressive clipping—MSE yields overly wide clipping ranges to preserve outliers, resulting in insufficient quantization resolution. This is because **distribution alignment ≠ semantic alignment**: maintaining distributional consistency does not guarantee consistency of semantic function in attention.

### Challenge 2: Quantization Reconstruction Ignores Prompt Semantics

Existing quantization reconstruction methods (e.g., BRECQ, QDrop) minimize response errors with respect to the full-precision model block-by-block within the encoder. However, a core characteristic of SAM is **prompt following**: image embeddings must interact with prompt embeddings in the mask decoder. Reconstructing purely image features locally ignores prompt intent, potentially introducing redundant information that disrupts image-prompt interaction.

Core insight: **Semantic misalignment in quantization is the primary bottleneck for low-bit quantization**, necessitating a shift from distribution alignment to semantic alignment.

## Method

### Overall Architecture

SAQ-SAM comprises two core techniques:

1. **Perceptual Consistency Clipping (PCC)**: Handles outliers in mask decoder QK activations by leveraging attention focus overlap as a semantic metric to guide clipping.
2. **Prompt-Aware Reconstruction (PAR)**: Learns quantization parameters for the image encoder by incorporating image-prompt interaction into the reconstruction process.

### Key Designs

#### 1. Perceptual Consistency Clipping (PCC)

**Attention Focus Overlap Metric**: High-attention regions are defined as "attention foci." A threshold factor $\theta$ filters salient attention values to produce a binarized attention focus mask:

$$M_A = \mathbf{1}\{A_w > \theta \cdot \max(A_w)\} \in \mathbb{R}^{N_q \times N_k}$$

The IoU between attention focus masks before and after quantization is then computed:

$$\text{IoU}_{AF}(A_w, \hat{A}_w) = \frac{|M_A \cap \hat{M}_A|}{|M_A \cup \hat{M}_A|}$$

The PCC distance function is: $\text{Dist}_{pcc} = 1 - \text{IoU}_{AF}(A_w, \hat{A}_w)$

This metric is used to determine the optimal clipping boundaries $x_{low}$ and $x_{up}$ for QK activations.

**Design Motivation**:
- The essence of the attention mechanism is to capture semantic information by allocating more attention to task-relevant regions. Using attention focus overlap instead of distribution matching to assess quantization quality therefore maintains consistency at the semantic level.
- This metric is magnitude-agnostic: even when outliers are aggressively clipped (e.g., $[-167, 177] \to [-1, 1]$), semantics are considered preserved as long as the attention focus distribution remains unchanged.
- Experiments demonstrate that $\theta = 0.5$ is robust to performance variation (Figure 6a), and PCC outperforms the baseline across all $\theta$ settings.

#### 2. Prompt-Aware Reconstruction (PAR)

**Interaction Response Reconstruction**: The off-the-shelf cross-attention module (Two-Way Transformer) in SAM's mask decoder is leveraged to inject prompt information into image tokens:

$$T_{ip}^k = \text{TwoWayTransformer}(T_i^k, T_p)$$

where $T_p$ denotes prompt tokens encoded by the prompt encoder and $T_i^k$ denotes image tokens output by stage $k$. The L2 distance between the mixed tokens and the full-precision response is then minimized:

$$\min_{s,z,\alpha} \|{\hat{T}}_{ip}^k - T_{ip}^k\|_2^2$$

**Design Motivation**: By reconstructing mixed image tokens (rather than pure image tokens), the quantized model preserves the correspondence between visual features and prompt intent during learning. This achieves alignment not only at the distributional level but also at the semantic level.

**Layer-Skipping Strategy**: The encoder is partitioned into multiple stages delimited by global attention layers (e.g., layers L0–L2 form stage 0 in SAM-B). The output of each stage bypasses subsequent layers and is fed directly into the Neck for interaction:

$$T_i^k = \text{Neck}\left(\left(\prod_{i=0}^k \text{Stage}^k\right)(E_i)\right)$$

**Design Motivation**: (1) This avoids the high computational cost of full forward passes. (2) Skipping deep layers preserves semantic information at different granularities from each stage. (3) Experiments show that "immature" tokens from early stages can still produce reasonable segmentation results (Figure 5), validating the feasibility of the layer-skipping design.

#### 3. Stage-wise Learning

Transformer layers are partitioned into multiple stages delimited by global attention layers, and quantization parameters within each stage are jointly optimized. This captures inter-block weight correlations more effectively than block-wise learning. The total iterations of stage-wise PAR learning are substantially fewer than those of the baseline PTQ4SAM (which uses block-wise learning for 20,000 iterations), resulting in higher computational efficiency.

### Loss & Training

- **PCC**: Uses attention focus IoU as the clipping metric; calibration requires only the first sample.
- **PAR**: L2 reconstruction loss; the encoder adopts stage-wise learning, the decoder adopts layer-wise learning (2,000 iterations), with 10,000 iterations for the final cross-attention block.
- **Calibration set**: 32 randomly sampled training images.
- **Quantization scheme**: Per-tensor asymmetric quantization for activations; per-channel asymmetric quantization for weights.

## Key Experimental Results

### Main Results

**Instance Segmentation (COCO, DINO detector)**:

| Method | Type | SAM-B 6/6 | SAM-B 4/4 | SAM-L 6/6 | SAM-L 4/4 | SAM-H 6/6 | SAM-H 4/4 |
|--------|------|-----------|-----------|-----------|-----------|-----------|-----------|
| MinMax | Statistical | 11.2 | - | 44.7 | - | 42.8 | - |
| PTQ4SAM-S | Statistical | 20.4 | - | 47.7 | 23.1 | 48.1 | 30.5 |
| SAQ-SAM★ | Statistical | 39.4 | 3.5 | 48.0 | 27.8 | 48.2 | 31.6 |
| QDrop | Learning | 38.9 | 11.2 | 47.5 | 27.5 | 48.3 | 41.7 |
| PTQ4SAM-L | Learning | 40.4 | 14.4 | 48.3 | 36.6 | 48.7 | 43.9 |
| **SAQ-SAM** | **Learning** | **42.4** | **33.8** | **48.3** | **46.3** | **48.9** | **47.4** |
| FP | - | 44.5 | 44.5 | 48.6 | 48.6 | 49.1 | 49.1 |

4-bit SAM-B gain: PTQ4SAM-L 14.4% → SAQ-SAM **33.8%** (+19.4%); 4-bit SAM-L achieves near-lossless performance (46.3% vs. FP 48.6%).

**Semantic Segmentation (ADE20K)**:

| Method | SAM-B 6/6 | SAM-B 4/4 | SAM-L 6/6 | SAM-L 4/4 |
|--------|-----------|-----------|-----------|-----------|
| PTQ4SAM-L | 32.65 | 31.85 | 33.66 | 32.82 |
| **SAQ-SAM** | **33.04** | **32.53** | **33.63** | **33.30** |
| FP | 33.15 | 33.15 | 33.61 | 33.61 |

### Ablation Study

**Component Ablation (YOLOX + COCO, SAM-B/L/H, 4-bit)**:

| Configuration | SAM-B W4A4 | SAM-L W4A4 | SAM-H W4A4 | Note |
|---------------|-----------|-----------|-----------|------|
| Baseline (PTQ4SAM-L) | 18.4 | 31.6 | 37.6 | Baseline |
| + PAR | 26.2 | 38.9 | 39.4 | PAR is effective |
| + PAR + PCC | **30.3** | **39.0** | **39.9** | PCC yields further gains |

**Consistent trend under DINO detector**:

| Configuration | SAM-B W4A4 | SAM-L W4A4 | SAM-H W4A4 |
|---------------|-----------|-----------|-----------|
| Baseline | 14.4 | 36.6 | 43.9 |
| + PAR | 30.2 | 46.1 | 47.4 |
| + PAR + PCC | **33.8** | **46.3** | **47.4** |

### Key Findings

- **Semantic clipping via PCC substantially outperforms distribution-based clipping**: Grid search shows that the $[-1, 1]$ clipping boundary improves performance by 12.4% over the MSE-derived $[-167, 177]$ range. Distribution preservation does not equate to functional preservation.
- **PAR outperforms QDrop at all granularity levels**: This validates the importance of image-prompt interaction in quantization reconstruction.
- **Stage-wise PAR achieves the optimal efficiency-accuracy trade-off**: It outperforms both block-wise and layer-wise learning.
- **PCC is insensitive to the threshold $\theta$**: It consistently outperforms the baseline across various settings, requiring no fine-grained hyperparameter tuning in production.
- **Smaller models benefit more**: The improvement for 4-bit SAM-B is the most pronounced (+19.4% mAP), as smaller models are more sensitive to quantization noise.
- **PCC can serve as an orthogonal technique**: It can be stacked on top of existing methods such as RepQ-ViT and QDrop for additional gains.

## Highlights & Insights

1. **The finding that aggressive outlier clipping is beneficial** challenges intuition: conventional wisdom favors preserving the activation range, yet the attention outliers in SAM's mask decoder are semantically irrelevant.
2. **The attention focus IoU metric** is elegantly designed: it reformulates the semantic preservation problem as a set overlap computation, which is conceptually simple and parameter-free.
3. **Incorporating prompt semantics into quantization reconstruction** breaks the convention of "local reconstruction": SAM's prompt-following characteristic requires that the quantization process respect this property.
4. **An interesting by-product of the layer-skipping strategy**: "immature" tokens output by intermediate stages can produce reasonable segmentations at varying granularities, suggesting a clear semantic specialization across encoder stages.

## Limitations & Future Work

- The approach targets PTQ only; quantization-aware training (QAT) is not explored.
- PCC's grid-search-based clipping boundary determination still requires a one-time calibration procedure.
- The stage partitioning in the layer-skipping strategy depends on model architecture (i.e., the positions of global attention layers) and is not automated.
- Mixed-precision quantization (different bit-widths for different layers) is not explored.
- Applicability to SAM 2 is not validated.

## Related Work & Insights

- The attention focus IoU metric could be generalized to quantization of other attention-intensive models (e.g., DETR, ViT-det).
- The prompt-aware reconstruction paradigm could be applied to quantization of other interactive models (e.g., LISA, Grounding DINO).
- The insight that "semantic alignment > distribution alignment" carries methodological significance and may drive a paradigm shift in the quantization community.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Both PCC and PAR are grounded in deep insights; the perspective of semantically-aligned quantization is distinctive.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three tasks, three model scales, multiple detectors, complete ablation and hyperparameter analysis.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Problem motivation is clearly articulated; the logical thread from observation to analysis to solution is coherent and complete.)
- Value: ⭐⭐⭐⭐⭐ (The leap from unusable to usable 4-bit SAM-B has direct practical significance for SAM deployment.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[ICCV 2025\] E-SAM: Training-Free Segment Every Entity Model](../../ICCV2025/segmentation/e-sam_training-free_segment_every_entity_model.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)

</div>

<!-- RELATED:END -->
