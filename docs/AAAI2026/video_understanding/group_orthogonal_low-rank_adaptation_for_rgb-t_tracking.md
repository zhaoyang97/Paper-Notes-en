---
title: >-
  [Paper Note] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking
description: >-
  [AAAI 2026][Video Understanding][RGB-T Tracking] This paper proposes the GOLA framework, which quantifies LoRA rank importance via SVD decomposition, freezes critical ranks to preserve pre-trained priors…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "RGB-T Tracking"
  - "LoRA"
  - "Low-Rank Adaptation"
  - "Orthogonal Constraint"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: c3ef27f0a1eb9e52
---

# Group Orthogonal Low-Rank Adaptation for RGB-T Tracking

**Conference**: AAAI 2026
**arXiv**: [2512.05359](https://arxiv.org/abs/2512.05359)  
**Code**: [GitHub](https://github.com/MelanTech/GOLA)  
**Area**: Video Understanding
**Keywords**: RGB-T Tracking, LoRA, Low-Rank Adaptation, Orthogonal Constraint, Parameter-Efficient Fine-Tuning

## TL;DR

This paper proposes the GOLA framework, which quantifies LoRA rank importance via SVD decomposition, freezes critical ranks to preserve pre-trained priors, clusters redundant ranks into groups, and imposes inter-group orthogonal constraints to enable more efficient RGB-T tracking adaptation.

## Background & Motivation

### State of the Field
RGB-T (visible + infrared) tracking enhances robustness in challenging scenarios such as low illumination and occlusion by fusing complementary information from two modalities. Recent methods primarily adopt the parameter-efficient fine-tuning (PEFT) paradigm, freezing pre-trained parameters and fine-tuning only a small subset for downstream tasks.

### Limitations of Prior Work
LoRA suffers from **severe rank space redundancy** in RGB-T tracking:
- SVD decomposition of LoRA parameter matrices after training reveals that only **a few ranks become dominant**, with the majority contributing negligible information.
- To achieve rapid convergence, the model tends to prioritize integrating a small number of critical ranks, **overwriting pre-trained priors**.
- The remaining redundant ranks lack targeted optimization signals and remain **unactivated**, failing to learn fine-grained features.
- This ultimately severely limits the model's ability to handle **diverse challenges** in RGB-T tracking (occlusion, illumination variation, deformation, etc.).

### Core Idea

**Protect critical ranks + activate redundant ranks**: (1) quantify rank importance via SVD and freeze critical ranks to preserve pre-trained priors; (2) cluster redundant ranks into groups; (3) impose inter-group orthogonal constraints to force each group to learn complementary, non-overlapping feature transformations, thereby fully exploiting the entire rank space.

## Method

### Overall Architecture

GOLA builds upon a single-stream tracking framework and applies modified LoRA to each linear layer of the backbone:
- Inputs: template images $(I_v^z, I_t^z)$, search images $(I_v^x, I_t^x)$, and online templates $(I_v^o, I_t^o)$
- After tokenization, tokens are concatenated: $h = [Z_v; Z_t; X_v; X_t; O_v; O_t]$
- Joint feature extraction through the encoder; GOLA is applied to each linear layer: $h' = \mathbf{W}h + \mathbf{BA}h$
- Parameters are merged at inference: $\mathbf{W}' = \mathbf{W} + \mathbf{BA}$ (no additional inference latency)

### Key Designs

#### 1. **Rank Decomposition Partition Strategy**

- **Mechanism**: Quantify the importance of each rank to distinguish critical ranks from redundant ranks; performed offline without increasing training overhead.
- **Steps**:
  1. Apply SVD to matrix $\mathbf{B}$: $\Sigma, \mathbf{V} \leftarrow \text{SVD}(\bar{\mathbf{B}})$
  2. Select the top-k singular vectors $V_k$ and corresponding singular values $\Sigma_k$ as references.
  3. Compute the importance score for each rank: $\mathbf{S} = \|\bar{\mathbf{B}}^\top \mathbf{V}_k^\top \odot \Sigma_k\|_2$
  4. Sort ranks by $\mathbf{S}$ in descending order; the top-k ranks are designated critical (**frozen**), and the remainder are redundant (**trainable**).
  5. Cluster redundant ranks into $n$ groups via constrained k-means.
- **Rationale for using $\mathbf{B}$ as reference**: $\mathbf{B}$ is more strongly associated with task-specific information, whereas $\mathbf{A}$ acts more as a general-purpose feature extractor.
- **Design Motivation**: Freezing critical ranks preserves learned generalization capability and prevents it from being overwritten during new modality adaptation.

#### 2. **Inter-Group Orthogonal Constraint Strategy**

- **Mechanism**: An orthogonality loss enforces orthogonal relationships between parameters of different rank groups, ensuring each group learns complementary features.
- **Orthogonal Loss**:

$$\mathcal{L}_{orth} = \sum_{i \neq j}\left(\left|\mathbf{A}_{u_i}^\top \mathbf{A}_{u_j}\right| + \left|\mathbf{B}_{u_i}^\top \mathbf{B}_{u_j}\right|\right)$$

- A **channel orthogonality** constraint is applied to $\mathbf{A}$: enhances diversity and discriminability of general features.
- A **rank orthogonality** constraint is applied to $\mathbf{B}$: ensures different ranks carry complementary task knowledge.
- **Efficiency Design**: At each iteration, only **one pair** of rank groups is randomly sampled to compute the orthogonality loss, reducing computational overhead.
- **Design Motivation**: Prevents redundant ranks from learning overlapping feature spaces, enabling the model to simultaneously address diverse challenges in RGB-T tracking.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{reg} + \lambda \cdot \mathcal{L}_{orth}$

- $\mathcal{L}_{cls}$: Binary cross-entropy classification loss
- $\mathcal{L}_{reg}$: GIoU regression loss
- $\lambda = 1.4 \times 10^{-3}$
- LoRA rank $r=64$, number of critical ranks $k=16$, number of redundant rank groups $n=8$
- Training: 10 epochs, batch size 128, 131,072 image pairs per epoch
- Online template update threshold $\tau=0.84$

## Key Experimental Results

### Main Results (4 Benchmark Datasets)

| Method | GTOT MPR/MSR | RGBT210 PR/SR | RGBT234 MPR/MSR | LasHeR PR/NPR/SR | Speed |
|--------|-------------|---------------|-----------------|------------------|-------|
| ViPT | -/- | -/- | 83.5/61.7 | 65.1/-/52.5 | - |
| TBSI | -/- | -/- | 87.1/63.7 | 69.2/65.7/55.6 | 36fps |
| CKD | 93.2/77.2 | 88.4/65.2 | 90.0/67.4 | 73.2/69.3/58.1 | 96fps |
| SUTrack-L384 | -/- | -/- | 93.7/70.3 | 76.9/-/61.9 | 12fps |
| **GOLA-B** | **92.8/78.5** | **90.9/67.0** | **92.2/69.5** | **77.5/73.9/61.6** | **125fps** |
| **GOLA-L** | **95.3/80.9** | **92.0/68.7** | **92.8/71.3** | **78.1/74.5/61.9** | **64fps** |

GOLA-B surpasses SUTrack-L384 (12fps) on LasHeR at 125fps, with only 99M parameters (10% trainable).

### Comparison with Fine-Tuning Methods

| Method | Trainable Params | LasHeR PR/SR | Inference Speed |
|--------|-----------------|-------------|-----------------|
| Full Fine-tune | 100% | 72.5/57.9 | 125fps |
| Adapter | 4% | 68.8/54.5 | 78fps |
| VPT | 3% | 70.8/56.3 | 85fps |
| LoRA | 13% | 76.3/60.7 | 125fps |
| DoRA | 13% | 63.7/49.3 | 125fps |
| **GOLA-B** | **10%** | **77.5/61.6** | **125fps** |

GOLA outperforms LoRA by 1.2%/0.9% (PR/SR) while reducing trainable parameters by 23%.

### Ablation Study

| Configuration | PR (%) | SR (%) | Notes |
|--------------|--------|--------|-------|
| w/o orthogonal constraint | 76.3 | 60.7 | LoRA baseline |
| $\mathbf{A}$ orthogonality only | 76.8 | 61.2 | Constrains general features only |
| $\mathbf{B}$ orthogonality only | 76.7 | 61.2 | Constrains task knowledge only |
| **$\mathbf{A}$+$\mathbf{B}$ orthogonality** | **77.5** | **61.6** | Best complementary effect |

| Sorting | Clustering | PR/SR | Notes |
|---------|-----------|-------|-------|
| ✓ | ✗ | 77.0/61.4 | Sorting only |
| ✗ | ✓ | 76.6/61.0 | Clustering only |
| **✓** | **✓** | **77.5/61.6** | Best with both |

| Key Hyperparameter | Optimal Value | Notes |
|-------------------|--------------|-------|
| Critical rank count $k$ | 16 | Too large → too many redundant ranks frozen; too small → pre-trained priors lost |
| Number of groups $n$ | 8 | Too many → insufficient expressiveness per group; too few → insufficient complementarity |
| Sampled group pairs per iteration | 1 | One pair suffices; more pairs do not improve performance but increase computation |

### Key Findings

1. **Orthogonal constraints on $\mathbf{A}$ and $\mathbf{B}$ are complementary**: Constraining both simultaneously outperforms constraining either alone.
2. **Rank sorting and clustering must be combined**: Sorting preserves generalization capacity; clustering promotes intra-group specialization.
3. **t-SNE visualizations confirm the effectiveness of orthogonal constraints**: Rank features from different groups in GOLA exhibit distinct clustering and separation in t-SNE plots.
4. **Near-optimal performance across 19 attributes**: GOLA shows particular advantages on challenging attributes such as HI (high illumination), HO (heavy occlusion), and SV (scale variation).

## Highlights & Insights

1. **In-depth analysis of LoRA redundancy**: SVD-based quantification reveals the rank space redundancy in LoRA, providing a theoretical foundation for improvement.
2. **Minimalist yet effective design**: Freezing critical ranks + grouping + orthogonal constraints — conceptually simple but empirically significant.
3. **No additional inference overhead**: The parameter merging strategy ensures inference speed is identical to standard LoRA (125fps).
4. **Strong generalizability**: Both GOLA-B and GOLA-L variants perform excellently, and the offline partitioning strategy does not affect the training pipeline.

## Limitations & Future Work

1. The critical rank count $k$ and group count $n$ require hyperparameter search; an adaptive mechanism is lacking.
2. Clustering relies on fixed constrained k-means; more flexible dynamic grouping methods could be explored.
3. Orthogonal constraints are applied only to randomly sampled group pairs, potentially leaving some pairs insufficiently constrained.
4. Validation is limited to RGB-T tracking; the approach could be extended to more multi-modal downstream tasks.
5. The offline partitioning strategy requires a preliminary LoRA training pass, adding upfront preparation cost.

## Related Work & Insights

- Unlike AdaLoRA, which dynamically adjusts rank, GOLA maintains a fixed rank while optimizing rank space utilization.
- The inter-group orthogonal constraint is conceptually analogous to the specialization of different experts in MoE, but without requiring a routing mechanism.
- The strategy of freezing critical ranks is transferable to LoRA applications in NLP.
- t-SNE visualization analysis provides a new perspective for evaluating LoRA variants.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The analysis of LoRA redundancy and the group orthogonal constraint solution are novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 datasets, 19 attribute analyses, comparisons with multiple PEFT methods, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, complete mathematical derivations, and rich visualization analysis.
- **Value**: ⭐⭐⭐⭐ — General insights on LoRA usage are transferable across multiple domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](../../CVPR2026/video_understanding/tcei_test_time_calibration_experience_intuition_mot.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](../../ICCV2025/video_understanding/xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[CVPR 2026\] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](../../CVPR2026/video_understanding/longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](../../ICCV2025/video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)

</div>

<!-- RELATED:END -->
