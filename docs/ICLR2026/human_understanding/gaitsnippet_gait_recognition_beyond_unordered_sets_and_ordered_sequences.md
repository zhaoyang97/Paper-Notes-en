---
title: >-
  [Paper Note] GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences
description: >-
  [ICLR2026][Human Understanding][gait recognition] This paper proposes a Snippet paradigm that organizes gait silhouette sequences into several "snippets," each formed by randomly sampling frames from a contiguous interval. This design captures both short-range temporal context and long-range temporal dependencies, achieving 77.5% Rank-1 on Gait3D with a 2D convolution backbone, surpassing all 3D convolution methods.
tags:
  - ICLR2026
  - Human Understanding
  - gait recognition
  - snippet paradigm
  - temporal modeling
  - silhouette
  - 2D convolution
date: 2026-05-08
content_hash: be4ea70688356cfb
---

# GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences

**Conference**: ICLR2026
**arXiv**: [2508.07782](https://arxiv.org/abs/2508.07782)
**Code**: To be confirmed
**Area**: Human Understanding
**Keywords**: gait recognition, snippet paradigm, temporal modeling, silhouette, 2D convolution

## TL;DR

This paper proposes a Snippet paradigm that organizes gait silhouette sequences into several "snippets," each formed by randomly sampling frames from a contiguous interval. This design captures both short-range temporal context and long-range temporal dependencies, achieving 77.5% Rank-1 on Gait3D with a 2D convolution backbone, surpassing all 3D convolution methods.

## Background & Motivation

Gait recognition takes silhouette sequences as input. Two mainstream modeling paradigms currently exist:

**Unordered Set**: Exemplified by GaitSet, this approach treats all frames as an unordered set, extracts features independently via 2D convolution, and aggregates them through Set Pooling. While efficient and robust to frame-order perturbations, independent per-frame processing **discards short-range temporal context between adjacent frames**.

**Ordered Sequence**: Exemplified by GaitGL and DeepGaitV2-3D, this approach jointly models spatiotemporal features using 3D/P3D convolutions. Although capable of capturing local temporal patterns, training typically samples only ~30 consecutive frames, making it **difficult to model long-range temporal dependencies** (real-world sequences can exceed 200 frames).

**Core Problem**: Can a new paradigm simultaneously achieve short-range temporal awareness and long-range temporal coverage?

The authors draw inspiration from human cognition—recognizing a person typically requires observing only a few key actions (not necessarily a complete gait cycle)—and propose viewing gait as a **combination of individualized actions**, with each action represented by a snippet.

## Method

### Overall Architecture

The silhouette sequence is divided into equal-length segments; frames are randomly sampled from each segment to form a snippet (representing a local action), and the overall gait feature is obtained by aggregating features from multiple snippets.

### 3.1 Snippet Sampling

**Training phase**:

- Divide the sequence into $K$ equal-length segments (length $L=16$, approximately one gait cycle)
- Randomly select $M=4$ segments; randomly sample $N=8$ frames from each segment to form one snippet
- Total sampled frames $S = M \times N = 32$
- The length of the first segment $L_1$ is randomly drawn from $\{1,\ldots,L\}$ to increase sampling diversity

**Inference phase**:

- All frames are used: all frames within each segment form a snippet, with $M=K, N=L$
- The first segment length is fixed at $L$ to ensure prediction stability

### 3.2 Snippet Modeling

The model, termed **GaitSnippet**, addresses three sub-problems:

#### (1) Intra-Snippet Modeling

Goal: capture local temporal context within a snippet to enhance frame-level features. The **Snippet Block** is designed as follows:

- **Gathering**: Treats frames within a snippet as an unordered set and aggregates them into a snippet-level representation via Temporal Max Pooling (non-parametric)
- **Smoothing**: Applies a $1\times1$ convolution to the aggregated features to suppress noise and reduce the semantic gap between frame-level and snippet-level representations
- **Residual**: Fuses snippet-level contextual information with frame-level features via residual connection

The Snippet Block is embedded between the two spatial convolution layers of a standard 2D residual block, forming the **Residual Snippet Block** as the backbone's basic building unit. The design is inspired by P3D, enabling frame-level features to continuously perceive local temporal context throughout the hierarchical feature extraction process.

#### (2) Cross-Snippet Modeling

- At the backbone output, Intra-Snippet Gathering is first applied to frame-level features to obtain snippet-level representations
- All snippets are then treated as an unordered set and aggregated into a sequence-level representation via Temporal Max Pooling
- This forms a **hierarchical unordered set** structure: frames → snippets → sequence

**Key distinction**: Although Set Pooling is applied at both levels, the temporal modeling within each snippet via the Snippet Block means the overall model is **not** frame-level permutation-invariant—local temporal information has been incorporated into the frame-level features.

#### (3) Snippet-Level Supervision

The snippet paradigm naturally produces two levels of representation (sequence-level and snippet-level). An additional supervision branch is introduced for snippet-level features:

- Sequence-level loss: Triplet Loss $\mathcal{L}_{tp}$ + Cross-Entropy Loss $\mathcal{L}_{ce}$ (with BNNeck)
- Snippet-level loss: $\mathcal{L}_{tp}^{\star}$ + $\mathcal{L}_{ce}^{\star}$, constructing positive/negative pairs at the snippet granularity
- Total loss: $\mathcal{L}_{all} = \mathcal{L}_{tp} + \mathcal{L}_{ce} + \alpha(\mathcal{L}_{tp}^{\star} + \mathcal{L}_{ce}^{\star})$, with $\alpha=0.75$
- The snippet-level branch is **used only during training** and incurs no inference overhead

### Backbone

Built upon DeepGaitV2-2D (a ResNet-style 2D convolution backbone), with standard residual blocks replaced by Residual Snippet Blocks. Horizontal Pyramid Mapping is adopted to extract multi-granularity local representations.

## Key Experimental Results

### Main Results (In-the-Wild Datasets)

| Method | Type | Backbone | Gait3D R1 | Gait3D mAP | GREW R1 | GREW R5 |
|------|------|------|-----------|------------|---------|---------|
| GaitSet | Set | 2D | 36.7 | 30.0 | 48.4 | 63.6 |
| GaitBase | Set | 2D | 64.6 | 55.3 | 60.1 | 75.5 |
| DeepGaitV2-2D | Set | 2D | 68.2 | 60.4 | 68.6 | 82.0 |
| DeepGaitV2-3D | Seq | 3D | 72.8 | 63.9 | 79.4 | 88.9 |
| VPNet | Seq | 3D | 75.4 | — | 80.0 | 89.4 |
| SwinGait-3D | Seq | Swin3D | 75.0 | 67.2 | 79.3 | 88.9 |
| **GaitSnippet** | **Snippet** | **2D** | **77.5** | **69.4** | **81.7** | **90.9** |

Key findings:

- GaitSnippet with a 2D convolution backbone **comprehensively outperforms** all 3D convolution methods
- Compared to DeepGaitV2-2D with the same backbone: Gait3D R1 improves by **+9.3%** and mAP by **+9.0%**
- Achieves AVG 95.1% on CCPG (clothing-change scenario) and R1 42.4% on CCGR-MINI, both state-of-the-art

### Ablation Study (Gait3D)

**Effect of Snippet Sampling**:

- Simply replacing DeepGaitV2-2D's sampling strategy from Set to Snippet raises R1 from 68.2% to 69.5% (+1.3%), indicating that snippet sampling itself provides a regularization effect
- Optimal hyperparameters: $L=16, M=4, N=8$

**Snippet Block components**:

- Removing Gathering: R1 drops to 73.3% (snippet-level supervision becomes infeasible)
- Removing Smoothing: R1 drops to 74.8% (increased noise and semantic gap)
- Removing Residual: R1 drops to 72.5% (loss of frame-level fine-grained information; largest impact)

**Snippet-Level Supervision**:

- $\alpha=0$ (no snippet-level supervision) still yields competitive performance, confirming the effectiveness of the Snippet Block itself
- Adding snippet-level supervision ($\alpha=0.75$) further improves performance

## Highlights & Insights

**Highlights**:

- Proposes a third paradigm between sets and sequences, with a clear concept grounded in cognitive science
- Outperforms all 3D methods using only a 2D convolution backbone at lower computational cost
- Snippet Block design is concise (non-parametric pooling + $1\times1$ conv + residual) and easy to integrate into existing architectures
- Hierarchical supervision (sequence-level + snippet-level) fully exploits the structural advantages of the snippet paradigm
- Achieves state-of-the-art performance in both controlled (CCPG) and unconstrained (Gait3D/GREW) scenarios

## Limitations & Future Work

- The snippet length $L$ is fixed at 16 frames, which may not be adaptive enough for individuals with highly variable cadence
- Cross-Snippet Modeling relies solely on Max Pooling for aggregation; more sophisticated inter-snippet relationship modeling (e.g., Transformer) remains unexplored
- During inference, all frames across all snippets must be processed, which may still incur high computational cost for long sequences
- Validation is limited to the silhouette modality; extension to skeleton, RGB, and other modalities has not been explored

## Rating

- Novelty: ⭐⭐⭐⭐ Proposes the snippet paradigm with a well-defined conceptual contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets with comprehensive ablation studies
- Value: ⭐⭐⭐⭐ 2D backbone surpassing 3D methods demonstrates strong practical utility

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](../../CVPR2026/human_understanding/mmgait_multi_modal_gait_recognition.md)
- [\[ICCV 2025\] CarGait: Cross-Attention based Re-ranking for Gait Recognition](../../ICCV2025/human_understanding/cargait_cross-attention_based_re-ranking_for_gait_recognition.md)
- [\[CVPR 2026\] Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](../../CVPR2026/human_understanding/beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[ICLR 2026\] NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)

<!-- RELATED:END -->
