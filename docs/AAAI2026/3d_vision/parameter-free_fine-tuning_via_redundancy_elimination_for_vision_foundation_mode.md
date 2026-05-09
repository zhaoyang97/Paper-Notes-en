---
title: >-
  [Paper Note] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models
description: >-
  [AAAI 2026][3D Vision][Vision Foundation Models] This work identifies pervasive redundant channels in vision foundation models (SAM/SAM2/DINOv2) and proposes a parameter-free adaptation method that requires no parameter updates: an output-difference-based channel selection algorithm identifies optimal replacement pairs, substituting redundant channels with effective ones to enhance feature representations for downstream tasks, yielding average mIoU gains of 5–11 points.
tags:
  - AAAI 2026
  - 3D Vision
  - Vision Foundation Models
  - Parameter-Free Fine-Tuning
  - Channel Redundancy
  - SAM
  - Feature Selection
date: 2026-05-08
content_hash: 0cbadcca9a612678
---

# Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models

**Conference**: AAAI 2026
**arXiv**: [2504.08915](https://arxiv.org/abs/2504.08915)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: Vision Foundation Models, Parameter-Free Fine-Tuning, Channel Redundancy, SAM, Feature Selection

## TL;DR

This work identifies pervasive redundant channels in vision foundation models (SAM/SAM2/DINOv2) and proposes a parameter-free adaptation method that requires no parameter updates: an output-difference-based channel selection algorithm identifies optimal replacement pairs, substituting redundant channels with effective ones to enhance feature representations for downstream tasks, yielding average mIoU gains of 5–11 points.

## Background & Motivation

Vision Foundation Models (VFMs) such as SAM and DINOv2, trained on large-scale data, possess powerful general-purpose visual representations. Adapting them to downstream tasks typically requires parameter fine-tuning:

- **Full fine-tuning**: updates all parameters at high computational cost.
- **Parameter-Efficient Fine-Tuning (PEFT/LoRA/Adapter)**: updates a small subset of parameters (thousands to millions), yet still requires backpropagation and computation graph maintenance.

**Key Observation** (control experiment in Table 1): On the PerSeg dataset with SAM, zeroing out certain channel activations reveals:
- Channel 6 zeroed: mIoU unchanged (50.6 → 50.6), indicating the channel is redundant.
- Channel 216 zeroed: mIoU improves (50.6 → 52.7), indicating the channel is even harmful.
- Channels 175/19/189 zeroed: mIoU drops, indicating these channels are beneficial for the task.

**Root Cause**: Among the general features learned by VFMs on large-scale data, many are irrelevant or even detrimental to specific downstream tasks. This redundancy arises because the model must generalize across a large variety of tasks.

**Core Problem**: Can downstream tasks be addressed without modifying any model parameters—solely by selecting, reusing, and enhancing existing features?

## Method

### Overall Architecture

The proposed approach contrasts sharply with conventional fine-tuning paradigms:
- **(a) Conventional decoder fine-tuning**: updates decoder parameters to adapt pre-trained features.
- **(b) Conventional encoder fine-tuning**: updates encoder parameters to modify pre-trained features.
- **(c) Proposed method**: updates no parameters; instead, redundant channels are replaced with more effective ones.

Pipeline: Search dataset → Encoder feature extraction → Pairwise channel replacement → Output difference comparison → Dictionary construction → Optimal combination search → Replacement application.

### Key Designs

1. **Problem Formulation**

   The objective is to find the optimal set of replacement pairs $P^*$ that maximizes performance on the downstream dataset $S$:
   $$P^* = \arg\max_P \text{mIoU}(S, P)$$
   where $P = \{(i,j)_1, (i,j)_2, \ldots, (i,j)_k\}$ and $(i,j)$ denotes replacing channel $i$ with channel $j$.

   Direct enumeration of all combinations is intractable: for $C=256$, it requires $2^{C^2}$ inference passes.

2. **Channel Selection Algorithm**

   **Three strategies to reduce search cost**:

   **(1) Output-difference-based search**:
   Given a search dataset $\mathbf{S}$, the encoder produces features $X \in \mathbb{R}^{D \times C \times W \times H}$. For each replacement pair $(i,j)$, the gain is computed as:
   $$\Delta\text{Acc}_{(i \to j)} = D(X') - D(X)$$
   where $D(X)$ and $D(X')$ denote decoder outputs for the original and replaced features, respectively.

   A dictionary $\mathcal{D} = \{(i,j): \Delta\text{Acc}_{(i \to j)}\}$ is constructed, and the top $N$ pairs form $\mathcal{D}_{topN}$.

   All $2^N - 1$ combinations within $\mathcal{D}_{topN}$ are then enumerated to find the optimal combination $P^*$.

   **Complexity reduction**: from $2^{C^2}$ to $C^2 + 2^N - 1$ (for $N=10$, approximately $65{,}536 + 1{,}023$ inference passes).

   **(2) Sample reduction**: only 50 images are used as the search dataset.

   **(3) Feature caching**: encoder features are precomputed and stored; each inference pass only modifies the cached features before passing them to the decoder, avoiding redundant encoding.

   **Design Motivation**: The output difference of a single replacement pair serves as a reliable predictor of that pair's contribution within a combination. The filter-then-combine strategy substantially reduces computation while preserving search effectiveness. The process requires only forward inference—no backpropagation—resulting in minimal GPU memory overhead.

3. **Channel Replacement Implementation**

   Given a replacement pair $(i,j)$, the feature transformation is:
   $$X'_{d,c,w,h} = X_{d, f_{i \to j}(c), w,h}$$
   where $f_{i \to j}(\cdot)$ is a mapping function that substitutes channel $i$ with channel $j$.

   This is not a random shuffle but a selective, deterministic replacement of redundant channels with effective ones.

### Loss & Training

During the search phase, evaluation uses the same Dice + CE loss as the baseline. Notably, the search process involves only model inference—**no gradient computation or backpropagation is required**.

Implementation details:
- Search dataset: 50 randomly sampled images.
- $N = 10$ (top-$N$ replacement pairs).
- Baseline fine-tuning comparisons use 25 epochs, Adam optimizer, and an initial learning rate of $10^{-4}$.

## Key Experimental Results

### Main Results

**Parameter-free fine-tuning results on various SAM versions** (average mIoU across 9 datasets):

| Model | Backbone | Parameters | Baseline Avg | +Ours Avg | Gain Δ |
|-------|----------|------------|--------------|-----------|--------|
| SAM | ViT-B | 91M | 49.14 | 58.08 | **+8.94** |
| SAM | ViT-L | 308M | 56.15 | 67.61 | **+11.46** |
| SAM | ViT-H | 636M | 55.54 | 60.68 | +5.14 |
| SAM2 | Hiera-T | 39M | 57.29 | 65.63 | **+8.34** |
| SAM2 | Hiera-S | 46M | 61.04 | 68.69 | +7.65 |
| SAM2 | Hiera-B+ | 81M | 61.62 | 66.94 | +5.32 |
| SAM2 | Hiera-L | 224M | 67.77 | 73.53 | +5.76 |

Gains of 5–11 mIoU points are achieved without updating any parameters.

**Combined with existing fine-tuning methods**:

| Fine-tuning Method | Baseline Avg | +Ours Avg | Additional Gain |
|--------------------|--------------|-----------|----------------|
| Decoder-only | 73.61 | 74.62 | +1.01 |
| SAMed (LoRA) | 78.56 | 79.72 | **+1.16** |
| SAM-COBOT | 78.73 | 79.32 | +0.59 |
| SAM-Adapter | 72.89 | 73.80 | +0.91 |
| SAM-PARSER | 60.96 | 65.39 | **+4.43** |
| DoRA | 79.12 | 79.92 | +0.80 |

These results demonstrate that channel redundancy persists even after parameter fine-tuning, and the proposed method can serve as a plug-and-play module for further improvement.

### Ablation Study

**Computational cost comparison**:

| Method | GPU Memory (GB) | Trainable Parameters (K) |
|--------|----------------|--------------------------|
| Encoder-only | 34.6 | 89,670 |
| Decoder-only | 13.7 | 4,057 |
| MedSAM | 34.7 | 93,735 |
| SAMed (LoRA) | 28.9 | 147 |
| SAM-PARSER | 15.9 | 0.5 |
| **Ours** | **11.1** | **0** |

The proposed method achieves the lowest GPU memory usage (11.1 GB vs. 13.7–34.7 GB for other methods) with zero trainable parameters.

**Effect of the number of replacement pairs**: Increasing the number of replacement pairs generally improves performance, with performance peaking at 6 pairs on the COCO dataset.

**Extension to other visual tasks**:

| Model | Backbone | NYUv2 MSE↓ / AbsRel↓ / δ₁↑ | CIFAR Acc↑ |
|-------|----------|---------------------------|-----------|
| DINOv2 | ViT-S | 0.225 / 0.126 / 0.893 | 80.41 |
| +Ours | ViT-S | **0.209 / 0.112 / 0.907** | **80.81** |
| DINOv2 | ViT-B | 0.210 / 0.110 / 0.900 | 88.08 |
| +Ours | ViT-B | **0.193 / 0.095 / 0.916** | **88.49** |

The method is effective for both depth estimation and image classification.

### Key Findings

- Feature maps of effective channels exhibit clearer structure, edges, and textures, whereas redundant channels appear blurry and noisy (visualized in Figure 5).
- Certain channels exhibit cross-domain consistency: e.g., Channel 19 is effective across natural, medical, and camouflaged scenarios, while Channels 20/98/162/226 are universally redundant.
- Larger models (ViT-H, Hiera-L) show slightly smaller gains, possibly because larger models exhibit relatively less redundancy.
- Gains on in-domain datasets (natural images) are larger than on out-of-domain datasets (medical images), consistent with SAM's training data distribution.

## Highlights & Insights

1. **Paradigm innovation**: This is the first work to demonstrate that VFMs can be adapted in a completely parameter-free manner—requiring no gradients, no backpropagation, and no additional parameters. Significant downstream performance gains are achieved solely through channel substitution.
2. **Minimal computational barrier**: Requiring only 11.1 GB of GPU memory and forward inference, the method is practically deployable on consumer-grade GPUs, substantially lowering the barrier to VFM adaptation.
3. **Orthogonal complementarity with PEFT**: The method functions as a plug-and-play post-processing step, delivering additional gains of 0.5–4.4 mIoU points on top of already fine-tuned models.
4. **Insights into channel redundancy**: The work reveals the pervasive feature redundancy in foundation models, offering a new perspective on understanding the efficiency of feature utilization in large models.
5. **Cross-task generalization**: The method generalizes from segmentation to depth estimation and classification, and from SAM to DINOv2, validating its universality.

## Limitations & Future Work

- The search process still requires iterating over $C^2$ (~65,536) pairs and $2^N - 1$ combinations; although only inference is needed, this incurs non-trivial time costs at scale.
- The choice of search dataset may influence the optimal replacement pairs; 50 images may not be sufficiently representative for all datasets.
- The value of $N=10$ is fixed; adaptive determination of $N$ has not been explored.
- Channel replacement is a hard substitution; softer channel reweighting schemes have not been investigated.
- Operations are limited to the last encoder layer; multi-layer channel replacement has not been explored.
- Validation is restricted to SAM/SAM2/DINOv2; generalizability to other VFMs such as CLIP and MAE remains unknown.

## Related Work & Insights

- **SAM-PARSER (2024)**: Compresses trainable parameters to as few as 512; this work goes further to zero parameters.
- **ShuffleNet**: Channel shuffling for cross-group information fusion during training—fundamentally different in objective and mechanism from the proposed method.
- **Channel-Exchanging Network**: Channel exchange for multimodal fusion; this work targets intra-modal redundancy elimination.
- **Network Pruning**: Removes redundancy but typically requires retraining; the proposed method does not.
- **Insight**: Feature redundancy is a universal phenomenon in foundation models; "subtraction" (redundancy elimination) can sometimes be more effective than "addition" (parameter augmentation). This perspective may generalize to the adaptation of NLP large language models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Parameter-free fine-tuning paradigm is proposed for the first time in the VFM domain; bold and effective.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (9 datasets × 7 backbones × 6 fine-tuning method combinations, plus depth estimation and classification extensions.)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, well-designed experiments, and informative visualizations.)
- Value: ⭐⭐⭐⭐⭐ (Highly practical, extremely low computational barrier, orthogonally complementary to existing methods.)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](../../NeurIPS2025/3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava_bench_atomic_visual_ability_vfm.md)

<!-- RELATED:END -->
