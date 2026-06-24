---
title: >-
  [Paper Note] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning
description: >-
  [ICCV 2025][3D Vision][3D Object Detection] This paper proposes tgGBC (trim keys gradually Guided By Classification scores), a zero-shot runtime pruning method that computes key importance by element-wise multiplication of classification scores and attention maps, progressively pruning unimportant keys across layers. It achieves nearly 2× acceleration of the Transformer decoder on multiple 3D detectors with less than 1% performance degradation.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Object Detection"
  - "Transformer Pruning"
  - "Attention Key Pruning"
  - "Zero-Shot Acceleration"
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 5317f79be5ca3f13
---

# Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning

**Conference**: ICCV 2025
**arXiv**: [2503.08101](https://arxiv.org/abs/2503.08101)  
**Code**: [https://github.com/iseri27/tg_gbc](https://github.com/iseri27/tg_gbc)  
**Area**: 3D Vision
**Keywords**: 3D Object Detection, Transformer Pruning, Attention Key Pruning, Zero-Shot Acceleration, Autonomous Driving

## TL;DR
This paper proposes tgGBC (trim keys gradually Guided By Classification scores), a zero-shot runtime pruning method that computes key importance by element-wise multiplication of classification scores and attention maps, progressively pruning unimportant keys across layers. It achieves nearly 2× acceleration of the Transformer decoder on multiple 3D detectors with less than 1% performance degradation.

## Background & Motivation
Query-based 3D object detection methods (e.g., PETR, StreamPETR, ToC3D) employ multi-layer Transformer decoders to process dense features from surround-view cameras and have become the dominant SOTA paradigm. However, the global feature interactions in dense methods incur substantial computational overhead, hindering deployment on edge devices.

Limitations of existing acceleration approaches:

**Static pruning methods** (e.g., FastV, SparseViT): require an additional forward pass for hyperparameter search and are designed for ViT classification models, making them difficult to transfer to 3D detectors.

**Runtime pruning methods** (e.g., ToMe, Zero-TPrune): ToMe's similarity matrix computation has complexity $O(N_k^2)$, which is prohibitively expensive for the large token counts in 3D detectors (4,224–24,000); Zero-TPrune assumes a square attention matrix, but in 3D detectors the numbers of queries and keys differ, yielding a non-square attention matrix.

**Key Challenge**: The token count in 3D detection far exceeds that in ViT models (3D: 4,224–24,000 vs. ViT: ~1,024), making direct transfer of existing methods suboptimal in both efficiency and effectiveness.

**Core Insight**: In 3D detectors, the final output is determined by predictions with the highest classification scores. Keys that contribute little to high-confidence predictions can therefore be safely pruned. Classification scores and attention maps arise naturally within the Transformer decoder, requiring no additional parameters.

**Core Idea**: Classification scores are broadcast and element-wise multiplied with the attention map; column-wise summation yields a per-key importance score, and the least important keys are pruned layer by layer.

## Method

### Overall Architecture
The tgGBC module is inserted between adjacent Transformer decoder layers. It receives the classification scores $C \in \mathbb{R}^{N_q \times N_C}$ and cross-attention weights $A \in \mathbb{R}^{N_q \times N_k}$ from the preceding layer, computes key importance, and prunes unimportant keys. A total of $r$ keys are removed, distributed across $n$ layers, with $\lfloor r/n \rfloor$ keys pruned per layer.

### Key Designs

1. **Key Importance Computation**:

    - Step 1: Obtain the maximum classification score per query: $\hat{C}_i = \max_j C_{i,j}$
    - Step 2: Expand to $\tilde{C} \in \mathbb{R}^{N_q \times N_k}$ by repeating along the key dimension
    - Step 3: Element-wise multiply: $S_0 = A \odot \tilde{C}$ (propagating classification quality to each key)
    - Step 4: Select the top-$k$ rows by classification score to obtain $S_1 \in \mathbb{R}^{k \times N_k}$
    - Step 5: Sum column-wise to obtain key importance: $S_j = \sum_{i=1}^k (S_1)_{i,j}$
    - **Design Motivation**: Not all queries are equally important; only high-confidence queries contribute to final predictions. Keys that are more relevant to these high-confidence queries are preferentially retained, avoiding the "democratic" assignment of equal weight to all queries.

2. **Gradual Layer-wise Pruning**: Rather than removing all target keys in a single layer, $\lfloor r/n \rfloor$ keys are pruned after each of the first $n$ Transformer layers.

    - **Design Motivation**: Progressive pruning allows subsequent layers to recompute attention over the already-pruned key set, resulting in less information loss compared to one-shot pruning.

3. **Feasibility Guarantee**: The attention module output shape $O \in \mathbb{R}^{N_q \times E}$ is independent of $N_k$; thus, pruning keys does not alter the output dimension. Provided that $K$ and $V$ are pruned synchronously (in 3D detectors, $V = K$), the original model parameters remain valid.

### Loss & Training
- **Training-free**: tgGBC does not modify model parameters and requires no training or fine-tuning.
- Plug-and-play: pruning layers are appended only during inference.
- Negligible additional computation: only a single element-wise matrix multiplication and column-wise summation are required.

## Key Experimental Results

### Main Results

| Model | Backbone | $N_k$ | $r$ | mAP | NDS | Dec. Time (ms) | Speedup |
|------|----------|-------|-----|-----|-----|----------------|--------|
| PETR | ResNet50 | 16896 | 0 | 31.74% | 0.367 | 47.09 | 1.00× |
| PETR+tgGBC | ResNet50 | 16896 | 12000 | 30.78% | 0.358 | 28.58 | **1.65×** |
| StreamPETR | VovNet | 24000 | 0 | 48.89% | 0.573 | 64.93 | 1.00× |
| StreamPETR+tgGBC | VovNet | 24000 | 21000 | 48.55% | 0.573 | 34.98 | **1.86×** |
| 3DPPE | VovNet | 6000 | 0 | 39.81% | 0.446 | 53.25 | 1.00× |
| 3DPPE+tgGBC | VovNet | 6000 | 3000 | 39.74% | 0.445 | 27.56 | **1.93×** |
| ToC3D | - | - | - | Latest model | - | - | **1.99×** |

A 1.99× Transformer decoder speedup is achieved on ToC3D with less than 1% mAP degradation.

### Ablation Study

| Pruning Method | Model | mAP | Decoder Latency |
|---------|------|-----|-----------|
| No pruning | FocalPETR (VovNet) | 42.36% | 24.65ms |
| ToMe | FocalPETR (VovNet) | 41.82% | 22.35ms |
| **tgGBC** | FocalPETR (VovNet) | **42.38%** | **17.08ms** |
| No pruning | StreamPETR (ResNet50) | 38.01% | 31.10ms |
| ToMe | StreamPETR (ResNet50) | 37.55% | 29.40ms |
| **tgGBC** | StreamPETR (ResNet50) | **37.93%** | **24.15ms** |

On FocalPETR, tgGBC even improves mAP (42.38% vs. 42.36%), suggesting that moderate key pruning may exert a regularization effect.

### Key Findings
- Deployment on edge devices (Orin) yields 1.18× and 1.19× inference speedup for FocalPETR and StreamPETR, respectively.
- tgGBC can prune up to 90% of keys within a single layer ($r$ approaching $N_k$), which is infeasible for ToMe due to its bipartite matching constraint.
- On certain models, tgGBC improves performance, indicating that redundant keys effectively act as noise.

## Highlights & Insights
- **Truly zero-shot**: no training data, hyperparameter search, or fine-tuning is required; the method is fully plug-and-play.
- The method leverages classification scores and attention maps already present in the model ("free" information) without introducing any additional parameters.
- Strong cross-model generality: validated on PETR, FocalPETR, StreamPETR, 3DPPE, MV2D, M-BEV, and ToC3D.
- Architecture-agnostic: applicable to any Transformer decoder employing dense global attention.
- Practical deployment efficacy verified on edge hardware.

## Limitations & Future Work
- Applicable only to dense methods with global attention; incompatible with sparse methods using Deformable Attention or Flash Attention.
- The pruning ratio $r$ must be set manually; no automatic selection strategy is provided.
- Evaluation is limited to the nuScenes dataset; validation on other autonomous driving benchmarks (e.g., Waymo, Argoverse) is absent.
- Key pruning for historical frames in temporal models (e.g., StreamPETR) is not thoroughly investigated.
- Query pruning in self-attention remains unexplored; the current work targets only cross-attention keys.

## Related Work & Insights
- **ToMe**: merges tokens via bipartite matching; the present work avoids the $O(N^2)$ similarity computation.
- **Zero-TPrune**: uses Markov chain convergence as a pruning criterion but requires a square attention map.
- **3D Detectors**: query-based methods such as the PETR family, OPEN, and ToC3D are the primary application targets.
- **Insight**: The paradigm of using task-specific signals (classification scores) as pruning criteria is generalizable to other Transformer tasks where analogous score signals exist.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegant use of classification-score-weighted attention maps for key pruning; the method is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven models × multiple configurations, including edge-device deployment validation.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play acceleration tool with direct practical value for autonomous driving deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Zoo3D: Zero-Shot 3D Object Detection at Scene Level](../../CVPR2026/3d_vision/zoo3d_zero-shot_3d_object_detection_at_scene_level.md)
- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ICCV 2025\] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)

</div>

<!-- RELATED:END -->
