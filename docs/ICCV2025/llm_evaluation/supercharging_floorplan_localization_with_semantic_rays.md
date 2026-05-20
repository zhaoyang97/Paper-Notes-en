---
title: >-
  [Paper Note] Supercharging Floorplan Localization with Semantic Rays
description: >-
  [ICCV 2025][LLM Evaluation][Indoor Localization] A semantics-aware floorplan localization framework is proposed that fuses predicted semantic rays with depth rays into a structural-semantic probability volume. Combined w…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Indoor Localization"
  - "Floorplan Localization"
  - "Semantic Rays"
  - "Probability Volume"
  - "Coarse-to-Fine"
date: 2026-05-08
content_hash: e0eedc094dd98d35
---

# Supercharging Floorplan Localization with Semantic Rays

**Conference**: ICCV 2025
**arXiv**: [2507.09291](https://arxiv.org/abs/2507.09291)  
**Code**: [GitHub](https://tau-vailab.github.io/SemRayLoc/)  
**Area**: LLM Evaluation
**Keywords**: Indoor Localization, Floorplan Localization, Semantic Rays, Probability Volume, Coarse-to-Fine

## TL;DR

A semantics-aware floorplan localization framework is proposed that fuses predicted semantic rays with depth rays into a structural-semantic probability volume. Combined with a coarse-to-fine refinement strategy, the method achieves 2–3× performance improvements on two standard benchmarks.

## Background & Motivation

Indoor localization is a classical problem in computer vision with important applications in augmented reality, navigation, and 3D reconstruction. Due to the absence of reliable GPS signals indoors, leveraging 2D floorplans for localization has emerged as a lightweight alternative.

Existing floorplan-based localization methods (e.g., F3Loc) primarily rely on depth rays for structural matching, but suffer from severe **ambiguity**: in environments with repetitive or symmetric layouts, different locations may produce nearly identical depth patterns, leading to high localization uncertainty. For instance, relying solely on wall information, depth rays from multiple corners may be entirely indistinguishable.

The key insight is that floorplans inherently contain **rich semantic information** (windows, doors, room types, etc.) that is typically overlooked. Semantic labels can effectively resolve depth-matching ambiguities — two corners may share identical depth profiles, yet one has a window and the other a door, making them distinguishable.

## Method

### Overall Architecture

The system takes as input a single RGB image $I \in \mathbb{R}^{h \times w \times 3}$ and a semantically labeled 2D floorplan $F \in \{0,1,...,C\}^{H \times W}$, with the goal of predicting the camera's 2D position $(x, y)$ and orientation angle $\theta$. The framework adopts a probabilistic modeling approach, constructing a probability volume $P \in \mathbb{R}^{\hat{H} \times \hat{W} \times O}$ over a discrete pose space $\mathcal{S}$, and outputs the maximum a posteriori estimate:

$$\hat{S}_{I,F} = \arg\max_{S_i \in \mathcal{S}} p(S_i \mid O_{I,F})$$

The pipeline consists of three core steps: (1) predicting depth rays and semantic rays from the image; (2) fusing both ray types to construct a structural-semantic probability volume; (3) refining localization via a coarse-to-fine strategy.

### Key Designs

1. **Semantic Ray Prediction Network**: The core innovation module. Unlike depth rays which are continuous-valued, semantic rays consist of **discrete class labels** (wall, window, door), requiring specialized network design. The network employs a pretrained ResNet50 for feature extraction followed by dimensionality reduction, and introduces two sets of learnable tokens: $l$ ray tokens and one CLS token. Cross-attention integrates spatial features, and the ray tokens produce per-token semantic logits after self-attention and MLP layers, yielding the semantic ray vector $\hat{r}_s \in \{1,...,C\}^l$. The CLS token is optionally used for room type prediction. This design enables accurate prediction of semantic label sequences from a single limited-viewpoint RGB image.

2. **Semantic Probability Volume Construction**: Since semantic rays are discrete labels, conventional linear interpolation (which produces semantically meaningless intermediate values) cannot be applied. A **voting-based interpolation scheme** is proposed: majority voting within a neighborhood is used to downsample the number of rays while preserving semantic consistency. The $L_1$ difference between predicted and reference semantic labels is then computed, exponentiated, and normalized to obtain the semantic probability volume $P_s$.

3. **Structural-Semantic Probability Volume Fusion and Coarse-to-Fine Localization**: The semantic probability volume $P_s$ and depth probability volume $P_d$ are fused with learned weights:

$$P_c = w_s \cdot P_s + w_d \cdot P_d$$

where $w_d = 1 - w_s$, with optimal weights determined on the validation set. The fused volume first extracts Top-$k$ candidate positions at low resolution (with a minimum inter-candidate spacing of $\delta_{\text{res}}$), followed by high-resolution ray-based refinement at multiple perturbed angles $[\pm \delta_{\text{ang}}, ..., \pm \Delta_{\text{max}}]$ for each candidate, selecting the highest-similarity candidate as the final prediction. This avoids information loss caused by low-resolution interpolation.

4. **Room Type Prediction (Optional)**: The CLS token predicts the room type (e.g., living room, bedroom). When the predicted probability exceeds threshold $T_{\text{room}}$, a room mask $M_{\text{room}}$ is constructed to zero out non-matching regions in the probability volume: $P = M_{\text{room}} \odot \tilde{P}$, substantially reducing the search space.

### Loss & Training

- Depth prediction: supervised with $L_1$ loss
- Semantic prediction: cross-entropy loss
- Room labels (if available): additional cross-entropy loss, jointly trained
- Optimizer: Adam with initial learning rate $1 \times 10^{-3}$
- Data augmentation: virtual roll-pitch augmentation for robustness to non-upright cameras
- Hyperparameters: $l=40$ predicted rays, downsampled to 7 at the coarse stage; $\delta_{\text{res}}=0.1$m, $\delta_{\text{ang}}=5°$, $\Delta_{\text{max}}=5°$, Top-5 refinement

## Key Experimental Results

### Main Results

Evaluated on S3D (synthetic dataset, 3,296 houses) and ZInD (real unfurnished residences, 1,575 units).

| Method | S3D R@0.1m | R@0.5m | R@1m | R@1m 30° |
|--------|-----------|--------|------|----------|
| LASER | 0.7 | 6.4 | 10.4 | 8.7 |
| F3Loc | 1.5 | 14.6 | 22.4 | 21.3 |
| **Ours_s** | **5.42** | **41.87** | **53.52** | **52.61** |
| **Ours_r** | **5.70** | **45.53** | **58.78** | **57.49** |
| Oracle | 61.00 | 93.84 | 94.87 | 94.57 |

| Method | ZInD R@0.1m | R@0.5m | R@1m | R@1m 30° |
|--------|-----------|--------|------|----------|
| LASER | 1.38 | 11.06 | 17.55 | 13.64 |
| F3Loc | 0.67 | 7.90 | 15.07 | 11.46 |
| **Ours_s** | **2.98** | **24.00** | **33.96** | **29.30** |
| **Ours_r** | **3.31** | **26.60** | **38.01** | **31.86** |

### Ablation Study

Component contributions analyzed on S3D:

| Configuration | R@0.1m | R@0.5m | R@1m | R@1m 30° | Description |
|--------------|--------|--------|------|----------|-------------|
| Base | 4.65 | 38.35 | 49.40 | 48.44 | Fused volume with argmax only |
| – Interpolation | 4.73 | 38.44 | 48.91 | 47.99 | Linear interpolation instead of voting |
| + Room | 5.12 | 42.92 | 55.57 | 54.04 | With room prediction |
| + Refine | 5.42 | 41.87 | 53.52 | 52.61 | With refinement module |
| + Room&Refine | 5.70 | 45.53 | 58.78 | 57.49 | Both combined |

### Key Findings

- Incorporating semantics doubles or triples performance across all thresholds (S3D R@1m 30°: 21.3% → 57.49%)
- Room prediction yields +9.2% on S3D and +8.7% on ZInD
- The refinement module improves R@1m 30° by 8.6%, confirming that low-resolution interpolation discards critical information
- Optimal semantic-depth weights are $[w_s, w_d] = [0.4, 0.6]$; using either modality alone underperforms fusion
- Inference is computationally tractable: approximately 0.778 seconds per image with Top-5 refinement (single CPU)

## Highlights & Insights

- **Core idea is simple yet powerful**: floorplans inherently encode semantic information (door and window positions), and exploiting it is essentially a "free lunch"
- **Voting-based semantic interpolation** elegantly resolves the problem that discrete labels cannot be linearly interpolated
- **Coarse-to-fine strategy** balances efficiency and accuracy — the coarse stage performs fast search at low resolution, while the fine stage performs high-resolution comparison on a small set of candidates
- Room type prediction serves as an optional module offering additional gains in both accuracy and efficiency

## Limitations & Future Work

- Currently limited to three semantic categories (wall, window, door); extending to more classes (stairs, columns) may yield further improvements
- Oracle results (S3D R@1m: 94.87%) indicate substantial headroom for improving ray prediction accuracy
- Only 2D localization (position + orientation) is supported; floor-level disambiguation is not addressed
- Inference speed remains approximately 0.8 seconds/frame on CPU, which may require optimization for real-time mobile deployment

## Related Work & Insights

- Built upon F3Loc, forming a complementary extension by adding a semantic channel
- Compared to LASER, the proposed ray prediction approach offers finer granularity than feature embedding matching
- The work may inspire semantic augmentation in other map-based localization tasks, such as HD Map localization in autonomous driving

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of fusing semantic rays into a probability volume is both novel and natural
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, multiple ablations, and runtime analysis are all provided
- **Writing Quality**: ⭐⭐⭐⭐ Clear presentation with intuitive figures and tables
- **Value**: ⭐⭐⭐⭐ Offers practical guidance to the indoor localization community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](../../AAAI2026/llm_evaluation/perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](../../NeurIPS2025/llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/llm_evaluation/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[ICLR 2026\] vCache: Verified Semantic Prompt Caching](../../ICLR2026/llm_evaluation/vcache_verified_semantic_prompt_caching.md)
- [\[AAAI 2026\] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/llm_evaluation/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)

</div>

<!-- RELATED:END -->
