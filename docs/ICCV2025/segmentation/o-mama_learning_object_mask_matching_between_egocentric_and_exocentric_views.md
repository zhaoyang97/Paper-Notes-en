---
title: >-
  [Paper Note] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views
description: >-
  [ICCV 2025][Segmentation][Cross-View Segmentation] This work reframes cross-view (ego-exo) object segmentation as a mask matching problem. It leverages FastSAM to generate candidate masks, DINOv2 to extract semantic features, and contrastive learning to match objects across views, achieving state-of-the-art performance on the Ego-Exo4D benchmark with only 1% of the trainable parameters used by prior methods.
tags:
  - ICCV 2025
  - Segmentation
  - Cross-View Segmentation
  - Mask Matching
  - Ego-Exo Correspondences
  - Contrastive Learning
  - DINOv2
date: 2026-05-08
content_hash: eed04e13745c628b
---

# O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views

**Conference**: ICCV 2025
**arXiv**: [2506.06026](https://arxiv.org/abs/2506.06026)
**Code**: [Maria-SanVil/O-MaMa](https://github.com/Maria-SanVil/O-MaMa)
**Area**: Image Segmentation
**Keywords**: Cross-View Segmentation, Mask Matching, Ego-Exo Correspondences, Contrastive Learning, DINOv2

## TL;DR

This work reframes cross-view (ego-exo) object segmentation as a mask matching problem. It leverages FastSAM to generate candidate masks, DINOv2 to extract semantic features, and contrastive learning to match objects across views, achieving state-of-the-art performance on the Ego-Exo4D benchmark with only 1% of the trainable parameters used by prior methods.

## Background & Motivation

Multi-agent collaboration scenarios — including multi-robot manipulation, AR assistants, and human-robot collaboration — require establishing object correspondences between egocentric (first-person) and exocentric (third-person) views. While single-image segmentation is well-studied, cross-view segmentation presents unique challenges:

**Severe viewpoint changes**: The egocentric view captures fine-grained hand-object interactions but suffers from high dynamics and motion blur, while the exocentric view covers the entire scene but exhibits large object scale variation.

**Occlusion and domain shift**: Differences in camera optics and imaging conditions introduce significant domain gaps.

**Failure of traditional geometric matching**: Even the state-of-the-art RoMa achieves only a 67.6% success rate in ego-exo settings.

Core insight: Rather than training a model to perform cross-view segmentation from scratch at the pixel level, one can leverage the zero-shot segmentation capability of SAM-based models to generate high-quality candidate masks, and then reduce the problem to determining *which* candidate mask corresponds to the target object — a matching problem.

## Method

### Overall Architecture

The O-MaMa pipeline proceeds as follows:
1. Generate $N$ candidate masks $\{\mathcal{M}_n\}_{n=1}^N$ in the target view using FastSAM.
2. Extract descriptors for each candidate mask via the Mask-Context Encoder.
3. Fuse global cross-view information via Ego↔Exo Cross Attention.
4. Learn view-invariant features using the Mask Matching Contrastive Loss.
5. At inference, select the candidate mask whose embedding is most similar to the source mask embedding.

### Key Designs

1. **Mask-Context Encoder**:

    - Dense feature maps $\psi(I)$ are extracted using DINOv2 ViT-B/14 and upsampled by 4× to preserve fine-grained spatial detail.
    - **Object descriptor** $\mathbf{o}_n$: average pooling of DINOv2 features over the mask region.
    - **Context descriptor** $\mathbf{c}_n$: average pooling over an expanded bounding box region, incorporating surrounding context to aid cross-view localization.
    - Design motivation: DINOv2's self-supervised features exhibit strong semantic understanding and object decomposition capability. Experiments confirm that Avg-Pool(mask) outperforms Avg-Pool(bbox), Max-Pool(bbox), centroid-based features, and CLIP features.

2. **Hard Negative Adjacent Mining**:

    - Problem: Neighboring objects share similar context but differ in identity; naive global negative sampling is insufficient for learning discriminative representations.
    - A Delaunay triangulation is applied to construct an adjacency graph over mask segments.
    - First- and second-order neighbors of each object are collected: $\mathcal{O}_n^- = \mathcal{N}(\mathbf{o}_n) \cup \mathcal{N}^2(\mathbf{o}_n)$.
    - Hard negatives are sampled from this neighborhood set for contrastive learning.
    - Ablations show this strategy yields gains of +4.2 IoU (Ego2Exo) and +1.2 IoU (Exo2Ego).

3. **Ego↔Exo Cross Attention**:

    - Candidate mask descriptors $\mathbf{o}_n$ serve as queries, while the full DINOv2 feature map $\psi(I^S)$ of the source image serves as keys and values.
    - Standard cross-attention is computed as: $\hat{\mathbf{o}}_n = \text{Softmax}(\frac{\mathbf{o}_n W_Q \cdot (\psi(I^S) W_K)^\top}{\sqrt{d}}) \cdot \psi(I^S) W_V$
    - Learnable positional encodings and LayerNorm are incorporated.
    - A cross-view embedding $\hat{\mathbf{o}}_S$ is similarly computed for the source mask in the target view.
    - Design motivation: Context embeddings encode only local information and lack global cross-view semantic associations.

### Loss & Training

- **Mask Matching Contrastive Loss**: based on InfoNCE, with hard negatives sampled from adjacent neighbors in a batch $\mathcal{B}$:

  $$\mathcal{L}_M(\rho^+, \rho_S) = -\log \frac{\exp(\text{sim}(f_\theta(\rho^+), f_\theta(\rho_S))/\tau)}{\sum_{n=1}^{|\mathcal{B}|} \exp(\text{sim}(f_\theta(\rho_n), f_\theta(\rho_S))/\tau)}$$

- The final descriptor $\rho_n = [\hat{\mathbf{o}}_n; \mathbf{c}_n; \mathbf{o}_n]$ (cross-view embedding + context + object) is projected into a shared latent space via a shallow MLP $f_\theta$.
- Optimizer: AdamW, lr=$8 \times 10^{-5}$, cosine annealing, batch size of 24 image pairs, 32 candidate masks sampled per target image.
- Hardware: 2× NVIDIA RTX 4090.

## Key Experimental Results

### Main Results (Tables)

**Ego-Exo4D Correspondences v2 Test Split**

| Method | Ego2Exo IoU ↑ | Exo2Ego IoU ↑ | Total IoU ↑ | Trainable Params (M) |
|--------|--------------|--------------|-------------|----------------------|
| XMem + XSegTx | 34.9 | 25.0 | 30.0 | 67.1 |
| PSALM (zero-shot) | 7.4 | 2.1 | 4.8 | 0 |
| k-NN baseline | 31.9 | 30.9 | 31.4 | 0 |
| **O-MaMa** | **42.6** | **44.1** | **43.4** | **11.6** |

**Ego-Exo4D Correspondences v1 Val Split**

| Method | Ego2Exo IoU ↑ | Exo2Ego IoU ↑ | Total IoU ↑ | Trainable Params (M) |
|--------|--------------|--------------|-------------|----------------------|
| PSALM (fine-tuned) | 41.3 | 44.1 | 42.7 | 1587.1 |
| ObjectRelator | 44.3 | 50.9 | 47.6 | 1587.3 |
| **O-MaMa** | **50.1** | **54.2** | **52.1** | **11.6** |

O-MaMa surpasses ObjectRelator (prev. SOTA) by +13.1% (Ego2Exo) and +6.5% (Exo2Ego) while using only **1%** of the trainable parameters.

### Ablation Study (Tables)

**Per-Module Ablation (10% Validation Set)**

| Config | $\mathcal{L}_M$ | Context | Adj.Neg | CrossAttn | Ego2Exo IoU | Exo2Ego IoU | Total IoU |
|--------|------|---------|---------|-----------|-------------|-------------|-----------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 35.2 | 34.9 | 35.1 |
| A | ✓ | ✗ | ✗ | ✗ | 42.2 | 44.7 | 43.5 |
| C | ✓ | ✓ | ✓ | ✗ | 46.9 | 45.6 | 46.3 |
| E (full) | ✓ | ✓ | ✓ | ✓ | **48.3** | **49.6** | **49.0** |

IoU gains over baseline: Ego2Exo +37.2%, Exo2Ego +42.1%.

**Mask Descriptor Comparison**

| Descriptor | k-NN Ego2Exo | k-NN Exo2Ego | Learned Ego2Exo | Learned Exo2Ego |
|------------|-------------|-------------|-----------------|-----------------|
| Avg-Pool(Mask)-DINOv2 | **35.2** | **34.9** | **42.2** | **44.7** |
| Avg-Pool(BBox)-DINOv2 | 21.8 | 21.2 | 27.8 | 44.1 |
| Avg-Pool(BBox)-CLIP | 24.5 | 23.9 | 27.5 | 40.4 |
| Centroid-DINOv2 | 25.6 | 24.1 | - | - |

DINOv2 mask-pooled features substantially outperform CLIP and alternative pooling strategies.

### Key Findings

- **Problem reformulation is the primary contribution**: Recasting cross-view segmentation as mask matching allows a zero-shot k-NN baseline (40.5 IoU) to already surpass many trained models.
- **Geometric constraints offer limited benefit**: RoMa achieves only a 67.6% success rate; geometric matching yields marginal improvement over contrastive learning (35.2→35.4 vs. 35.2→42.2).
- **DINOv2 > CLIP**: DINOv2's fine-grained semantic features outperform CLIP's coarser representations for this task.
- **Small objects remain challenging**: O-MaMa performs well on medium and large objects, but mask descriptors for very small objects carry insufficient discriminative information.
- **Inference speed**: approximately 250ms on average (FastSAM: 70ms).

## Highlights & Insights

1. **The power of problem reformulation**: Recasting the difficult pixel-level cross-view segmentation task as a mask-level retrieval/matching problem substantially reduces task difficulty, enabling a lightweight model to achieve state-of-the-art performance.
2. **DINOv2's object decomposition capability**: Self-supervised DINOv2 pretraining provides remarkably strong object-level semantic representations — even in a zero-shot setting, they surpass models trained specifically for this task.
3. **Delaunay triangulation for hard negative mining**: Spatial proximity is elegantly exploited to enhance the discriminability of contrastive learning, proving more effective than random negative sampling.
4. **Exceptional parameter efficiency**: 11.6M trainable parameters vs. ObjectRelator's 1587.3M, demonstrating that foundation model features are already of sufficient quality and require only minimal task-specific adaptation.

## Limitations & Future Work

- FastSAM may produce incomplete segmentations covering only part of an object, leading to correct matches with suboptimal IoU.
- Very small objects represent a primary bottleneck, as their mask descriptors contain insufficient information.
- Temporal information from video is not exploited (frames are processed independently); incorporating temporal continuity could further improve performance.
- The method depends on the quality of FastSAM proposals — if the target object is not covered by any candidate mask, matching is impossible.
- Stronger segmentation models such as SAM2 have not been explored as candidate generators.

## Related Work & Insights

- **Ego-Exo4D**: Provides a large-scale synchronized ego-exo video dataset and the associated Correspondences benchmark.
- **ObjectRelator**: Fine-tunes PSALM (LLM-based) for cross-view segmentation at very large parameter cost.
- **FastSAM / SAM**: Provide high-quality zero-shot segmentation, forming the foundation of this approach.
- **DINOv2**: A self-supervised visual foundation model that supplies object-level semantic representations.
- Insight: When foundation models already provide strong foundational capabilities (segmentation, feature extraction), lightweight task adaptation — such as contrastive learning combined with matching — may represent a more effective paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Reframing cross-view segmentation as mask matching is a concise and effective contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation on two dataset splits, comparison against multiple baselines, comprehensive ablations (modules, descriptors, geometric constraints), and task-level analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The method is intuitive, the architecture diagrams are clear, and the experimental analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Achieving state-of-the-art with 1% of prior parameters; the problem reformulation paradigm offers broad inspiration for cross-view understanding tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](../../CVPR2026/segmentation/learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](../../NeurIPS2025/segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)
- [\[ICCV 2025\] Refer to Any Segmentation Mask Group With Vision-Language Prompts](refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)

<!-- RELATED:END -->
