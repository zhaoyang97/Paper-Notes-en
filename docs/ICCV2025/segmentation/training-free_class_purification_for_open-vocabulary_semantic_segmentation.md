---
title: >-
  [Paper Note] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation
description: >-
  [ICCV 2025][Segmentation][Open-vocabulary semantic segmentation] This paper proposes FreeCP, a training-free class purification framework that addresses class redundancy and visual-language ambiguity arising from over-complete vocabularies in open-vocabulary semantic segmentation (OVSS), via a two-stage strategy of redundancy purification and ambiguity purification. As a plug-and-play module, FreeCP consistently improves existing methods across eight benchmarks.
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Open-vocabulary semantic segmentation"
  - "training-free"
  - "class purification"
  - "CLIP"
  - "class redundancy"
  - "visual-language ambiguity"
date: 2026-05-08
content_hash: d8cb2631e15de6aa
---

# Training-Free Class Purification for Open-Vocabulary Semantic Segmentation

**Conference**: ICCV 2025  
**arXiv**: [2508.00557](https://arxiv.org/abs/2508.00557)  
**Code**: [GitHub](https://github.com/chenqi1126/FreeCP)  
**Area**: Image Segmentation  
**Keywords**: Open-vocabulary semantic segmentation, training-free, class purification, CLIP, class redundancy, visual-language ambiguity

## TL;DR

This paper proposes FreeCP, a training-free class purification framework that addresses class redundancy and visual-language ambiguity arising from over-complete vocabularies in open-vocabulary semantic segmentation (OVSS), via a two-stage strategy of redundancy purification and ambiguity purification. As a plug-and-play module, FreeCP consistently improves existing methods across eight benchmarks.

## Background & Motivation

Open-vocabulary semantic segmentation (OVSS) aims to perform pixel-level segmentation of images based on arbitrary text category descriptions. Large-scale vision-language models such as CLIP have been widely adopted for this task owing to their strong generalization to novel categories. Existing methods are broadly divided into training-required and training-free approaches, with the latter attracting increasing attention due to zero computational overhead.

However, two critical yet overlooked challenges confront current training-free methods:

**Class Redundancy**: Inference requires a vocabulary containing a large number of potential categories, while in practice only a small subset of categories is present in the current test image. Absent categories generate false-positive activations that corrupt segmentation results. For example, in an image containing only sky, grass, and houses, categories such as "ocean" and "door" still produce non-trivial responses.

**Visual-Language Ambiguity**: Semantically similar categories in the vocabulary (e.g., "leaf/shrub/tree", "river/water") produce highly overlapping activation maps over the same visual regions, leading to classification confusion.

The authors validate the severity of these issues through a key experiment: restricting prediction to only ground-truth present categories yields substantial accuracy gains—from 59.4% to 72.0% on VOC21 and from 15.6% to 37.5% on ADE. This demonstrates that redundant and ambiguous categories are a critical performance bottleneck.

Further analysis reveals that after applying affinity-based refinement of CAMs using CLIP's self-attention matrix, activation maps of truly present categories remain spatially consistent, while those of redundant categories undergo significant distortion. Meanwhile, refined activation maps of ambiguous categories exhibit high spatial overlap with one another. These findings provide the theoretical basis for the subsequent class purification design.

## Method

### Overall Architecture

FreeCP is built upon the CLIP ViT backbone and comprises three stages:
1. Extract patch tokens $\mathbf{F}^p$ and text representations $\mathbf{T}$ using the CLIP image and text encoders, respectively.
2. Compute image-text affinity (CAM) and image self-affinity (self-attention matrix) to obtain class activation maps $\mathbf{M}$ and their refined counterparts $\tilde{\mathbf{M}}$.
3. Sequentially apply redundancy purification and ambiguity purification, followed by argmax to produce the final segmentation prediction.

### Key Design 1: CAM and Affinity Refinement

Class activation maps are computed via softmax-normalized image-text cosine similarity:

$$\mathbf{M}_j = \text{Reshape}\left(\frac{\exp(\text{Sim}(\mathbf{F}^p, \mathbf{T}_j))}{\sum_j \exp(\text{Sim}(\mathbf{F}^p, \mathbf{T}_j))}\right)$$

The image self-affinity matrix $SA$ is obtained by averaging self-attention matrices across multiple layers:

$$SA = \frac{1}{L}\sum_{l}^{L}\psi(A_l)$$

The refined activation map is then: $\tilde{\mathbf{M}}_i = \mathbf{M}_i \times SA$

### Key Design 2: Spatial Consistency Metric

IoU is introduced as a coarse-grained measure of Spatial Consistency (SC):

$$\text{SC}(\mathbf{X}, \mathbf{Y}) = \frac{\sum[\mathbf{X} \cdot \mathbf{Y}]}{\sum[\mathbf{X} + \mathbf{Y} - \mathbf{X} \cdot \mathbf{Y}]}$$

This metric is used for both redundancy detection (intra-class SC) and ambiguity detection (inter-class SC).

### Key Design 3: Redundancy Purification

The intra-class SC is computed for each category by comparing its activation maps before and after refinement:

$$S_i = \text{SC}(\mathbf{M}_i, \tilde{\mathbf{M}}_i)$$

If $S_i < T_{rp}$ (a predefined threshold), the category is considered redundant and removed from the candidate set. The intuition is that truly present categories maintain consistent spatial distributions after refinement, whereas redundant categories, lacking genuine visual correspondences, accumulate spurious responses upon refinement.

### Key Design 4: Ambiguity Purification

Over the purified category set $K'$ after redundancy purification, inter-class SC is computed for all category pairs:

$$P_{i,j} = \text{SC}(\tilde{\mathbf{M}}_i, \tilde{\mathbf{M}}_j)$$

After binarization with threshold $T_{ap}$, depth-first search (DFS) is applied to extract connected category groups, forming ambiguity groups. For each ambiguity group:
1. The activation maps of the group are averaged to localize high-response regions, from which bounding boxes are extracted.
2. The ambiguous region is cropped from the original image and fed into the CLIP image encoder to obtain visual features $\hat{\mathbf{F}}^c$.
3. Fine-grained textual descriptions $\hat{\mathbf{T}}_k$ for each candidate category are pre-generated using an LLM.
4. The final category for the local region is determined via cosine similarity: $k^* = \arg\max_k \text{Sim}(\hat{\mathbf{F}}^c, \hat{\mathbf{T}}_k)$

### Loss & Training

FreeCP operates entirely at inference time and requires no training; thus no loss function design is involved.

## Key Experimental Results

### Main Results

FreeCP is evaluated as a plug-and-play module on 8 benchmarks against existing training-free OVSS methods, yielding consistent improvements:

| Method | VOC21 | PC60 | Object | VOC20 | City | PC59 | ADE | Stuff | Avg. |
|------|-------|------|--------|-------|------|------|-----|-------|------|
| SCLIP | 59.1 | 30.4 | 30.5 | 80.4 | 32.2 | 34.2 | 16.1 | 22.4 | 38.2 |
| SCLIP + FreeCP | **65.8** | **35.3** | **37.2** | **84.3** | **33.3** | **38.0** | **18.4** | **24.9** | **42.1** |
| ClearCLIP | 51.8 | 32.6 | 33.0 | 80.9 | 30.0 | 35.9 | 16.7 | 23.9 | 38.1 |
| ClearCLIP + FreeCP | **64.5** | **35.7** | **36.9** | **81.5** | **34.4** | **39.3** | **18.9** | **26.1** | **42.2** |
| MaskCLIP | 43.4 | 23.2 | 20.6 | 74.9 | 24.9 | 26.4 | 11.9 | 16.7 | 30.3 |
| MaskCLIP + FreeCP | **64.4** | **34.7** | **36.2** | **84.1** | **32.5** | **36.6** | **17.6** | **23.3** | **41.2** |

FreeCP improves average mIoU by **+10.9%** for MaskCLIP, **+3.9%** for SCLIP, and **+4.1%** for ClearCLIP.

### Ablation Study: Class Purification Strategy

| Method | VOC21 | PC60 | Object | City | ADE | Stuff |
|------|-------|------|--------|------|-----|-------|
| Baseline | 59.8 | 31.6 | 34.5 | 32.0 | 17.2 | 23.2 |
| + Refine (no purification) | 27.5 | 21.1 | 11.9 | 26.0 | 9.1 | 14.3 |
| + RP (redundancy purification only) | 65.8 | 35.1 | 37.2 | 33.2 | 17.8 | 24.1 |
| + AP (ambiguity purification only) | 37.7 | 26.1 | 13.6 | 24.0 | 10.8 | 15.0 |
| + RP-AP (FreeCP) | **65.8** | **35.3** | **37.2** | **33.3** | **18.4** | **24.9** |

Key findings:
- Applying refinement without purification causes a **dramatic performance drop** (VOC21: 59.8→27.5).
- RP is the primary contributor to performance gains; AP provides additional improvement following RP, particularly on datasets with more fine-grained categories.
- AP must be applied after RP; reversing the order (AP→RP) leaves interference from redundant categories unresolved.

### Key Findings

1. **Strong generalizability**: FreeCP is effective regardless of the baseline method's initial performance level; MaskCLIP, with the lowest initial performance, achieves the largest gain.
2. **Effect of textual descriptions**: Fine-grained descriptions generated by different LLMs have a minor impact on results; Vicuna-13b slightly outperforms GPT-3.5, yet FreeCP surpasses the baseline under all description choices.
3. **No post-processing required**: FreeCP achieves state-of-the-art results without relying on post-processing techniques such as denseCRF or PAMR.

## Highlights & Insights

1. **Novel problem framing**: The paper is the first to systematically analyze the impact of class redundancy and visual-language ambiguity on OVSS performance, providing compelling evidence through GT vocabulary ablation experiments.
2. **Elegant solution**: Leveraging the change in spatial consistency before and after CAM refinement to distinguish genuine categories from redundant or ambiguous ones is intuitive and well-motivated.
3. **Plug-and-play design**: Requires no training and can be directly integrated into any CLIP-based training-free OVSS method.
4. **Threshold adaptivity**: Adjusting thresholds according to the semantic complexity of each dataset demonstrates flexibility across diverse scenarios.

## Limitations & Future Work

1. The thresholds $T_{rp}$ and $T_{ap}$ require manual tuning based on dataset-specific prior knowledge, lacking an adaptive strategy.
2. Ambiguity resolution depends on the quality of fine-grained descriptions generated by LLMs, implying an implicit dependency on LLM capability.
3. Gains are limited in simple scenarios with very few categories (e.g., only +1.1% on Cityscapes).
4. CAM computation and affinity refinement introduce additional inference overhead.

## Related Work & Insights

- **Training-free OVSS**: MaskCLIP, SCLIP, GEM, ClearCLIP, and others achieve pixel-level segmentation by modifying CLIP's self-attention; CaR progressively filters irrelevant text categories.
- **Prototype-based methods**: ReCo, OVDiff, and FreeDA leverage generative models to synthesize visual references.
- **Weakly supervised CAM**: AffinityNet and related works use affinity matrices to enhance class activation maps.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The class purification perspective is novel, though the specific designs for redundancy filtering and ambiguity resolution are relatively straightforward.
- **Technical Quality**: ⭐⭐⭐⭐ — Comprehensive experiments with thorough ablations, but threshold settings rely on dataset priors.
- **Practicality**: ⭐⭐⭐⭐⭐ — Training-free and plug-and-play; highly practical value.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and methodology are logically clear, with persuasive visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[CVPR 2026\] The Power of Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA](../../CVPR2026/segmentation/the_power_of_prior_training-free_open-vocabulary_semantic_segmentation_with_llav.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)

</div>

<!-- RELATED:END -->
