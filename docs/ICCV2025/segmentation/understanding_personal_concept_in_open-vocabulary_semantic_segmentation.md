---
title: >-
  [Paper Note] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation
description: >-
  [ICCV 2025][Segmentation][personalized segmentation] This paper introduces the Personalized Open-Vocabulary Semantic Segmentation (Personalized OVSS) task for the first time…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "personalized segmentation"
  - "open-vocabulary semantic segmentation"
  - "text prompt tuning"
  - "negative mask proposal"
  - "few-shot learning"
date: 2026-05-08
content_hash: dd89ef7c9b449125
---

# Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.11030](https://arxiv.org/abs/2507.11030)  
**Code**: None  
**Area**: Image Segmentation
**Keywords**: personalized segmentation, open-vocabulary semantic segmentation, text prompt tuning, negative mask proposal, few-shot learning

## TL;DR

This paper introduces the Personalized Open-Vocabulary Semantic Segmentation (Personalized OVSS) task for the first time, and proposes a plug-and-play method based on text prompt tuning. By incorporating negative mask proposals to suppress false positives and injecting visual embeddings to enrich personalized concept representations, the method enables recognition of user-specific object instances from only a few image-mask pairs, while preserving the original OVSS performance.

## Background & Motivation

Open-Vocabulary Semantic Segmentation (OVSS) enables segmentation with arbitrary text descriptions, but fails to understand personalized concepts — e.g., recognizing "my mug" requires distinguishing it from other mugs. This is critically important in practical applications:

**Limitations of OVSS**: Existing OVSS models are designed to distinguish between different categories (e.g., mug vs. bottle) rather than specific instances within the same category ("my mug" vs. "other mugs").

**Limitations of few-shot segmentation**: (a) It does not support open-vocabulary segmentation — only the given target category can be segmented; (b) it does not consider distinguishing specific instances among objects of the same category.

**Personalization demand**: In robotic assistant scenarios, users wish to simply say "bring my mug" without providing detailed descriptions each time.

**False positive problem in text prompt tuning**: Directly applying text prompt tuning can recognize the target concept, but also misidentifies other similar objects as the personalized concept (e.g., recognizing all birds as "my bird").

**Core motivation**: A plug-and-play method is needed that learns personal visual concepts from only a few image-mask pairs while maintaining normal segmentation performance for other categories.

## Method

### Overall Architecture

Three lightweight components are added on top of an off-the-shelf OVSS model (e.g., SAN, ODISE):
1. Learnable text embedding $\textbf{T}_{\text{per}}$ (for learning personalized concepts)
2. Negative mask proposals (to suppress false positives)
3. Visual embedding injection (to enrich personalized representations)

### Key Designs

1. **Text Prompt Tuning**:

    - A learnable text embedding $\textbf{T}_{\text{per}} \in \mathbb{R}^{1 \times D}$ is initialized using the text embedding of the target category name (e.g., "a photo of black footed albatross").
    - It is concatenated after the existing vocabulary text embeddings $\textbf{T}_{\text{open}}$: $\textbf{T} = [\textbf{T}_{\text{open}}; \textbf{T}_{\text{per}}]$.
    - Standard segmentation loss is used for training: $\mathcal{L}_{seg} = \lambda_1\mathcal{L}_{dice} + \lambda_2\mathcal{L}_{bce} + \lambda_3\mathcal{L}_{cls}$.
    - **Key finding**: Sole text prompt tuning improves recall (target can be recognized) but significantly degrades precision (other similar objects are also misidentified as the target), resulting in severe false positives.
    - **Design motivation**: Text prompt tuning is the most straightforward personalization approach, but must be coupled with negative masks to control false positives.

2. **Negative Mask Proposal**:

    - **Negative mask embedding**: Obtained via a learnable linear combination of existing mask embeddings: $\textbf{Z}_{\text{neg}} = \textbf{W}_{\text{Z}} \textbf{Z}_{\text{open}}$, $\textbf{W}_{\text{Z}} \in \mathbb{R}^{1 \times N}$.
    - **Negative mask**: Generated from existing mask proposals via a learnable convolutional layer $\textbf{W}_{\text{M}}$: $\textbf{M}_{\text{neg}} = \textbf{W}_{\text{M}} \textbf{M}_{\text{open}}$.
    - **Supervision signals**:
        - The negative mask embedding is trained to uniformly match all vocabulary entries except the personalized concept: $\mathcal{L}^{\text{neg}}_{\text{Z}} = -\sum_{i \neq k} \frac{1}{V-1}\log S[i,j]$.
        - The negative mask is supervised with BCE using $1 - \textbf{M}_{\text{gt}}$ as ground truth: $\mathcal{L}^{\text{neg}}_{\text{M}} = -(1-\textbf{M}_{\text{gt}})\log(\textbf{M}_{\text{neg}}) - \textbf{M}_{\text{gt}}\log(1-\textbf{M}_{\text{neg}})$.
    - After concatenation: $\textbf{Z} = [\textbf{Z}_{\text{open}}; \textbf{Z}_{\text{neg}}]$, $\textbf{M} = [\textbf{M}_{\text{open}}; \textbf{M}_{\text{neg}}]$.
    - **Design motivation**: Explicitly learning representations of "non-target regions" provides the model with contrastive signals. Unlike Yo'LLaVA, which requires collecting large numbers of negative samples, this method automatically generates negative masks from existing mask proposals without additional data.

3. **Injection of Visual Embeddings**:

    - The CLIP image encoder extracts feature maps: $\textbf{F} = \textbf{I}_{\text{enc}}(\textbf{X})$.
    - Target region features are extracted using the mask and averaged via masked pooling: $\textbf{F}_{\text{per}} = \frac{1}{\sum \mathbb{1}(\textbf{M}'_{\text{gt}}=1)} \sum \textbf{F} \odot \textbf{M}'_{\text{gt}}$.
    - $\textbf{F}_{\text{per}}$ is averaged across multiple reference images.
    - Interpolation fusion with text embeddings: $\textbf{T}_{\text{per}}^{vis} = \alpha \cdot \textbf{F}_{\text{per}} + (1-\alpha) \cdot \textbf{T}_{\text{per}}$.
    - **Design motivation**: Unimodal (pure text or pure visual) prompt tuning has limited representational capacity; text-visual joint encoding better captures fine-grained appearance features of personalized concepts.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda^{\text{neg}}_{\text{Z}}\mathcal{L}^{\text{neg}}_{\text{Z}} + \lambda^{\text{neg}}_{\text{M}}\mathcal{L}^{\text{neg}}_{\text{M}}$
- Only three parameter groups are trained — $\textbf{T}_{\text{per}}$, $\textbf{W}_{\text{M}}$, and $\textbf{W}_{\text{Z}}$ — while the OVSS model is fully frozen.
- Personalization training requires only 200 iterations.
- Supports K = 1, 3, 5 reference images.

## Key Experimental Results

### Main Results

| Dataset | Method | IoU$^{\text{per}}$ (K=5) | mIoU (K=5) | IoU$^{\text{per}}$ Gain |
|---|---|---|---|---|
| FSS$^{\text{per}}$ | SAN | 41.08 | 55.68 | baseline |
| FSS$^{\text{per}}$ | SAN + Ours | **56.80** | 55.85 | **+15.72** |
| CUB$^{\text{per}}$ | SAN | 68.25 | 77.32 | baseline |
| CUB$^{\text{per}}$ | SAN + Ours | **76.80** | 78.29 | **+8.55** |
| ADE$^{\text{per}}$ | SAN | 6.88 | 17.20 | baseline |
| ADE$^{\text{per}}$ | SAN + Ours | **26.15** | 17.19 | **+19.27** |
| FSS$^{\text{per}}$ | ODISE + Ours | **34.05** | 22.94 | +23.36 |
| ADE$^{\text{per}}$ | ODISE + Ours | **13.43** | 12.18 | +12.19 |

### Ablation Study

| Configuration | mIoU | IoU$^{\text{per}}$ | IoU$^{\text{per}}_{\text{precision}}$ | IoU$^{\text{per}}_{\text{recall}}$ | Note |
|---|---|---|---|---|---|
| No personalization | 77.32 | 68.25 | 92.25 | 72.95 | High precision, low recall |
| + Text prompt tuning | 77.89 | 69.70 | 74.75↓ | 91.04↑ | Recall improves but precision drops sharply |
| + Text + Negative mask | 77.89 | 73.71 | 80.07↑ | 90.17 | Negative mask effectively suppresses false positives |
| + Text + Visual injection | 77.65 | 65.94 | 70.06↓ | 91.58↑ | Visual injection further boosts recall but worsens precision |
| + Text + Negative mask + Visual injection | **78.29** | **76.80** | **84.51** | **89.07** | All three components combined yield optimal results |

### Key Findings

- **Negative mask is central**: Text prompt tuning alone causes precision to plummet from 92.25 to 74.75; the negative mask restores it to 80.07.
- **Visual injection requires negative mask**: Using visual injection alone actually degrades performance (65.94 < 68.25), but achieves the best results when combined with the negative mask (76.80).
- **mIoU remains stable**: Across all configurations, the original OVSS performance (mIoU) remains between 77 and 78, demonstrating that the method does not compromise existing capabilities.
- **K=1 is still effective**: Even a single reference image improves IoU$^{\text{per}}$ (SAN: 41.08→49.80), confirming practical utility.
- **Cross-model consistency**: Significant improvements are observed on both SAN and ODISE, two distinct OVSS model paradigms.

## Highlights & Insights

- **Valuable problem formulation**: Personalized OVSS is an overlooked yet practically important task direction; this paper is the first to define a complete task setup and evaluation framework.
- **Discovery and resolution of false positives**: The paper thoroughly analyzes why text prompt tuning causes false positives (via precision/recall decoupled analysis) and elegantly addresses the issue with negative mask proposals.
- **Plug-and-play design**: The method can be directly applied to any off-the-shelf OVSS model (SAN, ODISE, etc.) without modifying the model architecture.
- **Minimal parameter overhead**: Only a single text embedding vector and two linear layers need to be trained, converging within 200 iterations.
- **Benchmark contribution**: Three new benchmarks are established — FSS$^{\text{per}}$, CUB$^{\text{per}}$, and ADE$^{\text{per}}$.

## Limitations & Future Work

- When objects of the same category are visually very similar (e.g., mugs differing only by a small logo), the method's discriminative capacity may be insufficient.
- The evaluation benchmarks are primarily based on FSS-1000 and CUB-200; real-world scenarios may exhibit higher complexity.
- Visual embedding injection uses simple linear interpolation (fixed $\alpha$); more sophisticated fusion strategies could be explored.
- The simultaneous presence of multiple personalized concepts is not considered.
- The impact of occlusion and viewpoint variation in reference images on performance is not sufficiently analyzed.

## Related Work & Insights

- Related to the personalized VQA task in Yo'LLaVA, but Yo'LLaVA requires large numbers of negative samples, whereas this method generates them automatically.
- SAN and ODISE represent two distinct OVSS paradigms (CLIP-based vs. diffusion-based); the method's effectiveness on both demonstrates strong generalizability.
- The idea of negative mask proposals can be extended to other personalization tasks requiring false positive suppression (e.g., personalized detection, personalized VQA).
- The effectiveness of joint text-visual prompt tuning is consistent with multimodal prompt learning methods such as CoCoOp and MaPLe.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel and practical task definition; the negative mask proposal design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two models × three datasets × detailed ablations × qualitative analysis; comparison with more personalization methods is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem statement is clear, method description is intuitive, and figures are convincing.
- **Value**: ⭐⭐⭐⭐ Opens a new direction in personalized OVSS; both the benchmarks and the method serve as useful references.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation](stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)
- [\[ICCV 2025\] Auto-Vocabulary Semantic Segmentation](auto-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)

</div>

<!-- RELATED:END -->
