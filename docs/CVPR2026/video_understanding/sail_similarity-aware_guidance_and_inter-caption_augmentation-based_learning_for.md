---
title: >-
  [Paper Note] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning
description: >-
  [CVPR 2026][Video Understanding][Weakly-supervised dense video captioning] This paper proposes SAIL, which achieves state-of-the-art performance on both dense video captioning and event localization on ActivityNet and YouCook2 under a weakly-supervised setting (caption annotations only, no temporal boundaries), via cross-modal similarity-guided semantic-aware mask generation and auxiliary supervision from LLM-synthesized captions.
tags:
  - CVPR 2026
  - Video Understanding
  - Weakly-supervised dense video captioning
  - cross-modal alignment
  - LLM data augmentation
  - Gaussian mask
  - event localization
date: 2026-05-08
content_hash: 82d41b4520ee9824
---

# SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning

**Conference**: CVPR 2026
**arXiv**: [2603.05437](https://arxiv.org/abs/2603.05437)
**Code**: Unavailable
**Area**: Video Understanding
**Keywords**: Weakly-supervised dense video captioning, cross-modal alignment, LLM data augmentation, Gaussian mask, event localization

## TL;DR
This paper proposes SAIL, which achieves state-of-the-art performance on both dense video captioning and event localization on ActivityNet and YouCook2 under a weakly-supervised setting (caption annotations only, no temporal boundaries), via cross-modal similarity-guided semantic-aware mask generation and auxiliary supervision from LLM-synthesized captions.

## Background & Motivation
Dense Video Captioning (DVC) requires simultaneously localizing events and generating descriptions in untrimmed videos. Fully supervised methods rely on expensive temporal boundary annotations, whereas weakly-supervised DVC (WSDVC) trains using only caption annotations.

**Core problem with existing methods**: The current state-of-the-art method ILCACM employs a Gaussian mask strategy to achieve implicit event localization through complementary caption generation. However, its mask learning suffers from two fundamental deficiencies:

**Lack of semantic alignment in masks**: The model only learns non-overlapping mask distributions without considering the semantic relationship between masks and their corresponding events. Experiments show that even fixed, non-trainable uniform-distribution masks yield performance comparable to ILCACM — indicating that existing methods merely learn to cover different temporal regions rather than capturing semantically relevant ones.

**Annotation sparsity**: Event annotations in existing datasets are extremely sparse. For example, a 235-second video in ActivityNet may have only 3 event annotations, leaving a large number of potential events unannotated. Although annotations may span the entire video duration, event density remains consistently low.

## Method

### Overall Architecture
SAIL builds upon the Gaussian mask complementary caption generation framework of ILCACM and introduces two key components: (1) a cross-modal similarity-based mask guidance objective that encourages masks to focus on video regions semantically consistent with their corresponding captions; and (2) an LLM-generated synthetic caption augmentation mechanism with an inter-mask strategy to provide denser supervision signals.

### Key Designs

1. **Similarity-Aware Mask Guide**: Guides mask optimization through cross-modal alignment.

    - **Function**: Encourages Gaussian masks to emphasize video regions most semantically similar to their corresponding event captions.
    - **Mechanism**: After generating mask $M_i$, it is element-wise multiplied with video features to obtain the masked feature $\boldsymbol{v}'_i = \boldsymbol{v} \cdot M_i$. Leveraging CLIP's cross-modal alignment capability, the method maximizes the cosine similarity between the average-pooled masked feature $\bar{\boldsymbol{v}}'_i$ and the corresponding caption feature $\boldsymbol{c}_i$, while minimizing similarity to other event captions within the same video. A margin ranking loss is employed:
    $$\mathcal{L}_{\text{sim}} = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{N_s}\sum_{i=1}^{N_s}\max(0, \Delta - s^+_{b,i} + s^-_{b,i})$$
    where $s^+ = \text{sim}(\bar{\boldsymbol{v}}'_i, \boldsymbol{c}_i)$ is the positive-pair similarity and $s^- = \max_{j \neq i}\text{sim}(\bar{\boldsymbol{v}}'_i, \boldsymbol{c}_j)$ is the hard-negative similarity.
    - **Design Motivation**: Upgrades the weak constraint of "covering different regions" to a stronger constraint of "aligning with semantic content."

2. **LLM-Based Caption Augmentation**: Leverages LLM world knowledge to generate transitional event descriptions.

    - **Function**: For each pair of adjacent ground-truth captions $(C_i, C_{i+1})$, a synthetic caption $C^{syn}_i$ is generated for the temporal interval between them, yielding $N_s - 1$ synthetic captions per video.
    - **Mechanism**: A structured prompt is designed that defines the LLM as a "video context reasoning expert," tasked with analyzing the narrative flow between adjacent captions and inferring the most likely transitional action or state change. Qwen3-8B is used for generation.
    - **Design Motivation**: Addresses insufficient alignment signals caused by annotation sparsity, particularly for videos with only 1–2 event annotations.

3. **Inter-Mask Auxiliary Guidance**: An indirect utilization strategy for synthetic captions.

    - **Function**: Creates an "inter-mask" for each synthetic caption, localized to the temporal region between adjacent event masks.
    - **Mechanism**: For each pair of adjacent event centers $(c_i, c_{i+1})$, the inter-mask center is defined as their average $c^{inter}_i = \frac{c_i + c_{i+1}}{2}$, with width fixed at hyperparameter $w^{inter}$. After applying the inter-mask to video features, a cosine similarity loss aligns the augmented features with the synthetic captions:
    $$\mathcal{L}_{\text{aug}} = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{N_s-1}\sum_{i=1}^{N_s-1}(1 - \text{sim}(\bar{\boldsymbol{v}}'^{inter}_{b,i}, \boldsymbol{c}^{syn}_{b,i}))$$
    - **Design Motivation**: Directly incorporating synthetic captions as hard negatives introduces noise and degrades performance (validated in Table 6); treating them as an independent auxiliary signal is more robust.

### Loss & Training
- Final objective: $\mathcal{L} = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}} + \mathcal{L}_{\text{sim}} + \alpha_{\text{aug}}\mathcal{L}_{\text{aug}}$
- $\mathcal{L}_{\text{pos}}$/$\mathcal{L}_{\text{neg}}$: Positive/negative complementary caption generation losses (inherited from ILCACM).
- Hyperparameters: $\Delta=0.1$, $w^{inter}=0.6$, $\alpha_{\text{aug}}=0.25$.
- Caption decoder: Distilled-GPT2; optimizer: AdamW.
- ActivityNet: lr=1e-4, 10 epochs; YouCook2: lr=5e-5, 5+15 epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | SAIL | Prev. SOTA (ILCACM) | Gain |
|--------|------|------|-----------------|------|
| ActivityNet | CIDEr | 35.38 | 33.42 | +1.96 |
| ActivityNet | SODA_c | 6.29 | 6.08 | +0.21 |
| ActivityNet | F1 (Localization) | 57.00 | 56.20 | +0.80 |
| YouCook2 | CIDEr | 14.61 | 13.49 | +1.12 |
| YouCook2 | F1 (Localization) | 20.94 | 17.88 | +3.06 |

SAIL under weak supervision surpasses fully supervised methods CM2 and E2DVC on most metrics.

### Ablation Study

| Configuration | SODA_c | CIDEr | F1 | Note |
|------|--------|-------|-----|------|
| Baseline (ILCACM) | 6.08 | 33.42 | 56.20 | No semantic guidance |
| +Similarity-aware | 6.27 | 35.18 | 56.89 | Semantically aligned masks |
| +Synthetic captions | 6.29 | 34.92 | 56.79 | LLM-augmented supervision |
| +Both (SAIL) | 6.29 | 35.38 | 57.00 | Best combination |

### Key Findings
- The similarity-aware mask alone improves CIDEr by +1.76, demonstrating the effectiveness of the alignment loss.
- Using synthetic captions as auxiliary signals via inter-mask outperforms using them as hard negatives (+HN).
- Performance improves monotonically as the proportion of synthetic captions increases, with gains observed even at 25% usage.
- SAIL consistently improves performance across three mask designs — Gaussian, Hard Binary, and Cauchy — demonstrating the generality of the approach.
- Training overhead is negligible: 1h41m vs. ILCACM's 1h38m; inference is even slightly faster (7m01s vs. 7m11s).

## Highlights & Insights
1. **Insight from fixed-mask experiments**: Non-trainable uniform-distribution masks match ILCACM's performance, revealing that the masks learned by existing methods effectively lack semantic information.
2. **Elegant use of LLM augmentation**: Rather than mixing synthetic captions directly into the main loss (which introduces noise), the method employs inter-masks as an independent auxiliary signal — functioning as "soft narrative guidance" rather than "hard constraints."
3. **Weak supervision surpassing full supervision**: SAIL matches fully supervised methods on ActivityNet localization F1 and outperforms them on several captioning metrics, suggesting that semantic alignment is a more essential supervisory signal than temporal boundary annotations.

## Limitations & Future Work
- The improvement in SODA_c is modest (+0.21), indicating limited gains in narrative coherence.
- The quality of LLM-generated synthetic captions depends on the LLM's world knowledge and may be less accurate in specialized domains (e.g., cooking, sports).
- The inter-mask width $w^{inter}$ is a fixed hyperparameter rather than adaptively determined.
- Evaluation is limited to two datasets; generalization to larger-scale or more diverse datasets remains unverified.

## Related Work & Insights
- Built upon ILCACM (current WSDVC state-of-the-art) with minimal modifications, achieving substantial performance gains.
- The use of CLIP's cross-modal alignment capability to guide temporal mask learning is generalizable to other weakly-supervised video understanding tasks.
- The idea of using LLMs to generate transitional event descriptions is highly inspiring — it leverages LLMs' narrative reasoning ability to complement sparse annotations.
- The approach offers reference value for weakly-supervised video grounding, temporal action detection, and related tasks.

## Rating
- **Novelty**: ⭐⭐⭐ — The core intuition is clear, but the technical contribution is incremental (adding losses and augmentation on top of ILCACM).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations covering mask types, data proportions, and utilization strategies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation analysis is thorough; the fixed-mask experiment provides a particularly compelling insight.
- **Value**: ⭐⭐⭐ — Surpassing full supervision under weak supervision is practically meaningful, though the margin of improvement is modest.
- **Value**: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](stay_in_your_lane_role_specific_queries_with_overlap_suppression_loss_for_dense_.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](../../ACL2026/video_understanding/rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[CVPR 2026\] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding](mamba-vmr_multimodal_query_augmentation_via_generated_videos_for_precise_tempora.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
