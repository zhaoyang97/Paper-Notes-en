---
title: >-
  [Paper Note] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations
description: >-
  [ICCV 2025][Segmentation][Referring Video Object Segmentation] This paper proposes ReferDINO, which end-to-end adapts the GroundingDINO visual grounding foundation model to the Referring Video Object Segmentation (RVOS)…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Referring Video Object Segmentation"
  - "Visual Grounding"
  - "GroundingDINO"
  - "Deformable Attention"
  - "Query Pruning"
date: 2026-05-08
content_hash: 322ac2758e5d4898
---

# ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations

**Conference**: ICCV 2025
**arXiv**: [2501.14607](https://arxiv.org/abs/2501.14607)  
**Code**: [Project Page](https://isee-laboratory.github.io/ReferDINO)  
**Area**: Image Segmentation
**Keywords**: Referring Video Object Segmentation, Visual Grounding, GroundingDINO, Deformable Attention, Query Pruning

## TL;DR

This paper proposes ReferDINO, which end-to-end adapts the GroundingDINO visual grounding foundation model to the Referring Video Object Segmentation (RVOS) task. By introducing a grounding-guided deformable mask decoder, an object-consistent temporal enhancer, and a confidence-based query pruning strategy, ReferDINO significantly surpasses state-of-the-art methods across five benchmarks (e.g., +3.9% $\mathcal{J}\&\mathcal{F}$ on Ref-YouTube-VOS) while achieving real-time inference at 51 FPS.

## Background & Motivation

Referring Video Object Segmentation (RVOS) segments target objects in videos based on textual descriptions, requiring three capabilities: deep vision-language understanding, pixel-level dense prediction, and spatiotemporal reasoning. Existing methods face the following core challenges:

**Insufficient Vision-Language Capability**: Existing RVOS models frequently confuse visually similar objects when handling complex attribute descriptions (e.g., "shape + color" combinations), owing to the limited scale of RVOS training data.

**Limitations of Visual Grounding Foundation Models**: Although GroundingDINO provides strong object-level VL understanding, it (a) only predicts bounding boxes without producing masks (lacking pixel-level dense prediction), and (b) cannot reason about dynamic attributes (e.g., "a cat wagging its tail").

**Efficiency Bottleneck**: GroundingDINO employs 900 queries, making frame-by-frame video processing computationally expensive.

**Non-End-to-End Integration**: Existing approaches (e.g., Grounded-SAM-2) cascade GroundingDINO and SAM2 in a non-end-to-end, non-differentiable pipeline, precluding further joint optimization.

**Core Motivation**: An end-to-end framework is needed to seamlessly integrate GroundingDINO's open-world grounding knowledge (region-level VL alignment) with pixel-level segmentation and spatiotemporal reasoning capabilities.

## Method

### Overall Architecture

ReferDINO augments GroundingDINO with three new modules:
1. Confidence Query Pruning → reduces per-frame computation
2. Object-Consistent Temporal Enhancer → enables cross-frame interaction
3. Grounding-Guided Deformable Mask Decoder → produces high-quality masks

Workflow: video frames are fed into GroundingDINO frame-by-frame → query pruning → per-frame object feature collection → temporal enhancement → mask decoding → output mask sequences.

### Key Designs

1. **Grounding-Guided Deformable Mask Decoder**:

    - **Core Design**: Box prediction and mask prediction are cascaded into a grounding–deformation–segmentation pipeline rather than operating in parallel.
    - The predicted box center $\{b_x, b_y\}$ is directly used as the reference point for deformable attention, injecting grounding priors into mask prediction.
    - Object features $\tilde{\boldsymbol{o}}$ serve as Query, and FPN high-resolution features $\boldsymbol{F}_{\text{seg}}$ serve as Memory.
    - Sampling is performed via bilinear interpolation, making the process **end-to-end differentiable** so that segmentation gradients can back-propagate to optimize box predictions.
    - A cross-modal attention layer (Query = object features, K/V = text features) is further applied to suppress background noise.
    - **Comparison with Dynamic Mask Head**: The latter stores independent high-resolution feature maps per object, incurring substantial memory overhead; ReferDINO performs position-guided sampling over a shared feature map with zero additional memory cost.
    - The decoder consists of $L_m$ blocks, each comprising deformable cross-attention followed by cross-modal attention.

2. **Object-Consistent Temporal Enhancer**:

    - Composed of a memory-augmented tracker and a cross-modal temporal decoder.
    - **Memory-Augmented Tracker**: Applies the Hungarian algorithm with cosine similarity to align objects across adjacent frames; memory is updated via momentum as $\mathcal{M}^t = (1 - \alpha \cdot \boldsymbol{c}^t) \cdot \mathcal{M}^{t-1} + \alpha \cdot \boldsymbol{c}^t \cdot \hat{\mathcal{O}}^t$, where $\boldsymbol{c}^t$ denotes the object-text similarity, preventing frames in which the target is invisible from corrupting long-term memory.
    - **Cross-Modal Temporal Decoder**: Injects **time-varying text features** $\{\boldsymbol{f}_{\text{cls}}^t\}$ as frame proxies for cross-frame interaction, rather than relying on static text embeddings; dynamic information is extracted via temporal self-attention followed by cross-attention (text as Query, objects as K/V).
    - Design Motivation: (a) GroundingDINO processes each frame independently, lacking temporal consistency; (b) prior methods use static text embeddings, which cannot capture dynamic changes (e.g., action descriptions).

3. **Confidence Query Pruning**:

    - Low-confidence queries are progressively pruned at each layer of the cross-modal decoder.
    - Confidence is defined by two terms: $s_j = \frac{1}{N_l-1}\sum_{i \neq j} \boldsymbol{A}^s_{ij} + \max_k \boldsymbol{A}^c_{kj}$
        - First term: average attention received by the $j$-th query from all other queries (irreplaceability).
        - Second term: maximum probability that the $j$-th object is mentioned in the text.
    - The top $1/k$ high-confidence queries are retained at each layer, yielding $N_s \ll N_q = 900$ at the final layer.
    - Decoder time complexity is reduced from $O(L \cdot N^2 d)$ to $O(\frac{k^2}{k^2-1} N^2 d)$, independent of depth $L$.
    - At $k=2$, decoder computation is reduced to 24.7% of the original.

### Loss & Training

- Hungarian matching selects the prediction sequence with minimum cost as the positive sample.
- Total loss: $\mathcal{L}_{\text{total}} = \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{box}}\mathcal{L}_{\text{box}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$
- $\mathcal{L}_{\text{cls}}$: focal loss; $\mathcal{L}_{\text{box}}$: L1 + GIoU; $\mathcal{L}_{\text{mask}}$: DICE + binary focal + projection loss.
- The backbone is frozen; the cross-modal Transformer is fine-tuned using LoRA (rank = 32).
- The model is first pre-trained on RefCOCO/+/g image data, then fine-tuned on RVOS data.

## Key Experimental Results

### Main Results

| Dataset | Metric | ReferDINO (Swin-T) | DsHmp (CVPR'24) | SOC (NeurIPS'23) | Gain |
|---|---|---|---|---|---|
| Ref-YouTube-VOS | $\mathcal{J}\&\mathcal{F}$ | **67.5** | 63.6 | 62.4 | +3.9 |
| MeViS | $\mathcal{J}\&\mathcal{F}$ | **48.0** | 46.4 | - | +1.6 |
| Ref-DAVIS17 | $\mathcal{J}\&\mathcal{F}$ | **66.7** | 64.0 | 63.5 | +2.7 |
| Ref-YouTube-VOS (Swin-B) | $\mathcal{J}\&\mathcal{F}$ | **69.3** | 67.1 | 66.0 | +2.2 |
| MeViS (Swin-B) | $\mathcal{J}\&\mathcal{F}$ | **49.3** | - | - | - |

### Ablation Study

| Configuration | $\mathcal{J}\&\mathcal{F}$ (MeViS) | Note |
|---|---|---|
| ReferDINO | 48.0 | Full model |
| w/o CMA (remove cross-modal attention) | 47.6 (−0.4) | Cross-modal filtering provides auxiliary benefit |
| w/o DCA (remove deformable cross-attention) | 45.3 (−2.7) | Grounding-guided sampling is the core component |
| w/o Tracker | 47.6 (−0.4) | Tracker contributes temporal consistency |
| w/o Temporal Decoder | 45.8 (−2.2) | Temporal decoder is a critical component |
| 50% query pruning | 67.5 (vs. 67.6 full) | Negligible performance loss, −40.6% FLOPs |
| Random 50% pruning | 38.3 (−29.3) | Random pruning causes catastrophic degradation |

### Key Findings

- ReferDINO with Swin-T outperforms the GroundingDINO baseline with Swin-B backbone (67.5 vs. 66.7 $\mathcal{J}\&\mathcal{F}$).
- The query pruning strategy achieves a 10× speedup (4.9 → 51 FPS) with negligible performance loss.
- The grounding-guided deformable cross-attention (DCA) contributes the largest gain (−2.7%), validating the importance of box priors for mask quality.
- Comparisons against G-DINO+SH/DH baselines demonstrate that naive adaptation is insufficient to fully unlock the potential of the foundation model.

## Highlights & Insights

- **End-to-End Adaptation Paradigm**: The first work to adapt GroundingDINO end-to-end for RVOS, enabling differentiable joint optimization of box and mask predictions.
- **Grounding-to-Mask Cascade Design**: The pre-trained box predictions are elegantly leveraged as spatial priors to guide mask generation, outperforming a parallel design.
- **Time-Varying Text Embeddings**: The cross-modal encoder of GroundingDINO generates frame-specific text representations, capturing temporal dynamics more effectively than static text embeddings.
- **Efficient Query Pruning**: Confidence scores are computed by reusing decoder self-attention weights, introducing zero additional computational overhead.
- **Real-Time Performance**: The 51 FPS inference speed makes the method directly applicable to real-time video scenarios.

## Limitations & Future Work

- Freezing the backbone and applying LoRA fine-tuning may limit the depth of adaptation to RVOS-specific scenarios.
- The momentum-based update in the memory-augmented tracker may suffer from drift in long videos.
- The model supports only single-target RVOS (multi-target selection on MeViS is handled via thresholding); scalability to multi-target settings has not been thoroughly validated.
- Integration with segmentation foundation models such as SAM2 remains unexplored.

## Related Work & Insights

- Compared to Video-GroundingDINO (which merely adds temporal self-attention), ReferDINO presents a more thorough adaptation encompassing a mask decoder, temporal enhancer, and query pruning.
- The strategy of setting deformable attention reference points using box centers (replacing MLP-generated references) is a concise and effective design choice.
- The query pruning approach is generalizable to other DETR-like models for inference acceleration.
- The end-to-end foundation model adaptation paradigm offers valuable insights for other downstream tasks such as video grounding and temporal action detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The end-to-end foundation model adaptation paradigm is original, and the grounding-to-mask cascade design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, two backbone variants, detailed ablations, and efficiency analysis — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure with tightly coupled motivation and methodology; high-quality figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new state of the art on RVOS while achieving real-time inference, offering strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](../../NeurIPS2025/segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)

</div>

<!-- RELATED:END -->
