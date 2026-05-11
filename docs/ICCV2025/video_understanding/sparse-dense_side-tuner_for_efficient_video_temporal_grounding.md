---
title: >-
  [Paper Note] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding
description: >-
  [ICCV 2025][Video Understanding][Video Temporal Grounding] This paper proposes SDST (Sparse-Dense Side-Tuner), the first anchor-free side-tuning architecture for video temporal grounding (VTG). Through a sparse-dense dua…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Side-Tuning"
  - "Parameter-Efficient Fine-Tuning"
  - "Deformable Attention"
  - "InternVideo2"
date: 2026-05-08
content_hash: 8b6d005ba8707056
---

# Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding

**Conference**: ICCV 2025
**arXiv**: [2507.07744](https://arxiv.org/abs/2507.07744)
**Code**: [GitHub](https://github.com/davidpujol/SDST)
**Area**: Video Understanding
**Keywords**: Video Temporal Grounding, Side-Tuning, Parameter-Efficient Fine-Tuning, Deformable Attention, InternVideo2

## TL;DR

This paper proposes SDST (Sparse-Dense Side-Tuner), the first anchor-free side-tuning architecture for video temporal grounding (VTG). Through a sparse-dense dual-stream design, SDST jointly addresses moment retrieval (MR) and highlight detection (HD). A novel Reference-based Deformable Self-Attention (RDSA) module is introduced to resolve the context deficiency in standard deformable cross-attention. SDST achieves state-of-the-art or highly competitive results on QVHighlights, TACoS, and Charades-STA while reducing trainable parameters to 27% of the current SOTA.

## Background & Motivation

Video temporal grounding (VTG) requires localizing specific moments in a video (MR) and detecting highlight clips (HD) based on a natural language query. Existing methods face several key challenges:

**Limitations of frozen features**: Most methods rely solely on the last-layer features of a frozen pretrained backbone (e.g., CLIP). When a significant distribution shift exists between the pretraining domain and the downstream task, performance degrades substantially — a particularly pronounced issue when applying image-domain backbones to video-domain tasks.

**Infeasibility of full fine-tuning**: Fully fine-tuning large vision-language models incurs prohibitive computational costs. Although parameter-efficient fine-tuning (PEFT) methods such as Prompt tuning and Adapters reduce trainable parameters, they still require backpropagation through the entire backbone, resulting in high memory consumption.

**Deficiencies of existing side-tuning methods**: R2-Tuning is the first side-tuning method applied to VTG, but it employs an anchor-based design that treats the problem from a frame-level refinement perspective, neglecting the inherent sparsity of the MR task. Experiments demonstrate that such anchor-based approaches underperform on MR.

**Context limitation of deformable attention**: The deformable attention module — central to anchor-free methods such as DETR variants — exhibits an implicit context deficiency in cross-attention settings. Because queries and keys originate from different spaces, the CNN-based offset predictor cannot provide the query with contextual information from the key/value space, causing predicted offsets to collapse near their initial values.

## Method

### Overall Architecture

SDST is a dual-stream side-tuning architecture attached to the last $K$ intermediate layers of a frozen InternVideo2-1B backbone. Given input video and text, InternVideo2 first extracts $K$ intermediate visual-textual representations, which are then used to recursively refine two streams via weight-shared SDST layers:

- **Dense Stream ($\mathcal{D}$)**: Refines frame-level embeddings, suited for the HD task.
- **Sparse Stream ($\mathcal{S}$)**: Refines recurrent decoder queries, suited for the MR task.

The overall recurrence is formulated as:

$$\mathbf{D}^{\ell+1}, \mathbf{R}^{\ell+1}, \mathbf{H}^{\ell+1} = SDST(\mathbf{D}^{\ell}, \mathbf{R}^{\ell}, \mathbf{H}^{\ell}, \tilde{\mathbf{V}}^{\ell}, \tilde{\mathbf{T}}^{\ell})$$

### Key Designs

1. **Dense Learning Stream**:

    - **Function**: Progressively refines frame-level dense embeddings $\mathbf{D}^{\ell}$ by fusing multimodal information and modeling temporal relationships.
    - **Mechanism**: Intermediate visual and textual features are first projected into a shared $F$-dimensional space. Visual information is then fused into the dense embeddings via a weighted sum: $\mathbf{D}^{\ell} := \beta^{\ell}\mathbf{D}^{\ell} + (1-\beta^{\ell})\mathbf{V}^{\ell}$, where $\beta^{\ell}$ is a zero-initialized layer-wise scalar. Cross-attention and self-attention are subsequently applied to inject textual information and model temporal structure: $\mathbf{D}^{\ell+1} = PFFN(SA(CA(\mathbf{D}^{\ell}, \mathbf{T}^{\ell}, \mathbf{T}^{\ell})))$
    - **Design Motivation**: Frame-level embeddings are naturally suited for predicting per-frame saliency scores (HD task) and serve as the foundational signal for the sparse stream.

2. **Sparse Learning Stream**:

    - **Function**: Refines recurrent decoder queries, comprising center-width references $\mathbf{R}^{\ell} \in \mathbb{R}^{M \times 2}$ and corresponding hidden embeddings $\mathbf{H}^{\ell} \in \mathbb{R}^{M \times F}$.
    - **Mechanism**: Cross-attention and self-attention are first applied to inject textual information and facilitate information flow across moment proposals. RDSA then injects video-modal information into the queries: $\mathbf{H}^{\ell} = PFFN(RDSA(\mathbf{R}^{\ell}, \mathbf{H}^{\ell}, \mathbf{D}^{\ell+1}))$
    - **Design Motivation**: MR is an inherently sparse task (a video may contain very few ground-truth actions). DETR-based anchor-free architectures have been shown to outperform anchor-based methods on sparse detection tasks.

3. **Reference-based Deformable Self-Attention (RDSA)**:

    - **Function**: Replaces standard deformable cross-attention to resolve the context deficiency arising when queries lack information from the key/value space.
    - **Mechanism**: Rather than predicting offsets and attention scores from learnable queries $\mathbf{H}^{\ell}$, RDSA extracts three salient action embeddings from the dense embeddings — left endpoint (l), center (c), and right endpoint (r) — and uses them as new queries: $\hat{\mathbf{Q}} = \hat{\mathbf{X}}_{\mathcal{Q}} \mathbf{W}_{\mathcal{Q}}^{def}$, where $\hat{\mathbf{X}}_{\mathcal{Q}} = CNN(\mathbf{D}^{\ell})[l, c, r] \in \mathbb{R}^{M \times 3F}$. Since both queries and keys now originate from the same latent space (the dense embeddings), the context deficiency inherent in standard cross-attention is naturally resolved.
    - **Design Motivation**: In standard deformable CA, $\mathbf{X}_{\mathcal{Q}} \neq \mathbf{X}_{\mathcal{K}}$; the CNN offset predictor has no access to contextual information from the key space, causing offsets to collapse near their initial values. This also prevents the model from attending to frames outside the current estimated boundaries — a critical capability for refining short actions into longer ones.

4. **Intermediate Representation Extraction from InternVideo2**:

    - **Function**: Addresses the pooling challenge when extracting intermediate visual representations from the InternVideo2 backbone.
    - **Mechanism**: The frozen AdaptivePool module from the last layer of InternVideo2 is reused to pool spatiotemporal tokens from all intermediate layers: $\tilde{\mathbf{V}}^{\ell} = AdaptivePool(\hat{\mathbf{V}}^{\ell})$
    - **Design Motivation**: CLS pooling is unsuitable for intermediate layers of InternVideo2 as it constrains spatial aggregation capacity, whereas training separate AdaptivePool modules per layer requires full backpropagation, which is computationally infeasible. Reusing the frozen module leverages its enhanced multimodal alignment capability without incurring additional backpropagation or memory overhead.

### Loss & Training

The total loss comprises three terms: $\mathcal{L} = \lambda_5 \mathcal{L}_{HD} + \lambda_6 \mathcal{L}_{MR} + \lambda_7 \mathcal{L}_{align}$

- **HD Loss**: InfoNCE loss computed from cosine similarities between dense embeddings and pooled text representations.
- **MR Loss**: Following Hungarian matching, the loss includes FocalLoss (classification), L1 + IoU (boundary regression), and L1 (actionness score). Crucially, the loss is optimized across all refinement layers to facilitate convergence.
- **Alignment Loss**: SampledNCE-based visual-textual alignment applied along both the batch and intermediate-layer dimensions.

## Key Experimental Results

### Main Results

| Dataset | Metric | SDST (Ours) | SG-DETR | R2-Tuning | Parameters |
|--------|------|------------|---------|-----------|--------|
| QVHighlights val | R1@0.5 | 73.68 | — | 68.71 | 4.1M vs 15M vs 2.7M |
| QVHighlights val | R1@0.7 | 60.90 | — | 52.06 | — |
| QVHighlights val | mAP Avg | 55.60 | 55.64 | 47.59 | — |
| QVHighlights val | HD mAP | 44.00 | 43.91 | 40.59 | — |
| QVHighlights val | HD HIT@1 | 72.00 | 71.47 | 64.32 | — |
| Charades-STA | R@0.7 | **52.6** | 49.5 | 37.0 | — |
| Charades-STA | mIoU | **61.2** | 59.1 | 50.9 | — |
| TACoS | R@0.7 | **32.3** | 29.9 | 25.1 | — |
| TACoS | mIoU | **42.2** | 40.9 | 35.9 | — |

SDST achieves competitive or superior performance using only 27% of the parameters of SG-DETR.

### Ablation Study

| Configuration | MR mAP | HD HIT@1 | Notes |
|------|--------|----------|------|
| CLS pooling | 50.53 | 64.26 | Worst; restricts spatial aggregation |
| Average pooling | 53.44 | 69.68 | Partial improvement but insufficient |
| AdaptivePool (Ours) | **55.60** | **72.00** | Reusing frozen pooling module is optimal |
| Standard CA | 42.72 | 69.87 | Severe performance degradation |
| Def. CA | 54.27 | 70.58 | Baseline deformable CA |
| Def. CA + PureInit | 52.92 | 70.32 | Initialization is counterproductive |
| RDSA (Ours) | **55.60** | **72.00** | +1.33 mAP over Def. CA |

### Key Findings

- RDSA learns to attend to frames beyond the current action boundaries (offsets < −1 or > +1), which is critical for refining actions into longer temporal extents.
- The utility of intermediate features is not as straightforward as prior literature suggests; using only the last-layer features with multi-step refinement outperforms intermediate features at $K=2,3$.
- Sampling features from shallower layers introduces a depth-pooling trade-off: shallower features offer complementary information, but distribution shift degrades the effectiveness of the frozen AdaptivePool.

## Highlights & Insights

- This work is the first to combine side-tuning with an anchor-free DETR architecture, providing an elegant solution for sparse-dense multi-task learning.
- The analysis of RDSA offers a principled examination of the fundamental difference between deformable attention in CA vs. SA settings — in SA, the CNN offset predictor naturally has access to local context, whereas in CA it operates completely blind.
- This is the first successful integration of InternVideo2 into a side-tuning framework, resolving the non-trivial token pooling challenge.
- The in-depth empirical analysis of intermediate features vs. multi-step refinement provides a new perspective for understanding side-tuning dynamics.

## Limitations & Future Work

- The method relies on pre-extracted InternVideo2 features and does not support end-to-end training.
- The depth-pooling trade-off in AdaptivePool reuse limits the benefit of leveraging shallower-layer features.
- Evaluation is restricted to VTG-related benchmarks; generalization to broader video understanding tasks remains unexplored.
- The side-tuning paradigm is sensitive to backbone choice; transitioning from CLIP to InternVideo2 involves non-trivial engineering challenges.

## Related Work & Insights

- R2-Tuning, as the first side-tuning method for VTG, serves as the primary anchor-based baseline against which this work is compared.
- The RDSA formulation is generalizable to other tasks employing deformable cross-attention, such as Deformable DETR in object detection.
- The choice of feature pooling strategy is critical to side-tuning performance, a finding with broader implications for video understanding tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-stream side-tuning architecture and RDSA are meaningful contributions, though the overall framework builds upon existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three datasets, extensive ablation studies, and in-depth analysis of deformable attention behavior.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-motivated design choices, though the density of mathematical notation is relatively high.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for parameter-efficient video temporal grounding, achieving SOTA performance with a 73% reduction in trainable parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)

</div>

<!-- RELATED:END -->
