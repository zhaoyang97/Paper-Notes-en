---
title: >-
  [Paper Note] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding
description: >-
  [NeurIPS 2025][Segmentation][part-level understanding] This paper introduces the Partonomy part-level segmentation benchmark (862 part labels / 534 object labels) and the Plum model…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "part-level understanding"
  - "LMM"
  - "interpretable part segmentation"
  - "span tagging"
  - "mask feedback"
  - "benchmark"
date: 2026-05-08
content_hash: 23f14dad1a14c7ba
---

# PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2505.20759](https://arxiv.org/abs/2505.20759)
**Code**: [GitHub](https://github.com/AnselBlume/partonomy)
**Area**: Segmentation / Multimodal
**Keywords**: part-level understanding, LMM, interpretable part segmentation, span tagging, mask feedback, benchmark

## TL;DR
This paper introduces the Partonomy part-level segmentation benchmark (862 part labels / 534 object labels) and the Plum model, which replaces the [SEG] token with BIO span tagging and incorporates a mask feedback loop. The study reveals that state-of-the-art segmentation LMMs achieve only 5.9% gIoU on part understanding; Plum achieves significant improvements by avoiding distribution shift and leveraging historical predictions.

## Background & Motivation
**Background**: Segmentation-oriented LMMs (e.g., LISA, GLaMM, PixelLM) can generate segmentation masks from textual instructions and perform well on referring expression comprehension and reasoning segmentation tasks.

**Limitations of Prior Work**: Despite being trained on part-level datasets such as PACO and Pascal-Part, these LMMs are nearly incapable of performing part-level segmentation (LISA-13B achieves only 5.9% gIoU). Two architectural deficiencies are identified: (a) the use of special [SEG] tokens unseen during pretraining introduces distribution shift; (b) when generating multiple masks sequentially, prior predictions are discarded, preventing the model from exploiting historical information.

**Key Challenge**: Part understanding requires fine-grained compositional reasoning (e.g., which parts constitute an object? which parts are shared between two objects?), yet existing LMMs and evaluation benchmarks lack both the capability and design to support such reasoning.

**Goal**: (1) Construct a comprehensive part-level LMM evaluation benchmark; (2) Design a new model that addresses the two aforementioned architectural deficiencies.

**Key Insight**: The [SEG] token is introduced after pretraining and inevitably causes distribution shift. Using existing text spans to indicate segmentation regions preserves the pretrained representations.

**Core Idea**: Replace the [SEG] token with BIO span tagging to avoid distribution shift, and add a mask feedback loop to allow subsequent predictions to leverage historical mask information.

## Method

### Overall Architecture
Plum = LLaVA (vision-language model) + bidirectional Span Extractor (automatically tagging which text tokens correspond to parts requiring segmentation) + SAM mask decoder (with FiLM-based mask feedback). The pipeline proceeds as follows: image + part query → LMM generates textual response → Span Extractor tags part-name spans → spans are projected into mask queries → enhanced SAM decoder generates segmentation masks (conditioned on historical masks as feedback).

### Key Designs

1. **Span Extractor (replacing the [SEG] token)**:

    - **Function**: Performs BIO tagging on the token embeddings output by the LMM to automatically identify which text spans are part names requiring segmentation.
    - **Mechanism**: Token embeddings are first passed through an MLP, then through a bidirectional Transformer encoder to capture global context (since the causal attention mask in LLMs prevents access to future tokens), and finally projected to B/I/O labels.
    - **Design Motivation**: The [SEG] token is never seen during pretraining; its embedding is randomly initialized, and fine-tuning it perturbs the existing token distribution. Span tagging operates entirely within the existing vocabulary, introducing no new tokens.
    - **Effect**: Preserves the LMM's text reasoning capability, outperforming LISA and GLaMM on both zero-shot segmentation and VQA/hallucination benchmarks.

2. **KL-Constrained Query Projection**:

    - **Function**: When projecting B/I-tagged token embeddings into mask queries, KL divergence is used to constrain the projection from deviating from the frozen teacher embeddings pretrained representations.
    - **Mechanism**: $\mathcal{L}_{\text{KL}} = \frac{1}{N_+}\sum_{s} \frac{\|h^L_{i_s:j_s} - t^L_{i_s:j_s}\|_2^2}{2\sigma^2}$
    - **Design Motivation**: Prevents the hidden states from drifting away from the original language representation space during segmentation fine-tuning, thereby protecting text reasoning capability.

3. **Mask Feedback Loop**:

    - **Function**: Previously predicted masks are injected into SAM's mask encoder via FiLM layers, enabling subsequent mask predictions to be conditioned on historical predictions.
    - **Mechanism**: Each historical mask is encoded into a text-conditioned feature map using a modified mask encoder; all historical feature maps are then aggregated into a single feature map via patch-wise attention pooling and passed to the mask decoder.
    - **Design Motivation**: Parts exhibit spatial relationships (e.g., bilateral symmetry of wings); having observed the segmentation of the left wing aids in more accurately segmenting the right wing.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_1 \mathcal{L}_{\text{span}} + \lambda_2 \mathcal{L}_{\text{KL}} + \lambda_3 \mathcal{L}_{\text{seg}} + \lambda_4 \mathcal{L}_{\text{BCE}}$$
The segmentation loss employs Focal-Tversky loss with recall-biased weighting ($\alpha=0.7$, $\beta=0.3$). Training follows a two-stage fine-tuning strategy: first on mixed segmentation data, then on the Partonomy training set.

## Key Experimental Results

### Main Results (Partonomy-Core Segmentation gIoU, %)

| Method | Extra Seg. Data | Part Identification (macro) | Part Intersection (macro) | Part Difference (macro) |
|------|------------|---------------------------|--------------------------|------------------------|
| LISA-13B | ✗ | 7.0 | 7.5 | 7.1 |
| PixelLM-13B | ✓ | 8.4 | 4.8 | 4.8 |
| GLaMM | ✓ | 5.9 | 6.2 | 6.0 |
| **Plum-13B (zero-shot)** | ✗ | **27.4** | **29.9** | **24.8** |
| LISA-13B (ft) | ✗ | 35.4 | 38.4 | 31.6 |
| GLaMM (ft) | ✓ | 38.8 | 42.1 | 34.8 |
| **Plum-13B (ft)** | ✗ | **41.6** | **45.9** | **39.4** |
| Grounded SAM 2 (gt) | – | 16.8 | 23.6 | 17.1 |

### Ablation Study

| Configuration | Key Findings | Notes |
|------|---------|------|
| Span Extractor vs. [SEG] token | Span tagging preserves reasoning capability on VQA/hallucination benchmarks | Confirms distribution shift caused by [SEG] token |
| With vs. without mask feedback | Feedback improves Part-Whole reasoning segmentation | Historical masks provide spatial priors |
| With vs. without KL constraint | KL constraint protects text reasoning quality | Prevents hidden state drift |
| Zero-shot Plum vs. zero-shot LISA | Plum zero-shot (27.4) > LISA zero-shot (7.0) | 4× improvement |

### Key Findings
- State-of-the-art segmentation LMMs collapse comprehensively on part-level understanding, demonstrating that part understanding represents a critical blind spot in current LMMs.
- Distribution shift introduced by the [SEG] token is one of the primary causes of poor performance — Plum's span tagging scheme yields a 4× improvement in zero-shot performance.
- Partonomy-Core's 862 part labels constitute the largest part annotation set to date (4× PACO).
- Even GPT-4o achieves limited accuracy on questions requiring fine-grained part comparison.

## Highlights & Insights
- **Revealing the Part Understanding Bottleneck in LMMs**: The 5.9% gIoU figure is striking, demonstrating that the claim that "segmentation LMMs can understand images" does not hold at the part level. This poses new challenges for LMM evaluation methodology.
- **Generalizability of Span Tagging**: Using existing text spans rather than special tokens to indicate segmentation targets is a more principled approach to text–vision alignment, avoiding vocabulary mismatches between pretraining and fine-tuning stages. This paradigm is generalizable to any task requiring association between text and visual regions.
- **Part Comparison and Reasoning Tasks**: Partonomy goes beyond part identification, requiring models to compare parts across objects (shared/differing parts) and reason about part-whole relationships. This compositional reasoning capability is essential for fine-grained visual understanding.

## Limitations & Future Work
- After fine-tuning, Plum achieves only approximately 41% macro-gIoU on Partonomy-Core, leaving substantial room for improvement before practical deployment.
- Partonomy-Core contains only ~1K images, and part annotations may be incomplete, as some visible parts may not have been labeled.
- The model architecture relies on SAM's mask decoder, which may have insufficient resolution for extremely fine-grained parts (e.g., components on a circuit board).
- Integration with more recent segmentation models such as SAM 2 remains unexplored.

## Related Work & Insights
- **vs. LISA (Lai et al., 2024)**: LISA uses a [SEG] token to trigger segmentation; this paper demonstrates that such a design causes severe distribution shift. Plum's span tagging scheme provides a superior alternative.
- **vs. GLaMM (Rasheed et al., 2024)**: GLaMM is trained on more data yet performs poorly on parts, indicating that data scale cannot compensate for architectural deficiencies.
- **vs. SegLLM (based on HIPIE)**: SegLLM incorporates a similar mask feedback mechanism and achieves the best part segmentation performance among prior methods (32.4 macro-gIoU), validating the importance of mask feedback.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First part-level LMM benchmark + a new paradigm of span tagging as a replacement for the [SEG] token
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-baseline comparisons, comprehensive ablations, dual evaluation on text and segmentation
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem formulation with well-supported arguments
- **Value**: ⭐⭐⭐⭐⭐ Exposes an important blind spot in LMMs with dual contributions in benchmark and methodology

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](../../ICCV2025/segmentation/himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[CVPR 2026\] HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding](../../CVPR2026/segmentation/hippomm_hippocampal-inspired_multimodal_memory_for_long_audiovisual_event_unders.md)
- [\[ICCV 2025\] Advancing Visual Large Language Model for Multi-granular Versatile Perception](../../ICCV2025/segmentation/advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)

</div>

<!-- RELATED:END -->
