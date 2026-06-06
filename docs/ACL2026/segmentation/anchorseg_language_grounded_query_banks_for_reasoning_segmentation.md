---
title: >-
  [Paper Note] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation
description: >-
  [ACL 2026][Segmentation][Reasoning Segmentation] This paper proposes AnchorSeg, which reformulates reasoning segmentation as a structured conditional generation process based on language-grounded query banks. By explicit…
tags:
  - "ACL 2026"
  - "Segmentation"
  - "Reasoning Segmentation"
  - "Language Grounded Query Banks"
  - "Spatial Prior"
  - "Token-Mask Consistency"
  - "SAM"
date: 2026-05-08
content_hash: cc618c3192ce5e5f
---

# AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation

**Conference**: ACL 2026  
**arXiv**: [2604.18562](https://arxiv.org/abs/2604.18562)  
**Code**: [https://github.com/rui-qian/AnchorSeg](https://github.com/rui-qian/AnchorSeg)  
**Area**: Reasoning Segmentation / Multi-modal VLM  
**Keywords**: Reasoning Segmentation, Language Grounded Query Banks, Spatial Prior, Token-Mask Consistency, SAM

## TL;DR

This paper proposes AnchorSeg, which reformulates reasoning segmentation as a structured conditional generation process based on language-grounded query banks. By explicitly decoupling spatial localization and semantic reasoning via anchor queries combined with a Token-Mask Cycle Consistency training objective, the method achieves SOTA performance on ReasonSeg (67.7% gIoU, 68.1% cIoU).

## Background & Motivation

**Background**: Reasoning segmentation requires models to predict pixel-level masks based on complex, implicit textual queries (e.g., "the object providing shade in this scene"). Methods like LISA introduce a `<SEG>` token and feed its hidden state as a single query into a SAM decoder to predict the mask.

**Limitations of Prior Work**: Existing methods compress both semantic reasoning and spatial localization into the hidden representation of a single `<SEG>` token. This implicit compression limits the model's ability to explicitly distinguish "what to segment" (semantic reasoning) from "where to segment" (spatial localization), leading to restricted performance in complex reasoning scenarios.

**Key Challenge**: A single embedding must simultaneously encode two essentially different types of information: semantic understanding and spatial location. This creates a representation bottleneck—as reasoning complexity increases, it becomes harder for a single vector to carry both signals.

**Goal**: To redefine reasoning segmentation as a structured conditional generation problem, explicitly modeling spatial localization at the image token level and providing conditions through language-grounded queries.

**Key Insight**: Introduce multiple learnable tokens to form a "query bank," allowing different tokens to assume different roles—context queries handle semantic reasoning, while the anchor query handles spatial localization.

**Core Idea**: Replace the single SEG token with a language-grounded query bank to explicitly decouple spatial localization (anchor query) and semantic modulation (context queries) through a factorized conditional distribution.

## Method

### Overall Architecture

Given an input image and a text query, an LMM (e.g., LLaVA) autoregressively generates $K$ latent reasoning tokens and one segmentation anchor token `<SEG>`, forming the query bank $\mathbf{Q} = (\boldsymbol{q}_1, ..., \boldsymbol{q}_K, \boldsymbol{q}_{anc})$. The similarity between the anchor query and image tokens is calculated to generate a spatial prior. After injecting this into the visual features, the entire query bank is fed into the SAM decoder to predict the final mask.

### Key Designs

1. **Query Bank Construction**:

    - **Function**: Constructs a structured sequence of conditional queries to provide separated representations for subsequent spatial localization and semantic reasoning.
    - **Mechanism**: The LMM vocabulary is expanded to include $K$ latent reasoning tokens `<LAT_1>,...,<LAT_K>` and one segmentation token `<SEG>`. During autoregressive generation, `<SEG>` is explicitly conditioned on the preceding reasoning tokens. Context queries $\boldsymbol{q}_{1:K}$ encode intermediate reasoning states, while the anchor query $\boldsymbol{q}_{anc}$ serves as the spatial localization signal.
    - **Design Motivation**: By distributing the two signals across different tokens rather than compressing them into one, the model forms an internal ordered process: "reasoning first, localization second."

2. **Language Grounded Conditioning**:

    - **Function**: Converts the anchor query into an explicit spatial localization prior and injects it into visual features.
    - **Mechanism**: Spatial localization is modeled as a factorized conditional distribution over image tokens: $p(\boldsymbol{S}|\mathbf{Q}) = \prod_i p(s_i | \boldsymbol{i}_i, \boldsymbol{q}_{1:K}, \boldsymbol{q}_{anc})$. In practice, the spatial response $s_i = \boldsymbol{i}_i^\top \boldsymbol{q}_{anc}$ is calculated via the dot product of the anchor query and image tokens. This is reshaped into a spatial prior $\mathbf{P}$ and injected into visual features $\mathbf{f}$ via element-wise addition: $\tilde{\mathbf{f}} = \mathbf{f} \oplus \mathbf{P}$.
    - **Design Motivation**: The anchor query directly generates the localization signal, while context queries implicitly influence the anchor query generation through the autoregressive process, achieving semantic modulation of spatial information.

3. **Token-Mask Cycle Consistency (TMCC)**:

    - **Function**: Bridges the resolution gap between token-level spatial responses and pixel-level mask supervision.
    - **Mechanism**: Bidirectional constraints—(a) Token-to-Mask: Token-level responses are upsampled to image resolution and aligned with Gaussian-smoothed ground truth masks using BCE+Dice loss. (b) Mask-to-Token: Ground truth masks are downsampled to token resolution and aligned with token-level responses. This ensuring spatial reasoning remains consistent across language-vision hierarchies.
    - **Design Motivation**: Since token-level spatial responses and pixel-level masks operate at different resolutions, bidirectional consistency constraints are needed to prevent contradictions between the two levels.

### Loss & Training

The total loss consists of three parts: the autoregressive text generation loss $\mathcal{L}_{txt}$, the SAM mask prediction loss $\mathcal{L}_{mask}$ (BCE+Dice), and the TMCC loss $\mathcal{L}_{T2M} + \mathcal{L}_{M2T}$. The BCE and Dice weights for TMCC are shared with the mask loss.

## Key Experimental Results

### Main Results

Performance on the ReasonSeg test set:

| Method | gIoU | cIoU |
|------|------|------|
| LISA-7B | 54.3 | 58.1 |
| GSVA-7B | 55.6 | 59.4 |
| READ-7B | 57.2 | 60.5 |
| RSVP-7B | 63.7 | 64.8 |
| **AnchorSeg-7B (Ours)** | **67.7** | **68.1** |

### Ablation Study

| Configuration | gIoU | Description |
|------|------|------|
| Single SEG token (baseline) | 54.3 | Original LISA design |
| + Query Bank (No spatial prior) | ~62 | Multi-token reasoning helps |
| + Spatial Prior Injection | ~65 | Explicit localization signals yield large gains |
| + TMCC | 67.7 | Bidirectional consistency further improves results |

### Key Findings

- The improvement from a single SEG token to a query bank is most significant, indicating that the multi-token reasoning structure is the core contribution.
- Explicit injection of the spatial prior (rather than just using it as a query) provides substantial additional gains, verifying the necessity of the decoupling design.
- While the gain from TMCC's bidirectional consistency is smaller, it effectively prevents training instability.
- The method also shows competitiveness on RefCOCO/+/g, demonstrating strong generalization.

## Highlights & Insights

- The modeling via factorized conditional distribution is elegant: treating spatial localization explicitly as "relevance of each image token" is mathematically clear and physically intuitive. This token-level spatial reasoning can be transferred to other multi-modal tasks requiring precise localization.
- The division of roles within the query bank (context queries vs. anchor query) mimics human cognition: understanding the problem's semantics first, then performing spatial localization, and finally fine-grained segmentation.
- The cross-resolution consistency constraint of TMCC is a simple yet effective regularization technique applicable to any scenario involving alignment of representations across different resolutions.

## Limitations & Future Work

- The value of $K$ (number of latent reasoning tokens) in the query bank is a hyperparameter; queries of varying complexity may require different numbers of reasoning tokens.
- The spatial prior is only calculated through a simple inner product, which might not be powerful enough for complex spatial reasoning (e.g., occlusion relationships).
- Currently only evaluated on reasoning and referring segmentation; generalization to other tasks like VQA has not been explored.
- The method relies on SAM as the mask decoder, thus inheriting SAM's own limitations.

## Related Work & Insights

- **vs LISA**: Uses a single SEG token where semantic and spatial information are compressed together; AnchorSeg explicitly decouples them via a query bank, improving gIoU by 13.4 points.
- **vs GSVA**: Extends to multi-target reasoning and non-existent object rejection but remains based on the single-token paradigm; AnchorSeg fundamentally changes the representation structure.
- **vs RSVP**: Introduces multi-modal CoT reasoning, but the reasoning process is coupled with the segmentation module; AnchorSeg's factorized design is more modular and interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The design of the query bank and factorized spatial conditioning is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on ReasonSeg and RefCOCO.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization, though notation is somewhat heavy.
- Value: ⭐⭐⭐⭐ Provides a more structured paradigm for solving reasoning segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[ICCV 2025\] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation](../../ICCV2025/segmentation/veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)
- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](../../NeurIPS2025/segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[CVPR 2026\] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation](../../CVPR2026/segmentation/pixdlm_uav_reasoning_segmentation.md)
- [\[CVPR 2026\] VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation](../../CVPR2026/segmentation/virst_video-instructed_reasoning_assistant_for_spatiotemporal_segmentation.md)

</div>

<!-- RELATED:END -->
