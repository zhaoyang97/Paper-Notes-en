---
title: >-
  [Paper Note] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation
description: >-
  [ACL 2026][Segmentation][Reasoning Segmentation] This paper proposes AnchorSeg, which reformulates reasoning segmentation as a structured conditional generation process based on language-grounded query banks. By explicitly decoupling spatial localization and semantic reasoning through anchor queries, paired with a Token-Mask Cyclic Consistency training objective, it achieves SOTA on ReasonSeg (67.7% gIoU, 68.1% cIoU).
tags:
  - ACL 2026
  - Segmentation
  - Reasoning Segmentation
  - Language-Grounded Query Banks
  - Spatial Prior
  - Token-Mask Consistency
  - SAM
date: 2025-04-17
content_hash: b8da03f9e2f8e920
---

# AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation

**Conference**: ACL 2026  
**arXiv**: [2604.18562](https://arxiv.org/abs/2604.18562)  
**Code**: [https://github.com/rui-qian/AnchorSeg](https://github.com/rui-qian/AnchorSeg)  
**Area**: Reasoning Segmentation / Multimodal VLM  
**Keywords**: Reasoning Segmentation, Language-Grounded Query Banks, Spatial Prior, Token-Mask Consistency, SAM

## TL;DR

This paper proposes AnchorSeg, which reformulates reasoning segmentation as a structured conditional generation process based on language-grounded query banks. By explicitly decoupling spatial localization and semantic reasoning through anchor queries, paired with a Token-Mask Cyclic Consistency training objective, it achieves SOTA on ReasonSeg (67.7% gIoU, 68.1% cIoU).

## Background & Motivation

**Background**: Reasoning segmentation requires models to predict pixel-level masks from complex, implicit text queries (e.g., "the object that provides shade in this scene"). Methods like LISA introduce a `<SEG>` token, using its hidden state as a single query fed into the SAM decoder to predict masks.

**Limitations of Prior Work**: Existing methods compress both semantic reasoning and spatial localization into a single `<SEG>` token's hidden representation. This implicit compression limits the model's ability to explicitly distinguish "what to segment" (semantic reasoning) from "where to segment" (spatial localization), constraining performance in complex reasoning scenarios.

**Key Challenge**: A single embedding must simultaneously encode semantic understanding and spatial position — two fundamentally different types of information — creating a representation bottleneck. The more complex the reasoning, the harder it is for a single vector to carry both signals.

**Goal**: Redefine reasoning segmentation as a structured conditional generation problem, explicitly modeling spatial localization at the image token level with language-guided queries providing the condition.

**Key Insight**: Introduce multiple learnable tokens forming a "query bank," with different tokens assuming different roles — context queries handle semantic reasoning while anchor queries handle spatial localization.

**Core Idea**: Replace the single SEG token with a language-guided query bank, explicitly decoupling spatial localization (anchor queries) and semantic modulation (context queries) through a factorized conditional distribution.

## Method

### Overall Architecture

Given input image and text query, the LMM (e.g., LLaVA) autoregressively generates K latent reasoning tokens and 1 segmentation anchor token `<SEG>`, forming the query bank $\mathbf{Q} = (\boldsymbol{q}_1, ..., \boldsymbol{q}_K, \boldsymbol{q}_{anc})$. The anchor query computes similarity with image tokens to produce spatial priors, which are injected into visual features before the entire query bank is fed into the SAM decoder to predict the final mask.

### Key Designs

1. **Language-Grounded Query Bank Construction**:

    - Function: Construct a structured conditional query sequence providing separated representations for subsequent spatial localization and semantic reasoning
    - Mechanism: Extend the LMM vocabulary with K latent reasoning tokens `<LAT_1>,...,<LAT_K>` and one segmentation token `<SEG>`. During autoregressive generation, `<SEG>` is explicitly conditioned on preceding reasoning tokens. Context queries $\boldsymbol{q}_{1:K}$ encode intermediate reasoning states; the anchor query $\boldsymbol{q}_{anc}$ serves as the spatial localization signal
    - Design Motivation: Distributing two types of signals originally compressed into a single token across different tokens allows the model to internally form an ordered "reason first, then locate" process

2. **Language-Grounded Spatial Conditioning**:

    - Function: Transform the anchor query into an explicit spatial localization prior and inject it into visual features
    - Mechanism: Model spatial localization as a factorized conditional distribution over image tokens $p(\boldsymbol{S}|\mathbf{Q}) = \prod_i p(s_i | \boldsymbol{i}_i, \boldsymbol{q}_{1:K}, \boldsymbol{q}_{anc})$. In practice, compute spatial response through inner product of anchor query and image tokens $s_i = \boldsymbol{i}_i^\top \boldsymbol{q}_{anc}$, reshape to obtain spatial prior $\mathbf{P}$, and inject into visual features via element-wise addition $\tilde{\mathbf{f}} = \mathbf{f} \oplus \mathbf{P}$
    - Design Motivation: The anchor query directly produces localization signals while context queries implicitly influence anchor query generation through the autoregressive process, achieving semantic modulation of spatial information

3. **Token-Mask Cyclic Consistency (TMCC)**:

    - Function: Bridge the resolution gap between token-level spatial responses and pixel-level mask supervision
    - Mechanism: Bidirectional constraint — (a) Token-to-Mask: upsample token-level response to image resolution, align with Gaussian-smoothed GT mask using BCE+Dice loss; (b) Mask-to-Token: downsample GT mask to token resolution, align with token-level response. Ensures spatial reasoning consistency across language-vision hierarchies
    - Design Motivation: Token-level spatial responses and pixel-level masks operate at different resolutions, requiring bidirectional consistency constraints to prevent contradictions between the two levels

### Loss & Training

Total loss comprises three parts: autoregressive text generation loss $\mathcal{L}_{txt}$, SAM mask prediction loss $\mathcal{L}_{mask}$ (BCE+Dice), and TMCC losses $\mathcal{L}_{T2M} + \mathcal{L}_{M2T}$. TMCC BCE and Dice weights are shared with the mask loss.

## Key Experimental Results

### Main Results

Performance on ReasonSeg test set:

| Method | gIoU | cIoU |
|------|------|------|
| LISA-7B | 54.3 | 58.1 |
| GSVA-7B | 55.6 | 59.4 |
| READ-7B | 57.2 | 60.5 |
| RSVP-7B | 63.7 | 64.8 |
| **AnchorSeg-7B** | **67.7** | **68.1** |

### Ablation Study

| Config | gIoU | Note |
|------|------|------|
| Single SEG token (baseline) | 54.3 | Original LISA design |
| + Query bank (no spatial prior) | ~62 | Multi-token reasoning helps |
| + Spatial prior injection | ~65 | Explicit localization signal significant gain |
| + TMCC | 67.7 | Bidirectional consistency further improves |

### Key Findings

- The improvement from single SEG token to query bank is the most significant, indicating multi-token reasoning structure is the core contribution
- Explicit spatial prior injection (not just as query) brings clear additional gains, validating the necessity of the decoupled design
- TMCC bidirectional consistency constraint, while modest in improvement magnitude, effectively prevents training instability
- Competitive performance on RefCOCO/+/g as well, indicating good method generalizability

## Highlights & Insights

- The factorized conditional distribution modeling is highly elegant: explicitly modeling spatial localization as "relevance of each image token," with clear mathematical expression and physical meaning. This token-level spatial reasoning is transferable to other multimodal tasks requiring precise localization
- The role division within the query bank (context queries vs anchor queries) resembles human cognitive processes: first understand question semantics, then perform spatial localization, and finally fine-grained segmentation
- TMCC cross-resolution consistency constraint is a concise yet effective regularization technique applicable to any scenario involving alignment of representations at different resolutions

## Limitations & Future Work

- The K value (number of latent reasoning tokens) in the query bank is a hyperparameter; queries of different complexity may require different numbers of reasoning tokens
- Spatial priors are computed through simple inner products, which may be insufficient for complex spatial reasoning (e.g., occlusion relationships)
- Currently evaluated only on reasoning segmentation and referring segmentation; generalization to visual QA and other tasks is unexplored
- Method depends on SAM as the mask decoder, limited by SAM's own capabilities

## Related Work & Insights

- **vs LISA**: Uses a single SEG token where semantic and spatial information are compressed together; AnchorSeg explicitly decouples through query banks, achieving a 13.4-point gIoU improvement
- **vs GSVA**: Extends to multi-object reasoning and non-existent object rejection but remains based on the single-token paradigm; AnchorSeg fundamentally changes the representation structure
- **vs RSVP**: Introduces multimodal CoT reasoning but the reasoning process is coupled with the segmentation module; AnchorSeg's factorized design is more modular and interpretable

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Query bank + factorized spatial conditioning design is highly novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on ReasonSeg and RefCOCO
- Writing Quality: ⭐⭐⭐⭐ Formalization is clear, though notation is somewhat heavy
- Value: ⭐⭐⭐⭐ Provides a more structured solution paradigm for reasoning segmentation

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[ICCV 2025\] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation](../../ICCV2025/segmentation/veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)
- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](../../NeurIPS2025/segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[CVPR 2026\] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation](../../CVPR2026/segmentation/pixdlm_uav_reasoning_segmentation.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)

<!-- RELATED:END -->
