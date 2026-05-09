---
title: >-
  [Paper Note] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling
description: >-
  [CVPR 2026][Video Understanding][Video Temporal Grounding] HieraMamba proposes a Mamba-based hierarchical architecture for video temporal grounding. Its core contribution is the Anchor-MambaPooling (AMP) module, which employs Mamba's selective scanning to progressively compress video features into multi-scale anchor tokens. Complementary anchor-conditioned and segment-pooled contrastive losses enhance the compactness and discriminability of hierarchical representations, achieving state-of-the-art performance on Ego4D-NLQ, MAD, and TACoS.
tags:
  - CVPR 2026
  - Video Understanding
  - Video Temporal Grounding
  - State Space Models
  - Mamba
  - Hierarchical Representation
  - Contrastive Learning
date: 2026-05-08
content_hash: 1daeb003e82cc5a8
---

# HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling

**Conference**: CVPR 2026
**arXiv**: [2510.23043](https://arxiv.org/abs/2510.23043)
**Code**: [https://vision.cs.utexas.edu/projects/hieramamba](https://vision.cs.utexas.edu/projects/hieramamba)
**Area**: Video Understanding
**Keywords**: Video Temporal Grounding, State Space Models, Mamba, Hierarchical Representation, Contrastive Learning

## TL;DR

HieraMamba proposes a Mamba-based hierarchical architecture for video temporal grounding. Its core contribution is the Anchor-MambaPooling (AMP) module, which employs Mamba's selective scanning to progressively compress video features into multi-scale anchor tokens. Complementary anchor-conditioned and segment-pooled contrastive losses enhance the compactness and discriminability of hierarchical representations, achieving state-of-the-art performance on Ego4D-NLQ, MAD, and TACoS.

## Background & Motivation

1. **Background**: Video Temporal Grounding (VTG) requires localizing start and end timestamps in untrimmed videos given natural language queries. The task has evolved from predefined action localization to free-text queries, supporting applications such as VQA and automatic video editing. Recent methods including ActionFormer, SnAG, and DeCafNet have introduced multi-scale feature pyramids.

2. **Limitations of Prior Work**: Long videos (ranging from minutes to hours) present two intertwined challenges: (a) **Temporal fidelity** — many methods reduce computational cost via fixed-length pooling, naive downsampling, or fixed windows, all of which discard critical temporal cues or sever temporal structure at window boundaries; (b) **Multi-granularity** — different queries require different temporal granularities ("what did the detective do in the library?" requires coarse understanding, whereas "when did the detective pull the note from the shelf?" requires fine-grained localization), which single-resolution methods fail to address simultaneously.

3. **Key Challenge**: The quadratic attention cost of Transformers is the root cause driving downsampling and windowing heuristics — processing long sequences necessitates sacrificing temporal resolution. Although recent multi-scale models (SnAG, DeCafNet, OSGNet) incorporate multi-scale representations, these are still produced via uniform downsampling or coarse pooling, lacking content-aware compression.

4. **Goal**: (1) How to process full-length video sequences in linear time, avoiding downsampling or windowing? (2) How to construct content-aware multi-scale hierarchical representations rather than simple downsampled pyramids? (3) How to ensure that hierarchical anchors are both compact (faithfully summarizing local information) and discriminative (distinguishable from other events)?

5. **Key Insight**: Human episodic memory is inherently hierarchical — from the overall layout of a room to the precise movement of fingers, humans seamlessly switch between temporal scales. The authors identify quadratic Transformer attention as the root cause of temporal downsampling and propose replacing it with Mamba's linear-time selective scanning to enable full-resolution long-range modeling.

6. **Core Idea**: Leverage Mamba's selective scanning to build content-aware hierarchical anchor compression (rather than naive downsampling). Stacking AMP modules forms a fine-to-coarse multi-scale temporal pyramid that achieves accurate long-video temporal grounding at linear complexity.

## Method

### Overall Architecture

**Input**: Clip-level features $V \in \mathbb{R}^{L_V \times D_v}$ extracted by a frozen video backbone (e.g., EgoVLP) and query embeddings $Q \in \mathbb{R}^{L_Q \times D_q}$ from a frozen text encoder (e.g., CLIP). The video encoder consists of $L$ hierarchically stacked AMP modules, each producing refined features $\tilde{V}^{(l)}$ and anchors $A^{(l+1)}$ for the next layer, forming a feature pyramid $\mathcal{V}_{\text{pyr}} = \{\tilde{V}^{(0)}, \ldots, \tilde{V}^{(L-1)}\}$. The pyramid is fused with text embeddings via cross-modal attention, and a lightweight decoder regresses the temporal boundaries $(t_s, t_e)$.

### Key Designs

1. **Anchor-MambaPooling (AMP) Module**:

   - **Function**: Simultaneously performs feature refinement at the current resolution and content-aware compression to the next level.
   - **Mechanism**: A three-step pipeline — (a) **Anchor generation and interleaving**: An anchor token is initialized every $s$ frames (via local window pooling) and inserted before the frames it summarizes, forming an interleaved sequence $\hat{V} = [a_0, v_0, \ldots, v_{s-1}, a_1, v_s, \ldots] \in \mathbb{R}^{(L_0+L_1) \times D_v}$; (b) **Global encoding**: Hydra (bidirectional Mamba scanning) processes the interleaved sequence, enabling linear-complexity global context modeling — the forward scan allows anchors to receive information from preceding frames, and the backward scan from succeeding frames; (c) **Local encoding**: A narrow-window Transformer (window size 5) supplements short-range fine-grained attention patterns. The final outputs are refined current-layer features $\tilde{V}^{(l)}$ and compressed next-layer anchors $A^{(l+1)}$.
   - **Design Motivation**: The interleaving design allows anchors and frame features to share a single Mamba scan. Anchors broadcast coarse-grained context to neighboring frames, while frame features provide fine-grained details to refine anchors — a bidirectional information flow. The key distinction from conventional feature pyramids is that AMP produces multi-scale representations through token-level compression rather than naive downsampling, enabling content-aware abstraction.

2. **Gated Fusion and Decoupling**:

   - **Function**: Controls the quality of information propagation across hierarchy levels.
   - **Mechanism**: RMS normalization and residual connections are applied between global encoding, local encoding, and FFN. Between stages, learnable sigmoid gates $\boldsymbol{\sigma}$ replace unconditional residual addition, providing content-adaptive control over information propagation.
   - **Design Motivation**: Mamba captures global structure while the narrow-window Transformer captures local patterns — the roles of the two components are explicitly decoupled, avoiding the role ambiguity common in hybrid architectures. Gating ensures that only salient information propagates up the hierarchy.

3. **Anchor-Conditioned Contrastive (ACC) Loss**:

   - **Function**: A self-supervised objective ensuring anchors are compact and discriminative.
   - **Mechanism**: At each layer, each anchor $a_i^{(l+1)}$ is pulled toward the $s$ frame tokens it summarizes (positives $\mathcal{P}_i^{(l)}$) and pushed away from temporally distant anchors (negatives $\mathcal{N}_i^{(l)}$, separated by a temporal margin to avoid penalizing adjacent anchors):
     $$\mathcal{L}_{\text{acc}}(a_i^{(l+1)}) = -\log \frac{\sum_{p \in \mathcal{P}_i^{(l)}} \exp(a_i^{(l+1)} \cdot p / \tau)}{\sum_{c \in \mathcal{P}_i^{(l)} \cup \mathcal{N}_i^{(l)}} \exp(a_i^{(l+1)} \cdot c / \tau)}$$
   - **Design Motivation**: Compactness requires anchors to faithfully summarize their local window (alignment with intra-window frames), while discriminability requires different anchors to represent different events (separation from distant anchors). The multi-positive design avoids the information loss that can arise from single-positive contrastive objectives.

4. **Segment-Pooled Contrastive (SPC) Loss**:

   - **Function**: A supervised objective that distinguishes representations of ground-truth segments from surrounding non-target content.
   - **Mechanism**: At each layer, frame tokens within the GT segment $[t_{\text{start}}, t_{\text{end}})$ are pooled into a segment prototype $z_{\text{seg}}^{(l)}$, with intra-segment frames as positives and extra-segment frames as negatives. Using the pooled prototype rather than individual frames avoids forcing distinct sub-actions within a segment (e.g., "reach → grasp → retract") to align to a single representation.
   - **Design Motivation**: ACC provides structural consistency (intra-hierarchy self-supervision), while SPC provides semantic alignment (alignment with query annotations). The two losses are complementary: ACC ensures anchor quality, and SPC ensures anchors are semantically matched to the query.

### Loss & Training

The total contrastive loss is $\mathcal{L}_{\text{contrast}} = \lambda_{\text{ACC}} \mathcal{L}_{\text{ACC}} + \lambda_{\text{SPC}} \mathcal{L}_{\text{SPC}}$, jointly optimized with the standard temporal grounding task loss (boundary regression + classification).

## Key Experimental Results

### Main Results

Results on Ego4D-NLQ (with EgoVLP features):

| Method | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5 | Avg. |
|---|---|---|---|---|---|
| SnAG | 15.72 | 10.78 | 38.39 | 27.44 | 23.08 |
| DeCafNet | 18.10 | 12.55 | 38.85 | 28.27 | 24.44 |
| RGNet | 18.28 | 12.04 | 34.02 | 22.89 | 21.81 |
| OSGNet | 16.13 | 11.28 | 36.78 | 25.63 | 22.46 |
| **HieraMamba** | **18.81** | **13.04** | **40.82** | **29.96** | **25.66** |

State-of-the-art results are also reported on MAD and TACoS (detailed numbers in the paper).

### Method Characteristics Comparison

| Method | Naive Downsampling | Fixed Pooling | Quadratic Cost | Sliding Window | Ego4D Avg.R |
|---|---|---|---|---|---|
| 2D-TAN | ✓ | ✓ | ✓ | — | 6.46 |
| CONE | — | — | ✓ | ✓ | 17.67 |
| SnAG | ✓ | — | — | — | 23.08 |
| DeCafNet | ✓ | — | — | — | 24.44 |
| **HieraMamba** | **—** | **—** | **—** | **—** | **25.66** |

HieraMamba is the only method that simultaneously avoids all four undesirable properties.

### Key Findings

- The benefits of avoiding downsampling and windowing are especially pronounced on long videos — HieraMamba achieves the largest gains on Ego4D (8-minute average) and MAD (feature-length movies).
- ACC and SPC losses contribute complementarily — ACC primarily improves intra-hierarchy anchor quality and consistency, while SPC primarily improves semantic alignment with queries (ablation studies in the appendix).
- The global–local decoupling of Mamba and narrow-window Transformer outperforms either component alone.
- The gating mechanism (sigmoid gate) outperforms unconditional residual connections, demonstrating the importance of content-adaptive information propagation in hierarchical models.

## Highlights & Insights

- **The interleaving design of AMP is particularly elegant**: Inserting anchor tokens into the frame sequence for joint Mamba scanning allows anchors to naturally acquire global context summarization capability (via Mamba's state compression) while enabling frame features to obtain neighborhood summaries from anchors — achieving bidirectional information flow in a single scan at linear cost.
- **A methodology of eliminating all undesirable properties**: By systematically analyzing four limitations of prior methods (downsampling, fixed pooling, quadratic cost, sliding windows), the authors design an architecture that simultaneously avoids all of them, reflecting a principled engineering approach.
- **Multi-positive design in the ACC loss**: Conventional contrastive learning uses a single positive, but in temporal grounding an anchor must faithfully represent the content of multiple frames. The multi-positive InfoNCE formulation naturally accommodates this requirement.

## Limitations & Future Work

- The method relies on frozen video backbones (EgoVLP/InternVideo); insufficient clip feature quality from the backbone cannot be compensated by the subsequent hierarchical modeling.
- The stride $s$ in AMP is a fixed hyperparameter; different queries may benefit from different strides — adaptive stride selection warrants exploration.
- The unidirectional causal structure of Mamba requires compensation through bidirectional Hydra, which increases complexity — more native bidirectional SSM designs could be investigated.
- Inference speed is not discussed; while the theoretical complexity is linear, the practical latency of AMP interleaving, bidirectional scanning, and multi-layer stacking requires empirical validation.
- The sensitivity analysis of the temperature $\tau$ and the negative sampling strategy in the contrastive losses is insufficient.

## Related Work & Insights

- **vs. ActionFormer**: ActionFormer first introduced temporal feature pyramids but builds them via stride pooling, incurring information loss. HieraMamba replaces pooling with Mamba scanning for content-aware compression.
- **vs. SnAG / DeCafNet / OSGNet**: These are strong recent baselines, but all still rely on uniform downsampling for multi-scale construction. HieraMamba demonstrates that learned token compression surpasses these methods.
- **vs. CONE / RGNet**: Sliding-window methods suffer from boundary artifacts that disrupt temporal continuity. HieraMamba avoids this via Mamba's global state.
- This work suggests a broader direction: SSMs can serve not only as efficient alternatives to Transformers but also as tools for learning hierarchical compression.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The interleaved scanning design of AMP and the dual contrastive losses are clearly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ State-of-the-art on three benchmarks with a systematic method characteristics comparison.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; the comparison table is cleverly designed; figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Establishes a clean paradigm for long-video temporal grounding — linear complexity combined with hierarchical content compression.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding](mamba-vmr_multimodal_query_augmentation_via_generated_videos_for_precise_tempora.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](../../ICCV2025/video_understanding/hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)

<!-- RELATED:END -->
