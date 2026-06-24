---
title: >-
  [Paper Note] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling
description: >-
  [CVPR 2026][Video Understanding][Video Temporal Grounding] HieraMamba proposes a Mamba-based hierarchical video temporal grounding architecture centered on the Anchor-MambaPooling (AMP) module. This module uses Mamba's selective scanning to compress video features layer-by-layer into multi-scale anchor tokens. Combined with anchor-conditioned and segment-pooled contrastive losses, it enhances the compactness and discriminativeness of hierarchical representations…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "State Space Model"
  - "Mamba"
  - "Hierarchical Representation"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: b38ca71e4de86270
---

# HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling

**Conference**: CVPR 2026  
**arXiv**: [2510.23043](https://arxiv.org/abs/2510.23043)  
**Code**: [https://vision.cs.utexas.edu/projects/hieramamba](https://vision.cs.utexas.edu/projects/hieramamba)  
**Area**: Video Understanding  
**Keywords**: Video Temporal Grounding, State Space Model, Mamba, Hierarchical Representation, Contrastive Learning

## TL;DR

HieraMamba proposes a Mamba-based hierarchical video temporal grounding architecture centered on the Anchor-MambaPooling (AMP) module. This module uses Mamba's selective scanning to compress video features layer-by-layer into multi-scale anchor tokens. Combined with anchor-conditioned and segment-pooled contrastive losses, it enhances the compactness and discriminativeness of hierarchical representations, achieving SOTA on Ego4D-NLQ, MAD, and TACoS.

## Background & Motivation

1. **Background**: The Video Temporal Grounding task requires localizing start and end times in untrimmed videos based on natural language queries. Evolving from predefined action localization to free-form text queries, this task supports applications like VQA and automatic video editing. Existing methods such as ActionFormer, SnAG, and DeCafNet have introduced multi-scale feature pyramids.

2. **Limitations of Prior Work**: Long videos (minutes to hours) present two intertwined challenges: (a) **Temporal Fidelity issue**—many methods reduce computational costs through fixed-length pooling, naive downsampling, or fixed windows, but these operations discard critical temporal cues or sever temporal structures at window boundaries; (b) **Multi-granularity issue**—different queries require different temporal granularities ("What did the detective do in the library" requires coarse-grained understanding, while "When did the detective pull a note from the shelf" requires fine-grained localization), making single-resolution methods difficult to apply effectively.

3. **Key Challenge**: The quadratic attention cost of Transformers is the root cause of downsampling and windowing heuristics—temporal resolution must be sacrificed to handle long sequences. Existing multi-scale models (SnAG, DeCafNet, OSGNet), while introducing multiple scales, still generate them via uniform downsampling or coarse pooling, lacking content-aware compression.

4. **Goal**: (1) How to process full-length video sequences within linear time complexity without downsampling or windowing? (2) How to construct content-aware multi-scale hierarchical representations rather than simple downsampled pyramids? (3) How to ensure hierarchical anchors are both compact (faithfully summarizing local information) and discriminative (distinguishable from other events)?

5. **Key Insight**: Human episodic memory is naturally hierarchical—shifting seamlessly across temporal scales from the overall layout of a room to precise finger movements. The authors identify Transformer's quadratic attention as the cause of temporal downsampling and propose replacing it with Mamba's linear-time selective scanning to achieve full-resolution long-range modeling.

6. **Core Idea**: Utilize Mamba's selective scanning to build content-aware hierarchical anchor compression (rather than naive downsampling). Stacked AMP modules form a fine-to-coarse multi-scale temporal pyramid, achieving precise long-video temporal grounding with linear complexity.

## Method

### Overall Architecture

Input: Clip-level features $V \in \mathbb{R}^{L_V \times D_v}$ extracted from a frozen video backbone (e.g., EgoVLP) and query embeddings $Q \in \mathbb{R}^{L_Q \times D_q}$ from a frozen text encoder (e.g., CLIP). The video encoder is a hierarchical stack of $L$ AMP modules, generating refined features $\tilde{V}^{(l)}$ and anchors $A^{(l+1)}$ for the next layer at each step, forming a feature pyramid $\mathcal{V}_{\text{pyr}} = \{\tilde{V}^{(0)}, \ldots, \tilde{V}^{(L-1)}\}$. Within each layer, AMP first performs anchor generation and interleaving, followed by bidirectional Mamba global encoding, narrow-window Transformer local encoding, and finally gated fusion to pass salient information. Simultaneously, two contrastive losses, ACC and SPC, are applied to the anchors layer by layer. The pyramid and text embeddings are fused via cross-modal attention, and a lightweight decoder regresses the start and end times $(t_s, t_e)$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    V["Video Features V<br/>Frozen Backbone EgoVLP"] --> AMP
    subgraph AMP["Anchor-MambaPooling (AMP) Module · Layer Stack ×L"]
        direction TB
        S1["Anchor Generation & Interleaving<br/>Pooling anchors every s frames and inserting into sequence"]
        S2["Global Encoding<br/>Bidirectional Mamba (Hydra) Linear Scan"]
        S3["Local Encoding<br/>Narrow-Window Transformer for Short-Range Details"]
        S4["Gated Fusion & Role Decoupling<br/>Sigmoid Gating + RMSNorm Residuals"]
        S1 --> S2 --> S3 --> S4
    end
    S4 -.->|In-layer Self-supervision (Structure)| ACC["Anchor-Conditioned Contrastive (ACC) Loss<br/>Pull anchor to window frames, push away distant anchors"]
    S4 -.->|In-layer Supervision (Semantics)| SPC["Segment-Pooled Contrastive (SPC) Loss<br/>GT segment prototype contrast"]
    AMP -->|"Refined features + Compressed anchors for next layer"| PYR["Feature Pyramid V_pyr"]
    Q["Query Embedding Q<br/>Frozen CLIP Text"] --> FUSE["Cross-modal Attention Fusion"]
    PYR --> FUSE
    FUSE --> DEC["Lightweight Decoder<br/>Regress start/end times (t_s, t_e)"]
```

### Key Designs

**1. Anchor-MambaPooling (AMP) Module: Refining and Compressing in a Single Mamba Scan**

The core contradiction in long-video modeling is either retaining full resolution at a quadratic attention cost or downsampling to save computation at the expense of temporal detail. AMP bypasses this by performing three steps per layer. First is **Anchor Generation and Interleaving**: an anchor token is initialized using local window pooling every $s$ frames and inserted before the frames it summarizes, creating an interleaved sequence $\hat{V} = [a_0, v_0, \ldots, v_{s-1}, a_1, v_s, \ldots] \in \mathbb{R}^{(L_0+L_1) \times D_v}$. Second is **Global Encoding**: Hydra (bidirectional Mamba scanning) processes this sequence; the forward scan allows each anchor to absorb information from preceding frames, while the backward scan absorbs from subsequent frames, maintaining linear complexity. Third is **Local Encoding**: A narrow-window Transformer (window size 5) is added to supplement short-range fine-grained attention patterns. After one pass, AMP outputs both refined features $\tilde{V}^{(l)}$ at the current resolution and compressed anchors $A^{(l+1)}$ for the next layer.

The brilliance of this interleaved design is that anchors and frame features share the same Mamba scan, allowing bidirectional information flow—anchors broadcast coarse-grained context to neighboring frames, and frames feed back details to refine the anchors. Compared to traditional feature pyramids (like ActionFormer's stride pooling), AMP generates multi-scale representations through token-level content-aware compression rather than indiscriminate uniform downsampling, ensuring key moments are not smoothed out during pooling.

**2. Gated Fusion & Role Decoupling: Distinguishing Global Structure from Local Patterns**

AMP internally contains three stages: Mamba (global), narrow-window Transformer (local), and FFN. If connected using unconditional residual additions, a common issue in hybrid architectures arises—the roles of the branches become blurred. HieraMamba applies RMS normalization and residuals between stages but replaces unconditional residuals with a learnable sigmoid gate $\boldsymbol{\sigma}$, letting the content decide how much output from each stage to pass. This explicitly preserves the division of labor—Mamba for global long-range and Transformer for local details—while the gating ensures only salient information propagates up the hierarchy, preventing noise from amplifying through the pyramid.

**3. Anchor-Conditioned Contrastive (ACC) Loss: Enforcing "Faithfulness" and "Distinguishability"**

Anchors compressed solely by architecture might not be optimal—they might fail to faithfully summarize window content or look too similar to other events. ACC is an in-layer self-supervised objective targeting this. At each layer, it pulls the anchor $a_i^{(l+1)}$ toward the $s$ frame tokens it summarizes (positive samples $\mathcal{P}_i^{(l)}$) and pushes it away from distant anchors (negative samples $\mathcal{N}_i^{(l)}$, with deliberate temporal gaps to avoid harming adjacent anchors):

$$\mathcal{L}_{\text{acc}}(a_i^{(l+1)}) = -\log \frac{\sum_{p \in \mathcal{P}_i^{(l)}} \exp(a_i^{(l+1)} \cdot p / \tau)}{\sum_{c \in \mathcal{P}_i^{(l)} \cup \mathcal{N}_i^{(l)}} \exp(a_i^{(l+1)} \cdot c / \tau)}$$

Pulling positive samples ensures "compactness" (anchors resemble their window frames), while pushing negative samples ensures "discriminativeness" (anchors represent unique events). Using multiple positive samples instead of a single one fits the "one-to-many" proxy relationship where an anchor must faithfully represent multiple frames, preventing information loss from single-frame alignment.

**4. Segment-Pooled Contrastive (SPC) Loss: Separating Target Segments from Surroundings**

While ACC handles structural consistency, anchors must also align with query semantics—this is where SPC comes in. It is a supervised objective that pools frame tokens within the GT segment $[t_{\text{start}}, t_{\text{end}})$ into a segment prototype $z_{\text{seg}}^{(l)}$ at each layer, performing contrastive learning with in-segment frames as positives and out-of-segment frames as negatives. A key detail is using the pooled prototype as the positive anchor rather than frame-by-frame alignment. Since action segments often contain sub-actions (e.g., "reach → grasp → retract"), forcing them to a single representation is distortive; the prototype preserves this internal diversity. ACC (structural self-supervision) and SPC (semantic supervision) are thus complementary.

### Loss & Training

The total contrastive loss $\mathcal{L}_{\text{contrast}} = \lambda_{\text{ACC}} \mathcal{L}_{\text{ACC}} + \lambda_{\text{SPC}} \mathcal{L}_{\text{SPC}}$ is optimized jointly with standard temporal grounding losses (boundary regression + classification).

## Key Experimental Results

### Main Results

Results on Ego4D-NLQ (using EgoVLP features):

| Method | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5 | Avg. |
|------|------------|------------|------------|------------|------|
| SnAG | 15.72 | 10.78 | 38.39 | 27.44 | 23.08 |
| DeCafNet | 18.10 | 12.55 | 38.85 | 28.27 | 24.44 |
| RGNet | 18.28 | 12.04 | 34.02 | 22.89 | 21.81 |
| OSGNet | 16.13 | 11.28 | 36.78 | 25.63 | 22.46 |
| **HieraMamba** | **18.81** | **13.04** | **40.82** | **29.96** | **25.66** |

SOTA was also achieved on MAD and TACoS (detailed data reported in the paper).

### Method Characteristics Comparison

| Method | Naive Downsampling | Fixed Pooling | Quadratic Cost | Sliding Window | Ego4D Avg.R |
|------|-----------|---------|-----------|---------|------------|
| 2D-TAN | ✓ | ✓ | ✓ | — | 6.46 |
| CONE | — | — | ✓ | ✓ | 17.67 |
| SnAG | ✓ | — | — | — | 23.08 |
| DeCafNet | ✓ | — | — | — | 24.44 |
| **HieraMamba** | **—** | **—** | **—** | **—** | **25.66** |

HieraMamba is the only method that avoids all four undesirable characteristics simultaneously.

### Key Findings

- The benefits of avoiding downsampling and windowing are particularly evident in long videos—HieraMamba shows the largest gains on Ego4D (8-minute average) and MAD (multi-hour movies).
- The contributions of ACC and SPC losses are complementary—ACC primarily improves anchor quality and consistency within the hierarchy, while SPC improves semantic alignment with queries.
- The global-local decoupling of Mamba + narrow-window Transformer outperforms pure Mamba or pure Transformer setups.
- The gating mechanism (sigmoid gate) outperforms unconditional residual connections, indicating that content-adaptive information propagation is crucial for hierarchical models.

## Highlights & Insights

- **Ingenious Interleaved Design of AMP**: Inserting anchor tokens into the frame sequence for Mamba scanning allows anchors to naturally gain global context summary capabilities (Mamba's state compression) while frame features gain neighborhood summaries from anchors—bidirectional information flow in a single pass with linear complexity.
- **Methodology of "Avoiding All Bad Traits"**: By systematically analyzing the limitations of existing methods (downsampling, fixed pooling, quadratic cost, sliding windows), the authors designed an architecture that circumvents all issues, reflecting elegant engineering intuition.
- **Multi-positive Design in ACC Loss**: Traditional contrastive learning uses single positives, but in temporal grounding, an anchor must represent multiple frames. Multi-positive InfoNCE naturally fits this requirement, ensuring the anchor acts as a faithful proxy for the entire window.

## Limitations & Future Work

- Dependence on frozen video backbones (EgoVLP/InternVideo); if the distilled clip features are of poor quality, hierarchical modeling cannot fully compensate.
- The AMP stride $s$ is a fixed hyperparameter; different temporal scales of queries may require adaptive strides.
- Mamba's unidirectional causal structure requires compensation via bidirectional Hydra, which increases complexity—whether a more native bidirectional SSM design exists remains to be explored.
- Inference speed is not discussed in detail—though theoretically linear, the actual speed of interleaved, bidirectional scanning across multiple layers needs verification.
- Sensitivity analysis for the temperature $\tau$ and negative sample selection strategies in the contrastive losses is insufficient.

## Related Work & Insights

- **vs ActionFormer**: ActionFormer introduced temporal feature pyramids but built them via stride pooling—which is lossy. HieraMamba replaces pooling with Mamba scanning for content-aware compression.
- **vs SnAG / DeCafNet / OSGNet**: These recent strong baselines still rely on uniform downsampling for multi-scale construction. HieraMamba proves that learned token compression can outperform these methods.
- **vs CONE / RGNet**: These sliding window methods suffer from window boundary discontinuities. HieraMamba uses Mamba's global state to avoid this issue.
- This work suggests a direction: SSMs can serve not just as efficient alternatives to Transformers, but also as tools for learning hierarchical compression.

## Rating

- Novelty: ⭐⭐⭐⭐ The interleaved scanning design of the AMP module and the dual contrastive losses show clear innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ SOTA on three benchmarks with systematic characteristic comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, clever table design, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a clear paradigm for long-video temporal grounding—linear complexity + hierarchical content compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HERO: Hierarchical Embedding-Refinement for Open-Vocabulary Temporal Sentence Grounding in Videos](hero_hierarchical_embedding-refinement_for_open-vocabulary_temporal_sentence_gro.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] T2SGrid: Temporal-to-Spatial Gridification for Video Temporal Grounding](t2sgrid_temporal-to-spatial_gridification_for_video_temporal_grounding.md)
- [\[CVPR 2026\] MS-Temba: Multi-Scale Temporal Mamba for Understanding Long Untrimmed Videos](ms-temba_multi-scale_temporal_mamba_for_understanding_long_untrimmed_videos.md)
- [\[ICLR 2026\] HiTeA: Hierarchical Temporal Alignment for Training-Free Long-Video Temporal Grounding](../../ICLR2026/video_understanding/hitea_hierarchical_temporal_alignment_for_training-free_long-video_temporal_grou.md)

</div>

<!-- RELATED:END -->
