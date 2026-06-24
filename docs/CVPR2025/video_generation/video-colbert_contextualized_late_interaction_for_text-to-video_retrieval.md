---
title: >-
  [Paper Note] Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval
description: >-
  [CVPR 2025][Video Generation][Text-to-Video Retrieval] This paper proposes Video-ColBERT, which introduces ColBERT's late interaction in text retrieval into text-to-video retrieval (T2VR). By performing MeanMaxSim interaction at both the frame and video levels and employing a dual Sigmoid loss to train independent yet compatible multi-granularity representations, Video-ColBERT outperforms existing dual-encoder methods on multiple T2VR benchmarks.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Text-to-Video Retrieval"
  - "Late Interaction"
  - "Multi-vector Representation"
  - "ColBERT"
  - "Sigmoid Loss"
date: 2026-05-08
content_hash: 02d8ce5ff6d6d2a1
---

# Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2503.19009](https://arxiv.org/abs/2503.19009)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Text-to-Video Retrieval, Late Interaction, Multi-vector Representation, ColBERT, Sigmoid Loss

## TL;DR

This paper proposes Video-ColBERT, which introduces ColBERT's late interaction in text retrieval into text-to-video retrieval (T2VR). By performing MeanMaxSim interaction at both the frame and video levels and employing a dual Sigmoid loss to train independent yet compatible multi-granularity representations, Video-ColBERT outperforms existing dual-encoder methods on multiple T2VR benchmarks.

## Background & Motivation

**Background**: Most efficient T2VR solutions employ a dual-encoder architecture. Mainstream methods are adapted from CLIP, such as CLIP4Clip, X-CLIP, and DRL.

**Limitations of Prior Work**: (1) Single-vector representations struggle to encode complex video content and textual concepts; (2) Existing token-level interaction methods have over-complicated mechanisms or suffer from limited final representations (e.g., only performing interactions on features after temporal modeling, thereby ignoring spatial information of individual frames); (3) InfoNCE loss is sensitive to noisy data.

**Key Challenge**: Fine-grained interaction is effective but slow, while single-vector methods are fast but lack expressiveness. ColBERT-style late interaction offers a middle ground, but existing T2VR methods have not fully utilized it.

**Goal**: (1) Achieve bi-level token-level interaction across spatial and spatiotemporal dimensions; (2) Design appropriate training objectives to ensure both representation branches are sufficiently strong; (3) Maintain the efficiency of dual encoders.

**Key Insight**: ColBERT's MaxSim lets each query token scan the document independently. Extending this to videos: MaxSim is applied separately to independent frame features (spatial) and temporally modeled features (spatiotemporal), and the two scores are then summed.

**Core Idea**: Bi-level MeanMaxSim interaction ($MMS_F + MMS_V = MMS_{FV}$), trained with a dual Sigmoid loss to independently optimize both representation branches.

## Method

### Overall Architecture

Dual encoder: The text is encoded as a sequence of token representations. On the video side, the image encoder extracts the frame [CLS] tokens, and a temporal transformer generates contextualized video features. Similarity is computed via frame-level and video-level bi-level MeanMaxSim.

### Key Designs

1. **Bi-level MeanMaxSim Interaction**:

    - **Function**: Query-video matching at both spatial and spatiotemporal granularities.
    - **Mechanism**: $MMS_F = \frac{1}{M}\sum_j \max_i (\mathbf{q}_j \cdot \mathbf{f}_i)$ is computed for frame features, and the same operation is applied for $MMS_V$ over video features. Compared to ColBERT's summation, a mean operation is adopted to handle queries of varying lengths. The final similarity is $MMS_{FV} = MMS_F + MMS_V$. Interaction is unidirectional from query to video.
    - **Design Motivation**: Since MMS_F covers spatial information, the temporal transformer can focus on encoding temporal dynamics, achieving functional division.

2. **Query and Visual Expansion Tokens**:

    - **Function**: Enhance query and video representations.
    - **Mechanism**: Padding tokens are added to the query side to participate in MMS, learning implicit query expansion. Learnable visual expansion tokens are added to the temporal transformer on the video side.
    - **Design Motivation**: T2VR queries are short and abstract; query expansion infers concepts that are not explicitly expressed.

3. **Dual Sigmoid Loss**:

    - **Function**: Independently train frame-level and video-level representations.
    - **Mechanism**: Sigmoid losses are computed separately for the $MMS_F$ and $MMS_V$ similarity matrices: $\mathcal{L}_D = \lambda_F \mathcal{L}_F + \lambda_V \mathcal{L}_V$. Each Sigmoid loss formulates contrastive learning as independent binary classification.
    - **Design Motivation**: Computing loss directly on the combined score causes gradient imbalance. Calculating them separately allows both paths to learn discriminative features independently, which are then summed during inference.

### Loss & Training

Dual Sigmoid loss: Each loss is formulated as $\mathcal{L} = -\frac{1}{|B|}\sum_i\sum_j \log\frac{1}{1+e^{z_{ij}(-t \cdot MMS + b)}}$, where $z_{ij}$ is the label, $t$ is a learnable scaling factor, and $b$ is a learnable bias. Unidirectional MaxSim reflects the query-video asymmetry.

## Key Experimental Results

### Main Results

| Method | MSR-VTT R@1 | MSVD R@1 | VATEX R@1 |
|------|------------|---------|----------|
| X-CLIP (B/16) | 49.3 | 50.4 | — |
| DRL (B/16) | 50.2 | 50.0 | 65.7 |
| **V-ColBERT (CLIP-B/16)** | **51.0** | 50.2 | 66.8 |
| **V-ColBERT (SigLIP-B/16)** | **51.5** | **55.2** | **68.0** |

### Ablation Study

| Configuration | Description |
|------|------|
| MMS_F Only | Pure frame-level spatial matching, ~48 R@1 |
| MMS_V Only | Spatiotemporal features only, ~49 R@1 |
| MMS_FV Bi-level | **~51 R@1**, two levels are complementary |
| InfoNCE Loss | Inferior to Sigmoid |
| SMS (Sum) | Unfair for variable-length queries |
| Without Query Expansion | Loses implicit concept expansion |

### Key Findings

- Bi-level interaction improves R@1 by 2-3 points compared to single-level, indicating spatial (frame-level) and spatiotemporal information are complementary.
- Dual Sigmoid outperforms InfoNCE and single Sigmoid.
- Unidirectional MaxSim outperforms bidirectional MaxSim—irrelevant frames in the video should not drag down the score.
- SigLIP serves as a better backbone than CLIP.

## Highlights & Insights

- **Migration of ColBERT to video** proves that multi-vector late interaction is effective in the video modality.
- **Functional division in hierarchical interaction**: MMS_F covers spatial features, allowing the temporal transformer to focus on temporal dynamics.
- **Dual Sigmoid loss** is simple yet clever—independent training guarantees discriminativeness for each path, while summation ensures compatibility.

## Limitations & Future Work

- Larger storage overhead compared to single-vector methods, as double the features must be stored for each video.
- Efficient approximations, such as storing only TopK frame features, were not explored.
- Comparison with large-scale models like InternVideo2 was not conducted.
- The interpretability of query expansion tokens remains unexplored.

## Related Work & Insights

- **vs. DRL**: DRL uses weighted token-wise interaction but only on features after temporal modeling. Video-ColBERT's bi-level interaction is more comprehensive.
- **vs. X-CLIP**: X-CLIP's multi-granularity interaction is more complex, whereas Video-ColBERT is simpler yet more effective.
- **vs. CLIP4Clip**: Late interaction improves R@1 by 5-8 points compared to single-vector methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Clever combination of ColBERT migration to video, bi-level interaction, and dual Sigmoid loss.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 5 datasets, multiple backbones, and rich ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and concise methodology.
- **Value**: ⭐⭐⭐⭐ Provides a powerful and concise late interaction solution for video retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation](the_devil_is_in_the_prompts_retrieval-augmented_prompt_optimization_for_text-to-.md)
- [\[ICCV 2025\] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization](../../ICCV2025/video_generation/quantifying_and_narrowing_the_unknown_interactive_text-to-video_retrieval_via_un.md)
- [\[ACL 2025\] Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](../../ACL2025/video_generation/q2e_query-to-event_decomposition_for_zero-shot_multilingual_text-to-video_retrie.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)

</div>

<!-- RELATED:END -->
