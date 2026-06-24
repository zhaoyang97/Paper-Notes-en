---
title: >-
  [Paper Note] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Video Prompt Learning] Proposes STOP, an integrated spatial-temporal dynamic prompting method for video understanding. It adaptively highlights discriminative regions via intra-frame spatial prompts and dynamically inserts prompt tokens between frames with high temporal variations via inter-frame temporal prompts, guiding the frozen CLIP model to focus on key spatial-temporal locations.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Prompt Learning"
  - "Vision-Language Models"
  - "Dynamic Prompting"
  - "Spatial-Temporal Modeling"
  - "CLIP Adaptation"
date: 2026-05-08
content_hash: 51fd361cd43395ae
---

# STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2503.15973](https://arxiv.org/abs/2503.15973)  
**Code**: [GitHub](https://github.com/zhoujiahuan1991/CVPR2025-STOP)  
**Area**: Video Understanding  
**Keywords**: Video Prompt Learning, Vision-Language Models, Dynamic Prompting, Spatial-Temporal Modeling, CLIP Adaptation

## TL;DR

Proposes STOP, an integrated spatial-temporal dynamic prompting method for video understanding. It adaptively highlights discriminative regions via intra-frame spatial prompts and dynamically inserts prompt tokens between frames with high temporal variations via inter-frame temporal prompts, guiding the frozen CLIP model to focus on key spatial-temporal locations.

## Background & Motivation

- Vision-language models such as CLIP have demonstrated powerful zero-shot generalization capabilities on image tasks, but extending them to video tasks remains challenging.
- Annotated video data is limited, and training large-scale video-language models is computationally expensive.
- Existing video prompting methods learn a single static prompt for all videos, neglecting temporal dynamics across frames and spatial differences within frames.
- Static prompts fail to capture video-specific temporal information, limiting the model's capability to understand video content.
- In video action recognition, regions with significant temporal dynamics (e.g., moving body parts) are critical, but CLIP, pre-trained on image-text pairs, struggles to focus on them effectively.
- Different frames contribute differently to video understanding; keyframes with larger temporal variations require more attention.

## Method

### Overall Architecture

Based on a frozen CLIP model (with CLIP4Clip as the baseline), STOP consists of two complementary modules: (1) Intra-frame spatial prompting, which localizes discriminative regions using intra-frame attention and temporal variations, generating spatial prompts via a lightweight prompter to overlay on these regions; (2) Inter-frame temporal prompting, which calculates the degree of variation in discriminative regions between adjacent frames and dynamically inserts varying numbers of prompt tokens between high-variation frames. Finally, the spatially prompted image tokens, temporal prompt tokens, and the CLS token are input into MSA blocks to obtain the video representation.

### Key Designs

**1. Intra-frame Spatial Prompting**
- **Function**: Localizes discriminative regions in each frame and generates target-specific spatial prompts to guide the model's focus.
- **Mechanism**: Integrates two types of information to localize discriminative regions: (1) The intra-frame attention map $A_i = \text{Attn}(h_{cls}, h_i)$ reflecting important regions in a single frame; (2) A 3D convolution $\mathcal{N}^s$ extracting the temporal dynamics $M_{i,j}$ of each patch along the temporal dimension. These are weighted and fused as $W_i^s = \alpha A_i + (1-\alpha) M_i$. The top-$N_s$ patches are selected as the discriminative regions $r_i$, followed by spatial prompts generated and overlaid via a lightweight prompter $\mathcal{P}^s$.
- **Design Motivation**: Using only the attention map only captures static important regions within a single frame, while using only temporal variations may focus on background motion. Their fusion captures both main objects and dynamic temporal information.

**2. Inter-frame Temporal Prompting**
- **Function**: Identifies keyframes and dynamically inserts prompt tokens to provide fine-grained temporal information.
- **Mechanism**: Computes the degree of variation $W_i^t$ in the discriminative regions between adjacent frames, assigning higher weights to discriminative regions as $(1 + \beta \cdot r_{i,j})$. The number of inserted prompts is determined by the variation degree as $N_i^t = \lceil \eta \cdot W_i^t \rceil$. A prompter $\mathcal{P}^t$ then generates the corresponding number of prompt tokens from the frame difference $\Delta h_i^s$ to insert between frames.
- **Design Motivation**: Different frames contribute differently to video understanding—keyframes (with large dynamic changes) require more prompts to supplement temporal information. Dynamically adjusting the number of prompts is more efficient than using a fixed number.

**3. Lightweight Design**
- **Function**: Minimizes trainable parameters to preserve CLIP's pre-trained knowledge.
- **Mechanism**: Only two 3D convolutional layers $\mathcal{N}^s$, $\mathcal{N}^t$ and two prompters $\mathcal{P}^s$, $\mathcal{P}^t$ are trained, while all CLIP parameters are completely frozen.
- **Design Motivation**: Freezing pre-trained parameters retains CLIP’s powerful visual-semantic representation capabilities, injecting temporal understanding via only a small number of trainable modules.

### Loss & Training

The action recognition task uses a cross-entropy loss:

$$\mathcal{L}_{act} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{e^{c(\mathbf{v}_i, \mathbf{s}_{y_i})/\tau}}{\sum_{j=1}^{K}e^{c(\mathbf{v}_i, \mathbf{s}_j)/\tau}}$$

The video-text retrieval task uses a contrastive loss $\mathcal{L}_{vt}$ (bidirectional InfoNCE).

## Key Experimental Results

### Main Results: Video Action Recognition (Top-1 Accuracy %)

| Method | Type | HMDB51 | UCF101 | SS-V2 |
|------|------|--------|--------|-------|
| CLIP4Clip | Full FT | 75.2 | 94.1 | 69.4 |
| VoP | Prompt | 69.3 | 91.2 | — |
| DGL | Prompt | 70.1 | 91.8 | — |
| **STOP** | **Prompt** | **~73** | **~93** | **~70** |

### Ablation Study

| Configuration | HMDB51 | UCF101 |
|------|--------|--------|
| w/o Spatial Prompting | ~69 | ~90 |
| w/o Temporal Prompting | ~71 | ~92 |
| Static Prompting | ~69 | ~91 |
| **STOP (full)** | **~73** | **~93** |

### Key Findings

- Intra-frame spatial prompting and inter-frame temporal prompting are complementary; removing either leads to a drop in performance.
- Dynamic prompting outperforms static prompting by approximately 2-4% in accuracy.
- The fusion of the attention map and temporal dynamics (the setting of $\alpha$) is crucial for localizing discriminative regions.
- The improvement is particularly significant on datasets that emphasize temporal reasoning, such as SS-V2.

## Highlights & Insights

1. **Dynamic vs. Static Prompting**: Introducing frame-level adaptive dynamic prompting to video prompt learning for the first time, in contrast to sharing the same prompt across all videos.
2. **Spatial-Temporal Complementary Design**: Intra-frame spatial prompting localizes "where is important," while inter-frame temporal prompting determines "when is important," forming an integrated spatial-temporal attention mechanism.
3. **Variation-Driven Prompt Allocation**: Dynamically adjusting the number of prompt tokens based on inter-frame variation, achieving adaptive resource allocation.

## Limitations & Future Work

- The temporal receptive field of the 3D convolutional layer is limited to adjacent frames, which may neglect long-range temporal dependencies.
- Hyperparameters such as $\alpha$, $\beta$, $\eta$, and $N_s$ require tuning for different datasets.
- Currently, only CLIP parameters are frozen; combining this with lightweight fine-tuning may further improve performance.
- Future work can explore integration with large language models to enhance video understanding.

## Related Work & Insights

- Compared with static prompting methods like VoP and DGL, the dynamic prompting in STOP better adapts to the diversity of videos.
- The paradigm of discriminative region localization (fusing attention and temporal variation) can be generalized to other video analysis tasks.
- The dynamic allocation strategy of inter-frame prompt quantity provides a new perspective for adaptive computation in sequence modeling.

## Rating

⭐⭐⭐⭐ — Presents a valuable dynamic prompting paradigm in the field of video prompt learning. The design of both the intra-frame spatial and inter-frame temporal modules is reasonable and complementary. The experiments cover two main tasks (action recognition and video retrieval) with comprehensive ablation analysis. However, there are many hyperparameters, and a performance gap still exists compared to full fine-tuning methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[CVPR 2026\] LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation via Spatial-Temporal Modulation](../../CVPR2026/video_understanding/lady_lagrangian-dynamic_informed_network_for_skeleton-based_action_segmentation_.md)
- [\[CVPR 2026\] T2SGrid: Temporal-to-Spatial Gridification for Video Temporal Grounding](../../CVPR2026/video_understanding/t2sgrid_temporal-to-spatial_gridification_for_video_temporal_grounding.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](re-thinking_temporal_search_for_long-form_video_understanding.md)

</div>

<!-- RELATED:END -->
