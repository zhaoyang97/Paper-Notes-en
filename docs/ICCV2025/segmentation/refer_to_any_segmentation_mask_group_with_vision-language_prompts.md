---
title: >-
  [Paper Note] Refer to Any Segmentation Mask Group With Vision-Language Prompts
description: >-
  [ICCV 2025][Segmentation][Omni-modal referring segmentation] This paper proposes the Omni-modal Referring Expression Segmentation (ORES) task and the RAS framework…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Omni-modal referring segmentation"
  - "mask grouping"
  - "vision-language prompts"
  - "large multimodal models"
  - "non-autoregressive decoding"
date: 2026-05-08
content_hash: 617fdefd182a52e1
---

# Refer to Any Segmentation Mask Group With Vision-Language Prompts

**Conference**: ICCV 2025
**arXiv**: [2506.05342](https://arxiv.org/abs/2506.05342)  
**Code**: [Ref2Any](https://Ref2Any.github.io)  
**Area**: Image Segmentation
**Keywords**: Omni-modal referring segmentation, mask grouping, vision-language prompts, large multimodal models, non-autoregressive decoding

## TL;DR

This paper proposes the Omni-modal Referring Expression Segmentation (ORES) task and the RAS framework, which leverages a mask-level LMM with a non-autoregressive decoding mechanism to select target mask groups from a candidate pool based on vision-language hybrid prompts. The approach achieves state-of-the-art performance on the newly introduced ORES dataset as well as classical RES/GRES benchmarks.

## Background & Motivation

Referring Expression Segmentation (RES) associates textual descriptions with segmentation masks to enable language-driven object localization. However, in practical applications such as autonomous driving, robotic manipulation, AR, and image editing, users frequently need to express complex relations involving **visual reference entities** (e.g., "everything that has the same color as this object"), where text descriptions alone are insufficient to precisely identify the referent.

Existing methods exhibit three key limitations:

**Interactive segmentation models** (e.g., SEEM) support visual prompts, but visual prompts can only point directly to target entities and cannot express "other targets related to a reference entity."

**Grounding LMMs** (e.g., Groundhog) support region description tasks but cannot perform segmentation conditioned on mask prompts.

**Most methods produce only a single output mask per query**, making them incapable of handling multi-target scenarios.

Core motivation: define a new task, ORES, that allows users to specify **text + reference mask** hybrid prompts and retrieve a **group of masks** satisfying the specified conditions in a single pass, enabling more flexible and practical segmentation interaction.

## Method

### Overall Architecture

RAS (Refer to Any Segmentation Mask Group) extends LLaVA-1.5 (Vicuna-13B) with four core modules:

1. **Segmentation foundation model** (SAM/Co-DETR): generates a candidate mask pool
2. **Visual encoder ensemble**: CLIP + SigLIP + ConvNeXt-CLIP + DINOv2 + 2D positional encoding
3. **Mask projector**: maps mask-level features into the language embedding space
4. **Binary classifier**: performs binary classification on each candidate mask to determine group membership

### Key Design 1: Mask Tokenization

For each candidate mask, it is downsampled to the spatial resolution of each visual encoder's feature map, and mask-level features are obtained via average pooling within the masked region. Features from all encoders are concatenated and projected into the language feature space via the mask projector, forming **mask tokens**.

- Candidate mask tokens are prefixed with the `<mask-pool-pre>` special token.
- Reference mask tokens are prefixed with the `<mask-ref-pre>` special token.
- Both share the same tokenization pipeline and are distinguished by their respective special tokens.

### Key Design 2: Non-Autoregressive Mask Group Decoding

Conventional autoregressive approaches that predict mask embeddings one by one suffer from two issues: (a) LLMs inherently model discrete token distributions, making continuous embedding prediction unnatural; (b) unordered set prediction requires unstable bipartite matching.

RAS reformulates mask group prediction as a **per-mask binary classification problem**:

1. All context tokens (global visual + text + reference mask) are first fed as input.
2. Candidate mask tokens are then fed again to capture the LLM's output hidden states.
3. A binary classifier applied to these hidden states determines whether each candidate mask should be included in the target group.

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \text{BCE}(\hat{y}_i, y_i)$$

where positive samples are assigned a larger weight $w_i$ to address class imbalance. This strategy enables **single-pass forward inference** for classifying all candidate masks, with an inference latency of only 0.56s (vs. 2.13s for autoregressive decoding).

### Key Design 3: Multi-Stage Training

**Stage 1 — Mask Projector Pre-training**: All modules are frozen except the mask projector. Using only mask tokens (without global visual tokens), the LLM is trained to predict image captions, aligning mask representations with the language space.

**Stage 2 — Visual Instruction Fine-tuning**: All modules except the visual encoders are unfrozen and trained on the mask grouping task. Further fine-tuning on RES/GRES data can be applied to adapt to downstream tasks.

### Dataset Construction

- **MaskGroups-2M**: 2 million mask grouping samples automatically constructed from MS-COCO/LVIS/VG/RES/GRES datasets, covering four grouping criteria: category, attribute, location, and free-form description.
- **MaskGroups-HQ**: 100,299 high-quality mask groups annotated by humans (96,697 for training + 3,599 for evaluation), with 28% containing reference masks.

## Key Experimental Results

### Main Results: ORES Task (MaskGroups-HQ)

| Model | Text gIoU | Text cIoU | Hybrid gIoU | Hybrid cIoU | Overall cIoU |
|-------|:---:|:---:|:---:|:---:|:---:|
| ReLA | 34.93 | 43.22 | - | - | - |
| GSVA-13B | 41.98 | 49.55 | - | - | - |
| **RAS-13B (SAM)** | 55.82 | 60.12 | 35.91 | 37.77 | 53.93 |
| **RAS-13B (ORES-FT)** | **66.71** | **74.59** | **58.72** | **68.77** | **73.13** |

RAS is the only method capable of handling visual reference prompts. After ORES fine-tuning, cIoU improves substantially from 53.93 to 73.13.

### RES/GRES Benchmarks

| Model | RefCOCO val | RefCOCO+ val | RefCOCOg val | Avg. |
|-------|:---:|:---:|:---:|:---:|
| PSALM-1.3B | 83.6 | 72.9 | 73.8 | 77.1 |
| RAS-13B (RES-FT) | **81.0** | **75.1** | **76.0** | **77.8** |

RAS also achieves state-of-the-art results on GRES (gRefCOCO), with an overall cIoU of 71.79.

### Ablation Study

| Decoding Paradigm | cIoU | Inference Latency (s) |
|-------------------|:---:|:---:|
| Autoregressive | 45.34 | 2.13 |
| **Non-autoregressive** | **53.75** | **0.56** |

| Visual Encoder | Overall cIoU |
|----------------|:---:|
| CLIP only | 52.44 |
| DINOv2 only | 47.71 |
| **Four-encoder ensemble** | **53.75** |

### Key Findings

1. Candidate mask quality analysis: the Oracle cIoU of SAM/Co-DETR reaches 86–87, far exceeding the best final performance of existing methods (~77), indicating that the candidate mask pool is of very high quality.
2. Non-autoregressive decoding improves cIoU by +8.4 over autoregressive decoding while achieving 3.8× faster inference.
3. The four-encoder ensemble consistently outperforms any single encoder; ConvCLIP contributes the most on visual reference tasks.

## Highlights & Insights

- **Task definition innovation**: ORES is the first formulation to unify text and visual reference prompts while outputting mask groups rather than individual masks, closely reflecting real-world application needs.
- **Decoupled segmentation and understanding**: high-quality candidates are provided by segmentation foundation models, while the LMM handles semantic understanding and selection, leveraging the strengths of each.
- **Elegant non-autoregressive decoding design**: reformulating set prediction as per-element binary classification elegantly sidesteps the difficulties of unordered set matching.

## Limitations & Future Work

- Candidate mask quality is bounded by the segmentation foundation model; if the target is absent from the candidate pool, it cannot be recovered.
- The Vicuna-13B backbone entails substantial parameter count, leading to high deployment costs.
- The framework does not support text generation capabilities (e.g., explaining predictions), limiting interactivity.

## Related Work & Insights

- RES/GRES direction: ReLA, GSVA, PSALM, and others extend to multi-target/zero-target queries.
- Grounding LMMs: LISA, Groundhog, GLaMM, and others achieve pixel-level grounding.
- Segmentation foundation models: SAM provides class-agnostic, high-quality mask proposals.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both the ORES task formulation and non-autoregressive mask group decoding represent pioneering contributions.
- **Technical Depth**: ⭐⭐⭐⭐ — Mask tokenization, multi-encoder ensemble, and training strategy are thoroughly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-task evaluation across ORES/RES/GRES with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear exposition with well-motivated task formulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?](how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation](lawdis_language-window-based_controllable_dichotomous_image_segmentation.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)

</div>

<!-- RELATED:END -->
