---
title: >-
  [Paper Note] "Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment"
description: >-
  [CVPR2025][Model Compression][Paper Note] Academic paper note for "Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment".
tags:
  - CVPR2025
  - Model Compression
date: 2025-04-03
content_hash: 052c800c920ce175
---
# Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment

**Author**: Fatemeh Behrad, Farzad Khorasani, et al.  
**Institution**: KU Leuven  
**Conference**: CVPR 2025  

## Background & Motivation

Image Aesthetic Assessment (IAA) aims to assess the aesthetic quality of images automatically and is widely utilized in scenarios such as photo recommendation, automatic cropping, and image enhancement. While Vision Transformers (ViTs) perform exceptionally well on IAA tasks, they face the following core challenges:

**Limitations of Fixed Resolution**: ViTs typically resize input images to a fixed size (e.g., 224×224), which destroys the original composition and aspect ratio of the image—both of which are crucial factors in aesthetic assessment.

**Computational Cost of High Resolution**: Directly processing high-resolution images leads to a sharp increase in the number of tokens. With an attention computational complexity of $O(n^2)$, practical deployment becomes challenging.

**Loss of Compositional Information**: Traditional resizing and center cropping change the spatial relationship of subjects and the foreground-to-background ratio, preventing the model from learning genuine composition aesthetics.

**Lack of Multi-scale Information**: Aesthetic assessment requires simultaneous attention to global layouts (composition, color distribution) and local details (texture, sharpness), which is difficult to balance using a single-scale ViT.

The core idea of Charm is to significantly reduce the number of tokens while preserving composition, high-resolution, aspect ratio, and multi-scale information.

## Method

### CHARM Core Designs

Charm stands for the acronym of four key attributes:
- **C**omposition: Preserving the original composition
- **H**igh-resolution: Preserving high-resolution details
- **A**spect **R**atio: Preserving the original aspect ratio
- **M**ulti-scale: Integrating multi-scale information

### Dual-scale Token Strategy

Charm adopts a two-level tokenization scheme:

**Coarse Tokens**:
- Uses an expanded patch size $p' = p \times n$, where $p$ is the original patch size and $n$ is the scaling factor.
- Covers the entire image to maintain global composition and aspect ratio information.
- Token count = $\frac{H \times W}{(p \times n)^2}$

**Fine Tokens**:
- Uses the original patch size $p$, extracted only in selected key regions.
- Selects the most important patch locations via attention scores or content saliency.
- Provides high-resolution local details.

### Scale Embedding

To enable the model to distinguish between coarse and fine tokens, a learnable scale embedding is introduced:

$$e_{scale} = \begin{cases} e_{coarse} & \text{if token from coarse scale} \\ e_{fine} & \text{if token from fine scale} \end{cases}$$

Final token embedding: $e_{token} = e_{patch} + e_{pos} + e_{scale}$

### Token Reduction Efficiency

| Configuration | Token Count | Relative to Original |
|------|-----------|---------|
| Original ViT (224×224) | 196 | 100% |
| Original ViT (448×448) | 784 | 400% |
| Charm (448×448) | ~180 | ~23% (relative to high resolution) |
| Token Reduction Rate | - | **77.7%** |

### Training Strategy

1. Initialized with pre-trained ViTs (such as DINOv2 or CLIP).
2. The backbone network is frozen, fine-tuning only the scale embeddings and classification head.
3. Dynamic Batching: Since token counts can vary due to different aspect ratios of input images, padding and attention masks are used.

## Key Experimental Results

### AVA Dataset

| Method | SRCC ↑ | PLCC ↑ | ACC ↑ |
|------|--------|--------|-------|
| NIMA (Inception) | 0.636 | 0.642 | 0.815 |
| MUSIQ | 0.726 | 0.738 | 0.832 |
| VILA | 0.738 | 0.745 | 0.841 |
| Baseline ViT | 0.741 | 0.745 | - |
| **Charm (ours)** | **0.773** | **0.779** | - |
| Gain | +3.2% | **+4.5%** | - |

### TAD66k Dataset

| Method | SRCC ↑ | PLCC ↑ | ACC ↑ |
|------|--------|--------|-------|
| TANet | 0.432 | 0.441 | 0.692 |
| VILA | 0.521 | 0.534 | 0.743 |
| Baseline ViT | 0.538 | 0.547 | 0.752 |
| **Charm (ours)** | **0.612** | **0.625** | **0.794** |
| Gain | +7.4% | +7.8% | **+14.8%** (vs TANet) |

### Ablation Study

| Component | AVA PLCC | TAD66k ACC |
|------|----------|-----------|
| Coarse scale only | 0.753 | 0.768 |
| Fine scale only | 0.742 | 0.759 |
| Coarse + Fine (without scale embedding) | 0.769 | 0.783 |
| **Full Charm** | **0.779** | **0.794** |

## Highlights & Insights

1. **CHARM Design Philosophy**: Systematically preserving four key aesthetic properties (composition, high-resolution, aspect ratio, and multi-scale).
2. **Dual-scale Token Strategy**: Avoiding computational explosion caused by high resolution while retaining both global and local information.
3. **77.7% Token Reduction**: Significantly reducing computational overhead while improving aesthetic assessment accuracy.
4. **Plug-and-play**: Directly applicable to any pre-trained ViT without architectural modifications to the Transformer.

## Limitations & Future Work

- The selection strategy for fine-grained patches currently relies on heuristic approaches, which may not be optimal.
- For images with extreme aspect ratios (such as panoramas), the padding overhead matches are relatively large.
- The extension to video aesthetic assessment remains unexplored.

## Related Work & Insights

- MUSIQ: Multi-scale Image Quality assessment Transformer
- VILA: Vision-Language Alignment for aesthetic assessment
- TANet: Theme-Aware aesthetic Network
- Token pruning/merging: DynamicViT, ToMe, etc.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs](chapter-llama_efficient_chaptering_in_hour-long_videos_with_llms.md)
- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)

</div>

<!-- RELATED:END -->
