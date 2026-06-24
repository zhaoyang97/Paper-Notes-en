---
title: >-
  [Paper Note] Pixel-Level Reasoning Segmentation via Multi-turn Conversations
description: >-
  [Segmentation] Proposes a new task of pixel-level reasoning segmentation (Pixel-level RS) to achieve fine-grained segmentation by progressively understanding user intent through multi-turn conversations. A PRIST dataset containing 24k dialogue turns is constructed, and a MIRAS framework is designed to outperform existing baselines in both segmentation accuracy and reasoning capability.
tags:
  - "Segmentation"
date: 2026-05-08
content_hash: fc99ec8089d32b11
---

# Pixel-Level Reasoning Segmentation via Multi-turn Conversations

| Conference | arXiv | Code | Area | Keywords |
|------|-------|------|------|--------|
| ACL 2025 | [2502.09447](https://arxiv.org/abs/2502.09447) | [GitHub](https://github.com/ccccai239/PixelRIST) | segmentation | pixel-level reasoning segmentation, multi-turn conversation, MLLM, SAM, semantic region alignment |

## TL;DR

Proposes a new task of pixel-level reasoning segmentation (Pixel-level RS) to achieve fine-grained segmentation by progressively understanding user intent through multi-turn conversations. A PRIST dataset containing 24k dialogue turns is constructed, and a MIRAS framework is designed to outperform existing baselines in both segmentation accuracy and reasoning capability.

## Background & Motivation

- **Existing Problems**: Current visual perception systems only support region-level segmentation in single-turn conversations. They rely on complex and explicit query instructions, fail to reason at the pixel level, and cannot understand dynamically evolving user intents during interaction.
- **Key Gap**: Existing reasoning segmentation methods (e.g., LISA, PixelLM) suffer from two limitations: (1) relying on single-turn ambiguous queries and failing to fully understand user's changing intent; (2) lacking pixel-level segmentation capability, only achieving coarse region-level segmentation via one-step explanations.
- **Research Motivation**: Multi-turn interactions can progressively clarify ambiguous user instructions (e.g., "baking bread"), eventually focusing on specific objects through progressive dialogues to achieve precise pixel-level segmentation.
- **New Task Definition**: Pixel-level RS requires the system to track the evolving user intent through multi-turn conversations while simultaneously generating pixel-level segmentation masks and textual reasoning chains.

## Method

### Overall Architecture

The MIRAS (Multi-turn Interactive ReAsoning Segmentation) framework consists of three core components:
1. **Dual Visual Encoder**: Extracts multi-scale visual features
2. **Multimodal Large Language Model (MLLM)**: Performs multi-turn dialogue and reasoning based on LLaVA
3. **Mask Decoder**: Generates pixel-level segmentation masks based on SAM

By introducing a special token `[SEG]` as a placeholder for the formatted segments, end-to-end reasoning and segmentation are achieved.

### Key Designs

1. **Dual Visual Encoder Fusion**: High-resolution images (768×768) are processed by ConvNext-L and low-resolution images (336×336) by CLIP-L/14. Multi-scale features are fused via a cross-attention module to enhance visual detail capture.
2. **Semantic Region Alignment Strategy**: A segmentation prompt template `[OBJ]{CLASS}[SEG]` is designed, utilizing `[OBJ]` to extract relevant subsequences. Cross-attention is used to inject semantic information into the mask decoder, resolving the dimension mismatch issue caused by varying object description lengths.
3. **PRIST Dataset Construction**: A three-step progressive dialogue automatic generation pipeline based on reasoning trees: (Step 1) Extract visible elements, (Step 2) Construct reasoning questions and reasoning trees, (Step 3) Organize reasoning tree nodes into a multi-turn dialogue format.

### Loss & Training

$$\mathcal{L} = \lambda_t \mathcal{L}_t + \lambda_{bce} \text{BCE}(\mathcal{M}, \hat{\mathcal{M}}) + \lambda_{dice} \text{DICE}(\mathcal{M}, \hat{\mathcal{M}})$$

where $\lambda_t=1.0$, $\lambda_{bce}=2.0$, $\lambda_{dice}=0.5$. Two-stage training: Stage-1 mask-text alignment pre-training, and Stage-2 instruction tuning on the PRIST dataset. Only the mask decoder and projection layers are trained, while the image encoder and MLLM are frozen.

## Experiments

### Main Results

| Model | CIoU | Prec. | Recall | F1 | BLEU-4 | ROUGE_L | METEOR |
|------|------|-------|--------|-----|--------|---------|--------|
| GPT-4o (zero-shot) | 14.13 | 17.35 | 35.01 | 23.18 | 4.30 | 26.35 | 28.55 |
| OMG-LLaVA (zero-shot) | 9.67 | 16.67 | **77.80** | 27.46 | 8.70 | 23.47 | 27.90 |
| LISA (fine-tuned) | 11.23 | 26.23 | 29.22 | 27.64 | 7.81 | 27.84 | 30.74 |
| OMG-LLaVA (fine-tuned) | 13.84 | 21.54 | 49.31 | 29.98 | 11.21 | 30.59 | 39.18 |
| **MIRAS (Stage-2)** | **14.72** | **24.22** | 40.61 | **30.34** | 8.51 | **30.82** | **40.06** |

### Ablation Study (Reasoning Quality)

| Model | PR | LC | CC | TR | Win Rate(%) |
|------|-----|-----|-----|-----|-------------|
| Human | 4.03 | 4.04 | — | — | — |
| MIRAS | **Highest** | **Highest** | **Highest** | **Highest** | **42%** |

The average Win Rate of each model increases by about 10% after fine-tuning. MIRAS achieves SOTA performance on all four reasoning metrics.

### Key Findings

1. **Generality of PRIST Fine-Tuning**: After fine-tuning on PRIST, all segmentation models show significant improvements in CIoU and Precision (e.g., OMG-LLaVA CIoU ↑43%, LISA Precision ↑71%).
2. **Precision-Recall Trade-off**: After fine-tuning, the models prioritize improving segmentation specificity rather than generalization; while recall decreases, precision is substantially enhanced, which aligns with the objectives of pixel-level RS.
3. **Dual Capability of MIRAS**: Simultaneous optimization on both segmentation and dialog response leads to Dist-1/2 of 15.7/49.2, demonstrating the highest diversity in generated text.

## Highlights & Insights

- Defines a new task of pixel-level reasoning segmentation, filling the gap in multi-turn conversation-driven fine-grained segmentation.
- Constructs the PRIST dataset (24k dialogues, 8.3k scenes, 53% fine-grained targets), with an automated generation pipeline based on reasoning trees that is both highly efficient and quality-assured.
- The semantic region alignment strategy significantly improves the segmentation accuracy of the mask decoder by injecting target semantic information.

## Limitations & Future Work

- The dataset scale is relatively small (only 2,800 images), which may limit generalization capability.
- The absolute segmentation performance is still relatively low (maximal CIoU is only 14.72), leaving substantial room for improvement in pixel-level reasoning segmentation.
- Only training the mask decoder and projection layers while freezing MLLM and image encoders can be restrictive; end-to-end full-parameter fine-tuning could bring further improvements.
- The reasoning quality evaluation relies on GPT-4o as a referee, which may introduce evaluation bias.

## Related Work & Insights

- **Reasoning Segmentation Datasets**: ReasonSeg (Lai et al., 2023) first proposed a segmentation dataset based on complex queries, but it is small in scale and does not support multi-turn interactions. Subsequent works such as GREN (Yuan et al., 2024) extended this to multi-target scenarios but remain limited to single-turn reasoning.
- **Region-level Segmentation Models**: LISA integrates a segmentation module with LLMs to realize end-to-end training; PixelLM supports multi-target segmentation; OMG-LLaVA enhances regional understanding; however, all of these methods lack multi-turn reasoning capabilities.
- **Multimodal Large Language Models**: General MLLMs such as InternVL2 and Qwen2-VL possess strong visual perception but lack pixel-level segmentation capabilities.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Technical Depth | 7 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| **Overall** | **7.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](../../ICML2025/segmentation/actionpiece_contextually_tokenizing_action_sequences_for_generative_recommendati.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[CVPR 2025\] Audio-Visual Instance Segmentation](../../CVPR2025/segmentation/audio-visual_instance_segmentation.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](../../CVPR2025/segmentation/condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](../../CVPR2025/segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)

</div>

<!-- RELATED:END -->
