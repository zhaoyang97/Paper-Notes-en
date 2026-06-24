---
title: >-
  [Paper Note] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output
description: >-
  [CVPR 2025][Multimodal VLM][Medical Visual Question Answering] This paper proposes MIMO, the first medical vision-language model that simultaneously supports "visual referring multimodal input" (users specify regions of interest via points/boxes) and "pixel-level grounding multimodal output" (the model embeds segmentation masks within textual answers). It constructs the MIMOSeg dataset with 895K samples and demonstrates unique referring + grounding capabilities across various…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Medical Visual Question Answering"
  - "Visual Referring"
  - "Pixel-level Grounding"
  - "Multimodal Input/Output"
  - "Medical Image Segmentation"
date: 2026-05-08
content_hash: 793f8e689d889ce0
---

# MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output

**Conference**: CVPR 2025  
**arXiv**: [2510.10011](https://arxiv.org/abs/2510.10011)  
**Code**: [https://github.com/pkusixspace/MIMO](https://github.com/pkusixspace/MIMO)  
**Area**: Multimodal VLM  
**Keywords**: Medical Visual Question Answering, Visual Referring, Pixel-level Grounding, Multimodal Input/Output, Medical Image Segmentation

## TL;DR

This paper proposes MIMO, the first medical vision-language model that simultaneously supports "visual referring multimodal input" (users specify regions of interest via points/boxes) and "pixel-level grounding multimodal output" (the model embeds segmentation masks within textual answers). It constructs the MIMOSeg dataset with 895K samples and demonstrates unique referring + grounding capabilities across various medical VQA and segmentation tasks.

## Background & Motivation

Current Medical Vision-Language Models (MVLMs) primarily rely on text-only instruction inputs and text-only answer outputs in VQA. $\rightarrow$ Limitations of Prior Work 1 (Input side): Precisely describing regions of interest in medical images is highly difficult. Terms like "hilar region of the middle and lower lobes of the right lung" are unfriendly to non-expert users, and individual differences in imaging presentations exist across different patients. $\rightarrow$ Limitations of Prior Work 2 (Output side): Text-only answers lack association with critical image regions, preventing a "show and tell" communication style typical among doctors. $\rightarrow$ Key Challenge: Lack of a unified model architecture that simultaneously supports visual referring input and pixel grounding output, alongside a shortage of corresponding medical multimodal training data. $\rightarrow$ Key Insight: Design the MIMO architecture (incorporating Multi-modal Input Aligner + grounding tokens + SAM decoder) and construct the MIMOSeg dataset (895K samples across 4 perspectives) to enrich the multimodal capabilities of medical VLMs from both input and output ends.

## Method

### Overall Architecture

MIMO consists of 6 core modules: (1) A CLIP ViT-H/14 image encoder that generates image embeddings $X_{img} \in \mathbb{R}^{l_1 \times d}$; (2) A visual prompt encoder that encodes points/boxes into $X_q^v \in \mathbb{R}^{l_3 \times d}$; (3) A Multi-modal Input Aligner that fuses multimodal inputs via cross-attention; (4) A Vicuna-7B LLM that generates textual responses containing special tokens; (5) A SAM segmentation encoder that extracts image segmentation features; and (6) A SAM mask decoder that generates segmentation masks based on the embedding of the `<SEG>` token.

### Key Designs

1. **Multi-modal Input Aligner**:
    - **Function**: Extract instruction-guided visual information from multimodal inputs.
    - **Mechanism**: Use a learnable query embedding $X_q$ to interact with $(X_{img}, X_q^t, X_q^v)$ via cross-attention to extract image features focused on by both textual and visual prompts, which are then linearly projected and fed into the LLM.
    - **Design Motivation**: Directly concatenating multimodal features fails to effectively align visual prompts with image regions; cross-attention allows the model to selectively focus on regions pointed to by visual prompts. Ablation studies show that removing it significantly degrades performance in Perspectives II/IV that contain visual prompts (mIoU drops by 0.04 to 0.06).

2. **Grounding Token Mechanism (Mapping from Language to Pixels)**:
    - **Function**: Mark groundable medical entities in the LLM text output and associate them with segmentation masks.
    - **Mechanism**: Introduce three special tokens in the vocabulary: `<p>`, `</p>`, and `<SEG>`. `<p>...</p>` wraps groundable entities, followed closely by `<SEG>` to indicate segmentation is required. Extract the final hidden state $r_{seg}$ of the LLM corresponding to `<SEG>`, and feed it through a projection layer into the SAM mask decoder: $\mathcal{M} = \mathcal{V}(\mathcal{G}(I), \text{proj}(r_{seg}))$.
    - **Design Motivation**: Drawing inspiration from LISA/GLaMM to seamlessly link LLM semantic understanding with pixel-level segmentation, enabling each medical entity in the textual response to be "grounded" in image regions.

3. **MIMOSeg Dataset Construction (895K samples, 4 perspectives)**:
    - **Function**: Provide large-scale training data covering 8 imaging modalities (CT, X-ray, fundus, pathology, etc.).
    - **Mechanism**: Design 4 perspectives of progressive difficulty—I: Language-guided segmentation (text $\rightarrow$ mask); II: Visual-prompt-aware perception (text+visual $\rightarrow$ mask+entity identification); III: Complex reasoning QA with segmentation; IV: Visual-prompt-assisted reasoning + segmentation. Utilize knowledge base retrieval (Wikipedia/UMLS) + GPT-4o to generate complex scenario QAs.
    - **Design Motivation**: There is no off-the-shelf dataset in the medical domain containing both referring and grounding annotations; the four perspectives cover all interactive scenarios from basic to complex.

### Loss & Training

Total loss $L_{total} = \lambda_1 L_{text} + \lambda_2 L_{bce} + \lambda_3 L_{dice}$:
- $L_{text}$: Cross-entropy loss for text generation.
- $L_{bce}$: Pixel-level binary cross-entropy loss (segmentation mask supervision).
- $L_{dice}$: Dice loss (to handle foreground/background imbalance).
- Training data mixture ratio: 4 perspectives + LLaVA-Med VQA = 1:2:2:1:1, trained for 3 epochs.

## Key Experimental Results

### Main Results (Held-in: MIMOSeg Test Set, Segmentation Performance)

| Model | Perspective I (mIoU) | Perspective II (mIoU) | Perspective III (mIoU) | Perspective IV (mIoU) |
|------|------------|-------------|--------------|-------------|
| SAM-h | × | 0.571 | × | 0.583 |
| GLaMM | 0.556 | 0.496 | 0.421 | 0.564 |
| MIMO(w/o Aligner) | 0.607 | 0.622 | 0.468 | 0.526 |
| **MIMO** | **0.639** | **0.665** | **0.531** | **0.586** |

### Held-out Zero-shot Segmentation Experiments

| Dataset | Condition | MIMO (AP50) | GLaMM (AP50) | SAM-h (AP50) |
|--------|------|-----------|-------------|-------------|
| X-ray | w/o bbox | **0.507** | 0.335 | × |
| X-ray | with bbox | **0.989** | 0.859 | 0.989 |
| Fundus | with bbox | **0.988** | 0.946 | 0.973 |
| Skin Lesion | w/o bbox | **0.787** | 0.458 | × |
| Skin Lesion | with bbox | **0.985** | 0.719 | 0.982 |

### Ablation Study

| Configuration | Perspective II (mIoU) | Perspective IV (mIoU) | Description |
|------|-------------|-------------|------|
| MIMO | **0.665** | **0.586** | Full model |
| MIMO(w/o Aligner) | 0.622 | 0.526 | Remove Aligner, performance in visual prompt scenarios drops significantly |

| Training Data Ratio | mIoU Trend | AP50 Trend | F1 Trend |
|-------------|---------|---------|--------|
| 25%→50%→75%→100% | Steady improvement | Steady improvement | Steady improvement |

### Key Findings

- MIMO is the only medical VLM capable of simultaneous visual referring + pixel grounding.
- Under the text-only instruction segmentation (w/o bbox) scenario, MIMO significantly outperforms GLaMM, indicating stronger text instruction following capabilities.
- Held-out VQA average accuracy is 56.1%, which falls short of HuatuoGPT-Vision (66.1%), but the latter utilized significantly more VQA training data and does not support segmentation.
- The Multi-modal Input Aligner brings consistent gains in scenarios containing visual prompts.

## Highlights & Insights

- **Novelty**: First unified model in medical VLMs to simultaneously achieve visual referring input and pixel grounding output.
- **Data Construction Methodology**: 4 perspectives cover interactive scenarios from basic to complex. The knowledge-base + GPT-4o generation workflow is highly reusable.
- **High Practicality**: The seamless combination of text + segmentation outputs (e.g., `<p>smooth muscle<SEG></p>`) closely aligns with the actual clinical workflow of physicians.
- **Open Source**: Both the model and datasets have been open-sourced.

## Limitations & Future Work

- Held-out VQA performance is inferior to HuatuoGPT-Vision; general VQA ability is compromised by the multi-task segmentation objectives.
- QA generation relies on predefined knowledge bases, offering limited knowledge coverage.
- Only a 7B Vicuna is used as the LLM backbone; larger models would likely yield better performance.
- Visual prompts only support points and boxes, lacking support for free-form scribbles (e.g., like Ferret's free-form input).
- Segmentation masks only pair with individual entities, lacking support for complex multi-entity relationship modeling.

## Related Work & Insights

- **GLaMM**: A general-domain referring + grounding multimodal model. MIMO transfers this concept to medical scenarios and introduces the Multi-modal Input Aligner.
- **LISA**: A reasoning segmentation model connecting LLM and SAM using the `<SEG>` token; MIMO draws inspiration from this core design.
- **Ferret**: A general VLM supporting free-form referring, which inspired MIMO's visual prompt design.
- **Insights**: The richness of "input-output modalities" in medical VLMs is a crucial yet overlooked direction; the methodology for building multi-perspective datasets is highly generalizable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to pioneer a unified referring + grounding model in the medical field, though technical components are primarily combinations of existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers held-in/held-out and multi-dimensional evaluations for both segmentation and VQA, with reasonably comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear design logic for the 4 perspectives with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Offers direct utility for clinical medical AI, with a significant dataset contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[NeurIPS 2025\] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](../../NeurIPS2025/multimodal_vlm/bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)

</div>

<!-- RELATED:END -->
