---
title: >-
  [Paper Note] FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity
description: >-
  [CVPR 2025][Segmentation][Region Captioning] FineCaption proposes a vision-language model supporting arbitrary mask referencing and high-resolution image inputs. By integrating a mask-aware CLIP encoder, ConvNeXT and SAM high-resolution encoders, along with the newly constructed CompositionCap dataset, it realizes the multi-granularity compositional region image captioning task.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Region Captioning"
  - "Vision-Language Models"
  - "Mask Referencing"
  - "High-Resolution Encoding"
  - "Compositional Attributes"
date: 2026-05-08
content_hash: 4f685bceac3a61d3
---

# FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity

**Conference**: CVPR 2025  
**arXiv**: [2411.15411](https://arxiv.org/abs/2411.15411)  
**Code**: [https://hanghuacs.github.io/FineCaption/](https://hanghuacs.github.io/FineCaption/)  
**Area**: Segmentation / Multimodal VLM  
**Keywords**: Region Captioning, Vision-Language Models, Mask Referencing, High-Resolution Encoding, Compositional Attributes

## TL;DR

FineCaption proposes a vision-language model supporting arbitrary mask referencing and high-resolution image inputs. By integrating a mask-aware CLIP encoder, ConvNeXT and SAM high-resolution encoders, along with the newly constructed CompositionCap dataset, it realizes the multi-granularity compositional region image captioning task.

## Background & Motivation

**Background**: Pre-trained vision-language models (VLMs) demonstrate outstanding performance in multimodal tasks, including image captioning and visual question answering. Recent studies have begun exploring region-level understanding, where Kosmos-2 and Shikra achieve region referencing via bounding box coordinates, and Ferret and ViP-LLaVA use overlaid visual prompts for free-form region referencing.

**Limitations of Prior Work**: Existing region referencing methods exhibit obvious drawbacks. The IoU accuracy of bounding boxes is insufficient (averaging only 56.11% IoU with masks), failing to precisely refer to irregularly shaped regions. Free-form visual prompts (such as circles and arrows) overlaid on images are easily misunderstood by VLMs as part of the image's semantic content, leading to referencing failure. Moreover, most models only handle resolutions from 224x224 to 448x448, failing to perceive fine-grained compositional attribute information (such as materials, textures, poses, etc.).

**Key Challenge**: Precise region referencing requires mask-level inputs, but the vision encoders of existing VLMs do not support mask inputs; meanwhile, descriptions of fine-grained compositional attributes require high-resolution perception, but large resolutions introduce huge computational overhead.

**Goal**: (1) How to enable VLMs to precisely recognize referred regions of arbitrary shapes? (2) How to support high-resolution image inputs while maintaining efficiency? (3) How to train the model to generate multi-granularity compositional region descriptions?

**Key Insight**: The authors tackle the problem from both the encoder architecture and the dataset. On the model side, they introduce the mask-aware encoding mechanism of Alpha-CLIP and supplement it with high-resolution encoders. On the data side, they build CompositionCap, a high-quality human-annotated dataset containing 18 compositional attributes.

**Core Idea**: By merging mask-aware low-resolution encoding and dual high-resolution encoding (ConvNeXT + SAM), along with a specialized compositional attribute dataset, precise region-referenced, multi-granularity image captioning is achieved.

## Method

### Overall Architecture

The inputs of FineCaption include a low-resolution image (336×336), a high-resolution image (1024×1024), and a binary mask. The low-resolution image and mask are processed by a mask-aware CLIP encoder, while the high-resolution image is encoded by ConvNeXT and SAM encoders separately to extract features. The three-way features undergo channel-wise concatenation and fusion, followed by adapter mapping, and are then fed into a Large Language Model (LLM) to generate text output. Training is divided into three stages: pre-training alignment, mask-image alignment pre-training, and full fine-tuning.

### Key Designs

1. **Mask-Aware Encoding**:

    - **Function**: Enables the CLIP vision encoder to perceive mask-referred regions.
    - **Mechanism**: Following the method of Alpha-CLIP, an additional convolutional layer $\text{Conv}_\alpha$ is added alongside the patch embedding layer of the CLIP encoder to process the binary mask. The mask embedding and patch embedding are added and then fed into the Transformer encoder of CLIP: $\mathbf{E}_{\text{seq}} = \text{Flatten}(\mathbf{E}_{\text{patch}} + \mathbf{E}_{\text{mask}})^\top$. This preserves the original image semantics while injecting region referencing information.
    - **Design Motivation**: Directly overlaying masks on images destroys image semantics. Encoding the mask independently and fusing them at the embedding layer preserves image integrity while achieving precise region referencing.

2. **Dual High-Resolution Encoding**:

    - **Function**: Extracts fine-grained spatial and semantic features from the 1024×1024 resolution image.
    - **Mechanism**: Both ConvNeXT and SAM encoders are used simultaneously as high-resolution encoders. ConvNeXT excels at capturing hierarchical visual features, while the SAM encoder is adept at capturing spatial structural information. They independently extract features $\mathbf{F}_{\text{HR1}}$ and $\mathbf{F}_{\text{HR2}}$ from the same high-resolution input, and the complementary feature representations enhance the model's perception of compositional attributes such as materials, textures, and shapes.
    - **Design Motivation**: A single encoder struggles to capture details across all dimensions simultaneously; the combination of two complementary encoders can cover more comprehensive visual information.

3. **Channel-wise Fusion & Adaptation**:

    - **Function**: Maps the three-way encoded features uniformly into the embedding space of the LLM.
    - **Mechanism**: After upsampling the mask-aware features to the same spatial dimension as the high-resolution features, the three-way features are concatenated along the channel dimension: $\mathbf{F}_{\text{fusion}} = [\mathbf{F}'_M; \mathbf{F}_{\text{HR1}}; \mathbf{F}_{\text{HR2}}]$, and then mapped to the word embedding space of the LLM through an adapter module.
    - **Design Motivation**: Channel-wise concatenation is the simplest and most effective way of merging multi-source features, preserving the complete information of each encoder.

### Loss & Training

Three-stage training strategy: Stage 1 freezes the encoders and the LLM, training only the projection layer for vision-language alignment (using LLaVA-Pretrain data). Stage 2 freezes other parts and trains only the alpha channel of the mask-aware encoder for mask-image alignment (using CompositionCap + GranD + RefCOCO datasets). Stage 3 performs full fine-tuning. The loss function is the standard autoregressive negative log-likelihood loss.

## Key Experimental Results

### Main Results

CompositionCap dataset statistics:
- 14,590 entities, 5,392 images, 186,490 attribute descriptions
- The test set is from Open Images: 1,000 images, 7,215 mask entities, 19,326 attribute descriptions
- 18 types of compositional attributes: category name, body size, skin texture, clothing accessories, interaction relations, pose, relative position, color, material, view angle, shape, expression, hairstyle, age, etc.

| Task | Model | ROUGE-L | BLEU-4 |
|------|------|---------|--------|
| Attribute-Aware Region Captioning | GPT-4o | - | - |
| Attribute-Aware Region Captioning | LLaMA-3.2 | - | - |
| Attribute-Aware Region Captioning | FineCaption | **Best** | **Best** |

### Ablation Study

| Configuration | Description | Effect |
|------|------|------|
| CLIP-only (336x336) | No mask, no high-resolution | Baseline |
| + Alpha-CLIP mask | Adds mask-aware encoding | Significantly improves region localization accuracy |
| + Single high-resolution encoder | ConvNeXT-only or SAM-only | Increases attribute perception capability |
| + Dual high-resolution encoders | ConvNeXT + SAM | Further improvements, especially in material and texture attributes |

### Key Findings
- Mask referencing has a clear advantage over bounding box referencing in region description accuracy, especially for irregularly shaped regions.
- High-resolution encoding is critical for the perception of fine-grained attributes such as color, material, and texture; the description quality of these attributes falls significantly at 336x336 resolution.
- The improvement of the dual encoders over the single encoder is most prominent under material and texture attributes, indicating that their features are indeed complementary.
- Existing strong models (including GPT-4o) still have substantial room for improvement in compositional attribute description.

## Highlights & Insights
- **The fine-grained attribute definition in the CompositionCap dataset** is an important contribution—it expands region captioning from the coarse-grained "what is this" to 18 dimensions of "detailed descriptions of all its attributes," which is a more practical task definition than traditional referring expressions.
- **The mask encoding method of Alpha-CLIP** is highly elegant—by using additive fusion at the embedding layer rather than overlaying on the image, it avoids the issue of visual prompts being misinterpreted as semantic content.
- The design of Stage 2 in the three-stage training, which is dedicated to mask-image alignment, is noteworthy—this is a "decoupled step-by-step" training approach that prevents multiple objectives from interfering with each other in the early stages.

## Limitations & Future Work
- The inference overhead of three encoders is relatively large; under the 1024×1024 resolution, the computational costs of SAM and ConvNeXT are non-negligible.
- Masks must be provided externally (e.g., through interactive segmentation tools), and the model itself cannot automatically generate masks, which limits end-to-end applications.
- The scale of the CompositionCap dataset is relatively small (5K training images), which may be insufficient for generalizing to long-tail scenarios.
- The paper does not analyze the differences in model performance across different attribute categories in depth, nor does it explore conflicts when jointly describing multiple attributes.
- No fair comparison is conducted against the latest mask-based VLMs (such as OMG-LLaVA, RegionGPT) under a completely aligned setting.

## Related Work & Insights
- **vs Alpha-CLIP**: Alpha-CLIP proposes the core technology of mask-aware encoding. On this basis, this paper adds high-resolution encoding and a compositional attribute dataset, shifting the objective from "referencing a region" to "describing the region's attributes in detail."
- **vs Osprey**: Osprey also supports mask referencing and dense region description but is limited to low resolution. This paper enhances fine-grained perception through high-resolution encoders.
- **vs GLaMM**: GLaMM supports mask referencing and region attribute description but lacks high-resolution input; this paper achieves finer attribute perception through 1024×1024 encoding.

## Rating
- Novelty: ⭐⭐⭐ Although individual components (mask encoding, high-resolution, dataset) are not entirely novel on their own, their combination addresses a practical problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ The dataset construction is solid, and the multi-baseline comparison is thorough, though it lacks more quantitative ablations.
- Writing Quality: ⭐⭐⭐⭐ The task definition is clear, and the method description is comprehensive.
- Value: ⭐⭐⭐⭐ The CompositionCap dataset and the multi-granularity region description task definition possess significant research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Panoptic Captioning: An Equivalence Bridge for Image and Text](../../NeurIPS2025/segmentation/panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)
- [\[CVPR 2025\] Segment Any Motion in Videos](segment_any_motion_in_videos.md)
- [\[CVPR 2025\] SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)
- [\[CVPR 2025\] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [\[CVPR 2025\] Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)

</div>

<!-- RELATED:END -->
