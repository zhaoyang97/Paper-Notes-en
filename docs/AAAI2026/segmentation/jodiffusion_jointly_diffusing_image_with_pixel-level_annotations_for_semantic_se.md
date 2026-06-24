---
title: >-
  [Paper Note] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion
description: >-
  [AAAI 2026][Segmentation][Semantic Segmentation] This paper proposes JoDiffusion, a framework that jointly diffuses images and pixel-level annotation masks in latent space, enabling—for the first time—simultaneous generation of semantically consistent image–annotation pairs conditioned solely on text prompts. JoDiffusion substantially outperforms existing Image2Mask and Mask2Image methods on Pascal VOC, COCO, and ADE20K.
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Semantic Segmentation"
  - "Dataset Generation"
  - "Diffusion Models"
  - "Joint Generation"
  - "Annotation Masks"
  - "Latent Space"
date: 2026-05-08
content_hash: d7d7e03b616b1a51
---

# JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion

**Conference**: AAAI 2026
**arXiv**: [2512.13014](https://arxiv.org/abs/2512.13014)  
**Authors**: Haoyu Wang, Lei Zhang (Corresponding), Wenrui Liu, Dengyang Jiang, Wei Wei (Northwestern Polytechnical University), Chen Ding
**Code**: [GitHub](https://github.com/00why00/JoDiffusion)  
**Area**: Image Segmentation
**Keywords**: Semantic Segmentation, Dataset Generation, Diffusion Models, Joint Generation, Annotation Masks, Latent Space

## TL;DR

This paper proposes JoDiffusion, a framework that jointly diffuses images and pixel-level annotation masks in latent space, enabling—for the first time—simultaneous generation of semantically consistent image–annotation pairs conditioned solely on text prompts. JoDiffusion substantially outperforms existing Image2Mask and Mask2Image methods on Pascal VOC, COCO, and ADE20K.

## Background & Motivation

### State of the Field
Semantic segmentation relies on large-scale, high-quality image–pixel-level annotation pairs for training, yet manual annotation is extremely costly, especially in multi-object interaction or dense small-object scenarios. Generating synthetic datasets via diffusion models has emerged as a promising approach to alleviate the annotation bottleneck.

### Limitations of Prior Work
Existing methods fall into two pipeline categories, each with critical drawbacks:

- **Image2Mask (generate image first, then infer annotations)**: Methods such as DiffuMask, Dataset Diffusion, and SDS generate images from text and subsequently infer pseudo-annotations via cross-attention or similar mechanisms. Errors in text–image similarity computation and insufficient spatial resolution of feature maps lead to **semantic inconsistency between images and annotations**.
- **Mask2Image (provide annotations first, then generate images)**: Methods such as FreeMask and SegGen generate images conditioned on manually annotated masks. Although semantic consistency is better, **scalability is limited** by the available manual annotations, and the diversity of generated images is constrained by a finite set of annotation templates.

### Root Cause
The fundamental question is whether a unified model can **simultaneously** generate semantically consistent images and pixel-level annotations using only text prompts—thereby guaranteeing semantic consistency without being constrained by the number of manual annotations, and resolving the shortcomings of both pipeline types.

## Method

JoDiffusion comprises three stages: Annotation VAE training, joint diffusion modeling, and mask optimization.

### Stage 1: Annotation VAE

To map annotation masks into a latent space shared with images, a dedicated Annotation VAE is designed:

- **Input representation**: Per-pixel class indices are converted to binary encoding $M_{\text{bin}}$, avoiding the difficulty of distinguishing adjacent class values that are numerically close.
- **Lightweight architecture**: Encoder $E_M$ and decoder $D_M$ use only a small number of convolutional/transposed-convolutional layers, with approximately 50M parameters (compared to 300M for the image VAE).
- **Training objective**: Trained with cross-entropy loss only, without KL divergence regularization (since the VAE serves as a compression tool rather than a generative model):

$$\mathcal{L}_{\text{Annotation VAE}} = -\sum_{(i,j)}\sum_{c=0}^{N_C} M_{\text{one-hot},(i,j,c)} \log \bar{M}_{(i,j,c)}$$

- **Reconstruction quality**: mIoU exceeds 98% on all three datasets, confirming high-fidelity encoding.

### Stage 2: Joint Diffusion Modeling

Building on the UniDiffuser architecture, text, images, and annotation masks are jointly diffused and denoised in latent space:

1. **Encoding**: BLIP-2 generates caption text $T$ for each image; $z_T, z_I$ are obtained via CLIP text/image encoders and the image VAE, respectively; $z_M$ is obtained via the Annotation VAE.
2. **Forward diffusion**: **Shared noise** $\epsilon_{IM}$ is applied to both $z_I$ and $z_M$, ensuring structural consistency throughout the diffusion process:

$$q(z_I^t, z_M^t | z_I^0, z_M^0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\begin{bmatrix}z_I^0 \\ z_M^0\end{bmatrix}, (1-\bar{\alpha}_t)I\right)$$

3. **Joint denoising**: The network $\epsilon_\theta(z_I^t, z_M^t, z_T, t)$ learns to predict joint noise rather than estimating noise for each modality independently.
4. **Training loss**: Standard MSE denoising loss:

$$\mathcal{L}_{\text{denoising}} = \mathbb{E}_{t, z_I^0, z_M^0, \epsilon}\left[\|\epsilon_\theta(z_I^t, z_M^t, z_T, t) - \epsilon_{IM}\|^2\right]$$

**Key design**: Self-attention (rather than cross-attention) is used to concatenate text and image features, providing greater flexibility for fine-tuning. At inference, only a text prompt is required to simultaneously generate an image and its annotation mask.

### Stage 3: Mask Optimization

The diffusion process inevitably introduces annotation noise at small regions and object boundaries. A boundary-majority-based post-processing strategy is proposed:

1. Identify small regions $R$ whose area falls below threshold $\tau$.
2. Extract the boundary pixel set $\hat{R}$.
3. Compute the most frequent class among boundary pixels: $c^* = \arg\max_c \sum_{(i,j)\in\hat{R}} \mathbb{I}(x_{i,j}=c)$.
4. Reassign all pixels in region $R$ to class $c^*$.

This strategy exploits the prior that semantic regions in natural images are spatially continuous, and is statistically equivalent to maximum likelihood estimation of the true class for the region.

## Key Experimental Results

### Table 1: Comparison with Image2Mask Methods (Pascal VOC & MS-COCO)

| Segmentor | Backbone | Method | VOC Size | VOC mIoU (Syn) | VOC mIoU (Real+Syn) | COCO Size | COCO mIoU (Syn) | COCO mIoU (Real+Syn) |
|-----------|----------|--------|----------|----------------|---------------------|-----------|-----------------|----------------------|
| DeepLabV3 | ResNet50 | Raw Dataset | 11.5k | 77.4 | - | 118k | 48.9 | - |
| DeepLabV3 | ResNet50 | SDS | 26k | 60.4 | 77.6 | 50k | 31.0 | 50.3 |
| DeepLabV3 | ResNet50 | Dataset Diffusion | 40k | 61.6 | 77.6 | 80k | 32.4 | 54.6 |
| DeepLabV3 | ResNet50 | **JoDiffusion** | 40k | **72.5** | **78.3** | 80k | **42.6** | **56.4** |
| DeepLabV3 | ResNet101 | Raw Dataset | 11.5k | 79.9 | - | 118k | 54.9 | - |
| DeepLabV3 | ResNet101 | SDS | 26k | 59.1 | 79.8 | 50k | 31.8 | 56.8 |
| DeepLabV3 | ResNet101 | Dataset Diffusion | 40k | 64.8 | 80.3 | 80k | 34.2 | 57.4 |
| DeepLabV3 | ResNet101 | **JoDiffusion** | 40k | **75.8** | **80.7** | 80k | **44.9** | **59.1** |
| Mask2Former | ResNet50 | Raw Dataset | 11.5k | 77.3 | - | 118k | 57.8 | - |
| Mask2Former | ResNet50 | DiffuMask | 60k | 57.4 | 77.5 | - | - | - |
| Mask2Former | ResNet50 | SDS | 26k | 59.8 | 78.1 | 50k | 29.8 | 57.7 |
| Mask2Former | ResNet50 | Dataset Diffusion | 40k | 60.2 | 78.2 | 80k | 31.0 | 57.8 |
| Mask2Former | ResNet50 | **JoDiffusion** | 40k | **74.5** | **79.4** | 80k | **44.6** | **58.5** |

JoDiffusion achieves substantial gains when trained on synthetic data alone (Syn): approximately 10–14 mIoU points above the second-best method on VOC and 10–13 points on COCO. Consistent improvements are also observed in the Real+Syn mixed training setting.

### Table 2: Comparison with Mask2Image Methods (Pascal VOC & ADE20K)

| Backbone | Method | VOC Size | VOC mIoU | ADE20K Size | ADE20K mIoU |
|----------|--------|----------|----------|-------------|-------------|
| ResNet50 | Raw Data | 11.5k | 77.3 | 20k | 47.2 |
| ResNet50 | SegGen | - | - | 11M | 49.9 |
| ResNet50 | FreeMask | 40k | 77.9 | 40k | 48.2 |
| ResNet50 | **JoDiffusion** | 40k | **79.4** | 40k | **48.4** |
| Swin-S | Raw Data | 11.5k | 83.8 | 20k | 51.6 |
| Swin-S | FreeMask | 40k | 84.2 | 40k | 52.1 |
| Swin-S | **JoDiffusion** | 40k | **85.1** | 40k | **52.2** |

Without requiring manually annotated masks as input, JoDiffusion consistently outperforms Mask2Image methods.

### Ablation Study
- **Mask optimization threshold**: $\tau=20$ yields the best performance (72.47 mIoU), a gain of 1.1 points over no optimization ($\tau=0$, 71.37).
- **Synthetic data volume**: Increasing the data volume from 5k to 40k yields continuous mIoU improvements (68.54→72.47).

## Highlights & Insights

- **First joint image–annotation generation**: Unlike two-step pipelines that either generate images then infer annotations or require annotations to generate images, JoDiffusion produces semantically consistent image–annotation pairs simultaneously from text prompts alone, combining semantic consistency with scalability.
- **Elegant Annotation VAE design**: Binary encoding combined with a lightweight architecture (50M parameters) achieves reconstruction mIoU >98%, effectively mapping discrete class maps into a continuous latent space.
- **Shared noise mechanism**: Applying identical noise to both image and annotation latents enforces structural consistency at the diffusion process level, rather than relying on post-hoc alignment.
- **Significant performance gains**: Under synthetic-only training, JoDiffusion surpasses Dataset Diffusion by more than 10 mIoU points on VOC, demonstrating the fundamental advantage of joint generation for semantic consistency.

## Limitations & Future Work

- **Reliance on pre-trained text descriptions**: BLIP-2-generated captions are required for training images; caption quality directly affects generation performance.
- **Fixed-resolution training**: All images and annotations are resized to 512×512, limiting support for high-resolution fine-grained segmentation.
- **Simple mask optimization strategy**: The boundary-majority post-processing only addresses noise in small regions and is ineffective for large-area semantic confusion.
- **Limited dataset coverage**: Evaluation is conducted only on VOC (21 classes), COCO (81 classes), and ADE20K (150 classes), without covering finer-grained or domain-specific datasets.
- **Scalability ceiling unexplored**: At most 40k–80k synthetic samples are generated; quality trends at larger scales remain uninvestigated.
- **Marginal advantage over Mask2Image methods**: In the Real+Syn setting and on ADE20K, improvements over FreeMask are modest (<1 mIoU).

## Related Work & Insights

- **DiffuMask**: Infers annotations via cross-attention, yielding poor semantic consistency (VOC Syn 57.4 mIoU); JoDiffusion's joint generation avoids this issue.
- **Dataset Diffusion**: Incorporates LLM-generated diverse text and self-attention map refinement, but remains limited by feature map resolution (VOC Syn 60–65 mIoU); JoDiffusion exceeds it by more than 10 points.
- **SDS**: Introduces CLIP similarity filtering and class-balanced sampling, but is fundamentally still a two-step pipeline; JoDiffusion leads by 12+ points on VOC Syn.
- **FreeMask**: A representative Mask2Image method with good semantic consistency but constrained by the annotation library size; JoDiffusion achieves comparable or superior results without requiring manual annotations.
- **SegGen**: Trains an additional text-to-mask model to enhance diversity, yet requires 11M samples to reach 49.9 mIoU on ADE20K; JoDiffusion achieves 48.4 mIoU with only 40k samples.
- **UniDiffuser**: The diffusion architecture backbone of JoDiffusion; while UniDiffuser handles text–image bimodal modeling, JoDiffusion extends this to three-modal joint modeling of text, images, and annotations.

## Rating

- Novelty: ⭐⭐⭐⭐ — First proposal of joint image–annotation diffusion generation; the idea is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmark datasets, multiple backbone networks, comparisons against both method categories, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — The three-stage framework is described clearly, with complete mathematical derivations and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Proposes a unified paradigm for synthetic data generation in semantic segmentation with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[AAAI 2026\] From Attribution to Action: Jointly ALIGNing Predictions and Explanations](from_attribution_to_action_jointly_aligning_predictions_and_explanations.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](../../CVPR2025/segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[CVPR 2026\] Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding](../../CVPR2026/segmentation/joint_spectral_image_reconstruction_and_semantic_segmentation_with_cooperative_u.md)

</div>

<!-- RELATED:END -->
