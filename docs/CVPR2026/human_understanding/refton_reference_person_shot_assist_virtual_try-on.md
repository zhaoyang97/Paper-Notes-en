---
title: >-
  [Paper Note] RefTon: Reference Person Shot Assist Virtual Try-on
description: >-
  [CVPR 2026][Human Understanding][virtual try-on] This paper proposes RefTon, a person-to-person virtual try-on framework built on Flux-Kontext. By incorporating an additional reference image — a photo of another person wearing the target garment — RefTon provides richer garment detail information. Combined with a two-stage training strategy and a rescaled position index mechanism, the framework achieves end-to-end try-on without auxiliary conditions (e.g., DensePose, segmentation masks), attaining state-of-the-art performance on VITON-HD and DressCode.
tags:
  - CVPR 2026
  - Human Understanding
  - virtual try-on
  - reference image guidance
  - Flux-Kontext
  - mask-free try-on
  - diffusion models
date: 2026-05-08
content_hash: 281f1c42a73acc8e
---

# RefTon: Reference Person Shot Assist Virtual Try-on

**Conference**: CVPR 2026
**arXiv**: [2511.00956](https://arxiv.org/abs/2511.00956)
**Code**: [https://github.com/360CVGroup/RefTon](https://github.com/360CVGroup/RefTon)
**Area**: Human Understanding
**Keywords**: virtual try-on, reference image guidance, Flux-Kontext, mask-free try-on, diffusion models

## TL;DR
This paper proposes RefTon, a person-to-person virtual try-on framework built on Flux-Kontext. By incorporating an additional reference image — a photo of another person wearing the target garment — RefTon provides richer garment detail information. Combined with a two-stage training strategy and a rescaled position index mechanism, the framework achieves end-to-end try-on without auxiliary conditions (e.g., DensePose, segmentation masks), attaining state-of-the-art performance on VITON-HD and DressCode.

## Background & Motivation
1. **State of the Field**: Virtual try-on (ViTON) has evolved from GAN-based methods to diffusion model-based approaches, with significant improvements in garment warping and texture fidelity.
2. **Limitations of Prior Work**: (a) Many methods rely on complex external models — pose estimators, human parsers, segmentation models — to handle diverse conditional inputs, increasing framework complexity and making final results sensitive to mask quality. (b) More critically, a flat garment image alone cannot fully convey the style, texture, and design details of clothing — for instance, it is difficult to distinguish whether a garment is made of green translucent fabric or light green opaque fabric, or to identify lace collar designs.
3. **Root Cause**: In real-world shopping scenarios, users are more interested in how a garment looks when worn by a model than in flat lay images. However, existing methods do not support a "reference model image" as an additional input, partly due to the lack of such paired data in public datasets.
4. **Paper Goals**: (a) Eliminate dependency on external models and auxiliary conditions; (b) introduce reference images to more accurately convey the appearance of garments when worn; (c) construct training data that includes reference images.
5. **Starting Point**: Leveraging the powerful image editing capability of Flux-Kontext to automatically synthesize reference images showing the same garment worn by different people, thereby constructing a training dataset. Position encoding is also improved to support multi-condition, multi-resolution inputs.
6. **Core Idea**: By using reference images — photos of other individuals wearing the target garment — RefTon provides more intuitive visual guidance for virtual try-on. Combined with mask-free two-stage training and rescaled position indexing, the framework achieves a clean and efficient end-to-end try-on pipeline.

## Method

### Overall Architecture
RefTon is built on the Flux-Kontext backbone. Inputs include the source person image (or agnostic image), the target garment image, and an optional reference image. These images are encoded into latent representations via a VAE, concatenated into a sequence, and denoised by a DiT (Diffusion Transformer) to generate the target image. Training proceeds in two stages: Stage 1 trains a mask-based try-on model using synthesized unpaired data; Stage 2 trains a mask-free person-to-person model.

### Key Designs

1. **Two-Stage Training Strategy**:

    - **Function**: Enables direct person-to-person try-on using only paired data.
    - **Mechanism**: Existing datasets provide only paired samples $[\mathbf{c}_i, \mathbf{p}_{i,\mathbf{c}_i}]$ (garment + person wearing it), whereas training a person-to-person model requires unpaired triplets $[\bar{\mathbf{p}}_{i,\mathbf{c}_j}, \mathbf{c}_i, \mathbf{p}_{i,\mathbf{c}_i}]$ (person wearing a different garment + target garment + result). Stage 1 trains a mask-based try-on model with rich conditions including agnostic images, DensePose, and warp masks, then uses this model to synthesize images of each person wearing different garments. Stage 2 trains a person-to-person model on these synthesized unpaired images, randomly using either agnostic images or synthesized person images as input with 50% probability.
    - **Design Motivation**: This strategy is similar to CatVTON but enriches the conditional inputs to improve the quality of synthesized images. The key insight is that Stage 1 must generate sufficiently high-quality unpaired data to support Stage 2 training.

2. **Rescaled Position Index**:

    - **Function**: Enables unified handling of multi-type, multi-resolution conditional inputs.
    - **Mechanism**: The original Flux-Kontext position index has three channels — the first channel uses a binary flag to distinguish noisy from conditioning images, while the remaining two encode spatial coordinates. RefTon extends the first channel to discrete condition labels (distinguishing inputs such as person, garment, and reference). Position indices are generated independently for each condition, and spatial coordinates are rescaled by the resolution ratio between the target and conditioning images to maintain spatial alignment across resolutions.
    - **Design Motivation**: The original binary design cannot distinguish multiple heterogeneous conditional inputs. Generating indices independently per condition is more flexible than concatenating inputs on a pixel-space canvas (as in Any2AnyTryon) and supports an arbitrary number of conditions at varying resolutions. Ablation experiments confirm that rescaled position indexing outperforms the original scheme on both FID and KID.

3. **Reference Image Guidance Mechanism**:

    - **Function**: Transmits visual information about garment appearance through an additional reference image.
    - **Mechanism**: During training, a reference image $\mathbf{r}_i$ (a photo of another person wearing the target garment) is provided with 25% probability as an additional condition alongside the person and garment images. The reference image is integrated via its own independent position index. At inference time, the reference image is optional.
    - **Design Motivation**: Flat garment images cannot convey transparent fabrics, lace details, or the interaction between garment and body. Reference images bridge this information gap. Experiments show that incorporating reference images improves performance across all metrics (e.g., VITON-HD FID drops from 5.45 to 4.69).

4. **Reference Image Data Generation Pipeline**:

    - **Function**: Automatically constructs a training dataset containing reference images.
    - **Mechanism**: Qwen2.5-VL is used to describe the appearance of the person in the target image and generate an "opposite description" (different skin tone, hairstyle, etc.). Flux-Kontext then edits the target image using the opposite description as a positive prompt and the original description as a negative prompt, producing a reference image that preserves the garment while altering the person's appearance. Non-target garments and poses are also randomly sampled from a description pool to increase diversity.
    - **Design Motivation**: Three constraints ensure reference image quality — (i) faithful preservation of the target garment, (ii) a person appearance distinct from the target (to prevent the model from taking shortcuts by copying directly), and (iii) different non-target garments (to increase diversity). CLIP-based deduplication and VLM quality filtering further ensure data quality.

### Loss & Training
Standard flow matching loss is used for training. The VAE encoder and decoder of Flux-Kontext are frozen; only the Transformer blocks are fine-tuned using LoRA (rank=64, $\alpha=128$). Single-dataset experiments: 20k steps on VITON-HD and 48k steps on DressCode, with batch size 128 on 8×H100 GPUs. Mixed-dataset (VFR) training is also conducted to improve generalization.

## Key Experimental Results

### Main Results (VITON-HD + DressCode)

| Method | Input Conditions | VITON-HD LPIPS↓ | SSIM↑ | FID↓ (paired) | FID↓ (unpaired) |
|--------|-----------------|----------------|-------|--------------|----------------|
| CatVTON | Mask | 0.057 | 0.870 | 5.43 | 9.02 |
| IDM-VTON | Mask+Pose | 0.102 | 0.870 | 6.29 | — |
| **RefTon** | **Mask** | **0.057** | 0.873 | 5.45 | 8.58 |
| **RefTon+R** | **Mask+Ref** | **0.049** | **0.879** | **4.69** | **8.43** |
| RefTon/MF | Mask-free | 0.061 | 0.866 | 5.98 | 8.40 |
| RefTon+R/MF | Mask-free+Ref | 0.053 | 0.872 | 5.11 | **8.32** |

### Ablation Study

| Setting | VITON-HD FID↓ | DressCode FID↓ | Notes |
|---------|--------------|---------------|-------|
| w/ mask, w/o reference | 5.45 | 3.48 | Baseline |
| w/ mask, w/ reference | **4.69** | **2.94** | Reference significantly improves results |
| w/o mask, w/o reference | 5.98 | 3.84 | Slight drop without mask |
| w/o mask, w/ reference | 5.11 | 3.34 | Reference compensates for missing mask |
| Original position index (0.5×) | 5.29 | — | No rescaling |
| Rescaled position index (0.5×) | **5.09** | — | Improved with rescaling |

### Key Findings
- **Reference images consistently improve all metrics**: Under the masked setting, adding reference images reduces VITON-HD paired FID from 5.45 to 4.69 (↓14%) and LPIPS from 0.057 to 0.049 (↓14%). On DressCode, FID drops from 3.48 to 2.94 (↓15%).
- **Mask-free mode remains competitive**: Even without agnostic masks, performance is on par with or superior to mask-dependent baselines (FID 8.40 vs. CatVTON 9.02), demonstrating practical deployment convenience.
- **Cross-dataset generalization**: Models trained on the mixed VFR dataset surpass baselines such as OOTDiffusion without dedicated training on VITON-HD or DressCode.
- **Cross-domain evaluation on StreetTryOn**: State-of-the-art FID is achieved on the StreetTryOn dataset, which was never seen during training, demonstrating strong generalization capability.
- **Mask quality issues**: Ablation visualizations show that overly cropped masks discard items carried by the person (e.g., handbags), while conservative masks retain unwanted regions. The mask-free mode avoids both issues.

## Highlights & Insights
- **Reference image design grounded in real shopping behavior**: Real users in online shopping genuinely pay more attention to how a garment looks on a model than to flat lay images. Reference images capture information that flat garments cannot convey — transparent materials, lace details, and fabric drape. This design intuition is precise and well-motivated.
- **The reference data generation pipeline is cleverly designed**: By leveraging a VLM to automatically generate appearance descriptions and their opposites, and using Flux-Kontext for editing, the pipeline produces reference images that preserve the target garment while varying person appearance and other clothing. The three constraints (garment fidelity, distinct person, diverse garments) effectively prevent shortcut learning during training.
- **Unified framework for multiple input modes**: A single model supports four combinations of masked/mask-free × with/without reference, elegantly handled through condition labels and probability-based sampling.

## Limitations & Future Work
- Reference image generation relies on the editing quality of Flux-Kontext; if the editing model performs poorly on certain garment types, reference image quality will degrade accordingly.
- Evaluation is limited to static image try-on; extension to video try-on has not been explored.
- The expressive capacity of LoRA fine-tuning may be limited; whether full-parameter fine-tuning or higher LoRA rank could yield further improvements remains to be investigated.
- Reference images appear with only 25% probability during training; whether a more optimal sampling strategy exists has not been thoroughly studied.
- Fusion of multi-view reference images is not considered.

## Related Work & Insights
- **vs. CatVTON**: Both employ two-stage training and mask-free designs, but CatVTON lacks a reference image mechanism. RefTon builds on this foundation and significantly improves detail fidelity through reference image guidance.
- **vs. TryOffDiff/ViTON-GUN**: These methods adopt a "try-off then try-on" strategy, introducing error accumulation and loss of garment detail. RefTon directly leverages reference images to avoid the try-off stage.
- **vs. Any2AnyTryon**: Also supports person-to-person try-on but generates position indices over a concatenated pixel-space canvas. RefTon generates indices independently per condition, more flexibly accommodating multi-resolution inputs.
- **vs. OmniVTON**: Requires additional pose and text conditions; RefTon is comparatively more streamlined.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The reference image-guided virtual try-on concept is novel and practically motivated; the data generation pipeline is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-dataset, multi-setting (masked/mask-free × with/without reference), cross-domain evaluation, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, method descriptions are detailed, and figures and tables are informative.
- **Value**: ⭐⭐⭐⭐ Addresses the practical problem of insufficient garment detail information in virtual try-on, with direct application value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](referencefree_image_quality_assessment_for_virtual.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2026\] HandDreamer: Zero-Shot Text to 3D Hand Model Generation](handdreamer_zero_shot_text_to_3d_hand_model_generation.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)
- [\[ICLR 2026\] Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](../../ICLR2026/human_understanding/inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)

<!-- RELATED:END -->
