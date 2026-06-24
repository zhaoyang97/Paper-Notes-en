---
title: >-
  [Paper Note] MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis
description: >-
  [CVPR 2025][Image Generation][Shadow Detection] MetaShadow proposes the first three-in-one framework that synergistically combines a GAN-based Shadow Analyzer (shadow detection + removal) with a diffusion-based Shadow Synthesizer (shadow synthesis). It transfers shadow knowledge by guiding the diffusion model with intermediate GAN features, achieving SOTA performance across all three shadow tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Shadow Detection"
  - "Shadow Removal"
  - "Shadow Synthesis"
  - "Hybrid GAN-Diffusion"
  - "Object-Centered"
date: 2026-05-08
content_hash: 686b14a95c897e82
---

# MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2412.02635](https://arxiv.org/abs/2412.02635)  
**Code**: None  
**Area**: Image Editing / Diffusion Models  
**Keywords**: Shadow Detection, Shadow Removal, Shadow Synthesis, Hybrid GAN-Diffusion, Object-Centered

## TL;DR
MetaShadow proposes the first three-in-one framework that synergistically combines a GAN-based Shadow Analyzer (shadow detection + removal) with a diffusion-based Shadow Synthesizer (shadow synthesis). It transfers shadow knowledge by guiding the diffusion model with intermediate GAN features, achieving SOTA performance across all three shadow tasks.

## Background & Motivation

1. **Background**: Shadow manipulation in image editing (detection, removal, synthesis) is usually processed separately, with existing methods independently completing only one or two sub-tasks.
2. **Limitations of Prior Work**: (1) Independent models cannot share shadow knowledge, leading to inconsistent results when used in a cascade (e.g., the color/shape of the removed shadow does not match the synthesized one); (2) Existing detection methods are mostly global shadow detection, lacking object-centered instance-level operations; (3) Existing shadow synthesis methods either require extra illumination parameters or lack effective shadow feature extractors.
3. **Key Challenge**: Shadow detection, removal, and synthesis inherently share the same physical knowledge (light direction, intensity, softness/hardness), but existing methods treat them in isolation, failing to mutually benefit one another.
4. **Goal**: How to complete three shadow tasks in an object-centered manner within a unified framework, and improve performance for each task through shared knowledge?
5. **Key Insight**: GANs excel at efficiently and accurately detecting and removing shadows but struggle to synthesize realistic shapes. Diffusion models excel at generating realistic content but are difficult to control regarding shadow attributes. The two are complementary.
6. **Core Idea**: Learn shadow understanding knowledge using a GAN, and migrate it through a feature adapter to guide controllable shadow synthesis in a diffusion model.

## Method

### Overall Architecture
MetaShadow consists of two collaborative components: the Shadow Analyzer in Stage I (CM-GAN-based shadow detection + removal) and the Shadow Synthesizer in Stage II (DDPM-based shadow synthesis). Given an image and an object mask, the Shadow Analyzer detects and removes the shadow of that object. When the object is moved to a new location, the Shadow Synthesizer utilizes the shadow knowledge transferred from the Analyzer to synthesize the shadow at the new position.

### Key Designs

1. **Shadow Analyzer**:

    - **Function**: Object-centered joint shadow detection and removal.
    - **Mechanism**: Modified based on CM-GAN, containing an encoder to extract multi-scale features $F_e^i$ and a global style code $s$, and two parallel cascaded decoders (global modulation + spatial modulation) that output $F_g^i$ and $F_s^i$ respectively. A shadow detector is integrated alongside the spatial decoder, upsampling multi-scale features to a unified size and concatenating them, which then outputs a 256×256 shadow mask through a convolutional layer. Training uses adversarial loss + perceptual loss + L1 loss + Dice loss.
    - **Design Motivation**: Embedding the detector into the GAN decoder allows the encoder and decoder to learn to identify shadow regions under detection supervision. These rich intermediate features are precisely the "shadow knowledge" needed by the subsequent diffusion model.

2. **Shadow Synthesizer + Shadow Knowledge Transfer**:

    - **Function**: Controllable synthesis of realistic shadows based on reference shadows.
    - **Mechanism**: Based on DDPM, with inputs including the image $I_o$ with the moved object and the object mask $M_{\tilde{o}}$. The key innovation is the shadow knowledge transfer: extracting multi-scale features $F_{ms}$ (dimension $[N, 1348, 32, 32]$) from the Shadow Analyzer, converting them to a shadow embedding $E_s$ of dimension $[N, 1024, 2048]$ via an adapter $T(\cdot)$ (2D convolution -> 1D convolution -> MLP), and injecting it into the UNet of the diffusion model through cross-attention. The loss is the standard diffusion denoising loss: $\mathcal{L}_{syn} = \mathbb{E}[\|\epsilon - \epsilon_\theta(I_o^t, M_{\tilde{o}}, M_{\tilde{s}}, t, T(F_{ms}))\|^2]$.
    - **Design Motivation**: The intermediate features of the GAN implicitly encode critical information such as illumination direction, shadow softness, and color. Translating this information through the adapter into conditional embeddings understandable by the diffusion model ensures the synthesized shadows align with the scene in color, direction, and intensity.

3. **Multi-Source Dataset Training Strategy**:

    - **Function**: Overcome the scale limitation of existing shadow datasets.
    - **Mechanism**: Construct a synthetic dataset MOS (rendered in Blender, 200 scenes x 8 viewpoints x 5 movements = 8,000 pairs), combined with existing datasets such as DESOBA, ISTD+, and SRD for training. For datasets with only partial labels, an empty object mask is inputted to allow the model to learn general shadow detection. Three types of data augmentation are used: random shadow intensity augmentation, curve color adjustment, and random shadow dropout.
    - **Design Motivation**: Object-centered three-in-one labeled data is extremely scarce. Using a combination of synthetic data, multi-dataset mixing, and empty mask compatibility strategies maximizes the utility of existing resources.

### Loss & Training
- Stage I (Shadow Analyzer): Adversarial loss + perceptual loss + masked-R₁ regularization + L1 loss + Dice loss, 100 epochs, learning rate 0.001, batch size 16.
- Stage II (Shadow Synthesizer): Standard diffusion denoising loss $\mathcal{L}_{syn}$, with the Shadow Analyzer frozen, training the UNet and the adapter. The learning rate of UNet is 1e-4, multiplied by 0.01 after 200 epochs; the adapter's learning rate remains constant at 1e-4, for a total of 400 epochs.
- Two-stage training, 8×A100 GPUs, Adam optimizer.

## Key Experimental Results

### Main Results

**Shadow Detection (SOBA test set)**:

| Method | mIoU | mIoU_xs | mIoU_s | mIoU_l |
|------|------|---------|--------|--------|
| SSISv2 | 55.8 | 42.4 | 49.5 | 82.5 |
| **MetaShadow** | **71.0** | **60.4** | **72.6** | **87.8** |

**Shadow Removal (DESOBA test set)**:

| Method | Masked MAE↓ | Bbox PSNR↑ | PSNR↑ |
|------|------------|-----------|-------|
| ShadowDiffusion (GT mask) | 35.45 | 24.28 | 40.04 |
| **MetaShadow** | **21.32** | **32.97** | **42.20** |

**Shadow Synthesis (DESOBA test set)**:

| Method | Local RMSE↓ |
|------|------------|
| SGRNet | 56.44 |
| SGDiffusion | 51.73 |
| **MetaShadow** | **36.54** |

### Ablation Study

| Configuration | Description |
|------|------|
| Full MetaShadow | Detection mIoU is 71.0, optimal in all tasks |
| w/o MOS dataset | mIoU is 67.2 (↓3.8), proving synthetic data is effective |
| w/o Knowledge Transfer | Shadow synthesis quality drops significantly, direction and intensity are uncontrollable |
| Diffusion model without GAN features | Different random seeds produce inconsistent shadows with random directions |

### Key Findings
- Detection mIoU improved from 55.8 to 71.0 (+15.2), especially for extra-small shadows which improved from 42.4 to 60.4, showing outstanding performance in fine-grained detection.
- Shadow removal task achieved an 8.7dB improvement in Bbox PSNR, significantly surpassing methods requiring ground truth (GT) shadow masks even without using GT shadow masks.
- The GAN-to-diffusion knowledge transfer resolved the inconsistency issue in shadow direction under different random seeds in the diffusion model.
- The MOS synthetic dataset yielded a 3.8 mIoU point improvement for the detection task, validating the utility of synthetic data for low-shot shadow tasks.

## Highlights & Insights
- **Complementary GAN+Diffusion Design**: Accurately identifies the complementary properties of these two paradigms in shadow tasks—GANs excel at "understanding" shadows (detection/removal), while diffusion models excel at "generating" shadows (synthesis). By transferring features, the strengths of both are combined. This GAN-guided diffusion paradigm can be generalized to other image editing tasks that require precise control.
- **Empty Mask Compatibility Training**: By randomly supplying empty object masks, the model can perform both object-level and global-level shadow operations, enabling a single model to cover both scenarios.
- **Three-in-One Unified Framework**: Unifies shadow detection, removal, and synthesis under a single framework for the first time, sharing shadow knowledge. This unified approach can also be explored in other interrelated vision tasks.

## Limitations & Future Work
- The resolution of the Shadow Synthesizer is only 128×128, which limits the quality of fine shadow details.
- Dependency on a reference shadow as input limits applicability when no reference shadow is available (e.g., when inserting objects into a completely new scene).
- The complexity of the MOS synthetic dataset is limited (200 scenes), which still has a gap compared to real-world shadow diversity.
- Shadows of transparent or semi-transparent objects are not handled.
- Lack of explicit modeling for physical consistency (such as multi-light sources, ambient occlusion, etc.).

## Related Work & Insights
- **vs ObjectDrop**: ObjectDrop implicitly handles shadows using a bootstrap strategy but cannot control shadow attributes separately; MetaShadow provides explicit shadow detection, removal, and synthesis, offering better controllability.
- **vs SGDiffusion**: The pure diffusion scheme of SGDiffusion leads to inconsistent directions under different seeds, whereas MetaShadow solves this by conditioning on GAN features.
- **vs ShadowDiffusion**: ShadowDiffusion requires GT shadow masks as input, whereas MetaShadow requires only object masks to automatically locate and remove shadows.

## Rating
- Novelty: ⭐⭐⭐⭐ The GAN-guided shadow knowledge transfer design is novel, and the three-in-one framework is a first in this field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated comprehensively on three tasks across four benchmarks, with two new test sets constructed.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and elegant illustrations, though some mathematical notations are slightly complex.
- Value: ⭐⭐⭐⭐ Directly practical for shadow processing in image editing, though the 128×128 resolution limits practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BootPlace: Bootstrapped Object Placement with Detection Transformers](bootplace_bootstrapped_object_placement_with_detection_transformers.md)
- [\[ICLR 2026\] ReFocusEraser: Refocusing for Small Object Removal with Robust Context-Shadow Repair](../../ICLR2026/image_generation/refocuseraser_refocusing_for_small_object_removal_with_robust_context-shadow_rep.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](../../ICCV2025/image_generation/structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)
- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[CVPR 2025\] Composing Parts for Expressive Object Generation](composing_parts_for_expressive_object_generation.md)

</div>

<!-- RELATED:END -->
