---
title: >-
  [Paper Note] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation
description: >-
  [AAAI 2026][Segmentation][Camouflage Image Generation] CT-CIG is proposed as the first text-guided controllable camouflage image generation method. It leverages a VLM-based Camouflage-Revealing Dialogue Mechanism (CRDM)…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Camouflage Image Generation"
  - "Diffusion Models"
  - "Text-guided"
  - "Frequency Interaction"
  - "Vision Language Model"
date: 2026-05-08
content_hash: ae4090934236a407
---

# Text-guided Controllable Diffusion for Realistic Camouflage Images Generation

**Conference**: AAAI 2026
**arXiv**: [2511.20218](https://arxiv.org/abs/2511.20218)  
**Code**: [github.com/NikoNairre/CT-CIG](https://github.com/NikoNairre/CT-CIG)  
**Area**: Segmentation
**Keywords**: Camouflage Image Generation, Diffusion Models, Text-guided, Frequency Interaction, Vision Language Model

## TL;DR

CT-CIG is proposed as the first text-guided controllable camouflage image generation method. It leverages a VLM-based Camouflage-Revealing Dialogue Mechanism (CRDM) to generate high-quality text prompts, and combines a lightweight control network with a Frequency Interaction Refinement Module (FIRM) built upon the Stable Diffusion framework to produce logically coherent and texturally realistic camouflage images, establishing a new Text-guided CIG paradigm.

## Background & Motivation

Camouflage is an instinctive survival strategy by which organisms visually blend into their surroundings to avoid detection. Camouflage Image Generation (CIG) is important for augmenting training data for Camouflaged Object Detection (COD), yet natural camouflage images are difficult to collect.

**Two dominant CIG paradigms and their limitations**:

**Background fitting**: Alters the color and texture of objects to blend into arbitrary backgrounds (DCI, LCG-Net, PTDiffusion). The drawback is that it **destroys object appearance** and ignores the logical relationship between the foreground object and the background environment (e.g., a tiger's face appearing on a mountain), producing visual art rather than natural camouflage.

**Foreground guiding**: Uses generative models to outpaint backgrounds based on foreground object features (LAKE-RED, FACIG). The drawback is **lack of background semantic understanding**, leading to severe texture artifacts and unrealistic backgrounds.

**Core insight**: Natural camouflage requires not only visual consistency (color and texture similarity) but also **logical plausibility**—a semantically reasonable correspondence between the camouflaged object and its environment. This logical relationship cannot be learned directly from the pixel domain but can be explicitly introduced through text prompts in the semantic domain.

**Key challenge**: COD datasets lack paired textual descriptions, necessitating a method to automatically generate high-quality, camouflage-aware text prompts.

## Method

### Overall Architecture

CT-CIG is built upon Stable Diffusion (SDXL) and accepts three inputs:
- RGB camouflage image $x \in \mathbb{R}^{3 \times h \times w}$
- Binary mask $c_f \in \mathbb{R}^{1 \times h \times w}$
- Text prompt $c$ generated via CRDM + VLM

Pipeline: VAE encodes the image into latent space → Gaussian noise is added → a lightweight controller encodes the mask → FIRM performs frequency refinement → Cross Normalization aligns distributions → UNet diffusion denoising → VAE decoding. Only the controller, FIRM, and the linear projection layers of UNet cross-attention are trained (~4% of parameters).

### Key Designs

#### 1. **Camouflage-Revealing Dialogue Mechanism (CRDM)**

**Mechanism**: The VLM's visual perception and contextual understanding capabilities are leveraged through carefully designed multi-turn dialogues to generate camouflage-aware textual descriptions.

**Preprocessing**: Random semi-transparent colored contour lines are drawn on all images to annotate object boundaries—which are precisely the critical cues for camouflage. The semi-transparent effect helps the VLM locate camouflaged objects while preserving boundary pixel details.

**Dialogue design** (4-round Q&A):
- **Q1**: Obtain a description of the camouflaged object
- **Q2**: Obtain a description of the surrounding environment and its relationship to the object
- **Q3**: Reorganize the above descriptions into a detailed prompt $T_{detail}$
- **Q4**: Review all content and summarize into a single sentence $T_{simple}$

**Non-camouflage image handling**: For salient or general images, Q2 is modified to prompt the VLM to **imagine** an ideal scene in which the object could be successfully camouflaged, then generate the corresponding prompt.

$T_{detail}$ is used for training (richer information to prevent catastrophic forgetting), while $T_{simple}$ is used for inference (to increase generation diversity).

**VLM selection**: Based on CLIPScore evaluation, Qwen2.5-VL-7B achieves the best text-image alignment (0.3242), outperforming BLIP2, LLaVA, and Gemma3.

#### 2. **Frequency Interaction Refinement Module (FIRM)**

**Core problem**: Binary masks provide only coarse positional and geometric cues, lacking spatial hierarchy and intra-object appearance information. The controller-encoded $x_{cf}$ is informationally insufficient, potentially producing texture artifacts and unnatural hallucinations.

**Design Motivation**: Fourier transforms are applied to learn high-frequency texture information from the image latent representation $z_t$ to enhance control features.

**Pipeline**:
1. Apply FFT to $x_{cf}$ and $z_t$ to obtain frequency-domain representations
2. An attention generator (2-layer convolution) produces an attention map $A$ from the spectrum of $|z_t|$ (fftshift is applied first to make the spectrum contiguous for convolution)
3. Interaction-enhanced control spectrum: $\hat{x}_{facf} = \hat{x}_{cf} \otimes A$
4. Adaptively add the refinement gain via a learnable gate: $\hat{x}_{frcf} = \hat{x}_{cf} + gate \times (\hat{x}_{facf} - \hat{x}_{cf})$
5. IFFT transforms back to the feature domain

**Design Motivation**: According to Fourier spectral theory, low frequencies encode global structural information while high frequencies encode texture and fine-grained patterns. FIRM equips control features with detailed texture representations from the image, ensuring robustness in generating complex camouflage textures.

#### 3. **Cross Normalization (CN)**

A distributional discrepancy between FIRM-refined control features and the noisy latent representations can cause color instability. CN standardizes the control features and applies an affine transformation using the statistics of the latent representation:

$$x'_{frcf} = \mu_z + \frac{x_{frcf} - \mu_{cf}}{\sqrt{\sigma^2_{cf} + \varepsilon}} \times \sigma_z$$

This aligns the final control signal with the distribution of the noisy latent representation, replacing the zero-convolution layers in ControlNet.

### Loss & Training

The total loss combines the conditional diffusion loss and the LPIPS perceptual loss:

$$\mathcal{L} = \mathcal{L}_{SD} + \lambda_{Lpips} \cdot \mathcal{L}_{Lpips}$$

Where:
- $\mathcal{L}_{SD}$: Standard conditional diffusion noise prediction MSE loss
- $\mathcal{L}_{Lpips}$: LPIPS perceptual loss minimizing VGG feature differences between generated and input images
- $\lambda_{Lpips} = 1\text{e-3}$
- Controller and FIRM learning rate: 1e-4; UNet learning rate: 5e-6
- Trained for 80 epochs; ~8 hours on 4× RTX A5000
- Control scaling factor: 1.2

## Key Experimental Results

### Main Results

| Paradigm | Method | Camo FID↓ | Sal FID↓ | Gen FID↓ | Overall FID↓ | Overall KID↓ | CLIP↑ |
|----------|--------|-----------|----------|----------|--------------|--------------|-------|
| Background fitting | LCGNet | 129.80 | 136.24 | 132.64 | 129.88 | 0.0550 | — |
| Foreground guiding | LAKERED | 39.55 | 88.70 | 102.67 | 64.27 | 0.0355 | — |
| Text-guided | ControlNet | 39.67 | 81.72 | 102.94 | 59.52 | 0.0227 | 0.2950 |
| Text-guided | SOO | **30.92** | 89.46 | 117.31 | 59.75 | 0.0187 | 0.3043 |
| **Text-guided** | **CT-CIG** | 30.59 | **81.60** | **104.46** | **52.88** | **0.0169** | **0.3243** |

CT-CIG achieves a substantially leading overall FID of 52.88, and the highest CLIPScore indicates the best semantic alignment.

### Ablation Study

**Effect of FIRM and CN**:

| Configuration | FID↓ | KID↓ | Note |
|---------------|------|------|------|
| w/o FIRM & CN | 32.37 | 0.0079 | Baseline; visible texture artifacts |
| w/o CN | 33.99 | 0.0114 | Missing distribution alignment |
| w/o FIRM | 31.66 | 0.0080 | Missing high-frequency texture detail |
| **CT-CIG (full)** | **30.59** | **0.0085** | Best |

**Effect of text prompt configuration**:

| Configuration | CLIP↑ | FID↓ | KID↓ | Note |
|---------------|-------|------|------|------|
| Simple text training | 0.3183 | 54.92 | 0.0387 | Catastrophic forgetting; blurry outputs |
| No object contour | 0.3247 | 39.24 | 0.0112 | High CLIP but mismatched shape guidance |
| Contour mentioned | 0.3218 | 39.79 | 0.0138 | Line-drawing artifacts |
| **Silent contour (Ours)** | 0.3242 | **30.59** | **0.0085** | Best balance |

**VLM selection** (CLIPScore):

| VLM | CLIP simple | CLIP detail |
|-----|-------------|-------------|
| BLIP2-2.7B | 0.2461 | 0.2859 |
| LLaVA-13B | 0.2986 | 0.2969 |
| Gemma3-4B | 0.3127 | 0.3136 |
| **Qwen2.5-VL-7B** | **0.3183** | **0.3242** |

### Key Findings

1. The text-guided paradigm comprehensively outperforms background fitting and foreground guiding, demonstrating that semantic understanding is key to achieving natural camouflage.
2. Semi-transparent contour annotations help the VLM perceive camouflage but must remain "silent" in the text—mentioning the contours misleads the generation process.
3. The strategy of training with detailed prompts and inferring with simple prompts prevents catastrophic forgetting while preserving generation diversity.
4. FIRM's high-frequency texture enhancement and CN's distribution alignment are complementary in terms of visual quality; both components are indispensable.

## Highlights & Insights

1. **Establishing a new Text-guided CIG paradigm**: For the first time, camouflage generation is elevated from a purely visual task to a vision-language joint task, introducing logical plausibility as an explicit constraint.
2. **Elegant CRDM design**: The 4-round dialogue progressively guides the VLM through perception → understanding → description → summarization, with distinct dialogue strategies for camouflaged and non-camouflaged images.
3. **Ingenuity of semi-transparent contours**: These simultaneously assist the VLM in localizing camouflaged objects and preserve boundary pixel information, striking a balance between localization aid and information fidelity.
4. **Frequency-domain control signal enhancement**: Introducing Fourier transforms within a diffusion model to enrich informationally sparse binary mask control signals is an uncommon yet effective design choice.
5. **Parameter efficiency**: Fine-tuning only ~4% of parameters is sufficient to adapt the model to camouflage scenarios.

## Limitations & Future Work

1. The method depends on the mask quality of COD datasets; noisy mask annotations can degrade training performance.
2. The quality of VLM-generated text prompts has an inherent ceiling, and descriptions of complex scenes may be insufficiently accurate.
3. The resolution of generated images is limited by SDXL (512×512), making it difficult to meet high-resolution requirements.
4. The training data volume is limited (LAKE-RED contains only 4,040 training images), which may constrain generalization.
5. The actual data augmentation effect of generated images on downstream COD tasks has not been evaluated.

## Related Work & Insights

- **LAKE-RED** (ECCV 2024): A representative foreground-guided CIG method that generates color-consistent backgrounds via VQVAE and knowledge retrieval.
- **ControlNeXt**: The foundation for CT-CIG's control network, replacing ControlNet's parallel blocks with a lightweight network.
- **Qwen2.5-VL**: Serves as the VLM backbone; its vision-language alignment capability directly determines text prompt quality.
- **Camobj-LLaVA**: The first VLM focused on camouflage scene understanding, complementary in objective to CRDM.
- The multi-turn dialogue strategy of CRDM can be generalized to other image generation tasks requiring domain-specific descriptions.

## Rating

- Novelty: ⭐⭐⭐⭐ (Text-guided camouflage generation is a new paradigm, though individual components are relatively standard)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Compared against 11 methods with detailed ablations, but lacks downstream task evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, well-defined paradigm taxonomy)
- Value: ⭐⭐⭐⭐ (Establishes a new paradigm with practical value for COD data augmentation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion](ctrlfuse_mask-prompt_guided_controllable_infrared_and_visible_image_fusion.md)
- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](../../ICCV2025/segmentation/vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](../../NeurIPS2025/segmentation/seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)

</div>

<!-- RELATED:END -->
