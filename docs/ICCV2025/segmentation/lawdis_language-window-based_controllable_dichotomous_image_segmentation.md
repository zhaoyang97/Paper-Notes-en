---
title: >-
  [Paper Note] LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation
description: >-
  [ICCV 2025][Segmentation][Dichotomous image segmentation] This paper proposes LawDIS, a controllable dichotomous image segmentation framework built upon a latent diffusion model. It achieves high-quality foreground mask…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Dichotomous image segmentation"
  - "latent diffusion model"
  - "language control"
  - "window refinement"
  - "high-precision segmentation"
date: 2026-05-08
content_hash: f28799fe155eb8e9
---

# LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation

**Conference**: ICCV 2025
**arXiv**: [2508.01152](https://arxiv.org/abs/2508.01152)  
**Code**: [GitHub](https://github.com/XinyuYanTJU/LawDIS)  
**Area**: Image Segmentation
**Keywords**: Dichotomous image segmentation, latent diffusion model, language control, window refinement, high-precision segmentation

## TL;DR

This paper proposes LawDIS, a controllable dichotomous image segmentation framework built upon a latent diffusion model. It achieves high-quality foreground mask generation through the synergy of macro-level language control (LS) and micro-level window refinement (WR), comprehensively outperforming 11 state-of-the-art methods on the DIS5K benchmark.

## Background & Motivation

Dichotomous image segmentation (DIS) aims to accurately segment foreground objects from high-resolution images, requiring pixel-level precise boundary delineation. With the proliferation of high-quality imaging devices, segmentation tasks have evolved from coarse localization to fine-grained boundary description. DIS finds broad applications in 3D reconstruction, image editing, augmented reality, and medical image segmentation.

Existing DIS methods face two core challenges:

**Semantic ambiguity**: When an image contains multiple foreground entities, discriminative learning paradigms (per-pixel classification) cannot flexibly specify the target object, lacking user interaction capability.

**Geometric detail capture bottleneck**: To capture geometric details of high-resolution targets, existing methods typically introduce additional high-resolution data streams or split images into fixed-size patches, but cannot accommodate variable patch sizes. For instance, MVANet degrades significantly on local patches at non-training resolutions.

The core motivation of this paper is to reframe DIS as an **image-conditioned mask generation task**, leveraging the generative capability of latent diffusion models to seamlessly integrate user control and address both challenges.

## Method

### Overall Architecture

LawDIS builds upon pre-trained Stable Diffusion v2 and reformulates DIS as a conditional denoising diffusion process. The framework consists of three core components:
- **Mode Switcher**: A one-dimensional vector added to the diffusion model's time embedding via positional encoding, switching between macro and micro modes.
- **Macro Mode**: Language-controlled segmentation strategy (LS), generating an initial mask based on user language prompts.
- **Micro Mode**: Window-controlled refinement strategy (WR), refining masks within user-specified variable-size windows.

### Key Design 1: Generative DIS Paradigm

DIS is modeled as a conditional probability distribution $D(s|x)$, where $s$ is the segmentation mask and $x$ is the RGB image. A VAE encoder $\phi$ maps both the segmentation mask and image to a low-dimensional latent space, where the diffusion process is performed:

- **Forward process**: Starting from $\mathbf{z}_0^{(s)}$, Gaussian noise is incrementally added to construct a discrete Markov chain.
- **Reverse process**: A U-Net $f_\theta$ predicts the noise at each timestep, progressively denoising conditioned on image features $\mathbf{z}^{(x)}$.
- **Architectural modification**: The U-Net input layer is duplicated to accommodate concatenated image and noisy mask features; weights are copied and halved to prevent activation explosion.

### Key Design 2: Dual-Mode Joint Training

**Macro mode training**: Mode $\psi_a$ is activated; the model receives the full image $x$, segmentation mask $s$, and a VLM-generated language prompt $\mathcal{T}$. The prompt is encoded via CLIP into a control embedding $c_\mathcal{T}$ and injected into the U-Net via cross-attention:
$$\mathcal{L}_{macro} = \|\boldsymbol{\epsilon} - f_\theta(\mathbf{z}_t^{(s)}, \mathbf{z}^{(x)}, c_\mathcal{T}, t, \psi_a)\|_2^2$$

**Micro mode training**: Mode $\psi_b$ is activated; the minimum bounding rectangle of the foreground object is selected as a local window, cropped to obtain a local patch $x_p$ and local mask $s_p$. A null prompt $c_\varnothing$ is used to avoid semantic mismatch:
$$\mathcal{L}_{micro} = \|\boldsymbol{\epsilon}_p - f_\theta(\mathbf{z}_t^{(s_p)}, \mathbf{z}^{(x_p)}, c_\varnothing, t, \psi_b)\|_2^2$$

**Joint training**: $\mathcal{L}_u = \mathcal{L}_{macro} + \mathcal{L}_{micro}$. Both modes share the same U-Net, enabling mutual enhancement.

### Key Design 3: VAE Decoder Fine-tuning

After training the U-Net, the encoder and U-Net are frozen, and only the VAE decoder $\varphi$ is fine-tuned:
- Shortcut connections from the encoder to the decoder are added.
- The output channels are reduced from 3 to 1 (single-channel mask), with weights initialized via channel averaging.
- The **TCD (Trajectory Consistency Distillation)** scheduler is introduced to reduce sampling to a single step, saving memory and improving inference efficiency.

### Loss & Training

VAE decoder fine-tuning employs a structural loss:
$$\mathcal{L}_d = \mathcal{L}_{wbce}(\hat{s}, s) + \mathcal{L}_{wiou}(\hat{s}, s)$$
comprising weighted binary cross-entropy loss and weighted IoU loss, respectively.

### Inference Pipeline

1. **Language-controlled segmentation** (macro mode): Full image + language prompt → single-step TCD denoising → decoded initial segmentation map.
2. **Window-controlled refinement** (micro mode, optional): User selects an unsatisfactory region → crop local patch → use initial segmentation result (rather than pure noise) as the diffusion starting point → single-step denoising → refined mask replaces the original region. This process can be repeated indefinitely until satisfactory results are achieved.

## Key Experimental Results

### Main Results: DIS5K Benchmark (DIS-TE, 2000 images)

| Method | $F_\beta^\omega$ ↑ | $F_\beta^{mx}$ ↑ | $\mathcal{M}$ ↓ | $\mathcal{S}_\alpha$ ↑ | $E_\phi^{mn}$ ↑ |
|------|-----|-----|------|------|------|
| IS-Net (2022) | 0.726 | 0.799 | 0.070 | 0.819 | 0.858 |
| InSPyReNet (2022) | 0.838 | 0.891 | 0.039 | 0.900 | 0.923 |
| BiRefNet (2024) | 0.858 | 0.896 | 0.035 | 0.901 | 0.934 |
| GenPercept (2024) | 0.816 | 0.868 | 0.043 | 0.880 | 0.923 |
| MVANet (2024) | 0.862 | 0.907 | 0.034 | 0.909 | 0.938 |
| **Ours-S (LS only)** | **0.898** | **0.929** | **0.027** | **0.925** | **0.955** |
| **Ours-R (LS+WR)** | **0.908** | **0.932** | **0.024** | **0.926** | **0.959** |

- Ours-S surpasses MVANet by 6.6% in $F_\beta^\omega$ on DIS-TE1; Ours-R further improves by 2.0% on DIS-TE4.
- With both controls integrated, the improvement over MVANet on DIS-TE1 reaches 7.0%.

### Ablation Study

| Setting | $F_\beta^{mx}$ ↑ | $\mathcal{M}$ ↓ | $\mathcal{S}_\alpha$ ↑ | $E_\phi^{mn}$ ↑ |
|------|------|------|------|------|
| Baseline (no mode switcher / prompt / VAE fine-tuning) | 0.904 | 0.047 | 0.904 | 0.916 |
| Without micro mode training | 0.912 | 0.037 | 0.909 | 0.943 |
| Without VAE decoder fine-tuning | 0.919 | 0.040 | 0.915 | 0.933 |
| **Full Ours-S** | **0.926** | **0.032** | **0.920** | **0.955** |

**Macro control ablation**:

| Setting | $F_\beta^{mx}$ ↑ | $\mathcal{M}$ ↓ | $\mathcal{S}_\alpha$ ↑ |
|------|------|------|------|
| No prompt (train + test) | 0.912 | 0.036 | 0.908 |
| No prompt (test only) | 0.915 | 0.036 | 0.909 |
| **With prompt (train + test)** | **0.926** | **0.032** | **0.920** |

**Micro control ablation**:

| Setting | $F_\beta^\omega$ | $\mathcal{M}$ | $BIoU^m$ ↑ | $HCE_\gamma$ ↓ |
|------|------|------|------|------|
| Base Ours-S | 0.890 | 0.032 | 0.795 | 2481 |
| Initialized from Gaussian noise | -4.7% | +1.9% | -7.1% | -863 |
| Automatic window selection | +1.7% | -0.5% | +2.9% | -767 |
| Semi-automatic window selection | +2.0% | -0.6% | +3.2% | -871 |

### Key Findings

- Initializing the micro mode diffusion process from the initial segmentation result (rather than Gaussian noise) is critical, as it implicitly conveys global context information.
- Even with fully automatic window selection without user intervention, the WR strategy effectively improves boundary accuracy.
- Dual-mode joint training provides scalable geometric representation capability, enabling the model to adapt to varying input sizes.
- VAE decoder fine-tuning is essential for high-resolution segmentation, complementing denoised mask features with fine-grained details.

## Highlights & Insights

1. **Paradigm shift**: This work is the first to reformulate DIS from discriminative per-pixel classification to generative mask generation, leveraging the encyclopedic visual-language understanding of diffusion models.
2. **Controllability design**: Macro language control addresses the semantic ambiguity of "which object to segment," while micro window refinement addresses the geometric precision issue of "insufficiently fine boundaries." Both are unified within a single diffusion model via the mode switcher.
3. **Efficient inference**: The TCD scheduler enables single-step denoising, making diffusion models practically viable for segmentation in terms of inference speed.
4. **Clever initialization strategy**: The micro mode uses the macro mode's segmentation result (rather than pure noise) as the diffusion starting point, implicitly transferring contextual information between the two modes — a key factor in performance gains.
5. **Flexible user interaction**: Window refinement can be repeated indefinitely, supporting progressive refinement and making the framework suitable for high-precision personalized applications.

## Limitations & Future Work

- Built upon Stable Diffusion v2, the model has a large parameter count, incurring higher deployment costs than traditional discriminative methods.
- Language prompts rely on VLM-based automatic generation or manual user input, requiring an additional prompt generation module in fully automated scenarios.
- Automatic window selection in micro mode still relies on edge-detection heuristics, yielding slightly weaker results than manual user selection.
- All inputs are uniformly resized to $1024^2$, which may introduce distortion for images with extreme aspect ratios.
- Evaluation is conducted solely on the DIS5K benchmark; generalizability to other high-precision segmentation tasks (e.g., portrait matting, medical segmentation) remains unverified.

## Related Work & Insights

- **Dichotomous image segmentation**: IS-Net introduces intermediate supervision; FP-DIS exploits frequency priors; BiRefNet/MVANet enhance detail via multi-resolution patches. However, these discriminative methods lack flexible semantic control and adaptive local window capability.
- **Diffusion models for segmentation**: GenPercept converts generative models into a deterministic single-step paradigm; Wang et al. propose a diffusion refinement model to enhance mask quality. This paper is the first to extend a single Stable Diffusion model into a macro+micro dual-mode framework.
- **High-resolution segmentation**: InSPyReNet and BiRefNet enhance detail through additional resolution streams; MVANet employs fixed-size patch splitting, but none of these approaches adaptively handle variable patch sizes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reformulating DIS as a controllable generative task is a creative paradigm shift; the dual-mode switcher design is concise and effective.
- **Technical Depth**: ⭐⭐⭐⭐ — Adaptations of the diffusion model (input layer duplication, TCD single-step inference, VAE fine-tuning strategy) are well-considered; the micro mode initialization strategy is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Outperforms 11 methods across all metrics on DIS5K with ablations covering each component; however, evaluations on additional datasets and efficiency comparisons are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear; the macro-micro narrative logic flows naturally.
- **Recommendation**: ⭐⭐⭐⭐ — Introduces a novel generative paradigm for high-precision segmentation with convincing experimental results.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Refer to Any Segmentation Mask Group With Vision-Language Prompts](refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)
- [\[AAAI 2026\] CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion](../../AAAI2026/segmentation/ctrlfuse_mask-prompt_guided_controllable_infrared_and_visible_image_fusion.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
