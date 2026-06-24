---
title: >-
  [Paper Note] L-DiffER: Single Image Reflection Removal with Language-Based Diffusion Model
description: >-
  [ECCV 2024][Image Generation][Image Reflection Removal] L-DiffER is proposed, a language-guided diffusion model that addresses the issue of inaccurate control conditions through an iterative condition refinement strategy. It integrates a multi-condition constraint mechanism to ensure the color and structural fidelity of image restoration, while preserving the generative capability of diffusion models to handle low-transmission reflections.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image Reflection Removal"
  - "Diffusion Models"
  - "Language Guidance"
  - "Iterative Condition Refinement"
  - "Multi-condition Constraint"
date: 2026-05-08
content_hash: 6c0f4dbf560a7bbb
---

# L-DiffER: Single Image Reflection Removal with Language-Based Diffusion Model

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Image Reflection Removal, Diffusion Models, Language Guidance, Iterative Condition Refinement, Multi-condition Constraint

## TL;DR

L-DiffER is proposed, a language-guided diffusion model that addresses the issue of inaccurate control conditions through an iterative condition refinement strategy. It integrates a multi-condition constraint mechanism to ensure the color and structural fidelity of image restoration, while preserving the generative capability of diffusion models to handle low-transmission reflections.

## Background & Motivation

**Background**: Single Image Reflection Removal (SIRR) is a classic image restoration problem—separating an image taken through glass into a transmission layer (background) and a reflection layer. Existing methods are mainly based on end-to-end CNN regression or optimization utilizing prior knowledge (e.g., gradient sparsity, double images). Recently, Diffusion Models have shown powerful capabilities in image generation, naturally raising interest in applying them to image restoration tasks.

**Limitations of Prior Work**: Directly applying existing language-guided diffusion models (such as Stable Diffusion) to image restoration faces two core challenges: (1) **Inaccurate control conditions**: Image restoration requires precise input conditions to guide the restoration process. However, the degraded image (the mixed image containing reflection) as a condition inherently contains the reflection information to be removed, leading to residual reflection or background distortion in the generation results; (2) **Insufficient restoration fidelity**: The generative capability of diffusion models may cause the output to deviate from the color and structure of the original image, producing hallucinated content, which is unacceptable in image restoration tasks.

**Key Challenge**: The generative capability of diffusion models is a double-edged sword. It can handle severe degradation (such as low-transmission reflection where the background is almost completely obscured), but this powerful generative ability also means the model might "deviate freely" from the ground-truth content of the input image. A precise balance between generative capability and restoration fidelity is required.

**Goal**: (1) How to provide accurate control conditions during the denoising process of diffusion models? (2) How to constrain the outputs of diffusion models to be faithful to the color and structure of the original image? (3) How to retain sufficient generative capability to handle severe reflections while preserving fidelity?

**Key Insight**: The core observation is that in the iterative denoising process of diffusion models, the intermediate results of each step can serve as better conditional inputs for the next step. Although using the degraded image as a condition in the initial stage is inaccurate, the denoised intermediate results have already partially removed the reflection. Updating the condition with these results can provide more accurate guidance. This "self-refinement" idea can progressively improve condition quality.

**Core Idea**: Gradually refining the condition input with intermediate results during the iterative diffusion denoising process, while ensuring color/structure fidelity through multi-condition constraints, to achieve accurate and controllable reflection removal.

## Method

### Overall Architecture

L-DiffER is built upon a pre-trained language-guided diffusion model (such as Stable Diffusion). The inputs are the mixed image with reflection $I$ and a text description (e.g., "a clear photo without reflection"). The framework comprises three core innovations: (1) Iterative Condition Refinement (ICR), which dynamically updates control conditions during denoising; (2) Multi-condition Constraint (MCC), which ensures restoration fidelity through color and structural guidance; and (3) an adaptive generation-fidelity balancing strategy that adjusts the ratio of generative capability to constraint strength based on reflection intensity.

### Key Designs

1. **Iterative Condition Refinement (ICR)**:

    - **Function**: Address the issue of inaccurate control conditions, progressively providing clearer guidance.
    - **Mechanism**: The conditions of standard diffusion models remain fixed throughout the denoising process (i.e., always using the degraded image $I$ as the condition). ICR divides the denoising process into multiple stages. At the end of each stage, the denoising result of the current step $\hat{x}_t$ is decoded into a pixel-space image $\hat{I}_t$, and this intermediate result replaces the original conditional input as the control condition for the next stage. Since $\hat{I}_t$ has already partially removed reflection compared to the original $I$, the conditions for subsequent stages are more accurate, forming a positive feedback loop: better conditions $\rightarrow$ better denoising results $\rightarrow$ better conditions. The refinement frequency of ICR is a key hyperparameter—too frequent increases computational overhead and may introduce noise, while too sparse makes the refinement effect insignificant. Experiments show that refining once every 5-10 steps yields the best performance.
    - **Design Motivation**: Inaccurate conditions are a fundamental bottleneck for diffusion models in image restoration. ICR leverages the natural iterative structure of diffusion denoising, elegantly feeding back intermediate results to improve conditions at zero cost.

2. **Multi-condition Constraint (MCC)**:

    - **Function**: Ensure that the restored results are faithful to the color and structure of the original image.
    - **Mechanism**: Two extra conditional constraints are introduced. **Color Constraint**: Inject the low-frequency color information of the original image $I$ (extracted via Gaussian blur) into the denoising process to ensure that the overall tone of the restored image matches the input. Specifically, the color residual is superimposed onto each denoising step's result: $\hat{x}_{t}^{color} = \hat{x}_t + \gamma \cdot (I_{low} - \hat{x}_{t,low})$. **Structural Constraint**: Extract the edge map (Canny/Sobel) of the original image $I$ and inject the structural prior through a ControlNet-style auxiliary network to guarantee that the geometric layout of the restored image matches the input. The strength of the edge constraint decreases gradually as the denoising process progresses—strong constraints in the early stage ensure the global structure, and relaxed constraints in the later stage allow the model to refine texture details.
    - **Design Motivation**: Unconstrained diffusion models may alter the image's tone (due to priors trained under various lighting conditions) or structure (generating content inconsistent with the input). Color and structure are core information that must be preserved in image restoration; explicit constraints are more reliable than relying solely on implicit learning.

3. **Adaptive Generation-Fidelity Balancing**:

    - **Function**: Automatically adjust generative capability and constraint strength based on reflection severity.
    - **Mechanism**: Estimate the intensity of reflection in the input image (by analyzing gradient distribution or frequency characteristics of the image). When reflection is weak (background is clearly visible), the fidelity constraint weight is increased, and generative freedom is reduced—since only minor restoration is needed. When reflection is heavy (background is almost invisible), the constraint weight is decreased, and generative freedom is increased—since the strong generative capability of the diffusion model is required to "imagine" the obscured background content. The balance coefficient $\alpha$ can be represented as $\alpha = f(R_{intensity})$, where $f$ is a learned mapping function.
    - **Design Motivation**: A one-size-fits-all constraint strategy does not apply to all reflection intensities—weak reflections require high fidelity, while strong reflections require strong generation power. Adaptive balancing enables the model to handle both extreme cases simultaneously.

### Loss & Training

Training loss includes: (1) Diffusion denoising loss $L_{denoise} = \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$, the standard noise prediction objective; (2) Perceptual loss $L_{percep}$, based on VGG feature matching to ensure semantic-level fidelity; (3) L1 pixel loss $L_{pixel}$, to ensure color accuracy. The fine-tuning strategy adopts two stages—the first stage trains the basic reflection removal capability with a large batch size, and the second stage fine-tunes the hyperparameters of ICR and MCC with a small learning rate.

## Key Experimental Results

### Main Results

| Dataset | Metric | L-DiffER | IBCLN | DSRNet | Gain |
|--------|------|----------|-------|--------|------|
| SIR² | PSNR↑ | 24.83 | 22.18 | 23.47 | +1.36 |
| SIR² | SSIM↑ | 0.882 | 0.845 | 0.861 | +0.021 |
| Real20 | PSNR↑ | 25.41 | 22.96 | 24.15 | +1.26 |
| Real20 | SSIM↑ | 0.891 | 0.857 | 0.873 | +0.018 |
| Nature | PSNR↑ | 23.65 | 21.40 | 22.78 | +0.87 |
| CDR Dataset | PSNR↑ | 27.12 | 24.38 | 25.91 | +1.21 |

### Ablation Study

| Configuration | SIR² PSNR | SIR² SSIM | Description |
|------|-----------|-----------|------|
| Full L-DiffER | 24.83 | 0.882 | Full model |
| w/o ICR | 23.12 | 0.858 | No condition refinement, PSNR drops by 1.71 |
| w/o Color Constraint | 24.21 | 0.869 | Obvious color cast |
| w/o Structural Constraint | 23.85 | 0.862 | Structural distortion in some areas |
| w/o Adaptive Balancing | 24.35 | 0.874 | Fixed balancing coefficient |
| Directly use SD for restoration | 21.76 | 0.815 | Without any adaptation |

### Key Findings

- ICR is the most critical contribution, yielding a 1.71 dB PSNR improvement, which proves that condition refinement is crucial for diffusion models in image restoration.
- Within the multi-condition constraint, the color constraint and structural constraint contribute approximately 0.6 and 1.0 dB respectively, with the structural constraint having a larger impact.
- Directly using Stable Diffusion for reflection removal performs poorly (PSNR of only 21.76), indicating that unadapted diffusion models cannot handle precise image restoration tasks.
- In low-transmission reflection scenarios, L-DiffER exhibits a more pronounced advantage over traditional CNN methods—the generative capability of diffusion models is particularly critical under severe degradation.
- Qualitative results demonstrate that L-DiffER can better preserve high-frequency details such as text and fine lines while removing reflections.

## Highlights & Insights

- **Iterative Condition Refinement is a general strategy for diffusion-based image restoration**: The core idea of ICR—feeding back denoised intermediate results to update conditions—applies not only to reflection removal but can also be generalized to all diffusion-model-based image restoration tasks such as deraining, dehazing, and deblurring. Inaccurate conditions are a common bottleneck in these tasks, and ICR provides an elegant, zero-extra-parameter solution.
- **Multi-level constraint design for color and structure**: Decomposing fidelity into color fidelity and structural fidelity to constrain them independently ensures both flexibility (allowing independent adjustment of their weights) and comprehensiveness (without missing any aspects).
- **Adaptive control of generative capability**: Automatically adjusting the generation-fidelity balance according to the degradation level is a practical design principle. Since degradation levels vary significantly across different samples in restoration tasks, a fixed strategy is bound to fail in certain scenarios.

## Limitations & Future Work

- Iterative condition refinement increases inference time—each refinement requires additional decoding operations. Reducing the refinement frequency or optimizing implementation may be necessary in real-time application scenarios.
- The utilization of text prompts ("a clear photo without reflection") is relatively simple and fixed, which does not fully exploit the potential of language guidance. Finer-grained language descriptions (e.g., describing the background content) could be considered to provide more precise semantic guidance.
- For reflections in dynamic scenes (such as moving car window reflections), single-frame methods cannot leverage temporal consistency.
- The reflection synthesis method of the training data (alpha blending) differs from the physical formation process of real reflection, potentially leading to degraded performance when generalizing to real-world scenarios.
- The method relies on the prior knowledge of pre-trained Stable Diffusion, which may perform poorly in scenarios that are rare in the pre-training data.

## Related Work & Insights

- **vs IBCLN**: IBCLN utilizes an iterative "guessing + refinement" strategy within a CNN framework for reflection removal. L-DiffER implements a similar iterative concept within a diffusion model framework, but the powerful priors of diffusion models enable refinement performance that far exceeds the limitations of CNNs.
- **vs DSRNet**: DSRNet uses a dual-stream network to estimate the background and reflection layers separately. L-DiffER processes this with a single diffusion model, avoiding the competition between two branches in the dual-stream design.
- **vs IR-SDE**: IR-SDE uses a stochastic differential equation framework for image restoration. L-DiffER adds condition refinement and multi-condition constraints on top of this, addressing the condition inaccuracy issue that also exists in SDE frameworks. Integrating ICR and MCC into IR-SDE could be considered for further improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of iterative condition refinement and multi-condition constraints is innovative, though individual ideas are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on 4 datasets, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis and clear description of the methodology.
- Value: ⭐⭐⭐⭐ Provides a practical engineering solution for the application of diffusion models in image restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Implicit Concept Removal of Diffusion Models](implicit_concept_removal_of_diffusion_models.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models](styletokenizer_defining_image_style_by_a_single_instance_for_controlling_diffusi.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)

</div>

<!-- RELATED:END -->
