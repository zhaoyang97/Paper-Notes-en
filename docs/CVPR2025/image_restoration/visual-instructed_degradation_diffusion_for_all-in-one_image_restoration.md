---
title: >-
  [Paper Note] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration
description: >-
  [CVPR 2025][Image Restoration][All-in-One Image Restoration] Defusion proposes replacing textual instructions with "visual instructions" to guide all-in-one image restoration. By applying degradation effects to standardized visual elements, it constructs visual degradation descriptions. It performs diffusion denoising in the degradation space (instead of the image space), surpassing both task-specific and all-in-one methods across 8 restoration tasks.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "All-in-One Image Restoration"
  - "Visual Instructions"
  - "Degradation Diffusion"
  - "Degradation Space"
  - "Visual Prior"
date: 2026-05-08
content_hash: 31c1410b13c9b952
---

# Visual-Instructed Degradation Diffusion for All-in-One Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2506.16960](https://arxiv.org/abs/2506.16960)  
**Code**: None (Project Page available)  
**Area**: Image Restoration / All-in-One Restoration  
**Keywords**: All-in-One Image Restoration, Visual Instructions, Degradation Diffusion, Degradation Space, Visual Prior

## TL;DR

Defusion proposes replacing textual instructions with "visual instructions" to guide all-in-one image restoration. By applying degradation effects to standardized visual elements, it constructs visual degradation descriptions. It performs diffusion denoising in the degradation space (instead of the image space), surpassing both task-specific and all-in-one methods across 8 restoration tasks.

## Background & Motivation

**Background**: Image restoration encompasses various tasks such as denoising, deblurring, dehazing, and deraining. Traditional approaches train independent models for each type of degradation, whereas all-in-one methods attempt to handle multiple degradations using a single model. Current all-in-one methods primarily obtain degradation priors through two routes: implicit priors (extracted from LQ images via sub-networks or learnable prompts) and explicit priors (textual descriptions processed by language models).

**Limitations of Prior Work**: (1) Implicit priors (e.g., PromptIR, AirNet) are essentially equivalent to parameter scaling, offering limited generalization to complex or combined degradations. (2) Explicit priors based on textual descriptions (e.g., SUPIR, InstructIR) suffer from weak language-visual alignment (e.g., the instruction "enhance the image" can refer to brightening or denoising, and "underwater enhancement" actually involves both brightening and deblurring). Language cannot accurately convey low-level visual details.

**Key Challenge**: The specific visual representations of low-level degradations (such as the direction and density of rain streaks, or the intensity distribution of noise) are highly difficult to describe precisely in natural language, yet accurate information regarding degradation type and intensity is required to guide restoration.

**Goal**: To precisely describe visual degradation in a visual manner—letting visuals speak for visuals.

**Key Insight**: Constructing "visual instructions" by applying the degradation process to a standardized test chart (TE42 color chart) containing various visual primitives to generate visual descriptions that intuitively demonstrate degradation effects. These visual instructions naturally align with degradation patterns, bypassing the semantic gap of textual descriptions.

**Core Idea**: Replacing textual instructions with visual instructions to guide restoration, and performing diffusion in the degradation space (the LQ-HQ difference space) rather than the image space. This naturally aligns the degradation descriptions with the diffusion target, improving restoration efficacy and controllability.

## Method

### Overall Architecture

Defusion contains three core components: (1) Visual instruction construction: using the TE42 color chart as the "visual ground" and applying degradation to obtain visual instructions; (2) Degradation tokenizer: using a VQ-VAE encoder to compress visual instructions into discrete degradation tokens; (3) Degradation diffusion model: performing diffusion denoising in the degradation space $\mathbf{y}_0 = \mathbf{x}_{LQ} - \mathcal{T}_v(\mathbf{x}_{LQ})$, conditioned on the LQ image (via IRB) and visual instructions (via VIA). The restored result is obtained as $\hat{\mathcal{T}}_v(\mathbf{x}_{LQ}) = \mathbf{x}_{LQ} - \hat{\mathbf{y}}_0$.

### Key Designs

1. **Visual Instructions**:

    - **Function**: Precisely and intuitively describing the visual effects of image degradation.
    - **Mechanism**: Selecting four types of visual components from the TE42 chart (concentric textures, random textures, natural textures, color blocks) to randomly combine into a "visual ground," and applying the same degradation process (e.g., adding rain, adding noise) as the LQ image to obtain the degraded visual ground, which serves as the visual instruction. These visual instructions intuitively display the impact of degradation on various visual elements and are independent of image semantics (improving generalization). During inference, a clean visual ground ("null") is additionally encoded, and the difference between the two is used as the final degradation token.
    - **Design Motivation**: TE42 contains rich texture and color patterns that comprehensively reveal degradation effects. Unlike text, visual instructions are naturally aligned with low-level visual degradation patterns. Visual instructions for composite degradations also naturally display the combined effects (as shown in Fig. 1).

2. **Degradation Space Diffusion**:

    - **Function**: Performing diffusion in the LQ-HQ difference space instead of the image space to align the diffusion target with the degradation description.
    - **Mechanism**: Defining the degradation space $\mathcal{D}_v = \{\mathbf{x}_{LQ} - \mathcal{T}_v(\mathbf{x}_{LQ})\}$. The diffusion forward process is formulated as $p_t(\mathbf{y}_t | \mathbf{y}_0, \mathbf{x}_{LQ}, \mathbf{v}) = \mathcal{N}(\alpha_t \mathbf{y}_0, \sigma_t^2 I)$. A score model $\mathbf{s}_\theta(\mathbf{y}_t, t, \mathbf{x}_{LQ}, \mathbf{v})$ is trained, and during inference, the SDE is solved in reverse starting from $\mathbf{y}_1 \sim \mathcal{N}(0, I)$ to obtain $\hat{\mathbf{y}}_0$, yielding the final restoration $\hat{\mathbf{x}}_{HQ} = \mathbf{x}_{LQ} - \hat{\mathbf{y}}_0$.
    - **Design Motivation**: Three major advantages: (1) $\mathbf{y}_t$ is highly correlated with the degradation, leading to more accurate restoration; (2) The distribution in the degradation space is more consistent than in the image space, facilitating more stable training and requiring lower model capacity; (3) When visual instructions do not match the actual degradation, the model "does nothing," showing superior controllability. Furthermore, there are no special requirements for $\alpha_t$ and $\sigma_t$, allowing direct adaptation to pre-trained diffusion models.

3. **Condition Injection Mechanism (IRB + VIA)**:

    - **Function**: Injecting the LQ image and visual instructions separately into the diffusion U-Net.
    - **Mechanism**: **Image Restoration Bridge (IRB)**: A lightweight convolutional network encodes the LQ image into multi-scale feature maps, which are injected into each layer of the U-Net via AdaLN-Zero. **Visual Instruction Adapter (VIA)**: Replicates the cross-attention layers of the pre-trained text-to-image U-Net, where queries are U-Net feature maps and keys/values are the degradation tokens $\mathbf{z}_v$.
    - **Design Motivation**: IRB avoids using ControlNet or concatenation because the target in the degradation space differs significantly from the LQ image. VIA adopts an IP-Adapter-style design, preserving text conditioning capability and allowing text to act as an auxiliary guide.

### Loss & Training

The degradation tokenizer is trained using the VQ-VAE loss $\mathcal{L}_{inst} = \mathcal{L}_{rec} + \mathcal{L}_{VQ}$ (comprising MSE + LPIPS + hinge adversarial loss). The diffusion model is trained with standard denoising score matching. During training, visual instructions are replaced with a clean visual ground with a probability of 0.1. Multi-task joint training is performed.

## Key Experimental Results

### Main Results

**Comparison across 8 tasks (Defusion vs. best task-specific / best all-in-one)**:

| Task | Metric | Task-specific SOTA | All-in-one SOTA | Defusion |
|------|------|-------------------|----------------|----------|
| Motion Deblurring (GoPro) | PSNR | 33.20 (DiffIR) | 32.49 (MPerceiver) | **34.53** |
| Defocus Deblurring (DPDD) | PSNR | 26.18 (FocalNet) | 28.21 (NAFNet*) | **29.68** |
| Desnowing (Snow100K) | PSNR | 30.43 (WeatherDiff) | 31.02 (MPerceiver) | **32.11** |
| Dehazing (Dense-Haze) | PSNR | 17.07 (FocalNet) | 16.72 (NAFNet*) | **17.55** |
| Denoising (SIDD) | PSNR | 40.02 (Restormer)  | 40.60 (Restormer*) | **40.89** |

Defusion surpasses both task-specific and all-in-one methods on **all 8 tasks**, while being a single all-in-one model.

### Ablation Study

Based on the paper structure and data, the key ablations lie in the comparisons of visual vs. textual instructions and degradation space vs. image space. The degradation space exhibits clear advantages in training stability and controllability.

### Key Findings

- Visual instructions are more effective than textual instructions in low-level vision tasks because they naturally align with degradation patterns.
- Diffusion in the degradation space is more stable and accurate than in the image space.
- IRB injection via AdaLN-Zero outperforms ControlNet and concatenation methods.
- Subtracting the "null" visual ground enhances the discriminativeness of degradation tokens.
- A single all-in-one model outperforms all task-specific expert models, demonstrating the value of cross-task knowledge sharing.

## Highlights & Insights

- The design philosophy of "letting visuals speak for visuals" is elegant and practical, completely bypassing the semantic gap of textual descriptions.
- Utilizing the TE42 color chart as the visual ground is a clever choice, as industrial standard test charts are pre-designed with rich texture and color patterns.
- The concept of degradation space diffusion is highly inspiring—what is diffused and predicted should align with the conditioning information.
- The design of compressing visual instructions into discrete tokens via a VQ-VAE tokenizer aligns with the concept of tokenization in LLMs.

## Limitations & Future Work

- The construction of visual instructions relies on known degradation processes; "black-box" degradations (e.g., real-world unknown degradations) require an additional degradation detection step.
- Although the TE42 color chart is comprehensive, it is not specifically designed for image restoration and might overlook certain degradation patterns.
- The efficacy of visual instructions in generative restoration tasks such as super-resolution has not been discussed.
- The degradation space diffusion assumption $\mathbf{y}_0 = \mathbf{x}_{LQ} - \mathbf{x}_{HQ}$ assumes perfectly paired data; its generalization to unpaired scenarios requires further research.

## Related Work & Insights

- It complements **ResFlow** from the same research group—ResFlow employs deterministic flow for single-task restoration, while Defusion uses SDE diffusion for multi-task restoration.
- The cross-attention injection concept of **IP-Adapter** is leveraged to import visual instructions.
- The concept of visual instructions can be extended to other low-level vision tasks (e.g., video restoration, 3D restoration).

## Rating

- **Novelty**: 9/10 — Outstanding insights in both visual instructions and degradation space diffusion.
- **Experimental Thoroughness**: 9/10 — Comprehensive evaluation across 8 tasks on both synthetic and real-world datasets.
- **Writing Quality**: 8/10 — Clear motivation and detailed methodology.
- **Value**: 9/10 — Highly practical, with the all-in-one model completely outperforming expert models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[ICLR 2026\] RestoreVAR: Visual Autoregressive Generation for All-in-One Image Restoration](../../ICLR2026/image_restoration/restorevar_visual_autoregressive_generation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[CVPR 2026\] Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration](../../CVPR2026/image_restoration/degradation-consistent_test-time_adaptation_for_all-in-one_image_restoration.md)
- [\[CVPR 2026\] Retrieve-to-Restore: Efficient All-in-One Image Restoration with a Retrieval-Based Degradation Bank](../../CVPR2026/image_restoration/retrieve-to-restore_efficient_all-in-one_image_restoration_with_a_retrieval-base.md)

</div>

<!-- RELATED:END -->
