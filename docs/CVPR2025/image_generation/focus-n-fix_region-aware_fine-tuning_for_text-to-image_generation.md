---
title: >-
  [Paper Note] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation
description: >-
  [CVPR 2025][Image Generation][Region-aware fine-tuning] Proposes Focus-N-Fix, a region-aware fine-tuning method for T2I models. By localizing problematic regions and constraining non-problematic regions to remain unchanged, it achieves precise correction of local quality issues (such as artifacts, over-sexualization, and violence) while avoiding catastrophic forgetting and reward hacking induced by global fine-tuning.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Region-aware fine-tuning"
  - "reward fine-tuning"
  - "safe generation"
  - "artifact correction"
  - "diffusion models"
date: 2026-05-08
content_hash: 0fc08e4306fc24c2
---

# Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2501.06481](https://arxiv.org/abs/2501.06481)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Region-aware fine-tuning, reward fine-tuning, safe generation, artifact correction, diffusion models

## TL;DR
Proposes Focus-N-Fix, a region-aware fine-tuning method for T2I models. By localizing problematic regions and constraining non-problematic regions to remain unchanged, it achieves precise correction of local quality issues (such as artifacts, over-sexualization, and violence) while avoiding catastrophic forgetting and reward hacking induced by global fine-tuning.

## Background & Motivation

**Background**: Current quality improvements in text-to-image (T2I) generation models mostly rely on Reinforcement Learning from Human/AI Feedback (RLHF/RLAIF), which trains reward models to evaluate the quality of generated images and then fine-tunes the models using methods like DRaFT or DPO to improve reward scores.

**Limitations of Prior Work**: Existing reward fine-tuning methods suffer from three main issues: (1) optimizing for a specific quality dimension (e.g., safety) often degrades other dimensions (e.g., prompt alignment) or introduces new artifacts, leading to catastrophic forgetting; (2) models may resort to "reward hacking" by finding shortcuts to boost scores without addressing the actual issues, such as generating completely different images to evade artifacts; (3) using coarse-grained scalar rewards fails to provide precise guidance for pixel-level local improvements.

**Key Challenge**: The mismatch between global optimization strategies and local issues. Safety issues and artifacts usually appear only in local regions of an image, yet existing methods apply global optimization pressure to the entire image, causing uncontrollable shifts in the model's solution space.

**Goal**: How to precisely correct problematic regions when improving specific quality dimensions, while keeping the rest of the regions unchanged?

**Key Insight**: The authors observe that problematic regions (such as artifacts or over-exposed body parts) can be localized. Using existing quality heatmap models or gradient saliency maps can yield a problematic mask. During fine-tuning, only modifications within these regions are allowed, while the rest remains consistent with the pre-trained model.

**Core Idea**: To incorporate region constraints into the reward fine-tuning objective, penalizing pixel changes in non-problematic regions to achieve focused correction.

## Method

### Overall Architecture
Focus-N-Fix is based on the DRaFT framework. Given a text prompt and noise, it simultaneously generates two images using both the pre-trained model and the fine-tuning model. A localization method is applied to the output of the pre-trained model to mark the problematic region mask $\mathcal{M}$. Subsequently, a region constraint is added to the optimization objective: while maximizing the reward score, it penalizes pixel differences in regions outside the mask. Fine-tuning only updates the LoRA parameters. During inference, no heatmap input is required, and standard forward propagation is sufficient.

### Key Designs

1. **Region-Constrained Objective Function**:

    - **Function**: Forces non-problematic regions to remain unchanged while improving the reward score.
    - **Mechanism**: The objective function is formulated as $J(\theta) = r(\hat{I}, \mathbf{c}) - \beta \|(1-\mathcal{M}(\hat{I}_0)) \odot (\hat{I}_0 - \hat{I})\|_F$, where the first term maximizes the reward, and the second term uses the Frobenius norm to penalize changes in regions outside the mask. The hyperparameter $\beta$ controls the strength of the constraint.
    - **Design Motivation**: This is a minimal modification to the DRaFT objective, elegantly converting global optimization into local correction. Since pixels in non-problematic regions are "locked", the model is forced to search for better solutions exclusively within the problematic regions, effectively preventing reward hacking.

2. **Problematic Region Localization**:

    - **Function**: Generates binary masks indicating the regions in the image that require correction.
    - **Mechanism**: Supports two localization approaches: (1) using models that directly predict heatmaps (such as the Rich Human Feedback model) to identify artifacts and misaligned regions; (2) computing gradient saliency maps (Grad-CAM style) on scalar reward models (such as safety classifiers) and mapping the gradients back to the image space. The heatmaps are converted into binary masks via thresholding, and a dilation operation is applied to slightly relax the boundaries of the modified regions.
    - **Design Motivation**: The gradient saliency map approach allows any classifier or reward model that only outputs scalar scores to be utilized, greatly expanding the scope of applicability without requiring additional training of specialized localization models.

3. **Parameter-Efficient Fine-Tuning with LoRA**:

    - **Function**: Updates only the low-rank adaptation parameters while keeping the main model weights frozen.
    - **Mechanism**: Utilizes a LoRA decomposition with rank=64, combined with a DRaFT-K (K=2) truncated backpropagation strategy to propagate gradients only through the last 2 steps of the sampling chain.
    - **Design Motivation**: LoRA itself restricts the parameter update space, which, combined with the region constraint, forms a double safeguard to further reduce the risk of catastrophic forgetting. It adds no computational overhead during inference.

### Loss & Training
The total loss is the aforementioned region-constrained objective function. During training, noise is sampled for each prompt, and images are generated by both the pre-trained and fine-tuning models. The mask is obtained from the output of the pre-trained model, and gradients are backpropagated to update the LoRA parameters. The mask is only utilized during the training phase and is not required during inference.

## Key Experimental Results

### Main Results

| Reward Model | Method | Safety Score↑ | Artifact Score↑ | T2I Alignment Score↑ |
|----------|------|-----------|----------|-------------|
| Over-sexualization | SD v1.4 (baseline) | 0 | 0 | 0 |
| Over-sexualization | SLD | 0.439 | 0.092 | -0.081 |
| Over-sexualization | DRaFT | 0.361 | -0.097 | -0.146 |
| Over-sexualization | **Focus-N-Fix** | **0.479** | 0.042 | **0.004** |
| Artifacts | DRaFT | - | 0.207 | 0.012 |
| Artifacts | **Focus-N-Fix** | - | **0.294** | **0.100** |

### Ablation Study (Voting-based Human Evaluation)

| Method | Safety Improvement↑ | Safety Degradation↓ | Degradation in Other Dimensions↓ |
|------|----------|----------|-------------|
| SLD | 63% | 8% | 41% |
| DRaFT | 59% | 11% | 52% |
| **Focus-N-Fix** | **69%** | **1%** | **26%** |

### Key Findings
- Focus-N-Fix significantly outperforms all baselines with a safety improvement rate of 69%, while its safety degradation rate is only 1% (compared to 11% for DRaFT).  
- The most prominent advantage is that the "Degradation in Other Dimensions" metric is substantially lower than that of competitors—only 26% for Focus-N-Fix, compared to 52% for DRaFT.  
- In the PartiPrompts catastrophic forgetting test, the decrease in VNLI alignment scores for Focus-N-Fix in categories such as basic, perspective, and properties & positioning is significantly smaller than that of DRaFT.  
- In artifact experiments, Focus-N-Fix simultaneously improves T2I alignment (+0.100), as correcting text-rendering artifacts enhances prompt alignment.

## Highlights & Insights
- **Elegant Design of Region Constraints**: A single Frobenius norm penalty term achieves the effect of "only modifying what needs to be modified." It is simple to implement but highly effective. This approach can be transferred to any fine-tuning scenario for generative models requiring local improvements.
- **Gradient Saliency Map Localization Scheme**: Converting an arbitrary scalar classifier into a regional localizer is highly practical. For scenarios lacking fine-grained annotations (such as safety filters), this provides zero-cost localization capability.
- **Decoupling of Training and Inference**: The mask is used only during training, incurring zero extra overhead during inference. The fine-tuned model "internalizes" the correction capability, indicating that LoRA learns generalizable abilities to fix specific problematic regions rather than hard-coding mask information.

## Limitations & Future Work
- Validated only on Stable Diffusion v1.4, lacking experiments on more recent models such as SDXL and SD3.
- Regional localization depends heavily on mask quality; if the localization model itself is inaccurate, the correction effectiveness may degrade.
- For global issues (such as overall style shift or global color issues), regional constraints might instead restrict the model's room for improvement.
- The choice of hyperparameter $\beta$ affects performance, and the paper lacks a thorough sensitivity analysis.
- Potential improvements: extending regional constraints to concept erasing tasks or combining them with preference learning methods like DPO.

## Related Work & Insights
- **vs DRaFT**: DRaFT globally optimizes rewards, whereas Focus-N-Fix incorporates region constraints on top of it. Focus-N-Fix is essentially a "constrained version" of DRaFT, but it is precisely this constraint that prevents over-optimization.
- **vs SLD (Safe Latent Diffusion)**: SLD addresses safety issues via concept erasing, but it is prone to erasing correlated safe concepts, causing a substantial drop in T2I alignment. Focus-N-Fix does not erase concepts; it only modifies problematic pixels.
- **vs Image Editing Methods**: Image editing works as a post-processing scheme without improving the model itself, whereas Focus-N-Fix directly enhances the model's capabilities, requiring no editing steps during inference.

## Rating
- Novelty: ⭐⭐⭐⭐ The region-constrained idea is intuitive and clear; although the technical barrier is not high, it successfully addresses a practical pain point.
- Experimental Thoroughness: ⭐⭐⭐⭐ The human evaluation is comprehensive (100 prompts × 11 annotators), but it lacks experiments on newer models and more diverse benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the problem definition and motivation are extremely well-articulated.
- Value: ⭐⭐⭐⭐ Direct application value for the safe deployment of T2I models; the region-constrained concept is widely transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RAD: Region-Aware Diffusion Models for Image Inpainting](rad_region-aware_diffusion_models_for_image_inpainting.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
