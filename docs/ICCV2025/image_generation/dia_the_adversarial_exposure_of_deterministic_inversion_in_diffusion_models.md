---
title: >-
  [Paper Note] DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models
description: >-
  [ICCV 2025][Image Generation][Adversarial Attack] This paper proposes DDIM Inversion Attack (DIA), which disrupts the image editing capability of diffusion models by directly attacking the DDIM inversion trajectory. DIA effectively defends against malicious deepfake generation and privacy-violating content synthesis, substantially outperforming existing defenses such as AdvDM and Photoguard across diverse editing methods.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Adversarial Attack"
  - "DDIM Inversion"
  - "Diffusion Model Defense"
  - "Image Editing Protection"
  - "Deepfake Defense"
date: 2026-05-08
content_hash: 4380e753b42cf882
---

# DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models

**Conference**: ICCV 2025
**arXiv**: [2510.00778](https://arxiv.org/abs/2510.00778)  
**Code**: [https://anonymous.4open.science/r/DIA-13419/](https://anonymous.4open.science/r/DIA-13419/)  
**Area**: Diffusion Models / AI Security
**Keywords**: Adversarial Attack, DDIM Inversion, Diffusion Model Defense, Image Editing Protection, Deepfake Defense

## TL;DR

This paper proposes DDIM Inversion Attack (DIA), which disrupts the image editing capability of diffusion models by directly attacking the DDIM inversion trajectory. DIA effectively defends against malicious deepfake generation and privacy-violating content synthesis, substantially outperforming existing defenses such as AdvDM and Photoguard across diverse editing methods.

## Background & Motivation

**Background**: Diffusion models, particularly DDIM-based models, have emerged as powerful tools for image generation and editing. The deterministic inversion of DDIM allows real images to be mapped back into latent space, enabling downstream editing operations such as style transfer and attribute manipulation.

**Limitations of Prior Work**: This powerful editing capability is exploited by malicious actors to generate misinformation and deepfake content, posing serious threats to privacy and copyright. Existing defenses such as AdvDM and Photoguard can partially disrupt the diffusion process, yet their optimization objectives are misaligned with the iterative denoising trajectory at inference time, resulting in limited protection.

**Key Challenge**: The perturbation direction optimized by existing defenses during training does not correspond to the actual trajectory traversed by DDIM inversion at test time. AdvDM targets the single-step denoising process, while Photoguard targets the encoder output; both neglect the multi-step deterministic inversion that is central to DDIM-based editing.

**Goal**: To design an adversarial attack method that directly targets the DDIM inversion trajectory, causing severe distortion in protected images after the inversion–editing pipeline and thereby preventing malicious editing.

**Key Insight**: The authors observe that DDIM inversion is a deterministic multi-step process in which errors introduced at early steps accumulate and propagate along the trajectory. Adding imperceptible perturbations to the input image to deviate the entire inversion trajectory from its normal path can fundamentally undermine subsequent editing operations.

**Core Idea**: Adversarial perturbations are optimized to attack the complete DDIM inversion trajectory rather than a single denoising step or encoder output, achieving defense objectives that are closely aligned with the actual inference procedure.

## Method

### Overall Architecture

The core pipeline of DIA is as follows. Given an image to be protected, the complete DDIM inversion process is first applied to obtain the latent trajectory. Adversarial perturbations are then optimized to cause the inverted latent representations to deviate substantially from the normal trajectory. Specifically, the input is the original image $x_0$, and the output is the protected image $x_0 + \delta$, where the perturbation $\delta$ causes any DDIM inversion–based editing operation to produce severe visual artifacts.

### Key Designs

1. **Trajectory Attack**:

    - **Function**: Applies adversarial attacks directly to the complete DDIM inversion trajectory.
    - **Mechanism**: Unlike AdvDM, which attacks a single denoising step, or Photoguard, which attacks the image encoder output, DIA computes gradients at every step of the DDIM inversion process $x_0 \to x_1 \to \cdots \to x_T$ and integrates them into a unified perturbation on the input image. The optimization objective is to maximize the deviation of the adversarial inversion trajectory from the normal trajectory at each timestep: $\max_{\delta} \sum_{t=1}^{T} \|f_\theta(x_t^{adv}) - f_\theta(x_t)\|$, where $x_t^{adv}$ denotes the adversarial sample's inversion result at step $t$.
    - **Design Motivation**: Because DDIM inversion is deterministic, small deviations introduced at early steps are amplified in subsequent steps; attacking the full trajectory is therefore more effective than attacking a single step.

2. **Integrated Gradient Optimization**:

    - **Function**: Efficiently computes adversarial gradients through the entire inversion trajectory.
    - **Mechanism**: The chain rule is applied to differentiate the cumulative loss with respect to the input image across all timesteps, and gradient accumulation is employed to avoid the memory overhead of storing all intermediate activations simultaneously. Perturbations are iteratively optimized via PGD (Projected Gradient Descent), with $\ell_\infty$ norm projection applied at each update to ensure visual imperceptibility (e.g., $\epsilon = 8/255$).
    - **Design Motivation**: End-to-end optimization of the entire DDIM trajectory requires storing all intermediate activations and incurs prohibitive memory costs; the integrated gradient strategy preserves attack effectiveness while significantly reducing memory consumption.

3. **Edit-agnostic Defense**:

    - **Function**: Ensures that the defense transfers across different editing methods.
    - **Mechanism**: Because DIA attacks the inversion step shared by all DDIM inversion–based editing methods rather than any specific editing operation, the generated adversarial perturbation is effective against diverse editing methods (e.g., SDEdit, Prompt2Prompt, Null-text Inversion) without requiring separate optimization for each.
    - **Design Motivation**: In realistic threat scenarios, adversaries may employ arbitrary editing methods; a practical defense must generalize across editing pipelines.

### Loss & Training

The total loss of DIA combines a trajectory deviation loss and a perceptual loss. The trajectory deviation loss measures the sum of L2 distances between the adversarial inversion trajectory and the normal trajectory at each timestep, encouraging maximal deviation. Optimization follows PGD iterations, performing gradient ascent (maximizing deviation) at each step followed by projection onto the $\ell_\infty$ ball. The perturbation budget is typically set to $\epsilon = 8/255$, and the number of iterations is chosen in the range of 50–200 depending on available computational resources.

## Key Experimental Results

### Main Results

| Editing Method | Metric (LPIPS↑) | DIA | AdvDM | Photoguard | No Defense |
|---|---|---|---|---|---|
| SDEdit | LPIPS | **0.682** | 0.423 | 0.371 | 0.215 |
| Prompt2Prompt | LPIPS | **0.715** | 0.456 | 0.398 | 0.198 |
| Null-text Inv. | LPIPS | **0.691** | 0.441 | 0.385 | 0.203 |
| InstructPix2Pix | LPIPS | **0.658** | 0.412 | 0.356 | 0.187 |

### Ablation Study

| Configuration | LPIPS (SDEdit) | Description |
|---|---|---|
| Full DIA | **0.682** | Full model, attacking the complete trajectory |
| w/o Trajectory (single-step) | 0.478 | Only attacks single-step denoising; large performance drop |
| w/o Integrated Gradient | 0.623 | Uses naive gradient descent; slight performance drop |
| First half of trajectory only | 0.591 | Attacks $t \in [T/2, T]$ only; second half also contributes |
| Second half of trajectory only | 0.548 | Attacks $t \in [0, T/2]$ only; first half is more critical |
| $\epsilon = 4/255$ | 0.542 | Halved perturbation budget limits effectiveness |

### Key Findings

- Trajectory attack improves over single-step attack by more than 40%, demonstrating the necessity of exploiting the complete deterministic inversion trajectory of DDIM.
- DIA substantially outperforms AdvDM and Photoguard across all tested editing methods without requiring per-method tuning.
- The first half of the trajectory (steps closer to the original image) contributes more to the attack, as early deviations accumulate and amplify through subsequent steps.
- A perturbation budget of $\epsilon = 8/255$ strikes a favorable balance between visual imperceptibility and defense effectiveness.

## Highlights & Insights

- **Elegant Trajectory-Level Attack Design**: Unlike step-wise or single-point attacks, DIA exploits the deterministic nature of DDIM inversion to attack the entire trajectory. This approach of targeting the most critical component of the pipeline is precise and efficient, and is worth adapting to other adversarial attack scenarios involving iterative processes.
- **Alignment Between Defense and Inference**: The central insight of DIA is that the optimization objective of a defense method must be aligned with the actual operations performed at inference time. This principle generalizes to adversarial defense in any iterative generative pipeline.
- **Edit-Agnostic Generalization**: By targeting the shared inversion process rather than any specific editing operation, DIA naturally generalizes across editing methods. This design philosophy is transferable to protecting other generative models from misuse.

## Limitations & Future Work

- The computational overhead is substantial: protecting a single image requires multiple forward–backward passes through the complete DDIM inversion trajectory.
- The defense effectiveness against editing methods that bypass DDIM inversion (e.g., methods relying solely on text-conditioned generation) has not been validated.
- As diffusion models and editing techniques evolve rapidly, the robustness of DIA against future novel editing methods requires ongoing evaluation.
- Potential directions for improvement include more efficient gradient estimation methods to reduce computational cost, and extension to protecting video content from diffusion model–based editing.

## Related Work & Insights

- **vs. AdvDM**: AdvDM targets the noise prediction network in the single-step denoising process, which is misaligned with the multi-step inversion trajectory used at inference time, yielding weaker protection. DIA attacks the full trajectory directly, achieving superior alignment and effectiveness.
- **vs. Photoguard**: Photoguard attacks the VAE encoder or initial encoding output, but these perturbations may be washed out during multi-step inversion. DIA targets the inversion process itself, making perturbation effects more persistent.
- **vs. Glaze/Mist**: Glaze and Mist primarily protect artistic style from imitation, whereas DIA defends against image content editing. The two approaches have different objectives but complementary technical insights.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Trajectory-level attack is a meaningful and well-motivated innovation, though the overall framework remains grounded in PGD-based adversarial attack.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple editing methods and baselines; ablation study is well-designed.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and the method is described accessibly.
- **Value**: ⭐⭐⭐⭐ Defending against the misuse of diffusion models is an important practical problem, and DIA provides an effective protection tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](../../CVPR2025/image_generation/lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [\[ICCV 2025\] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICML 2026\] Adversarial Flow Models](../../ICML2026/image_generation/adversarial_flow_models.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
