---
title: >-
  [Paper Note] Does FLUX Already Know How to Perform Physically Plausible Image Composition?
description: >-
  [ICLR2026][Image Generation][image composition] This paper proposes SHINE, a training-free image composition framework that leverages the intrinsic physical priors of pretrained T2I models (e.g.…
tags:
  - "ICLR2026"
  - "Image Generation"
  - "image composition"
  - "training-free"
  - "diffusion model"
  - "FLUX"
  - "physically plausible"
date: 2026-05-08
content_hash: 722bbc82c4f80f46
---

# Does FLUX Already Know How to Perform Physically Plausible Image Composition?

**Conference**: ICLR2026  
**arXiv**: [2509.21278](https://arxiv.org/abs/2509.21278)  
**Code**: [GitHub](https://github.com/ZhumingLian/SHINE)  
**Area**: Image Generation  
**Keywords**: image composition, training-free, diffusion model, FLUX, physically plausible

## TL;DR
This paper proposes SHINE, a training-free image composition framework that leverages the intrinsic physical priors of pretrained T2I models (e.g., FLUX) via three components — Manifold-Steered Anchor Loss, Degradation-Suppression Guidance, and Adaptive Background Blending — to achieve high-quality object insertion under complex lighting conditions (shadows, water reflections, etc.).

## Background & Motivation
Image composition aims to seamlessly insert a user-specified object into a new scene. Despite remarkable progress in multimodal large models (GPT-5, Gemini-2.5, etc.), they still perform poorly on image composition tasks, frequently exhibiting imprecise object placement, lighting inconsistency, and subject identity drift.

Existing methods face two major challenges:

1. **Limitations of training-based methods**: Fine-tuned composition models are constrained by the quality of synthetic training data, struggle with complex lighting (e.g., accurate shadow generation and water reflections), and are tied to fixed resolutions. A key observation is that these problems do not exist in foundation models, suggesting that physical priors are already encoded within them but are degraded during fine-tuning.
2. **Bottlenecks of training-free methods**: (a) Inversion-based methods lock the object pose to the reference image orientation and perform poorly on CFG-distilled models (e.g., FLUX); (b) attention-surgery-based methods are unstable and sensitive to hyperparameters.

Core insight: Modern T2I diffusion models have encoded rich physical and resolution priors; the key challenge lies in releasing these priors without corrupting them.

## Core Problem
How to fully exploit the physical priors of pretrained T2I models — without additional training, inversion, or attention manipulation — to achieve physically plausible (with correct shadows, reflections, etc.) high-fidelity image composition?

## Method

### Overall Architecture
SHINE (Seamless, High-fidelity Insertion with Neutralized Errors) consists of three core components, designed to be model-agnostic and relying only on standard functionality of modern generative models.

### 1. Non-Inversion Latent Preparation
Rather than relying on traditional image inversion, a single forward diffusion step is used to obtain the noisy latent:

- A VLM (BLIP-3) generates a description of the subject image.
- An inpainting model (FLUX.1-Fill) generates an initial image $\bm{x}^{\text{init}}$ in the target region of the background using the description.
- Noise is added via a single forward diffusion step: $\bm{z}_t = (1 - \sigma_t)\bm{z}^{\text{init}} + \sigma_t \bm{\epsilon}$

This avoids the pose-locking problem caused by inversion and allows the object to appear in a scene-appropriate orientation.

### 2. Manifold-Steered Anchor (MSA) Loss
Core Idea: Pretrained customization adapters (e.g., IP-Adapter, InstantCharacter) are used as implicit priors to optimize the noisy latent during denoising, so that the result remains faithful to the reference subject while preserving the background structure.

$$\mathcal{L}_{\text{MSA}}(\bm{z}_t) = \|\bm{v}_{\bm{\theta}+\bm{\Delta\theta}}(\bm{z}_t, t, \bm{c}, \bm{z}^{\text{subj}}) - \text{sg}[\tilde{\bm{v}}_t]\|_2^2$$

- $\tilde{\bm{v}}_t$ is the base model's prediction on the original latent, serving as a fixed anchor to preserve background structure.
- $\bm{v}_{\bm{\theta}+\bm{\Delta\theta}}$ is the adapter-augmented model prediction, guiding the latent toward the reference subject.
- Gradients are computed only within the masked region; the Jacobian term is omitted (analogous to SDS).

Key mathematical intuition: Optimizing the latent with respect to a frozen generative model implicitly projects it onto the learned data manifold of that model.

### 3. Degradation-Suppression Guidance (DSG)
Addresses occasional color oversaturation and identity consistency degradation arising from MSA optimization.

$$\bm{v}_t^{\text{dsg}} = \bm{v}_t + \eta(\bm{v}_t - \bm{v}_{\bm{\theta}+\Delta\bm{\theta}}^{\text{neg}})$$

Key finding: Negative text prompts are ineffective for FLUX (the model generates high-quality images even from nonsensical text). The authors systematically investigate the effect of applying blur perturbations to different attention components:

- Blurring $Q_{\text{txt}}$/$K_{\text{txt}}$/$V_{\text{txt}}$: negligible effect
- Blurring $V_{\text{img}}$: complete output collapse
- Blurring $K_{\text{img}}$: moderate impact
- **Blurring $Q_{\text{img}}$: produces clear degradation while preserving structure** (optimal choice)

Theoretically, blurring $Q_{\text{img}}$ is equivalent to blurring the self-attention weights, consistent with the known finding that suppressing attention activations reduces output quality.

### 4. Adaptive Background Blending (ABB)
Replaces the fixed user mask with a semantically guided attention mask to eliminate visible seams at composition boundaries:

- Early denoising steps ($t > \tau$): adaptive mask $M^{\text{attn}}$ derived from cross-attention maps
- Late denoising steps ($t \leq \tau$): revert to user mask $M^{\text{user}}$

Using the adaptive mask in early steps eliminates seams, while reverting to the user mask in late steps prevents cropping of shadows and reflections.

## Key Experimental Results

### Benchmark
The paper introduces the ComplexCompo benchmark (300 composition pairs), covering multi-resolution, portrait/landscape orientations, and challenging conditions such as low light, harsh illumination, complex shadows, and water reflections — addressing the limitations of existing 512×512 fixed-resolution benchmarks.

### Main Results (DreamEditBench, 220 pairs)

| Method | DINOv2↑ | DreamSim↓ | ImageReward↑ | VisionReward↑ |
|--------|---------|-----------|--------------|---------------|
| AnyDoor | 0.7283 | 0.3764 | 0.4511 | 3.3946 |
| UniCombine | 0.7332 | 0.3984 | 0.4565 | 3.6108 |
| EEdit | 0.6590 | 0.6160 | 0.0216 | 3.3606 |
| **SHINE-Adapter** | **0.7415** | **0.3730** | **0.5709** | **3.6234** |
| **SHINE-LoRA** | **0.7452** | **0.3577** | **0.5906** | **3.6161** |

SHINE outperforms all baselines across human preference alignment metrics (DreamSim, ImageReward, VisionReward). Advantages are even more pronounced on ComplexCompo, where competing methods degrade substantially on non-square resolutions and complex scenes.

### Ablation Study
- MSA contributes most significantly, substantially improving subject identity consistency (DINOv2: 0.6745 → 0.7204).
- DSG improves image quality scores (ImageReward and VisionReward).
- ABB effectively eliminates visible seams (clear visual improvement, though LPIPS/SSIM do not fully capture this).

## Highlights & Insights
1. **Training-free framework design**: Fully leverages pretrained model priors, avoiding the synthetic data contamination problems of data-driven approaches.
2. **Elegant DSG design**: Systematic experiments identify blurring $Q_{\text{img}}$ as the optimal strategy for constructing negative velocity, with a theoretically clean explanation.
3. **Comprehensive model-agnosticism**: Compatible with FLUX, SDXL, SD3.5, and PixArt, relying only on standard model functionality.
4. **ComplexCompo benchmark contribution**: Fills a gap in evaluation for image composition under complex lighting conditions.

## Limitations & Future Work
1. When the inpainting prompt specifies an incorrect color, the final result inherits that error.
2. Subject similarity to the reference depends on the quality of the customization adapter; LoRA requires per-concept fine-tuning at test time.
3. MSA optimization requires multiple forward passes ($k$ gradient descent steps), incurring non-trivial computational overhead.
4. The paper does not sufficiently discuss the dependence on VLM description quality.

## Related Work & Insights
- **vs. training-based methods (AnyDoor, UniCombine)**: Training-based methods are constrained by synthetic data quality and perform poorly under complex lighting; AnyDoor tends to copy-paste subjects, producing unnatural results.
- **vs. training-free inversion methods (TF-ICON, EEdit)**: Inversion locks object pose and degrades on CFG-distilled models such as FLUX.
- **vs. SDS**: The MSA loss borrows the strategy of omitting the Jacobian term from SDS, but serves a different purpose — SDS targets 3D generation, while MSA addresses constrained 2D composition.

The core principle — "pretrained models already possess the required priors; the key is how to release them" — is broadly applicable and transferable to video composition, 3D scene editing, and related tasks. The systematic analysis of FLUX attention components in DSG provides valuable reference for understanding the internal workings of MMDiT architectures. The manifold projection idea (using a frozen model to constrain the optimization direction) can be applied to other tasks requiring a balance between fidelity and editing flexibility.

## Rating
- Novelty: 8/10 — Each of the three components is innovative; the attention perturbation analysis in DSG is particularly elegant.
- Experimental Thoroughness: 9/10 — Multiple benchmarks, metrics, and baselines; complete ablations; a new benchmark is introduced.
- Writing Quality: 8/10 — Clear structure with effective integration of mathematical derivations and intuitive explanations.
- Value: 8/10 — The training-free approach is highly practical; the ComplexCompo benchmark has long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)
- [\[ICLR 2026\] Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study](does_semantic_noise_initialization_transfer_from_images_to_videos_a_paired_diagn.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](../../CVPR2026/image_generation/consistcompose_multimodal_layout_control.md)
- [\[ICLR 2026\] Steer Away From Mode Collisions: Improving Composition In Diffusion Models](steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
