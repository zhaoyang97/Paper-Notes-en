---
title: >-
  [Paper Note] FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior
description: >-
  [ECCV 2024][Image Generation][Image Composition] FreeCompose is proposed, leveraging the generative prior of pretrained diffusion models to achieve generic zero-shot image composition. It unifies image harmonization (appearance editing) and semantic image composition (semantic editing) under a single framework without any extra training.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image Composition"
  - "Diffusion Prior"
  - "Zero-shot"
  - "Image Harmonization"
  - "Semantic Image Composition"
date: 2026-05-08
content_hash: 876cd810277b6bc4
---

# FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior

**Conference**: ECCV 2024  
**arXiv**: [2407.04947](https://arxiv.org/abs/2407.04947)  
**Code**: [GitHub](https://github.com/aim-uofa/FreeCompose)  
**Area**: Image Generation  
**Keywords**: Image Composition, Diffusion Prior, Zero-shot, Image Harmonization, Semantic Image Composition

## TL;DR

FreeCompose is proposed, leveraging the generative prior of pretrained diffusion models to achieve generic zero-shot image composition. It unifies image harmonization (appearance editing) and semantic image composition (semantic editing) under a single framework without any extra training.

## Background & Motivation

Image composition is a fundamental task in computer vision, aiming to merge foreground objects from one image with the background of another to generate a natural and coherent image. It finds widespread application in image editing, artistic design, game development, and virtual reality.

Key limitations faced by existing methods:
- **Data Scarcity**: Traditional learning-based methods rely on triplets of (foreground, background, composition) training data, which are difficult to acquire, thereby limiting generalization capability.
- **Task Fragmentation**: Image harmonization (adjusting only low-level statistics like color and illumination) and semantic image composition (involving structural changes) are typically handled separately by different models, lacking a unified framework.
- **Domain Specificity**: Existing models are usually trained on domain-specific datasets, making it difficult to generalize to open-world scenarios.

The authors identify a key insight: **pretrained diffusion models can automatically detect unnatural boundary regions caused by simple copy-and-paste operations**. During the denoising process, these regions are identified as low-density areas (corresponding to larger gradient updates), which align closely with actual unharmonized regions. Based on this discovery, the composition process can be optimized towards high-density regions to achieve natural image composition.

## Method

### Overall Architecture

The core idea of FreeCompose is to leverage the image prior of pretrained diffusion models for generic image composition, consisting of three stages:

1. **Object Removal Stage**: Removes objects within a designated region $M_s$ from the background image $I_s$ to generate a clean background $I_b$.
2. **Image Harmonization Stage**: Integrates the target object into the background and adjusts lighting, color, etc., to achieve natural blending.
3. **Semantic Image Composition Stage**: Performs semantic-level structural editing based on additional conditions (text or sketch).

Each stage shares the same general pipeline: input image $I_i$, original prompt $P_o$, and target prompt $P_t$, which optimizes image pixels through specific loss functions.

### Key Designs

#### 1. Core Observation: Diffusion Prior Automatically Locates Unnatural Regions

The authors conduct key experimental verifications: after adding varying levels of noise to simple copy-and-pasted composite images, a frozen diffusion model predicts a gradient to update the image. The results reveal:
- Low-density regions (areas with large gradient updates) correspond highly to the unharmonized regions caused by copy-and-paste operations.
- This implies that the prior of the diffusion model can automatically "sense" which regions are unnatural.

#### 2. Optimization Framework Based on DDS Loss

The design is optimized based on the Delta Denoising Score (DDS) loss. The gradient form of the DDS loss is:

$$\nabla_\theta \mathcal{L}_{DDS} = (\epsilon_\phi^w(\mathbf{z_t}, y, t) - \epsilon_\phi^w(\hat{\mathbf{z_t}}, \hat{y}, t)) \frac{\partial \mathbf{z_t}}{\partial \theta}$$

By utilizing two image-text pairs, the differential denoising score is calculated using matched timesteps and noise to guide the direction of image optimization.

#### 3. Mask-Guided Loss in the Object Removal Stage

In the object removal stage, using only the DDS loss is insufficient to completely eliminate the object. The authors propose an innovative mask-guided DDS loss:
- In the self-attention layer of the UNet, Key (K) and Value (V) values corresponding to the target region are selectively discarded using a mask $M$.
- Specifically: the mask is resized to the sequence length $l$, and indices where $v_i > threshold$ are selected to replace the semantic information of the masked region.

The gradient of the mask-guided loss is:
$$\nabla_\theta \mathcal{L}_{DDS}^{rmv} = (\epsilon_\phi^w(\mathbf{z_t}, y, t) - \epsilon_\phi^w(\hat{\mathbf{z_t}}, \hat{y}, t, M)) \frac{\partial \mathbf{z_t}}{\partial \theta}$$

The total loss for object removal is:
$$\mathcal{L}_{rmv} = \mathcal{L}_{DDS}^{rmv}(I_s, I_t, P_o, P_t, M_s) + \lambda_{per} \mathcal{L}_{per}(I_s \otimes M_s', I_t \otimes M_s')$$

The perceptual loss is used to maintain consistency in the background region outside the mask.

#### 4. Image Harmonization Stage

After copying and pasting the target object onto the clean background to form $I_p$, three loss functions are used for optimization:

$$\mathcal{L}_{har} = \mathcal{L}_{DDS}(I_p, I_t, P_o, P_t) + \lambda_{bak}\mathcal{L}_{per}(I_p \otimes M_p', I_t \otimes M_p') + \lambda_{for}\mathcal{L}_{per}(I_p \otimes M_p, I_t \otimes M_p)$$

Key Design: **Foreground and background utilize perceptual losses with different weights**. A smaller foreground weight $\lambda_{for}=0.1$ is employed (allowing more variations to blend into the background), while a larger background weight $\lambda_{bak}=0.3$ is used (to keep the background stable). The default prompts are empty strings and "A harmonious scene.".

#### 5. Semantic Image Composition Stage

This stage supports conditional inputs in the form of text or other modalities (sketch, canny edge converted via T2I-Adapter). The key innovation is a **K, V value replacement strategy** used to maintain object identity consistency:

$$\text{Attention}(Q, K_i, V_i), \text{if } t > T \text{ and } l > L$$

where $T=400$ and $L=10$ are hyperparameters controlling the start time and layer depth of the replacement. The replacement is performed only in the early stages of optimization and in deeper layers to balance both identity consistency and editing flexibility.

### Loss & Training

**Key Feature: Training-free**, requiring no training of the diffusion model itself. Composite image generation is accomplished by directly optimizing image pixels (in the latent space).

Optimization Details:
- Stable Diffusion V2.1 is used as the pretrained model for real images, while AnyLoRA is used for anime/cartoon styles.
- The input resolution is aligned to 512×512.
- Adam optimizer is employed with a fixed learning rate of $5 \times 10^{-2}$.
- Object removal: 150 steps. The DDS loss is multiplied by 0.2 outside the mask to suppress background changes, with $\lambda_{per}=0.3$.
- Image harmonization: 200 steps, with $\lambda_{bak}=0.3$ and $\lambda_{for}=0.1$.
- Semantic composition: 500 steps for text conditions; 200 steps for sketch/canny conditions.

## Key Experimental Results

### Main Results

The paper quantitatively evaluates the performance via user studies. Each task involves 20+ volunteers across 5 cases, scored on a scale of 1 to 5.

| Method | Image Harmony ↑ | Object Removal Completeness ↑ |
|------|------------|---------------|
| Repaint | 3.24±1.23 | 3.82±1.35 |
| SD Inpainting | 2.99±1.37 | 3.55±1.34 |
| Lama | 3.47±1.16 | 4.14±0.94 |
| **FreeCompose** | **3.85±1.01** | **4.47±0.73** |

| Method | Image Harmony ↑ | Object Identity Preservation ↑ |
|------|------------|-------------|
| Diff Harmonization | 3.11±1.04 | 3.83±1.10 |
| DucoNet | 3.14±1.17 | **4.16±1.04** |
| **FreeCompose** | **3.69±1.07** | 4.11±0.92 |

### Ablation Study

Decomposition validation of components in the object removal stage:

| Component | Effect |
|------|------|
| Perceptual loss only | Image remains almost unchanged |
| Vanilla DDS only | Object partially changes but cannot be completely eliminated |
| DDS + mask | Object successfully removed, but background is affected |
| DDS + mask + perceptual | Object removed while background is preserved (Full Method) |

Ablation in the image harmonization stage:

| Component | Effect |
|------|------|
| Perceptual only | Remains consistent with raw copy-and-paste, no blending |
| DDS only | Blending achieved but foreground/background features may be lost |
| Full Method | Optimal balance among blending degree, object identity, and background features |

### Key Findings

1. **Mask-guided K,V manipulation** is crucial for successful object removal—DDS loss and text prompts alone cannot completely eliminate the object.
2. **Separated foreground and background perceptual loss** effectively balances the trade-off between blending degree and fidelity.
3. The method can be plugged-and-played into more powerful models like SDXL to obtain better performance (especially for reflection/shadow effects during the harmonization stage).
4. Runtime: On an RTX 3090 FP16, the first 50 steps take approximately 30 seconds, and each subsequent 50 steps take about 25 seconds.

## Highlights & Insights

1. **Unified Framework**: Offers the first unified, zero-shot framework for both image harmonization and semantic image composition, eliminating the need to train separate models for different tasks.
2. **Universality of Key Insight**: The discovery that diffusion priors can automatically identify unnatural regions holds broad theoretical and practical value.
3. **Mask-guided KV Manipulation**: Controlling the flow of semantic information through masks at the self-attention layer is a novel approach, extendable to other tasks requiring spatial control.
4. **Diverse Applications**: Beyond basic composition, the framework can be applied to downstream tasks such as object stylization and multi-character customization.

## Limitations & Future Work

1. **Object Identity Preservation Inferior to Training-based Methods**: To achieve better blending in image harmonization, the weight of the foreground perceptual loss is lowered, which slightly degrades identity preservation compared to DucoNet.
2. **Slow Execution Speed**: Hundreds of optimization iterations are required per stage, limiting real-time application.
3. **Dependence on Mask Quality**: High-quality input object masks are required, while automatic segmentation in complex scenes is not addressed.
4. **Future Directions**: Exploring extensions to video composition and application to a broader range of synthesis tasks.

## Related Work & Insights

- **DDS (Delta Denoising Score)**: The primary loss function source for this work, adapted from text-guided editing to image composition scenarios.
- **SDS/VSD**: Similar ideas in 3D generation (DreamFusion/ProlificDreamer), demonstrating the universal value of diffusion priors across multiple fields.
- **Prompt-to-Prompt / MasaCtrl**: A family of works utilizing attention manipulation for editing, which complements the KV manipulation mechanism in this paper.

## Rating

- **Novelty**: ★★★★☆ — The insight of utilizing diffusion priors to locate unnatural regions is novel, and the mask-guided KV design is elegant.
- **Value**: ★★★☆☆ — Zero-shot and plug-and-play, but speed constraints impact practical deployment.
- **Experimental Thoroughness**: ★★★☆☆ — The user study is well-designed, but large-scale quantitative evaluation is lacking.
- **Writing Quality**: ★★★★☆ — Clear methodology presentation and rigorous logic in ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators](dreamdrone_text-to-image_diffusion_models_are_zero-shot_perpetual_view_generator.md)
- [\[ECCV 2024\] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](multigen_zero-shot_image_generation_from_multi-modal_prompts.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](zero-shot_detection_of_ai-generated_images.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)

</div>

<!-- RELATED:END -->
