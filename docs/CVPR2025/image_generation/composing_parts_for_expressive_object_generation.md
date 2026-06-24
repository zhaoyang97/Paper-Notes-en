---
title: >-
  [Paper Note] Composing Parts for Expressive Object Generation
description: >-
  [CVPR 2025][Image Generation][Part-level control] Proposes PartComposer, a training-free method that localizes object parts from attention maps via parallel "part diffusion" and uses regional diffusion to independently generate user-specified fine-grained attributes (color, style, description) for each part, achieving part-level controllable image synthesis.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Part-level control"
  - "Training-free generation"
  - "Attention map segmentation"
  - "Regional diffusion"
  - "Rich-Text"
date: 2026-05-08
content_hash: f334e28e4053cafd
---

# Composing Parts for Expressive Object Generation

**Conference**: CVPR 2025  
**arXiv**: [2406.10197](https://arxiv.org/abs/2406.10197)  
**Code**: None (uses standard SD 1.5/2.1/XL, no additional code required)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Part-level control, Training-free generation, Attention map segmentation, Regional diffusion, Rich-Text

## TL;DR
Proposes PartComposer, a training-free method that localizes object parts from attention maps via parallel "part diffusion" and uses regional diffusion to independently generate user-specified fine-grained attributes (color, style, description) for each part, achieving part-level controllable image synthesis.

## Background & Motivation

**Background**: Models like Stable Diffusion can generate high-quality images via text prompts, and methods like ControlNet/GLIGEN provide spatial control (edge maps, bounding boxes, etc.). However, the control granularity remains at the object level—unable to specify the attributes of individual parts of an object (e.g., a bird with a red beak and blue wings).

**Limitations of Prior Work**: When part details are added to the prompt, SD either completely ignores them (e.g., "a bird with a red beak and blue wings" generates a normal bird) or generates images completely different from the basic prompt. While Rich-Text methods support local attributes, they operate at the object level rather than the part level, modifying the entire object region. InstructPix2Pix performs global editing and cannot precisely modify a single part.

**Key Challenge**: Controlling the attributes of each part individually without disrupting the overall structure of the object is necessary, but the attention maps of pretrained diffusion models work well at the object level while carrying weak signals at the part level.

**Goal**: How to localize object parts in a zero-shot manner from pretrained diffusion models and generate content for each part based on user-specified fine-grained attributes.

**Key Insight**: By running a parallel "part diffusion" process (denoising only within the object region and conditioned on part tokens), the U-Net is forced to focus the attention of part tokens on the correct spatial regions, thereby obtaining part localization masks. Then, regional diffusion is used to generate and combine specified attributes for each part independently.

**Core Idea**: Use parallel part diffusion to denoise within the object mask region to activate part-level attention localization, and then use masked regional diffusion to generate and harmoniously combine specified attributes for each part.

## Method

### Overall Architecture
A two-stage pipeline. **Part Localization Stage**: First run standard diffusion to obtain the object mask $\mathcal{M}_o$, and then after $T_{th} \approx T/2$, run parallel part diffusion (the U-Net conditioned on part prompts denoises only within the object region). Spectral clustering is applied to the self-attention maps of part diffusion to obtain K segmentation maps, and these segmentation maps are assigned to part tokens using a dot-product agreement over cross-attention to obtain each part mask $\mathbf{M}_{\mathbf{p}_i}$. **Part Generation Stage**: Run independent regional diffusion for each part (based on attribute descriptions from the Rich-Text interface), combine the noise predictions via mask weighting, and blend them with the background from the base generation.

### Key Designs

1. **Part Diffusion Localization**

    - **Function**: Extracts spatial masks of object parts from pretrained diffusion models in a zero-shot manner.
    - **Mechanism**: Within the object mask region, the output of the base diffusion is replaced with the output of a U-Net conditioned on a list of part tokens (e.g., "beak crown wings"): $\epsilon_t = \alpha \mathcal{M}_o \odot D(x_t, \hat{\mathbf{p}}, t) + (1-\alpha\mathcal{M}_o) \odot D(x_t, \hat{\mathbf{b}}, t)$. This forces the part U-Net to learn to denoise each part within a restricted region, making the cross-attention of each part token focus on the correct location. Spectral clustering is performed on the self-attention maps to obtain K spatial segmentations, and these segmentations are assigned to parts using a dot-product protocol (rather than average attention)—the dot product favors attention maps with high local activation, avoiding noise interference.
    - **Design Motivation**: Part token attention in standard SD is extremely weak and imprecise. By restricting the U-Net to denoisng only in the object region, the part tokens are forced to focus on specific regions.

2. **Rich-Text Interface + Regional Diffusion Generation**

    - **Function**: Independently generates user-specified fine-grained attributes for each part.
    - **Mechanism**: Users specify the attributes of each part via a Rich-Text interface (supporting footnote descriptions, RGB color values, style, and size). An independent diffusion process is run for each part and combined via masks: $\epsilon_t = \sum_i \mathbf{M}_{\mathbf{p}_i} \odot D(x_t, f(\mathbf{p}_i, \mathbf{a}_i), t)$. The background region is blended with the base generation to maintain the overall structure. Color attributes are precisely realized with RGB values using gradient guidance.
    - **Design Motivation**: Part-level generation requires finer control than text (such as precise RGB colors), and the Rich-Text interface provides a natural way to specify multiple attributes.

3. **Localization Quality Assurance Mechanism**

    - **Function**: Prevents misallocating part masks to unrelated regions.
    - **Mechanism**: The maximum value of cross-attention is used to determine whether a part is successfully localized: $L(j) = \mathds{1}\{\max(\hat{\mathbf{m}}_j) \geq (1-\delta)/K\}$. Unlocalized parts remain in their original state to avoid erroneous modifications. Independent text embedding initialization ("A photo of {part} of a {object}") makes part token embeddings more meaningful.
    - **Design Motivation**: Part attention maps are inherently noisy, and it is better not to localize than to localize erroneously.

### Loss & Training
Completely training-free method, using only pretrained SD models. Null-Text Inversion is used on real images to obtain inverted latents. DDIM with 50 steps and CFG scale of 8.5.

## Key Experimental Results

### Main Results

| Method | LPIPS↓ (Localization) | CLIP↑ (Consistency) | Aesthetic |
|------|-------------|-------------|----------------|
| **PartComposer** | **0.168** | **0.201** | 5.66 |
| StableDiffusion | 0.467 | 0.183 | 5.68 |
| InstructPix2Pix | 0.189 | 0.193 | 5.63 |
| Rich-Text | 0.243 | 0.187 | 5.65 |

Zero-shot unsupervised part segmentation (CUB200, FG-NMI/ARI):

| Method | FG-NMI | FG-ARI |
|------|--------|--------|
| SD baseline | 8.0 | 0.6 |
| Rich-Text | 3.1 | 0.3 |
| **PartComposer** | **20.5** | **9.2** |
| Unsup-Parts (Trained) | 46.0 | 21.0 |

### Ablation Study

| Configuration | FG-NMI | FG-ARI |
|------|--------|--------|
| Full PartComposer | 35.4 | 11.0 |
| W/o Null-Text Inversion | 23.1 | 5.2 |
| W/o Max Localization | 21.3 | 2.8 |
| W/o Dot-Product Allocation | 23.7 | 5.0 |

### Key Findings
- Part diffusion localization improves FG-NMI by 12.5 over the SD baseline, proving that denoising inside the object region indeed activates part-level attention.
- A user study (28 participants) showed strong preference for PartComposer's localization and consistency.
- The method generalizes to SDXL and across style domains (Monet paintings, Pixar characters, etc.).
- Null-Text Inversion is crucial for localization quality on real images (FG-NMI increases from 23.1 to 35.4).

## Highlights & Insights
- **Inherent insight of Part Diffusion** is clever: limiting the denoising region forces part tokens to compete for spatial locations, naturally producing part localization. This is a general strategy to extract finer-grained knowledge from diffusion models.
- **Dot-product allocation protocol** is superior to average attention: in noisy attention maps, only locally highly activated regions are reliable, and the dot product favors this pattern.
- **Completely training-free + highly versatile**: The same method works across all domains, including birds, humans, cartoon characters, and paintings.

## Limitations & Future Work
- Part generation quality is limited by part localization—if localization fails, generation fails.
- Attention maps are inherently noisy; some parts (especially small or overlapping parts) might not be localized correctly.
- Real images require Null-Text Inversion, which increases computational overhead.
- Limited to parts that the diffusion model already has semantic understanding of—unconventional parts might not be recognized.

## Related Work & Insights
- **vs Rich-Text**: Rich-Text operates at the object level, modifying the entire object region. PartComposer goes a step further by decomposing to the part level and modifying only specified parts.
- **vs InstructPix2Pix**: Editing methods perform global modifications. PartComposer achieves strictly local modification through masks.
- **vs ControlNet + Mask**: Requires manually provided part masks. PartComposer automatically extracts part masks from the diffusion model.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea of part diffusion localization is creative, and zero-shot part segmentation is a new setting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quite comprehensive, including localization evaluation, generation evaluation, user study, ablations, and cross-domain generalization.
- Writing Quality: ⭐⭐⭐⭐ Clearly described method and rich illustrations.
- Value: ⭐⭐⭐⭐ Provides unprecedented part-level control capabilities for creative design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2026\] ExpPortrait: Expressive Portrait Generation via Personalized Representation](../../CVPR2026/image_generation/expportrait_expressive_portrait_generation_via_personalized_representation.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[CVPR 2025\] BootPlace: Bootstrapped Object Placement with Detection Transformers](bootplace_bootstrapped_object_placement_with_detection_transformers.md)

</div>

<!-- RELATED:END -->
