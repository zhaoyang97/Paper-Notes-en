---
title: >-
  [Paper Note] LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models
description: >-
  [CVPR 2026][Image Generation][Lighting Control] This paper identifies a semantic gap in T2I model text encoders that prevents understanding of standard lighting terminology (e.g., tungsten, 6500K), and proposes LumiCtrl, which learns illumination prompts via three components — physics-based lighting augmentation, edge-guided prompt disentanglement, and masked reconstruction loss — enabling precise text-guided lighting control while preserving subject identity.
tags:
  - CVPR 2026
  - Image Generation
  - Lighting Control
  - Text-to-Image
  - Personalized Generation
  - Illumination Prompt Learning
  - ControlNet
date: 2026-05-08
content_hash: d30256da2d5deef5
---

# LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models

**Conference**: CVPR 2026
**arXiv**: [2512.17489](https://arxiv.org/abs/2512.17489)
**Code**: None
**Area**: Diffusion Models / Image Generation
**Keywords**: Lighting Control, Text-to-Image, Personalized Generation, Illumination Prompt Learning, ControlNet

## TL;DR

This paper identifies a semantic gap in T2I model text encoders that prevents understanding of standard lighting terminology (e.g., tungsten, 6500K), and proposes LumiCtrl, which learns illumination prompts via three components — physics-based lighting augmentation, edge-guided prompt disentanglement, and masked reconstruction loss — enabling precise text-guided lighting control while preserving subject identity.

## Background & Motivation

**Background**: Diffusion model-based text-to-image (T2I) generation has reached high quality. Lighting is a key factor influencing image aesthetics and atmosphere — software such as Photoshop provides standard presets including daylight, tungsten, and fluorescent. T2I personalization methods (Textual Inversion, DreamBooth, Custom Diffusion) can learn new concepts from a few images, but tend to entangle lighting information from the training images.

**Limitations of Prior Work**: Through systematic experiments, the authors reveal a fundamental issue — the text encoder of T2I models suffers from an **illumination semantic gap**: standard lighting terms (e.g., "tungsten") do not cluster with illumination concepts, and Kelvin temperature values (e.g., "2850K") are treated as ordinary numbers rather than photometric quantities. This is confirmed via t-SNE visualization and silhouette coefficient analysis across four CLIP encoders. As a result, regardless of how lighting instructions are written in prompts, generated images consistently favor a default daylight prior.

**Key Challenge**: T2I models are visually lighting-aware, yet their text encoders lack the capacity to map lighting terminology to visual illumination variation. Existing post-processing methods (IC-Light, Instruct Pix2Pix) either destroy background structure or introduce artifacts.

**Goal**: How can T2I models accept lighting conditions directly via text prompts while preserving subject identity and spatial structure?

**Key Insight**: Since standard lighting terminology is not properly understood in text space, LumiCtrl learns illumination prompts from scratch — using physics-based lighting augmentation as training signal, ControlNet to decouple lighting from structure, and masked loss to focus learning on foreground illumination.

**Core Idea**: By learning new text token embeddings corresponding to standard lighting conditions, precise lighting control is embedded directly into the prompt space of T2I generation.

## Method

### Overall Architecture

The input is a single image and text description of the target concept. The method comprises three stages: (1) **Temperature Mapping** — physics-based lighting augmentation using the Planckian locus to generate training variants under 7 standard lighting conditions; (2) **Weight Optimization** — introducing a concept token $[v]$ and an independent illumination token $[c_i^*]$ per lighting condition, optimizing the key/value projection matrices of cross-attention to learn illumination representations; (3) **Inference** — generating concept images under specified lighting using the learned illumination tokens. During training, a frozen pretrained ControlNet provides edge constraints; it is discarded at inference.

### Key Designs

1. **Physics-Based Lighting Augmentation (Temperature Mapping + Flat Light Adaptation)**:

    - Function: Generate training samples of the same concept under multiple standard lighting conditions from a single image.
    - Mechanism: Seven standard lighting conditions are defined along the Planckian locus (blackbody radiation trajectory): tungsten (2850K), 3300K, fluorescent (3800K), 4500K, cloudy (6500K), 7000K, and shade (7500K). Each condition corresponds to an RGB color vector; global illumination shifts are applied via the von Kries model (diagonal matrix transform) to produce training images under each condition.
    - Design Motivation: Learning illumination prompts requires paired data of the same concept under different lighting, which is extremely difficult to collect in practice. Although Flat Light Adaptation is a simple global uniform transform, combined with the masked loss it allows the model to accurately learn foreground lighting while leveraging the diffusion prior to adaptively generate plausible background lighting.

2. **Edge-Guided Prompt Disentanglement**:

    - Function: Prevent structural information from leaking into the illumination prompt during learning.
    - Mechanism: A pretrained, parameter-frozen ControlNet conditioned on Canny edge maps is incorporated during training. Since the ControlNet already supplies structural information (edges, shapes), the illumination prompt learning is forced to focus exclusively on color/lighting variation, preventing the encoding of specific layout or object position details from training images into the prompt. ControlNet is not used at inference.
    - Design Motivation: A common failure mode of T2I personalization is prompt overfitting to the specific structure of training images — generated results may incorrectly replicate training image layouts or add/remove objects. The ControlNet serves as a "structural anchor," separating the learning pathways for structural and appearance (lighting) information.

3. **Masked Reconstruction Loss**:

    - Function: Focus illumination learning on the foreground subject to achieve context-aware lighting adaptation.
    - Mechanism: The loss function is defined as $\mathcal{L}_{mrl} = (1-\lambda)\cdot\mathcal{L}_{rec}\cdot(1-\mathcal{M}) + \lambda\cdot\mathcal{L}_{rec}\cdot\mathcal{M}$, where $\mathcal{M}$ is the foreground mask of the target concept and $\lambda$ controls the foreground/background weight balance. By assigning higher loss weight to foreground pixels, the model is compelled to precisely learn the illumination color variation of the foreground subject.
    - Design Motivation: Flat Light Adaptation applies a uniform global transform to the entire image, whereas in real scenes different regions respond differently to lighting due to material properties, shadows, and reflections. The masked loss strictly enforces accurate foreground lighting while relaxing the constraint on the background, allowing the diffusion prior to naturally adapt background regions — this constitutes the proposed Contextual Light Adaptation.

### Loss & Training

The method is built on Stable Diffusion v1.5, fine-tuning the key/value projection matrices of cross-attention and the illumination token embeddings using the Custom Diffusion framework. AdamW optimizer, batch size 2, learning rate $10^{-5}$, 3000 training steps. The masked loss is computed at $64\times64$ latent resolution. Inference uses DDPM with 200 steps and CFG scale 6.0.

## Key Experimental Results

### Main Results

| Category | Method | Angular Error↓ | SSIM↑ | MSE↓ |
|----------|--------|---------------|-------|------|
| T2I Personalization | Textual Inversion | 15.35 | 0.57 | 38.50 |
| T2I Personalization | DreamBooth | 12.76 | 0.71 | 34.10 |
| T2I Personalization | Custom Diffusion | 13.34 | 0.61 | 38.20 |
| T2I Editing | IC-Light | 10.39 | 0.58 | 35.90 |
| T2I Editing | PnP+P2P | 11.24 | 0.67 | 33.20 |
| **Ours w/ CtrlNet** | **LumiCtrl** | **4.51** | **0.77** | **16.80** |
| Ours w/o CtrlNet | LumiCtrl | 6.87 | 0.74 | 22.40 |

LumiCtrl reduces Angular Error by 56.6% relative to the strongest baseline IC-Light (10.39 → 4.51) and reduces MSE by 53.2% (33.20 → 16.80).

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| w/o Temperature Mapping + w/o Masked Loss | Incorrect lighting, deviates from prompt | Both training data and loss function are necessary |
| w/o ControlNet | Structural artifacts appear | Edge guidance is critical for disentangling lighting and structure |
| $\lambda$ too high | Unnatural background | Foreground/background weight balance is required |

### Key Findings

- The illumination semantic gap in text encoders is a systematic issue — consistently observed across all four CLIP models.
- Kelvin temperature value embeddings cluster with ordinary numbers rather than illumination concepts.
- A user study (15 participants, 320 questions, 2AFC protocol + Thurstone Case V model) confirms that LumiCtrl significantly outperforms all baselines.
- Contextual Light Adaptation is effective — foreground lighting is accurate while the background adapts naturally.

## Highlights & Insights

- The diagnosis of the CLIP text encoder's illumination semantic gap is thorough: beyond identifying the problem, quantitative evidence is provided via t-SNE visualization and silhouette coefficient analysis.
- Elevating lighting control from post-processing (image space) to prompt space represents a paradigm shift.
- The three components combine elegantly: physics-based augmentation provides training signal → ControlNet blocks structural leakage → masked loss focuses illumination learning → diffusion prior handles the background.
- ControlNet is not required at inference, keeping deployment lightweight.

## Limitations & Future Work

- Only 7 discrete lighting presets are covered; practical needs may include more diverse conditions (e.g., colored light sources, directional lighting).
- Lighting is a continuous spectrum; discrete tokens may lack sufficient granularity — continuous illumination embeddings could offer greater flexibility.
- The method is built on Stable Diffusion v1.5 and has not been validated on more advanced architectures (SD3, FLUX).
- Flat Light Adaptation applies a globally uniform transform, which may be insufficient for scenes with complex lighting and shadows.
- Independent 3000-step training is required per concept, making scaling to large numbers of concepts challenging.

## Related Work & Insights

- The masked diffusion loss from Break-a-Scene is adapted into the foreground masked reconstruction loss.
- The use of ControlNet as a "structural anchor" — discarded at inference — is worth borrowing for other disentangled learning settings.
- The discovery of the text encoder semantic gap also has implications for other T2I control scenarios — specific terms for materials, weather, and similar attributes may exhibit analogous issues.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to reveal the illumination semantic gap; first to achieve prompt-space lighting control.
- Experimental Thoroughness: ⭐⭐⭐⭐ 20 concepts × 7 lighting conditions × 42 seeds; quantitative evaluation + user study + ablation; reasonably comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The diagnostic analysis in the motivation section is particularly compelling; method description is clear.
- Value: ⭐⭐⭐⭐ Practically valuable for controllable T2I generation; lighting control addresses a genuine need for content creators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TokenLight: Precise Lighting Control in Images using Attribute Tokens](tokenlight_precise_lighting_control_in_images_using_attribute_tokens.md)
- [\[CVPR 2026\] PureCC: Pure Learning for Text-to-Image Concept Customization](purecc_pure_learning_for_text-to-image_concept_customization.md)
- [\[ICCV 2025\] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](../../ICCV2025/image_generation/autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)
- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](../../ICLR2026/image_generation/directional_textual_inversion_for_personalized_text-to-image_generation.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)

</div>

<!-- RELATED:END -->
