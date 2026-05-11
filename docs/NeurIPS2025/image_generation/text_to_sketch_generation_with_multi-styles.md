---
title: >-
  [Paper Note] Text to Sketch Generation with Multi-Styles
description: >-
  [NeurIPS 2025][Image Generation][Sketch Generation] This paper proposes M3S (Multi-Style Sketch Synthesis), a training-free framework that achieves single- and multi-style sketch generation conditioned on text prompts an…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Sketch Generation"
  - "Multi-Style Synthesis"
  - "Diffusion Models"
  - "K/V Injection"
  - "AdaIN"
date: 2026-05-08
content_hash: 0b0112dbe95c39be
---

# Text to Sketch Generation with Multi-Styles

**Conference**: NeurIPS 2025
**arXiv**: [2511.04123](https://arxiv.org/abs/2511.04123)
**Code**: [GitHub](https://github.com/CMACH508/M3S)
**Area**: Image Generation, Style Transfer, Sketch Synthesis
**Keywords**: Sketch Generation, Multi-Style Synthesis, Diffusion Models, K/V Injection, AdaIN

## TL;DR
This paper proposes M3S (Multi-Style Sketch Synthesis), a training-free framework that achieves single- and multi-style sketch generation conditioned on text prompts and reference style sketches, via linearly smoothed K/V feature injection, joint AdaIN style tendency control, and style-content disentangled guidance.

## Background & Motivation
- Sketches serve as a cross-lingual visual medium with broad applications ranging from industrial prototyping to artistic expression.
- The scarcity of high-quality sketch datasets (requiring professional skills and substantial time) constrains model training.
- Limitations of existing approaches:
    - CLIPasso/DiffSketcher: lack precise control over style attributes.
    - K/V replacement methods (e.g., MasaCtrl): suffer from misalignment between Q and the substituted K/V in cross-domain settings, leading to content leakage and structural incoherence.
    - StyleAligned: aligns statistical distributions via AdaIN but performs poorly in structurally divergent domains such as sketches.
- Text-conditioned style control lacks expressiveness and cannot accurately match specific styles.

## Method

### Overall Architecture
A training-free framework built upon Stable Diffusion v1.5/SDXL, supporting both single-style and multi-style sketch generation.

### Key Designs

#### 1. Style Feature Injection
**Against direct replacement**: $Attention(Q_{tar}, K_{ref}, V_{ref})$ introduces structural incoherence in cross-domain settings.

**Against AdaIN alignment**: $Q_{tar} = AdaIN(Q_{tar}, Q_{ref})$ is detrimental to sketch generation.

**M3S solution: linearly smoothed feature concatenation**
$$Attention\left(Q_{tar}, \begin{bmatrix}K_{tar}\\\hat{K}_{ref}\end{bmatrix}, \begin{bmatrix}V_{tar}\\\hat{V}_{ref}\end{bmatrix}\right)$$
$$\hat{K}_{ref} = \lambda K_{tar} + (1-\lambda)K_{ref}, \quad \hat{V}_{ref} = \lambda V_{tar} + (1-\lambda)V_{ref}$$
- $\lambda \in [0,1]$ balances content fidelity and style consistency.
- Increasing $\lambda$ enhances aesthetics and text alignment, but excessively large values may cause style degradation.

#### 2. Multi-Style Tendency Control (Joint AdaIN)
$$z_t^{tar} = \eta \cdot AdaIN(z_t^{tar}, z_t^{ref_1}) + (1-\eta) \cdot AdaIN(z_t^{tar}, z_t^{ref_2})$$
- $\eta \in [0,1]$: style tendency parameter.
- Intuition: dense-stroke sketches have lower mean → AdaIN biases toward detailed output; sparse sketches have higher mean → minimalist results.
- Even at $\eta=0$ or $\eta=1$, multi-style characteristics are preserved because the self-attention module incorporates K/V features from both styles.

#### 3. Style-Content Disentangled Guidance
$$\tilde{\epsilon}_t = \epsilon_\theta(z_t^{tar}, t, \emptyset) + \omega_1 \cdot \underbrace{[\epsilon_\theta^\times(\cdot, text, K_{ref}, V_{ref}) - \epsilon_\theta(\cdot, \emptyset)]}_{\text{content guidance}} + \omega_2 \cdot \underbrace{[\epsilon_\theta^\times(\cdot, \emptyset, K_{ref}, V_{ref}) - \epsilon_\theta(\cdot, \emptyset)]}_{\text{style guidance}}$$
- $\omega_1, \omega_2$ control the strength of content and style guidance, respectively.
- $\omega_2$ is linearly increased from $\omega_2/3$ to $\omega_2$ throughout the denoising process (gradually reinforcing style).

#### 4. Contour-Based Regularization Guidance (SD v1.5)
- The denoised latent is estimated as $z_{0|t}^{tar}$ via the Tweedie formula and decoded into an image.
- Sobel operators are applied to extract directional gradients, maximizing edge responses.
- $\mathcal{L}_{edge} = -|grad_x| - |grad_y|$
- Suppresses artifacts in abstract sketch generation.

## Key Experimental Results

### Main Results: Quantitative Comparison Across 6 Styles

| Method | Impl. | Style1 CLIP-T↑ | Style1 DINO↑ | Style1 VGG↓ | Style5 CLIP-T↑ | Style5 DINO↑ |
|-----|------|---------------|-------------|------------|---------------|-------------|
| StyleAligned | - | 0.3130 | 0.6691 | 0.0308 | 0.3004 | 0.5428 |
| AttentionDistill | - | 0.3305 | **0.7738** | 0.0930 | 0.3377 | **0.6221** |
| InstantStyle | - | **0.3512** | 0.4934 | 0.0417 | **0.3480** | 0.4408 |
| CSGO | - | 0.3336 | 0.5276 | 0.0571 | 0.3298 | 0.4288 |
| **M3S (SD v1.5)** | - | 0.3507 | 0.6383 | **0.0200** | 0.3494 | 0.5777 |
| **M3S (SDXL)** | - | **0.3607** | 0.6545 | 0.0165 | 0.3467 | 0.5332 |

### Human Preference Scores (1–8)

| Method | Avg. Score |
|-----|---------|
| StyleAligned | 2.77 |
| CSGO | 3.83 |
| StyleStudio | 4.22 |
| AttentionDistill | 4.28 |
| InstantStyle | 5.08 |
| **M3S (SD v1.5)** | **5.44** |
| **M3S (SDXL)** | **6.19** |

### Multi-Style Generation (Validation of $\eta$ Control)

| $\eta$ | DINO-ref1↑ | DINO-ref2↑ | CLIP-T↑ |
|--------|-----------|-----------|---------|
| 0 | 0.3936 | **0.4944** | 0.3442 |
| 0.25 | 0.4180 | 0.4821 | 0.3514 |
| 0.5 | 0.4408 | 0.4556 | 0.3495 |
| 0.75 | 0.4578 | 0.4221 | 0.3499 |
| 1.0 | **0.4693** | 0.3975 | 0.3470 |

### Key Findings
- M3S (SDXL) substantially outperforms all baselines in human preference with a score of 6.19.
- Linear smoothing ($\lambda$) effectively reduces content leakage compared to direct replacement and AdaIN alignment.
- AttentionDistillation achieves high DINO scores but suffers from severe content leakage, mixing reference image content into the target output.
- The multi-style parameter $\eta$ reliably controls style tendency: DINO-ref1 increases monotonically with $\eta$ while DINO-ref2 decreases monotonically.
- Even at $\eta=0$ or $\eta=1$, the outputs retain characteristics of both styles due to the two sets of K/V features present in self-attention.

## Highlights & Insights
1. **Training-free framework**: operates directly on pretrained diffusion models without any fine-tuning.
2. **Elegant simplicity**: linear smoothing replaces complex feature alignment or distillation schemes.
3. **Controllable multi-style synthesis**: the first method to achieve multi-style fusion and continuous style interpolation for sketches.
4. **Cross-domain robustness**: maintains generation quality even when reference and target structures differ substantially — a known failure mode of K/V replacement methods.
5. **Dual-backbone validation**: verified on both SD v1.5 and SDXL.

## Limitations & Future Work
- $\omega_1, \omega_2, \lambda$ require manual tuning per style type (e.g., abstract sketches in Style 6 require different parameter settings).
- Abstract sketch generation on SD v1.5 may produce artifacts, necessitating contour regularization guidance.
- The base model is trained on natural images; excessively large $\lambda$ may cause outputs to revert to a naturalistic appearance.
- 100-step DDIM sampling introduces considerable inference latency.
- Video and animated sketch generation remain unexplored.

## Related Work & Insights
- DiffSketcher: a pioneering method for text-driven sketch synthesis, but lacks style control.
- Cross-image/MasaCtrl: establishes the K/V replacement paradigm — this paper analyzes the root causes of its cross-domain failure.
- B-LoRA/IP-Adapter/CSGO: style adapter approaches primarily targeting natural images.
- AdaIN (Huang 2017): a classical method for real-time arbitrary style transfer — extended in this paper to multi-style sketch control.

## Rating
- Novelty: ⭐⭐⭐⭐ (first exploration of multi-style sketch generation with an elegant method design)
- Technical Depth: ⭐⭐⭐⭐ (thorough analysis of style injection mechanisms with comprehensive ablations)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 styles × 8 baselines + multi-style + human evaluation)
- Writing Quality: ⭐⭐⭐⭐ (rich visualizations and clear comparisons)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](../../CVPR2026/image_generation/multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)

</div>

<!-- RELATED:END -->
