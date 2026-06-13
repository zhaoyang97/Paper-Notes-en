---
title: >-
  [Paper Note] CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation
description: >-
  [ICCV 2025][Image Generation][attribute disentanglement] This paper proposes CompSlider, a compositional slider model that generates conditional priors to enable simultaneous, independent…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "attribute disentanglement"
  - "slider control"
  - "text-to-image generation"
  - "conditional prior"
  - "multi-attribute manipulation"
date: 2026-05-08
content_hash: a88a88b823deeea2
---

# CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation

**Conference**: ICCV 2025
**arXiv**: [2509.01028](https://arxiv.org/abs/2509.01028)  
**Code**: None  
**Area**: Image Generation
**Keywords**: attribute disentanglement, slider control, text-to-image generation, conditional prior, multi-attribute manipulation

## TL;DR

This paper proposes CompSlider, a compositional slider model that generates conditional priors to enable simultaneous, independent, and fine-grained control over multiple attributes in T2I foundation models. It addresses inter-attribute entanglement via a disentanglement loss and a structural consistency loss.

## Background & Motivation

In text-to-image (T2I) generation, controlling the intensity of image attributes (e.g., age, degree of smile) through text prompts alone is inherently imprecise. Slider-based generation methods (e.g., ConceptSliders, PromptSlider) have emerged to allow users to continuously adjust attributes via sliders. However, existing methods train a separate adapter for each attribute and overlook inter-attribute entanglement:

**Attribute entanglement**: Composing sliders in different orders yields different results — e.g., applying "smile" before "age" produces a different outcome than the reverse order.

**Structural inconsistency**: Adjusting one attribute inadvertently alters unrelated factors such as background or hairstyle.

**Poor scalability**: $N$ attributes require $N$ forward passes, imposing significant computational overhead.

## Method

### Overall Architecture

CompSlider replaces the role of the CLIP image encoder in the T2I foundation model. It takes user-defined slider values and a text prompt as input, and outputs an image condition $\bm{c}^{\mathcal{I}}$ as a multi-attribute prior, which is fed into the base diffusion model for image generation:

$$\bm{c}^{\mathcal{I}} = \text{CompSlider}(\bm{c}^{\mathcal{S}}, \bm{c}^{\mathcal{T}})$$

where $\bm{c}^{\mathcal{S}}$ denotes slider embeddings and $\bm{c}^{\mathcal{T}}$ denotes T5 text tokens. The base model requires no fine-tuning.

### Key Designs

1. **DiT as the CompSlider backbone**: A Diffusion Transformer (DiT) is adopted, using a reparameterization trick to directly predict the clean image condition $\bm{c}_0^{\mathcal{I}}$ rather than noise. Since the image condition is a 1024-dimensional vector with no need for downsampling, DiT is more suitable than U-Net. The model consists of 10 DiT blocks, taking 128 text tokens and 16 slider tokens as input, with a total of 277M parameters.

2. **Slider value embedding**: Attribute scores are obtained via pretrained attribute classifiers and normalized to $[0,1]$. Positional encoding (sinusoidal encoding) maps continuous slider values to vectors $\bm{p}^{\mathcal{S}} \in \mathbb{R}^{N \times \frac{dim}{2}}$, and learnable category embeddings $\bm{w} \in \mathbb{R}^{N \times \frac{dim}{2}}$ are introduced to distinguish different attributes. The final slider embedding is their concatenation: $\bm{c}^{\mathcal{S}} = [\bm{p}^{\mathcal{S}}, \bm{w}]$.

3. **Random attribute combination training strategy**: A key innovation is that the method does not rely on paired data (i.e., images of the same subject at varying attribute intensities). Instead, randomly sampled attribute value combinations $\bm{v}^{\mathcal{S}*}$ are introduced during training, ensuring the model generalizes to arbitrary combinations rather than only the co-occurrence patterns observed in training data.

### Loss & Training

The total loss comprises three terms: $\mathcal{L} = \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{st}} + \mathcal{L}_{\text{clss}}$

- **Diffusion loss** $\mathcal{L}_{\text{diff}}$: Ensures the generated condition aligns with the output distribution of the CLIP image encoder: $\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\bm{c}_0^{\mathcal{I}} - \text{DiT}(\bm{c}_t^{\mathcal{I}}, \bm{c}^{\mathcal{S}}, \bm{c}^{\mathcal{T}}, t)\|^2]$
- **Disentanglement loss** $\mathcal{L}_{\text{clss}}$: An MLP classifier is trained to recover attribute differences from the conditional discrepancy between original and randomly combined attribute outputs. The difference is discretized into $B=20$ bins and supervised with a cross-entropy loss.
- **Structural loss** $\mathcal{L}_{\text{st}}$: When the attribute difference satisfies $|\Delta v_i| \leq 0.1$, the L2 distance between the two conditional outputs is constrained to preserve local structural consistency.

Training data comprises approximately 3 million images. The diffusion and structural losses train the DiT, while the disentanglement loss jointly trains both the DiT and the MLP classifier.

## Key Experimental Results

### Main Results

Quantitative comparison on human-related sliders (300 prompts × 5 attributes × 5 values = 7,500 images):

| Method | Cont.%↑ | Cons.%↑ | Scope%↑ | Entang.%↓ | LPIPS↓ | CLIP↑ |
|------|---------|---------|---------|-----------|--------|-------|
| Prompt2Prompt | - | 88.47 | 49.46 | 28.99 | 0.19 | 4.15 |
| PromptSlider | 61.17 | 80.23 | 46.25 | 24.31 | 0.10 | 4.79 |
| ConceptSlider | 73.41 | 83.17 | 54.43 | 27.22 | 0.16 | 5.76 |
| **CompSlider** | **81.07** | **90.95** | **59.02** | **14.04** | 0.12 | **6.20** |

A/B user study on non-human sliders (Vector Style + Scene Complexity): CompSlider achieves a user preference of 54.66% vs. 34.16% for ConceptSlider.

### Ablation Study

Ablation of the disentanglement loss and structural loss:

| $\mathcal{L}_{\text{diff}}$ | $\mathcal{L}_{\text{clss}}$ | $\mathcal{L}_{\text{st}}$ | Cont.%↑ | Cons.%↑ | Scope%↑ | Entang.%↓ |
|:---:|:---:|:---:|---------|---------|---------|-----------|
| ✓ | | | 68.96 | 63.21 | 42.06 | 36.68 |
| ✓ | ✓ | | 76.49 | 49.29 | 63.27 | 19.87 |
| ✓ | ✓ | ✓ | **81.07** | **90.95** | **59.02** | **14.04** |

### Key Findings

- Using only the diffusion loss results in an entanglement rate as high as 36.68%; adding the disentanglement loss reduces it to 19.87%, but structural consistency collapses to 49.29%.
- The structural loss substantially recovers consistency from 49.29% to 90.95% (+41.66%) and further reduces entanglement to 14.04%.
- CompSlider supports controlling all sliders in a single forward pass, offering substantially better inference efficiency than per-attribute methods.

## Highlights & Insights

- **Core innovation**: Operating in the latent space of conditional priors eliminates the need to fine-tune the base model, significantly reducing training and inference costs.
- **No paired data required**: Disentanglement is learned via randomly sampled attribute combinations during training, elegantly circumventing the difficulty of acquiring paired data of the same subject at varying attribute intensities.
- **Four new evaluation metrics**: Continuity, Scope, Consistency, and Entanglement are proposed, providing a more comprehensive assessment of slider-based generation quality than LPIPS/CLIP alone.
- The approach is extensible to video generation.

## Limitations & Future Work

- The method relies on pretrained attribute classifiers to obtain ground-truth slider values; classifier quality directly impacts training.
- The attribute set is closed (16 predefined attributes), with no support for open-domain attributes.
- Compatibility with more recent diffusion models (e.g., SDXL, SD3) has not been validated.
- Automated evaluation metrics for non-human attributes are absent; assessment relies solely on user studies.

## Related Work & Insights

- ConceptSliders and PromptSlider are direct predecessors, employing LoRA adapters and textual inversion for single-attribute sliders, respectively.
- eDiff-I provides the foundational framework for conditional image priors.
- The disentanglement strategy offers inspiration for other multi-condition controllable generation tasks, such as simultaneously controlling composition, style, and content.

## Rating

- Novelty: ⭐⭐⭐⭐ Compositional sliders operating in the conditional prior space is a novel perspective, and the disentanglement loss design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation covers both human and non-human attributes, new metrics are proposed, and ablation studies and extensions are included.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, the method is described in detail, and figures and tables are informative.
- Value: ⭐⭐⭐⭐ Addresses a practical problem of simultaneous multi-attribute control with clear application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] All-in-One Slider for Attribute Manipulation in Diffusion Models](../../CVPR2026/image_generation/all_in_one_slider_attribute_manipulation.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[NeurIPS 2025\] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans](../../NeurIPS2025/image_generation/multihuman-testbench_benchmarking_image_generation_for_multiple_humans.md)
- [\[ICCV 2025\] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)

</div>

<!-- RELATED:END -->
