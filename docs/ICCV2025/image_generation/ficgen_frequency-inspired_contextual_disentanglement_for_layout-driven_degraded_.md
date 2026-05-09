---
title: >-
  [Paper Note] FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation
description: >-
  [ICCV 2025][Image Generation][Layout-to-Image] FICGen is proposed as the first method to address the "contextual illusion dilemma" in Layout-to-Image (L2I) generation for degraded scenes (low-light, underwater, remote sensing, adverse weather, etc.). It extracts high- and low-frequency prototypes of degraded scenes via a learnable dual-query mechanism, injects them into the latent diffusion space through visual-frequency enhanced attention, and achieves foreground-background disentanglement using instance consistency maps and spatial-frequency adaptive aggregation. FICGen comprehensively outperforms existing L2I methods across five degraded-scene datasets.
tags:
  - ICCV 2025
  - Image Generation
  - Layout-to-Image
  - Degraded Image Generation
  - Frequency Disentanglement
  - Low-light
  - Remote Sensing
  - Underwater
date: 2026-05-08
content_hash: 4c8eca1cb5ab44d6
---

# FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation

**Conference**: ICCV 2025
**arXiv**: [2509.01107](https://arxiv.org/abs/2509.01107)
**Code**: None (not mentioned)
**Area**: Image Generation / Layout-to-Image / Degraded Scene Synthesis
**Keywords**: Layout-to-Image, Degraded Image Generation, Frequency Disentanglement, Low-light, Remote Sensing, Underwater

## TL;DR
FICGen is proposed as the first method to address the "contextual illusion dilemma" in Layout-to-Image (L2I) generation for degraded scenes (low-light, underwater, remote sensing, adverse weather, etc.). It extracts high- and low-frequency prototypes of degraded scenes via a learnable dual-query mechanism, injects them into the latent diffusion space through visual-frequency enhanced attention, and achieves foreground-background disentanglement using instance consistency maps and spatial-frequency adaptive aggregation. FICGen comprehensively outperforms existing L2I methods across five degraded-scene datasets.

## Background & Motivation

### Problem Background
Visual perception tasks in degraded scenes (low-light, underwater, remote sensing, adverse weather, etc.) suffer from severe data scarcity. For example, the ExDARK low-light dataset contains only 7,363 images, roughly 1/20 the size of COCO. Layout-to-Image (L2I) generation is a promising approach for synthesizing training data from layout conditions.

### Core Challenge: Contextual Illusion Dilemma
Existing L2I methods perform well on natural scenes but face serious issues when applied to degraded scenarios:
- Remote sensing objects (e.g., vehicles) are small and visually similar to surrounding structures (e.g., bridges).
- Underwater species (e.g., fish) frequently merge with nearby organisms (e.g., coral).
- This leads to **hallucinations in object count, position, and interaction** during generation.

### Frequency-Domain Analysis
In natural images, high-frequency (HF) and low-frequency (LF) components are relatively balanced, and foreground-background distinction is clear. In degraded images, **high-frequency details of foreground instances are attenuated**, while **low-frequency background components dominate the overall frequency distribution**. This explains why instances tend to be "submerged" in degraded scenes.

### Motivation
The paper motivates contextual disentanglement from a frequency perspective: extracting high-frequency (instance boundaries/textures) and low-frequency (background color/atmosphere) knowledge from degraded scenes, injecting them into the diffusion generation process, and achieving latent-space foreground-background disentanglement via instance-level masks.

## Method

### Overall Architecture
FICGen comprises three core modules:
1. **Frequency Perceiver Resamplers** — extract HF/LF frequency prototypes via a dual-query mechanism.
2. **Visual-Frequency Enhanced Attention** — inject frequency knowledge into the latent diffusion space.
3. **Adaptive Spatial-Frequency Aggregation** — blend spatial and frequency information to reconstruct degraded representations.

### Frequency Prototype Extraction
**Step 1: Constructing Frequency Prototypes**
Degraded instances are sampled per category from the training set. Intermediate feature maps $\mathbf{X} \in \mathbb{R}^{H \times W}$ are transformed to the frequency domain via DFT:

$$\textbf{X}_{\mathcal{F}}(u,v) = \frac{1}{H \times W}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}\textbf{X}(h,w)e^{-j2\pi(uh+vw)}$$

A binary mask $\mathbf{M}_{\mathcal{F}}$ separates HF/LF regions; learnable channel weights are applied before inverse DFT back to the spatial domain:

$$\textbf{X}^{\uparrow} = \mathcal{F}^{-1}(\textbf{X}_{\mathcal{F}}\textbf{M}_{\mathcal{F}})\cdot\mathbf{W}_{\mathcal{F}}$$

Average pooling over HF/LF feature maps yields frequency prototypes: $\textbf{p}^{\uparrow} = \{p_i^{\uparrow}\}_{i=1}^N$ (instance HF) and $p^{\downarrow}$ (background LF).

**Step 2: Dual-Query Frequency Resamplers**
Inspired by the Perceiver architecture, two independent learnable queries interact with the frequency prototypes via Transformer blocks:

$$\textbf{q}_i^{\uparrow} = \text{HF-Resampler}(\mathcal{Q}^{\uparrow}, \phi_{k1}^r(p_i^{\uparrow}), \phi_{v1}^r(p_i^{\uparrow}))$$
$$\textbf{q}^{\downarrow} = \text{LF-Resampler}(\mathcal{Q}^{\downarrow}, \phi_{k1}^g(p^{\downarrow}), \phi_{v1}^g(p^{\downarrow}))$$

The dual-query mechanism simultaneously captures instance boundary textures (HF) and environmental atmosphere/color (LF).

### Contextual Frequency Knowledge Transfer

**Visual-Frequency Enhanced Attention**: Frequency-aware tokens are fused with layout conditions and injected into the diffusion U-Net:
- Instance representation: $\textbf{R}_i = [\textbf{q}_i^{\uparrow}; \textbf{E}_{clip}(l_i); \textbf{E}_{box}(\text{Fourier}(b_i))]$ (HF token + semantics + position)
- Background representation: $\textbf{G} = [\textbf{q}^{\downarrow}; \textbf{E}_{clip}(\mathcal{Y})]$ (LF token + global description)

**Instance Consistency Maps** decouple foreground from background:

$$\hat{\mathbf{M}}_i(x,y) = \begin{cases} 1, & \text{if } [x,y] \in b_i \\ 0, & \text{otherwise} \end{cases}$$
$$\hat{\mathbf{M}}^g = 1 - \sum_{i=1}^{N}\hat{\mathbf{M}}_i$$

Mask constraints ensure each layout condition influences only its corresponding local region, preventing attribute leakage.

### Adaptive Spatial-Frequency Aggregation
Rather than simple summation or purely spatial-domain fusion, FICGen aggregates degraded instances and background simultaneously in both spatial and frequency domains:

$$\textbf{F}^s = \textbf{SAM}([\sum_{i=1}^N \textbf{f}_i^r, \textbf{f}^g]), \quad \textbf{F}^f = \textbf{FAM}([\sum_{i=1}^N \textbf{f}_i^r, \textbf{f}^g])$$

where SAM captures spatial relational dependencies via standard self-attention, and FAM employs frequency attention to emphasize fine-grained cross-instance attributes (boundary sharpness, texture). The two streams are fused via a learnable depthwise convolution $\zeta$ and softmax-weighted aggregation to produce the final degraded representation $\delta^{final}$.

### Loss & Training
Pre-trained LDM parameters are frozen; only FICGen modules are trained:

$$\min_{\theta'} \mathcal{L}_{FICGen} = \mathbb{E}_{z_0, \epsilon, t, \mathcal{Y}}[\|\epsilon - \mathcal{G}_{\theta,\theta'}(z_t, t, \mathcal{Y}, \mathcal{B}, \mathcal{Q})\|_2^2]$$

## Key Experimental Results

### Experimental Setup
- **Base Model**: SDv1.5; FICGen is deployed only at the 8×8 and 16×16 resolution decoder layers.
- **Training**: AdamW, lr=1e-4, 300 epochs, 8×A100, batch size=320.
- **Evaluation**: Five degraded-scene datasets — ExDARK (low-light), RUOD (underwater), DIOR-H (remote sensing), DAWN (adverse weather), blurred VOC2012 (blur).
- **Metrics**: FID (fidelity), COCO-style AP (alignment), downstream detector mAP (trainability).

### Main Results: Fidelity and Alignment

| Dataset | Method | FID↓ | mAP↑ | AP_50↑ | AP_75↑ |
|--------|------|------|------|--------|--------|
| DIOR-H (Remote Sensing) | MIGC | 31.64 | 21.8 | 38.4 | 17.5 |
| DIOR-H | CC-Diff | 30.88 | 23.6 | 42.4 | 21.4 |
| DIOR-H | **FICGen** | 31.25 | **27.6** | **48.7** | **27.6** |
| RUOD (Underwater) | MIGC | 26.50 | 27.2 | 54.1 | 24.6 |
| RUOD | CC-Diff | 25.21 | 29.7 | 58.4 | 27.9 |
| RUOD | **FICGen** | **25.10** | **37.0** | **68.6** | **36.5** |
| ExDARK (Low-light) | MIGC | 45.76 | 32.4 | 63.5 | 29.5 |
| ExDARK | CC-Diff | 44.26 | 35.1 | 65.6 | 34.1 |
| ExDARK | **FICGen** | **42.40** | **42.5** | **73.0** | **45.1** |

FICGen substantially leads in alignment (mAP) across all degraded scenes. Notably, on ExDARK it even surpasses the real test-set baseline (42.5 vs. 37.2 mAP), demonstrating that generated degraded instances precisely follow the layout.

### DIOR-H Remote Sensing Comparison (Multiple Methods)

| Method | FID↓ | YOLO mAP↑ | AP_50↑ | AP_75↑ |
|------|------|----------|--------|--------|
| LayoutDiffusion | 45.31 | 20.0 | 37.4 | 19.3 |
| GLIGEN | 41.31 | 25.8 | 44.4 | 27.8 |
| AeroGen | 38.57 | 29.8 | 54.2 | 31.6 |
| CC-Diff | 30.88 | 26.4 | 44.2 | 28.5 |
| **FICGen** | 31.25 | **31.2** | **49.9** | **34.6** |

FICGen achieves the best mAP under YOLO evaluation as well.

### Downstream Trainability (Data Augmentation Effect)
When FICGen-synthesized data is used to augment downstream detector training:
- Consistent improvement of ~2.0 mAP.
- Significant gains on specific categories: "airport" class in remote sensing **+6.0 AP** (32.2→38.1).
- On ExDARK, Cascade R-CNN mAP improves from 37.2 to 42.5.

### Deformable-DETR Validation

| Dataset | Method | mAP↑ | AP_50↑ | AP_75↑ |
|--------|------|------|--------|--------|
| ExDARK | CC-Diff | 31.3 | 61.8 | 28.8 |
| ExDARK | **FICGen** | **38.5** | **68.5** | **39.5** |
| RUOD | CC-Diff | 29.7 | 57.8 | 28.0 |
| RUOD | **FICGen** | **37.1** | **67.1** | **36.7** |

FICGen's advantage is even more pronounced when evaluated with the stronger Deformable-DETR detector.

## Highlights & Insights
- **First systematic treatment of L2I generation for degraded scenes**: introduces the concept of the "contextual illusion dilemma" and provides a frequency-domain solution.
- **Elegant design of frequency prototypes**: explicitly models the frequency characteristics of degraded scenes (HF instance attenuation + LF background dominance) as learnable prototypes.
- **Clean dual-query architecture**: the HF query focuses on instance details while the LF query captures environmental atmosphere, with a clear division of roles.
- **Simple yet effective instance consistency maps**: binary masks enable latent-space disentanglement, preventing attribute leakage and object merging.
- **Broad validation across five degraded scenarios**: from severe low-light to mild blur, demonstrating the generality of the approach.
- **Practical data augmentation capability**: synthesized data directly improves downstream detector performance.

## Limitations & Future Work
- Frequency prototypes are constructed by sampling degraded instances from the training set, making the method dependent on the coverage of degradation patterns in the training data.
- The implementation is based on SDv1.5; adaptation to newer foundation models is not explored.
- The $\gamma$ parameter (controlling HF region size) requires manual specification.
- Robustness to extreme degradation (e.g., near-total darkness in low-light) is not thoroughly validated.
- Downstream trainability is verified only on object detection; other tasks such as segmentation remain unexplored.

## Related Work & Insights
- **Text-driven image synthesis**: diffusion/autoregressive models such as DALL-E and LDM.
- **Layout-driven image synthesis**: GLIGEN, LayoutDiffusion, MIGC (multi-instance control), CC-Diff (contextual consistency).
- **Degraded scene generation**: AeroGen (remote sensing) is a pioneer but is limited by semantic ambiguity and insufficient layout controllability.

## Rating
- Novelty: ⭐⭐⭐⭐ — The frequency-domain perspective on degraded-scene L2I generation is original.
- Technical Depth: ⭐⭐⭐⭐ — A complete design chain of dual-query + frequency prototypes + instance disentanglement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, multiple detectors, and downstream training validation.
- Practical Value: ⭐⭐⭐⭐ — Data augmentation for degraded scenes addresses a genuine engineering need.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation](the_silent_assistant_noisequery_as_implicit_guidance_for_goal-driven_image_gener.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[ICCV 2025\] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)
- [\[NeurIPS 2025\] Contextual Thompson Sampling via Generation of Missing Data](../../NeurIPS2025/image_generation/contextual_thompson_sampling_via_generation_of_missing_data.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)

</div>

<!-- RELATED:END -->
