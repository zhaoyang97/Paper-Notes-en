---
title: >-
  [Paper Note] GeoRemover: Removing Objects and Their Causal Visual Artifacts
description: >-
  [NeurIPS2025][Image Generation][Object Removal] GeoRemover is a geometry-aware two-stage framework that decouples object removal into geometric removal (depth domain) and appearance rendering (RGB domain). By modifying t…
tags:
  - "NeurIPS2025"
  - "Image Generation"
  - "Object Removal"
  - "Causal Visual Artifacts"
  - "Geometry-Awareness"
  - "Depth Maps"
  - "Diffusion Models"
  - "DPO"
date: 2026-05-08
content_hash: 581b090ffea52a32
---

# GeoRemover: Removing Objects and Their Causal Visual Artifacts

**Conference**: NeurIPS2025
**arXiv**: [2509.18538](https://arxiv.org/abs/2509.18538)  
**Code**: [Project Page](https://buxiangzhiren.github.io/GeoRemover)  
**Area**: Image Generation
**Keywords**: Object Removal, Causal Visual Artifacts, Geometry-Awareness, Depth Maps, Diffusion Models, DPO

## TL;DR
GeoRemover is a geometry-aware two-stage framework that decouples object removal into geometric removal (depth domain) and appearance rendering (RGB domain). By modifying the scene's geometric representation, it implicitly eliminates causal visual artifacts—such as shadows and reflections—left by the removed object.

## Background & Motivation

Object removal is one of the core tasks in image editing. Existing methods face a critical challenge: when an object is removed, its **causal visual artifacts** (e.g., shadows, reflections) often persist in the image, producing unnatural editing results.

Existing methods fall into two categories:

1. **Strictly Mask-Aligned Methods**: Edit only within the user-annotated mask region and cannot handle shadow/reflection artifacts outside the mask. Eliminating these artifacts requires manual annotation of all artifact regions, which is neither scalable nor user-friendly.
2. **Loosely Mask-Aligned Methods**: Allow the model to modify regions outside the mask to infer and remove artifacts, but lack explicit boundary guidance, making them prone to **over-erasure**—inadvertently removing unrelated surrounding objects.

A key observation of this work is that visual artifacts such as shadows and reflections are causal consequences of an object's **geometry presence** under specific lighting conditions. Removing the object's presence from the scene geometry naturally eliminates its associated lighting effects.

## Core Problem

How can an object be removed while automatically eliminating its causal visual artifacts (shadows, reflections) without causing uncontrolled modifications to unannotated regions?

## Method

### Overall Architecture: Geometry–Appearance Decoupling

GeoRemover decouples object removal into two sub-tasks:

$$x_0^- = \mathcal{D}(I^-), \quad x_0^+ = s_\theta(x_0^-, M) \quad \text{(Stage 1: Geometric Removal)}$$

$$I^+ = \mathcal{G}(I^-, x_0^-, x_0^+) \quad \text{(Stage 2: Appearance Rendering)}$$

where $\mathcal{D}$ is a depth estimator (Depth Anything), $s_\theta$ is the geometric removal diffusion model, and $\mathcal{G}$ is the appearance rendering diffusion model.

### Stage 1: Geometric Removal

**Mechanism**: The object is removed in the depth domain. Since shadows and reflections are invisible in depth maps, the depth domain is naturally suited to strictly mask-aligned training.

- Input: Depth estimate $x_0$ of the RGB image + object mask $M$
- Objective: Predict the edited depth map $\hat{x}_0$, removing object geometry within the mask while preserving regions outside it
- Backbone: FLUX.1-Fill-dev + LoRA (rank=64) fine-tuning

**Preference-Guided Geometric Completion (DPO)**:

Direct training of a diffusion model for depth completion causes the model to frequently **hallucinate new structures** (e.g., inserting nonexistent geometry in the masked region). To address this, a DPO strategy is introduced:

- **Positive samples** $x_0^+$: The object is successfully removed; depth in the masked region is smooth with depth flow approaching zero
- **Negative samples** $x_0^-$: The object remains present; depth discontinuities exist within the masked region

Depth flow is defined as the spatial gradient:

$$F_{ij}(x) = \{|d_{i+1,j} - d_{i,j}|, \; |d_{i,j+1} - d_{i,j}|\}$$

The flow loss measures the discrepancy between predicted and ground-truth flows:

$$\mathcal{L}_{\text{flow}}(\hat{x}_0, x_0) = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega} \|F_{ij}(\hat{x}_0) - F_{ij}(x_0)\|_1$$

Preference probability is modeled via the Bradley–Terry model, and the final loss is:

$$\mathcal{L} = \mathcal{L}_{\text{DSM}} + \lambda \mathcal{L}_{\text{BT}}$$

### Stage 2: Appearance Rendering

**Mechanism**: The masked RGB image is translated into an object-free image conditioned on geometric changes.

- Input: Masked RGB image + pre-edit depth map $x_0^-$ + post-edit depth map $x_0^+$
- The difference between the two depth maps allows the model to localize the removed object and learn the causal relationship between objects and their artifacts from paired images

**Bidirectional Training**: Both removal and insertion directions are trained simultaneously to enhance the model's understanding of geometry–appearance correspondence:

$$I^+ = \mathcal{G}(I^- \mid x_0^-, x_0^+), \quad I^- = \mathcal{G}(I^+ \mid x_0^+, x_0^-)$$

Implementation concatenates the RGB image and two colorized depth maps along the width dimension into an $H \times 3W \times 3$ tensor, which is fed directly into the diffusion model.

### Failure Case Handling: Local Max Depth Fill-in

When motion blur or transparent/reflective surfaces cause unreliable depth estimation, the pre- and post-edit depth difference may be insufficient to trigger removal in Stage 2. In such cases, **Local Max Depth Fill-in** is applied: for pixels within the mask that lack reliable estimates, the maximum depth value is propagated from a $10 \times 10$ local neighborhood to restore boundary contrast.

## Key Experimental Results

### Main Results (RemovalBench & RORD-Val)

| Method | FID↓ | CMMD↓ | LPIPS↓ | PSNR↑ |
|--------|------|-------|--------|-------|
| LaMa | 99.88 | 0.351 | 0.156 | 18.72 |
| Attentive-Eraser | 55.49 | 0.232 | 0.146 | 20.60 |
| OmniEraser | 39.52 | 0.208 | 0.133 | 21.11 |
| **GeoRemover** | **29.88** | **0.089** | **0.124** | **25.52** |

On RemovalBench, GeoRemover reduces FID by 24.4% and improves PSNR by 4.41 dB over OmniEraser.

### Ablation Study (RORD-Val)

| Method | FID↓ | PSNR↑ | Insert.↓ |
|--------|------|-------|----------|
| One-Stage | 56.24 | 17.52 | 2.81% |
| Two-Stage w/o DPO | 34.24 | 22.81 | 5.09% |
| Two-Stage w/ DPO | **31.15** | **23.70** | **1.48%** |

DPO reduces the structural hallucination insertion rate from 5.09% to 1.48%.

### Causal Artifact Removal (CausRem)

| Method | IoU% ↑ |
|--------|--------|
| OmniEraser | 68.29 |
| **GeoRemover** | **73.76** |

### Geometric Removal Accuracy (MAE)

| Method | MAE↓ |
|--------|------|
| Input Depth | 0.0827 |
| Two-Stage w/o DPO | 0.0490 |
| Two-Stage w/ DPO | **0.0387** |

## Highlights & Insights

1. **Causal Reasoning Perspective**: Object removal is reframed as a causal reasoning process—geometric presence is the cause, visual artifacts are the effect; eliminating the cause naturally removes the effect.
2. **Geometry–Appearance Decoupling**: Strictly mask-aligned geometric editing is performed in the depth domain (avoiding the uncontrollability of loose alignment), while implicit artifact elimination is handled in the RGB domain (avoiding the limited capability of strict alignment).
3. **DPO for Geometric Completion**: DPO is innovatively applied to depth map completion, constructing positive/negative sample pairs via depth flow smoothness to effectively suppress structural hallucinations in the diffusion model.
4. **Bidirectional Rendering Training**: Simultaneously training on both removal and insertion directions strengthens the model's understanding of the correspondence between geometric and appearance changes.
5. **Comprehensive Experimental Superiority**: GeoRemover outperforms all baselines across three benchmarks: RemovalBench, RORD-Val, and CausRem.

## Limitations & Future Work

1. **Depth Estimation Failure Cases**: Motion blur, high transparency, and specular reflections lead to unreliable depth estimates, requiring the additional Fill-in strategy as a remedy.
2. **Two-Stage Computational Cost**: Compared to single-stage methods, two diffusion inference passes are required; Stage 1 training takes approximately 24 hours and Stage 2 approximately 60 hours (8×H100).
3. **Self-Luminous Objects**: For self-luminous objects such as colored light bulbs, Stage 2 may generate diffuse glow rather than clean removal.
4. **Incomplete Masks**: When the mask does not fully cover the object, Stage 1 may hallucinate a complete object rather than removing it.
5. **Residual Reflections**: Color bleeding from semi-transparent objects on reflective surfaces may still persist.
6. **Training Data Dependency**: Training relies solely on the RORD dataset of indoor scenes; generalization to outdoor scenes remains to be verified.

## Related Work & Insights

| Method | Type | Artifact Handling | Controllability | Key Difference |
|--------|------|------------------|-----------------|----------------|
| CLIPAway | Strict Alignment | ✗ | ✓ | CLIP-guided but cannot handle artifacts outside the mask |
| Attentive-Eraser | Strict Alignment | Partial | ✓ | Attention redirection but limited artifact handling |
| OmniEraser | Loose Alignment | ✓ | Partial | Video frame paired data but limited controllability |
| ObjectDrop | Loose Alignment | ✓ | Partial | Counterfactual bootstrapping; code not released |
| **GeoRemover** | Geometry Decoupled | ✓ | ✓ | Strict alignment in geometry domain + implicit artifact removal in RGB domain |

GeoRemover's core advantage lies in combining the strengths of both categories: precise controllability via strict alignment in the geometry domain, and implicit artifact elimination guided by geometry in the appearance domain.

The following broader insights are also noted:

1. **Universal Value of Geometry as an Intermediate Representation**: Depth maps, as scene representations free of visual artifacts, offer a valuable intermediate representation for various lighting-related editing tasks (e.g., relighting, scene composition).
2. **DPO in Low-Level Vision**: Extending RLHF/DPO from language models and image generation to low-level vision tasks such as geometric completion suggests that preference learning can be introduced into a wider range of traditional computer vision tasks.
3. **Causal Analysis-Driven Method Design**: Rather than learning an end-to-end mapping directly, analyzing the causal chain in the task (geometry → lighting → artifacts) and designing a modular pipeline accordingly is an approach worth adopting in other editing tasks.
4. **Unexpected Extension to Watermark Removal**: By injecting pseudo depth cues (Local Max Depth Fill-in), the framework generalizes to non-geometric inpainting tasks such as watermark removal, suggesting the methodology is more broadly applicable than initially anticipated.

## Rating
- Novelty: 8/10 — The causal reasoning perspective and geometry–appearance decoupling design are novel; applying DPO to geometric completion is innovative.
- Experimental Thoroughness: 8/10 — Three benchmarks, multiple metrics, comprehensive ablation studies, and failure case analysis.
- Writing Quality: 8/10 — Motivation is clearly articulated, the methodological logic chain is complete, and figures are intuitive.
- Value: 8/10 — Establishes a new paradigm for object removal; the geometry decoupling approach offers insights for other editing tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)
- [\[NeurIPS 2025\] V-CECE: Visual Counterfactual Explanations via Conceptual Edits](v-cece_visual_counterfactual_explanations_via_conceptual_edits.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](../../ICML2026/image_generation/caracal_causal_architecture_via_spectral_mixing.md)

</div>

<!-- RELATED:END -->
