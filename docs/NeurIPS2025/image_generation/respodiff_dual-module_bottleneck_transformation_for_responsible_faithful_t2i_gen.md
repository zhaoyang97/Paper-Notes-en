---
title: >-
  [Paper Note] RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation
description: >-
  [NeurIPS 2025][Image Generation][Responsible Generation] This paper proposes RespoDiff, a framework that introduces two learnable transformation modules at the bottleneck layer of a diffusion model UNet — a Responsibility Alignment Module (RAM) and a Semantic Alignment Module (SAM) — trained via score matching objectives to achieve fair and safe text-to-image generation while preserving image quality and semantic fidelity.
tags:
  - NeurIPS 2025
  - Image Generation
  - Responsible Generation
  - Fairness
  - Safety
  - Bottleneck Transformation
  - Score Matching
date: 2026-05-08
content_hash: 35eb8a877b44e22e
---

# RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation

**Conference**: NeurIPS 2025
**arXiv**: [2509.15257](https://arxiv.org/abs/2509.15257)
**Code**: [Project Page](https://vssilpa.github.io/respodiff_project_page)
**Area**: Diffusion Models / Image Generation
**Keywords**: Responsible Generation, Fairness, Safety, Bottleneck Transformation, Score Matching

## TL;DR

This paper proposes RespoDiff, a framework that introduces two learnable transformation modules at the bottleneck layer of a diffusion model UNet — a Responsibility Alignment Module (RAM) and a Semantic Alignment Module (SAM) — trained via score matching objectives to achieve fair and safe text-to-image generation while preserving image quality and semantic fidelity.

## Background & Motivation

T2I models such as Stable Diffusion, SDXL, and FLUX achieve high generation quality but exhibit serious social biases:
- **Gender bias**: Prompts like "a doctor" predominantly generate male figures.
- **Racial bias**: Generated outputs skew toward specific skin tones.
- **Safety risks**: Models may produce violent, explicit, or otherwise inappropriate content.

**Limitations of Prior Work**:

**Prompt modification methods** (harmful token removal, prompt tuning): Limited capability; cannot achieve precise control.

**Model fine-tuning methods** (concept erasure, weight fine-tuning): Risk degrading original model performance; require retraining per prompt.

**Classifier guidance methods**: Require no additional training but offer insufficient fine-grained control.

**Latent injection methods** (e.g., SDisc): Lack explicit reference to neutral denoising latents during bottleneck-space manipulation, resulting in insufficient control precision.

**Root Cause**: Existing methods tend to sacrifice semantic fidelity and image quality when improving fairness or safety. Achieving responsible generation without compromising generation quality is the central challenge.

**Starting Point**: The paper introduces a **dual-path transformation** at the UNet bottleneck layer (demonstrated to be a semantic latent space): one path steers generation toward a target concept (e.g., "female"), while the other maintains consistency with the original diffusion trajectory, with the two paths mutually constraining each other.

## Method

### Overall Architecture

The UNet is decomposed into an encoder $e: \mathcal{Z} \times \mathcal{Y} \to \mathcal{H}$ and a decoder $g: \mathcal{H} \times \mathcal{Y} \to \mathcal{Z}$. A dual-module transformation is applied to the bottleneck representation $\boldsymbol{h}_{neu} \in \mathcal{H}$:

$$\hat{f}(y_{neu}) = g(\mathcal{T}_\theta^{resp,s}(\boldsymbol{h}_{neu}) + \mathcal{T}_\theta^{sem,s}(\boldsymbol{h}_{neu}))$$

### Key Designs

1. **Responsibility Alignment Module (RAM, $\mathcal{T}_\theta^{resp,s}$)**:

    - **Function**: Modifies the bottleneck representation of a neutral prompt (e.g., "a person") so that its diffusion trajectory aligns with a target concept (e.g., "a woman").
    - **Score matching loss**: At a randomly sampled timestep $t$, the neutral denoising latent $\boldsymbol{z}_{t,neu}$ is first obtained via reverse diffusion through the RAM-augmented model; then:
    $\mathcal{L}_{resp} = \mathbb{E}_{\boldsymbol{z}_{t,neu}} \left[\|\epsilon_{f_{resp}}(\boldsymbol{z}_{t,neu}, y_{neu}) - \epsilon_f(\boldsymbol{z}_{t,neu}, y_{tar}^s)\|_2^2\right]$
    - **Novelty**: The neutral denoising latent serves as a stable anchor, enabling precise directional guidance by contrasting UNet predictions under the neutral and target concepts.
    - Only $\mathcal{T}_\theta^{resp,s}$ is updated at this stage; the full dual-module transformation is not yet involved.

2. **Semantic Alignment Module (SAM, $\mathcal{T}_\theta^{sem,s}$)**:

    - **Function**: Prevents excessive deviation caused by the RAM transformation by maintaining semantic consistency with the original diffusion model trajectory.
    - **Score matching loss**:
    $\mathcal{L}_{sem} = \mathbb{E}_{\boldsymbol{z}_{t,neu}} \left[\|\epsilon_{\hat{f}}(\boldsymbol{z}_{t,neu}, y_{neu}) - \epsilon_f(\boldsymbol{z}_{t,neu}, y_{neu})\|_2^2\right]$
    - The full dual-module transformation $\hat{f}$ is used at this stage, but only $\mathcal{T}_\theta^{sem,s}$ is updated.
    - The loss is weighted by $\lambda$ (default $\lambda=0.5$).

3. **Alternating Training Strategy**:

    - Two alternating steps: first update RAM (using only $\mathcal{L}_{resp}$), then update SAM (using only $\lambda \mathcal{L}_{sem}$).
    - Backpropagation is not performed through the reverse diffusion process, reducing computational overhead.
    - As training progresses, the neutral denoising latent progressively aligns with both the target concept and the original diffusion trajectory.

4. **Flexible Inference**:

    - **Fairness**: A transformation is learned for each sensitive concept $s \in \mathcal{S}_c$; at inference time, one is randomly selected to achieve a uniform distribution.
    - **Safety**: Negative concepts (violence, nudity) are treated as negative prompts; "anti-violence" and "anti-explicit" transformations are learned and aggregated at inference.

### Loss & Training

- Transformations are implemented as constant functions added linearly to bottleneck activations.
- Fairness training: 5,000 iterations, batch size 1.
- Safety training: 1,500 iterations, batch size 1.
- Training uses neutral prompts "a person" (fairness) or "a scene" (safety); **no occupation- or scene-specific data is required**.
- UNet weights are frozen throughout training; only the lightweight transformation modules are updated.

## Key Experimental Results

### Main Results — Gender Fairness (SD v1.4, Winobias 36 Occupations)

| Method | DevRat↓ | WinoAlign↑ | FID(30K)↓ | CLIP(30K)↑ |
|--------|---------|-----------|-----------|-----------|
| SD (baseline) | 0.68 | 27.51 | 14.09 | 31.33 |
| SDisc | 0.17 | 26.61 | 23.59 | 29.94 |
| FDF | 0.40 | 23.90 | 15.22 | 30.63 |
| BAct | 0.57 | 27.67 | 17.07 | 30.54 |
| **RespoDiff** | **0.14** | **27.30** | **14.91** | **30.67** |

### Racial Fairness (SD v1.4)

| Method | DevRat↓ | WinoAlign↑ | FID(30K)↓ | CLIP(30K)↑ |
|--------|---------|-----------|-----------|-----------|
| SD (baseline) | 0.56 | 27.51 | 14.09 | 31.33 |
| SDisc | 0.23 | 26.80 | 17.47 | 30.27 |
| **RespoDiff** | **0.16** | **27.53** | **12.82** | **31.02** |

### Safe Generation (I2P Benchmark, SD v1.4)

| Method | I2P Inappropriate↓ | FID(30K)↓ | CLIP(30K)↑ |
|--------|-------------------|-----------|-----------|
| SD | 0.27 | 14.09 | 31.33 |
| SLD | 0.20 | 18.76 | 29.75 |
| ESD | 0.32 | 13.68 | 30.43 |
| **RespoDiff** | **0.16** | **17.89** | **31.10** |

### Ablation Study

| Configuration | DevRat↓ | WinoAlign↑ | FID↓ | CLIP↑ |
|--------------|---------|-----------|------|-------|
| RAM only | 0.12 | 26.12 | 15.63 | 29.93 |
| RAM+SAM (full) | 0.14 | 27.30 | 14.91 | 30.67 |
| Shared module | 0.16 | 26.12 | 15.63 | 29.93 |
| $\lambda=0$ (no SAM) | 0.12 | 26.12 | 15.63 | 29.93 |
| $\lambda=0.5$ (default) | 0.14 | 27.30 | 14.91 | 30.67 |
| $\lambda=4$ | 0.29 | 27.53 | 14.17 | 31.24 |

### Key Findings

- RespoDiff outperforms SLD in inappropriate content filtering by approximately 20%, while also achieving superior text-image alignment.
- Training solely on "a person" generalizes to 36 specific occupations without requiring occupation-specific data.
- Transfer to SDXL: gender DevRat decreases from 0.72 to 0.26; racial DevRat from 0.57 to 0.23.
- Separating the two modules outperforms a shared-module design, as each module can focus exclusively on its own optimization objective.
- $\lambda=0.5$ is the inflection point: smaller values cause over-steering toward the target concept; larger values result in excessive conservatism.

## Highlights & Insights

- **Decoupled dual-module design**: RAM and SAM fulfill distinct roles, avoiding the zero-sum trade-off between fairness and image quality.
- **Neutral denoising latent as anchor**: Provides more precise directional control than indirect reconstruction from target images.
- **No occupation- or scene-specific data required**: Strong generalization from generic neutral prompts to diverse scenarios.
- **Scalable to large models such as SDXL**: The lightweight module design enables low-cost adaptation to larger architectures.
- **Composable fairness and safety**: The modular design supports independent learning and flexible combination at inference time.

## Limitations & Future Work

- Fairness and safety concepts must be predefined — the framework cannot automatically discover emerging bias categories.
- The approach relies on predefined neutral prompts; although experiments demonstrate robustness to alternative prompts, automatic selection remains an open problem.
- Transformations are implemented as constant functions; more expressive structures may enable finer-grained control.
- Cross-attribute biases (e.g., gender × race) require composing existing modules, but have not been systematically evaluated.

## Related Work & Insights

- RespoDiff operates in the bottleneck space similarly to SDisc, but introduces dual modules and explicit trajectory alignment to achieve a better balance.
- The score matching objective elegantly reformulates trajectory alignment as a noise prediction matching problem, yielding a theoretically principled formulation.
- The modular design paradigm is extensible to other concept control scenarios beyond fairness and safety.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-module score matching framework represents a substantive improvement over prior bottleneck-space manipulation methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage of Winobias and I2P benchmarks, including SDXL transfer experiments and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The framework is clearly described, though the dense notation requires some patience on first reading.
- **Value**: ⭐⭐⭐⭐ — A significant contribution to responsible AI generation with strong practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[NeurIPS 2025\] Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)
- [\[CVPR 2026\] Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](../../CVPR2026/image_generation/adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)

<!-- RELATED:END -->
