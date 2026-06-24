---
title: >-
  [Paper Note] JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation
description: >-
  [ECCV 2024][3D Vision][Text-to-3D] Proposes Joint Score Distillation (JSD), which models the joint distribution of multi-view denoised images through an energy function, extending SDS from single-view independent optimization to multi-view joint optimization. This effectively resolves the Janus problem in 3D generation while maintaining generation fidelity for complex text prompts.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Text-to-3D"
  - "Score Distillation"
  - "Janus Problem"
  - "Multi-view Consistency"
  - "Energy Functions"
date: 2026-05-08
content_hash: 2ea0680c7366a051
---

# JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation

**Conference**: ECCV 2024  
**arXiv**: [2407.12291](https://arxiv.org/abs/2407.12291)  
**Code**: [Project Page](https://jointdreamer.github.io)  
**Area**: 3D Vision  
**Keywords**: Text-to-3D, Score Distillation, Janus Problem, Multi-view Consistency, Energy Functions

## TL;DR

Proposes Joint Score Distillation (JSD), which models the joint distribution of multi-view denoised images through an energy function, extending SDS from single-view independent optimization to multi-view joint optimization. This effectively resolves the Janus problem in 3D generation while maintaining generation fidelity for complex text prompts.

## Background & Motivation

**Background**: Score Distillation Sampling (SDS) is the mainstream paradigm for text-to-3D generation, utilizing the image distribution prior of pretrained 2D diffusion models to optimize 3D representations such as NeRF. Methods like DreamFusion, Magic3D, and ProlificDreamer have made significant progress.

**Limitations of Prior Work**: SDS optimizes each rendered view independently, inheriting the view-agnostic nature of 2D diffusion models, which results in the severe **Multi-Face Janus Problem**—wherein 3D assets exhibit repeated content (such as multiple faces) from different viewing angles.

**Key Challenge**: Existing solutions either offer limited efficacy (such as prompt engineering) or overfit and lose text fidelity when fine-tuned on limited 3D data (e.g., MVDream missing semantic components when handling complex prompts). **It is difficult to balance both geometric consistency and text congruence.**

**Goal**: Starting from the SDS optimization paradigm itself, introduce multi-view consistency constraints to eliminate the Janus problem without sacrificing the generalization ability of the diffusion model.

**Key Insight**: Model the joint distribution of multi-view images using an energy function, theoretically deriving the multi-view KL divergence to obtain the Joint Score Distillation function.

**Core Idea**: SDS is a special case of JSD when the energy term is zero—introducing a vision-aware energy function allows a natural transition from single-view optimization to multi-view joint optimization.

## Method

### Overall Architecture

The JointDreamer framework is built upon the Instant-NGP NeRF representation, core components include:

1. **Joint Score Distillation (JSD)**: Extends the single-view KL divergence of SDS to a multi-view version.
2. **General Vision Perception Model as the Energy Function**: Three models (binary classifier / image-to-image translation / multi-view generation).
3. **Geometry Fading scheme**: Focuses on geometry early on and texture later.
4. **CFG Switching strategy**: Low CFG to preserve shape early on, high CFG to enhance texture later.

### Key Designs

1. **Joint Score Distillation (JSD)**: 
   SDS minimizes the single-view KL divergence $D_{KL}(q_t^\theta(\mathbf{x}_t|c,y) \| p_t(\mathbf{x}_t|y))$. JSD extends this to a multi-view joint distribution. An energy function $\mathcal{C}(\tilde{\mathbf{x}}, \tilde{\mathbf{c}})$ is introduced to measure consistency among multi-view denoised images:

$$p_0(\tilde{\mathbf{x}}|\tilde{\mathbf{c}}, y) \propto \exp(\mathcal{C}(\tilde{\mathbf{x}}, \tilde{\mathbf{c}})) \prod_{i=1}^{V} p_0(\mathbf{x}^i|c^i, y)$$

   Larger values of $\mathcal{C}$ indicate stronger consistency among views. From this, the multi-view KL divergence and JSD gradient are derived:

$$\nabla_\theta L_{JSD}(\theta) = \sum_{i=1}^{V} \mathbb{E}_{t, \epsilon^i_\Phi} \left[w(t)\left(\hat{\epsilon}_\Phi(\mathbf{x}_t^i, y) - \frac{\partial \mathcal{C}(\tilde{\mathbf{x}})}{\partial \mathbf{x}_t^i} - \epsilon^i\right) \frac{\delta g(\theta, c^i)}{\delta\theta}\right]$$

   Key insight: When $\mathcal{C} \equiv 0$, JSD degenerates to SDS—implying that **SDS is a special case of JSD** that lacks inter-view consistency constraints.

   Design Motivation: Image distributions of 2D diffusion models are view-agnostic, and independent sampling across views naturally leads to inconsistency. Refining the independent distribution into a joint distribution via the energy term elegantly addresses the root cause of the Janus problem from a probabilistic perspective.

2. **Three Vision Perception Energy Functions**: To demonstrate the generalizability of JSD, the paper instantiates three energy functions:

   **(a) Binary Classification Model $M_{CLS}$**: Based on a DINO-ViT/s16 backbone, it determines whether two views originate from the same 3D object. Given an image pair $(x^i, x^j)$ and relative pose $\Delta(c^j, c^i)$, it outputs a consistency binary classification score:
    $\mathcal{C}_{CLS}(\tilde{\mathbf{x}}, \tilde{\mathbf{c}}) = \sum_{i,j; i\neq j} M_{CLS}(\mathbf{x}_t^i, \mathbf{x}_t^j, \Delta(c^j, c^i))$

   **(b) Image Translation Model $M_{I2I}$**: Uses Wonder3D to synthesize a target view from a reference view, using reconstruction loss to measure consistency:
    $\mathcal{C}_{I2I} = -\sum_i \|M_{I2I}(\mathbf{x}_t^{ref}, \Delta(c^i, c^{ref})) - \mathbf{x}_t^i\|_2^2$

   **(c) Multi-view Generation Model $M_{MVS}$**: Uses MVDream to directly generate multi-view images and calculate reconstruction loss:
    $\mathcal{C}_{MVS} = -\|M_{MVS}(y, \tilde{\mathbf{c}}) - \tilde{\mathbf{x}}\|_2^2$

   Design Motivation: Different energy functions provide different perspectives of 3D perception—the classifier provides coarse-grained structural judgment, the translation model provides fine-grained reconstruction reference, and the multi-view generation model provides the most direct multi-view consistency. The JSD framework is naturally compatible with various choices of energy functions.

3. **Geometry Fading and CFG Switching**: 

    - **Geometry Fading**: Starting from the 5K iteration, the learning rate of the NeRF density network is decreased from $1\times10^{-2}$ to $1\times10^{-6}$, and the orientation loss is set to 0. Focusing on geometry convergence in the early stage allows resources to be freed up for texture optimization in the late stage.
    - **CFG Switching**: In the first 5K iterations, a small CFG $s=30$ is used to maintain shape integrity and allow the consistency guidance of JSD to take effect; subsequently, it switches to $s=50$ to enhance texture fidelity.

### Loss & Training

- NeRF representation based on Instant-NGP + Volume Renderer.
- JSD loss replaces traditional SDS loss, using MVDream as the default energy function.
- Uses common techniques such as time-annealing and resolution scaling-up.
- The default rendering resolution is $64 \times 64$, and a 3D asset can be generated in 5K iterations.
- In CFG, $\hat{\epsilon}_\Phi := (1+s)\epsilon_\Phi(\mathbf{x}_t, t, y) - s\epsilon_\Phi(\mathbf{x}_t, t, \emptyset)$.

## Key Experimental Results

### Main Results — Quantitative Evaluation of Text Congruence (MS-COCO 153 prompts)

| Method | CLIP Score↑ | R-Precision(%)↑ | User Study(%)↑ |
|------|-------------|-----------------|----------------|
| DreamFusion | 20.1 | 27.7 | 18.2 |
| ProlificDreamer | 25.0 | 18.7 | 16.2 |
| MVDream | 20.8 | 33.6 | 23.5 |
| **JointDreamer** | **27.7** | **88.5** | **42.1** |

### Ablation Study — CFG Switching + Geometry Fading

| SDS | JSD | CFGS | GF | CLIP Score↑ | FID↓ |
|-----|-----|------|----|-------------|------|
| ✓ | | | | 20.0 | 429.2 |
| | ✓ | | | 27.6 | 360.7 |
| | ✓ | ✓ | | 28.2 | 357.6 |
| | ✓ | ✓ | ✓ | **28.8** | **353.9** |

### Ablation Study — Janus Elimination Rate of Different Energy Functions

| Method | Janus Rate↓ | GPU Memory | Training Time |
|------|------------|------------|---------|
| SDS (Baseline) | 100% | 16.1G | 50 min |
| JSD + $\mathcal{C}_{CLS}$ | 12.5% | 22.1G | 80 min |
| JSD + $\mathcal{C}_{I2I}$ | 31.2% | 16.0G | 119 min |
| JSD + $\mathcal{C}_{MVS}$ | **6.2%** | 19.4G | 54 min |

### Key Findings

- JSD's R-Precision reaches 88.5%, which is a 60.8% gain over DreamFusion and a 54.9% gain over MVDream.
- The Janus Rate of SDS is 100% (occurring in all 16 complex prompts), whereas JSD + MVDream reduces it to 6.2%.
- The JSD training loss curve is significantly smoother and has more stable convergence—multi-view optimization eliminates single-view randomness.
- While VSD in ProlificDreamer enhances photorealism, the weak pose-image association of LoRA actually worsens geometric inconsistency.
- The image translation model ($\mathcal{C}_{I2I}$) performs poorly, possibly due to camera range mismatches.
- Simply combining "SDS + MVDream" (weighted sum) fails to balance geometry and text, whereas JSD naturally unifies them from a distribution perspective.

## Highlights & Insights

- Elegantly proves from a probabilistic perspective that SDS is a special case of JSD, with rigorous theoretical derivation.
- JSD features strong generalizability to energy functions—even a simple binary classifier can drastically reduce the Janus Rate.
- The insight of Geometry Fading is intuitive and effective: geometry and texture need to be prioritized in a phased manner.
- R-Precision jumps directly from 27.7% to 88.5%, a monumental leap from qualitative to quantitative change.

## Limitations & Future Work

- Training time is acceptable but still has room for acceleration; highly efficient representations like 3D Gaussian Splatting have not been explored.
- The energy function models still require 3D data training; dependency on data has not been fully eliminated.
- Currently, only object-centric generation has been validated; it has not been extended to scene-level text-to-3D.
- The thresholds for CFG Switching (5K iterations, s=30 $\to$ 50) are manually set.

## Related Work & Insights

- **DreamFusion / SDS**: The foundational paradigm of JSD; this work fundamentally addresses its missing multi-view constraint.
- **MVDream**: Solves consistency by fine-tuning multi-view diffusion models, but overfitting limits generalization. JSD utilizes it as an energy function instead of direct distillation, retaining the original diffusion model's generalization capability.
- **ProlificDreamer / VSD**: Enhances texture quality but worsens geometric inconsistency, which is orthogonal to JSD's approach.
- **Energy-Based Modeling**: Models the joint distribution leveraging Energy-Based Models (EBM) concepts, demonstrating a successful cross-domain application of inspiration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — JSD elegantly remedies the fundamental shortcoming of SDS, offering outstanding theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative, qualitative, ablation, and user studies, though the scene coverage could be broader.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The step-by-step derivation from SDS to JSD is progressive, and the comparative design of the three energy functions is ingenious.
- **Value**: ⭐⭐⭐⭐⭐ — Solves the Janus problem at a paradigm level, with a profound impact on future 3D generation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)
- [\[ICCV 2025\] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](../../ICCV2025/3d_vision/advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] GVGEN: Text-to-3D Generation with Volumetric Representation](gvgen_text-to-3d_generation_with_volumetric_representation.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
