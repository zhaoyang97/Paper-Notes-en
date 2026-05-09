---
title: >-
  [Paper Note] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation
description: >-
  [ICCV 2025][Image Generation][Long-sequence motion generation] InfiniDreamer leverages a pretrained short-sequence motion diffusion model as a prior and proposes Segment Score Distillation (SSD), an optimization method that iteratively refines overlapping short segments within a coarsely initialized long motion sequence, enabling arbitrarily long human motion generation without requiring additional long-sequence training data.
tags:
  - ICCV 2025
  - Image Generation
  - Long-sequence motion generation
  - Score Distillation
  - sliding window
  - training-free
  - motion diffusion model
date: 2026-05-08
content_hash: f3fe79d3dd3ba7a2
---

# InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation

**Conference**: ICCV 2025
**arXiv**: [2411.18303](https://arxiv.org/abs/2411.18303)
**Code**: To be confirmed
**Area**: Human Motion Generation / Diffusion Models
**Keywords**: Long-sequence motion generation, Score Distillation, sliding window, training-free, motion diffusion model

## TL;DR
InfiniDreamer leverages a pretrained short-sequence motion diffusion model as a prior and proposes Segment Score Distillation (SSD), an optimization method that iteratively refines overlapping short segments within a coarsely initialized long motion sequence, enabling arbitrarily long human motion generation without requiring additional long-sequence training data.

## Background & Motivation

**Background**: Text-to-motion generation, driven by methods such as MDM, MLD, and T2M-GPT, has achieved high-quality short-sequence motion synthesis of approximately 10 seconds. However, practical applications in game animation, film production, and AR/VR typically demand continuous motion spanning several minutes or even hours.

**Limitations of Prior Work**:
   - **Data bottleneck**: High-quality long-sequence motion data is extremely scarce; existing datasets (HumanML3D, BABEL) predominantly consist of short sequences.
   - **Autoregressive methods** (TEACH, Multi-Act) accumulate errors, leading to motion drift, repetitive patterns, and "freezing" artifacts.
   - **Diffusion infilling methods** (PriorMDM/DoubleTake, DiffCollage) apply hard modifications at segment boundaries, often causing abrupt transitions, motion distortion, and overwriting of previously generated content.

**Key Challenge**: Generating motion sequences that exceed the length of training data requires long-sequence supervision signals that are unavailable, and existing compositional approaches handle boundaries in a heavy-handed manner.

**Goal**: Given a series of text descriptions, generate semantically coherent, smoothly transitioning, continuous motion sequences of arbitrary length.

**Key Insight**: Inspired by Score Distillation Sampling (SDS) from DreamFusion—whose key advantage is a progressive, smooth distillation process that maintains consistency across different viewpoints—this work transfers that advantage to the temporal dimension to achieve smooth transitions in long motion sequences.

**Core Idea**: The long motion sequence is parameterized as differentiable variables. A sliding window samples short segments, and Score Distillation aligns each segment to the distribution of the pretrained motion diffusion prior, achieving local realism alongside global coherence.

## Method

### Overall Architecture
The input is a series of text prompts $Y = \{y_1, y_2, ..., y_n\}$, and the output is a long motion sequence $M = \{m_1, t_1, m_2, t_2, ..., m_n\}$, where $m_i$ denotes the sub-motion corresponding to each text prompt and $t_i$ denotes the transition segment. The framework comprises three modules: motion sequence initialization, motion segment sampling, and segment score distillation.

### Key Designs

1. **Motion Sequence Initialization**

    - Function: Constructs a coarse long motion sequence as the starting point for optimization.
    - Mechanism: The entire long sequence is first randomly initialized, after which a pretrained MDM generates a short motion clip for each sub-motion $m_i$ conditioned on its corresponding text $y_i$; these clips are then inserted into the sequence. Transition segments $t_i$ remain randomly initialized, and neighboring regions are smoothed via linear interpolation.
    - Design Motivation: Providing an initial structure with random transition segments establishes a reasonable starting point for subsequent optimization. Gradient masks $Mask_l = 0.1$ (slower updates for sub-motion regions) and $Mask_h = 0.8$ (faster updates for transition regions) govern the optimization intensity across different regions.

2. **Motion Segment Sampling**

    - Function: Iteratively samples overlapping short segments from the long sequence for optimization.
    - Mechanism: A sliding window of size $W$ moves along the long sequence with stride $S$, sampling overlapping short segments $x_0^i$. For windows spanning multiple sub-motions, the text condition of one sub-motion is selected at random with equal probability.
    - Design Motivation: Overlap ensures continuity between adjacent segments, and the sliding window strategy guarantees that optimization covers the entire sequence.

3. **Segment Score Distillation (SSD)**

    - Function: Optimizes each short segment using the prior distribution of the pretrained motion diffusion model.
    - Mechanism: For each sampled segment $x_0^i$, a timestep $t$ is randomly sampled, noise is added to obtain $x_t^i$, and the diffusion model predicts the denoised result $\hat{x}_0^i = \phi(x_t^i; t, \varnothing)$. Optimization then proceeds via an alignment loss:
    $\mathcal{L}_{align} = \mathbf{E}_{t,\epsilon}[w(t) \|\hat{x}_0^i - x_0^i\|_2^2]$
    - Three geometric regularization losses are also applied: positional constraint $\mathcal{L}_{pos}$ (maintaining accurate joint positions via forward kinematics), foot contact constraint $\mathcal{L}_{foot}$ (preventing foot sliding), and velocity regularization $\mathcal{L}_{vel}$ (encouraging smooth transitions).
    - Design Motivation: Unlike infilling methods that apply hard modifications at boundaries, SSD performs progressive global optimization with small per-step updates, preserving existing motion structure. The overlapping windows ensure that transition segments are repeatedly refined throughout the optimization process.

### Loss & Training
The total loss is $\mathcal{L}_{ssd} = \mathcal{L}_{align} + \lambda_{pos}\mathcal{L}_{pos} + \lambda_{foot}\mathcal{L}_{foot} + \lambda_{vel}\mathcal{L}_{vel}$, where $\lambda = 0$ on HumanML3D and $\lambda = 0.1$ on BABEL. The AdamW optimizer is used with a learning rate of 0.002 for 20,000 iterations. Window size is set to $W=120$ and stride to $S=30$.

## Key Experimental Results

### Main Results (HumanML3D)

| Metric | DoubleTake | DiffCollage | **InfiniDreamer** |
|--------|-----------|-------------|-------------------|
| R-precision ↑ | 0.603 | 0.605 | **0.679** |
| FID ↓ (Motion) | 1.36 | 1.07 | **0.47** |
| Diversity → | 9.33 | 9.34 | **9.58** |
| MM-Dist ↓ | 4.27 | 3.62 | **3.15** |
| FID ↓ (Transition) | 3.19 | 4.27 | **2.04** |
| Diversity → (Trans) | 8.09 | 7.47 | **8.69** |

### Ablation Study (HumanML3D)

| Configuration | R-precision ↑ | FID ↓ | Trans FID ↓ |
|---------------|--------------|-------|------------|
| Full model | **0.679** | **0.47** | **2.04** |
| w/o gradient masks | 0.643 | 0.64 | 2.25 |

### BABEL Dataset Results

| Metric | TEACH | DoubleTake | DiffCollage | **InfiniDreamer** |
|--------|-------|-----------|-------------|-------------------|
| R-precision ↑ | 0.461 | 0.483 | 0.487 | **0.543** |
| FID ↓ | 1.43 | 1.14 | 1.83 | **0.97** |
| Trans FID ↓ | 4.23 | 3.54 | 4.62 | **2.07** |

### Key Findings
- InfiniDreamer outperforms all prior training-free methods across every metric, with motion FID dropping from 1.07 to 0.47 on HumanML3D.
- Transition quality improves substantially (Trans FID from 3.19 to 2.04), demonstrating that the progressive optimization of SSD is better suited to transition generation than infilling-based approaches.
- Ablation results confirm that gradient masks make a significant contribution to sub-motion quality.
- Learning rate requires careful tuning: too high causes motion collapse, while too low results in underfitting and motion distortion.
- Window size $W=120$ and stride $P=30$ constitute the optimal configuration; transition quality degrades sharply when $P \geq W$.

## Highlights & Insights
- **Transferring SDS from 3D generation to motion generation** is the paper's primary contribution. The progressive nature of SDS is inherently well-suited to tasks requiring global consistency—multi-view consistency in 3D and temporal consistency in motion.
- **The gradient mask design** elegantly allows sub-motion regions to update slowly (preserving text alignment) while transition regions update rapidly (converging quickly from random initialization), enabling differentiated optimization.
- **The decoupled design of short-sequence models and long-sequence generation** implies that future, more capable short-sequence motion diffusion models can be directly plugged into the framework for improved performance without retraining.

## Limitations & Future Work
- Optimization over 20,000 iterations is time-consuming, and efficiency warrants further improvement.
- The method still relies on the quality ceiling of short-sequence models such as MDM and does not itself learn new motion knowledge.
- Geometric losses (e.g., foot contact) may be insufficient for certain scenarios, such as complex interaction motions.
- The framework has only been validated with MDM as the prior; stronger latent diffusion models such as MLD remain unexplored.
- Window size is constrained by the maximum context of the short-sequence model (approximately 200 frames for MDM).

## Related Work & Insights
- **vs DoubleTake/PriorMDM**: These methods apply hard modifications at boundaries via infilling, resulting in abrupt transitions and overwritten motion; InfiniDreamer's progressive optimization avoids this issue.
- **vs DiffCollage**: Also a stitching-based approach but without global optimization; InfiniDreamer achieves global coherence through overlapping windows and SSD.
- **vs TEACH**: Autoregressive generation accumulates errors leading to motion drift; InfiniDreamer circumvents error accumulation through global optimization.
- **vs FlowMDM**: FlowMDM achieves seamless composition via Blended Positional Encodings but requires specialized training; InfiniDreamer is entirely training-free.

## Rating
- Novelty: ⭐⭐⭐⭐ Transferring SDS to the motion generation domain represents an elegant cross-domain innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both HumanML3D and BABEL with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method is described clearly, and the connection between SSD and SDS is well articulated.
- Value: ⭐⭐⭐⭐ A practical training-free solution for long-sequence motion generation; the decoupling of short-sequence models from long-sequence composition leaves room for future improvement.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[ICCV 2025\] HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)

<!-- RELATED:END -->
