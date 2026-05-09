---
title: >-
  [Paper Note] GENMO: A GENeralist Model for Human MOtion
description: >-
  [ICCV 2025][Human Understanding][Human motion modeling] This paper proposes GENMO, the first generalist model that unifies human motion estimation (recovering motion from video/2D keypoints) and motion generation (synthesizing motion from text/music/keyframes) within a single framework. Through a dual-mode training paradigm (regression + diffusion), GENMO achieves both precise estimation and diverse generation in a single model.
tags:
  - ICCV 2025
  - Human Understanding
  - Human motion modeling
  - motion estimation
  - motion generation
  - diffusion models
  - multimodal conditioning
date: 2026-05-08
content_hash: 39eb07f741265eba
---

# GENMO: A GENeralist Model for Human MOtion

**Conference**: ICCV 2025
**arXiv**: [2505.01425](https://arxiv.org/abs/2505.01425)
**Code**: [Project Page](https://research.nvidia.com/labs/dair/genmo)
**Area**: Human Understanding
**Keywords**: Human motion modeling, motion estimation, motion generation, diffusion models, multimodal conditioning

## TL;DR

This paper proposes GENMO, the first generalist model that unifies human motion estimation (recovering motion from video/2D keypoints) and motion generation (synthesizing motion from text/music/keyframes) within a single framework. Through a dual-mode training paradigm (regression + diffusion), GENMO achieves both precise estimation and diverse generation in a single model.

## Background & Motivation

Human motion modeling is a long-standing research topic in computer vision and graphics, with broad applications in gaming, animation, and 3D content creation. Consider a practical creation scenario: a user wants to generate a motion sequence starting from a video clip, transitioning to text-described actions, synchronizing with music beats, and finally aligning with another video—all while maintaining fine-grained keyframe control. This demands a model capable of both faithfully reproducing observed motion and generating diverse plausible motion.

**Root Cause**: Motion estimation and motion generation have fundamentally different objectives:
- **Estimation** requires precise, deterministic output—given the same video, a unique motion sequence should be recovered.
- **Generation** requires diverse output—given the same text description, multiple plausible motions should be producible.

This tension has led the two tasks to be handled by separate models, limiting cross-task knowledge transfer.

**Key Insights**:
1. Generative priors can improve estimation—under challenging conditions such as occlusion, the motion distribution learned by generative models provides useful constraints.
2. Diverse video data can enhance generation—large-scale in-the-wild videos (with only 2D annotations) broaden the motion distribution available to generative models.
3. Diffusion models provide a natural framework for unification—estimation can be viewed as "maximum-likelihood generation."

## Method

### Overall Architecture

GENMO is built upon the diffusion model framework and reformulates motion estimation as constrained motion generation: given conditioning signals (video, 2D keypoints, text, music, 3D keyframes), the model generates motion sequences $x = \{x^i\}_{i=1}^N$ satisfying the given constraints. Each frame's motion representation is:
$$x^i = (\Gamma_{\text{gv}}^i, v_{\text{root}}^i, \theta^i, \beta^i, t_{\text{root}}^i, \pi^i, p^i)$$
corresponding to gravity-view orientation (6D), root velocity (3D), SMPL joint angles (24×6D), shape parameters (10D), root translation (3D), camera pose, and contact labels.

### Key Designs

1. **Dual-Mode Training Paradigm**:

    - **Function**: Enables a single model to achieve both estimation accuracy and generation diversity.
    - **Mechanism**:
        - **Estimation mode**: The model input is set to pure Gaussian noise $z \sim \mathcal{N}(\mathbf{0}, I)$ with the timestep set to the maximum value $T$, and the model directly regresses the clean motion:
       $$\mathcal{L}_{\text{est}} = \mathbb{E}_{z \sim \mathcal{N}(\mathbf{0}, I)} [\|x_0 - \mathcal{G}(z, T, \mathcal{C}, \mathcal{M})\|^2]$$
       This is equivalent to maximum likelihood estimation, forcing the model to predict the most probable motion in a single step from pure noise.
        - **Generation mode**: Standard DDPM training, progressively denoising from a noisy motion:
       $$\mathcal{L}_{\text{gen}} = \mathbb{E}_{t, x_t} [\|x_0 - \mathcal{G}(x_t, t, \mathcal{C}, \mathcal{M})\|^2]$$
    - **Design Motivation**: The authors observe that diffusion models under video conditioning exhibit high determinism—the first-step prediction is already very close to the final result—whereas text-conditioned predictions exhibit high variance. For estimation tasks, improving the quality of the "first-step prediction" (estimation mode) is therefore critical, without sacrificing diverse generation capability (generation mode).

2. **Multimodal Conditioning Architecture**:

    - **Function**: Supports arbitrary combinations of multimodal conditioning inputs including video, music, 2D keypoints, and text.
    - **Mechanism**:
        - **Frame-aligned conditions** (video, music, 2D skeleton): An Additive Fusion Block projects features from each modality via separate MLPs, sums them, and fuses the result with the noisy motion to produce a token sequence.
        - **Text conditions**: A novel **Multi-Text Injection Block** supports the injection of multiple text segments over different temporal windows:
       $$f_{\text{out}} = \sum_{k=1}^K \text{MaskedMHA}(f_{\text{in}}, c_{\text{text}}^k, \Omega_k)$$
       where $\Omega_k(i,j)$ is a binary mask restricting the $k$-th text segment to influence only motion frames within its corresponding temporal window.
        - The backbone network uses a RoPE-based Transformer, supporting variable-length sequences and sliding-window attention at inference time.
    - **Design Motivation**: Text and motion frames lack a frame-wise alignment relationship and cannot be naively concatenated (which would introduce positional bias). Multi-text injection elegantly addresses the need for temporally segmented text control via masked attention.

3. **Estimation-Guided 2D Training**:

    - **Function**: Leverages in-the-wild videos with only 2D annotations to enhance the diversity of the generative model.
    - **Mechanism**: 2D data is exploited in two steps:
      1. The estimation mode first generates pseudo 3D motion from 2D conditions: $\hat{x}_0 = \mathcal{G}(z, T, \mathcal{C})$.
      2. The pseudo motion is noised and used for generation-mode training, with the loss computed via 2D reprojection:
       $$\mathcal{L}_{\text{gen-2D}} = \mathbb{E} [\|x_{\text{2d}} - \Pi(\mathcal{G}(\hat{x}_t, t, \mathcal{C}))\|^2]$$
    - **Design Motivation**: 3D motion capture data is scarce and limited in diversity, whereas 2D annotations can be obtained at scale via detectors. Converting in-the-wild 2D video into training data through the estimation capability simultaneously enriches the generative distribution and avoids the noise inherent in 3D pseudo-labels.

### Loss & Training

- **Estimation mode**: $\mathcal{L}_{\text{est}} + \mathcal{L}_{\text{geo}}$ (including geometric regularization terms such as 3D joint/vertex constraints and contact constraints).
- **Generation mode**: $\mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{geo}}$ (3D data) or $\mathcal{L}_{\text{gen-2D}} + \mathcal{L}_{\text{geo}}$ (2D data).
- **Mode selection strategy**: Strong conditions (video/2D skeleton) employ both estimation and generation modes simultaneously; weak conditions (text/music) use generation mode only.
- Inference supports sliding-window attention (window of $W$ frames) for arbitrarily long sequence generation.

## Key Experimental Results

### Main Results

**Global Motion Estimation (EMDB-2 dataset)**:

| Method | WA-MPJPE100 | W-MPJPE100 | RTE | Foot Sliding |
|------|------------|------------|-----|-------------|
| WHAM (DPVO) | 135.6 | 354.8 | 6.0 | 4.4 |
| GVHMR (DPVO) | 111.0 | 276.5 | 2.0 | 3.5 |
| TRAM (DROID) | 76.4 | 222.4 | 1.4 | - |
| **GENMO (DROID)** | **74.3** | **202.1** | **1.2** | 8.8 |

**Music-Driven Dance Generation (AIST++)**:

| Method | FIDk↓ | FIDm↓ | PFC↓ | BAS↑ |
|------|-------|-------|------|------|
| Bailando | 28.16 | 9.62 | 1.754 | 0.2332 |
| EDGE | 42.16 | 22.12 | 1.5363 | 0.2334 |
| **GENMO (music only)** | **16.10** | **13.91** | 0.7340 | 0.2282 |
| **GENMO (generalist)** | 40.91 | 18.51 | **0.3702** | **0.2708** |

### Ablation Study

**Contribution of Dual-Mode Training (motion estimation, RICH dataset)**:

| Configuration | WA-MPJPE100 | W-MPJPE100 | Note |
|------|------------|------------|------|
| Diffusion-only | 88.9 | 143.9 | Standard diffusion training only |
| Regression-only | 87.0 | 141.0 | Regression training only |
| **Dual-mode** | **75.3** | **118.6** | Best performance |

**Effect of 2D Training Data (text-to-motion generation, Motion-X)**:

| Configuration | FID↓ | R@3↑ | MM Dist↓ | Note |
|------|------|------|---------|------|
| MDM baseline | 2.389 | 0.313 | 6.745 | Baseline method |
| w/o 2D Training | 0.515 | 0.401 | 5.210 | Without 2D data |
| **w/ 2D Training** | **0.207** | **0.472** | **4.801** | With 2D data |

### Key Findings

1. **The unified model outperforms task-specific models**: GENMO surpasses the dedicated estimation method TRAM on motion estimation (W-MPJPE 202.1 vs. 222.4), benefiting from the generative prior as a constraint on motion plausibility.
2. **Both modes in dual-mode training are indispensable**: Neither diffusion-only nor regression-only training matches the dual-mode configuration. Diffusion provides generation diversity, while regression ensures estimation precision.
3. **2D data training comprehensively improves generation quality**: On Motion-X, FID drops from 0.515 to 0.207 and R@3 improves from 0.401 to 0.472, validating the effectiveness of extracting training signal from in-the-wild videos.
4. The generalist model achieves a higher FIDk than the task-specific model on music-driven dance (40.91 vs. 16.10), but exhibits superior diversity, physical plausibility, and music beat alignment (PFC 0.37 vs. 0.73, BAS 0.27 vs. 0.23), reflecting the mutual benefits of multi-task training.
5. Motion interpolation experiments further validate the synergistic benefits of joint estimation and generation training.

## Highlights & Insights

- **Paradigm-level innovation**: This work is the first to demonstrate that motion estimation and generation can be unified within a single diffusion framework with mutual performance benefits.
- **The theoretical basis of dual-mode training is compelling**: The estimation mode corresponds to denoising at the maximum timestep in a diffusion model, making it fully compatible with the generation mode.
- **The multi-text injection design** addresses the practical challenge of aligning text with motion frames, enabling temporally segmented control such as "run for 5 seconds, then dance for 10 seconds."
- **Variable-length sequence support** is achieved via RoPE combined with sliding-window attention, producing naturally coherent long sequences without post-hoc stitching.
- Being an NVIDIA product, the engineering implementation and data scale are both substantial.

## Limitations & Future Work

- The use of SMPL parametric representation leads to inferior performance on HumanML3D metrics compared to methods using task-specific representations (representation mismatch issue).
- The generalist model generally underperforms task-specific models on individual tasks (e.g., FIDk on music-driven dance), though its overall performance is superior.
- Training involves multiple datasets and multiple training modes, resulting in considerable implementation complexity.
- The model supports only single-person motion and does not handle multi-person interaction.
- Only the SMPL skeleton representation is supported; facial expressions and hand details are excluded.

## Related Work & Insights

- The gravity-view coordinate system proposed in GVHMR (NeurIPS 2024) is directly adopted by GENMO, demonstrating the general utility of this coordinate formulation.
- MDM (ICLR 2023) provides the foundational diffusion-based motion generation paradigm; GENMO extends it with the estimation mode and a multimodal architecture.
- Unlike large-model approaches such as MotionGPT that focus on language understanding, GENMO prioritizes precise geometric constraints and physical plausibility.
- The dual-mode training concept is generalizable to other tasks requiring unified estimation and generation (e.g., 3D scene understanding, audio).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Unifying estimation and generation represents a significant paradigm contribution; both dual-mode training and estimation-guided 2D training are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple tasks spanning estimation (global/local) and generation (text/music/interpolation) with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Content is rich but lengthy; some architectural details could be presented more concisely.
- **Value**: ⭐⭐⭐⭐⭐ Substantially advances the field of human motion modeling; the unified framework with bidirectional mutual benefits constitutes a compelling research direction.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[ICCV 2025\] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds](egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)

<!-- RELATED:END -->
