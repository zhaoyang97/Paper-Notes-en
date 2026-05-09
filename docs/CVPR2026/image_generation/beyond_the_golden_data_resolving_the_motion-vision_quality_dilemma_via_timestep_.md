---
title: >-
  [Paper Note] Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training
description: >-
  [CVPR 2026][Image Generation][Video Diffusion Models] This paper identifies a "Motion-Vision Quality Dilemma" where motion quality (MQ) and visual quality (VQ) are negatively correlated in video data. Through gradient analysis, it reveals that imbalanced data can produce equivalent learning signals at appropriate timesteps. The proposed TQD framework enables training on imbalanced data to surpass training on "golden data."
tags:
  - CVPR 2026
  - Image Generation
  - Video Diffusion Models
  - Data Quality Dilemma
  - Timestep Selective Training
  - Motion-Vision Quality Balance
  - Flow Matching
date: 2026-05-08
content_hash: feada004f288b567
---

# Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training

**Conference**: CVPR 2026  
**arXiv**: [2603.25527](https://arxiv.org/abs/2603.25527)  
**Code**: None  
**Area**: Image Generation/Video Generation  
**Keywords**: Video Diffusion Models, Data Quality Dilemma, Timestep Selective Training, Motion-Vision Quality Balance, Flow Matching

## TL;DR
This paper identifies a "Motion-Vision Quality Dilemma" where motion quality (MQ) and visual quality (VQ) are negatively correlated in video data. Through gradient analysis, it reveals that imbalanced data can produce equivalent learning signals at appropriate timesteps. The proposed TQD framework enables training on imbalanced data to surpass training on "golden data."

## Background & Motivation
**Background**: Video generation models (e.g., CogVideoX, Wan-T2V) rely on "golden data" that possesses both high visual quality (VQ) and high motion quality (MQ). However, such data is **statistically scarce**.

**Key Challenge**—Motion-Vision Quality Dilemma: Analysis on Koala36M reveals a **negative correlation** between MQ and VQ ($r=-0.2419$). High-VQ data tends to be static (low MQ), while high-MQ data tends to have artifacts (low VQ). Only 21.9% of data satisfies both high standards simultaneously.

**Limitations of Prior Work**: Existing practices use strict filtering to retain only golden data—discarding a vast amount of videos that excel in only one dimension, leading to severe data waste.

**Key Insight**: Shifting the perspective from "which data to keep" to "how to use imperfect data more effectively."

**Core Idea**: The denoising process of diffusion models is **hierarchical**—high-noise timesteps establish motion and composition, while low-noise timesteps refine detailed textures. Gradient analysis confirms: VQ-degraded data produces gradients close to golden data at high timesteps, and MQ-degraded data produces gradients close to golden data at low timesteps.

## Method

### Overall Architecture
Pre-compute MQ/VQ scores for each video (VideoAlign) → Normalization → Sample-level quality dropout → Timestep-level Beta distribution adaptive sampling → Training diffusion/flow matching models.

### Key Designs
1.  **Sample-level Weighting (Absolute Quality)**:
    - Retention probability $p_{sample} = \max(vq_{norm}, mq_{norm})$
    - **Design Motivation**: Data performing well in at least one dimension is valuable; "double-low" samples should be suppressed.
    - Effect: Naturally filters the low-quality tail.

2.  **Timestep-level Distribution Adjustment (Relative Quality)**:
    - Sample timesteps according to a quality-dependent Beta distribution: $p(t|x_0) \propto \text{Beta}(t; \mu \kappa, (1-\mu)\kappa)$
    - Central parameter $\mu = 0.5 + 0.5 \times (mq_{norm} - vq_{norm})$
        - High MQ / Low VQ → $\mu > 0.5$ → Biased towards large timesteps (motion learning stage).
        - Low MQ / High VQ → $\mu < 0.5$ → Biased towards small timesteps (detail refinement stage).
    - Concentration parameter $\kappa = \kappa_{base} + (\kappa_{max} - \kappa_{base}) \times |mq_{norm} - vq_{norm}|$
        - Greater quality imbalance → More concentrated distribution → Stronger timestep specialization.
    - **Why use Beta distribution**: Flexible shape (can be symmetric, skewed, or concentrated); degrades to baseline sampling when $mq=vq$.

3.  **Combined Quality Weighted Distribution**:
    $p(t) \propto \max(vq_{norm}, mq_{norm}) \cdot \text{Beta}(t; \mu\kappa, (1-\mu)\kappa)$
    Behavior of four types of data: HMLV → large timesteps, LMHV → small timesteps, HMHV → uniform coverage, LMLV → dropped out.

### Loss & Training
- Flow Matching objective (Wan-T2V): $\mathcal{L} = \mathbb{E}[\|v_\theta(x_t, t, c) - (x_1 - x_0)\|^2]$
- Diffusion objective (CogVideoX): Standard noise prediction.
- $\kappa_{base}$ aligns with original sampling strategies (uniform → 2, logit-normal → 4).

## Key Experimental Results

### Main Results (Wan-T2V 1.3B)

| Data Setup | Method | VBench Dynamic↑ | VideoAlign MQ↑ | VideoAlign VQ↑ |
| :--- | :--- | :--- | :--- | :--- |
| Set-A (All) | Baseline | 0.5312 | 2.1388 | 3.2537 |
| Set-A (All) | **TQD** | **0.6384** | **2.2557** | **3.3450** |
| Set-B (Imbalanced Only) | Baseline | 0.5224 | 2.0905 | 3.2338 |
| Set-B (Imbalanced Only) | **TQD** | 0.5447 | **2.1477** | **3.2679** |
| Set-C (Golden Only) | Baseline | 0.5268 | 2.1917 | 3.3378 |
| Set-C (Golden Only) | **TQD** | **0.6473** | **2.2200** | **3.3743** |

### Ablation Study

| Component | MQ | VQ | Description |
| :--- | :--- | :--- | :--- |
| Baseline | 2.1388 | 3.2537 | Baseline |
| + Adaptive Timestep | 2.2193 | 3.3044 | Core source of gain |
| + Quality Dropout | 2.1909 | 3.2921 | Auxiliary data filtering |
| + Both (TQD) | **2.2557** | **3.3450** | Synergistic gain |

### Key Findings
- **Imbalanced data + TQD can surpass conventional training on golden data**: Set-B with TQD MQ (2.1477) > Set-A Baseline MQ (2.1388).
- TQD still provides significant improvements on golden data (Set-C), indicating the universality of the method beyond imbalanced scenarios.
- Physical reasoning capabilities are improved (both SA and PC on VideoPhy2 increased).

## Highlights & Insights
- **Paradigm Shift**: From "data filtering" to "data routing," significantly expanding usable training data.
- Gradient analysis provides a solid theoretical foundation: imbalanced data aligns with golden data gradients at specific timesteps.
- Challenges the inherent assumption that "video generation must use golden data."
- The Beta distribution parameterization is elegant; balanced data naturally degrades to baseline sampling.

## Limitations & Future Work
- Requires pre-computation of MQ/VQ scores (VideoAlign), increasing data preparation costs.
- Gains are smaller under LoRA fine-tuning (CogVideoX); full-parameter training may benefit more.
- More fine-grained quality dimensions (e.g., audio quality, text alignment) were not explored.
- $\kappa_{max}$ requires hyperparameter tuning for different models.

## Related Work & Insights
- Consistent with but more specific than the idea of Ambient Diffusion (utilizing imperfect data).
- Timestep-aware training strategies can be generalized to image diffusion models (e.g., image restoration training for different degradation types).
- Provides direct guidance for industrial-level video generation data pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Original problem discovery (MV Dilemma) and solution (timestep routing) with deep insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two architectures, three data configurations, with detailed ablations and qualitative comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Smooth narrative from problem discovery → gradient analysis → method design → validation.
- Value: ⭐⭐⭐⭐⭐ Potential for a transformative impact on the training paradigm of video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](../../ICLR2026/image_generation/verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)

</div>

<!-- RELATED:END -->
