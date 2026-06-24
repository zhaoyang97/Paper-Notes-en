---
title: >-
  [Paper Note] Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts
description: >-
  [CVPR 2025][Image Generation][Diffusion Model Artifacts] This work identifies a three-stage (Profiling-Mutation-Refinement) process in diffusion generation and the "score trap" mechanism responsible for artifact formation. It proposes ASCED, which monitors anomalous score dynamics to detect and correct artifacts in real-time without training, matching or exceeding supervised methods.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Model Artifacts"
  - "Score Dynamics"
  - "Unsupervised Detection"
  - "Online Correction"
  - "Generation Quality"
date: 2026-05-08
content_hash: 750acd4e550df829
---

# Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts

**Conference**: CVPR 2025  
**arXiv**: [2503.16218](https://arxiv.org/abs/2503.16218)  
**Code**: [Project Page](https://YuCao16.github.io/ASCED)  
**Area**: Image Generation / Others  
**Keywords**: Diffusion Model Artifacts, Score Dynamics, Unsupervised Detection, Online Correction, Generation Quality

## TL;DR

This work identifies a three-stage (Profiling-Mutation-Refinement) process in diffusion generation and the "score trap" mechanism responsible for artifact formation. It proposes ASCED, which monitors anomalous score dynamics to detect and correct artifacts in real-time without training, matching or exceeding supervised methods.

## Background & Motivation

### Background

**Background**: Even when trained on large-scale datasets, diffusion models still generate images with visual artifacts (anomalous local textures/structures).

### Key Challenge

**Key Challenge**: Existing methods primarily rely on supervised classifiers (trained on annotated datasets or leveraging large multimodal models) without explaining why artifacts occur.

### Limitations of Prior Work

**Limitations of Prior Work**: Uncertainty-based methods only analyze the spatial variance $Var(x_0)$ of the final output, ignoring the crucial temporal dynamics during the generation process.

### Proposed Approach

**Proposed Approach**: Post-processing correction methods (e.g., noise addition and denoising post-generation) operate on already completed generation results, which is inefficient and of limited effectiveness.

### Supplementary Notes

**Supplementary Notes**: There is a lack of fundamental understanding regarding the mechanism of artifact formation—why do these regions become artifacts, and when are they formed?

## Method

### Overall Architecture

ASCED consists of detection and correction steps integrated into the standard diffusion inference process. Starting from the detection onset step $T_d$, score values at each step are recorded in a score bank. At the correction step $T_c$, temporal dynamics in the score bank are analyzed to identify anomalous regions $\Omega^a$. Trajectory-aware targeted perturbations are applied to the artifact regions to perform real-time correction without interrupting the inference pipeline.

### Key Designs

**1. Discovery of the Three Stages in Diffusion Generation (Profiling-Mutation-Refinement)**
- **Function**: Reveals the intrinsic mechanism of the diffusion generation process, providing a theoretical foundation for artifact detection.
- **Core Idea**: It is discovered that diffusion generation essentially undergoes three stages: (1) Profiling—recovering global mean templates and foundational semantic layouts; (2) Mutation—introducing local pixel-level changes to establish local structures, where anomalous score dynamics emerge in artifact regions during this stage; (3) Refinement—integrating local changes into contextually coherent visual details.
- **Design Motivation**: Artifacts form during the Mutation stage when certain regions undergo drastic score changes and become "locked" within "score traps," making them irrecoverable during the Refinement stage. Analyzing only the final output fails to capture this temporal process.

**2. Anomalous Score Dynamics Detection**
- **Function**: Pinpoints potential artifact regions in real-time by monitoring the temporal evolution of scores.
- **Core Idea**: Score dynamics are defined as the difference between scores of adjacent steps $\Delta s_\theta(x_t^{i,j}, t)$. A score bank $\mathcal{S}$ is maintained to record historical score values. A temporal weight function $w(t) = \frac{1-\bar{\alpha_t}}{\sqrt{\bar{\alpha_t}}}$ is used to compensate for the natural decay of score magnitude. When the weighted score change exceeds an adaptive threshold $\tau = \max\{\text{MAD}(\Delta(\cdot)), \text{mean}(\mathcal{S})\}$, the region is flagged as an artifact.
- **Design Motivation**: Artifact regions exhibit a characteristic "rapid acceleration followed by sudden deceleration" pattern on the score acceleration curve, whereas normal regions maintain stable evolution. The temporal weight $w(t)$ is supported by theoretical analysis.

**3. Trajectory-Aware Targeted Correction (TTC)**
- **Function**: Corrects artifact regions without interrupting the inference process.
- **Core Idea**: Controlled perturbations are applied exclusively to the detected artifact regions $\Omega^a$: $\hat{x}_{T_c} = x_{T_c} \cdot \mathbb{1}_{\bar{\Omega}^a} + (\sqrt{\bar{\alpha}_{T_c}} x_0' + \sqrt{1-\bar{\alpha}_{T_c}} \epsilon) \cdot \gamma \xi \cdot \mathbb{1}_{\Omega^a}$, where $x_0'$ is the clean image predicted from the current step. Non-artifact regions remain completely untouched.
- **Design Motivation**: Although state replacement and score clipping can fix artifacts, they degrade generation diversity. TTC breaks the rigid pattern of score traps by injecting stochastic perturbations, allowing these regions to re-couple and evolve with surrounding areas while maintaining the original trajectories of non-artifact regions.

### Loss & Training

ASCED is entirely unsupervised and requires no training. Detection is based on the statistical analysis of score dynamics (MAD threshold), and correction is based on control-theoretic perturbation injection.

## Key Experimental Results

### Main Results

| Method | Type | FID↓ | Artifact Rate↓ |
|------|------|------|----------------|
| BayesDiff | UnS | — | Higher |
| SARGD | Sup | — | Moderate |
| State Replacement | UnS | — | Low but poor diversity |
| Score Clipping | UnS | — | Low but poor diversity |
| **ASCED (TTC)** | **UnS** | **Lowest** | **Lowest** |

*ASCED matches or outperforms the supervised method (SARGD) across five datasets while maintaining generation diversity.*

### Ablation Study

| Correction Strategy | FID↓ | Diversity | Artifact Rate↓ |
|----------|------|--------|---------|
| No Correction | baseline | High | High |
| State Replacement | Improved | Significantly Reduced | Low |
| Score Clipping | Slightly Improved | Reduced | Moderate |
| **TTC** | **Optimal** | **Maintained** | **Lowest** |

### Key Findings

- Diffusion models cannot autonomously recognize artifacts during the generation process—confusing artifacts with normal Mutations.
- Denoising-after-noise-addition in SDEdit can repair artifact regions, indirectly demonstrating that artifacts are indistinguishable from normal states in the feature space after noise injection.
- The theoretical derivation of the temporal weight $w(t)$ aligns with experimental observations.
- TTC effectively eliminates artifacts while maintaining diversity, outperforming simple state replacement and score clipping.

## Highlights & Insights

1. **Mechanism-Level Analysis**: Explains the formation mechanism of diffusion artifacts (score traps) for the first time from the perspective of score dynamics, moving beyond phenomenological detection.
2. **"Online" Intervention**: Embeds artifact correction directly into the generation process itself rather than post-processing, which is highly efficient and preserves diversity.
3. **Unsupervised Outperforms Supervised**: Achieves artifact elimination metrics matching or exceeding supervised methods without requiring any annotated data.

## Limitations & Future Work

- Detection requires storing the score bank $\mathcal{S}$, which increases memory overhead.
- The selection of $T_d$ and $T_c$ requires empirical tuning.
- Handling semantic hallucinations (e.g., extra limbs) is out of scope.
- Investigating the application of the three-stage analysis to controllable generation is a potential future direction.

## Related Work & Insights

- Compared with the space uncertainty analysis of BayesDiff, the temporal analysis of ASCED can locate artifacts more precisely.
- The concept of score traps can be generalized to understand failure modes in other generative models.
- The perturbation injection strategy of TTC serves as a valuable reference for other generative tasks requiring localized restoration.

## Rating

⭐⭐⭐⭐⭐ — Characterized by both profound theoretical insight (three-stage + score trap) and a practical unsupervised solution, this paper presents extremely high quality. It elegantly embeds artifact detection and correction into the inference process, outperforming supervised methods without requiring additional training. It has pioneering significance for understanding the internal mechanisms of diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2026\] C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis](../../CVPR2026/image_generation/c2fg_control_classifier-free_guidance_via_score_discrepancy_analysis.md)
- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[CVPR 2025\] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
