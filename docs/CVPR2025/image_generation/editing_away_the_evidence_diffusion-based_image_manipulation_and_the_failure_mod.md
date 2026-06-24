---
title: >-
  [Paper Note] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking
description: >-
  [CVPR 2025][Image Generation][Diffusion-based Image Editing] A unified theoretical and empirical analysis of how diffusion-based image editing "unintentionally" destroys robust invisible watermarks: forward noising decays the watermark SNR exponentially, and the manifold contraction effect of reverse denoising eliminates the watermark signal as an "unnatural residual." Even state-of-the-art watermarks like VINE drop to near random guessing (~60% bit accuracy) under strong edi…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion-based Image Editing"
  - "Robust Watermarking"
  - "Watermark Removal"
  - "Information-theoretic Analysis"
  - "Content Provenance"
date: 2026-05-08
content_hash: 891f880affcc51fb
---

# Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking

**Conference**: CVPR 2025  
**arXiv**: [2603.12949](https://arxiv.org/abs/2603.12949)  
**Code**: None  
**Area**: Image Generation / Digital Watermarking  
**Keywords**: Diffusion-based Image Editing, Robust Watermarking, Watermark Removal, Information-theoretic Analysis, Content Provenance

## TL;DR

A unified theoretical and empirical analysis of how diffusion-based image editing "unintentionally" destroys robust invisible watermarks: forward noising decays the watermark SNR exponentially, and the manifold contraction effect of reverse denoising eliminates the watermark signal as an "unnatural residual." Even state-of-the-art watermarks like VINE drop to near random guessing (~60% bit accuracy) under strong editing ($t^*=0.8$).

## Background & Motivation

**Background**: Deep learning-based watermarking systems (e.g., StegaStamp, TrustMark, VINE) maintain high robustness (>95% bit accuracy) against traditional post-processing such as JPEG compression, scaling, and cropping through end-to-end training and differentiable noise layers.

**Limitations of Prior Work**: Diffusion-based image editing (e.g., InstructPix2Pix, DragDiffusion, TF-ICON) introduces an entirely new category of transformations, which injects heavy noise first and then reconstructs via generative priors. This is fundamentally different from traditional post-processing, and watermarking systems are not trained against it.

**Key Challenge**: Watermarks are essentially "low-amplitude structured perturbations," whereas diffusion denoisers are trained to remove all "unnatural residuals"—and watermarks happen to fit this exact description. Editors are not deliberately attacking the watermark, but the editing process itself inherently destroys it.

**Goal**: Under what conditions does diffusion editing render watermarks unrecoverable? What theoretical principles explain this collapse?

**Key Insight**: Model diffusion editing as a Markov kernel (forward noising + conditional denoising), derive the theoretical bounds for watermark SNR decay and mutual information decay, and establish Fano-type unrecoverability conditions.

**Core Idea**: Robustness to traditional post-processing $\neq$ robustness to generative transformations—the manifold contraction effect of diffusion editing systematically eliminates watermark signals.

## Method

### Overall Architecture

Three components: (1) modeling diffusion-based editing of watermarked images as a Markov kernel $K_\mathcal{T}(\tilde{x} | x_w, y)$, (2) deriving theoretical bounds for SNR decay and mutual information decay, and (3) designing a standardized evaluation protocol DEW-ST spanning 7 diffusion editors $\times$ 3 watermarking systems.

### Key Designs

1. **Watermark Signal Model**:

    - $x_w = x + \gamma \mathbf{s}(\mathbf{m}, \mathbf{k}, x)$: Watermark is a low-amplitude additive perturbation.
    - After forward noising: $x_t = \sqrt{\bar\alpha_t} x_w + \sqrt{1-\bar\alpha_t} \epsilon$
    - Watermark component SNR = $\gamma^2 \bar\alpha_t \|\mathbf{s}\|^2 / (1-\bar\alpha_t)$ $\rightarrow$ decays exponentially with $t$.

2. **Mutual Information Decay Bound**:

    - Derive the decay of $I(\mathbf{m}; \tilde{x})$ along the diffusion trajectory, connecting it to Fano's inequality $\rightarrow$ yielding a lower bound on the bit error rate.
    - Key conclusion: When the editing strength $t^*$ exceeds a critical threshold, watermark unrecoverability becomes an information-theoretic certainty.

3. **DEW-ST Evaluation Protocol**:

    - 7 Editors: InstructPix2Pix, UltraEdit, DragDiffusion, DragFlow, InstantDrag, TF-ICON, SHINE
    - 3 Watermarks: StegaStamp (Physically robust), TrustMark (Multi-resolution), VINE (Diffusion-aware training)
    - Various editing strengths $t^* \in \{0.2, 0.4, 0.6, 0.8\}$

### Frequency-Domain Analysis

Analysis of the retention rate $\rho_\Omega$ of watermark signals in different frequency bands: diffusion editing imposes the strongest suppression on high- and mid-frequency watermark signals, while low-frequency signals are relatively well retained.

## Key Experimental Results

### Main Results

| Transformation | Strength | StegaStamp | TrustMark | VINE | PSNR(dB) |
|---|---|---|---|---|---|
| None | — | 99.4% | 99.7% | 99.8% | 41.2 |
| JPEG Q50 | — | 96.1% | 98.2% | 98.9% | 33.5 |
| InstructPix2Pix | $t^*$=0.4 | 71.5% | 76.1% | 85.4% | 29.8 |
| InstructPix2Pix | $t^*$=0.8 | **53.2%** | **55.0%** | **60.7%** | 25.1 |
| DragDiffusion | medium | 63.4% | 67.9% | 78.6% | 28.7 |
| TF-ICON Synthesis | — | 58.9% | 63.2% | 74.8% | 28.1 |

(Note: The authors state that the experimental data are "hypothetical but realistic", reflecting literature trends)

### By Edit Type

| Edit Type | StegaStamp | TrustMark | VINE |
|---|---|---|---|
| Local Edit | ~75% | ~80% | ~88% |
| Global Edit | ~55% | ~58% | ~63% |

### Key Findings

- **Huge Gap Between Diffusion Editing and Traditional Post-processing**: Under JPEG Q50, StegaStamp maintains 96.1% bit accuracy, but drops to 71.5% under InstructPix2Pix with $t^*=0.4$. At $t^*=0.8$, it falls to near random guessing.
- **VINE is the most robust but still insufficient**: VINE utilizes diffusion-aware training, keeping 85%+ accuracy under mild editing, but still drops to 60% under strong editing.
- **Composition/Insertion operations are particularly lethal**: Even if TF-ICON and SHINE maintain global realism, the watermarks collapse (55-74%).
- **Local editing can also destroy global watermarks**: The denoising coupling in the diffusion latent space affects pixels outside the edited region.
- **Frequency-Domain Analysis**: High-frequency watermark signals are strongly suppressed while low-frequency signals are relatively preserved—yet most watermarks are encoded in the mid-to-high frequencies.

## Highlights & Insights

- **The perspective of "unintentional removal" is highly important**: This is not an adversarial attack—users are merely editing images normally. However, the editing process itself systematically destroys the watermark. This poses a fundamental challenge to the reliability of content provenance infrastructures.
- **Information-theoretic analysis goes straight to the core**: Instead of stating "this specific watermarking method performs poorly," the paper derives that "under this editing strength, no watermarking method can achieve reliable recovery." This is an impossibility result.
- **Constructive suggestions for watermark design**: (a) Diffusion-native fingerprints (e.g., Tree-Ring, embedded in initial noise) are more robust than post-processing watermarks; (b) Optimize semantic invariance rather than pixel-level robustness.

## Limitations & Future Work

- **The experimental data is "hypothetical but realistic"**: The authors admit that the data in the tables are not from real experiments but rather simulated values based on literature trends. This requires validation with real-world experiments.
- **Diffusion-native watermarks like Tree-Ring were not tested**: Theoretical predictions suggest they should be more robust, which warrants actual empirical comparison.
- **Adversarial fine-tuning of watermarking systems was not considered**: If the watermarking system incorporates diffusion editing as a noise layer during training, robustness could potentially improve.
- **Privacy tension**: There is a fundamental conflict between strong watermarking, freedom of editing, and privacy, which is not explored deeply in this paper.

## Related Work & Insights

- **vs VINE (W-Bench)**: VINE already identified the threat of diffusion editing to watermarks and incorporated diffusion-aware training; this paper advances this by providing theoretical analysis and a systematic evaluation across multiple editors.
- **vs ForensicZip**: ForensicZip deals with compression of forensic tokens, whereas this paper addresses watermarking vs. editing. However, both share a key insight: the denoising process of diffusion models tends to remove "off-manifold" (unnatural) signals.
- **Insights**: Future watermark designs should consider the "generative manifold" instead of the "pixel space"—watermark signals must be compatible with the data manifold to survive diffusion-based editing.

## Rating

- Novelty: ⭐⭐⭐⭐ The systematic analysis is comprehensive, and the information-theoretic bound holds theoretical value; however, the observation that "diffusion destroys watermarks" is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ Covers 7 editors and 3 watermarking systems, but the experimental data are simulated rather than real.
- Writing Quality: ⭐⭐⭐⭐⭐ The three-stage structure (theoretical derivation + experiments + design recommendations) is exceptionally clear.
- Value: ⭐⭐⭐⭐ Holds important cautionary and guiding significance for the fields of content provenance and watermark design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](../../ECCV2024/image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](../../NeurIPS2025/image_generation/gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges](../../NeurIPS2025/image_generation/finegrain_evaluating_failure_modes_of_text-to-image_models_with_vision_language_.md)
- [\[CVPR 2025\] EmoEdit: Evoking Emotions through Image Manipulation](emoedit_evoking_emotions_through_image_manipulation.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)

</div>

<!-- RELATED:END -->
