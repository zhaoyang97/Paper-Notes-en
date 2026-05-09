---
title: >-
  [Paper Note] Targeted Data Protection for Diffusion Model by Matching Training Trajectory
description: >-
  [AAAI 2026][Image Generation][Diffusion model data protection] TAFAP achieves, for the first time, effective Targeted Data Protection (TDP) for diffusion models by generating adversarial perturbations via training trajectory matching, redirecting unauthorized fine-tuning outputs toward a user-specified target concept while maintaining high image quality.
tags:
  - AAAI 2026
  - Image Generation
  - Diffusion model data protection
  - training trajectory matching
  - adversarial perturbation
  - dataset distillation
  - targeted protection
date: 2026-05-08
content_hash: 9c1ad872448863b7
---

# Targeted Data Protection for Diffusion Model by Matching Training Trajectory

**Conference**: AAAI 2026
**arXiv**: [2512.10433](https://arxiv.org/abs/2512.10433)
**Code**: N/A
**Area**: Image Generation / AI Security
**Keywords**: Diffusion model data protection, training trajectory matching, adversarial perturbation, dataset distillation, targeted protection

## TL;DR
TAFAP achieves, for the first time, effective Targeted Data Protection (TDP) for diffusion models by generating adversarial perturbations via training trajectory matching, redirecting unauthorized fine-tuning outputs toward a user-specified target concept while maintaining high image quality.

## Background & Motivation

**State of the Field**: Personalized fine-tuning of diffusion models allows anyone to replicate specific individuals or styles using only a small set of images. Existing protection methods (e.g., Anti-DreamBooth, GLAZE) passively degrade the quality of fine-tuned outputs through imperceptible perturbations.

**Limitations of Prior Work**: Passive quality degradation is uncontrollable, and attackers may still obtain usable results. Prior TDP attempts based on "snapshot matching" perform poorly — the protective effect is diluted as training continues.

**Root Cause**: Snapshot matching affects only a single instant during training, whereas fine-tuning is a continuous process.

**Paper Goals**: Achieve effective TDP by controlling the entire training trajectory.

**Starting Point**: Drawing inspiration from training trajectory matching techniques used in dataset distillation.

**Core Idea**: Replace snapshot matching with training trajectory matching to generate adversarial perturbations, ensuring that the fine-tuning trajectory globally aligns with the training trajectory of the target concept.

## Method

### Overall Architecture
Two stages: (1) fine-tune the diffusion model on target concept images to record the full training trajectory, then optimize adversarial perturbations on the source images so that the resulting fine-tuning trajectory matches the target trajectory at every step; (2) release the perturbed images, such that unauthorized fine-tuning will produce outputs resembling the target concept.

### Key Designs

1. **Training Trajectory Matching**:

    - Function: Ensures the protection effect remains active throughout the entire fine-tuning process.
    - Mechanism: Obtain the "target trajectory" $\{\theta_t^*\}$ from target concept images; optimize perturbation $\delta$ so that fine-tuning on $x+\delta$ matches the target trajectory at each step. This is realized by backpropagating through training steps following the MTT approach.
    - Design Motivation: Snapshot matching yields exponentially decaying protection, whereas trajectory matching provides persistent protection.

2. **Adversarial Perturbation Optimization**:

    - Function: Generate imperceptible perturbations capable of redirecting fine-tuning outputs.
    - Mechanism: Iteratively optimize under an $L_\infty$ constraint via PGD, computing the gradient of the trajectory matching loss at each step and projecting onto the feasible set.
    - Design Motivation: Perturbations must be imperceptible (PSNR > 30 dB) while remaining sufficiently effective.

3. **Dual Control over Identity and Visual Pattern**:

    - Function: Simultaneously control both identity and visual style outputs of the fine-tuned model.
    - Mechanism: The target concept encodes both identity and visual patterns; synchronous control is achieved by jointly matching training trajectories along both dimensions.
    - Design Motivation: Stronger controllability makes protection verifiable — generating the preset target concept serves as evidence of unauthorized data usage.

### Loss & Training
Trajectory matching loss $\mathcal{L} = \sum_t \|\theta_t(x+\delta) - \theta_t^*\|^2$ combined with a perceptual loss to enforce imperceptibility. PGD optimization under $L_\infty$ constraint of 8/255 or 16/255.

## Key Experimental Results

### Main Results

| Setting | Metric | TAFAP | Prev. SOTA TDP | Gain |
|---------|--------|-------|----------------|------|
| Identity redirection | Target ID matching | High | Extremely low | First effective TDP |
| Visual pattern redirection | Style matching | High | Uncontrollable | First achieved |
| Image quality | FID | Maintained | Degraded | Better quality |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| Trajectory matching | Best | Effective control throughout |
| Snapshot matching | Poor | Protection diluted during training |
| Varying trajectory length | Longer is better | More constraints = stronger protection |

### Key Findings
- First successful TDP method for diffusion models.
- Trajectory matching vs. snapshot matching represents a fundamental distinction.
- Perturbations are visually imperceptible (PSNR > 30 dB).

## Highlights & Insights
- **Inverse use of dataset distillation**: Distillation enables a small dataset to reproduce a large dataset's trajectory; TAFAP inversely uses perturbed data to redirect the trajectory — an elegant conceptual reversal.
- **First successful TDP**: Identifies the root cause (snapshot vs. trajectory) and provides a principled solution.
- **Verifiable protection**: Generation of the target concept itself constitutes evidence of unauthorized data usage.

## Limitations & Future Work
- Trajectory matching requires simulating the full fine-tuning process, incurring high computational cost.
- Robustness to alternative fine-tuning protocols (e.g., LoRA) remains to be validated.
- The trade-off between perturbation strength and protection efficacy warrants further investigation.

## Related Work & Insights
- **vs. Anti-DreamBooth**: Passive quality degradation vs. active redirection — a more advanced protection paradigm.
- **vs. GLAZE/Mist**: Passive defense vs. verifiable, active protection.
- **vs. Dataset Distillation (MTT)**: Technically inspired by MTT, but with opposite objectives — efficiency vs. security.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First successful TDP; innovative inverse application of trajectory matching.
- Experimental Thoroughness: ⭐⭐⭐⭐ Protection efficacy thoroughly validated.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology clearly presented.
- Value: ⭐⭐⭐⭐⭐ Significant implications for AI security and copyright protection.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[AAAI 2026\] RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation](retrysql_text-to-sql_training_with_retry_data_for_self-correcting_query_generati.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)

<!-- RELATED:END -->
