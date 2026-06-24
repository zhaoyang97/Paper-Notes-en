---
title: >-
  [Paper Note] Augmenting Perceptual Super-Resolution via Image Quality Predictors
description: >-
  [CVPR 2025][Image Restoration][Super-Resolution] No-reference image quality assessment (NR-IQA) models are leveraged to replace human annotations. By improving perceptual super-resolution quality through weighted sampling and direct optimization, the proposed method outperforms state-of-the-art methods that rely on human feedback, without requiring any human-labeled data.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Super-Resolution"
  - "No-Reference Image Quality Assessment"
  - "Perceptual Quality"
  - "NR-IQA"
  - "Perception-Distortion Trade-off"
date: 2026-05-08
content_hash: 736a6385eea8e3c4
---

# Augmenting Perceptual Super-Resolution via Image Quality Predictors

**Conference**: CVPR 2025  
**arXiv**: [2504.18524](https://arxiv.org/abs/2504.18524)  
**Code**: None  
**Area**: Image Restoration / Super-Resolution  
**Keywords**: Super-Resolution, No-Reference Image Quality Assessment, Perceptual Quality, NR-IQA, Perception-Distortion Trade-off

## TL;DR

No-reference image quality assessment (NR-IQA) models are leveraged to replace human annotations. By improving perceptual super-resolution quality through weighted sampling and direct optimization, the proposed method outperforms state-of-the-art methods that rely on human feedback, without requiring any human-labeled data.

## Background & Motivation

Single image super-resolution (SISR) is a classic ill-posed inverse problem, where a single low-resolution input corresponds to multiple plausible high-resolution solutions. Models trained with traditional pixel-level losses ($L_1$/$L_2$) tend to output the mean of the target distribution—resulting in blurry images that exhibit high PSNR but poor perceptual quality. To address this, the community has introduced perceptual loss and GANs to enhance perceptual quality; however, these methods are prone to generating high-frequency artifacts.

Prior work, HGGT, proposed generating multiple augmented ground truths (GTs), which human annotators then score to filter "positive" GTs for training, thereby achieving significant perceptual improvements. However, human annotation is: (1) coarse-grained (limited to only three categories: positive/negative/similar), (2) non-differentiable, preventing direct gradient optimization, and (3) expensive and difficult to scale.

**Core Motivation**: Can existing NR-IQA models be utilized to replace human annotators? NR-IQA models possess three key advantages over human annotators: fine-grained continuous scoring, differentiability, and the capacity for online dynamic evaluation.

## Method

### Overall Architecture

The proposed method comprises two complementary modules: (1) NR-IQA-based weighted sampling, which modifies the multi-GT selection strategy during training; and (2) NR-IQA-based direct optimization, which incorporates the quality score as a differentiable loss term. The combination of both yields the AMO+FT scheme.

### Key Designs

#### 1. Analysis and Selection of NR-IQA Metrics

The authors systematically evaluated over 20 NR-IQA metrics on two human preference datasets, SBS180K and HGGT. A two-stage screening process was conducted:

- **Phase I**: Evaluated 42 metric variants on 1,212 image pairs, selecting the Top-7 (PaQ-2-PiQ, NIMA, MUSIQ, LIQE, ARNIQA, Q-Align, and TOPIQ-NR).
- **Phase II**: Validated on the complete SBS180K dataset, where MUSIQ stood out with an 82.73% test accuracy.
- **Complementary Analysis**: On samples where MUSIQ failed, NIMA and Q-Align performed the best, and are thus utilized as complementary metrics for evaluation.

Ultimately, MUSIQ is selected as the core IQA model for both sampling and optimization.

#### 2. Reweighted Sampling Strategy

Base formula: $I \sim \mathcal{P}[S_I \mid \text{SoftMax}_\tau(Q(S_I))]$

Three variants:
- **SMA (Softmax-All)**: Performs weighted sampling across all GTs (including original and augmented) based on IQA scores, without requiring human labels.
- **SMP (Softmax-Positives)**: Performs weighted sampling only over the human-annotated positive GTs, thereby utilizing human-labeled data.
- **AMO (Argmax-Online)**: Extracts a patch from each GT first, then runs IQA at the patch level to select the best one, enabling finer-grained online decision-making.

The key innovation of AMO lies in shifting the quality assessment from the "image level" down to the "patch level," enabling it to identify quality discrepancies that human annotators cannot distinguish (e.g., as shown in Figure 2, for two GTs both labeled as "positive" by humans, MUSIQ yields scores of 36.13 vs. 54.19).

#### 3. Direct Optimization

The NR-IQA model $Q$ is incorporated into the objective function: $\widetilde{\mathcal{L}}(\phi|\hat{I},I) = \mathcal{L}(\phi|\hat{I},I) - \lambda_Q Q(\hat{I})$

**Key Challenge**: Directly optimizing a neural network-based IQA model can lead to adversarial exploitation, where gradient descent deceives $Q$ into giving high scores while actually introducing structural artifacts.

**Solution**: Regularization is achieved via LoRA (Low-Rank Adaptation). The main network parameters $\theta$ are frozen, and only the LoRA parameters $\phi$ are trained, thereby limiting the model's modification capacity. This approach is inspired by human feedback-guided tuning in text-to-image generation.

### Loss & Training

Base loss (identical to HGGT): $\mathcal{L}(\theta|\hat{I},I) = \lambda_{\ell_1}\|I-\hat{I}\|_1 + \lambda_P d_P(\hat{I},I) + \lambda_A D(\hat{I})$

During the fine-tuning phase, the NR-IQA term is introduced, and the GAN loss is disabled by default (as the IQA itself plays a similar role). By adjusting the ratio of $\lambda_P$ to $\lambda_Q$, one can control the trade-off between mid-level perceptual metrics and high-level NR quality.

## Key Experimental Results

### Main Results

| Model | Without Human Labels | PSNR↑ | LPIPS-ST↓ | MUSIQ↑ | NIMA↑ | Q-Align↑ | TOPIQ↑ |
|------|-----------|-------|-----------|--------|-------|----------|--------|
| SwinIR-UPos (HGGT SOTA) | ✗ | 22.30 | 0.129 | 66.39 | 5.16 | 3.56 | 0.62 |
| SwinIR-AMO | ✓ | 22.08 | 0.124 | 68.08 | 5.21 | 3.67 | 0.66 |
| SwinIR-AMO+FT | ✓ | 21.77 | 0.121 | **70.81** | **5.29** | **3.75** | **0.70** |
| Gold Standard | - | - | - | 69.64 | 5.28 | 3.78 | 0.69 |
| RESRGAN-UPos | ✗ | 21.54 | 0.192 | 65.93 | 5.25 | 3.47 | 0.63 |
| RESRGAN-AMO+FT | ✓ | 21.02 | 0.169 | **71.67** | **5.35** | **3.68** | **0.71** |

AMO+FT outperforms UPos (the HGGT SOTA) across all NR-IQA metrics without requiring any human labels. On SwinIR, it even surpasses the NR upper bound of the Gold Standard.

### Ablation Study

| Experiment | Conclusion |
|------|------|
| SMA vs SMP vs AMO | AMO achieves the best consistency; patch-level online evaluation is superior to image-level sampling. |
| FT vs FT_HP vs FT_IG | Increasing the perceptual loss weight (FT_HP) can recover mid-level metrics but sacrifices NR quality; incorporating GAN (FT_IG) yields no noticeable benefits. |
| MUSIQ vs PaQ-2-PiQ as Optimization Target | PaQ-2-PiQ leads to degradation across all NR metrics, validating MUSIQ as the optimal choice. |
| UPos+FT (with human data) vs AMO+FT (without human data) | AMO+FT is superior on SwinIR and achieves comparable performance on RealESRGAN. |

### Key Findings

1. **NR-IQA Sampling Can Outperform Human Annotation**: AMO uses no human data, yet outperforms UPos in perceptual metrics.
2. **Existence of a Three-Tier Perception-Distortion Trade-off**: Pixel-level (PSNR) $\rightarrow$ mid-level perceptual (LPIPS) $\rightarrow$ high-level NR-IQA. FT allows flexible tuning along this spectrum.
3. **LPIPS-ST is More "Perceptual" than LPIPS**: The behavior of LPIPS-ST aligned more closely with NR-IQA metrics, suggesting that shift invariance is crucial for perceptual evaluation.
4. **Discriminators Are Not Good IQAs**: Simply scaling up GAN loss cannot substitute for NR-IQA optimization.

## Highlights & Insights

- **Complete Pipeline for Replacing Human Annotation with Automation**: Outlines a systematic approach spanning metric selection $\rightarrow$ sampling strategy $\rightarrow$ direct optimization.
- **AMO's Patch-Level Online Evaluation** is an elegant design: It harnesses the unique advantage of NR-IQA being runnable online.
- **LoRA Regularization Tackles Adversarial Exploitation**: Simply and effectively addresses the core challenge of directly optimizing IQA.
- Identifies LPIPS-ST as a more perceptually meaningful mid-level metric.

## Limitations & Future Work

- Currently restricted to a single IQA model (MUSIQ); combining multiple complementary IQAs (e.g., NIMA, Q-Align as analyzed by the authors) could yield further improvements.
- Biases inherent in the IQA models might limit potential gains—fine-tuning IQA models explicitly for SR tasks could produce better results.
- The applicability to diffusion-based models as SR backbones remains unexplored.
- Degradations in PSNR/SSIM must be carefully evaluated, as pixel fidelity remains critical for certain downstream applications.

## Related Work & Insights

- **HGGT** provides the multi-GT framework and human-annotation baseline.
- **The success of RLHF in text-to-image generation** inspired the utilization of NR-IQA for SR fine-tuning.
- **LoRA**, originally designed for LLM adaptation, is ingeniously utilized here as a regularization mechanism for optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Upgrades NR-IQA from an evaluation tool to a training signal. Both the patch-level online evaluation in AMO and LoRA regularization present novel ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Systematic analysis of IQA metrics (42 variants) + two architectures + comprehensive ablation studies + user studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear, progressing systematically from analysis to methodology and experiments.
- **Value**: ⭐⭐⭐⭐ — Establishes a scalable, human-annotation-free training paradigm for perceptual super-resolution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/image_restoration/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[CVPR 2025\] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution](pidsr_complementary_polarized_image_demosaicing_and_super-resolution.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[AAAI 2026\] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](../../AAAI2026/image_restoration/temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)
- [\[CVPR 2026\] Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective](../../CVPR2026/image_restoration/rethinking_knowledge_transfer_in_image_quality_assessment_a_perceptual_preferenc.md)

</div>

<!-- RELATED:END -->
