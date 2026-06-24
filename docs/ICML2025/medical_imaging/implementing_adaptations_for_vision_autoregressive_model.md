---
title: >-
  [Paper Note] Implementing Adaptations for Vision AutoRegressive Model
description: >-
  [ICML 2025][Medical Imaging][Vision AutoRegressive] This paper presents the first systematic implementation and evaluation of various adaptation methods (FFT/LoRA/LNTuning) and differential privacy (DP) adaptation for the Vision AutoRegressive (VAR) model. It finds that VAR significantly outperforms diffusion model adaptation (DiffFit) in non-DP scenarios with faster convergence and higher computational efficiency, but its DP adaptation performance remains poor…
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Vision AutoRegressive"
  - "VAR fine-tuning"
  - "LoRA"
  - "differential privacy"
  - "parameter-efficient fine-tuning"
  - "image generation"
  - "DiffFit"
date: 2026-05-08
content_hash: eb04a24acf1927e7
---

# Implementing Adaptations for Vision AutoRegressive Model

**Conference**: ICML 2025  
**arXiv**: [2507.11441](https://arxiv.org/abs/2507.11441)  
**Code**: [https://github.com/sprintml/finetuning_var_dp](https://github.com/sprintml/finetuning_var_dp)  
**Area**: Medical Imaging  
**Keywords**: Vision AutoRegressive, VAR fine-tuning, LoRA, differential privacy, parameter-efficient fine-tuning, image generation, DiffFit

## TL;DR

This paper presents the first systematic implementation and evaluation of various adaptation methods (FFT/LoRA/LNTuning) and differential privacy (DP) adaptation for the Vision AutoRegressive (VAR) model. It finds that VAR significantly outperforms diffusion model adaptation (DiffFit) in non-DP scenarios with faster convergence and higher computational efficiency, but its DP adaptation performance remains poor, revealing an important research gap in the field of privacy-preserving image generation.

## Background & Motivation

The Vision AutoRegressive (VAR) model was recently proposed as a strong alternative to Diffusion Models (DMs) in the field of image generation. VAR redefines "next-token prediction" as "next-scale prediction", progressively generating 2D token grids from low to high resolution, which is faster. However:

**Lack of Adaptation Methods**: While rich fine-tuning techniques (DiffFit, DreamBooth, Textual Inversion) exist for diffusion models, adaptation methods for VAR remain largely unexplored.

**Gap in Differentially Private Adaptation**: Privacy protection is required when fine-tuning data is sensitive (e.g., medical images). While DP adaptation has been widely studied for DMs, no such solutions exist for VAR.

**Implementation Challenges**: The attention operators and forward functions in the original VAR codebase require patching to incorporate LoRA and DP-SGD.

*Core Motivation*: Bridging the gap between vision autoregressive models and diffusion models in model adaptation and privacy protection.

## Method

### Overall Architecture

Based on a pre-trained VAR model (ImageNet-1K class-conditional, 256×256), three adaptation strategies are systematically implemented:

1. **Full Fine-Tuning (FFT)**: Updates all parameters of the model.
2. **LoRA**: Inserts low-rank matrices $\Delta W = BA$ into the self-attention Q/K/V and projection layers, with $r=16$ and $\alpha=2r$.
3. **LayerNorm Tuning (LNTuning)**: Only updates the parameters of the Adaptive LayerNorm module.

### Key Designs

#### 1. VAR Adaptation

**LoRA Implementation**:
- Target modules: the query, key, value matrices and projection layers of self-attention.
- Simultaneously fine-tunes the Adaptive LayerNorm module.
- Low-rank decomposition: $\Delta W \in \mathbb{R}^{d \times k}$, where $\Delta W = BA$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$.

**LNTuning Implementation**:
- Only updates the newly introduced trainable parameters in the Adaptive LayerNorm module.
- All other weights are frozen.

*Implementation Challenges*: The original attention operator in VAR requires patching to integrate LoRA adapters (see Appendix F for details).

#### 2. Differentially Private Adaptation

Using the DP-SGD algorithm:

$\theta_{i+1} = \theta_i - \eta \left(\frac{1}{L}\sum_{k=1}^{L} \text{clip}(g(x_k)) + \mathcal{N}(0, \sigma^2 C^2 I)\right)$

where $\text{clip}(g(x_k)) = g(x_k) / \max(1, \|g(x_k)\|_2 / C)$

**Augmentation Multiplicity**: Generates $k$ augmented views for each sample and averages their gradients to improve the signal-to-noise ratio.

*Implementation Challenges*: Resolving issues with model-specific buffers and non-standard forward functions in the VAR codebase is required.

### Evaluation Metrics

- **FID** (Fréchet Inception Distance): Quantifies the generation quality.
- **PFLOPs**: Quantifies the computational cost.

## Key Experimental Results

### Main Results: VAR vs DiffFit (FID↓)

| Model | Method | Food-101 | CUB-200 | Oxford Flowers | Stanford Cars | Trainable Params |
|------|------|----------|---------|----------------|---------------|-----------|
| DiT-XL-2 | DiffFit | 6.96 | 5.48 | 20.18 | 9.90 | 0.83M (0.12%) |
| **VAR d16** | **FFT** | **6.11** | 5.74 | **12.08** | **7.42** | 309.6M |
| VAR d16 | LoRA | 6.94 | 7.84 | 13.18 | 8.87 | 6.02M (1.91%) |
| **VAR d20** | **FFT** | **5.38** | **5.58** | **11.65** | **6.31** | 599.7M |
| VAR d20 | LoRA | 6.97 | 6.29 | 11.16 | 9.42 | 9.42M (1.54%) |

Key Findings:
- VAR FFT completely outperforms DiffFit on all datasets.
- VAR LoRA can also outperform or match DiffFit on most datasets.
- **VAR converges extremely fast**: it reaches the final FID in only thousands of steps, whereas diffusion models require long-term training.

### Differentially Private Adaptation (Oxford Flowers, LoRA)

| Model | $k=1$ | $k=128$ |
|------|-------|---------|
| VAR-d16 | 69.92 | 63.24 |
| VAR-d20 | 68.92 | 59.29 |

DP-LoRA under different $\epsilon$ ($k=32$):

| Model | $\epsilon=1$ | $\epsilon=10$ | $\epsilon=100$ | $\epsilon=1000$ |
|------|-------------|--------------|----------------|-----------------|
| VAR-d16 | 196.52 | 60.24 | 41.63 | 35.36 |
| VAR-d20 | 160.33 | 63.38 | 43.35 | 35.06 |

Key Findings:
- Under DP fine-tuning, the model struggles to converge, requiring extremely high $\epsilon$ values to obtain acceptable generation quality.
- Augmentation multiplicity ($k=128$) only brings modest improvements but increases the computational cost by 128 times.
- LoRA outperforms LNTuning in DP scenarios, likely due to having fewer trainable parameters.

### Computational Cost

- FFT has the highest computational cost (approximately 4.5 times that of PEFT on Food-101).
- LNTuning has the lowest computational cost.
- LoRA achieves the best balance between performance and cost.

## Highlights & Insights

1. **First systematic benchmark for VAR adaptation**: Fills the evaluation gap in model adaptation for vision autoregressive models.
2. **Convergence speed advantage**: VAR converges in only a few update steps, in contrast to DMs which require extensive training; this stems from VAR's deterministic prediction target (free from input noise stochasticity).
3. **Unveiled challenges of DP adaptation**: The gradient clipping and noise injection in DP-SGD affect VAR more severely than DMs, opening up new research directions.
4. **Practical value of open-source code**: Releases implementations and patches for all adaptation methods, lowering the barrier for future research.

## Limitations & Future Work

1. Evaluated only on class-conditional VAR, without exploring text-conditional or unconditional settings.
2. Poor DP adaptation performance without an effective solution presented yet.
3. Evaluated only at 256×256 resolution, leaving higher-resolution scenarios unexplored.
4. Augmentation multiplicity yields limited improvement and incurs massive computational overhead.

## Related Work & Insights

- **Vision Autoregressive Models**: VAR, Infinity, iGPT
- **Diffusion Model Adaptation**: DiffFit, DreamBooth, Textual Inversion
- **Parameter-Efficient Fine-Tuning**: LoRA, LNTuning
- **Differentially Private Generative Models**: DPDM, DP-LDM

## Rating

⭐⭐⭐ — The work is solid and the open-source code holds practical value, but the primary contribution lies in "implementation and benchmarking" rather than methodological innovation. The issue of poor DP adaptation performance is identified but remains unresolved. While foundational as the first VAR adaptation benchmark, its depth and novelty are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Autoregressive Visual Decoding from EEG Signals](../../ICLR2026/medical_imaging/autoregressive_visual_decoding_from_eeg_signals.md)
- [\[CVPR 2026\] GeneVAR: Causal MeanFlow for Autoregressive Gene-to-WSI Tile Synthesis](../../CVPR2026/medical_imaging/genevar_causal_meanflow_for_autoregressive_gene-to-wsi_tile_synthesis.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](../../NeurIPS2025/medical_imaging/toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](../../CVPR2026/medical_imaging/tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[ICLR 2026\] CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation](../../ICLR2026/medical_imaging/cerebragloss_instruction-tuning_a_large_vision-language_model_for_fine-grained_c.md)

</div>

<!-- RELATED:END -->
