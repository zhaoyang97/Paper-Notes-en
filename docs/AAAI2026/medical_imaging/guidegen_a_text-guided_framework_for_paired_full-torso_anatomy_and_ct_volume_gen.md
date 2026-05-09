---
title: >-
  [Paper Note] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation
description: >-
  [Medical Imaging] GuideGen proposes a controllable framework that requires only text input. It synthesizes full-torso anatomical masks via a categorical diffusion model, and combines an anatomy-aware high-dynamic-range autoencoder with a latent feature generator to produce paired full-torso CT volumes, providing high-quality synthetic training data for downstream segmentation tasks.
tags:
  - Medical Imaging
date: 2026-05-08
content_hash: 6a5e48401ea44f39
---

# GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation

## Paper Information

- **Conference**: AAAI 2026
- **arXiv**: [2403.07247](https://arxiv.org/abs/2403.07247)
- **Code**: [https://github.com/OvO1111/GuideGen](https://github.com/OvO1111/GuideGen)
- **Area**: Medical Imaging
- **Keywords**: CT generation, text guidance, anatomy mask synthesis, full-torso, categorical diffusion model, high dynamic range, segmentation data augmentation

## TL;DR

GuideGen proposes a controllable framework that requires only text input. It synthesizes full-torso anatomical masks via a categorical diffusion model, and combines an anatomy-aware high-dynamic-range autoencoder with a latent feature generator to produce paired full-torso CT volumes, providing high-quality synthetic training data for downstream segmentation tasks.

## Background & Motivation

Acquisition of large-scale medical image datasets is constrained by privacy concerns and annotation costs; conditional generative models offer a promising solution. Existing methods suffer from the following limitations:

**Semantic-conditioned methods** (e.g., MAISI): limited sample diversity; rely on costly fine-grained anatomical masks.

**Text-conditioned methods** (e.g., GenerateCT): flexible but unable to fully capture precise spatial relationships among anatomical structures.

**Limitations of hybrid methods**: Kim et al. require both masks and text at inference time (limiting applicability); MedSyn covers only the thorax (cannot scale to the complex anatomy of the full torso).

**Insufficient downstream usability**: existing text-guided generation primarily serves classification tasks, with inadequate support for segmentation.

The core goal of GuideGen: text input only → automatic generation of full-torso anatomical masks + corresponding CT volumes → construction of segmentation training datasets.

## Method

### Overall Architecture (Three-Stage Pipeline)

1. **Text-Conditioned Semantic Synthesizer (TCSS)**: text → discrete anatomical mask
2. **Anatomy-Aware HDR Autoencoder**: CT → high-fidelity latent features
3. **Latent-Guided Feature Generator**: semantic + text latent features → CT latent features → CT image

### Key Designs

#### 1. Text-Conditioned Semantic Synthesizer (TCSS)

**Disambiguated categorical modeling**:

Existing methods (e.g., MedGen3D) generate masks with continuous diffusion models, which fail to capture sharp transitions near semantic boundaries, leading to ambiguity. TCSS adopts a **categorical diffusion model** to directly model discrete label indices:

- Diffusion variable $\mathbf{x}_0 = \mathbf{m} \in \{1,...,N\}^{H \times W \times D}$ ($N$ = number of semantic classes)
- The forward process gradually converts class labels into uniform categorical noise:
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{C}_N(\mathbf{x}_t; (1-\beta_t)\mathbf{e}(\mathbf{x}_{t-1}) + \beta_t \cdot \frac{\mathbf{1}}{N})$$
- The training objective is KL divergence minimization, using a reparameterized form that directly predicts $\mathbf{x}_0$

**Knowledge injection module**:

- An ERNIE-series medical text encoder maps structured prompts into a latent space
- Learnable task-specific queries $Q$ interact with Transformer decoder blocks to extract task-relevant responses $R_{\text{task}}$
- Layer-wise responses $R_{\text{layer}}$ are further derived, attending to global anatomy or local structures respectively
- Layer-wise guidance is injected into the diffusion backbone via cross-attention

#### 2. Anatomy-Aware High Dynamic Range (HDR) Autoencoder

**Anatomy preservation**:
- A pyramid scheme resamples semantic masks to the resolution of latent features at each autoencoder layer and concatenates them
- Helps the model attend to semantic details during encoding and reconstruct semantically accurate images during decoding
- Addresses the problem of small tumors being ignored in low-resolution latent spaces

**HDR adaptation**:
- Rather than clipping CT intensity to a fixed range (e.g., lung window or abdominal window only), the full dynamic range is preserved
- An intensity transformation module $h(\mathbf{x})$ randomly samples window center $w_c$ and window width $w_r$:
$$h(\mathbf{x}) = k \max\{\min\{\frac{\mathbf{x} - w_c + w_r}{2w_r}, 1\}, 0\} + b$$
- Learnable coefficients $k$, $b$ map the clipped result back to the input space

**Training loss**:
$$\mathcal{L}_2 = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{perc}} + \mathcal{L}_{\text{disc}}(\mathbf{D}_f) + \mathcal{L}_{\text{disc}}(\mathbf{D}_v)$$

Includes a frame discriminator $\mathbf{D}_f$ (random slices) and a volumetric discriminator $\mathbf{D}_v$ (3D), as well as VGG-16 perceptual loss.

#### 3. Latent-Guided Feature Generator

- A standard Gaussian diffusion model operating in the autoencoder's latent space
- Resampled semantic masks are concatenated to the diffusion variable $\mathbf{z}_t$
- Textual information is injected via knowledge injection and cross-attention
- Training objective: standard $\epsilon$-prediction loss
$$\mathcal{L}_3 = \mathbb{E}_{t,\boldsymbol{\epsilon}}[\|\boldsymbol{\epsilon} - f_\varphi(\mathbf{z}_t; \text{Resample}(\hat{\mathbf{m}}), \mathbf{p})\|_2^2]$$

### Loss & Training

The three stages are optimized separately: $\mathcal{L}_1$ (KL divergence for TCSS), $\mathcal{L}_2$ (composite loss for the HDR autoencoder), and $\mathcal{L}_3$ (denoising loss for the latent generator).

## Key Experimental Results

### Data Construction

- **Training set**: 12 public TCIA datasets + 1 private colorectal cancer dataset (RJ), totaling 4,534 training / 1,179 validation samples
- **Inference evaluation**: BTCV (multi-organ segmentation), AMOS22 (abdominal multi-organ), MSD-LU/CO (tumors), KiTS21 (kidney tumors)
- Text prompts are generated from structured records by a medical LLM, formatted as: "The patient is {demographics}. In this imaging, ..."
- Pseudo-labels are generated by TotalSegmentator and nnU-Net

### Main Results

**Mask generation quality**:

| Method | Params | Full-Anatomy LPIPS↓/FID↓ | Tumor LPIPS↓/FID↓ |
|--------|--------|--------------------------|-------------------|
| MedGen3D | 48.8M | 0.70/201 | 0.29/33.5 |
| LDM | 115.2M | 0.67/98.6 | 0.30/69.1 |
| **GuideGen** | 51.5M | **0.33/7.1** | **0.29/27.9** |

Full-anatomy FID drops from the second-best 98.6 to 7.1 — a substantial improvement.

**CT image generation quality**:

| Method | Inference Condition | LPIPS↓ | FID↓ | FVD↓ |
|--------|---------------------|--------|------|------|
| MedSyn (text-only) | Text | 0.396 | 50.0 | 2012 |
| MedSyn (mask+text) | Mask + Text | 0.282 | 26.7 | 1288 |
| MAISI | Mask | 0.393 | 54.6 | 1791 |
| **GuideGen** | **Text only** | **0.248/0.256** | **20.2/19.4** | **791/745** |

GuideGen with text-only input outperforms all methods that require masks.

**Downstream multi-organ segmentation (DSC)**:

| Method | Training Samples | Spleen | Kidney | Liver | Stomach | Pancreas | Mean |
|--------|-----------------|--------|--------|-------|---------|----------|------|
| Real | 24/240 | 0.92/0.95 | 0.79/0.94 | 0.94/0.96 | 0.86/0.89 | 0.70/0.81 | 0.74/0.84 |
| MAISI | 200 | 0.91/0.83 | 0.89/0.84 | 0.94/0.91 | 0.80/0.74 | 0.61/0.60 | 0.69/0.65 |
| **GuideGen** | 200 | **0.96/0.95** | **0.91/0.92** | **0.98/0.95** | **0.90/0.90** | **0.76/0.70** | **0.79/0.78** |

On BTCV, the model trained on synthetic data even surpasses the model trained on real data (0.79 vs. 0.74), a remarkable result.

### Ablation Study

| Configuration | LPIPS↓ | FID↓ | DSC↑ | Acc.↑ |
|---------------|--------|------|------|-------|
| w/o mask input | 0.42 | 54.3 | - | 0.32 |
| w/o knowledge injection | 0.26 | 21.7 | 0.25 | 0.57 |
| w/o anatomy preservation | 0.27 | 32.4 | 0.40 | 0.61 |
| w/o HDR adaptation | 0.33 | 40.9 | 0.36 | 0.64 |
| **Full GuideGen** | **0.25** | **20.2** | **0.52** | **0.69** |

All three core components (TCSS, knowledge injection, HDR) are indispensable.

### Key Findings

1. **Categorical vs. continuous diffusion**: categorical modeling shows a clear advantage when the number of semantic classes $N$ is large (full-anatomy FID: 7.1 vs. 98.6); the gap narrows with fewer classes.
2. **Mask quality determines CT quality**: CT generation quality is positively correlated with the quality of the input mask.
3. **Remarkable effectiveness of synthetic data**: 200 synthetic samples outperform 24 real samples on BTCV; synthetic data even surpass real data for lung tumor segmentation (DSC 0.71 vs. 0.69).
4. **Augmentation gains persist with scale**: downstream performance improves continuously as the number of generated samples increases from 100 → 200 → 500 → 1000.

## Highlights & Insights

1. **Full-torso coverage**: the first framework to achieve full-torso CT generation from thorax to pelvis, substantially broadening applicable scenarios.
2. **Disambiguating via categorical diffusion**: elegantly resolves the boundary ambiguity of continuous diffusion models; discrete label modeling is more natural for semantic masks.
3. **Clever HDR adaptation design**: simulates how clinicians apply different window width/level settings to inspect different anatomical regions; random intensity window sampling during training preserves full dynamic range information.
4. **Closed loop from text to segmentation data**: users input a text description → masks and CT volumes are generated automatically → directly used to train segmentation models.
5. **Knowledge injection outperforms naive cross-attention**: learnable queries extract task-relevant information more precisely than direct cross-attention, focusing on critical descriptions (e.g., tumor location) over less informative ones (e.g., demographics).

## Limitations & Future Work

- Cannot accept free-form text input directly (requires structured prompt format, converted by an LLM).
- Generation resolution is limited to $128^3$ (mask) and $256^3$ (CT), below clinical resolution.
- Pyramid-based anatomical label injection increases computational overhead in the autoencoder.
- Tumor segmentation DSC for colorectal and kidney cancers remains substantially lower than with real data (CO: 0.21 vs. 0.47; KI: 0.64 vs. 0.72).
- Training data relies on pseudo-labels (TotalSegmentator); label noise may affect the performance upper bound.

## Related Work & Insights

- **Semantically guided CT generation**: MAISI, MedGen3D, Label-efficient GAN
- **Text-guided CT generation**: GenerateCT, MedSyn, RoentGen
- **Hybrid-condition generation**: Kim et al. (requires paired inputs), MedSyn (null mask)
- **Categorical diffusion models**: Argmax Flows, Stochastic Segmentation

## Rating

⭐⭐⭐⭐⭐ (5/5)

- The problem formulation carries significant practical value (text-driven automatic construction of medical datasets).
- The three-stage design is logically coherent, with each component having a clear design motivation.
- Experimental evaluation is comprehensive: generation quality + conditional alignment + downstream segmentation usability.
- The result showing synthetic data surpassing real data is highly convincing.
- The only limitations are the resolution constraint and dependency on structured prompts, which do not diminish the academic contribution.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](../../ICLR2026/medical_imaging/protein_as_a_second_language_for_llms.md)

<!-- RELATED:END -->
