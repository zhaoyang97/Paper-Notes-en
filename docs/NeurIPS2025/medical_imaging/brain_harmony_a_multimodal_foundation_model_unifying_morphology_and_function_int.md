---
title: >-
  [Paper Note] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens
description: >-
  [NeurIPS 2025][Medical Imaging][brain imaging foundation model] The first multimodal brain foundation model that unifies structural morphology (T1 sMRI) and functional dynamics (fMRI), compressing high-dimensional neuroimaging data into compact 1D token representations via Geometric Harmonics Pre-alignment and Temporally Adaptive Patch Embedding (TAPE). The model consistently outperforms prior methods on neurodevelopmental/neurodegenerative disease diagnosis and cognitive prediction tasks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - brain imaging foundation model
  - multimodal fusion
  - fMRI
  - sMRI
  - 1D token
date: 2026-05-08
content_hash: 3254f7dfd5b303b6
---

# Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens

**Conference**: NeurIPS 2025
**arXiv**: [2509.24693](https://arxiv.org/abs/2509.24693)
**Code**: [hzlab/Brain-Harmony](https://github.com/hzlab/Brain-Harmony)
**Area**: Medical Imaging
**Keywords**: brain imaging foundation model, multimodal fusion, fMRI, sMRI, 1D token

## TL;DR

The first multimodal brain foundation model that unifies structural morphology (T1 sMRI) and functional dynamics (fMRI), compressing high-dimensional neuroimaging data into compact 1D token representations via Geometric Harmonics Pre-alignment and Temporally Adaptive Patch Embedding (TAPE). The model consistently outperforms prior methods on neurodevelopmental/neurodegenerative disease diagnosis and cognitive prediction tasks.

## Background & Motivation

**Single-modality limitations**: Existing brain foundation models either model structure alone (BrainMVP) or function alone (BrainLM, Brain-JEPA, BrainMass), failing to capture complementary structure–function information.

**Structure constrains function**: Neuroscientific evidence shows that functional brain activity propagates along cortical geometry ("function follows structure"), yet existing functional models entirely ignore this prior.

**Heterogeneous TR problem**: fMRI sampling rates (TR) vary across scanners and protocols. BrainLM and Brain-JEPA only handle fixed TR, severely limiting multi-dataset joint pretraining and clinical deployment.

**Information loss**: BrainMass relies on static functional connectivity, discarding non-stationary dynamics in BOLD signals (e.g., state transitions and co-activation pattern evolution).

**Data scale**: Pretraining employs the largest neuroimaging dataset to date—64,594 T1 3D volumes (~14 million images) and 70,933 fMRI time series.

**Compact representation**: There is a need to deeply compress high-dimensional neuroimaging data into compact, information-dense 1D continuous tokens forming a unified latent space of the human brain.

## Method

### Overall Architecture

BrainHarmonix adopts a two-stage pretraining pipeline: **Unimodal Encoding (UE)** → **Multimodal Fusion (MF)**.

- **BrainHarmonix-S** (structural): ViT-B backbone + 3D Masked Autoencoder (MAE), pretrained on 64,594 T1 images.
- **BrainHarmonix-F** (functional): ViT-B backbone + JEPA framework, pretrained on 252,961 fMRI time series.
- **Harmonizer**: ViT-B encoder + MAE-style decoder, fusing the latent representations of both modalities via learnable 1D brain hub tokens.

### Key Designs

1. **Geometric Harmonics Pre-alignment**
   - Computes the eigendecomposition of the Laplace-Beltrami Operator (LBO) on a population-level cortical surface mesh, yielding a set of orthogonal geometric harmonics $\psi_i$.
   - Downsamples the geometric harmonics to ROI level ($\bar{\psi}_i \in \mathbb{R}^{N\times 1}$) and projects them via a linear layer to serve as Transformer positional encodings.
   - Injects physical priors (cortical morphological constraints) into fMRI representations, enhancing alignment across subjects and datasets.

2. **Temporally Adaptive Patch Embedding (TAPE)**
   - Defines a unified time window $\tau$ and dynamically computes the patch size according to TR: $k = \text{round}(\tau / s)$.
   - Adaptively rescales embedding weights via pseudo-inverse resizing (PI-resize): $\omega = ((B_k^{k^*})^T)^\dagger \cdot \omega^*$.
   - Handles variable-length time series with zero-padding and attention masking, enabling fMRI pretraining with arbitrary TR for the first time.
   - Gives rise to the first data augmentation method for fMRI time series: downsampling high-resolution scans to multiple TR levels (e.g., UKB 0.735s → 1.47s/2.205s/2.94s).

3. **Brain Hub Token Fusion**
   - Introduces $N_H=128$ learnable continuous 1D brain hub tokens $\mathbf{H}_0 \in \mathbb{R}^{N_H \times d}$.
   - Concatenates hub tokens with structural/functional tokens as input to the Harmonizer: $\mathbf{Z}_0 = [\mathbf{H}_0; \mathbf{Z}_S; \mathbf{Z}_F]$.
   - Self-attention allows hub tokens to simultaneously aggregate information from both modalities and facilitate cross-modal interaction.

### Loss & Training

- **UE stage**: MAE reconstruction loss for structure; JEPA prediction loss for function (standard implementations).
- **MF stage**: Dual-modality reconstruction MSE loss:

$$\mathcal{L}_{\text{fusion}} = \|\mathcal{D}_S(\tilde{\mathbf{H}}) - \mathbf{Z}_S\|_2^2 + \|\mathcal{D}_F(\tilde{\mathbf{H}}) - \mathbf{Z}_F\|_2^2$$

## Key Experimental Results

### Datasets

| Stage | Dataset | Scale |
|-------|---------|-------|
| T1 Pretraining | UKB + ABCD | 64,594 scans |
| fMRI Pretraining | UKB + ABCD (with TR augmentation) | 252,961 time series |
| Fusion | UKB + ABCD paired | 69,360 pairs |

### Main Results

**Neurodevelopmental disorder diagnosis (Table 1)**:

| Model | ABIDE-I ACC/F1 | ABIDE-II ACC/F1 | ADHD-200 ACC/F1 |
|-------|---------------|-----------------|-----------------|
| BrainMass | 65.64/69.07 | 59.35/71.86 | 65.99/61.27 |
| Brain-JEPA | Not testable (fixed TR) | — | — |
| BrainHarmonix-F | 57.39/71.24 | 62.90/72.76 | 67.69/68.75 |
| **BrainHarmonix** | **63.13/72.63** | **66.67/74.88** | **70.09/66.72** |

**Neurodegenerative disease & cognitive prediction (Table 2)**:

| Model | PPMI ACC/F1 | ADNI ACC/F1 | HCP-A MAE/ρ |
|-------|------------|------------|-------------|
| Brain-JEPA | 60.36/48.76 | 59.60/60.78 | 5.62/0.26 |
| BrainHarmonix-F | 62.79/52.90 | 61.62/64.80 | 5.77/0.30 |
| **BrainHarmonix** | **64.34/56.40** | **64.65/68.75** | 6.56/**0.42** |

### Key Findings

1. **Multimodal fusion consistently yields gains**: BrainHarmonix achieves the best results on 5 of 6 benchmarks (p<0.05 significance).
2. **Token scaling**: Performance improves steadily as the number of 1D tokens increases from 32 to 256, saturating between 128 and 256.
3. **Linear probing surpasses prior SOTA**: Linear probing with only 0.0015M trainable parameters already outperforms previous best methods.
4. **Ablation validation**: Both Geometric Harmonics Pre-alignment and TR data augmentation yield significant and consistent improvements.
5. **Attention analysis**: Among 128 hub tokens, 93 attend predominantly to fMRI, 30 to T1, and 5 exhibit cross-modal attention, demonstrating automatic modality specialization.

## Highlights & Insights

- **First brain foundation model unifying structure and function**, with clear neuroscientific motivation throughout both problem formulation and technical design.
- **Elegant TAPE design**: A concise formulation resolves the long-standing heterogeneous-TR problem and naturally gives rise to fMRI data augmentation.
- **Brain hub token as an information bottleneck** is compelling—compressing complex brain structure and function into 128 1D tokens while enabling effective reconstruction.
- **Interpretability analysis is substantive**: Attention patterns align closely with ASD-related brain regions and networks (temporoparietal junction, default mode network, etc.).
- Pretraining requires only 8 H100 GPUs, with the fusion stage taking approximately 10 hours—a reasonable computational cost for the field.

## Limitations & Future Work

- The pretraining data age distribution skews toward middle-aged and older adults (UKB: 44–83 years) and children (ABCD: 8–11 years), with limited coverage of infants and young adults.
- The unimodal encoders and fusion module are trained separately rather than jointly end-to-end, potentially introducing representational inconsistencies.
- On HCP-A cognitive prediction, BrainHarmonix MAE (6.56) underperforms BrainMVP2 (5.39), indicating that multimodal fusion does not yield uniform gains across all metrics.
- Validation is limited to the Schaefer-400 parcellation; the effect of finer, coarser, or parcellation-free schemes remains unexplored.
- The model currently addresses only resting-state fMRI and has not been extended to task-based fMRI or other modalities (DTI/EEG).

## Related Work & Insights

| Model | Modality | Heterogeneous TR | Pretraining Strategy |
|-------|----------|-----------------|---------------------|
| BrainLM | fMRI time series | ✗ | MAE |
| Brain-JEPA | fMRI time series | ✗ | JEPA |
| BrainMass | fMRI connectivity | ✓ (static) | Self-supervised |
| BrainMVP | T1/T2 structural | N/A | Multi-parameter MRI contrastive |
| BDO | fMRI dynamics | — | Stochastic optimal control |
| **BrainHarmonix** | **T1 + fMRI** | **✓** | **MAE + JEPA + Hub Token Fusion** |

## Rating

- Novelty: ⭐⭐⭐⭐ — First multimodal brain foundation model; TAPE and hub token designs are original
- Experimental Thoroughness: ⭐⭐⭐⭐ — 6 downstream benchmarks + ablations + token scaling + interpretability analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, polished figures, well-articulated neuroscientific motivation
- Value: ⭐⭐⭐⭐ — Establishes a multimodal foundation model paradigm for neuroimaging, though absolute performance gains are modest

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)

</div>

<!-- RELATED:END -->
