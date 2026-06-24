---
title: >-
  [Paper Note] Finetuning Stellar Spectra Foundation Models with LoRA
description: >-
  [ICML 2025][Physics & Scientific Computing][LoRA] This work applies LoRA to the stellar spectra foundation model SpecCLIP for the first time, achieving efficient adaptation of models pre-trained on LAMOST/Gaia XP to DESI survey data with only approximately 100-200 labeled samples, demonstrating that LoRA is a lightweight and effective strategy for cross-survey spectral migration.
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "LoRA"
  - "Foundation Models"
  - "Stellar Spectra"
  - "Cross-Survey Adaptation"
  - "Few-Shot Learning"
date: 2026-05-08
content_hash: a8d499eb98f0f20f
---

# Finetuning Stellar Spectra Foundation Models with LoRA

**Conference**: ICML 2025  
**arXiv**: [2507.20972](https://arxiv.org/abs/2507.20972)  
**Code**: None  
**Area**: AI4Science / Astrophysics  
**Keywords**: LoRA, Foundation Models, Stellar Spectra, Cross-Survey Adaptation, Few-Shot Learning

## TL;DR

This work applies LoRA to the stellar spectra foundation model SpecCLIP for the first time, achieving efficient adaptation of models pre-trained on LAMOST/Gaia XP to DESI survey data with only approximately 100-200 labeled samples, demonstrating that LoRA is a lightweight and effective strategy for cross-survey spectral migration.

## Background & Motivation

**Background**: Large-scale spectroscopic surveys (e.g., LAMOST, Gaia, DESI) have driven advancements in Galactic research. Stellar parameter estimation methods have evolved from traditional template matching (e.g., UlySS, LSP3) to machine learning approaches (The Cannon, The Payne, TransformerPayne). Recently, spectral foundation models like SpecCLIP have begun to emerge.

**Limitations of Prior Work**: Different spectroscopic surveys vary significantly in wavelength coverage, resolution, and signal-to-noise ratio (SNR). Most existing methods rely on heavy supervision and are bound to specific surveys, making it difficult to achieve consistent parameter estimation across heterogeneous stellar spectra. Although foundation models exhibit potential for generalization, how to adapt them to a new survey with minimal supervision remains an open challenge.

**Key Challenge**: The pre-training cost of spectral foundation models is high, making full fine-tuning impractical. Furthermore, labeled data is extremely scarce in few-shot scenarios. There is an urgent need for a parameter-efficient adaptation scheme to rapidly deploy these models on new surveys.

**Goal**: (1) Can LoRA effectively adapt spectral foundation models to completely new surveys? (2) How do fine-tuning different modules (foundation model, projection head, downstream MLP) affect performance? (3) Can cross-modal knowledge from Gaia XP embedded during pre-training aid DESI adaptation?

**Key Insight**: Spectral data possesses structured properties similar to language (local features correspond to physical information). Since LoRA has proven effective in NLP and CV, the authors introduce it to the field of astronomical spectra for the first time. SpecCLIP's multimodal contrastive pre-training provides a rich baseline of cross-survey knowledge.

**Core Idea**: Utilizing LoRA to efficiently migrate spectral foundation models to new surveys with minimal parameters (less than 3% of the total model parameters) and extremely few samples (approximately 100 labeled instances).

## Method

### Overall Architecture

The input consists of DESI spectra (normalized and interpolated to the LAMOST wavelength grid of 400-560nm), which pass through the pre-trained SpecCLIP foundation model to extract embeddings (768-dimensional). These embeddings are then mapped to a shared contrastive learning space via a projection network, and finally, a downstream MLP predicts the iron abundance [Fe/H]. LoRA modules are selectively inserted at four different locations for fine-tuning.

### Key Designs

1. **SpecCLIP Pre-training Foundation (Frozen Backbone)**:
    - **Function**: Establish foundation models for the LAMOST LRS and Gaia XP modalities and align them via contrastive learning.
    - **Mechanism**: The LAMOST LRS foundation model is a 6-layer Transformer encoder (42.7M parameters) that tokenizes 1462 flux points (using a window size of 20 and a stride of 10 to obtain 146 tokens) and is pre-trained using masked language modeling. The Gaia XP model is an MLP autoencoder processing 343-dimensional spectra. Contrastive training aligns the embeddings of 820K paired spectra to a shared space via modality-specific projection networks.
    - **Design Motivation**: Contrastive pre-training allows the model to learn physically meaningful representations shared across surveys, laying a solid foundation for subsequent migration.

2. **Four-Module LoRA Fine-Tuning Strategy**:
    - **Function**: Insert and fine-tune LoRA modules in four distinct components of the architecture, either individually or in combination.
    - **Mechanism**: LoRA decomposes weight updates as $\Delta W = AB$ ($A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$). LoRA1 is inserted into all self-attention layers of the LRS foundation model (rank=4, α=8, 129K parameters / 0.30%); LoRA2 is inserted into the LRS downstream MLP (rank=8, α=16, 31.7K parameters / 2.30%); LoRA3 is inserted into the projection network (rank=16, α=32, 147K parameters / 0.29%); LoRA4 is inserted into the post-projection downstream MLP (configured identically to LoRA2).
    - **Design Motivation**: Different modules carry different levels of knowledge—the foundation model encodes spectral features, the projection network encodes cross-modal alignment, and the MLP encodes label mapping. Testing them individually reveals the key pathways of knowledge transfer.

3. **Cross-Survey Data Adaptation Pipeline**:
    - **Function**: Standardize and feed DESI spectra into the SpecCLIP pipeline.
    - **Mechanism**: DESI spectra are retrieved via SPARCL, normalized using the same pipeline as LAMOST LRS, and interpolated to the 400-560nm wavelength grid. Cross-matching with APOGEE DR17 yields high-precision [Fe/H] labels for 495 stars, with 89 used for training, 9 for validation, and 396 for testing. LoRA1/3 fine-tuning additionally utilizes 164 unlabeled DESI samples (SNR > 50).
    - **Design Motivation**: Deliberately using different subsets for foundation model fine-tuning and downstream fine-tuning increases adaptation difficulty to rigorously test generalization capabilities.

### Loss & Training

The downstream MLP is trained using an [Fe/H] regression loss. Performance is evaluated using the robust standard deviation (Tukey Biweight Scale Estimator) and the $R^2$ metric. Each experiment is completed on a single NVIDIA V100 GPU within 10 to 180 seconds.

## Key Experimental Results

### Main Results

| Method | Full Test Set σ↓ | Full Test Set R²↑ | Metal-Rich Stars σ↓ | Metal-Rich Stars R²↑ |
|------|-----------|------------|-----------|------------|
| Zero-shot (MLP1) | 0.2730 | 0.7358 | 0.2479 | 0.0702 |
| LoRA2 | 0.2663 | 0.7156 | 0.2272 | 0.2378 |
| LoRA1+LoRA2 | 0.2227 | 0.7719 | 0.1924 | 0.4173 |
| Zero-shot (MLP2) | 0.2560 | 0.7203 | 0.2371 | 0.0725 |
| **LoRA4** | **0.2023** | **0.7937** | **0.1621** | **0.5106** |
| LoRA1+LoRA3+LoRA4 | 0.2297 | 0.7801 | 0.1851 | 0.4274 |

### Ablation Study

| Configuration | Metal-Poor Stars σ↓ | Metal-Poor Stars R²↑ | Description |
|------|-----------|------------|------|
| Zero-shot (MLP1) | **0.4444** | **-0.5130** | Zero-shot is paradoxically best on the metal-poor end |
| LoRA2 | 0.5872 | -1.2881 | Fine-tuning only the MLP leads to overfitting in sparse regions |
| LoRA1+LoRA2 | 0.5151 | -0.9143 | Joint fine-tuning mitigates but does not resolve the issue |
| LoRA4 | 0.5803 | -0.8357 | Even the best configuration degrades on the metal-poor end |
| LoRA1+LoRA3+LoRA4 | 0.5970 | -0.8159 | Full-module fine-tuning also fails in this region |

### Key Findings

- **LoRA4 (fine-tuning only the downstream MLP on the Gaia XP alignment path) achieves the best performance**, indicating that the cross-modal Gaia XP knowledge introduced during SpecCLIP pre-training provides critical information for DESI adaptation, even though resolution and bands differ substantially between DESI and Gaia XP.
- Jointly fine-tuning the foundation model (LoRA1) can bring additional gains, but it is not always optimal—it indeed degrades performance for metal-poor stars, suggesting that few-shot fine-tuning easily overfits in label-sparse regions.
- All methods perform poorly on metal-poor stars ([Fe/H] < -1, only 60 test stars), yielding negative $R^2$ values.

## Highlights & Insights

- **First to introduce LoRA into stellar spectroscopy**, demonstrating the transferability of parameter-efficient fine-tuning techniques from NLP/CV to astronomy, establishing a standard paradigm for cross-survey deployment of spectral foundation models.
- **Indirect transfer of Gaia XP knowledge** is particularly intriguing—the success of LoRA4 suggests that cross-modal information from contrastive pre-training embeddings (Gaia XP $\rightarrow$ shared space) can indirectly assist DESI through the projection pathway, despite DESI itself never being involved in the pre-training phase.
- Significant performance improvements are achieved with very few annotations (89 training samples), highlighting the immense potential of foundation models combined with LoRA in data-scarce scientific fields.

## Limitations & Future Work

- Heavy degradation in predicting metal-poor stars highlights the need for better regularization strategies or targeted data augmentation.
- With only 89 labeled samples in the training set, there is a clear data scarcity bottleneck; larger cross-matched catalogs could lead to substantial improvements.
- Only a single parameter ([Fe/H]) is predicted, leaving its effectiveness on other stellar parameters (e.g., effective temperature $T_{\text{eff}}$, surface gravity $\log g$, $[\alpha/\text{Fe}]$) unverified.
- The wavelength range is limited to 400-560nm (LAMOST grid), leaving DESI's full wavelength coverage (360-980nm) unutilized, which potentially discards valuable spectral information.

## Related Work & Insights

- **vs The Cannon/The Payne**: Traditional methods require large numbers of labeled samples and are locked to specific surveys, whereas the LoRA + foundation model approach achieves order-of-magnitude improvements in annotation efficiency.
- **vs Full Fine-Tuning**: LoRA updates less than 3% of the parameters and completes training in seconds, whereas full fine-tuning would suffer from severe overfitting under such small-sample conditions.
- **vs AstroCLIP**: AstroCLIP first introduced CLIP to astronomy, while SpecCLIP specializes it for the spectroscopic domain and incorporates mechanisms to preserve modality-specific information.

## Rating

- **Novelty**: ⭐⭐⭐ Although LoRA is not a new technique itself, its first application to astronomical spectra is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐ Systematically compares different LoRA configurations, but only evaluates one downstream task and one target survey.
- **Writing Quality**: ⭐⭐⭐⭐ Clear method diagrams and reasonable experimental design.
- **Value**: ⭐⭐⭐⭐ Establishes a standard parameter-efficient fine-tuning pipeline for cross-survey deployment of spectral foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[NeurIPS 2025\] The Platonic Universe: Do Foundation Models See the Same Sky?](../../NeurIPS2025/physics/the_platonic_universe_do_foundation_models_see_the_same_sky.md)
- [\[ICML 2025\] OmniArch: Building Foundation Model For Scientific Computing](omniarch_building_foundation_model_for_scientific_computing.md)
- [\[ICML 2026\] Foundation Inference Models for Ordinary Differential Equations](../../ICML2026/physics/foundation_inference_models_for_ordinary_differential_equations.md)

</div>

<!-- RELATED:END -->
