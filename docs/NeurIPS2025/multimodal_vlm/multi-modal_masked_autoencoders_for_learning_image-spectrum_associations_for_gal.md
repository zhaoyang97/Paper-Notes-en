---
title: >-
  [Paper Note] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology
description: >-
  [NeurIPS 2025][Multimodal VLM][Multi-modal masked autoencoder] This work constructs GalaxiesML-Spectra, a large-scale multi-modal dataset of 134,533 galaxies with images, spectra, and redshifts, and adapts a Multi-Modal Masked Autoencoder (MMAE) for joint image–spectrum reconstruction and redshift regression. It demonstrates that at test time, even with spectra entirely absent, using only 25% masked images achieves a redshift prediction scatter of $\sigma_{NMAD} = 0.016$, surpassing AstroCLIP.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Multi-modal masked autoencoder
  - galaxy images
  - spectral reconstruction
  - redshift regression
  - missing modality learning
date: 2026-05-08
content_hash: 5f728a609072979b
---

# Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology

**Conference**: NeurIPS 2025
**arXiv**: [2510.22527](https://arxiv.org/abs/2510.22527)
**Code**: Available (GitHub + Zenodo dataset)
**Area**: Multi-modal Learning / Astronomy
**Keywords**: Multi-modal masked autoencoder, galaxy images, spectral reconstruction, redshift regression, missing modality learning

## TL;DR

This work constructs GalaxiesML-Spectra, a large-scale multi-modal dataset of 134,533 galaxies with images, spectra, and redshifts, and adapts a Multi-Modal Masked Autoencoder (MMAE) for joint image–spectrum reconstruction and redshift regression. It demonstrates that at test time, even with spectra entirely absent, using only 25% masked images achieves a redshift prediction scatter of $\sigma_{NMAD} = 0.016$, surpassing AstroCLIP.

## Background & Motivation

**State of the Field**: Next-generation astronomical surveys (LSST, Euclid, etc.) will image billions of galaxies, yet acquiring spectra requires roughly 100× more observing time than imaging. Redshift $z$—quantifying spectral shifts due to cosmic expansion—is a fundamental cosmological observable, but precise redshifts require spectroscopy. Existing ML approaches include CNN/MLP-based photometric redshift estimation, AstroMAE (single-modal image MAE), and AstroCLIP (contrastive learning for joint image–spectrum embeddings).

**Limitations of Prior Work**: (1) Most methods model only a single modality and cannot learn cross-modal associations. (2) AstroCLIP performs contrastive alignment without reconstruction and is validated only at low redshift $z < 0.5$. (3) MAE has not been explored in astronomical multi-modal settings.

**Root Cause**: Upcoming surveys will generate vast numbers of images but almost no spectra. A method is needed that can learn physically meaningful spectral representations from images alone. The MAE training objective of "recovering the whole from partial observations" naturally simulates spectral absence scenarios.

**Paper Goals**: (1) Construct a large-scale astronomical dataset with images, spectra, and redshifts; (2) validate the feasibility of MMAE for multi-modal astronomical reconstruction and redshift regression; (3) evaluate model performance under complete spectral absence.

**Starting Point**: The MultiMAE framework is adopted to unify 5-band galaxy images and 1D spectra as patch tokens, trained with 75% masking for joint reconstruction, with an integrated redshift regression head. During training, spectra are fully masked for 50% of samples to simulate real survey conditions.

**Core Idea**: A multi-modal masked autoencoder learns shared representations of galaxy images and spectra, enabling redshift prediction at test time without any spectral input.

## Method

### Overall Architecture

Input: 5-band images $(64\times64\times5)$ + 1D spectrum (259 pixels) → patchified separately → 75% random masking → independent Transformer encoding → cross-attention fusion → attention pooling for joint representation → three task heads: image decoder, spectrum decoder, redshift regressor.

### Key Designs

1. **Dual-Modality Patch Tokenization + Independent Encoding**:

   - Function: Converts images and spectra into token sequences, extracting intra-modal features independently.
   - Mechanism: Images are patchified using $8\times8\times5$ 2D convolutions and projected to 256 dimensions with 2D learnable positional encodings. Spectra are divided into 1D patches of length 8 and linearly projected to the same dimension. Each modality is encoded by an independent 1D Transformer (depth 4, 8 heads, dropout 0.1) with 75% of tokens masked.
   - Design Motivation: Independent encoding allows each modality to capture its own internal structure. The high 75% masking rate forces the model to learn strong representations rather than memorizing inputs.

2. **Cross-Attention Fusion**:

   - Function: Establishes information flow between modalities, enabling spectra to inform image morphology understanding and images to guide spectral inference.
   - Mechanism: Across 4 cross-attention layers, image tokens query spectral tokens and vice versa. After fusion, attention pooling produces global image and spectral embeddings, which are concatenated into a joint representation.
   - Design Motivation: Cross-modal attention enables the model to learn physical associations such as "emission line positions indicate galaxy type" and "galaxy morphology constrains the redshift range."

3. **Joint Training Objective (Reconstruction + Regression)**:

   - Function: Simultaneously optimizes reconstruction and redshift prediction, so that learned representations capture both fine-grained detail and high-level semantics.
   - Mechanism: The loss is a weighted sum $\mathcal{L} = 0.1 \cdot \mathcal{L}_{img} + 0.01 \cdot \mathcal{L}_{spec} + 1.0 \cdot \mathcal{L}_z$. Reconstruction uses MSE over masked tokens only. The redshift loss is $\mathcal{L}_z = 1 - 1/(1+(dz/0.15)^2)$, where $dz = (z_{pred}-z_{spec})/(1+z_{spec})$. Spectra are fully zeroed for 50% of training samples.
   - Design Motivation: Embedding redshift regression directly into MAE training (rather than conventional pre-train-then-fine-tune) guides the encoder to extract physically meaningful features during reconstruction. The 50% spectral masking directly simulates real-world spectral absence.

### Loss & Training

AdamW (weight decay 0.01, lr 0.0001) with gradient clipping. The dataset is split 70/15/15 into train/validation/test (~94k/20k/20k). Spectral preprocessing: normalization + downsampling to 259 pixels.

## Key Experimental Results

### Main Results

| Method | Test Condition | Redshift Range | $\sigma_{NMAD}$ |
|--------|---------------|----------------|----------------|
| MMAE (25% img mask, 100% spec mask) | Image only | $z \lesssim 0.4$ | **0.016** |
| MMAE (0% img mask, 100% spec mask) | Image only | $z \lesssim 0.4$ | 0.026 |
| AstroCLIP | Image + spectrum | $z \lesssim 0.4$ | 0.020 |
| Fine-tuned BCNN | Image only | $z \lesssim 0.4$ | 0.012 |

### Ablation Study

| Reconstruction Target | Captured | Limitations |
|----------------------|----------|-------------|
| Image reconstruction | Galaxy shape/color ✓ | Nearby galaxy details/background noise ✗ |
| Spectral reconstruction | Continuum shape ✓, H-α/Ly-α positions ✓ | Line width severely overestimated, line strength underestimated |
| Redshift regression | Accurate for $z<1$ | Degrades for $z>1$, staircase artifacts |

### Key Findings

- **25% image masking outperforms no masking**: $\sigma_{NMAD}$ decreases from 0.026 to 0.016. Moderate masking acts as regularization, preventing overfitting to small-scale features. This differs from the standard MAE finding that higher masking rates are optimal, possibly because astronomical images have lower information density.
- **Physical features in spectral reconstruction**: The model learns that "a specific emission line should appear at a given redshift" (e.g., H-α position error of 24Å), but line widths are overestimated by 15× (34.5Å → 528Å) and line strengths are underestimated by 5×. Line ratios—an important physical diagnostic—fail entirely.
- **Staircase structure in redshift predictions**: Corresponds to redshift intervals where strong spectral lines enter or exit the spectrograph range (e.g., Lyman-α at $z\sim2$), indicating high model sensitivity to the visibility of specific lines.
- A gap remains relative to BCNN ($\sigma_{NMAD}=0.012$): Inception-style CNNs are more robust on redshift estimation tasks.

## Highlights & Insights

- **Natural alignment between MAE training and missing-modality surveys**: Training with random masking (50% fully masked spectra) directly simulates the operational reality of surveys lacking spectroscopic coverage. This principle of "designing training strategies for deployment scenarios" generalizes to any missing-modality setting.
- **Masking as regularization**: In astronomical images with relatively low information density, 25% masking improves performance. The optimal masking rate should be calibrated according to the information density of the data.
- This is the first work to jointly perform multi-modal reconstruction and regression within a single astronomical framework, extending the redshift range to $z\sim4$—far beyond AstroCLIP's $z\lesssim0.5$.
- GalaxiesML-Spectra (134k galaxies, HSC images + DESI spectra) is an independent contribution.

## Limitations & Future Work

- **Poor emission line reconstruction quality**: Line widths and strengths cannot be accurately recovered, and line ratios fail entirely. Physics-informed loss terms (e.g., parametric spectral line constraints, auxiliary line detection losses) should be incorporated.
- **Gap relative to CNN baseline**: Transformers underperform Inception-style CNNs on small-scale redshift tasks; larger datasets or deeper models are needed.
- **Insufficient high-redshift data**: GalaxiesML skews toward low redshift and high brightness, limiting generalization at high redshift. High-redshift sources from DESI Legacy Imaging Surveys should be incorporated.
- MSE reconstruction loss assigns insufficient weight to sharp spectral peaks; weighted MSE or perceptual losses should be considered.
- The model is small (depth 4, embedding dim 256); no scaling ablations are performed.
- Transfer to other downstream tasks (morphology classification, star formation rate estimation) has not been evaluated.

## Related Work & Insights

- **vs. AstroCLIP**: Contrastive learning aligns images and spectra without reconstruction. The proposed MMAE achieves lower redshift scatter (0.016 vs. 0.020) over the same redshift range while also performing reconstruction, though the comparison is not entirely fair.
- **vs. AstroMAE**: Operates on single-modal images only. This work extends to multi-modal joint spectral reconstruction.
- **vs. BCNN**: BCNN achieves superior $\sigma_{NMAD}=0.012$ but is specifically fine-tuned for redshift. MMAE's advantage lies in learning general-purpose representations that can scale to additional modalities and tasks.
- The framework naturally extends to additional modalities such as textual metadata and multi-epoch observations.

## Rating

- Novelty: ⭐⭐⭐ The MMAE framework is not new, but its astronomical multi-modal application is a first.
- Experimental Thoroughness: ⭐⭐⭐ Dataset construction is solid, but ablations lack depth.
- Writing Quality: ⭐⭐⭐ Structure is clear; analysis of some results is relatively shallow.
- Value: ⭐⭐⭐ A meaningful cross-domain application that establishes feasibility, though findings are limited.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[NeurIPS 2025\] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)

<!-- RELATED:END -->
