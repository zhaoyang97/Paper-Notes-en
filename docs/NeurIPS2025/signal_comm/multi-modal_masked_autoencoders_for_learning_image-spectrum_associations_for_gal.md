---
title: >-
  [Paper Note] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology
description: >-
  [NeurIPS 2025][Signal & Communication][multi-modal masked autoencoder] A multi-modal image–spectrum–redshift dataset (GalaxiesML-Spectra) comprising 134,533 galaxies is constructed and adapted for a Multi-Modal Masked Autoencoder (MMAE) that performs joint reconstruction of images and spectra alongside redshift regression. Experiments demonstrate that, even when spectra are entirely absent at test time, using only 25% masked images achieves a redshift prediction scatter of $\sigma_{NMAD} = 0.016$, outperforming AstroCLIP.
tags:
  - NeurIPS 2025
  - "Signal & Communication"
  - multi-modal masked autoencoder
  - galaxy images
  - spectral reconstruction
  - redshift regression
  - missing modality learning
date: 2026-05-08
content_hash: c31d13bb46e0edbc
---

# Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology

**Conference**: NeurIPS 2025
**arXiv**: [2510.22527](https://arxiv.org/abs/2510.22527)
**Code**: Available (GitHub + Zenodo dataset)
**Area**: Signal Communication
**Keywords**: multi-modal masked autoencoder, galaxy images, spectral reconstruction, redshift regression, missing modality learning

## TL;DR

A multi-modal image–spectrum–redshift dataset (GalaxiesML-Spectra) comprising 134,533 galaxies is constructed and adapted for a Multi-Modal Masked Autoencoder (MMAE) that performs joint reconstruction of images and spectra alongside redshift regression. Experiments demonstrate that, even when spectra are entirely absent at test time, using only 25% masked images achieves a redshift prediction scatter of $\sigma_{NMAD} = 0.016$, outperforming AstroCLIP.

## Background & Motivation

**Background**: Next-generation astronomical surveys (LSST, Euclid, etc.) will image billions of galaxies, yet acquiring spectra takes roughly 100× longer than imaging. Redshift $z$, which quantifies spectral shifts caused by cosmic expansion, is a fundamental cosmological observable, but precise redshift measurements require spectroscopy. Existing ML approaches include CNN/MLP-based photometric redshift estimation, AstroMAE (single-modality image MAE), and AstroCLIP (contrastive joint embedding of images and spectra).

**Limitations of Prior Work**: (1) Most methods model only a single modality and cannot learn cross-modal associations. (2) AstroCLIP performs contrastive alignment without reconstruction and is validated only at low redshift $z < 0.5$. (3) MAE has not been explored in astronomical multi-modal settings.

**Key Challenge**: Upcoming surveys will generate massive image archives with virtually no accompanying spectra. A method is needed that can learn physically meaningful representations from images that are associated with spectral information. The MAE training objective of "recovering the whole from partial observations" naturally simulates the spectral-missing scenario.

**Goal**: (1) Construct a large-scale astronomical dataset combining images, spectra, and redshifts. (2) Validate the feasibility of MMAE for astronomical multi-modal reconstruction and redshift regression. (3) Evaluate model performance when spectra are entirely absent.

**Key Insight**: The MultiMAE framework is adopted to unify 5-band galaxy images and 1D spectra as patch tokens. Joint reconstruction is trained under 75% masking, with an integrated redshift regression head. During training, 50% of samples have their spectra fully masked to simulate realistic survey conditions.

**Core Idea**: A multi-modal masked autoencoder is used to learn a shared representation of galaxy images and spectra, enabling redshift prediction at test time without any spectral input.

## Method

### Overall Architecture

Input: 5-band images $(64\times64\times5)$ + 1D spectra (259 pixels) → patchified separately → 75% random masking → independently encoded by Transformers → cross-attention fusion → attention pooling to obtain joint representation → three task heads: image decoder, spectrum decoder, redshift regressor.

### Key Designs

1. **Dual-Modality Patch Tokenization + Independent Encoding**:

    - Function: Unify images and spectra as token sequences and extract intra-modal features independently.
    - Mechanism: Images are patchified via $8\times8\times5$ 2D convolution and projected to 256 dimensions with learnable 2D positional encodings. Spectra are split into 1D patches of length 8 and linearly projected to the same dimension. Each modality is encoded by an independent 1D Transformer (depth 4, 8 heads, dropout 0.1). 75% of tokens are masked.
    - Design Motivation: Independent encoding allows each modality to learn its own intra-modal structure. The high masking rate of 75% forces the model to learn strong representations rather than memorizing inputs.

2. **Cross-Attention Fusion**:

    - Function: Establish information flow between modalities, enabling spectra to inform image morphology understanding and images to guide spectral inference.
    - Mechanism: In 4 cross-attention layers, image tokens serve as queries over spectral tokens and vice versa. After fusion, attention pooling generates global image and spectral embeddings, which are concatenated into a joint representation.
    - Design Motivation: Cross-modal attention enables the model to learn physical associations such as "emission line positions imply galaxy type" and "galaxy morphology implies redshift range."

3. **Joint Training Objective (Reconstruction + Regression)**:

    - Function: Simultaneously optimize reconstruction and redshift prediction so that learned representations are both detail-rich and semantically meaningful.
    - Mechanism: The loss is a weighted sum $\mathcal{L} = 0.1 \cdot \mathcal{L}_{img} + 0.01 \cdot \mathcal{L}_{spec} + 1.0 \cdot \mathcal{L}_z$. Reconstruction uses MSE over masked regions only. The redshift loss is $\mathcal{L}_z = 1 - 1/(1+(dz/0.15)^2)$, where $dz = (z_{pred}-z_{spec})/(1+z_{spec})$. During training, 50% of samples have their spectra fully zeroed out.
    - Design Motivation: Embedding redshift regression directly into MAE training (rather than conventional pre-train-then-finetune) guides the encoder to extract physically relevant features during reconstruction. The 50% spectral masking simulates realistic missing-modality scenarios.

### Loss & Training

AdamW (weight decay 0.01, lr 0.0001) with gradient clipping. The dataset is split 70/15/15 into train/validation/test (~94k/20k/20k). Spectral preprocessing: normalization + downsampling to 259 pixels.

## Key Experimental Results

### Main Results

| Method | Test Condition | Redshift Range | $\sigma_{NMAD}$ |
|--------|---------------|----------------|----------------|
| MMAE (25% img mask, 100% spec mask) | Image only | $z \lesssim 0.4$ | **0.016** |
| MMAE (0% img mask, 100% spec mask) | Image only | $z \lesssim 0.4$ | 0.026 |
| AstroCLIP | Image + Spectrum | $z \lesssim 0.4$ | 0.020 |
| Fine-tuned BCNN | Image only | $z \lesssim 0.4$ | 0.012 |

### Ablation Study

| Reconstruction Target | Captured | Limitations |
|----------------------|----------|-------------|
| Image reconstruction | Galaxy shape/color ✓ | Nearby galaxy details / background noise ✗ |
| Spectral reconstruction | Continuum shape ✓, H-α/Ly-α positions ✓ | Line width severely overestimated, line strength underestimated |
| Redshift regression | Accurate for $z<1$ | Degrades for $z>1$, staircase artifacts |

### Key Findings

- **25% image masking outperforms no masking**: $\sigma_{NMAD}$ decreases from 0.026 to 0.016. Mild masking acts as regularization, preventing overfitting to small-scale features. This differs from standard MAE behavior where high masking rates are optimal, likely because astronomical images have lower information density.
- **Physical features in spectral reconstruction**: The model learns that "a given emission line should appear at a specific redshift" (e.g., H-α position error of 24Å), but severely overestimates line width by 15× (34.5Å → 528Å) and underestimates line strength by 5×. Line ratios, an important physical diagnostic, fail completely.
- **Staircase structure in redshift predictions**: Steps correspond to redshift intervals where strong spectral lines enter or leave the spectrograph range (e.g., Lyman-α at $z\sim2$), indicating that the model is highly sensitive to the visibility of specific lines.
- A performance gap remains relative to BCNN ($\sigma_{NMAD}=0.012$): the Inception-style CNN is more robust for the redshift regression task.

## Highlights & Insights

- **Natural alignment between MAE training and missing-modality surveys**: Random masking during training (50% fully masked spectra) directly simulates the unavailability of spectra in real surveys. The design philosophy of "aligning training strategy with deployment scenarios" generalizes to any missing-modality setting.
- **Masking as regularization**: In astronomical images with relatively low information density, 25% masking improves performance. The optimal masking rate should be calibrated to the information density of the data.
- This work is the first to use a single framework for simultaneous multi-modal reconstruction and regression in astronomy, extending the redshift range to $z\sim4$ (far beyond AstroCLIP's $z\lesssim0.5$).
- The GalaxiesML-Spectra dataset (134k galaxies, HSC images + DESI spectra) constitutes an independent contribution.

## Limitations & Future Work

- **Poor emission line reconstruction quality**: Line widths and strengths cannot be accurately recovered, and line ratios fail entirely. Physically motivated constraints are needed, such as parameterized spectral line losses or auxiliary line detection objectives.
- **Performance gap relative to CNN baseline**: Transformers underperform Inception-style CNNs on the redshift task at this data scale; larger datasets or deeper models are required.
- **Insufficient high-redshift data**: GalaxiesML is biased toward low redshift and high luminosity, limiting generalization at high redshift. Supplementing with high-redshift sources from the DESI Legacy Imaging Surveys is needed.
- MSE reconstruction loss assigns insufficient weight to sharp spectral line peaks; weighted MSE or perceptual losses should be considered.
- The model is small (depth 4, embedding dimension 256); no scaling ablation is performed.
- Transferability of learned representations to other downstream tasks (morphological classification, star formation rate estimation) is not evaluated.

## Related Work & Insights

- **vs. AstroCLIP**: AstroCLIP aligns images and spectra via contrastive learning without reconstruction. The proposed MMAE performs both reconstruction and regression simultaneously, achieving lower redshift scatter over the same redshift range (0.016 vs. 0.020), though the comparison is not entirely fair.
- **vs. AstroMAE**: AstroMAE applies MAE to single-modality images only. This work extends the framework to multi-modal inputs with spectral reconstruction.
- **vs. BCNN**: BCNN achieves superior $\sigma_{NMAD}=0.012$ but is specifically fine-tuned for redshift. The advantage of MMAE lies in learning general representations that are extensible to additional modalities and tasks.
- The framework naturally extends to additional modalities such as textual metadata and multi-epoch observations.

## Rating

- Novelty: ⭐⭐⭐ — The MMAE framework itself is not new, but its application to astronomical multi-modal data is a first.
- Experimental Thoroughness: ⭐⭐⭐ — Dataset construction is solid, but ablation studies lack depth.
- Writing Quality: ⭐⭐⭐ — Structure is clear; analysis of some results is shallow.
- Value: ⭐⭐⭐ — Cross-domain application that establishes feasibility, though findings remain limited.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[NeurIPS 2025\] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)

<!-- RELATED:END -->
