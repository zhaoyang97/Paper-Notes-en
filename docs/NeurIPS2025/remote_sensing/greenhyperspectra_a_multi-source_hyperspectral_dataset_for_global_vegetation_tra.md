---
title: >-
  [Paper Note] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction
description: >-
  [NeurIPS 2025][Remote Sensing][Hyperspectral dataset] GreenHyperSpectra constructs a pretraining dataset of 140,000+ multi-source hyperspectral vegetation samples spanning proximal, airborne…
tags:
  - "NeurIPS 2025"
  - "Remote Sensing"
  - "Hyperspectral dataset"
  - "vegetation trait prediction"
  - "semi-supervised learning"
  - "cross-sensor generalization"
  - "masked autoencoder"
date: 2026-05-08
content_hash: 8a614c65630c1689
---

# GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2507.06806](https://arxiv.org/abs/2507.06806)  
**Code**: [https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra](https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra)  
**Area**: Remote Sensing / Self-Supervised Learning
**Keywords**: Hyperspectral dataset, vegetation trait prediction, semi-supervised learning, cross-sensor generalization, masked autoencoder

## TL;DR

GreenHyperSpectra constructs a pretraining dataset of 140,000+ multi-source hyperspectral vegetation samples spanning proximal, airborne, and satellite platforms. Label-efficient regression models trained via semi-supervised and self-supervised methods (MAE, GAN, RTM-AE) comprehensively outperform fully supervised baselines on 7 plant trait prediction tasks, with particularly pronounced advantages under label-scarce and out-of-distribution scenarios.

## Background & Motivation

**Background**: Plant functional traits (e.g., leaf mass per area, chlorophyll content, water content) are core variables for biodiversity assessment and ecosystem monitoring. Hyperspectral remote sensing can non-destructively predict these traits by measuring reflectance across hundreds of narrow spectral bands. Traditional approaches primarily rely on partial least squares regression (PLSR) or fully supervised deep learning methods to map hyperspectral data to multiple trait values.

**Limitations of Prior Work**: Annotating plant traits is extremely costly, requiring field sampling and laboratory analysis. Existing labeled datasets are geographically and ecologically limited (e.g., covering only temperate forests in the Northern Hemisphere), and sampling strategies and measurement protocols are inconsistent across studies. More critically, different hyperspectral sensors—ground spectrometers vs. airborne AVIRIS vs. satellite EnMAP—differ substantially in spectral resolution, spatial resolution, and observation geometry, causing severe domain shift and poor cross-sensor generalization in fully supervised models.

**Key Challenge**: Hyperspectral data itself is abundant (unlabeled data is easy to acquire), but annotations are scarce. Existing self/semi-supervised learning in hyperspectral remote sensing primarily targets classification tasks (e.g., land cover mapping), whereas trait prediction is a multi-output regression problem that must additionally handle input heterogeneity across sensors. No large-scale pretraining dataset or self/semi-supervised framework has been designed specifically for plant trait prediction.

**Goal**: (1) Construct a large-scale hyperspectral pretraining dataset spanning multiple sensors and ecosystems; (2) Establish a benchmark framework of semi/self-supervised methods for multi-output regression tasks; (3) Validate the advantages of pretraining under label-scarce and out-of-distribution settings.

**Key Insight**: Leverage 140,000+ unlabeled hyperspectral samples from diverse platforms (proximal/airborne/satellite), time periods (1992–2024), and ecosystems for pretraining, in order to learn spectrally robust, cross-domain representations.

**Core Idea**: Construct GreenHyperSpectra, a multi-source heterogeneous hyperspectral pretraining dataset, and combine it with self-supervised methods such as MAE to address the dual challenges of label scarcity and domain shift in plant trait prediction.

## Method

### Overall Architecture

The system consists of two stages: (1) self/semi-supervised pretraining on GreenHyperSpectra (140,000+ unlabeled samples) to learn general spectral representations; (2) fine-tuning on an annotated dataset (7,900 labeled samples from 50 experimental sources, covering 7 traits) for multi-output regression. Evaluation covers four scenarios: full-spectrum (400–2450 nm), half-spectrum (400–900 nm), in-distribution, and out-of-distribution.

### Key Designs

1. **GreenHyperSpectra Dataset Construction**:

    - Function: Provides cross-domain hyperspectral pretraining data.
    - Mechanism: Data are collected from three platform types—proximal spectrometers (ASD FieldSpec etc., <1 m resolution, 1–4 nm spectral resolution, 5,620 samples), airborne sensors (AVIRIS-NG, NEON AOP etc., 1–20 m resolution, 96,699 samples), and satellites (PRISMA, EnMAP, EMIT etc., 30–60 m resolution, 36,059 samples). All data are processed to surface reflectance level and spectrally resampled to a consistent band grid. The multi-platform composition introduces variability in spatial resolution, spectral resolution, sun-sensor geometry, and background conditions—precisely what single-platform datasets lack.
    - Design Motivation: Cross-sensor domain shift is the primary obstacle to generalizable trait prediction. By exposing models to multi-source data during pretraining, domain-invariant spectral features can be learned.

2. **Masked Autoencoder (MAE) for 1D Spectral Reconstruction**:

    - Function: A self-supervised pretraining strategy that learns spectral representations via masked reconstruction.
    - Mechanism: Hyperspectral data are divided into tokens (patches); a random subset is masked, and a Transformer encoder–decoder reconstructs the full spectrum. The reconstruction loss combines MSE and cosine similarity (weight $\alpha$), capturing both spectral magnitude and shape. After pretraining, the encoder is either frozen with a multi-output regression head for linear probing (MAE_LP) or fully fine-tuned (MAE_FT). For half-spectrum inputs, MAE's masking mechanism naturally supports variable-length input—a model pretrained on full-spectrum data can be directly transferred to half-spectrum data.
    - Design Motivation: By reconstructing masked tokens, MAE learns both local correlations and long-range dependencies in hyperspectral data, which serve as useful priors for downstream regression tasks.

3. **Physics-Constrained Autoencoder (RTM-AE)**:

    - Function: Uses a radiative transfer model (PROSAIL-PRO) as a non-learnable decoder to align the latent space with physical traits.
    - Mechanism: The encoder compresses spectra into a latent vector whose dimensions directly correspond to PROSAIL-PRO input parameters (chlorophyll content, LAI, etc.). PROSAIL-PRO then simulates reflectance, which is compared against the original spectrum. A learnable correction layer bridges the gap between simulated and real spectra, and a supervised loss on labeled samples is additionally applied.
    - Design Motivation: Incorporating physical priors makes the latent space interpretable (each dimension corresponds to a physical trait), while reconstruction learning on unlabeled data enhances representation robustness.

### Loss & Training

MAE uses MSE combined with weighted cosine similarity as the reconstruction loss. RTM-AE uses reconstruction loss plus supervised regression loss. SR-GAN uses adversarial loss, feature contrastive loss, and label regression loss. Labeled data are split 80/20 for training; 20 non-overlapping subsets of GreenHyperSpectra ensure consistent proportions of each data source across splits.

## Key Experimental Results

### Main Results (Full Spectrum, 400–2450 nm)

| Method | Mean R² (↑) | Mean nRMSE (↓) | Notes |
|--------|------------|----------------|-------|
| Supervised (fully supervised) | 0.587 | 13.697 | EfficientNet-B0 baseline |
| SR-GAN | 0.592 | 13.589 | Semi-supervised GAN |
| RTM-AE | 0.592 | 13.557 | Physics-constrained autoencoder |
| MAE_FR_LP (linear probing) | 0.466 | 15.499 | Frozen encoder + linear head |
| **MAE_FR_FT (fine-tuning)** | **0.641** | **12.777** | Full fine-tuning, best overall |

### Ablation Study (Label-Scarce Scenarios)

| Label Ratio | Supervised R² | MAE_FT R² | Gain |
|-------------|--------------|-----------|------|
| 20% | ~0.40 | ~0.55 | +37.5% |
| 40% | ~0.48 | ~0.58 | +20.8% |
| 60% | ~0.53 | ~0.61 | +15.1% |
| 100% | 0.587 | 0.641 | +9.2% |

### Key Findings
- **MAE fine-tuning is the clear optimal strategy**: Under the full-spectrum setting, MAE_FT achieves 9% higher R² and 6% lower nRMSE than the fully supervised baseline. However, linear probing (MAE_LP) performs worst, indicating that pretrained representations must be adapted through fine-tuning to capture the fine-grained spectral dependencies required for trait regression.
- **Advantage grows with fewer labels**: At only 20% labeled data, MAE_FT shows the largest relative gain over the baseline (+37.5%), confirming the core value of self-supervised pretraining under data scarcity.
- **Cross-spectrum transfer is effective**: The full-spectrum pretrained MAE applied directly to half-spectrum data (MAE_FR_HR_FT) achieves R²=0.566—far exceeding the half-spectrum-specific fully supervised baseline of 0.163—demonstrating that MAE learns transferable spectral priors.
- **Out-of-distribution generalization**: MAE_FT achieves R²=0.311 vs. 0.243 for the fully supervised baseline in leave-one-dataset-out cross-validation across 50 datasets, a 29% improvement.
- **Noise robustness**: Under additive Gaussian noise with σ=0.05, MAE_FT maintains R²=0.331, while the fully supervised baseline drops to −0.065.

## Highlights & Insights
- The **cross-platform dataset design philosophy** is the paper's most central contribution: rather than simply accumulating more data, the authors deliberately introduce sensor heterogeneity (proximal/airborne/satellite at three spatial scales, spanning different time periods and ecosystems), enabling the model to learn domain-invariant features during pretraining. The stratified sampling strategy that ensures consistent source proportions across splits is a critical engineering detail for experimental reliability.
- The **RTM-AE design using PROSAIL-PRO as a decoder** is particularly elegant: embedding a physical model within a deep learning framework yields a latent space that naturally corresponds to physical traits, balancing interpretability and predictive performance. Although its performance is slightly below MAE's, its physical grounding makes it more trustworthy in ecological applications.
- **MAE's natural support for variable-length input** is an underappreciated advantage: a model pretrained on full-spectrum data can be directly transferred to half-spectrum scenarios without repretraining, greatly enhancing the practical utility of a single pretrained model.

## Limitations & Future Work
- Increasing the pretraining data volume has limited impact on performance (Fig. 5), possibly because spectral variability within the current 140,000 samples is near saturation; data from additional underrepresented ecosystems and geographic regions would be needed.
- The poor performance of MAE_LP suggests a gap between pretrained representations and trait regression; introducing auxiliary trait prediction tasks during pretraining (e.g., weak supervision via PLSR-derived pseudo-labels) could help bridge this gap.
- The 50 sources of labeled data are predominantly concentrated in the Northern Hemisphere; coverage of biodiversity hotspots such as Africa and South America is insufficient, and OOD generalization experiments may overestimate real-world performance.
- More advanced self-supervised methods (e.g., DINOv2, I-JEPA) have not been explored for their applicability to 1D spectral data.

## Related Work & Insights
- **vs. HySpecNet/HyperSIGMA**: These hyperspectral benchmark datasets are based on a single sensor and include non-vegetation pixels, making them unsuitable for trait prediction tasks. GreenHyperSpectra specifically targets vegetation and spans multiple sensors.
- **vs. SpectralEarth**: SpectralEarth provides temporal series but is limited to a single sensor (EnMAP); GreenHyperSpectra holds a fundamental advantage in sensor diversity.
- **vs. Cherif et al. (2023) (fully supervised baseline)**: GreenHyperSpectra uses the same annotated dataset and EfficientNet-B0 architecture as a baseline; MAE_FT achieves a 9% R² improvement on top of this, demonstrating the value of pretraining.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-source hyperspectral pretraining dataset and self-supervised benchmark framework targeting plant trait prediction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage is exceptionally comprehensive, including full/half-spectrum, label sensitivity, noise robustness, OOD generalization, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction and experimental design are described in thorough detail.
- Value: ⭐⭐⭐⭐ Establishes an important benchmark at the intersection of remote sensing and plant science; data and code are publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[NeurIPS 2025\] C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction](c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)

</div>

<!-- RELATED:END -->
