---
title: >-
  [Paper Note] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images
description: >-
  [ICCV 2025][Remote Sensing][Remote sensing foundation model] This paper proposes SMARTIES, a unified sensor-agnostic foundation model for remote sensing that maps heterogeneous sensor data into a shared space via spectru…
tags:
  - "ICCV 2025"
  - "Remote Sensing"
  - "Remote sensing foundation model"
  - "multi-sensor"
  - "spectrum-aware"
  - "masked autoencoder"
  - "sensor-agnostic representation"
date: 2026-05-08
content_hash: 0bb4270fd4392b33
---

# SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images

**Conference**: ICCV 2025
**arXiv**: [2506.19585](https://arxiv.org/abs/2506.19585)  
**Code**: [https://gsumbul.github.io/SMARTIES](https://gsumbul.github.io/SMARTIES)  
**Area**: Remote Sensing
**Keywords**: Remote sensing foundation model, multi-sensor, spectrum-aware, masked autoencoder, sensor-agnostic representation

## TL;DR

This paper proposes SMARTIES, a unified sensor-agnostic foundation model for remote sensing that maps heterogeneous sensor data into a shared space via spectrum-aware projection. Combined with cross-sensor token mixing and masked reconstruction for self-supervised pre-training, SMARTIES surpasses sensor-specific models on both unimodal and multimodal tasks and generalizes to sensors unseen during pre-training.

## Background & Motivation

Remote sensing data originates from diverse sensors (optical, SAR, thermal infrared, etc.) with substantial variation in spectral coverage, radiometric resolution, and spatial resolution. Existing deep learning models face the following challenges:

**Sensor-specific models**: Trained independently for each sensor, precluding cross-sensor transfer.

**Limitations of multi-sensor foundation models**:
   - Dual/tri-modal models (e.g., CROMA, SkySense) employ sensor-specific encoders; adding new sensors requires architectural modifications and incurs high computational overhead.
   - Dynamic weight methods (e.g., DOFA) rely on hypernetworks and massive pre-training data (8 million images), limiting scalability.

**Generalization bottleneck**: Training on fixed sensor combinations introduces bias and prevents transfer to unseen sensors.

**Mechanism**: All remote sensing sensors fundamentally capture different subsets of the electromagnetic spectrum. A unified projection layer can be defined based on wavelength ranges to map different sensors into a shared spectrum-aware space.

## Method

### Overall Architecture

SMARTIES consists of four components:
1. Spectrum-aware image projection: projects data from different sensors into a shared space.
2. Cross-sensor token mixing: swaps tokens from different sensors to break sensor-specific bias.
3. Spectrum-aware image reconstruction: performs masked image modeling with a standard ViT encoder-decoder.
4. Downstream transfer to diverse sensors: includes interpolation-based adaptation for unseen sensors.

### Key Designs

1. **Spectrum-aware projection layer**: Seventeen projection layers $\mathcal{F} = \{f_1, ..., f_{17}\}$ are defined according to wavelength ranges, where $f_1$–$f_{12}$ correspond to the 12 bands of Sentinel-2, $f_{13}$–$f_{15}$ to Maxar RGB, and $f_{16}$–$f_{17}$ to Sentinel-1 SAR. Each projection layer $f_i: \mathbb{R}^{S \times S} \to \mathbb{R}^D$ is a fully connected layer. For each patch of a given sensor image, the corresponding projection layers for its bands are applied separately and then averaged, followed by multiplication by $C_{\text{max}}=12$ to prevent imbalance across sensors with different band counts. Adding a new sensor requires only the addition of new projection layers.

2. **Cross-sensor token mixing**: Given a co-registered image pair $(\mathbf{I}_a, \mathbf{I}_b)$ from different sensors over the same region, tokens are exchanged via a binary mask $\mathcal{M}$:
    $\mathbf{T}_{a'} = \mathcal{M} \odot \mathbf{T}_a + (1-\mathcal{M}) \odot \mathbf{T}_b$
   A mirrored mixing is applied simultaneously to retain all information. This prevents the model from developing bias toward specific spectral combinations.

3. **Interpolation-based transfer to unseen sensors**: For unseen bands of a new sensor, if the center wavelength $\lambda_n^c$ falls between the center wavelengths of two learned layers $f_i$ and $f_j$, the outputs of the two projection layers are combined via distance-weighted averaging. This approach is restricted to interpolation within the pre-trained spectral range and does not support extrapolation.

### Loss & Training

- Self-supervised MAE loss: MSE reconstruction loss computed over masked regions for both mixed patch sets:
  $$\mathcal{L} = \mathcal{L}_{a'} + \mathcal{L}_{b'}, \quad \mathcal{L}_{a'} = \frac{\sum(\mathbf{P}_{a'}^{\text{mask}} - \hat{\mathbf{P}}_{a'}^{\text{mask}})^2}{R \cdot N_W N_H}$$
- Pre-training uses only 496K images (60K fMoW RGB-S2 pairs + 188K BigEarthNet S1-S2 pairs), 300 epochs.
- ViT-B/L backbone, AdamW optimizer, batch size 2048, 8× A100 GPUs.
- Masking ratio 75%, mixing ratio 50%, input resolution 224×224.

## Key Experimental Results

### Main Results (Tables)

BigEarthNet multi-label scene classification (10% training data, mAP):

| Method | Backbone | S1 (LP) | S2 (FT) | MM (LP) |
|--------|----------|---------|---------|---------|
| SatMAE (S2) | ViT-B | 68.4 | 85.9 | 77.8 |
| SpectralGPT | ViT-B | 57.1 | 85.6 | 68.5 |
| CROMA | ViT-B×2 | 79.8 | 87.6 | 85.2 |
| **SMARTIES** | **ViT-B** | **78.9** | **86.9** | **85.4** |
| **SMARTIES** | **ViT-L** | **80.5** | **87.7** | **86.7** |

EuroSAT scene classification Top-1 accuracy:

| Method | LP | FT |
|--------|-----|-----|
| SatMAE (S2) ViT-B | 96.6 | 99.2 |
| CROMA ViT-B×2 | 97.6 | 99.2 |
| **SMARTIES ViT-B** | **98.4** | **99.4** |
| **SMARTIES ViT-L** | **98.9** | **99.6** |

Semantic segmentation on the PANGAEA benchmark (frozen backbone UPerNet, mIoU):

| Method | BurnScars | DEN | SpaceNet7 |
|--------|-----------|-----|-----------|
| CROMA | 81.8 | 38.3 | 59.9 |
| DOFA | 80.6 | 39.3 | 61.8 |
| **SMARTIES** | **82.8** | **38.5** | **62.2** |

### Ablation Study (Tables)

Cross-sensor token mixing ablation (EuroSAT kNN, 50-epoch pre-training):

| Setting | kNN Accuracy |
|---------|-------------|
| No mixing | 91.0 |
| Mixing (BEN only) | 91.1 |
| Mixing (full) | **93.2** |

Multimodal fusion strategies (BEN-MM LP, mAP):

| Strategy | 1% data | 10% data |
|----------|---------|----------|
| Image Stacking | 75.9 | 83.1 |
| Feature Concatenation | 77.0 | 84.7 |
| **Mixup Concatenation** | **79.2** | **86.7** |

### Key Findings

- **Single model surpasses sensor-specific models**: SMARTIES with a single ViT-B simultaneously outperforms dedicated models on both SAR (S1) and optical (S2) tasks.
- **Data efficiency**: Pre-training requires only 496K images, far fewer than DOFA (8 million) and CROMA (large-scale data).
- **Generalization to unseen sensors**: For Landsat-8 thermal infrared bands (unseen during pre-training), projection interpolation with a frozen backbone achieves 50.2 mIoU, surpassing U-Net (47.7) and DeepLabV3+ (48.5) trained from scratch.
- **Multi-scale robustness**: Outperforms Scale-MAE and Cross-Scale MAE, which are specifically designed for multi-scale evaluation.
- Cross-sensor token mixing yields a multimodal fusion gain of +2.2% mAP.

## Highlights & Insights

- **Physics-grounded design philosophy**: The continuous nature of the electromagnetic spectrum and the physical correspondence of sensor bands are exploited to unify representations, providing a more principled basis than purely data-driven approaches.
- **Minimal yet effective**: Compared to complex architectures requiring hypernetworks (DOFA) or sensor-specific encoders (CROMA), SMARTIES adds only lightweight projection layers (+5.9M parameters), maintaining computational complexity comparable to vanilla MAE.
- **Zero-cost addition of new sensors**: Defining new projection layers corresponding to the target wavelength range requires no modification to the backbone architecture.
- **Cross-sensor token mixing** is conceptually simple yet effective, offering insights for other multi-modal learning scenarios.

## Limitations & Future Work

- Projection interpolation is restricted to the pre-trained spectral range (no extrapolation); bands such as X-band radar may require additional learning.
- Temporal modeling is not addressed; extension to tasks requiring multi-temporal analysis (e.g., change detection) remains future work.
- Only amplitude-related physical quantities are considered; SAR phase information (e.g., InSAR) is excluded.
- Independent fully connected projection layers are used per band, without exploiting spectral continuity across adjacent bands.
- Pre-training scenes are predominantly European (BigEarthNet); generalization to other geographic regions requires further validation.

## Related Work & Insights

- The MAE family (SatMAE, SpectralGPT, S2MAE) constitutes the mainstream SSL framework for remote sensing.
- CROMA and SkySense are representative multi-modal remote sensing foundation models but depend on sensor-specific encoders.
- DOFA dynamically generates weights via a hypernetwork but requires large-scale data and a complex architecture.
- The spectrum-aware projection design of SMARTIES offers inspiration for other multi-modal learning settings, such as multi-sequence MRI in medical imaging.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Spectrum-aware space and cross-sensor token mixing represent a significant paradigm advance for remote sensing foundation models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 10 datasets covering classification, segmentation, multi-scale, and unseen sensor scenarios.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with tight integration of design motivation and physical principles.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and scalable solution for unified multi-sensor modeling in remote sensing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)

</div>

<!-- RELATED:END -->
