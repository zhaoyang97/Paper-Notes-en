---
title: >-
  [Paper Note] Towards a Unified Copernicus Foundation Model for Earth Vision
description: >-
  [ICCV 2025][Remote Sensing][Earth observation foundation model] This work presents a unified Earth observation foundation model system covering all major Copernicus Sentinel tasks, comprising the Copernicus-Pretrain dataset with 18.7 million aligned images, the Copernicus-FM model supporting arbitrary spectral and non-spectral sensors, and the Copernicus-Bench evaluation benchmark spanning 15 hierarchical downstream tasks.
tags:
  - ICCV 2025
  - Remote Sensing
  - Earth observation foundation model
  - multimodal pretraining
  - Copernicus Sentinel
  - dynamic hypernetwork
  - atmospheric monitoring
date: 2026-05-08
content_hash: 675b473bc2340a80
---

# Towards a Unified Copernicus Foundation Model for Earth Vision

**Conference**: ICCV 2025
**arXiv**: [2503.11849](https://arxiv.org/abs/2503.11849)
**Code**: [GitHub](https://github.com/zhu-xlab/Copernicus-FM)
**Area**: Remote Sensing
**Keywords**: Earth observation foundation model, multimodal pretraining, Copernicus Sentinel, dynamic hypernetwork, atmospheric monitoring

## TL;DR

This work presents a unified Earth observation foundation model system covering all major Copernicus Sentinel tasks, comprising the Copernicus-Pretrain dataset with 18.7 million aligned images, the Copernicus-FM model supporting arbitrary spectral and non-spectral sensors, and the Copernicus-Bench evaluation benchmark spanning 15 hierarchical downstream tasks.

## Background & Motivation

The development of Earth observation (EO) foundation models faces three major bottlenecks:

**Insufficient sensor diversity**: Existing pretraining datasets predominantly focus on high- and medium-resolution sensors such as Sentinel-1/2 and Landsat, neglecting low-resolution but high-temporal-frequency atmospheric monitoring sensors such as Sentinel-3 and Sentinel-5P.

**Limited model flexibility**: Most models adopt rigid architectures tailored to specific sensor modalities, and cannot dynamically adapt to new spectral bands or non-spectral inputs (e.g., atmospheric composition data).

**Narrow evaluation scope**: Existing benchmarks primarily target surface-level applications with RGB/multispectral/SAR sensors, overlooking coarse-resolution sensors and atmospheric tasks.

These limitations hinder the development of general-purpose multimodal foundation models that integrate EO with weather and climate research. This paper aims to overcome these barriers through contributions across data, model, and benchmark dimensions.

## Method

### Overall Architecture

The project consists of three synergistic components: **Copernicus-Pretrain** (large-scale pretraining dataset) → **Copernicus-FM** (unified foundation model) → **Copernicus-Bench** (systematic evaluation benchmark). The model adopts a ViT backbone, processes multimodal inputs via a dynamic hypernetwork, and is trained with masked image modeling (MIM) combined with continual distillation.

### Key Designs

1. **Copernicus-Pretrain Dataset**: The globe is partitioned into approximately 310K grid cells following the $0.25° \times 0.25°$ grid of the ERA5 reanalysis dataset, covering eight Sentinel modalities:

    - Sentinel-1 GRD (SAR, 10 m, 264×264×2, ~4.2M images)
    - Sentinel-2 TOA (multispectral, 10 m, 264×264×13, ~4.2M images)
    - Sentinel-3 OLCI (multispectral, 300 m, 96×96×21, ~2.2M images)
    - Sentinel-5P (atmospheric variables: CO/NO2/SO2/O3, 1 km, 28×28, ~7.8M images)
    - Copernicus DEM (elevation, 30 m, 960×960, ~0.3M images)

    The dataset totals approximately 18.7 million images, constituting the largest and most diverse EO pretraining dataset to date. A Gaussian sampling strategy is applied to sample S1/S2 local patches around the top 10K most populous cities globally, while also covering polar regions.

2. **Sensor-Aware Hypernetwork with Dynamic Patch Embedding**: The key module for resolving differences in input size and channel count across modalities:

    - **Spectral hypernetwork**: The center wavelength $\lambda$ and bandwidth $\delta$ of each channel are mapped to a $D$-dimensional vector via Fourier encoding, and then passed through an MLP and multi-head attention to generate convolutional kernel weights $\mathbf{K}_{\text{conv}} \in \mathbb{R}^{D \times C \times p \times p}$.

    $\text{FE}(x) = [\cos \frac{2\pi x}{\omega_i}, \sin \frac{2\pi x}{\omega_i}], \quad \omega_i = \exp(\log \omega_{\min} + i \cdot \frac{\log \omega_{\max} - \log \omega_{\min}}{D/2-1})$

    - **Variable hypernetwork** (novel contribution): Non-spectral modalities (e.g., S5P atmospheric composition, DEM elevation) lack wavelength attributes. A frozen Llama 3.2 LLM is used to encode variable names into $D$-dimensional vectors, which are then passed through a similar MLP pipeline to generate the corresponding patch embedding weights. This is a one-time preprocessing step with zero additional inference cost.
    - **FlexiViT dynamic patch size**: The convolutional kernel patch size is dynamically adjusted according to the ground sampling distance (GSD), ranging from 10 m to 1 km (16×16 for S1/S2, 8×8 for S3, 4×4 for S5P, 64×64 for DEM).

3. **Unified Fourier Encoding with Metadata Integration**: In addition to positional encoding, three types of optional metadata encodings are introduced, all processed uniformly via Fourier encoding:

    - **Geolocation**: Latitude and longitude are encoded and concatenated as $\text{Loc} \in \mathbb{R}^D$
    - **Spatial coverage area**: Area encoding $\text{Area} \in \mathbb{R}^D$ computed from GSD and patch size
    - **Timestamp**: Day-offset encoding from a reference date $\text{Time} \in \mathbb{R}^D$
    - During training, metadata is randomly dropped with probability 0.7, with learnable tokens serving as substitutes for missing values.

### Loss & Training

- **Masked Image Modeling (MIM)**: MAE-style 70% masking ratio, with independent reconstruction of masked patches per modality.
- **Continual distillation**: DINOv2 and SoftCon are used as teachers to distill S2-RGB and S1/S2 representations, with loss weights of 0.1 and 0.2, respectively.
- ViT-Base backbone, 100 epochs, trained on a 220K full-modality grid subset.
- Data augmentation: random crop and scale with horizontal flipping.

## Key Experimental Results

### Main Results (Copernicus-Bench)

Representative results across 15 downstream tasks (frozen encoder evaluation):

| Task | Metric | Random | SoftCon | CROMA | DOFA | **Copernicus-FM** |
|------|--------|--------|---------|-------|------|-------------------|
| EuroSAT-S1 | OA↑ | 75.4 | 83.6 | 83.9 | 81.7 | **87.2** |
| EuroSAT-S2 | OA↑ | 92.5 | 96.7 | 97.0 | 97.2 | **97.9** |
| BigEarthNet-S1 | mAP↑ | 63.8 | **78.7** | 70.8 | 70.5 | 77.9 |
| LC100Cls-S3 | mAP↑ | 88.9 | - | - | 89.5 | **93.3** |
| LC100Seg-S3 | mIoU↑ | 18.2 | - | - | 16.5 | **24.1** |
| AQ-NO2-S5P | RMSE↓ | 3.4 | - | - | 3.3 | **2.8** |
| AQ-O3-S5P | RMSE↓ | 1741.6 | - | - | 1755.6 | **789.4** |

Improvements are particularly pronounced on S3 and S5P tasks (e.g., O3 prediction RMSE reduced from 1741.6 to 789.4). Copernicus-FM surpasses supervised training on 11 out of 15 tasks.

### Ablation Study

Ablation results with incremental component additions:

| Component | EuroSAT-S1 | EuroSAT-S2 | EuroSAT-RGB | LC100-S3 | AQ-O3-S5P |
|-----------|------------|------------|-------------|----------|-----------|
| Baseline (DOFA + dynamic patch) | 56.3 | 87.6 | 62.2 | 86.7 | 2218.0 |
| + Bandwidth Fourier encoding | 56.5 | 88.9 | 65.4 | 87.1 | 1710.7 |
| + Variable hypernetwork | 57.5 | 88.9 | 65.8 | 86.6 | 1598.1 |
| + Metadata encoding | **77.9** | 88.9 | **78.5** | **90.7** | **839.3** |
| + Continual distillation | **81.0** | **89.5** | **78.9** | 90.7 | **811.6** |

Metadata encoding yields the most substantial gains (EuroSAT-S1: 57.5→77.9, +20.4), particularly for non-optical modalities. This underscores the critical importance of metadata (e.g., geolocation) in remote sensing applications.

### Key Findings

- Cross-modal pretraining simultaneously improves performance on both surface and atmospheric applications.
- The contribution of metadata encoding substantially exceeds that of architectural improvements; geolocation is the most important among all metadata types.
- A metadata dropout probability of 0.7 is optimal (higher values encourage the model to not rely on metadata).
- Grid embeddings used for climate parameter prediction can complement the limitations of geographic coordinates (temperature prediction RMSE reduced from 3.99 to 1.98).
- LLM-encoded variable names provide meaningful initialization for non-spectral modalities.

## Highlights & Insights

- **Breaking traditional EO barriers**: This is the first work to unify surface and atmospheric observations within a single pretraining framework.
- **LLM-assisted variable hypernetwork**: The approach of encoding non-spectral variable names using a language model is elegant and concise, incurring zero additional cost.
- **ERA5-aligned data organization**: The gridded design naturally connects EO with weather and climate data, paving the way for cross-domain research.
- **Copernicus-Bench fills a critical gap**: 15 tasks cover three application tiers (preprocessing → basic applications → specialized applications), with 6 newly curated datasets.
- **Global grid embedding dataset**: Copernicus-Embed-025deg provides semantically rich geographic representations for climate modeling.

## Limitations & Future Work

- Coverage is limited to the Sentinel series; other important satellite sources such as Landsat and MODIS are not incorporated.
- The temporal span is approximately one year (around 2021), limiting long-term temporal sequence modeling.
- Native support for multimodal fusion and time-series processing is absent; modalities are currently encoded independently.
- The ViT-Base scale is constrained; the scalability to larger models (Large/Huge) remains to be verified.
- Some newly curated datasets in Copernicus-Bench are relatively small in scale (e.g., S5P tasks contain only ~1,500 samples).

## Related Work & Insights

- DOFA's wavelength-conditioned dynamic patch embedding serves as the foundation for the proposed model; this work extends it to bandwidth and non-spectral inputs.
- SatCLIP provides a baseline for location encoders, but Copernicus-FM's grid embeddings demonstrate superior performance on climate prediction tasks.
- MMEarth integrates multi-source data but with limited modality coverage; the full Sentinel coverage in this work is more systematic.
- The grid embeddings produced by this work can be directly used as static variable extensions in numerical weather prediction models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The framework unifying all Sentinel tasks represents a pioneering contribution to the EO field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 18.7M dataset, comprehensive ablations, 15-task benchmark, and climate application exploration.
- **Writing Quality**: ⭐⭐⭐⭐ The three-component structure is clearly organized, though the density of information requires consulting the appendix for certain details.
- **Value**: ⭐⭐⭐⭐⭐ The combined contribution of dataset, model, and benchmark represents a significant advancement for the EO community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[ICLR 2026\] AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](../../ICLR2026/remote_sensing/autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)
- [\[ICLR 2026\] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](../../ICLR2026/remote_sensing/earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)

<!-- RELATED:END -->
