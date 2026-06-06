---
title: >-
  [Paper Note] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction
description: >-
  [NeurIPS 2025][Remote Sensing][Ionospheric prediction] This paper constructs an open, ML-ready ionospheric prediction dataset that integrates 8 heterogeneous data sources (solar observations, geomagnetic indices…
tags:
  - "NeurIPS 2025"
  - "Remote Sensing"
  - "Ionospheric prediction"
  - "dataset"
  - "solar activity"
  - "time series forecasting"
  - "space weather"
date: 2026-05-08
content_hash: 336f34ee81662c88
---

# Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2511.15743](https://arxiv.org/abs/2511.15743)  
**Code**: N/A  
**Area**: Time Series / Space Weather
**Keywords**: Ionospheric prediction, dataset, solar activity, time series forecasting, space weather

## TL;DR

This paper constructs an open, ML-ready ionospheric prediction dataset that integrates 8 heterogeneous data sources (solar observations, geomagnetic indices, TEC maps, etc.) spanning approximately 14 years (2010–2024). Three spatiotemporal baseline models—LSTM, SFNO, and GraphCast—are trained on this dataset, achieving TEC forecasts with lead times up to 12 hours.

## Background & Motivation

- **Importance of ionospheric prediction**: Modern society critically depends on GNSS navigation, LEO satellite communications, aviation networks, and power grids. Solar events such as solar flares and coronal mass ejections directly perturb the ionosphere, causing degraded GNSS accuracy, radio communication outages, and even power grid failures.
- **Pain points of data fragmentation**:
    - Ionospheric observations originate from diverse platforms (satellites, ground stations, smartphone crowdsourcing) with widely varying formats, temporal resolutions, and spatial coverage.
    - Existing data products are not designed for machine learning; missing value representations are inconsistent (e.g., different sentinel values across OMNI dataset channels), requiring extensive preprocessing.
    - The absence of standardized ML-ready datasets prevents systematic comparison across models.
- **Paper's positioning**: As part of the 2025 NASA Heliolab collaboration, this work constructs the first unified dataset that aligns sparse/dense TEC maps with solar and geomagnetic driver data, filling a critical gap in the field.

## Method

### Overall Architecture

The core design philosophy of the dataset is *heterogeneous alignment*: eight data sources with native temporal resolutions ranging from 15 seconds to daily cadence are aligned to a common time axis (2010-05-13 to 2024-08-01) and stored in a modular structure. The final product supports multi-resolution temporal queries, includes a built-in PyTorch Dataset class and normalization scheme, and is directly usable for model training.

### Key Designs

1. **Data collection and fusion**: Eight heterogeneous data sources are integrated, covering the complete solar–terrestrial coupling chain.

    | Data Source | Features | Native Frequency | Time Range |
    |--------|------|----------|----------|
    | OMNI2 (NASA) | AU/AL/AE, SYM-H, IMF Bx/By/Bz, solar wind velocity | 1 min | 2010.05–2024.08 |
    | NOAA/GFZ Kp | Ap, Kp geomagnetic indices | 3 h | 1997–2025 |
    | JPL Dense TEC | 1°×1° global TEC grid | 15 min | 2010.05–2024.07 |
    | Madrigal Sparse TEC | 1°×1° GNSS sparse TEC | 5 min | 2010–2024 |
    | SDO-FM | EUV irradiance embeddings (NVAE) | 15 s | 2010.05–2024.08 |
    | Space Env. Tech. | F10.7/S10.7/M10.7/Y10.7 solar flux | Daily | 1997–2025 |
    | Orbital mechanics | Solar/lunar zenith angles, Sun–Earth distance | Variable | Variable |
    | Quasi-dipole coordinates | Georeference coordinates for magnetic field projection | Yearly | 2010–2024 |

2. **Missing value handling and temporal alignment**:
    - All missing values are unified as NaN, resolving inconsistent sentinel values across sources such as OMNI.
    - Forward-filling is applied to handle short gaps, with a maximum rewind time defined per data stream.
    - For most streams, the maximum rewind time equals the native sampling frequency; OMNI is an exception at 50 minutes.
    - Gaps exceeding the rewind time are skipped entirely to prevent propagation of stale data.
    - The same forward-filling logic also serves as the interpolation strategy for resampling all features to a unified frequency.

3. **Geomagnetic storm event catalog**:
    - Geomagnetic activity levels are classified based on Kp thresholds following the NOAA G-level standard.

    | Event ID Format | NOAA G-level (Kp range) | Description |
    |-----------|----------------------|------|
    | G0Hℓ | Kp < 5 | Quiet period |
    | G1Hℓ | 5 ≤ Kp < 6 | Minor storm |
    | G2Hℓ | 6 ≤ Kp < 7 | Moderate storm |
    | G3Hℓ | 7 ≤ Kp < 8 | Strong storm |
    | G4Hℓ | 8 ≤ Kp < 9 | Severe storm |
    | G5Hℓ | Kp ≥ 9 | Extreme storm |

    - Event IDs encode G-level + "H" + duration ℓ (in hours); e.g., G2H6 denotes an event reaching G2 level for at least 6 consecutive hours.
    - This catalog is used to ensure that the same storm event is not split across training and validation sets, thereby preventing data leakage.

### Loss & Training

- Training data uses the 15-minute aligned data product, with JPL Dense TEC as the prediction target.
- The codebase provides a built-in PyTorch Dataset class supporting user-specified time ranges and dataset-specific normalization schemes.
- Models perform autoregressive forecasting with lead times up to 12 hours.
- Data splits are performed based on the event catalog to ensure coverage of both geomagnetically quiet and active periods.

## Key Experimental Results

### Main Results

Three IonCast baseline models are trained for global TEC prediction:

| Model | Architecture | Characteristics |
|------|----------|------|
| IonCast-LSTM | LSTM | Classic sequential modeling baseline |
| IonCast-SFNO | Spherical Fourier Neural Operator | Spectral modeling on the sphere, adapted to Earth's curved surface |
| IonCast-GraphCast | GraphCast | Graph network architecture inspired by recent advances in weather prediction |

- All models outperform the persistence forecast baseline.
- Strong performance is demonstrated across lead times up to 12 hours.
- Models are evaluated under both geomagnetically quiet and active conditions.

### Ablation Study

As a dataset paper, no traditional ablation experiments are conducted. However, the following design choices are left flexible for future research:
- The maximum rewind time for forward-filling is user-adjustable.
- Temporal frequency is selectable (ranging from 15 seconds to daily).
- Data sources can be modularly added or removed.

### Key Findings

- After aligning heterogeneous data to a unified time axis, all three architectures successfully train and surpass the baseline, validating the dataset's practical utility.
- The physics-informed event catalog-based data splitting effectively prevents data leakage, ensuring reliable evaluation on rare geomagnetic storm events.
- The dataset spans approximately 14 years (2010–2024), covering a complete solar cycle (Cycles 24–25), providing sufficient data for long-term trend analysis.

## Highlights & Insights

- **First comprehensive solar–ionosphere ML dataset**: Fills the gap in the complete data chain from solar surface observations to ionospheric responses, unifying SDO EUV embeddings, solar wind parameters, geomagnetic indices, and TEC maps into a single data product.
- **Strong engineering value**: Provides a complete open-source pipeline (GitHub + public Google Cloud storage bucket), including data alignment, preprocessing, PyTorch data loading, and model training examples, substantially lowering the barrier to entry in this field.
- **Elegantly designed event catalog**: The MESTICI classification scheme jointly encodes storm intensity (G-level) and duration, more richly characterizing geomagnetic events than simple Kp thresholds alone, and provides physically meaningful data splits for model evaluation.
- **Crowdsourced data integration**: Incorporating TEC measurements from Android smartphones into the sparse TEC source demonstrates the potential for low-cost sensor extension.

## Limitations & Future Work

- **No quantitative metrics reported**: As a NeurIPS ML4PS Workshop paper, model performance is described only qualitatively as "outperforming the persistence baseline," without specific RMSE/MAE values or comprehensive cross-model comparisons.
- **Simplistic missing value handling**: Forward-filling may be insufficient for data streams with rapidly varying characteristics (e.g., sudden solar wind changes); physics-constrained interpolation or learned imputation methods could be explored.
- **Limited spatial resolution**: The 1°×1° TEC grid has limited capacity to capture regional ionospheric disturbances such as traveling ionospheric disturbances (TIDs).
- **Black-box nature of SDO embeddings**: NVAE-compressed EUV irradiance embeddings discard original spectral details, and embedding quality depends on the performance of the pretrained model.
- **Temporal coverage constraints**: Although spanning ~14 years, extreme geomagnetic events (G4/G5 level) are extremely rare in the record, limiting the ability to fully validate model generalization under extreme conditions.

## Related Work & Insights

- **JPL GIM-TEC**: Provides dense global TEC maps but is a single data product unaligned with driver data.
- **Madrigal/MIT Haystack**: Provides sparse TEC and plasma parameters, requiring additional processing before ML use.
- **SDO Foundation Model** (Walsh et al., 2024): Compresses full-disk solar observations from SDO into low-dimensional embeddings; this paper directly uses those embeddings as input features.
- **GraphCast** (Lam et al., 2023): Originally designed for medium-range weather forecasting; this paper adapts its graph network architecture to spherical ionospheric prediction.
- **FourCastNet/SFNO** (Bonev et al., 2025): Spherical Fourier Neural Operator, naturally suited for global field prediction on the sphere.
- **Takeaway**: The modular design philosophy of this dataset is applicable to other multi-source time series fusion scenarios (e.g., ocean monitoring, climate modeling). The event catalog approach to preventing data leakage is worth adopting broadly in all event-driven forecasting tasks.

## Rating

- **Novelty**: ⭐⭐⭐ — Dataset construction rather than methodological innovation, but the positioning as the "first ML-ready ionospheric dataset" carries unique value.
- **Technical Depth**: ⭐⭐⭐ — Data engineering details are thorough (missing value handling, temporal alignment strategies), though the modeling component is relatively thin.
- **Experimental Thoroughness**: ⭐⭐ — Lacks quantitative experimental results; only qualitative descriptions are provided.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, with detailed tables of data sources and event classification schemes.
- **Value**: ⭐⭐⭐⭐ — The complete pipeline of open data + code + PyTorch interface offers strong utility for the space weather ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)
- [\[NeurIPS 2025\] C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction](c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[AAAI 2026\] Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](../../AAAI2026/remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
