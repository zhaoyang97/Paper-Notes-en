---
title: >-
  [Paper Note] Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models
description: >-
  [NeurIPS 2025 (Workshop on ML for Physical Sciences)][Remote Sensing][Ionospheric forecasting] As a product of the 2025 NASA Frontier Development Lab (FDL) Heliolab program, this paper presents the first comprehensive ML-ready dataset for ionospheric forecasting. It unifies seven categories of heterogeneous data sources — Solar Dynamics Observatory (SDO) extreme ultraviolet (EUV) irradiance embeddings, solar wind parameters, interplanetary magnetic field (IMF), geomagnetic activity indices, JPL dense TEC global ionospheric maps (GIMs), Madrigal sparse TEC, solar flux indices, and orbital mechanics parameters — into a consistent temporal-spatial structure. Building on this dataset, multiple spatiotemporal forecasting architectures are trained, including LSTM, Spherical Fourier Neural Operator (SFNO), and GraphCast, achieving autoregressive prediction of global vertical total electron content (vTEC) up to 12 hours ahead under both quiet and geomagnetically active conditions, surpassing the persistence baseline.
tags:
  - NeurIPS 2025 (Workshop on ML for Physical Sciences)
  - Remote Sensing
  - Ionospheric forecasting
  - global TEC prediction
  - multi-source heterogeneous data alignment
  - space weather ML dataset
  - spatiotemporal forecasting
date: 2026-05-08
content_hash: 7e698c26807425ec
---

# Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models

**Conference**: NeurIPS 2025 (Workshop on ML for Physical Sciences)
**arXiv**: [2511.15743](https://arxiv.org/abs/2511.15743)
**Code**: [GitHub](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere)
**Area**: Time Series / Space Weather / Remote Sensing
**Keywords**: Ionospheric forecasting, global TEC prediction, multi-source heterogeneous data alignment, space weather ML dataset, spatiotemporal forecasting

## TL;DR

As a product of the 2025 NASA Frontier Development Lab (FDL) Heliolab program, this paper presents the first comprehensive ML-ready dataset for ionospheric forecasting. It unifies seven categories of heterogeneous data sources — Solar Dynamics Observatory (SDO) extreme ultraviolet (EUV) irradiance embeddings, solar wind parameters, interplanetary magnetic field (IMF), geomagnetic activity indices, JPL dense TEC global ionospheric maps (GIMs), Madrigal sparse TEC, solar flux indices, and orbital mechanics parameters — into a consistent temporal-spatial structure. Building on this dataset, multiple spatiotemporal forecasting architectures are trained, including LSTM, Spherical Fourier Neural Operator (SFNO), and GraphCast, achieving autoregressive prediction of global vertical total electron content (vTEC) up to 12 hours ahead under both quiet and geomagnetically active conditions, surpassing the persistence baseline.

## Background & Motivation

**State of the Field**: Modern society relies heavily on complex technological infrastructure — global navigation satellite systems (GNSS) provide precise positioning for billions of devices, low Earth orbit (LEO) satellite constellations (e.g., Starlink) carry an increasing share of communications, aviation networks depend on GNSS for precision approach navigation, and power grids must guard against geomagnetically induced currents. All of these are highly vulnerable to space weather disturbances driven by solar activity. Solar flares, coronal mass ejections (CMEs), and energetic particle events not only threaten space operations but can also trigger geo-effective disturbances that directly impact terrestrial infrastructure. The ionosphere — the ionized layer of the atmosphere at approximately 60–1000 km altitude — is a critical link in the Sun–Earth coupling chain. Solar radiation and particle energy are transferred through the magnetosphere into the ionosphere–thermosphere system, causing dramatic variations in electron density that directly affect the propagation of radio signals passing through the ionosphere. The key physical quantity used to quantify this effect is Total Electron Content (TEC), which measures the total number of free electrons per unit area along the line-of-sight path from the ground to a satellite, expressed in TECU (1 TECU = $10^{16}$ electrons/m²). Accurate spatiotemporal prediction of TEC is essential for GNSS positioning error correction, radio communication link budgets, and space situational awareness.

**Limitations of Prior Work**: Over the past several decades, observational capabilities for monitoring near-Earth space have expanded substantially. NASA's SDO satellite continuously provides high-resolution solar EUV observations; the ACE and Wind spacecraft measure solar wind and IMF parameters at the Sun–Earth L1 point; hundreds of global GNSS ground stations provide extensive TEC measurements; and networks of geomagnetic observatories record indices such as Kp, AE, and SYM-H. However, these rich multi-source data face a fundamental challenge: they are scattered across disparate platforms (NASA GSFC's OMNIWeb, NOAA SWPC, JPL, MIT Haystack Observatory's Madrigal database, etc.), adopt widely different data formats and temporal resolutions (ranging from SDO's 15-second cadence to the 3-hour Kp index and daily solar flux values), and employ incompatible spatial grids. More critically, these data products were not originally designed for machine learning workflows — missing value encodings are inconsistent (e.g., the OMNI dataset uses different sentinel values for different channels), timestamp formats vary, and substantial domain expertise and preprocessing effort are required before data can be used directly in model training. This situation severely impedes ML model development and systematic comparison.

**Root Cause**: There is a fundamental gap between the scientific richness of multi-source heterogeneous observational data and the engineering requirements of machine learning pipelines for unified, clean, plug-and-play datasets. The causal chain from solar activity to ionospheric response is complex — solar EUV radiation directly ionizes the upper atmosphere, the IMF carried by the solar wind injects energy into the magnetosphere and ionosphere via magnetic reconnection, and geomagnetic indices reflect the intensity of this energy injection. Building predictive models capable of capturing this complete causal chain requires fusing data spanning different temporal scales and spatial domains into a coherent framework. The absence of a standardized ML-ready dataset forces researchers to independently perform laborious data preprocessing, resulting in limited comparability between models and constraining overall community progress.

**Paper Goals**: The goals of this paper decompose into three concrete sub-problems: (1) How can heterogeneous data from more than seven sources, covering multiple physical quantities with different temporal resolutions and spatial coverages, be systematically aligned into a unified temporal-spatial structure? (2) How should pervasive missing values, data gaps, and non-standard encodings across multi-source data be handled to yield a clean and reliable final product? (3) How can meaningful ML benchmarks be established on this dataset, covering diverse geomagnetic conditions while preventing data leakage?

**Starting Point**: The authors are members of the 2025 NASA Frontier Development Lab (FDL) Heliolab project, a research initiative jointly supported by NASA, Trillium Technologies, Google Cloud, NVIDIA, and Pasteur Labs. This context afforded direct access to all relevant data sources and domain expert support from institutions including JPL and NASA Goddard. The team's key observation is that, while the ionospheric modeling community possesses rich multimodal observational data, "connecting the dots" to form a complete Sun–Earth causal chain data product is a necessary prerequisite for unlocking the potential of data-driven methods.

**Core Idea**: By constructing a modular, spatiotemporally aligned, ML-ready data pipeline that unifies seven categories of heterogeneous data sources spanning the solar–solar wind–magnetosphere–ionosphere causal chain into a single data structure, this work provides standardized infrastructure for ML benchmarking of ionospheric forecasting.

## Method

### Overall Architecture

The methodological framework is an end-to-end dataset construction and benchmarking pipeline. The input consists of raw heterogeneous data streams from seven different platforms, including satellite remote sensing, ground-based networks, and online databases. These pass through stages of data acquisition, cleaning, missing value handling, temporal alignment, and spatial grid mapping. The output is a unified, temporally indexed, modular data product stored on Google Cloud Bucket, accompanied by open-source PyTorch data loading code. Building on this, the team trained a suite of spatiotemporal forecasting models called IonCast (LSTM baseline, SFNO, GraphCast), using JPL dense TEC GIMs as the prediction target, performing autoregressive global vTEC forecasting at 15-minute temporal resolution with a 12-hour forecast horizon.

The design philosophy of the entire pipeline is **modularity** and **reproducibility**: the processing logic for each data source is an independent module, allowing users to selectively use subsets of data sources; all processing parameters (e.g., the maximum rewind time for missing value filling) are user-configurable; and the complete code is publicly available on GitHub to ensure reproducibility.

### Key Designs

1. **Systematic Integration of Multi-Source Heterogeneous Data**

   - **Function**: Unify raw data streams from seven categories of data sources into a modular data structure indexed by time and aligned on a latitude–longitude grid.
   - **Mechanism**: Data integration follows the physical logic of the Sun–Earth causal chain — proceeding from the driver end (solar radiation, solar wind) to the response end (geomagnetic activity, ionospheric TEC). Specifically, the seven data source categories are: (a) **NASA/GSFC OMNI2 solar wind and magnetic field data** — obtained via OMNIWeb, containing geomagnetic indices (AU, AL, AE reflecting auroral electrojets; SYM-H reflecting the ring current; ASY-D reflecting partial ring current asymmetry) and solar wind/IMF parameters (IMF components $B_x$, $B_y$, $B_z$ and solar wind velocity components $v_x$, $v_y$, $v_z$), at 1-minute native resolution. (b) **NOAA SWPC/GFZ geomagnetic indices** — providing Kp and Ap indices at 3-hour resolution; Kp is a globally integrated quasi-logarithmic geomagnetic activity indicator and Ap is its linearized counterpart. (c) **JPL Dense TEC GIM** — global ionospheric maps produced operationally by the NASA Jet Propulsion Laboratory from global GNSS station networks, on a 1°×1° latitude–longitude grid at 15-minute temporal resolution in TECU; this is the primary prediction target. (d) **Madrigal Sparse TEC** — GNSS receiver-derived TEC maps hosted by MIT Haystack Observatory, also mapped to a 1°×1° grid but sparser, at 5-minute temporal resolution. (e) **SDO-FM EUV irradiance embeddings** — full-disk EUV observations from the Solar Dynamics Observatory compressed into low-dimensional embedding vectors via the Nouveau Variational Autoencoder (NVAE) approach, with a native cadence of 15 seconds — the fastest data stream in the dataset. (f) **Solar flux indices** — multi-wavelength solar flux values including F10.7, S10.7, M10.7, Y10.7 (in solar flux units), and the JB08 thermospheric heating rate dSt/dt (in K), at daily resolution. (g) **Orbital mechanics parameters** — derived features computed from the geometric positions of the Sun and Moon relative to Earth, including solar/lunar zenith angles, sub-solar/sub-lunar point coordinates, solar/lunar conjugate foot-point positions, and Sun–Earth and Moon–Earth distances. These geometric features are essential for capturing the diurnal variation and tidal modulation of TEC. The dataset also includes **Quasi-Dipole (QD) coordinates** — annual lookup tables mapping geomagnetic field lines to a geographic reference frame, which aid ionospheric modeling at high latitudes. The temporal coverage of all data sources is uniformly aligned to the period spanned by the SDO foundation model: May 13, 2010 to August 1, 2024.
   - **Design Motivation**: The physical drivers of the ionosphere are not singular but arise from the complete Sun–solar wind–magnetosphere–ionosphere coupling chain. Using a single data source (as most prior ML work has done, relying solely on TEC history for autoregressive prediction) cannot capture variations in external forcing. The value of this dataset lies in integrating the key observational variables from all critical links of the causal chain into a standardized format that can be directly "plugged into" ML training pipelines, enabling researchers to systematically explore which combinations of data sources are most valuable for TEC prediction under different conditions.

2. **Missing Value Handling and Temporal Alignment Strategy**

   - **Function**: Systematically handle pervasive missing values, data gaps, and non-standard encodings across multi-source data, and align data streams with different temporal resolutions to a unified time step.
   - **Mechanism**: Missing value handling adopts a three-tier strategy. The first tier is **standardized missing value encoding**: different data sources use different conventions to mark missing values (e.g., the OMNI dataset uses different sentinel numbers for different physical channels — values that appear numerically valid but indicate missingness), all of which are uniformly converted to NaN. The second tier is **removal of columns with large gaps**: if a feature channel exhibits persistent data gaps spanning multiple years, it is removed from the dataset entirely to avoid introducing excessive noise. The third tier is **forward-filling of small gaps**: for brief data interruptions, the most recent valid value is propagated forward. The key parameter is the **maximum rewind time**, which for most data streams is set equal to the native sampling interval (e.g., 3 hours for 3-hourly data), ensuring that only minor interruptions are filled. The sole exception is OMNI data, for which the rewind time is set to 50 minutes (native resolution: 1 minute) to accommodate the relatively frequent brief interruptions in that source. If a gap exceeds the rewind time threshold, the corresponding timestamp is skipped without filling, to avoid propagating stale data. This forward-fill logic also serves as a simple interpolation strategy for resampling all features to the standard time step (e.g., 15 minutes).
   - **Design Motivation**: Missing value handling is one of the most easily overlooked yet most critical aspects of multi-source data fusion. Without careful treatment, models may learn spurious patterns (e.g., treating sentinel values as physically meaningful) or suffer training instability due to widespread NaNs. The choice of forward-filling is a pragmatic compromise — it is simple, physically reasonable (assuming physical quantities do not change dramatically over short intervals), and introduces no future information. The introduction of the maximum rewind time parameter provides users with flexibility to balance data quality and completeness according to their specific application.

3. **Geomagnetic Storm Event Catalog and Data Splitting**

   - **Function**: Construct a physics-based geomagnetic storm event classification catalog (MESTICI scale) to guide train/validation/test set partitioning and prevent data leakage.
   - **Mechanism**: Based on the Kp index time series, simple threshold-based segmentation is applied to divide the entire 14-year span (2010–2024) into sub-intervals, each assigned an event identifier consisting of two components: the **NOAA G-level** (a geomagnetic disturbance classification based on Kp, ranging from G0-Calm ($K_p < 5$) to G5-Extreme ($K_p \ge 9$)) and the **duration** $\ell$ (in hours). For example, event ID "G2H6" denotes a geomagnetic event reaching G2 level (moderate storm, $6 \le K_p < 7$) lasting at least 6 hours. The NOAA G-level classification is: G0 ($K_p < 5$, quiet), G1 ($5 \le K_p < 6$, minor storm), G2 ($6 \le K_p < 7$, moderate storm), G3 ($7 \le K_p < 8$, strong storm), G4 ($8 \le K_p < 9$, severe storm), and G5 ($K_p \ge 9$, extreme storm). The primary use of this event catalog is to ensure that, when partitioning into train/validation/test sets, a single geomagnetic storm event is never split across different subsets — a particularly insidious form of data leakage in ionospheric forecasting, given that a major geomagnetic storm can persist for days with its onset, main phase, and recovery phase being physically highly correlated.
   - **Design Motivation**: Under conventional random temporal splitting, a model may accurately predict the peak phase of a storm on the test set because it has already "seen" the recovery phase of the same storm in the training set — such performance is spurious and does not reflect genuine generalization. Event-based splitting ensures that the model faces complete geomagnetic events it has never encountered during evaluation, more faithfully reflecting operational forecast performance. The inclusion of event duration adds granularity to the classification, enabling researchers to analyze model behavior as a function of storm intensity and duration.

### Loss & Training

As the primary contribution of this paper is the dataset rather than the models, training details are described concisely. All IonCast models are trained on the aligned data at 15-minute time steps, with JPL Dense TEC GIM as the prediction target. Training proceeds autoregressively: at each time step, the model predicts the next global TEC map on a 1°×1° grid, which is then fed as input for the subsequent step, enabling iterative forecasting up to 12 hours (48 autoregressive steps). The PyTorch dataset class allows users to specify temporal ranges for train/validation/test splits and provides normalization schemes for each data source. Dataset splitting is based on the geomagnetic storm event catalog to prevent intra-event leakage. The baseline is the persistence forecast (assuming TEC remains constant), and all IonCast models are required to surpass this baseline.

## Key Experimental Results

### Dataset Scale and Coverage Statistics

| Data Source | Feature Channels | Native Temporal Resolution | Temporal Span | Spatial Coverage / Format |
|---|---|---|---|---|
| NASA/GSFC OMNI2 | AU, AL, AE, SYM-D, SYM-H, ASY-D, $B_x$, $B_y$, $B_z$, $v_x$, $v_y$, $v_z$, etc. | 1 min | 2010.05–2024.08 | Point data (L1 point) |
| NOAA/GFZ Kp/Ap | Kp (dimensionless), Ap (nT) | 3 hours | 1997.01–2025.10 | Global composite index |
| JPL Dense TEC GIM | TEC (TECU), 1°×1° grid | 15 min | 2010.05–2024.07 | Global 180×360 grid |
| Madrigal Sparse TEC | TEC (TECU), 1°×1° grid | 5 min | 2010.01–2024.08 | Sparse global grid |
| SDO-FM EUV embeddings | Low-dimensional embedding (NVAE) | 15 sec | 2010.05–2024.08 | Solar full disk |
| Solar flux indices | F10.7, S10.7, M10.7, Y10.7 (sfu), JB08 dSt/dt (K) | Daily | 1997.01–2025.10 | Scalar time series |
| Orbital mechanics parameters | Solar/lunar zenith angles, sub-solar/sub-lunar point coordinates, Sun/Moon–Earth distances, etc. | Follows input features | Variable | Lat–lon grid |
| Quasi-Dipole coordinates | Latitude, longitude (QD reference frame) | Annual | 2010–2024 | Lat–lon grid |

### Baseline Model Performance Comparison

Detailed numerical results for the IonCast model suite are reported in a companion paper (Kelebek et al., NeurIPS 2025 Workshop). As a dataset paper, this work makes the following core claims:

| Model Architecture | Type | Prediction Target | Forecast Horizon | vs. Persistence Baseline |
|---|---|---|---|---|
| LSTM baseline | Sequence model | Global vTEC (15 min) | 12 hours | Outperforms |
| SFNO (Spherical Fourier Neural Operator) | Frequency-domain spatiotemporal model | Global vTEC (15 min) | 12 hours | Outperforms |
| GraphCast variant | Graph neural network | Global vTEC (15 min) | 12 hours | Outperforms |
| Persistence forecast | Baseline | Global vTEC (15 min) | — | Baseline |

### Ablation Study

As this paper is positioned primarily as a dataset contribution, detailed ablation results are deferred to the companion IonCast model paper. However, the following dimensions constitute implicit ablation axes from the perspective of dataset design:

| Data Combination | Expected Effect | Notes |
|---|---|---|
| TEC history only (autoregressive) | Base performance | No external forcing information |
| TEC + solar wind/IMF | Performance gain | Captures solar wind-driven ionospheric disturbances |
| TEC + geomagnetic indices | Performance gain | Reflects magnetosphere–ionosphere coupling |
| TEC + SDO EUV embeddings | Performance gain | Captures direct photoionization by solar EUV |
| All data sources combined | Best performance | Complete causal chain information |
| Quiet conditions ($K_p < 5$) | Better predictions | More regular physical processes |
| Active conditions ($K_p \ge 5$) | Degraded predictions | Enhanced nonlinear coupling, greater physical complexity |

### Key Findings

- **Geomagnetically active periods present greater prediction challenges**: During geomagnetic storms with $K_p \ge 5$, ionospheric variations are more intense and nonlinear, and TEC prediction errors increase substantially. This is consistent with physical expectations — storm-time processes such as high-latitude particle precipitation and convection electric field penetration to equatorial regions introduce additional complexity. Nevertheless, the inclusion of multi-source forcing data (particularly solar wind parameters and geomagnetic indices) helps improve predictions during storm periods.
- **The modular dataset design supports flexible experimentation**: Researchers can readily select different combinations of data sources to systematically investigate which physical drivers are most valuable for TEC forecasting under different conditions — something that was difficult to achieve with previously non-standardized data.
- **The event catalog is critical for preventing data leakage**: The Kp threshold-based MESTICI event classification scheme provides a principled train/test splitting strategy, ensuring that model evaluation reflects genuine generalization rather than memorization of different phases of the same storm event.
- **The potential of crowdsourced Android GNSS data**: The dataset incorporates sparse TEC data from Madrigal, a portion of which derives from GNSS receivers in Android smartphones. Although individual phone measurements are noisy, the crowdsourcing effect from a massive number of devices can substantially increase the spatial density of TEC observations, particularly in ocean regions and developing countries where traditional GNSS station networks are sparse.
- **SDO foundation model embeddings provide an efficient pathway for integrating satellite imagery**: Directly using full-disk SDO EUV images introduces extremely high data dimensionality, whereas compressing them into compact embedding vectors via the SDO Foundation Model (SDO-FM) substantially reduces computational burden while retaining key solar activity information.
- **The 14-year temporal span covers more than one complete solar cycle** (the declining phase of Cycle 24 and the ascending phase of Cycle 25), which is essential for training robust models capable of handling different levels of solar activity.

## Highlights & Insights

- **Full Sun–Earth causal chain data integration**: This is the most central contribution of the paper. Prior ionospheric ML work has largely relied on TEC history alone for autoregressive prediction, or at most incorporated one or two forcing indices. This paper systematically integrates all key observational quantities across the complete causal chain — from the solar surface to the ionospheric response — into a single "one-stop" data product ready for immediate use. This "data as infrastructure" paradigm is precisely what the ML for Science field currently needs most: many scientific domains do not lack observational data, but rather lack the systematic engineering effort to transform heterogeneous data into ML-ready formats.

- **Physics-based data splitting strategy (MESTICI event catalog)**: This is a particularly elegant design. In meteorological and space weather forecasting, data leakage is a frequently overlooked problem — temporally adjacent data points are physically highly correlated, and random splits almost inevitably introduce information leakage. By partitioning data according to the natural boundaries of geomagnetic events, the physical independence between training and test sets is ensured. This strategy generalizes naturally to other event-driven time series forecasting problems.

- **Introduction of the maximum rewind time parameter in the forward-fill mechanism**: This is an exceptionally pragmatic engineering design. Different data sources have different autocorrelation timescales — 1-minute resolution solar wind data can change dramatically within minutes (especially during interplanetary shock arrivals), while daily solar flux values change negligibly over a single day. By assigning different maximum rewind times to each data source, the pipeline preserves data integrity while avoiding inappropriate extrapolation.

- **Innovative inclusion of crowdsourced Android GNSS data**: Although this aspect receives limited discussion in the paper, incorporating consumer smartphone GNSS measurements into ionospheric datasets is a highly forward-looking concept. As the number of smartphones supporting raw GNSS observations continues to grow, this data source has the potential to fundamentally address the spatial coverage limitations of traditional ionospheric observing networks, particularly in regions with limited GNSS infrastructure.

- **SDO-FM embeddings as an efficient compression mechanism for solar observational data**: Directly incorporating full-disk EUV solar images (high-resolution 2D data) into ionospheric prediction pipelines would incur enormous computational and storage overhead. Using a pretrained solar foundation model (SDO-FM) to generate low-dimensional embeddings is an elegant solution — it retains critical physical information about solar activity while avoiding the curse of dimensionality. This approach generalizes to other time series forecasting tasks that require integrating high-dimensional remote sensing data.

## Limitations & Future Work

- **Limited benchmark depth as a workshop paper**: Although LSTM, SFNO, and GraphCast are mentioned, specific experimental designs, hyperparameter choices, training details, and detailed numerical comparisons are all deferred to the companion IonCast paper. The dataset paper itself lacks systematic ablation experiments across different data source combinations, making it impossible to quantitatively answer the central question of which data sources contribute most to TEC prediction.

- **Spatial resolution constrained by GNSS station network distribution**: Although JPL Dense TEC GIM provides global coverage, its 1°×1° grid resolution is obtained by interpolating sparse station data, and data quality is lower over oceans and high-latitude regions of the Southern Hemisphere. Madrigal Sparse TEC more directly reflects the uneven distribution of the GNSS station network. As a result, model prediction performance in these regions may be unreliable — a limitation that the dataset itself cannot resolve at the observational infrastructure level.

- **Simplistic missing value handling strategy**: Although forward-filling is practical, in scenarios where solar wind parameters change rapidly (e.g., parameters may jump by several times within 1–2 minutes during interplanetary shock arrivals), the 50-minute maximum rewind time may result in filled values that deviate substantially from true values. More advanced interpolation methods — such as physics-constrained interpolation or neural network-based imputation — could provide better data quality.

- **Absence of thermospheric and neutral atmosphere model parameters**: Ionospheric state is influenced not only by solar and geomagnetic forcing but also by the neutral density, temperature, and composition of the thermosphere (through collisional ionization and recombination processes). The dataset does not include outputs from empirical thermospheric models such as NRLMSISE-00 or JB2008, limiting the model's capacity to capture thermosphere–ionosphere coupling effects.

- **Prediction target limited to vertical TEC (vTEC)**: vTEC is a path-integrated quantity that discards information about the vertical structure of ionospheric electron density. For certain applications (e.g., precise point positioning (PPP) or radio occultation inversion), predictions of electron density profiles rather than column-integrated values are required. Future work could consider incorporating ionosonde data and GPS radio occultation (RO) data to support three-dimensional ionospheric modeling.

- **Temporal coverage spans ~1.5 solar cycles**: Solar Cycle 24 (approximately 2008–2019) was one of the weakest cycles in nearly a century, while Cycle 25 (from 2019 onward) exhibits substantially higher activity. This imbalance in solar activity levels may result in insufficient training data for high solar activity conditions. Extending temporal coverage to earlier solar cycles (e.g., by integrating pre-SDO legacy data) would increase the diversity of the dataset.

- **Future extensions**: These include incorporating ionospheric scintillation data to support GNSS signal quality forecasting, integrating COSMIC-2 constellation GPS RO data to provide three-dimensional electron density information, and leveraging side-view solar observations from the forthcoming ESA Vigil L5 spacecraft to improve CME and solar wind forecast lead times.

## Related Work & Insights

- **vs JPL GIM (Mannucci et al., 1998; Martire et al., 2024)**: JPL global ionospheric maps serve as one of the primary prediction targets in this dataset. The JPL GIM is itself a complex data product (estimating global TEC distribution from sparse GNSS observations via Kalman filtering and spherical harmonic analysis), but its output format (IONEX files) is not designed for ML workflows. The value of this dataset lies in aligning GIM TEC with other forcing data sources and providing a standardized PyTorch interface.

- **vs traditional physics-based models (IRI, NeQuick, SAMI3)**: Empirical and semi-empirical models such as the International Reference Ionosphere (IRI) and NeQuick use parameterized physical formulas to describe the ionosphere, while first-principles models such as SAMI3 numerically solve the ionospheric continuity and momentum equations. These models perform reasonably well at capturing mean ionospheric behavior but struggle with geomagnetically active periods. ML methods have the potential to complement physics-based models by learning from large observational datasets, and this dataset provides the necessary foundation for such efforts.

- **vs SDO Foundation Model (Walsh et al., 2024)**: SDO-FM is an upstream dependency for this dataset. SDO-FM uses an NVAE architecture to compress AIA and EVE instrument data from the SDO into compact embedding representations, intended to serve downstream solar physics tasks. This dataset demonstrates the applicability of SDO-FM embeddings to a novel downstream task — ionospheric forecasting.

- **vs GraphCast (Lam et al., 2023) and FourCastNet (Bonev et al., 2025)**: These are state-of-the-art ML architectures for numerical weather prediction, whose design concepts this work transfers to the space weather domain. GraphCast employs a graph neural network architecture to process gridded data on the sphere, while SFNO (a successor to FourCastNet) uses spherical Fourier transforms for efficient processing of global-scale physical fields. Applying these architectures to ionospheric forecasting is a natural extension, as the global distribution of TEC is likewise a physical field on the sphere.

- **vs existing ionospheric ML work**: The majority of prior work has used single data sources (e.g., TEC history alone or Kp/Dst indices alone), with inconsistent preprocessing and no standardized evaluation protocols. By providing a unified data product and benchmarking framework, this dataset makes systematic comparison between different models possible for the first time.

- **Insights**: This paper exemplifies an important trend in ML for Science — "datasets as papers." In many scientific disciplines, a high-quality ML-ready dataset may be as valuable as methodological innovation, as it lowers the barrier to entry, promotes comparability, and accelerates overall community progress. For other Earth science domains with multi-source heterogeneous observational data (e.g., oceanography, seismology, atmospheric chemistry), this paper provides a replicable paradigm for dataset construction: identify the complete physical causal chain → collect data from each link in the chain → unify spatiotemporal alignment → construct an event catalog to prevent leakage → provide open-source tools and benchmark models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first ionospheric dataset to systematically integrate multi-source heterogeneous data spanning the Sun–Earth causal chain into an ML-ready format; the MESTICI event catalog's anti-leakage design has practical significance.
- **Experimental Thoroughness**: ⭐⭐⭐ — The workshop paper format imposes clear limitations; detailed benchmark numerics are deferred to the companion paper, and ablation experiments across data source combinations are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The dataset description is systematic and well-organized; Table 1 summarizes data sources clearly and efficiently; the overall logical flow is coherent.
- **Value**: ⭐⭐⭐⭐ — Provides important infrastructure value for the space weather ML community; lays a data foundation for ionospheric digital twin construction; open-source data and code lower the barrier to entry.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)
- [\[NeurIPS 2025\] EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)
- [\[AAAI 2026\] Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](../../AAAI2026/remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)
- [\[AAAI 2026\] Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India](../../AAAI2026/remote_sensing/machine_learning_for_sustainable_rice_production_region-scale_monitoring_of_wate.md)
- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)

<!-- RELATED:END -->
