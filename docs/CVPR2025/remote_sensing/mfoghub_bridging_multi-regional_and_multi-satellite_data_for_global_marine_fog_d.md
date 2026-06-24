---
title: >-
  [Paper Note] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting
description: >-
  [CVPR 2025][Remote Sensing][Marine Fog Detection] MFogHub constructs the first multi-regional (15 coastal regions) and multi-satellite (6 geostationary satellites) global marine fog detection and forecasting dataset, containing over 68,000 high-resolution samples and 11,600+ pixel-level annotations. Extensive experiments on 16 baseline models reveal the influence of regional differences and satellite variations on model generalization.
tags:
  - "CVPR 2025"
  - "Remote Sensing"
  - "Marine Fog Detection"
  - "Marine Fog Forecasting"
  - "Multi-Regional and Multi-Satellite"
  - "Remote Sensing Dataset"
  - "Generalization Evaluation"
date: 2026-05-08
content_hash: 9e5238fe219341c3
---

# MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting

**Conference**: CVPR 2025  
**arXiv**: [2505.10281](https://arxiv.org/abs/2505.10281)  
**Code**: [https://github.com/kaka0910/MFogHub](https://github.com/kaka0910/MFogHub)  
**Area**: Remote Sensing  
**Keywords**: Marine Fog Detection, Marine Fog Forecasting, Multi-Regional and Multi-Satellite, Remote Sensing Dataset, Generalization Evaluation

## TL;DR
MFogHub constructs the first multi-regional (15 coastal regions) and multi-satellite (6 geostationary satellites) global marine fog detection and forecasting dataset, containing over 68,000 high-resolution samples and 11,600+ pixel-level annotations. Extensive experiments on 16 baseline models reveal the influence of regional differences and satellite variations on model generalization.

## Background & Motivation

1. **Background**: Marine fog is a complex marine meteorological phenomenon that reduces visibility to below 1 km, significantly impacting shipping, port operations, and coastal activities. Deep learning methods have outperformed traditional methods in marine fog detection and forecasting.
2. **Limitations of Prior Work**: (1) Existing datasets are almost exclusively limited to a single region (mostly the Yellow Sea and Bohai Sea) and a single satellite (mostly H8/9), making it impossible to evaluate model generalization under different conditions; (2) The scale of datasets is small (mostly < 5,000 samples), and even fewer are open-source; (3) Single-region data restricts the exploration of the intrinsic characteristics of marine fog formation and dissipation.
3. **Key Challenge**: While marine fog is a global phenomenon, research data remains highly localized. Marine fog spatial distribution patterns differ across regions (concentrated vs. dispersed), and spectral bands and imaging capabilities vary among satellites. High performance of models on single-source data may simply be due to overfitting.
4. **Goal**: (1) Construct a unified dataset covering global fog zones; (2) Support cross-regional and cross-satellite generalization evaluations; (3) Analyze the sensitivity of spectral bands to marine fog detection.
5. **Key Insight**: Statistical analysis of global marine fog frequency is conducted using 9.5 million records from ICOADS (International Comprehensive Ocean-Atmosphere Data Set) to select 15 high-frequency fog zones, collecting multispectral data from 6 satellites. A cube-stream data structure is proposed to uniformly organize spatiotemporal data.
6. **Core Idea**: By constructing the first global, multi-regional, and multi-satellite marine fog dataset, systematic evaluation of model generalization and research on cross-regional characteristics of marine fog are made possible.

## Method

### Overall Architecture
The construction pipeline of the MFogHub dataset is as follows: (1) Global marine fog frequency distribution is analyzed from 9.5 million ICOADS observation records to filter out 15 coastal high-frequency fog regions; (2) Multispectral L1 data from 6 geostationary satellites (FY4A, FY4B, GOES16, GOES17, H8/9, MeteoSat) are collected and unified to 1 km spatial resolution; (3) Data are organized into a cube-stream structure (timestamps $\times$ spectral bands $\times$ latitude $\times$ longitude), yielding 21 data streams; (4) Meteorological experts perform pixel-level annotations on 11,600+ samples. The dataset simultaneously supports both detection (semantic segmentation) and forecasting (spatiotemporal prediction) tasks.

### Key Designs

1. **Multi-Regional Data Collection and Selection Strategy**:

    - **Function**: Ensure the dataset covers major global fog zones, reflecting the regional diversity of marine fog.
    - **Mechanism**: Marine fog observation records from ICOADS (2015–2024) are extracted and accumulated into a $0.25^\circ \times 0.25^\circ$ global grid for frequency statistics. A sliding window of $12.8^\circ$ is used to scan global regions, combined with prior research to select 15 coastal regions with high shipping traffic and high fog frequencies. Analysis indicates that the spatial distribution of marine fog varies significantly across regions—e.g., concentrated distribution in the Baja California region versus dispersed distribution in the Gulf of Alaska region.
    - **Design Motivation**: Subjective selection alone is prone to omitting critical fog zones. A data-driven selection strategy ensures comprehensive and representative coverage. Regional difference analysis directly proves the necessity of cross-regional evaluation.

2. **Cube-stream Data Structure**:

    - **Function**: Uniformly organize spatiotemporal data from multiple regions and satellites, facilitating flexible slicing and deep learning integration.
    - **Mechanism**: Data for each region-satellite pair is organized as a cube-stream of $\mathbb{R}^{T \times C \times H \times W}$ (timestamps $\times$ spectral bands $\times$ latitude $\times$ longitude), with all region-satellite pairs constituting 21 cube-streams. Key attributes (region, satellite, time) can be customized for retrieval, supporting flexible slicing—e.g., slicing by spectral bands for sensitivity analysis, or by time for forecasting tasks. The minimum temporal interval is 30 minutes, spatial resolution is 1 km, and size is $1024 \times 1024$.
    - **Design Motivation**: Marine fog is a spatiotemporally dynamic process (formation $\rightarrow$ maintenance $\rightarrow$ dissipation), and continuous data is essential for forecasting tasks. The cube-stream structure preserves temporal continuity while supporting flexible multi-dimensional slicing, which is more suitable for sequential tasks than traditional independent image organization.

3. **Multi-Satellite Spectral Analysis and Annotation**:

    - **Function**: Reveal the varying impacts of different satellites and spectral bands on marine fog detection.
    - **Mechanism**: Spectral differences are analyzed using FY4A and H8/9 as examples—H8/9 is stronger in the infrared spectrum (16 bands), while FY4A covers more near-infrared bands (14 bands). The $0.65\,\mu\text{m}$ band of H8/9 exhibits a bimodal distribution, whereas FY4A exhibits a unimodal distribution. Marine fog exhibits two features across different bands: in visible light bands ($0.4\text{--}0.7\,\mu\text{m}$), fog areas show high brightness due to water droplet scattering; in infrared/water vapor bands, fog areas show low brightness due to weak absorption. Separability analysis reveals distinct separation in the pixel value distribution of fog/non-fog in certain bands, which benefits detection. Meteorological experts performed pixel-level annotations on 11,600+ samples.
    - **Design Motivation**: Different bands have different sensitivities to marine fog, and understanding these differences helps guide feature selection and model design. Data discrepancies when multiple satellites cover the same region highlight the challenges of cross-satellite generalization.

### Loss & Training
- Detection task: Standard semantic segmentation training (binary classification), evaluation metrics: CSI, recall, precision, mAcc, mIoU.
- Forecasting task: Standard spatiotemporal forecasting training (input $T$ frames to predict $T'$ frames), evaluation metrics: MSE, MAE, SSIM, PSNR.
- 8 detection baselines (DeepLabv3+, UNet, UNet++, ViT, DlinkViT, Unetformer, BANet, ABCNet) + 8 forecasting baselines (ConvLSTM, PredRNN, MIM, PhyDNet, SimVPv2, Uniformer, VAN, TAU).

## Key Experimental Results

### Main Results (Detection, Three GOES Satellite Sub-regions)

| Method | B.C. CSI↑ | C.C. CSI↑ | G.A. CSI↑ | Cross-Regional Fluctuation |
|------|---------|---------|---------|-----------|
| DeepLabv3+ | 20.73 | 51.17 | 27.40 | Extreme (30.44) |
| UNet | 31.30 | 46.53 | 27.01 | Moderate (19.52) |
| BANet | 37.74 | 63.03 | 43.22 | Large (25.29) |
| ViT | 43.97 | 50.88 | 36.64 | Moderate (14.24) |
| DlinkViT | 40.36 | 51.46 | 42.05 | Small (11.10) |

### Ablation Study (Impact of Spectral Bands)

| Dimension of Analysis | Findings |
|---------|------|
| Visible vs. Infrared Bands | Fog is bright in visible light (scattering) and dark in infrared (weak absorption) |
| Single Band vs. Multi-band | Multi-band combinations improve detection performance (complementary information) |
| FY4A vs. H8/9 (Same Region) | A significant performance discrepancy exists for models trained on different satellites in the same region |
| Positive-to-Negative Sample Ratio | Positive samples are extremely sparse; the ratio clearly impacts model performance |

### Key Findings
- Huge fluctuations in cross-regional generalization: The difference in CSI for the same method across different regions can reach up to 30 points (e.g., DeepLabv3+ scores only 20.73 in B.C. vs. 51.17 in C.C.), indicating that single-region evaluation is highly misleading.
- ViT-based methods exhibit better cross-regional stability (fluctuation of 14.24 vs. 30.44 for DeepLabv3+), suggesting that global attention is beneficial for regional generalization.
- Significant performance differences occur for models trained on different satellite data within the same region, with spectral band discrepancies being the primary driver.
- BANet achieves the highest CSI of 63.03 in the C.C. region but drops to 37.74 in B.C., showing that method efficacy is strongly region-dependent.
- The dataset scale (68,000 samples) and annotation volume (11,600+) far exceed all existing marine fog datasets.

## Highlights & Insights
- **Data-Driven Fog Region Selection**: Marine fog frequency is statistically analyzed from 9.5 million ICOADS records, allowing data to speak rather than relying on subjective choices. This ensures the representativeness and authority of the 15 selected regions. This methodology can be generalized to any remote sensing task requiring study area selection.
- **Cube-stream Data Structure**: Integrating spatiotemporal spectral data into a streaming structure of $\mathbb{R}^{T \times C \times H \times W}$ inherently supports both detection (single-frame slicing) and forecasting (time-window slicing), as well as multi-dimensional analysis. This data organization scheme serves as a highly valuable reference for other meteorological remote sensing datasets.
- **Systematic Vulnerability Exposure**: The experimental design cleverly controls regional and satellite variables to systematically expose model generalization deficiencies, rather than simply reporting performance on a single test set. This evaluation paradigm is highly recommended for other remote sensing tasks.

## Limitations & Future Work
- Dataset annotation relies on meteorological experts; the annotation process is time-consuming and may suffer from inter-expert inconsistency.
- Polar marine fog is still not covered among the 15 regions (though reasons are provided), and tropical sea areas are also excluded.
- Only geostationary satellites are utilized; high-resolution data from polar-orbiting satellites (e.g., MODIS, VIIRS) are not incorporated.
- Cross-regional transfer learning or domain adaptation methods are not explored, illustrating generalization challenges without providing a solution.
- The detection and forecasting tasks remain independent; exploring detection-assisted forecasting or joint representation learning is a potential future research direction.
- The 30-minute temporal resolution may not be fine-grained enough for rapidly changing marine fog.

## Related Work & Insights
- **vs. Huang et al. (2023)**: Limited to the Yellow Sea and Bohai Sea using H8/9 with only 4,291 samples. MFogHub covers 15 regions and 6 satellites with 68,000 samples, representing a massive leap in both scale and diversity.
- **vs. Zhou et al. (2022)**: GOCI single-satellite dataset with only 1,040 samples at $512 \times 512$ resolution. MFogHub's $1024 \times 1024$ resolution and multi-satellite coverage are much more suited for real-world applications.
- **vs. Bari et al. (2023)**: Focused only on the coastal waters of Morocco using MeteoSat for forecasting. MFogHub supports both detection and forecasting tasks simultaneously.
- This dataset is also highly valuable for domain adaptation and transfer learning fields—serving as a natural multi-domain benchmark.

## Rating
- Novelty: ⭐⭐⭐⭐ First global multi-regional and multi-satellite marine fog dataset, with a highly practical cube-stream design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 16 baselines across multiple regions and satellites, coupled with in-depth spectral sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ The dataset construction process is clearly described, supported by rich and analytical visualizations.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in data infrastructure within the marine fog domain, exerting a long-term impact on meteorological remote sensing and domain adaptation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](../../NeurIPS2025/remote_sensing/greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)
- [\[CVPR 2025\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)
- [\[CVPR 2025\] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)
- [\[CVPR 2026\] GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization](../../CVPR2026/remote_sensing/geobridge_a_semantic-anchored_multi-view_foundation_model_bridging_images_and_te.md)
- [\[CVPR 2026\] Orthogonal Spatial-Aware Multi-View Anchor Graph Clustering for Incomplete Remote Sensing Data](../../CVPR2026/remote_sensing/orthogonal_spatial-aware_multi-view_anchor_graph_clustering_for_incomplete_remot.md)

</div>

<!-- RELATED:END -->
