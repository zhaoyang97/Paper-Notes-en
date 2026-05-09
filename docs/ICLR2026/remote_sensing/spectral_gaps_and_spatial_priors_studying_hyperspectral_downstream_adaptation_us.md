---
title: >-
  [Paper Note] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind
description: >-
  [ICLR 2026][Remote Sensing][Hyperspectral Imaging] This paper investigates whether TerraMind, a multimodal geospatial foundation model not pretrained on hyperspectral data, can be effectively adapted to hyperspectral downstream tasks via channel adaptation strategies (naive band selection vs. SRF-based grouping). Results demonstrate that naive band selection consistently outperforms the physically-informed SRF approach, with the performance gap widening as the spectral complexity of the task increases.
tags:
  - ICLR 2026
  - Remote Sensing
  - Hyperspectral Imaging
  - Geospatial Foundation Models
  - Channel Adaptation
  - TerraMind
  - Spectral Response Function
date: 2026-05-08
content_hash: 11debe1707c97b7b
---

# Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind

**Conference**: ICLR 2026
**arXiv**: [2603.06690](https://arxiv.org/abs/2603.06690)
**Code**: None
**Area**: Remote Sensing
**Keywords**: Hyperspectral Imaging, Geospatial Foundation Models, Channel Adaptation, TerraMind, Spectral Response Function

## TL;DR

This paper investigates whether TerraMind, a multimodal geospatial foundation model not pretrained on hyperspectral data, can be effectively adapted to hyperspectral downstream tasks via channel adaptation strategies (naive band selection vs. SRF-based grouping). Results demonstrate that naive band selection consistently outperforms the physically-informed SRF approach, with the performance gap widening as the spectral complexity of the task increases.

## Background & Motivation

1. Geospatial foundation models (GFMs) have emerged as a paradigm shift in remote sensing, transitioning from task-specific architectures toward general-purpose pretrained models with transferable representations across tasks.
2. Hyperspectral imaging (HSI) provides rich spectral detail through hundreds of narrow spectral channels, which is critical for precision agriculture, mineral exploration, and environmental monitoring, yet remains underrepresented in existing GFMs due to data complexity.
3. Existing HSI-specific GFMs (e.g., HyperSIGMA, SpectralEarth) are largely unimodal; multimodal GFMs (e.g., DOFA) incorporate HSI pretraining data but rarely address the three-dimensional feature extraction required for HSI.
4. The availability and diversity of HSI downstream datasets are limited, with most benchmark datasets being single-scene, making it difficult to support standard deep learning training pipelines.
5. Core research question: Can a multimodal GFM not pretrained on HSI serve as an effective baseline for HSI-specific tasks?
6. This paper selects TerraMind as the subject of study and systematically compares two channel adaptation strategies across four HSI downstream tasks.

## Method

### Overall Architecture

The high-dimensional HSI input ($X_{\text{HSI}} \in \mathbb{R}^{H \times W \times C_{in}}$) is projected into the 12-band spectral space of Sentinel-2 L2A ($\hat{X} \in \mathbb{R}^{H \times W \times 12}$), followed by fine-tuning for downstream tasks using TerraMind's pretrained weights.

### Key Designs

#### 1. Naive Band Selection

For each Sentinel-2 target band $k$, the HSI band whose center wavelength is closest is selected:

$$\hat{X}_{:,:,k} = X_{\text{HSI}}\left[:,:,\arg\min_j |\lambda_j - \mu_k|\right]$$

This preserves the original radiance values of specific narrow bands while discarding the remaining spectral information.

#### 2. SRF-based Spectral Resampling

The Sentinel-2 spectral response function (SRF) is used to simulate physically realistic S2 signals by constructing a weight matrix $\mathbf{W} \in \mathbb{R}^{C_{in} \times 12}$:

$$\hat{w}_{j,k} = \frac{\phi_k(\lambda_j)}{\sum_{m=1}^{C_{in}} w_{m,k}}$$

All HSI bands falling within the S2 spectral range are aggregated via weighted summation, yielding a smoother, physically-informed representation.

### Loss & Training

- Segmentation tasks use a Fully Convolutional decoder (256 channels) with cross-entropy loss
- Regression tasks use a linear head (hidden dimension 256) with MSE loss
- AdamW optimizer, trained for 100 epochs with early stopping (patience=20)
- Cosine Annealing for segmentation; ReduceOnPlateau for regression
- All experiments are repeated 10 times with different random seeds to ensure statistical robustness

## Key Experimental Results

### Main Results

| Model | Adaptation | EnMAP-BNETD (Easy) | EnMAP-CDL (Moderate) | EnMAP-BDForet (Hard) | Hyperview-1 (V.Hard) |
|------|----------|-------|--------|---------|---------|
| TerraMind | Naive Selection | 0.465±0.002 | 0.693±0.006 | 0.657±0.007 | 0.813 (Rank #6) |
| TerraMind | SRF Grouping | 0.461±0.003 | 0.679±0.006 | 0.623±0.006 | 0.831 (Rank #25) |
| SpectralEarth | Full HSI (Upper) | **0.495±0.001** | **0.774±0.003** | **0.766±0.005** | **0.810** (Rank #5) |

### Ablation Study

| Comparison Dimension | Finding |
|----------|------|
| Naive vs. SRF | Naive consistently outperforms SRF, with 0.4%–3.4% higher mIoU on segmentation tasks |
| Performance gap vs. spectral complexity | Easy: ~3% gap, Moderate: ~8% gap, Hard: ~11% gap |
| Hyperview-1 regression | TerraMind Naive (0.813) approaches SpectralEarth (0.810); spatial representations partially compensate for spectral reduction |

### Key Findings

1. **Naive selection consistently outperforms SRF**: TerraMind pretraining establishes strong anchoring at S2 center wavelengths; naive selection preserves the original radiance distributions at these anchor points, while SRF-weighted averaging acts as a low-pass filter that smooths out critical narrow-band features.
2. **Performance gap positively correlates with spectral complexity**: Simple tasks (land cover) allow spatial features to compensate for spectral loss, whereas complex tasks (tree species classification) cannot be adequately addressed with only 12 bands.
3. **Unexpected competitiveness on regression tasks**: Soil parameters (K, P₂O₅, Mg, pH) can be indirectly detected via proxy signals such as organic matter and clay minerals, whose broad spectral responses align well with S2 bands.

## Highlights & Insights

1. This is the first systematic evaluation of the adaptation capability of a non-HSI-pretrained multimodal GFM on HSI downstream tasks, establishing an important baseline reference.
2. The paper reveals a counterintuitive finding: the physically-informed SRF method underperforms compared to simple band selection, which is attributed to the representation anchoring mechanism established during model pretraining.
3. The experimental design is rigorous—10 random seed repetitions spanning four tasks of varying difficulty from "easy" to "very hard."
4. The in-depth analysis of Hyperview-1 regression results from a soil spectroscopy perspective demonstrates noteworthy interdisciplinary insight.

## Limitations & Future Work

1. Only TerraMind is studied; the conclusions may be model-specific and require validation on other GFMs (e.g., DOFA).
2. Channel adaptation is limited to simple selection/weighting strategies; learnable spectral projection or adapter modules are not explored.
3. Native HSI tokenizer design is not investigated, though the paper identifies this as a future direction.
4. The downstream datasets are relatively small (1,600–2,550 chips), which may affect the assessment of fine-tuning effectiveness.

## Related Work & Insights

- **SpectralEarth**: A native HSI foundation model utilizing all 202 bands, serving as an upper-bound performance reference.
- **DOFA**: A multimodal GFM that does not adequately address three-dimensional HSI feature extraction.
- **TerraMind**: A multimodal GFM pretrained on multispectral/SAR/DEM/LULC data, with the optical modality based on Sentinel-2.
- Insight: Future multimodal GFMs should natively support hyperspectral tokenization rather than relying on simple dimensionality reduction adaptation.

## Rating

- ⭐ Novelty: 3/5 — The research question is valuable, but the methodology itself is relatively straightforward (band selection + fine-tuning)
- ⭐ Experimental Thoroughness: 4/5 — Four datasets, two strategies, and 10 repetitions yield statistically rigorous results
- ⭐ Writing Quality: 4/5 — Clear structure, in-depth analysis, and well-reasoned interpretation of results
- ⭐ Value: 3.5/5 — Provides important baselines and insights for integrating HSI into multimodal GFMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction](../../AAAI2026/remote_sensing/m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction.md)
- [\[CVPR 2026\] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](../../CVPR2026/remote_sensing/no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)
- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](../../CVPR2026/remote_sensing/exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[CVPR 2026\] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](../../CVPR2026/remote_sensing/metaspectra_a_compact_broadband_metasurface_camera.md)
- [\[AAAI 2026\] Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification](../../AAAI2026/remote_sensing/perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)

</div>

<!-- RELATED:END -->
