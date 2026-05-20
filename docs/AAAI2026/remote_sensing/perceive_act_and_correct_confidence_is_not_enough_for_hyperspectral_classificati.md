---
title: >-
  [Paper Note] Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification
description: >-
  [AAAI 2026][Remote Sensing][hyperspectral] This paper proposes the CABIN framework, which employs a closed-loop cognitive perceive–act–correct learning mechanism. By replacing naive confidence with epistemic uncertainty…
tags:
  - "AAAI 2026"
  - "Remote Sensing"
  - "hyperspectral"
  - "semi-supervised"
  - "uncertainty"
  - "pseudo-label"
  - "evidential deep learning"
date: 2026-05-08
content_hash: 2c832f581b3a3050
---

# Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification

**Conference**: AAAI 2026
**arXiv**: [2511.10068](https://arxiv.org/abs/2511.10068)  
**Code**: To be confirmed  
**Area**: Remote Sensing / Hyperspectral Classification
**Keywords**: hyperspectral, semi-supervised, uncertainty, pseudo-label, evidential deep learning

## TL;DR

This paper proposes the CABIN framework, which employs a closed-loop cognitive perceive–act–correct learning mechanism. By replacing naive confidence with epistemic uncertainty to guide sample selection and pseudo-label management in semi-supervised hyperspectral image classification, CABIN significantly outperforms fully supervised baselines while using only 75% of the labeled data.

## Background & Motivation

Hyperspectral image (HSI) classification relies on capturing rich spectral features for fine-grained land-cover analysis, with broad applications in urban planning, military reconnaissance, and precision agriculture. However, existing deep learning methods generally rest on a flawed assumption: **high-confidence predictions are inherently reliable**. In practice, due to the limited spatial resolution of HSI, individual pixels may correspond to mixtures of multiple materials, introducing semantic ambiguity into the annotations themselves.

Existing semi-supervised methods such as FixMatch and CGMatch adopt fixed confidence thresholds or historical consistency as criteria for pseudo-label selection, yet these approaches:

- Ignore the model's current **cognitive state**, failing to distinguish genuine certainty from blind overconfidence
- Relying solely on confidence leads to **confirmation bias**, where the model grows increasingly confident in erroneous predictions
- Static thresholds or lagged feedback cannot adapt to the dynamic evolution of uncertainty throughout training

**Core Insight**: Confidence alone is insufficient—epistemic uncertainty must be used to measure what the model "knows it does not know."

## Method

### Overall Architecture: CABIN (Cognitive-Aware Behavior-Informed learNing)

CABIN establishes a closed loop of **Perceive → Act → Correct**:

1. **Perceive**: Epistemic uncertainty $u_i = K / S_i$ is estimated for each sample via Evidential Deep Learning (EDL), where $S_i$ is the total evidence of the Dirichlet distribution.
2. **Act**: The UGDSS module partitions the candidate set into high-uncertainty samples (exploration) and low-uncertainty samples (exploitation), performing dual-path sampling accordingly.
3. **Correct**: The FDAS module introduces the Uncertainty-Gap metric to partition pseudo-labeled data into reliable, ambiguous, and noisy subsets, applying differentiated losses to each.

### Key Design 1: Uncertainty-Guided Dual Sampling Strategy (UGDSS)

- **Test-Time Augmentation**: $K$ spectral–spatial variants are generated per sample and averaged over EDL outputs to stabilize uncertainty estimation.
- **Adaptive Threshold**: Histogram density analysis is used to locate the first local minimum of the uncertainty distribution as the splitting threshold $T_u$, rather than a fixed percentile.
- **DRQS (Diversity-Representative Query Selection)**: K-means++ clustering is applied to semantic embeddings of high-uncertainty samples; the sample closest to each cluster centroid is selected, eliminating spatial redundancy.
- **GFP (Gaussian Feature Perturbation)**: Gaussian noise proportional to the estimated uncertainty is injected into the feature space of selected samples, enhancing model robustness in uncertain regions.

### Key Design 2: Fine-Grained Dynamic Assignment Strategy (FDAS)

- The **Uncertainty-Gap (UG)** is defined as: $UG_i^\alpha = \max_k(\bar{\alpha}_{ik}) - \text{second\_max}_k(\bar{\alpha}_{ik})$, measuring the gap between the strongest and second-strongest evidence classes.
- EMA smoothing is applied to the evidence vector $\bar{\alpha}_i$ to reduce batch-level fluctuations.
- Combining softmax confidence $c_i$ and $UG_i^\alpha$ with dual thresholds $(\tau_c, \tau_e)$, pseudo-labeled samples are partitioned into three subsets:
    - **Reliable set $\mathcal{D}_{re}$**: High confidence + high UG → trained with EDL loss
    - **Ambiguous set $\mathcal{D}_{am}$**: Intermediate → trained with noise-robust GCE loss
    - **Noisy set $\mathcal{D}_{no}$**: Low confidence + low UG → discarded

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{EDL}}(\mathcal{D}_L \cup \hat{\mathcal{D}}_{\text{aug}}) + \lambda_r \mathcal{L}_{\text{EDL}}(\mathcal{D}_{re}) + \lambda_a \mathcal{L}_{\text{GCE}}(\mathcal{D}_{am})$$

where $\lambda_r = \lambda_a = 0.3$.

## Key Experimental Results

### Experimental Setup

- **Datasets**: Five HSI benchmarks (Indian Pines, Salinas, Pavia University, WHU-Hi-HongHu, WHU-Hi-LongKou), spanning agricultural, urban, and UAV remote sensing scenarios.
- **Baselines**: CABIN is inserted into four SOTA methods (CNN-based: ReS², CLOLN; Transformer-based: SSFTT, GSC-ViT).
- **Core Condition**: CABIN uses only 75% of the original annotations, i.e., 240 samples vs. 320 for baselines.
- **Metrics**: OA, AA, Cohen's Kappa (κ).

### Main Results (Table 1: Indian Pines)

| Method | OA (%) | AA (%) | κ×100 |
|--------|--------|--------|-------|
| ReS² | 79.70 | 88.98 | 76.71 |
| ReS² + CABIN | **87.39** (+7.69) | **93.80** (+4.82) | **85.68** (+8.97) |
| SSFTT | 87.35 | 93.65 | 85.48 |
| SSFTT + CABIN | **90.08** (+2.73) | **94.20** (+0.55) | **88.66** (+3.18) |
| GSC-ViT | 84.64 | 91.33 | 82.49 |
| GSC-ViT + CABIN | **87.40** (+2.76) | **93.31** (+1.98) | **85.60** (+3.11) |

### Multi-Dataset Results (Table 3 Summary)

| Dataset | Best OA Gain | Best κ Gain |
|---------|-------------|------------|
| Salinas | +1.92 (CLOLN) | +2.11 (CLOLN) |
| PaviaU | +2.43 (ReS²) | +3.11 (ReS²) |
| LongKou | +1.09 (SSFTT) | +1.41 (SSFTT) |
| HongHu | +2.44 (CLOLN) | +1.94 (ReS²) |

### Ablation Study (Table 2: Indian Pines, SSFTT backbone)

| UGDSS | FDAS | OA (%) | κ×100 |
|-------|------|--------|-------|
| ✗ | ✗ | 87.35 | 85.48 |
| ✓ | ✗ | 89.80 | 88.30 |
| ✗ | ✓ | 88.87 | 87.27 |
| ✓ | ✓ | **90.08** | **88.66** |

Each module is individually effective; combining both yields complementary gains.

## Highlights & Insights

- **Confidence ≠ Reliability**: A profound insight that, for the first time, reframes the bottleneck of semi-supervised HSI classification from the perspective of cognitive-behavioral consistency.
- **Uncertainty-Gap Metric**: Elegantly combines behavioral confidence and evidential gap to differentiate pseudo-label quality, offering greater sensitivity than confidence alone or historical consistency.
- **Model-Agnostic Plug-and-Play Design**: CABIN requires no modification to the backbone network and can be directly inserted into arbitrary existing methods, yielding consistent improvements across all four baselines.
- **High Annotation Efficiency**: CABIN surpasses fully supervised baseline performance with only 75% of the labeled data, and peak performance is achieved with as little as 50% of annotations.

## Limitations & Future Work

- Experiments are validated solely on hyperspectral image classification; generalization to natural images or other remote sensing tasks (e.g., object detection, change detection) is not explored.
- EDL-based epistemic uncertainty estimation is unstable during early training and requires test-time augmentation to mitigate this, increasing inference overhead.
- Histogram parameters in the adaptive threshold (bin count $N$, tolerance $\delta$) require manual specification, with no sensitivity analysis provided.
- The three-way partition (reliable/ambiguous/noisy) is governed by two thresholds, and adaptability to extreme class imbalance scenarios is not thoroughly examined.
- The number of GFP augmentation samples requires manual tuning (experiments show that excessive augmentation degrades performance), with no automated mechanism proposed.

## Related Work & Insights

- **Evidential Deep Learning**: Sensoy et al. (2018) introduced the EDL framework, parameterizing predictive uncertainty via Dirichlet distributions.
- **Semi-Supervised Learning**: FixMatch (Sohn et al., 2020) applies fixed confidence thresholds for pseudo-labeling; CGMatch (Cheng et al., 2025) introduces historical consistency.
- **Hyperspectral Image Classification**: SSFTT (Sun et al., 2022) employs spectral–spatial attention; GSC-ViT (Zhao et al., 2024) uses a global spectral context Vision Transformer; IGroupSS-Mamba (He et al., 2024) adopts state space modeling.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: Reexamines semi-supervised HSI classification from a cognitive closed-loop perspective; the UG metric is elegantly designed.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Five datasets, four baselines, comprehensive ablation and efficiency analysis; overall convincing.
- **Writing Quality** ⭐⭐⭐⭐: Narrative is clear and coherent, with the perceive–act–correct analogy sustained throughout.
- **Value** ⭐⭐⭐⭐⭐: Plug-and-play design with high performance under low annotation budgets; low barrier to practical deployment.
- **Deduction**: Task scope is narrow (HSI classification only); generalization is insufficiently validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](../../CVPR2026/remote_sensing/pretrained_image_matchers_for_sar_optical_satellite_registration.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/remote_sensing/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[CVPR 2026\] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](../../CVPR2026/remote_sensing/metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)
- [\[CVPR 2026\] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](../../CVPR2026/remote_sensing/lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)
- [\[ICLR 2026\] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](../../ICLR2026/remote_sensing/spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us.md)

</div>

<!-- RELATED:END -->
