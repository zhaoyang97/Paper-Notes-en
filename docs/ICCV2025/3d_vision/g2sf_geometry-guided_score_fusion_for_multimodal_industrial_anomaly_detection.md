---
title: >-
  [Paper Note] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection
description: >-
  [ICCV 2025][3D Vision][multimodal anomaly detection] G2SF reinterprets memory-bank-based anomaly scores as isotropic Euclidean distances in a local feature space, and progressively evolves them into an anisotropic unified fusion metric by learning direction-aware scaling factors via a Local Scale Prediction Network (LSPN), achieving state-of-the-art performance on multimodal industrial anomaly detection.
tags:
  - ICCV 2025
  - 3D Vision
  - multimodal anomaly detection
  - metric learning
  - anisotropic distance
  - industrial inspection
  - point cloud–RGB fusion
date: 2026-05-08
content_hash: 67d5545ce5ecc4c9
---

# G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection

**Conference**: ICCV 2025
**arXiv**: [2503.10091](https://arxiv.org/abs/2503.10091)
**Code**: [GitHub](https://github.com/ctaoaa/G2SF)
**Area**: 3D Vision
**Keywords**: multimodal anomaly detection, metric learning, anisotropic distance, industrial inspection, point cloud–RGB fusion

## TL;DR

G2SF reinterprets memory-bank-based anomaly scores as isotropic Euclidean distances in a local feature space, and progressively evolves them into an anisotropic unified fusion metric by learning direction-aware scaling factors via a Local Scale Prediction Network (LSPN), achieving state-of-the-art performance on multimodal industrial anomaly detection.

## Background & Motivation

Industrial quality inspection is a critical component of modern manufacturing. Because defective products are extremely rare in practice, unsupervised anomaly detection (trained on normal samples only) has become the dominant paradigm. However:

**Incomplete unimodal information**: 3D point clouds provide rich geometric detail but lack texture/color information; 2D RGB images capture rich appearance cues but lack geometric context. Certain anomalies are only detectable in a specific modality (e.g., cutting defects in foam are nearly invisible in RGB, while thread anomalies in cable glands are insufficiently discriminative in point clouds).

**Insufficient discriminability of unimodal anomaly scores**: Memory-bank-based methods (e.g., PatchCore) use the Euclidean distance to the nearest prototype as the anomaly score. The key limitation is that this distance is **isotropic**—it ignores directional distribution patterns in the local feature space. Normal regions may also exhibit high Euclidean distances, degrading post-fusion discriminability.

**Coarse existing fusion strategies**: Score addition (BTF), score maximization (Shape-Guided), and OCSVM (M3DM) all fail to adequately handle the unreliability of unimodal anomaly scores. Feature adaptation methods (LSFA) may discard detection-critical information during dimensionality reduction.

**Core geometric insight**: The anomaly score in memory-bank methods is essentially an isotropic hypersphere centered at the prototype in the local feature space. However, normal sample distributions are typically directional—more elongated along certain directions. Learning an anisotropic distance metric can suppress scores along normal directions and amplify scores along anomalous directions, substantially improving discriminability.

## Method

### Overall Architecture

G2SF employs frozen DINO (image) and Point-MAE (point cloud) encoders to extract features and construct per-modality memory banks. Features are decomposed into (prototype, direction, distance) triplets via geometric encoding. LSPN predicts direction-aware scaling factors, which are combined with raw distances to form a unified anisotropic metric.

### Key Designs

1. **Geometric Feature Encoding**: Each feature $\mathbf{f}^m_i$ is decomposed relative to its nearest memory prototype $\mathbf{m}^m_{i,j}$ into a triplet:
    $(\mathbf{m}^m_{i,j}, \mathbf{d}^m_{i,j}, s^m_{i,j}) = \mathcal{E}_{\mathbf{m}^m_{i,j}}(\mathbf{f}^m_i)$
   where $s^m_{i,j} = \|\mathbf{f}^m_i - \mathbf{m}^m_{i,j}\|$ is the Euclidean distance (i.e., the conventional anomaly score), and $\mathbf{d}^m_{i,j} = (\mathbf{f}^m_i - \mathbf{m}^m_{i,j}) / s^m_{i,j}$ is the unit direction vector. This encoding preserves all information in the original feature, avoiding the information loss inherent in feature adaptation.

2. **Local Scale Prediction Network (LSPN)**: A lightweight MLP predicts direction-aware scaling factors:
    $[w^P_{i,j}, w^R_{i,j}] = \Phi(\mathbf{m}^P_{i,j}, \mathbf{m}^R_{i,j}, \mathbf{d}^P_{i,j}, \mathbf{d}^R_{i,j})$
   LSPN consists of two parallel branches processing prototype and direction inputs respectively, with an $\exp(\tanh(\cdot))$ activation ensuring $w^m_{i,j} \in [e^{-1}, e^1]$, symmetrically distributed around 1. The network is trained to produce $w < 1$ (suppression) for normal samples and $w > 1$ (amplification) for anomalous samples.

3. **Anisotropic Local Distance Metric**: The fusion metric is defined as:
    $l_{i,j} = \sum_{m \in \{P,R\}} w^m_{i,j} s^m_{i,j} \sigma^m$
   where $\sigma^m$ is a learnable global modality weight. When LSPN is initialized with $w \approx 1$, the metric degrades to a Euclidean metric in the joint feature space, then gradually evolves into an anisotropic metric during training, mitigating the overfitting risk associated with learning a metric from scratch.

### Loss & Training

The loss consists of a **discriminative loss** and a **geometry-preserving loss**:

- **Separation loss $\mathcal{L}_{sep}$**: minimizes $l_{i,0}$ for normal samples and maximizes $l_{i,0}^{-1}$ for anomalous samples.
- **Margin loss $\mathcal{L}_{mar}$**: reduces the overlap region between normal and anomalous distributions.
- **Consistency loss $\mathcal{L}_{cns}$**: enforces that metrics for neighboring prototypes do not vary drastically (neighbor consistency), while metrics for distant prototypes remain sufficiently large (distal separation).
- **Scale loss $\mathcal{L}_{sc}$**: asymmetrically constrains $w \leq 1$ for normal samples and $w \to e^1$ for anomalous samples.
- **Cross-modal alignment loss $\mathcal{L}_{cma}$**: constructs negative samples via random index permutation, compelling LSPN to exploit cross-modal correspondences.

Training employs synthetic anomaly injection based on a CutPaste strategy. At inference, the final anomaly score for patch $i$ is $s_i = \min\{l_{i,j} \mid j=0,\ldots,k\}$, i.e., the minimum metric over the $k+1$ nearest-neighbor local spaces.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (DAUP) | Gain |
|--------|------|------|-----------------|------|
| MVTec-3D AD | I-AUROC | 0.971 | 0.970 | +0.001 |
| MVTec-3D AD | AUPRO@30% | — | 0.969 (DAUP) | Competitive |
| MVTec-3D AD (Mean) | I-AUROC | 0.971 | 0.970 | SOTA |

Per-category I-AUROC shows Cable Gland improving from 0.889 (DAUP) to 0.923, and Foam reaching 0.991 (SOTA), demonstrating the advantage of cross-modal fusion.

### Ablation Study

| Configuration | I-AUROC | Notes |
|------|---------|------|
| Isotropic baseline (Euclidean) | ~0.945 | PatchCore baseline |
| + LSPN direction-aware scaling | ~0.960 | Significant gain from anisotropy |
| + Consistency loss | ~0.965 | Neighbor consistency constraint |
| + Cross-modal alignment | ~0.970 | Exploiting cross-modal correspondence |
| Full G2SF | 0.971 | All components |

Ablation of geometric reasoning (score aggregation) comparing mean, max, and min strategies confirms that the min operator is optimal, corresponding to the distance to the normal data manifold.

### Key Findings

- Gains are particularly pronounced on AUPRO@1% (a stricter metric), indicating superior precision in localizing anomalous regions.
- G2SF reaches competitive performance within very few training epochs, benefiting from the progressive evolution from the Euclidean metric.
- The most challenging categories—Cable Gland and Foam—exhibit the largest improvements, validating the effectiveness of cross-modal complementarity.
- The symmetry of the $\exp(\tanh(\cdot))$ activation ensures that scaling factors do not deviate excessively from the Euclidean baseline.

## Highlights & Insights

- Reinterpreting memory-bank anomaly scores as geometric distances is an elegant conceptual shift.
- The progressive evolution from isotropic to anisotropic metric is a clever strategy to avoid overfitting in high-dimensional metric learning.
- Fully preserving the original feature information via geometric encoding—without feature compression or adaptation—is an important design choice.
- The LSPN architecture is simple yet computationally efficient thanks to its dual-branch design for direction and position inputs.

## Limitations & Future Work

- As an MLP, LSPN may fail to capture complex nonlinear directional dependencies.
- The quality and diversity of synthetic anomalies may limit LSPN's generalization ability.
- Extension to more than two modalities (e.g., RGB + point cloud + thermal imaging) remains unexplored.
- Memory bank size directly impacts inference speed; efficiency must be considered for large-scale deployment.

## Related Work & Insights

- PatchCore's memory-bank framework provides a solid foundation upon which G2SF builds a stronger metric.
- The idea of local Mahalanobis distance is simplified into direction-aware scaling factors, reducing computational complexity.
- The synthetic anomaly injection strategy (CutPaste) converts the unsupervised problem into a semi-supervised one, offering a transferable insight for other tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel perspective of geometric metric evolution; elegant LSPN design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, extensive ablations, per-category analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Derives rigorously from geometric intuition with clear exposition.
- Value: ⭐⭐⭐⭐ Practical industrial value for multimodal fusion-based anomaly detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] Stable Score Distillation](stable_score_distillation.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[ICCV 2025\] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)

</div>

<!-- RELATED:END -->
