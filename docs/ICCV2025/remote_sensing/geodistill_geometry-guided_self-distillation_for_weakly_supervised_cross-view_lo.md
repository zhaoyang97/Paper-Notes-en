---
title: >-
  [Paper Note] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization
description: >-
  [ICCV 2025][Remote Sensing][cross-view localization] This paper proposes GeoDistill, a framework that enhances locally discriminative feature learning via a Field-of-View (FoV) occlusion-based teacher-student self-distillation paradigm. Under weakly supervised conditions (requiring only coarse GPS annotations), it achieves robust cross-view localization with performance improvements exceeding 10%, and can be applied as a plug-and-play component to different localization frame…
tags:
  - "ICCV 2025"
  - "Remote Sensing"
  - "cross-view localization"
  - "weakly supervised learning"
  - "self-distillation"
  - "FoV occlusion"
  - "orientation estimation"
date: 2026-05-08
content_hash: d917716fd7ae7610
---

# GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization

**Conference**: ICCV 2025
**arXiv**: [2507.10935](https://arxiv.org/abs/2507.10935)  
**Code**: [https://github.com/tongshw/GeoDistill](https://github.com/tongshw/GeoDistill)  
**Area**: Remote Sensing / Cross-View Localization
**Keywords**: cross-view localization, weakly supervised learning, self-distillation, FoV occlusion, orientation estimation

## TL;DR

This paper proposes GeoDistill, a framework that enhances locally discriminative feature learning via a Field-of-View (FoV) occlusion-based teacher-student self-distillation paradigm. Under weakly supervised conditions (requiring only coarse GPS annotations), it achieves robust cross-view localization with performance improvements exceeding 10%, and can be applied as a plug-and-play component to different localization frameworks.

## Background & Motivation

Cross-view localization aims to estimate a camera's 3-DoF pose (planar position + yaw angle) by matching ground-level images with satellite imagery, and is critical for large-scale outdoor applications such as autonomous driving and augmented reality. Existing state-of-the-art methods predominantly rely on fully supervised learning, requiring precise ground camera pose annotations—typically obtained through dedicated mapping vehicles equipped with expensive sensors traversing the environment, which is both costly and poorly generalizable across regions.

Weak supervision offers a more practical alternative, as coarse location information can be acquired via smartphone GPS. However, existing weakly supervised methods have notable limitations: (1) they require data from the test region for fine-tuning, which is infeasible in practice; and (2) image-level metric learning fails to provide sufficiently strong supervision for precise localization. The core challenge is: how can discriminative local features be learned given only paired images without precise pose annotations?

The key insight of GeoDistill is that a panoramic image and its partial-FoV counterpart depict the same geographic location and must therefore map to identical satellite coordinates—this geometric consistency requirement itself constitutes a powerful supervisory signal.

## Method

### Overall Architecture

GeoDistill adopts a two-stage pipeline:
- **Stage 1 – Orientation Estimation**: Predicts the relative yaw angle $\hat{\theta}$ between the ground and satellite images, and aligns orientation by horizontally shifting the panorama.
- **Stage 2 – Location Estimation**: Applies the proposed geometry-guided self-distillation paradigm on top of any cross-view localization network. At inference, the teacher model, continuously refined via EMA, is used.

### Key Designs

1. **Orientation Estimation Network**: To align structural cues shared between ground and aerial views (e.g., road layouts), the panorama $I_g$ is first projected into a Bird's-Eye View (BEV) image $I_g^b$ via spherical transformation, correcting road curvature introduced by perspective projection. Satellite and BEV features are then extracted using non-weight-sharing EfficientNet-B0 encoders:
    $\mathbf{F}_s = \mathcal{E}_s(\mathbf{I}_s), \quad \mathbf{F}_g = \mathcal{E}_g(\mathbf{I}_g^b)$
   The concatenated channel features are passed through an MLP for per-degree classification, trained with cross-entropy loss.

2. **FoV-Based Occlusion Strategy**: Unlike random patch masking or activation-based masking, FoV occlusion simulates a camera with a limited field of view: $\tilde{I_g}^m = I_g \odot M_{\text{mask}}$, where the mask retains a contiguous horizontal field (e.g., 180°–240°) and discards the remainder. The key advantage is that FoV occlusion always preserves coherent scene geometry (including discriminative features such as lane markings and buildings), whereas patch masking may destroy critical structures or retain uninformative regions (e.g., sky). Notably, applying FoV occlusion purely as a data augmentation strategy degrades performance; it is effective only within the teacher-student distillation framework.

3. **Geometry-Guided Self-Distillation**:

    - **Teacher model** $f_t$: Receives the complete panorama $\tilde{I_g}$ and satellite image $I_s$, generating heatmaps using full global context.
    - **Student model** $f_s$: Shares the same architecture but receives the FoV-occluded image $\tilde{I_g}^m$ and $I_s$.
    - **Uncertainty Preservation**: Heatmaps from both branches are sharpened via low-temperature softmax ($\tau < 1$): $P_t = \text{Softmax}(H_t/\tau)$, $P_s = \text{Softmax}(H_s/\tau)$. Sharpening serves a dual purpose: suppressing noisy low-confidence signals while preserving the teacher's "dark knowledge" (distributional shape and relative activation intensity).
    - **Self-Distillation Loss**: Cross-entropy $\mathcal{L}_{SD} = \mathbb{E}[-\sum_i P_t(i)\log P_s(i)]$
    - **Bidirectional Knowledge Flow**: Teacher weights absorb the robust features learned by the student via EMA: $\theta_t \leftarrow \alpha\theta_t + (1-\alpha)\theta_s$ ($\alpha=0.9$), enabling continuous self-improvement.

### Loss & Training

The student is updated by minimizing the self-distillation loss via gradient descent; the teacher is updated via EMA. Training uses batch size 8, learning rate 0.0001, and the Adam optimizer. Temperature $\tau=0.06$, EMA ratio $\alpha=0.9$. The FoV range is randomly sampled between 180° and 240°. All experiments are conducted on a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results

| Dataset | Method | Cross-Area Mean (m) | Cross-Area Median (m) | Same-Area Mean (m) |
|--------|------|-------------------|---------------------|-------------------|
| VIGOR | CCVPE | 4.97 | 1.68 | 3.60 |
| VIGOR | +GeoDistill | **4.05** (↓18.5%) | **1.57** (↓6.5%) | **3.21** (↓10.8%) |
| VIGOR | G2SWeakly (VGG) | 5.20 | 1.44 | 4.81 |
| VIGOR | +GeoDistill (VGG) | **4.49** (↓13.6%) | **1.22** (↓15.3%) | **4.26** (↓11.4%) |
| VIGOR | G2SWeakly (DINO) | 3.58 | 1.45 | 3.61 |
| VIGOR | +GeoDistill (DINO) | **2.68** (↓25.1%) | **1.20** (↓17.2%) | **3.08** (↓14.7%) |

**Comparison with Fully Supervised Methods (VIGOR Cross-Area, 0° noise)**:

| Method | Supervision | Mean (m) | Median (m) |
|------|---------|---------|-----------|
| HC-Net (current SOTA) | Fully Supervised* | 3.35 | 1.59 |
| CCVPE | Fully Supervised* | 4.97 | 1.68 |
| **GeoDistill (DINO)** | **Weakly Supervised** | **2.68** | **1.20** |

### Ablation Study

| Occlusion Strategy | Backbone | Mean (m) | Median (m) | Note |
|---------|---------|---------|-----------|------|
| FoV Occlusion | CNN (VGG) | **4.49** | **1.22** | Best |
| Max-Activation Occlusion | CNN | 5.14 | 1.42 | Destroys structure |
| Random Patch Occlusion | CNN | 5.21 | 1.44 | No improvement |
| FoV Occlusion | ViT (DINO) | **2.68** | **1.20** | Backbone-agnostic validation |
| Random Patch Occlusion | ViT | 3.10 | 1.33 | — |

| Uncertainty Strategy | Mean (m) | Median (m) | Note |
|-----------|---------|-----------|------|
| With Sharpening | **4.49** | **1.22** | Balances denoising and dark knowledge retention |
| Without Sharpening | 5.23 | 1.44 | Raw dark knowledge too noisy to converge |
| Unimodal | 4.96 | 1.36 | Dark knowledge discarded but functional |
| Baseline (no distillation) | 5.20 | 1.44 | — |

| FoV Used As… | G2SWeakly | CCVPE | Note |
|---------|-----------|-------|------|
| Data Augmentation | 5.64 (↑ degraded) | 5.37 (↑ degraded) | Harmful as standalone augmentation |
| Teacher-Student Distillation | **4.49** (↓ improved) | **4.05** (↓ improved) | Effective only within the framework |

### Key Findings

- Weakly supervised GeoDistill (DINO) surpasses all fully supervised SOTA methods on Cross-Area localization, achieving a mean error of 2.68 m versus HC-Net's 3.35 m.
- FoV occlusion as a standalone data augmentation degrades performance (+0.44 m mean), but yields significant improvements within the distillation framework (−0.71 m mean), validating the necessity of the teacher-student design.
- Stronger backbones (e.g., DINO vs. VGG) yield larger gains from GeoDistill (DINO: ↓25.1% vs. VGG: ↓13.6%).
- The optimal FoV range is 180°–240°; performance degrades when the range is too narrow (insufficient information) or too wide (insufficient discrepancy from the teacher).

## Highlights & Insights

- The idea of converting geometric consistency constraints into self-supervisory signals is elegant—panoramic and partial views of the same location must point to identical coordinates, and this "free" constraint suffices to drive local feature learning.
- The comparison between FoV occlusion and patch occlusion reveals an important principle: masking strategies should preserve "identifiable scene coherence" rather than introducing random structural destruction.
- The sharpening temperature $\tau$ refines "dark knowledge" more effectively than either extreme (full dark knowledge vs. complete discarding), reflecting the noise-signal trade-off in information theory.
- The plug-and-play property is noteworthy: without modifying the underlying architecture, the distillation training paradigm alone improves multiple existing methods.

## Limitations & Future Work

- The FoV occlusion strategy is validated only on panoramic inputs; for inputs that are inherently limited-FoV (e.g., pinhole cameras in KITTI), the gains from GeoDistill are smaller than on VIGOR.
- The orientation estimation network relies on BEV-projected road structure alignment, which may be less effective in non-road environments (e.g., mountainous or forested areas).
- The EMA update coefficient $\alpha$ is fixed at 0.9; adaptive adjustment strategies merit further exploration.
- Validation is limited to two datasets (VIGOR and KITTI), constraining the breadth of empirical evidence.

## Related Work & Insights

- Self-distillation paradigms (e.g., Born-Again Networks) have proven effective in the knowledge distillation literature; this paper is the first to extend this approach to cross-view localization.
- The FoV occlusion concept shares conceptual parallels with MAE's random masking, but is critically adapted to the task's geometric properties.
- The result of weakly supervised performance surpassing fully supervised methods suggests that, for certain tasks, carefully designed self-supervisory signals can be more effective than expensive annotated data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of FoV occlusion and self-distillation is concise and effective; the exploitation of geometric consistency constraints is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Plug-and-play generality is validated across multiple baselines with comprehensive ablations, though only two datasets are used.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; ablation design is meticulous, with controlled experiments supporting every design choice.
- **Value**: ⭐⭐⭐⭐ Weak supervision surpassing full supervision offers a practically viable, low-cost solution for deploying cross-view localization systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis](../../CVPR2026/remote_sensing/geo2_geometry-guided_cross-view_geo-localization_and_image_synthesis.md)
- [\[ECCV 2024\] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](../../ECCV2024/remote_sensing/weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)
- [\[CVPR 2026\] MOGeo: Beyond One-to-One Cross-View Object Geo-localization](../../CVPR2026/remote_sensing/mogeo_beyond_one-to-one_cross-view_object_geo-localization.md)
- [\[ICCV 2025\] GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration](geoexplorer_active_geo-localization_with_curiosity-driven_exploration.md)
- [\[ECCV 2024\] Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](../../ECCV2024/remote_sensing/adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)

</div>

<!-- RELATED:END -->
