---
title: >-
  [Paper Note] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets
description: >-
  [CVPR 2026][Autonomous Driving][mmWave radar] This paper proposes EMDUL, a pipeline that expands mmWave HPE datasets in scale and diversity by (1) annotating unlabeled mmWave data via pseudo-labels with a novel unsupervi…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "mmWave radar"
  - "human pose estimation"
  - "dataset expansion"
  - "LiDAR point cloud conversion"
  - "semi-supervised learning"
date: 2026-05-08
content_hash: 96dec2d629988844
---

# EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets

**Conference**: CVPR 2026
**arXiv**: [2603.14507](https://arxiv.org/abs/2603.14507)
**Code**: [GitHub](https://github.com/Shimmer93/EMDUL)
**Area**: Autonomous Driving
**Keywords**: mmWave radar, human pose estimation, dataset expansion, LiDAR point cloud conversion, semi-supervised learning

## TL;DR

This paper proposes EMDUL, a pipeline that expands mmWave HPE datasets in scale and diversity by (1) annotating unlabeled mmWave data via pseudo-labels with a novel unsupervised temporal consistency loss (UTCL), and (2) converting LiDAR datasets to mmWave point clouds through a closed-form converter with flow-based point filtering (FPF). The approach reduces in-domain error by 15.1% and cross-domain error by 18.9%.

## Background & Motivation

mmWave radar has attracted sustained attention for human pose estimation (HPE) owing to its 3D perception capability, privacy preservation, and robustness to lighting conditions. Mainstream methods use processed 3D point clouds as input, yet current mmWave HPE datasets suffer from two fundamental bottlenecks:

**Data scarcity**: Annotated mmWave HPE datasets are extremely limited, and data collection and annotation are costly.

**Insufficient diversity**:
   - **Homogeneous point cloud attributes**: Limited device models and environments result in narrow distributions of detection noise, point density, and motion sensitivity.
   - **Monotonic poses**: Subjects predominantly adopt standing postures facing the radar, lacking complex actions such as squatting or swimming.

Meanwhile, **unlabeled mmWave data** is easy to collect, and **LiDAR HPE datasets** (e.g., LiDARHuman26M, HmPEAR) are abundant and posturally diverse. However, LiDAR and mmWave point clouds differ fundamentally in physical sensing principles—LiDAR can detect static objects, whereas mmWave relies on the Doppler effect and is more sensitive to moving targets—making naive mixing ineffective. Prior expansion methods either enlarge skeleton diversity without improving point cloud attribute distributions, or require paired mmWave–LiDAR data that is difficult to collect.

## Method

### Overall Architecture

EMDUL consists of two independent modules:

1. **Pseudo-label estimator**: Trained on annotated mmWave data with the UTCL loss to generate pseudo-labels $D^{\text{pl}}$ for unlabeled mmWave data.
2. **Closed-form point cloud converter**: Converts LiDAR datasets into mmWave-style point clouds $D^{\text{conv}}$.

The workflow proceeds as follows: LiDAR data is first converted to $D^{\text{conv}}$ and merged with the original $D^{\text{mm}}$ to form $D^{\text{lab}}$; the pseudo-label estimator is then trained on $D^{\text{lab}}$ to annotate unlabeled data, yielding $D^{\text{pl}}$; the final expanded dataset is $D^{\text{exp}} = D^{\text{lab}} \cup D^{\text{pl}}$. At each training epoch, $D^{\text{conv}}$ is regenerated with a new random seed and $D^{\text{pl}}$ is updated, enabling iterative refinement.

### Key Designs

1. **Unsupervised Temporal Consistency Loss (UTCL)**

    - **Function**: Imposes temporal consistency constraints on unlabeled mmWave data to improve pseudo-label quality.
    - **Mechanism**: Exploits the physical prior of mmWave motion detection—joints near detected points are likely moving, while those far from detected points are likely static. UTCL comprises two complementary components:
        - **Dynamic Consistency Loss (DCL)**: Encourages joints near the point cloud to exhibit non-zero optical flow.
       $$L^{\text{dyn}} = \frac{1}{|F_t^{\text{dyn}}|} \sum_{k} \max(0, \eta - \|F_t^{\text{dyn}}[k]\|_2)$$
        - **Static Consistency Loss (SCL)**: Encourages joints far from the point cloud to exhibit near-zero optical flow.
       $$L^{\text{sta}} = \frac{1}{|F_t^{\text{sta}}|} \sum_{k} \|F_t^{\text{sta}}[k]\|_2$$
    - **Design Motivation**: Using DCL or SCL in isolation biases predictions toward excessive motion or excessive stillness, respectively; their combination maintains balance and is used alongside the supervised loss $L^{\text{lab}}$.

2. **Flow-based Point Filtering (FPF)**

    - **Function**: Simulates the motion detection mechanism of mmWave radar to transform LiDAR point clouds into point clouds with mmWave-like characteristics.
    - **Mechanism**: The 3D flow vector of each point is computed via inverse-distance-weighted interpolation from skeleton optical flow, and points are then probabilistically filtered by flow magnitude—low-flow (static) points are discarded with higher probability:
    $$\mathcal{P}(P_t[i] \in P_t^{\text{conv}}) = \min\left(\frac{\|F_t^P[i]\|_2}{\upsilon_t}, 1\right)$$
    - **Design Motivation**: mmWave radar relies on the Doppler effect, making moving body parts more detectable. FPF directly simulates this physical mechanism at the point cloud level.

3. **Complete Point Cloud Conversion Pipeline (NPA→FPF→RS→NI)**

    - **NPA (Noise Point Addition)**: Adds a fixed number of noise points to simulate environmental clutter.
    - **FPF (Flow-based Point Filtering)**: Simulates the motion detection mechanism (see above).
    - **RS (Random Sampling)**: Reduces point density to simulate the sparsity of mmWave point clouds.
    - **NI (Noise Injection)**: Adds random noise to coordinates to simulate the lower spatial resolution of mmWave radar.

### Loss & Training

The total loss for the pseudo-label estimator:

$$L = L^{\text{lab}} + \lambda^{\text{con}} L^{\text{con}}$$

- $L^{\text{lab}}$: MSE loss on annotated data.
- $L^{\text{con}} = L^{\text{dyn}} + L^{\text{sta}}$: UTCL.
- $\lambda^{\text{con}} = 0.01$.

Training runs for 100 epochs with the AdamW optimizer, a learning rate of $10^{-4}$, cosine annealing, and linear warmup. The inference model $\theta^{\text{infer}}$ is trained jointly with the estimator; at each epoch, $\theta^{\text{pl}}$ is updated first, followed by $\theta^{\text{infer}}$.

## Key Experimental Results

### Main Results

Using 10% annotated MM-Fi (F) or mmBody (B) data under the full setting (+unlabeled data +HmPEAR), MPJPE (cm):

| Setting | Method | Backbone | MPJPE | PA-MPJPE |
|---------|--------|----------|-------|----------|
| F→F (in-domain) | P4T baseline | P4T | 12.23 | 7.95 |
| F→F | EMDUL | P4T | **10.06** | 7.01 |
| F→F | EMDUL | SPiKE | 10.40 | 7.23 |
| B→F (cross-domain) | P4T baseline | P4T | 33.62 | 15.85 |
| B→F | EMDUL | SPiKE | **22.80** | 14.09 |
| B→F | Mean Teacher | SPiKE | 32.71 | 15.74 |

- In-domain (F→F) MPJPE reduced by **15.1%** (12.23→10.06; with SPiKE, 17.5% over Mean Teacher).
- Cross-domain (B→F) MPJPE reduced by **18.9%**; with SPiKE, outperforms Mean Teacher by **30.3%**.

### Ablation Study

| Configuration | F→B MPJPE | PA-MPJPE | Note |
|---------------|-----------|----------|------|
| No pseudo-labels (PCC only) | 15.22 | 11.51 | LiDAR conversion alone yields significant gains |
| +$L^{\text{lab}}$ only | 15.12 | 11.44 | Supervised by annotated data |
| +$L^{\text{lab}}$+DCL+SCL (full UTCL) | **14.89** | **11.11** | Complementary combination |
| PCC w/o FPF | 15.85 | 12.23 | FPF contributes most |
| Full PCC | 14.89 | 11.11 | Components are complementary |

**Realism validation of LiDAR point cloud conversion**: A binary classifier trained to distinguish mmWave from LiDAR point clouds classifies 60.46% of converted point clouds as mmWave when FPF is used, compared to only 43.06% without FPF.

### Key Findings

- FPF is the single most impactful component in the point cloud conversion pipeline, directly simulating the physical motion detection mechanism of mmWave radar.
- DCL and SCL in isolation do not improve cross-domain performance, but their combination yields significant gains—each alone biases predictions toward excessive motion or excessive stillness.
- EMDUL's advantage is most pronounced at 1% annotated data (MPJPE drops from 18.40 to 14.77).
- Cross-domain gains exceed in-domain gains, indicating that EMDUL genuinely improves data diversity rather than overfitting.

## Highlights & Insights

- **Elegant exploitation of physical priors**: UTCL translates the Doppler-based motion detection mechanism of mmWave radar into an unsupervised loss constraint, serving as an exemplar of physics-informed semi-supervised learning.
- **Closed-form conversion without paired data**: The LiDAR→mmWave conversion requires no paired cross-modal data, relying solely on independent LiDAR datasets.
- **Per-epoch re-randomized conversion data**: Effectively increases training data diversity, analogous to online data augmentation.

## Limitations & Future Work

- The point cloud conversion pipeline depends on empirically set parameters (thresholds $\gamma$, $\delta$, etc.), which may be suboptimal for different scenes; future work could explore adaptive or learnable conversion.
- UTCL's modeling of complex motion patterns remains limited; more sophisticated temporal models could be incorporated.
- The current framework supports only single-person HPE; extension to multi-person scenarios involving occlusion and interaction is an important future direction.

## Related Work & Insights

- Unlike video-to-point-cloud methods such as Video2mmPoint, EMDUL operates directly in the 3D point cloud domain and improves point cloud attribute distributions.
- The motion detection simulation concept underlying FPF is generalizable to other Doppler-based sensors (e.g., FMCW radar).
- The temporal consistency idea in UTCL can inspire other self-supervised 3D estimation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of cross-modal point cloud conversion from LiDAR to mmWave for HPE dataset expansion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two mmWave and two LiDAR datasets, comprehensive ablations, and validation across multiple backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated physical intuitions.
- Value: ⭐⭐⭐⭐ Addresses the data bottleneck in mmWave HPE with a generalizable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[CVPR 2026\] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[CVPR 2026\] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)

</div>

<!-- RELATED:END -->
