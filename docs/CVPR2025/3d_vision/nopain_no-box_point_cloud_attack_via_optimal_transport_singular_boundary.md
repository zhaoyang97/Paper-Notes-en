---
title: >-
  [Paper Note] NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary
description: >-
  [CVPR 2025][3D Vision][Point cloud adversarial attack] NoPain proposes the first no-box adversarial attack method for point clouds. By leveraging semi-discrete optimal transport (OT) to calculate the mapping from noise to feature space, it samples adversarial perturbations at the singular boundaries (non-differentiable points) of the mapping. This approach requires no target classifier or surrogate model, achieving 100% ASR on PointNet and maintaining a generation speed of on…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point cloud adversarial attack"
  - "no-box attack"
  - "optimal transport"
  - "singular boundary"
  - "transferability"
date: 2026-05-08
content_hash: bf0a58f0ada91af2
---

# NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary

**Conference**: CVPR 2025  
**arXiv**: [2503.00063](https://arxiv.org/abs/2503.00063)  
**Code**: [https://github.com/cognaclee/nopain](https://github.com/cognaclee/nopain)  
**Area**: 3D Vision  
**Keywords**: Point cloud adversarial attack, no-box attack, optimal transport, singular boundary, transferability

## TL;DR

NoPain proposes the first no-box adversarial attack method for point clouds. By leveraging semi-discrete optimal transport (OT) to calculate the mapping from noise to feature space, it samples adversarial perturbations at the singular boundaries (non-differentiable points) of the mapping. This approach requires no target classifier or surrogate model, achieving 100% ASR on PointNet and maintaining a generation speed of only 28ms/sample.

## Background & Motivation

1. **Background**: Point cloud adversarial attacks are categorized into white-box (requiring target model gradients), black-box (requiring target model queries), and transfer-based attacks (requiring surrogate models). All existing methods depend on some form of classifier information.
2. **Limitations of Prior Work**: (1) The success rate of transfer-based attacks (e.g., AdvPC, SI-ADV) drops significantly across architectures (13-54%); (2) Iterative optimization is required, resulting in slow generation speed (6-12s/sample); (3) The choice of surrogate model heavily influences the attack effectiveness.
3. **Key Challenge**: Attacks relying on classifier information are inherently limited by the classifier's specificity—perturbations effective on one model may not transfer to another.
4. **Goal**: To generate adversarial examples completely without any classifier information, leveraging solely the geometric properties of the data manifold.
5. **Key Insight**: The singular boundary in optimal transport theory—the Brenier potential function is non-differentiable on certain hyperplanes. These locations correspond to unstable regions on the data manifold, which are naturally vulnerable points for classifiers.
6. **Core Idea**: OT to compute hyperplane sets $\rightarrow$ dihedral angles to detect singular boundaries $\rightarrow$ sampling along the singular boundaries to generate adversarial examples.

## Method

### Overall Architecture

A pre-trained encoder-decoder (PointFlow or Point-Diffusion) extracts the features of clean point clouds $\rightarrow$ a semi-discrete OT solver computes the Brenier potential $\rightarrow$ gradient descent optimizes hyperplane parameters $\rightarrow$ dihedral angles detect singular boundaries $\rightarrow$ adversarial perturbations are expanded along singular boundaries $\rightarrow$ the decoder reconstructs the adversarial point clouds.

### Key Designs

1. **Semi-discrete Optimal Transport Solving**

    - **Function**: Computes the optimal mapping from continuous noise distribution to discrete feature sets.
    - **Mechanism**: The Brenier potential is $u_h(x) = \max_i \{\langle y_i, x \rangle + h_i\}$, and gradient descent is utilized to optimize the energy $E(h) = \sum_i (w_i(h) - 1/N)^2$ so that each target point receives equal area weight.
    - **Design Motivation**: Non-differentiable points of the OT mapping (intersection of hyperplanes) correspond exactly to regions of classification uncertainty—minor perturbations near these points can cause classification flips.

2. **Singular Boundary Detection**

    - **Function**: Identifies the most adversarial perturbation locations from the OT mapping.
    - **Mechanism**: Calculates the dihedral angle between adjacent hyperplanes $\theta_{ik} = \frac{\langle y_i, y_{ik} \rangle}{||y_i|| \cdot ||y_{ik}||}$; the intersection lines of hyperplanes with dihedral angles smaller than a threshold $\tau$ are identified as singular boundaries—where the transition probability to different categories is maximized.
    - **Design Motivation**: A small dihedral angle implies that the two hyperplanes are nearly parallel—a tiny displacement near their intersection can cross different Voronoi cells.

3. **Expanded Perturbation Generation**

    - **Function**: Generates physical adversarial point clouds along the direction of singular boundaries.
    - **Mechanism**: $\hat{y} = \lambda_i y_i + \lambda_{ik} y_{ik}$, interpolating between two hyperplanes by smoothing the OT mapping.
    - **Design Motivation**: Direct sampling on singular points is too sparse; expanding to band-like regions near boundaries increases coverage.

### Loss & Training

Training-free—only requires pre-trained point cloud encoders-decoders (PointFlow or Point-Diffusion). Hyperparameters are K=11, $\tau$=1.6 (PF) / 0.9 (PD).

## Key Experimental Results

### Main Results

| Method | PointNet ASR↑ | DGCNN ASR↑ | PCT ASR↑ | Generation Speed |
|------|-------------|-----------|---------|---------|
| AdvPC | 13.0% | 23.3% | 15.8% | 6.2s |
| SI-ADV | 54.5% | 67.3% | 91.3% | 8.9s |
| **NoPain-PD** | **100%** | **88.7%** | **85.7%** | **0.026s** |

### Ablation Study

| Defense Method | NoPain-PD ASR |
|---------|--------------|
| No Defense | 100% |
| SRS | 98.4% |
| SOR | 90.7% |
| DUP-Net | 85.0% |
| IF-Defense | 70.0% |

### Key Findings

- 100% ASR on PointNet—perfect attack, completely without any classifier information.
- Generation speed of 26-28ms, which is 200-300x faster than iterative methods—enabling real-time attacks.
- Maintains 70% ASR even against strong defenses like IF-Defense—singular boundary attacks possess inherent robustness.

## Highlights & Insights

- **Mathematical Elegance**: Rigorous theoretical foundation of the OT singular boundary—the argument that non-differentiable points correspond to classification uncertainty regions is highly natural.
- **Pioneering the No-Box Paradigm**: Attacks that completely bypass classifier information are a first in the point cloud domain.
- **Extreme Speed**: The generation speed of 28ms/sample makes real-time adversarial attacks feasible, which holds significant value for robustness evaluation.

## Limitations & Future Work

- Evaluated only on classification tasks; detection and segmentation tasks have not been tested.
- Relies on the quality of pre-trained encoders-decoders—a performance gap exists between PointFlow and Point-Diffusion.
- Hyperparameters (K, τ) are dataset-specific.

## Related Work & Insights

- **vs SI-ADV**: Transfer attacks require a surrogate model, achieving 54-91% ASR. NoPain requires no surrogate model, achieving 85-100% ASR.
- **vs AdvPC**: A classic white-box method, with cross-architecture transfer ASR of only 13-30%.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ OT singular boundary attack is a brand-new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ ModelNet40 + ShapeNet + 4 classifiers + 4 defenses + ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations.
- Value: ⭐⭐⭐⭐⭐ Profound impact on point cloud safety evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](../../ICML2026/3d_vision/avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ICML 2026\] Streaming Sliced Optimal Transport](../../ICML2026/3d_vision/streaming_sliced_optimal_transport.md)
- [\[CVPR 2025\] 3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)
- [\[CVPR 2025\] Sketchy Bounding-Box Supervision for 3D Instance Segmentation](sketchy_bounding-box_supervision_for_3d_instance_segmentation.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)

</div>

<!-- RELATED:END -->
