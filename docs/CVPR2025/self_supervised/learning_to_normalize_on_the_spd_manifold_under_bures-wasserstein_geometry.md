---
title: >-
  [Paper Note] Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry
description: >-
  [CVPR 2025][Self-Supervised Learning][SPD Manifold] This paper proposes GBWBN, the first batch normalization method for the SPD manifold based on generalized Bures-Wasserstein geometry. By introducing learnable metric parameters and matrix power non-linear transformations to effectively handle ill-conditioned covariance matrices, it achieves SOTA performance on skeleton-based action recognition and EEG classification.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "SPD Manifold"
  - "Bures-Wasserstein Metric"
  - "Riemannian Batch Normalization"
  - "Ill-conditioned Matrices"
  - "Learnable Geometry"
date: 2026-05-08
content_hash: 54209ba4bad48109
---

# Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry

**Conference**: CVPR 2025  
**arXiv**: [2504.00660](https://arxiv.org/abs/2504.00660)  
**Code**: [https://github.com/jjscc/GBWBN](https://github.com/jjscc/GBWBN)  
**Area**: Manifold Learning / SPD Networks  
**Keywords**: SPD Manifold, Bures-Wasserstein Metric, Riemannian Batch Normalization, Ill-conditioned Matrices, Learnable Geometry

## TL;DR
This paper proposes GBWBN, the first batch normalization method for the SPD manifold based on generalized Bures-Wasserstein geometry. By introducing learnable metric parameters and matrix power non-linear transformations to effectively handle ill-conditioned covariance matrices, it achieves SOTA performance on skeleton-based action recognition and EEG classification.

## Background & Motivation

**Background**: Covariance matrices (SPD matrices) are widely used in fields such as brain-computer interfaces, action recognition, and UAV recognition. SPD matrices lie on a Riemannian manifold rather than in Euclidean space, requiring dedicated Riemannian neural networks (e.g., SPDNet). Riemannian Batch Normalization (RBN) has been proven to improve the performance of SPD networks.

**Limitations of Prior Work**: Existing RBNs use the Affine-Invariant Metric (AIM) or the Log-Euclidean Metric (LEM), which perform poorly on ill-conditioned SPD matrices (with extremely large condition numbers, e.g., $\kappa > 10^5$). Han et al. proved that AIM has a quadratic dependence on SPD matrices, leading to low learning efficiency on ill-conditioned matrices. Ill-conditioned matrices are extremely common in real-world data: 100% of samples in the HDM05 and NTU RGB+D datasets have condition numbers exceeding $10^5$.

**Key Challenge**: Widely used regularization ($X \leftarrow X + \lambda I$) only guarantees positive definiteness but cannot effectively alleviate ill-conditioning. The underlying metrics (AIM, LEM, LCM) of existing RBNs are unfriendly to ill-conditioned matrices.

**Goal**: Design an RBN based on the Bures-Wasserstein (BW) metric, which is friendlier to ill-conditioned matrices, and introduce learnable geometric parameters.

**Key Insight**: BWM has a linear dependence on SPD matrices (vs. the quadratic dependence of AIM), making it naturally more suitable for ill-conditioned scenarios. The Generalized BWM (GBWM) parameterizes BWM through an SPD parameter, allowing for more flexible geometric representation.

**Core Idea**: Construct RBN based on GBWM, making the metric parameters learnable to adapt to data geometry, and introduce matrix power transformations to enhance representation capability.

## Method

### Overall Architecture
The batch normalization layers in the SPD network are replaced by GBWBN. The Riemannian mean and variance under BW geometry are calculated for the SPD features in each batch, followed by normalization and rescaling. The underlying geometric structure is dynamically adjusted using learnable SPD parameters.

### Key Designs

1. **Batch Normalization based on BW Geometry**:

    - Function: Perform normalization on the SPD manifold to handle ill-conditioned matrices.
    - Mechanism: Use the Riemannian operators of BWM (geodesics, logarithmic mapping, exponential mapping, Fréchet mean) instead of the corresponding operators under AIM to calculate batch mean and variance. Solving the Lyapunov operators under BWM involves element-wise operations after eigendecomposition, which is more stable than matrix inversion under AIM.
    - Design Motivation: The linear dependence of BWM prevents numerical explosion for small eigenvalues of ill-conditioned matrices.

2. **Learnable Generalized BW Metric (GBWM)**:

    - Function: Adapt the geometric structure of the normalized space to the data distribution.
    - Mechanism: GBWM introduces an SPD parameter $M$ to parameterize BWM, and $M$ is set as a learnable parameter trained together with the network. GBWM is locally equivalent to AIM, and thus possesses both the robustness of BWM and the geometric flexibility of AIM.
    - Design Motivation: A fixed metric may not adapt to variations in data distribution across different layers/tasks; a learnable metric enhances adaptive capability.

3. **Matrix Power Non-linear Deformation**:

    - Function: Further enhance the representation capability of GBWM.
    - Mechanism: Apply matrix power deformation $d_p(X,Y) = d(X^p, Y^p)^{1/p}$ to GBWM, introducing additional non-linearities to alter the manifold's geometry.
    - Design Motivation: Inspired by how matrix power deformation improves AIM performance in LieBN, the same idea is applied to GBWM.

### Loss & Training
Standard cross-entropy classification loss is used. GBWBN is embedded as a plug-and-play module into backbone networks such as SPDNet.

## Key Experimental Results

### Main Results
Classification accuracy on three datasets:

| Dataset | Task | GBWBN | Second-best RBN | Gain |
|--------|------|-------|---------|------|
| HDM05 | Action Recognition | SOTA | LieBN | Significant |
| NTU RGB+D | Action Recognition | SOTA | LieBN | Significant |
| MAMEM-SSVEP-II | EEG Classification | SOTA | SPDBN | Significant |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| BWM (Fixed Geometry) | Outperforms AIM | BWM is more robust to ill-conditioning |
| + Learnable GBWM | Further Improvement | Adaptive geometry is effective |
| + Matrix Power Deformation | Best | Nonlinear enhancement helps |
| No RBN | Significant drop | RBN remains crucial for SPD networks |

### Key Findings
- The higher the degree of ill-conditioning in the dataset, the more pronounced the advantages of GBWBN.
- The learnable metric parameters indeed undergo meaningful changes during training, rather than remaining static.
- GBWBN is plug-and-play and can replace existing RBN layers in any SPD network.

## Highlights & Insights
- **Geometry-driven Normalization Design**: Instead of simply mapping Euclidean concepts onto manifolds, the appropriate underlying geometry is selected based on data properties (ill-conditioning), reflecting the philosophy that "geometry should serve the data."
- **Learnable Metric Concept**: Allowing the network to learn the most suitable manifold geometry itself, which is similar to learning a "distance metric" in feature space.

## Limitations & Future Work
- Computations under GBWM are more complex than those under AIM (requiring the solution of Lyapunov equations).
- Only classification tasks are validated; scenarios like generative or contrastive learning remain unexplored.
- The training stability of learnable metrics requires further investigation.

## Related Work & Insights
- **vs. SPDBN/LieBN**: These use AIM/LEM metrics, which are unfriendly to ill-conditioned matrices.
- **vs. ManifoldNorm**: Uses first- and second-order statistics but under AIM/LEM, whereas ours is under BWM.

## Rating
- Novelty: ⭐⭐⭐⭐ The first RBN based on BW geometry; the learnable metric is novel.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated on three datasets, but all are relatively small; larger-scale validation is needed.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations, with sufficient background introduction.
- Value: ⭐⭐⭐⭐ An important improvement for the SPD network community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bures-Isotropy Alignment: Manifold Learning of Generalized Category Discovery](../../ICLR2026/self_supervised/bures-isotropy_alignment_manifold_learning_of_generalized_category_discovery.md)
- [\[ICCV 2025\] WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](../../ICCV2025/self_supervised/wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [\[NeurIPS 2025\] SegMASt3R: Geometry Grounded Segment Matching](../../NeurIPS2025/self_supervised/segmast3r_geometry_grounded_segment_matching.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](../../ICML2025/self_supervised/generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
