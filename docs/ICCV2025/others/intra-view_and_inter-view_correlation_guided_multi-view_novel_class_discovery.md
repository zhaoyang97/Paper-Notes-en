---
title: >-
  [Paper Note] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery
description: >-
  [ICCV 2025][Novel Class Discovery] This paper proposes IICMVNCD, the first framework extending Novel Class Discovery (NCD) to the multi-view setting. It captures distributional consistency between known and novel classes…
tags:
  - "ICCV 2025"
  - "Novel Class Discovery"
  - "Multi-view Learning"
  - "Matrix Factorization"
  - "View Weighting"
  - "Clustering"
date: 2026-05-08
content_hash: 8f915dc1e1452c59
---

# Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery

**Conference**: ICCV 2025
**arXiv**: [2507.12029](https://arxiv.org/abs/2507.12029)  
**Code**: N/A  
**Area**: Other
**Keywords**: Novel Class Discovery, Multi-view Learning, Matrix Factorization, View Weighting, Clustering

## TL;DR

This paper proposes IICMVNCD, the first framework extending Novel Class Discovery (NCD) to the multi-view setting. It captures distributional consistency between known and novel classes via intra-view matrix factorization, and transfers view relationships learned from known classes to novel classes through inter-view weight learning, eliminating the need for pseudo-labels.

## Background & Motivation

**Novel Class Discovery (NCD)** is an important learning paradigm: given a labeled dataset of known classes and an unlabeled dataset of novel classes (with non-overlapping categories), the goal is to leverage knowledge from known classes to cluster the novel ones. This emulates the human cognitive process of using prior knowledge to understand new concepts—analogous to a child who, familiar with smartphones and tablets, can correctly distinguish computers from smartwatches.

Existing NCD methods suffer from two major limitations:

**Limitation 1: Single-view data only.** Multi-view data are increasingly prevalent in practice—medical diagnosis, for instance, requires the joint interpretation of multi-omics features such as gene expression and imaging. Single-view data may not provide sufficient information, and existing NCD methods cannot effectively handle information fusion in multi-view scenarios.

**Limitation 2: Reliance on pseudo-labels.** Most NCD methods supervise novel-class clustering via pseudo-labels, whose quality is sensitive to data noise, feature dimensionality, and other factors, leading to unstable performance. In multi-view settings, pseudo-label generation becomes even more challenging.

**Core Mechanism**: The paper addresses multi-view NCD at two levels—intra-view and inter-view:

- **Intra-view**: Exploits distributional similarity between known and novel classes by learning a shared feature basis matrix.
- **Inter-view**: Leverages supervision from known classes to learn view weights, which are then transferred to novel-class view fusion.

## Method

### Overall Architecture

IICMVNCD is an end-to-end, single-stage method comprising three core components:
1. Intra-view information extraction (matrix factorization)
2. Inter-view information extraction (view-weighted fusion + label prediction)
3. Joint optimization objective (with class separation constraints)

### Key Designs

1. **Intra-view Shared Basis Matrix Factorization**:

    - **Function**: Learns a shared basis matrix for known and novel classes within each view to improve feature representation quality.
    - **Mechanism**: The core assumption of NCD is that the data distributions of known and novel classes are similar. Based on this assumption, for each view $v$, the feature matrix $\mathbf{X}_v = [\mathbf{X}_v^l, \mathbf{X}_v^u]$ (concatenating known- and novel-class data) is factorized as:
    $$\min_{\mathbf{W}_v, \mathbf{Z}_v} \|\mathbf{X}_v - \mathbf{W}_v \mathbf{Z}_v\|_F^2 \quad \text{s.t.} \quad \mathbf{W}_v^\top \mathbf{W}_v = \mathbf{I}_k$$
      where $\mathbf{W}_v \in \mathbb{R}^{d_v \times k}$ is the view-specific shared basis matrix, $\mathbf{Z}_v \in \mathbb{R}^{k \times n}$ is the factor matrix, and $k = k_l + k_u$ denotes the total number of classes.
    - **Design Motivation**: The shared basis $\mathbf{W}_v$ captures distributional consistency between the two datasets; the orthogonality constraint prevents redundancy and stabilizes optimization. The factor matrix $\mathbf{Z}_v$ encodes inter-sample relationships, providing the foundation for subsequent label prediction.

2. **Inter-view Weight Learning and Label Prediction**:

    - **Function**: Learns optimal view weights from supervision on known classes and fuses multi-view information to generate consistent predicted labels.
    - **Mechanism**: Learnable view weights $\boldsymbol{\alpha}$ are introduced, and the factor matrix is further decomposed into a view-specific centroid matrix $\mathbf{A}_v$ and a consensus predicted label matrix $\mathbf{Y}$:
    $$\min_{\boldsymbol{\alpha}, \mathbf{W}_v, \mathbf{A}_v, \mathbf{Y}} \sum_{v=1}^V \alpha_v^2 \|\mathbf{X}_v - \mathbf{W}_v \mathbf{A}_v \mathbf{Y}\|_F^2 + \lambda_1 \|\mathbf{Y}_l - \mathbf{G}_l\|_F^2$$
      subject to $\boldsymbol{\alpha}^\top \mathbf{1} = 1, \boldsymbol{\alpha} \geq \mathbf{0}$. View weights are automatically updated based on reconstruction error:
    $$\alpha_v = \frac{1/r_v^2}{\sum_{v=1}^V 1/r_v^2}$$
      where $r_v^2 = \|\mathbf{X}_v - \mathbf{W}_v \mathbf{A}_v \mathbf{Y}\|_F^2$.
    - **Design Motivation**: Different views vary in quality and importance; fixed weights cannot adapt to specific data. The ground-truth labels $\mathbf{G}_l$ of known classes constrain the learning of $\mathbf{Y}_l$, indirectly optimizing the view weights, which are then transferred to the novel classes.

3. **Class Separation Constraint**:

    - **Function**: Prevents novel-class samples from being incorrectly assigned to known classes.
    - **Mechanism**: A repulsion term is added to the final objective to maximize the distance between novel-class predicted labels and known-class ground-truth labels:
    $$\mathcal{L} = \sum_v \alpha_v^2 \|\mathbf{X}_v - \mathbf{W}_v \mathbf{A}_v \mathbf{Y}\|_F^2 + \lambda_1 \|\mathbf{Y}_l - \mathbf{G}_l\|_F^2 - \lambda_2 \sum_{\mathbf{g}^i \in \mathbf{G}_l} \sum_{\mathbf{y}^j \in \mathbf{Y}_u} \|\mathbf{g}^i - \mathbf{y}^j\|_F^2$$
    - **Design Motivation**: Due to the distributional similarity between known and novel classes, novel-class samples are prone to being incorrectly allocated to known-class clusters during joint learning. The repulsion term encourages novel-class labels to move away from known-class labels.

### Loss & Training

- **Optimization**: Alternating optimization over four steps, each fixing all other variables and updating one:
    - $\mathbf{W}_v$: Closed-form solution via SVD: $\mathbf{W}_v = \mathbf{S}_v \mathbf{V}_v^\top$
    - $\mathbf{A}_v$: Solved by setting the derivative to zero: $\mathbf{A}_v = \mathbf{W}_v^\top \mathbf{X}_v \mathbf{Y}^\top (\mathbf{Y}\mathbf{Y}^\top)^{-1}$
    - $\mathbf{Y}$: Per-sample discrete optimization (one-hot constraint)
    - $\boldsymbol{\alpha}$: Closed-form update based on the Cauchy–Schwarz inequality
- **Convergence Guarantee**: The objective function decreases monotonically at each iteration and is lower-bounded by $\mathcal{J} \geq -\lambda_2 n_l n_u \sqrt{2}$.
- **Time Complexity**: $\mathcal{O}(d(nk + k^2) + Vk^3)$, linear in the number of samples $n$, ensuring scalability.

## Key Experimental Results

### Main Results

**ACC comparison on 8 datasets** (vs. multi-view clustering and NCD methods):

| Dataset | AEVC (MVC) | CKD (NCD) | IICMVNCD | Gain |
|---------|-----------|-----------|-----------|------|
| BRCA | 86.67 | 84.32 | **98.79** | +12.12 |
| uci-digit | 92.60 | 92.50 | **95.30** | +2.70 |
| Cora | 63.20 | 35.23 | **76.36** | +13.16 |
| Wiki | 37.30 | 61.05 | **65.42** | +4.37 |
| STL10 | 98.74 | 96.11 | **99.02** | +0.28 |
| YTB10 | 91.59 | 93.01 | **94.55** | +1.54 |

### Ablation Study

**NMI comparison**:

| Dataset | Best MVC | Best NCD | IICMVNCD | Notes |
|---------|---------|---------|-----------|-------|
| BRCA | 87.81 | 86.93 | **90.45** | Large margin |
| Cora | 29.96 | 2.12 | **44.59** | NCD methods severely degrade |
| Wiki | 49.98 | 35.10 | **66.44** | Significant improvement |
| CCV | 17.81 | 16.83 | **19.24** | Modest improvement |

### Key Findings

1. **Multi-view NCD differs substantially from single-view NCD**: Existing single-view NCD methods (e.g., CKD) exhibit unstable performance on multi-view data and may even underperform classical multi-view clustering methods (e.g., ACC of only 35.23% vs. AEVC's 63.20% on Cora).
2. **View weight learning is critical**: Optimal view weights vary considerably across datasets; adaptive weighting substantially outperforms fixed uniform weights.
3. **The pseudo-label-free design** effectively avoids noisy label issues, particularly on high-dimensional multi-view data.
4. The method achieves especially strong results on medical multi-omics datasets (BRCA: 98.79%, KIPAN: 92.51% ACC), validating the practical value of multi-view NCD in biomedical applications.
5. Theoretical convergence guarantees ensure stable performance across diverse data conditions.

## Highlights & Insights

- **First formulation of the multi-view NCD task**, filling an important research gap. Existing NCD methods are limited to single-view data, whereas multi-view data are ubiquitous in fields such as bioinformatics.
- **Elegant pseudo-label-free design**: The label matrix is predicted directly via matrix factorization and view weighting, fundamentally eliminating pseudo-label noise.
- **Alternating optimization with closed-form solutions** renders the method both efficient and convergence-guaranteed—a rare property in the NCD literature.
- The view weight learning mechanism transfers view relationships from known classes to novel classes, leveraging the unique advantage of labeled data available in the NCD setting.

## Limitations & Future Work

- The matrix factorization framework assumes a linear feature space, which may be insufficiently expressive for complex nonlinear feature distributions.
- The number of novel classes $k_u$ must be known in advance, which may be impractical in real-world applications.
- The dataset splitting strategy is simplistic (first/second half of categories); in realistic scenarios, the relationship between known and novel classes may be considerably more complex.
- Comparisons with deep learning baselines (e.g., DINOv2 features + clustering) are absent.
- The class separation constraint relies on simple distance-based repulsion, which may push the novel-class label distribution excessively far from the known-class distribution.

## Related Work & Insights

- Pioneer NCD works such as AutoNovel (ICLR 2021) and UNO focus on single-view image data; the multi-view extension proposed here carries significant methodological value.
- The approach is directly related to classical multi-view NMF clustering methods (e.g., DiNMF, OPMC), but incorporates the supervisory signal from NCD.
- Strong performance on multi-omics data suggests promising cross-disciplinary research directions with bioinformatics.
- The view weight learning strategy is generalizable to settings such as multi-modal few-shot learning.

## Rating

- Novelty: ⭐⭐⭐⭐ First to propose the multi-view NCD setting; method design is clean and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 datasets broadly, but lacks deep-feature baselines.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and theoretical analysis is complete.
- Value: ⭐⭐⭐⭐ Opens a new research direction with direct applicability to multi-view scenarios such as bioinformatics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-view Gaze Target Estimation](multi-view_gaze_target_estimation.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] Thermal Polarimetric Multi-view Stereo](thermal_polarimetric_multi-view_stereo.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)

</div>

<!-- RELATED:END -->
