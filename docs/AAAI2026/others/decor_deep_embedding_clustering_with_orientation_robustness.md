---
title: >-
  [Paper Note] DECOR: Deep Embedding Clustering with Orientation Robustness
description: >-
  [AAAI 2026 (KGML Bridge, non-archival)][deep clustering] This paper proposes DECOR, a framework that achieves orientation-robust clustering of wafer map defect patterns via a rotation-invariant equivariant convolutional…
tags:
  - "AAAI 2026 (KGML Bridge, non-archival)"
  - "deep clustering"
  - "rotation invariance"
  - "wafer defect detection"
  - "non-parametric clustering"
  - "anomaly detection"
date: 2026-05-08
content_hash: 4ddcf5e4f4321af0
---

# DECOR: Deep Embedding Clustering with Orientation Robustness

**Conference**: AAAI 2026 (KGML Bridge, non-archival)
**arXiv**: [2510.03328](https://arxiv.org/abs/2510.03328)  
**Code**: None  
**Area**: Other
**Keywords**: deep clustering, rotation invariance, wafer defect detection, non-parametric clustering, anomaly detection

## TL;DR

This paper proposes DECOR, a framework that achieves orientation-robust clustering of wafer map defect patterns via a rotation-invariant equivariant convolutional autoencoder (RCAE), non-parametric clustering (DeepDPM), and an ensemble anomaly detection mechanism.

## Background & Motivation

In semiconductor manufacturing, early detection of wafer defects is critical to product yield. However, raw data from wafer quality testing presents several challenges:

**Unlabeled and complex data**: Wafer map patterns are complex and typically lack manual annotations.

**Class imbalance**: Different defect types occur at vastly different frequencies.

**Multi-defect co-occurrence**: A single wafer may exhibit multiple defect patterns simultaneously.

**Orientation variability**: Due to variations in wafer placement or handling, identical defect patterns may appear at different rotation angles. Standard clustering methods treat rotated instances as distinct patterns, leading to cluster fragmentation.

Traditional clustering methods (K-Means, DBSCAN, etc.) rely on fixed assumptions (e.g., number of clusters $k$, distance threshold $\varepsilon$), making them impractical in dynamic environments where wafer defect distributions evolve over time. This motivates the authors to adopt a **non-parametric approach** to adaptively discover clusters while using **equivariant networks** to handle orientation variability.

Core motivation: to design a deep clustering framework that requires no pre-specified number of clusters and is robust to rotation, so that spatially similar defects are consistently grouped regardless of their orientation.

## Method

### Overall Architecture

DECOR is a three-stage framework: (A) a rotation- and flip-invariant embedding extractor RCAE, (B) the non-parametric clustering module DeepDPM, and (C) an ensemble anomaly detection mechanism.

Input wafer map → Normalization & masking → RCAE feature extraction (128-dim) → DeepDPM clustering → Ensemble anomaly detection

### Key Designs

1. **Rotation-Invariant Convolutional Autoencoder (RCAE)**:

   The equivariant encoder is built using `R2Conv` layers from the `e2cnn` library. It consists of three equivariant blocks, each comprising R2Conv + ReLU + PointwiseAvgPool (stride=2), with channel counts increasing as 8→16→32. The key design is achieving equivariance with respect to the **dihedral group $D_4$**, covering four discrete rotations (0°, 90°, 180°, 270°) and two mirror reflections.

   **GroupPooling** is applied to convert equivariant features into orientation-invariant descriptors (collapsing all orientation channels), followed by a linear layer mapping to a 128-dimensional latent vector. The decoder mirrors the encoder structure and uses `ConvTranspose2D` for image reconstruction.

   Design motivation: Compared to contrastive learning methods such as MoCo, RCAE has built-in symmetry handling, producing more compact and separable clusters. Experiments confirm that RCAE yields better-separated clusters in the latent space.

2. **Non-Parametric Clustering (DeepDPM)**:

   A deep clustering framework based on the Dirichlet Process Mixture Model (DPMM) that **requires no pre-specified number of clusters**. Unlike K-Means, DeepDPM infers the optimal number of clusters from the data distribution.

   Implemented as a two-layer MLP: Linear(128, 50) → Linear(50, K), where $K$ is determined adaptively during training. The model produces **soft cluster membership probabilities**, with hard labels obtained via argmax.

   Hyperparameter settings: $\nu = d + 2$ (where $d$ is the input dimensionality), $k_{init} = 30$, maximum training epochs = 200.

3. **Ensemble Anomaly Detection**:

   Two detectors — Isolation Forest (IF) and Local Outlier Factor (LOF) — are combined, with a sample flagged as anomalous only when both detectors agree ($\mathbf{1}_{final} = \mathbf{1}_{IF} \wedge \mathbf{1}_{LOF}$).

   A key contribution is the **robust adaptive threshold**:
   $$\tau = \text{median}(s) + k \cdot \text{MAD}(s)$$
   Using the median and MAD instead of the mean and standard deviation ensures robustness to heavy-tailed distributions and pre-existing outliers.

   The number of neighbors for LOF is selected adaptively: $k_{LOF} = \text{clip}(\sqrt{N}, k_{min}, k_{max})$, adjusting automatically based on cluster size. IF uses a conservative contamination prior (hi\_cont = 0.20).

### Loss & Training

- RCAE is trained with MSE loss and Adam optimizer (lr=$10^{-3}$), batch size=128, for 1000 epochs.
- Input images are normalized and preprocessed with edge masking and Gaussian blurring.
- Wafer maps are uniformly rescaled to 128×128 pixels.
- `MultilabelStratifiedShuffleSplit` is used to ensure consistent defect type distributions across training and test sets.

## Key Experimental Results

### Main Results

Evaluated on the MixedWM38 dataset (38,000+ wafer maps, 38+ defect pattern combinations):

| Embedding | Clustering | Final/Best K | NMI ↑ | ARI ↑ |
|-----------|------------|--------------|-------|-------|
| CAE | K-Means^p | 24 | 0.503±0.00 | 0.199±0.00 |
| MoCo | K-Means^p | 18 | 0.409±0.00 | 0.173±0.01 |
| RCAE | K-Means^p | 30 | 0.529±0.00 | 0.205±0.00 |
| CAE | DeepDPM | 25 | 0.498±0.05 | 0.218±0.01 |
| MoCo | DeepDPM | 30 | 0.273±0.00 | 0.117±0.00 |
| **RCAE (Ours)** | **DeepDPM** | **22** | **0.543±0.03** | **0.296±0.00** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| RCAE vs CAE (K-Means) | NMI: 0.529 vs 0.503 | Equivariant encoding improves clustering quality |
| RCAE vs MoCo (K-Means) | NMI: 0.529 vs 0.409 | Reconstruction-based learning outperforms contrastive learning |
| DeepDPM vs K-Means (RCAE) | ARI: 0.296 vs 0.205 | Non-parametric method significantly surpasses fixed-K approach |
| MoCo + DeepDPM | NMI: 0.273 | Contrastive embeddings are incompatible with DeepDPM |

### Key Findings

1. RCAE embeddings combined with DeepDPM achieve the best performance on both NMI and ARI.
2. RCAE-derived clusters demonstrably exhibit **rotation invariance** — identical defect patterns at different orientations are correctly assigned to the same cluster.
3. Clustering quality correlates with the degree of separation among defect patterns in the latent space.
4. Non-parametric clustering automatically determines $K=22$ without manual tuning.
5. The ensemble anomaly detection successfully identifies anomalous defects within clusters (e.g., donut defects anomalous within a center-defect cluster).

## Highlights & Insights

- **Elegant design of equivariance vs. invariance**: The encoder uses equivariant convolutions to preserve orientation information, while GroupPooling converts equivariance to invariance — each operating at the appropriate level of abstraction.
- **Fully unsupervised end-to-end pipeline**: From embedding learning to clustering to anomaly detection, no labels are required at any stage.
- **Strong practical applicability**: The model is lightweight (small RCAE parameter count), incurs low training cost (~8 hours on a single H100), and is suitable for industrial deployment.
- **Robust ensemble detection strategy**: IF provides global partition-based isolation while LOF captures local density deviations; AND-fusion reduces false positives.

## Limitations & Future Work

1. **Multiple DeepDPM runs required**: Determining an appropriate initial $K$ and optimal training epoch requires repeated experimentation.
2. **Difficulty in multi-label evaluation**: NMI and ARI require reducing multi-labels to a dominant label, which may not fully reflect performance.
3. **Single dataset evaluation**: Validation is limited to MixedWM38; generalization capability remains unknown.
4. **Limited baseline comparisons**: Methods such as GMM and HDBSCAN are not included.
5. **Non-archival venue**: As a KGML Bridge paper, the research depth is inherently limited.

## Related Work & Insights

DECOR integrates advances from three areas: equivariant networks (e2cnn/$D_4$ group), non-parametric Bayesian clustering (DPMM/DeepDPM), and robust anomaly detection. The framework offers valuable reference for other domains requiring rotation-robust clustering, such as cell image analysis and remote sensing. Future work could explore multi-label-aware clustering metrics and temporal defect tracking.

## Rating

- Novelty: ⭐⭐⭐ (Limited innovation in individual components; novelty lies in the combination)
- Experimental Thoroughness: ⭐⭐⭐ (Single dataset, limited baseline comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Clear and accessible, with detailed method descriptions)
- Value: ⭐⭐⭐ (Meaningful for industrial applications, but contributions are limited given the non-archival venue)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](../../ICML2026/others/towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)
- [\[AAAI 2026\] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](../../ICLR2026/others/the_price_of_robustness_stable_classifiers_need_overparameterization.md)

</div>

<!-- RELATED:END -->
