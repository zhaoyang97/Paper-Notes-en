---
title: >-
  [Paper Note] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment
description: >-
  [AAAI 2026][LLM Evaluation][Incomplete multi-view clustering] This paper proposes DIMVC-HIA, a deep incomplete multi-view clustering framework that integrates hierarchical imputation with dual alignment. The method first…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Incomplete multi-view clustering"
  - "hierarchical imputation"
  - "energy-based model"
  - "contrastive alignment"
  - "missing views"
date: 2026-05-08
content_hash: d747a6b60a43ece2
---

# Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment

**Conference**: AAAI 2026
**arXiv**: [2601.09051](https://arxiv.org/abs/2601.09051)
**Code**: [Available](https://github.com/YMBest/DIMVC-HIA)
**Area**: LLM Evaluation
**Keywords**: Incomplete multi-view clustering, hierarchical imputation, energy-based model, contrastive alignment, missing views

## TL;DR

This paper proposes DIMVC-HIA, a deep incomplete multi-view clustering framework that integrates hierarchical imputation with dual alignment. The method first imputes missing cluster assignments and then imputes missing features in a coarse-to-fine manner, maintaining robust performance under high missing rates (up to 70%).

## Background & Motivation

Multi-view clustering (MVC) integrates heterogeneous multi-source data to provide richer insights than single-view approaches. However, **the assumption that all views are fully observable rarely holds in practice**—sensor failures, data corruption, and transmission errors commonly result in missing views.

Existing incomplete multi-view clustering (IMVC) methods fall into two categories:

**Imputation-based methods**: Reconstruct missing views but suffer from **error propagation**—poor imputation distorts structural patterns, which in turn degrades subsequent imputation quality, forming a vicious cycle.

**Imputation-free methods**: Learn a shared latent space directly from available views, but face challenges including instance misalignment, information imbalance, and representation uncertainty under high missing rates.

Core challenge: How to accurately impute missing views without introducing bias, while preserving cross-view semantic consistency and intra-cluster compactness?

## Method

### Overall Architecture

DIMVC-HIA comprises four key components:

1. **View-specific Autoencoders**: Independent encoder-decoder per view with a shared cluster predictor
2. **Hierarchical Imputation Module**: Imputes cluster assignments first, then latent features (coarse-to-fine)
3. **Energy-Based Semantic Alignment Module**: Enhances intra-cluster compactness via EBM
4. **Contrastive Assignment Alignment Module**: Enhances cross-view consistency and clustering confidence

### Key Designs

**1. View Feature Learning and Prediction**

Each view $v$ uses an independent AutoEncoder to extract latent features:

$$H_v = E_v(X_v; \phi_v^e), \quad \hat{X}_v = D_v(H_v; \phi_v^d)$$

Latent features are mapped to soft cluster assignments via a **shared** cluster predictor: $Q_v(i) = F(H_v(i); \vartheta) \in \mathbb{R}^K$.

**2. Hierarchical Imputation — Core Innovation**

**First Level: Cluster Assignment Imputation (Core of Coarse-to-Fine)**

Core insight: The soft cluster assignment space encodes richer semantic information than raw features and more directly reflects the underlying cluster structure. Cross-view information transfer is therefore performed first in the assignment space.

Steps:
- For each co-observed view pair $(v, v')$, extract soft cluster assignments of co-observed samples
- Compute cross-view similarity matrix: $S_{v,v'} = Q_v^{v,v'} (Q_{v'}^{v',v})^\top$
- Compute inter-view semantic alignment scores using **label-aware contrastive similarity** (excluding false negatives—samples from the same cluster but different indices)
- Rank reference views $\mathcal{R}_v$ in descending order of similarity
- For each missing sample, select the most semantically aligned available view for assignment imputation

$$Q_v^*(i) = \begin{cases} Q_v(i), & \text{if } G(i,v) = 1 \\ Q_{\pi_v^i}(i), & \text{otherwise} \end{cases}$$

**Second Level: Latent Feature Imputation**

Based on the imputed cluster assignments, missing features are reconstructed using intra-cluster statistics:
- Determine the most probable cluster label for each missing sample: $\hat{y}_v(i) = \arg\max_k Q_v^*(i, k)$
- Compute cluster prototypes (mean latent features of all available samples in the cluster)
- Use cluster prototypes as imputed values for missing features

$$H_v^*(i) = \begin{cases} H_v(i), & \text{if } G(i,v) = 1 \\ \mathcal{C}_v(\hat{y}_v(i)), & \text{otherwise} \end{cases}$$

**3. Energy-Based Semantic Alignment (EBM)**

A view-shared energy function $\mathcal{E}_{\theta_k}: \mathbb{R}^d \to \mathbb{R}^+$ is defined for each cluster, where lower energy indicates stronger compatibility with the cluster.

An "anchor" (the most reliable feature with minimum energy) is identified per cluster, and all intra-cluster features are encouraged to converge toward this anchor's energy level:

$$\mathcal{L}_\text{EBM}^k = \frac{1}{|\mathcal{H}_k|} \sum_{\mathbf{h} \in \mathcal{H}_k} |\mathcal{E}_{\theta_k}(\mathbf{h}) - \varepsilon_\text{min}^k|$$

Unlike conventional centroid-distance regularization, EBM allows flexible shaping of a continuous energy landscape.

**4. Contrastive Assignment Alignment (CAA)**

Two sub-objectives are included:

- **Contrastive alignment loss**: Pulls the cluster assignment distributions of the same sample across different views closer together
- **Entropy regularization**: Promotes balanced cluster assignments and prevents degenerate solutions (all samples assigned to one cluster)

$$\mathcal{L}_\text{CAA} = \frac{1}{2}\sum_v \sum_{v' \neq v} [\text{sim}(v,v') \cdot \mathcal{L}_\text{ca}^{v,v'} + \mathcal{L}_\text{reg}^{v,v'}]$$

Here $\text{sim}(v, v')$ serves as an adaptive weight, assigning higher alignment weight to view pairs with greater semantic alignment.

### Loss & Training

Overall objective: $\mathcal{L} = \mathcal{L}_\text{REC} + \alpha \cdot \mathcal{L}_\text{EBM} + \beta \cdot \mathcal{L}_\text{CAA}$

- AutoEncoders are pre-trained independently for 100 epochs (with $\mathcal{L}_\text{REC}$ only)
- Joint fine-tuning for 200 epochs (full loss)
- $\alpha = 0.1$, $\beta = 0.01$ (fixed across all datasets)
- lr = 0.0001, NVIDIA RTX 3080

## Key Experimental Results

### Main Results

**Table 1: Clustering Performance at Various Missing Rates (ACC/NMI/PUR)**

| Dataset | Method | η=0.1 ACC | η=0.3 ACC | η=0.5 ACC | η=0.7 ACC |
|--------|------|-----------|-----------|-----------|-----------|
| BDGP | DSIMVC | 98.00 | 96.08 | 93.56 | 91.12 |
| BDGP | **DIMVC-HIA** | **98.40** | **96.25** | **95.16** | **92.32** |
| MNIST-USPS | DCG | 99.05 | 97.48 | 96.09 | 92.58 |
| MNIST-USPS | **DIMVC-HIA** | **99.10** | **97.54** | **96.48** | **93.66** |
| Fashion | ProImp | 96.26 | 93.48 | 91.01 | 86.74 |
| Fashion | **DIMVC-HIA** | **98.84** | **97.16** | **96.51** | **95.27** |
| Handwritten | GIMVC | 92.14 | 93.58 | 90.73 | 86.10 |
| Handwritten | **DIMVC-HIA** | **96.85** | **96.35** | **95.15** | **94.05** |

**Table 2: Best Baseline Comparison at High Missing Rate η=0.7**

| Dataset | Best Baseline | Baseline ACC | DIMVC-HIA ACC | Gain |
|--------|-------------|-------------|---------------|------|
| BDGP | PMIMC | 91.72 | 92.32 | +0.60 |
| MNIST-USPS | ProImp | 93.42 | 93.66 | +0.24 |
| Fashion-MNIST | ProImp | 86.74 | **95.27** | **+8.53** |
| Handwritten | GIMVC | 86.10 | **94.05** | **+7.95** |

### Ablation Study

Impact of removing each component on Fashion-MNIST (η=0.5):

- Removing $\mathcal{L}_\text{CAA}$: **Largest** performance drop, confirming that contrastive assignment alignment is the most critical component
- Removing $\mathcal{L}_\text{EBM}$: Notable performance degradation, confirming the importance of energy-based alignment
- Removing $\mathcal{L}_\text{REC}$: Performance drop, indicating that reconstruction loss is indispensable for stable training

Hyperparameter sensitivity analysis: Performance remains stable without significant fluctuation within $\alpha \in [0.01, 0.10]$ and $\beta \in [0.01, 0.05]$.

### Key Findings

- **More pronounced advantage at high missing rates**: DIMVC-HIA outperforms ProImp by 8.53% on Fashion-MNIST at η=0.7, demonstrating the effectiveness of hierarchical imputation under severe missing scenarios
- **Minimal accuracy degradation**: On BDGP, accuracy drops only from 98.40 to 92.32 (~6%) as η increases from 0.1 to 0.7, compared to nearly 7% for DSIMVC
- **Stable convergence**: Loss curves converge smoothly after a rapid decrease in the first 25 epochs

## Highlights & Insights

1. **Elegant hierarchical imputation design**: Imputing cluster assignments at the semantic level first, then using the imputed assignments to guide feature-level imputation in a coarse-to-fine manner, avoiding the noise issues associated with direct feature imputation
2. **Label-aware contrastive similarity**: Excluding false negatives (same-cluster samples with different indices) improves the quality of cross-view semantic matching
3. **EBM as an alternative to distance regularization**: Modeling intra-cluster compactness via an energy landscape rather than simple centroid-distance constraints offers greater flexibility
4. **Bridging imputation and imputation-free paradigms**: Combines the advantages of both approaches—performing imputation while mitigating imputation errors through alignment

## Limitations & Future Work

- Cluster prototype imputation is non-parametric and may cause all missing samples to collapse toward cluster centroids in the feature space, limiting diversity
- Experiments are conducted on only 4 standard benchmarks with relatively small data scales (up to 10,000 samples)
- Scalability with increasing numbers of views is not discussed (at most 6 views evaluated)
- EBM maintains independent energy functions per cluster, which may increase computational burden when the number of clusters $K$ is large
- Missing patterns are assumed to be missing completely at random (MCAR); missing not at random (MNAR) scenarios are not considered

## Related Work & Insights

- Compared to SOTA methods such as DSIMVC and ProImp, DIMVC-HIA's hierarchical imputation strategy demonstrates clear advantages at high missing rates
- The application of EBM to clustering is relatively novel and transferable to other unsupervised learning tasks
- The false negative exclusion strategy in label-aware contrastive learning is worth adopting in other contrastive learning settings

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The hierarchical strategy of imputing assignments before features is novel; the application of EBM to clustering is innovative
- **Technical Depth**: ⭐⭐⭐⭐ — Four well-designed components with clear mathematical derivations
- **Experimental Thoroughness**: ⭐⭐⭐ — Four datasets with four missing rates, though data scales are relatively small
- **Value**: ⭐⭐⭐ — Practically meaningful for multimodal data fusion scenarios, though computational overhead may limit large-scale applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/llm_evaluation/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[AAAI 2026\] SpikCommander: A High-Performance Spiking Transformer with Multi-View Learning for Efficient Speech Command Recognition](spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)
- [\[AAAI 2026\] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)
- [\[AAAI 2026\] Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering](break_the_tie_learning_cluster-customized_category_relationships_for_categorical.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)

</div>

<!-- RELATED:END -->
