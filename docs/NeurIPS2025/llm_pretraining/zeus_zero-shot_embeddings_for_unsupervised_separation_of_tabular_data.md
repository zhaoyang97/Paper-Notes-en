---
title: >-
  [Paper Note] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data
description: >-
  [NeurIPS 2025][LLM Pretraining][Zero-shot Learning] ZEUS is the first zero-shot clustering method for tabular data. By pretraining a Transformer encoder on synthetic datasets…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Zero-shot Learning"
  - "Tabular Data"
  - "Clustering"
  - "Transformer"
  - "Prior-data Fitted Networks"
date: 2026-05-08
content_hash: 7fa6b2bf9fa63330
---

# ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data

**Conference**: NeurIPS 2025
**arXiv**: [2505.10704](https://arxiv.org/abs/2505.10704)  
**Code**: [GitHub](https://github.com/gmum/zeus)  
**Area**: Tabular Data Clustering / Unsupervised Learning
**Keywords**: Zero-shot Learning, Tabular Data, Clustering, Transformer, Prior-data Fitted Networks

## TL;DR

ZEUS is the first zero-shot clustering method for tabular data. By pretraining a Transformer encoder on synthetic datasets, it learns generalizable representations that enable high-quality clustering of new datasets in a single forward pass, requiring no additional training or hyperparameter tuning.

## Background & Motivation

Tabular data clustering has long been a core challenge in unsupervised learning. Unlike image data, tabular data lacks inherent spatial or semantic structure, and the notion of "similarity" varies greatly across datasets, making it difficult for clustering methods to generalize.

Existing methods suffer from two major limitations:

**Classical methods (e.g., k-means, GMM)** rely on predefined distance metrics, fail to capture complex nonlinear relationships, and perform poorly on high-dimensional heterogeneous data.

**Deep learning methods (e.g., DEC, IDEC)** can learn richer representations but are extremely sensitive to hyperparameters. In unsupervised settings, the absence of label signals makes tuning unreliable, leading to unstable performance and requiring per-dataset training.

ZEUS draws inspiration from the Prior-data Fitted Networks (PFNs) framework: given that TabPFN has demonstrated the potential of in-context learning on supervised tasks, can this paradigm be extended to unsupervised settings? The key challenges are: (1) how to generate synthetic training data with clear yet non-trivial cluster structure; and (2) how to encode clustering priors without label supervision.

## Method

### Overall Architecture

ZEUS operates in three stages:
- **Synthetic data generation**: A large collection of synthetic datasets with known cluster structure is sampled from latent variable models.
- **Pretraining**: A Transformer encoder is trained on synthetic data to learn mappings from inputs to a representation space amenable to clustering.
- **Inference**: A single forward pass is performed on a new dataset to obtain representations, followed by standard k-means to produce cluster assignments.

### Key Designs

1. **Probabilistic cluster representation learning**: Rather than directly predicting cluster labels, ZEUS trains an encoder $f_\theta$ to map input $\mathbf{x}$ to representation vectors $z(x_i) = f_\theta(\mathbf{x})_i \in \mathbb{R}^D$. Inspired by GMMs, $K$ cluster centers $c_k$ are defined in the representation space, and distances are converted to soft cluster membership probabilities via softmax:
    $p_k(x_i) = \frac{\hat{\pi}_k \exp(\alpha_k(x_i))}{\sum_{j=1}^K \hat{\pi}_j \exp(\alpha_j(x_i))}, \quad \alpha_k(x_i) = -\|z(x_i) - \hat{c}_k\|^2$
   where $\hat{\pi}_k$ is the class prior estimated from training labels and $\hat{c}_k$ is the mean representation of samples in cluster $k$. The training objective maximizes the log-likelihood of correct cluster assignments: $\mathcal{L}_{prob} = -\sum_i \log p_{y_i}(x_i)$. This design elegantly reformulates the unsupervised clustering problem as a supervised classification problem during pretraining.

2. **Synthetic data generation priors**: ZEUS samples training data from latent variable models, where each dataset is drawn from a mixture of $K$ (ranging from 2 to 10) components. Three types of priors are included:

    - **Gaussian mixtures**: Multivariate Gaussian distributions with sufficient separation constraints between components.
    - **Neural network transformations**: Gaussian samples are passed through a randomly initialized ResNet to produce non-Gaussian cluster shapes. ResNet is chosen because it defines invertible transformations that preserve cluster structure in the output.
    - **Categorical features**: Discrete features appended in one-hot encoded form.

   The core intuition is to train the model to "invert the data generation process," rather than optimizing arbitrary heuristic objectives as in k-means or DEC.

3. **Regularization**: Two auxiliary regularization terms supplement the primary loss $\mathcal{L}_{prob}$ to improve inference quality:

    - **Compactness regularization $\mathcal{L}_{cp}$**: Minimizes the distance from same-class representations to their cluster center, promoting intra-cluster compactness analogous to the within-cluster sum of squares in k-means:
    $\mathcal{L}_{cp} = \sum_k \sum_{i:y_i=k} \alpha_k(x_i)$
    - **Separation regularization $\mathcal{L}_{sep}$**: Maximizes pairwise distances between cluster centers with a truncation threshold $T$ to prevent unbounded divergence, avoiding center collapse:
    $\mathcal{L}_{sep} = -\sum_{k=1}^K \sum_{j=k+1}^K \min(\|\hat{c}_k - \hat{c}_j\|^2, T)$

### Loss & Training

The final loss is a weighted combination of three terms:
$$\mathcal{L} = \mathcal{L}_{prob} + \lambda_{cp}\mathcal{L}_{cp} + \lambda_{sep}\mathcal{L}_{sep}$$
where $\lambda_{cp} = \lambda_{sep} = 1$. The model uses a Transformer architecture with 12 attention blocks, 6 heads, and 512 hidden dimensions (following TabPFN). Training uses the Adam optimizer with a learning rate of 2e-5 and a cosine scheduler with warm-up. During pretraining, Gaussian and transformed data are sampled in equal proportions, with 1000 datasets sampled per epoch. At inference, numerical features are normalized to $[-1, 1]$, categorical features are one-hot encoded, inputs exceeding 30 dimensions are reduced via PCA, and k-means is applied to the normalized Transformer outputs.

## Key Experimental Results

### Main Results

Evaluation metric is ARI (×100) across 34 real OpenML datasets, 20 synthetic Gaussian datasets, and 20 synthetic transformed datasets, comparing against 11 methods.

| Dataset Group | ZEUS | DEC | k-means | GMM | TabPFN | Best Baseline |
|---|---|---|---|---|---|---|
| Real (OpenML) | **57.43** | 55.93 | 55.54 | 48.49 | 31.32 | 55.93 (DEC) |
| Syn. Gauss. | 89.03 | 89.35 | **89.90** | 76.93 | 55.97 | 89.90 (KM) |
| Syn. Transf. | **86.33** | 79.94 | 75.04 | 75.88 | 15.66 | 79.94 (DEC) |

| Dataset Group | ZEUS Avg. Rank | DEC Rank | KM Rank | Worst Method |
|---|---|---|---|---|
| Real | **4.13** | 4.69 | 4.69 | SCARF (8.85) |
| Syn. Gauss. | 2.92 | 3.23 | **2.65** | SCARF (11.00) |
| Syn. Transf. | **2.20** | 3.20 | 4.80 | SCARF (11.00) |

### Ablation Study

| Loss Combination | Real ARI | Syn. Gauss. ARI | Syn. Transf. ARI |
|---|---|---|---|
| $\mathcal{L}_{prob}$ only | 44.80 | 83.37 | 79.85 |
| $+\mathcal{L}_{sep}$ | 51.60 | 81.88 | 79.29 |
| $+\mathcal{L}_{cp}$ | 48.65 | **90.59** | **88.58** |
| $+\mathcal{L}_{sep}+\mathcal{L}_{cp}$ (full) | **57.43** | 89.03 | 86.33 |

Data prior ablation:

| Prior Combination | Real | Syn. Gauss. | Syn. Transf. |
|---|---|---|---|
| Gauss. + Cat. | 40.59 | **92.61** | 73.34 |
| NN-transf. + Cat. | 50.90 | 89.90 | **87.04** |
| Gauss. + NN-transf. | 52.00 | 75.25 | 71.29 |
| All three (ZEUS) | **57.43** | 89.03 | 86.33 |

### Key Findings

- ZEUS ranks first on the most challenging real and synthetic transformed datasets, and top-three on simple Gaussian data.
- $\mathcal{L}_{cp}$ is critical for synthetic data (+7pp); $\mathcal{L}_{sep}$ is important for real data (+9pp); their combination yields the largest improvement on real data.
- Combining all three priors is essential for generalization to real data: using only Gaussian priors achieves an ARI of merely 40.59 on real datasets.
- ZEUS inference time is nearly constant, only marginally slower than basic k-means and far faster than deep clustering methods.
- Brier score evaluation confirms excellent calibration quality of ZEUS's probabilistic assignments, with a substantial lead on synthetic transformed data.

## Highlights & Insights

- **Paradigm innovation**: Framing clustering as "inverting the data generation process" rather than optimizing heuristic objectives aligns with the philosophy of generative modeling.
- **Theoretical grounding**: ZEUS is shown to conform to the PFN framework, implicitly performing approximate Bayesian inference.
- **Practical utility**: Plug-and-play, single forward pass, no hyperparameter tuning — directly addressing the poor usability of deep clustering methods.
- The use of known labels during pretraining to estimate cluster centers and priors elegantly sidesteps the hyperparameter sensitivity inherent to unsupervised settings.

## Limitations & Future Work

- Inherits TabPFN's limitations: maximum 30 input features (PCA required beyond this) and maximum 2000 samples.
- Clustering quality is strongly dependent on the synthetic data prior design; performance may degrade when real data distributions diverge significantly from the priors.
- The model itself does not perform clustering — it only provides a representation space, and an external algorithm such as k-means is still required.
- Future work may explore richer data generation priors and scaling to larger datasets (drawing on ideas from TabPFN v2).

## Related Work & Insights

- Integration with TabPFN v2 could overcome current scale limitations.
- The prior design methodology could extend to other unsupervised tabular tasks such as anomaly detection and missing value imputation.
- For domain-specific data, customized prior distributions could further improve performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First zero-shot clustering method for tabular data; clear paradigm innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 34 real datasets + 40 synthetic datasets with comprehensive ablations; large-scale scenarios remain unvalidated.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical structure, rigorous theoretical derivations, and well-organized experiments.
- **Value**: ⭐⭐⭐⭐ Represents a new benchmark in tabular data clustering with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[AAAI 2026\] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](../../AAAI2026/llm_pretraining/granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](predict_training_data_quality_via_its_geometry_in_metric_space.md)

</div>

<!-- RELATED:END -->
