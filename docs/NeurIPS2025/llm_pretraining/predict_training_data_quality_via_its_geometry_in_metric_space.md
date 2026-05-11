---
title: >-
  [Paper Note] Predict Training Data Quality via Its Geometry in Metric Space
description: >-
  [NeurIPS 2025][LLM Pretraining][Persistent Homology] This paper proposes a training data diversity metric based on Persistent Homology (PH)…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Persistent Homology"
  - "Data Diversity"
  - "Training Data Quality"
  - "Hill Numbers"
  - "Topological Data Analysis"
date: 2026-05-08
content_hash: da7c01cf12fec447
---

# Predict Training Data Quality via Its Geometry in Metric Space

**Conference**: NeurIPS 2025
**arXiv**: [2510.15970](https://arxiv.org/abs/2510.15970)
**Code**: None
**Area**: Data Quality / Topological Data Analysis
**Keywords**: Persistent Homology, Data Diversity, Training Data Quality, Hill Numbers, Topological Data Analysis

## TL;DR

This paper proposes a training data diversity metric based on Persistent Homology (PH), demonstrating that geometric and topological structural features of data can effectively predict model performance, outperforming traditional entropy-based metrics such as Vendi Score.

## Background & Motivation

High-quality training data is fundamental to machine learning. Recent studies have shown a strong correlation between training data diversity and model performance. However, existing diversity metrics (e.g., Vendi Score) are based on entropy or eigenvalue spectra, measuring distributional uniformity without capturing the geometric structure of data.

Core problems:
- Beyond class balance, is more data always better if all data points are equally important?
- When augmenting a dataset, which type of data is more valuable — samples that contract, expand, maintain, or shift the data range?
- How do the geometric properties of data affect model generalization?

These questions motivate the authors to adopt a Topological Data Analysis (TDA) perspective, leveraging persistent homology to quantify the structural diversity of training data and establish its connection to model performance.

## Method

### Overall Architecture

1. Construct Vietoris-Rips complexes from the dataset in metric space.
2. Extract topological features via persistent homology — connected components ($H_0$) and loops ($H_1$).
3. Define diversity metrics based on persistence lifetimes.
4. Validate the association between metrics and model performance through transfer learning experiments.

### Key Designs

**PH-based Diversity Metric**: Given a dataset $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ with pairwise distance matrix $D$, a Vietoris-Rips filtration complex is constructed, yielding persistence interval sets $\mathcal{B}_k = \{(b_i, d_i)\}_{i=1}^{m_k}$. Each interval has lifetime $l_i = d_i - b_i$ and normalized weight $p_i = l_i / L$.

The following metrics are defined:
- **Rényi Persistence Entropy**: $\text{PE}_k^{(q)} = \frac{1}{1-q} \log(\sum_{i=1}^{m_k} p_i^q)$
- **Shannon Persistence Entropy** (as $q \to 1$): $\text{PE}_k^{(1)} = -\sum_{i=1}^{m_k} p_i \log p_i$
- **PH-based Hill Numbers**: $\text{PEH}_k^q(X) = \exp(\text{PE}_k^{(q)})$

$H_0$ features capture connected components (cluster structure), while $H_1$ features capture loop structures (higher-order geometric information).

**Axiomatic Validation**: The PH metrics are proven to satisfy four core axioms of diversity:
- **Effective Size**: Diversity is minimized when data points coincide and increases as they disperse.
- **Twin Property**: Adding a duplicate data point does not change diversity.
- **Multi-scale**: Different homological dimensions capture features at multiple scales.
- **Symmetry**: Invariant to the ordering of data points.

**Subset Construction Strategies**: Samples are ranked by their maximum distance to all other points, and three types of balanced subsets are constructed:
- **Closest**: Sampled from the lower half of the distance ranking (core samples).
- **Farthest**: Sampled from the upper half of the distance ranking (peripheral samples).
- **Random**: Uniformly sampled at random.

### Loss & Training

Validation is performed via transfer learning: a dropout layer and a softmax classifier are appended to BERT-base, trained for 8 epochs with a learning rate of 1e-6 and a dropout rate of 10%. Each training set is fixed at 500 samples (250 per class), evaluated across multiple text classification datasets.

## Key Experimental Results

### Main Results — PH Diversity vs. Model Performance

| Subset Type | Accuracy (avg ± std) | PEH₀¹ | PEH₀²⁰ | H₀ min | PEH₁¹ | PEH₁²⁰ | H₁ mean | Vendi Score |
|-------------|----------------------|--------|---------|--------|--------|---------|---------|-------------|
| Closest | 0.836 ± 0.021 | 489 | 347 | 0.0215 | 376 | 126 | 0.0025 | 1.143 |
| Farthest | 0.832 ± 0.014 | 478 | 244 | 0.0191 | 291 | 86 | 0.0029 | 1.160 |
| Random | **0.845 ± 0.013** | 485 | 287 | **0.0234** | 331 | 123 | 0.0028 | 1.151 |

### Ablation Study — Consistency Across Datasets

| Feature | Correlation with Model Accuracy | Notes |
|---------|----------------------------------|-------|
| PH-based ($H_0$) metric | Positive correlation ✅ | Higher connectivity diversity → better performance |
| PH-based ($H_1$) metric | Positive correlation ✅ | Higher loop structure diversity → better performance |
| Vendi Score | **Negative correlation** ❌ | Higher distributional entropy → lower performance |
| $H_0$ minimum | Negatively correlated with accuracy std | Greater geometric diversity → more stable training |

Experiments cover five datasets: Complaints (TC), SUBJectivity (SUBJ), SentEval (SE), Arxiv-10, and Medical.

### Key Findings

1. **PH Metrics vs. Vendi Score**: PH metrics are positively correlated with model accuracy, whereas Vendi Score exhibits a negative correlation. Entropy-based distributional metrics cannot reliably predict data quality.
2. **Random Subsets Are Optimal**: Among the three construction strategies, random subsets achieve the best accuracy and stability, as they strike a balance in both $H_0$ (moderate cluster separation) and $H_1$ (stable loop structures).
3. **Value of Higher-Order Topological Features**: $H_1$ features play a key role in capturing meaningful structural patterns, beyond $H_0$ connected components alone.
4. **Data Efficiency**: Using only 6%–19% of the original data achieves 91%–98.6% of full-dataset fine-tuning performance, indicating that structural diversity matters more than data volume.

## Highlights & Insights

- **New Perspective**: Persistent homology is introduced from TDA into data quality assessment, providing structural information beyond distributional metrics.
- **Counter-intuitive Finding**: Vendi Score is negatively correlated with performance, challenging the intuition that higher distributional entropy implies better data.
- **Practical Guidance**: High-quality datasets should exhibit well-separated clusters (high $H_0$ min) and moderately stable loop structures (moderate $H_1$ mean), avoiding extreme redundancy or sparsity.
- **Theoretical Completeness**: PH metrics are rigorously proven to satisfy the axiomatic definition of diversity.

## Limitations & Future Work

- Experiments are limited to transfer learning for BERT-based text classification; generalization to other modalities (image, multimodal) and larger-scale settings remains to be validated.
- The computational complexity of persistent homology grows with data size (Vietoris-Rips complexes are generally exponential in cost), requiring approximate algorithms for large-scale application.
- Only $H_0$ and $H_1$ are considered; the role of higher-dimensional features (e.g., $H_2$ voids) remains unexplored.
- The experimental scale of 250 samples per class is relatively small; large-scale validation is needed to strengthen the conclusions.

## Related Work & Insights

- Contrasted with Vendi Score (entropy metric based on eigenvalue spectra) and MAGAREA/MAGDIFF (magnitude-based metrics).
- Leverages the fundamental connection between persistent homology and agglomerative hierarchical clustering to define diversity metrics.
- Future directions: integrating topological features directly into data augmentation, selection, and synthesis pipelines to enable topology-guided data engineering.

## Rating

- Novelty: ⭐⭐⭐⭐ (PH-based data quality assessment is a novel perspective)
- Technical Depth: ⭐⭐⭐⭐ (complete axiomatic proof, solid methodology)
- Experimental Thoroughness: ⭐⭐⭐ (limited scale, single modality)
- Practicality: ⭐⭐⭐ (high computational overhead, but a valuable direction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](../../ICCV2025/llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)

</div>

<!-- RELATED:END -->
