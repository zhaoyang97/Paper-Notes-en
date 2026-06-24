---
title: >-
  [Paper Note] S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling
description: >-
  [ACL 2025][Topic modeling] Proposes S2WTM, a topic model based on a spherical sliced-Wasserstein autoencoder. It aligns the aggregated posterior and prior distributions on a hyperspherical latent space, effectively avoiding the posterior collapse problem of VAEs while outperforming existing SOTAs in topic coherence and diversity.
tags:
  - "ACL 2025"
  - "Topic modeling"
  - "Spherical latent space"
  - "Wasserstein distance"
  - "Posterior collapse"
  - "von Mises-Fisher distribution"
date: 2026-05-08
content_hash: aa03109637ec362a
---

# S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling

**Conference**: ACL 2025  
**arXiv**: [2507.12451](https://arxiv.org/abs/2507.12451)  
**Code**: Yes ([https://github.com/AdhyaSuman/S2WTM](https://github.com/AdhyaSuman/S2WTM))  
**Area**: Others  
**Keywords**: Topic modeling, Spherical latent space, Wasserstein distance, Posterior collapse, von Mises-Fisher distribution  

## TL;DR

Proposes S2WTM, a topic model based on a spherical sliced-Wasserstein autoencoder. It aligns the aggregated posterior and prior distributions on a hyperspherical latent space, effectively avoiding the posterior collapse problem of VAEs while outperforming existing SOTAs in topic coherence and diversity.

## Background & Motivation

Topic modeling aims to discover latent topics from document corpora. From LDA to neural topic models (VAE-NTMs), the field has made significant progress. However, current methods face two core challenges:

**Challenge 1: Limitations of Euclidean Latent Spaces**
- The "soap bubble effect" of Gaussian distributions in high-dimensional spaces: probability mass concentrates on the surface of the hypersphere rather than around the mean.
- Euclidean distances tend to be uniform in high dimensions, leading to a degradation in discriminative capacity (curse of dimensionality).
- In actual text data, **directional similarity** (cosine similarity) is more meaningful than distance-based similarity.
- This motivates the use of the von Mises-Fisher (vMF) distribution to model latent representations on a **hypersphere**.

**Challenge 2: Posterior Collapse**
- The KL divergence regularization term in VAEs can be decomposed into a mutual information term and an aggregated posterior-prior distance term.
- Minimizing KL divergence simultaneously reduces mutual information, causing latent representations to lose input details.
- Existing vMF-VAE models (e.g., vONT) still employ KL divergence and fail to completely avoid posterior collapse.

The authors' solution is a two-pronged approach: **spherical latent space** + **replacing KL divergence with Wasserstein distance**. Specifically, the Spherical Sliced-Wasserstein (SSW) distance is used to align the aggregated posterior and hyperspherical prior, preserving spherical geometry while avoiding posterior collapse.

## Method

### Overall Architecture

S2WTM adopts the Wasserstein Autoencoder (WAE) framework, with its core components including:
1. **Encoder**: Maps document BoW representations to latent representations on a hypersphere.
2. **Decoder**: Reconstructs the document-word distribution from the latent representation.
3. **Regularization**: Aligns the aggregated posterior with the hyperspherical prior using the SSW distance.
4. **Prior Selection**: Supports three hyperspherical priors (vMF / MvMF / Uniform distribution).

### Key Designs

1. **Hyperspherical Prior Distribution Selection**

    - **Function**: Provides structured prior constraints for the latent space.
    - Three options:
        - **vMF Distribution**: A unimodal directional distribution parameterized by mean direction $\mu$ and concentration parameter $\kappa$.
        - **MvMF Distribution**: A mixture of multiple vMFs, which can model more complex multi-modal structures.
        - **Uniform Distribution $U(S^{K-1})$**: A completely unbiased prior, sampled from a standard Gaussian and then L2-normalized.
    - **Design Motivation**: Treats prior selection as a hyperparameter, allowing different datasets to select the most suitable prior structure.

2. **Deterministic Encoder**

    - **Function**: Maps the document BoW representation $x \in R^V$ to the hypersphere $z \in S^{K-1}$.
    - Network Architecture: Linear(V,H') → Dropout → ReLU → Linear(H',H'') → Dropout → ReLU → Linear(H'',K) → **L2Norm**
    - **Key**: The final L2 normalization step projects the output onto the unit hypersphere.
    - **Design Motivation**: Uses a deterministic encoder (instead of a stochastic encoder in VAE) to learn the aggregated posterior, which, coupled with SSW regularization, avoids posterior collapse.

3. **Spherical Sliced-Wasserstein (SSW) Distance Regularization**

    - **Function**: Measures the divergence between the aggregated posterior $q(\theta)$ and the prior $p(\theta)$ on the hypersphere.
    - **Mechanism**:
        - The computational complexity of the standard Wasserstein distance is $O(n^3 \log n)$.
        - Sliced Wasserstein (SW) approximates this by projecting onto 1D, but is unsuitable for spherical data.
        - SSW utilizes the **spherical Radon transform** instead of linear projection to project the distribution on the sphere onto great circles.
        - It then efficiently computes the Wasserstein distance on 1D via sorting.
    - **Approximate Computation**: $SSW_2^2(q_\theta, p_\theta) \approx \frac{1}{M}\sum_{i=1}^{M} W_2^2(\tilde{R}_i q_\theta, \tilde{R}_i p_\theta)$, where $M$ is the number of random projections.
    - **Design Motivation**: Replaces KL divergence with SSW directly operating on the aggregated posterior (rather than the single-sample posterior), fundamentally avoiding posterior collapse.

4. **Training Objective**

    - **Total Loss**: $\mathcal{L} = \mathcal{L}_{RL} + \lambda \mathcal{L}_{OT}$
    - **Reconstruction Loss** $\mathcal{L}_{RL}$: Cross-entropy between input $x$ and reconstruction $\hat{x}$.
    - **Regularization Loss** $\mathcal{L}_{OT}$: $SSW_2^2(q_\theta, p_\theta)$
    - $\lambda$ is a hyperparameter balancing reconstruction quality and latent space regularization.

### Loss & Training

- Uses the Adam optimizer; the learning rate requires tuning.
- **Key Hyperparameters**: Number of projections $M$ (500–8000), prior type, batch size, dropout rate, and $\lambda$.
- Optimal priors vary across datasets: 20NG/BBC perform best with vMF, M10/Bio/DBLP with the Uniform distribution, and SS/Pascal with MvMF.

## Key Experimental Results

### Main Results (Topic Coherence, Median, 5 Runs)

| Model | 20NG-NPMI | BBC-NPMI | M10-NPMI | SS-NPMI | Pascal-NPMI | Bio-NPMI | DBLP-NPMI |
|------|-----------|----------|----------|---------|-------------|----------|-----------|
| LDA | 0.092 | 0.076 | -0.047 | -0.066 | -0.072 | 0.019 | 0.015 |
| NMF | 0.118 | 0.065 | 0.050 | 0.019 | -0.042 | 0.100 | 0.016 |
| ProdLDA | 0.107 | 0.010 | 0.027 | -0.009 | -0.023 | 0.107 | -0.065 |
| CombinedTM | 0.107 | 0.017 | 0.059 | 0.018 | -0.002 | 0.133 | -0.065 |
| WTM | 0.046 | -0.006 | -0.052 | -0.013 | -0.089 | 0.052 | -0.044 |
| vONT | 0.045 | -0.001 | -0.053 | -0.015 | -0.090 | 0.052 | -0.043 |
| ECRTM | -0.089 | 0.170 | -0.445 | -0.333 | -0.414 | -0.421 | -0.248 |
| **S2WTM** | **0.167** | **0.252** | **0.101** | **0.146** | **0.045** | **0.191** | **0.133** |

S2WTM achieves the highest NPMI across all 7 datasets, outperforming baselines by a large margin.

### Ablation Study (Prior Type Comparison, NPMI)

| Prior Type | 20NG | BBC | M10 | SS | Pascal | Bio | DBLP |
|---------|------|-----|-----|----|----|-----|------|
| vMF | **Best** | **Best** | Moderate | Moderate | Moderate | Moderate | Moderate |
| MvMF | Moderate | Moderate | Moderate | **Best** | **Best** | Moderate | Moderate |
| Uniform | Moderate | Moderate | **Best** | Moderate | Moderate | **Best** | **Best** |

**Key Findings**: Different datasets suit different prior distributions; treating the prior as a hyperparameter is a sound design design.

### Key Findings

1. **Completely Outperforming SOTA**: S2WTM achieves optimal or suboptimal performance across all coherence and diversity metrics on 7 datasets.
2. **Underperformance of WTM and vONT**: Indicates that utilizing only a spherical latent space (vONT) or only Wasserstein distance (WTM) is insufficient; combining both is essential.
3. **High Instability of ECRTM on Multiple Datasets**: Though it performs best on BBC, its scores turn significantly negative on other datasets.
4. **Significant Impact of Prior Selection**: vMF is suitable for datasets with clear topic structures, MvMF for multimodal data, and the Uniform distribution for scattered/dispersed topics.
5. **Effective Mitigation of Posterior Collapse**: Under SSW regularization, the aggregated posterior is effectively aligned with the prior, avoiding the typical degradation of VAEs.

## Highlights & Insights

- **Solid Theoretical Motivation**: Starting from the theoretical analysis of the "soap bubble effect" and posterior collapse, the combination of a spherical latent space and SSW distance is naturally derived.
- **Organic Combination of Two Key Ideas**: Spherical geometry (vMF prior) and optimal transport (SSW distance) complement each other's limitations.
- **Simplicity and Effectiveness**: The deterministic encoder, L2 normalization, and SSW regularization are not complex to implement.
- **First to Introduce SSW Distance to Topic Modeling**: Demonstrates its advantages over KL divergence and standard SW distance.

## Limitations & Future Work

1. **Limitations of BoW Representation**: Still relies on traditional bag-of-words (BoW) representations, failing to exploit contextualized embeddings from pretrained language models.
2. **Selection of the Number of Projections $M$**: The range of $M$ is wide (500–8000), and too many projections increase computational overhead.
3. **Limited to Small-Scale Datasets**: The largest dataset (DBLP) contains only 54K documents, leaving its performance on million-scale corpora unexplored.
4. **Lack of Comparison with Latest LLM-based Topic Models**: Such as BERT-based approaches like BERTopic.
5. **Number of Components $T$ in MvMF is a Hyperparameter Requiring Tuning**: This increases hyperparameter tuning complexity.

## Related Work & Insights

- SAM (Reisinger et al., 2010) pioneered spherical topic modeling; S2WTM is a modern neural network variant in this direction.
- WAE (Tolstikhin et al., 2018) provides a theoretical framework to replace VAE; S2WTM extends it to spheres.
- SSW distance (Bonet et al., 2023) is the core tool, applied here to topic modeling for the first time.
- Inspires the use of SSW as an alternative to KL divergence in other latent variable models to alleviate posterior collapse.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of spherical WAE and SSW distance is a first in topic modeling, with the choice of three priors enhancing flexibility.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Highly comprehensive coverage with 7 datasets, 12+ baselines, and 4 categories of metrics (coherence, diversity, downstream tasks, LLM evaluation).
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous and clear mathematical derivations; the step-by-step introduction from Wasserstein to SW to SSW makes it easy to follow.
- **Value**: ⭐⭐⭐ — Solid work in the topic modeling field, though the field itself is receiving declining attention, potentially limiting its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)
- [\[ICLR 2026\] Mixed-Curvature Tree-Sliced Wasserstein Distance](../../ICLR2026/others/mixed-curvature_tree-sliced_wasserstein_distance.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](../../ICML2025/others/lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[ACL 2025\] Persistent Homology of Topic Networks for the Prediction of Reader Curiosity](persistent_homology_of_topic_networks_for_the_prediction_of_reader_curiosity.md)

</div>

<!-- RELATED:END -->
