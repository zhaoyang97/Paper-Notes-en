---
title: >-
  [Paper Note] Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning
description: >-
  [ICML 2025][LLM Pretraining][data mixing] Introduces the Chameleon framework, which utilizes kernel ridge leverage scores (KRLS) to quantify the importance of each training domain in the embedding space of a proxy model. It achieves comparable or superior data mixing performance at only 1/10 of DoReMi's computational cost, eliminates the need to retrain the proxy model when introducing new domains, and unifiedly handles both pretraining and finetuning scenarios.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "data mixing"
  - "domain reweighting"
  - "kernel ridge leverage scores"
  - "pretraining"
  - "finetuning"
date: 2026-05-08
content_hash: 3e55394286da5daa
---

# Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning

**Conference**: ICML 2025  
**arXiv**: [2505.24844](https://arxiv.org/abs/2505.24844)  
**Code**: [GitHub - Chameleon](https://github.com/LIONS-EPFL/Chameleon)  
**Area**: LLM Pretraining / Data Mixing  
**Keywords**: data mixing, domain reweighting, kernel ridge leverage scores, pretraining, finetuning

## TL;DR

Introduces the Chameleon framework, which utilizes kernel ridge leverage scores (KRLS) to quantify the importance of each training domain in the embedding space of a proxy model. It achieves comparable or superior data mixing performance at only 1/10 of DoReMi's computational cost, eliminates the need to retrain the proxy model when introducing new domains, and unifiedly handles both pretraining and finetuning scenarios.

## Background & Motivation

**Background**: The data sources for pretraining LLMs are diverse (web text, academic papers, code, books, etc.), and the mixing ratio of different domains significantly affects model generalization performance. Recently, domain reweighting methods have emerged as a core research direction in data mixing. DoReMi and DoGE are representative works that use small proxy models to derive domain weights for training larger models.

**Limitations of Prior Work**: DoReMi requires training two models (a reference model and a proxy model) and optimizes domain weights using Group DRO. DoGE tracks domain-specific gradients, which incurs high computational costs. The core limitations of both methods are: (1) high computational expense—proxy model training itself is equivalent to a full LM training; (2) sensitivity to domain changes—when datasets add or remove domains, the proxy model must be retrained from scratch; (3) lack of support for finetuning—these methods are designed only for pretraining, leaving a gap in domain reweighting for finetuning scenarios.

**Key Challenge**: Obtaining high-quality domain weights requires iterative optimization of the proxy model, but this iterative optimization itself is the primary computational bottleneck. Furthermore, domain weights are coupled with the training process of the proxy model, making it impossible to reuse prior computations when domains change.

**Goal**: To design a method that decouples domain weight calculation from proxy model training, satisfying three objectives: (1) obtaining high-quality domain weights at a low computational cost; (2) avoiding retraining when new domains are introduced; and (3) unifiedly handling both pretraining and finetuning.

**Key Insight**: Starting from the intrinsic characteristics of the data itself (data-centric) rather than the model optimization process. The core observation is that if a proxy model is already available, domain relationships can be directly extracted from the embedding representations of the data without indirect derivation through the training process.

**Core Idea**: Directly quantify the uniqueness and representativeness of each domain using kernel ridge leverage scores in the embedding space of a proxy model. Inverse KRLS (emphasizing common domains) is used during pretraining, while standard KRLS (emphasizing unique domains) is used during finetuning.

## Method

### Overall Architecture

The pipeline of Chameleon: (1) Train a small proxy model with uniform weights; (2) extract intermediate layer embeddings of each domain's data, averaging them to obtain domain embedding vectors $x_i$; (3) construct the domain affinity matrix $\Omega_\mathcal{D} = XX^\top$, where $X = [x_1, ..., x_k]^\top$; (4) calculate the KRLS score $S_\lambda(D_i) = [\Omega_\mathcal{D}(\Omega_\mathcal{D} + k\lambda I)^{-1}]_{ii}$; (5) use the inverse score $\alpha^{PT} \propto \text{softmax}(S_\lambda^{-1})$ for pretraining, and the positive score $\alpha^{FT} \propto \text{softmax}(S_\lambda)$ for finetuning.

### Key Designs

1. **Domain Embeddings and Affinity Matrix**:

    - **Function**: Represents each training domain as a vector and quantifies semantic relationships between domains.
    - **Mechanism**: For data in domain $D_i$, the domain embedding $x_i = \frac{1}{|D_i|}\sum_{a \in D_i} h_{\theta_p}^{(L)}(a)$ is obtained by averaging the $L$-th layer hidden states of the proxy model. Each element of the domain affinity matrix $\Omega_\mathcal{D} = XX^\top$, $\Omega_{ij} = x_i^\top x_j$, measures the similarity between domain $i$ and domain $j$ in the embedding space. UMAP visualization shows that semantically similar domains (e.g., CC and C4) cluster together in the embedding space, while unique domains (e.g., ArXiv, Github) are separated.
    - **Design Motivation**: Domain embeddings are the most natural domain-level representation—computationally orders of magnitude cheaper than sample-level embeddings (requiring only one forward pass + average) while retaining the core information of cross-domain relationships.

2. **Kernel Ridge Leverage Scores (KRLS) for Quantifying Domain Uniqueness**:

    - **Function**: Computes a scalar score for each domain, representing how "unique" the domain is within the overall embedding space.
    - **Mechanism**: KRLS is defined as $S_\lambda(D_i) = [\Omega_\mathcal{D}(\Omega_\mathcal{D} + k\lambda I)^{-1}]_{ii}$, which is the diagonal element of the hat matrix in kernel ridge regression. Domains with high KRLS occupy unique locations in the embedding space and cannot be well-approximated by linear combinations of other domains; domains with low KRLS exhibit high redundancy and can be adequately represented by other domains. The regularization parameter $\lambda$ controls the strictness of uniqueness determination.
    - **Design Motivation**: KRLS has a solid theoretical foundation in statistics—its inverse is proportional to the Christoffel function, which precisely characterizes the local density of the data distribution in the feature space. This provides theoretical justification for domain weights rather than relying on heuristic rules.

3. **Dual Strategy of Inverse KRLS for Pretraining and standard KRLS for Finetuning**:

    - **Function**: Adaptively adjusts the direction of domain weights based on the different objectives of each training stage.
    - **Mechanism**: The goal of pretraining is to learn general knowledge—domains in densely populated regions that are widely shared should be sampled more (low KRLS = high representativeness). Thus, inverse KRLS is used as the weight $\alpha^{PT} = \text{softmax}(S_\lambda^{-1})$. The goal of finetuning is to learn domain-specific knowledge—unique and distinct domains should be sampled more (high KRLS = high uniqueness). Thus, KRLS is directly used as the weight $\alpha^{FT} = \text{softmax}(S_\lambda)$.
    - **Design Motivation**: Pretraining and finetuning have completely different requirements for the same data distribution, but DoReMi/DoGE are designed solely for pretraining. The duality of KRLS (positive and inverse measuring uniqueness and representativeness, respectively) naturally yields a unified framework.

### Loss & Training

The proxy model is trained using the standard language model cross-entropy loss with uniform domain weights. KRLS computation involves no additional training and requires only matrix operations, with a complexity of $O(k^3)$ (where $k$ is the number of domains, typically $<20$). The entire pipeline makes no modifications to the training process of the proxy model.

## Key Experimental Results

### Main Results Table (SlimPajama, 684M parameters, pretraining domain perplexity)

| Domain | Uniform | DoReMi | DoGE | **Chameleon** | RegMix |
|----|---------|--------|------|-------------|--------|
| ArXiv | 8.16 | 9.16 | 9.07 | **8.31** | 11.35 |
| Book | 42.55 | 46.48 | 40.30 | **39.23** | 41.52 |
| CC | 45.26 | **40.62** | 38.99 | 40.11 | 37.32 |
| C4 | 49.00 | 43.92 | 40.65 | 42.59 | 43.85 |
| Github | **3.99** | 4.10 | 4.09 | 4.20 | 4.99 |
| StackExchange | 7.99 | 8.35 | **7.39** | 7.94 | 10.63 |
| Wikipedia | **12.42** | 10.78 | 15.74 | 13.90 | 20.88 |
| **Average PPL↓** | 24.20 | 23.34 | 22.32 | **22.31** | 24.36 |
| **Computational Cost (FLOPs)** | 0 | 1.34×10¹⁸(10×) | 6.68×10¹⁷(5×) | **1.36×10¹⁷(1×)** | 1.20×10¹⁸(9×) |

### Downstream Inference Task Accuracy (Average over 13 benchmarks)

| Method | ARC-E | COPA | HellaSwag | Lambada | PiQA | WinoGrande | **Average↑** |
|------|-------|------|-----------|---------|------|------------|---------|
| Uniform | 36.8 | 55.7 | 26.5 | 13.5 | 59.2 | 50.5 | 37.9 |
| DoReMi | 37.6 | 59.3 | 27.0 | 13.6 | 59.5 | 51.3 | 38.4 |
| DoGE | 38.0 | 62.3 | 27.2 | 14.7 | 60.0 | 52.0 | 39.4 |
| **Chameleon** | 37.8 | 61.9 | 27.1 | **15.1** | **60.5** | **52.1** | **39.6** |
| RegMix* | 39.1 | 63.0 | 27.0 | 16.5 | 57.6 | 50.9 | 39.3 |

*RegMix utilizes downstream task information; other methods do not.

### Key Findings

- **5x–10x Reduction in Computational Cost**: Chameleon requires only 1.36×10¹⁷ FLOPs (approximately 1/10 of DoReMi's cost) while achieving the best average PPL (22.31 vs. DoGE's 22.32).
- **Highest Downstream Inference Accuracy**: On the average of 13 benchmarks (39.6%), Chameleon outperforms all methods that do not use downstream information, and even slightly exceeds RegMix (39.3%), which utilizes downstream info.
- **Zero Retraining under Domain Changes**: When adding new domains, one only needs to compute the new domain embeddings and update the affinity matrix, without retraining the proxy model. Experiments show that even if the number of domains doubles, Chameleon still outperforms baselines that require full retraining, at only 1% of the retraining cost.
- **Consistent Improvement in Finetuning Scenarios**: In finetuning domain reweighting experiments, KRLS weights consistently improved the test perplexity across all domains, validating the effectiveness of the dual strategy.

## Highlights & Insights

- **The application of KRLS to data mixing** represents a key methodological innovation—introducing the mature statistical tool of leverage scores to LLM training, providing a theoretically grounded calculation method for domain weights rather than relying on heuristic searches.
- **"Requiring no training signals from the proxy model" is a fundamental simplification over DoReMi**: Domain weights are determined entirely by the embedding geometry of the data, independent of dynamic signals like loss or gradients from the training process.
- **The dual strategy for pretraining/finetuning** elegantly leverages the complementary meanings of positive and inverse KRLS scores: inverse score $\rightarrow$ representativeness $\rightarrow$ general knowledge $\rightarrow$ pretraining; positive score $\rightarrow$ uniqueness $\rightarrow$ specialized knowledge $\rightarrow$ finetuning.

## Limitations & Future Work

- The selection of the embedding layer $L$ affects the quality of domain embeddings; currently, it is selected empirically based on experiments, lacking theoretical guidance.
- Only a linear kernel ($\kappa(x_i, x_j) = x_i^\top x_j$) is used. Although the model itself provides non-linearity, non-linear kernels might potentially capture more complex cross-domain relationships.
- Scalability under ultra-large numbers of domains ($k > 100$) has not been validated, though the $O(k^3)$ complexity is negligible when $k < 20$.
- Averaging domain embeddings might lose diversity information of in-domain distributions—if a domain contains multiple sub-clusters, the mean representational vector might not be representative.

## Related Work & Insights

- **vs. DoReMi**: DoReMi dynamically adjusts domain weights during proxy training using Group DRO, which causes domain weights to fluctuate significantly during training. In contrast, Chameleon computes static weights once from embedding geometry, which is stable and computationally efficient.
- **vs. DoGE**: DoGE tracks domain-specific gradients to capture domain contributions to training, which incurs high computational overhead. Chameleon uses KRLS to directly measure domain uniqueness from data representations, requiring no gradient computation.
- **vs. Data Pruning methods (e.g., DSIR)**: Data pruning operates at the sample level to select samples that match a target distribution. Chameleon operates at the domain level to adjust sampling probabilities of domains. These two paradigms are complementary and can be used jointly.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of KRLS in data mixing, with a clear theoretical motivation
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of pretraining, finetuning, and domain-changing scenarios, with extensive comparisons to mainstream baselines
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, coherent methodological derivations, and highly informative figures/tables
- Value: ⭐⭐⭐⭐⭐ Reduces the computational cost of data mixing by an order of magnitude, offering extraordinarily high practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](../../ICLR2026/llm_pretraining/scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](../../ICML2026/llm_pretraining/ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[ICML 2025\] How to Synthesize Text Data without Model Collapse?](how_to_synthesize_text_data_without_model_collapse.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](metadata_conditioning_accelerates_language_model_pre-training.md)

</div>

<!-- RELATED:END -->
