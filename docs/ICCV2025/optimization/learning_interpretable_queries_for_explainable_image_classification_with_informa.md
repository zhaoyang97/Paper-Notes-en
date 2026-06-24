---
title: >-
  [Paper Note] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit
description: >-
  [ICCV 2025][Optimization][Explainable classification] This paper parameterizes the query dictionary of Information Pursuit (IP) as learnable vectors in the CLIP semantic embedding space, and learns a task-sufficient interpretable query dictionary via an alternating optimization algorithm, substantially closing the performance gap between interpretable classifiers and black-box classifiers.
tags:
  - "ICCV 2025"
  - "Optimization"
  - "Explainable classification"
  - "Information Pursuit"
  - "sparse dictionary learning"
  - "CLIP"
  - "query dictionary optimization"
date: 2026-05-08
content_hash: 5a523e2d6e51b7c3
---

# Learning Interpretable Queries for Explainable Image Classification with Information Pursuit

**Conference**: ICCV 2025
**arXiv**: [2312.11548](https://arxiv.org/abs/2312.11548)  
**Code**: None  
**Area**: Explainable AI / Image Classification
**Keywords**: Explainable classification, Information Pursuit, sparse dictionary learning, CLIP, query dictionary optimization

## TL;DR

This paper parameterizes the query dictionary of Information Pursuit (IP) as learnable vectors in the CLIP semantic embedding space, and learns a task-sufficient interpretable query dictionary via an alternating optimization algorithm, substantially closing the performance gap between interpretable classifiers and black-box classifiers.

## Background & Motivation

Information Pursuit (IP) is an interpretable-by-design classification framework: given a predefined dictionary of semantic queries, IP selects the most informative query subset in order of information gain and makes predictions based on query–answer pairs. However, IP faces critical limitations:

**Manual dependency of query dictionaries**: Prior methods rely on expert-annotated concepts (e.g., bird attributes in CUB-200) or LLM prompt-generated queries, whose quality depends heavily on domain expertise.

**Suboptimality of LLM-generated queries**: Reliance on prompt engineering heuristics produces query sets that may be redundant, irrelevant, or insufficient.

**Performance gap**: A significant accuracy gap exists between IP with handcrafted dictionaries and black-box classifiers.

Core Problem: **How to learn a query dictionary that is task-sufficient?**

## Method

### Overall Architecture

The method leverages CLIP's semantic embedding space by parameterizing each query as a learnable vector $\theta_i$ in that space, maintaining interpretability via nearest-neighbor projection $q^{(\theta_i)} = \arg\min_{q \in \mathcal{U}} \|\theta_i - q\|_2^2$ (each learned query always corresponds to a natural language concept in the query universe). An alternating optimization algorithm is adopted: freeze the dictionary to update the V-IP network → freeze the V-IP network to update the dictionary.

### Key Designs

1. **Query parameterization in CLIP space**: The query universe $\mathcal{U} = \{E_T(c) | c \in \mathcal{T}\}$ consists of approximately 300,000 CLIP text embeddings (derived from multiple LLM prompts and COCO dataset captions). $K$ learnable embeddings $\theta = \{\theta_i\}_{i=1}^K$ are projected to interpretable queries via nearest-neighbor projection. The straight-through estimator (STE) enables backpropagation through the $\arg\min$ operation. The dictionary-augmented V-IP objective is: $\arg\min_{\theta,\psi,\eta} J_{Q_\theta}(\psi, \eta)$.

2. **Alternating optimization algorithm (Algorithm 1)**: Directly optimizing all three components ($\theta$, $\psi$, $\eta$) jointly is problematic: once the dictionary is updated, query semantics change, invalidating the existing querier policy. Accordingly, the method alternates every $t=4$ V-IP network update steps with 1 dictionary update step. During V-IP updates, the dictionary is frozen while the querier and classifier are trained; during dictionary updates, the V-IP network is frozen and only $\theta$ is updated.

3. **Connection to sparse dictionary learning**: The method bears deep connections to classical sparse dictionary learning algorithms such as K-SVD: (a) IP query subset selection ≈ OMP sparse coding (selecting the most informative atoms); (b) V-IP updates ≈ sparse coding step (computing semantic codes); (c) dictionary updates ≈ dictionary atom updates (minimizing classification error rather than reconstruction error). Proposition 1 proves that under biased sampling, the optimal dictionary parameters simultaneously minimize the sum of KL divergences across all query budgets.

4. **Query answering mechanism**: Soft answers are computed using CLIP ViT-L/14 via normalized dot products: $\hat{q}^{(\theta_i)}(X) = (\langle q^{(\theta_i)}/\|q\|, E_I(X)/\|E_I(X)\| \rangle - m_\theta) / (M_\theta - m_\theta)$, then binarized into hard answers by thresholding at 0.5 (ensuring interpretability).

### Loss & Training

- V-IP loss: $J_{Q_\theta}(\psi, \eta) = \mathbb{E}_{X,S}[D_{KL}(P(Y|X) \| P_\psi(Y|S, A_\eta(X,S)))]$
- Both the querier and classifier are two-layer MLPs with masking to handle variable-length inputs
- Adam optimizer; V-IP updates and dictionary updates are performed alternately
- Hyperparameters are tuned based on validation accuracy AUC

## Key Experimental Results

### Main Results — Query Dictionary Learning Improves V-IP Accuracy

K-Learned vs. K-LLM across 6 datasets at a fixed query budget $\tau$:

| Dataset | Query Budget $\tau$ | K-LLM | K-Learned (best init) | Black-Box |
|---------|---------------------|-------|----------------------|-----------|
| RIVAL-10 | 10 | ~96% | **~98.7%** | ~99% |
| CIFAR-10 | 10 | ~90% | **~95.1%** | ~97% |
| CIFAR-100 | 50 | ~70% | **~75.2%** | ~82% |
| ImageNet-100 | 50 | ~79% | **~84.0%** | ~91% |
| CUB-200 | 100 | ~69% | **~74.5%** | ~82% |
| Stanford-Cars | 100 | ~77% | **~82.4%** | ~87% |

K-Learned consistently and significantly outperforms K-LLM on all datasets and substantially closes the gap with black-box models.

### Ablation Study

Alternating optimization vs. joint optimization:

| Dataset | Query Budget | Alternating | Joint |
|---------|-------------|-------------|-------|
| RIVAL-10 | 10 | **98.73%** | 98.26% |
| CIFAR-10 | 10 | **95.12%** | 87.00% |
| CUB-200 | 100 | **74.52%** | 72.14% |
| Stanford-Cars | 100 | **82.39%** | 79.18% |

Alternating optimization consistently outperforms joint optimization, with a gap of up to 8% on CIFAR-10.

Comparison with 4 state-of-the-art CBMs (using RN50 CLIP with soft answers):

| Dataset | K-Learned | PCBM | LaBo | Label-free | Res-CBM |
|---------|-----------|------|------|------------|---------|
| CIFAR-10 | **88.55%** | 82.08% | 87.52% | 86.77% | 88.03% |
| CIFAR-100 | **68.02%** | 56.00% | 67.36% | 67.45% | 67.91% |

K-Learned outperforms or is competitive with all four state-of-the-art concept bottleneck models.

### Key Findings

- All three initialization strategies (K-LLM, K-Random, K-Medoids) benefit from learning, with performance differences within 5 percentage points.
- Quantization (hard answers + nearest-neighbor projection) reduces performance but guarantees interpretability.
- In a jellyfish classification case study, V-IP progressively reduces posterior entropy through 8 queries (e.g., "Wings? No", "Swims? Yes", "UFO-like? Yes"), providing a transparent decision process.
- CLIP as a query-answering mechanism introduces noise (e.g., answering "anemone? Yes" for jellyfish).

## Highlights & Insights

- **Bridging dictionary learning from signal processing to explainable AI**: A formal connection between IP query selection and OMP sparse coding is established (Proposition 1).
- **Interpretability constraints built into the parameterization**: Nearest-neighbor projection onto the query universe guarantees that all learned queries remain expressible in natural language.
- **Necessity of alternating optimization**: The work reveals the coupling problem between the querier and the dictionary; joint optimization leads to semantic drift.
- **Progressive explanation**: The IP decision process resembles a "20 Questions" game, where the posterior distribution change is observable at each step, offering more intuitive explanations than the static representations of CBMs.

## Limitations & Future Work

- The method relies heavily on CLIP's query-answering quality; noisy CLIP responses constrain final performance.
- The query universe must be constructed in advance (~300K queries), and its quality affects the learned dictionary.
- Hard-answer quantization loses information, yet removing quantization compromises interpretability—a fundamental tension.
- Extension to larger-scale classification tasks (e.g., full ImageNet-1K) remains unexplored.
- The query budget $\tau$ has a large impact on performance but must be set manually.

## Related Work & Insights

- Distinction from Res-CBM: Res-CBM compensates for an incomplete dictionary via a residual module, whereas this work directly learns a sufficient dictionary.
- Sparse CLIP (SPLICE) decomposes images into sparse linear combinations of concepts, sharing a similar spirit but targeting a different task.
- The approach may inspire other tasks requiring interpretable intermediate representations, such as explainable VQA and medical diagnosis.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The bridge between sparse dictionary learning and Information Pursuit is highly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six datasets with comparisons across multiple initialization strategies and optimization schemes.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with in-depth connections to classical methods.
- **Value**: ⭐⭐⭐⭐ Provides a principled learning approach for query design in interpretable classifiers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](../../NeurIPS2025/optimization/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[NeurIPS 2025\] From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD](../../NeurIPS2025/optimization/from_information_to_generative_exponent_learning_rate_induces_phase_transitions_.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](../../NeurIPS2025/optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[ICML 2025\] Synonymous Variational Inference for Perceptual Image Compression](../../ICML2025/optimization/synonymous_variational_inference_for_perceptual_image_compression.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)

</div>

<!-- RELATED:END -->
