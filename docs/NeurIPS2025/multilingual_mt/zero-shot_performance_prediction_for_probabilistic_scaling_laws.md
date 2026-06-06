---
title: >-
  [Paper Note] Zero-Shot Performance Prediction for Probabilistic Scaling Laws
description: >-
  [NeurIPS 2025][Multilingual & Machine Translation][scaling laws] This paper frames NLP learning curve prediction as a multi-task learning problem…
tags:
  - "NeurIPS 2025"
  - "Multilingual & Machine Translation"
  - "scaling laws"
  - "learning curves"
  - "Gaussian processes"
  - "zero-shot prediction"
  - "hierarchical modeling"
  - "active learning"
  - "multi-task learning"
date: 2026-05-08
content_hash: 11a45150a452926f
---

# Zero-Shot Performance Prediction for Probabilistic Scaling Laws

**Conference**: NeurIPS 2025
**arXiv**: [2510.16743](https://arxiv.org/abs/2510.16743)  
**Code**: Authors state it will be released (link not yet confirmed)  
**Area**: Multilingual Translation
**Keywords**: scaling laws, learning curves, Gaussian processes, zero-shot prediction, hierarchical modeling, active learning, multi-task learning

## TL;DR

This paper frames NLP learning curve prediction as a multi-task learning problem, employing a latent-variable multi-output Gaussian process (MaGP) to capture the bi-level hierarchical structure of datasets and inter-task correlations, enabling zero-shot prediction of learning curves and deriving probabilistic scaling laws via Monte Carlo simulation.

## Background & Motivation

### State of the Field
Deriving scaling laws requires training a large number of models under varying configurations, incurring substantial computational cost. Zero-shot prediction of learning curves for untrained configurations from a small set of existing curves would significantly reduce this cost.

### Limitations of Prior Work

**Parametric fitting methods** (power law, exponential functions, etc.): require independent fitting for each curve and cannot transfer across tasks.

**Bayesian Neural Networks (BNNs)**: require a large number of configurations (Klein et al. use a minimum of 256), performing poorly on small datasets (fewer than 30 curves).

**Conventional GP methods**: assume a flat internal data structure and do not exploit hierarchical organization.

**Existing scaling law studies**: computationally intensive empirical investigations lacking uncertainty quantification.

### Core Hypothesis

NLP learning curve datasets exhibit a **bi-level hierarchy** that is **exchangeable**. Exploiting this structure together with inter-task correlations enables zero-shot prediction.

## Method

### Overall Architecture

The pipeline proceeds as follows: data modeling (bi-level hierarchy) → MaGP training → zero-shot prediction of missing curves → Monte Carlo simulation → probabilistic scaling law.

### Definition of the Hierarchical Structure

Three experimental scenarios correspond to different hierarchy definitions:

| Dataset | Task $t$ (Level 1) | Data instance $d$ (Level 2) | Model size |
|--------|-------------------|----------------------|---------|
| nanoGPT | Number of embedding parameters | Number of layers | $n_i$ determined by both |
| Bilingual translation | Source language | Target language | Fixed model |
| Multilingual translation | Source language | Target language + model size | Various M2M100 sizes |

### Key Designs: MaGP Model

The paper adopts the latent-variable multi-output Gaussian process proposed by Ma et al. (2023):

**Generative process**:

$$g(\mathbf{x}) \sim \mathcal{GP}(0, k_g(\mathbf{x}, \mathbf{x}'))$$

$$l_t^d(\mathbf{x}) \sim \mathcal{GP}(g(\mathbf{x}), k_l(\mathbf{x}, \mathbf{x}'))$$

$$y_t^d(\mathbf{x}) = l_t^d(\mathbf{x}, \mathbf{h}_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2), \quad \mathbf{h}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Component descriptions:
- $g(\mathbf{x})$: **shared mean function** — encodes common prior information across all tasks
- $k_g$: covariance kernel for the shared mean
- $l_t^d(\mathbf{x})$: the $d$-th learning curve of task $t$, with $g(\mathbf{x})$ as its mean
- $k_l$: curve-level covariance kernel
- $\mathbf{h}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: **latent variable** — captures inter-task correlations
- $\epsilon_t$: observation noise

**Predictive distribution** (via variational inference):

$$q(\mathbf{l}^* | \mathbf{X}^*) = \int q(\mathbf{l}^* | \mathbf{X}^*, \mathbf{H}) q(\mathbf{H}) \text{d}\mathbf{H}$$

$$q(\mathbf{l}^* | \mathbf{X}^*, \mathbf{H}) = \mathcal{N}(\mathbf{l}^* | \tilde{\mathbf{m}}_*, \tilde{\mathbf{K}}_*)$$

The posterior is multivariate Gaussian, directly yielding both the mean and uncertainty.

### Probabilistic Scaling Law

The standard scaling law is linear in log-log space:

$$l^{\log}(c) \sim \mathcal{N}(\beta_0 + \beta_1 c, \sigma^2)$$

Probabilistic estimation procedure ($R$ Monte Carlo iterations):
1. Train MaGP on $N$ known curves.
2. Zero-shot predict $M$ missing curves.
3. Fit the compute-efficient frontier on known and predicted curves.
4. Repeat $R$ times and average over $\beta_0, \beta_1$:

$$\hat{\beta}_0 = \frac{1}{R} \sum_{r=1}^{R} p(\beta_0^r | \mathcal{D}^*), \quad \hat{\beta}_1 = \frac{1}{R} \sum_{r=1}^{R} p(\beta_1^r | \mathcal{D}^*)$$

### Active Learning Strategy

Starting from an initial training set, the most uncertain curve is queried at each step:

$$\text{mvar} = \frac{1}{11} \sum_{j=1}^{11} \text{var}_j, \quad \text{var} = \frac{1}{10} \sum_{i=1}^{10} (\mathbf{l}^{m_i*} - \overline{\mathbf{l}^{m_i*}})^2$$

The curve with the highest mvar is selected for querying, after which the model is retrained and predictions are updated.

## Key Experimental Results

### Main Results: Zero-Shot Prediction on nanoGPT

| Method | Split | MSE ↓ | MAE ↓ | MNLPD ↓ |
|------|------|-------|-------|---------|
| **MaGP** | Quad | **0.03±0.01** | **0.12±0.02** | **2.58±2.56** |
| DHGP | Quad | 0.07±0.00 | 0.20±0.01 | 3.25±0.24 |
| BNN (LC) | Quad | 10.69±0.43 | 2.82±0.03 | 596.05±23.79 |
| BNN (orig) | Quad | 13.96±0.75 | 3.06±0.05 | 778.95±41.90 |
| **MaGP** | Tri | **0.02±0.01** | **0.10±0.02** | **0.87±0.29** |
| DHGP | Tri | 0.07±0.00 | 0.19±0.00 | 1.85±0.63 |
| **MaGP** | T1 | **0.04±0.05** | **0.12±0.08** | **0.80±1.54** |
| DHGP | T1 | 0.08±0.00 | 0.18±0.00 | 0.87±0.09 |

**MaGP consistently outperforms all baselines across all splits and metrics.** BNN methods perform extremely poorly due to insufficient training data (only 29 curves × 11 points).

### Scaling Law Prediction

| Train–Test Split | Predicted Scaling Law | AbC ↓ | Computational Cost (PetaFLOPs) |
|-------------|-----------------|-------|---------------------|
| Quad | $(-0.043\pm0.002)c + (2.957\pm0.074)$ | 0.521±0.069 | $1.28 \times 10^5$ |
| Tri | $(-0.052\pm0.005)c + (3.352\pm0.202)$ | 0.160±0.172 | $2.06 \times 10^5$ |
| T1 | $(-0.059\pm0.006)c + (3.636\pm0.245)$ | **0.111±0.222** | $5.15 \times 10^5$ |

Ground truth: $-0.056c + 3.51$.

**Findings**:
- The Tri and T1 splits yield scaling law predictions very close to the ground truth.
- More training data leads to lower AbC (greater accuracy), though uncertainty may also increase.
- The Quad split is the most computationally efficient but exhibits the largest deviation.

### Ablation Study: Validation of Hierarchical Exchangeability

| Method | Split | MSE ↓ | MAE ↓ |
|------|------|-------|-------|
| MaGP (swapped) | Quad | 0.04±0.04 | 0.11±0.07 |
| DHGP (swapped) | Quad | 0.11±0.00 | 0.26±0.01 |
| MaGP (swapped) | T1 | **0.01±0.02** | **0.08±0.04** |
| DHGP (swapped) | T1 | 0.16±0.02 | 0.34±0.02 |

MaGP retains its advantage over the baseline after swapping the hierarchy levels, performing even better on T1 — validating the **exchangeability** hypothesis.

### Comparison of Active Learning Query Strategies

Performance of four strategies in terms of AbC:
- **Active Learning**: most stable, lowest uncertainty; AbC remains lowest or near-lowest across all query counts.
- **Largest First**: AbC close to Active Learning but at higher cost and greater uncertainty.
- **Smallest First**: AbC is higher in early queries.
- **Random**: highest uncertainty overall.

### Zero-Shot Prediction on Bilingual Translation

On the bilingual translation dataset (mBART50 + Transformer):
- MaGP achieves the lowest RMSE on both BLEU and ChrF metrics.
- The naive baseline (averaging source/target language curves) exhibits high variance.
- DHGP improves over the naive baseline but retains high uncertainty.
- Results confirm the advantage of modeling the bi-level hierarchical structure.

### Key Findings

1. On extremely small datasets (at most 30 curves, 11 points each), MaGP substantially outperforms both BNN and DHGP.
2. Hierarchical exchangeability holds: different hierarchy definitions yield comparable predictive performance.
3. The active learning strategy effectively reduces uncertainty and provides the most reliable scaling law predictions.
4. Probabilistic scaling laws offer both point estimates and uncertainty quantification, conveying more information than traditional deterministic fits.

## Highlights & Insights

1. **Novel problem framing**: Scaling law derivation is recast as a multi-task learning and zero-shot prediction problem, departing from conventional parametric fitting.
2. **Strong validation of the hierarchical hypothesis**: The effectiveness of the bi-level hierarchical structure is consistently demonstrated across three distinct NLP datasets.
3. **Probabilistic scaling laws as a key contribution**: Conventional methods yield only point estimates; MaGP combined with MC simulation produces a full distribution, which is more valuable for decision-making.
4. **Active learning strategy**: Provides a principled approach to selecting which model to train next, with practical deployment value.
5. **Effective with minimal data**: Meaningful predictions can be made from as few as 29 learning curves × 11 points, which is highly significant for resource-constrained researchers.
6. **Model selection intuition**: MaGP captures inter-task correlations through latent variables, while DHGP relies on hierarchy to discover clusters — MaGP prevails when inter-task correlations are strong.

## Limitations & Future Work

1. **Limited dataset scale**: The largest dataset contains only 30 curves; the approach has not been validated in truly large-scale settings (e.g., Chinchilla-scale).
2. **NLP-only evaluation**: Although the framework is general, experiments are restricted to NLP tasks.
3. **nanoGPT is a small model**: A gap remains between this setting and the large models (70B+) of practical industry interest.
4. **Hierarchy definition requires prior knowledge**: Users must determine which factors constitute the hierarchy, which may not be intuitive in all domains.
5. **Computational complexity**: The $O(n^3)$ complexity of GPs may become a bottleneck as the number of curves or observation points grows.
6. **Future directions**:
    - Validate on larger-scale models (Llama-70B class).
    - Develop automatic discovery of hierarchical structure rather than relying on manual specification.
    - Integrate Chinchilla optimality theory for more refined scaling law prediction.
    - Extend the framework to other domains such as CV and speech.

## Related Work & Insights

- **Relation to Hoffmann et al. (2022, Chinchilla)**: Chinchilla derives scaling laws through exhaustive training; the present work substantially reduces cost via zero-shot prediction.
- **Comparison with Klein et al. (2016, BNN)**: BNNs require 256+ configurations, whereas this work requires fewer than 30 curves.
- **Complementary to alternative training techniques proposed by Hägele et al.**: The latter reduces cost from the training side; this work reduces it from the prediction side.
- **Hensman et al. (2013, DHGP)**: Hierarchical GPs were applied to gene expression data; the present work is the first to introduce analogous ideas to NLP learning curves.
- **Ma et al. (2023, MaGP)**: Provides the underlying model; the contribution of this paper lies in applying it to the scaling law derivation setting and identifying the hierarchical properties of NLP datasets.

**Insight**: When conducting scaling law research, it is unnecessary to train all configurations — by modeling the correlations among already-trained models, scaling laws close to ground truth can be derived with far less computation than traditional methods require.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying hierarchical GPs to scaling law prediction is a well-motivated combinatorial innovation; the MaGP model itself is not a contribution of this paper, but its application context is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple baselines, exchangeability validation, and active learning strategy comparisons provide comprehensive coverage; however, dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is rigorous and the narrative of progressively validating three hypotheses is clear; equations are numerous but well-explained.
- **Value**: ⭐⭐⭐⭐ — Currently at proof-of-concept stage; larger-scale validation is needed before it can practically replace traditional scaling law research, but the direction is promising.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](../../ICLR2026/multilingual_mt/atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[NeurIPS 2025\] MergeBench: A Benchmark for Merging Domain-Specialized LLMs](mergebench_a_benchmark_for_merging_domain-specialized_llms.md)
- [\[NeurIPS 2025\] Adaptive Originality Filtering: Rejection-Based Prompting and RiddleScore for Culturally Grounded Multilingual Riddle Generation](adaptive_originality_filtering_rejection_based_prompting_and_riddlescore_for_cul.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](exploring_the_translation_mechanism_of_large_language_models.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)

</div>

<!-- RELATED:END -->
