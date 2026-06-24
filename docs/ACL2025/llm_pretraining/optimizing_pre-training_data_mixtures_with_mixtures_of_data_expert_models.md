---
title: >-
  [Paper Note] Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models
description: >-
  [ACL2025][LLM Pretraining][Data Mixture] Proposes the Mixture of Data Experts (MDE) method, which independently trains expert models on each data domain and aggregates them via probability-level ensemble using mixture weights. This efficiently approximates language model loss under different data mixture ratios, significantly improving the search efficiency and prediction accuracy of pre-training data mixture proportions.
tags:
  - "ACL2025"
  - "LLM Pretraining"
  - "Data Mixture"
  - "Mixture of Data Experts"
  - "Proxy Model"
  - "Regression"
  - "Pretraining"
date: 2026-05-08
content_hash: e7b1a99302b8d6bb
---

# Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models

**Conference**: ACL2025  
**arXiv**: [2502.15950](https://arxiv.org/abs/2502.15950)  
**Code**: Not released  
**Area**: LLM Pre-training / Data Mixture Optimization  
**Keywords**: Data Mixture, Mixture of Data Experts, Proxy Model, Regression, Pretraining  

## TL;DR

Proposes the Mixture of Data Experts (MDE) method, which independently trains expert models on each data domain and aggregates them via probability-level ensemble using mixture weights. This efficiently approximates language model loss under different data mixture ratios, significantly improving the search efficiency and prediction accuracy of pre-training data mixture proportions.

## Background & Motivation

- **Importance of Data Mixture Ratios**: LLM pre-training data originates from multiple heterogeneous sources (e.g., Wikipedia, GitHub, CommonCrawl, etc.). The sampling proportions of different sources significantly impact the generalization performance of the model.
- **Large Search Space**: For $k$ data domains, the mixture ratio defines $k-1$ real-valued hyperparameters. Since large-scale LLMs are typically trained only once, exhaustively evaluating a vast number of mixture plans is infeasible.
- **Limitations of Prior Work**:
    - Online methods (DoGe, DoReMi) require modifications to the training algorithm and cannot reuse the same set of proxy models for different optimization objectives.
    - Regression methods (RegMix, etc.) only use mixture weights $\lambda$ as features, resulting in limited prediction accuracy and the need to train 30-500 proxy models.
- **Core Problem**: Can the generalization performance of models under arbitrary data mixture ratios be efficiently predicted using a minimal number of proxy models?

## Method

### Overall Architecture

The MDE method consists of three levels:
1. Independently train $k$ data expert models (one per domain).
2. Use the MDE ensemble to approximate model performance for any mixture ratio $\lambda$.
3. (Optional) Inject MDE features into regression models to further improve prediction accuracy.

### Mixture of Data Experts (MDE) Approximation

**Core Idea**: Uses $k$ expert models $\theta_1^*, \dots, \theta_k^*$, each trained on an individual data domain, to approximate the model trained on mixed data through a probability-level weighted ensemble:

$$P_{MDE}(x_t|x_{1\cdots t-1},\lambda) := \sum_{i=1}^{k}\lambda_i P_{\theta_i^*}(x_t|x_{1\cdots t-1})$$

For each candidate mixture scheme $\lambda$, the MDE loss is defined as the cross-entropy loss of this ensemble model across the validation domains.

**Efficient Implementation**:
- Pre-compute and cache the next-token probability of each expert $\theta_i^*$ on all validation domain tokens.
- For each new candidate $\lambda$, only $O(k)$ weighted addition and logarithm operations need to be executed on the CPU.
- No neural network inference needs to be executed for each candidate mixture scheme, making the computational cost negligible.

### MDE as Regression Features

Input the MDE-approximated domain losses as additional features into the regression models:

$$\hat{L}_j(\lambda) = M_j(\lambda; L_{MDE}^1, ..., L_{MDE}^m)$$

Three categories of regression models are investigated:
- **Linear Model**: Regularized weighted sum of features.
- **Gradient Boosting (GBM)**: Regression tree ensemble (referencing RegMix).
- **Multi-Task Gaussian Process (MTGP)**: Exploits inter-domain task correlations.

### Theoretical Foundation

**Proposition 3.1**: For any mixture weight $\lambda$, the optimal distribution $p^*_\lambda$ that minimizes the weighted mixture loss can be expressed as a weighted combination of the optimal distributions of individual data experts:

$$p^*_\lambda(y|x) = \sum_{i=1}^{k}\lambda'_i(x)p^*_i(y|x)$$

where $\lambda'_i(x) \propto D_i(x)\lambda_i$. When the prefix distributions of each domain are identical, $\lambda'_i = \lambda_i$, which is perfectly consistent with the MDE approximation. This provides theoretical justification for the probability-level ensemble.

### Optimization Objectives

Generalization performance is defined as the aggregation of validation domain losses:
- **avg-sp**: Average loss of 7 SlimPajama validation domains (training domains).
- **avg-et**: Average loss of 11 end-task validation domains (downstream tasks).
- **avg-et+sp**: Joint average loss of 18 validation domains.

Optimization is performed using the Vizier framework within a $k$-dimensional non-negative parameter search space, without requiring the objective to be differentiable.

## Key Experimental Results

### Experimental Setup

- **Dataset**: SlimPajama (7 training domains)
- **Model Scales**: 70M, 150M, 280M, 510M, 1B parameters
- **Training Tokens**: 5-25B for proxy models, 100B for full-scale models
- **Proxy Model Selection**: 280M/10K steps serves as the proxy for 1B/200K steps
- **Validation Domains**: SlimPajama validation set (7 domains) + ARC/OpenBookQA/MultiRC (11 domains)
- **Downstream Tasks**: 10 tasks including TriviaQA, NQ, SQuAD 2.0, LAMBADA, etc.

### Loss Prediction Accuracy

| Method | MSE(SP)↓ | Spearman ρ(SP)↑ | MSE(ET+SP)↓ | ρ(ET+SP)↑ |
|------|----------|-----------------|-------------|-----------|
| Empirical Mean | 0.01151 | N/A | 0.01250 | N/A |
| Linear | 0.01637 | 0.234 | 0.00655 | 0.646 |
| GBM-RegMix | 0.00242 | 0.923 | 0.00431 | 0.814 |
| DML | 0.00296 | 0.920 | 0.00116 | 0.892 |
| **MDE (7 models only)** | 0.02809 | 0.912 | 0.00391 | 0.886 |
| **Linear+MDE** | **0.00050** | **0.976** | **0.00048** | **0.953** |
| **MTGP+MDE** | **0.00053** | **0.984** | 0.00116 | 0.935 |
| **GBM+MDE** | 0.00140 | 0.950 | **0.00089** | **0.955** |

**Key Findings**:
- As an independent predictor, MDE achieves ranking capability comparable to the best regressors utilizing 3x more proxy models.
- MDE features bring significant improvements to all regression models: Spearman correlation of the Linear model increases from 0.65 to 0.95.
- Leveraging 25 training samples paired with MDE features outperforms existing methods.

### Proxy Model Scale Analysis

- For ranking on a single training domain, there is little difference between the 70M and 280M proxy models.
- For ranking on cross-domain aggregated metrics, the performance of the 70M model and proxies with training steps < 6K is significantly inferior to that of the 280M model.
- This validates the conditional nature of the "approximate rank consistency" hypothesis.

### Learning Curves

- Under low-data regimes, MDE (with only $k=7$ models) consistently outperforms all regression methods.
- Diminishing returns are observed after exceeding 25 training samples.
- Regression models combined with MDE features show steady improvement as the number of samples increases.

### Downstream Task Performance

Mixture schemes selected using avg-et+sp as the optimization objective:
- Outperform heuristic methods such as DoGe and DoReMi on both generation and ranking tasks.
- Validate the value of incorporating end-task validation domains into the optimization objective.

### Correlation between Validation Loss and Downstream Performance

- The correlation between a single end-task validation domain and downstream performance varies greatly (Self-correlation ranging from 0.245 to 0.903).
- Aggregating losses across multiple validation domains yields more stable correlation with all downstream tasks (0.77-0.85).
- This validates the rationality of using a diverse validation set to define optimization objectives.

## Highlights & Insights

1. **Extreme Efficiency**: Requiring only $k$ data expert models (one per domain), MDE achieves a mixture ratio ranking capability comparable to schemes using 30+ proxy models.
2. **Plug-and-Play**: MDE features can enhance any regression model (Linear, GBM, MTGP), yielding significant and robust improvements.
3. **Theoretically Driven**: Proposition 3.1 mathematically justifies the probability-level ensemble approximation from an information theoretic perspective and points out directions for improvement (prefix-dependent weights).
4. **High Practical Value**: MDE does not require modifications to the LM training algorithm, allowing the same set of proxy models to serve multiple optimization objectives.
5. **Caching & CPU Computation**: The core computation of the MDE approximation is virtually cost-free; once token probabilities are cached, the evaluation of infinitely many candidate schemes is performed entirely on the CPU.
6. **Effective Across Scales**: The mixture weights optimized with the 280M proxy model can be effectively transferred to the 1B model.

## Limitations & Future Work

1. **Constraint on Number of Domains**: Experiments are only verified on 7 training domains; scaling to dozens or more domains might present new challenges.
2. **Static Mixture**: Only fixed sampling ratios are considered, without exploring dynamic curriculum-based mixtures.
3. **Single SlimPajama Dataset**: Experiments are concentrated on a single dataset, leaving performance on larger or more diverse datasets unexplored.
4. **Idealized Assumptions of Approximation Theory**: Theoretical results indicate that when prefix distributions differ across domains, the optimal weights become $x$-dependent, which may bias actual performance.
5. **Cross-Entropy vs. Generation Performance**: The method optimizes cross-entropy rather than downstream task accuracy, and the relationship between the two is not perfectly linear.

## Related Work & Insights

- **Online Methods**: DoGe (first-order bilevel optimization), DoReMi (worst-case gap optimization)
- **Regression Methods**: RegMix (GBM regression), DML (exponential model), BiMix (power law model)
- **Model Blending Approximation**: Parameter averaging methods, merging models pre-trained from scratch
- **Data Selection & Curriculum**: Data selection at different granularities, such as domain-level and sample-level

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐⭐ The MDE concept is simple and elegant, achieving a solid unification of theory and practice.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive analysis across multiple scales, regression models, and learning curves.
- **Value**: ⭐⭐⭐⭐ Direct instructional significance for optimizing data recipes in large-scale LLM pre-training.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with highly readable theoretical sections.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks](../../ICLR2026/llm_pretraining/duet_optimizing_llm_training_data_mixtures_via_noisy_feedback_from_unseen_downst.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](data-constrained_synthesis_of_training_data_for_de-identification.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->
