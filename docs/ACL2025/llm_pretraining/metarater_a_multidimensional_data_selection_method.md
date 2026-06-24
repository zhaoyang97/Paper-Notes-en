---
title: >-
  [Paper Note] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models
description: >-
  [ACL 2025][LLM Pretraining][Data Selection] Proposes the Meta-rater multi-dimensional data selection framework, defining four quality dimensions under PRRC (Professionalism, Readability, Reasoning, and Cleanliness). By using a proxy model regression to learn the optimal weighted combination of multiple quality scores, it doubles the training convergence speed of a 1.3B model and improves downstream task performance by 3.23%.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Data Selection"
  - "Pre-training"
  - "Multi-dimensional Quality"
  - "Proxy Model"
  - "Data Curation"
date: 2026-05-08
content_hash: 8f5991222c6529d1
---

# Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models

**Conference**: ACL 2025  
**arXiv**: [2504.14194](https://arxiv.org/abs/2504.14194)  
**Code**: [https://github.com/opendatalab/Meta-rater](https://github.com/opendatalab/Meta-rater)  
**Area**: LLM Pre-training Data / Data Quality  
**Keywords**: Data Selection, Pre-training, Multi-dimensional Quality, Proxy Model, Data Curation

## TL;DR

Proposes the Meta-rater multi-dimensional data selection framework, defining four quality dimensions under PRRC (Professionalism, Readability, Reasoning, and Cleanliness). By using a proxy model regression to learn the optimal weighted combination of multiple quality scores, it doubles the training convergence speed of a 1.3B model and improves downstream task performance by 3.23%.

## Background & Motivation
**Background**: Pre-training data quality is a critical driver of LLM performance, making data selection a core research direction.

**Limitations of Prior Work**: Existing methods are limited to single-dimensional evaluations—natural language quality methods focus on surface features, diversity methods prioritize deduplication, and classifier methods employ single-dimensional filters.

**Key Challenge**: How to systematically integrate complementary quality dimensions to achieve globally optimal data selection?

**Goal**: Propose a multi-dimensional data quality evaluation framework and its optimal fusion method.

**Key Insight**: Search for the optimal weight combination of multi-dimensional quality scores by training proxy models.

**Core Idea**: Train multiple proxy models to fit the mapping from "quality weights to validation loss," thereby finding the optimal weighting scheme for multi-dimensional quality scores.

## Method

### Overall Architecture
1) Annotate 25 quality scores (natural language features + data importance + model scores) for the SlimPajama-627B corpus; 2) Randomly generate weight combinations to select data for training proxy models; 3) Fit a LightGBM regression model to predict validation loss; 4) Search for the optimal weight combination. The entire workflow can be executed automatically without manual hyperparameter tuning.

### Key Designs
1. **PRRC Four-Dimensional Evaluation**: Professionalism (density of professional knowledge), Readability (difficulty of text comprehension), Reasoning (depth of logical reasoning), and Cleanliness (formatting correctness with minimal noise). Each dimension is annotated on 500K samples using Llama-3.3-70B-Instruct, followed by fine-tuning ModernBERT scoring models, achieving F1 scores of 87-92%.
2. **Proxy Model Training**: Generate $N$ sets of random weights $\rightarrow$ select top-$k$ data according to the weighted score for each set $\rightarrow$ train small models $\rightarrow$ record validation loss to obtain (weights, loss) data pairs.
3. **Regression Model Prediction**: Fit a non-linear mapping from weights to loss using LightGBM, and search the larger weight space to find the optimal weights corresponding to the minimum loss: $$\mathbf{w}^* = \arg\min_{\tilde{\mathbf{w}}} f(\tilde{\mathbf{w}})$$

### Loss & Training
- Aggregated Quality Score: $Q_{agg} = \sum_{j=1}^{m} w_j \cdot Q_j(x)$
- Optimal Weight Search: $\mathbf{w}^* = \arg\min_{\mathbf{w}} J(\theta)$, where $J(\theta)$ represents the validation loss.
- Robustness is enhanced using Top-$k$ averaging.
- The main experiments use a 1.3B model trained on 30B tokens, and scale to 3.3B and 7.2B models to verify scalability.

## Key Experimental Results

### Main Results (1.3B Model Downstream Performance)

| Method | General Know. | Commonsense | Reading Comp. | Average |
|------|----------|----------|----------|---------|
| Random (30B) | 52.79 | 43.94 | 30.02 | 43.78 |
| Random (60B) | 56.01 | 44.87 | 31.47 | 45.70 |
| QuRating-Edu | 57.66 | 46.72 | 28.10 | 46.16 |
| Fineweb-Edu | 55.79 | 45.51 | 31.10 | 45.76 |
| MATES | 53.15 | 43.25 | 30.55 | 43.79 |
| **Meta-rater (All 25)** | **58.90** | **45.41** | **31.55** | **47.01** |

### Ablation Study (Different Dimensional Combinations)

| Meta-rater Configuration | Average | Gain over Random |
|--------------|---------|-----------|
| PRRC (4-dim) | 46.35 | +2.57 |
| Model (11-dim) | 46.60 | +2.82 |
| All (25-dim) | 47.01 | +3.23 |
| Single Dimension - Professionalism | 45.26 | +1.48 |
| Single Dimension - Readability | 45.89 | +2.11 |
| Single Dimension - Reasoning | 45.28 | +1.50 |
| Single Dimension - Cleanliness | 45.68 | +1.90 |

### Key Findings
- PPL-based selection leads to a performance drop (-2.25% average), emphasizing that low perplexity does not equate to high quality.
- Semdedup also slightly degrades performance (-0.81%), showing that deduplication alone is insufficient for quality enhancement.
- DSIR relies on target domain selection, yielding significantly different results between the Book and Wikipedia domains.

- Meta-rater (trained on 30B tokens) outperforms Random selection (trained on 60B tokens), doubling the convergence speed.
- Multi-dimensional fusion significantly outperforms any single-dimension selection.
- F1 scores for the four PRRC scoring models: Professionalism 91.57%, Readability 87.47%, Reasoning 89.59%, and Cleanliness 87.88%.
- Traditional methods such as PPL and Semdedup may actually degrade performance.
- The advantages scale effectively to 3.3B and 7.2B models.

## Highlights & Insights
- The "Meta-rater" concept is highly generalizable: using a proxy model to search for the optimal combination of multi-dimensional scores can be extended to other data selection scenarios.
- The PRRC dimensions are specifically targeted: the reasoning dimension addresses the demand for reasoning data in the o1 era, and the cleanliness dimension replaces rule-based methods with model-based approaches to handle long-tail anomalies.
- First release of the fully annotated SlimPajama-627B corpus (containing 25 quality metrics), representing a valuable resource for data-centric AI research.
- Validates the core hypothesis that multi-dimensional integration is far superior to single-dimension methods—All (25 dimensions) achieves an improvement of 0.85 to 1.75 percentage points over any single-dimension method.
- Meta-rater selection with 30B tokens outperforms random selection with 60B tokens, indicating that quality is far more important than quantity.
- All four PRRC scoring models achieve over 87% F1 scores, ensuring the reliability of automatic annotation.

## Limitations & Future Work
- The training cost for the proxy model remains relatively high (requiring training over a hundred small models). Although cheaper than large models, the total overhead is substantial.
- LightGBM regression might not fully capture the complex non-linear relationships in the weight space; stronger regression models could be explored.
- Validated only on SlimPajama; generalization to other corpora (such as multilingual or code data) remains to be confirmed.
- The optimal set of quality dimensions may vary depending on the downstream tasks, and generalizability has yet to be verified.
- PRRC annotations rely on the judgments of Llama-3.3-70B-Instruct; different annotation models may produce different scores.
- Combination effects between different data points (such as the complementarity of certain topic pairs) are not considered.

## Related Work & Insights
- Complements works like QuRating (Wettig et al., 2024) and Fineweb-Edu (Penedo et al., 2024).
- Generalizes domain mixing weight optimization concepts like DoReMi (Xie et al., 2023).
- The multi-dimensional definition of "data quality" can inspire exploration into additional dimensions.
- Provides a reproducible and systematic experimental paradigm for data-centric AI.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-dimensional fusion framework is elegantly designed, though individual components (classifier-based scoring/proxy models) are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons and ablation studies; validated across multiple scales (1.3B/3.3B/7.2B).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed methodology descriptions, and formal algorithmic pseudocode.
- Value: ⭐⭐⭐⭐⭐ Open-source datasets and methods offer highly valuable contributions to the community.
- Overall: A highly practical work in the field of data-centric AI; the conclusion that "quality is far more important than quantity" offers significant guidance.
- Reproducibility: Fully open-sourced code, data, and models, allowing immediate community adoption.
- Extensibility: Integration with other quality dimensions (such as safety or factuality) can be further explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [\[ACL 2025\] Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models](optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)

</div>

<!-- RELATED:END -->
