---
title: >-
  [Paper Note] Predictive Data Selection: The Data That Predicts Is the Data That Teaches
description: >-
  [ICML2025][Model Compression][Pre-training data selection] The PreSelect method is proposed based on the hypothesis that "the data that can predict model capability is the data that can teach the model." By leveraging the rank correlation of multi-model losses to quantify document predictive strength, a fastText classifier is trained for efficient data selection. On a 1B model, training with 30B tokens selected by PreSelect outperforms random selection with 300B tokens…
tags:
  - "ICML2025"
  - "Model Compression"
  - "Pre-training data selection"
  - "compression is intelligence"
  - "predictive strength"
  - "fastText classifier"
  - "data quality"
date: 2026-05-08
content_hash: 76d1460429713f2f
---

# Predictive Data Selection: The Data That Predicts Is the Data That Teaches

**Conference**: ICML2025  
**arXiv**: [2503.00808](https://arxiv.org/abs/2503.00808)  
**Code**: [hkust-nlp/PreSelect](https://github.com/hkust-nlp/PreSelect)  
**Area**: Data Selection  
**Keywords**: Pre-training data selection, compression is intelligence, predictive strength, fastText classifier, data quality

## TL;DR

The PreSelect method is proposed based on the hypothesis that "the data that can predict model capability is the data that can teach the model." By leveraging the rank correlation of multi-model losses to quantify document predictive strength, a fastText classifier is trained for efficient data selection. On a 1B model, training with 30B tokens selected by PreSelect outperforms random selection with 300B tokens, achieving a 10x compute saving.

## Background & Motivation

Pre-training large language models requires training on massive web datasets, where data quality directly impacts the efficiency of scaling laws. Existing data selection methods primarily rely on manual heuristic rules:

- **Rule-based filtering**: FineWeb-Edu uses LLMs to score educational quality, favoring educational documents.
- **Reference data alignment**: DCLM trains a fastText classifier using SFT data as positive examples.
- **Perplexity filtering**: CCNet retains low-perplexity documents.

These methods introduce strong human priors, which may deviate from the optimal selection. This paper takes a different path, starting from the findings of Huang et al. (2024): the compression efficiency (normalized loss) of different models on specific texts is highly correlated with downstream performance. For example, the loss on GitHub code is close to being linearly correlated with coding tasks, while the loss on Common Crawl correlates with knowledge-intensive tasks. Consequently, the core hypothesis is proposed: **the more the compression on a certain dataset reflects model capability, the more this data helps the model learn that capability**.

## Method

### Predictive Strength Definition

Given $N$ open-source pre-trained models $\{M_1, M_2, \ldots, M_N\}$ and their downstream average scores $\{S_1 < S_2 < \ldots < S_N\}$, for document $d$, the normalized character loss $\{C_1, C_2, \ldots, C_N\}$ of each model is computed, and the predictive strength is defined as:

$$\mathbf{S} = \sum_{1 \le i < N} \sum_{i < j \le N} \mathbb{I}\{C_i > C_j\} / Z$$

where $Z = \frac{N^2 - N}{2}$ is the normalization factor to ensure $\mathbf{S} \in [0, 1]$. Intuitively, when the ranking of model losses is inversely aligned with the ranking of downstream performance (i.e., stronger capability corresponds to lower loss), the score is higher. $\mathbf{S}=1$ indicates that the document's loss perfectly predicts the model capability ranking.

### Comparison with Pearson Correlation Coefficient

The authors choose a rank-based matching score over Pearson correlation because single-document loss calculation is highly sensitive to noise (especially for short documents). Numerical correlation estimation is easily affected by outliers, whereas rank correlation is more robust.

### Overall Architecture

1. **Sampling Calculation Set**: Sample 300 documents from each of the 3,000 most frequent domains in the pre-training corpus, totaling ~900,000 documents.
2. **Calculating Predictive Strength**: Use 6 models from the Llama 1/2 series (7B-65B) to compute the normalized loss of each document, and combine this with the average rank scoring of 12 benchmarks to calculate predictive strength.
3. **Constructing Training Set**: Select ~200,000 documents with the highest predictive strength as positive examples, and ~200,000 with the lowest as negative examples.
4. **Training fastText Classifier**: Train a lightweight fastText scorer based on the positive and negative examples.
5. **Large-scale Filtering**: Score the entire corpus using fastText and select the top 10% of documents for pre-training.

### Key Designs

- **Same-Family Models**: Use only the 6 models from the Llama series to avoid evaluation noise across different model families (different families have high variance in sensitivity to prompts).
- **Document-level Granularity**: Operate directly at the document level rather than grouping by domain first (domain-level granularity is too coarse).
- **fastText-only Deployment**: No heavy LLM inference is required, making it easy to scale to trillion-token corpora.

## Key Experimental Results

### Main Results: RefinedWeb Corpus (1B model, 30B tokens, selecting 10%)

| Method | ARC-E | ARC-C | MMLU | LAMBADA | RACE | SciQ | BBH | Average |
|------|-------|-------|------|---------|------|------|-----|------|
| Random (300B) | 42.2 | 27.8 | 24.5 | 27.6 | 22.3 | 70.9 | 12.8 | 31.3 |
| Random (30B) | 39.2 | 24.4 | 26.0 | 19.0 | 21.9 | 64.8 | 7.8 | 28.1 |
| PPL Filtering | 42.5 | 24.6 | 25.8 | 18.8 | 22.6 | 67.5 | 8.5 | 29.1 |
| FineWeb-Edu | 48.3 | 26.1 | 26.0 | 18.2 | 24.4 | 69.0 | 12.8 | 31.1 |
| DCLM | 45.2 | 24.8 | 26.3 | 22.2 | 24.3 | 70.0 | 12.6 | 31.2 |
| **PreSelect** | **48.0** | **26.8** | **26.0** | **23.5** | **27.7** | **71.5** | **16.2** | **33.4** |

Key Findings:

- The model trained with 30B tokens using PreSelect (33.4) outperforms Random trained with 300B tokens (31.3), **achieving a 10x compute saving**.
- Outperforms the strongest baseline DCLM by **+2.2%** absolute.
- The improvement is most significant on BBH (16.2 vs 12.6), indicating that the selected data substantially enhances reasoning capability.

### 3B Model (100B tokens)

| Method | ARC-E | SciQ | BBH | Average | Math (BPC↓) | Code (BPC↓) |
|------|-------|------|-----|------|-------------|-------------|
| Random | 51.2 | 79.5 | 15.3 | 34.7 | 0.818 | 0.726 |
| DCLM | 55.7 | 82.5 | 20.5 | 37.8 | 0.712 | 0.664 |
| **PreSelect** | **61.2** | **85.6** | **23.3** | **39.5** | **0.694** | **0.648** |

The advantage is maintained at a larger scale, outperforming DCLM by +1.7%.

### C4 Corpus Validation (410M Pythia Model)

Compared with methods like DSIR, DsDm, QuRating, and MATES, PreSelect achieves the best performance on C4 as well, validating its cross-corpus generalization capability.

## Highlights & Insights

1. **Unique Theoretical Insight**: The hypothesis "the data that predicts is the data that teaches" elevates data selection from heuristic rules to an information-theoretic perspective, establishing a triangular relationship among compression, intelligence, and data quality.
2. **Extremely Lightweight**: Only requires inference on 6 open-source models + fastText training. It does not train any deep learning models for selection, and deployment solely relies on fastText, which is far superior to methods that require heavy LLM inference.
3. **10x Compute Savings**: Training on 30B tokens outperforms 300B tokens under random training, demonstrating immense practical value.
4. **Same-Family Model Design**: Discovered that evaluation noise across different model families is a key obstacle. Restricting the evaluation to the Llama series effectively avoids this issue.
5. **Document-level vs. Domain-level**: Experiments demonstrate that document-level granularity is significantly better than domain-level (PPL Correlation DD/DP), validating the necessity of fine-grained selection.

## Limitations & Future Work

1. **Model Family Dependency**: Predictive strength is calculated using only the Llama 1/2 series, which might introduce biases inherent to this model family; its effectiveness when switching to other model families remains uncertain.
2. **Benchmark Dependency**: Downstream ranking is based on the average score of 12 fixed benchmarks. Choosing a different combination of benchmarks could lead to different data preferences.
3. **English-Centric**: Experiments are validated only on English corpora (RefinedWeb, C4), and the performance in multilingual scenarios is unknown.
4. **Static Selection**: A one-time selection is performed and then fixed for training, which has not yet been combined with dynamic data selection methods (e.g., MATES).
5. **Scale Ceiling**: The maximum experimental scale is a 3B model, and it remains to be verified whether the advantage holds at the 7B+ scale.

## Related Work & Insights

- **Huang et al. (2024)**: Empirical discovery of the correlation between compression efficiency and downstream performance, which serves as the core inspiration for this work.
- **Thrush et al. (2024) Perplexity Correlation**: A pioneer in domain-level correlation for data selection, though this paper offers fundamental improvements in document-level granularity and model selection strategies.
- **DCLM (Li et al., 2024a)**: The current strongest baseline, which uses SFT data to guide fastText training.
- **FineWeb-Edu**: Educational quality scoring approach, which contrasts with the unsupervised approach presented in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel hypothesis, successfully translating the compression-intelligence correlation into a data selection principle.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-scale validation (400M/1B/3B) across two corpora and 17 benchmarks with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, concise methodology, and detailed experiments.
- Value: ⭐⭐⭐⭐⭐ The 10x compute saving holds high practical value; both the classifier and datasets are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](../../ACL2025/model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ICML 2025\] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams](lego_sketch_a_scalable_memory-augmented_neural_network_for_sketching_data_stream.md)
- [\[ICML 2025\] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)
- [\[NeurIPS 2025\] Geometric Data Valuation via Leverage Scores](../../NeurIPS2025/model_compression/geometric_data_valuation_via_leverage_scores.md)
- [\[ICML 2025\] WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training](wildchat-50m_a_deep_dive_into_the_role_of_synthetic_data_in_post-training.md)

</div>

<!-- RELATED:END -->
