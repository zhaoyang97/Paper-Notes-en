---
title: >-
  [Paper Note] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph
description: >-
  [LLM Evaluation] Proposes the GraphFilter method, which models SFT datasets as sentence–n-gram bipartite graphs and simultaneously optimizes data quality and diversity through a multiplicative priority function, comprehensively outperforming 9 baseline methods across 3 models and 6 benchmarks.
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: b2a58e645f1462c3
---

# The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph

> **arXiv**: [2410.12458](https://arxiv.org/abs/2410.12458)
> **Conference**: ICML 2025
> **Area**: LLM Evaluation
> **Author**: Minghao Wu, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari (Monash University)

## TL;DR

Proposes the GraphFilter method, which models SFT datasets as sentence–n-gram bipartite graphs and simultaneously optimizes data quality and diversity through a multiplicative priority function, comprehensively outperforming 9 baseline methods across 3 models and 6 benchmarks.

## Background & Motivation

The performance of Large Language Models (LLMs) during the Supervised Fine-Tuning (SFT) phase heavily relies on the quality and diversity of the training data. However, existing data selection methods often struggle to balance the two:

- **Quality-oriented** methods (e.g., AlpaGasus, Deita, SuperFilter) focus on selecting high-quality data but might neglect the diversity of linguistic patterns, leading to overfitting.
- **Diversity-oriented** methods (e.g., K-means clustering, InsTag topic tagging) focus on coverage breadth but may introduce low-quality data.
- Imbalances in either direction lead to suboptimal performance of the fine-tuned models.

Key Challenge: How to **simultaneously maximize quality and diversity** in data selection?

## Method

### Problem Formulation

Given an SFT dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the objective is to select a subset $\mathcal{S}_\pi$ of size $k$ that maximizes the performance of the fine-tuned model $f_\theta$ on downstream tasks:

$$\pi^* = \arg\max_\pi \mathcal{R}(f_\theta; \mathcal{D}_{\text{tst}}), \quad \text{s.t. } |\mathcal{S}_\pi| = k$$

### Bipartite Graph Modeling

The dataset is modeled as a bipartite graph $\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})$:

- **Sentence nodes** $\mathcal{U} = \{u_i\}_{i=1}^N$: The instruction part of each training sample.
- **n-gram nodes** $\mathcal{V} = \{v_j\}_{j=1}^M$: Including unigram (n=1), bigram (n=2), and trigram (n=3).
- **Edges** $\mathcal{E} \subseteq \mathcal{U} \times \mathcal{V}$: An edge exists if a sentence contains a specific n-gram.

### Iterative Selection Algorithm (GraphFilter)

Essentially, this is a **greedy solution to the set cover problem**:

1. Start with an empty set $\mathcal{S} = \emptyset$.
2. In each round, select the sentence with the highest priority $u^* = \arg\max_{u \in \mathcal{U}} \phi(u)$.
3. Add $u^*$ to $\mathcal{S}$, and remove all n-gram nodes associated with $u^*$ along with their connected edges from the graph.
4. Recalculate the priorities of the remaining sentences.
5. Repeat until $|\mathcal{S}| = k$.

A **max-heap** is utilized to optimize the selection process, reducing the complexity per round from $O(N)$ to $O(\log N)$.

### Priority Function

Quality and diversity are combined multiplicatively:

$$\phi(u) = \text{Quality}(u) \times \text{Diversity}(u)$$

**Quality Metric** — SuperFilter (IFD):

$$\text{Quality}(u) = \frac{\text{ppl}(y|x)}{\text{ppl}(y)}$$

Where $\text{ppl}(y|x)$ represents the perplexity of the response given the instruction, and $\text{ppl}(y)$ is the perplexity of the response alone. A larger ratio indicates a higher information gain from the instruction to the response.

**Diversity Metric** — TF-IDF Accumulation:

$$\text{Diversity}(u) = \sum_{v \in \mathcal{V}_u} \text{TF-IDF}(v)$$

$$\text{TF-IDF}(v) = \text{TF}(v) \times \log\frac{N}{d_v}$$

Note: $\mathcal{V}_u$ changes dynamically as the graph updates; covered n-grams no longer contribute to the score.

### Key Designs: Applying GraphFilter Only to Instructions

GraphFilter is applied exclusively to the instruction part of the SFT data, leaving the response part unprocessed. This design is based on the observation that "instruction diversity is more critical than response diversity."

## Experiments

### Main Results

Using the Magpie 300K dataset, a subset of 10K samples is selected for fine-tuning. Evaluations are conducted on 4 standardized benchmarks (MMLU, ARC, HellaSwag, GSM8K) and 2 LLM-as-Judge benchmarks (AlpacaEval-2.0, MT-Bench).

**Improvement of GraphFilter vs. the Strongest Baselines ($\mu_{\text{all}}$ overall score)**:

| Model | GraphFilter | Strongest Baseline | Gain |
|------|------------|---------|------|
| Gemma-2-2b | 35.06 | 34.36 (SuperFilter) | +0.70 |
| Mistral-7B-v0.3 | 39.66 | 38.40 (SuperFilter) | +1.26 |
| Llama-3-8B | — | — | +3.38 |

GraphFilter consistently outperforms all 9 baseline methods across **all three models**.

**Computational Efficiency**: GraphFilter accomplishes data selection without requiring a GPU, relying solely on CPU computation. This is significantly superior to methods requiring LLM inference (e.g., AlpaGasus requires calling ChatGPT, while ArmoRM requires a GPU for scoring).

### Ablation Study

1. **Quality vs. Diversity**:
    - Using only quality (SuperFilter): Performs poorly due to a lack of coverage.
    - Using only diversity (n-gram degree): Performs better than using only quality, demonstrating the importance of diversity.
    - Multiplicative combination: Dynamic combination achieves optimal performance, validating their complementarity.

2. **Instruction vs. Response Diversity**:
    - Applying GraphFilter to instructions: Achieves optimal results.
    - Applying to responses: Leads to a decline in performance.
    - This validates the dominant role of instruction diversity.

3. **Impact of Subset Size**:
    - Small subset (1K-5K): Quality is more critical.
    - Large subset (10K-50K): Diversity becomes more critical.
    - GraphFilter consistently maintains its advantage across all sizes.

4. **Analysis of Selected Subset Characteristics**:
    - The subset selected by GraphFilter yields the highest n-gram coverage.
    - The selected instructions are shorter yet more diverse.

## Highlights & Insights

- **Theoretical Support**: Formulates data selection as a set cover problem, where the greedy algorithm is guaranteed by an $H(r)$ approximation factor.
- **Extreme Efficiency**: No GPU is required, as the process runs entirely on CPU, vastly outperforming methods requiring LLM inference.
- **Simple Yet Effective Design**: The multiplication priority function offers clear intuition—selecting samples that are both high-quality and unique.
- **Comprehensive Experiments**: Evaluated across 3 models, 6 benchmarks, and 9 baselines, demonstrating reliable conclusions.
- **High Practicality**: Directly applicable to any SFT data selection scenario.

## Limitations & Future Work

- Validated only on the Magpie dataset; other synthetic or human-annotated SFT datasets were not covered.
- The quality metric is fixed to SuperFilter; compatibility with other quality metrics (such as reward model scores) remains unexplored.
- Uses only n-grams as a proxy for diversity, without accounting for semantic-level diversity.
- The greedy set-cover algorithm assumes n-gram independence, ignoring the semantic correlations between different n-grams.
- Evaluations are primarily focused on English tasks, leaving multilingual scenarios unverified.

## Rating

⭐⭐⭐⭐ (4/5)

The method is simple, efficient, and thoroughly experimented, elegantly formulating the data selection problem as a set cover problem. Although not highly complex technically, it achieves significant practical performance and strong usability, making it a solid contribution to the field of data selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](../../ACL2025/llm_evaluation/a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](../../ACL2025/llm_evaluation/atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](../../ACL2025/llm_evaluation/ecomscriptbench.md)

</div>

<!-- RELATED:END -->
