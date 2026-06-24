---
title: >-
  [Paper Note] Untitled
description: >-
  [ACL 2025][LLM (Other)][NLU] Academic paper note for Untitled.
tags:
  - ACL 2025
  - LLM (Other)
  - NLU
  - LLM
date: 2026-05-08
content_hash: 0002c50b0b8ead3c
---
# Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding

## Basic Information

- **Conference**: ACL 2025
- **arXiv**: [2502.13738](https://arxiv.org/abs/2502.13738)
- **Code**: [https://github.com/Romainpkq/CD_ICL](https://github.com/Romainpkq/CD_ICL)
- **Area**: LLM / NLP (LLM NLP)
- **Keywords**: In-Context Learning, Contrastive Decoding, Input-Label Mapping, NLU, LLM
- **TL;DR**: Proposes ICCD (In-Context Contrastive Decoding), which enhances LLM's utilization of input-label mapping information in ICL by contrasting the output distributions of positive and negative in-context examples, yielding consistent and significant performance improvements across 7 NLU tasks without requiring training.

## Background & Motivation

In-Context Learning (ICL) is a core capability that allows LLMs to adapt to new tasks using a few examples. Existing research identifies two key factors that contribute to the success of ICL:

**Task Recognition (TR)**: Identifying the task type from the examples and leveraging pre-trained knowledge to make predictions.

**Task Learning (TL)**: Directly learning the input-label mapping relationships from the examples.

**Core Problem**: LLMs **over-rely on pre-trained knowledge** and **neglect input-label mapping information** in ICL.

**Classic Finding**: Min et al. (2022) showed that even when the labels of ICL examples are randomly shuffled, model performance does not drop significantly. This indicates that the model relies more on the signal of "observing a sentiment analysis task" (Task Recognition) rather than the specific mapping of "this positive sentence corresponds to the positive label" (Task Learning).

**Practical Impact**: When the task is inconsistent with the pre-training distribution (such as custom label mappings), the model makes erroneous predictions due to neglecting the mapping relationships in the examples.

## Method

### Core Idea

Drawing inspiration from contrastive decoding techniques, it **extracts and reinforces the input-label mapping information** by contrasting the output distributions of positive and negative in-context examples.

### ICCD Formula Derivation

Standard ICL decoding: $y \sim p_\theta(y | \mathbf{c}, \mathcal{T}(x))$

ICCD enhanced decoding:
$$y_t \sim \text{softmax}(\mathbf{z}_t + \alpha(\mathbf{z}_t - \mathbf{z}_t^-))$$

Equivalent form:
$$\tilde{p}_\theta(y|\mathbf{c}, \mathbf{c}^-, \mathcal{T}(x)) \propto p_\theta(y|\mathbf{c}, \mathcal{T}(x)) \left(\frac{p_\theta(y|\mathbf{c}, \mathcal{T}(x))}{p_\theta(y|\mathbf{c}^-, \mathcal{T}(x))}\right)^\alpha$$

**Intuitive Understanding**:
- $\mathbf{z}_t$: Output distribution of positive examples (containing correct mapping + pre-trained knowledge)
- $\mathbf{z}_t^-$: Output distribution of negative examples (containing incorrect mapping + same pre-trained knowledge)
- $\mathbf{z}_t - \mathbf{z}_t^-$: The difference represents the **pure input-label mapping signal**
- Adding this signal back to the original output enhances the model's focus on the mapping information.

### Negative Example Construction (Key Designs)

**Why modify the inputs instead of the labels?**
- Directly modifying labels introduces completely different label biases, distorting the mapping information.
- Modifying inputs maintains the label distribution unchanged while only altering the mapping relationships.

**Specific Method**:
For each example $(x_i, y_i)$:
1. Randomly select a different label $y_j$ ($y_j \neq y_i$).
2. Randomly select an input $x_j$ with label $y_j$ from the exemplar pool.
3. Construct the negative example $(x_j, y_i)$ — where the input and label do not match.

**$\alpha$ Parameter**: Controls the importance of the input-label mapping information, set to 1 by default.

## Experiments

### Experimental Setup
- **Models**: Llama3.2-1B/3B, Llama3.1-8B, Qwen2-0.5B/1.5B/7B (6 scales in total)
- **Tasks**: 7 NLU tasks — SST-2, SST-5, CR, Subj, QNLI, MNLI, AG_NEWS
- **Exemplar Selection Methods**: Random, BM25, TopK
- **ICL Setup**: 16-shot, $\alpha=1$

### Main Results (Table 2)

| Model | Method | SST2 | Subj | QNLI | MNLI | Avg |
|------|------|------|------|------|------|-----|
| Llama3.2-1B | Regular | 89.8 | 72.8 | 53.5 | 36.6 | 66.1 |
| | **ICCD** | **91.1** | **83.0** | **53.8** | **39.2** | **68.3 (+2.1)** |
| Llama3.2-3B | Regular | 93.7 | 86.0 | 54.2 | 56.9 | 72.9 |
| | **ICCD** | **94.0** | **92.1** | **57.2** | **57.0** | **74.6 (+1.7)** |
| Llama3.1-8B | Regular | 96.7 | 94.0 | 60.3 | 65.3 | 77.6 |
| | **ICCD** | **96.5** | **96.1** | **65.4** | **67.5** | **79.4 (+1.8)** |
| Qwen2-1.5B | Regular | 95.2 | 72.3 | 60.2 | 61.8 | 72.3 |
| | **ICCD** | **95.1** | **81.5** | **61.8** | **65.2** | **74.6 (+2.3)** |
| Qwen2-7B | Regular | 96.0 | 82.3 | 71.4 | 78.7 | 79.4 |
| | **ICCD** | **96.3** | **90.4** | **72.8** | **79.9** | **81.3 (+1.9)** |

### Compatibility with Different Exemplar Selection Methods (Table 1)

| Model | Method | Random | BM25 | TopK |
|------|------|--------|------|------|
| Llama3.1-8B | Regular | 77.6 | 79.7 | 80.2 |
| | ICCD | **79.4** | **80.8** | **80.9** |

ICCD brings improvements across all exemplar selection methods with lower variance (greater stability).

### Larger Label Spaces (Table 3)

| Model | Method | TREC (6 classes) | Dbpedia (14 classes) |
|------|------|-----------|---------------|
| Llama3.2-1B | Regular | 40.0 | 85.6 |
| | ICCD | **46.2 (+6.2)** | **90.5 (+4.9)** |
| Llama3.1-8B | Regular | 41.0 | 87.5 |
| | ICCD | **46.6 (+5.6)** | **93.8 (+6.3)** |

The larger the label space, the greater the improvement brought by ICCD (TREC +5-6%, Dbpedia +5-8%).

### Alignment (Chat) Model Validation

ICCD is equally effective on Llama3.2-1B/3B-Instruct and Llama3.1-8B-Instruct, demonstrating its applicability to instruction-tuned and RLHF-aligned models.

### Key Analysis

**1. Comparison of Negative Example Construction Methods (Table 4)**

| Method | Random | BM25 | TopK |
|------|--------|------|------|
| Regular | 77.6 | 79.7 | 80.2 |
| +NULL (No exemplars/blank) | 73.0 | 75.8 | 76.5 |
| +Label (Modified label) | 77.3 | 79.5 | 80.0 |
| **+Input (Modified input)** | **79.4** | **80.8** | **80.9** |

- NULL objective reduces performance (removing the contribution of pre-trained knowledge).
- Modifying labels is nearly ineffective (introducing label bias counteracts the mapping signal).
- **Modifying inputs is optimal** (maintaining the label distribution unchanged while only altering the mapping relationships).

**2. KL Divergence Analysis (Table 5)**
The KL divergence between the output distributions of positive and negative examples is significant on most tasks (SST2: 0.64, AGNEWS: 0.79), validating that the contrast indeed isolates distinct mapping information.

**3. Influence of Number of Shots**
As the number of shots increases from 1 to 16, the improvement margin of ICCD expands. This is because more examples provide richer input-label mapping information for contrast.

**4. $\alpha$ Sensitivity Analysis (Table 6)**
- Performance continuously improves as $\alpha$ increases from 0 to 1.
- Performance stabilizes once $\alpha \geq 1.0$.
- For advanced selection methods (e.g., TopK), an excessively large $\alpha$ may slightly degrade performance.

## Highlights & Insights

1. **Extremely Simple Method**: The core idea is summarized in a single formula; it requires no training and is plug-and-play.
2. **Clear Theoretical Motivation**: Starting from the separation of TR and TL in ICL, it precisely targets the problem (the neglect of input-label mapping).
3. **Clever Design in Negative Example Construction**: Modifying inputs instead of labels avoids introducing label bias.
4. **Broad Compatibility**: Effective across model series (Llama/Qwen), model scales (0.5B-8B), exemplar selection methods, and alignment states.
5. **Open Source Code**: Facilitates reproduction and extension.
6. **Contribution to Understanding ICL Mechanisms**: Provides a new perspective for understanding the internal mechanisms of ICL using analytical tools such as KL divergence.

## Limitations & Future Work

1. **Doubled Computational Cost**: Each prediction requires two forward passes (positive + negative examples), doubling the inference time.
2. **While $\alpha = 1$ Works Decently, It May Not Be Globally Optimal**: Tuning may be required for different tasks.
3. **Only Validated on Classification Tasks**: Not yet validated on generative tasks (such as translation, summarization, or dialogue).
4. **Randomness of Negative Examples**: The construction method relies on random sampling, which may introduce variance.
5. **Upper Limit of Model Scale**: Tested up to 8B/7B models, not yet validated on 70B+ models.
6. **Limited Improvement on Fine-Grained Tasks Like SST-5**: Potentially due to the inherently blurry boundaries between labels.

## Related Work & Insights

- **ICL Mechanism Research**: Label randomization experiments in Min et al. (2022), TR vs TL in Pan et al. (2023).
- **Contrastive Decoding**: Contrastive Decoding (Li et al., 2023), DoLa, CD-ICL.
- **ICL Exemplar Selection**: Random, BM25, TopK (Liu et al., 2022), KATE.
- **LLMs**: Llama-3 series, Qwen2 series.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — Cleverly addresses the mapping neglect problem in ICL from a contrastive decoding perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Exhaustive evaluation with 6 model scales, 7 tasks, 3 selection methods, and multi-dimensional analysis.
- **Value**: ⭐⭐⭐⭐ — No training required and the code is open-source, though doubled computational cost remains a practical roadblock.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations and comprehensive ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)
- [\[ACL 2025\] When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)

</div>

<!-- RELATED:END -->
