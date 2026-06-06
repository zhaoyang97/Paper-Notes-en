---
title: >-
  [Paper Note] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners
description: >-
  [ACL 2026][LLM Reasoning][Multilingual Reasoning] This paper systematically investigates the latent reasoning behavior of Large Reasoning Models (LRMs) across 11 languages. It finds that latent reasoning capabilities exi…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Multilingual Reasoning"
  - "Latent Reasoning"
  - "Chain-of-Thought Truncation"
  - "Representation Analysis"
  - "Reasoning Models"
date: 2026-05-08
content_hash: 7aa7a6fb1526e9c8
---

# Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners

**Conference**: ACL 2026  
**arXiv**: [2601.02996](https://arxiv.org/abs/2601.02996)  
**Code**: [https://github.com/cisnlp/multilingual-latent-reasoner](https://github.com/cisnlp/multilingual-latent-reasoner)  
**Area**: LLM Reasoning  
**Keywords**: Multilingual Reasoning, Latent Reasoning, Chain-of-Thought Truncation, Representation Analysis, Reasoning Models

## TL;DR

This paper systematically investigates the latent reasoning behavior of Large Reasoning Models (LRMs) across 11 languages. It finds that latent reasoning capabilities exist in a multilingual context but are unevenly distributed (strong in high-resource languages, weak in low-resource ones), and internal reasoning dynamics tend to follow an English-centric shared path.

## Background & Motivation

**Background**: Large Reasoning Models (e.g., DeepSeek-R1) have achieved breakthroughs in tasks like mathematical reasoning by generating explicit Chain-of-Thought (CoT). Recent studies suggest that before completing explicit reasoning steps, these models already form correct answers within their hidden states through "latent reasoning"—the model "thinks ahead" of the result.

**Limitations of Prior Work**: Existing research on latent reasoning focuses almost exclusively on English, leaving its performance in multilingual scenarios unknown. At the explicit reasoning level, multilingual performance is known to vary significantly, with reasoning quality being notably poorer for low-resource languages.

**Key Challenge**: If explicit reasoning performs unevenly across languages, does latent reasoning exhibit a similar imbalance? Or does latent reasoning follow a language-agnostic internal mechanism?

**Goal**: Two research questions—(RQ1) Do LRMs exhibit latent reasoning capabilities across various languages, and how does the strength vary? (RQ2) Do different languages follow distinct internal latent reasoning paths, or do they share a unified mechanism?

**Key Insight**: Utilize a reasoning trajectory truncation strategy—providing the model with only partial reasoning steps to observe if it can provide the correct answer at the truncation point. Correct answers given after seeing only a few reasoning steps strongly indicate that the model has internally computed the answer (i.e., latent reasoning exists).

**Core Idea**: Reveal the multilingual characteristics of latent reasoning through multilingual truncation experiments and representation analysis—showing that it exists but is uneven, and internally converges toward an English-centric shared path.

## Method

### Overall Architecture

Truncation experiments are conducted on three scales of models (7B/14B/32B) distilled from DeepSeek-R1 across 11 languages (covering high/mid/low resources). Two mathematical reasoning benchmarks are used: MGSM (simple) and Multilingual AIME (hard). Reasoning steps are controlled by a truncation ratio $r \in [0,1]$. Accuracy is evaluated under partial reasoning information, and internal reasoning dynamics are analyzed via logit lens and hidden state similarity.

### Key Designs

1.  **Truncation-based Probing**:
    - **Function**: To quantify the extent to which the model relies on explicit reasoning steps to arrive at the correct answer.
    - **Mechanism**: For each problem $x$, a complete reasoning trajectory $c = (t_1, ..., t_T)$ is generated. For different truncation ratios $r$, the first $\lfloor r \cdot T \rfloor$ reasoning steps are kept, then the model is forced to output the final answer. Language consistency is ensured by inserting language-specific prefixes after `<think>`. The contribution of explicit versus latent reasoning is distinguished by comparing the truncated accuracy with the ratio of "whether the gold answer has already appeared in the visible trajectory."
    - **Design Motivation**: If a model answers correctly when seeing only 10% of reasoning steps and the answer has not yet appeared in the visible text, it strongly suggests the model has internally computed the result through latent reasoning.

2.  **Multidimensional Metrics (AUTC/AUGC/LRS)**:
    - **Function**: To quantify the strength of latent reasoning and decouple it from explicit reasoning.
    - **Mechanism**: (a) AUTC (Area Under the Truncation-accuracy Curve) = $\int_0^1 a_k(r) dr$, measuring the earliness and robustness of correct predictions; (b) AUGC (Area Under the Gold-occurrence Curve) = $\int_0^1 g_k(r) dr$, measuring the explicit appearance of the correct answer in the trajectory; (c) LRS (Latent Reasoning Score) = $\int_0^1 a_k(r)(1-g_k(r)) dr$, weighting accuracy by the proportion where the "answer has not appeared," specifically measuring non-explicit latent reasoning capability.
    - **Design Motivation**: Truncated accuracy alone is insufficient—a model might be "correct" simply because it wrote the answer in early steps. LRS provides a purer measure of latent reasoning by excluding this possibility.

3.  **Representation Analysis (Logit Lens + Hidden State Similarity)**:
    - **Function**: To reveal whether internal reasoning paths are shared across different languages.
    - **Mechanism**: (a) Logit lens: Projecting hidden states into the vocabulary space layer-by-layer to track how the rank of the correct answer token evolves; (b) Calculating the cosine similarity between hidden states of different languages and English hidden states across layers and reasoning steps to analyze cross-lingual representation convergence.
    - **Design Motivation**: Highly similar logit lens trajectories and high alignment of non-English hidden states with English would indicate the existence of a shared, English-centric latent reasoning path.

### Loss & Training

This is an analytical study and does not involve training. Inference and analysis are performed using three model scales: DeepSeek-R1-Distill-Qwen-{7B, 14B, 32B}.

## Key Experimental Results

### Main Results

**Truncation Metrics on MGSM (R1-Qwen-32B)**

| Language | AUTC | AUGC | LRS |
| :--- | :--- | :--- | :--- |
| EN (High) | 0.75 | 0.25 | 0.53 |
| ZH (High) | 0.70 | 0.30 | 0.45 |
| DE (High) | 0.67 | 0.20 | 0.51 |
| JA (Mid) | 0.63 | 0.21 | 0.47 |
| BN (Mid) | 0.61 | 0.23 | 0.44 |
| SW (Low) | 0.38 | 0.20 | 0.30 |
| TE (Low) | 0.39 | 0.23 | 0.30 |

**Truncation Metrics on Multilingual AIME (R1-Qwen-32B)**

| Language | AUTC | AUGC | LRS |
| :--- | :--- | :--- | :--- |
| EN | 0.18 | 0.61 | 0.06 |
| ZH | 0.13 | 0.75 | 0.03 |
| SW | 0.01 | 0.05 | 0.00 |

### Ablation Study

**Impact of Model Scale on LRS (MGSM, English)**

| Model | AUTC | LRS |
| :--- | :--- | :--- |
| R1-Qwen-7B | 0.52 | 0.38 |
| R1-Qwen-14B | 0.59 | 0.44 |
| R1-Qwen-32B | 0.75 | 0.53 |

### Key Findings

- **Latent reasoning exists but is uneven**: On MGSM, high-resource languages like English/Chinese achieve a pass@1 of ~0.2 at 0% truncation, showing the model computes answers internally without explicit reasoning. However, the LRS for low-resource languages (Swahili, Telugu) is only about 60% of that of high-resource languages.
- **Task difficulty determines latent reasoning detectability**: LRS drops sharply on Multilingual AIME (EN from 0.38→0.06), indicating complex problems require more explicit reasoning steps.
- **Internal reasoning paths tend to be English-centric**: Logit lens shows highly similar layer-wise answer rank evolution across languages; cosine similarity between high-resource languages and English hidden states is significantly higher than that of low-resource languages.
- **Model scale enhances latent reasoning but does not eliminate the gap**: LRS increases for all languages from 7B to 32B, but the gap between high and low-resource languages persists.

## Highlights & Insights

- This is the first systematic study of LRM latent reasoning behavior in a multilingual dimension, filling a significant knowledge gap.
- The LRS metric is ingeniously designed to provide a pure measure of latent reasoning by decoupling accuracy from the explicit occurrence of answers in the trajectory.
- The discovery of an "English-centric shared reasoning path" has profound implications for understanding the internal mechanisms of multilingual LLMs—suggesting that even with Chinese input, the model may be "thinking in English" internally.
- The non-zero accuracy at a 0% truncation ratio provides the strongest evidence for latent reasoning.

## Limitations & Future Work

- Evaluation is limited to mathematical tasks, excluding other types like logical or code reasoning.
- Analysis was mainly performed on distilled models; behaviors of the original DeepSeek-R1 might differ.
- Causal mechanism analysis is limited—observed correlations (between entropy and correctness) do not equal causality.
- Future work could explore enhancing latent reasoning in low-resource languages through post-training with multilingual reasoning data.

## Related Work & Insights

- Connects two research lines: multilingual reasoning (known language gaps at the explicit level) and latent reasoning (a new dimension at the implicit level).
- The logit lens analysis method can be generalized to comparative multilingual studies of other model capabilities.
- Provides a representation-level explanation for the "translate-then-solve" strategy—models inherently lean toward reasoning via English paths.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First multilingual latent reasoning study with creative problem definition and metric design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 11 languages, 3 model scales, and 2 benchmarks, though lacking non-mathematical tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear research questions, rigorous experimental design, and precise formulation of conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)
- [\[ICLR 2026\] mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](../../ICLR2026/llm_reasoning/mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)

</div>

<!-- RELATED:END -->
