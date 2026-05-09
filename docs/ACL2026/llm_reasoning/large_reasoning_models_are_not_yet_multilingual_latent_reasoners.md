---
title: >-
  [Paper Note] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners
description: >-
  [ACL 2026][LLM Reasoning][multilingual reasoning] This paper systematically investigates the latent reasoning behavior of large reasoning models (LRMs) across 11 languages, finding that latent reasoning capability exists multilingually but is unevenly distributed (stronger for high-resource languages, weaker for low-resource ones), and that internal reasoning dynamics tend toward an English-centric shared pathway.
tags:
  - ACL 2026
  - LLM Reasoning
  - multilingual reasoning
  - latent reasoning
  - chain-of-thought truncation
  - representation analysis
  - reasoning models
date: 2026-05-08
content_hash: f4e6b11231d7c537
---

# Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners

**Conference**: ACL 2026  
**arXiv**: [2601.02996](https://arxiv.org/abs/2601.02996)  
**Code**: [https://github.com/cisnlp/multilingual-latent-reasoner](https://github.com/cisnlp/multilingual-latent-reasoner)  
**Area**: LLM Reasoning  
**Keywords**: multilingual reasoning, latent reasoning, chain-of-thought truncation, representation analysis, reasoning models

## TL;DR

This paper systematically investigates the latent reasoning behavior of large reasoning models (LRMs) across 11 languages, finding that latent reasoning capability exists multilingually but is unevenly distributed (stronger for high-resource languages, weaker for low-resource ones), and that internal reasoning dynamics tend toward an English-centric shared pathway.

## Background & Motivation

**State of the Field**: Large reasoning models (e.g., DeepSeek-R1) have achieved breakthrough performance on mathematical reasoning by generating explicit chain-of-thought (CoT). Recent studies have found that, before completing explicit reasoning steps, these models already form correct answers in their hidden states via "latent reasoning"—the model can "think ahead" to the result.

**Limitations of Prior Work**: Existing research on latent reasoning focuses almost exclusively on English, leaving multilingual latent reasoning behavior entirely unexplored. At the level of explicit reasoning, significant performance disparities across languages are already well established, with low-resource languages exhibiting notably lower reasoning quality.

**Root Cause**: If explicit reasoning varies across languages, does latent reasoning exhibit similar imbalance, or does it follow some language-agnostic internal mechanism?

**Paper Goals**: Two research questions—(RQ1) Do LRMs exhibit latent reasoning across languages, and how does its strength vary? (RQ2) Do different languages follow distinct internal latent reasoning pathways, or do they share a unified mechanism?

**Starting Point**: A reasoning trajectory truncation strategy is employed—the model is provided only partial reasoning steps, and its ability to produce a correct answer at the truncation point is observed. If the model answers correctly after seeing only a small fraction of reasoning steps, this indicates that the answer has been internally computed (i.e., latent reasoning is present).

**Core Idea**: Through multilingual truncation experiments and representation analysis, the paper reveals the multilingual characteristics of latent reasoning—present but uneven, with internal dynamics tending toward an English-centric shared pathway.

## Method

### Overall Architecture

Truncation experiments are conducted across 11 languages (covering high/medium/low-resource) on three model scales of DeepSeek-R1-distilled models (7B/14B/32B). Two mathematical reasoning benchmarks are used: MGSM (simple) and Multilingual AIME (difficult). The truncation ratio $r \in [0,1]$ controls the proportion of retained reasoning steps; model accuracy under partial reasoning information is evaluated. Internal reasoning dynamics are analyzed via logit lens and hidden state similarity.

### Key Designs

1. **Truncation-based Probing**:

    - Function: Quantifies the degree to which the model relies on explicit reasoning steps to arrive at the correct answer.
    - Mechanism: For each problem $x$, a full reasoning trajectory $c = (t_1, ..., t_T)$ is generated. For each truncation ratio $r$, the first $\lfloor r \cdot T \rfloor$ reasoning steps are retained, and the model is forced to output a final answer. A language-specific prefix is inserted after `<think>` to ensure consistency in the reasoning language. The truncated accuracy is compared against the rate at which the gold answer already appears in the visible trajectory, disentangling explicit and latent reasoning contributions.
    - Design Motivation: If the model answers correctly after seeing only 10% of the reasoning steps, and the answer has not yet appeared in the visible text, this strongly suggests that the model has internally computed the result via latent reasoning.

2. **Multi-dimensional Metrics (AUTC/AUGC/LRS)**:

    - Function: Quantifies the strength of latent reasoning and decouples it from explicit reasoning.
    - Mechanism: (a) AUTC (Area Under the Truncation-accuracy Curve) $= \int_0^1 a_k(r) dr$, measuring the earliness and robustness of correct predictions; (b) AUGC (Area Under the Gold-answer occurrence Curve) $= \int_0^1 g_k(r) dr$, measuring the degree to which the correct answer appears explicitly in the trajectory; (c) LRS (Latent Reasoning Score) $= \int_0^1 a_k(r)(1-g_k(r)) dr$, weighting accuracy by the proportion of steps where the answer has not yet appeared, specifically measuring non-explicit latent reasoning ability.
    - Design Motivation: Truncated accuracy alone is insufficient—the model may answer "correctly" simply because the answer was written in an early step. LRS provides a purer measure of latent reasoning by excluding this possibility.

3. **Representation Analysis (Logit Lens + Hidden State Similarity)**:

    - Function: Reveals whether internal reasoning pathways are shared across languages.
    - Mechanism: (a) Logit lens: hidden states are projected layer-by-layer into the vocabulary space to track how the rank of the correct answer token evolves across layers; (b) Cosine similarity between each language's hidden states and English hidden states is computed at each layer and reasoning step, analyzing representational convergence across languages.
    - Design Motivation: If logit lens trajectories are highly similar across languages, and non-English hidden states are strongly aligned with English, this indicates the existence of an English-centric shared latent reasoning pathway.

### Loss & Training

This is an analytical study; no training is involved. Inference and analysis are conducted using DeepSeek-R1-Distill-Qwen-{7B, 14B, 32B}.

## Key Experimental Results

### Main Results

**Truncation Metrics on MGSM (R1-Qwen-32B)**

| Language | AUTC | AUGC | LRS |
|----------|------|------|-----|
| EN (high-resource) | 0.75 | 0.25 | 0.53 |
| ZH (high-resource) | 0.70 | 0.30 | 0.45 |
| DE (high-resource) | 0.67 | 0.20 | 0.51 |
| JA (mid-resource) | 0.63 | 0.21 | 0.47 |
| BN (mid-resource) | 0.61 | 0.23 | 0.44 |
| SW (low-resource) | 0.38 | 0.20 | 0.30 |
| TE (low-resource) | 0.39 | 0.23 | 0.30 |

**Truncation Metrics on Multilingual AIME (R1-Qwen-32B)**

| Language | AUTC | AUGC | LRS |
|----------|------|------|-----|
| EN | 0.18 | 0.61 | 0.06 |
| ZH | 0.13 | 0.75 | 0.03 |
| SW | 0.01 | 0.05 | 0.00 |

### Ablation Study

**Effect of Model Scale on LRS (MGSM, English)**

| Model | AUTC | LRS |
|-------|------|-----|
| R1-Qwen-7B | 0.52 | 0.38 |
| R1-Qwen-14B | 0.59 | 0.44 |
| R1-Qwen-32B | 0.75 | 0.53 |

### Key Findings

- **Latent reasoning exists but is unevenly distributed**: On MGSM, high-resource languages such as English and Chinese achieve a pass@1 of approximately 0.2 at 0% truncation, indicating that the model can internally compute answers without any explicit reasoning. LRS for low-resource languages (Swahili, Telugu) is only about 60% of that for high-resource languages.
- **Task difficulty determines the detectability of latent reasoning**: LRS drops sharply on Multilingual AIME (EN: 0.53→0.06), indicating that complex problems require more explicit reasoning steps.
- **Internal reasoning pathways tend to be English-centric**: Logit lens analysis shows that layer-wise answer rank evolution trajectories are highly similar across languages; high-resource languages exhibit significantly higher cosine similarity with English hidden states than low-resource languages.
- **Larger model scale improves latent reasoning but does not close the gap**: LRS increases for all languages from 7B to 32B, but the high-/low-resource disparity persists.

## Highlights & Insights

- This is the first systematic study of LRM latent reasoning behavior across multiple languages, filling an important gap in the field.
- The LRS metric is elegantly designed: by decoupling accuracy from whether the answer has already appeared in the trajectory, it provides a clean measure of latent reasoning.
- The finding of an "English-centric shared reasoning pathway" has far-reaching implications for understanding the internal workings of multilingual LLMs—even when the input is in Chinese, the model may be "thinking in English" internally.
- Non-zero accuracy at 0% truncation constitutes the most compelling evidence for latent reasoning.

## Limitations & Future Work

- Evaluation is limited to mathematical reasoning tasks; other reasoning types such as logical reasoning and code reasoning are not covered.
- Analysis is conducted primarily on distilled models; the behavior of the original DeepSeek-R1 may differ.
- Causal mechanism analysis is limited—the observed correlations (entropy and correctness) do not imply causation.
- Future work could explore improving latent reasoning in low-resource languages through post-training on multilingual reasoning data.

## Related Work & Insights

- This paper bridges two lines of research: multilingual reasoning (known language gaps at the explicit level) and latent reasoning (a new dimension at the implicit level).
- The logit lens analysis methodology can be generalized to multilingual comparative studies of other model capabilities.
- The findings provide a representational-level explanation for the "translate-then-solve" strategy—the model internally tends to reason through an English pathway regardless.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First study of multilingual latent reasoning; both the problem formulation and metric design are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 11 languages, 3 model scales, and 2 benchmarks, though non-mathematical tasks are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions are clearly articulated, experimental design is rigorous, and conclusions are precisely stated.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](../../ICLR2026/llm_reasoning/mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)
- [\[ACL 2026\] TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)

<!-- RELATED:END -->
