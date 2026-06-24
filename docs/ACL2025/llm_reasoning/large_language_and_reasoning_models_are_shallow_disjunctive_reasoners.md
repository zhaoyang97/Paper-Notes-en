---
title: >-
  [Paper Note] Large Language and Reasoning Models are Shallow Disjunctive Reasoners
description: >-
  [ACL 2025][Reasoning][Systematic Reasoning] This paper evaluates the systematic generalization capabilities of LLMs and LRMs on disjunctive rule reasoning tasks that require composing multiple reasoning paths, using a synthetic spatial and temporal reasoning benchmark (STaR). It finds that even reasoning models like o3-mini can only handle single-path reasoning, with performance degrading drastically in multi-path disjunctive reasoning scenarios.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Systematic Reasoning"
  - "Disjunctive Rules"
  - "Spatial-Temporal Reasoning"
  - "Large Reasoning Models"
  - "OOD Generalization"
date: 2026-05-08
content_hash: 61af3ebdae00fb34
---

# Large Language and Reasoning Models are Shallow Disjunctive Reasoners

**Conference**: ACL 2025  
**arXiv**: [2503.23487](https://arxiv.org/abs/2503.23487)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Systematic Reasoning, Disjunctive Rules, Spatial-Temporal Reasoning, Large Reasoning Models, OOD Generalization

## TL;DR

This paper evaluates the systematic generalization capabilities of LLMs and LRMs on disjunctive rule reasoning tasks that require composing multiple reasoning paths, using a synthetic spatial and temporal reasoning benchmark (STaR). It finds that even reasoning models like o3-mini can only handle single-path reasoning, with performance degrading drastically in multi-path disjunctive reasoning scenarios.

## Background & Motivation

**Background**: In recent years, LLMs have demonstrated strong capabilities in math and coding tasks. "Large Reasoning Models" (LRMs) like DeepSeek-R1 and o3-mini, post-trained via reinforcement learning, have further improved analytical reasoning through Chain-of-Thought (CoT). The academic community has high expectations for the reasoning capabilities of these models.

**Limitations of Prior Work**: However, existing evaluations primarily focus on math and coding, which are highly susceptible to dataset contamination and memorization of training data. More importantly, these tasks usually only require deriving answers along a single reasoning chain (Horn rules), failing to verify the true capability of models in complex reasoning scenarios that require combining multiple reasoning paths. Static benchmarks like GSM8k and MMLU also present the risk of leaking into training corpora.

**Key Challenge**: Chain-of-Thought (CoT) naturally fits Horn-style single reasoning chain problems. However, many real-world reasoning tasks involve **disjunctive rules**—where known facts only lead to a disjunction (OR) of candidate conclusions, requiring the combination of information from multiple paths to pinpoint the unique answer. Existing evaluations cannot distinguish whether a model is truly reasoning or merely exploiting shallow pattern-matching shortcuts.

**Goal**: Systematically evaluate the disjunctive reasoning capabilities of LLMs and LRMs on precisely controlled synthetic benchmarks, specifically quantifying their performance in out-of-distribution (OOD) generalization scenarios.

**Key Insight**: The authors select qualitative spatial reasoning (RCC-8) and temporal reasoning (Interval Algebra) as the evaluation domains. The composition rules for these two domains can be precisely described via disjunctive rules. Problem difficulty is strictly controlled by path length $k$ and path count $b$, and is polynomial-time solvable—theoretically within the capability of LRMs.

**Core Idea**: Utilizing the controllable difficulty parameters of the STaR benchmark to reveal that LLMs and LRMs are merely "shallow disjunctive reasoners"—capable of handling single reasoning paths but failing to effectively combine multiple paths.

## Method

### Overall Architecture

This paper does not propose a new method but designs a systematic evaluation framework. The input is a directed labeled graph $\mathcal{G}$, where vertices represent entities and edges denote RCC-8 or IA relations. The goal is to infer the unique relation between specified head and tail entities. The model needs to: (1) understand the candidate outcomes for each relation composition in the composition table; (2) perform chain reasoning along each path to obtain the set of candidate relations; (3) take the intersection of outcomes from all paths to obtain the unique answer.

### Key Designs

1. **Difficulty Control Mechanism of the STaR Benchmark**:

    - Function: Precisely control problem difficulty using two parameters.
    - Mechanism: $k$ controls the length of each simple path (requiring $k-1$ rule compositions), and $b$ controls the number of independent paths between the head and tail entities (requiring the intersection of disjunctive outcomes from $b$ paths). The training set spans $k \in \{2,3,4\}, b \in \{1,2,3\}$, while the test set extends to $k \in \{2,...,10\}, b \in \{1,2,3,4\}$.
    - Design Motivation: To ensure the OOD test set is structurally more complex than the training set, while guaranteeing that all atomic knowledge required to solve the puzzles has appeared in the training set.

2. **Three Evaluation Settings: Zero-shot / Few-shot / Fine-tuned**:

    - Function: Evaluate the instruction-following, in-context learning, and post-learning generalization capabilities of the models, respectively.
    - Mechanism: Zero-shot and Few-shot settings directly provide the composition table (encoded as integers) in the prompt. The Fine-tuned setting utilizes the full training set (~57,600 examples for RCC-8, ~93,400 for IA) for fine-tuning.
    - Design Motivation: Distinguish whether the model can reason from rules provided in the prompt (zero/few-shot) vs. whether it can learn and generalize from data (fine-tuned).

3. **Dedicated Analysis on LRMs: o3-mini vs. Distilled R1 Models**:

    - Function: Evaluate specialized performance patterns of reasoning models on disjunctive reasoning.
    - Mechanism: Perform zero-shot evaluation on o3-mini and Qwen R1 distilled models under identical configurations, and analyze the relationship between the number of generated tokens and problem difficulty.
    - Design Motivation: Verify whether the CoT mechanism of LRMs is effective in multi-path disjunctive reasoning, and whether thinking-time allocation is reasonable.

### Loss & Training

Fine-tuning is conducted using QLoRA (4-bit quantization), AdamW optimizer (learning rate $2 \times 10^{-4}$), gradient accumulation of 4 steps, for exactly 1 epoch. LoRA adapters are applied to Q/K/V/O/Gate/Up/Down projections.

## Key Experimental Results

### Main Results

| Model | Setting | RCC-8 (k=9,b=1) Acc | RCC-8 (k=9,b=2) Acc | RCC-8 (k=9,b=3) Acc |
|------|------|---------------------|---------------------|---------------------|
| o3-mini | Zero-shot | 0.90 | 0.48 | 0.30 |
| Qwen2.5-7B (R1) | Zero-shot | 0.08 | 0.06 | 0.12 |
| Qwen2.5-14B (R1) | Zero-shot | 0.07 | 0.02 | 0.07 |
| Qwen2.5-14B | Fine-tuned | ~0.40 | ~0.35 | ~0.35 |
| Qwen2.5-72B | Zero-shot | ~0.15 | ~random | ~random |

### Ablation Study (Analysis of Fine-tuned Models by Relation Category, k=9, b=2)

| Relation | Qwen2.5-14B F1 | o3-mini F1 | Description |
|------|------|------|------|
| EQ | 1.00 | 0.60 | The fine-tuned model leverages the "EQ-chain" shortcut to predict perfectly, whereas o3-mini is imperfect |
| NTPPI | 0.81 | 0.33 | Fine-tuned model leverages learned rules |
| NTPP | 0.00 | 0.33 | Fine-tuned model fails completely; o3-mini shows basic capability |
| PO | 0.16 | 0.33 | Hard relation requiring true disjunctive reasoning |

### Key Findings

- o3-mini achieves 0.90 accuracy when $b=1$ (single path), but drops sharply to 0.48/0.30 when $b \geq 2$, indicating that the CoT mechanism only fits single-path reasoning.
- Fine-tuned LLMs differ qualitatively from o3-mini: they achieve perfect accuracy on certain relations by learning simple shortcuts (e.g., EQ transitive chains) but fail completely on relations requiring true disjunctive reasoning.
- R1 distilled models perform worse than random guessing across all settings with $b \geq 2$.
- Fine-tuned Qwen-series models consistently outperform Llama-series; even a 7B Qwen beats a 70B Llama-3.3.

## Highlights & Insights

- **Precise Control of Synthetic Benchmarks**: The dual parameters $k$ and $b$ precisely control reasoning chain length and disjunctive complexity, making the boundaries of model reasoning performance clearly measurable. This design paradigm is transferable to any scenario requiring the evaluation of compositional generalization.
- **"Orthogonal" Failure Modes of LRMs vs. Fine-tuned LLMs**: o3-mini reasons by systematically applying rules but is error-prone (showing non-zero, uniform performance across relations), whereas fine-tuned models exploit shortcuts to perform perfectly on simple relations but fail entirely on complex ones. This insight is highly valuable for understanding reasoning mechanisms under different training paradigms.
- **Counter-intuitive Finding on CoT Token Allocation**: R1 models generate *fewer* thinking tokens as $b$ increases, seemingly "giving up" on reasoning in multi-path scenarios—implying that current CoT mechanisms lack a deep understanding of problem structure.

## Limitations & Future Work

- Only two qualitative reasoning domains (RCC-8 and IA) were evaluated; generalization to other disjunctive reasoning tasks remains unknown.
- Due to compute limits, reasoning models were only evaluated on a small subset of $(k,b)$ configurations.
- Larger models (e.g., o3, Claude, etc.) might perform better but were not included in the evaluation.
- Specialized training strategies targeting disjunctive reasoning (e.g., incorporating explicit supervision of multi-path intersections during training) were not explored.

## Related Work & Insights

- **vs CLUTRR (Kinship relation reasoning)**: CLUTRR only involves Horn-clause reasoning along a chain, whereas STaR introduces a disjunctive dimension, making it more challenging.
- **vs Path-of-Thoughts**: The latter helps LLMs handle relational reasoning through graph structure extraction but does not involve the composition of disjunctive rules.
- The evaluation framework presented in this paper can serve as a standard benchmark for evaluating the disjunctive capabilities of future reasoning models.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of disjunctive reasoning is a crucial contribution to LRM evaluation, though the experimental design is a direct application of the existing STaR benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers 11 models, 3 settings, and fine-grained analyses, but lacks more recent models.
- Writing Quality: ⭐⭐⭐⭐⭐ The structure is clear, showcasing a complete logical chain from problem definition to experimental analysis.
- Value: ⭐⭐⭐⭐ Highly inspiring for understanding the reasoning limitations of LRMs, but lacks proposed solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ranked Voting based Self-Consistency of Large Language Models](ranked_voting_based_self-consistency_of_large_language_models.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](../../ACL2026/llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)

</div>

<!-- RELATED:END -->
