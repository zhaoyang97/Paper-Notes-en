---
title: >-
  [Paper Note] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency
description: >-
  [NeurIPS 2025][LLM Alignment][Jailbreak Attack] This paper proposes JAIL-CON, a jailbreak attack framework based on task concurrency. By interleaving harmful and benign tasks at the word level, it exploits LLMs' ability to handle concurrent tasks to bypass safety mechanisms, while the resulting concurrent outputs exhibit stronger evasiveness against guardrails.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - Jailbreak Attack
  - Task Concurrency
  - LLM Safety
  - Guardrail Bypass
  - Adversarial Robustness
date: 2026-05-08
content_hash: 1d701475653ab7e6
---

# Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency

**Conference**: NeurIPS 2025
**arXiv**: [2510.21189](https://arxiv.org/abs/2510.21189)
**Code**: None
**Area**: LLM Alignment
**Keywords**: Jailbreak Attack, Task Concurrency, LLM Safety, Guardrail Bypass, Adversarial Robustness

## TL;DR

This paper proposes JAIL-CON, a jailbreak attack framework based on task concurrency. By interleaving harmful and benign tasks at the word level, it exploits LLMs' ability to handle concurrent tasks to bypass safety mechanisms, while the resulting concurrent outputs exhibit stronger evasiveness against guardrails.

## Background & Motivation

Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, yet remain vulnerable to misuse for generating harmful content. Jailbreak attacks further amplify this risk.

**Limitations of Prior Work**:
- Existing jailbreak attacks predominantly follow **sequential logic**: LLMs comprehend and respond to tasks one at a time.
- These include role-playing, encoding obfuscation, prompt injection, and similar techniques.
- Their outputs are also sequential, making them relatively easy for guardrails to detect.

**Key Observation**:
- **Concurrency** is a natural extension beyond sequential scenarios, yet it has been largely overlooked in LLM safety research.
- LLMs are in fact capable of processing concurrent tasks—i.e., responding to multiple distinct tasks simultaneously within a single output.
- Interleaving harmful and benign tasks at the word level can substantially reduce the probability of being intercepted by safety filters.

**Security Threat**: Harmful content embedded within concurrent outputs is interspersed among benign content, making it difficult for traditional keyword- or semantic-matching-based guardrails to detect effectively.

## Method

### Overall Architecture

JAIL-CON is an **iterative attack framework** comprising the following core steps:

1. **Task Encoding**: Interleave harmful and benign tasks at the word level.
2. **Concurrent Request**: Submit the encoded concurrent request to the target LLM.
3. **Response Decoding**: Extract the harmful task's response from the LLM's concurrent output.
4. **Iterative Optimization**: Iteratively refine the encoding strategy based on attack success rate and response quality.

### Key Designs

**Word-level Task Interleaving**

Unlike conventional multi-task prompting—which presents two tasks in separate segments (e.g., "first answer A, then answer B")—JAIL-CON interleaves tasks at the word level.

Given a harmful task $T_h = [w_1^h, w_2^h, ..., w_n^h]$ and a benign task $T_b = [w_1^b, w_2^b, ..., w_m^b]$, the encoded input is:

$$T_{interleaved} = [w_1^h, w_1^b, w_2^h, w_2^b, ..., w_k^h, w_k^b, ...]$$

where adjacent words encode divergent intents (Adjacent Words, Divergent Intents).

**Verification of Concurrent Capability**

Prior to launching the attack, the authors first verify that LLMs are genuinely capable of handling concurrent tasks:
- Word-level interleaved tasks are tested on mathematical QA and general QA benchmarks.
- LLMs are found to answer both interleaved tasks simultaneously with reasonably high accuracy.
- This capability provides the empirical foundation for concurrent jailbreak attacks.

**Guardrail Bypass Mechanism**

Harmful content in concurrent outputs appears in a non-contiguous manner, exhibiting the following characteristics:
- Harmful tokens are separated by benign tokens, reducing detection rates for keyword-level matching.
- The overall semantics are diluted by benign task content, lowering the confidence of semantic-level detectors.
- Guardrails are typically designed for sequential outputs and exhibit weaker detection capability against concurrent outputs.

### Loss & Training

JAIL-CON requires no training and operates as an **optimization-based attack**:

The iterative optimization objective is:
$$\max_{T_b} P(harmful\_response | T_{interleaved}(T_h, T_b))$$

where $T_b$ denotes the choice of benign task, and the objective is to identify the benign task that best conceals the harmful task.

The iterative process considers:
- Semantic relevance between the benign and harmful tasks.
- Variations in interleaving patterns (e.g., interleaving interval, group size).
- The quality of the LLM's responses to concurrent tasks.

## Key Experimental Results

### Main Results

Attack success rates (ASR) across multiple mainstream LLMs:

| Method | GPT-4 ASR (%) | Claude-3 ASR (%) | Llama-3 ASR (%) | Avg. ASR (%) |
|--------|--------------|-----------------|----------------|-------------|
| GCG | 31.2 | 18.7 | 42.5 | 30.8 |
| AutoDAN | 47.8 | 29.3 | 56.1 | 44.4 |
| PAIR | 52.6 | 35.4 | 61.3 | 49.8 |
| DeepInception | 55.3 | 38.7 | 58.9 | 51.0 |
| JAIL-CON (Ours) | **68.9** | **52.1** | **73.4** | **64.8** |

### Guardrail Evasion Experiment

Detection rates of generated content from each attack method after applying guardrail defenses:

| Method | Llama Guard Detection (%) | Keyword Detection (%) | Semantic Detection (%) | Overall Evasion (%) |
|--------|--------------------------|----------------------|----------------------|-------------------|
| GCG | 78.5 | 85.2 | 72.1 | 21.5 |
| AutoDAN | 65.3 | 71.8 | 68.4 | 34.7 |
| PAIR | 61.2 | 68.5 | 65.7 | 38.8 |
| DeepInception | 58.9 | 64.3 | 62.8 | 41.1 |
| JAIL-CON (Sequential Output) | 55.1 | 62.7 | 58.3 | 44.9 |
| JAIL-CON (Concurrent Output) | **32.4** | **41.6** | **38.5** | **67.6** |

### Ablation Study

| Variable | ASR (%) | Concurrent Task Quality |
|----------|---------|------------------------|
| No benign task concealment | 35.2 | — |
| Fixed benign task | 58.7 | Medium |
| Adaptive benign task selection | 68.9 | High |
| Interleaving interval: 1 word | 68.9 | High |
| Interleaving interval: 3 words | 62.3 | Medium |
| Interleaving interval: sentence-level | 48.5 | Low |

### Key Findings

1. **JAIL-CON substantially outperforms existing attacks**: It achieves an average ASR of 64.8% across all tested LLMs, surpassing the best baseline by approximately 14 percentage points.
2. **Concurrent outputs are highly covert**: The overall guardrail evasion rate of concurrent outputs reaches 67.6%, far exceeding the 44.9% of sequential outputs.
3. **Word-level interleaving is most effective**: Finer interleaving granularity yields stronger attack performance (word-level > phrase-level > sentence-level).
4. **LLMs genuinely possess concurrent processing capability**: On benign benchmarks, LLMs maintain high accuracy when handling concurrently interleaved mathematical problems.
5. **Adaptive benign task selection is critical**: An appropriately chosen benign task significantly improves the pass rate of the harmful task.

## Highlights & Insights

1. **A novel attack perspective**: This work is the first to introduce the concept of "task concurrency" into LLM jailbreak attacks, opening an entirely new attack surface.
2. **Dual threat**: JAIL-CON achieves not only a high attack success rate but also produces outputs that are significantly harder to detect—posing greater challenges for defenders.
3. **Exposing a fundamental security vulnerability**: LLMs' concurrent processing capability itself constitutes a security weakness.
4. **Comprehensive experimental design**: Evaluations span multiple dimensions, including attack success rate, output covertness, and verification of concurrent capability.

## Limitations & Future Work

1. **Absence of defensive countermeasures**: The paper focuses primarily on the attack surface and does not propose effective defense strategies.
2. **Practical usability of concurrent outputs**: Attackers require an additional decoding step to extract harmful content from concurrent outputs.
3. **Model updates may weaken the attack**: LLM providers can mitigate the attack through subsequent safety training to improve detection of concurrent requests.
4. **Ethical concerns**: Although this is security research, publicly disclosing detailed attack methods risks malicious exploitation.
5. **Scalability**: For very long harmful requests, word-level interleaving doubles the input length, potentially exceeding token limits.

## Related Work & Insights

- **GCG (Zou et al.)**: Universal jailbreak suffixes via gradient-based optimization.
- **AutoDAN (Liu et al.)**: Automated jailbreak prompt generation.
- **PAIR (Chao et al.)**: Jailbreaking through conversational interaction.
- **Guardrail Defenses**: LlamaGuard, NeMo Guardrails, etc.
- **Insights**: Task concurrency is a direction worthy of in-depth investigation by the security community; defenders need to develop detection methods tailored to non-sequential outputs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Entirely new attack perspective; the concept of task concurrency is highly original.)
- Technical Depth: ⭐⭐⭐⭐ (Well-designed methodology with a sound iterative optimization framework.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-model, multi-dimensional evaluation including covertness analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Clever title; clear and well-structured exposition.)
- Overall: ⭐⭐⭐⭐☆ (High-quality security research that exposes an important vulnerability.)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[NeurIPS 2025\] GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs](gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)

<!-- RELATED:END -->
