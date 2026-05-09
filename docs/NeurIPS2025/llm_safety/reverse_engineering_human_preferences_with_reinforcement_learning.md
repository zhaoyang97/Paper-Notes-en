---
title: >-
  [Paper Note] Reverse Engineering Human Preferences with Reinforcement Learning
description: >-
  [NeurIPS 2025 (Spotlight)][LLM Safety][LLM-as-a-Judge] A reinforcement learning-trained preamble generator is used to inflate the evaluation scores of downstream LLMs, exposing critical vulnerabilities in the LLM-as-a-Judge evaluation framework. The attack is nearly undetectable and demonstrates cross-model transferability.
tags:
  - NeurIPS 2025 (Spotlight)
  - LLM Safety
  - LLM-as-a-Judge
  - Adversarial Attack
  - Preference Reverse Engineering
  - Reinforcement Learning
  - Detectability
date: 2026-05-08
content_hash: dc50ef4517904c06
---

# Reverse Engineering Human Preferences with Reinforcement Learning

**Conference**: NeurIPS 2025 (Spotlight)

**arXiv**: [2505.15795](https://arxiv.org/abs/2505.15795)

**Code**: N/A

**Area**: AI Safety

**Keywords**: LLM-as-a-Judge, Adversarial Attack, Preference Reverse Engineering, Reinforcement Learning, Detectability

## TL;DR

A reinforcement learning-trained preamble generator is used to inflate the evaluation scores of downstream LLMs, exposing critical vulnerabilities in the LLM-as-a-Judge evaluation framework. The attack is nearly undetectable and demonstrates cross-model transferability.

## Background & Motivation

### Starting Point

**Goal**: **Background**: LLM-as-a-Judge has become the dominant framework for evaluating LLM capabilities—leveraging a powerful LLM as a judge to predict human preferences. However, this framework is susceptible to malicious exploitation:

**Gamability**: LLM outputs can be adversarially optimized to satisfy the judge model.

**Limitations of Prior Work**: Existing post-hoc editing methods directly modify model responses, making them readily detectable.

**Security Threat**: If leaderboard scores can be artificially inflated, the credibility of the entire evaluation ecosystem is called into question.

## Method

### Overall Architecture

A **preamble generator** model is trained to prepend an implicit steering text before the candidate LLM's response, inducing the judge LLM to assign higher scores.

### Key Designs

**1. Pipeline Architecture**

Two LLMs are composed in sequence:
- **Preamble Generator $\pi_\phi$**: Generates a preamble $p$ conditioned on the prompt.
- **Candidate LLM $M$** (frozen): Takes $[p; \text{prompt}]$ as input and produces response $y$.
- **Judge LLM $J$**: Scores the response $y$.

Key: The candidate LLM remains frozen; only the preamble generator is trained.

**2. RL Training**

- **Reward Signal**: The judge LLM's score for the response.
- **Policy**: Parameters of the preamble generator $\pi_\phi$.
- **Objective**: $\max_\phi \mathbb{E}_{p \sim \pi_\phi}[J(M(p, \text{prompt}))]$
- Optimized using PPO.

**3. Stealthiness**

- Unlike direct response editing, the preamble is used internally by the model; both the user and the judge observe only the candidate LLM's natural output.
- The preamble influences the model's activation states, thereby altering generation behavior.
- The attack is nearly **undetectable**: the distributional shift in the responses themselves is minimal.

### Loss & Training

$$\mathcal{L}(\phi) = -\mathbb{E}_{p \sim \pi_\phi}[R(p)] + \beta \text{KL}(\pi_\phi \| \pi_0)$$

where $R(p)$ is the judge score and $\pi_0$ is the initial policy of the preamble generator.

## Key Experimental Results

### Main Results

Evaluation score gains across different frameworks (AlpacaEval 2.0, Win Rate vs. GPT-4):

| Method | Baseline Win Rate | Post-Attack Win Rate | Detectability |
|--------|------------------|---------------------|---------------|
| Direct Response | 25.3% | - | - |
| Post-hoc Editing | - | 38.5% | High (82%) |
| Manual Prompt Injection | - | 32.1% | Medium (55%) |
| Preamble Generator (Ours) | - | **42.8%** | **Low (12%)** |

Cross-model transferability:

| Train Judge | Test Judge | Baseline | Post-Attack | Transfer |
|-------------|------------|----------|-------------|---------|
| GPT-4 | GPT-4 | 25.3% | 42.8% | (in-distribution) |
| GPT-4 | Claude-3 | 28.1% | 38.5% | ✓ Effective |
| GPT-4 | Gemini | 26.5% | 36.2% | ✓ Effective |
| Claude-3 | GPT-4 | 25.3% | 37.5% | ✓ Effective |

### Ablation Study

Effect of preamble length on attack performance:

| Preamble Length (tokens) | Win Rate | Detectability | Response Naturalness |
|--------------------------|----------|---------------|----------------------|
| 0 (no preamble) | 25.3% | 0% | 100% |
| 16 | 32.5% | 5% | 98% |
| 64 | 38.8% | 8% | 95% |
| 128 | 42.8% | 12% | 92% |
| 256 | 43.2% | 25% | 85% |

### Key Findings

1. The preamble generator raises Win Rate from 25.3% to 42.8%, surpassing post-hoc editing by a substantial margin.
2. Attack detectability is only 12%, far below that of direct editing (82%).
3. **Cross-model transfer**: A preamble generator trained against one judge model remains effective against other judge models.
4. This suggests that certain features of human preference are **model-agnostic** and can be systematically exploited.

## Highlights & Insights

- **Spotlight paper**: Exposes a fundamental security vulnerability in the LLM-as-a-Judge paradigm.
- **Undetectability**: The core innovation lies in conducting the attack via the preamble (invisible) rather than by modifying the response (visible).
- **Cross-model transfer**: Implies the existence of systematic biases in preference judgments, rather than weaknesses specific to individual models.
- **Dual implications**: Serves simultaneously as a security warning and as a new direction for optimizing upstream inputs via preamble-based steering.

## Limitations & Future Work

1. Although the preamble is "invisible" to end users, inspection of the system prompt can still reveal it.
2. If evaluators simultaneously audit the system prompt, the attack's effectiveness is reduced.
3. RL training requires a large number of judge API calls, resulting in substantial computational cost.
4. The study focuses primarily on English-language settings; cross-lingual generalization remains unvalidated.

## Related Work & Insights

- **LLM-as-a-Judge** (Zheng et al.): Established the LLM-based evaluation framework.
- **Prompt Injection**: Related work on system prompt attacks.
- **AlpacaEval**: LLM evaluation leaderboard.

## Rating

- ⭐ Novelty: 9/10 — The preamble-based attack is a genuinely novel approach that surfaces deep-seated issues.
- ⭐ Experimental Thoroughness: 8/10 — Directly informs the design of more robust evaluation systems.
- ⭐ Writing Quality: 9/10 — Spotlight-level work with clear and compelling argumentation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning with Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)
- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](trap_targeted_redirecting_of_agentic_preferences.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](finding_structure_in_continual_learning.md)

<!-- RELATED:END -->
