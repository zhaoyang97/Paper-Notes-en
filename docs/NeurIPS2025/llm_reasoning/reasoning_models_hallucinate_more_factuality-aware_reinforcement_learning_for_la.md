---
title: >-
  [Paper Note] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models
description: >-
  [NeurIPS 2025][LLM Reasoning][Hallucination] This paper reveals that RL-trained reasoning models (e.g., DeepSeek-R1) hallucinate significantly more than non-reasoning models…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Hallucination"
  - "Reasoning Models"
  - "Reinforcement Learning"
  - "Factuality Verification"
  - "GRPO"
  - "Step-level Reward"
date: 2026-05-08
content_hash: 749f4e324bc99cd1
---

# Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.24630](https://arxiv.org/abs/2505.24630)  
**Code**: [GitHub](https://github.com/nusnlp/FSPO)  
**Area**: LLM Reasoning
**Keywords**: Hallucination, Reasoning Models, Reinforcement Learning, Factuality Verification, GRPO, Step-level Reward

## TL;DR
This paper reveals that RL-trained reasoning models (e.g., DeepSeek-R1) hallucinate significantly more than non-reasoning models, theoretically identifies three root causes (high-variance gradients, entropy constraints, and spurious local optima), and proposes the FSPO algorithm, which adjusts token-level advantages via step-level factuality verification to reduce hallucination while maintaining or even improving reasoning capability.

## Background & Motivation
**Background**: Reasoning models exemplified by DeepSeek-R1 and OpenAI o1 are trained via RL (e.g., GRPO) to produce long chain-of-thought reasoning, achieving breakthrough performance on complex tasks such as mathematics and coding.

**Limitations of Prior Work**: The authors identify a critically overlooked problem — RL-trained reasoning models exhibit substantially higher hallucination rates. Empirically, R1-Distill-Qwen-7B achieves only 6.9% truthfulness on TruthfulQA (vs. 36.7% for Qwen2.5-7B-Instruct) and only 11.6% on HaluEval-QA (vs. 48.0%). Beneath the appearance of "confident reasoning," these models produce pervasive factual errors.

**Key Challenge**: Existing RL training relies solely on binary outcome rewards (0/1) based on final answer correctness, entirely ignoring the factuality of intermediate reasoning steps. This sparse reward signal leads to three theoretical issues: (1) extremely high gradient variance when the probability of a correct answer is low → training instability; (2) the need for high-entropy exploration to find correct answers → increased hallucination probability; (3) the model may converge to a "confident but wrong" spurious local optimum → zero gradient prevents escape.

**Goal**: Design an RL training algorithm that jointly optimizes reasoning capability and factuality, significantly reducing hallucination while improving mathematical reasoning performance.

**Key Insight**: Integrate step-level factuality verification signals (NLI-based) into GRPO's advantage computation, providing much denser gradient signals than pure outcome rewards.

**Core Idea**: An automated factuality verifier scores each reasoning sentence; the token-level advantages for steps that contain hallucinated reasoning despite a correct final answer are flipped, guiding the model to learn "correct reasoning processes" rather than "coincidentally correct answers."

## Method

### Overall Architecture
FSPO augments GRPO with step-level factuality feedback. Given a question $x$ and associated evidence $\mathcal{K}$ (e.g., Wikipedia passages), the model generates an output comprising a reasoning chain $\{z_1, \ldots, z_N\}$ and a final answer $y$. Training employs two reward signals: (1) answer correctness reward $\mathcal{R}_{\text{answer}} \in \{0, 1\}$; and (2) step-level factuality reward $\mathcal{R}_{\text{factuality}}(z_j) \in \{-1, 0, 1\}$ (entailment / neutral / contradiction).

### Key Designs

1. **Step-level Factuality Verifier**:

    - Function: Determines the relationship between each sentence $z_j$ in the reasoning chain and the evidence $\mathcal{K}$.
    - Mechanism: HHEM-2.1 (a natural language inference model) automatically classifies each sentence as entailed by the evidence (+1), neutral (0), or contradictory (−1). Neutral includes connective phrases and exploratory tokens such as "Aha" and "Wait."
    - Design Motivation: Provides a far denser gradient signal than outcome-only rewards, directly addressing the high-variance problem established in Theorem 4.1.

2. **Factuality-Aware Advantage Adjustment**:

    - Function: Flips or retains GRPO-computed token advantages based on sentence-level factuality scores.
    - Mechanism: Let $A_i$ denote the original GRPO advantage. For each token $o_{i,t} \in z_j$: when $A_i > 0$ but $\mathcal{R}_{\text{factuality}}(z_j) = -1$ (correct answer but hallucinated reasoning), the advantage is flipped to $-A_i$; when $A_i < 0$ but $\mathcal{R}_{\text{factuality}}(z_j) = 1$ (incorrect answer but factually correct reasoning step), the advantage is flipped to $-A_i$ to encourage such steps.
    - Design Motivation: Addresses reward hacking — models may arrive at correct answers via erroneous reasoning, and standard GRPO would reinforce these hallucinated tokens. FSPO ensures that only factually correct reasoning steps are reinforced.

3. **Mixed Training Data Strategy**:

    - Function: Combines knowledge-intensive QA data (2K HotpotQA) with mathematical reasoning data (8K SimpleRL).
    - Mechanism: QA data provides factuality training signal; math data preserves reasoning capability. Factuality rewards are computed only for the QA portion; the math portion uses answer reward exclusively.
    - Design Motivation: As few as 2K factuality examples suffice to substantially reduce hallucination without degrading mathematical reasoning.

### Theoretical Analysis (Three Theorems)
- **Theorem 4.1**: Under binary rewards, gradient variance $\propto p(1-p)\|\nabla\log\pi\|^2$; when correctness probability $p$ is small, variance is extremely high → training instability.
- **Theorem 4.2**: To avoid zero-reward collapse, the policy must maintain high-entropy exploration $H_\theta(x) \geq H_{\min}(\epsilon)$ → increased hallucination probability.
- **Theorem 4.3**: A deterministic policy that produces incorrect answers is a stationary point (zero gradient); binary rewards cannot escape this trap.

### Loss & Training
- Built on the verl framework; batch size 8, 8 rollouts per prompt, maximum length 2048.
- Learning rate 4e-7, KL coefficient 1e-3, clip ratio 0.2.
- Trained for 1 epoch on a mixture of HotpotQA (2K) and SimpleRL (8K).

## Key Experimental Results

### Main Results

| Model | GSM8K | MATH500 | TruthfulQA↑ | HaluEval-QA↑ | HalluQA↑ |
|-------|-------|---------|-------------|-------------|----------|
| Qwen2.5-7B-Base | 65.2 | 35.7 | 38.2 | 48.0 | 39.5 |
| R1-Distill-Qwen-7B | 84.3 | 92.8 | **6.9** | **11.6** | **3.1** |
| FSPO (Qwen-Base) | **89.5** | 75.5 | **58.4** | **83.0** | **52.0** |
| Llama3.1-8B-Inst | 77.5 | 33.1 | 26.4 | 36.7 | 12.2 |
| R1-Distill-Llama-8B | 82.1 | 89.1 | 8.8 | 14.6 | 4.6 |
| FSPO (Llama-Inst) | 86.2 | 68.3 | 41.1 | 67.1 | 42.0 |

Key comparison: R1-Distill-Qwen-7B exhibits extremely high hallucination rates (only 6.9% on TruthfulQA). FSPO raises this from 6.9% to 58.4% while achieving a GSM8K score that surpasses the distilled model.

### Ablation Study

| Configuration | MATH-500 | HaluEval-QA↑ | Note |
|--------------|----------|-------------|------|
| GRPO (answer only) | 74.2 | 62.0 | Answer correctness reward only |
| GRPO w/ factuality reward | 74.8 | 72.0 | Factuality reward added without advantage flipping |
| FSPO (full) | **75.5** | **83.0** | Full method with advantage flipping |

### Key Findings
- Reasoning models (R1-Distill series) perform substantially worse than non-reasoning models on all hallucination benchmarks, corroborating the central finding that reasoning models hallucinate more.
- As few as 2K factuality QA examples suffice to significantly reduce hallucination; using 4K/8K is counterproductive and degrades mathematical reasoning performance.
- FSPO is effective with both GRPO and Reinforce++, demonstrating its generality.
- Factuality scores rise steadily during training while response length remains stable, indicating that FSPO improves quality rather than merely increasing verbosity.

## Highlights & Insights
- **Dual theoretical and empirical justification**: Three theorems clearly explain why binary-reward RL induces hallucination — the solution addresses root causes rather than superficially adding regularization.
- **The advantage-flipping mechanism is particularly elegant**: when the final answer is correct but the reasoning contains hallucinated sentences, the token-level advantages for those sentences are negated, directly penalizing "coincidentally correct but factually erroneous" reasoning. This is a minimal yet highly effective modification to GRPO.
- **The 2K-data sufficiency finding** is practically valuable — large-scale factuality annotation is unnecessary.
- The paper reveals a fundamental trade-off in RL-trained reasoning models: reasoning capability ↑ but factuality ↓, serving as an important warning to the broader reasoning LLM community.

## Limitations & Future Work
- Factuality verification depends on external evidence (Wikipedia passages) and does not directly apply to settings lacking a knowledge base (e.g., pure mathematical reasoning).
- The HHEM-2.1 verifier is imperfect and may misclassify factuality — stronger verifiers are needed.
- Experiments are limited to the 7B/8B scale; performance at 32B+ is unknown.
- FSPO achieves 75.5% on MATH-500, far below R1-Distill-Qwen-7B's 92.8%, indicating a non-trivial cost in pure mathematical reasoning.
- The theoretical analysis covers only binary rewards; extensions to more complex reward shaping scenarios warrant further investigation.

## Related Work & Insights
- **vs. DeepSeek-R1**: R1 is trained with pure outcome rewards; FSPO reveals the associated hallucination cost and proposes a step-level remedy.
- **vs. post-hoc methods (e.g., Self-CheckGPT)**: Such approaches detect hallucinations after inference; FSPO penalizes hallucinated reasoning during training, addressing the problem more fundamentally.
- **vs. RLHF**: RLHF uses human feedback but typically at the sequence level; FSPO operates at sentence-level factuality granularity, providing finer-grained supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic theoretical and empirical analysis of hallucination in RL-trained reasoning models; the advantage-flipping design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models and benchmarks with ablations and training dynamics analysis, though large-scale model validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical flow from theory → empirics → method → experiments is clear, and figures are rich and intuitive.
- Value: ⭐⭐⭐⭐⭐ Raises an important hallucination alarm for the entire reasoning LLM community; FSPO is a practical and efficient solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)
- [\[NeurIPS 2025\] DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)

</div>

<!-- RELATED:END -->
