---
title: >-
  [Paper Note] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention
description: >-
  [ICLR2026][LLM Reasoning][LRM safety alignment] This paper proposes Intervened Preference Optimization (IPO), which constructs preference pairs for training by replacing compliance cues with safety triggers at critical steps during the reasoning process, significantly improving the safety of the chain-of-thought (CoT) reasoning process itself in large reasoning models (LRMs).
tags:
  - ICLR2026
  - LLM Reasoning
  - LRM safety alignment
  - reasoning process safety
  - process supervision
  - preference optimization
  - jailbreak defense
date: 2026-05-08
content_hash: ce0a836cabf166da
---

# Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention

**Conference**: ICLR2026
**arXiv**: [2509.24393](https://arxiv.org/abs/2509.24393)
**Code**: Not released
**Area**: LLM Reasoning
**Keywords**: LRM safety alignment, reasoning process safety, process supervision, preference optimization, jailbreak defense

## TL;DR

This paper proposes Intervened Preference Optimization (IPO), which constructs preference pairs for training by replacing compliance cues with safety triggers at critical steps during the reasoning process, significantly improving the safety of the chain-of-thought (CoT) reasoning process itself in large reasoning models (LRMs).

## Background & Motivation

- LRMs such as DeepSeek-R1 excel at complex problem solving, yet their CoT reasoning processes frequently contain harmful content.
- Even when the final response appears safe, harmful content within the reasoning trace can still be exploited by malicious users.
- Existing safety alignment methods (e.g., RealSafe, STAR) focus primarily on the safety of final responses, neglecting the safety of the reasoning process itself.
- Experiments demonstrate that safe reasoning almost invariably leads to safe responses, but the converse does not hold — cases where reasoning is unsafe yet the final response appears safe are common.
- Directly applying RL methods such as GRPO to reward reasoning safety yields limited results, as rollout diversity is low and safe reasoning trajectories are difficult to sample for approximately 50% of harmful prompts.

## Core Problem

How can the reasoning process of LRMs be aligned to be safe? The core challenges are: (1) optimizing reasoning-process safety is harder than optimizing response safety; and (2) RL methods suffer from insufficient rollout diversity, resulting in weak training signals.

## Method

### Three Key Findings

**Finding 1: Safety Triggers**
- A Continuation Safety Ratio (CSR) is defined: for the $i$-th token in a reasoning trajectory $z_s$, the subsequent safety probability is estimated via 32 samples:
$$S_i(x, z_s) = \mathbb{E}_{z_c \sim \pi_\theta(\cdot|x, z_s^{\leq i})}[\mathbb{I}(z_s^{\leq i} \| z_c \text{ is safe})]$$
- Over 90% of safe trajectories exhibit a turning point where the CSR sharply rises to 100%, corresponding to sentences where the model explicitly recognizes risk, reformulates the task, or invokes safety criteria.
- These sentences are referred to as "safety triggers" and constitute critical steps for reasoning safety.

**Finding 2: Compliance Cues**
- In unsafe trajectories, turning points where the CSR sharply drops are highly correlated with the first compliance cue — a sentence expressing a tendency to comply with a malicious request.
- The Pearson correlation between the token position of compliance cues and the CSR turning point reaches **0.85**.

**Finding 3: Effectiveness of Corrective Intervention**
- Replacing the first compliance cue in an unsafe trajectory with a safety trigger substantially reduces the proportion of harmful content in the subsequent generation.
- Interventions can be applied iteratively, yielding stronger cumulative effects.

### IPO Pipeline

1. **Compliance cue detection**: GPT-4o is used to automatically detect the position $h$ of the first compliance cue in a reasoning trajectory (agreement with human annotation exceeds 80%).
2. **Replacement and continuation**: The compliance cue is replaced with a trigger $\tau$ sampled from a safety trigger pool $\mathcal{T}$, and the model continues generation: $\tilde{z}^{\geq h} \sim \pi_\theta(\cdot|x, z^{<h}, \tau)$.
3. **Preference pair construction**: The corrected safe trajectory $\tilde{z}$ and the original unsafe trajectory $z$ form a preference pair $(x, \tilde{z} \succ z, h)$.
4. **Preference learning**: DPO training is applied to the portion after the divergence point:

$$\mathcal{L} = -\mathbb{E}_{(x, \tilde{z} \succ z, h) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(\tilde{z}^{\geq h}|x, z^{<h})}{\pi_{\theta_{\text{ref}}}(\tilde{z}^{\geq h}|x, z^{<h})} - \beta \log \frac{\pi_\theta(z^{\geq h}|x, z^{<h})}{\pi_{\theta_{\text{ref}}}(z^{\geq h}|x, z^{<h})}\right)\right]$$

### Connection to Reward Shaping

- The CSR is essentially the value function for the safety label, $V^\pi(s_t) = \Pr[S(x,z)=1|s_t]$.
- IPO is equivalent to injecting intermediate reward signals at safety-critical steps, analogous to potential-based reward shaping.
- This approach is more efficient than the sparse terminal rewards used in GRPO.

## Key Experimental Results

### Reasoning Safety (Reasoning Harmful Ratio ↓)

| Method | JBB | StrongReject | WildJailbreak | Avg. |
|--------|-----|-------------|---------------|------|
| DS-8B Base | 69.0% | 63.2% | 82.4% | 71.5% |
| SafeChain | 56.1% | 55.3% | 66.7% | 59.4% |
| RealSafe | 20.7% | 34.7% | 47.1% | 34.2% |
| STAR | 8.0% | 21.9% | 37.8% | 22.6% |
| GRPO | 0.3% | 19.0% | 36.3% | 18.5% |
| **IPO (Ours)** | **5.7%** | **16.7%** | **23.4%** | **15.3%** |

### Reasoning Capability Retention

| Method | AIME | MATH-500 | GPQA | HumanEval | Avg. |
|--------|------|----------|------|-----------|------|
| DS-8B Base | 50.7% | 91.8% | 44.9% | 79.5% | 66.7% |
| STAR | 46.0% | 89.4% | 47.0% | 77.1% | 64.9% |
| GRPO | 50.0% | 92.8% | 50.5% | 79.9% | 68.3% |
| **IPO (Ours)** | **54.0%** | **91.6%** | **49.0%** | **79.5%** | **68.5%** |

- On DS-8B, IPO reduces the WildJailbreak reasoning harmful ratio from 82.4% to 23.4% (a 71.6% reduction).
- Average reasoning capability improves by 1.8%, with a 3.3% gain on AIME.
- Consistent effectiveness is observed on DS-7B and Qwen3-8B.

## Highlights & Insights

1. **Unique perspective**: This is the first work to systematically advance safety alignment from "response safety" to "reasoning process safety."
2. **In-depth empirical analysis**: CSR curve analysis reveals the critical roles of safety triggers and compliance cues; a Pearson correlation of 0.85 provides strong quantitative evidence.
3. **Simple yet effective method**: IPO achieves substantial safety improvements without complex RL training, relying solely on "replacement + DPO."
4. **Theoretical connection**: Linking IPO to reward shaping provides a theoretical explanation for why IPO is more efficient than GRPO.
5. **Safety–capability synergy**: IPO improves safety while maintaining or enhancing reasoning capability, breaking the conventional safety–utility trade-off.

## Limitations & Future Work

- The safety trigger pool relies on GPT-4o for detection; more automated approaches could be explored in future work.
- The XsTest compliance rate declines (DS-8B: 98.4% → 80.0%), indicating a degree of over-refusal.
- Validation is limited to 8B-scale models; effectiveness on larger models remains to be investigated.
- Safety evaluation depends on a GPT-4o automatic evaluator, which may introduce bias.
- The intervention strategy is relatively simple (replacing only the first compliance cue); more refined multi-step intervention strategies could further improve performance.

## Related Work & Insights

| Method | Alignment Target | Training | Reasoning Safety | Response Safety | Capability Retention |
|--------|-----------------|----------|-----------------|-----------------|----------------------|
| RealSafe | Response | SFT (distillation) | Moderate | Strong | Good |
| STAR | Response | SFT (distillation) | Good | Good | Good |
| GRPO | Reasoning + Response | RL | Good | Good | Strong |
| **IPO** | **Reasoning process** | **DPO (intervention)** | **Best** | **Strong** | **Strong** |

- The CSR analysis methodology provides a new tool for understanding safety behavior in reasoning models and can be generalized to other process supervision scenarios.
- The "critical step replacement + preference learning" framework has potential applications in improving reasoning quality (e.g., correcting critical steps in mathematical reasoning).
- The findings offer important guidance for the safety of LRM-based agents.

## Rating

- Novelty: ⭐⭐⭐⭐ — Approaching safety from the perspective of reasoning process safety, the corrective intervention idea in IPO is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three models, three safety benchmarks, and four reasoning benchmarks provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ — Analysis is deep, logic is clear, and figures are well-crafted.
- Value: ⭐⭐⭐⭐ — Reasoning safety is a core concern for LRM deployment; the method is practical and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_solution_complexity.md)
- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
