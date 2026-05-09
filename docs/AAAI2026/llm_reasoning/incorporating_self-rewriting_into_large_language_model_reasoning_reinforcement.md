---
title: >-
  [Paper Note] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement
description: >-
  [AAAI 2026][LLM Reasoning][Reasoning quality improvement] This paper proposes a Self-Rewriting framework that enables large reasoning models (LRMs) to rewrite their own reasoning traces for "easy" samples (queries where all responses are correct) during RL training and learn from the rewritten versions. With only ~10% additional training overhead, the approach reduces reasoning length by 46% while maintaining accuracy, improves internal reasoning quality (LLM-as-Judge) by 7.2 points, and effectively mitigates issues such as over-thinking and redundant thinking.
tags:
  - AAAI 2026
  - LLM Reasoning
  - Reasoning quality improvement
  - self-rewriting
  - GRPO
  - reasoning length control
  - over-thinking
date: 2026-05-08
content_hash: e5dde0b43cbec291
---

# Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement

**Conference**: AAAI 2026
**arXiv**: [2511.16331](https://arxiv.org/abs/2511.16331)
**Code**: N/A
**Area**: LLM Reasoning
**Keywords**: Reasoning quality improvement, self-rewriting, GRPO, reasoning length control, over-thinking

## TL;DR
This paper proposes a Self-Rewriting framework that enables large reasoning models (LRMs) to rewrite their own reasoning traces for "easy" samples (queries where all responses are correct) during RL training and learn from the rewritten versions. With only ~10% additional training overhead, the approach reduces reasoning length by 46% while maintaining accuracy, improves internal reasoning quality (LLM-as-Judge) by 7.2 points, and effectively mitigates issues such as over-thinking and redundant thinking.

## Background & Motivation

**Background**: RL-based large reasoning models (e.g., DeepSeek-R1, o1) have achieved strong performance on complex reasoning tasks by scaling test-time computation, but training relies solely on outcome correctness rewards, lacking supervision over the quality of internal reasoning processes.

**Limitations of Prior Work**:
   - **Over-thinking**: The model reasons extensively over irrelevant or trivial parts.
   - **Under-thinking**: The model skips or over-simplifies complex and relevant parts.
   - **Redundant thinking**: The model repeats essentially identical ideas.
   - **Disordered thinking**: Multiple reasoning threads are interleaved incoherently.
   - Existing length-control methods focus on being "short" rather than being "good."

**Key Challenge**: Outcome correctness rewards are coarse-grained—any response that reaches the correct final answer receives a positive reward, regardless of how verbose or disorganized the reasoning process is.

**Goal**: To introduce a fine-grained reasoning quality improvement mechanism while preserving the standard RL training framework (GRPO).

**Key Insight**: Drawing on the idea of LLM self-rewarding, but adopting a generative approach (self-rewriting) rather than a discriminative one (self-scoring)—the model rewrites its own reasoning traces and then learns from the rewritten versions.

**Core Idea**: Enable LRMs to rewrite their own reasoning and learn from the rewritten outputs, rather than learning solely from correct/incorrect feedback signals.

## Method

### Overall Architecture
A selective rewriting step is embedded within the GRPO training loop: for each query, $G/2$ responses are first sampled → if all are correct ("easy" query), the reasoning traces are rewritten and answers are generated from the rewritten versions → rewritten samples receive higher rewards → if not all correct ("hard" query), the remaining $G/2$ responses are sampled normally → optimization proceeds via standard GRPO.

### Key Designs

1. **Selective Rewriting**:

    - Function: Rewriting is applied only to "easy" queries (i.e., all $G/2$ responses are correct), which the model has already fully mastered.
    - Mechanism: When $\mathbf{r}_{1:G/2} = \mathbf{1}$, the reasoning traces are rewritten as $\mathbf{t}_{G/2+1:G} \leftarrow \mathcal{M}(\mathbf{t}_{1:G/2})$; the original query is then concatenated with the rewritten reasoning to generate the final answer. Otherwise, responses are sampled normally.
    - Design Motivation: (1) Minimizing disruption to GRPO—fully correct samples yield near-zero advantage in standard GRPO and provide no gradient signal; rewriting converts them into informative training samples. (2) Practicing concise expression on easy problems while preserving normal exploration on hard ones.

2. **Rewriting Reward Design**:

    - Function: Assigns higher rewards to samples that remain correct after rewriting.
    - Mechanism: $\hat{r}_i = 1$ (rewritten and correct), $\hat{r}_i = 0$ (original samples from fully-correct queries that were not rewritten), $\hat{r}_i = r_i$ (non-fully-correct queries retain their original rewards).
    - Design Motivation: Establishes a preference signal favoring rewritten reasoning over original reasoning, incentivizing the model to produce more refined traces.

3. **Efficient Implementation**:

    - Function: Compiles rewriting and normal sampling into the same batch to minimize additional overhead.
    - Mechanism: After the initial $G/2$ sampling, queries requiring normal sampling and those requiring rewriting are batched together for inference. Continued generation after rewriting only produces the short output following `</think>`. The total overhead is approximately 10%.
    - Design Motivation: Running rewriting as a separate inference pass would incur substantial overhead. Batch compilation preserves the scalability of the RL training pipeline.

### Loss & Training
- PPO-clip objective based on GRPO, with advantages computed from rewriting rewards.
- The rewriting prompt is minimal—only requesting "rewrite to improve quality," with no explicit instruction to shorten the output.
- Compared against six baselines: GRPO, LenPen1/2, ShortBetter, LPO, and TOPS.

## Key Experimental Results

### Main Results

Performance of Qwen3-4B on four reasoning benchmarks (Acc / reasoning length in tokens / LLM-Judge score):

| Method | Avg Acc | Avg Len | Avg Judge |
|--------|---------|---------|-----------|
| Original | 76.1 | 4025 | 67.1 |
| GRPO | 75.8 (−0.3) | 3681 (−9%) | 68.6 (+1.5) |
| ShortBetter | — | −26% | +4.3 |
| TOPS | — | −35% | +3.3 |
| **Rewrite** | 76.2 (+0.1) | 2835 (**−35%**) | **73.5 (+6.4)** |

A similar trend is observed for Qwen3-1.7B: Self-Rewrite achieves Acc −0.7, Len −35%, Judge +8.1.

### Ablation Study

| Configuration | Acc Δ | Len Δ | Judge Δ | Notes |
|---------------|-------|-------|---------|-------|
| Self-Rewrite (full) | +0.6 | −46% | +7.2 | Full method |
| GRPO only | +0.0 | −11% | +1.0 | RL baseline |
| LenPen1 (length penalty) | −1.6 | −29% | +4.8 | Explicit length penalization |
| TOPS (truncation) | −2.3 | −35% | +3.3 | Truncates long reasoning |

### Key Findings
- **No explicit length control is required**: The rewriting prompt contains no instruction to shorten output, yet the model naturally learns to express the same reasoning in fewer tokens—achieving a 35–46% length reduction.
- **Comprehensive improvement in reasoning quality**: LLM-Judge scores improve significantly across all four internal reasoning defects (over/under/redundant/disordered thinking).
- **Accuracy is maintained or slightly improved**: Unlike explicit length-penalization methods (LenPen, TOPS) that trade off accuracy, Self-Rewrite preserves or marginally improves accuracy.
- **Practice on easy samples generalizes to hard samples**: Rewriting is applied only to fully-correct samples, yet the learned concise reasoning habits generalize across all queries.

## Highlights & Insights
- **Generative vs. discriminative self-improvement**: Self-rewarding has the model assign scores (discriminative); self-rewriting has the model rewrite (generative). The generative approach provides richer improvement signals.
- **Repurposing "wasted" samples in GRPO**: Fully-correct queries yield near-zero advantage in GRPO and are effectively discarded. Self-Rewriting converts them into valuable training signal.
- **Rewriting as implicit distillation**: The rewriting process essentially compresses verbose "thinking experiments" into refined reasoning chains.

## Limitations & Future Work
- **Overly simple rewriting prompt**: The generic instruction "improve quality" may be insufficient; prompts targeting specific reasoning defects could be more effective.
- **Rewriting is only applied to easy samples**: Hard samples arguably have greater need for reasoning quality improvement, but the current strategy avoids them.
- **Reliability of LLM-as-Judge**: Reasoning quality evaluation relies on LLM scoring, and the reliability of this evaluation protocol is not thoroughly discussed.
- **Only smaller models are evaluated**: Experiments are conducted on 1.7B and 4B models; effectiveness at larger scales remains to be confirmed.

## Related Work & Insights
- **vs. L1/ShortBetter and similar length-control methods**: Explicitly penalizing length sacrifices accuracy; Self-Rewrite achieves length reduction indirectly through quality improvement, offering a more principled solution.
- **vs. Self-Rewarding (Yuan et al.)**: Self-Rewarding has the model self-score and trains with DPO; Self-Rewrite has the model self-rewrite and trains with GRPO, more directly improving the reasoning process.
- **vs. TOPS (truncation-based)**: TOPS truncates long reasoning traces and trains on the shortened versions; Self-Rewrite allows the model to autonomously determine how to reformulate its reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of embedding self-rewriting into RL training is novel, and the selective rewriting design that repurposes discarded samples is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, multiple baselines, multiple model sizes, and fine-grained analysis are provided, though the model scales evaluated remain relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures and tables are clear; the taxonomy of four reasoning defects with illustrative examples is intuitive; the algorithmic pseudocode is well-presented.
- Value: ⭐⭐⭐⭐ A practical improvement to LRM reasoning efficiency and quality; the ~10% overhead is highly attractive from an engineering perspective.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](serl_self-examining_reinforcement_learning_on_open-domain.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](../../NeurIPS2025/llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](../../NeurIPS2025/llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](../../ACL2026/llm_reasoning/dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)

<!-- RELATED:END -->
