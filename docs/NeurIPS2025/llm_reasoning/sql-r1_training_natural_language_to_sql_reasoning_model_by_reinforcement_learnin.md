---
title: >-
  [Paper Note] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning
description: >-
  [NeurIPS 2025][LLM Reasoning][NL2SQL] This work presents the first systematic application of GRPO-based reinforcement learning to NL2SQL tasks. Through a four-level progressive reward function and a training strategy com…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "NL2SQL"
  - "Reinforcement Learning"
  - "GRPO"
  - "Cold Start"
  - "Synthetic Data"
  - "Reasoning Capability"
date: 2026-05-08
content_hash: a16764603fb4c94c
---

# SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2504.08600](https://arxiv.org/abs/2504.08600)  
**Code**: [https://github.com/IDEA-FinAI/SQL-R1](https://github.com/IDEA-FinAI/SQL-R1)  
**Area**: LLM Reasoning
**Keywords**: NL2SQL, Reinforcement Learning, GRPO, Cold Start, Synthetic Data, Reasoning Capability

## TL;DR
This work presents the first systematic application of GRPO-based reinforcement learning to NL2SQL tasks. Through a four-level progressive reward function and a training strategy combining 200K cold-start data with 5K complex-sample RL fine-tuning, the 7B model achieves 88.7% on Spider and 66.6% on BIRD, surpassing GPT-4-based methods at comparable scale.

## Background & Motivation

**Background**: NL2SQL enables non-expert users to query databases directly in natural language. Current approaches primarily rely on SFT, while closed-source models (e.g., GPT-4) are costly and opaque.

**Limitations of Prior Work**: SFT methods underperform in complex database scenarios (multi-table joins, nested queries), and fixed generation strategies struggle with ambiguous semantics, resulting in poor generalization.

**Key Challenge**: The "data scale bottleneck" of SFT — large volumes of high-quality annotated data are required, yet annotation is expensive, and simply scaling data does not necessarily improve reasoning on complex queries.

**Goal**: ① Can RL replace SFT for NL2SQL? ② How should NL2SQL-specific rewards be designed? ③ What is the minimum data requirement?

**Key Insight**: Inspired by the R1 series, this paper transfers the GRPO algorithm to SQL generation and designs reward signals incorporating execution verification.

**Core Idea**: Cold-start SFT to activate basic capabilities + GRPO training on 5K complex samples + four-level progressive rewards = high performance with minimal data.

## Method

### Overall Architecture
Two-stage training: ① Cold start — SFT on 200K synthetic data to establish basic NL2SQL capability; ② RL stage — GRPO optimization on 5K complex samples to refine reasoning strategies. At inference time, 8 SQL candidates are generated and the best is selected via self-consistency voting.

### Key Designs

1. **Four-Level Progressive Reward Function**:

    - Function: Format → Execution → Result → Length, evaluating SQL quality layer by layer.
    - Mechanism: Format reward $S_f$ (±1) checks think/answer tags; execution reward $S_e$ (±2) verifies SQL executability; result reward $S_r$ (±3) judges query result correctness (highest priority); length reward $S_l$ encourages detailed reasoning traces.
    - Design Motivation: Strict priority ordering ensures the model does not sacrifice correctness for length, while the progressive learning signal guides the model from format → syntax → semantics incrementally.

2. **GRPO Reinforcement Learning Algorithm**:

    - Function: Generates $G=8$ SQL candidates per query and computes advantages based on within-group relative performance.
    - Mechanism: Requires no value model; policy updates are controlled via importance sampling ratio $r_i^{ratio}$, clipping, and KL divergence regularization. Lighter-weight than PPO.
    - Design Motivation: The discrete rewards of NL2SQL (execution success/failure) are naturally suited to GRPO's within-group comparison mechanism.

3. **Cold Start + 5K Complex Data Strategy**:

    - Function: 200K synthetic SFT data activates basic capability; 5K complex samples are used for RL training.
    - Mechanism: Cold start uses CoT-annotated synthetic data (SynSQL-200K); the RL stage intentionally withholds CoT, forcing the model to autonomously explore reasoning strategies.
    - Design Motivation: Demonstrates the data efficiency of RL — only 5K complex samples are needed to significantly outperform SFT trained on the full 200K dataset.

### Loss & Training
Standard SFT loss is used for cold start; the GRPO objective is used for the RL stage. The base model is Qwen2.5-Coder-7B; 8-way self-consistency voting is applied at inference.

## Key Experimental Results

### Main Results (Spider & BIRD)

| Model | Params | Spider-Test | BIRD-Dev |
|-------|--------|-------------|----------|
| CodeS (SFT) | 15B | 85.4% | 59.4% |
| GPT-4 (DAIL-SQL) | — | 86.6% | 57.4% |
| OmniSQL (SFT) | 7B | 88.9% | 66.1% |
| **SQL-R1** | **7B** | **88.7%** | **66.6%** |
| SQL-R1 | 14B | 88.1% | **67.1%** |

The 7B model surpasses most GPT-4-based approaches and matches or exceeds the strongest SFT baselines.

### Ablation Study (BIRD Stratified by Difficulty)

| Difficulty | SQL-R1 (7B) | SQL-R1 (14B) | GPT-4 |
|------------|-------------|--------------|-------|
| Simple | 72.1% | 72.4% | 66.9% |
| Moderate | 60.8% | 59.7% | 46.5% |
| **Challenging** | 51.0% | **56.5%** (+5.5%) | 43.8% |

### Key Findings
- **RL outperforms SFT**: SQL-R1 (RL) 7B exceeds OmniSQL (SFT) 7B on BIRD using the same base model, demonstrating the advantage of RL.
- **Remarkable efficiency of 5K data**: RL training on only 5K complex samples achieves SOTA, substantially reducing data requirements.
- **Targeted gains from scaling**: Scaling from 7B to 14B primarily improves performance on challenging queries (+5.5%), with minimal gains on simple/moderate ones, suggesting that more parameters help handle long-range dependencies.
- **Cost-effectiveness**: The 7B open-source model reduces cost by over 70% compared to GPT-4 without sacrificing performance.
- **Interpretable reasoning**: The `<think>` output provides transparency suitable for compliance requirements in high-stakes domains.

## Highlights & Insights
- **First RL-based NL2SQL work**: Systematically validates the effectiveness of GRPO for structured output generation, opening a new research direction.
- **Elegant four-level reward design**: The format → execution → result → length progression aligns perfectly with NL2SQL task requirements — a SQL query must be executable before correctness can even be assessed.
- **Critical role of cold start**: 200K SFT data provides a stable foundation, after which RL requires only 5K carefully selected samples, demonstrating the efficiency of the two-stage SFT+RL paradigm.
- **Transferability**: The RL + reward design framework is generalizable to other structured generation tasks (JSON, XML, regular expressions, etc.).

## Limitations & Future Work
- BIRD 67.1% vs. GPT-4o 73% — a notable gap remains in complex real-world scenarios.
- Synthetic data may not cover edge cases encountered in real DBA workloads.
- Computational cost of RL training (8-candidate generation × execution evaluation) is not disclosed.
- Execution accuracy (EX) cannot handle cases where semantically equivalent SQL queries differ syntactically.
- Ablation of the cold-start strategy is missing (no SFT vs. SFT, data scaling curves).
- Generalization across real cross-domain settings (e.g., finance, healthcare) has not been validated.

## Related Work & Insights
- **vs. OmniSQL**: Using the same base model under an SFT paradigm, SQL-R1 surpasses OmniSQL by 0.5% on BIRD via RL, demonstrating the additional gains from reinforcement learning.
- **vs. SQL-o1**: SQL-o1 improves reasoning through long chain-of-thought, whereas SQL-R1 leverages reward signals — the two approaches are complementary.
- **vs. MCTS-SQL**: Tree search vs. policy learning; SQL-R1 requires no search overhead, making inference more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic application of GRPO to NL2SQL with a novel four-level reward design, though GRPO itself is not original to this work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on Spider, BIRD, derived benchmarks, and difficulty-stratified analysis; cold-start ablation and training cost analysis are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-defined problem formulation; some details (e.g., the self-consistency voting mechanism) are insufficiently described.
- Value: ⭐⭐⭐⭐⭐ An open-source, low-cost, high-performance NL2SQL solution with direct practical value for enterprise deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [\[NeurIPS 2025\] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning](ktae_a_model-free_algorithm_to_key-tokens_advantage_estimation_in_mathematical_r.md)

</div>

<!-- RELATED:END -->
