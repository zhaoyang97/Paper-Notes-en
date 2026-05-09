---
title: >-
  [Paper Note] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] This paper proposes RLVRR, a framework that extends RLVR (reinforcement learning with verifiable rewards) from mathematical/code reasoning to open-ended text generation. It extracts hierarchical keyword sequences (content rewards) and executable Python checking functions (style rewards) from high-quality reference answers, forming a "reward chain" to replace single-point verification signals. On 10+ benchmarks, RLVRR trained on 10K examples outperforms 100K SFT and advanced reward models.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - RLVR
  - open-ended generation
  - reward chain
  - verifiable rewards
  - GRPO
date: 2026-05-08
content_hash: 6a505c0c2d3e112d
---

# From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation

**Conference**: ICLR 2026  
**arXiv**: [2601.18533](https://arxiv.org/abs/2601.18533)  
**Code**: [https://github.com/YJiangcm/RLVRR](https://github.com/YJiangcm/RLVRR)  
**Area**: Reinforcement Learning / LLM Alignment  
**Keywords**: RLVR, open-ended generation, reward chain, verifiable rewards, GRPO

## TL;DR

This paper proposes RLVRR, a framework that extends RLVR (reinforcement learning with verifiable rewards) from mathematical/code reasoning to open-ended text generation. It extracts hierarchical keyword sequences (content rewards) and executable Python checking functions (style rewards) from high-quality reference answers, forming a "reward chain" to replace single-point verification signals. On 10+ benchmarks, RLVRR trained on 10K examples outperforms 100K SFT and advanced reward models.

## Background & Motivation

**Background**: RLVR approaches (e.g., DeepSeek-R1, GRPO) have achieved remarkable success in mathematics and code generation by providing reward signals through verification of final answer correctness—a single "verifiable dot." RLHF employs preference reward models to guide alignment in open-ended generation tasks.

**Limitations of Prior Work**: (a) RLVR cannot be directly applied to open-ended generation—open-ended responses have no unique correct answer, making single-point verification inapplicable; (b) reward models in RLHF are prone to reward hacking (overfitting to surface features), require large-scale preference annotation data, and incur high training costs with instability.

**Key Challenge**: Open-ended generation requires simultaneous evaluation across multiple quality dimensions (content completeness, format, style), yet lacks deterministic verification signals analogous to mathematical answers.

**Goal**: Design a method that automatically extracts multi-dimensional verifiable signals from reference answers, enabling the RLVR paradigm to generalize to open-ended generation.

**Key Insight**: Reference answers are treated as a "source of rules"—analogous to how mathematical reasoning derives rules from ground truth, high-quality references are used to extract ordered linguistic signals (reward chains), upgrading single-point supervision to chain-based supervision.

**Core Idea**: Decompose reference answers into keywords (content) and Python verification functions (style), using rule-based rewards along these two verifiable dimensions to replace reward models.

## Method

### Overall Architecture

RLVRR operates in two stages:
- **Data Construction**: Given a question $x$ and reference answer $z$, GPT-4o-mini is used to extract: (a) hierarchical keywords along the content dimension; (b) executable Python checking code along the style dimension.
- **RL Training**: GRPO is used to optimize policy $\pi_\theta$, with reward defined as the average of content reward $r_c$ and style reward $r_s$.
- Total reward: $r_\phi(x,y) = \mathcal{F}(r_c(x,y,z), r_s(x,y,z))$

### Key Designs

1. **Two-Level Hierarchical Keyword Extraction (Content Reward)**:

    - Function: Extracts verifiable keywords capturing core content from reference answers.
    - Mechanism: An LLM first extracts $M$ key points (e.g., "explain risks," "refuse harmful requests"), then extracts specific keywords (<3 words) under each key point. The content reward is computed using LCS (Longest Common Subsequence) to measure alignment between rollout and reference keywords: $r_c = \frac{1}{M}\sum_{m=1}^{M}\frac{\text{len}(\text{LCS}(K_z^m, K_y^m))}{\max(\text{len}(K_z^m), \text{len}(K_y^m))}$
    - Design Motivation: Two-level extraction yields broader and more systematic coverage than direct keyword extraction; LCS preserves keyword order and repetition, providing finer granularity than bag-of-words; keywords constitute only ~15% of the reference, preserving expressive flexibility.

2. **Python Verification Functions (Style Reward)**:

    - Function: Evaluates whether a rollout satisfies the style attributes of the reference answer.
    - Mechanism: An LLM generates $N$ Python CodeEval functions per reference (checking length, markdown formatting, etc.), each assigned a weight $w_n$. Style reward: $r_s = \sum_{n=1}^{N} w_n \cdot \text{CodeEval}_n(y)$
    - Design Motivation: Python-based checking is deterministic, verifiable, and zero-cost—more reliable than reward models.

3. **Multi-Reference Tolerance**:

    - Supports $I=3$ reference answers, taking the highest alignment score per key point.
    - Ablation studies confirm that multiple references yield better consistency than a single reference.

### Loss & Training

- Optimization algorithm: GRPO (Group Relative Policy Optimization)
- KL divergence constraint: $\beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]$
- Training data: Only 10K open-ended instruction-response pairs (filtered from 100K), with data construction via GPT-4o-mini.
- Quality filtering: Samples with combined content and style reward < 0.7 are discarded.

## Key Experimental Results

### Main Results

Comparison on 5 open-ended benchmarks using Qwen2.5-3B-Instruct:

| Method | Data Size | AlpacaEval2 (LC%) | ArenaHard (WR%) | MTBench | IFEval | FollowBench |
|--------|-----------|-------------------|-----------------|---------|--------|-------------|
| SFT | 100K | 25.1 | 32.9 | 7.5 | 35.9 | 51.3 |
| RM (Skywork-8B) | 10K | 28.8 | 32.3 | 7.6 | 34.5 | 51.4 |
| GRM (GPT-4o-mini) | 10K | 27.1 | 28.7 | 7.4 | 35.2 | 50.9 |
| DPO | 10K | 24.8 | 28.8 | 7.5 | 35.5 | 49.5 |
| **RLVRR** | **10K** | **31.5** | **36.2** | **7.7** | **36.8** | **53.1** |

RLVRR with 10K examples outperforms 100K SFT and an 8B reward model across all metrics.

### Ablation Study

| Configuration | AlpacaEval2 | ArenaHard | Notes |
|---------------|-------------|-----------|-------|
| Full RLVRR | 31.5 | 36.2 | Complete framework |
| w/o hierarchical extraction (direct keywords) | 30.6 | 35.0 | Hierarchical contribution: +0.9 |
| w/o style reward | 29.8 | 33.1 | Style signal is effective |
| w/o multi-reference ($I=1$) | 30.2 | 34.5 | Multi-reference improves robustness |
| BLEU as reward | 24.3 | 27.5 | n-gram far inferior to keywords |
| Random reward | 22.5 | 25.1 | Baseline |

### Key Findings

- RLVRR incurs negligible computational overhead: only 0.71% additional cost compared to random rewards, whereas loading a reward model requires extra GPU memory and computation.
- RLVRR integrates seamlessly with RLVR, enabling unified training on both reasoning tasks and open-ended generation tasks.
- In-depth analysis shows that RLVRR improves output quality while preserving diversity, unlike SFT which tends to produce mode-collapsed outputs.
- BLEU performs poorly as a reward signal—n-gram precision fails to capture the key content aligned with human preferences.

## Highlights & Insights

- **Elegant "reward chain" concept**: The transition from "verifying a single point" to "verifying a chain" represents a natural extension of the RLVR paradigm. The keyword chain preserves the deterministic verifiability of content while allowing expressive freedom—combining the precise guidance of SFT with the exploratory nature of RL.
- **Eliminating the reward model**: Replacing billion-parameter reward models with rule-based checking (regex matching, Python code) substantially reduces RL training cost and instability. This idea generalizes to any scenario with reference answers.
- **High data efficiency**: Outperforming 100K SFT with only 10K examples demonstrates that the exploration mechanism in RL is far more data-efficient than supervised learning for alignment tasks.

## Limitations & Future Work

- **Dependence on reference quality**: Both keyword extraction and style checking are derived from references; if reference quality is poor or biased, RLVRR will learn incorrect patterns accordingly.
- **Reliance on GPT-4o-mini for extraction**: The data construction stage requires a powerful LLM; the effectiveness of open-source alternatives remains unvalidated.
- **Shallow style checking**: Current checks cover only surface attributes such as length and formatting; deeper stylistic properties such as tone and logical coherence cannot be verified with simple code.
- **Validated only on ≤7B models**: Whether RLVRR retains its advantage on larger models (e.g., 70B+) remains unknown.

## Related Work & Insights

- **vs. RLHF/DPO**: RLHF requires preference data and a reward model, incurring high cost and susceptibility to reward hacking; DPO performs offline optimization but lacks online exploration. RLVRR preserves the online exploration advantage of RL while eliminating the reward model.
- **vs. BLEU-as-reward** (Chang et al. 2025): BLEU measures n-gram precision and cannot distinguish key content from filler text. RLVRR uses hierarchical keywords to precisely capture core concepts.
- **vs. RLPR** (Yu et al. 2025): RLPR uses the model's own token probabilities as rewards but is effective only for short answers. RLVRR is applicable to long-form open-ended generation.

## Rating

- Novelty: ⭐⭐⭐⭐ The "reward chain" concept is novel, though the content reward is essentially keyword matching and does not constitute a breakthrough technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 10+ benchmarks, multiple model families, detailed ablations, diversity analysis, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Narrative is clear; the "dot→chain" analogy is intuitive.
- Value: ⭐⭐⭐⭐⭐ Highly practical—provides a low-cost, scalable RL training solution for alignment tasks without ground-truth answers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](../../NeurIPS2025/reinforcement_learning/generalizing_verifiable_instruction_following.md)
- [\[ICLR 2026\] References Improve LLM Alignment in Non-Verifiable Domains](references_improve_llm_alignment_in_non-verifiable_domains.md)

</div>

<!-- RELATED:END -->
