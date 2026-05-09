---
title: >-
  [Paper Note] Reward Modeling for Scientific Writing Evaluation
description: >-
  [ACL 2026][LLM Alignment][Reward Model] This paper proposes SciRM and SciRM-Ref, two open-source reward models specifically designed for scientific writing evaluation. Through a two-stage reinforcement learning approach (GRPO), the models are optimized separately for evaluation preference alignment and reasoning capability, enabling fine-grained multi-aspect evaluation across diverse scientific writing tasks while generalizing to unseen evaluation tasks and criteria.
tags:
  - ACL 2026
  - LLM Alignment
  - Reward Model
  - Scientific Writing Evaluation
  - GRPO
  - Multi-aspect Evaluation
  - Reasoning Enhancement
date: 2026-05-08
content_hash: 22b3f6e0ad749701
---

# Reward Modeling for Scientific Writing Evaluation

**Conference**: ACL 2026
**arXiv**: [2601.11374](https://arxiv.org/abs/2601.11374)
**Code**: [https://github.com/UKPLab/acl2026-expert-rm](https://github.com/UKPLab/acl2026-expert-rm)
**Area**: LLM Alignment / Scientific Writing Evaluation
**Keywords**: Reward Model, Scientific Writing Evaluation, GRPO, Multi-aspect Evaluation, Reasoning Enhancement

## TL;DR

This paper proposes SciRM and SciRM-Ref, two open-source reward models specifically designed for scientific writing evaluation. Through a two-stage reinforcement learning approach (GRPO), the models are optimized separately for evaluation preference alignment and reasoning capability, enabling fine-grained multi-aspect evaluation across diverse scientific writing tasks while generalizing to unseen evaluation tasks and criteria.

## Background & Motivation

**State of the Field**: LLMs have been widely applied to scientific text generation tasks such as related work writing, review generation, and paper revision. However, evaluating the quality of such generated outputs remains an open challenge. The most common approach is LLM-as-a-judge, where an LLM directly scores the generated content.

**Limitations of Prior Work**: (1) General-purpose LLM judges struggle to reason over domain knowledge and task-specific preferences in scientific writing evaluation, frequently producing self-contradictory assessments (as illustrated in Figure 1); (2) existing reward models are optimized for general benchmarks (e.g., mathematical reasoning, code, helpfulness) and are ill-suited to the nuanced requirements of scientific writing; (3) most reward models rely on pairwise comparisons and cannot perform independent evaluation against explicit criteria; (4) existing models are optimized for fixed scoring rubrics, and performance degrades when rubrics change.

**Root Cause**: Scientific writing evaluation requires dynamic adaptation to varying tasks, aspects, and scoring criteria—even within a single document, different aspects may call for conflicting standards. Yet existing models encode evaluation preferences statically during training, lacking the flexibility to adapt at inference time.

**Paper Goals**: To develop open-source scientific writing evaluation reward models capable of dynamically adapting at inference time to explicit constitutions (evaluation criteria, scoring rules, and examples).

**Starting Point**: Evaluation is framed as a conditional generation task, where the model receives a constitution as contextual conditioning and interprets and follows the evaluation criteria through an explicit reasoning process. The two-stage training teaches the model to first "score according to criteria" and then "reflect on criteria to self-correct its reasoning."

**Core Idea**: A two-stage GRPO training pipeline is employed to train a reward reasoning model: the first stage teaches the model to follow constitutions for evaluation, and the second stage teaches self-reflection and self-correction. Joint multi-task training further improves cross-task generalization.

## Method

### Overall Architecture

The input consists of three components: a task query $q$ (the scientific text to be evaluated), an evaluation constitution $c$ (containing scoring rules and criterion descriptions), and scoring examples $e$. The model produces a reasoning process $j$ (wrapped in `<reasoning>` tags) and a final score $s$ (wrapped in `<score>` tags). Training data spans multiple tasks, including related work evaluation (binary labels) and review quality assessment (1–5 scale).

### Key Designs

1. **Stage 1: Evaluation Preference Optimization**

   - **Function**: Trains the model to perform accurate scientific writing evaluation conditioned on a given constitution.
   - **Mechanism**: The GRPO algorithm is used for optimization. The reward function adopts a hierarchical structure: missing `<score>` tags yield $-0.5$; non-numeric output yields $0$; a numeric output outside the valid range yields $0.25$; a valid but incorrect score yields $0.5$; a correct score yields $1.5$. An additional length penalty function $f(L, T)$ applies quadratic penalties for outputs that are too short or too long, preventing reward hacking.
   - **Design Motivation**: The hierarchical reward distinguishes different error types (formatting errors vs. semantic errors), guiding the model toward incremental improvement. The length penalty discourages the degenerate behavior of outputting only a score while bypassing the reasoning process.

2. **Stage 2: Reasoning Enhancement via Self-Reflection**

   - **Function**: Strengthens the model's capacity for self-reflection and correction, enabling it to revisit evaluation criteria under uncertainty.
   - **Mechanism**: The output of the Stage 1 model is taken with the score portion removed, retaining only the reasoning; a reflection prompt is then appended, asking the model to re-examine the criteria before producing a final score. The reward function considers both the initial score $s_i$ and the final score $s_f$: self-correction ($s_i \neq s^*$ and $s_f = s^*$) receives the highest reward of $1.0$, while degradation ($s_i = s^*$ and $s_f \neq s^*$) incurs the heaviest penalty of $-1.0$.
   - **Design Motivation**: The model is encouraged to actively correct errors during reasoning while being penalized for unstable behavior that turns correct answers into incorrect ones. This addresses the limitation of constitutional AI approaches that internalize rules into weights and cannot dynamically adapt to new criteria.

3. **Joint Multi-task Training**

   - **Function**: Improves the model's generalization across diverse scoring criteria and evaluation dimensions.
   - **Mechanism**: Training data encompasses multiple scientific writing tasks (related work evaluation covering consistency, citation type, and citation consistency; review quality covering actionability, grounding, verifiability, and helpfulness) across heterogeneous scoring scales (binary and 1–5).
   - **Design Motivation**: Single-task training tends to overfit specific criteria; joint training enables the model to acquire meta-level evaluation competency rather than memorizing task-specific patterns.

### Loss & Training

The models are built on Qwen2.5-7B with LoRA fine-tuning. Both stages use the GRPO algorithm. Inference temperature is set to $1.0$ with top-p $0.95$. Each experiment is repeated 5 times, and mean and standard deviation are reported. The Stage 1 model is referred to as SciRM; the two-stage model is referred to as SciRM-Ref.

## Key Experimental Results

### Main Results

| Task | SciRM-Ref | Qwen2.5-7B | Qwen3-8B | GPT-5.2 | Prometheus |
|------|-----------|------------|----------|---------|------------|
| Review – Actionability | **Best** | Low | Mid | High | Low |
| Review – Verifiability Extraction | **Best** | Mid | Mid | High | Mid |
| Related Work – Consistency | **Best** | Low | Mid | Mid | Low |
| Related Work – Citation Consistency | **Near-perfect** | Mid | Mid | High | Low |

### Ablation Study (Unseen Aspects / Task Generalization)

| Configuration | Performance | Note |
|---------------|-------------|------|
| SciRM-Masked (2 aspects removed) | Exceeds most baselines on unseen aspects | Demonstrates generalization, not overfitting |
| Unseen task – Novelty evaluation | $0.71+$ alignment accuracy | Generalizes to completely unseen tasks |
| Unseen task – Revision evaluation | Surpasses most baselines | Effective cross-task transfer |

### Key Findings

- Two-stage training consistently improves performance; the second stage (reflection) contributes most on tasks requiring stronger reasoning.
- SciRM-Masked outperforms most baselines on unseen evaluation aspects, demonstrating that the model learns a general evaluation structure rather than overfitting to specific aspects.
- On completely unseen tasks—novelty evaluation and paper revision evaluation—SciRM still outperforms general-purpose baselines, demonstrating strong generalization.
- Reasoning-capable models such as Qwen3 and o3-mini perform exceptionally well on certain aspects (e.g., Grounding), likely owing to their inherent reasoning abilities.

## Highlights & Insights

- The design philosophy of "constitution-conditioned evaluation" is particularly valuable: rather than internalizing evaluation criteria into model weights, criteria are provided as explicit inference-time conditions. This allows a single model to evaluate diverse tasks under different standards, greatly enhancing practical utility.
- The reflection reward design in Stage 2 is elegant: it rewards not only whether the final answer is correct, but also whether the model corrects an initial error ($+1.0$) or degrades from a correct answer ($-1.0$), effectively promoting stable self-correction behavior.
- The hierarchical reward function design can be transferred to other RLHF tasks requiring structured outputs—assigning differentiated penalties according to the severity of different error types.

## Limitations & Future Work

- Only 7B-scale models are evaluated; different scaling behaviors may emerge with larger models.
- Training data primarily covers scientific literature in NLP/ML; generalization to other disciplines (e.g., biology, physics) remains unverified.
- The quality of the constitution directly affects evaluation outcomes; low-quality criteria may mislead the model.
- The hyperparameter $k$ in the length penalty requires manual tuning; an adaptive scheme may be preferable.

## Related Work & Insights

- **vs. Prometheus / Selene**: General-purpose LLM-as-judge models, not optimized for scientific writing. SciRM substantially outperforms these through domain-specific training data and constitution-conditioned design.
- **vs. DeepSeek-GRM**: A general reward model employing pairwise evaluation, which cannot perform pointwise evaluation against explicit criteria. SciRM's independent multi-aspect evaluation is better suited to scientific writing scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of reward reasoning models specifically to scientific writing evaluation; the two-stage training design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers seen/unseen aspects and tasks, multiple baselines, and multiple metrics with detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-substantiated motivation.
- Value: ⭐⭐⭐⭐ Provides a practical open-source solution for automated scientific writing evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](../../ICLR2026/llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)

<!-- RELATED:END -->
