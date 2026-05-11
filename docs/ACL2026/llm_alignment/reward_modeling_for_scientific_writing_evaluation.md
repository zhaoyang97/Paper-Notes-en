---
title: >-
  [Paper Note] Reward Modeling for Scientific Writing Evaluation
description: >-
  [ACL 2026][LLM Alignment][Reward Model] This paper proposes SciRM and SciRM-Ref, two open-source reward models tailored for scientific writing evaluation. Through two-stage reinforcement learning (GRPO) that separately o…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Reward Model"
  - "Scientific Writing Evaluation"
  - "GRPO"
  - "Multi-Aspect Evaluation"
  - "Reasoning Enhancement"
date: 2026-05-08
content_hash: f8086db4eec1443f
---

# Reward Modeling for Scientific Writing Evaluation

**Conference**: ACL 2026
**arXiv**: [2601.11374](https://arxiv.org/abs/2601.11374)
**Code**: [https://github.com/UKPLab/acl2026-expert-rm](https://github.com/UKPLab/acl2026-expert-rm)
**Area**: LLM Alignment / Scientific Writing Evaluation
**Keywords**: Reward Model, Scientific Writing Evaluation, GRPO, Multi-Aspect Evaluation, Reasoning Enhancement

## TL;DR

This paper proposes SciRM and SciRM-Ref, two open-source reward models tailored for scientific writing evaluation. Through two-stage reinforcement learning (GRPO) that separately optimizes evaluation preference and reasoning ability, these models achieve fine-grained multi-aspect evaluation across diverse scientific writing tasks and generalize to unseen evaluation tasks and criteria.

## Background & Motivation

**State of the Field**: LLMs have been widely applied to scientific text generation tasks such as related work writing, review generation, and paper revision, yet evaluating these generated outputs remains an open challenge. The most common approach is LLM-as-a-judge, where an LLM directly scores the output.

**Limitations of Prior Work**: (1) General-purpose LLM judges struggle to reason over domain knowledge and task-specific preferences in scientific writing evaluation, frequently producing self-contradictory assessments (as illustrated in Figure 1); (2) existing reward models are optimized for general benchmarks (mathematical reasoning, code, helpfulness, etc.) and are ill-suited to the nuanced requirements of scientific writing; (3) most reward models rely on pairwise comparisons and cannot perform independent evaluation against explicit criteria; (4) existing models are optimized for fixed rubrics and degrade when the rubric changes.

**Root Cause**: Scientific writing evaluation requires dynamic adaptation to varying tasks, aspects, and rubrics—even conflicting criteria across aspects of the same text—yet existing models encode evaluation preferences rigidly during training, lacking the flexibility to adapt at inference time.

**Paper Goals**: To construct open-source scientific writing evaluation reward models capable of dynamically adapting at inference time based on an explicit constitution (evaluation criteria + scoring rules + examples).

**Starting Point**: Evaluation is framed as a conditional generation task—the model receives a constitution as context, and interprets and follows the evaluation criteria through a chain-of-thought reasoning process. The two training stages teach the model to "score according to criteria" and then to "reflect on criteria and revise its own reasoning," respectively.

**Core Idea**: A two-stage GRPO training pipeline is used to train a reward reasoning model: the first stage teaches the model to follow a constitution for evaluation; the second stage teaches self-reflection and self-correction; joint multi-task training improves cross-task generalization.

## Method

### Overall Architecture

The input consists of three components: a task query $q$ (the scientific text to be evaluated), evaluation criteria $c$ (the constitution, including scoring rules and criterion descriptions), and scoring examples $e$. The model outputs a reasoning process $j$ (wrapped in `<reasoning>` tags) and a final score $s$ (wrapped in `<score>` tags). Training data spans multiple tasks, including related work evaluation (binary labels) and review quality assessment (1–5 scale).

### Key Designs

1. **Stage 1: Evaluation Preference Optimization**

    - **Function**: Teaches the model to perform accurate scientific writing evaluation conditioned on a given constitution.
    - **Mechanism**: Optimized using the GRPO algorithm. The reward function adopts a hierarchical structure: outputs missing a `<score>` tag receive $-0.5$; non-numeric outputs receive $0$; numeric outputs outside the valid range receive $0.25$; valid but incorrect scores receive $0.5$; correct scores receive $1.5$. A length penalty function $f(L,T)$ is additionally introduced to apply a quadratic penalty when outputs are too short or too long, preventing reward hacking.
    - **Design Motivation**: The hierarchical reward design distinguishes between different error types (format errors vs. semantic errors) and guides the model toward incremental improvement. The length penalty addresses the degenerate behavior of outputting only a score while skipping the reasoning process.

2. **Stage 2: Reasoning Enhancement via Self-Reflection**

    - **Function**: Enhances the model's capacity for self-reflection and correction, enabling it to revisit criteria when uncertain.
    - **Mechanism**: The output of the Stage 1 model is taken with the score stripped, retaining only the reasoning; a reflection prompt is appended to instruct the model to re-examine the criteria before producing a final score. The reward function considers both the initial score $s_i$ and the final score $s_f$: self-correction ($s_i \neq s^*$ and $s_f = s^*$) receives the highest reward of $1.0$, while regression ($s_i = s^*$ and $s_f \neq s^*$) incurs the heaviest penalty of $-1.0$.
    - **Design Motivation**: This encourages the model to actively correct errors during reasoning while penalizing unstable behavior of switching from correct to incorrect. It addresses the inability of constitutional AI approaches—which internalize rules into weights—to dynamically adapt to new criteria.

3. **Multi-Task Joint Training**

    - **Function**: Improves generalization across different rubrics and evaluation dimensions.
    - **Mechanism**: Training data encompasses multiple scientific writing tasks (consistency, positioning type, and positioning consistency for related work evaluation; actionability, grounding, verifiability, and helpfulness for review quality assessment) across different scoring scales (binary and 1–5).
    - **Design Motivation**: Single-task training tends to overfit to specific rubrics, whereas joint training enables the model to acquire meta-evaluation capabilities rather than memorizing task-specific patterns.

### Loss & Training

The model is fine-tuned from Qwen2.5-7B using LoRA. Both stages use the GRPO algorithm, with inference temperature $1.0$ and top-p $0.95$. Each experiment is repeated 5 times; mean and standard deviation are reported. The model trained through Stage 1 alone is referred to as SciRM; the model trained through both stages is referred to as SciRM-Ref.

## Key Experimental Results

### Main Results

| Task | SciRM-Ref | Qwen2.5-7B | Qwen3-8B | GPT-5.2 | Prometheus |
|------|-----------|------------|----------|---------|------------|
| Review – Actionability | **Best** | Low | Mid | High | Low |
| Review – Verifiability | **Best** | Mid | Mid | High | Mid |
| Related Work – Consistency | **Best** | Low | Mid | Mid | Low |
| Related Work – Positioning Consistency | **Near-perfect** | Mid | Mid | High | Low |

### Ablation Study (Unseen Aspects / Task Generalization)

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| SciRM-Masked (2 aspects removed) | Still outperforms most baselines on unseen aspects | Demonstrates generalization rather than overfitting |
| Unseen Task – Novelty Evaluation | $0.71+$ alignment accuracy | Generalizes to completely unseen tasks |
| Unseen Task – Revision Evaluation | Outperforms most baselines | Effective cross-task transfer |

### Key Findings

- Two-stage training consistently improves performance; Stage 2 (reflection) yields the greatest benefit on tasks requiring strong reasoning.
- SciRM-Masked still outperforms most baselines on unseen evaluation aspects, demonstrating that the model learns a general evaluation structure rather than overfitting to specific aspects.
- On completely unseen tasks—novelty evaluation and paper revision assessment—SciRM still outperforms general-purpose baselines, exhibiting strong generalization.
- Reasoning-capable models such as Qwen3 and o3-mini perform exceptionally well on certain aspects (e.g., Grounding), likely attributable to their inherent reasoning capacity.

## Highlights & Insights

- The "constitution-conditioned evaluation" design is highly valuable—evaluation criteria are not internalized into model weights but are instead supplied as explicit conditions at inference time. This allows a single model to evaluate different tasks under different rubrics, greatly enhancing practical utility.
- The reflection reward design in Stage 2 is elegant: it rewards not merely whether the final answer is correct, but whether the model corrected an earlier error (reward $1.0$) or regressed from a correct answer (penalty $-1.0$), effectively encouraging stable self-correction behavior.
- The hierarchical reward function design is transferable to other RLHF tasks requiring structured outputs—assigning different penalties based on the severity of different error types.

## Limitations & Future Work

- Only 7B-scale models are evaluated; larger models may exhibit different scaling behavior.
- Training data remain predominantly drawn from NLP/ML literature; generalization to other disciplines (e.g., biology, physics) has not been validated.
- The quality of the constitution directly affects evaluation performance; low-quality criteria may mislead the model.
- The hyperparameter $k$ in the length penalty requires manual tuning; an adaptive scheme may be preferable.

## Related Work & Insights

- **vs. Prometheus / Selene**: General-purpose LLM-as-judge models not optimized for scientific writing. SciRM substantially outperforms them via domain-specific training data and constitution-conditioned design.
- **vs. DeepSeek-GRM**: A general-purpose reward model employing pairwise evaluation, which cannot perform pointwise assessment against explicit criteria. SciRM's independent multi-aspect evaluation is better suited to scientific writing scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of reward reasoning models specifically to scientific writing evaluation; the two-stage training design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers seen/unseen aspects and tasks, multiple baselines, and multiple metrics; analysis is thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with well-motivated argumentation.
- **Value**: ⭐⭐⭐⭐ — Provides a practical open-source solution for automatic scientific writing evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](../../ICLR2026/llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)

</div>

<!-- RELATED:END -->
