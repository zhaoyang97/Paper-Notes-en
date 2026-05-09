---
title: >-
  [Paper Note] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning
description: >-
  [ICLR 2026][Model Compression][planning benchmark] This paper introduces ACPBench Hard — an open-ended generative planning reasoning benchmark comprising 8 task types grounded in PDDL formal systems (13 domains × 8 tasks = 1,040 questions), equipped with symbolic validators that provide rigorous correctness guarantees. A systematic evaluation of 15 LLMs reveals that even the strongest reasoning model, o1-preview, achieves accuracy ≤66% on half the tasks, and all models fail almost completely on the most fundamental task of enumerating applicable actions, exposing fundamental deficiencies in current LLMs' planning reasoning capabilities.
tags:
  - ICLR 2026
  - Model Compression
  - planning benchmark
  - PDDL
  - generative evaluation
  - symbolic validator
  - action reasoning
date: 2026-05-08
content_hash: 56144ece10099091
---

# ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning

**Conference**: ICLR 2026
**arXiv**: [2503.24378](https://arxiv.org/abs/2503.24378)
**Code**: [https://ibm.github.io/ACPBench](https://ibm.github.io/ACPBench)
**Area**: Model Compression
**Keywords**: planning benchmark, PDDL, generative evaluation, symbolic validator, action reasoning

## TL;DR

This paper introduces ACPBench Hard — an open-ended generative planning reasoning benchmark comprising 8 task types grounded in PDDL formal systems (13 domains × 8 tasks = 1,040 questions), equipped with symbolic validators that provide rigorous correctness guarantees. A systematic evaluation of 15 LLMs reveals that even the strongest reasoning model, o1-preview, achieves accuracy ≤66% on half the tasks, and all models fail almost completely on the most fundamental task of enumerating applicable actions, exposing fundamental deficiencies in current LLMs' planning reasoning capabilities.

## Background & Motivation

Existing LLM planning evaluations face two layers of bottlenecks. **First**: benchmarks such as PlanBench and AutoPlanBench focus solely on end-to-end plan generation/validation and cannot pinpoint the specific cause of failure when a black-box model fails. ACPBench v1 decomposed the planning process into 7 atomic reasoning tasks (applicability, state progression, reachability, etc.), but used a **Boolean/multiple-choice** format. **Second**: the multiple-choice format is misaligned with the requirements of real planners — planners must *generate* answers from a large action space rather than selecting from 4 options. Answering correctly on multiple-choice questions does not imply the ability to complete generation tasks, and evaluating open-ended generation is itself far more challenging (the verification complexity of some tasks is PSPACE-complete).

The core idea of this paper is to upgrade ACPBench's 7 tasks from multiple-choice to **open-ended generation**, add a "next action" task (corresponding to optimal planning), and design a PDDL-based **symbolic validator** for each of the 8 tasks, entirely avoiding the unreliability of LLM-as-judge evaluation.

## Method

### 8 Atomic Planning Reasoning Tasks

ACPBench Hard decomposes planning capability into 8 atomic tasks spanning three levels: action-level, state-level, and plan-level.

| Level | Task | Abbr. | Generation Target | Verification Complexity |
|-------|------|-------|-------------------|------------------------|
| Action | Applicability | App | Enumerate **all** executable actions in the current state | $O(|A|)$ |
| Action | Progression | Prog | Given an action, list positive effects (newly true) and negative effects (become false) | $O(|F|)$ |
| State | Proposition Reachability | Reach | Identify propositions that can **never** become true from the current state | PSPACE-complete |
| State | Action Reachability | AReach | Identify actions that can **never** become executable | PSPACE-complete |
| Plan | Plan Validation | Val | Identify the **first** non-executable action in an action sequence | $O(1)$ |
| Plan | Plan Justification | Just | Remove 1–2 redundant actions from a plan and output the simplified plan | $O(|\pi| \cdot |F|)$ |
| Plan | Landmarks | Land | Identify **necessary subgoals** that every valid plan must pass through | PSPACE-complete |
| Plan | Next Action (new) | NextA | Select an action that reduces the optimal goal distance by 1 | PSPACE-complete |

### Key Designs

1. **Symbolic Validator System**: Each task is equipped with a dedicated verification algorithm. Simple tasks (App/Prog/Val) rely on set comparison; harder tasks (Reach/AReach/Land/NextA) first query a precomputed cache and invoke a PDDL planner upon a cache miss, thereby guaranteeing **completeness** and **correctness** of verification. For example, Landmarks verification is performed by constructing an auxiliary planning task $\Pi'$ (adding a marker proposition $p_\text{nach}$) to check whether a valid plan exists that bypasses the candidate landmark.

2. **Data Construction Pipeline**: Based on the 13 PDDL domains from ACPBench, 10 questions are generated per domain per task (1,040 questions total). Plans are generated by a top-quality planner, with a diverse planner as a fallback. Questions are converted from PDDL to natural language via templates.

3. **Lenient Syntax Parser**: To handle inconsistent model output formats, a grammar-based lenient parser is designed that automatically discards syntactically invalid tokens to maximally extract valid answers.

## Key Experimental Results

### Small/Medium Model Results

| Model | App | AReach | Just | Land | NextA | Prog | Reach | Val |
|-------|-----|--------|------|------|-------|------|-------|-----|
| Granite 3.1 8B | 0.00 | 0.00 | 0.21 | 0.08 | 0.22 | 0.36 | 0.33 | 0.09 |
| Llama 3.1 8B | 0.00 | 0.00 | 0.22 | 0.06 | 0.25 | 0.40 | 0.33 | 0.13 |
| DeepSeek Coder 33B | 0.02 | 0.02 | 0.21 | 0.10 | 0.17 | 0.42 | 0.18 | 0.15 |
| Granite 34B Code | 0.02 | 0.00 | 0.17 | 0.11 | 0.18 | 0.43 | 0.28 | 0.12 |

Small models achieve near-zero scores on App and AReach; the highest score across all tasks, Prog, reaches only 43%.

### Large Model & Reasoning Model Results

| Model | App | AReach | Just | Land | NextA | Prog | Reach | Val |
|-------|-----|--------|------|------|-------|------|-------|-----|
| Mixtral 8x22B | 0.10 | 0.02 | 0.31 | 0.26 | 0.32 | 0.68 | **0.37** | 0.23 |
| Llama 3.1 70B | 0.12 | 0.02 | 0.44 | 0.20 | 0.42 | 0.65 | 0.28 | 0.20 |
| GPT-4o mini | 0.07 | 0.01 | 0.14 | 0.04 | 0.35 | 0.59 | 0.22 | 0.27 |
| Llama 3.1 405B | 0.14 | 0.04 | 0.59 | 0.15 | 0.48 | 0.74 | 0.26 | 0.48 |
| **GPT-4o** | **0.25** | 0.01 | 0.54 | **0.29** | **0.55** | **0.78** | 0.32 | **0.62** |
| DeepSeek V3 | 0.21 | **0.05** | **0.65** | 0.12 | 0.47 | 0.76 | 0.32 | 0.56 |
| o1-mini | 0.38 | 0.06 | 0.44 | 0.38 | 0.64 | 0.70 | 0.60 | **0.78** |
| GPT OSS 20B | 0.03 | 0.09 | 0.14 | 0.47 | 0.62 | 0.72 | 0.50 | 0.14 |
| GPT OSS 120B | 0.00 | 0.13 | 0.05 | 0.49 | 0.78 | 0.79 | 0.68 | 0.70 |
| **o1-preview** | **0.44** | **0.12** | 0.46 | **0.56** | **0.80** | **0.89** | **0.66** | 0.26 |
| DeepSeek R1 | 0.05 | 0.01 | 0.52 | 0.20 | 0.36 | 0.77 | 0.24 | 0.53 |

### Representation Ablation (DeepSeek V3)

| Representation | App | AReach | Just | Land | NextA | Prog | Reach | Val | Avg. |
|----------------|-----|--------|------|------|-------|------|-------|-----|------|
| NL | 0.21 | 0.05 | 0.65 | 0.12 | 0.47 | 0.76 | 0.32 | 0.56 | 0.39 |
| PDDL | 0.31 | 0.07 | 0.74 | 0.21 | 0.53 | 0.87 | 0.33 | 0.55 | 0.44 |
| PDDL+NL | 0.32 | 0.09 | 0.68 | 0.19 | 0.60 | 0.88 | 0.37 | 0.61 | **0.47** |

Including formal PDDL descriptions improves average accuracy from 39% to 47%; however, when PDDL is available, using a classical planner directly is more appropriate.

### Key Findings

- **Applicability is near-universally failed**: On the most fundamental task of enumerating all executable actions, small models score ≈0% and the strongest model, o1-preview, reaches only 44%. The strict requirement to generate a **complete** set is the primary cause — switching to Jaccard similarity scoring raises o1-preview to 57% and Mixtral from 10% to 38%.
- **Action Reachability is the hardest task**: It requires reasoning about the **joint reachability** of multiple propositions in action preconditions; o1-preview scores only 12%. Most correct answers arise from recognizing the "None" case (all actions are reachable).
- **No universally dominant model**: No model achieves the best performance across all 8 tasks. GPT-4o leads on 5 of 8 tasks, but is outperformed by DeepSeek V3 by 11% on Just and 4% on AReach.
- **Questionable cost-effectiveness of reasoning models**: The o1 series incurs far greater computational cost than standard LLMs, yet achieves significant advantages only on Prog (89%) and NextA (80%), with scores ≤66% on half the tasks.
- **Anomalously low Val score for o1-preview (26%)**: In 86% of error cases, the predicted index differs from the correct index by exactly 1, indicating the model is close but consistently off by one step.
- **Progression is the easiest task** (o1-preview: 89%), yet models still make basic errors such as failing to recognize that after stacking a block, the top block becomes *clear*.

## Highlights & Insights

- **Precise diagnosis of capability deficits**: By decomposing the planning process into 8 atomic tasks, the framework enables exact identification of where models fail. App ≈ 0% indicates that LLMs cannot even enumerate executable actions — a prerequisite for all planning.
- **Methodological significance of symbolic validators**: The framework provides a fully reliable automatic evaluation scheme for open-ended generation tasks; the construction of certain verification algorithms (e.g., the auxiliary planning task for Landmarks) constitutes an independent technical contribution.
- **The gap between generation and multiple-choice**: The same model (GPT-4o) exhibits far lower error rates under multiple-choice format than under generative format (except for Val), demonstrating that multiple-choice questions substantially overestimate models' planning reasoning capabilities.

## Limitations & Future Work

- Template-based natural language is insufficiently naturalistic and diverges from real-world planning scenarios
- Coverage of only 13 PDDL domains may be insufficient to represent all planning reasoning patterns
- Only the final answer of a single generation is evaluated, without considering iterative self-correction over multiple steps
- The lenient parser extracts only the first answer, potentially missing subsequent attempts by the model
- Future directions include constructing training data with chain-of-thought (CoT) reasoning and extending to new task types such as object counting

## Related Work & Insights

- **vs. ACPBench v1**: v1 uses Boolean/multiple-choice format; this work upgrades to open-ended generation — a qualitative increase in difficulty
- **vs. PlanBench / AutoPlanBench**: Those works focus on end-to-end plan generation/validation and cannot diagnose atomic capability deficits
- **vs. ActionReasoningBench**: That benchmark conflates multiple capabilities into a single question and relies on LLM-as-judge evaluation; this work assigns one atomic capability per task with a corresponding symbolic validator

## Rating

- Novelty: ⭐⭐⭐⭐ Generative planning reasoning benchmark + complete symbolic validators
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models × 8 tasks + representation ablation + complexity analysis + per-domain analysis
- Writing Quality: ⭐⭐⭐⭐ Task definitions and verification algorithms are clearly presented; observations and analyses are detailed
- Value: ⭐⭐⭐⭐⭐ A benchmark platform for diagnosing LLM planning reasoning capabilities

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Efficient Reasoning with Balanced Thinking](efficient_reasoning_with_balanced_thinking.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [\[AAAI 2026\] Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)](../../AAAI2026/model_compression/satisficing_and_optimal_generalised_planning_via_goal_regression_extended_versio.md)
- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)

<!-- RELATED:END -->
