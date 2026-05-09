---
title: >-
  [Paper Note] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards
description: >-
  [ACL 2026][LLM Reasoning][Process Reward Models] This paper proposes leveraging the Planning Domain Definition Language (PDDL) to automatically generate large-scale, high-precision step-level reward datasets for training Process Reward Models (PRMs), achieving significant improvements on both mathematical and non-mathematical reasoning benchmarks.
tags:
  - ACL 2026
  - LLM Reasoning
  - Process Reward Models
  - PDDL
  - Planning Problems
  - Step-Level Rewards
  - Reasoning Evaluation
date: 2026-05-08
content_hash: 6dcc2b983535eda4
---

# Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards

**Conference**: ACL 2026
**arXiv**: [2604.17957](https://arxiv.org/abs/2604.17957)
**Code**: [https://github.com/Babelscape/prm-meets-planning/](https://github.com/Babelscape/prm-meets-planning/)
**Area**: LLM Reasoning
**Keywords**: Process Reward Models, PDDL, Planning Problems, Step-Level Rewards, Reasoning Evaluation

## TL;DR
This paper proposes leveraging the Planning Domain Definition Language (PDDL) to automatically generate large-scale, high-precision step-level reward datasets for training Process Reward Models (PRMs), achieving significant improvements on both mathematical and non-mathematical reasoning benchmarks.

## Background & Motivation

**State of the Field**: Process Reward Models (PRMs) have become a crucial tool for evaluating the reasoning quality of large language models. By providing reward signals at each step of a chain-of-thought (CoT), PRMs can detect errors in intermediate reasoning steps—even when the final answer is correct, intermediate steps may still contain logical flaws.

**Limitations of Prior Work**: Existing PRM training datasets suffer from three core issues. First, PRM800k relies on human annotation, which is costly and difficult to scale; inter-annotator agreement is required only at 75%, meaning up to 25% of labels may be incorrect. Second, Math-Shepherd employs automatic labeling (estimating step quality by sampling multiple continuations from an LLM), but incurs extremely high computational costs and yields only coarse-grained binary labels. Third, nearly all existing datasets are confined to the mathematical domain, lacking coverage of broader logical reasoning.

**Root Cause**: High-quality step-level reward data requires precise annotation and broad domain coverage, yet human annotation is expensive and unreliable, while automatic methods (e.g., LLM sampling) lack precision and diversity.

**Paper Goals**: To design a scalable, high-precision, cross-domain framework for automatic PRM dataset generation.

**Starting Point**: The authors observe that planning problems naturally exhibit the structure of logical reasoning—each planning action corresponds to a reasoning step, and the correctness of an action can be determined precisely by rules (executability, optimality, whether it leads to a dead end, etc.). This structured nature makes PDDL an ideal source for generating precise step-level rewards.

**Core Idea**: Transform PDDL planning problems into natural-language reasoning chains and use a planner to automatically assign five-level rewards (0.0–1.0) to each step, thereby constructing a large-scale, high-quality PRM training dataset.

## Method

### Overall Architecture
The pipeline consists of three stages: (1) **PDDL Problem Generation**: approximately 15,000 planning problems are automatically generated across 11 distinct PDDL domains; (2) **Dataset Construction**: for each step of each problem, random actions are sampled and optimal actions are computed, with precise five-level rewards assigned to each action; (3) **PRM Training**: the PDDL-generated data is combined with existing datasets (PRM800k or Math-Shepherd) to train the PRM.

### Key Designs

1. **Five-Level Reward Classification Scheme**

    - **Function**: Precisely assigns each reasoning step to one of five categories with a corresponding reward value.
    - **Mechanism**: Non-executable (reward 0.0) → Dead-end (0.25) → Backtracking (0.5) → Suboptimal (0.75) → Optimal (1.0). An external planner (Fast Downward + A* + LM-Cut) is used to determine the category of each action exactly.
    - **Design Motivation**: Compared to existing binary or three-level annotations, the five-level reward provides a richer supervisory signal, enabling finer discrimination among "poor," "mediocre," and "good" reasoning steps.

2. **Diverse Action Sampling Strategy (Algorithm 1)**

    - **Function**: Generates reasoning steps of varying quality at each state to construct positive and negative samples.
    - **Mechanism**: At each state $s$, $y$ actions (both executable and non-executable) are randomly sampled; each action is evaluated via `eval_action` and translated to natural language via `translate`; the optimal action is then executed to advance to the next state, repeating until the goal state is reached.
    - **Design Motivation**: By sampling actions of diverse quality at the same state, the dataset is enriched with contrastive positive and negative pairs, helping the PRM learn to distinguish reasoning steps of different quality.

3. **Regression-Based Training Objective**

    - **Function**: Reformulates PRM training as a regression problem rather than a classification problem.
    - **Mechanism**: Exploiting the continuous reward values (0.0–1.0) of the PDDL dataset, a scalar regression head is added on top of the decoder to directly predict the reward of each step, rather than classifying it as "good/bad."
    - **Design Motivation**: The five-level continuous rewards are naturally suited to regression training; experiments demonstrate that the regression objective outperforms the classification objective on most benchmarks.

### Loss & Training
A head-based architecture is adopted, with a regression head appended to the decoder to predict scalar rewards. During training, PDDL2PRM data is mixed with PRM800k or Math-Shepherd—PDDL data provides precise reasoning signals, while existing datasets contribute more naturalistic reasoning expression patterns.

## Key Experimental Results

### Main Results (ProcessBench Benchmark)

| Model | GSM8K F1 | MATH F1 | OlympiadBench F1 | Omni-MATH F1 | Avg. F1 |
|-------|----------|---------|------------------|--------------|---------|
| Qwen2.5-Math-7B-PRM800k | 68.2 | 62.6 | 50.7 | 44.3 | 56.5 |
| +PDDL (regression) | **77.3** | **70.1** | **52.4** | **51.6** | **62.9** |
| Llama-3.1-8B-PRM800k | 61.1 | 51.3 | 36.5 | 38.6 | 46.9 |
| +PDDL (regression) | **71.4** | **54.6** | 34.2 | 36.8 | **49.3** |

### Ablation Study

| Configuration | Avg. F1 | Note |
|---------------|---------|------|
| Qwen2.5-Math-PRM800k+PDDL (regression) | 62.9 | Full model |
| Qwen2.5-Math-PRM800k (regression) | 60.6 | −2.3 without PDDL data |
| Qwen2.5-Math-PRM800k (classification) | 57.2 | Classification objective lags regression by 3.4 |
| Qwen2.5-Math-MathShepherd+PDDL (regression) | 34.2 | Math-Shepherd baseline also benefits |

### Key Findings
- Across all benchmarks, adding PDDL data consistently improves PRM performance, with average F1 gains of 2.4–6.4 percentage points.
- The regression training objective outperforms the classification objective in most settings, with particularly pronounced differences on harder benchmarks (OlympiadBench, Omni-MATH).
- The improvement from PDDL data is more substantial on non-mathematical reasoning tasks (cross-domain generalization is validated using the Rooms domain as a held-out test domain).
- The dataset contains approximately one million reasoning steps spanning 11 PDDL domains; the entire generation process runs on CPU without requiring any GPU resources.

## Highlights & Insights
- The idea of **using planning problems as a source of reasoning supervision** is notably elegant—the formal nature of PDDL allows step-level rewards to be determined precisely through rules, entirely avoiding the uncertainty inherent in human annotation and LLM sampling. This paradigm can be extended to other formal tasks with verifiable intermediate steps.
- The **five-level reward classification scheme** better reflects the actual quality distribution of reasoning steps than simple binary annotation. The intermediate categories "backtracking" and "suboptimal" are particularly valuable, as they capture reasoning that is "not entirely wrong but not good enough."
- The PDDL data generation pipeline requires no GPU and is highly parallelizable, making it far more cost-efficient than LLM-sampling-based approaches.

## Limitations & Future Work
- The natural-language reasoning chains generated from PDDL rely on template-based translation, lacking the fluency and diversity of authentic CoT, and thus can only serve as augmentation data rather than a standalone training source.
- Only 11 planning domains have been evaluated; more complex PDDL domains may produce excessively long reasoning chains that exceed the PRM's processing capacity.
- The paper does not explore the effect of combining PDDL data with larger-scale PRMs (e.g., 70B).
- Future work could consider designing finer-grained reward levels or leveraging planning heuristic functions to provide continuous reward values instead of discrete five-level labels.

## Related Work & Insights
- **vs. PRM800k**: PRM800k relies on human annotation (at \$25/hr); the proposed method generates data at zero cost automatically, with higher annotation precision (rule-based determination vs. 75% human inter-annotator agreement).
- **vs. Math-Shepherd**: Math-Shepherd estimates step quality via repeated LLM sampling, incurring high computational cost and yielding only binary labels; the proposed method directly assigns five-level precise rewards via a planner.
- **vs. VersaPRM**: VersaPRM also attempts to extend beyond the mathematical domain but still relies on an LLM as a judge, introducing annotation noise.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing PDDL planning problems into PRM data generation is a highly novel and elegant idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Experiments across multiple models and benchmarks are fairly comprehensive, though validation on larger-scale models is missing.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear and the formalization is rigorous, though the method section involves heavy notation.
- **Value**: ⭐⭐⭐⭐ — Provides a low-cost, high-precision paradigm for PRM data generation with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards](../../NeurIPS2025/llm_reasoning/smaller_models_smarter_rewards_a_two-sided_approach_to_process_and_outcome_rewar.md)
- [\[ACL 2026\] Generating Effective CoT Traces for Mitigating Causal Hallucination](generating_effective_cot_traces_for_mitigating_causal_hallucination.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)

<!-- RELATED:END -->
