---
title: >-
  [Paper Note] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards
description: >-
  [ACL 2026][LLM Reasoning][Process Reward Models] This paper proposes using Planning Domain Definition Language (PDDL) to automatically generate large-scale…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Process Reward Models"
  - "PDDL"
  - "Planning Problems"
  - "Step-Level Rewards"
  - "Reasoning Evaluation"
date: 2026-05-08
content_hash: 9bc9c7b1dafd65c9
---

# Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards

**Conference**: ACL 2026  
**arXiv**: [2604.17957](https://arxiv.org/abs/2604.17957)  
**Code**: [https://github.com/Babelscape/prm-meets-planning/](https://github.com/Babelscape/prm-meets-planning/)  
**Area**: LLM Reasoning  
**Keywords**: Process Reward Models, PDDL, Planning Problems, Step-Level Rewards, Reasoning Evaluation

## TL;DR
This paper proposes using Planning Domain Definition Language (PDDL) to automatically generate large-scale, high-precision step-level reward datasets for training Process Reward Models (PRMs), achieving significant improvements across both mathematical and non-mathematical reasoning benchmarks.

## Background & Motivation

**Background**: Process Reward Models (PRMs) have become essential tools for evaluating the reasoning quality of Large Language Models (LLMs). By providing reward feedback for each step in a Chain-of-Thought (CoT), PRMs can detect errors in intermediate reasoning steps—even if the final answer is correct, intermediate steps may still contain logical flaws.

**Limitations of Prior Work**: Existing PRM training datasets suffer from three core issues. First, PRM800k relies on human annotation, which is costly, difficult to scale, and has an inter-annotator agreement of only 75%, implying up to 25% of annotations may be incorrect. Second, although Math-Shepherd uses automatic annotation (estimating step quality via LLM-generated completions), it is computationally expensive and only provides coarse-grained binary labels. Third, almost all existing datasets are limited to the mathematics domain, lacking coverage of broader logical reasoning.

**Key Challenge**: High-quality step-level reward data requires precise annotation and broad domain coverage, but human annotation is expensive and unreliable, while automated methods like LLM sampling lack precision and diversity.

**Goal**: To design a scalable, high-precision, cross-domain automatic generation framework for PRM datasets.

**Key Insight**: Planning problems naturally possess the structure of logical reasoning—each planning action corresponds to a reasoning step, and the correctness of an action can be precisely determined by rules (e.g., whether it is executable, optimal, or leads to a dead-end). This structural characteristic makes PDDL an ideal source for generating precise step-level rewards.

**Core Idea**: Transform PDDL planning problems into natural language reasoning chains and use planners to automatically and precisely assign five-level rewards (0.0–1.0) to each step, thereby constructing large-scale, high-quality PRM training data.

## Method

### Overall Architecture
The entire pipeline is divided into three stages: (1) PDDL Problem Generation: Approximately 15,000 planning problems are automatically generated across 11 different PDDL domains; (2) Dataset Construction: For each step of every problem, random action sampling and optimal action calculation are performed, assigning precise five-level rewards to each action; (3) PRM Training: The PDDL-generated data is combined with existing datasets (PRM800k or Math-Shepherd) to train the PRM.

### Key Designs

1.  **Five-Level Reward Classification System**:

    - **Function**: Precisely categorizes each reasoning step into five classes and assigns corresponding reward values.
    - **Mechanism**: Non-executable (0.0) → Dead-end (0.25) → Backtracking (0.5) → Suboptimal (0.75) → Optimal (1.0). Each action's category is precisely determined using an external planner (Fast Downward + A* + LM-Cut).
    - **Design Motivation**: Compared to existing binary or three-level annotations, five-level rewards provide richer supervisory signals, better distinguishing between "poor," "average," and "good" reasoning steps.

2.  **Diverse Action Sampling Strategy (Algorithm 1)**:

    - **Function**: Generates reasoning steps of various qualities at each state to construct positive and negative samples.
    - **Mechanism**: At each state $s$, $y$ actions (including executable and non-executable) are first randomly sampled; each action is evaluated using eval_action and translated into natural language. Then, the optimal action is executed to proceed to the next state until the goal state is reached.
    - **Design Motivation**: By sampling multiple actions of different qualities at the same state, the dataset ensures a rich set of positive and negative contrastive samples, helping the PRM learn to distinguish between varying qualities of reasoning steps.

3.  **Regression Training Objective**:

    - **Function**: Converts PRM training from a classification problem to a regression problem.
    - **Mechanism**: Utilizing the continuous reward values (0.0–1.0) from the PDDL dataset, a scalar regression head is added to the decoder to directly predict the reward value of each step, rather than classifying them as "good/bad."
    - **Design Motivation**: Five-level continuous rewards are naturally suited for regression training. Experiments show that the regression objective outperforms classification objectives on most benchmarks.

### Loss & Training
A head-based architecture is adopted, adding a regression head on top of the decoder to predict scalar rewards. During training, PDDL2PRM data is mixed with PRM800k or Math-Shepherd—PDDL data provides precise reasoning signals, while existing datasets provide more natural reasoning expression patterns.

## Key Experimental Results

### Main Results (ProcessBench Benchmark)

| Model | GSM8K F1 | MATH F1 | OlympiadBench F1 | Omni-MATH F1 | Average F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Math-7B-PRM800k | 68.2 | 62.6 | 50.7 | 44.3 | 56.5 |
| +PDDL (Regression) | **77.3** | **70.1** | **52.4** | **51.6** | **62.9** |
| Llama-3.1-8B-PRM800k | 61.1 | 51.3 | 36.5 | 38.6 | 46.9 |
| +PDDL (Regression) | **71.4** | **54.6** | 34.2 | 36.8 | **49.3** |

### Ablation Study

| Configuration | Average F1 | Description |
| :--- | :--- | :--- |
| Qwen2.5-Math-PRM800k+PDDL (Regression) | 62.9 | Full model |
| Qwen2.5-Math-PRM800k (Regression) | 60.6 | 2.3 drop without PDDL data |
| Qwen2.5-Math-PRM800k (Classification) | 57.2 | Classification is 3.4 worse than regression |
| Qwen2.5-Math-MathShepherd+PDDL (Regression) | 34.2 | Math-Shepherd baseline also improves |

### Key Findings
- Across all benchmarks, PRM performance consistently improves after adding PDDL data, with an average F1 Gain of 2.4–6.4 percentage points.
- The regression training objective outperforms classification in most cases, with significant differences on difficult benchmarks (OlympiadBench, Omni-MATH).
- The Gain from PDDL data is even more pronounced on non-mathematical reasoning tasks (the Rooms domain served as a held-out test domain, verifying cross-domain generalization).
- The dataset contains approximately 1 million reasoning steps covering 11 PDDL domains; the generation process is entirely CPU-based, requiring no GPUs.

## Highlights & Insights
- The idea of using **planning problems as a source of reasoning supervision** is ingenious—the formal nature of PDDL allows step-level rewards to be precisely determined via rules, completely avoiding the uncertainty of human annotation and LLM sampling. This approach can be extended to other formal tasks with verifiable intermediate steps.
- The **five-level reward classification system** is more aligned with the actual quality distribution of reasoning steps than simple binary labels. The "Backtracking" and "Suboptimal" categories are particularly valuable as they represent reasoning that is "not entirely wrong but not good enough."
- PDDL data generation requires no GPUs and is highly parallelizable, making it far superior to LLM sampling methods in terms of cost-efficiency.

## Limitations & Future Work
- The natural language reasoning chains generated by PDDL are based on template translation, lacking the fluency and diversity of real CoT; thus, they currently serve as augmentation rather than independent training data.
- Only 11 planning domains have been verified; more complex PDDL domains might generate excessively long reasoning chains beyond the PRM's processing capacity.
- The paper does not explore the effects of combining PDDL data with larger-scale PRMs (e.g., 70B).
- Future work could consider designing finer-grained reward levels or utilizing planning heuristic functions to provide continuous reward values instead of discrete levels.

## Related Work & Insights
- **vs PRM800k**: PRM800k relies on human annotation (\$25/hr), whereas Ours automatically generates data at zero cost with higher annotation precision (rule-based vs 75% human agreement).
- **vs Math-Shepherd**: Math-Shepherd estimates step quality via multiple LLM samplings, which is computationally expensive and yields only binary labels; Ours directly provides five-level precise rewards via a planner.
- **vs VersaPRM**: VersaPRM also attempts to expand into non-mathematical domains but still relies on LLMs as judges, which introduces annotation noise.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Introducing PDDL planning problems into PRM data generation is a highly novel and elegant idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Experiments across multiple models and benchmarks are relatively complete, though validation on larger-scale models is missing.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear and the formulation is rigorous, though the method section contains many symbols.
- **Value**: ⭐⭐⭐⭐ Provides a low-cost, high-precision paradigm for PRM data generation with strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ACL 2026\] C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences](c2_scalable_rubric-augmented_reward_modeling_from_binary_preferences.md)
- [\[ACL 2026\] Stabilizing Efficient Reasoning with Step-Level Advantage Selection](stabilizing_efficient_reasoning_with_step-level_advantage_selection.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[NeurIPS 2025\] Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards](../../NeurIPS2025/llm_reasoning/smaller_models_smarter_rewards_a_two-sided_approach_to_process_and_outcome_rewar.md)

</div>

<!-- RELATED:END -->
