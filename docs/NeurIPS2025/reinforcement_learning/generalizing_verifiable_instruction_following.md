---
title: >-
  [Paper Note] Generalizing Verifiable Instruction Following
description: >-
  [NeurIPS 2025][Reinforcement Learning][Instruction Following] This paper introduces IFBench, a benchmark for evaluating generalization in precise instruction following, demonstrating that current SOTA models severely overfit to the 25 constraint templates of IFEval. It further proposes IF-RLVR, a training method based on GRPO with verifiable rewards, which significantly improves both in-domain and out-of-domain instruction following performance.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Instruction Following
  - Verifiable Constraints
  - RLVR
  - GRPO
  - Generalization
date: 2026-05-08
content_hash: 535ae99550681650
---

# Generalizing Verifiable Instruction Following

**Conference**: NeurIPS 2025
**arXiv**: [2507.02833](https://arxiv.org/abs/2507.02833)
**Code**: [Available](https://github.com/allenai/IFBench)
**Area**: Reinforcement Learning
**Keywords**: Instruction Following, Verifiable Constraints, RLVR, GRPO, Generalization

## TL;DR

This paper introduces IFBench, a benchmark for evaluating generalization in precise instruction following, demonstrating that current SOTA models severely overfit to the 25 constraint templates of IFEval. It further proposes IF-RLVR, a training method based on GRPO with verifiable rewards, which significantly improves both in-domain and out-of-domain instruction following performance.

## Background & Motivation

Precise Instruction Following (IF) is a critical capability for effective human–LLM interaction. Users frequently embed output constraints in their instructions, such as "answer only with yes or no" or "mention 'abracadabra' at least 3 times." IFEval is the most widely used evaluation benchmark, comprising 25 verifiable constraint templates, yet it has become saturated—many 2B-parameter models already exceed 80% accuracy.

**Core Finding**: Most models severely overfit to IFEval's 25 constraint types and fail to generalize to unseen output constraints. This is largely because mainstream training approaches (as described in, e.g., the Nemotron-4 technical report) directly synthesize instruction-following data from the IFEval taxonomy. This paper exposes such overfitting by constructing IFBench, on which leading models such as GPT-4.1 and Claude 3.7 Sonnet score below 50%.

## Method

### Overall Architecture

The work comprises three interrelated contributions:

1. **IFBench** (Evaluation): 58 novel, diverse, and challenging verifiable constraints spanning 7 categories: counting, ratio, words, sentence, format, custom, and copy.
2. **IFTrain** (Training Constraints): 29 new manually annotated training constraints with corresponding verification functions.
3. **IF-RLVR** (Training Method): RL training using GRPO with verifiable rewards.

### Key Designs

#### IFBench Benchmark Construction

- **Constraint Sources**: Collected from LM user feedback and hand-crafted to cover core IF skills.
- **Selection Criteria**: Each constraint must be accompanied by a Python verification function to ensure reproducible evaluation.
- **Test Prompt Construction**: Instantiated constraints are appended to held-out prompts from WildChat to prevent train–test leakage.
- **Evaluation Setup**: 300 prompts, each with 1–2 constraints.
  - **Single-turn**: Instruction and constraint are provided simultaneously.
  - **Multi-turn**: The model first responds to the instruction, then is asked in a second turn to revise the response to satisfy the constraint.

**Representative Constraint Examples**: Maintaining a 2:1 ratio of declarative to interrogative sentences, using only unique words, copying a specific portion of the input, etc.

#### IF-RLVR Training Pipeline

**Data Construction**:
- Prompts are randomly sampled from Tülu-3-SFT.
- Each prompt is appended with 1 to $n$ constraints ($n \in \{1,2,3,4,5,6\}$).
- Constraints are drawn from IFTrain and IFEval with extended variable ranges.
- A constraint conflict dictionary is maintained to prevent contradictory constraint combinations.
- Approximately 60k–100k training prompts are generated in total.

**Training**:
- GRPO (Group Region Policy Optimization) is used for outcome supervision.
- Each output is scored based on whether the constraints are satisfied.

### Loss & Training

**Multi-Constraint Reward Function**:
$$\text{Instance Reward} = \sum_{i=1}^{n} \text{verifiable\_reward}_i \cdot \text{reward\_multiplier}_i \cdot \text{reward\_weight}_i$$

The reward multiplier and weight are generally set to 1, and can be adjusted to up- or down-weight specific rewards.

**Training Hyperparameters**: max_token_length=2048, temperature=1, lr=5e-7, 16 samples/prompt, 8 H100 GPUs, local mini-batch=32, approximately 2,000 training steps (~1 day). For base models, a reasoning chat template is used: max_token_length=10240, beta=0.

## Key Experimental Results

### Main Results

**Large Gap Between SOTA Models on IFEval vs. IFBench**:

| Model | IFEval (%) | IFBench (%) |
|-------|-----------|-------------|
| o3 | ~95 | ~55 |
| Claude 4 Sonnet | ~90 | <50 |
| Qwen3-32B | ~90 | <50 |
| GPT-4.1 | ~88 | <50 |

**IF-RLVR Training Results**:

| Model | IFEval Before→After | IFBench Before→After |
|-------|---------------------|----------------------|
| Tülu-3-8B-DPO | 82.4 → **92.2** | 28.9 → **44.6** |
| Qwen2.5-7B (base) | N/A → **87.8** | N/A → **53.7** |
| Llama3.1-8B (base) | N/A → **88.2** | N/A → **54.1** |
| OLMo2-instruct | 61.7 → **74.5** | 16.7 → **44.6** |

### Ablation Study

**Effect of Multi-Constraint Training** (Qwen2.5 policy):

| Constraints per Prompt | IFBench | IFEval |
|------------------------|---------|--------|
| 1 | 48.9 | 71.2 |
| 2 | 53.1 | 79.9 |
| 3 | **59.5** | 77.8 |
| 5 | 55.8 | 79.9 |
| 6 | 54.1 | **85.8** |

**Effect of Training Constraint Diversity**: Combining IFTrain (out-of-domain) and IFEval (in-domain) constraints yields the best overall performance. Training with only the 29 out-of-domain constraints already improves IFBench scores; adding all 25 IFEval constraints achieves the highest IFEval scores.

**Generalization Across Variable Ranges**: Training with a wider variable range (covering and extending beyond the test range) performs ≥ training on the same range > training on a disjoint range.

**GRPO vs. DPO Comparison** (same data and policy):

| Training Method | IFEval | IFBench |
|-----------------|--------|---------|
| DPO after DPO | 79.67 | 29.3 |
| **GRPO after DPO** | **89.65** | **30.6** |

**Base vs. Instruct Models under IF-RLVR**: Base models trained with a reasoning chat template generalize better on IFBench (54.1 vs. 44.6), suggesting that RLVR combined with reasoning is beneficial for IF generalization.

### Key Findings

1. **Severe Overfitting**: SOTA models exceed 90% on IFEval but fall below 50% on IFBench.
2. **Constraint Diversity Is Critical**: Increasing both the variety of training constraints and the number of constraints per prompt substantially improves generalization.
3. **GRPO Substantially Outperforms DPO**: Under identical data, GRPO consistently outperforms DPO, as RLVR can produce accurate training signals for prompts of arbitrary difficulty.
4. **Constraint–Task Trade-off**: After IF-RLVR training, models tend to prioritize satisfying constraints at the expense of response quality (LLM-as-judge scores drop from 7.0 to 6.4).
5. **RLVR Is Viable for Base Models**: Strong IF capability can be acquired via direct RLVR on base models without SFT or DPO pretraining.

## Highlights & Insights

- **Exposing the Generalization Illusion**: Saturation on IFEval reflects memorization of 25 constraint types rather than genuine IF capability.
- **IFBench Targets Long-Tail Constraints**: It covers skills where models are genuinely weak, including counting, ratio, and copy constraints.
- **Unique Advantage of RLVR**: Unlike DPO, which requires chosen/rejected pairs that are difficult to construct, RLVR only requires a verification function to generate training signal for prompts of any difficulty.
- **Instruction Hierarchy Finding**: Different models prioritize constraints and tasks differently—Qwen2.5 tends to prioritize constraints, while Tülu-3 tends to prioritize task quality.
- **Multi-Turn Training**: Mixing single-turn and multi-turn training data yields the best results.

## Limitations & Future Work

1. The paper focuses exclusively on verifiable constraints; many real-world user constraints are difficult to verify automatically.
2. Some constraints may appear unnatural or contrived.
3. IF-RLVR training slightly degrades performance on other downstream tasks (e.g., AlpacaEval).
4. Balancing strategies when constraints conflict with task requirements warrant further investigation—incorporating a preference reward model signal is suggested.
5. Joint training of IF-RLVR with other RLVR tasks such as mathematics and coding remains unexplored.

## Related Work & Insights

- **IFEval** (Zhou et al., 2023): An IF evaluation benchmark with 25 verifiable constraints; now largely saturated.
- **FollowBench** (Jiang et al., 2023): Tests IF ability under incrementally increasing numbers of constraints, but relies on LLM-as-judge evaluation.
- **VFF** (Wang et al.): Automatically generates verifiable training/test data and trains with SFT and DPO.
- **Tülu-3** (Lambert et al., 2024): First demonstrates the potential of RLVR for instruction following.
- **DeepSeek-R1** (Guo et al., 2025): A successful application of RLVR to mathematical reasoning.
- **Takeaway**: Verifiable rewards combined with diverse constraint combinations constitute the key paradigm for improving IF generalization.

## Rating

- **Novelty**: ★★★★☆ — IFBench fills an important gap; the IF-RLVR training method is systematic and comprehensive.
- **Experimental Thoroughness**: ★★★★★ — Ablations are exceptionally detailed, covering constraint count, variable ranges, training methods, and multi-turn settings.
- **Value**: ★★★★★ — IFBench, IFTrain, and IF-RLVR code are all open-sourced and directly reusable.
- **Writing Quality**: ★★★★☆ — Content is rich but information-dense; some experiments require careful cross-referencing.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] Financial Instruction Following Evaluation (FIFE)](financial_instruction_following_evaluation_fife.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](../../ICLR2026/reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)

<!-- RELATED:END -->
