---
title: >-
  [Paper Note] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving
description: >-
  [ICLR 2026][LLM Reasoning][RLVR] This paper proposes Re², a pure reinforcement learning method that trains LLMs to actively abandon unproductive reasoning chains and restart the solving process during inference. The appr…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "RLVR"
  - "chain-of-thought optimization"
  - "re-solving"
  - "overthinking"
date: 2026-05-08
content_hash: dd2f0a5fa65dbab8
---

# $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving

**Conference**: ICLR 2026
**arXiv**: [2603.07197](https://arxiv.org/abs/2603.07197)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: RLVR, LLM reasoning, chain-of-thought optimization, re-solving, overthinking

## TL;DR
This paper proposes Re², a pure reinforcement learning method that trains LLMs to actively abandon unproductive reasoning chains and restart the solving process during inference. The approach amplifies the rare redo behavior from ~0.5% to over 30%, achieving significant improvements over standard RLVR methods under the same training compute budget.

## Background & Motivation

The reasoning capabilities of large language models can be improved through reinforcement learning with verifiable rewards (RLVR), which enhances performance by increasing test-time computation. However, even after extensive RLVR training, models tend to generate unnecessary and low-quality reasoning steps when producing chain-of-thought (CoT) outputs, leading to the "overthinking" problem—consuming large numbers of tokens while degrading the quality of final answers.

A key observation is that **when the initial direction or quality of a CoT is poor, the model often fails to reach the correct answer**, even when it generates several times more tokens than it would in cases where the initial CoT is of good quality. This reveals a critical limitation: models trained with standard RLVR lack the ability to "cut losses" and "start over"—they persistently attempt to complete reasoning paths that have already gone astray.

The core idea of this paper is to teach LLMs to flexibly abandon unproductive reasoning paths during inference and restart the solving process when necessary, rather than always committing to the current trajectory through to a final answer.

## Method

### Overall Architecture
Re² (Reinforcement Learning with Re-solving) adopts a pure reinforcement learning approach without any prior supervised fine-tuning (SFT). The overall pipeline is: given a math/reasoning problem → the model generates an extended reasoning chain that may include multiple re-solving episodes → the final answer is evaluated via a verifiable reward → the policy is updated via reinforcement learning.

### Key Designs
1. **Re-solving Mechanism**: The core of Re² lies in enabling the model to insert "re-solve" markers during the reasoning process. When the model perceives that the current reasoning direction may be erroneous, it can choose to abandon the current path and restart from the beginning of the problem. The design motivation stems from observations of vanilla models, in which redo behavior occurs spontaneously but rarely (~0.5%), and these rare redo instances tend to correlate with better reasoning outcomes.

2. **Pure RL Training Strategy**: Unlike methods that require prior SFT data collection, Re² amplifies the model's existing but extremely rare redo behavior entirely through reinforcement learning. During training, when the model spontaneously adopts a re-solving strategy in a rollout and ultimately arrives at the correct answer, this behavior receives a positive reward, reinforcing it in subsequent training iterations. This approach eliminates the need to manually engineer re-solving formats.

3. **Progressive Behavior Amplification**: Training begins from the vanilla model's very low redo rate (~0.5%) and, through sustained RL training, the model gradually learns to employ the re-solving strategy more frequently. The redo rate ultimately rises to over 30%. This progressive process occurs naturally without requiring special curriculum design.

### Loss & Training
Re² employs a standard RLVR training framework using verifiable rewards as the training signal. The reward function checks the correctness of the model's final answer, assigning a positive reward for correct answers and a negative reward for incorrect ones. Crucially, Re² imposes no additional constraints on the format of the model's output—the model is free to decide whether to perform re-solving, and the reward signal naturally guides the model to adopt this strategy at appropriate moments.

## Key Experimental Results

### Main Results

| Dataset | Metric | Re² | Standard RLVR | Gain |
|--------|------|-----|-----------|------|
| Math reasoning benchmarks | Accuracy | Significantly above baseline | Baseline | Substantial improvement |
| Same training compute budget | Pass@1 | Higher | Lower | Consistent improvement |
| Test-time scaling | Multi-sample sampling | Performance continues to improve as sample count increases | Improvement plateaus | Superior scaling behavior |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Vanilla model | Redo rate ~0.5% | Redo behavior is extremely rare in the baseline |
| After Re² training | Redo rate >30% | Successfully amplifies redo behavior |
| With SFT pre-training | Comparison | Re²'s pure RL approach yields better results |
| Different compute budgets | Convergence curves | Re² achieves superior performance under the same budget |

### Key Findings
- When the initial CoT direction is poor, the model struggles to correct errors even when generating several times more tokens than usual, demonstrating the necessity of re-solving.
- Pure RL is sufficient to raise the redo rate from 0.5% to 30%+, without requiring SFT data.
- Re² exhibits superior test-time scaling behavior: performance continues to improve as the number of samples increases.
- Re-solving improves not only accuracy but also reasoning efficiency by reducing the generation of invalid tokens.

## Highlights & Insights
- **Simple yet effective design philosophy**: Rather than engineering more complex reasoning structures, Re² endows models with the ability to "start over," which aligns naturally with how humans approach problem-solving.
- **Pure RL training without SFT data**: This demonstrates that reinforcement learning alone can surface and amplify beneficial reasoning patterns latent in the model, offering a new direction for future LLM training.
- **In-depth analysis of the overthinking problem**: The paper clearly exposes the fragility of standard RLVR models when the initial CoT direction is suboptimal.
- **Test-time compute efficiency**: Re² improves not only pass@1 but also performs strongly in pass@k settings requiring multiple samples, indicating that the reasoning paths it generates are more diverse.

## Limitations & Future Work
- The paper focuses primarily on mathematical reasoning tasks; effectiveness in other reasoning domains such as code generation and logical reasoning remains to be validated.
- The re-solving mechanism increases the average output length, which may be undesirable in latency-sensitive inference scenarios.
- The decision of when to trigger re-solving is learned entirely implicitly by the model, and explicit analysis of triggering conditions is lacking.
- For simple problems, the re-solving mechanism may introduce unnecessary computational overhead.
- Whether Re² can be combined with more advanced CoT optimization methods such as tree-of-thought warrants further exploration.

## Related Work & Insights
- **RLVR methods**: Works such as DeepSeek-R1 improve LLM reasoning via verifiable rewards; Re² builds on this foundation to address the overthinking problem.
- **CoT optimization**: Unlike self-reflection and backtracking methods, Re² adopts a more radical "restart from scratch" strategy rather than performing local corrections.
- **Test-time compute optimization**: Re²'s test-time behavior suggests a positive effect of re-solving on sample diversity, which is synergistic with best-of-N sampling strategies.
- **Insight**: Rare but beneficial behavioral patterns latent in a model can be effectively amplified through RL training—a principle that may generalize to other domains.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](../../ICML2026/llm_reasoning/when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](../../AAAI2026/llm_reasoning/well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)
- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](../../NeurIPS2025/llm_reasoning/expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
