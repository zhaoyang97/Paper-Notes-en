---
title: >-
  [Paper Note] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints
description: >-
  [ICLR 2026][Reinforcement Learning][Tool Use] This paper proposes AutoTool, which addresses reasoning collapse in direct RL training for LLM tool use and the overthinking problem in scaled models via a decoupled adaptive entropy constraint strategy. AutoTool enables automatic switching between long and short reasoning modes based on problem difficulty, achieving a 9.8% accuracy improvement while reducing reasoning token overhead by ~81%.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Tool Use
  - Test-Time Scaling
  - Entropy Constraint
  - Automatic Reasoning Scaling
date: 2026-05-08
content_hash: 2992c385b3e5e447
---

# AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints

**Conference**: ICLR 2026
**arXiv**: [2603.13348](https://arxiv.org/abs/2603.13348)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Tool Use, Reinforcement Learning, Test-Time Scaling, Entropy Constraint, Automatic Reasoning Scaling

## TL;DR
This paper proposes AutoTool, which addresses reasoning collapse in direct RL training for LLM tool use and the overthinking problem in scaled models via a decoupled adaptive entropy constraint strategy. AutoTool enables automatic switching between long and short reasoning modes based on problem difficulty, achieving a 9.8% accuracy improvement while reducing reasoning token overhead by ~81%.

## Background & Motivation

1. **State of the Field**: Integration of LLMs with external tools is a key capability for AI agents. RLVR (Reinforcement Learning with Verifiable Rewards) has successfully enabled test-time scaling on math and code tasks, but its effectiveness in tool use remains unvalidated.

2. **Limitations of Prior Work**: (1) Direct RL training in tool-use settings leads to "reasoning collapse" — the model fails to sufficiently extend its reasoning length to solve complex problems; (2) Distilled models generate lengthy reasoning for all problems, wasting substantial tokens on simple queries.

3. **Root Cause**: While RL training on math tasks naturally increases reasoning length, it shortens reasoning length in tool-use tasks. The root cause is that low entropy causes the model to prematurely converge to short-reasoning strategies.

4. **Paper Goals**: Design a training method that automatically selects reasoning modes based on problem difficulty — extended thinking for complex problems and direct answers for simple ones.

5. **Starting Point**: The paper identifies a strong positive correlation between low information entropy and reasoning collapse, while also finding that a naive entropy constraint is extremely sensitive to its coefficient.

6. **Core Idea**: Decouple the policy losses for long and short reasoning, applying an adaptive entropy constraint to long reasoning to maintain exploration capability, while applying a fixed constraint to short reasoning to prevent over-exploration.

## Method

### Overall Architecture
A three-stage pipeline: (1) Data preparation — constructing the PubTool mixed dataset; (2) Warm-up SFT — mixing long and short reasoning data to make the model difficulty-aware; (3) Decoupled adaptive entropy constraint RL — GRPO combined with decoupled entropy regularization.

### Key Designs

1. **Warm-up SFT with Mixed Reasoning Data**:
    - Function: Enable the model to initially perceive problem difficulty.
    - Mechanism: Training data is annotated by sampling 8 times each from a non-thinking model (Qwen2.5-7B-Instruct) and a thinking model (Qwen3-32B). If the non-thinking model answers correctly, its short reasoning is used as the label; otherwise, the thinking model's long reasoning is used. An auto-thinking template is designed to support mode switching.
    - Design Motivation: Direct RL training leads to collapse; SFT warm-up is needed to establish initial long and short reasoning capabilities.

2. **Decoupled Adaptive Entropy Constraint**:
    - Function: Differentially control the exploration capacity of long and short reasoning.
    - Mechanism: The entropy coefficient $\beta_i$ in the policy loss $\mathcal{L}_p$ is decoupled by reasoning mode: short reasoning uses a fixed $\beta_s$, while long reasoning uses an adaptive $\beta_l$. $\beta_l$ is dynamically adjusted via an auxiliary loss $\mathcal{L}_\beta^l = \frac{1}{N}\sum (1-m_i)\cdot\beta_l\cdot(H_i - H_l)$, which increases $\beta_l$ when $H_i < H_l$ to encourage exploration.
    - Design Motivation: Low entropy in direct RL leads to reasoning collapse, yet globally high entropy causes over-exploration on simple problems.

3. **Asymmetric Reward Design**:
    - Function: Encourage short reasoning for simple problems and long reasoning for complex ones.
    - Mechanism: Correct + no-thinking: +1.0; Correct + thinking: +0.5; Incorrect + thinking: −0.5; Incorrect + no-thinking: −1.0.
    - Design Motivation: Higher rewards for correct short reasoning encode efficiency preference; heavier penalties for incorrect short reasoning encourage the model to think when needed.

### Loss & Training
- Base model: Qwen2.5-7B-Instruct
- RL algorithm: GRPO
- Data: PubTool (8.2k SFT + 7k RL), sourced from ToolACE + xLAM + Hermes
- Data quality optimization: Removal of trivially easy or excessively hard samples; filtering based on reward variance across multiple training rounds.

## Key Experimental Results

### Main Results

| Model | BFCL Overall | Non-Live | Live | Multi-Turn |
|------|-------------|----------|------|-----------|
| Qwen2.5-7B (Base) | 53.69 | 86.46 | 67.44 | 7.62 |
| PubTool-SFT | 58.17 | 88.98 | 77.28 | 9.68 |
| PubTool-Distilled | 60.30 | 87.73 | 78.64 | 15.65 |
| **AutoTool-7B** | **70.12** | **89.76** | **80.22** | **38.18** |
| GPT-4o | 70.42 | 87.67 | 79.88 | 43.00 |

### Ablation Study

| Configuration | Overall | Change |
|------|---------|------|
| Full method | 70.12 | — |
| w/o data refine | 63.69 | −6.43 |
| w/o decouple | 64.23 | −5.89 |
| w/o adapt coeff | 67.78 | −2.34 |

### Key Findings
- The thinking rate reaches 45% in Multi-Turn settings and 0% in Non-Live settings, demonstrating that the model has learned to automatically judge problem difficulty.
- Reasoning trajectories for complex problems are extended by 5×, while simple problems remain concise.
- Data quality optimization is the most critical component (−6.43% upon removal).
- AutoTool-7B approaches GPT-4o on BFCL (70.12 vs. 70.42).

## Highlights & Insights
- This work is the first to identify and address the "reasoning collapse" phenomenon in RL training for tool use.
- The collapse mechanism is understood through the lens of information entropy, revealing a strong positive correlation with entropy rather than with data difficulty distribution.
- The asymmetric reward design elegantly encodes efficiency preference.
- The 7B model outperforms most SFT/RLVR models of comparable scale and approaches frontier model performance.

## Limitations & Future Work
- Validation is limited to tool-use scenarios and has not been extended to other agent tasks.
- The warm-up SFT stage relies on an external reasoning model (Qwen3-32B) to generate long reasoning data.
- The target entropy value for the adaptive entropy constraint still requires manual pre-specification.
- Data sources are limited to mixtures of public datasets; effectiveness on domain-specific tools has not been validated.

## Related Work & Insights
- **vs. DeepSeek-R1**: R1 successfully scales reasoning on math/code tasks but faces collapse in tool-use settings.
- **vs. Thinkless**: Thinkless also performs reasoning mode switching, but relies on SFT distillation rather than RL.
- **vs. AdaCtrl**: AdaCtrl performs adaptive reasoning control; AutoTool achieves better automation through decoupled entropy constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of reasoning collapse and the decoupled entropy constraint solution represent significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, multiple baselines, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The logical chain of preliminary analysis → findings → method is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Directly informative for LLM agent training practices.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [\[ICLR 2026\] Learning to Orchestrate Agents in Natural Language with the Conductor](learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)
- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](entropy-preserving_reinforcement_learning.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_diverse_behaviors.md)
- [\[ICLR 2026\] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation](momagen_generating_demonstrations_under_soft_and_hard_constraints_for_multi-step.md)

<!-- RELATED:END -->
