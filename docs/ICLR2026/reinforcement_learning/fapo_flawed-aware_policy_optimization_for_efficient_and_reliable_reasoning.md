---
title: >-
  [Paper Note] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning
description: >-
  [ICLR2026][Reinforcement Learning][RLVR] To address the problem of flawed-positive rollouts in RLVR training—where the model reaches a correct answer through unreliable reasoning—this paper proposes the FAPO algorithm. FAPO employs a GenRM to detect flawed reasoning and applies a parameter-free reward penalty mechanism that realizes a natural "exploit-then-suppress" learning trajectory, simultaneously improving outcome correctness, process reliability, and training stability.
tags:
  - ICLR2026
  - Reinforcement Learning
  - RLVR
  - flawed positives
  - reward shaping
  - generative reward model
  - process reward
  - GRPO
date: 2026-05-08
content_hash: 9b58a3ba9b73a0bf
---

# FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning

**Conference**: ICLR2026  
**arXiv**: [2510.22543](https://arxiv.org/abs/2510.22543)  
**Code**: [fapo-rl.github.io](https://fapo-rl.github.io)  
**Area**: Reinforcement Learning  
**Keywords**: RLVR, flawed positives, reward shaping, generative reward model, process reward, GRPO

## TL;DR
To address the problem of flawed-positive rollouts in RLVR training—where the model reaches a correct answer through unreliable reasoning—this paper proposes the FAPO algorithm. FAPO employs a GenRM to detect flawed reasoning and applies a parameter-free reward penalty mechanism that realizes a natural "exploit-then-suppress" learning trajectory, simultaneously improving outcome correctness, process reliability, and training stability.

## Background & Motivation
RLVR (Reinforcement Learning with Verifiable Rewards) is the prevailing paradigm for enhancing the reasoning capabilities of LLMs, in which models optimize their policy by exploring reasoning trajectories and treating correct answers as positive signals. However, standard rule-based outcome rewards only verify whether the final answer is correct, and cannot distinguish the quality of the underlying reasoning process.

This gives rise to a critical problem: **flawed-positive rollouts**—instances where the model happens to produce the correct answer through unreliable mechanisms such as answer-guessing or jump-in-reasoning, yet receives the same positive reward as a fully correct reasoning chain. These flawed reasoning patterns are continuously reinforced during training, ultimately capping the model's reasoning potential.

The authors' analysis of models including Qwen2.5-Math-7B and Llama3.3-70B reveals that flawed positives account for 20%–40% of correct rollouts and persist throughout the entire RL training process (remaining at approximately 30% with little change).

## Core Problem
1. **The dual nature of flawed positives**: In early training, when the model lacks the capacity to produce fully correct reasoning, flawed positives serve as a "stepping stone" that facilitates rapid capability acquisition; in later stages, however, they impede the model's evolution toward genuine problem-solving ability.
2. **How to detect flawed positives**: Existing models either over-criticize (high recall, low precision) or are too large for practical use in online RL.
3. **How to balance exploitation and suppression timing**: An adaptive mechanism is needed that permits exploitation during the warm-up phase and gradually suppresses flawed positives during the refinement phase.

## Method

### 1. Flawed-Positive Detection: FAPO-GenRM
A compact and efficient generative reward model (GenRM) is trained via RL on top of Qwen3-4B-Instruct, with the following reward design:

$$R_{\text{FAPO-GenRM}} = R_{\text{Outcome}} + R_{\text{Process}}$$

- **$R_{\text{Outcome}}$**: Outcome reward for predicting correct/incorrect (+1/−1).
- **$R_{\text{Process}}$**: Step-level penalty, applied only when a flawed positive is correctly detected, defined as $-|\hat{t}_\theta - t^*|/n$, where $\hat{t}_\theta$ is the predicted error position, $t^*$ is the ground-truth error position, and $n$ is the total number of steps.

Two key aspects of this design:
- **Beyond guessing**: The process penalty forces the model to genuinely localize the error rather than merely predicting whether a flaw exists.
- **Natural reward shift**: In early training, outcome correctness dominates (large gain from $-1 \to 1$); as outcome rewards saturate, optimization naturally shifts toward process quality.

The training dataset FAPO-Critic-85K is constructed by generating rollouts on DAPO-Math-17K using multiple LLaMA/Qwen models (7B–70B), with step-level error positions annotated by Qwen3-32B.

### 2. Flawed-Positive Penalization: Adaptive Reward Adjustment
Upon detecting flawed positives, the rewards in RL training are adjusted as follows:

$$R_{\text{FAPO}}(o, a^* | \theta) = R_{\text{RLVR}}(o, a^*) + R_\Delta(o, a^* | \theta)$$

where $R_\Delta = -\lambda$ when a rollout is detected as a flawed positive, and 0 otherwise. With the default $\lambda = 1$, the reward of a flawed positive is reduced from +1 to 0.

**Natural optimization transition mechanism**: Let $\alpha$ denote the proportion of positive samples and $\beta$ the proportion of negative samples in the current rollout batch, and define the learning progress as $\rho = \alpha/\beta$.
- When $\rho < 1$ (negatives dominate, i.e., warm-up phase): flawed positives still carry positive advantage and are exploited.
- When $\rho > 1$ (positives dominate, i.e., refinement phase): the advantage of flawed positives approaches or falls below zero and is naturally suppressed.
- When $\rho > 3$: the advantage of positive samples is scaled down, yielding more stable training.

The choice of $\lambda = 1$ is derived from a majority-guided strategy, placing the transition point exactly at $\rho = 1$ without introducing additional hyperparameters.

### 3. Engineering Architecture
- The GenRM is deployed as an external LLM service on the compute cluster, decoupled asynchronously from rollout inference and actor training.
- Load balancing is achieved via multiple workers and a router.
- Token budget for the GenRM is controlled through an overlong reward strategy and checkpoint selection.
- Total training time increases by less than 20%.

## Key Experimental Results

### GenRM Detection Performance
- FAPO-GenRM-4B outperforms the teacher model Qwen3-32B on both FlawedPositiveBench and ProcessBench.
- It achieves significant gains over the Qwen3-4B-Instruct baseline and strong baselines such as Qwen2.5-Math-PRM-72B.
- It resolves the "over-criticism" problem (high recall, low precision) exhibited by existing models.

### Reasoning Performance (Qwen2.5-Math-7B + GRPO baseline)
- On **AIME24 / AIME25 / GPQA-Diamond**, FAPO outperforms the baseline at nearly all intermediate checkpoints.
- The proportion of flawed positives decreases substantially (from approximately 30%).
- Training curves are smoother, with no notable performance degradation in later stages.
- No increase in token budget is required (improvements are not attributed to longer responses).

### Ablation Study
- A stronger GenRM leads to better final RL performance, indicating a positive correlation between detection accuracy and downstream performance.
- Self-correction analysis shows that FAPO naturally transitions toward fully correct rollouts in later training, with shorter response lengths and more efficient reasoning.
- Step-ratio reward (scoring proportional to the fraction of correct steps) leads to reward hacking—the model outputs only high-confidence steps and skips uncertain ones.

## Highlights & Insights
1. **Systematic analysis of flawed positives**: This work is the first to reveal their dual role as an "early stepping stone and later obstacle," providing a new perspective on RLVR training.
2. **Parameter-free adaptive mechanism**: The choice of $\lambda=1$ is theoretically derived without introducing additional hyperparameters; the optimization direction shifts naturally as training progresses.
3. **Compact and efficient GenRM**: The 4B-parameter model surpasses the 32B teacher model and is decoupled asynchronously from RL training, adding less than 20% to total training time.
4. **Comprehensive validation**: Beyond reporting final performance, the paper evaluates all intermediate checkpoints, thoroughly demonstrating training stability.

## Limitations & Future Work
1. The GenRM introduces additional inference overhead; while currently kept below 20%, it may become a bottleneck in larger-scale systems.
2. FlawedPositiveBench is constructed based on ProcessBench, limiting the breadth of evaluation coverage.
3. Experiments are conducted primarily on mathematical reasoning and general QA; the approach has not been thoroughly explored on more complex verifiable tasks such as code generation.
4. The GenRM itself is also susceptible to reward hacking; although the paper discusses this risk, long-term training robustness requires further investigation.
5. The asynchronous architecture is an engineering compromise; a fully synchronous approach may yield better system efficiency.

## Related Work & Insights

| Method | Reward Type | Handles Flawed Positives | Parameter-Free | Characteristics |
|--------|-------------|--------------------------|----------------|-----------------|
| Standard RLVR | Binary outcome | No | Yes | Simple but reinforces flawed reasoning |
| PRM (discriminative) | Step-level scores | Indirectly | No | Dense reward, prone to hacking |
| Step-ratio reward | Step proportion | Indirectly | No | Induces jump-in reasoning |
| **FAPO** | Outcome + penalty | **Direct detection + adaptive penalty** | **Yes** | Natural learning trajectory, stable and efficient |

- The "exploit-then-suppress" paradigm of FAPO can be generalized to the handling of erroneous signals in other RL settings.
- The step-wise RL training methodology for GenRM can be applied to improve any process-level evaluation model (e.g., code review models).
- The paper's analysis of reward hacking (the failure case of step-ratio reward) serves as a cautionary reference for designing novel reward signals.
- The asynchronous GenRM architecture provides a practical reference for integrating external evaluators into large-scale RL systems.

## Rating
- Novelty: ⭐⭐⭐⭐ — The systematic analysis of flawed positives and the parameter-free penalty mechanism are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Full checkpoint evaluation, multi-dimensional ablations, human verification, and reward hacking analysis.
- Writing Quality: ⭐⭐⭐⭐ — Fluent exposition with a coherent motivation–analysis–method–experiment narrative.
- Value: ⭐⭐⭐⭐ — Practically meaningful for improving RLVR training quality; the GenRM solution is directly integrable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](../../ACL2026/reinforcement_learning/bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)
- [\[AAAI 2026\] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)

<!-- RELATED:END -->
