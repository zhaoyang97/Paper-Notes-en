---
title: >-
  [Paper Note] CAMEL: Confidence-Gated Reflection for Reward Modeling
description: >-
  [ICML 2026][Reinforcement Learning][Reward Model] This paper observes that the log-probability margin of the verdict token is highly correlated with judgment accuracy. Based on this…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Reward Model"
  - "Confidence Gating"
  - "Reflection Mechanism"
  - "GRPO"
  - "Counterfactual Prefix Augmentation"
date: 2026-05-08
content_hash: fb4a324e51947bf3
---

# CAMEL: Confidence-Gated Reflection for Reward Modeling

**Conference**: ICML 2026  
**arXiv**: [2602.20670](https://arxiv.org/abs/2602.20670)  
**Code**: Not yet released  
**Area**: Alignment RLHF / Reward Models / LLM Reasoning  
**Keywords**: Reward Model, Confidence Gating, Reflection Mechanism, GRPO, Counterfactual Prefix Augmentation

## TL;DR
This paper observes that the log-probability margin of the verdict token is highly correlated with judgment accuracy. Based on this, it proposes CAMEL—using a single token to provide a fast preference judgment first, only triggering reflection generation under low confidence. It utilizes counterfactual prefix augmentation in GRPO training to enhance self-correction capabilities. With 14B parameters, CAMEL achieves an average accuracy of 82.9% across three reward model benchmarks, exceeding previous 70B state-of-the-art models by 3.2%.

## Background & Motivation
**Background**: Reward models (RM) used as alignment signals for LLMs primarily follow two paradigms. Scalar discriminative models (e.g., Skywork-Reward, ArmoRM) are stable to train and fast during inference but output only a score and lack interpretability. Generative judges (e.g., J1, RM-R1) generate rationales before providing a judgment, which yields higher accuracy but requires generating hundreds to thousands of tokens per sample.

**Limitations of Prior Work**: The cost of generative RMs is hard to sustain in industrial deployments. RM-R1-DeepSeek-32B generates approximately 900 tokens on RewardBench and 1100 tokens on RM-Bench, even though many samples comprise "one good response and one bad response," which are easy cases that do not require lengthy reasoning. Treating simple and difficult samples equally with full reasoning generation is a waste of computational budget.

**Key Challenge**: Reward models face a clear efficiency-expressiveness trade-off. Simple samples should be processed instantly like scalar models, while difficult samples require reflection like generative models. However, there has been no suitable signal to inform the model whether a problem is difficult enough to require reflection.

**Goal**: (1) Identify a proxy indicator for "problem difficulty" that can be obtained without additional reasoning; (2) Create an adaptive routing reward model based on this indicator that only incurs generation costs for truly difficult samples; (3) Ensure that reflection actually corrects errors instead of merely echoing the original answer.

**Key Insight**: The authors observe that when a prompt requires a model to choose between A and B, the log-probability margin at the verdict token ($c(x) = |\log P(A|x) - \log P(B|x)|$) naturally characterizes the model's "certainty." Statistics on Qwen3-14B using Skywork-80K show that higher confidence correlates monotonically with higher prediction accuracy, with errors almost entirely concentrated in low-margin regions.

**Core Idea**: Use the single-token margin as a "zero-cost difficulty estimator" to construct a two-stage pipeline: "fast judgment first → reflect if confidence is low." Employ GRPO with counterfactual prefix augmentation during training to enable genuine self-correction.

## Method

### Overall Architecture
CAMEL splits reward modeling into two stages. Given $(q, r_a, r_b)$, the model first outputs an initial verdict $v_0 \in \{\texttt{A}, \texttt{B}\}$. The confidence score $c(x)$ is calculated from the two candidate probabilities of this verdict token. If $c(x) \geq \tau$ (high confidence), the process terminates immediately with $v_1 = v_0$, consuming only 1 generated token. If $c(x) < \tau$, the prompt triggers a brief reflection $J$ ("think again...") followed by the final verdict $v_1$. This structure of "scoring first then deciding whether to explain" is trained using GRPO with counterfactual prefix augmentation.

### Key Designs

1. **Confidence Score as a Difficulty Estimator**:
    - **Function**: Replaces explicit difficulty labeling with a single-token margin to decide whether to trigger reflection.
    - **Mechanism**: Define $c(x) = |\log P_\theta(v=\texttt{A}|x) - \log P_\theta(v=\texttt{B}|x)|$, representing the model's "potential difference" regarding preference. Plotting $c(x)$ against accuracy on the training distribution reveals a strong monotonic function, meaning accuracy/cost can be balanced simply by adjusting the threshold $\tau$.
    - **Design Motivation**: Eliminates the need for training a separate difficulty estimator or performing a second forward pass. Obtaining a high-quality routing signal for free is key to the zero-overhead framework.

2. **Confidence-Gated Two-Stage Judging Prompt**:
    - **Function**: Reconstructs the traditional "long rationale → final verdict" into a factorized structure: $v_0 \rightarrow \text{optional } J \rightarrow v_1$, allowing the confidence gate to be inserted mid-generation.
    - **Mechanism**: The prompt forces the model to output a verdict placeholder $v_0$ without explanation. During inference, the generation either stops or continues based on $c(x)$. If reflection is triggered, the model writes rationale $J$ and outputs $v_1$ after seeing its own $v_0$. The margin for $v_0$ is obtained in a single forward pass.
    - **Design Motivation**: Externalizes the discrete "to think or not" decision to the token probability level rather than letting the model judge its own need for reflection via language. This factorization allows easy and hard samples to follow different paths while sharing the same policy.

3. **Counterfactual Prefix Augmentation + GRPO**:
    - **Function**: Teaches the model when to stick to the initial judgment and when to be overturned by reflection.
    - **Mechanism**: Each sample $(x, z)$ is duplicated: one forces $v_0 = \texttt{A}$ and the other $v_0 = \texttt{B}$. RL credit is applied only to the reflection $J$ and final verdict $v_1$; $v_0$ is treated as context and not optimized. The reward is binary $R = +1/-1$ (whether $v_1$ matches ground truth). GRPO optimizes $\max_\theta \mathbb{E}[R(v_1, z)] - \beta \mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$.
    - **Design Motivation**: Without counterfactual augmentation, models easily learn the shortcut of "reflection = repeating $v_0$" since most initial judgments are correct. Forcing a wrong starting point compels the model to actually compare evidence and overturn itself during reflection, allowing self-correction to "emerge."

### Loss & Training
The training follows two phases: First, SFT is performed on Qwen3-14B using three preference datasets (Skywork-Reward-Preference-80K + Code-Preference-Pairs + Math-Step-DPO-10K) to learn preference formats. Next, one epoch of GRPO is conducted with counterfactual prefixes, using a KL coefficient $\beta$ to control deviation from the reference policy. During inference, $\tau = 5$ is used by default (adjustable).

## Key Experimental Results

### Main Results
Evaluation across three reward model benchmarks (RewardBench / RM-Bench / JudgeBench) comparing scalar and generative RMs:

| Model | RewardBench | RM-Bench | JudgeBench | Avg |
|------|-------------|----------|------------|-----|
| INF-ORM-Llama3.1-70B (Prev. SOTA) | 95.1 | 73.8 | 70.2 | 79.7 |
| RM-R1-Qwen-Instruct-32B (Generative) | 89.0 | 73.1 | 64.8 | 75.6 |
| J1-Llama-70B | 93.3 | 82.7 | 60.0 | 78.7 |
| **CAMEL-Fast (14B, 1 token)** | 90.5 | 74.8 | 65.2 | 76.8 |
| **CAMEL-Reflection (14B, always)** | 92.8 | **84.2** | **71.6** | **82.9** |
| **CAMEL (gated, $\tau=5$)** | 92.4 | 81.9 | 69.1 | 81.1 |

CAMEL-Reflection is 3.2% higher than the previous SOTA on average. CAMEL-Fast matches or exceeds RM-R1-Qwen-Instruct-32B using only 1 token. The 14B model matches or surpasses 70B baselines.

### Ablation Study

| Configuration | RewardBench | RM-Bench | JudgeBench | Avg |
|------|-------------|----------|------------|-----|
| Qwen3-14B (No tune) | 81.9 | 71.1 | 62.6 | 71.9 |
| Qwen3-14B + Reflection | 83.3 | 73.2 | 65.0 | 73.8 |
| Qwen3-14B-SFT | 90.6 | 72.7 | 64.8 | 76.0 |
| Qwen3-14B-GRPO (No Counterfactual) | 91.2 | 83.5 | 62.9 | 79.2 |
| Qwen3-14B-GRPO + Reflection | 90.0 | 84.0 | 74.2 | 82.7 |
| **CAMEL (full)** | 92.4 | 81.9 | 69.1 | 81.1 |

### Key Findings
- Reflection gains are most significant in reasoning-intensive benchmarks: from always-fast to always-reflect, accuracy increased by +2.3% on RewardBench, +9.4% on RM-Bench, and +6.4% on JudgeBench, indicating more difficult samples in the latter two.
- Counterfactual prefixes are critical: removing them causes a 5% drop on JudgeBench (74.2 → 69.1) for GRPO+Reflection, showing that without them, reflection degrades into "repeating the initial judgment."
- Pareto Frontier: CAMEL strictly outperforms RM-R1 on RewardBench and RM-Bench. RM-R1-DeepSeek-32B generates 900–1100 tokens to reach ~87/74, while CAMEL is close with 1 token and surpasses it with fewer total tokens by adjusting $\tau$.
- Post-training confidence distributions shift left (more conservative), matching expectations that the model learns to distinguish certainty. Self-correction confusion matrices show reflection yields a net gain of 77 correct samples on RewardBench and 1233 on RM-Bench.

## Highlights & Insights
- **"Free Difficulty Signals"**: The single-token margin carries almost no overhead but reliably predicts accuracy. This is a highly portable trick: any pairwise discriminative task (multiple choice QA, safety classification, tool selection) can reuse this for routing, abstention, or uncertainty estimation.
- **Externalizing "Whether to Think"**: Previous chain-of-thought works often let the model decide whether to think, resulting in either always thinking or never thinking. CAMEL uses hard decisions at the token probability level—clean, adjustable, and zero-regression.
- **Counterfactual Prefixes as a Secret Weapon for RL**: Many self-correction works struggle with "models refusing to change answers" because $v_0$ is correct in most of the training distribution. Forcing a wrong starting point is a general fix transferable to self-refinement and self-debate scenarios.
- Overall, restructuring reward modeling into "adaptive two-stage computation" is more engineering-oriented than strictly choosing between scalar or generative paradigms.

## Limitations & Future Work
- The threshold $\tau$ is globally fixed, but confidence distributions across different tasks/domains are not aligned—safety tasks generally have higher confidence, while math tasks have lower. Ideal implementation would use dynamic or bucketed $\tau$.
- Validation is limited to Qwen3-14B; scaling laws (whether the "low confidence = difficult" correlation holds for 70B/100B+) are unclear. For small models, the margin might be too noisy to be effective.
- Reflection token length is not strictly controlled; the average length of the reflection segment is not reported. Long reflections could partially offset savings from gating.
- Future directions: (a) Learning $\tau$ automatically; (b) Introducing multi-level reflection (short/long/very long) for finer-grained routing; (c) Embedding this architecture into actor-critic RLHF pipelines as the critic.

## Related Work & Insights
- **vs RM-R1 (Generative RM SOTA)**: RM-R1 uses distilled rubrics + RL with verifiable rewards, generating long rationales for every sample. CAMEL shares similar data but adds gating, achieving higher accuracy with fewer tokens, establishing a strictly superior Pareto frontier.
- **vs Generative RM (J1, Critic-RM)**: J1/Critic-RM emphasize explicit reasoning traces to improve judgment. CAMEL borrows the reflection mechanism but rejects "indiscriminate reasoning."
- **vs Self-Consistency / Self-Refine**: These methods rely on multi-sample voting or iterative correction. CAMEL uses the margin from a single forward pass to decide on refinement, avoiding repeated sampling costs.
- **vs Uncertainty-based Abstention**: Traditional selective prediction uses confidence to decide whether to answer; CAMEL uses it to decide whether to "think again," representing another paradigm of conditional compute.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of single-token margin for difficulty estimation and counterfactual prefixes is a refreshing framework, though individual components are not entirely disruptive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on three mainstream benchmarks + Pareto curves + ablations + self-correction analysis, though lacking multi-backbone validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic from motivation to observation to method to experiment is exceptionally smooth. Formulas and figures are well-coordinated.
- Value: ⭐⭐⭐⭐⭐ A deployment-friendly 14B reward model exceeding 70B baselines is directly usable for industrial RLHF pipelines. The tricks are transferable across tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](../../ICLR2026/reinforcement_learning/rm-r1_reward_modeling_as_reasoning.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[ACL 2026\] LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations](../../ACL2026/reinforcement_learning/lovec_reinforcement_learning_for_better_verbalized_confidence_in_long-form_gener.md)
- [\[ICML 2026\] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](one_bias_after_another_mechanistic_reward_shaping_and_persistent_biases_in_langu.md)
- [\[ICLR 2026\] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](../../ICLR2026/reinforcement_learning/stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)

</div>

<!-- RELATED:END -->
