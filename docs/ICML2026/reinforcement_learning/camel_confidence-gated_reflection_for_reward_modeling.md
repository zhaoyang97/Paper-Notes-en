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
content_hash: 5ba61a89522edb16
---

# CAMEL: Confidence-Gated Reflection for Reward Modeling

**Conference**: ICML 2026  
**arXiv**: [2602.20670](https://arxiv.org/abs/2602.20670)  
**Code**: Not yet released  
**Area**: Alignment RLHF / Reward Modeling / LLM Reasoning  
**Keywords**: Reward Model, Confidence Gating, Reflection Mechanism, GRPO, Counterfactual Prefix Augmentation

## TL;DR
This paper observes that the log-probability margin of the verdict token is highly correlated with judgment accuracy. Based on this, CAMEL is proposed: it first makes a quick preference judgment using a single token, only triggering reflective generation when confidence is low. Counterfactual prefix augmentation is used to enhance GRPO training for self-correction. On three reward model benchmarks, a 14B parameter model achieves an average accuracy of 82.9% (surpassing the previous best 70B model by 3.2%).

## Background & Motivation
**Background**: Reward models used as alignment signals for LLMs mainly follow two paradigms. Scalar discriminators (e.g., Skywork-Reward, ArmoRM) are stable to train and fast at inference, but only output a score and lack interpretability. Generative judges (e.g., J1, RM-R1) generate reasoning before making a judgment, achieving higher accuracy, but require generating hundreds to thousands of tokens per sample.

**Limitations of Prior Work**: The cost of generative RMs is prohibitive for industrial deployment—RM-R1-DeepSeek-32B generates about 900 tokens per sample on RewardBench and about 1100 tokens on RM-Bench. Many samples are "one good, one bad reply" and do not require lengthy reasoning. Treating both easy and hard samples equally with full reasoning wastes computational resources.

**Key Challenge**: There is a clear "efficiency vs. expressiveness" trade-off in reward modeling. Simple samples should be handled quickly like scalar models, while difficult ones require reflection like generative models. However, there is currently no suitable signal to tell the model "how hard is this sample, and should it reflect?"

**Goal**: (1) Find a "difficulty" proxy that requires no extra inference; (2) Use it to build an adaptive routing reward model, incurring generation cost only for truly difficult samples; (3) Ensure that reflection genuinely enables self-correction rather than echoing the initial answer.

**Key Insight**: The authors observe that when the prompt asks the model to choose between A/B, the log-probability margin at the verdict token ($c(x) = |\log P(A|x) - \log P(B|x)|$) naturally captures the model's "certainty." Statistics on Skywork-80K with Qwen3-14B show that higher confidence samples have higher prediction accuracy (monotonically increasing), and errors are almost entirely concentrated in low-margin regions.

**Core Idea**: Use the single-token margin as a "zero-cost difficulty estimator" to construct a two-stage process of "quick judgment → reflection only at low confidence," and use counterfactual prefix RL training to enable genuine self-correction during reflection.

## Method

### Overall Architecture
CAMEL decomposes reward modeling into two stages. Given $(q, r_a, r_b)$, the model first outputs an initial verdict $v_0 \in \{\texttt{A}, \texttt{B}\}$. The confidence $c(x)$ is computed from the two candidate probabilities at the verdict token. If $c(x) \geq \tau$ (high confidence), terminate immediately, $v_1 = v_0$, requiring only 1 generated token. If $c(x) < \tau$, the prompt triggers a brief reflection $J$ ("think again..."), and then outputs the final verdict $v_1$. This "score first, then decide whether to explain" structure is trained with GRPO and counterfactual prefix augmentation.

### Key Designs

1. **Confidence Score as Difficulty Estimator**:

    - **Function**: Uses the single-token margin instead of explicit difficulty labels to decide whether to trigger reflection.
    - **Mechanism**: Defines $c(x) = |\log P_\theta(v=\texttt{A}|x) - \log P_\theta(v=\texttt{B}|x)|$, representing the "potential difference" of the model's preference. The authors plot $c(x)$ vs. accuracy on the training distribution and find a strong monotonic relationship—tuning a single threshold $\tau$ allows free movement along the accuracy/cost curve.
    - **Design Motivation**: No need to train an extra difficulty estimator or perform a second forward pass. This provides a high-quality routing signal at zero extra cost, which is key to the framework.

2. **Confidence-Gated Two-Stage Judging Prompt**:

    - **Function**: Reconstructs the traditional "long rationale → final verdict" into a factorized $v_0$ → optional $J$ → $v_1$ structure, allowing confidence gating to be inserted mid-generation.
    - **Mechanism**: The prompt forces the model to first output a verdict $v_0$ without explanation. During inference, $c(x)$ determines whether to proceed to the reflection segment. If reflection is triggered, the model, after seeing its own $v_0$, writes the rationale $J$ and produces $v_1$. The process allows obtaining $v_0$'s logit in a single forward pass, then deciding whether to continue decoding as needed.
    - **Design Motivation**: Externalizes the discrete decision of "whether to reflect" to the token probability level, rather than letting the model decide in language. This factorization allows easy/hard samples to follow different token paths but share the same policy.

3. **Counterfactual Prefix Augmentation + GRPO**:

    - **Function**: Trains the model to learn "when to stick to the initial judgment, when to be overturned by reflection."
    - **Mechanism**: For each sample $(x, z)$, create two copies: one forcing $v_0 = \texttt{A}$, the other $v_0 = \texttt{B}$. RL credit is applied only to the reflection $J$ and final $v_1$, treating $v_0$ as context and not optimizing it. The reward is binary $R = +1/-1$ (whether the final verdict matches ground truth), and GRPO optimizes $\max_\theta \mathbb{E}[R(v_1, z)] - \beta \mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$.
    - **Design Motivation**: Without counterfactual augmentation, the model easily learns the shortcut "reflection = repeat $v_0$" (since most initial judgments are correct). Forcing incorrect starting points compels the model to genuinely compare evidence and overturn itself during reflection, enabling emergent self-correction.

### Loss & Training
Two-stage training: first, supervised fine-tuning (SFT) on three preference datasets (Skywork-Reward-Preference-80K + Code-Preference-Pairs + Math-Step-DPO-10K) for Qwen3-14B to learn basic preference formats; then one epoch of GRPO with counterfactual prefix, with KL coefficient $\beta$ controlling deviation from the reference policy. At inference, $\tau = 5$ is used by default (tunable).

## Key Experimental Results

### Main Results
On three reward model benchmarks (RewardBench / RM-Bench / JudgeBench), compared with various scalar and generative RMs:

| Model | RewardBench | RM-Bench | JudgeBench | Avg |
|------|-------------|----------|------------|-----|
| INF-ORM-Llama3.1-70B (Prev. SOTA) | 95.1 | 73.8 | 70.2 | 79.7 |
| RM-R1-Qwen-Instruct-32B (Generative) | 89.0 | 73.1 | 64.8 | 75.6 |
| J1-Llama-70B | 93.3 | 82.7 | 60.0 | 78.7 |
| **CAMEL-Fast (14B, 1 token)** | 90.5 | 74.8 | 65.2 | 76.8 |
| **CAMEL-Reflection (14B, always)** | 92.8 | **84.2** | **71.6** | **82.9** |
| **CAMEL (gated, $\tau=5$)** | 92.4 | 81.9 | 69.1 | 81.1 |

CAMEL-Reflection outperforms the previous SOTA by 3.2% on average. CAMEL-Fast matches or exceeds the fully generative RM-R1-Qwen-Instruct-32B using only 1 token, and the 14B model matches/exceeds 70B baselines.

### Ablation Study

| Configuration | RewardBench | RM-Bench | JudgeBench | Avg |
|------|-------------|----------|------------|-----|
| Qwen3-14B (no tune) | 81.9 | 71.1 | 62.6 | 71.9 |
| Qwen3-14B + Reflection | 83.3 | 73.2 | 65.0 | 73.8 |
| Qwen3-14B-SFT | 90.6 | 72.7 | 64.8 | 76.0 |
| Qwen3-14B-GRPO (no counterfactual) | 91.2 | 83.5 | 62.9 | 79.2 |
| Qwen3-14B-GRPO + Reflection | 90.0 | 84.0 | 74.2 | 82.7 |
| **CAMEL (full)** | 92.4 | 81.9 | 69.1 | 81.1 |

### Key Findings
- Reflection brings the most gain on reasoning-intensive benchmarks: from always-fast to always-reflect, RewardBench +2.3%, RM-Bench +9.4%, JudgeBench +6.4%, indicating that the latter two benchmarks contain more "hard-to-judge-at-a-glance" samples.
- Counterfactual prefix is crucial: removing it causes GRPO+Reflection to drop 5 points on JudgeBench (74.2 → 69.1), showing that without counterfactuals, reflection degenerates into "repeating the initial judgment."
- Pareto frontier: On RewardBench and RM-Bench, CAMEL strictly outperforms RM-R1—RM-R1-DeepSeek-32B generates 900–1100 tokens on average to reach 87/74, while CAMEL approaches this with just 1 token, and surpasses it with fewer tokens by tuning $\tau$.
- After training, the confidence distribution shifts left (more conservative), consistent with the expectation that "the model learns to distinguish certainty from uncertainty." The self-correction confusion matrix shows reflection yields a net increase of 77 correct cases on RewardBench and 1233 on RM-Bench.

## Highlights & Insights
- **"Free Difficulty Signal"**: The single-token margin incurs almost no extra cost yet reliably predicts accuracy. This is a highly portable trick—any binary discrimination task (multiple choice QA, safety classification, tool selection) can directly reuse it for routing, abstention, or uncertainty estimation.
- **Explicitly Externalizing "Whether to Reflect"**: Previous chain-of-thought work often lets the model decide whether to think, resulting in either always thinking or never thinking. CAMEL makes a hard decision at the token probability level—clean, tunable, and regression-free.
- **Counterfactual Prefix as a Hidden Weapon for RL Training**: Many self-correction works struggle with "the model is reluctant to change its answer," essentially because $v_0$ is almost always correct in the training distribution. Forcing incorrect starting points is a general fix, transferable to self-refinement, self-debate, etc.
- Overall, reward modeling is restructured as "adaptive two-stage computation," which is more engineering-friendly than the scalar vs. generative dichotomy.

## Limitations & Future Work
- The threshold $\tau$ is globally fixed, but confidence distributions differ across tasks/domains—safety tasks tend to have higher confidence, math tasks lower. Ideally, $\tau$ should be dynamic or bucketed.
- Only validated on Qwen3-14B; scaling law (whether "low confidence = hard sample" still holds for 70B/100B+ models) is unclear. For small models, the margin may be too noisy to be effective.
- Reflection token length is not strictly controlled; the authors do not report the average length of the reflection segment. If reflection is lengthy, the cost savings from gating may be partially offset.
- Future directions: (a) Learn $\tau$ as well; (b) Introduce multi-level reflection (short/long/very long) for finer-grained routing; (c) Embed this architecture into actor-critic style RLHF pipelines as the critic.

## Related Work & Insights
- **vs RM-R1 (Generative RM SOTA)**: RM-R1 uses distilled rubric + RL with verifiable reward, generating long rationales for each sample. CAMEL shares the same training data but adds gating, achieving higher accuracy with fewer tokens, and establishes a strictly better Pareto frontier.
- **vs Generative RM (J1, Critic-RM)**: J1 / Critic-RM emphasize explicit reasoning traces to improve judgment quality. CAMEL borrows the reflection mechanism but avoids "indiscriminate reasoning."
- **vs Self-Consistency / Self-Refine**: Those methods rely on multiple sampling and voting or self-correction. CAMEL uses the single forward margin to directly decide whether to elaborate, avoiding repeated sampling costs.
- **vs uncertainty-based abstention**: Traditional selective prediction uses confidence to decide whether to answer; CAMEL uses it to decide whether to "think again," representing another paradigm of conditional compute.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of single-token margin for difficulty estimation and counterfactual prefix is a refreshing new framework, though each component alone is not disruptive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three mainstream benchmarks + Pareto curve + ablation + self-correction analysis are comprehensive, but lacks multi-backbone validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic chain from motivation–observation–method–experiment is particularly clear, with formulas and figures well integrated; the core idea can be reproduced after reading.
- Value: ⭐⭐⭐⭐⭐ A deployment-friendly 14B reward model surpasses 70B baselines, directly usable in industrial RLHF pipelines, and the trick is transferable across tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](../../ICLR2026/reinforcement_learning/rm-r1_reward_modeling_as_reasoning.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[ICLR 2026\] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](../../ICLR2026/reinforcement_learning/stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](../../ICLR2026/reinforcement_learning/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
