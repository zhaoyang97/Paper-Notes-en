---
title: >-
  [Paper Note] RLP: Reinforcement as a Pretraining Objective
description: >-
  [ICLR 2026][Reinforcement Learning][Pretraining] Ours proposes RLP (Reinforcement Learning Pretraining), an information gain-driven RL pretraining objective. By rewarding Chain-of-Thought (CoT) that increases the probability of next-token prediction, it shifts RL from the post-training stage to the pretraining stage, achieving dense reward signals without a verifier.
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Pretraining"
  - "Information Gain"
  - "Chain-of-Thought"
  - "Next-Token Prediction"
date: 2026-05-08
content_hash: 1272061817ccec22
---

# RLP: Reinforcement as a Pretraining Objective

**Conference**: ICLR 2026  
**arXiv**: [2510.01265](https://arxiv.org/abs/2510.01265)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Pretraining, Information Gain, Chain-of-Thought, Reinforcement Learning, Next-Token Prediction

## TL;DR

Ours proposes RLP (Reinforcement Learning Pretraining), an information gain-driven RL pretraining objective. By rewarding Chain-of-Thought (CoT) that increases the probability of next-token prediction, it shifts RL from the post-training stage to the pretraining stage, achieving dense reward signals without a verifier.

## Background & Motivation

The standard training pipeline for current LLMs is "Pretraining (NTP) $\to$ SFT $\to$ RLHF/RLVR," where reinforcement learning only appears in the final stage and relies on task-specific verifiers or human feedback. However, human understanding of text is not processed linearly token-by-token; instead, it integrates input with prior knowledge in parallel. Standard NTP pretraining lacks this mechanism, limiting the model's ability to perform reasoning and knowledge grounding during the learning process.

Core Problem: **Can the exploratory spirit of RL (exploratory CoT generation) be brought into the pretraining stage?**

The Core Idea of this paper is to treat CoT as an "action": before predicting each next token, the model first samples an internal thought segment. The reward signal is the degree to which this thought improves the prediction accuracy (information gain). This design requires no verifier and can be trained on general text.

## Method

### Overall Architecture

RLP embeds the "think before acting" spirit of reinforcement learning into ordinary next-token prediction. In standard NTP, the model predicts $x_t$ directly upon seeing the context $x_{<t}$; RLP instead makes the model sample an internal thought $c_t$ (equivalent to an "action") from $x_{<t}$ at each position $t$, and then predicts $x_t$ based on $(x_{<t}, c_t)$. The quality of this thought is not scored by an external verifier but by how much it increases the prediction probability of $x_t$—the more it improves over a "non-thinking" reference baseline, the higher the reward. Thus, a dense scalar reward can be generated at every position of an entire general text sequence, allowing RL to run on unlabeled corpora during the pretraining stage. The entire process consists of three parts: first, use "Information Gain Reward" to quantify the value of thinking into a scalar; the "non-thinking" reference in the reward is provided by the "EMA Teacher Baseline"; finally, use "Group Relative Baseline & Clipping Proxy" to transform this high-variance signal into stable policy updates.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    X["Input Sequence<br/>(General text, unlabeled)"] --> S["Sample G Chain-of-Thought (CoT)<br/>(Exploratory actions)"]
    X --> EMA["EMA Teacher Baseline<br/>Non-thinking prediction, yields S_ema"]
    S --> P["Thinking-augmented prediction<br/>p_θ(x_t | x, c_t), yields S_pred"]
    P --> R["Information Gain Reward<br/>r = S_pred − S_ema"]
    EMA --> R
    R --> A["Group Relative Baseline & Clipping Proxy<br/>In-group advantage A + Clipping loss"]
    A --> U["Gradient backprop to thought tokens only<br/>Update policy θ"]
    U -.-.|"φ ← τφ + (1−τ)θ Moving update"| EMA
```

### Key Designs

**1. Information Gain Reward: Quantifying "Utility of Thinking" into an Optimizable Scalar**

To move RL to pretraining, the first hurdle is the lack of a verifier to tell the model if the answer is correct. RLP bypasses this by directly taking the "change in prediction probability before and after thinking" as the reward: $r(c_t) = S_{\text{pred}}(c_t) - S_{\text{ema}}$, where $S_{\text{pred}}(c_t) = \log p_\theta(x_t \mid x_{<t}, c_t)$ is the log-probability of the true next token with thinking, and $S_{\text{ema}} = \log \bar{p}_\phi(x_t \mid x_{<t})$ is the log-probability given by a non-thinking baseline. The log-likelihood ratio is positive if and only if thinking truly makes the prediction more accurate. Proposition 1 in the paper further explains that the expectation of this reward equals the reduction in cross-entropy. Therefore, optimizing it is equivalent to letting the model learn that "correct thinking leads to more accurate prediction"—and this can be calculated at every position without learning a value function or using any external scoring.

**2. EMA Teacher Baseline: Providing a Stable and Non-degenerate Reference Point for Rewards**

The level of the reward depends entirely on what is used as the "non-thinking" reference $\bar{p}_\phi$. If a frozen old model is used as the baseline, the current model will deviate further from it as training progresses, causing rewards to be artificially inflated and inducing reward hacking. Conversely, if the baseline is perfectly synchronized with the current model, the probabilities on both sides will be the same, the log-likelihood ratio will drop to zero, and the reward signal will disappear. RLP uses an Exponential Moving Average (EMA) teacher as a compromise: $\phi \leftarrow \tau\phi + (1-\tau)\theta$, with $\tau=0.999$, initialized to the current model. This is equivalent to letting the baseline follow the policy smoothly with a one-step delay, retaining enough information without being misled by instantaneous policy fluctuations, thus balancing reward effectiveness and training stability.

**3. Group Relative Baseline & Clipping Proxy: Variance Reduction and Stable Updates**

The reward noise for a single thought sample is large, and direct optimization leads to high variance. RLP samples $G$ thoughts at each position and uses the mean in-group reward $\bar{r}$ as a baseline to calculate the advantage. However, the naive inclusive mean introduces a shrinkage bias of $(1-1/G)$, so a corrected form $A^{(i)} = \frac{G}{G-1}\big(r(c_t^{(i)}) - \bar{r}\big)$ is used to eliminate this bias. Furthermore, a PPO-style clipping proxy loss $\mathcal{L}_{\text{clip}}$ is applied to the thought tokens to limit the step size of policy updates, avoiding large gradients that could destabilize the policy—essentially migrating mature GRPO/PPO stabilization techniques to token-level thought optimization.

### Loss & Training

RLP **does not stack standard NTP loss** but only optimizes the information gain objective itself: $\max_\theta J(\theta) = \mathbb{E}[r(c_t)]$. Gradients are only backpropagated to thought tokens; both $p_\theta$ and $\bar{p}_\phi$ in the reward calculation use stop-gradient to prevent the model from "cheating" the reward by changing the prediction distribution instead of actually improving the thinking. To control costs in engineering, RLP is applied to only one randomly selected token position per document, while other positions follow regular prediction.

## Key Experimental Results

### Main Results (qwen3-1.7b-base, average of 8 benchmarks)

| Model | Math Avg | Science Avg | Total Avg |
|------|---------|---------|--------|
| $\mathcal{M}_{\text{base}}$ | 24.35 | 34.50 | 30.32 |
| $\mathcal{M}_{\text{CPT}}$ (Continual Pretraining) | 30.77 | 32.01 | 30.85 |
| $\mathcal{M}_{\text{RLP}}$ | **31.74** | **39.68** | **36.03** |
| $\mathcal{M}_{\text{base}}$+Post | 34.29 | 42.38 | 39.34 |
| $\mathcal{M}_{\text{CPT}}$+Post | 34.63 | 42.73 | 39.90 |
| $\mathcal{M}_{\text{RLP}}$+Post | **36.03** | **45.74** | **42.51** |

### Ablation Study (Nemotron-Nano-12B-v2 Extension)

| Configuration | Total Avg | Note |
|------|--------|------|
| Base model | 42.81% | Strong baseline |
| +RLP | **61.32%** | +18.5 percentage points |
| Science reasoning improvement | +23% | Generalizes to non-math domains |

### Key Findings
- RLP improves by 19% relative to the base model and 17% relative to continual pretraining, confirming that the Gain comes from the method rather than computation.
- Post-training gains are not washed out but compounded: RLP+Post is 7-8% higher than CPT+Post.
- The greatest benefits are seen in reasoning-intensive benchmarks like AIME25 (5.02 vs 3.96 vs 2.25).
- Training on general web corpora is also effective—not limited to mathematical data.

## Highlights & Insights

- **Paradigm Innovation**: Shifting RL from post-training to pretraining changes the fixed "Pretraining $\to$ SFT $\to$ RL" pipeline.
- **Verifier-free, General Text**: Rewards are calculated entirely from the model's own predictive capability, applicable to any text.
- **Theoretical Guarantees for Information Gain**: Propositions 1 and 2 establish the relationship between reward, cross-entropy reduction, and marginalized thinking.
- **Orthogonal to Post-training**: The reasoning foundation established by RLP is not only maintained but amplified after SFT/RLVR.

## Limitations & Future Work

- RLP is applied to only 1 position per document; the effect and cost of applying it to all positions deserve exploration.
- The impact of CoT length on performance needs more systematic analysis.
- The current EMA decay $\tau=0.999$ is a fixed value; adaptive adjustment might be superior.
- More validation is needed in non-English and non-STEM domains.

## Related Work & Insights

- RPT (Dong et al., 2025) also performs RL pretraining but uses sparse binary rewards and relies on proxy models for filtering; RLP provides continuous signals at every position.
- Key difference from RLHF/RLVR: RLP does not require any external verifier or human annotation.
- Insight: Injecting "thinking habits" during the pretraining phase might be more fundamental than teaching the model to reason only during the post-training phase.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The design of RL pretraining with information gain rewards is paradigmatic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes multiple model scales, multiple data domains, post-training validation, and comparative ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory-Method-Experiment sections are closely linked.
- Value: ⭐⭐⭐⭐⭐ Opens the new direction of RL pretraining with broad potential impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning](a_reward-free_viewpoint_on_multi-objective_reinforcement_learning.md)
- [\[ICLR 2026\] Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels](webscale-rl_automated_data_pipeline_for_scaling_rl_data_to_pretraining_levels.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations](dual-objective_reinforcement_learning_with_novel_hamilton-jacobi-bellman_formula.md)
- [\[ICLR 2026\] AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)

</div>

<!-- RELATED:END -->
