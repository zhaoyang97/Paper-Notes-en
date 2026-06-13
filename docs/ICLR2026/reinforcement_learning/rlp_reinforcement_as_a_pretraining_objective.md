---
title: >-
  [Paper Note] RLP: Reinforcement as a Pretraining Objective
description: >-
  [ICLR 2026][Reinforcement Learning][Pretraining] This paper proposes RLP (Reinforcement Learning Pretraining), an information-gain-driven RL pretraining objective that rewards Chain-of-Thought (CoT) reasoning when it imp…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Pretraining"
  - "Information Gain"
  - "Chain-of-Thought"
  - "Next-Token Prediction"
date: 2026-05-08
content_hash: 4de4445d207c2769
---

# RLP: Reinforcement as a Pretraining Objective

**Conference**: ICLR 2026
**arXiv**: [2510.01265](https://arxiv.org/abs/2510.01265)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Pretraining, Information Gain, Chain-of-Thought, Reinforcement Learning, Next-Token Prediction

## TL;DR

This paper proposes RLP (Reinforcement Learning Pretraining), an information-gain-driven RL pretraining objective that rewards Chain-of-Thought (CoT) reasoning when it improves next-token prediction probability. RLP shifts reinforcement learning from the post-training stage into pretraining, enabling dense reward signals without any verifier.

## Background & Motivation

The standard LLM training pipeline follows "Pretraining (NTP) → SFT → RLHF/RLVR," where reinforcement learning appears only at the final stage and relies on task-specific verifiers or human feedback. However, human text comprehension is not a linear token-by-token process; rather, it involves parallel integration of input with prior knowledge. Standard NTP pretraining lacks such a mechanism, limiting the model's ability to reason and ground knowledge during learning.

**Core Problem**: Can the exploratory spirit of RL (exploratory CoT generation) be introduced into the pretraining stage?

The core idea of this paper is to treat CoT as an "action": before predicting each next token, the model samples an internal thought, and the reward signal reflects the degree to which that thought improves prediction accuracy (information gain). This design requires no verifier and can be applied to general-domain text.

## Method

### Overall Architecture

RLP inserts a CoT sampling step at each position $t$ of NTP. The model first generates a thought $c_t$ from the context $x_{<t}$, then predicts $x_t$ conditioned on $(x_{<t}, c_t)$. The reward is the log-likelihood ratio relative to a "no-thinking" baseline.

### Key Designs

1. **Information Gain Reward**:

    - Function: Measures how much the CoT improves next-token prediction.
    - Mechanism: $r(c_t) = S_{\text{pred}}(c_t) - S_{\text{ema}}$, where $S_{\text{pred}}(c_t) = \log p_\theta(x_t|x_{<t},c_t)$ is the log probability with thinking, and $S_{\text{ema}} = \log \bar{p}_\phi(x_t|x_{<t})$ is the EMA teacher (no-thinking) baseline.
    - Design Motivation: The reward is positive when thinking genuinely improves prediction (Proposition 1: expected reward equals cross-entropy reduction), and provides a scalar signal at every position — eliminating the need for a value function or external verifier.

2. **EMA Teacher Baseline**:

    - Function: Provides a counterfactual "no-thinking" reference.
    - Mechanism: Teacher parameters $\phi \leftarrow \tau\phi + (1-\tau)\theta$, with $\tau=0.999$, initialized from the current model.
    - Design Motivation: A frozen baseline diverges too far and causes reward hacking; full synchronization collapses the log-likelihood ratio to zero. The EMA provides a one-step-delayed smoothed reference, balancing informativeness and training stability.

3. **Group-Relative Baseline and Clipped Surrogate**:

    - Function: Reduces variance and stabilizes training.
    - Mechanism: $G$ thoughts are sampled per position; a corrected inclusive-mean baseline $A^{(i)} = \frac{G}{G-1}(r(c_t^{(i)}) - \bar{r})$ is used. A PPO-style clipped surrogate loss $\mathcal{L}_{\text{clip}}$ is applied to thought tokens.
    - Design Motivation: The group-relative baseline eliminates the $(1-1/G)$ shrinkage bias of the inclusive mean; clipping prevents excessively large policy updates.

### Loss & Training

RLP **does not include a standard NTP loss** and optimizes solely the information gain objective: $\max_\theta J(\theta) = \mathbb{E}[r(c_t)]$. Gradients are applied only to thought tokens; gradients through $p_\theta$ and $\bar{p}_\phi$ in the reward computation are stop-gradiented. In practice, one token position per document is randomly selected for RLP application.

## Key Experimental Results

### Main Results (qwen3-1.7b-base, average over 8 benchmarks)

| Model | Math Avg. | Science Avg. | Overall Avg. |
|-------|-----------|--------------|--------------|
| $\mathcal{M}_{\text{base}}$ | 24.35 | 34.50 | 30.32 |
| $\mathcal{M}_{\text{CPT}}$ (Continued Pretraining) | 30.77 | 32.01 | 30.85 |
| $\mathcal{M}_{\text{RLP}}$ | **31.74** | **39.68** | **36.03** |
| $\mathcal{M}_{\text{base}}$+Post | 34.29 | 42.38 | 39.34 |
| $\mathcal{M}_{\text{CPT}}$+Post | 34.63 | 42.73 | 39.90 |
| $\mathcal{M}_{\text{RLP}}$+Post | **36.03** | **45.74** | **42.51** |

### Ablation Study (Nemotron-Nano-12B-v2 Scale)

| Configuration | Overall Avg. | Note |
|---------------|--------------|------|
| Base model | 42.81% | Strong baseline |
| +RLP | **61.32%** | +18.5 percentage points |
| Science reasoning gain | +23% | Generalizes beyond math |

### Key Findings

- RLP improves over the base model by 19% and over continued pretraining by 17%, confirming the gain originates from the method rather than additional compute.
- Post-training gains compound rather than wash out: RLP+Post outperforms CPT+Post by 7–8%.
- The largest gains appear on reasoning-intensive benchmarks such as AIME25 (5.02 vs. 3.96 vs. 2.25).
- Training on general web corpora remains effective — the approach is not limited to mathematical data.

## Highlights & Insights

- **Paradigm-level Innovation**: RLP shifts RL from post-training into pretraining, fundamentally altering the conventional "Pretraining → SFT → RL" pipeline.
- **Verifier-free, General-domain**: The reward is computed entirely from the model's own predictive capacity and can be applied to arbitrary text.
- **Theoretical Guarantees for Information Gain**: Propositions 1 and 2 establish formal connections between the reward signal, cross-entropy reduction, and marginalized thoughts.
- **Orthogonal Composability with Post-training**: The reasoning foundation established by RLP is not only preserved but amplified after SFT/RLVR.

## Limitations & Future Work

- Only one token position per document is currently used for RLP; the effect and cost of full-position application warrants exploration.
- The impact of CoT length on performance requires more systematic analysis.
- The EMA decay $\tau=0.999$ is fixed; adaptive scheduling may be more effective.
- Validation on non-English and non-STEM domains remains limited.

## Related Work & Insights

- RPT (Dong et al., 2025) also performs RL pretraining but uses sparse binary rewards and relies on a proxy model for filtering; RLP provides a continuous signal at every position.
- The key distinction from RLHF/RLVR is that RLP requires no external verifier or human annotation.
- Insight: Instilling "thinking habits" during pretraining may be more fundamental than teaching models to reason only at the post-training stage.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of RL pretraining and information gain reward carries paradigm-level significance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple model scales, data domains, post-training validation, and comparative ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory, method, and experiments are tightly integrated.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction of RL pretraining with broad potential impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](../../NeurIPS2025/reinforcement_learning/multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[ICML 2026\] Probing RLVR Training Instability through the Lens of Objective-Level Hacking](../../ICML2026/reinforcement_learning/probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)

</div>

<!-- RELATED:END -->
