---
title: >-
  [Paper Note] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models
description: >-
  [NeurIPS 2025][LLM Pretraining][temporal credit assignment] This paper proposes RICL (Retrospective In-Context Learning), which leverages the pre-trained knowledge of LLMs to convert sparse environmental feedback into dense advantage function signals via retrospective in-context learning, achieving up to 100× sample efficiency over conventional Monte Carlo methods. Building on this, the paper further introduces RICOL, an online learning framework.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - temporal credit assignment
  - in-context learning
  - advantage function
  - LLM policy
  - online learning
date: 2026-05-08
content_hash: 018c55029d6bda76
---

# Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2602.17497](https://arxiv.org/abs/2602.17497)
**Code**: None
**Area**: LLM Pre-training / Reinforcement Learning
**Keywords**: temporal credit assignment, in-context learning, advantage function, LLM policy, online learning

## TL;DR

This paper proposes RICL (Retrospective In-Context Learning), which leverages the pre-trained knowledge of LLMs to convert sparse environmental feedback into dense advantage function signals via retrospective in-context learning, achieving up to 100× sample efficiency over conventional Monte Carlo methods. Building on this, the paper further introduces RICOL, an online learning framework.

## Background & Motivation

### State of the Field

**Background**: Online learning for LLM agents faces the challenge of **sparse rewards** — in multi-step tasks, consecutive correct actions are required before any reward signal is received.

### Limitations of Prior Work

**Limitations of Prior Work**: Temporal credit assignment aims to decompose sparse feedback into dense per-step training signals.

### Root Cause

**Key Challenge**: Traditional RL approaches must learn task-specific value functions for credit assignment, resulting in low sample efficiency and poor generalization.

### Resolution Direction

**Resolution Direction**: Existing LLM self-correction methods (e.g., Reflexion) operate on trajectory-level feedback, which is coarse-grained and assumes feedback can transfer across trajectories.

### Additional Notes

**Additional Notes**: Core insight: LLM pre-trained knowledge + retrospective in-context updates → accurate estimation of the advantage function.

## Method

### Overall Architecture

The RICL credit assignment pipeline (for each state $s_t$):

1. **Trajectory Collection**: Sample $n$ trajectories from $s_t$ using the current policy $\pi_0$.
2. **Retrospective Reflection**: For each trajectory, feed the subsequent $(s_{t:T}, a_{t:T-1}, r_{t:T-1})$ into a reflector LLM to generate textual feedback $f_t$.
3. **In-Context Update**: Incorporate feedback $f_t$ into the prompt to obtain the updated policy $\pi'(a|s_t) = \pi_0(a|s_t, f_t)$.
4. **Advantage Estimation**: $\bar{A}_r^{\pi_0}(s,a) = \frac{\beta}{n}\sum_{i=1}^n (\log\frac{\pi'^{(i)}(a|s)}{\pi_0(a|s)} + \log Z^{(i)}(s))$

RICOL embeds RICL into an iterative online learning loop: RICL → advantage-weighted regression → parameter update.

### Key Designs

1. **Theorem-Grounded Advantage Inference**:
    - Function: Infers the advantage function from the log-probability ratio before and after policy update.
    - Mechanism: Theorem 4.1 proves that for any two policies $\pi_0$ and $\pi'$, there exists a reward function such that $\beta \log\frac{\pi'(a|s)}{\pi_0(a|s)} \propto A_r^{\pi_0}(s,a)$.
    - Design Motivation: Inverts the KL-regularized policy update — if in-context learning implicitly performs policy improvement, its log-ratio encodes advantage information.

2. **Retrospective Design Reduces Uncertainty**:
    - Function: Feedback is used only to update states within the same trajectory that generated it, rather than new trajectories.
    - Mechanism: Standard ICL assumes that experience from one trajectory transfers to new states, which is unreliable for LLMs; RICL updates only states that have already been visited.
    - Design Motivation: Reduces uncertainty in LLM in-context learning; RICL achieves 7.2% higher accuracy than ICL.

### Loss & Training

RICOL policy improvement objective (trust-region-constrained advantage-weighted regression):

$$\min_\pi \mathbb{E}_{s \sim d_{\pi_0}}\left[D_{KL}\left(\frac{1}{Z(s)} \odot \exp((1-\alpha)\log\pi_0 + \alpha\log\pi') \| \pi\right)\right]$$

- $\alpha$ controls the trust region size to prevent overfitting to noisy feedback.
- Policy model: LLaMA-3.2-3B-Instruct
- Reflector: GPT-4o mini
- Discrete action space enables exact computation of KL divergence.

## Key Experimental Results

### Main Results

| Method | goto Success | pickup Success | pick_up_seq Success | open Success |
|--------|-------------|---------------|---------------------|-------------|
| GPT-4o mini (zero-shot) | ~35% | ~15% | ~10% | ~20% |
| Reflexion | ~40% | ~20% | ~10% | ~25% |
| PPO (3B) | ~55% @ 20k steps | ~40% @ 20k steps | ~20% @ 20k steps | ~40% @ 20k steps |
| PPO (10M param MLP) | ~60% @ 100k steps | ~45% @ 100k steps | ~25% @ 100k steps | ~45% @ 100k steps |
| **RICOL** | **~55% @ 2k steps** | **~40% @ 2k steps** | **~20% @ 2k steps** | **~40% @ 2k steps** |

### Ablation Study

- RICL vs. Monte Carlo credit assignment: RICL achieves with 10 trajectories the accuracy that MC requires 1,000 trajectories (100× sample efficiency).
- RICL vs. ICL: On BabyAI goto, RICL achieves 7.2% higher accuracy in predicting expert actions.
- RICOL vs. RWR (no credit assignment): RWR is effective only on tasks with high initial success rates and degrades under sparse rewards.
- Noise robustness: RICOL remains effective even when feedback accuracy drops to 70%.

### Key Findings

- RICOL achieves approximately 10× better sample efficiency than PPO (3B) and ~50× over PPO (MLP).
- In-context learning implicitly performs KL-regularized policy updates.
- RICOL outperforms Reflexion because RICL provides state-level rather than trajectory-level feedback; Reflexion's gains saturate quickly.
- A 3B model can surpass GPT-4o mini zero-shot performance through interactive learning.

## Highlights & Insights

- **LLM Pre-trained Knowledge → Value Estimation**: This work is the first to demonstrate that LLMs can accurately estimate advantage functions via in-context learning, without training a value network.
- The retrospective design elegantly addresses the unreliability of in-context learning.
- Theoretically grounded: Theorem 4.1 provides a principled justification for the method.
- Well-suited for settings with limited simulation budgets (1k–10k steps).

## Limitations & Future Work

- Supports only discrete, finite action spaces (requires enumerating all actions to compute the normalization term $Z$).
- BabyAI tasks are relatively simple; effectiveness on more complex reasoning tasks remains unknown.
- Requires an additional reflector LLM (GPT-4o mini), increasing inference cost.
- Not evaluated on token-level MDPs or reasoning tasks.

## Related Work & Insights

- Reflexion applies trajectory-level in-context learning; RICL performs retrospective updates at the state level, offering finer granularity.
- RICO-GRPO estimates advantages using trajectory-level rewards but does not perform explicit credit assignment.
- AWR (Advantage-Weighted Regression) serves as the foundational method for the policy improvement stage.

## Rating

- Theoretical Innovation: ⭐⭐⭐⭐⭐
- Experimental Validation: ⭐⭐⭐⭐
- Practical Value: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)

</div>

<!-- RELATED:END -->
