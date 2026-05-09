---
title: >-
  [Paper Note] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification
description: >-
  [ICLR 2026][Reinforcement Learning][SFT generalization] This paper mathematically proves, from an RL policy gradient perspective, that SFT gradients implicitly encode a pathological reward structure with inverse probability weighting ($1/\pi_\theta$), causing excessively large gradients on low-probability tokens and limiting generalization. The paper proposes DFT (Dynamic Fine-Tuning), which requires only a one-line code modification (multiplying the CE loss by the token probability: $-p\log p$) to eliminate inverse probability weighting. DFT substantially outperforms SFT on mathematical reasoning, code generation, and multimodal tasks, and even surpasses GRPO/PPO in the offline RL setting.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - SFT generalization
  - policy gradient
  - inverse probability weighting
  - Dynamic Fine-Tuning
  - reward rectification
date: 2026-05-08
content_hash: 5271ac624a4bcdca
---

# On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**Conference**: ICLR 2026
**arXiv**: [2508.05629](https://arxiv.org/abs/2508.05629)
**Code**: [GitHub](https://github.com/yongliang-wu/DFT)
**Area**: Reinforcement Learning
**Keywords**: SFT generalization, policy gradient, inverse probability weighting, Dynamic Fine-Tuning, reward rectification

## TL;DR
This paper mathematically proves, from an RL policy gradient perspective, that SFT gradients implicitly encode a pathological reward structure with inverse probability weighting ($1/\pi_\theta$), causing excessively large gradients on low-probability tokens and limiting generalization. The paper proposes DFT (Dynamic Fine-Tuning), which requires only a one-line code modification (multiplying the CE loss by the token probability: $-p\log p$) to eliminate inverse probability weighting. DFT substantially outperforms SFT on mathematical reasoning, code generation, and multimodal tasks, and even surpasses GRPO/PPO in the offline RL setting.

## Background & Motivation

**Background**: SFT is the standard paradigm for LLM post-training, efficiently acquiring expert-like behavior; RL explores diverse strategies via reward signals to achieve better generalization, but requires substantial computation, careful hyperparameter tuning, and explicit reward functions.

**Limitations of Prior Work**:
- (1) SFT causes performance degradation on challenging benchmarks (OlympiadBench/AIME24): Qwen2.5-Math-1.5B drops from 15.88 to 12.63 on OlympiadBench after SFT.
- (2) The phenomenon of "SFT memorizes, RL generalizes" has been empirically observed across multiple works (Chu et al., 2024), but lacks a mathematical root-cause analysis.
- (3) Hybrid SFT+RL methods (InstructGPT/DPO, etc.) do not improve SFT itself — when only positive samples are available without reward or preference data, SFT is the only option.

**Key Challenge**: Why does SFT generalize poorly with only positive samples? Can the theoretical gap between SFT and RL be rigorously characterized?

**Key Insight**: By rewriting SFT gradients strictly in the form of policy gradients via importance sampling, the paper reveals that SFT's implicit "reward" is a sparse indicator function with inverse probability weighting — the mathematical root cause of limited generalization.

**Intuitive Analogy**: Focal Loss $-(1-p)^\gamma\log p$ emphasizes hard samples (in the underfitting era) → DFT $-p\log p$ de-emphasizes hard samples (in the LLM overfitting era) → a fundamental inversion of the objective design philosophy.

**Practical Motivation**: Without a reward model, preference data, or online sampling (using only positive samples), can SFT be made to approach RL-level performance?

## Method

### Core Theory: SFT Gradient = Policy Gradient with Inverse Probability Weighting

The standard SFT gradient is transformed into an on-policy policy gradient form via importance sampling:

$$\nabla\mathcal{L}_{SFT} = -\mathbb{E}_{y\sim\pi_\theta}\left[\frac{\mathbf{1}[y=y^*]}{\pi_\theta(y|x)} \nabla\log\pi_\theta(y|x)\right]$$

Compared to the standard policy gradient $\nabla J = \mathbb{E}[\nabla\log\pi_\theta \cdot r(x,y)]$, the implicit reward in SFT is:
- **Reward function** $r(x,y)=\mathbf{1}[y=y^*]$: nonzero only for exact matches → **extremely sparse**
- **Importance weight** $w=1/\pi_\theta(y|x)$: lower model probability → larger weight → **gradient explosion / instability**

These two factors jointly cause SFT optimization to over-focus on low-probability exact-match samples, leading to overfitting rather than generalization.

### DFT (Dynamic Fine-Tuning)

**Rectification Strategy**: Multiplying the reward by $1/w = \pi_\theta$ to neutralize the inverse probability weight yields the token-level DFT loss:

$$\mathcal{L}_{DFT} = -\sum_{t=1}^{|y^*|} \text{sg}\big(\pi_\theta(y_t^*|y_{<t}^*,x)\big) \log\pi_\theta(y_t^*|y_{<t}^*,x)$$

where $\text{sg}(\cdot)$ denotes the stop-gradient operator. This is equivalent to multiplying the standard cross-entropy by the token probability — **a one-line code change**.

### Key Design Decisions

1. **Token-level vs. Sentence-level Weighting**: The sentence-level probability $\pi(y|x)=\prod_t \pi(y_t)$ is extremely small, leading to numerical instability and uninformative loss; geometric mean variants also show limited effectiveness. Token-level weighting improves performance from 15.92 to 31.58 (vs. 15.75 for sentence-level).

2. **Stop-gradient**: Gradients do not flow through the weighting term → DFT retains the standard SFT implementation form → no additional sampling, reward model, or reference model is required.

3. **Rectified Reward = 1**: DFT is equivalent to assigning a uniform reward of 1 to all expert trajectories, analogous to RLVR assigning uniform reward to all correct samples → avoids over-concentration on low-probability tokens.

## Key Experimental Results

### Table 1: Mathematical Reasoning Main Results (Avg@16, NuminaMath-CoT 100K samples)

| Model | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|-------|---------|---------|---------------|--------|-------|-----|
| Qwen2.5-Math-1.5B (base) | 31.66 | 8.51 | 15.88 | 4.16 | 19.38 | 15.92 |
| + SFT | 43.76 | 13.04 | 12.63↓ | 1.87↓ | 18.75↓ | 18.01 |
| + **DFT** | **64.89** | **20.94** | **27.08** | **6.87** | **38.13** | **31.58** |
| Qwen2.5-Math-7B (base) | 40.12 | 14.39 | 17.12 | 6.68 | 27.96 | 21.25 |
| + SFT | 53.96 | 16.66 | 18.93 | 2.48↓ | 26.09↓ | 23.62 |
| + **DFT** | **68.20** | **30.16** | **33.83** | **8.56** | **45.00** | **37.15** |

DFT achieves an average gain of +15.66 on Qwen2.5-Math-1.5B, **5.9×** that of SFT (+2.09). SFT degrades on AIME24/OlympiadBench, while DFT consistently improves across all benchmarks.

### Table 2: Offline RL Comparison (Qwen2.5-Math-1.5B)

| Method | Setting | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|--------|---------|---------|---------|---------------|--------|-------|-----|
| DPO | Offline | 46.89 | 11.53 | 22.86 | 4.58 | 30.16 | 23.20 |
| RFT | Offline | 48.23 | 14.19 | 22.29 | 4.37 | 30.78 | 23.97 |
| PPO | Online | 56.10 | 15.41 | 26.33 | 7.50 | 37.97 | 28.66 |
| GRPO | Online | 62.86 | 18.93 | 28.62 | 8.34 | 41.25 | 32.00 |
| **DFT** | **Offline** | **64.71** | **25.16** | **30.93** | **7.93** | **48.44** | **35.43** |

As an offline SFT method, DFT surpasses all offline RL baselines (+11.46 vs. RFT) and even outperforms online GRPO (+3.43 avg).

### Table 3: Code Generation (UltraFeedback 10K)

| Model | HumanEval | HE+ | MultiPL-E Avg |
|-------|-----------|-----|---------------|
| Qwen2.5-Coder-7B base | 62.2 | 53.0 | 57.76 |
| + SFT | 54.9↓ | 48.8↓ | 57.62 |
| + **DFT** | **67.7** | **59.8** | **62.30** |

SFT degrades across all metrics on Qwen2.5-Coder-7B, while DFT consistently improves performance.

## Key Findings

1. **Mathematical Explanation of SFT Degradation**: SFT degrades on difficult benchmarks because the implicit $1/\pi_\theta$ inverse probability weighting biases gradients toward low-probability tokens, causing overfitting to exact matches in the training set at the expense of generalization.

2. **Token Probability Polarization Effect**: After DFT training, token probabilities exhibit a bimodal distribution — the probabilities of semantically critical high-probability tokens are further increased, while those of low-probability connectives and punctuation ('the', 'let', ',', '.') are suppressed → the model learns to distinguish semantic from syntactic tokens.

3. **Significantly Faster Convergence**: DFT reaches its peak performance within the first 120 steps, already surpassing SFT's final accuracy within 10–20 steps → gradient updates carry more information → avoids the optimization stagnation observed in SFT.

4. **Consistent Gains Across Tasks and Models**: Consistent improvements are observed across 5 base models (LLaMA/DeepSeek/Qwen) × 4 task types (math/code/multimodal/RL), demonstrating strong robustness.

5. **Failure on Factual Knowledge Tasks**: On the Natural Questions dataset, DFT (30.14%) underperforms SFT (36.62%) — DFT reinforces the model's existing beliefs, which impedes learning when the model lacks domain knowledge.

## Highlights & Insights
- **"SFT = RL with pathological rewards"** is a profound theoretical insight: rather than an empirical observation, it is a mathematically precise derivation that reveals the root cause of the "SFT memorizes, RL generalizes" phenomenon.
- **The power of a one-line change**: replacing $-\log p$ with $-p\log p$ is remarkably concise yet highly effective — combining aesthetic elegance with practical value.
- **No additional resources required**: no reward model, preference data, reference model, large batch size, or online sampling is needed → the optimal choice under resource-constrained settings.
- **Focal Loss inversion reflects a paradigm shift**: the CV era emphasized hard samples (underfitting) → the LLM era de-emphasizes hard samples (overfitting) → objective function design philosophy warrants fundamental reconsideration.

## Limitations & Future Work
- **Limited factual knowledge learning**: DFT relies on the model's own confidence for weighting → when the model lacks prior knowledge, it reinforces incorrect beliefs (performance drops 1%+ on NQ).
- **Disadvantageous for hard samples / low-resource domains**: DFT suppresses the weights of samples with low initial probability → potentially discarding rare but important training signals.
- **Limited evaluation scope**: validation on larger-scale LLMs (70B+) and additional task types (dialogue/summarization, etc.) has not been conducted.
- **No quality awareness**: a uniform reward of 1 is assigned to all positive samples → differences in example quality are not exploited → non-uniform reward allocation is a promising direction for future work.

## Related Work & Insights

| Comparison Method | Key Difference |
|-------------------|---------------|
| **DPO** (Rafailov 2023) | DPO requires preference pairs (positive/negative samples) to optimize the policy via implicit rewards → DFT requires only positive samples with a one-line loss modification; DFT outperforms DPO by +12.23 avg in the offline RL setting. |
| **GRPO** (DeepSeek 2024) | GRPO is online RL requiring sampling multiple responses and reward verification → DFT requires no online sampling or reward model; DFT surpasses GRPO by +3.43 avg in the offline setting at comparable scale. |
| **iw-SFT** (Qin & Springenberg 2025) | iw-SFT introduces importance weighting based on the data-generating policy → requires estimating the unknown $\pi_b$; DFT directly uses current model probabilities as weights → simpler and free of additional assumptions. |
| **Focal Loss** (Lin 2017) | Focal Loss emphasizes hard samples $-(1-p)^\gamma\log p$ → DFT inversely de-emphasizes hard samples $-p\log p$ → more appropriate under LLM overfitting scenarios. |

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified SFT-RL theory with a profoundly simple one-line code modification
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 5 math benchmarks + code + multimodal + offline RL + ablation
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and practice are seamlessly integrated with a clear and coherent narrative
- Value: ⭐⭐⭐⭐⭐ Direct and significant impact on LLM SFT training practice with extremely low implementation overhead

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[ICLR 2026\] How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)
- [\[ICLR 2026\] ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)

</div>

<!-- RELATED:END -->
