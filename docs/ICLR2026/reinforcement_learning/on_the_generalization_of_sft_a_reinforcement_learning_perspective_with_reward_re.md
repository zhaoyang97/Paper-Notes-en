---
title: >-
  [Paper Note] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification
description: >-
  [ICLR 2026][Reinforcement Learning][SFT Generalization] This work mathematically proves from an RL policy gradient perspective that the SFT gradient implicitly encodes a pathological reward structure of inverse probability weighting ($1/\pi_\theta$). This causes excessively large gradients for low-probability tokens, which limits generalization. The authors propose DFT (Dynamic Fine-Tuning), which eliminates this weighting via a one-line code modification (multiplying CE loss…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "SFT Generalization"
  - "Policy Gradient"
  - "Inverse Probability Weighting"
  - "Dynamic Fine-Tuning"
  - "Reward Rectification"
date: 2026-05-08
content_hash: 6372ef8417d9ba80
---

# On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**Conference**: ICLR 2026  
**arXiv**: [2508.05629](https://arxiv.org/abs/2508.05629)  
**Code**: [GitHub](https://github.com/yongliang-wu/DFT)  
**Area**: Reinforcement Learning  
**Keywords**: SFT Generalization, Policy Gradient, Inverse Probability Weighting, Dynamic Fine-Tuning, Reward Rectification

## TL;DR
This work mathematically proves from an RL policy gradient perspective that the SFT gradient implicitly encodes a pathological reward structure of inverse probability weighting ($1/\pi_\theta$). This causes excessively large gradients for low-probability tokens, which limits generalization. The authors propose DFT (Dynamic Fine-Tuning), which eliminates this weighting via a one-line code modification (multiplying CE loss by the token probability: $-p\log p$). DFT significantly outperforms SFT in mathematical reasoning, code generation, and multimodal tasks, and even surpasses GRPO/PPO in offline RL settings.

## Background & Motivation

**Background**: SFT is the standard paradigm for LLM post-training to efficiently acquire expert-like behavior. RL enables better generalization by exploring diverse strategies through reward signals but requires substantial computation, meticulous hyperparameter tuning, and explicit reward functions.

**Limitations of Prior Work**:
   - (1) SFT exhibits performance degradation on difficult benchmarks (OlympiadBench/AIME24): Qwen2.5-Math-1.5B accuracy on OlympiadBench drops from 15.88 to 12.63 after SFT.
   - (2) The "SFT memorizes, RL generalizes" phenomenon has been empirically observed by multiple works (Chu et al., 2024) but lacks root-cause analysis at the mathematical level.
   - (3) Hybrid SFT+RL methods (InstructGPT, DPO, etc.) do not improve the SFT process itself. SFT remains the only choice when only positive samples are available without reward or preference data.

**Core Problem**: Why does SFT generalize poorly when only positive samples are available? Can the root cause of the difference between SFT and RL be revealed theoretically?

**Key Insight**: By re-writing the SFT gradient in the form of a policy gradient using importance sampling, it is discovered that the implicit "reward" in SFT is a sparse indicator function weighted by the inverse probability. This is the mathematical root cause of limited generalization.

**Mechanism**: Focal Loss $-(1-p)^\gamma\log p$ emphasizes hard samples (underfitting era) → DFT $-p\log p$ de-emphasizes hard samples (LLM overfitting era). This represents a fundamental reversal in objective design philosophy.

**Goal**: Can SFT approach RL performance under conditions without reward models, preference data, or online sampling (using only positive samples)?

## Method

### Overall Architecture

This paper examines SFT within the RL policy gradient framework. By applying importance sampling to rewrite the standard SFT gradient into an on-policy policy gradient form, the authors identify the implicit reward structure behind SFT. The diagnosis reveals that this implicit reward is both sparse and amplified by the inverse probability $1/\pi_\theta$, which is the root cause of poor generalization. The proposed solution, DFT (Dynamic Fine-Tuning), neutralizes the inverse probability term by multiplying the cross-entropy by the token probability—a one-line code modification.

### Key Designs

**1. Policy Gradient Re-derivation of SFT: Quantifying Why Generalization Fails**

The "SFT memorizes, RL generalizes" observation is moved from empirical to mathematical. By applying importance sampling to the standard SFT gradient, it can be strictly rewritten as an on-policy policy gradient: $\nabla\mathcal{L}_{SFT} = -\mathbb{E}_{y\sim\pi_\theta}\big[\tfrac{\mathbf{1}[y=y^*]}{\pi_\theta(y|x)} \nabla\log\pi_\theta(y|x)\big]$. Comparing this with the standard policy gradient $\nabla J = \mathbb{E}[\nabla\log\pi_\theta \cdot r(x,y)]$, the implicit reward of SFT consists of two parts: a reward function $r(x,y)=\mathbf{1}[y=y^*]$ that is non-zero only for exact matches (extremely sparse), and an importance weight $w=1/\pi_\theta(y|x)$ that increases as the model probability for a token decreases. This causes the gradient to pivot aggressively toward low-probability exact-match tokens, leading the model to overfit the training set rather than learning generalizable skills.

**2. DFT Rectification Loss: Neutralizing Inverse Probability Weights with Token Probability**

Since the root cause is the $1/\pi_\theta$ term, the most direct solution is to multiply by $\pi_\theta$ to eliminate it. The resulting token-level DFT loss is: $\mathcal{L}_{DFT} = -\sum_{t=1}^{|y^*|} \text{sg}\big(\pi_\theta(y_t^*|y_{<t}^*,x)\big) \log\pi_\theta(y_t^*|y_{<t}^*,x)$, where $\text{sg}(\cdot)$ is the stop-gradient operator. This ensures the weight only scales the gradient without introducing additional optimization paths. Formally, this replaces the standard cross-entropy $-\log p$ with $-p \log p$. DFT maintains the implementation simplicity of SFT without requiring reward models, preference data, reference models, or online sampling.

**3. Token-level Weighting vs. Uniform Reward: Why Token Granularity is Necessary**

The choice of weighting granularity is critical. Using sentence-level probability $\pi(y|x)=\prod_t\pi(y_t)$ causes weights to vanish due to numerical underflow in long sequences, making the loss uninformative. Experimentally, sentence-level weighting yielded a score of 15.75, equivalent to the unweighted 15.92. In contrast, token-level weighting improved the mean from 15.92 to 31.58. From a reward perspective, neutralizing the weight results in a uniform reward of 1 for every expert trajectory, similar to assigning the same reward to all correct samples in RLVR. This prevents updates from concentrating on a few low-probability tokens, ensuring stability and generalization.

## Key Experimental Results

### Main Results: Mathematical Reasoning (Avg@16, NuminaMath-CoT 100K samples)

| Model | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|------|---------|---------|---------------|--------|-------|-----|
| Qwen2.5-Math-1.5B (base) | 31.66 | 8.51 | 15.88 | 4.16 | 19.38 | 15.92 |
| + SFT | 43.76 | 13.04 | 12.63↓ | 1.87↓ | 18.75↓ | 18.01 |
| + **DFT** | **64.89** | **20.94** | **27.08** | **6.87** | **38.13** | **31.58** |
| Qwen2.5-Math-7B (base) | 40.12 | 14.39 | 17.12 | 6.68 | 27.96 | 21.25 |
| + SFT | 53.96 | 16.66 | 18.93 | 2.48↓ | 26.09↓ | 23.62 |
| + **DFT** | **68.20** | **30.16** | **33.83** | **8.56** | **45.00** | **37.15** |

DFT achieved a mean improvement of +15.66 on Qwen2.5-Math-1.5B, which is **5.9x** the gain of SFT (+2.09). While SFT degraded on AIME24 and OlympiadBench, DFT showed consistent positive improvements.

### Main Results: Offline RL Comparison (Qwen2.5-Math-1.5B)

| Method | Setting | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|------|------|---------|---------|---------------|--------|-------|-----|
| DPO | Offline | 46.89 | 11.53 | 22.86 | 4.58 | 30.16 | 23.20 |
| RFT | Offline | 48.23 | 14.19 | 22.29 | 4.37 | 30.78 | 23.97 |
| PPO | Online | 56.10 | 15.41 | 26.33 | 7.50 | 37.97 | 28.66 |
| GRPO | Online | 62.86 | 18.93 | 28.62 | 8.34 | 41.25 | 32.00 |
| **DFT** | **Offline** | **64.71** | **25.16** | **30.93** | **7.93** | **48.44** | **35.43** |

As an offline method, DFT outperformed all offline RL baselines (+11.46 vs. RFT) and even surpassed online GRPO (+3.43).

### Main Results: Code Generation (UltraFeedback 10K)

| Model | HumanEval | HE+ | MultiPL-E Avg |
|------|-----------|-----|---------------|
| Qwen2.5-Coder-7B base | 62.2 | 53.0 | 57.76 |
| + SFT | 54.9↓ | 48.8↓ | 57.62 |
| + **DFT** | **67.7** | **59.8** | **62.30** |

SFT performance degraded across the board on Qwen2.5-Coder-7B, whereas DFT consistently improved.

## Key Findings

1. **Mathematical Explanation for SFT Degradation**: SFT degrades on difficult benchmarks because the implicit $1/\pi_\theta$ inverse probability weight biases gradients toward low-probability tokens, favoring training set exact matches at the expense of generalization.

2. **Token Probability Polarization Effect**: After DFT training, token probabilities exhibit a bimodal distribution. Probabilities for semantic keywords increase, while probabilities for functional tokens (e.g., 'the', 'let', ',', '.') are suppressed. The model learns to distinguish semantic vs. syntactic tokens.

3. **Significant Convergence Efficiency**: DFT reaches peak performance within 120 steps, surpassing SFT's final accuracy within the first 10-20 steps. The gradient updates are more informative, avoiding the optimization stagnation typical of SFT.

4. **Cross-Task/Model Consistency**: Improvements are consistent across 5 base models (LLaMA/DeepSeek/Qwen) and 4 task categories (Math/Code/Multimodal/RL), demonstrating strong robustness.

5. **Failure on Factual Knowledge Tasks**: On the Natural Questions dataset, DFT (30.14%) performed worse than SFT (36.62%). DFT reinforces the model's existing beliefs, which hinders learning when the model lacks domain-specific knowledge.

## Highlights & Insights
- **"SFT = RL with pathological rewards"** is a profound theoretical insight. It moves beyond empirical observation to a rigorous mathematical derivation that reveals exactly why "SFT memorizes while RL generalizes."
- **The Power of One Line**: Replacing $-\log p$ with $-p\log p$ is elegantly simple yet functionally powerful, combining academic beauty with practical utility.
- **Zero Resource Overhead**: No requirements for reward models, preference data, reference models, large batches, or online sampling makes this the optimal choice for resource-constrained scenarios.
- **Focal Loss Reversal**: Reflects the shift from the CV era (emphasizing hard samples for underfitting) to the LLM era (de-emphasizing hard samples for overfitting).

## Limitations & Future Work
- **Constrained Factual Knowledge Learning**: DFT relies on the model's own confidence for weighting. If the model lacks prior knowledge, it may reinforce incorrect beliefs (1%+ degradation on NQ).
- **Disadvantage for Truly Hard/Low-resource Samples**: Weights for samples with low initial probabilities are suppressed by DFT, potentially losing rare but important training signals.
- **Evaluation Scope**: Not yet validated on larger scale LLMs (70B+) or a wider variety of task types (e.g., dialogue, summarization).
- **Quality Agnosticism**: All positive samples receive a uniform reward of 1. It does not exploit quality differences between examples; non-uniform reward assignment is a direction for future work.

## Related Work & Insights

| Method | Key Difference |
|----------|----------|
| **DPO** (Rafailov 2023) | Requires preference pairs (positive/negative) to optimize policy via implicit rewards. DFT uses only positive samples with a one-line loss change. DFT outperforms DPO by +12.23 avg in offline settings. |
| **GRPO** (DeepSeek 2024) | An online RL method requiring multiple sampled responses and reward verification. DFT requires no online sampling/reward model. DFT outperforms GRPO by +3.43 avg at similar scales. |
| **iw-SFT** (Qin & Springenberg 2025) | Uses importance weighting based on data generation policies, necessitating the estimation of an unknown $\pi_b$. DFT uses the current model probability as the weight directly, making it simpler. |
| **Focal Loss** (Lin 2017) | Emphasizes hard samples with $-(1-p)^\gamma\log p$. DFT conversely de-emphasizes them with $-p\log p$, which is more appropriate for LLM overfitting scenarios. |

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Theoretical SFT-RL unification + profound simplicity of the one-line fix.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models across 5 math benchmarks, code, multimodal, offline RL, and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless integration of theoretical derivation and practice with a clear narrative.
- Value: ⭐⭐⭐⭐⭐ Significant direct impact on LLM SFT practices with an extremely low barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective](benefits_and_pitfalls_of_reinforcement_learning_for_language_model_planning_a_th.md)
- [\[ICLR 2026\] Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT](getting_your_llms_ready_for_reinforcement_learning_with_lightweight_sft.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[ICLR 2026\] Leveraging Explanation to Improve Generalization of Meta Reinforcement Learning](leveraging_explanation_to_improve_generalization_of_meta_reinforcement_learning.md)
- [\[ICLR 2026\] R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](r1-reward_training_multimodal_reward_model_through_stable_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
