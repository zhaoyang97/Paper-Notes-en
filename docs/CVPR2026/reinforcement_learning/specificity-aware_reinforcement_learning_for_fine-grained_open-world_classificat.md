---
title: >-
  [Paper Note] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification
description: >-
  [CVPR2026][Reinforcement Learning][open-world classification] This paper proposes SpeciaRL—a specificity-aware reinforcement learning framework that guides reasoning-capable large multimodal models to simultaneously impr…
tags:
  - "CVPR2026"
  - "Reinforcement Learning"
  - "open-world classification"
  - "fine-grained recognition"
  - "large multimodal models"
  - "GRPO"
  - "specificity-aware reward"
date: 2026-05-08
content_hash: c69d5ab34a11fe88
---

# Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification

**Conference**: CVPR2026
**arXiv**: [2603.03197](https://arxiv.org/abs/2603.03197)  
**Code**: [s-angheben/SpeciaRL](https://github.com/s-angheben/SpeciaRL)  
**Area**: Reinforcement Learning
**Keywords**: open-world classification, fine-grained recognition, reinforcement learning, large multimodal models, GRPO, specificity-aware reward

## TL;DR

This paper proposes SpeciaRL—a specificity-aware reinforcement learning framework that guides reasoning-capable large multimodal models to simultaneously improve prediction specificity and correctness in open-world fine-grained image classification, via a dynamic reward signal derived from the best prediction among online rollouts.

## Background & Motivation

**Growing demand for open-world classification**: Traditional image classification operates under a closed vocabulary, but real-world scenarios require handling emerging categories and novel concepts, rendering the fixed-vocabulary assumption invalid.

**LMMs are strong reasoners but tend to generalize**: State-of-the-art reasoning large multimodal models (e.g., Qwen2.5VL) exhibit strong visual understanding, yet tend to produce overly generic predictions on fine-grained classification tasks (e.g., outputting "flower" instead of "daisy").

**Naively enforcing specificity hurts correctness**: Prompting models to "be more specific," or applying SFT/standard RFT fine-tuning, can improve specificity but simultaneously increases incorrect predictions, revealing a non-trivial trade-off between the two objectives.

**Models do not lack knowledge**: A Best-of-N analysis shows that the best prediction of Qwen2.5VL-7B across 64 rollouts substantially surpasses single-pass inference in both correctness and specificity, indicating that models already possess fine-grained prior knowledge but fail to express it reliably in a single sample.

**Existing RLVR methods are ill-suited for open-world settings**: Standard RLVR employs binary exact-match rewards, which cannot provide appropriate signals for predictions that are "correct but insufficiently specific," and risk driving models toward overconfidence.

**Lack of systematic study**: How to improve specificity without sacrificing correctness in an open-world setting is a severely underexplored problem.

## Method

### Overall Architecture

SpeciaRL is built upon the GRPO (Group Relative Policy Optimization) online policy optimization framework:

1. For each input image $I$, the policy model generates $N$ open-ended predictions $\{p_1, \dots, p_N\}$.
2. An LLM judge (Qwen3-30B) classifies the relationship between each prediction and the ground truth into one of six categories $\mathcal{C} = \{W, A, G, S^-, S, S^+\}$ (wrong, abstain, generic, less specific, specific, more specific).
3. A minimum specificity requirement $c^*$ is dynamically set based on the best prediction category $c_{best}$ among the current $N$ rollouts.
4. Predictions that meet or exceed $c^*$ receive a positive reward; otherwise, the reward is zero.
5. Policy parameters are updated via GRPO to reinforce maximally specific predictions within the model's capability.

### Key Designs: Specificity-aware Dynamic Reward

**Prediction categorization**: An ordered category set $W \prec A \prec G \prec S^- \prec S \prec S^+$ is defined, and an LLM-as-a-judge automatically evaluates which category each prediction belongs to.

**Dynamic reference level**: The minimum requirement $c^*$ is adaptively determined by the best category $c_{best}$ among the current rollouts:

$$c^* = \begin{cases} S, & \text{if } c_{best} = S^+ \\ A, & \text{if } c_{best} = W \\ c_{best}, & \text{otherwise} \end{cases}$$

**Reward function**:

$$r_I^*(p, y) = \begin{cases} 1, & \text{if } c_y(p) \succeq c^* \\ 0, & \text{otherwise} \end{cases}$$

Core intuition: if the best prediction for a given sample is itself Generic, penalizing other Generic predictions from that sample for insufficient specificity would push the model toward generating more incorrect outputs. The dynamic reward ensures that maximal specificity within the model's capability is encouraged.

### Loss & Training

The standard GRPO objective is adopted, embedding the dynamic reward into group-relative advantage estimation with a KL divergence regularization term ($\lambda = 0.01$) to prevent policy drift. During training, $N=10$ rollouts are used simultaneously for both reward computation and policy updates, incurring no additional inference overhead.

## Key Experimental Results

### Main Results

**Cross-domain fine-grained classification (training domain: CUB birds → test domains: flowers/food/pets/cars/aircraft)**:

| Method | Specificity↑ | Correctness↑ | HM↑ |
|--------|-------------|-------------|-----|
| Qwen2.5VL-7B (zero-shot) | 0.742 | 0.846 | 0.790 |
| Qwen2.5VL-7B ("Be specific") | 0.816 | 0.832 | 0.822 |
| Qwen2.5VL-7B (SFT) | 0.935 | 0.807 | 0.866 |
| Qwen2.5VL-7B (RFT) | 0.875 | 0.785 | 0.825 |
| **SpeciaRL-7B** | **0.920** | **0.848** | **0.883** |
| BoN-64 (upper bound) | 0.889 | 0.984 | 0.933 |

On highly fine-grained datasets (StanfordCars, FGVCAircraft), SpeciaRL also achieves the best HM (0.830), outperforming SFT (0.814) and RFT (0.821).

### Ablation Study

**Static vs. dynamic reward**: Compared against four static reward variants, SpeciaRL's dynamic reward achieves the best overall HM of 0.883. The standard binary reward yields only HM=0.825, demonstrating that graded rewards for "correct but insufficiently specific" predictions are critical.

**Number of rollouts $N$**: $N=5$ and $N=10$ perform comparably (HM=0.883), while performance degrades at $N=15$ (HM=0.824), possibly due to limitations of the batch-based grouping strategy.

**Compatibility across RL algorithms**: Across three online policy optimization algorithms—GRPO, Dr.GRPO, and DAPO—the SpeciaRL dynamic reward consistently improves HM (+1.5%~+5.8%), demonstrating the generality of the approach.

### Key Findings

- On the fine-grained datasets, SpeciaRL simultaneously improves both specificity (+0.178) and correctness (+0.002), being the only method to achieve joint improvement on both dimensions.
- SFT achieves very high specificity (0.935) but at a severe cost to correctness (0.807), yielding a lower HM than SpeciaRL.
- Training on only 3,000 samples from a single domain (birds) generalizes to completely different domains including flowers, food, pets, cars, and aircraft.
- Under the general evaluation protocol of [10] (TI/LI/SS/CS), SpeciaRL achieves the best performance on 6 out of 8 metrics.

## Highlights & Insights

1. **Clear and novel problem formulation**: This is the first work to systematically identify the specificity–correctness trade-off in open-world classification as an independent research problem.
2. **Elegant dynamic reward design**: The reward threshold is adaptively adjusted based on online rollouts, leveraging GRPO's existing multi-sample mechanism without additional computational overhead.
3. **Strong cross-domain generalization**: Training solely on bird data generalizes to entirely different domains such as food and cars, suggesting the method learns a reasoning strategy rather than domain-specific knowledge.
4. **Practical six-level categorization scheme**: The $\{W, A, G, S^-, S, S^+\}$ taxonomy comprehensively covers possible prediction relationships in open-world settings and provides a reusable evaluation protocol.

## Limitations & Future Work

1. **Dependence on LLM judge quality**: The reward signal relies on LLM-as-a-judge; if the judge makes errors in certain fine-grained domains (e.g., confusing closely related species), erroneous signals may propagate.
2. **Validation limited to classification**: The approach has not been extended to more complex visual tasks such as detection or segmentation.
3. **Restricted base model**: Validation is conducted solely on Qwen2.5VL-7B; larger models or different architectures remain unexplored.
4. **Small training data scale**: With only 3,000 training samples, a gap remains between SpeciaRL and the BoN-64 upper bound on highly fine-grained datasets (HM 0.830 vs. 0.868).
5. **Sensitivity to rollout count**: Performance degrades at $N=15$, indicating a degree of dependence on GRPO's batch grouping strategy.

## Related Work & Insights

- **Visual-RFT [34]**: Applies RLVR to closed-set classification using binary exact-match rewards—SpeciaRL extends this to open-world settings and introduces graded rewards.
- **Conti et al. [10]**: Proposes an open-world classification evaluation benchmark and reveals the generalization tendency of LMMs—this paper builds upon that foundation to propose a solution.
- **DeepSeek-R1 [16]**: A canonical application of the GRPO algorithm—SpeciaRL integrates specificity-aware rewards into this framework.
- **Hierarchical precision/recall [44]**: Evaluates prediction quality based on explicit taxonomic trees—this paper does not assume a predefined hierarchy, using an LLM judge instead.
- **CaSED [9]**: CLIP retrieval-based open-world classification—exhibits good specificity but lower correctness compared to reasoning-based LMMs.

## Rating

- Novelty: ⭐⭐⭐⭐ (novel dynamic reward design; clearly formulated problem)
- Experimental Thoroughness: ⭐⭐⭐⭐ (cross-domain evaluation + multiple ablations + multi-algorithm validation, but only one base model tested)
- Writing Quality: ⭐⭐⭐⭐⭐ (complete and coherent logical chain from analysis → insight → method → validation)
- Value: ⭐⭐⭐⭐ (practical framework for open-world fine-grained classification with strong cross-domain generalization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](../../ICLR2026/reinforcement_learning/dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](../../ICLR2026/reinforcement_learning/rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)
- [\[ACL 2026\] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](../../ACL2026/reinforcement_learning/beyond_majority_voting_towards_fine-grained_and_more_reliable_reward_signal_for_.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](../../AAAI2026/reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](../../ICLR2026/reinforcement_learning/from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
