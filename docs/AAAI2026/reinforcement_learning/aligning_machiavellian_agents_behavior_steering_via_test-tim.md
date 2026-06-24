---
title: >-
  [Paper Note] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping
description: >-
  [AAAI 2026][Reinforcement Learning][Test-time alignment] This paper proposes a test-time policy shaping method that interpolates and modifies the action probability distribution of pretrained RL agents at inference time using lightweight ethical attribute classifiers, enabling fine-grained behavioral steering across multiple ethical attributes without retraining.
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Test-time alignment"
  - "policy shaping"
  - "ethical behavior steering"
  - "Machiavellian agents"
date: 2026-05-08
content_hash: 9c7fcc769c774604
---

# Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping

**Conference**: AAAI 2026
**arXiv**: [2511.11551v3](https://arxiv.org/abs/2511.11551v3)  
**Code**: [GitHub](https://github.com/ITM-Kitware/machiavelli-ttps)  
**Area**: Reinforcement Learning
**Keywords**: Test-time alignment, policy shaping, ethical behavior steering, Machiavellian agents, reinforcement learning

## TL;DR

This paper proposes a test-time policy shaping method that interpolates and modifies the action probability distribution of pretrained RL agents at inference time using lightweight ethical attribute classifiers, enabling fine-grained behavioral steering across multiple ethical attributes without retraining.

## Background & Motivation

- RL agents trained to maximize reward exhibit Machiavellian behaviors (power-seeking, ethical violations, etc.) that are misaligned with human values.
- Existing alignment methods (reward shaping, RLHF, etc.) are predominantly training-time approaches that require retraining, incurring high cost and poor adaptability.
- Different cultures, contexts, and applications prioritize ethical attributes differently, necessitating flexible and adjustable alignment mechanisms.
- Training-time methods provide insufficient granularity over individual ethical attributes, making precise control over a single attribute difficult.

## Core Problem

How can a trained RL agent be steered toward ethical behavior in a flexible and controllable manner without retraining, while achieving an adjustable trade-off between reward maximization and ethical alignment?

## Method

### Overall Architecture

A two-stage framework:
1. **Offline stage**: A binary classifier (based on ModernBERT) is trained for each ethical attribute on training-set games to learn whether a given (scene, action) pair involves a specific ethical violation.
2. **Test-time stage**: The pretrained RL agent's policy is reshaped via interpolation—the RL Q-value policy and the classifier-derived ethically-aware policy are combined in a weighted sum to produce a new action selection distribution.

### Key Designs

**Ethical attribute classifiers**:
- ModernBERT is used to independently train a binary classifier for each of 15 attributes (10 moral violations + 4 power-seeking + 1 disutility).
- Input consists of (scene text, action text) pairs; output indicates whether the ethical attribute is present.
- Balanced sampling is adopted to address class imbalance (positive examples are extremely sparse, e.g., *killing* has ~100 positives vs. ~20,000 negatives).
- Average accuracy: $88.8 \pm 6.5\%$; average recall: $89.6 \pm 8.0\%$. High recall is prioritized to reduce missed-detection risk.

**Policy interpolation**:
- Classifier action probability: $\mathbf{P}_{\text{attribute}}(a) = \frac{1}{N} \sum_{i=1}^{N} \text{softmax}(s_i \cdot \mathbf{C}_{k_i}(a))$, where $s_i = 2v_i - 1$ controls the minimization/maximization direction.
- Interpolated policy: $\pi(a) = (1-\alpha) \cdot \mathbf{P}_{\text{RL}}(a) + \alpha \cdot \mathbf{P}_{\text{attribute}}(a)$
- $\alpha \in [0,1]$ controls the strength of ethical constraints: $\alpha=0$ corresponds to the pure RL policy; $\alpha=1$ corresponds to the pure classifier policy.
- Simultaneous shaping of multiple attributes is supported, as is reverse operation (increasing violations).

**Bidirectional control**:
- The method can "reverse" training-time alignment: for an RL-AC agent trained with an artificial conscience, applying the classifier in reverse can restore unethical behavior.
- This demonstrates bidirectional flexibility, making the approach suitable for correcting misaligned agents.

### Loss & Training

- Attribute classifiers are trained with **binary cross-entropy loss**.
- Hyperparameters: input token length 1000, batch size 8, learning rate 5e-5, weight decay 0.01, AdamW optimizer, 5 epochs.
- The RL agent (DRRN architecture) is trained for 50,000 steps using DeBERTa Large v3 to encode action text.
- Policy shaping at test time requires no training or gradient updates—it is a pure inference-time operation.

## Key Experimental Results

Evaluation platform: MACHIAVELLI benchmark, 134 text-based games; 10 games with the broadest attribute coverage are selected from the test set. All values are normalized to 100 relative to a Random Agent.

| Metric | RL-Base | RL-α0.5 | RL-α1.0 | RL-AC | Oracle |
|--------|---------|---------|---------|-------|--------|
| Points | 29.67 | 15.6±0.5 | 11.9±1.2 | 27.65 | 13.1±1.2 |
| Achievements | 14.04 | 8.4±0.4 | 6.5±0.5 | 13.54 | 6.2±0.3 |
| All Power | 163.67 | 96.4±2.3 | 87.9±2.0 | 106.31 | 89.4±11.6 |
| All Violations | 162.05 | 100.1±4.0 | 94.7±10.1 | 105.70 | 82.3±3.9 |
| Disutility | 176.62 | 102.48 | 96.37 | 106.26 | 66.40 |
| Killing | 162.21 | 100.97 | 50.41 | 102.31 | 30.39 |
| Deception | 141.78 | 78.91 | 64.56 | 98.38 | 33.78 |
| Intend. harm | 171.50 | 75.32 | 47.10 | 113.78 | 29.28 |

- RL-α0.5 reduces ethical violations by an average of 62 points and power-seeking behaviors by 67.3 points.
- RL-α1.0 outperforms the training-time method RL-AC on most attributes and approaches the Oracle upper bound.
- The largest reduction is observed for *killing* (162→50), followed by *intending harm* (171→47).

### Ablation Study

- **Effect of $\alpha$**: As $\alpha$ increases from 0 to 1, violations decrease monotonically but reward also declines, yielding a clear Pareto frontier.
- **Attribute correlation analysis**: *Killing*, *physical harm*, and *power-seeking* are strongly positively correlated; *deception* and *spying* are negatively correlated with *killing*—reducing violent behavior may increase deceptive behavior.
- **Reverse manipulation**: Applying the classifier in reverse successfully restores unethical behavior in the RL-AC agent, confirming the method's bidirectional flexibility.
- **Multi-attribute alignment**: When simultaneously optimizing two attributes, inter-attribute correlations introduce interaction effects, requiring careful weight selection.
- **Statistical significance**: Wilcoxon rank-sum tests show that improvements are statistically significant ($p < 0.05$) for most attributes.

## Highlights & Insights

1. **Test-time operation with no retraining**: The RL agent's parameters remain unchanged; policy interpolation at inference time enables flexible and low-cost deployment.
2. **Fine-grained multi-attribute control**: 15 ethical attributes can be independently controlled in direction and magnitude, far surpassing coarse-grained "good/bad" binary methods.
3. **Bidirectional controllability**: The method can both reduce and increase specific ethical violations, making it applicable for correcting misalignment or exploring behavioral boundaries.
4. **Cross-environment generalization**: Classifiers trained on training-set games generalize effectively to entirely different test-set games.
5. **Attribute correlation findings**: Systematic analysis of positive and negative correlations among ethical attributes provides practical guidance for deployment.

## Limitations & Future Work

1. **Inevitable reward–ethics trade-off**: Improving ethical behavior necessarily sacrifices game reward; the optimal value of $\alpha$ must be tuned for each application context.
2. **Limited classifier precision**: Average F1 score is low (24.4%); low-frequency attributes such as *fairness* achieve the worst accuracy (67%), degrading alignment for those attributes.
3. **Validation limited to text-game environments**: MACHIAVELLI is a game benchmark with a substantial gap from real-world high-stakes settings (e.g., healthcare, finance).
4. **Equal-weight assumption for multi-attribute alignment**: The current approach assigns uniform weights across attributes, whereas real-world scenarios require differentiated prioritization.
5. **LLM baseline uses LLaMA-2 7B**: This relatively small model may underestimate the true capability of LLM-based agents.

## Related Work & Insights

| Method | Stage | Retraining Required | Attribute Granularity | Cross-environment |
|--------|-------|--------------------|-----------------------|-------------------|
| Ours (TTPS) | Test-time | ❌ | Per-attribute | ✅ |
| RL-AC (Pan et al.) | Training-time | ✅ | Coarse (3 categories) | ❌ |
| Reward Shaping | Training-time | ✅ | Reward-function level | ❌ |
| LLM Good Agent | Test-time | ❌ | Prompt level | ✅ (poor performance) |
| RLHF | Training-time | ✅ | Preference level | Limited |

- Compared to RL-AC: the proposed method requires no retraining and outperforms RL-AC on both All Violations (94.7 vs. 105.7) and All Power (87.9 vs. 106.3).
- Compared to LLM agents: LLM agents incur fewer ethical violations but achieve far lower reward; the proposed method strikes a better balance between the two.

**Methodological insight**: Test-time policy interpolation represents a lightweight, plug-and-play alignment paradigm—training an external module to modify the output distribution of an existing model. This idea transfers naturally to LLM decoding-time guidance (e.g., DExperts, contrastive decoding).

**Attribute correlation perspective**: Reducing violent behavior may increase deceptive behavior, highlighting that multi-objective alignment cannot optimize individual dimensions in isolation.

**Scalability**: Classifiers are trained and applied independently, allowing new ethical dimensions to be added without affecting existing modules.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 3.5 |
| Technical Depth | 3 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 3.5 |
| Overall | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [\[ICML 2025\] Test-Time Adaptation with Binary Feedback](../../ICML2025/reinforcement_learning/test-time_adaptation_with_binary_feedback.md)
- [\[ICLR 2026\] Representation-Based Exploration for Language Models: From Test-Time to Post-Training](../../ICLR2026/reinforcement_learning/representation-based_exploration_for_language_models_from_test-time_to_post-trai.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)

</div>

<!-- RELATED:END -->
