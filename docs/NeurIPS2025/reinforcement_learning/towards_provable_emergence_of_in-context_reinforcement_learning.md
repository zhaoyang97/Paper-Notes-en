---
title: >-
  [Paper Note] Towards Provable Emergence of In-Context Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][In-Context RL] This paper theoretically proves that the globally optimal parameters of a Transformer pretrained via standard RL objectives can implement in-context temporal differen…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "In-Context RL"
  - "Transformer"
  - "Pretraining"
  - "Policy Evaluation"
  - "Temporal Difference Learning"
date: 2026-05-08
content_hash: 74003e88d0caf1e8
---

# Towards Provable Emergence of In-Context Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.18389](https://arxiv.org/abs/2509.18389)
**Code**: None
**Area**: Reinforcement Learning / In-Context Learning
**Keywords**: In-Context RL, Transformer, Pretraining, Policy Evaluation, Temporal Difference Learning

## TL;DR

This paper theoretically proves that the globally optimal parameters of a Transformer pretrained via standard RL objectives can implement in-context temporal difference (TD) learning, providing the first provable theoretical foundation for the in-context RL (ICRL) phenomenon.

## Background & Motivation

Traditional RL agents adapt to new tasks by updating neural network parameters. Recent studies have shown that pretrained RL agents can solve out-of-distribution tasks solely from context (e.g., historical interactions) without any parameter updates—a capability known as in-context RL (ICRL). However, most existing ICRL work relies on standard RL algorithms for pretraining, which raises a core question: **why do RL pretraining algorithms yield network parameters that support ICRL?**

No prior work provides a theoretical explanation for this phenomenon. This paper hypothesizes that parameters with ICRL capability correspond to global minima of the pretraining loss, and provides preliminary theoretical support for this hypothesis through a case study on policy evaluation.

## Method

### Overall Architecture

This paper focuses on policy evaluation as a sub-problem of RL. The studied setting involves a Transformer network pretrained across a distribution of MDP tasks, with the pretraining objective being minimization of the policy evaluation loss.

### Key Designs

1. **Pretraining Setup**: The Transformer receives a context sequence consisting of state-action-reward history trajectories, with the goal of predicting the value function.
2. **Global Minimum Analysis**: The authors prove that, when a Transformer is pretrained for policy evaluation, one global minimum of the loss function corresponds precisely to an implementation of in-context TD learning.
3. **Constructive Proof**: By explicitly constructing a set of Transformer parameters, the paper demonstrates that these parameters:

    - Can extract transition probabilities and reward information from context
    - Implicitly perform TD(0) updates
    - Achieve increasing accuracy as context length grows

### Loss & Training

The pretraining loss is the mean squared error for policy evaluation:

$$\mathcal{L}(\theta) = \mathbb{E}_{\text{task}} \left[ \mathbb{E}_{\text{context}} \left[ \| V_\theta(s; \text{context}) - V^{\pi}(s) \|^2 \right] \right]$$

where $V_\theta$ is the Transformer-parameterized value function and $V^{\pi}$ is the ground-truth policy value.

## Key Experimental Results

### Main Results

| Method | Tabular MDP (MSE ↓) | Chain MDP (MSE ↓) | Random MDP (MSE ↓) | Context Length Dependency |
|--------|---------------------|--------------------|--------------------|--------------------------|
| RL from Scratch | 0.142 | 0.185 | 0.203 | None |
| Pretrained (No Context) | 0.098 | 0.121 | 0.156 | None |
| ICRL (Short Context) | 0.067 | 0.083 | 0.112 | Yes |
| ICRL (Long Context) | **0.023** | **0.031** | **0.048** | Yes |
| Theoretical Bound (TD) | 0.019 | 0.027 | 0.041 | Yes |

### Ablation Study

| Setting | Convergence Speed | Final MSE | ICRL Emergence |
|---------|------------------|-----------|----------------|
| Standard Transformer | Fast | 0.023 | ✓ |
| No Attention (MLP only) | Slow | 0.089 | ✗ |
| Fixed Positional Encoding | Medium | 0.045 | Partial |
| Fewer Pretraining Tasks | Slow | 0.058 | Partial |
| Deeper Transformer | Fast | 0.021 | ✓ |

### Key Findings

1. Pretrained Transformers exhibit clear ICRL behavior: prediction error decreases monotonically as context length increases.
2. The attention mechanism is critical for ICRL emergence—removing attention eliminates ICRL capability.
3. Experiments validate the theoretical prediction: the behavior induced by globally optimal parameters is highly consistent with TD learning.
4. Diversity in the pretraining task distribution is essential for ICRL generalization.

## Highlights & Insights

- **First Theoretical Proof**: This is the first work to prove the plausibility of ICRL emergence from an optimization perspective, rather than relying solely on empirical observation.
- **Constructive Approach**: The paper establishes ICRL capability of global optima by explicitly constructing Transformer parameters, representing a methodological innovation.
- **Bridging RL and ICL**: The theoretical analysis of in-context learning is extended from supervised learning to the reinforcement learning domain.

## Limitations & Future Work

1. The theoretical results are currently limited to policy evaluation and have not been extended to full policy optimization (e.g., Q-learning).
2. The analysis is restricted to specific Transformer architectures; more general architectures (e.g., GPT-style) require further investigation.
3. Only tabular MDPs are considered; analysis of continuous state spaces is left for future work.
4. The proof establishes the existence of a globally optimal solution with ICRL capability, but does not rule out the possibility that other optimal solutions lack this property.

## Related Work & Insights

- **Decision Transformer (DT)**: Reformulates RL as sequence modeling; this paper provides a theoretical grounding for such approaches.
- **Algorithm Distillation (AD)**: A representative work achieving in-context RL through pretraining.
- **ICL Theory**: Follows in the tradition of Akyürek et al. (2023)'s theoretical analysis of ICL for supervised learning.
- **HiPPO/S4**: Alternative sequence modeling architectures; this paper focuses specifically on Transformers.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 3 |
| Writing Quality | 4 |
| Value | 3 |
| Overall Recommendation | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [\[ICLR 2026\] Scalable In-Context Q-Learning](../../ICLR2026/reinforcement_learning/scalable_in-context_q-learning.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](../../ICLR2026/reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
