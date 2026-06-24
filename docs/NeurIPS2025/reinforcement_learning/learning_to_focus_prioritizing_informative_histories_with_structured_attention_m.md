---
title: >-
  [Paper Note] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Partially Observable RL] Two structured temporal priors—Memory-Length Prior and Gaussian Distributional Prior—are embedded into the self-attention mechanism of a Transformer world model. Under partially observable RL settings, Gaussian Attention achieves a 77% relative improvement in human-normalized score over UniZero on the Atari 100k benchmark with negligible computational overhead.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Partially Observable RL"
  - "Transformer World Models"
  - "Attention Prior"
  - "Gaussian Attention"
  - "Sample Efficiency"
date: 2026-05-08
content_hash: 1ce7cf05991d77d1
---

# Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2511.06946](https://arxiv.org/abs/2511.06946)  
**Code**: [GitHub](https://github.com/daniallegue/learning-to-focus)  
**Area**: Reinforcement Learning
**Keywords**: Partially Observable RL, Transformer World Models, Attention Prior, Gaussian Attention, Sample Efficiency

## TL;DR

Two structured temporal priors—Memory-Length Prior and Gaussian Distributional Prior—are embedded into the self-attention mechanism of a Transformer world model. Under partially observable RL settings, Gaussian Attention achieves a 77% relative improvement in human-normalized score over UniZero on the Atari 100k benchmark with negligible computational overhead.

## Background & Motivation

Transformers as world models are increasingly popular in model-based reinforcement learning (MBRL). In particular, UniZero replaces MuZero's recurrent dynamics with a Transformer backbone, leveraging masked self-attention to capture long-range dependencies. However, a fundamental assumption mismatch exists:

- **NLP setting**: Data is abundant and balanced; long-range dependencies are frequent, and standard self-attention can implicitly learn them.
- **RL setting**: Trajectories are sparse and reward-driven; the vast majority of transitions carry no useful information, with only a small number of critical transitions driving decisions.

Standard self-attention assigns uniform initial weights to all historical tokens, making it difficult to rapidly identify which transitions are truly important in data-scarce RL environments. This leads to poor sample efficiency for Transformer world models in low-data regimes such as Atari 100k.

Core Problem: **Can structured temporal priors be directly encoded into self-attention so that the model knows "where to focus" from the outset?**

## Method

### Overall Architecture

Two inductive biases—Memory-Length Prior and Distributional Prior—are introduced into the self-attention layers of UniZero's dynamics head, which is responsible for predicting the next latent state $\hat{z}_{t+1}$ and immediate reward $\hat{r}_t$ from historical latent state–action pairs.

### Key Designs

1. **Memory-Length Prior (Adaptive Attention)**: Motivated by the assumption of limited effective memory in partially observable environments. Each attention head $h$ learns a scalar parameter $s_h$, transformed via softplus into a positive lookback span $L_h = \text{softplus}(s_h)$. A hard mask is constructed as:
$$M_{ij}^{(h)} = \begin{cases} 0, & i - j \leq L_h \\ -\infty, & i - j > L_h \end{cases}$$
The attention weights become $\text{Attention}^{(h)} = \text{softmax}\left(\frac{Q^{(h)} K^{(h)\top}}{\sqrt{d_k}} + M^{(h)}\right)$. An $\ell_1$ penalty is added to prevent all spans from growing unboundedly, encouraging each head to learn a minimal yet sufficient window.

2. **Distributional Prior (Gaussian Attention)**: In partially observable environments, only a sparse subset of tokens contributes to prediction. Each head learns Gaussian parameters $\mu_h, \sigma_h > 0$ and defines a positional kernel:
$$G_{ij}^{(h)} = -\frac{(i - j - \mu_h)^2}{2\sigma_h^2}$$
This is added to the attention logits as a bias. Unconstrained $\mu_h$ and $\sigma_h$ allow each head to acquire a smooth, learnable saliency distribution—concentrating on specific temporal offsets or spreading over a broad range. Different heads capture different temporal scales.

3. **Combined Prior (Gaussian Adaptive Attention)**: The two priors are summed as $B_{ij}^{(h)} = G_{ij}^{(h)} + M_{ij}^{(h)}$, maintaining smooth saliency within a bounded window. However, experiments show that the hard truncation cuts off the Gaussian tails and degrades performance.

### Loss & Training

- Training follows UniZero's joint model–policy optimization framework with soft-target world models for stable learning.
- Adaptive attention spans are regularized with $\ell_1$, $\ell_2$, or max-norm penalties; $\ell_2$ yields the most robust generalization.
- Gaussian priors are initialized with $\mu_h=6, \sigma_h=1$; narrow priors ($\sigma_h=1$) consistently outperform wide priors ($\sigma_h=3$).

## Key Experimental Results

### Main Results

Atari 100k benchmark, 26 games, 5 random seeds, compared against UniZero and MuZero:

| Method | HNS Mean | HNS Median | Games Won |
|---|---|---|---|
| MuZero | 0.44 | 0.13 | — |
| UniZero ST (Baseline) | 0.13 | 0.05 | — |
| Adaptive UniZero | 0.095 | 0.05 | Partial |
| **Gaussian UniZero** | **0.23** | **0.10** | **19/26** |
| Gaussian Adaptive UniZero | 0.00 | 0.02 | Very few |

Representative per-game score comparison:

| Game | UniZero | Gaussian UniZero | Gain |
|---|---|---|---|
| KungFuMaster | 2019 | **9424** | +367% |
| Kangaroo | 843 | **1636** | +94% |
| Assault | 342 | **487** | +42% |
| Jamesbond | 202 | **362** | +79% |

### Ablation Study

| Configuration | Pong | MsPacman | Jamesbond | Freeway |
|---|---|---|---|---|
| $L_h=2$ | -18.5 | 716.7 | 180.0 | 2.2 |
| $L_h=6$ | -19.6 | **1103.3** | 156.7 | 0.7 |
| $\sigma_h=1$ (narrow) | -7.9 | 726.7 | **362.1** | 0.1 |
| $\sigma_h=3$ (wide) | -15.1 | 638.7 | 196.7 | 0.0 |
| $\ell_1$ regularization | Moderate | Moderate | Moderate | Occasionally best |
| $\ell_2$ regularization | **Most robust** | **Most robust** | Moderate | Moderate |

### Key Findings

- **Gaussian priors substantially outperform hard truncation windows**: Smooth positional weights flexibly adapt to diverse temporal dependency structures, whereas hard cutoffs frequently misidentify the relevant context range.
- **Combining the two priors degrades performance**: Hard masking truncates useful signals in the Gaussian tails, producing conflicting priors.
- **Narrow Gaussian ($\sigma_h=1$) consistently outperforms wide Gaussian ($\sigma_h=3$)**: A strong initial prior is more effective than a weak one.
- **Computational overhead is negligible**: All prior variants increase MFLOPs by no more than 0.002%.

## Highlights & Insights

- The paper reveals a fundamental distinction between NLP and RL in sequence modeling: RL trajectories are sparse and reward-driven, and directly transferring the uniform attention assumption from NLP is inappropriate in low-data RL regimes.
- Smooth distributional priors are better suited than discrete memory windows for the irregular temporal dependency structures encountered in RL—a valuable design principle.
- The number of added parameters is minimal (only 2–3 scalars per head), yet the performance gains are substantial, demonstrating the power of well-chosen inductive biases.

## Limitations & Future Work

- Validation is limited to Atari environments; extension to continuous control or multi-task settings remains unexplored.
- Adaptive attention spans require regularization to avoid degenerate solutions (attending to everything or nothing).
- Gaussian priors are isotropic; more complex spatiotemporal structures may require more flexible distributional forms.
- No comparison is made against other positional encoding methods such as RoPE or ALiBi.

## Related Work & Insights

- **UniZero**: Replaces MuZero's recurrent dynamics with a Transformer; serves as the base architecture for this work.
- **Adaptive Attention Span**: Learns per-head context lengths, but is designed for NLP rather than RL.
- **Influence-Based Abstraction (IBA)**: Formalizes the concept of minimal sufficient history in POMDPs, providing the theoretical foundation for the Memory-Length Prior.
- **Insight**: Domain-specific inductive biases in Transformer architecture design are more efficient than general large-model architectures—this is particularly evident in RL.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing structured temporal priors into RL world models is a natural and effective contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 26 Atari games with comprehensive ablations, though continuous control experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, experimental analysis is thorough, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides practical design guidelines for Transformer world models in RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](../../ICML2025/reinforcement_learning/pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[ICLR 2026\] PAMDP: Interact to Persona Alignment via a Partially Observable Markov Decision Process](../../ICLR2026/reinforcement_learning/pamdp_interact_to_persona_alignment_via_a_partially_observable_markov_decision_p.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [\[ACL 2025\] Learning to Generate Structured Output with Schema Reinforcement Learning](../../ACL2025/reinforcement_learning/learning_to_generate_structured_output_with_schema_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
