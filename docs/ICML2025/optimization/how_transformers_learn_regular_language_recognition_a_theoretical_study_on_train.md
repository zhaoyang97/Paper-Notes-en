---
title: >-
  [Paper Note] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias
description: >-
  [ICML2025][Optimization][Transformer theory] This work theoretically characterizes the two-stage training dynamics of a single-layer Transformer learning two types of regular language recognition tasks, "even pairs" and "parity check". It proves that the linear layer implicitly converges to the max-margin hyperplane under gradient descent, and reveals the critical role of CoT in solving the parity problem.
tags:
  - "ICML2025"
  - "Optimization"
  - "Transformer theory"
  - "training dynamics"
  - "implicit bias"
  - "regular language recognition"
  - "Chain-of-Thought"
  - "max-margin"
date: 2026-05-08
content_hash: c0ffa67e0c7899a3
---

# How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias

**Conference**: ICML2025  
**arXiv**: [2505.00926](https://arxiv.org/abs/2505.00926)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Transformer theory, training dynamics, implicit bias, regular language recognition, Chain-of-Thought, max-margin

## TL;DR

This work theoretically characterizes the two-stage training dynamics of a single-layer Transformer learning two types of regular language recognition tasks, "even pairs" and "parity check". It proves that the linear layer implicitly converges to the max-margin hyperplane under gradient descent, and reveals the critical role of CoT in solving the parity problem.

## Background & Motivation

- **Formal language recognition** serves as a fundamental benchmark in NLP, widely used to measure the reasoning capabilities of LLMs, and acts as an important entry point for understanding the internal mechanisms of Transformers.
- Existing work heavily focuses on the **expressiveness** and **learnability** of Transformers on formal languages, but theoretical analysis of their **training dynamics** (how parameters evolve under gradient descent) remains virtually unexplored.
- This study specifically focuses on two representative tasks:
    - **Even pairs**: Determining whether the total number of "ab" and "ba" subsequences in a binary sequence is even (equivalent to checking if the first and last tokens are equal).
    - **Parity check**: Determining whether the number of "b"s in a sequence is even.
- Existing theoretical works on parity check (Kim & Suzuki 2024b; Wen et al. 2024) only analyze isolated attention layers or finite-step training, lacking a complete convergence analysis of the **joint training** of the attention layer and the linear layer.

## Method

### Model Architecture

A **single-layer Transformer** consisting of an attention layer and a linear layer is adopted, formalized as:

$$\mathtt{T}_\theta(X) = u^\top \sum_{\ell=1}^{L} x_\ell \varphi_\ell$$

where $\varphi_\ell = [\phi(X^\top W x_L / \lambda)]_\ell$ represents the softmax attention weights, and $\lambda$ is a scaling factor; the trainable parameters are $\theta = (u, W)$, where $u$ denotes the linear layer and $W$ denotes the attention layer.

### Loss & Training

Logistic loss is employed for binary classification:

$$\mathcal{L}(u,W) = \sum_{L=1}^{L_{\max}} \frac{1}{|I_L|} \sum_{n \in I_L} \log\left(1 + \exp(-y_n \mathtt{T}_\theta(X^{(n)}))\right)$$

A two-stage gradient descent is used: in Phase 1, the learning rate of the attention layer is scaled up by $\lambda$, and Phase 2 reverts to standard GD.

### Key Theoretical Results: Two-Stage Dynamics of Even Pairs

**Phase 1 (Rapid Growth Phase)**:

- **Linear layer**: The score of the first token $\langle u_t, E_1^w \rangle = \Theta(\eta t)$ grows rapidly; the scores of non-first tokens $\langle u_t, E_\ell^w \rangle = -\Theta(\eta^2 t^2)$ decrease rapidly.
- **Attention layer**: The first token receives the highest attention in positive samples, while the second token receives the highest attention in negative samples.
- By the end of Phase 1, the attention layer output satisfies **separability** (Proposition 4.3).

**Phase 2 (Margin Maximization Phase)**:

- The attention layer remains almost unchanged: $\|W_t - W_{t_0}\| \leq O(1)$.
- The norm of the linear layer grows logarithmically: $\|u_t\| \geq \Omega(\log t)$.
- The direction of the linear layer converges to the **max-margin hyperplane** $u_{EP}^*$ (Theorem 4.4).
- The loss converges at a rate of $O(L_{\max}\|u_{EP}^*\|^2 / (\eta\sqrt{t}))$ (Theorem 4.5).

### Two CoT Methods for Parity Check

**Method 1: Truncated CoT Inference (Zero-Shot)**

- Leverages the equivalence relation between even pairs and checking if the first and last tokens are equal.
- The pre-trained even pairs Transformer directly solves parity via iterative truncated CoT (comparing the first and last tokens at each step, appending the prediction, and removing the first token) **without requiring additional training**.

**Method 2: CoT Training under Teacher Forcing**

- Training loss = CoT loss + Even Pairs regularization loss: $\mathcal{L}_{Parity} = \mathcal{L}_{CoT} + \mathcal{L}_{Reg}$.
- The even pairs loss acts as a regularization to stabilize training (directly training CoT leads to vanishing gradients).
- This also exhibits two-stage dynamics, where the linear layer converges to the max-margin solution $u_{CoT}^*$ corresponding to CoT.

### Key Analytical Techniques

- Utilizing **high-order Taylor expansion** to precisely analyze the coupling effect of the gradients of both layers.
- Utilizing **implicit bias** theory to characterising the convergence direction in Phase 2.
- Carefully designing the scaling factor $\lambda$ to suppress attention layer updates in Phase 2, ensuring training stability.

## Key Experimental Results

### Experimental Setup

| Parameter | Value |
|------|------|
| $L_{\max}$ | 6 |
| $L_0$ (parity) | 4 |
| Learning rate $\eta$ | 0.1 |
| Phase 1 steps $t_0$ | 100 |
| Scaling factor $\lambda$ | 2 |
| Device | i5-12400F + 16GB |

### Main Results

| Phenomenon | Even Pairs | Parity Check |
|------|-----------|-------------|
| Rapid loss decay | ✓ | ✓ |
| Growth of the first token score | ✓ | ✓ |
| Decrease of non-first token scores | ✓ | ✓ |
| Higher attention on the first token for positive samples | ✓ | ✓ |
| Logarithmic growth of $\|u_t\|$ after Phase 2 | ✓ ($t_2 \approx 600$) | ✓ ($t_2 \approx 600$) |

### Ablation Study

- The consistency of the two-stage training dynamics is verified under different $\lambda$ configurations.
- The boundaries between Phase 1 and Phase 2 align with theoretical predictions.

## Highlights & Insights

1. **Establishes the first complete training dynamics theory for the even pairs problem**, filling a gap in the literature.
2. **Joint training analysis**: Unlike existing works that only analyze the attention layer, this study analyzes the coupled training process of both the attention layer and the linear layer.
3. **Surprising zero-shot CoT results**: Transformers trained on even pairs can solve parity check in a zero-shot manner, revealing deep connections between the two tasks.
4. **Implicit bias $\to$ max-margin**: Extends the classical implicit bias theory in classification tasks to language recognition tasks with unique structural features.
5. **Key role of the scaling factor**: $\lambda$ not only stabilizes training but also serves as the core mechanism controlling the transition between the two phases.

## Limitations & Future Work

- Only analyzes a **single-layer Transformer**; multi-layer scenarios remain unexplored.
- The experimental scale with a maximum sequence length $L_{\max}=6$ is relatively small, and scalability to longer sequences has not been verified.
- The theoretical analysis relies on the orthogonal embedding assumption, whereas practical token embeddings are usually non-orthogonal.
- More complex context-free languages or general automaton tasks are not discussed.
- The transition condition from Phase 1 to Phase 2 (the choice of $t_0$) relies on hyperparameter settings.

## Related Work & Insights

- **Training dynamics**: Tarzanagh et al. (2023) proved that attention training is equivalent to SVM; Vasudeva et al. (2024) established convergence rates.
- **CoT theory**: Kim & Suzuki (2024b) analyzed attention models learning parity with CoT; Wen et al. (2024) studied sample complexity but did not establish complete convergence.
- **Implicit bias**: Soudry et al. (2018) pioneered the theory of implicit bias of GD in classification; Huang et al. (2024a) extended it to NTP tasks.
- This work provides new analytical tools for Transformer theoretical analysis (the combination of high-order Taylor expansion and implicit bias), which can be generalized to other structured learning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First to completely characterize even pairs training dynamics, with novel zero-shot CoT results.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments validate the theory, but are small in scale.
- Writing Quality: ⭐⭐⭐⭐ Clear theory with a well-structured two-stage framework.
- Value: ⭐⭐⭐⭐ Provides an important theoretical foundation for understanding Transformer training mechanisms and CoT principles.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](../../NeurIPS2025/optimization/do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](can_transformers_learn_full_bayesian_inference_in_context.md)
- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICML 2025\] Transformative or Conservative? Conservation Laws for ResNets and Transformers](transformative_or_conservative_conservation_laws_for_resnets_and_transformers.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)

</div>

<!-- RELATED:END -->
