---
title: >-
  [Paper Note] Bandit and Delayed Feedback in Online Structured Prediction
description: >-
  [NeurIPS 2025][Reinforcement Learning][Online Structured Prediction] This paper is the first to study bandit and delayed feedback settings in online structured prediction. By designing a novel pseudo-inverse matrix gradi…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Online Structured Prediction"
  - "Bandit Feedback"
  - "Delayed Feedback"
  - "Surrogate Regret"
  - "Fenchel-Young Loss"
date: 2026-05-08
content_hash: a6dc3e0227ea1eee
---

# Bandit and Delayed Feedback in Online Structured Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2502.18709](https://arxiv.org/abs/2502.18709)  
**Code**: None  
**Area**: Online Learning / Structured Prediction
**Keywords**: Online Structured Prediction, Bandit Feedback, Delayed Feedback, Surrogate Regret, Fenchel-Young Loss

## TL;DR

This paper is the first to study bandit and delayed feedback settings in online structured prediction. By designing a novel pseudo-inverse matrix gradient estimator, it achieves an $O(T^{2/3})$ surrogate regret bound that does not explicitly depend on the output set size $K$.

## Background & Motivation

Online structured prediction is the task of sequentially predicting outputs with complex structure, encompassing problems such as online classification, multi-label classification, and ranking. Given an input space $\mathcal{X}$ and a finite output space $\mathcal{Y}$ (with $K = |\mathcal{Y}|$), the learner must predict an output based on the input at each round and incur a target loss $L(\hat{y}_t; y_t)$.

**Surrogate Loss Framework**: Since directly optimizing the target loss over a structured output space is generally intractable, surrogate losses such as the Fenchel-Young loss are employed in practice. Surrogate regret measures the cumulative excess of the learner's target loss over the optimal surrogate loss.

**Limitations of Prior Work**:
- Prior work (Sakaue et al., 2024) achieves bounded surrogate regret only under full-information feedback.
- Full-information feedback requires immediate access to the complete structured output, which is often impractical.
- For example, in online advertising ranking, only click-through rates can be observed (bandit feedback), not the identities of which ads were clicked.

**Core Challenges**:
1. Under bandit feedback, the true output $y_t$ is unobserved, making it impossible to compute the true gradient of the surrogate loss.
2. When the output set is extremely large (e.g., $K = \binom{d}{m}$ in multi-label classification or $K = m!$ in ranking), bounds depending on $K$ are highly unfavorable.
3. Delayed feedback introduces additional complexity into algorithm design.

## Method

### Overall Architecture

This paper builds upon the online structured prediction framework with Fenchel-Young losses and develops four classes of algorithms to handle different combinations of feedback and delay settings. The core pipeline is: receive input → compute score vector via linear estimator → generate prediction via randomized decoding → receive feedback → estimate gradient and update parameters.

### Key Designs

**1. Randomized Decoding with Uniform Exploration (RDUE)**:
- With probability $q$, select an output uniformly at random from $\mathcal{Y}$.
- With probability $1-q$, use standard randomized decoding.
- This guarantees that each output is selected with probability at least $q/K$, providing the basis for inverse-weighting estimation.
- The loss satisfies: $\mathbb{E}[L(\psi_\Omega(\theta); y)] \leq \frac{4\gamma}{\lambda\nu}(1-q)S_\Omega(\theta; y) + q\frac{K-1}{K}$

**2. Inverse-Weighting Gradient Estimator** ($O(\sqrt{KT})$ algorithm):
$$\tilde{G}_t = \frac{\mathbb{1}[\hat{y}_t = y_t]}{p_t(y_t)} G_t$$
This estimator is unbiased, but its variance is proportional to $K/q$, causing the regret bound to depend on $K$.

**3. Pseudo-Inverse Matrix Gradient Estimator** ($O(T^{2/3})$ algorithm, core contribution):

For the special subclass of Structured Encoding Loss Functions (SELF), the following surrogate label is constructed:
$$\tilde{y}_t = V^{-1} P_t^+ \hat{y}_t \langle \hat{y}_t, Vy_t \rangle$$
where $P_t = \mathbb{E}[\hat{y}_t \hat{y}_t^\top]$ is the second-moment matrix. The gradient estimator is then defined as:
$$\tilde{G}_t = (\hat{y}_\Omega(\theta_t) - \tilde{y}_t) x_t^\top$$

Key property: $\mathbb{E}_t[\|\tilde{G}_t\|_F^2] \leq 2bS_t(W_t) + 2C_x^2 \omega / q$, where $\omega = \text{tr}(V^{-1}Q^+(V^{-1})^\top)$ does not depend on $K$.

**4. Handling Delayed Feedback**:
- The ODAFTRL (Optimistic Delayed Adaptive FTRL) algorithm is employed to handle fixed $D$-step delays.
- This yields an AdaGrad-type regret bound that fully exploits the adaptive structure of the gradients.

### Loss & Training

- **Surrogate Loss**: Fenchel-Young loss $S_\Omega(\theta; y) = \Omega^*(\theta) + \Omega(y) - \langle\theta, y\rangle$
- **Target Loss**: Structured Encoding Loss Function (SELF), $L(y_t; \hat{y}_t) = \langle\hat{y}_t, Vy_t + b\rangle + c(y_t)$
- **Update Strategy**: Adaptive online gradient descent (bandit setting) or ODAFTRL (delayed setting), with learning rate $\eta_t = B / \sqrt{2\sum_{i=1}^t \|\tilde{G}_i\|_F^2}$

## Key Experimental Results

### Main Results: Summary of Surrogate Regret Bounds

| Problem Setting | Feedback Type | Target Loss | Surrogate Regret Bound | Source |
|----------------|--------------|-------------|----------------------|--------|
| Multi-class classification | Bandit | 0-1 loss | $O(\sqrt{KT})$ | Van der Hoeven et al. (2021) |
| Structured prediction | Full-information | SELF | Finite bound | Sakaue et al. (2024) |
| **Structured prediction** | **Bandit** | **SELF** | $O(\sqrt{KT})$ | **Ours (Thm 3.5, 3.6)** |
| **Structured prediction** | **Bandit** | **SELF*** | $O(T^{2/3})$ | **Ours (Thm 3.8)** |
| **Structured prediction** | **Full-info + Delay** | **SELF** | $O(D^{2/3}T^{1/3})$ | **Ours (Thm 4.3, 4.4)** |
| **Structured prediction** | **Bandit + Delay** | **SELF** | $O(\sqrt{DKT})$ | **Ours (Thm 5.1)** |
| **Structured prediction** | **Bandit + Delay** | **SELF*** | $O(D^{1/3}T^{2/3})$ | **Ours (Thm 5.2)** |

### Ablation Study: Numerical Comparison under Bandit Feedback (MNIST)

| Algorithm | Surrogate Loss | Misclassification Rate (Median) | Designed for Multi-class Classification |
|-----------|---------------|--------------------------------|----------------------------------------|
| Gaptron | Hinge loss | Higher | Yes |
| Gappletron | Logistic loss | Moderate | Yes |
| **Ours** | **FY loss** | **Lowest** | **No (general structured prediction)** |

On the MNIST dataset ($K=10$), the proposed algorithm outperforms methods specifically designed for multi-class classification, despite being designed for general structured prediction.

### Key Findings

1. **$K$-independent bound**: The pseudo-inverse matrix estimator eliminates explicit dependence on $K$ in the regret bound. This improvement is particularly significant in multi-label classification ($K = \binom{d}{m}$) and ranking ($K = m!$).
2. **Cost of $O(T^{2/3})$**: Eliminating the $K$ dependence comes at the cost of increasing the exponent of $T$ from $1/2$ to $2/3$. This trade-off is favorable when $K$ is extremely large.
3. **Effect of delay**: A fixed delay of $D$ introduces a multiplicative factor of $D^{1/3}$ (pseudo-inverse estimator) or $\sqrt{D}$ (inverse-weighting estimator).
4. **Synthetic data experiments**: The proposed method is competitive when $K \leq 24$; for $K = 48, 96$, specialized algorithms hold an advantage due to the use of different surrogate losses.
5. **High-probability bounds**: Theorems 3.6 and 4.4 establish high-probability regret bounds via Bernstein's inequality.

## Highlights & Insights

1. **Elegant design of the pseudo-inverse estimator**: Unlike the standard inverse-weighting estimator, the pseudo-inverse estimator cleverly exploits the structure of SELF, ensuring that the second moment of the gradient estimator does not depend on $K$.
2. **Completeness of the theoretical framework**: The paper systematically covers three feedback scenarios—bandit, delayed, and bandit with delay—as well as two classes of loss functions, SELF and SELF*.
3. **Practical relevance**: Settings such as online advertising ranking and recommendation systems typically involve limited (bandit) or delayed feedback, making the proposed methods directly applicable.

## Limitations & Future Work

1. **Absence of lower bounds**: Lower bounds on surrogate regret under bandit and delayed feedback have not been established, and the optimal dependence on $d$ remains unknown.
2. **Fixed delay assumption**: Only a fixed delay $D$ is considered, whereas delays in practice are typically variable.
3. **Optimality of $O(T^{2/3})$**: Whether $T^{2/3}$ can be improved to $\sqrt{T}$ (at the cost of some polynomial dependence on $K$) remains an open question.
4. **Linear estimator restriction**: The model is assumed to be a linear estimator; extensions to deep learning settings warrant further exploration.
5. **Restrictiveness of SELF***: The $O(T^{2/3})$ bound holds only for a subclass of SELF; general SELF still requires dependence on $K$.

## Related Work & Insights

- **Online multi-class classification**: Van der Hoeven (2020) introduced the concept of surrogate regret; Gaptron and Gappletron are representative algorithms.
- **Combinatorial bandits**: The pseudo-inverse estimator of Cesa-Bianchi & Lugosi (2012) inspired the design in this paper.
- **Delayed online learning**: Flaspohler et al. (2021)'s ODAFTRL provides the core algorithmic foundation for handling delayed feedback.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 3 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](../../AAAI2026/reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)

</div>

<!-- RELATED:END -->
