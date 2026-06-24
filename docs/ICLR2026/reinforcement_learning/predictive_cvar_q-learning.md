---
title: >-
  [Paper Note] Predictive CVaR Q-Learning
description: >-
  [ICLR 2026][Reinforcement Learning][CVaR] This paper proposes Predictive CVaR Q-learning (PCVaR-Q), which reformulates the CVaR objective—originally only evaluable at the end of a trajectory—into a step-by-step recursive Bellman form by introducing a pair of "predictive tail value/probability functions." Combined with a "bi-directional exploration" strategy that simultaneously explores actions and risk budgets, it significantly improves the sample efficiency and training stab…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "CVaR"
  - "Risk-Sensitive RL"
  - "Q-learning"
  - "Bellman equation"
  - "Sample efficiency"
date: 2026-05-08
content_hash: 12a65d7629fca3af
---

# Predictive CVaR Q-Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=B4SCegRJOA](https://openreview.net/forum?id=B4SCegRJOA)  
**Code**: None  
**Area**: Reinforcement Learning / Risk-Sensitive RL  
**Keywords**: CVaR, Risk-Sensitive RL, Q-learning, Bellman equation, Sample efficiency

## TL;DR
This paper proposes Predictive CVaR Q-learning (PCVaR-Q), which reformulates the CVaR objective—originally only evaluable at the end of a trajectory—into a step-by-step recursive Bellman form by introducing a pair of "predictive tail value/probability functions." Combined with a "bi-directional exploration" strategy that simultaneously explores actions and risk budgets, it significantly improves the sample efficiency and training stability of risk-sensitive RL, approaching the CVaR-optimal policy in both decision trees and stochastic grid worlds.

## Background & Motivation
**Background**: In high-risk sequential decision-making such as autonomous driving, robotic surgery, and finance, rare but catastrophic outcomes cannot be ignored. Standard RL optimizes expected returns and is risk-neutral, which is unsuitable. Among various risk measures, Conditional Value-at-Risk (CVaR, the expected loss within the worst $q$-quantile of the return distribution) has become the most common objective due to its mathematical tractability and focus on worst-case scenarios. Methods for optimizing CVaR are mainly divided into policy gradient and value-based methods.

**Limitations of Prior Work**: CVaR RL is notoriously sample-inefficient, usually attributed to its focus on a small fraction of worst-case trajectories. However, this paper points out two more fundamental issues. One is **noisy policy evaluation**: many value-based methods treat CVaR as a single, non-decomposable reward realized only at the end of an episode, represented as $-(\eta-R_{1:T})_+$. The learning signal for the entire trajectory is compressed into a delayed terminal outcome, preventing the agent from evaluating the immediate impact of each action and leading to extremely noisy evaluation. The second is **ineffective exploration**: because CVaR is driven by the worst-case scenarios, learning signals almost exclusively come from "failure" trajectories. High-return trajectories outside the low-risk quantiles are ignored, meaning the agent fails to learn how to continue improving successful behaviors—a phenomenon termed "blindness to success" in the literature, which stalls training on overly conservative sub-optimal policies.

**Key Challenge**: The CVaR objective $-(\eta-R_{1:T})_+$ is **non-decomposable** in time, delaying reward realization to the end of the horizon. Simultaneously, its non-zero effective rewards only appear in approximately a $(1-q)$ proportion of tail trajectories; the vast majority of trajectories have zero effective rewards and contribute nothing to learning. The lack of decomposability and tail sparsity are the root causes of the dilemma.

**Goal**: (1) Find a recursive structure for the CVaR objective that can propagate learning signals at every step to eliminate evaluation noise; (2) Design an exploration mechanism capable of escaping over-conservatism.

**Key Insight**: The authors shift the perspective from "terminal tail expectation" to "step-wise tail prediction." Since $-(\eta-R_{1:T})_+$ is hard to decompose, they instead recursively predict the "probability that the remaining return from the current step falls into the tail" and the "return value when it falls into the tail." These two quantities happen to satisfy risk-neutral Bellman recursions.

**Core Idea**: Rewrite the CVaR objective into a temporally decomposable Bellman recursion using a pair of "predictive tail value function $f^\chi$ + predictive tail probability function $g^\chi$," combined with randomized exploration of the risk budget $\eta$, transforming CVaR optimization into a trainable problem similar to standard Q-learning.

## Method

### Overall Architecture
PCVaR-Q is built upon the **variational representation** of CVaR and **state augmentation**. Given a risk level $q\in(0,1]$, CVaR has the variational form:

$$q\cdot \mathrm{CVaR}_q^\pi[R_{1:T}] = \max_{\eta\in\mathbb{R}}\Big\{q\eta + \mathbb{E}^\pi\big[-(\eta-R_{1:T})_+\big]\Big\},$$

This decomposes CVaR maximization into a two-level optimization over the tail budget $\eta$ (outer) and the strategy $\pi$ (inner). By introducing the "residual budget" as an augmented state $Y_t^\eta := \eta - R_{1:t-1}$, a Markov kernel $\chi_t:\mathcal{S}\times\mathbb{R}\to\Delta_{|\mathcal{A}|}$ is obtained on the augmented state space, where actions are selected according to $A_t\sim\chi_t(\cdot|S_t,Y_t^\eta)$. Prior work (Pflug & Pichler 2016, etc.) applied dynamic programming to the terminal reward $-(\eta-R_{1:T})_+$ in this augmented space, but sample efficiency was poor due to non-decomposability and sparse rewards.

Instead of the "terminal settlement" value function, this paper defines a pair of **predictive functions** to rewrite the learning objective, summarized in three steps: (1) Replacing the terminal value function $u^\chi$ with $f^\chi$ (tail value) and $g^\chi$ (tail probability) and proving they satisfy risk-neutral Bellman recursions, allowing learning signals to propagate step-by-step; (2) Fitting $\hat f_\theta$ and $\hat g_\phi$ using two sets of TD losses and periodically updating the risk budget $\eta$ within a Generalized Policy Iteration (GPI) framework; (3) Designing bi-directional exploration by randomizing actions ($\epsilon$-greedy) and the initial risk budget (sampled from $\mathcal{N}(\eta,\sigma_k^2)$). The three key designs correspond to these points.

### Key Designs

**1. Predictive Tail Value/Probability Functions: Reformulating Non-Decomposable CVaR into Recursive Bellman Form**

This is the theoretical foundation addressing the "noisy evaluation" issue. The authors define two functions (assuming $\eta=0$ and proving they are invariant to $\eta$): the predictive tail probability function

$$g_t^\chi(s,y,a) := \mathbb{P}^{\chi}\big(R_{t:T}\le y \mid S_t=s, Y_t=y, A_t=a\big),$$

characterizing the probability that the remaining return $R_{t:T}$ starting from $t$ falls below threshold $y$ (i.e., enters the CVaR tail); and the predictive tail value function

$$f_t^\chi(s,y,a) := \mathbb{E}^{\chi}\big[\mathbb{I}\{R_{t:T}\le y\}\,R_{t:T} \mid S_t=s, Y_t=y, A_t=a\big],$$

which is a risk-sensitive version of the standard action-value function (Q-function), capturing the remaining rewards weighted by the probability of staying in the tail, reflecting both the **magnitude and occurrence probability** of tail outcomes. Crucially, as $R_{t:T}$ is recursively decomposable, under Assumption 1 (the remaining return distribution has no probability mass points), the authors prove that $f^\chi$ satisfies a **Bellman equation with an immediate reward term** (Theorem 1):

$$f_t^\chi(s,y,a) = \mathbb{E}\Big[f_{t+1}^\chi(S_{t+1}, y-R_t, A_{t+1}) + g_{t+1}^\chi(S_{t+1}, y-R_t, A_{t+1})\cdot R_t\Big].$$

Compared to prior Bellman equations ($u_t^\chi = \mathbb{E}[u_{t+1}^\chi]$, lacking an immediate reward term), the extra $g_{t+1}^\chi\cdot R_t$ term corresponds to the immediate reward in standard Bellman equations, meaning the "expected contribution of the current reward to the final objective" is explicitly propagated back at each step, weighted by the tail probability. Thus, learning signals are dense across the trajectory rather than realized only at the end, significantly reducing evaluation noise. Proportionally, Proposition 1 provides the temporal decomposition $f_t^\chi = \mathbb{E}[\sum_{\tau\ge t} g_{\tau+1}^\chi\cdot R_\tau]$, $g_t^\chi=\mathbb{E}[g_{t+1}^\chi]$ (martingale property), and rewrites the objective as $\mathbb{E}[-(\eta-R_{1:T})_+] = \mathbb{E}_{A_1}[f_1^\chi(s_1,\eta,A_1) - g_1^\chi(s_1,\eta,A_1)\cdot\eta]$. On this structure, the authors further establish the Bellman optimality equation (Theorem 2) and a Policy Improvement Theorem (Theorem 3).

**2. Decoupled Estimation of Tail Probability and Value: Lower Variance and Stable Learning**

Decomposing CVaR into $g$ (probability) and $f$ (value) for separate estimation, rather than directly regressing tail expectation, is emphasized as an advantage addressing environments where "tail samples are sparse and high-variance." Direct regression of tail expectation requires fitting return magnitudes; when non-zero effective reward samples are rare and high-variance, training becomes unstable. Tail probability $g$ can be estimated more stably (in extreme cases, by counting the proportion of trajectories crossing a threshold), and $g\in[0,1]$ is naturally bounded, allowing stable learning objectives like log-loss or KL divergence. This explicit decoupling allows the probability component to learn more stably while the value component benefits from the variance reduction of the decomposition.

**3. Bi-directional Randomized Exploration + Periodic Risk Budget Updates: Exploring Risk Preferences to Solve "Blindness to Success"**

This addresses exploration difficulties. Conventional $\epsilon$-greedy only perturbs at the **action level**. This paper adds exploration in the **augmented state space**: the initial residual budget for each episode is not fixed but sampled from $Y_1\sim\mathcal{N}(\eta,\sigma_k^2)$, where $\eta$ is the current estimate of the optimal risk budget. By sampling around this center, the agent experiences trajectories under different risk sensitivities (sometimes aggressive, sometimes conservative), effectively exploring the "risk preference" dimension and avoiding premature convergence to overly safe sub-optimal policies. $\sigma_k$ is annealed throughout training. The training follows GPI: parameters $\theta, \phi$ are updated with two TD losses derived from Theorem 1 / Proposition 1:

$$L_f(\theta) = \tfrac{1}{B|H|}\sum_{\eta'\in H}\sum_j\Big(\hat f_j^\theta - [\hat f_{j+1}^\theta + \hat g_{j+1}^\phi\cdot R_j]\Big)^2,\quad L_g(\phi) = \tfrac{1}{B|H|}\sum_{\eta'\in H}\sum_j\Big(\hat g_{j+1}^\phi - \hat g_j^\phi\Big)^2,$$

A practical trick is used: losses for each sample are computed simultaneously across a set of candidate budgets $H\in\mathbb{R}$, allowing the function approximator to generalize across risk levels. The risk budget $\eta$ is updated every $c$ episodes via the outer variational optimization: $\eta\leftarrow\arg\max_{\eta'\in H}\max_a\{\hat f_1^\theta(s_1,\eta',a) + \eta'(q-\hat g_1^\phi(s_1,\eta',a))\}$. Optional warm-starting with existing trajectories (e.g., risk-neutral data) further mitigates early blindness-to-success.

### Loss & Training
The core training objectives are $L_f(\theta)$ and $L_g(\phi)$ (optimized via Adam). Actions are selected using $\epsilon$-greedy + the greedy kernel for $\hat f_\theta - \hat g_\phi\cdot Y_t$. Experiments use tabular function approximators with learning rates $\alpha_\theta=0.01, \alpha_\phi=0.0001$, $\epsilon_t=0.1\cdot0.9^{\lfloor t/100\rfloor}$, a batch size of 8 trajectories, and risk budget updates every 500 episodes. PCVaR-Q and the baseline CVaR-Q use identical hyperparameters for fairness.

## Key Experimental Results

Experiments compare three policies in two controllable environments: RN (Risk-Neutral Optimal), CVaR-Q (terminal settlement Q-learning baseline), and PCVaR-Q (Ours).

### Main Results: CVaR Performance ($q=0.1$)

| Environment | Metric | RN | PCVaR-Q (Ours) | CVaR Optimal |
|------|------|------|------|------|
| Sequential Decision Tree | CVaR@0.1 | 1.96 | **2.45** | 2.50 |
| Stochastic Grid World | CVaR@0.1 | −58.37 | **−55.84** | −53.34 |

In the decision tree, RN always chooses the high-expectation but high-risk "up" path (mean return 5.0, CVaR 1.96), while PCVaR-Q learns the risk-sensitive optimal policy, reaching a CVaR of 2.45, close to the theoretical 2.50. In the grid world (success rate 0.7, obstacle penalty $\mathcal{N}(-50,1)$, target reward $\mathcal{N}(50,1)$), PCVaR-Q learns a safer path to avoid obstacles, improving the tail from RN's −58.37 to −55.84, approaching the optimal −53.34.

### Stability / Sample Efficiency Analysis

| Configuration | Learning Curve Performance | Notes |
|------|------|------|
| PCVaR-Q | Smooth, fast, monotonic convergence to near-optimal | Mean of 10 runs |
| CVaR-Q (Baseline) | High variance, converges to sub-optimal | Significant fluctuations |

### Key Findings
- **Recursive structure is the source of stability**: Under identical hyperparameters, PCVaR-Q exhibits smooth convergence while CVaR-Q remains noisy and sub-optimal—confirming that the reformulated Bellman recursion effectively reduces evaluation noise.
- **Approaching theoretical optimum**: CVaR values in both environments remarkably approach theoretical limits, indicating the framework finds the true CVaR-optimal policy.
- **CVaR-Q can converge under ideal conditions**, but the robustness and stability of PCVaR-Q are the primary advantages under equivalent conditions.

## Highlights & Insights
- **Replacing "Terminal Settlement" with "Step-wise Tail Prediction"**: The introduction of $f,g$ pairs allows a non-decomposable CVaR objective to gain a Bellman recursion. The additional $g_{t+1}\cdot R_t$ term acts as an immediate reward, naturally extending Q-learning theory.
- **Engineering benefits of decoupling Probability/Value**: $g\in[0,1]$ is bounded and easier to train, transforming a "sparse+high-variance" tail problem into a manageable probability estimate and a variance-reduced value estimate.
- **Exploration in augmented state space**: Treating the risk budget $\eta$ as a "preference knob" addresses "blindness to success" by allowing the agent to experience varied risk-conservative trajectories.

## Limitations & Future Work
- **Increased model complexity**: Requires learning two function approximators and tracking the residual threshold.
- **Limited experimental scale**: Verified only on small tabular environments; performance in deep RL or high-dimensional control is unverified.
- **Dependence on Assumption 1**: The validity of the assumption in environments with deterministic rewards or discrete returns remains a question.
- **Future Work**: Scaling to deep RL, integrating with model-based planning, and applying to safety-critical real-world scenarios.

## Related Work & Insights
- **vs CVaR-Q (Pflug & Pichler 2016 / Wang et al. 2023)**: Baselines use non-decomposable terminal rewards; this work propagates learning signals at every step, improving efficiency.
- **vs Predictive CVaR Policy Gradient (Kim & Min 2024)**: That work uses risk-conditional probabilities for policy gradients; this work adapts it for value-based learning and action selection.
- **vs Distributional RL (Lim & Malik 2022, etc.)**: Instead of learning the whole distribution, this work focuses specifically on tail probability and value, providing a direct interface with Q-learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] Wavelet Predictive Representations for Non-Stationary Reinforcement Learning](wavelet_predictive_representations_for_non-stationary_reinforcement_learning.md)
- [\[ICLR 2026\] TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning](td-jepa_latent-predictive_representations_for_zero-shot_reinforcement_learning.md)
- [\[ICLR 2026\] RAMPS: Robust Adaptive Multi-step Predictive Shield](robust_adaptive_multi-step_predictive_shielding.md)
- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](../../AAAI2026/reinforcement_learning/deep_predictive_discounted_counterfactual_regret_minimization.md)

</div>

<!-- RELATED:END -->
