---
title: >-
  [Paper Note] On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning
description: >-
  [ICML 2025][Reinforcement Learning][FTRL] This paper proposes the OptFPRL algorithm, which introduces a **history gradient pruning** mechanism into the Follow the Regularized Leader (FTRL) framework. This establishes the first data-dependent dynamic regret guarantee for FTRL on compact sets, where the dynamic regret is fully controlled by the prediction error and can reach zero regret under perfect predictions.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "FTRL"
  - "Dynamic Regret"
  - "Online Convex Optimization"
  - "Optimistic Learning"
  - "History Pruning"
date: 2026-05-08
content_hash: 993e212ff243ef0f
---

# On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning

**Conference**: ICML 2025  
**arXiv**: [2505.22899](https://arxiv.org/abs/2505.22899)  
**Code**: Unreleased  
**Area**: Online Learning / Online Convex Optimization  
**Keywords**: FTRL, Dynamic Regret, Online Convex Optimization, Optimistic Learning, History Pruning  

## TL;DR

This paper proposes the OptFPRL algorithm, which introduces a **history gradient pruning** mechanism into the Follow the Regularized Leader (FTRL) framework. This establishes the first data-dependent dynamic regret guarantee for FTRL on compact sets, where the dynamic regret is fully controlled by the prediction error and can reach zero regret under perfect predictions.

## Background & Motivation

Online Convex Optimization (OCO) is a classic paradigm for sequential decision-making: at each time step $t$, the learner chooses an action $\bm{x}_t \in \mathcal{X}$, and the environment reveals a cost function $f_t(\cdot)$. The most stringent metric to evaluate the learner's performance is the **dynamic regret**:

$$\mathcal{R}_T = \sum_{t=1}^{T} f_t(\bm{x}_t) - f_t(\bm{u}_t)$$

where $\{\bm{u}_t\}$ is an arbitrary sequence of comparators, whose complexity is quantified by the **path length** $P_T = \sum_{t=1}^{T-1} \|\bm{u}_{t+1} - \bm{u}_t\|$.

The two main algorithmic families of OCO are **FTRL** and **OMD (Online Mirror Descent)**. For static regret (with a fixed comparator), both can achieve the optimal $\mathcal{O}(\sqrt{T})$ rate. However, regarding dynamic regret, FTRL has long been considered inferior to OMD due to the following reasons:

- **State bloating issue**: The state of FTRL is the cumulative gradient $\bm{g}_{1:t}$, which grows boundlessly over time.
- **Known negative results**: Jacobsen et al. proved that standard FTRL cannot achieve sublinear dynamic regret even when the path length is constant.
- **Lack of data-dependent bounds**: The previous best result for FTRL's dynamic regret was $\mathcal{O}(P^{1/3} T^{2/3})$ (Ahn et al.), which is data-independent and suboptimal.

The core insight of this paper is that **the dynamic regret bottleneck of FTRL does not stem from its "lazy projection" nature, but rather from the decoupling between the state (linearized history) and the iterates, which allows the state to grow infinitely**. By pruning to synchronize both, this issue can be resolved.

## Method

### Overall Architecture: OptFPRL

The update rule of OptFPRL is:

$$\bm{x}_{t+1} = \arg\min_{\bm{x}} \langle \bm{p}_{1:t}, \bm{x} \rangle + r_{1:t}(\bm{x}) + \tilde{f}_{t+1}(\bm{x}) + I_{\mathcal{X}}(\bm{x})$$

where $\bm{p}_{1:t}$ is the pruned state vector, $r_{1:t}(\bm{x}) = \frac{\sigma_{1:t}}{2}\|\bm{x}\|^2$ is the incremental regularization, $\tilde{f}_{t+1}$ is the prediction of the next-step cost, and $I_{\mathcal{X}}$ is the indicator function of the set.

### Key Designs: History Pruning Mechanism

At each step, the state vector is $\bm{p}_t = \bm{g}_t + \bm{g}_t^I$, where $\bm{g}_t \in \partial f_t(\bm{x}_t)$ is the subgradient of the cost, and $\bm{g}_t^I \in \mathcal{N}_{\mathcal{X}}(\bm{x}_t)$ belongs to the normal cone. The pruning rule is:

$$\bm{g}_t^I = \begin{cases} -(\bm{p}_{1:t-1} + \tilde{\bm{g}}_t + \sigma_{1:t-1}\bm{x}_t) & \text{if } \bm{x}_t^{\text{uc}} \notin \mathcal{X} \\ 0 & \text{otherwise} \end{cases}$$

Intuition: When the unconstrained iterate lies outside the feasible region (and is projected back to the boundary), it indicates that the accumulated state has pushed the iterate "too far". In this case, an element from the normal cone is chosen to **prune the cumulative history**, replacing the original $\bm{g}_{1:t}$ with a surrogate state $\bm{p}_{1:t}$ of smaller norm, while ensuring it generates the same iterate.

### Core Lemma: Optimistic Bounded State

**Lemma 4.3** proves that the norm of the pruned state is bounded:

$$\|\bm{p}_{1:t}\| \leq R\sigma_{1:t-1} + \epsilon_t$$

where $R$ is the set radius, and $\epsilon_t = \|\bm{g}_t - \tilde{\bm{g}}_t\|$ is the prediction error. This implies that the state growth is constrained by the regularization parameter (which can be controlled), rather than growing linearly with $t$ as in standard FTRL.

### Four Regularization Strategies and Their Regret Bounds

1. **$P_T$-Independent Regularization** (Theorem 3.1): $\sigma_{1:t} \propto \sqrt{E_t}$
   $$\mathcal{R}_T \leq (5.8R + \tfrac{1}{2}P_T)\sqrt{E_T} + H_T = \mathcal{O}((1+P_T)\sqrt{E_T})$$

2. **Regularization with Known $P_T$** (Theorem 3.2): $\sigma \propto 1/\sqrt{P_T}$
   $$\mathcal{R}_T = \mathcal{O}((1+\sqrt{P_T})\sqrt{E_T})$$
   This matches the minimax optimal rate $\mathcal{O}(\sqrt{(1+P_T)T})$.

3. **Unknown but Observable $P_T$** (Theorem 3.3): Online estimation of $\sqrt{E_t/P_t}$
   $$\mathcal{R}_T = \mathcal{O}((1+\sqrt{P_T})\sqrt{E_T} + A_T)$$

4. **Recurrent Regularization (AdaFTRL-style)** (Theorem 3.4): $\sigma_t \propto \delta_t$ (local regret)
   $$\mathcal{R}_T \leq 1.1\,\delta_{1:T} + \sum_{t=1}^{T-1}\tfrac{1}{4R}\delta_{1:t}\|\bm{u}_{t+1}-\bm{u}_t\| + H_T$$
   This provides a more refined bound, since $\delta_{1:T} \leq \mathcal{O}(\sqrt{E_T})$ but is typically much smaller.

### Loss & Training

This is an analytical work and does not involve training losses. The core inequality of the analysis comes from the **Strong Dynamic Optimistic FTRL Lemma** (Lemma 4.1), which decomposes the regret into three parts:
- **(I)** Penalty for choosing $\bm{x}_t$ without knowing $\bm{g}_t$ $\rightarrow$ controlled by the prediction error $\epsilon_t$
- **(II)** Penalty for comparator non-stationarity $\rightarrow$ controlled by $\|\bm{p}_{1:t}\| \cdot \|\bm{u}_{t+1}-\bm{u}_t\|$
- **Regularization term** $r_t(\bm{u}_t)$ $\rightarrow$ balances the first two terms

## Key Experimental Results

This is a purely theoretical paper with no numerical experiments. A comparison of the main theoretical contributions is as follows:

| Method | Dynamic Regret Bound | Under Perfect Prediction | FTRL? |
|------|-----------|-----------|-------|
| Jadbabaie et al. (OMD) | $\mathcal{O}((1+P_T)\sqrt{E_T+1})$ | $\mathcal{O}(P_T)$ | ✗ |
| Scroccaro et al. (OMD) | $\mathcal{O}((1+P_T)(1+\sqrt{D_T}))$ | $\mathcal{O}(1+P_T)$ | ✗ |
| Zhang et al. (meta) | $\mathcal{O}(\sqrt{T(1+P_T)})$ | $\mathcal{O}(\sqrt{T(1+P_T)})$ | ✗ |
| Ahn et al. (FTRL) | $\mathcal{O}(P^{1/3}T^{2/3})$ | $\mathcal{O}(P^{1/3}T^{2/3})$ | ✓ |
| **OptFPRL (Ours)** | $\mathcal{O}((1+\sqrt{P_T})\sqrt{E_T})$ | **0** | ✓ |

### Key Findings

- **First minimax optimal dynamic regret guarantee for FTRL**: Matches the $\Omega(\sqrt{(1+P_T)T})$ lower bound when $P_T$ is known.
- **Zero regret under perfect prediction**: All regret terms (including $P_T$) are scaled by the prediction error, whereas the $P_T$ term in prior OMD methods was independent of prediction quality.
- **Mixed term $H_T$**: $H_T = \sum_t \epsilon_t \|\bm{u}_{t+1}-\bm{u}_t\|$ precisely characterizes the interaction between prediction error and environmental changes.

## Highlights & Insights

1. **FTRL is not "inherently disadvantaged" in dynamic environments**: It was previously believed that FTRL's "lazy" updates are unsuitable for non-stationary environments. This work proves that the bottleneck lies in state growth rather than the projection mechanism.
2. **Pruning = Selective Forgetting**: By cleverly "pruning" redundant gradients from the accumulated history using normal cone elements, FTRL essentially achieves selective forgetting.
3. **Dual Perspective**: From a dual perspective, OptFPRL is equivalent to only keeping the explicit gradient history since the last pruning, while the prior history is implicitly encoded through the dual map at $\bm{x}_{t-k}$.
4. **Advantages of Recurrent Regularization**: AdaFTRL-style regularization provides an "exactly right" amount of regularization, showing greater advantages in dynamic environments (since the regularization terms are nested inside the summation over $[T]$).

## Limitations & Future Work

1. **Purely theoretical work**: No numerical experiments are conducted to verify practical performance gains.
2. **Bounded set assumption**: The results rely on $\mathcal{X}$ being bounded ($\|\bm{x}\| \leq R$); unbounded domains would require different techniques.
3. **Requires prediction $\tilde{f}_{t+1}$**: How to obtain high-quality predictions in practical scenarios remains an independent challenge.
4. **Euclidean regularizer**: Only scaled Euclidean regularizers are considered; exploring more general Bregman divergences is a valuable direction.
5. **Assumption of known $P_T$**: The optimal rate requires knowing or online monitoring of $P_T$ beforehand.

## Related Work & Insights

- **Optimistic Online Learning**: From AdaGrad $\rightarrow$ Optimistic FTRL/OMD $\rightarrow$ Data-dependent regret bounds.
- **OMD-line for Dynamic Regret**: Jadbabaie et al., Zhang et al. (meta-learning), Zhao et al., Scroccaro et al.
- **FTRL for Dynamic Regret**: Ahn et al. (geometric discounting), Jacobsen et al. (centered OMD).
- **Connections between OMD and FTRL**: Fang et al. (Dual Averaging $\approx$ improved OMD), McMahan survey.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contribution is significant and elegant—establishing the first minimax optimal dynamic regret guarantee for FTRL on compact sets, with a pruning mechanism that is design-wise simple yet profound. While the lack of empirical validation is a limitation, it remains acceptable for a purely theoretical work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](../../NeurIPS2025/reinforcement_learning/dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[ICML 2025\] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](../../ICML2026/reinforcement_learning/parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](../../NeurIPS2025/reinforcement_learning/generalizing_verifiable_instruction_following.md)
- [\[ICML 2025\] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)

</div>

<!-- RELATED:END -->
