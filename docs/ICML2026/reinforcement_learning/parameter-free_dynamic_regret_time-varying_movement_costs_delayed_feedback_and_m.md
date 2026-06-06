---
title: >-
  [Paper Note] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory
description: >-
  [ICML2026][Reinforcement Learning][Dynamic Regret] This work provides the first parameter-free algorithm for **unconstrained** Online Convex Optimization (OCO) under **time-varying movement costs** and **dynamic comparat…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Dynamic Regret"
  - "Movement Cost"
  - "Delayed Feedback"
  - "Memory in Online Learning"
  - "Unconstrained OCO"
date: 2026-05-08
content_hash: ffebaff9325a0b88
---

# Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory

**Conference**: ICML2026  
**arXiv**: [2602.06902](https://arxiv.org/abs/2602.06902)  
**Code**: None (Theoretical paper)  
**Area**: Online Convex Optimization / Dynamic Regret / Parameter-free Algorithms  
**Keywords**: Dynamic Regret, Movement Cost, Delayed Feedback, Memory in Online Learning, Unconstrained OCO  

## TL;DR
This work provides the first parameter-free algorithm for **unconstrained** Online Convex Optimization (OCO) under **time-varying movement costs** and **dynamic comparator sequences**. It unifies and improves dynamic regret bounds for three scenarios—movement costs, delayed feedback, and time-varying memory—by reducing the latter two to the time-varying movement cost setting.

## Background & Motivation

**Background**: Standard OCO assumes bounded decision domains and static optimal points, measuring static regret $R_T = \sum_t f_t(w_t) - \min_u \sum_t f_t(u)$. However, real-world applications—portfolio optimization, video streaming, load balancing, and optimal control—violate these assumptions: domains are unconstrained (leverage/shorting), objectives drift (market structural changes), and adjusting decisions incurs movement costs proportional to $\|w_t - w_{t-1}\|$ (transaction fees, switching costs), where the cost coefficient $\lambda_t$ fluctuates significantly (liquidity, volatility).

**Limitations of Prior Work**: These three directions have been studied in silos: Zhang et al. (2022) handle unconstrained plus movement costs but focus on static regret; Zhang et al. (2021) provide dynamic regret for bounded domains with fixed movement costs; Wan et al. (2024) address dynamic regret with delayed feedback in bounded domains, requiring "in-order arrival" to tighten the delay dependence from $d_{\max}T$ to $d_{\mathrm{tot}}$. The intersection—**unconstrained + dynamic + time-varying movement costs**—remained an open problem.

**Key Challenge**: Tension exists between being parameter-free (not knowing the comparator norm $M$ and path length $P_T$ in advance) and operating in unconstrained domains. Without bounded domains, $M$ has no prior upper bound, yet algorithms must adapt to $M$ and $P_T$. Movement costs intensify this; they encourage stability, while parameter-free methods often require aggressive non-strongly convex regularizers to explore unknown radii.

**Goal**: (1) Design the first parameter-free algorithm for unconstrained OCO with time-varying movement costs; (2) prove adaptability of regret to $M$, $P_T$, $\{\lambda_t\}$, and $\{\|g_t\|\}$; (3) reduce delayed feedback and time-varying memory to this unified framework.

**Key Insight**: In the composite mirror descent framework of Jacobsen & Cutkosky (2022), a correction term $\varphi_t(w)$ is used to stabilize unconstrained mirror descent. By extending the coefficient of this term from $\eta\|g_t\|^2$ to $\eta(\|g_t\|+\lambda_{t+1})^2$, movement costs are encoded as "dynamic friction coefficients"—the algorithm automatically reduces step size when high future movement costs are anticipated.

**Core Idea**: Reformulate the regularizer with $\beta_t = \|g_t\| + \lambda_{t+1}$ to lift unconstrained parameter-free tools to the time-varying movement cost setting. Then, use adaptive batching to compress second-order $\lambda_t^2$ dependence into first-order $\lambda_t\|g_t\|$ dependence, ensuring movement costs do not penalize excessively in mild environments with small gradients.

## Method

### Overall Architecture
The architecture follows a three-layer stack:

1.  **Base Layer (Algorithm 1)**: Composite Mirror Descent operating on a single learning rate $\eta$, using a log-linear regularizer $\psi(w) = \tfrac{2}{\eta}\int_0^{\|w\|}\log(x/\alpha+1)\,dx$ with a time-varying correction term $\varphi_t(w) = (\eta\beta_t^2 + \gamma)\|w\|$.
2.  **Meta Algorithm Layer (Algorithm 2)**: Maintains $\mathcal{O}(\log T)$ parallel instances of Algorithm 1 with learning rates on a geometric grid $\eta_i = 2^i / (L\sqrt{T})$. It uses a standard hedging trick where outputs are **directly summed**, ensuring regret adapts to the optimal learning rate plus an $\mathcal{O}(\log T)$ term.
3.  **Batching Layer (Algorithm 3)**: An adaptive epoch partitioning scheme over Algorithm 2. It triggers mirror descent updates only when the cumulative gradient $\|H_\tau\| = \|\sum_{t\in I_\tau} g_t\|$ exceeds the current movement cost $\lambda_{t+1}$; otherwise, the decision is frozen. This is key to achieving first-order $\lambda_t\|g_t\|$ dependence.

Section 5 uses two independent reductions (Algorithm 4 and 5) to translate delayed feedback and time-varying memory into inputs for Algorithm 3.

### Key Designs

1.  **Encoding Time-varying Movement Costs into Regularizers**:
    - **Function**: Forces single-instance mirror descent to be cautious when facing high future movement costs.
    - **Mechanism**: Define $\beta_t \triangleq \|g_t\| + \lambda_{t+1}$ and correction $\varphi_t(w) = (\eta\beta_t^2 + \gamma)\|w\|$. The update is $w_{t+1} = \arg\min_w \langle g_t, w\rangle + D_\psi(w \mid w_t) + \varphi_t(w)$. $\varphi_t$ acts as a "return-to-origin spring" whose stiffness scales with gradients and movement costs.
    - **Design Motivation**: Extends Jacobsen & Cutkosky (2022) by observing that the stability term $\varphi_t$ coefficient can be enlarged to embed extra $\lambda_{t+1}^2$ terms without breaking parameter-free properties.

2.  **Adaptive First-order Batching (Algorithm 3)**:
    - **Function**: Improves the regret's leading term from $\sqrt{\sum_t (\|g_t\|^2 + \lambda_t^2)\|u_t\|}$ to $\sqrt{\sum_t (\|g_t\|^2 + \lambda_t\|g_t\|)\|u_t\|}$.
    - **Mechanism**: Maintains a cumulative buffer $H_\tau$. Decisions $w_t = \tilde{w}_\tau$ remain unchanged while accumulating $g_t$. Updates occur only when $\|H_\tau\| > \lambda_{t+1}$.
    - **Design Motivation**: Adapts the batching strategy of Zhang et al. (2022b) to time-varying $\lambda_t$. Remark 4.3 proves this first-order bound is **never worse** than the second-order one and strictly tighter when $\|g_t\| \ll \lambda_t$.

3.  **Reductions: Delays/Memory → Movement Costs**:
    - **Function**: Mechanically translates issues unrelated to movement costs into Algorithm 3's input.
    - **Mechanism (Delayed Feedback)**: Lemma 5.1 bounds $R_T^{\mathrm{del}}(u_{1:T}) \le \sum_t \langle \sum_{\tau\in o_{t+1}\setminus o_t} g_\tau, w_t - u_t\rangle + G\sum_t |m_t|\|w_t - w_{t-1}\| + GP_T \sigma_{\max}$. Treating the collection of already arrived gradients as a pseudo-gradient and $\lambda_t = G|m_t|$ as a pseudo-movement cost yields $\widetilde{\mathcal{O}}(\sqrt{(M^2 + MP_T)(T + d_{\mathrm{tot}})})$ regret.
    - **Mechanism (Time-varying Memory)**: Lemma 5.6 uses a unary loss $\hat{f}_t(w) = f_t(w,\dots,w)$ to derive $R_T^{\mathrm{mem}} \le \sum_t \langle h_t, w_t - u_t\rangle + G\sum_t \xi_t \|w_t - w_{t-1}\| + GP_T B^2$, where $\xi_t$ is a computable value related to future memory length.
    - **Design Motivation**: Proves that "unconstrained + time-varying movement cost" is a **primitive**. The delay reduction is notable as it bypasses the "in-order arrival" requirement for $d_{\mathrm{tot}}$ dependence.

### Loss & Training
This is a theoretical paper providing worst-case regret bounds. Implementation complexity is $\mathcal{O}(d\log T)$ per round. It requires knowing $L \ge G + 2\lambda_{\max}$, though Remark 4.2 uses a doubling trick to remove this dependence.

## Key Experimental Results

### Main Results: Dynamic Regret Bounds for Three Settings

| Setting | Regret Bound (Ours) | Domain | Key Adaptation |
| :--- | :--- | :--- | :--- |
| Unconstrained OCO + TV Movement Cost (Th 4.1) | $\widetilde{\mathcal{O}}\bigl(\sqrt{(M+P_T)\sum_t(\|g_t\|^2 + \lambda_t\|g_t\|)\|u_t\|}\bigr)$ | $\mathbb{R}^n$ | $M, P_T, \|g_t\|, \lambda_t, \|u_t\|$ |
| Unconstrained OCO + Delayed Feedback (Th 5.2) | $\widetilde{\mathcal{O}}\bigl(\sqrt{(M^2 + MP_T)(T + d_{\mathrm{tot}})}\bigr)$ | $\mathbb{R}^n$ | $M, P_T, d_{\mathrm{tot}}$ |
| Unconstrained OCO + TV Memory $b_t$ (Th 5.7) | $\widetilde{\mathcal{O}}\bigl(\sqrt{(M^2 + MP_T)(H^2 T + GH\sum_t b_t^2)}\bigr)$ | $\mathbb{R}^n$ | $M, P_T, \{b_t\}, G, H$ |

All results recover known optimal dynamic regret $\widetilde{\mathcal{O}}(\sqrt{(M+P_T)\sum_t\|g_t\|^2\|u_t\|})$ when movement costs/delays/memory are zero.

### Ablation Study: Comparison with Prior Work

| Setting | Prev. SOTA | Ours | Gain |
| :--- | :--- | :--- | :--- |
| Static unconstrained + Move Cost (Zhang 2022b) | $\sqrt{G^2 T + \lambda GT}$ | $\sqrt{\sum_t\|g_t\|^2 + \lambda \sum_t \|g_t\|}$ | Problem-dependent adaptation vs worst-case $G^2T$ |
| Dynamic bounded + Fixed Memory (Zhao 2023) | $\sqrt{(1+P_T)(\sqrt{G}H^2 B + GHB^2)T}$ | $\sqrt{(M^2+MP_T)(H^2 T + GH\sum_t b_t^2)}$ | Unconstrained domain + time-varying memory |
| Dynamic bounded + Delay (Wan 2024) | $\sqrt{(1+P_T)(T + d_{\max}T)}$ or $\sqrt{(1+P_T)(T+d_{\mathrm{tot}})}$ (In-order) | $\sqrt{(M^2+MP_T)(T + d_{\mathrm{tot}})}$ | No in-order assumption + unconstrained |
| Static unconstrained + Delay (van der Hoeven 2022) | Second-order "lag" dependence (Static) | Dynamic version (with $P_T$) | Support for comparator drift |

### Key Findings
- First-order movement cost dependence $\lambda_t\|g_t\|$ is never worse than second-order $\lambda_t^2$ and tightens strictly as $\|g_t\|\to 0$.
- Delay is fundamentally a "penalty for missing information," where $\lambda_t = G|m_t|$ effectively translates missing gradients into movement penalties.
- In memory-constrained settings, the $b_t \equiv B$ case recovers the known minimax lower bound $\Omega(B\sqrt{(1+P_T)T})$ and extends it to the динамический regret bound.

## Highlights & Insights
- **"Movement Cost as a Hidden OCO Primitive"**: This work's insight that time-varying movement cost is an atomic primitive for other structures (delays, memory) is a paradigm shift.
- **Extensibility of $\varphi_t(w)$**: Re-scaling the correction term $\beta_t$ inherits parameter-free benefits without breaking the Jacobsen-Cutkosky analysis skeleton.
- **Coexistence of Batching and Parameter-free**: Proves that adaptive triggering ($\|H_\tau\| > \lambda_{t+1}$) allows batching and parameter-free exploration to coexist and benefit each other.

## Limitations & Future Work
- **Limitations**: Pure movement penalties are unavoidable if the environment is strictly static ($g_t=0$) but $\lambda_t$ is large.
- **Implicit Assumptions**: $\lambda_{t+1}$ must be known at time $t$, which fits finance but not adversarial cost settings.
- **Future Work**: Combining delays and memory into a single setting; implicit mirror descent variants for smoother $\beta_t$ transitions; high-probability regret extensions.

## Related Work & Insights
- **vs Jacobsen & Cutkosky (2022)**: Adds movement costs to their unconstrained dynamic regret framework.
- **vs Zhang et al. (2022b)**: Generalizes from static/fixed to dynamic/time-varying.
- **vs Wan et al. (2024)**: Bypasses "in-order arrival" via reduction to movement costs.
- **vs Zhao et al. (2023)**: Handles unconstrained domains and time-varying memory simultaneously.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First in intersection of unconstrained + dynamic + TV costs; establishes reduction primitives.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical paper, no numerical experiments; logic is verified via optimality and recovery of known bounds.
- Writing Quality: ⭐⭐⭐⭐ Clear three-layer stack; provides insightful remarks but notation is heavy.
- Value: ⭐⭐⭐⭐⭐ Refreshes SOTA for three sub-areas; essential reading for OCO theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](../../NeurIPS2025/reinforcement_learning/bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](../../NeurIPS2025/reinforcement_learning/dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](../../AAAI2026/reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)
- [\[ICML 2026\] Flow-Equivariant World Models: Memory for Partially Observed Dynamic Environments](flow_equivariant_world_models_memory_for_partially_observed_dynamic_environments.md)
- [\[ICML 2026\] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)

</div>

<!-- RELATED:END -->
