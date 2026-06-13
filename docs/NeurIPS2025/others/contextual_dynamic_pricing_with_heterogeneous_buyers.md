---
title: >-
  [Paper Note] Contextual Dynamic Pricing with Heterogeneous Buyers
description: >-
  [NeurIPS 2025][Dynamic Pricing] This paper presents the first systematic study of contextual dynamic pricing with heterogeneous buyers of $K_\star$ unknown types. It proposes an Optimistic Posterior Sampling (OPS)-based…
tags:
  - "NeurIPS 2025"
  - "Dynamic Pricing"
  - "Heterogeneous Buyers"
  - "Contextual Bandits"
  - "Posterior Sampling"
  - "Disagreement Coefficient"
  - "Lipschitz Bandits"
date: 2026-05-08
content_hash: cc35052a3b8503b5
---

# Contextual Dynamic Pricing with Heterogeneous Buyers

**Conference**: NeurIPS 2025
**arXiv**: [2512.09513](https://arxiv.org/abs/2512.09513)  
**Area**: Others
**Keywords**: Dynamic Pricing, Heterogeneous Buyers, Contextual Bandits, Posterior Sampling, Disagreement Coefficient, Lipschitz Bandits

## TL;DR
This paper presents the first systematic study of contextual dynamic pricing with heterogeneous buyers of $K_\star$ unknown types. It proposes an Optimistic Posterior Sampling (OPS)-based algorithm achieving an $\tilde{O}(K_\star\sqrt{dT})$ regret bound (optimal in $d$ and $T$), and further introduces ZoomV—a variance-aware adaptive discretization algorithm—achieving the optimal $\tilde{O}(\sqrt{K_\star T})$ regret in the non-contextual setting.

## Background & Motivation

**Background**: In online contextual pricing, a seller sets price $p_t$ for a product based on a $d$-dimensional feature vector $u_t$, and a buyer purchases if their valuation $v_t = \langle \theta^*, u_t \rangle$ meets or exceeds $p_t$. Existing algorithms—based on multidimensional binary search or regression—uniformly assume homogeneous buyers sharing a single type $\theta^*$.

**Limitations of Prior Work**: (a) Binary search methods cannot handle type-varying feedback, as knowledge sets are corrupted by contradictory observations; (b) regression methods assume i.i.d. noise, whereas the randomness induced by heterogeneous types is context-dependent; (c) the regret of naively discretized EXP4 explodes due to the infinite action space.

**Key Challenge**: The buyer type $\theta_t$ is unobservable, making it impossible to associate feedback with the uncertainty set of a specific type. Moreover, the revenue function under pricing is discontinuous—a price increase of $\epsilon$ may cause the buyer to abandon the purchase entirely.

**Key Insight**: The problem is formulated as learning the type distribution $D_\star \in \Delta(\Theta)$, exploiting the disagreement coefficient within the OPS framework to carefully control the exploration–exploitation tradeoff.

## Method

### Problem Setup
At each round $t$, the buyer type $\theta_t \sim D_\star$ with $|{\rm supp}(D_\star)| = K_\star$ unknown, valuation $v_t = \langle u_t, \theta_t \rangle$, and purchase decision $y_t = \mathbb{1}\{v_t \geq p_t\}$. The seller's objective is to minimize pricing regret:

$$R(T) = \sum_{t=1}^T \left[\mathsf{rev}_\star(\mathsf{br}_\star(u_t), u_t) - \mathsf{rev}_\star(p_t, u_t)\right]$$

where $\mathsf{rev}_Q(p) = p \cdot \mathsf{dem}_Q(p)$ is the expected revenue and $\mathsf{br}_Q$ is the optimal pricing policy.

### OPS for Finite Model Classes (Algorithm 1)
A posterior $\mu_t \in \Delta(\mathcal{D})$ over models is maintained. At each round:
1. Sample model $D_t \sim \mu_t$ and set price $p_t = \mathsf{br}_{D_t}(u_t)$.
2. Observe $y_t$ and update the posterior via a composite loss:

$$\ell_\lambda(Q, p, y) = \underbrace{(y - \mathsf{dem}_Q(p))^2}_{\text{model mismatch}} - \underbrace{\lambda \cdot \mathsf{rev}_Q(\mathsf{br}_Q)}_{\text{optimistic bias}}$$

The model mismatch term penalizes deviation between predictions and observations, while the optimistic bias encourages exploration of high-potential models.

### Controlling the Disagreement Coefficient (Core Technical Contribution)
The disagreement coefficient is defined as:

$$\mathsf{dis}(\mathcal{F}) = \sup_{\varepsilon, \delta > 0} \sup_{\nu \in \Delta(\mathcal{X})} \frac{\delta^2}{\varepsilon^2} \mathbb{P}_{p \sim \nu}\left(\exists f \in \mathcal{F}: \mathbb{E}_{q \sim \nu}[f(q)^2] \leq \varepsilon^2 \wedge |f(p)| > \delta\right)$$

**Key Insight**: For a fixed context $u$, the aggregate demand function $\mathsf{dem}_\star(\cdot, u)$ has at most $K_\star$ jump points, so $\mathsf{dem}_D(\cdot, u) - \mathsf{dem}_\star(\cdot, u)$ is monotonically decreasing on each of $K_\star + 1$ intervals. By:
- **Lemma 3.4**: $\mathsf{dis} \leq 2$ for the class of monotone decreasing functions;
- **Lemma 3.5**: the disagreement coefficient of an $N$-composite function class is amplified by at most a factor of $N$,

one obtains $\mathsf{dis}(\mathcal{D}, D_\star) \leq 2(K_\star + 1)$. Combined with the OPS regret bound $\tilde{O}(\lambda \cdot \mathsf{dis} \cdot T + \log|\mathcal{D}|/\lambda)$, Theorem 3.2 follows.

### Perturbed OPS for the General Setting (POPS, Algorithm 2)
To handle infinite model classes and unknown $K_\star$:
1. **Smoothed demand**: $\mathsf{dem}_Q^\varepsilon(p) = \mathbb{E}_{\delta \sim U[0,\varepsilon]}[\mathsf{dem}_Q(p - \delta)]$, which smooths discrete jumps via random perturbation.
2. **Price discretization with conservative perturbation**: $p_t = \max\{\hat{p}_t - \delta_t, 0\}$ with $\delta_t \sim U[0,\varepsilon]$, exploiting the one-sided Lipschitz property of revenue (a price increase yields at most $p' - p$ additional revenue) to bound the perturbation cost.
3. **Trajectory coupling**: It is shown that $D_\star$ has a Lévy-metric neighbor within a finite cover $\mathcal{D}$, ensuring the purchase sequences under POPS agree with high probability.
4. **Non-uniform prior for unknown $K_\star$**: Models with larger support sizes are assigned smaller prior weights.

### ZoomV for the Non-Contextual Setting (Algorithm 4)
When $d = 1$, variance-aware confidence intervals are combined with adaptive discretization:
- Standard zooming dimension $\text{ZoomDim}(c) \leq 1$ yields $\tilde{O}(T^{2/3})$.
- **Variance-aware zooming dimension**: using $\sigma^2(p) = p^2 \mathsf{dem}_\star(p)(1 - \mathsf{dem}_\star(p))$, confidence intervals become extremely narrow in high-price regions where demand is near zero for low-demand types.
- It is proven that $\text{ZoomDimV}(10K_\star) = 0$, yielding $\tilde{O}(\sqrt{K_\star T})$.

## Key Experimental Results

### Summary of Theoretical Regret Bounds

| Setting | Algorithm | Upper Bound | Lower Bound |
|---------|-----------|-------------|-------------|
| Contextual ($d>1$, $K_\star$ known) | OPS | $\tilde{O}(K_\star\sqrt{dT})$ | — |
| Contextual ($d>1$, $K_\star$ unknown) | POPS | $\tilde{O}(K_\star\sqrt{dT})$ | $\Omega(\sqrt{K_\star dT})$ |
| Non-contextual ($d=1$) | ZoomV | $\tilde{O}(\sqrt{K_\star T})$ | $\Omega(\sqrt{K_\star T})$ |
| Type-observable (identifier) | Alg 5 | $\tilde{O}(K_\star\sqrt{dT})$ | — |
| Type-observable (full vector) | Interpolation estimator | $\tilde{O}(\sqrt{\min\{K_\star, d\}T})$ | — |

### Key Findings
- The contextual regret $\tilde{O}(K_\star\sqrt{dT})$ is optimal in $d$ and $T$ (up to a $\sqrt{K_\star}$ factor).
- The non-contextual regret $\tilde{O}(\sqrt{K_\star T})$ is fully optimal, matching the lower bound.
- When types are fully observable, regret drops to $\tilde{O}(\sqrt{\min\{K_\star, d\}T})$, quantifying the substantial benefit of richer feedback.
- The computational complexity of POPS scales with the model class size (exponentially), constituting a theoretical bottleneck.

## Highlights & Insights
- The **composite disagreement coefficient decomposition** is the most elegant technical contribution: it reduces the exploration complexity over an infinite action space to the finite type count $K_\star$.
- The **one-sided Lipschitz property** is systematically exploited throughout: since raising prices only reduces demand, the regret cost of conservative pricing is bounded.
- **ZoomV's variance-aware dimension** reveals a special structure of the pricing problem: high-price intervals, though numerous, have near-zero variance and thus low regret contribution.
- The problem formulation is clean, providing an elegant connection between heterogeneous buyer pricing and contextual bandits.

## Limitations & Future Work
- The runtime of POPS scales with the size of the discretized model class, making it computationally infeasible for large $K_\star$ and $d$.
- A gap remains in the contextual setting regarding the optimal dependence on $K_\star$ (upper bound $K_\star$ vs. lower bound $\sqrt{K_\star}$).
- Noise robustness is limited: when the Lévy distance from $D_\star$ to the nearest $K_\star$-support distribution is $\delta$, regret incurs an additional $\sqrt{\delta} T^2$ term.
- Only linear valuation models are considered; extensions to nonlinear valuation structures are not discussed.
- The paper is purely theoretical with no empirical validation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic treatment of heterogeneous contextual pricing, with multiple novel technical tools.
- Theoretical Depth: ⭐⭐⭐⭐⭐ Upper and lower bound analyses across a hierarchy of methods (OPS/POPS/ZoomV).
- Experimental Thoroughness: ⭐⭐ Purely theoretical; no experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with progressive development; notation is heavy but consistent.
- Overall: ⭐⭐⭐⭐ An important theoretical breakthrough with broad implications for online learning and pricing.

## Related Work & Insights
- **vs. Cohen et al. (2020) / Lobel et al. (2018)**: Classical multidimensional binary search methods, restricted to homogeneous buyers ($K_\star = 1$). Contradictory feedback from heterogeneous types corrupts the knowledge set.
- **vs. Javanmard & Nazerzadeh (2019)**: Regression-based approach assuming i.i.d. noise, which cannot accommodate the context-dependent randomness introduced by heterogeneous types.
- **vs. Cesa-Bianchi et al. (2019)**: A pioneering work on non-contextual heterogeneous pricing, but its regret depends on an instance parameter $V$ that can blow up in the worst case. ZoomV eliminates this dependency via variance awareness.
- **vs. Krishnamurthy et al. (2023)**: Corruption-robust contextual search tolerating $C = o(T)$ corrupted responses, which does not apply to genuinely large-scale heterogeneous populations.
- The composite disagreement coefficient decomposition is transferable to other structured contextual bandit problems, such as auction design with piecewise linear revenue or heterogeneous user modeling in recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[ICCV 2025\] You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception](../../ICCV2025/others/you_share_beliefs_i_adapt_progressive_heterogeneous_collaborative_perception.md)
- [\[ICCV 2025\] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes](../../ICCV2025/others/adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](../../AAAI2026/others/area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->
