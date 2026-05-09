---
title: >-
  [Paper Note] Lipschitz Bandits with Stochastic Delayed Feedback
description: >-
  [ICLR 2026][Lipschitz bandit] This paper provides the first systematic study of Lipschitz bandits over continuous arm spaces under stochastic delayed feedback. For bounded delays, it proposes the Delayed Zooming algorithm, which employs a lazy update mechanism to maintain the suboptimality gap bound $\Delta(x) \leq 6r_t(x)$. For unbounded delays, it proposes DLPP, a phased pruning strategy whose regret is tied to the delay quantile $Q(p)$. Instance-dependent lower bounds are established to prove that DLPP is nearly optimal.
tags:
  - ICLR 2026
  - Lipschitz bandit
  - delayed feedback
  - zooming algorithm
  - phased elimination
  - regret lower bound
  - quantile
date: 2026-05-08
content_hash: 1d47a8f215cd6096
---

# Lipschitz Bandits with Stochastic Delayed Feedback

**Conference**: ICLR 2026
**arXiv**: [2510.00309](https://arxiv.org/abs/2510.00309)
**Code**: None
**Area**: Online Learning / Bandit Algorithms
**Keywords**: Lipschitz bandit, delayed feedback, zooming algorithm, phased elimination, regret lower bound, quantile

## TL;DR

This paper provides the first systematic study of Lipschitz bandits over continuous arm spaces under stochastic delayed feedback. For bounded delays, it proposes the Delayed Zooming algorithm, which employs a lazy update mechanism to maintain the suboptimality gap bound $\Delta(x) \leq 6r_t(x)$. For unbounded delays, it proposes DLPP, a phased pruning strategy whose regret is tied to the delay quantile $Q(p)$. Instance-dependent lower bounds are established to prove that DLPP is nearly optimal.

## Background & Motivation

**State of the Field**: Lipschitz bandits extend classical MAB to a continuous metric space $(\mathcal{A}, \mathcal{D})$, where the reward function $\mu$ satisfies a 1-Lipschitz condition. The classical Zooming algorithm achieves the optimal regret rate $\tilde{O}(T^{(d_z+1)/(d_z+2)})$ (with $d_z$ being the zooming dimension) via adaptive discretization in the delay-free setting. Delayed feedback has been extensively studied in MAB, linear bandits, and kernel bandits, yet the delayed feedback problem for Lipschitz bandits remains entirely unexplored.

**Limitations of Prior Work**: (1) The combination of continuous arm spaces and delayed feedback introduces dual complexity — each sampled point represents a neighborhood region whose estimate depends on delayed observations. (2) The core analysis of the Zooming algorithm relies on the property that confidence radii shrink only upon pulling an arm; this property breaks under delay, since delayed rewards arriving between pulls also reduce confidence radii. (3) Delayed methods for finite-arm settings (e.g., Delayed-UCB1) do not involve covering arguments for continuous spaces and cannot be directly extended.

**Root Cause**: How can efficient adaptive exploration over a continuous metric space be maintained when feedback is delayed and may be permanently absent?

**Paper Goals**: To design Lipschitz bandit algorithms that achieve near-optimal regret under both bounded and unbounded stochastic delays, and to prove their optimality.

**Starting Point**: The problem is decomposed into two sub-problems based on the support of the delay distribution — bounded delay is handled via "zooming + lazy update," while unbounded delay is handled via "phased accumulation of reliable feedback + elimination."

**Core Idea**: For bounded delays, lazy updates control the rate at which confidence radii shrink; for unbounded delays, phased round-robin sampling and the delay quantile $Q(p)$ are used to relate the number of pulls to the number of observations, thereby recovering the optimal delay-free regret rate.

## Method

### Overall Architecture

The problem is formalized as a triple $(\mathcal{A}, \mathcal{D}, \mu)$: $\mathcal{A}$ is a compact doubling metric space, $\mathcal{D}$ is a metric, and $\mu: \mathcal{A} \to [0,1]$ is an unknown 1-Lipschitz reward function. At each round $t$, the learner selects arm $x_t$ and receives reward $y_t = \mu(x_t) + \epsilon_t$ (where $\epsilon_t$ is sub-Gaussian noise), but the reward is only observed after a random delay $\tau_t$. The delays $\tau_t \sim f_\tau$ are independent of arms and rewards. The cumulative regret is $R(T) = \sum_{t=1}^T (\mu^* - \mu(x_t))$. Two algorithms are proposed for bounded and unbounded delays respectively.

### Key Designs

1. **Delayed Zooming Algorithm (Bounded Delay $\tau_t \leq \tau_{\max}$)**:
    - **Function**: Preserves the adaptive discretization advantage of the Zooming algorithm while handling information disorder caused by delays.
    - **Mechanism**: The observation count $v_t(x)$ replaces the pull count $n_t(x)$ when computing the confidence radius $r_t(x) = \sqrt{\frac{4\log T + 2\log(2/\delta)}{1 + v_t(x)}}$. A **lazy update mechanism** is introduced: a buffer queue $Q[x]$ is maintained for each active arm, and arriving feedback is cached rather than immediately incorporated when $v_t(x)+1 > 4 v_s(x)$ (where $s$ is the time of the last pull). The buffered rewards are processed together at the next pull of that arm. This ensures $r_t(x) \geq \frac{1}{2} r_s(x)$, thereby recovering the suboptimality gap bound $\Delta(x) \leq 6r_t(x)$ (compared to $3r_t(x)$ in the classical delay-free case).
    - **Design Motivation**: Under delay, $r_t(x)$ shrinks between pulls as delayed rewards arrive, breaking the core of classical zooming analysis. The lazy update resolves this by controlling the growth rate of the observation count. The resulting regret bound is $\tilde{O}\big(T^{\frac{d_z+1}{d_z+2}} + \tau_{\max} T^{\frac{d_z}{d_z+2}}\big)$.

2. **DLPP Algorithm (Unbounded Delays, Including $\tau = \infty$)**:
    - **Function**: Handles scenarios where delays may be unbounded or feedback may be permanently absent, providing a near-optimal regret guarantee.
    - **Mechanism**: Learning proceeds in phases. In phase $m$, a covering set $\mathcal{B}_m$ of balls with radius $r_m = 2^{-m}$ is maintained. Uniform round-robin sampling is performed over each ball until $v_m = \frac{2\log T + \log(2/\delta)}{2r_m^2}$ observations have been accumulated, at which point sampling for that ball ceases. At the end of each phase, regions whose empirical mean falls far below the best ball are eliminated (pruning rule: $\hat\mu_m^* - \hat\mu_m(B) \geq 8r_m$), and surviving regions are further subdivided. The number of pulls is related to the number of delayed observations via a Chernoff bound: $\Pr(v_{t+Q(p)}(B) \leq \frac{p}{2} n_t(B)) \leq \exp(-\frac{p}{8} n_t(B))$, linking regret to the delay quantile $Q(p)$.
    - **Design Motivation**: DLPP does not rely on real-time feedback updates; instead, it waits until sufficient statistics are accumulated before making decisions, and thus remains effective even when some feedback is missing. The regret bound is $R(T) \lesssim \min_{p \in (0,1]} \left\{ \frac{1}{p} T^{\frac{d_z+1}{d_z+2}} (c\log\frac{T}{\delta})^{\frac{1}{d_z+2}} + Q(p) \right\}$.

3. **Instance-Dependent Regret Lower Bound**:
    - **Function**: Proves that the regret rate of DLPP is nearly optimal (up to logarithmic factors).
    - **Mechanism**: A specific delay distribution is constructed (delay equals a fixed value $\tau_0$ with probability $p$, and $\infty$ otherwise). Via a Bernoulli sampling coupling, the delayed Lipschitz bandit is reduced to its delay-free counterpart. The resulting lower bound is $R(T) \gtrsim \frac{T^{(d_z+1)/(d_z+2)}(c\log T)^{1/(d_z+2)}}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$, where $\bar\Delta = \int_\mathcal{A} \Delta(x) / \int_\mathcal{A} 1$ is the average suboptimality gap.
    - **Design Motivation**: The final term $\bar\Delta \cdot Q(p)$ reflects the unavoidable regret incurred during the first $Q(p)$ rounds when no feedback is available, matching the form of the upper bound.

### Loss & Training

Both algorithms are theoretical constructs and require no gradient-based training. All key parameters are determined analytically: the lazy update threshold $4v_s(x)$ in Delayed Zooming, and the required observation count $v_m$ and pruning threshold $8r_m$ per phase in DLPP. Optimal regret rates are obtained by setting $\rho = (\log T / T)^{1/(d_z+2)}$ (bounded delay) or $M = \frac{\log(T/(c\log T))}{d_z+2}$ (DLPP).

## Key Experimental Results

### Main Results (Three Reward Functions × Two Delay Distributions, $T=60000$, 30 Independent Trials)

| Reward Function | Algorithm | No Delay | Uniform $\mathbb{E}[\tau]=20$ | Uniform $\mathbb{E}[\tau]=50$ | Geometric $\mathbb{E}[\tau]=20$ | Geometric $\mathbb{E}[\tau]=50$ |
|-----------------|-----------|:--------:|:--------:|:--------:|:--------:|:--------:|
| Trigonometric (1D) | Delayed Zooming | 138.97 | 154.55 | 171.07 | 159.30 | 152.98 |
| Trigonometric (1D) | DLPP | 304.60 | 314.87 | 326.71 | 312.44 | 325.74 |
| Sinusoidal (1D) | Delayed Zooming | 130.64 | 137.31 | 148.69 | 132.88 | 144.08 |
| Sinusoidal (1D) | DLPP | 178.05 | 195.35 | 209.97 | 186.28 | 208.80 |
| Bimodal (2D) | Delayed Zooming | 1445.86 | 1843.05 | 1858.45 | 1463.38 | 1828.15 |
| Bimodal (2D) | DLPP | **1120.64** | **1159.85** | **1136.46** | **1120.63** | **1142.55** |

### Theoretical Results Comparison

| Setting | Algorithm | Regret Upper Bound | Degenerate Case |
|---------|-----------|-------------------|-----------------|
| Bounded delay $\tau_{\max}$ | Delayed Zooming | $\tilde{O}(T^{(d_z+1)/(d_z+2)} + \tau_{\max} T^{d_z/(d_z+2)})$ | Recovers classical Lipschitz bandit when $\tau_{\max}=0$ |
| Bounded delay, $d_z=0$ | Delayed Zooming | $O(\sqrt{cT\log T} + c\tau_{\max})$ | Recovers finite-arm MAB with delay |
| Unbounded delay | DLPP | $\min_p \{ \frac{1}{p} T^{(d_z+1)/(d_z+2)} (\cdot)^{1/(d_z+2)} + Q(p) \}$ | Depends on median $\tau_{\text{med}}$ when $p=0.5$ |
| Lower bound | — | $\frac{R}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$ | Matches DLPP upper bound up to logarithmic factors |

### Key Findings

- Both algorithms maintain sublinear regret under bounded/unbounded delays; the additional regret due to delay appears only as an additive term.
- Delayed Zooming achieves lower regret in 1D settings; DLPP's phased pruning and discretization strategy is superior in 2D settings (where DLPP already outperforms Zooming even without delay).
- Delayed Zooming performs well empirically under geometric (unbounded) delay, but its theoretical guarantee is restricted to bounded delays — an open problem noted by the authors.
- The regret curve of DLPP exhibits a piecewise-linear shape, stemming from the phased accumulation of regret during uniform within-phase sampling.

## Highlights & Insights

- **Technical Contribution of Lazy Update**: Despite appearing straightforward — replacing $n_t$ with $v_t$ and buffering — the analysis is highly non-trivial. The analysis shows that the factor-of-two increase in the constant of $\Delta(x) \leq 6r_t(x)$ is tight: when $v_t(x)+1 > 4v_s(x)$, $r_t(x)$ may fall just below $r_s(x)/2$, necessitating the suspension of updates.
- **Elegance of Quantile Characterization**: The DLPP regret bound is expressed as a $\min_{p \in (0,1]}$ optimization over quantiles, adapting to the "central mass" of the delay distribution rather than its worst case — taking $p=0.5$ yields median dependence, while smaller $p$ accommodates heavy tails.
- **Matching Upper and Lower Bounds**: The lower bound construction employs a Bernoulli coupling technique — simulating a delay-free algorithm with probability $p$ — proving that the delayed regret is at least $\Omega(R/(p\log T))$, plus the $\bar\Delta \cdot Q(p)$ term reflecting the first $Q(p)$ rounds with no information, demonstrating that DLPP is nearly unimprovable.

## Limitations & Future Work

- The theoretical guarantee of Delayed Zooming is restricted to bounded delays; extension to unbounded delays is an open problem explicitly identified by the authors.
- DLPP requires a covering oracle, and computing coverings in high-dimensional spaces may be computationally expensive.
- Experiments are conducted only on 1D and 2D synthetic functions (trigonometric, sinusoidal, and 2D bimodal), without more realistic application scenarios.
- Both algorithms require knowledge of the time horizon $T$; anytime variants are not discussed.
- The adversarial delay setting is not addressed.

## Related Work & Insights

- **vs Delayed-UCB1 (Joulani et al., 2013)**: This finite-arm delayed MAB approach does not involve covering-activation mechanisms, making direct generalization infeasible.
- **vs BLiN (Feng et al., 2022)**: The first batched Lipschitz bandit, but it assumes instantaneous/batched feedback and is restricted to $[0,1]^d$ with axis-aligned cube partitions.
- **vs Lancewicki et al. (2021)**: That work characterizes regret for unbounded-delay MAB via quantile functions; this paper extends the approach to continuous spaces and proves a matching lower bound for DLPP.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to study the intersection of Lipschitz bandits and delayed feedback, though the core technical approach builds on extensions and combinations of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Theoretically complete (upper bounds + matching lower bounds); experimental validation is adequate but limited to synthetic settings.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, the two algorithms are presented in a well-organized hierarchy, and theorem statements are concise and elegant.
- Value: ⭐⭐⭐⭐ — Makes a solid foundational contribution to bandit theory, filling an important theoretical gap.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](../../NeurIPS2025/others/infrequent_exploration_in_linear_bandits.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)

<!-- RELATED:END -->
