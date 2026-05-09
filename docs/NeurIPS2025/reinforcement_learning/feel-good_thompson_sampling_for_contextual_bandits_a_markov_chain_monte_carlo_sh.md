---
title: >-
  [Paper Note] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown
description: >-
  [NeurIPS 2025][Reinforcement Learning][Thompson Sampling] This paper presents the first systematic empirical evaluation of Feel-Good Thompson Sampling (FG-TS) and its smoothed variant SFG-TS under approximate posteriors, spanning linear, logistic, and neural contextual bandit settings across fourteen benchmarks. The study finds that FG-TS outperforms standard TS when exact posteriors are available (linear/logistic), but degrades in neural bandits, revealing a critical trade-off between optimistic bias and sampling noise.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Thompson Sampling
  - Contextual Bandits
  - MCMC
  - Exploration-Exploitation
  - Posterior Sampling
date: 2026-05-08
content_hash: 0ed68177d13e8b68
---

# Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown

**Conference**: NeurIPS 2025
**arXiv**: [2507.15290](https://arxiv.org/abs/2507.15290)
**Code**: [GitHub](https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown)
**Area**: Reinforcement Learning
**Keywords**: Thompson Sampling, Contextual Bandits, MCMC, Exploration-Exploitation, Posterior Sampling

## TL;DR
This paper presents the first systematic empirical evaluation of Feel-Good Thompson Sampling (FG-TS) and its smoothed variant SFG-TS under approximate posteriors, spanning linear, logistic, and neural contextual bandit settings across fourteen benchmarks. The study finds that FG-TS outperforms standard TS when exact posteriors are available (linear/logistic), but degrades in neural bandits, revealing a critical trade-off between optimistic bias and sampling noise.

## Background & Motivation

**Background**: Thompson Sampling (TS) is among the most popular exploration-exploitation algorithms for contextual bandits, valued for its practicality and ease of implementation. However, in high-dimensional settings, TS suffers from insufficient exploration, achieving a theoretical regret of $O(d\sqrt{dT})$, which falls short of the information-theoretic lower bound $\Omega(d\sqrt{T})$.

**Limitations of Prior Work**: FG-TS (Zhang, 2022) introduces an optimistic bias by adding a "feel-good bonus" to the likelihood, enforcing more aggressive exploration and achieving optimal $O(d\sqrt{T})$ regret in the linear setting. However, its theoretical analysis assumes exact posteriors, whereas large-scale or neural network settings require approximate posteriors (e.g., MCMC). **The behavior of FG-TS under approximate posteriors remains entirely unknown.**

**Key Challenge**: Optimistic bias aids exploration under exact posteriors, but when posterior sampling itself is noisy, this bias may amplify errors — the compounding of two noise sources can degrade decision quality.

**Goal**: How does approximate posterior inference affect the performance of FG-TS? Under what conditions does FG-TS outperform or underperform standard TS? How do hyperparameters such as bonus magnitude, prior strength, and preconditioning interact?

**Key Insight**: A spectrum from exact to coarse posteriors is constructed — linear (closed-form posterior) → logistic (near-Gaussian) → neural (highly nonlinear) — with systematic comparison across multiple MCMC samplers.

**Core Idea**: The first systematic FG-TS benchmark, revealing that the trade-off between bonus magnitude and posterior accuracy is the decisive factor governing performance.

## Method

### Overall Architecture
At each round $t$, the agent observes a context $\mathcal{X}_t \subseteq \mathbb{R}^d$, samples model parameters $\theta_t$ from the posterior, greedily selects action $x_t = \arg\max_{x \in \mathcal{X}_t} f_{\theta_t}(x)$, observes reward $r_t$, and updates the posterior. The key distinction lies in the likelihood function used to define the posterior.

### Key Designs

1. **Feel-Good Thompson Sampling (FG-TS)**:

    - **Function**: Introduces a preference for high-reward models (optimistic bias) by augmenting the standard TS likelihood.
    - **Mechanism**: The likelihood is modified to $L^{\text{FG}}(\theta, x, r) = \eta(f_\theta(x) - r)^2 - \lambda \min(b, f_\theta(x))$, where $\lambda > 0$ controls the bonus magnitude and $b$ is a truncation upper bound. The negative term biases sampling toward models with higher $f_\theta$.
    - **Design Motivation**: Standard TS under-explores in high dimensions; the optimistic bias steers sampling toward favorable directions, theoretically reducing regret from $O(d\sqrt{dT})$ to the optimal $O(d\sqrt{T})$.

2. **Smoothed Feel-Good TS (SFG-TS)**:

    - **Function**: Smooths the min truncation in FG-TS to yield a posterior more amenable to MCMC sampling.
    - **Mechanism**: Replaces min with softplus: $L^{\text{SFG}}(\theta, x, r) = \eta(f_\theta(x) - r)^2 - \lambda(b - \Phi_s(b - f_\theta^\star))$, where $\Phi_s(u) = \log(1 + \exp(su))/s$.
    - **Design Motivation**: The min operator renders the posterior non-smooth and difficult to sample from via MCMC; softplus smoothing preserves the theoretical regret guarantee while improving sampling quality.

3. **MCMC Sampler Family**:

    - **LMC (Langevin MC)**: $\theta_{t,k+1} = \theta_{t,k} - \eta_t \nabla \mathcal{L}_t(\theta_{t,k}) + \sqrt{2\eta_t \beta_t^{-1}} \epsilon_{t,k}$, the most basic gradient-plus-noise sampler.
    - **MALA (Metropolis-Adjusted LMC)**: Augments LMC with a Metropolis accept/reject step to correct discretization bias.
    - **HMC (Hamiltonian MC)**: Introduces momentum variables to simulate Hamiltonian dynamics, $H(\theta, v) = \mathcal{L}_t(\theta) + \frac{1}{2}\|v\|^2$, with theoretically faster mixing.
    - **Preconditioned variants**: Apply the design matrix $\mathbf{V}_t^{-1}$ to precondition gradients, reducing the impact of the condition number $\kappa_t$ by one order of magnitude.
    - **SVRG variants**: Use variance-reduced gradient estimates with control variates to reduce stochastic gradient noise.

4. **Underdamped LMC (ULMC)**:

    - **Function**: A Langevin variant incorporating momentum and a damping coefficient.
    - **Mechanism**: $v_{t,k+1/2} = (1-\gamma\eta)v_{t,k} - \eta\nabla U(\theta_{t,k}) + \sqrt{2\gamma\eta}\xi_{t,k}$; the auxiliary velocity variable enables faster mixing over rough posterior landscapes.
    - **Design Motivation**: Provides a contrast with overdamped LMC for evaluating performance on complex posteriors.

### Experimental Setup
- Linear bandits: $d=20/40$, $K=5$ arms, $T=10000$, Gaussian noise $\sigma=0.5$
- Logistic bandits: $d=20$, $K=50$ arms, $T=10000$, Bernoulli rewards
- Neural bandits: UCI datasets (Adult, Mushroom, Shuttle, MNIST, etc.), 2/4-layer MLP
- 10 seeds (linear/logistic) / 5 seeds (neural)

## Key Experimental Results

### Main Results (Selected Linear and Logistic Results)

| Algorithm | Linear-20d ($\beta=10^3$) | Linear-40d | Logistic-20d |
|-----------|--------------------------|------------|-------------|
| LinUCB | 73.0 ± 13.8 | 126.3 ± 19.3 | 176.9 ± 41.9 |
| LinTS | 114.7 ± 8.8 | 204.6 ± 19.1 | 179.9 ± 53.2 |
| LMCTS | 62.6 ± 9.5 | 129.1 ± 16.1 | 202.7 ± 44.1 |
| MALATS | **61.3 ± 26.6** | **100.6 ± 10.0** | 194.0 ± 76.9 |
| FGLMCTS | 213.0 ± 126.0 | 163.6 ± 22.4 | **184.8 ± 38.0** |
| PHMCTS | 90.0 ± 9.2 | 162.2 ± 12.5 | 218.9 ± 16.1 |
| SFGMALATS | 189.3 ± 135.3 | 142.1 ± 19.5 | 198.4 ± 52.3 |

### Neural Bandit Key Results

| Dataset | LMCTS | FGLMCTS | SFGLMCTS | Neural-εGreedy | NeuralUCB |
|---------|-------|---------|----------|---------------|-----------|
| Adult | 2456.6 | 3505.0 | 4505.6 | 2658.0 | **2444.4** |
| Mushroom | 324.6 | **283.2** | 440.6 | 124.0 | 145.6 |
| Shuttle | **210.2** | 214.4 | 1503.0 | 372.4 | 2981.2 |
| MNIST | **2854.6** | 2542.6 | 2935.0 | 3248.0 | 5442.8 |

### Ablation Study

| Ablation Dimension | Key Findings |
|-------------------|-------------|
| Feel-good bonus $\lambda$ | $\lambda=0.5$ yields no improvement in the linear setting; at $\lambda=0.01$, SFGMALATS achieves 56.2±22.8, outperforming all vanilla TS variants |
| Preconditioning | Preconditioned HMC is effective in the linear setting (90.0 vs. 241.2), but preconditioned LMC performs worse (134.4 vs. 62.6) |
| Inverse temperature $\beta$ | $\beta=10^3$ with a decay schedule generally outperforms $\beta=1$ |
| SVRG | Effective at $d=20$ (73.2 vs. 62.6, comparable), but collapses at $d=40$ (19236.8) |

### Key Findings
- **FG-TS is effective in linear/logistic bandits** (especially at low $\lambda$), with MALATS achieving the best overall performance.
- **FG-TS consistently degrades in neural bandits**: the compounding of approximate posterior noise and optimistic bias deteriorates decision quality.
- **Core trade-off**: larger bonus is more beneficial under exact posteriors but more harmful under approximate posteriors.
- **Preconditioning is a double-edged sword**: it improves HMC but may disrupt the natural diffusion behavior of LMC.
- Neural-εGreedy and NeuralUCB are more robust in neural settings.

## Highlights & Insights
- **Discovery of the "bonus × posterior accuracy" trade-off**: This is the paper's central empirical contribution — the more accurate the posterior, the more beneficial the bonus; the more approximate the posterior, the more harmful the bonus. This insight has broad implications for all optimistic exploration methods.
- **Comprehensive MCMC sampler comparison**: A systematic comparison of LMC, MALA, HMC, ULMC, SVRG, and their preconditioned variants constitutes the most thorough MCMC benchmark in the contextual bandit literature.
- **Open-source framework**: A PyTorch implementation covering all algorithms, reproducible and extensible, offering lasting practical value for future research.

## Limitations & Future Work
- **The root cause of FG-TS degradation in neural settings is insufficiently analyzed** — is it gradient estimation bias, posterior multimodality, or network capacity issues?
- Non-parametric posterior methods (e.g., particle filters, Bayesian neural networks) are not considered as alternatives.
- Performance gains from FG-TS in linear and logistic settings are modest, and some results exhibit high variance.
- Adaptive bonus scheduling strategies (e.g., $\lambda$ decaying with $t$) are not evaluated.
- Validation in real-world applications such as recommendation systems is absent.

## Related Work & Insights
- **vs. Zhang (2022) original FG-TS paper**: The original work provides only theoretical analysis under exact posteriors; this paper complements it with a full empirical landscape under approximate posteriors, demonstrating that the theoretical advantage is conditional in practice.
- **vs. Riquelme et al. (2018)**: A seminal bandit benchmark paper; this work follows its experimental protocol while extending evaluation to the FG-TS dimension.
- **vs. Xu et al. (2022)**: The original LMC-TS paper; this work builds upon it by incorporating FG variants and a broader set of sampler comparisons.
- **Implications for future research**: Any method that theoretically improves performance by modifying the objective function should be re-evaluated under approximate optimization conditions.

## Rating
- Novelty: ⭐⭐⭐ No new algorithmic designs at the method level (HMC for SFG-TS is novel); the core contribution is empirical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 benchmarks, ~20 algorithm variants, and multi-dimensional ablations — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and thorough background, though some tables are overly dense and difficult to parse quickly.
- Value: ⭐⭐⭐⭐ The "bonus × posterior accuracy" trade-off is a practically relevant and broadly applicable insight; the benchmark and codebase offer long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)

</div>

<!-- RELATED:END -->
