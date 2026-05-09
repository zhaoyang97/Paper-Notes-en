---
title: >-
  [Paper Note] Thompson Sampling in Function Spaces via Neural Operators
description: >-
  [NeurIPS 2025][Reinforcement Learning][Thompson Sampling] This paper extends Thompson Sampling (TS) from finite-dimensional parameter spaces to infinite-dimensional function spaces, leveraging neural operators as approximate samplers of Gaussian process posteriors to efficiently solve functional optimization problems involving partial differential equations (PDEs).
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Thompson Sampling
  - Function Space
  - Neural Operators
  - Bayesian Optimization
  - PDE
date: 2026-05-08
content_hash: f9c3fa20ed4d3ad9
---

# Thompson Sampling in Function Spaces via Neural Operators

**Conference**: NeurIPS 2025
**arXiv**: [2506.21894](https://arxiv.org/abs/2506.21894)
**Code**: None
**Area**: Reinforcement Learning / Bayesian Optimization
**Keywords**: Thompson Sampling, Function Space, Neural Operators, Bayesian Optimization, PDE

## TL;DR

This paper extends Thompson Sampling (TS) from finite-dimensional parameter spaces to infinite-dimensional function spaces, leveraging neural operators as approximate samplers of Gaussian process posteriors to efficiently solve functional optimization problems involving partial differential equations (PDEs).

## Background & Motivation

### Functional Optimization Problems
Many scientific and engineering problems can be formulated as optimizing a known functional $J$ applied to the output of an unknown operator $\mathcal{G}$. Examples include:
- Designing initial temperature distributions that optimize heat conduction
- Finding boundary conditions that minimize fluid drag
- Optimizing control functions that maximize chemical reaction yield

These problems share several characteristics:
- **High query cost**: Evaluating the operator $\mathcal{G}$ (e.g., running high-fidelity simulators or physical experiments) is extremely expensive
- **Cheap functional evaluation**: Once the output of $\mathcal{G}$ is known, computing $J$ is inexpensive
- **Infinite-dimensional inputs and outputs**: The optimization space is infinite-dimensional

### Limitations of Thompson Sampling
Traditional Thompson Sampling has been widely and successfully applied in finite-dimensional parameter spaces, but extending it to function spaces poses fundamental challenges:
- Defining and sampling from posterior distributions in infinite-dimensional spaces is non-trivial
- Gaussian process (GP) inference in function spaces incurs prohibitive computational costs
- No established theoretical connection exists between neural operators and GPs

## Method

### Overall Architecture

The algorithm adopts a **sample-then-optimize** strategy:

```
Round t:
1. Train neural operator f_t ≈ sample from GP posterior p(G|D_{1:t})
2. Select the next query function u_{t+1} by optimizing J(f_t(·)) over the sampled operator f_t
3. Query the true operator: v_{t+1} = G(u_{t+1})
4. Update dataset: D_{1:t+1} = D_{1:t} ∪ {(u_{t+1}, v_{t+1})}
```

### Key Designs

#### Neural Operators as GP Posterior Samplers
The core contribution is treating a trained neural operator as an approximate sample from the GP posterior:

- **Theoretical connection**: Under appropriate prior distributions and training procedures, a neural operator with random initialization trained to a specified accuracy can be interpreted as an approximate sample from the GP posterior
- **Avoiding explicit uncertainty quantification**: There is no need to maintain a full GP posterior (which is computationally prohibitive); training the neural operator suffices
- **Source of stochasticity**: Different random initializations are used in each round to introduce the necessary exploratory behavior

#### Theoretical Framework in the Infinite-Dimensional Setting
The paper establishes the following theoretical results:

**Proposition (Neural Operator–GP Connection)**: Let $\mathcal{G}: \mathcal{U} \to \mathcal{V}$ be an unknown operator and $f_\theta$ a neural operator with parameters $\theta$. Under specific regularity conditions and prior distributions, the trained $f_\theta$ approximately follows the marginal distribution of the GP posterior.

**Theorem (Regret Bound)**: The cumulative regret of the proposed algorithm satisfies sublinear growth:
$$R_T = \sum_{t=1}^T [J(\mathcal{G}(u^*)) - J(\mathcal{G}(u_t))] \leq \tilde{O}(\sqrt{T \gamma_T})$$
where $\gamma_T$ denotes the maximum information gain.

#### Functional Optimization Sub-step
Given the sampled neural operator $f_t$, the inner optimization problem is:
$$u_{t+1} = \arg\max_{u \in \mathcal{U}} J(f_t(u))$$
Since $f_t$ is a differentiable neural network, this can be solved directly via gradient-based optimization.

### Loss & Training

- **Neural operator training objective**: Minimize prediction error on observed data
  $$\mathcal{L}(\theta) = \sum_{i=1}^{t} \| f_\theta(u_i) - v_i \|_{\mathcal{V}}^2 + \lambda \|\theta - \theta_0\|^2$$
  where the second term is a regularizer and $\theta_0$ denotes the randomly initialized parameters
- **Neural operator architecture**: DeepONet or Fourier Neural Operator (FNO)
- **Retraining each round**: Ensures independence across successive samples

## Key Experimental Results

### Main Results

Comparison with Bayesian optimization baselines on PDE functional optimization tasks:

| Method | Darcy Flow (obj. ↑) | Heat Conduction (obj. ↑) | Burgers' Eq. (obj. ↑) | Query Budget |
|--------|--------------------|--------------------------|-----------------------|-------------|
| Random Search | 0.32 | 0.28 | 0.25 | 100 |
| GP-UCB (finite-dim) | 0.58 | 0.53 | 0.47 | 100 |
| EI (finite-dim) | 0.61 | 0.55 | 0.49 | 100 |
| Functional GP-TS | 0.67 | 0.62 | 0.54 | 100 |
| **Ours (NO-TS)** | **0.78** | **0.73** | **0.65** | 100 |

Sample efficiency comparison (number of queries to reach objective value 0.6):

| Method | Darcy Flow | Heat Conduction | Burgers' Eq. | Avg. Speedup |
|--------|-----------|-----------------|--------------|-------------|
| GP-UCB | 78 | 85 | 92 | 1.0× |
| EI | 72 | 80 | 88 | 1.1× |
| Functional GP-TS | 55 | 63 | 70 | 1.4× |
| **Ours (NO-TS)** | **35** | **42** | **48** | **2.1×** |

### Ablation Study

| Component | Darcy Flow (obj.) | Query Efficiency Change |
|-----------|------------------|------------------------|
| Full NO-TS | **0.78** | Baseline |
| Fixed initialization (no stochasticity) | 0.52 | −33% |
| No regularization | 0.71 | −9% |
| DeepONet → FNO | 0.76 | −3% |
| Reduced training epochs | 0.68 | −13% |
| Increased regularization $\lambda$ | 0.74 | −5% |

### Key Findings

1. **Substantial improvement in sample efficiency**: The proposed method reduces the number of required queries by approximately 50% compared to conventional Bayesian optimization approaches
2. **Random initialization is critical**: Fixing the initialization eliminates exploratory behavior and leads to significant performance degradation
3. **Dual role of regularization**: Regularization simultaneously controls overfitting and maintains the approximation quality between the neural operator and GP posterior sampling
4. **Architectural robustness**: DeepONet and FNO yield comparable performance, indicating that the method is insensitive to the choice of neural operator architecture

## Highlights & Insights

1. **Deep integration of theory and practice**: The paper establishes a rigorous mathematical connection between neural operators and GPs in the infinite-dimensional setting, and derives formal regret bounds
2. **Avoiding explicit uncertainty quantification**: Training stochasticity is cleverly leveraged as a surrogate for costly posterior inference
3. **Broad applicability**: The framework applies to any operator optimization problem amenable to neural operator modeling
4. **An important bridge for scientific computing**: Modern Bayesian optimization methods are brought to bear on PDE-constrained optimization
5. **Sublinear regret guarantee**: Theoretical convergence guarantees are provided

## Limitations & Future Work

1. **Full retraining each round**: The computational overhead is non-trivial; incremental training strategies could be explored
2. **Prior assumptions**: Theoretical results rely on specific prior distributions and regularity conditions that may not be fully satisfied in practice
3. **High-dimensional function spaces**: When the input function space has very high dimensionality, the inner optimization sub-step may become intractable
4. **Lack of adaptive exploration–exploitation balance**: Unlike UCB-type methods, the proposed approach does not offer explicit control over exploration intensity
5. **Limited experimental scale**: Validation is conducted on only a few PDE benchmarks; performance on larger-scale problems remains to be examined

## Related Work & Insights

- **Neural Operators**: DeepONet, FNO, and related architectures provide practical tools for learning in function spaces
- **Bayesian Optimization**: Classical methods such as GP-UCB and EI perform well in finite dimensions but face significant challenges when generalized to function spaces
- **Thompson Sampling Theory**: Theoretical analyses by Russo & Van Roy and others provide the foundation for the regret bounds derived in this work
- **Future Directions**: Extending this framework to multi-objective or constrained optimization settings

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First work to combine TS with neural operators for function-space optimization
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Establishes a rigorous infinite-dimensional theoretical framework with regret bounds
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple PDE tasks
- **Practical Impact**: ⭐⭐⭐⭐ — Significant potential value for scientific computing and engineering optimization
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical sections are somewhat dense, but overall presentation is clear

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Complexity Scaling Laws for Neural Models using Combinatorial Optimization](complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](learning_from_demonstrations_via_capability-aware_goal_sampling.md)

</div>

<!-- RELATED:END -->
