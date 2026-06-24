---
title: >-
  [Paper Note] A Generalization Result for Convergence in Learning-to-Optimize
description: >-
  [ICML 2025 Spotlight][Optimization][Learning-to-Optimize] A probabilistic framework is proposed to combine PAC-Bayesian generalization theory with the Kurdyka-Łojasiewicz (KL) convergence theorem from variational analysis, proving for the first time with high probability that learned optimization algorithms converge to critical points without restricting the algorithm design.
tags:
  - "ICML 2025 Spotlight"
  - "Optimization"
  - "Learning-to-Optimize"
  - "PAC-Bayesian"
  - "Convergence Guarantees"
  - "Kurdyka-Łojasiewicz"
  - "Non-smooth Non-convex Optimization"
date: 2026-05-08
content_hash: 6a93dc53be66b443
---

# A Generalization Result for Convergence in Learning-to-Optimize

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2410.07704](https://arxiv.org/abs/2410.07704)  
**Code**: [GitHub](https://github.com/MichiSucker/COLA_2024)  
**Area**: Optimization  
**Keywords**: Learning-to-Optimize, PAC-Bayesian, Convergence Guarantees, Kurdyka-Łojasiewicz, Non-smooth Non-convex Optimization

## TL;DR

A probabilistic framework is proposed to combine PAC-Bayesian generalization theory with the Kurdyka-Łojasiewicz (KL) convergence theorem from variational analysis, proving for the first time with high probability that learned optimization algorithms converge to critical points without restricting the algorithm design.

## Background & Motivation

Learning-to-Optimize (L2O) leverages machine learning to accelerate optimization algorithms, demonstrating significant performance improvements over classical optimization algorithms in practice. However, a long-standing core issue is that **learned optimization algorithms lack theoretical convergence guarantees**.

Convergence proofs in classical optimization rely on geometric arguments (such as sufficient descent and relative error conditions). While these arguments are effective for hand-designed algorithms, they cannot be directly applied to learned algorithms for the following reasons:

**Problem instances are functions and cannot be globally observed**: The regions explored during training are restricted by initialization and the maximum number of iterations.

**Limits and mathematical induction are hindered**: Since only a finite number of iterations can be observed, classical limit concepts cannot be directly used to prove convergence.

**Convergence is a tail event**: Convergence is independent of any finite number of iterations, making it impossible to directly observe during training.

Existing solutions primarily rely on safeguards—restricting update steps to satisfy specific geometric properties (such as descent direction constraints) so that the learned algorithms can be analyzed similarly to hand-designed ones. However, this approach **severely limits the design space and performance of learned algorithms**, causing them to degrade to performance levels close to hand-designed algorithms.

This work shifts the perspective: **leveraging the unique advantage of being able to observe algorithmic behavior during training**, it derives convergence guarantees through generalization theory rather than directly transferring classical convergence results.

## Method

### Overall Architecture

The core mechanism can be summarized as a concise logical chain:

1. **Identify observable sufficient conditions**: Find a set of properties $a$ and $b$ that are verifiable during training and imply the target property $c$ (convergence to a critical point), i.e., $a \wedge b \Rightarrow c$.
2. **Probabilistic handling**: In a probability space, convert the above implication into set inclusion $\mathsf{A} \cap \mathsf{B} \subset \mathsf{C}$, yielding $\mu\{\mathsf{A} \cap \mathsf{B}\} \leq \mu\{\mathsf{C}\}$.
3. **Generalization to unseen problems**: Utilize the PAC-Bayesian theorem to ensure that the empirical estimate of $\mu\{\mathsf{A} \cap \mathsf{B}\}$ on the training set generalizes to unseen problems.

Specifically, this paper combines two core theoretical tools:

- **The abstract convergence theorem of Attouch et al. (2013) (Theorem 6.5)**: For KL functions, the iterative sequence converges to a critical point when sufficient descent, relative error, and boundedness conditions are satisfied.
- **The PAC-Bayesian generalization theorem of Sucker & Ochs (2024) (Theorem 6.3)**: Measurable properties of trajectories can generalize within the PAC-Bayesian framework.

### Key Designs

#### Problem Setting

Consider a parameterized loss function $\ell: \mathscr{Z} \times \mathscr{P} \to [0, \infty]$, where $\mathscr{Z} = \mathbb{R}^d$ is the optimization space and $\mathscr{P} = \mathbb{R}^q$ is the parameter space. The algorithm update rule is:

$$z^{(t+1)} = \mathcal{A}(h, p, z^{(t)}, r^{(t+1)})$$

where $h \in \mathscr{H}$ represents the hyperparameters (learned), $p \in \mathscr{P}$ represents the problem parameters, and $r^{(t+1)}$ denotes the internal randomness of the algorithm.

**Key Observation**: When $h$ and $p$ are fixed, the above iteration can be viewed as a functional equation of a Markov process, thereby defining a probability distribution on the trajectory space $\mathscr{Z}^{\mathbb{N}_0}$.

#### Construction of Three Measurable Sets

The technical core of this paper lies in constructing the three conditions required by the convergence theorem as measurable sets:

**1. Sufficient descent set $\mathsf{A}_{\mathrm{desc}}$**: The sequence satisfies $\ell(z^{(t+1)}, p) + a\|z^{(t+1)} - z^{(t)}\|^2 \leq \ell(z^{(t)}, p)$, meaning that every iteration step strictly decreases the loss function value, with the reduction being at least proportional to the squared step size.

**2. Relative error set $\mathsf{A}_{\mathrm{err}}$**: There exists a subgradient selection $v(z,p) \in \partial_1 \ell(z,p)$ satisfying $\|v(z^{(t+1)}, p)\| \leq b\|z^{(t+1)} - z^{(t)}\|$, meaning that the magnitude of the subgradient is controlled by the step size.

**3. Bounded set $\mathsf{A}_{\mathrm{bound}}$**: The sequence $(z^{(t)})$ remains bounded, $\|z^{(t)}\| \leq c$.

Proving the measurability of these three sets (especially $\mathsf{A}_{\mathrm{conv}}$ and $\mathsf{A}_{\mathrm{desc}}$) is highly non-trivial, as it requires handling product $\sigma$-algebras on infinite sequence spaces. The core technique exploits the separability and completeness of Polish spaces to express these sets as countable intersections/unions of measurable sets.

#### Core Corollary (Corollary 7.5)

Under the assumption of KL functions:

$$\mathsf{A}_{\mathrm{desc}} \cap \mathsf{A}_{\mathrm{err}} \cap \mathsf{A}_{\mathrm{bound}} \subset \mathsf{A}_{\mathrm{conv}}$$

That is, any sequence satisfying sufficient descent, relative error, and boundedness necessarily converges to a critical point.

#### Main Theorem (Theorem 7.6)

Combining with the PAC-Bayesian generalization theorem, letting $\mathsf{A} = \mathsf{A}_{\mathrm{desc}} \cap \mathsf{A}_{\mathrm{err}} \cap \mathsf{A}_{\mathrm{bound}}$, then with a probability of at least $1 - \varepsilon$:

$$\rho[\mathbb{P}_{(P,\xi)|H}\{\mathsf{A}_{\mathrm{conv}}\}] \geq 1 - \Phi_{\lambda/N}^{-1}\left(\frac{1}{N}\sum_{n=1}^N \rho[\mathbb{P}_{(P_n,\xi_n)|H,P_n}\{\mathsf{A}^c\}] + \frac{D_{\mathrm{KL}}(\rho \| \mathbb{P}_H) + \log(1/\varepsilon)}{\lambda}\right)$$

where $\Phi_a^{-1}(p) = \frac{1 - \exp(-ap)}{1 - \exp(-a)}$.

Intuitive interpretation: When the training set size $N$ is sufficiently large, the second term of the bound is suppressed by $\lambda$, making the bound approximately equal to the empirical probability of satisfying condition $\mathsf{A}$ on the training set plus a regularization term.

### Loss & Training

The algorithm $\mathcal{A}$ is trained on an i.i.d. dataset of problem parameters $S = (P_1, \ldots, P_N)$. The training objective is to optimize the hyperparameters $h$ such that the algorithm performs exceptionally well on the given family of problems.

**Verification Strategies in Practice**:

- The sufficient descent condition is verified step-by-step within a finite number of iterations $t_{\mathrm{train}}$.
- The relative error condition is directly satisfied by calculating $b = \max_t \|v(z^{(t)}, p)\| / \min_t \|z^{(t)} - z^{(t-1)}\|$.
- The boundedness condition is automatically satisfied in strongly convex problems, while requiring extra verification in non-convex problems.

It is worth emphasizing that **the update steps of the learning algorithm are completely unconstrained**, which is the core advantage of the proposed method over safeguard approaches.

## Key Experimental Results

### Main Results

#### Experiment 1: Quadratic Problems

| Configuration | Metric | Learned Algorithm | HBF (Baseline) | Description |
|------|------|---------|-----------|------|
| $d=200$, $N=250$ | Convergence Speed | ~100 steps to reach $10^{-16}$ | Not reached even at ~500 steps | Learned algorithm is significantly faster |
| 250 test sets | PAC bound on $\mathbb{P}\{\mathsf{A}\}$ | ~75% | - | Guarantees convergence on 75% of problems |
| 250 test sets | $\mathbb{P}\{\mathsf{A}_{\mathrm{conv}}\}$ estimate | ~95% | - | Actual convergence probability is higher |

- The optimization variable is $z \in \mathbb{R}^{200}$, and the loss is $\frac{1}{2}\|Az - b\|^2$.
- Strong convexity constants and smoothness constants are randomly sampled from the intervals $[m_-, m_+]$ and $[L_-, L_+]$, respectively.
- The number of training iterations is $t_{\mathrm{train}} = 500$.

#### Experiment 2: Neural Network Training

| Configuration | Metric | Learned Algorithm | Adam (Baseline) | Description |
|------|------|---------|-----------|------|
| $d=351$, $K=50$ | Loss Descent Speed | Faster | Slower | Learned algorithm outperforms Adam |
| 250 test sets | PAC bound | ~92% | - | Guarantees convergence on 92% of problems |
| 250 test sets | $\mathbb{P}\{\mathsf{A}\}$ estimate | ~94% | - | Bound is extremely tight |

- Train a two-layer fully connected ReLU network for regression.
- Use the MSE loss (non-smooth and non-convex with respect to parameter $\beta$).
- The Adam step size is tuned via a grid search over 100 values ($\kappa = 0.008$).
- The number of training iterations is $t_{\mathrm{train}} = 250$.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Sufficient descent condition only | Cannot guarantee convergence | Counterexample provided in Appendix B of the paper |
| No boundedness assumption | Sequence may diverge | e.g., $z^{(t)} = \sum_{k=1}^t 1/k$, which satisfies sufficient descent but is unbounded |
| Gap between $\mathsf{A}$ and $\mathsf{A}_{\mathrm{conv}}$ | Quadratic problems: ~20% | Shows that the bound is tight but not optimal |
| Gap between $\mathsf{A}$ and $\mathsf{A}_{\mathrm{conv}}$ | Neural networks: ~2% | The bound is tighter in non-convex settings |

### Key Findings

1. **The PAC bound is quite tight in both experiments**: The gap between the empirical estimation on the training set and the generalization bound is small.
2. **The learned algorithm still converges with high probability without constraints**: This validates the theoretical predictions.
3. **The gap $\mathsf{A}_{\mathrm{conv}} \setminus \mathsf{A}$ is smaller in the non-convex setting**: This indicates that the selected sufficient conditions are closer to being necessary and sufficient under non-convex setups.
4. **Convergence probability increases as the training set size grows**: This aligns with the expectations of the PAC-Bayesian theory.

## Highlights & Insights

1. **Profound shift in perspective**: Unlike traditional approaches that strive to prove "convergence for all problems" (worst-case), this work shifts to "convergence with high probability" (average-case), which is more aligned with the practical application scenarios of L2O.
2. **No restrictions on algorithm design**: The core breakthrough lies in eliminating the need for safeguards, giving the learning algorithm complete freedom in its update steps without constraining performance.
3. **Generality of the method**: The framework is not limited to optimization and can be generalized to any sequence prediction model with a Markovian structure.
4. **Technical depth of the measurability proof**: Proving the measurability of $\mathsf{A}_{\mathrm{conv}}$ is highly non-trivial, requiring the structural properties of Polish spaces.
5. **Consistency between theory and experiments**: Experiments perfectly validate the theoretically predicted chain of inequalities: $1 - \Phi^{-1}(\ldots) \leq \mathbb{P}\{\mathsf{A}\} \leq \mathbb{P}\{\mathsf{A}_{\mathrm{conv}}\}$.

## Limitations & Future Work

1. **Approximation of finite iterations**: Testing infinite iteration conditions is impossible in practice; they can only be checked within $t_{\mathrm{train}}$ steps, which serves as an approximation of the theoretical results.
2. **Inapplicability to stochastic optimization**: The sufficient descent condition requires strict descent of the loss at each step, which does not hold in stochastic setups such as mini-batch SGD.
3. **Training difficulty**: The training process to make the learned algorithm simultaneously satisfy all three conditions (especially sufficient descent) across most problems is difficult and time-consuming.
4. **Tightness of the bound**: The exact gap $\mathbb{P}\{\mathsf{A}_{\mathrm{conv}} \setminus \mathsf{A}\}$ is unknown, meaning that the bound's estimation of the convergence probability might be conservative.
5. **Choice of prior distribution**: The quality of the PAC-Bayesian bound strongly relies on the choice of the prior $\mathbb{P}_H$.

## Related Work & Insights

- **Safeguard methods** (Möller et al. 2019; Prémont-Schwarz et al. 2022): Guarantee convergence by constraining the update steps, but at the cost of performance.
- **Unrolling methods** (Gregor & LeCun 2010): Fix the number of iterations and provide generalization guarantees using Rademacher complexity or stability analysis.
- **PAC-Bayesian learning** (Sucker & Ochs 2024): The present paper builds directly upon this work, applying trajectory-level generalization to convergence analysis for the first time.
- **KL inequality** (Attouch et al. 2013): Provides abstract convergence conditions in non-smooth, non-convex scenarios.
- **Algorithm design principles** (Castera & Ochs 2024): Extract geometric properties from classical algorithms to guide L2O design.

**Inspiration for future research**: Extending this framework to stochastic optimization scenarios (such as SGD), or incorporating convergence rates (rather than just convergence status) into the generalization analysis.

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | First to prove L2O convergence without restricting algorithm design |
| Theoretical depth | ⭐⭐⭐⭐⭐ | Merging fields of PAC-Bayesian, variational analysis, and stochastic processes |
| Experimental Thoroughness | ⭐⭐⭐ | Only two experiments of small scale, lacking validation in large-scale practical applications |
| Utility | ⭐⭐⭐ | Significant theoretical breakthrough, but training is highly difficult and hard to apply directly |
| Writing Quality | ⭐⭐⭐⭐ | Clear logic, but the technical barrier is relatively high |
| Overall | ⭐⭐⭐⭐ | Resolves a long-standing open theoretical problem in the L2O field |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Generalization and Robustness of the Tilted Empirical Risk](generalization_and_robustness_of_the_tilted_empirical_risk.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](../../ICML2026/optimization/interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[NeurIPS 2025\] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](../../NeurIPS2025/optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)

</div>

<!-- RELATED:END -->
