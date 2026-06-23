---
title: >-
  [Paper Note] Poisson Midpoint Method for Log-Concave Sampling: Beyond the Strong Error Lower Bounds
description: >-
  [ICLR 2026][learning_theory][Wasserstein-2] This paper provides a sharp $W_2$ convergence analysis for Poisson Midpoint Discretization (PLMC) in strongly log-concave sampling. It proves that PLMC further compresses the precision $\epsilon$ dependency from $\tilde O(\epsilon^{-2/3})$ to $\tilde O(\epsilon^{-1/3})$ under both overdamped and underdamped Langevin dy
tags:
  - ICLR 2026
  - learning_theory
  - Wasserstein-2
date: 2026-05-08
content_hash: 9613f67d226773f0
---
# Poisson Midpoint Method for Log-Concave Sampling: Beyond the Strong Error Lower Bounds

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=j2wEu2ycTg](https://openreview.net/forum?id=j2wEu2ycTg)  
**Code**: To be confirmed  
**Area**: Learning Theory / Sampling Algorithms / Langevin Monte Carlo  
**Keywords**: Strongly log-concave sampling, Poisson Midpoint Method, Langevin Dynamics, Wasserstein-2, oracle complexity

## TL;DR
This paper provides a sharp $W_2$ convergence analysis for Poisson Midpoint Discretization (PLMC) in strongly log-concave sampling. It proves that PLMC further compresses the precision $\epsilon$ dependency from $\tilde O(\epsilon^{-2/3})$ to $\tilde O(\epsilon^{-1/3})$ under both overdamped and underdamped Langevin dynamics—an order of magnitude faster than the previously assumed optimal randomized midpoint method. This work also signifies the first proof that $W_2$ weak error complexity can be **lower** than the $L^2$ strong error complexity lower bound established in the literature.

## Background & Motivation
**Background**: Sampling from $\pi(x)\propto\exp(-f(x))$ is a fundamental problem in physics, finance, and Bayesian statistics. The standard approach is Langevin Monte Carlo (LMC), which applies Euler-Maruyama discretization to the continuous-time Langevin stochastic differential equation (SDE). Later, Shen & Lee proposed the **Randomized Midpoint Method** (RLMC), which reduces discretization bias at the cost of variance by evaluating gradients at random midpoints, achieving faster convergence. Kandasamy & Nagaraj further introduced the **Poisson Midpoint Method** (PLMC), a variant of RLMC that converges wherever LMC does, enabling analysis beyond log-concavity.

**Limitations of Prior Work**: Cao et al. (2021) proved a **strong $L^2$ error** lower bound of $\Omega(\epsilon^{-2/3})$ for randomized algorithms in Underdamped Langevin Dynamics (ULD)—where "strong error" refers to the $L^2$ distance between the algorithm output and the SDE solution driven by the **same Brownian motion**. Since RLMC achieves exactly this $\tilde O(\epsilon^{-2/3})$ rate, it was **widely believed** in the literature that $\epsilon^{-2/3}$ was also the optimal rate for $W_2$ weak convergence, and it remained unchallenged.

**Key Challenge**: Strong error lower bounds do not strictly bound weak errors. The $W_2$ distance is the infimum of the $L^2$ distance when the algorithm output and the true solution can be driven by **different, arbitrarily coupled** Brownian motions. By taking the infimum over couplings, it can be much smaller than the strong error. Mistaking "strong error optimality" for "sampling optimality" led to using an overly tight lower bound that restricted potentially faster algorithms.

**Goal**: The authors aim to (1) revisit the strongly log-concave convergence of PLMC to provide a tighter $W_2$ upper bound than existing randomized midpoint methods, and (2) clarify the true relationship between this upper bound and the strong error lower bound.

**Key Insight**: By rewriting a single PLMC iteration as "standard LMC + a perturbed Gaussian noise," the problem is reduced to measuring the $W_2$ distance between "a Gaussian" and "a Gaussian perturbed by a one-dimensional random vector." This quantity has a **sharp upper bound** borrowed from Zhai’s (2018) proof of the high-dimensional CLT, which is significantly tighter than naive coupling.

**Core Idea**: Replace naive coupling with Zhai's sharp $W_2$ Gaussian perturbation lemma (reducing error from $\nu$ to $\nu^2$), and combine it with the contractivity of gradient mappings to establish a step-by-step coupling recursion. This eventually yields an oracle complexity of $\tilde O(\epsilon^{-1/3})$ for both overdamped and underdamped PLMC.

## Method

### Overall Architecture
Ours is a purely theoretical analysis with no new algorithm pipeline; the analysis focuses on the PLMC iteration format given by prior work. The contribution lies in a tighter convergence proof. The logic follows: first, review the PLMC iterative definition and **rewrite** it as "LMC with perturbed noise." Then, use three tools (Zhai’s sharp $W_2$ lemma + gradient contraction + optimal coupling) to construct a contractive recursion for $W_2^2$. Finally, accumulate the single-step errors and substitute LMC's own convergence rate to derive the end-to-end oracle complexity.

PLMC iterations run in groups with batch size $k$, which can be viewed as a stochastic approximation of LMC with step size $\eta/k$. The core recursion for overdamped PLMC is:

$$\tilde X_{t,i+1}=\tilde X_{t,i}-\tfrac{\eta}{k}\nabla f(\tilde X_{t,0})+\eta H_{t,i}\big(\nabla f(\tilde X_{t,0})-\nabla f(\tilde X^+_{t,i})\big)+\sqrt{\tfrac{2\eta}{k}}\,Z_{t,i},$$

where $\tilde X^+_{t,i}$ is the midpoint, $H_{t,i}\sim\mathrm{Bernoulli}(1/k)$ determines whether to evaluate the gradient at the midpoint, and $Z_{t,i}$ is standard Gaussian. Since $\mathbb{E}[N_t]=\mathbb{E}\sum_i H_{t,i}=1$, each batch only requires 2 gradient calls in expectation. Thus, $tk$ iterations cost only $O(t)$ oracle calls, which is why PLMC is more efficient than LMC.

### Key Designs

**1. Rewriting PLMC as "LMC with perturbed Gaussian noise": Making midpoint randomness an analyzable noise term**

PLMC is difficult to analyze because the Bernoulli correction term $\eta H_{t,i}(\cdots)$ makes the iteration deviate from a clean Langevin step. The first step involves algebraically folding it into the noise, writing it in standard LMC form $\tilde X_{t,i+1}=\tilde X_{t,i}-\frac{\eta}{k}\nabla f(\tilde X_{t,i})+\sqrt{\frac{2\eta}{k}}\,\tilde Z_{t,i}$, where the **perturbed Gaussian** is:

$$\tilde Z_{t,i}=\sqrt{\tfrac{\eta k}{2}}(H_{t,i}-\tfrac1k)\big(\nabla f(\tilde X_{t,0})-\nabla f(\tilde X^+_{t,i})\big)+\sqrt{\tfrac{\eta}{2k}}\big(\nabla f(\tilde X_{t,i})-\nabla f(\tilde X^+_{t,i})\big)+Z_{t,i}.$$

Conditioned on the history, $\tilde Z_{t,i}$ consists of a Gaussian $Z_{t,i}$ plus a deterministic mean shift $B_{t,i}$ and a **zero-mean random vector $S_{t,i}$ that almost surely falls into a one-dimensional subspace**: $S_{t,i}=\sqrt{\frac{\eta k}{2}}(H_{t,i}-\frac1k)(\nabla f(\tilde X_{t,0})-\nabla f(\tilde X^+_{t,i}))$. This rewrite precisely isolates the "randomized midpoint" stochasticity into $S_{t,i}$, providing the prerequisite for sharp analysis.

**2. Zhai's sharp $W_2$ lemma: Reducing the distance between Gaussian and perturbed Gaussian from $\nu$ to $\nu^2$**

Measuring how much $\tilde Z_{t,i}$ deviates from a pure Gaussian is equivalent to measuring the $W_2$ distance between $Z\sim N(0,I_d)$ and $Z+V$ (where $V$ is the perturbation). Naive coupling of two Gaussians only yields $W_2^2(\mathrm{Law}(Z),\mathrm{Law}(Z+V))\le \nu$, where $\nu=\mathrm{Tr}(\Sigma)$. Ours adopts a lemma adapted from Zhai (2018) for high-dimensional CLT (Lemma 1): If $V$ satisfies $\|V\|\le\beta$ almost surely, $\mathbb{E}[V]=0$, $\mathbb{E}[VV^\top]=\Sigma$, and $V$ almost surely lies in a one-dimensional subspace, then:

$$W_2^2\big(\mathrm{Law}(Z),\mathrm{Law}(Z+V)\big)\le \tfrac{11}{2}\nu^2+15\,\mathbf{1}_{\beta^2>1}\cdot 2\nu.$$

Critically, when $\nu\ll1$, the leading term is $\nu^2$ rather than $\nu$, and $\nu^2$ can be much smaller than $\nu$. Intuitively, taking $V\sim N(0,\nu)$ and $Z\sim N(0,1)$ in 1D yields $W_2^2=2+\nu-2\sqrt{1+\nu}=\Theta(\nu^2)$, illustrating this quadratic improvement. Compared to the version used by Kandasamy & Nagaraj, this lemma **avoids high-order moments**, which is the core engine for improving the rate to $\epsilon^{-1/3}$.

**3. Gradient contraction + optimal coupling: Building the step-by-step $W_2^2$ contraction recursion**

A single-step noise bound is insufficient; it must accumulate without exploding. The authors use the fact that under well-conditioned $f$ ($\alpha$-strongly convex, $L$-smooth), the gradient descent map $T(x)=x-\eta\nabla f(x)$ is $(1-\alpha\eta)$-Lipschitz (i.e., contractive). During coupling, the PLMC iteration $\tilde X_{t,i}$ and the LMC iteration $X_{t,i}$ with step size $\eta/k$ are first optimally coupled. Then, their noises $Y_{t,i}$ and $\tilde Z_{t,i}$ are **optimally coupled** according to the bound provided by Lemma 1, resulting in the contraction recursion:

$$W_2^2(\mathrm{Law}(X_{t,i+1}),\mathrm{Law}(\tilde X_{t,i+1}))\le\Big(1-\tfrac{\alpha\eta}{2k}\Big)W_2^2(\mathrm{Law}(X_{t,i}),\mathrm{Law}(\tilde X_{t,i}))+E_{t,i},$$

where $E_{t,i}$ is the single-step discretization error. Expanding $E_{t,i}$ involves moments like $\mathbb{E}\|\tilde X_{t,i}-\tilde X_{t,0}\|^p$. These are reduced to $\mathbb{E}\|\nabla f(\tilde X_{t,0})\|^2$ and finalized using a tight gradient bound $\sum_{t}\mathbb{E}\|\nabla f(\tilde X_{t,0})\|^2\lesssim\frac1\eta\mathbb{E}[f(\tilde X_{0,0})-f(\tilde X_{N,0})]+LdN$ (Lemma 2). Since $\int\|\nabla f\|^2\mathrm{d}\pi\le Ld$ at stationarity, the dominance of $LdN$ indicates this bound is near the limit.

**4. Underdamped case: Coordinate transformation to turn ULMC into "noisy contraction"**

Underdamped Langevin (ULD) has position $U$ and momentum $V$ components, where the deterministic part does not contract directly. The authors apply a coordinate transformation $\binom{x}{y}\mapsto M\binom{x}{y}$ with $M=\begin{psmallmatrix}I_d&0\\I_d&\frac2\gamma I_d\end{psmallmatrix}$, defining $W_{t,i}=U_{t,i}+\frac2\gamma V_{t,i}$ and $X_{t,i}=[U_{t,i},W_{t,i}]^\top$. With a suitable step size, the transformed deterministic map $T$ is $(1-\frac{\alpha\eta}{\gamma}+L\eta^2)$-Lipschitz (citing Zhang et al. 2023 Lemma 16), which contracts for small $\eta$. Thus, ULMC also becomes a "noisy contraction," allowing for the same coupling argument as the overdamped case, with additional control over $\mathbb{E}\|\nabla f(\tilde U_{t,0})\|^p$ and $\mathbb{E}\|\tilde V_{t,0}\|^p$ moments (given by Theorem 4). The result includes an arbitrary non-negative integer $p$ to control low-probability events; setting $p\ge3$ yields $\tilde O(\epsilon^{-1/3})$.

## Key Experimental Results
Ours is a purely theoretical work with no numerical experiments; "Key Experimental Results" are presented as complexity comparison tables ($\kappa=L/\alpha$ is the condition number, $d$ is the dimension, and $\epsilon$ is the target precision).

### Main Results: Oracle Complexity Comparison for Overdamped Langevin

| Algorithm | Assumption | Metric | Oracle Complexity (w.r.t. $\epsilon$) |
| :--- | :--- | :--- | :--- |
| LMC (Durmus 2019) | Strong Log-Concave | $W_2^2\le\epsilon^2/\alpha$ | $\kappa d/\epsilon^2$ |
| RLMC (Shen & Lee; Yu 2024) | Strong Log-Concave | $W_2^2\le\epsilon^2/\alpha$ | $\kappa\sqrt d/\epsilon+\kappa^{4/3}d^{1/3}/\epsilon^{2/3}$ |
| **PLMC (Ours)** | Strong Log-Concave | $W_2^2\le\epsilon^2/\alpha$ | $(\kappa^{4/3}d^{1/3}+\kappa d^{2/3})/\epsilon^{2/3}$ |

Compared to the $\tilde O(\epsilon^{-2})$ of overdamped LMC, PLMC provides a **cubic** improvement in $\epsilon$.

### Main Results: Oracle Complexity Comparison for Underdamped Langevin

| Algorithm | Assumption | Metric | Oracle Complexity (w.r.t. $\epsilon$) |
| :--- | :--- | :--- | :--- |
| LMC (Dalalyan & Riou-Durand 2020) | Strong Log-Concave | $W_2^2\le\epsilon^2/\alpha$ | $\kappa^{3/2}\sqrt d/\epsilon$ |
| RLMC (Shen & Lee; Yu 2024) | Strong Log-Concave | $W_2^2\le\epsilon^2/\alpha$ | $\kappa d^{1/3}/\epsilon^{2/3}+\kappa^{7/6}d^{1/6}/\epsilon^{1/3}$ |
| PLMC (Kandasamy & Nagaraj 2024) | LSI | $\mathrm{TV}\le\epsilon$ | $\kappa^{17/12}d^{5/12}/\sqrt\epsilon$ |
| **PLMC (Ours)** | Strong Log-Concave | $W_2^2\le\epsilon^2/\alpha$ | $\kappa^{7/6}d^{1/3}/\epsilon^{1/3}+\cdots$ ($\tilde O(\epsilon^{-1/3})$ for $p\ge3$) |

Compared to the $\tilde O(\epsilon^{-1})$ of underdamped LMC, PLMC similarly offers a cubic improvement and a quadratic improvement over the $\epsilon^{-2/3}$ of RLMC.

### Key Findings
- **Breaking the $\epsilon^{-2/3}$ Barrier**: PLMC is the first known algorithm to reduce the $\epsilon$ complexity of strongly log-concave sampling to $\tilde O(\epsilon^{-1/3})$, which is lower than the long-believed $\tilde\Omega(\epsilon^{-2/3})$.
- **Weak Error < Strong Error Lower Bound**: Cao et al.'s $\Omega(\epsilon^{-2/3})$ is a **strong $L^2$ error** lower bound. Ours proves that the $W_2$ weak error only requires $\tilde O(\epsilon^{-1/3})$, clarifying that "strong error optimality $\neq$ sampling optimality."
- **Trade-off in $d$ vs. $\kappa$**: Compared to the parallel work by Altschuler et al. (2025) which achieves $\tilde O(\kappa^{5/6}d^{5/3}/\epsilon^{2/3})$, ours improves the $\epsilon$ dependency but has a worse dependency on dimension $d$. The authors suggest that tighter bounds for high-order algorithm moments could further optimize the $\kappa$ dependency.

## Highlights & Insights
- **The Core Lever of "Quadratic vs. Linear"**: While naive coupling gives $W_2^2\lesssim\nu$, Zhai's lemma gives $\nu^2$. When $\nu\ll1$, this is a qualitative leap—the entire acceleration essentially stems from replacing this single inequality.
- **"Purifying" Randomness into 1D Perturbation**: When rewriting PLMC as perturbed Gaussian LMC, the authors intentionally argued that the perturbation $S_{t,i}$ falls into a one-dimensional subspace. This precisely matches the prerequisite for Zhai's lemma, showcasing a deliberate design in reformulating the iteration.
- **Conceptual Contribution of Separating Strong/Weak Errors**: The biggest insight is not algorithmic but cognitive—pointing out that the community mistook a strong error lower bound for a weak one, thereby closing off a space that could have been explored. This "debunking" of a fake lower bound provides guidance for the entire sampling theory field.
- **Transferability**: The paradigm of "target algorithm = base algorithm + controllable perturbation noise" paired with Zhai's sharp $W_2$ bound can, in principle, be used to analyze weak convergence in other stochastic discretizations, such as diffusion model samplers.

## Limitations & Future Work
- The authors admit that the dependency on dimension $d$ is worse than parallel work, and the dependency on condition number $\kappa$ might not be optimal, requiring tighter high-order moment bounds for improvement.
- The underdamped bound introduces an arbitrary integer $p$ to control a low-probability event, resulting in a cumbersome expression with terms like $\epsilon^{p+2/(4p+3)}$, only cleaning up to $\epsilon^{-1/3}$ when $p\ge3$. A more fundamental treatment of this parameter is missing.
- The analysis strictly relies on **strong log-concavity + well-conditioning** ($\alpha$-strong convexity, $L$-smoothness). Whether the cubic acceleration can be generalized beyond log-concavity (where PLMC was originally established under LSI) remains an open question.
- Bounds for TV and $W_2^2$ are not strictly interchangeable; comparisons with TV-based prior works are only approximate per literature convention and should be viewed with caution.

## Related Work & Insights
- **vs. RLMC (Shen & Lee; Yu et al.)**: RLMC uses randomized midpoints to reduce bias, reaching $\tilde O(\epsilon^{-2/3})$ and hitting the strong $L^2$ error lower bound. Ours uses the PLMC variant + Zhai's weak error analysis to achieve $\tilde O(\epsilon^{-1/3})$ in $W_2$, winning by a quadratic margin because it analyzes weak rather than strong error.
- **vs. Cao et al. (2021) Lower Bound**: They proved that randomized algorithms approximating ULD require $\Omega(\epsilon^{-2/3})$ strong $L^2$ error. Ours does not contradict this but points out that the bound doesn't apply to $W_2$, allowing it to be bypassed under a weak metric.
- **vs. Kandasamy & Nagaraj (2024)**: The original PLMC provided TV convergence under LSI via entropy CLT and required high-order moments. Ours provides $W_2$ convergence under strong log-concavity, avoids high-order moments with the improved Zhai lemma, and results in faster, more applicable bounds.
- **vs. Altschuler et al. (2025, Parallel Work)**: They used dual-midpoint RLMC at low friction to give $\tilde O(\kappa^{5/6}d^{5/3}/\epsilon^{2/3})$ in KL divergence. $\kappa$ is better, but $d$ is worse, and $\epsilon$ is not improved. Ours is better in $\epsilon$ but worse in $d$, making the two complementary across different parameter regimes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to break the widely believed $\epsilon^{-2/3}$ barrier and clarify conceptual confusion over strong/weak error lower bounds.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical but with complete proofs and clear complexity comparisons against several baselines, missing only numerical validation.
- Writing Quality: ⭐⭐⭐⭐ Intuition and proof logic (Section 4) are clear, though the underdamped bound is cluttered by the $p$ parameter.
- Value: ⭐⭐⭐⭐⭐ Advances the understanding of fundamental computational limits in sampling algorithms; the paradigm is transferable to other stochastic discretization analyses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and Beyond](complexity_analysis_of_normalizing_constant_estimation_from_jarzynski_equality_t.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](../../ICML2026/learning_theory/on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)
- [\[ICLR 2026\] Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond](bounds_of_chain-of-thought_robustness_reasoning_steps_embed_norms_and_beyond.md)

</div>

<!-- RELATED:END -->
