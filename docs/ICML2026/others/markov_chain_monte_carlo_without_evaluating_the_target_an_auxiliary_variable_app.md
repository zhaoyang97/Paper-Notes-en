---
title: >-
  [Paper Note] Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach
description: >-
  [ICML 2026][Auxiliary Variables] The authors unify three types of MCMC that "sample without evaluating the target distribution" (Exchange, PoissonMH…
tags:
  - "ICML 2026"
  - "Auxiliary Variables"
  - "Minibatch MCMC"
  - "Gradient-based Proposal"
  - "Doubly-intractable"
  - "Peskun Ordering"
date: 2026-05-08
content_hash: 2e56f11f8523ad23
---

# Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach

**Conference**: ICML 2026 Oral  
**arXiv**: [2406.05242](https://arxiv.org/abs/2406.05242)  
**Code**: https://github.com/ywwes26/Auxiliary-MCMC  
**Area**: Sampling / Bayesian Inference / MCMC  
**Keywords**: Auxiliary Variables, Minibatch MCMC, Gradient-based Proposal, Doubly-intractable, Peskun Ordering  

## TL;DR
The authors unify three types of MCMC that "sample without evaluating the target distribution" (Exchange, PoissonMH, and TunaMH) into a meta-algorithm using auxiliary variables. By introducing auxiliary randomness in both the proposal and the acceptance rate, they design gradient-based MCMC methods (Poisson–Barker, Poisson–MALA, Tuna–SGLD) that maintain exact stationary distributions under minibatch settings, significantly outperforming baselines like PoissonMH, TunaMH, and SGLD in empirical tests.

## Background & Motivation

**Background**: Sampling from a Bayesian posterior $\pi(\theta\mid x)\propto\pi(\theta)\prod_{i=1}^N \mathsf{p}_\theta(x_i)$ becomes expensive in two scenarios: first, when it is doubly-intractable, where the likelihood contains a $\theta$-dependent normalization constant $Z(\theta)$; second, with tall data, where $N$ is extremely large and requires a full data scan at each step. The Exchange algorithm (Murray 2006) addresses the former, while PoissonMH (Zhang & De Sa 2019) and TunaMH (Zhang et al. 2020) handle the latter using Poisson minibatches.

**Limitations of Prior Work**: These three algorithms appear to operate independently—one generates synthetic data to cancel $Z(\theta)$, one decomposes the likelihood into Poisson factors, and another uses minorization tricks. However, they are all restricted to random-walk proposals, which mix slowly in high dimensions. Conversely, gradient-based methods like MALA and HMC must scan the full dataset, contradicting the goal of reducing data access. SGLD attempts to bypass the MH acceptance step with noisy SGD but suffers from persistent bias due to fixed step sizes; "minibatch-based MH correction" has long remained an open problem.

**Key Challenge**: To guarantee an exact stationary distribution, traditional MH must calculate expensive ratios like $\pi(\theta'\mid x)/\pi(\theta\mid x)$. To achieve scalability, it must only observe partial data or synthetic samples. Existing algorithms use techniques like "variable exchange," "Poisson thinning," or "unbiased estimation" to avoid this, but each only solves half the problem (either the acceptance rate does not evaluate the target, or the proposal does not use gradients).

**Goal**: (i) Identify the common structure behind Exchange, PoissonMH, and TunaMH; (ii) extend this structure so the proposal can also utilize auxiliary variables, enabling gradient-based proposals to maintain exact stationary distributions under minibatches; (iii) establish a supporting theory to quantify the gap between the new framework and the "ideal full-data chain."

**Key Insight**: Explicitly decompose the randomness of each MH step into two auxiliary variables $\omega_1$ and $\omega_2$—where $\omega_1$ determines the proposal and $\omega_2$ estimates the target ratio—then prove detailed balance through the lens of involutive MCMC.

**Core Idea**: Use "cheap estimates" to simultaneously replace all expensive terms in both proposal design and acceptance rate calculation. As long as the joint distribution $\mathbb{P}_{\theta, \theta'}(\omega_1, \omega_2)$ matches according to specific rules in the numerator and denominator of the acceptance rate when indices are swapped, $\pi$ remains the invariant distribution.

## Method

### Overall Architecture

The authors first provide a **common substructure** in Section 2: any MH step that uses auxiliary variables to replace $\pi(\theta'\mid x)/\pi(\theta\mid x)$ can be written as "sample $\omega\sim P_{\theta\to\theta'}$ → use $R_{\theta\to\theta'}(\omega)$ as the ratio estimate → accept with $\min\{1,r\}$." Proposition 1 provides the necessary and sufficient conditions: if $R_{\theta\to\theta'}(\omega)\pi(\theta\mid x)P_{\theta\to\theta'}(\omega)=\pi(\theta'\mid x)P_{\theta'\to\theta}(\omega)$, then $R$ is unbiased relative to the true ratio, and the chain is reversible with respect to $\pi$. Exchange, PoissonMH, and TunaMH all satisfy this condition.

In Section 3, this architecture is extended into a "**Dual Auxiliary Variable Meta-Algorithm**" (Algorithm 1):

1. Sample $\omega_1$ from $\mathbb{P}_{\theta_t}(\cdot)$ to determine the proposal kernel $q_{\omega_1}(\theta_t,\cdot)$.
2. Propose $\theta'\sim q_{\omega_1}(\theta_t,\cdot)$.
3. Sample $\omega_2$ from $\mathbb{P}_{\theta_t,\theta'}(\cdot\mid\omega_1)$ to estimate the target ratio.
4. Decide acceptance using the rate $r=\dfrac{\pi(\theta'\mid x)\mathbb{P}_{\theta',\theta_t}(\omega_1,\omega_2)}{\pi(\theta_t\mid x)\mathbb{P}_{\theta_t,\theta'}(\omega_1,\omega_2)}\cdot\dfrac{q_{\omega_1}(\theta',\theta_t)}{q_{\omega_1}(\theta_t,\theta')}$.

Setting $\Omega_1$ or $\Omega_2$ as the singleton space $\mathsf{NULL}$ "turns off" the corresponding auxiliary variable: (Null, Null) is standard MH; (Null, present) reduces to the Section 2 framework; $\omega_1=\omega_2$ is auxiliary MH from the Metropolis-within-Gibbs perspective (Titsias & Papaspiliopoulos 2018); $\omega_1\perp\omega_2$ allows "proposal design" and "ratio estimation" to use independent minibatches. Proposition 2 provides a unified proof of detailed balance by treating $(\theta,\omega_1,\theta',\omega_2)$ as an involution $(\theta',\omega_1, \theta, \omega_2)$ within involutive MCMC.

### Key Designs

1. **Dual Auxiliary Variable Meta-Algorithm**:
    - **Function**: Simultaneously attaches auxiliary variables to both the proposal and the acceptance rate, allowing "gradient-based proposals + minibatch ratio estimation" to coexist while maintaining an exact stationary distribution for the first time.
    - **Mechanism**: All randomness of each MH step is written as $(\omega_1,\theta',\omega_2)$, where the Jacobian of the involution $f(\theta,\omega_1,\theta',\omega_2)=(\theta',\omega_1,\theta,\omega_2)$ is 1. In the acceptance rate $r$, as long as the joint density of $\omega_1, \omega_2$ can cancel expensive terms when $(\theta, \theta')$ are swapped (e.g., the likelihood part of $\mathbb{P}_\theta(\omega_1)$ in PoissonMH is canceled by $\pi(\theta\mid x)$, leaving only minibatch contributions), the entire $r$ depends only on minibatch data.
    - **Design Motivation**: Existing methods are "two-step"—either the proposal uses gradients but the acceptance rate scans full data, or the acceptance rate uses minibatches but the proposal is restricted to random walks. Dual auxiliary variables unify both, allowing the overhead of gradient estimation and ratio estimation to share the same minibatch, reducing the cost per step.

2. **Poisson–Barker / Poisson–MALA: locally balanced PoissonMH**:
    - **Function**: Replaces the random-walk proposal of PoissonMH with gradient-based locally balanced proposals (Barker or MALA type), while reusing the Poisson minibatch sampling of PoissonMH to cancel normalization terms.
    - **Mechanism**: Corresponds to Case 2 ($\omega_1=\omega_2$) of the meta-algorithm. First, draw $\omega_1=(s_1,\dots,s_N)\sim\bigotimes_i \mathsf{Poi}(\lambda M_i/L+\phi_i(\theta;x))$ to form a minibatch $S=\{i\mid s_i>0\}$. The proposal uses a one-dimensional decomposition $q_{\omega_1}^{(g)}(\theta,\theta')=\prod_i q_{\omega_1,i}^{(g)}$, where $q_{\omega_1,i}^{(g)}\propto g(e^{\partial_{\theta_i}\log(\pi(\theta\mid x)\mathbb{P}_\theta(\omega_1))(\theta_i'-\theta_i)})\mu_i(\theta_i'-\theta_i)$. Crucially, the proxy function $\pi(\theta\mid x)\cdot\mathbb{P}_\theta(\omega_1)$ only depends on the minibatch $S$, so each gradient calculation only scans a few thousand points. $g(t)=t/(1+t)$ yields Poisson–Barker, and $g(t)=\sqrt{t}$ yields Poisson–MALA.
    - **Design Motivation**: Locally balanced proposals (Zanella 2020; Livingstone & Zanella 2022) have been proven more robust than MALA under full data. This work brings the benefits of "shaping proposals with gradient information" to minibatch scenarios and cancels full-data calculations by embedding $\omega_1$ into both the proposal and acceptance rate.

3. **Tuna–SGLD: Giving SGLD Exact MH Correction**:
    - **Function**: Replaces the random-walk proposal of TunaMH with an SGLD-style proposal $q_{\omega_1}(\theta,\cdot)\sim\mathcal{N}(\theta-\tfrac{\epsilon^2}{2}\tfrac{N}{K}\sum_{i\in B}\nabla_\theta U_i(\theta;x),\epsilon^2 I)$, and then uses TunaMH’s Poisson minibatch $\omega_2$ to estimate the target ratio, transforming SGLD into an exact sampler for $\pi$.
    - **Mechanism**: Corresponds to Case 3, where $\omega_1=B$ is a uniform minibatch of size $K$ independent of $\omega_2$. Since the marginal distribution of $\omega_1$ does not depend on $\theta$, the $\omega_1$ component in the acceptance rate formula is eliminated, resulting in $r=\dfrac{\pi(\theta'\mid x)\mathbb{P}_{\theta',\theta_t}(\omega_2)}{\pi(\theta_t\mid x)\mathbb{P}_{\theta_t,\theta'}(\omega_2)}\cdot\dfrac{q_{\omega_1}(\theta',\theta_t)}{q_{\omega_1}(\theta_t,\theta')}$. Every step only observes minibatches.
    - **Design Motivation**: SGLD's fixed step-size bias has lacked a clean MH correction scheme for a long time—a problem noted by Welling & Teh (2011). Tuna-SGLD uses TunaMH's auxiliary variables as a "corrector," providing the first solution that transforms SGLD into an exact sampler using only minibatch data.

### Loss & Training
There is no explicit loss function—all algorithms are iterative samplers. Hyperparameters to tune include $\lambda$ for the PoissonMH series (controlling expected minibatch size, typically $\lambda=0.0005L^2$ to $0.01L^2$), the batch size $K$ and step size $\epsilon$ for Tuna–SGLD, and the function $g$ for locally balanced types. Pilot runs adjust the step size to target acceptance rates of 0.25 / 0.4 / 0.55.

## Key Experimental Results

### Main Results

Experiments cover three types of tasks: (i) 20-dimensional heterogeneous truncated Gaussian with $N=10^5$, $\Sigma=\mathrm{diag}(1,0.95,\dots,0.05)$, and tempered posterior ($\beta=10^{-5}$); (ii) 10-dimensional robust Student-$t$ linear regression with $N=10^5, \nu=4$; (iii) Bayesian logistic regression on MNIST. Metrics include MSE over time and min/median/max ESS/s per dimension.

| Task | Method | Best ESS/s (Min, Med, Max) |
|------|------|------|
| Heterogeneous Gaussian | MH | (0.05, 0.08, 0.47) |
| Heterogeneous Gaussian | MALA | (0.10, 0.19, 2.77) |
| Heterogeneous Gaussian | Barker | (0.12, 0.22, 1.53) |
| Heterogeneous Gaussian | PoissonMH | (0.40, 0.66, 4.67) |
| Heterogeneous Gaussian | Poisson–Barker | (0.91, 1.65, 12.16) |
| Heterogeneous Gaussian | Poisson–MALA | (0.84, 1.65, 23.84) |

Poisson–Barker improves by 1.37–7.12× over PoissonMH, 4.39–9.80× over MALA, 6.58–15.58× over Barker, and 13.62–70.12× over random-walk MH on the heterogeneous Gaussian task. On robust linear regression, Poisson–{MALA, Barker} improves by 1.80–8.79× over PoissonMH and reaches nearly 100× improvement over full-data methods. SGLD's MSE drops quickly early on but soon hits a plateau caused by fixed step-size bias; Tuna–SGLD converges fastest on MNIST Bayesian logistic regression without the bias plateau of SGLD.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Full Poisson–Barker | Full gradient-based minibatch MH, best ESS/s | Benefits from both gradient proposal and Poisson minibatch ratio estimation. |
| Removing Gradient (= PoissonMH) | ESS/s drops by 1.4–7× | Validates the contribution of locally balanced proposals. |
| Removing MH Correction (= SGLD) | MSE drops fast early on but converges to a biased distribution | Without auxiliary variable correction, it returns to the standard SGLD bias issue. |
| Replacing Barker with MALA | Matches best at high acceptance rates; worse than PoissonMH at low rates (0.25) | MALA is more sensitive to step size, whereas Barker is more robust, consistent with Livingstone & Zanella 2022. |
| Different $\lambda$ (Minibatch size) | Monotonic change in acceptance rate/ESS | Smaller $\lambda$ is noisy but cheap; larger $\lambda$ approaches full-data costs. |

### Key Findings

- The core contribution is the coupling of "gradient proposals" and "minibatch acceptance rates"; missing either results in degradation (missing gradients = PoissonMH/TunaMH, missing correction = SGLD).
- Poisson–Barker is the least sensitive to acceptance rates and is the recommended default; Poisson–MALA is comparable at high acceptance rates but becomes fragile at lower ones.
- Tuna–SGLD provides the first feasible solution for "correcting SGLD with minibatches," answering the open question from Welling & Teh (2011).
- The Peskun ordering $\mathbb{P}_{\mathsf{aux}}\prec\mathbb{P}_{\mathsf{MwG}}\prec\mathbb{P}_{\mathsf{ideal}}$ guarantees that the asymptotic variance of the ideal full-data chain is always the lower bound for the new chain, ensuring no "free lunch."

## Highlights & Insights

- **Unified Perspective**: Reducing three seemingly unrelated classes of algorithms (Exchange for doubly-intractable, PoissonMH and TunaMH for tall-data) to a single detailed balance equation $R\cdot\pi P=\pi' P'$ is a clean conceptual contribution, identifying them as special cases where $\omega_1=\mathsf{NULL}$.
- **Involutive MCMC Lens**: By involuting $(\omega_1, \theta', \omega_2)$ together, the validity proof is reduced to a single line, leaving room for future extensions (e.g., generalizing acceptance functions $a(t)$ or non-trivial involutions $\varphi$).
- **Proxy Function Ingenuity**: In Poisson–Barker, replacing the gradient $\partial_\theta\log\pi$ with $\partial_\theta\log(\pi\cdot\mathbb{P}_\theta(\omega_1))$ appears to simply add a $\log \mathbb{P}_\theta(\omega_1)$ term. However, because $\pi(\theta\mid x)\mathbb{P}_\theta(\omega_1)$ only depends on the minibatch $S$, the gradient computation cost remains identical to PoissonMH—this "adding a term to make expensive cheap" proxy design is a transferable trick for other minibatch algorithms.
- **Theoretical Byproducts**: Using the unified framework to reprove spectral gaps for PoissonMH and TunaMH. The PoissonMH bound $\mathrm{Gap}(\mathbb{P}_{\mathsf{aux}})\ge \max\{\tfrac12 e^{-L^2/(\lambda+L)},e^{-1/e}e^{-L^2/(2\lambda)}\}\mathrm{Gap}(\mathbb{P}_{\mathsf{ideal}})$ is strictly better than Zhang & De Sa (2019) when $\lambda>L$, and the TunaMH bound $e^{-1/(2\chi)}$ is strictly better than $e^{-1/\chi-2\sqrt{\log 2/\chi}}$.

## Limitations & Future Work

- **Dependence on PoissonMH/TunaMH Technical Assumptions**: The likelihood must be expressible in a form that can be canceled by Poisson minibatches (e.g., bounded Lipschitz $U_i$). Efficiency on non-exponential families or heavy-tailed likelihoods has not been verified.
- **Small Experimental Scale**: The largest dataset is MNIST + Bayesian logistic regression. Evaluation on deep Bayesian neural networks or LLM-scale posteriors is missing, leaving a gap before practical large-model Bayesian inference.
- **Gradient Estimation Variance**: The proposal in Tuna-SGLD uses standard minibatch gradients; high variance can lead to decreased acceptance rates. Natural extensions include control variates, SVRG-style gradient estimation, or preconditioning.
- **Relative Theoretical Bounds**: All spectral gap bounds are multiplicative constants relative to $\mathbb{P}_{\mathsf{ideal}}$; the mixing speed of the ideal chain itself still requires separate analysis. Empirical comparisons with PDMP-type samplers (Bouncy Particle, Zig-Zag) are missing.

## Related Work & Insights

- **vs Exchange / PoissonMH / TunaMH**: These are all special cases of this framework where $\omega_1=\mathsf{Null}$. The extension here allows $\omega_1$ to enter the proposal, enabling gradient information in minibatch MH for the first time.
- **vs Pseudo-marginal MCMC (Andrieu & Roberts 2009)**: Pseudo-marginal incorporates estimator values into the chain state, whereas this work retains only $\theta$ and regenerates auxiliary variables at each step, representing a different augmentation construction.
- **vs Involutive MCMC (Neklyudov et al. 2020)**: This paper provides a specific and practical family of instances for involutive MCMC, clarifying concrete applications for "dual auxiliary variables" (Locally Balanced + Minibatch).
- **vs PDMP Subsampling (Bouchard-Côté et al. 2018; Bierkens et al. 2019)**: They use continuous time + Poisson thinning to avoid full data scans. This work follows the discrete MH route, making the two approaches complementary in engineering.
- **vs SGLD (Welling & Teh 2011)**: Tuna-SGLD is an exact version of SGLD, directly answering the open question of "minibatch-based MH correction" left in the original paper.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies doubly-intractable and tall-data MCMC routes into a dual auxiliary variable framework for the first time, constructing exact gradient-based minibatch MCMC.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks cover synthetic and real data with sufficient baseline comparisons; lacks posterior experiments on deep models.
- Writing Quality: ⭐⭐⭐⭐⭐ Framework introduction is clear, Propositions 1–2 are concise and powerful, and the mappings to Case 1/2/3 and Algorithms 2/3 are easy to follow.
- Value: ⭐⭐⭐⭐⭐ Represents both a conceptual contribution unifying old algorithms and a practical algorithm replacement for PoissonMH/TunaMH/SGLD, while providing sharper spectral gap bounds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[ICML 2026\] Variable Clustering via Distributionally Robust Nodewise Regression](variable_clustering_via_distributionally_robust_nodewise_regression.md)
- [\[ICML 2026\] Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification](target-agnostic_calibration_under_distribution_shift_with_frequency-aware_gradie.md)
- [\[AAAI 2026\] Learning Network Dismantling Without Handcrafted Inputs](../../AAAI2026/others/learning_network_dismantling_without_handcrafted_inputs.md)

</div>

<!-- RELATED:END -->
