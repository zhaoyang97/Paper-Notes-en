---
title: >-
  [Paper Note] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration
description: >-
  [ICML2025][Fairness/Online Learning][multicalibration] Proposes to reduce online $\ell_1$-multicalibration to a newly defined Online Linear-Product Optimization (OLPO) problem, achieving improved and oracle-efficient online multicalibration error bounds of $\widetilde{O}(T^{-1/3})$ and $\widetilde{O}(T^{-1/4})$, respectively.
tags:
  - "ICML2025"
  - "Fairness/Online Learning"
  - "multicalibration"
  - "online learning"
  - "fairness"
  - "oracle-efficient"
  - "calibration"
  - "omniprediction"
date: 2026-05-08
content_hash: e0ef51611fc796f9
---

# Improved and Oracle-Efficient Online $\ell_1$-Multicalibration

**Conference**: ICML2025  
**arXiv**: [2505.17365](https://arxiv.org/abs/2505.17365)  
**Code**: To be confirmed  
**Area**: Fairness/Online Learning  
**Keywords**: multicalibration, online learning, fairness, oracle-efficient, calibration, omniprediction

## TL;DR
Proposes to reduce online $\ell_1$-multicalibration to a newly defined Online Linear-Product Optimization (OLPO) problem, achieving improved and oracle-efficient online multicalibration error bounds of $\widetilde{O}(T^{-1/3})$ and $\widetilde{O}(T^{-1/4})$, respectively.

## Background & Motivation

### Calibration and Multicalibration
- **Calibration** is a classic metric for measuring the quality of probabilistic predictions: for each predicted value $p$, the proportion of positive instances among the samples where this prediction occurs should converge to $p$. Foster & Vohra (1998) proved that online calibration under the $\ell_1$ metric is achievable at a rate of $O(T^{-1/3})$.
- **Multicalibration**, introduced by Hébert-Johnson et al. (2018), requires that predictions are calibrated not only overall but also across multiple subpopulations defined by a hypothesis class $\mathcal{H}$. This directly relates to algorithmic fairness, avoiding discriminatory predictions for specific subpopulations defined by gender, race, age, etc.

### Limitations of Prior Work

**Indirect methods leading to rate loss**: Prior works (Gupta et al. 2022; Lee et al. 2022) first obtain multicalibration guarantees under the $\ell_\infty$ or $\ell_2$ norm, and then convert them to $\ell_1$, resulting in additional loss. The best indirect rate is only $\widetilde{O}(T^{-1/4})$.

**High computational complexity**: Most algorithms have running times linear in $|\mathcal{H}|$, which is infeasible when the hypothesis class is exponentially large (e.g., $d$-dimensional linear functions).

**Poor rate of oracle-efficient methods**: The oracle-efficient algorithm of Garg et al. (2024) only achieves a rate of $\widetilde{O}(T^{-1/8})$ and requires a stronger online regression oracle.

### Core Problem
> Can we design an online multicalibration algorithm that directly operates under the $\ell_1$ metric, achieves a rate of $O(T^{-1/3})$, and is oracle-efficient?

## Method

### Core Reduction: Multicalibration to OLPO

The key insight of this paper is defining a new online learning problem called **Online Linear-Product Optimization (OLPO)**:

- In each round $t$, the learner chooses $(h_t, \boldsymbol{\theta}_t) \in \mathcal{H} \times \mathbb{B}_\infty$.
- The adversary reveals context $\mathbf{x}_t$ and reward vector $\boldsymbol{f}_t \in \mathbb{R}^M$.
- The learner receives a product-form reward $\langle \boldsymbol{\theta}_t, h_t(\mathbf{x}_t) \cdot \boldsymbol{f}_t \rangle$.

**Theorem 3.1** proves that any algorithm with an OLPO regret bound of $R_T(\mathcal{L}; \mathcal{H})$ can be efficiently transformed into an online multicalibration algorithm with a multicalibration error of at most
$$\frac{B}{m} + \frac{R_T(\mathcal{L}; \mathcal{H})}{T} + 4B\sqrt{\frac{m\log(6T|\mathcal{H}|)}{T}} + \frac{4mB}{T}$$
The reduction relies on a **Halfspace Oracle**, which originates from the connection between calibration and Blackwell approachability proposed by Abernethy et al. (2011).

### Improving Rate Method: Lin-OLPO

To solve OLPO (whose reward function involves the product of variables and is not standard linear optimization), the paper introduces a **linearized version (Lin-OLPO)**:

1. **Space expansion**: Expand the decision variables from $\mathcal{H} \times \mathbb{B}_\infty$ to the mixed-norm ball $\mathbb{B}_{1,\infty}$ in $\mathbb{R}^{|\mathcal{H}| \times M}$ space.
2. **Mixed Norm**: Define $\|\mathbb{z}\|_{1,\infty} = \sum_{h \in \mathcal{H}} \|\mathbb{z}(h)\|_\infty$, whose dual norm is $\|\mathbb{z}\|_{\infty,1} = \max_{h} \|\mathbb{z}(h)\|_1$.
3. **Consistency guarantee**: Ensure consistency between the optimal actions of OLPO and Lin-OLPO by probabilistically sampling $h_t = h$ with probability $\gamma_t(h) = \|\widetilde{\boldsymbol{\theta}}_t(h)\|_\infty$ and scaling $\boldsymbol{\theta}_t$.

**Algorithm for solving Lin-OLPO (Algorithm 2)**:
- Run $|\mathcal{H}|$ parallel instances of Online Gradient Descent (OGD) $\mathcal{A}^h$, each responsible for the calibration optimization of hypothesis $h$.
- Use Multiplicative Weights Update (MWU) as the expert algorithm $\mathcal{E}$ to select "harder" hypotheses (i.e., $h$ with larger calibration error).
- Finally achieve a regret of $\widetilde{O}(\sqrt{T \log |\mathcal{H}|})$.

**Theorem 1.1**: For a finite hypothesis class $\mathcal{H}$, setting $m = T^{1/3}$ yields
$$\mathbb{E}[K(\pi_T, \mathcal{H})] \leq O(BT^{-1/3}\sqrt{\log(6T|\mathcal{H}|)})$$

### Extension to Infinite Hypothesis Classes

**Theorem 1.2**: By utilizing the 1-Lipschitz property of the $\ell_1$-multicalibration error with respect to $\mathcal{H}$ and constructing a $\beta$-covering $\mathcal{H}_\beta$, the error bound is generalized to:
$$\mathbb{E}[K(\pi_T, \mathcal{H})] \leq O(BT^{-1/3}\sqrt{\log(6T|\mathcal{H}_\beta|)}) + \beta$$

**Corollary 1.3**: For a class of $d$-dimensional bounded linear functions, the rate is $O(Bd^{1/2}T^{-1/3}\log(BT))$.

### Oracle-Efficient Method

Lin-OLPO requires enumerating all $h \in \mathcal{H}$, which is infeasible for large hypothesis classes. The paper directly applies the oracle-efficient framework to OLPO:

- Adopts the **Follow-the-Perturbed-Leader (FTPL)** family of algorithms (Dudík et al. 2020).
- Exploits the special structure of OLPO: the decisions can be restricted to the Boolean hypercube $\{±1\}^M$.
- Defines an offline oracle: given the history sequence, solves the optimal $(h^*, \boldsymbol{\theta}^*)$ in one shot—essentially equivalent to offline multicalibration error evaluation.
- Invokes the offline oracle only once per round.

**Theorem 1.4**: Under the transductive or sufficiently separated contexts assumption, for a binary hypothesis class $\mathcal{H}: \mathcal{X} \to \{0,1\}$,
$$\mathbb{E}[K(\pi_T, \mathcal{H})] \leq \widetilde{O}(T^{-1/4}\sqrt{\log T})$$

## Key Experimental Results

This paper is a purely theoretical work with no experimental section. Comparison of the main theoretical results:

| Method | $\ell_1$-Multicalibration Rate | Computational Complexity | Oracle Type |
|------|---------------------|-----------|-------------|
| Gupta et al. (2022) via $\ell_\infty$ conversion | $\widetilde{O}(T^{-1/4})$ | $O(\|\mathcal{H}\|)$/round | None |
| Garg et al. (2024) | $\widetilde{O}(T^{-1/8})$ | Oracle-efficient | Online regression oracle |
| **Ours (Improved Rate)** | $\widetilde{O}(T^{-1/3})$ | $O(\|\mathcal{H}\|)$/round | None |
| **Ours (Oracle-Efficient)** | $\widetilde{O}(T^{-1/4})$ | Polynomial/round | Offline oracle |
| Online calibration lower bound reference | $O(T^{-1/3})$ | — | — |

Key Improvements:
- The improved rate almost matches the optimal $O(T^{-1/3})$ rate of online calibration (where online calibration is a special case of multicalibration).
- The oracle-efficient rate is improved from $T^{-1/8}$ to $T^{-1/4}$, and the oracle requirement is downgraded from online to offline (which is weaker and easier to implement).

## Highlights & Insights

1. **Elegance of the OLPO reduction**: Reducing multicalibration, a complex fairness-constrained problem, to a clearly structured online optimization problem not only unifies the two goals of improved rate and oracle efficiency but also yields independent theoretical value.
2. **Linearization technique**: Linearizing the non-linear reward containing product terms through an elegant design in the mixed-norm space while maintaining consistency of optimal actions—this is the core technical innovation.
3. **Exploitation of the Lipschitz property**: The 1-Lipschitz property of the $\ell_1$-multicalibration error with respect to $\mathcal{H}$ is an unexplored structural property, facilitating a natural generalization from finite to infinite hypothesis classes.
4. **Offline vs. online oracle**: Lowering the oracle requirement from online regression to offline evaluation is a significant improvement in practicality—an offline oracle only requires one-shot optimization without maintaining online states.

## Limitations & Future Work

1. **Assumption constraints on the oracle-efficient method**: Theorem 1.4 requires (a) binary hypothesis class $\mathcal{H}: \mathcal{X} \to \{0,1\}$, and (b) transductive or sufficiently separated contexts. Whether these assumptions can be relaxed remains an open question.
2. **Rate gap**: The improved rate $T^{-1/3}$ requires enumerating $\mathcal{H}$, while the oracle-efficient version degrades to $T^{-1/4}$—whether a rate of $T^{-1/3}$ can be achieved under oracle efficiency remains unresolved.
3. **Lack of experimental validation**: As a purely theoretical work, no empirical validation on real-world datasets is provided.
4. **Relation to Noarov et al. (2025)**: A concurrent work obtained similar results for finite hypothesis classes via a different framework, but required a small-loss regret bound, whereas this paper only requires a worst-case regret bound, resulting in a simpler algorithm.
5. **Weaker data assumptions**: The authors suggest that the results can be adapted to smoothed data or K-hint data assumptions (Haghtalab et al. 2022; Block et al. 2022), but this has not been completed in this paper.

## Related Work & Insights

- **Connection between calibration and online learning** (Abernethy et al. 2011): The halfspace oracle in this paper is directly inherited from this work's connection between calibration and Blackwell approachability.
- **Multicalibration and Omniprediction** (Gopalan et al. 2022): An $\ell_1$-multicalibrated predictor automatically becomes an omnipredictor, meaning the improvements in this paper directly enhance the rate of online omniprediction.
- **Oracle-efficient online learning** (Syrgkanis et al. 2016; Dudík et al. 2020): The FTPL method in this paper is built directly upon the generalized FTPL framework of Dudík et al.
- **Online multigroup learning** (Acharya et al. 2024; Deng et al. 2024): A related but distinct problem; the latter is designed for binary labels and loss functions and does not directly apply to OLPO.

## Rating
- Novelty: ⭐⭐⭐⭐ (The OLPO reduction and linearization technique are novel contributions)
- Experimental Thoroughness: ⭐⭐ (Pure theoretical work, no experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure with well-defined reduction steps)
- Value: ⭐⭐⭐⭐ (Significantly improves the optimal rate of online multicalibration, with direct implications for fair ML)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Oracle-Efficient Hybrid Online Learning with Constrained Adversaries](../../ICLR2026/learning_theory/oracle-efficient_hybrid_learning_with_constrained_adversaries.md)
- [\[ICML 2025\] Avoiding Catastrophe in Online Learning by Asking for Help](avoiding_catastrophe_in_online_learning_by_asking_for_help.md)
- [\[ICML 2025\] Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)
- [\[ICML 2025\] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)
- [\[ICLR 2026\] Efficient Testing for Correlation Clustering: Improved Algorithms and Optimal Bounds](../../ICLR2026/learning_theory/efficient_testing_for_correlation_clustering_improved_algorithms_and_optimal_bou.md)

</div>

<!-- RELATED:END -->
