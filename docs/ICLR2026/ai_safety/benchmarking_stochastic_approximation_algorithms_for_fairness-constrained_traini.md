---
title: >-
  [Paper Note] Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks
description: >-
  [ICLR 2026][AI Safety][benchmark] This paper formalizes the "training of fair deep networks" as a **stochastic optimization problem with inequality constraints** (specifically, loss differences between subgroups). It points out that no existing algorithm currently provides convergence guarantees across the full spectrum of "stochastic + inequality + no
tags:
  - ICLR 2026
  - AI Safety
  - benchmark
date: 2026-05-08
content_hash: 20a7f836e2a8b3f3
---
# Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=JxmjzC6syB](https://openreview.net/forum?id=JxmjzC6syB)  
**Code**: https://github.com/humancompatible/train  
**Area**: AI Safety / Fair Machine Learning  
**Keywords**: Fairness constraints, constrained ERM, stochastic approximation, augmented Lagrangian, benchmark

## TL;DR
This paper formalizes the "training of fair deep networks" as a **stochastic optimization problem with inequality constraints** (specifically, loss differences between subgroups). It points out that no existing algorithm currently provides convergence guarantees across the full spectrum of "stochastic + inequality + non-convex + non-smooth" scenarios. Consequently, it selects three types of stochastic approximation algorithms from the literature that best fit this scenario but lacked implementation, integrates them into a Python toolbox, and provides the first systematic comparison of their optimization performance and fairness behavior on large-scale real-world US Census data (Folktables/ACSIncome).

## Background & Motivation
**Background**: Training fair models using constraints is a widely studied direction. Regulations such as the EU AI Act require the elimination of bias, and a natural way to translate "bias elimination" into a training objective is to add constraints to Empirical Risk Minimization (ERM) that limit the empirical risk differences between subgroups. Over the past five years, numerous algorithms have been proposed to solve such constrained ERM problems (e.g., sequential quadratic programming, augmented Lagrangian, switching subgradient).

**Limitations of Prior Work**: Despite the abundance of algorithms, there is **no standard, recognized method for constrained training**, nor is there a **unified toolbox** to implement and compare these algorithms horizontally. Furthermore, there is a lack of **real-world large-scale benchmarks** to test the combined effects of different design choices (sampling methods, first-order/higher-order derivatives, globalization strategies). Many theoretically analyzed algorithms have **never been implemented** and remain purely theoretical.

**Key Challenge**: Fairness-constrained training simultaneously possesses three properties that make optimization difficult: both the objectives and constraints are **large-scale stochastic** (requiring sampling), constraints are **inequalities** rather than equalities, and both are **non-convex and non-smooth** due to the use of neural networks. Most algorithms in the literature cover only one or two of these properties: most do not consider stochastic constraints; among those that do, only three allow inequality constraints; and among those, only one allows non-smooth objectives. In other words, **an algorithm that satisfies all scenarios with convergence guarantees does not currently exist**.

**Goal**: (1) Unify fairness-constrained training into a computable constrained ERM template; (2) filter out implementable algorithms from theoretical literature and implement them; (3) compare their optimization and fairness performance on real data.

**Key Insight**: Rather than inventing a new algorithm, it is better to first build a **benchmark that honestly reflects the difficulty**—using real, large-scale data (like US Census data) known to exhibit bias under ERM, and comparing "hard constraint" approaches against "penalty regularization" approaches on the same scale.

**Core Idea**: Treat fairness as **hard constraints** on loss differences rather than soft penalties, and use a unified toolbox to implement three classes of stochastic approximation algorithms, providing the community with a reproducible "constrained training arena."

## Method

### Overall Architecture
The paper addresses the constrained empirical risk minimization problem: minimize a stochastic objective $\mathbb{E}[f(x,\xi)]$ over parameters $x\in\mathbb{R}^n$, subject to stochastic inequality constraints $\mathbb{E}[c(x,\zeta)]\le 0$, where $\xi$ controls objective sampling and $\zeta$ controls constraint sampling. The pipeline is: **Translate fairness requirements into constraint functions $c$** (Table 1, subgroup loss difference $\le\delta$) → **Screen algorithms from literature candidates that can handle "stochastic + inequality + non-convex + non-smooth"** (Table 3) → **Implement the three selected algorithm classes into a unified toolbox** → **Execute a unified evaluation protocol on Folktables/ACSIncome data** (fairness metrics + inaccuracy + Wasserstein distance), comparing them with unconstrained SGD and penalty-regularized SGD baselines.

As a benchmark/toolbox paper, the core contribution lies in converging a fragmented algorithm lineage into a comparable experimental framework, focusing on the "formalization-screening-implementation-evaluation" chain.

### Key Designs

**1. Unifying Fairness as Loss-Difference Constrained ERM**

Fairness definitions vary (Statistical Parity, Equal Opportunity, Equalized Odds, etc.). Directly adding constraints to discrete quantities like "accuracy/pass rate" makes labels **piecewise constant and non-differentiable**, causing first-order methods to fail. The paper's approach is to revert to the **loss level**: use a continuous loss $\ell$ to measure the average risk of each subgroup $s_i$: $\ell_{s_i}(\theta)=\frac{1}{|D[s_i]|}\sum_{X,Y\in D[s_i]}\ell(f_\theta(X),Y)$, and then constrain the deviation of each subgroup's loss from the mean of all groups within a bandwidth $\delta$:

$$\min_{\theta}\ \frac{1}{N}\sum_{i=1}^N \ell(f_\theta(X_i),Y_i)+R(\theta)\quad\text{s.t.}\quad -\delta\le \ell_{s_i}(\theta)-\frac{1}{m}\sum_{j=1}^m\ell_{s_j}(\theta)\le\delta,\ i=1,\dots,m.$$

$m$ subgroups yield $m$ inequality constraints. Since the loss is continuous, constraints can be integrated into a "non-smooth non-convex optimization" framework. The paper emphasizes the "hard constraint vs. soft penalty" debate: although hard constraints make optimization harder, **each constraint corresponds to a clear target, allowing practitioners to specify them individually**, whereas penalty weights depend on the dataset and hide the "accuracy vs. fairness" trade-off in an opaque weighted sum.

**2. Screening Algorithms via a "Full Scenario Coverage" Criterion**

The paper uses a "hypothesis table" (Table 3) to categorize constrained ERM algorithms by **assumptions on objective $F$, assumptions on constraint $C$, and handling of stochastic/inequality constraints**. The conclusion is stark: most algorithms do not consider **stochastic constraints**; among those that do, only **three** allow **inequality constraints**; and except for Huang & Lin (2023), all require $F$ to be at least $C^1$ (continuously differentiable), failing the **non-smoothness** of neural networks. Following Davis et al., assuming the objective and constraints are "tame and locally Lipschitz" is appropriate, but an algorithm satisfying this while providing convergence guarantees **does not yet exist**. Consequently, the paper identifies three algorithms that best approximate the full scenario.

**3. Implementing Three Classes of Previously Unimplemented Stochastic Approximation Algorithms**

The three selected algorithm classes follow distinct technical routes, all implemented in the unified toolbox (with Augmented Lagrangian divided into ALM with and without smooth terms, totaling four solvers):

- **Stochastic Ghost (StGh)**: Solves a quadratic programming subproblem for the descent direction $d$ ($\min_d \nabla_J f(x_k)^\top d+\frac{\tau}{2}\|d\|^2$, subject to $c_J(x_k)+\nabla_J c(x_k)^\top d\le \kappa_k^J e$ and $\|d\|_\infty\le\beta$), followed by a line search. It uses **geometric distribution sampling + a multi-level Monte Carlo combination of four mini-batches** to construct an **unbiased estimate** $d(x_k)$: sample $N\sim G(p_0)$, split a mini-batch of size $2^{N+1}$ into odd and even halves, and de-bias via $\frac{d(X^{2^{N+1}})-\frac12[d(\text{odd})+d(\text{even})]}{(1-p_0)^N p_0}+d(X^1)$. This requires larger mini-batches.

- **SSL-ALM (Stochastic Smoothed Linearized ALM)**: Based on the augmented Lagrangian $L_\rho(x,y)=F(x)+y^\top C(x)+\frac{\rho}{2}\|C(x)\|^2$, it adds a smoothing variable $z$ to obtain a proximal AL function $K_{\rho,\mu}(x,y,z)=L_\rho(x,y)+\frac{\mu}{2}\|x-z\|^2$, interpreted as inexact gradient descent on the Moreau envelope. Its main advantage is **not requiring large mini-batches**: each step samples one $\xi$ for the objective and two $\zeta$ for constraints and their Jacobians, alternatingly updating $(x,y,z)$. Inequality constraints are handled via slack variables. Removing the smoothing term ($\mu=0$) reverts to standard ALM.

- **Stochastic Switching Subgradient (SSw)**: The only algorithm allowing **non-smooth, weakly convex** objectives by using subgradients. Each step estimates constraint values $c_J(x_k)$: if below tolerance $\epsilon_k$ (current point is feasible), it takes a **subgradient step for the objective**; otherwise, it takes a **subgradient step for the constraint**—switching between "reducing objective" and "repairing feasibility."

**4. Folktables Real-World Large-Scale Benchmark and Unified Protocol**

Realism is supported by the ACSIncome dataset (from the American Community Survey, ~3.5 million households annually). The task is binary classification of individual income > $50k. The paper provides two settings: **binary protected attributes** (Oklahoma, race binarized as "White/Non-White", 17,917 samples) and **multi-valued protected attributes** (Virginia, 5 marital status values). It can define up to 5.7 billion protected subgroups, scaling to industrial levels. The protocol reports three fairness metrics: **Independence (Ind)** (Statistical Parity), **Separation (Sp)** (Equalized Odds), and **Sufficiency (Sf)**, plus Inaccuracy (Ina) and subgroup Wasserstein distance (Wd). Results are averaged over 10 runs.

### Loss & Training
The loss $\ell$ is **Binary Cross-Entropy with logits**: $\ell(f_\theta(X_i),Y_i)=-Y_i\log\sigma(f_\theta(X_i))-(1-Y_i)\log(1-\sigma(f_\theta(X_i)))$. The network consists of two hidden layers (64, 32) + ReLU. No regularization is added to the constraints ($R=0$). In binary settings, the difference between the two subgroups' losses is constrained. Hyperparameters are grid-searched (e.g., SSL-ALM: $\mu=2, \rho=1, \tau=0.01, \eta=0.05, \beta=0.5$). Baselines include unconstrained SGD and SGD-Fairret (using the Fairret library for accuracy fairness regularization).

## Key Experimental Results

### Main Results (Binary Protected Attribute, ACSIncome / Oklahoma / Race)
The following table shows fairness metrics (Ind, Sp, Sf), Inaccuracy (Ina), and subgroup Wasserstein distance (Wd) on the test set (10-run mean, **all lower is better**):

| Method | Ind | Sp | Ina | Sf | Wd |
|------|------|------|------|------|------|
| SGD (Unconstrained) | 0.098 | 0.155 | **0.209** | 0.061 | 0.183 |
| SGD-Fairret (Penalty) | 0.091 | 0.141 | 0.211 | 0.056 | 0.059 |
| StGh | 0.086 | 0.123 | 0.239 | 0.057 | 0.161 |
| ALM | 0.058 | 0.114 | 0.244 | 0.221 | 0.158 |
| **SSL-ALM** | 0.083 | **0.108** | 0.223 | **0.046** | 0.170 |
| SSw | 0.103 | 0.168 | 0.212 | 0.066 | **0.018** |

Key readings: Constrained methods generally suppress fairness metrics better than unconstrained SGD. **SSL-ALM / ALM provides the best trade-off**, improving Ind, Sp, and Sf with only a modest increase in Ina (from 0.209 to ~0.223). **StGh** achieves good sufficiency but higher inaccuracy and trajectory variance. **SSw** achieves the best constraint satisfaction (Wd only 0.018, almost perfect feasibility), but fails to reduce the objective significantly, with fairness metrics and inaccuracy returning to levels similar to unconstrained SGD.

### Ablation Study (Multi-valued Attributes + Hard Constraints vs. Soft Penalties)

| Configuration | Observation | Explanation |
|------|------|------|
| SGD (Unconstrained) | Subgroup constraints continuously violated | Real data learn bias; benchmark difficulty is authentic |
| SGD-Fairret (Penalty) | Requires extensive $\lambda$ tuning | Performance is highly sensitive to $\lambda$ (affects trade-off and optimization) |
| Three Constrained Algorithms | Stable loss reduction under constraints | Less sensitive to hyperparameters (esp. SSL-ALM); trade-offs are controllable |

### Key Findings
- **No Silver Bullet**: The four algorithms have different trade-offs—SSL-ALM/ALM is most balanced, SSw favors feasibility, and StGh favors sufficiency but is unstable. No single algorithm dominates.
- **Value of Hard Constraints**: Penalty regularization requires expensive tuning of $\lambda$, and trade-offs are opaque. Constrained methods are more robust and offer controllable trade-offs.
- **Generalization Bias**: AL-based methods satisfy constraints on training data, but constraint values slightly "overshoot" on test sets, reflecting the generalization behavior of fairness constraint estimators.

## Highlights & Insights
- **Honest "Difficulty Map"**: Using a hypothesis table (Stochastic/Inequality/Non-convex/Non-smooth) to map the literature and conclude that a comprehensive algorithm is missing is more valuable than proposing a single new algorithm.
- **Bridging Theory and Practice**: StGh, SSL-ALM, and SSw previously only existed "on paper." This paper provides a unified pip-installable implementation.
- **Real-world Scale**: Use of Folktables/ACSIncome avoids the pitfall of "toy" fairness datasets that do not consistently exhibit bias.
- **Transferable Mechanisms**: Techniques like Multilevel Monte Carlo de-biasing in StGh and switching logic in SSw are transferable to any task with stochastic inequality constraints (e.g., robustness, Lipschitz, or resource constraints).

## Limitations & Future Work
- **Limitations**: No convergence guarantees are provided for the specific fairness-constrained problem setup. The experimental network is small (194 params), and the scale is limited to specific census subsets. Generalization guarantees for fairness constraint estimators were explicitly out of scope.
- **Future Work**: Developing algorithms with guarantees for the "stochastic + inequality + non-convex + non-smooth" quadrant is the natural next step. The benchmark could extend to larger networks, diverse domains (credit, medical), and more complex constraint forms.

## Related Work & Insights
- **vs. Penalty Methods (Fairret / HSIC)**: These treat fairness as a soft penalty. Hard constraints provide clearer targets and more robust hyperparameters.
- **vs. Rate Constraints (Cotter et al. 2019)**: They constrain piecewise constant prediction rates, making first-order methods unusable. This paper uses continuous loss differences to enable first-order optimization.
- **vs. Existing Toolboxes (AIF360 / FairLearn)**: Existing toolboxes often focus on evaluation or a single Lagrangian route. This paper fills the gap of a unified platform for diverse stochastic approximation algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ (High value in visualizing research gaps and unified implementation of paper-only algorithms)
- Experimental Thoroughness: ⭐⭐⭐ (Solid across several settings, but limited by small network size and single dataset type)
- Writing Quality: ⭐⭐⭐⭐ (Clear formalization and logical screening via the hypothesis table)
- Value: ⭐⭐⭐⭐ (A reproducible open-source benchmark and toolbox to catalyze research in constrained optimization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](../../ECCV2024/ai_safety/unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICLR 2026\] Robustify Spiking Neural Networks via Dominant Singular Deflation under Heterogeneous Training Vulnerability](robustify_spiking_neural_networks_via_dominant_singular_deflation_under_heteroge.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
