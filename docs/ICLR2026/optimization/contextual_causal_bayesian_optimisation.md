---
title: >-
  [Paper Note] Contextual Causal Bayesian Optimisation
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper proposes CoCa-BO, which unifies the selection of "which variables to intervene on" (causal scope selection) and "what values to intervene with" (contextual Bayesian optimization) into a single search process. It uses a multi-armed bandit to select scopes among all "possibly-optimal mixed policy scopes" (POMP
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 44b69f31946f03ea
---
# Contextual Causal Bayesian Optimisation

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=QW0PchhVaD](https://openreview.net/forum?id=QW0PchhVaD)  
**Code**: Available (Pyro implementation, paper promises release)  
**Area**: optimization  
**Keywords**: Bayesian optimization, causal inference, contextual optimization, mixed policy scope, multi-armed bandits, regret bounds

## TL;DR
This paper proposes CoCa-BO, which unifies the selection of "which variables to intervene on" (causal scope selection) and "what values to intervene with" (contextual Bayesian optimization) into a single search process. It uses a multi-armed bandit to select scopes among all "possibly-optimal mixed policy scopes" (POMPS) and Gaussian processes to select intervention values within each scope, providing the first sublinear regret bound covering both causal BO and contextual BO.

## Background & Motivation

**Background**: Bayesian Optimisation (BO) is a mainstream framework for optimizing expensive black-box functions, where the target function is often expressed as the "conditional expectation of the target variable after assigning values to certain intervenable variables." When these variables exhibit causal relationships, **Causal Bayesian Optimisation (CaBO)** utilizes known causal graphs to intervene only on carefully selected subsets of variables, thereby achieving lower regret. In parallel, **Contextual Bayesian Optimisation (CoBO)** exploits observable contextual variables by defining intervention values as functions of the context.

**Limitations of Prior Work**: Both approaches have critical flaws. CaBO treats all non-intervention variables as unobservable, **completely discarding contextual information**. Conversely, CoBO is constrained by a **fixed policy scope** (a hardcoded set of intervenable and contextual variables). If this scope is poorly chosen, it can result in **provably linear regret** under certain structural causal models (SCMs), meaning it fails to converge to the optimum. Furthermore, the number of possible policy scopes grows **exponentially** with the number of variables, making exhaustive search infeasible.

**Key Challenge**: Causal information helps determine "which variables to intervene on," while contextual information determines "to what value to intervene given observations." Existing methods use only one type of information, and the lack of either can lead to linear regret. This paper proves with a specific SCM (Fig. 1) that ignoring context results in regret of at least $T/12$, ignoring causal structure results in regret of at least $T/3$, while a strategy utilizing both can achieve an expected reward of $1/3$.

**Goal**: To design a framework that (1) jointly optimizes across multiple candidate scopes to escape the linear regret of fixed scopes; (2) compresses exponential scope search to a manageable scale; (3) incorporates causal graphs and contextual variables into decision-making with convergence guarantees.

**Core Idea**: Combine "scope selection" and "intervention value selection" into a **two-layer search**. The outer layer uses a multi-armed bandit to select scopes from the set of "possibly-optimal mixed policy scopes" (POMPS), while the inner layer uses Gaussian Process BO to pick context-dependent intervention values within the selected scope. These layers are synchronized using a new scope selection criterion based on UCB moving averages.

## Method

### Overall Architecture

CoCa-BO addresses a sequential decision problem: given a causal graph $G$ of a structural causal model (SCM) $M=(\mathbf V,\mathbf U,\mathbf F,P_{\mathbf U})$ with known topology but unknown mechanisms, a target variable $Y\in\mathbf V$, and a set of intervenable variables $\mathbf X^*$. At each step $t$, the agent must: (a) select a **mixed policy scope** $S_t$ and a policy $\pi_t$ defined over it; (b) the environment generates samples according to the intervened distribution $P_{\mathbf V}^{\pi_t}$; (c) the agent observes context $c_t$ and reward $y_t$. The goal is to minimize cumulative regret $R_T=T\sup_{\pi}\mu(\pi)-\sum_{t=1}^T\mu(\pi_t)$, where $\mu(\pi)=\mathbb E_{P_{\mathbf V}^\pi}[Y]$.

The algorithm (Alg. 1) runs an outer loop embedding three components that jointly optimize "scope + intervention values":

1. **POMPS Enumeration**: One-time calculation of the set $S^*[G]$ of all "possibly-optimal mixed policy scopes" compatible with $G$, fixing the context set to $\mathbf C^*=\mathbf V\setminus(\mathbf X^*\cup\{Y\})$ to ensure the optimal scope is not missed.
2. **Multi-Armed Bandit Scope Selection**: Treats each POMPS in $S^*$ as an "arm," using an MAB strategy $A:S^* \to \mathbb R$ to pick the most promising scope $S_t$ at each step.
3. **Per-Scope Gaussian Process BO**: Maintains an independent contextual BO optimizer for each POMPS (using GP surrogate + HEBO acquisition). After selecting $S_t$ and observing context $c_t$, it provides the intervention values $x_{S_t}$ within that scope.

After observing $(c_t,x_t,y_t)$ at each step, the BO optimizer of the selected scope and the MAB arm value are updated; others remain unchanged. This structure is abstracted as a more general **multi-function Bayesian optimisation (multi-function BO)** setting for theoretical analysis: $m$ arms correspond to $m$ POMPS, each with its own function $f_i$, context distribution $P_i$, and action space, with reward $Y=f_{i_t}(X_t,C_t)+\epsilon_t$.

### Key Designs

**1. Mixed Policy Scopes (MPS) and Possibly-Optimal Mixed Policy Scopes (POMPS): Making "which variables to intervene on" optimizable**

This is the foundation for unifying CaBO and CoBO. A **mixed policy scope** $S$ is a set of pairs $(X;\mathbf C_X)$, where $X$ is an intervenable variable and $\mathbf C_X$ is the set of contextual variables used to condition it. The constraint is that the graph $G_S$, obtained by removing incoming edges to $X$ in $G$ and adding edges from $\mathbf C_X$ to $X$, remains acyclic (Def. 1). A **mixed policy** $\pi=\{\pi_{X\mid \mathbf C_X}\}$ on $S$ maps each context value to an intervention value (Def. 2). Thus, CoBO (fixed scope) and CaBO (intervention subset only, no context) become special cases.

Directly searching over all MPS is too costly. The authors use the **POMPS (possibly-optimal MPS)** concept from Lee & Bareinboim (2020) to narrow the candidates: $S$ is a POMPS if and only if there exists an SCM compatible with $G$ such that $S$ is strictly superior to all other scopes (Def. 3). The algorithm only selects from $S^*[G]$. Crucially, this set cannot be further pruned: for any $S\in S^*[G]$, there exists a compatible SCM where $S$ is optimal. When only $G$ is known without the true mechanism, discarding any POMPS risks losing the optimal strategy and incurring linear regret.

**2. Nested Three-Component Solver: POMPS Enumeration × MAB Arm Selection × Per-Scope GP BO**

POMPS enumeration is a brute-force search over all MPS, with complexity growing exponentially with $|\mathbf V|$. However, this step is highly parallelizable and is handled as such in the implementation. Once $S^*$ is obtained, the outer layer treats each POMPS as a bandit arm to find the scope maximizing average reward $\mu_S^*=\sup_{\pi\in\Pi_S}\mu(\pi)$. The inner layer maintains an independent contextual GP for each scope $S$. Specifically, let $[t]_S=\{i\le t:S_i=S\}$ be the history of steps where $S$ was selected. Assuming $y_t=g(x_t,C_t)+\sigma_n\xi_t$, the closed-form posterior at test point $(x,c)$ is:

$$\mu_{\text{post}}(x,c)=k_{[t]_S}^\top(x,c)\big(K_{[t]_S}+\sigma_n^2 I\big)^{-1}y_{[t]_S},$$
$$\sigma_{\text{post}}^2(x,c)=k\big((x,c),(x,c)\big)-k_{[t]_S}^\top(x,c)\big(K_{[t]_S}+\sigma_n^2 I\big)^{-1}k_{[t]_S}(x,c).$$

Inner optimization uses **HEBO** for its ability to handle contextual and mixed variables. Posterior updates require inverting a $T\times T$ kernel matrix, costing $O(T^3)$. This nested "outer discrete arm selection, inner continuous value selection" architecture decouples the causal (which variables) and contextual (conditioning on what) aspects.

**3. Replacing Causal Acquisition Functions with UCB Moving Averages: Making "cross-context comparison" valid**

This is the most critical correction relative to CaBO. When CaBO selects the next intervention subset, it directly compares the **causal acquisition function (cAF)** values of different GPs. The intuition is that an "optimal subset + optimal intervention value" should yield the highest reward, reflected by the acquisition function in low-uncertainty regions. However, in a contextual setting, the optimal value of the acquisition function **inherently depends on the currently observed context**. Thus, CaBO effectively compares acquisition functions evaluated under different contexts, which is **invalid** and causes the optimization to switch erratically between scopes, leading to instability or divergence.

CoCa-BO adopts a new selection criterion $\bar U_{it}$: it is the **moving average of UCB values** of each POMPS instance over its historical evaluation points, plus an exploration term $\rho_i(n_i)/\sqrt{n_i}$ ($n_i$ is the number of times the scope has been selected). Lemma 2 proves this quantity bounds the empirical mean of the target under the optimal strategy (within a controllable additive error), and these additive terms decay fast enough so that sub-optimal POMPS are selected only a few times (Lemma 4).

**4. Regret Bounds for Multi-function BO: Worst-case and Instance-dependent**

The paper provides convergence guarantees under the more general multi-function BO setting. The regret $R_T=\sum_{t=1}^T\big(\mu^*-\mathbb E[f_{i_t}(X_t,C_t)\mid f,D_{t-1}]\big)$ compares the algorithm against an oracle that always selects the arm with the highest context-averaged reward and takes the optimal action. This analysis is more difficult than MAB (where arm rewards are constant) or standard BO (where the context-averaged maximum of $f_{i_t}$ must be compared across functions). Introducing the **maximum information gain** $\gamma_i(n)$ for the $i$-th GP, Theorem 1 provides two high-probability bounds under standard assumptions (Asm. 1–2):

$$R_T\le A_2\sqrt{mT}\big(\sqrt{\beta_T\bar\gamma_T}+\rho_T\big)+\|\Delta\|_1,$$

$$R_T\le A_3\Big\{\sqrt{|I^*|T}\big(\sqrt{\beta_T\gamma_T}+\rho_T\big)+(\beta_T\gamma_T+\rho_T^2)\textstyle\sum_{i\notin I^*}\tfrac{1}{\Delta_i}+m(\sqrt{\beta_1}+\rho_T)\Big\}+\|\Delta\|_1.$$

The first is a **worst-case bound** independent of function instances: for kernels with exponential feature decay (e.g., SE kernel) $\gamma_T$ is poly-logarithmic, so regret is approximately $\sqrt{mT}\max_i\bar d_i$. The second is **instance-dependent**: when the optimal arm set $I^*$ is a singleton, the dominant term reduces to $\sqrt T\max_i\bar d_i$, an improvement of $\sqrt m$ over the worst-case. This is the **first** regret bound covering not only this framework but also the established CaBO setting.

## Key Experimental Results

Experiments used **110 random seeds and 700 iterations**. Results report time-normalized cumulative regret $\bar R_T=\tfrac1T\sum_t\hat r_t$ and the cumulative selection frequency of each scope. Baselines include CaBO and CoBO in two configurations: Config I (baselines can reach the optimum) and Config II (baselines fail to reach the optimum).

### Main Results

| Comparison / Config | Can baseline converge? | CoCa-BO Performance | Key Observations |
|------|------|------|------|
| CaBO vs CoCa-BO, Config I (PSA graph) | Recovers optimum only with specific alignment | Significantly lower regret | CaBO selects POMIS via cAF; selection is highly jittery, hindering convergence |
| CaBO vs CoCa-BO, Config II | Does not converge (cannot adjust by Age/BMI) | Converges, sublinear regret | CaBO marginalizes context and fails to make context-dependent interventions |
| CoBO vs CoCa-BO, Config I (Fig. 1) | Converges slightly faster than CoCa-BO | Converges | CoBO is faster with few variables; CoCa-BO excels in high dimensions |
| CoBO vs CoCa-BO, Config II (Fig. 1 original SCM) | Linear regret (no convergence) | Converges to $\langle X_1;C\rangle$, sublinear | Validates theory: fixed scopes cause linear regret |

In the PSA example with 6 variables, although only Aspirin and Statin are intervenable, targetting PSA requires conditioning on Age and BMI. CoCa-BO correctly identifies the optimal POMPS.

### Ablation Study

| Configuration | Observation | Explanation |
|------|------|------|
| High-dim environments | Optimization space reduced by **~22x** | Causal graph independencies prune irrelevant dimensions |
| Noise $\sigma(\epsilon_Y)$ from 0.01 to 1.25 | Still converges | Robust to target variable observation noise |
| Noise $\sigma(\epsilon_Y)=6.25$ | Convergence trend becomes fuzzy | Extreme noise interferes with scope selection |

### Key Findings
- **Stability of scope selection is the deciding factor**: CaBO's use of cAF leads to volatile selection frequencies in contextual environments, ruining inner optimizer convergence. CoCa-BO stabilizes selection via the UCB moving average.
- **Advantages scale with variable count**: While CoBO may be faster for few variables, CoCa-BO leverages causal independence to compress the search space (approx. 22x reduction), leading to superior sample complexity in high dimensions.
- **Theoretical-Empirical alignment**: The linear regret of CoBO in Config II confirms that fixed scopes result in failure to converge under certain causal structures.

## Highlights & Insights
- **Promoting scope selection to a first-class citizen**: Unlike previous methods that treat subset selection and policy optimization separately, CoCa-BO embeds them into a single search space solved via a nested "outer MAB + inner BO" structure.
- **Diagnosing and fixing a specific "bug"**: The paper identifies the invalidity of "cross-context acquisition function comparison" in prior work and provides a provable fix using the moving average of UCB values.
- **Filling theoretical gaps**: Provides the first regret bounds that apply both to the contextual-causal BO setting and the classical CaBO setting, distinguishing between worst-case and instance-dependent scenarios.
- **Focus on reproducibility**: Implemented in Pyro with a benchmark environment, explicitly addressing the lack of reproducibility in prior works like Functional Causal BO.

## Limitations & Future Work
- **Exponential POMPS Enumeration**: Complexity grows exponentially with $|\mathbf V|$; while parallelizable, it remains a bottleneck for very large graphs.
- **Reliance on Prior Causal Knowledge**: Assumes the causal graph $G$ is known and correct. Causal discovery is outside the scope.
- **$O(T^3)$ GP Posterior complexity**: High cost for large $T$; lacks discussion on sparse GP or other scalable approximations.
- **Extreme Noise Sensitivity**: Convergence degrades at very high noise levels ($\sigma(\epsilon_Y)=6.25$).
- **Information Sharing between Scopes**: How to share information across different POMPS (e.g., via multitask GPs) without strengthening assumptions remains an open problem.

## Related Work & Insights
- **vs Causal BO**: CaBO marginalizes context; CoCa-BO explicitly utilizes it and fixes the instability of acquisition-based selection.
- **vs Contextual BO**: CoBO uses fixed scopes; CoCa-BO searches over POMPS to avoid suboptimality.
- **vs Functional Causal BO**: Avoids finite RKHS expansions and provides a specified, reproducible implementation with theoretical guarantees.
- **vs Model-Based Causal BO**: Provides better scaling by utilizing POMPS to reduce the search space compared to assuming additive noise SCMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies causal scope selection and contextual BO with the first comprehensive regret bound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid validation across diverse configurations and noise levels, though environments are relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent motivation using counter-examples; builds logic rigorously from definitions to theorems.
- Value: ⭐⭐⭐⭐ Strong theoretical and practical framework for causal-contextual decision making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

- **Causal Bayesian Optimization** (Aglietti et al., AISTATS 2020)
- **Characterizing and Learning Strategies for Optimal Policy Space Search** (Lee & Bareinboim, ICML 2020)
- **Contextual Bayesian Optimization** (Krause & Ong, NIPS 2011)

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[ICLR 2026\] Combinatorial Bandit Bayesian Optimization for Tensor Outputs](combinatorial_bandit_bayesian_optimization_for_tensor_outputs.md)
- [\[ICLR 2026\] BoGrape: Bayesian optimization over graphs with shortest-path encoded](bogrape_bayesian_optimization_over_graphs_with_shortest-path_encoded.md)
- [\[ICML 2026\] Efficient Stochastic Optimisation via Sequential Monte Carlo](../../ICML2026/optimization/efficient_stochastic_optimisation_via_sequential_monte_carlo.md)
- [\[ICLR 2026\] Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models](adaptive_acquisition_selection_for_bayesian_optimization_with_large_language_mod.md)

</div>

<!-- RELATED:END -->
