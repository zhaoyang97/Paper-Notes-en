---
title: >-
  [Paper Note] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification
description: >-
  [ICML 2026][Interpretability][conjoint analysis] This paper reformulates traditional conjoint analysis from "estimating AMCE marginal effects" to "learning interpretable product-form Categorical stochastic policies over…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "conjoint analysis"
  - "factored stochastic policy"
  - "minimax"
  - "Delta method"
  - "AMCE"
date: 2026-05-08
content_hash: d4de05a37fab334e
---

# MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification

**Conference**: ICML 2026  
**arXiv**: [2504.19043](https://arxiv.org/abs/2504.19043)  
**Code**: TBD  
**Area**: Interpretability / Offline Policy Learning / Conjoint Analysis / Minimax Games  
**Keywords**: conjoint analysis, factored stochastic policy, minimax, Delta method, AMCE

## TL;DR
This paper reformulates traditional conjoint analysis from "estimating AMCE marginal effects" to "learning interpretable product-form Categorical stochastic policies over an exponential factored action space." It provides a closed-form solution with an $L_2$ trust region under a second-order interaction model, a differentiable general solution, and a two-player minimax extension incorporating primary election systems. By propagating uncertainty through the Delta method to policy probabilities and values, it successfully brings the adversarial equilibrium "vote share" back into historical ranges in the 2016 US presidential conjoint experiment for the first time.

## Background & Motivation

**Background**: Conjoint analysis is a mainstay in social sciences for studying multi-attribute preferences. Respondents are typically presented with two multi-attribute profiles (e.g., candidate characteristics) and forced to choose one. Analysis usually summarizes marginal effects as AMCE (Average Marginal Component Effect): fixing one attribute level and averaging over others according to some distribution. AMCE is the de facto standard in journals like *Political Analysis*.

**Limitations of Prior Work**: AMCE assumes other attributes are drawn independently from a distribution (usually uniform). However, real-world candidate pools are neither uniform nor selected in a "strategic vacuum"—Democratic and Republican profiles emerge through mutual strategic competition. Consequently, "optimal profile combinations" suggested by AMCE often diverge from historical results. Moreover, AMCE identifies single-attribute effects but fails to answer the decision problem: "What kind of candidate should be fielded?"

**Key Challenge**: The decision object is a joint distribution over $D$ attributes, where the action space size $|\mathcal{T}|=\prod_d L_d$ explodes exponentially. Since the sample size $n$ is much smaller than $|\mathcal{T}|$, learning a policy for every profile is neither feasible nor interpretable. Researchers often sacrifice expressiveness (only looking at margins), interpretability (black-box models), or strategic realism (ignoring opponents).

**Goal**: (1) Reformulate the estimation problem as offline policy optimization; (2) Identify a policy class that spans exponential action spaces while remaining interpretable to political scientists; (3) Model the "opponent" as a strategic agent undergoing simultaneous optimization; (4) Provide confidence intervals for journal-quality reporting.

**Key Insight**: Conjoint random assignment naturally provides a logging policy, enabling the use of the offline contextual bandit framework. The authors observe that "product-of-Categoricals" distributions serve as a natural restricted family under a mean-field variational approximation of the Gibbs optimal policy. It allows "attribute weights" to be read directly, building interpretability into the inductive bias of the policy class.

**Core Idea**: Replace AMCE with a family of "product-form Categorical stochastic policies." Derive closed-form optimal solutions under linear probability approximations and propagate regression uncertainty to policies and values via the Delta method. Extend this to a restricted minimax objective incorporating primary systems, solved via adversarial ascent–descent to find restricted equilibria.

## Method

### Overall Architecture
Input consists of conjoint data $(C_i, \mathbf{T}_i^a, \mathbf{T}_i^b)_{i=1}^n$ with forced-choice labels ($\mathbf{T}^c \in \mathcal{T}=\{1,\dots,L\}^D$ is a $D$-dimensional profile; $C_i\in\{0,1\}$ indicates if $a$ was chosen). The pipeline is: (1) Fit an outcome model $\sigma(\eta_i)$ with interaction terms, expressing logits as differences in main effects $\beta_{dl}$ and second-order interactions $\gamma_{dl,d'l'}$; (2) Optimize the policy $\Pr_{\bm{\pi}^c}(\mathbf{T}^c=\mathbf{t})=\prod_d \pi^c_{d,t_d}$ subject to an $L_2$ trust region constraint $\|\bm{\pi}^c-\mathbf{p}\|_2^2 \le \epsilon_n$; (3) Use closed-form solutions for average cases and logit reparameterization with synchronous ascent–descent for adversarial cases; (4) Propagate the outcome model's variance–covariance matrix $\hat{\Sigma}$ to policy and value standard errors via the Jacobian $\mathbf{J}=\nabla_{\hat\beta,\hat\gamma}\{\hat Q,\hat{\bm\pi}^*\}$ and the Delta method.

### Key Designs

1.  **Product-of-Categoricals Policy Class + $L_2$ Trust Region**:
    *   **Function**: Defines a family of "interpretable and estimable" stochastic interventions over exponential action spaces, relaxing the fragile "optimal single profile" goal to an "optimal distribution."
    *   **Mechanism**: Restricts policies to $\Pr_{\bm\pi}(\mathbf{t})=\prod_d \pi_{d,t_d}$ (independent Categoricals across attributes). Target: $\max_{\bm\pi} Q(\bm\pi)-\lambda_n\|\bm\pi-\mathbf{p}\|_2^2$, where $Q(\bm\pi)=\sum_{\mathbf{t}}\mathbb{E}[Y_i(\mathbf{t})]\Pr_{\bm\pi}(\mathbf{t})$ and $\mathbf{p}$ is the logging policy. The authors prove that while the full simplex optimal solution is Gibbs-form $\sigma^\star(\mathbf{t})\propto p(\mathbf{t})\exp\{u(\mathbf{t})/\lambda\}$, the product-form constraint is equivalent to a classical mean-field variational approximation (Wainwright & Jordan, 2008).
    *   **Design Motivation**: (i) Factored forms make policies "attribute-readable"—e.g., "the model assigns 0.7 probability to outsiders"—meeting political science standards for interpretability. (ii) $L_2/KL$ trust regions control off-policy variance. (iii) Stochastic policies aggregate a family of high-performing profiles rather than picking a single optimal point that is statistically unstable in high dimensions.

2.  **Closed-form Average-case Optimal Solution + Delta Method UQ**:
    *   **Function**: Provides analytical expressions for optimal $\bm{\pi}^{a*}$ under linear probability approximations of second-order interactions, automatically propagating regression uncertainty.
    *   **Mechanism**: Setting the derivative of the objective w.r.t. $\pi_{dl}$ to zero yields a linear system $\mathbf{C}\bm{\pi}^{a*}=\mathbf{B}$, where $B_{r(dl),1}=-\bar\beta_{dl}-4\lambda_n p_{dl}-2\lambda_n\sum_{l'\ne l}p_{dl'}$, $C_{r(dl),r(dl)}=-4\lambda_n$, and $C_{r(dl),r(d'l')}=\bar\gamma_{dl,d'l'}$ (Proposition 3.1). For large $\lambda_n$, this is the unique global optimum. Since $\bm{\pi}^{a*}=\mathbf{C}^{-1}\mathbf{B}$ is a differentiable function of $(\hat\beta,\hat\gamma)$, Var-Cov$(\hat Q, \hat{\bm\pi}^{a*})=\mathbf{J}\hat\Sigma\mathbf{J}'$ via the Delta method. For iterative solvers, the paper supports implicit differentiation $\partial\bm\alpha^*/\partial\theta=-H^{-1}\nabla_\theta F$ at convergence to avoid long-range backpropagation.
    *   **Design Motivation**: Standard errors are mandatory in social science. Analytical solutions provide "analysis-friendliness," allowing reviewers to verify optimality. Implicit differentiation generalizes the UQ framework to GLM/BNN/Minimax models.

3.  **Adversarial Minimax Extension for Primary Systems**:
    *   **Function**: Upgrades the "opponent" from a static distribution to a simultaneous agent and encodes "Primary-then-General" election structures into the objective.
    *   **Mechanism**: Defines a zero-sum payoff $Q(\bm\pi^A,\bm\pi^B)$. Institutional parameters $\beth$ (primary sets $\mathcal{I}^A,\mathcal{I}^B$ and general voters $\mathcal{E}$) are injected via "nomination pushforward" operators $\bar{\bm\pi}^A(\bm\pi^A,\bm\pi^{A'},\beth)$. Algorithm 1 performs synchronous ascent–descent on logit parameters $\bm\alpha^A,\bm\alpha^B$: $\bm\alpha^{A,(s)}\leftarrow\bm\alpha^{A,(s-1)}+\gamma\nabla_{\bm\alpha^A}\Phi$, $\bm\alpha^{B,(s)}\leftarrow\bm\alpha^{B,(s-1)}-\gamma\nabla_{\bm\alpha^B}\Phi$. A **Policy Divergence Factor** $\mathcal{D}_\varepsilon(\mathbf{t})=|\log\frac{\Pr_{\bm\pi^A}(\mathbf{t})+\varepsilon}{\Pr_{\bm\pi^B}(\mathbf{t})+\varepsilon}|$ was introduced to measure strategic distance.
    *   **Design Motivation**: AMCE fails to account for opponents strategically optimizing their candidates. Encoding institutional structures via pushforward operators prevents "optimal profiles" that could never win a primary from being selected.

### Loss & Training
Average case: $O(\bm\pi)=Q(\bm\pi)-\lambda\|\mathbf{p}-\bm\pi\|^2$ solved via closed-form or projected gradient. Adversarial: $\Phi(\pi^A,\pi^B)=Q_{\text{inst}}-\lambda R(\pi^A\|\mathbf{p})+\lambda R(\pi^B\|\mathbf{p})$ using logit reparameterization + synchronous ascent–descent. Jacobians for inference are computed via $S$-step unrolling or implicit differentiation, with standard errors clustered at the respondent level.

## Key Experimental Results

### Main Results
Two types: Synthetic data and the 2016 US Presidential conjoint (Ono & Burden 2019). Synthetic grids spanned $n\in\{500,\dots,10000\}$ and $K\in\{5,10,20\}$.

| Scenario | Samples / Dim | Metric | Ours (Closed-form + Delta) | AMCE Baseline | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Avg Case Sync ($R^2{=}0.7$) | $n{=}3500, K{=}10$ | RMSE($\hat{\bm\pi}^*$) | Rapid decay / Negligible bias | — | Fig 3–4 |
| Avg Case Sync | Above | Expected Win Rate $Q$ | Significantly higher than AMCE argmax | Baseline | Fig 4 |
| Avg Case Sync | Above | 95% CI Coverage | Close to 0.95 | — | §B.4 |
| Adv Case Sync | $n{=}10000$ | RMSE($\hat{\bm\pi}^R$) | Primarily $n$-driven, weak $p_R$ dependence | — | Fig 1 |
| 2016 US Conjoint | Neural Outcome | Avg Case Latent Vote Share | **Outside historical 1976–2020 range** | — | Fig 2 |
| 2016 US Conjoint | Neural Outcome | Adv Minimax Vote Share | **Falls back to historical range** | — | Key selling point |

### Ablation Study
| Configuration | Key Observation |
| :--- | :--- |
| GLM vs. Transformer | GLM is more efficient/calibrated for near-linear data; Transformer has better RMSE but poor CI coverage under mismatch. |
| No-Adv vs. Minimax | Average policies yield unrealistic vote shares; adversarial policies align with historical reality. |
| Closed-form vs. Implicit Diff | Solutions match; implicit differentiation is memory-efficient but can be unstable if $H$ is ill-conditioned. |
| Data-driven Clustering | Clustered versions endogenously recover Democrat-Independent-Republican preference structures. |

### Key Findings
*   **Empirical Realism**: The average-case optimal profile results in vote shares outside historical bounds (unbelievable). The adversarial restricted-equilibrium results align with historical ranges since 1976 and the 2016 actual result—providing a **falsifiable** criterion.
*   **AMCE Failure**: In cases where main effects are positive but interactions are negative (e.g., outsider and moderate as substitutes), AMCE argmax selects (outsider, moderate), while Ours spreads probability to valid strategic alternatives, achieving higher win rates.
*   **Sample Sensitivity**: In adversarial settings, RMSE is dominated by $n$ rather than $p_R$, suggesting the bottleneck is utility estimation, not game complexity.

## Highlights & Insights
*   **Reformulating social science estimation as policy learning**: Shifting from AMCE to factored stochastic policies connects conjoint analysis to the offline contextual bandit/MARL toolbox (Delta method, implicit differentiation, Mirror-Prox).
*   **Theoretical Mean-Field Link**: Grounding the policy family as a variational approximation of the Gibbs solution provides the first theoretical gap bound for restricted policy classes in conjoint.
*   **Institutional Structure as Pushforward**: Modeling primary rules as a nomination pushforward operator $\bar{\bm\pi}^c$ allows for a unified framework across different political systems.
*   **Policy Divergence Factor**: A simple log-ratio diagnosis $\mathcal{D}_\varepsilon$ quantifies how far real candidates deviate from partisan optimality.

## Limitations & Future Work
*   **Two-step Approach**: The outcome model and policy optimization are decoupled; misspecification in the first step contaminates the CI.
*   **Optimality Gap**: Product-form Categoricals are not unconstrained optimal; the gap under complex interactions is not fully quantified.
*   **Institutional Priors**: Parameters $\beth$ (e.g., primary turnout) must be known a priori; errors here bias the minimax solution.
*   **Global Convergence**: Due to non-convexity in factored policy spaces, only steady points (diagnosed by exploitability) are guaranteed, not global optima.

## Related Work & Insights
*   **vs. AMCE/AMIE**: Moves beyond uniform marginal assumptions to strategic optimization and game equilibria.
*   **vs. Offline Policy Learning (Athey & Wager)**: Transitions from deterministic rules for binary treatments to stochastic policies for multi-dimensional factored action spaces.
*   **vs. Minimax RL (Kallus & Zhou)**: Uses minimax for strategic competition rather than worst-case unobserved confounding.
*   **vs. PSRO/Markov Games**: Tailors multi-agent RL to the constraints of offline experimental data common in social sciences.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ Systematically bridges conjoint analysis with minimax policy learning.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid synthetic grids and historical validation; needs more global convergence analysis.
*   **Writing Quality**: ⭐⭐⭐⭐ Rigorous but high barrier to entry for cross-disciplinary readers.
*   **Value**: ⭐⭐⭐⭐⭐ High potential to replace AMCE as the gold standard in political science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](../../ICLR2026/interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)

</div>

<!-- RELATED:END -->
