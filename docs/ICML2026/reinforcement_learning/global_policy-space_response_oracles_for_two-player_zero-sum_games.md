---
title: >-
  [Paper Note] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games
description: >-
  [ICML 2026][Reinforcement Learning][PSRO] This paper points out that mainstream PSRO methods, when expanding the strategy population, rely only on local information from the "restricted game…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "PSRO"
  - "Nash Equilibrium"
  - "Population Exploitability"
  - "Zero-sum Games"
  - "DRL"
date: 2026-05-08
content_hash: 7f3b49ccd886dc90
---

# Global Policy-Space Response Oracles for Two-Player Zero-Sum Games

**Conference**: ICML 2026  
**arXiv**: [2605.28273](https://arxiv.org/abs/2605.28273)  
**Code**: https://github.com/Zhangjy1997/GlobalPSRO (Available)  
**Area**: Reinforcement Learning / Game Theory RL  
**Keywords**: PSRO, Nash Equilibrium, Population Exploitability, Zero-sum Games, DRL

## TL;DR
This paper points out that mainstream PSRO methods, when expanding the strategy population, rely only on local information from the "restricted game," which can lead to a worst-case requirement of adding nearly $N$ pure strategies to achieve convergence. It proposes Global PSRO, a two-stage exploration-selection framework that first samples multiple candidate best responses and then uses *post-expansion Population Exploitability (PE)* to directly score and select the best expansion. The costs of multi-candidate training and evaluation are reduced to an acceptable range using a conditional policy network with shared parameters.

## Background & Motivation
**Background**: Policy-Space Response Oracle (PSRO) is a standard approach for finding Nash equilibria in large-scale zero-sum games. It maintains a restricted strategy population $\Pi^r$, uses a meta-strategy solver (MSS) to solve for a mixed strategy $\sigma$ in this restricted game, and then employs DRL to learn a best response (BR) $\pi$ against $\sigma$, which is iteratively added to $\Pi^r$. Representative MSSs include restricted-game Nash, PRD, AlphaRank, and Uniform.

**Limitations of Prior Work**: MSSs decide which BR to train next based only on the payoff matrix within $\Pi^r$. This results in strategies that are "locally strong but globally useless"—adding a strategy that is strong against the current restricted game often contributes almost nothing to the global exploitability reduction. While Anytime PSRO / EPSRO mitigate this by expanding exploitability evaluation to the full game to calculate SRGB-MSS, they still follow the "train one BR for one meta-strategy" paradigm and do not truly evaluate "how much the population as a whole improves after adding this strategy."

**Key Challenge**: The ultimate goal of PSRO is to make the *population*, rather than a single strategy, approach a Nash equilibrium. However, the selection criteria of all MSSs are based on the "BR to a single meta-strategy," creating a misalignment with the goal. The authors prove a strong negative theorem: for any RGB-MSS $\mathcal{M}$ and any $N$, there exists an $N \times N$ zero-sum game such that $\mathcal{M}$ either fails to converge or requires at least $N-1$ iterations, whereas a specific oracle MSS requires only $\min\{S+2, N-1\}$ iterations (where $S$ is the support size of the equilibrium).

**Goal**: To shift the expansion decision from "training a BR for the current meta-strategy" to "directly minimizing the PE of the population after adding a candidate," while ensuring the cost of multi-candidate training and evaluation remains acceptable in large-scale DRL scenarios.

**Key Insight**: The Population Exploitability $\mathcal{PE}(\Pi^r;\mathcal{G})=\min_{\sigma\in\Delta(\Pi^r)}\epsilon(\sigma)$ is the true measure of "population quality." Since the best response possesses local stability over the simplex (Proposition 4.1: if the BR of $\sigma$ is unique, it remains unchanged in a certain neighborhood), a finite candidate pool is sufficient to cover large regions of the BR space.

**Core Idea**: In each round, $K$ meta-strategies are sampled to train $K$ candidate BRs. Then, RM-BR is used to estimate the PE for each *hypothetically expanded population*, and the one with the lowest PE is added to $\Pi^r$. A single conditional policy network $\pi_\theta(a\mid s,\sigma)$ is used to amortize the training and evaluation costs of $K$ candidates into one set of parameters.

## Method

### Overall Architecture
Global PSRO splits each PSRO iteration into two phases. The inputs are the current restricted strategy set $\mathbf{\Pi}^r$ and the restricted game payoff matrix $\widehat{\mathbf{U}}$, and the output is the expanded $\mathbf{\Pi}^r \cup \{\pi_{k^\star}, \beta_{k^\star}\}$:

1. **Phase I (Exploration)**: Construct a meta-strategy pool $\mathcal{S}_t=\{\sigma^{\mathcal{M}}\}\cup\{\sigma_k\sim\text{Dirichlet}(\mathbf{1})\}_{k=2}^{K}$ (one from the base RGB-MSS $\mathcal{M}$, others sampled uniformly from the simplex). Use a conditional policy $\pi_\theta(a\mid s,\sigma)$ to simultaneously train $K$ candidate BRs via $\max_\theta\frac{1}{K}\sum_k U(\pi_\theta(\cdot\mid\sigma_k),\sigma_k)$. The candidate populations are $\mathbf{\Pi}_k^+=\mathbf{\Pi}^r\cup\{\pi_k\}$.
2. **Phase II (Selection)**: For each $\mathbf{\Pi}_k^+$, estimate $\mathcal{PE}(\mathbf{\Pi}_k^+;\mathcal{G})=\min_\sigma\max_\pi U(\pi,\sigma)$. Select the $k^\star$ with the lowest PE and add both the candidate $\pi_{k^\star}$ and the evaluation byproduct $\beta_{k^\star}$ (the full-game BR against $\rho_{k^\star}$, equivalent to an Anytime-PSRO expansion) to the population.

> Two strategies are added per round, which counts as two iterations in PSRO terms. Theoretical guarantees (Theorems 4.2-4.3): As long as $\sigma^{\mathcal{M}}$ is included and ties are broken conservatively, the finite-step convergence of the base MSS is inherited. On the adversarial game families mentioned in Sec 3.1, the expected number of iterations is reduced to $\mathbb{E}[T^\star]\le \min\{2+\tfrac{2S}{1-(1-c)^{K-1}},N-1\}$. As $K\to\infty$, this tightens to $\min\{2+2S,N-1\}$, an order of magnitude faster than the "must add $N-1$ pure strategies" case.

### Key Designs

1. **Global Selection Criteria based on Post-expansion PE**:
    - **Function**: Changes the objective function of the expansion decision from "BR utility against a meta-strategy" to "population PE assuming the candidate is added," i.e., $\pi^\star\in\arg\min_{\pi\in\mathcal{B}(\Delta(\mathbf{\Pi}^r))}\mathcal{PE}(\mathbf{\Pi}^r\cup\{\pi\};\mathcal{G})$.
    - **Mechanism**: During evaluation, the candidate population is treated as a new restricted game. The goal is to solve $\min_{\sigma\in\Delta(\mathbf{\Pi}_k^+)}\max_\pi U(\pi,\sigma)$. In large games where exact solutions are infeasible, RM-BR (McAleer 2022) is used for approximation—inner-loop regret matching maintains a mixture $\rho_k$, while an outer-loop best response learner tracks $\rho_k$, yielding $\widehat{\mathcal{PE}}_k=U(\beta_k,\rho_k)$.
    - **Design Motivation**: Because RGB-MSS only considers local payoffs, it may require adding almost the entire strategy space to converge in the worst case. Post-expansion PE directly aligns with the original goal of PSRO (making the population approach Nash), transforming the problem of "which strategy to pick" into direct optimization of a global metric.

2. **Multi-candidate Training and Evaluation via Shared Parameter Conditional NN**:
    - **Function**: Amortizes the training of $K$ candidate BRs and $K$ evaluation BRs into a single network, avoiding the need to train $K$ independent models per iteration.
    - **Mechanism**: A single conditional explorer $\pi_\theta(a\mid s,\sigma)$ outputs $K$ responses simultaneously, using the objective $J(\theta)=\tfrac{1}{K}\sum_k U(\pi_\theta(\cdot\mid\sigma_k),\sigma_k)$. Similarly, a conditional evaluator $\beta_\psi(a\mid s,\sigma)$ is used in the evaluation phase, where the BR learner for candidate $k$ is the slice $\beta_k \triangleq \beta_\psi(\cdot\mid\sigma_k)$. The entire RM-BR evaluation process shares $\psi$.
    - **Design Motivation**: A naive implementation would require training $K$ independent strategies and $K$ independent evaluators, multiplying the DRL computation per iteration by $K$. Conditional parameter sharing makes the relative overhead of multiple candidates negligible, making the exploration-selection framework practical in DRL settings.

3. **Regularized PE Evaluation Metric**:
    - **Function**: Prevents "false optimality" where a new strategy $\pi_k$ has nearly zero weight in $\rho_k$, yet $\widehat{\mathcal{PE}}_k$ appears small.
    - **Mechanism**: Define $\hat p_k \triangleq \rho_k(\pi_k)$ and score using $\widehat{\mathcal{PE}}_k^{\mathrm{reg}}=(1-\hat p_k)(\widehat{\mathcal{PE}}(\mathbf{\Pi}^r;\mathcal{G})-U(\beta_k,\rho^r))+\widehat{\mathcal{PE}}_k$. Since $U(\beta_k,\rho^r) \le \widehat{\mathcal{PE}}(\mathbf{\Pi}^r;\mathcal{G})$, the term in brackets is non-negative, meaning a smaller $\hat p_k$ results in a larger "penalty."
    - **Design Motivation**: In large games, both $\rho_k$ and $\beta_k$ are approximations. If RM converges slowly, the new strategy might not be utilized, and looking only at $\widehat{\mathcal{PE}}_k$ could be biased by estimation noise. This regularization term explicitly encodes "whether the newly added strategy is actually utilized" into the score, making selection more robust to estimation errors.

### Loss & Training
- The Explorer uses PPO to optimize $J(\theta)=\tfrac{1}{K}\sum_k U(\pi_\theta(\cdot\mid\sigma_k),\sigma_k)$. In the evaluation phase, it alternates between "sampling trajectories with $\beta_\psi$ against $\rho_k \to$ updating $\rho_k$ with RM $\to$ updating $\psi$ with DRL" for several steps.
- Each round adds two strategies $\{\pi_\theta(\cdot\mid\sigma_{k^\star}), \beta_\psi(\cdot\mid\sigma_{k^\star})\}$. Under parallel implementation, it is compared against base PSRO with the same environment-step budget.

## Key Experimental Results

### Main Results
Environment: Kuhn Poker / Liar's Dice / Leduc Poker / Goofspiel (5, 13 cards) two-player zero-sum extensive-form games using OpenSpiel. PPO is used for all BRs, and all methods are aligned by environment steps budget. Metric: Population Exploitability (lower is better).

| Dataset / Steps | Ours (Global PSRO) | PSRO w/ Nash | PSRO w/ AlphaRank | Anytime PSRO | NeuPL w/ AlphaRank | PSD-PSRO w/ AlphaRank |
|---|---|---|---|---|---|---|
| Goofspiel-13 @ $19.2\times 10^6$ | **0.305** | 0.579 | 0.459 | 0.404 | 0.244 | 0.599 |
| Goofspiel-13 @ $76.8\times 10^6$ | **0.056** | 0.284 | 0.188 | 0.223 | 0.169 | 0.251 |
| Goofspiel-13 @ $153.6\times 10^6$ | **0.046** | 0.193 | 0.178 | 0.191 | 0.132 | 0.160 |

In Goofspiel-13, the largest game, Global PSRO generally reduces PE to 1/3 or 1/4 of all baselines in mid-to-late stages. NeuPL w/ AlphaRank is the strongest competitor besides the proposed method but still lags by more than 2x. In tiny games like Kuhn Poker, restricted-game Nash is already close to the full game, and Global PSRO performs similarly or slightly worse—consistent with the intuition that "small games do not require global info."

### Ablation Study
RQ4 conducts five ablations on Kuhn / Liar's / Leduc / Goofspiel-5:

| Configuration | Key Phenomenon | Explanation |
|---|---|---|
| Full Global PSRO | Best overall | Complete method |
| Exploitation only | Significant degradation | Candidate pool contains only $\sigma^{\mathcal{M}}$; equivalent to single-MSS PSRO; confirms gain from multi-candidate selection |
| Random selection | Significant degradation | Using the same candidate pool but picking randomly; shows PE scoring is critical |
| w/o PE regularization | Moderate degradation | Using bare $\widehat{\mathcal{PE}}_k$ scores; estimation noise biases the selection |
| Exact PE evaluation | Slight improvement | Provides upper bound for selection; implies practical RM-BR estimation is sufficiently informative |
| Neighbor-search exploration | Degradation | Replacing Dirichlet uniform sampling with local perturbations around $\sigma^{\mathcal{M}}$; insufficient diversity |

Additionally, RQ2 incorporates the diversity regularization from PSD-PSRO into the Global PSRO explorer (conditional input $(\sigma, \lambda)$, $\lambda \sim \text{Uniform}[0, 0.1]$). This shows clear further gains in Leduc Poker, though gains in other games are inconsistent, suggesting diversity-driven and the proposed framework are orthogonal and stackable.

### Key Findings
- The gain primarily comes from "changing the scoring criterion" rather than "changing the parameterization": Global PSRO consistently achieves lower PE than NeuPL, which also uses a conditional population, proving that post-expansion PE selection is the core contribution.
- Alignment of theory and practice: The negative theorem in Sec 3 predicts that "RGB-MSS requires $N-1$ steps on certain adversarial games." Fig.1 confirms this progressive stall on a specially constructed $100 \times 100$ game.
- Final performance is not sensitive to evaluation accuracy: Replacing estimated PE with exact PE provides only a minor boost, indicating that RM-BR + regularization already correctly identifies the relative ranking for selection.

## Highlights & Insights
- **The Minimalist Idea of "Scoring by the Goal Itself"**: PSRO research has focused heavily on MSS (how to pick the meta-strategy), but picking an MSS is always a proxy goal. This paper directly replaces the proxy with the true goal, PE, effectively moving the search from "proxy space" to "target space." This design of "aligning selection criteria with the final objective" is transferable to other multi-stage training pipelines.
- **Conditional Policy = Free Lunch for Multi-candidates**: Amortizing the training and evaluation of $K$ candidates into single networks $\pi_\theta(a\mid s,\sigma)$ / $\beta_\psi(a\mid s,\sigma)$ turns "sampling then selection" from a luxury to an affordable process under DRL budgets. This trick is suitable for any "generate-then-select" pipeline (diverse RL, ensembling, population-based training, etc.).
- **Regularized Scoring Against Estimation Noise**: Using the "new strategy weight $\hat p_k$" as a credibility correction is a small but valuable design—many selection-based methods (neural architecture selection, active learning) face similar issues where noisy estimators lead to incorrect choices.

## Limitations & Future Work
- Limited to two-player zero-sum: PE does not have a saddle-point structure in multi-player general-sum games, making $\min$-$\max$ estimation non-trivial; authors note this as future work.
- $\widehat{\mathcal{PE}}_k$ remains a biased estimator in real DRL: When the budget per round is tight, RM-BR's $\rho_k$ might be far from the true least-exploitable mixture. While regularization helps, no error analysis is provided; switching to a Confidence-Bound style selection (UCB-PSRO) might further save budget.
- No adaptive mechanism for candidate count $K$: Currently a fixed hyperparameter. Theorem 4.3 suggests larger $K$ is better, but training objectives for the explorer might dilute gradient signals. A dynamic $K$ based on current PE estimation variance would be more elegant.

## Related Work & Insights
- **vs Anytime PSRO / EPSRO (SRGB-MSS)**: They also use full-game information by finding the meta-strategy hardest to exploit by the full game, but still "train one BR for one $\sigma$." This paper elevates the expansion decision to the population level and adds $\beta_k$ as an additional byproduct.
- **vs PSD-PSRO (diversity-driven)**: They force new strategies to be different from existing ones via diversity regularization. This paper evaluates candidates based on whether they "truly reduce PE." The two are orthogonal and combined successfully in Leduc Poker.
- **vs NeuPL / Simplex-NeuPL (conditional population)**: They also use conditional NNs to share PSRO population parameters but retain traditional selection mechanisms. This paper adopts the same parameter sharing technology but contributes the core innovation in the selection criterion—NeuPL is the strongest baseline, and Global PSRO’s consistently lower PE proves the improvement comes from selection, not just parameterization.

## Rating
- Novelty: ⭐⭐⭐⭐ "Aligning selection criteria with final goal" + negative theorem is a concise and powerful perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five games + four baseline categories + complete RQ1-4 ablations + theory-check experiments (Fig 1) + diversity stacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Sec 3 clearly explains the limitations of RGB-MSS through the lens of tools/goals misalignment.
- Value: ⭐⭐⭐⭐ Provides a "new standard" for selection criteria in the PSRO family, directly impacting open-ended multi-agent training and self-play infrastructure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games](revisiting_regularized_policy_optimization_for_stable_and_efficient_reinforcemen.md)
- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](../../AAAI2026/reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)
- [\[ICML 2026\] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICML 2026\] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](convergence_of_two-timescale_markovian_stochastic_approximations_with_applicatio.md)

</div>

<!-- RELATED:END -->
