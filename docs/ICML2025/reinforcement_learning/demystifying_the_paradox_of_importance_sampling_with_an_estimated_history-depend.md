---
title: >-
  [Paper Note] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation
description: >-
  [ICML2025][Reinforcement Learning][off-policy evaluation] This work theoretically demystifies the fundamental reason behind the paradox that "using an estimated history-dependent behavior policy in OPE is paradoxically better than using the true behavior policy"—estimating the behavior policy implicitly projects the IS estimator onto a more constrained space, reducing asymptotic variance at the cost of increasing finite-sample bias.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "off-policy evaluation"
  - "importance sampling"
  - "behavior policy estimation"
  - "history-dependent"
  - "bias-variance trade-off"
date: 2026-05-08
content_hash: 63bcf5d5ca82939d
---

# Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation

**Conference**: ICML2025  
**arXiv**: [2505.22492](https://arxiv.org/abs/2505.22492)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: off-policy evaluation, importance sampling, behavior policy estimation, history-dependent, bias-variance trade-off

## TL;DR
This work theoretically demystifies the fundamental reason behind the paradox that "using an estimated history-dependent behavior policy in OPE is paradoxically better than using the true behavior policy"—estimating the behavior policy implicitly projects the IS estimator onto a more constrained space, reducing asymptotic variance at the cost of increasing finite-sample bias.

## Background & Motivation
- **Off-Policy Evaluation (OPE)** is a core problem in reinforcement learning: evaluating the expected return of a target policy $\pi_e$ using historical data collected by a behavior policy $\pi_b$.
- **Importance Sampling (IS)** is the most fundamental OPE method, which reweights returns using the importance ratio $\pi_e/\pi_b$. It is theoretically unbiased but can suffer from high variance.
- **The Paradox**: Existing empirical studies (Hanna et al., 2021) show that even when the true behavior policy is first-order Markovian, constructing the IS ratio with an estimated **history-dependent** behavior policy paradoxically reduces the Mean Squared Error (MSE). More perplexingly, using longer history windows yields better results.
- **Unresolved Questions**: Why is the "estimated" policy better than the "known ground truth"? Why does introducing historical information that is irrelevant to the current state help? This work provides the first systematic theoretical answers to these questions.

## Method

### Core Intuition: From Bandit to MDP
Taking context-free bandits as an example, three types of IS estimators are defined:
1. **Oracle IS** $\hat{v}_{IS}^\dagger$: uses the true behavior policy $\pi_b$.
2. **Context-Agnostic IS** $\hat{v}_{IS}^{CA}$: uses sample frequency estimation $\hat{\pi}_b(a) = n(a)/n$.
3. **Context-Dependent IS** $\hat{v}_{IS}^{CD}$: uses conditional frequency estimation $\hat{\pi}_b(a|s) = n(s,a)/n(s)$.

**Lemma 1** establishes a strict ordering of asymptotic MSE:

$$\text{MSE}_A(\hat{v}_{IS}^{CD}) \le \text{MSE}_A(\hat{v}_{IS}^{CA}) \le \text{MSE}_A(\hat{v}_{IS}^\dagger)$$

**Key Insight**: Using an IS with an estimated behavior policy is equivalent to a Doubly Robust estimator. Taking $\hat{v}_{IS}^{CD}$ as an example:

$$\hat{v}_{IS}^{CD} = \mathbb{E}_n\left\{\sum_a \pi_e(a)\hat{r}(S,a) + \frac{\pi_e(A)}{\hat{\pi}_b(A|S)}[R - \hat{r}(S,A)]\right\}$$

The first term is the direct method estimator, and the second term is the augmentation term. Their combination achieves: (1) debiasing; and (2) variance reduction by contrasting the observed rewards with predicted rewards.

### History-Dependent Behavior Policy Estimation in MDPs
Define the $k$-step history $H_{t-k:t} = (S_{t-k}, A_{t-k}, \ldots, S_{t-1}, A_{t-1}, S_t)$, which is estimated via maximum likelihood:

$$\hat{\pi}_b^{(k)} = \arg\max_{\pi \in \Pi_k} \mathbb{E}_n\left[\sum_{t=0}^T \log \pi(A_t | H_{t-k:t})\right]$$

This requires the policy class to satisfy **monotonicity** $\Pi_0 \subseteq \Pi_1 \subseteq \Pi_2 \subseteq \cdots$ (which is satisfied by commonly used models such as logistic regression and neural networks).

### Unified Analysis of Four Classes of OPE Estimators

**Theorem 2 (Bias-Variance Decomposition of OIS)**:

$$\text{MSE}(\hat{v}_{OIS}(k)) = \frac{1}{n}\text{Var}\left(\text{Proj}_{\mathbb{T}(k)}(\lambda_T G_T)\right) + O\left(\frac{(k+1)C^{2T}R_{\max}^2}{n^{3/2}\varepsilon^2}\right)$$

- The first term (variance): $O(n^{-1})$, which is **monotonically decreasing** with respect to the history length $k$.
- The second term (bias): $O(n^{-3/2})$, which **increases** as $k$ gets larger and the time horizon $T$ grows.
- Projection Interpretation: Estimating the behavior policy is equivalent to projecting $\lambda_T G_T$ onto the subspace $\mathbb{T}(k)$ orthogonal to the tangent space spanned by the score function.

Summary of effects on SIS, DR, and MIS:

| Estimator | Bias Change | Variance Change |
|--------|---------|---------|
| OIS    | ↑       | ↓       |
| SIS    | ↑       | ↓       |
| DR (Q misspecified) | ↑ | ↓ |
| DR (Q correctly specified) | ↑ | → unchanged |
| MIS    | -       | ↑ increases |

**Counter-intuitive Conclusion for MIS (Theorem 8)**: For the MIS estimator, increasing the history length paradoxically **increases** the MSE. Intuitively, when $k=T$, MIS degenerates into SIS, losing the advantage of marginalization.

### Extension to Nonparametric Estimation (Section 5)
The parametric policy estimation is extended to sieve nonparametric estimation, relaxing the realizability assumption (allowing approximation errors to converge at $o(n^{-1/4})$). It is proved that even under the nonparametric setting, estimating the behavior policy still reduces the asymptotic variance of OIS, SIS, and DR.

### History Length Selection
A selection criterion based on BIC is proposed:

$$h^* = \arg\min_h \left[2n\hat{\text{Var}}(h) - h\log(n)\right]$$

## Key Experimental Results

Experimental Environment: CartPole + MuJoCo (Inverted Pendulum, Double Inverted Pendulum, Swimmer)

| Experimental Findings | Details |
|---------|------|
| SIS History-Dependence | Longer history $\rightarrow$ significantly reduced MSE under large sample sizes, validating Theorem 4 |
| DR (Q misspecified) | History-dependence reduces MSE, but the degree of improvement is conditioned on the quality of Q |
| MIS History-Dependence | Increasing history length $\rightarrow$ MSE **consistently worsens**, validating Theorem 8 |
| Bias Trend | Across all estimators, the bias of the history-dependent version is higher than that of the oracle version |
| Consistency | The MSE of all estimators converges to zero as the sample size increases |
| MuJoCo Extension | Conclusions remain consistent in continuous action spaces and high-dimensional state spaces |

## Highlights & Insights
- **Theoretical Depth**: This work is the first to systematically explain the long-standing "estimation is better than ground truth" paradox in OPE from the perspective of bias-variance decomposition.
- **Unified Framework**: A single theoretical framework covers four major classes of estimators (OIS/SIS/DR/MIS), revealing the **differentiated impacts** of history-dependence on diverse estimators.
- **Projection Perspective**: Estimating the behavior policy $\approx$ implicit doubly robust estimation $\approx$ projection onto a constrained subspace—unifying these three perspectives.
- **Counter-intuitive Conclusion for MIS**: History-dependence is harmful to MIS because marginalization itself already eliminates trajectory-level variance; introducing history instead disrupts this benefit.
- **Practical Guidance**: The BIC criterion provides a principled way to choose the history length in practical applications.

## Limitations & Future Work
- The theoretical analysis assumes finite state/action spaces and finite time horizons; generalization to continuous or infinite time horizons remains incomplete.
- The conclusions for nonparametric estimation only prove asymptotic superiority, without providing explicit finite-sample bias bounds.
- Although the history length selection criterion (BIC) is empirically validated, it lacks theoretical optimality guarantees.
- The experimental scale is limited (CartPole and simple MuJoCo) and has not been verified on high-dimensional complex tasks (such as Atari or robotic manipulation).
- The effect of history-dependent estimation under partially observable (POMDP) settings is not considered.

## Related Work & Insights
- **Hanna et al. (2019, 2021)**: Empirically found that estimating a history-dependent behavior policy reduces the MSE of OIS, but without a theoretical explanation; this paper formally answers their open question.
- **Hirano et al. (2003); Henmi et al. (2007)**: Documented the phenomenon in causal inference where estimating propensity scores outperforms known scores; this work generalizes it to the sequential decision-making scenario.
- **Rowland et al. (2020)**: Conditional IS (CIS) is closely related to the history-dependent MIS ratio analyzed in this paper.
- **Liu et al. (2018)**: Proposed MIS methods to alleviate the horizon curse; this work reveals the negative effects of incorporating history-dependence on top of it.
- **Insight**: **Imperfections in estimators can sometimes be advantageous**—adding noise or estimation error can act as a regularizer/variance reducer, a principle that can be generalized to other statistical estimation problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to thoroughly demystify a classic paradox in OPE theoretically.
- Experimental Thoroughness: ⭐⭐⭐ — Validates the theory but environmental complexity is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Systematically builds intuition step-by-step from bandit examples with a clear structure.
- Value: ⭐⭐⭐⭐ — Holds significant guiding importance for practical OPE applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning](log-sum-exponential_estimator_for_off-policy_evaluation_and_learning.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[ICLR 2026\] Breaking Safety Paradox with Feasible Dual Policy Iteration](../../ICLR2026/reinforcement_learning/breaking_safety_paradox_with_feasible_dual_policy_iteration.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](wasserstein_policy_optimization.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](../../NeurIPS2025/reinforcement_learning/bootstrap_off-policy_with_world_model.md)

</div>

<!-- RELATED:END -->
