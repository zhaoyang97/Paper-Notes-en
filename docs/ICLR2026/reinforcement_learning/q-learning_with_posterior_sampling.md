---
title: >-
  [Paper Note] Q-learning with Posterior Sampling
description: >-
  [ICLR 2026][Reinforcement Learning][Q-learning] This paper proposes PSQL—the first Q-learning algorithm that uses "maintaining Gaussian posteriors over Q-values and taking argmax after sampling" for exploration. By modifying the target value calculation to "optimistic multi-sample sampling," the authors prove a near-optimal regret upper bound of $\tilde{O}(H^2\sqrt{
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Q-learning
date: 2026-05-08
content_hash: 725c720c37b79d84
---
# Q-learning with Posterior Sampling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=a4z7OlgSxC](https://openreview.net/forum?id=a4z7OlgSxC)  
**Code**: TBD  
**Area**: Reinforcement Learning / Exploration Theory  
**Keywords**: Posterior Sampling, Q-learning, Thompson Sampling, Regret Upper Bound, Tabular MDP

## TL;DR
This paper proposes PSQL—the first Q-learning algorithm that uses "maintaining Gaussian posteriors over Q-values and taking argmax after sampling" for exploration. By modifying the target value calculation to "optimistic multi-sample sampling," the authors prove a near-optimal regret upper bound of $\tilde{O}(H^2\sqrt{SAT})$ for this natural posterior-sampling-style Q-learning.

## Background & Motivation
**Background**: Managing the exploration-exploitation trade-off in online tabular episodic RL follows two main paths. One is UCB (Upper Confidence Bound), which adds an explicit exploration bonus to Q-value estimates, represented by UCBQL (Jin et al., 2018), achieving near-optimal regret. The other is posterior sampling (Thompson Sampling in multi-armed bandits), which implicitly expresses uncertainty by maintaining and sampling from a posterior over parameters. Extensive empirical studies show that posterior sampling often outperforms UCB in practice.

**Limitations of Prior Work**: While posterior sampling is effective, its **theoretical analysis is extremely difficult**, especially in complex RL scenarios involving bootstrapping. Previously, there were no model-free algorithms with provable regret for the "most natural" approach—applying a posterior directly to Q-values. Existing provable results either bypass this path (e.g., Staged-RandQL by Tiapkin et al., 2023, uses a Dirichlet posterior on transition probabilities and implicit sampling via learning rate randomization) or rely on approximate value iteration like RLSVI, which requires least-squares fitting of historical data (high computational/storage overhead, degrades to model-based in tabular settings, and has loose regret dependence on the number of states $S$, i.e., $\tilde{O}(H^3 S^{1.5}\sqrt{AT})$).

**Key Challenge**: The biggest obstacle to integrating posterior sampling into Q-learning TD updates is **error accumulation from bootstrapping**. The Q-learning target $z = r_h + \hat{V}_{h+1}$ is constructed using "current estimates of the next step," propagating errors layer by layer. More critically, posterior sampling lacks the "high-probability optimism" of UCB: UCB estimates are naturally high-probability upper bounds of the true value, whereas posterior sampling usually only guarantees **constant-probability** optimism. When multiplied across $H$ recursive steps, the probability of optimism decays exponentially to $p^H$.

**Goal**: Design a simple Q-learning algorithm that truly "maintains a posterior over Q-values and samples for argmax" and prove a near-optimal regret bound with optimal dependence on $S$.

**Key Insight**: The authors re-interpret Q-learning update rules using Bayesian inference, finding them to be solutions to ELBO optimization with regularization. This provides a principled design for "where and how to inject posterior sampling." A key observation is that to break the exponential decay of optimism probability, it is **not necessary** for the sampling at the decision point to be high-probability optimistic; it only needs to ensure that the **value estimate of the next step used in the target value** is high-probability optimistic.

**Core Idea**: Use "single-sample" posterior sampling for decisions (maintaining natural exploration and efficiency) while constructing targets using "multi-sample maximum + specialized argmax action" to inject limited optimism. These two components work together to bypass the exponential error of bootstrapping.

## Method

### Overall Architecture
PSQL maintains a Gaussian posterior $\mathcal{N}(\hat{Q}_h(s,a),\sigma(N_h(s,a))^2)$ for each $(s,a,h)$ regarding the true value $Q_h(s,a)$, where $\hat{Q}$ is the posterior mean and variance decays with the number of visits $n$:

$$\sigma(n)^2 = \frac{64H^3}{n+1}\log(KH/\delta).$$

At each step $h$ of every episode, the algorithm performs three actions: (1) **Decision**—samples **one** instance $\tilde{Q}_h(s_h,a)\sim\mathcal{N}(\hat{Q}_h(s_h,a),\sigma(N_h(s_h,a))^2)$ for each action at the current state $s_h$ and plays $a_h=\arg\max_a\tilde{Q}_h(s_h,a)$; (2) **Target Construction**—after observing $r_h,s_{h+1}$, uses an **optimistic multi-sample** process to calculate the next-step value estimate $\overline{V}_{h+1}(s_{h+1})$, yielding target $z=r_h+\overline{V}_{h+1}(s_{h+1})$; (3) **Update Posterior Mean**—uses a Q-learning rule with a specific learning rate:

$$\hat{Q}_h(s_h,a_h)\leftarrow(1-\alpha_n)\hat{Q}_h(s_h,a_h)+\alpha_n z,\qquad \alpha_n=\frac{H+1}{H+n}.$$

The "asymmetry" of this method is its essence: **the decision point uses vanilla single-sample posterior sampling** (natural and efficient), while **the target point uses optimistic multi-sample posterior sampling** (to obtain high-probability optimism for analysis).

### Key Designs

**1. Re-deriving Q-learning from Bayesian ELBO: Explaining the update rule and learning rate**

The authors first address what the Q-learning update rule actually "is." By setting the parameter to be inferred $\theta$ as $Q_h(s,a)$, given prior $p$, log-likelihood $\ell(\theta,z)$, and sample $z$, the Bayesian posterior $q(\theta)\propto p(\theta)\exp(\ell(\theta,z))$ is also the optimal solution to the ELBO: $\max_q\, \mathbb{E}_{\theta\sim q}[\ell(\theta,z)]-\mathrm{KL}(q\|p)$. With a Gaussian prior $\mathcal{N}(\hat{\mu},\frac{\sigma^2}{n-1})$ and likelihood $\mathcal{N}(\theta,\sigma^2)$, the posterior becomes $\mathcal{N}(\hat{\mu},\frac{\sigma^2}{n})$, and its mean update is the Q-learning rule with learning rate $\alpha_n=\frac1n$. This suggests standard Q-learning is essentially Bayesian posterior inference.

However, Q-learning's $z$ is **biased** due to bootstrapping. Jin et al. (2018) manually adjusted the learning rate to $\alpha_n=\frac{H+1}{H+n}$ to make the regret analysis work. This paper provides a principled explanation: by adding an entropy regularization term to the ELBO:

$$\max_q\, \mathbb{E}_{\theta\sim q}[\ell(\theta,z)]-\mathrm{KL}(q\|p)+\lambda_n\mathcal{H}(q),$$

when $\lambda_n=\frac{H}{n}$ and the likelihood variance is $\frac{\sigma^2}{H+1}$, the optimal posterior yields exactly $\alpha_n=\frac{H+1}{H+n}$. The intuition is clear: bootstrapping introduced extra bias, so one must **artificially retain more posterior uncertainty**; as the sample size $n$ increases and target bias decreases, the regularizer $\lambda_n=H/n$ naturally decays.

**2. Single-sample posterior sampling for decisions: Bringing Thompson sampling naturally into Q-learning**

Addressing the pain point that posterior sampling is better in practice but lacks theory for Q-value versions, PSQL adopts the most basic Thompson-style approach at decision points: sample one value for each action and play the one with the maximum sampled Q-value. Actions with fewer visits $(s,a)$ have larger posterior variance $\sigma(n)^2\propto H^3/(n+1)$, making samples more likely to deviate from the mean (exploration); actions with more visits have smaller variance, making samples closer to the mean (exploitation). This exploration relies on posterior sampling rather than explicit bonuses or perturbations.

Only **one sample** is taken at the decision point because the authors demonstrate that if multi-sample maximization were used there to force optimism, bootstrapping would amplify errors by $\epsilon\sqrt{J^{H}\log(1/\delta)}$, exploding exponentially with $H$ even for $J=2$. Thus, decisions must remain vanilla, and optimism must be sourced elsewhere.

**3. Optimistic multi-sample target construction: Injecting limited optimism in the target to bypass exponential decay**

This is the technical heart of the paper, solving the problem of constant-probability optimism decaying over $H$ steps. The insight is: to break the recursive multiplication, one **does not need** the decision-point estimate to be optimistic; one only needs the **next-step value $\overline{V}_{h+1}$ used in the target** to be optimistic with high probability. Target construction (Algorithm 2) does two things:

First, it **samples multiple times for a single specific action $\hat{a}$ and takes the maximum**. It first selects $\hat{a}=\arg\max_a\big(\hat{Q}_{h+1}(s',a)+\sigma(N_{h+1}(s',a))\big)$ (posterior mean plus one standard deviation offset), then takes $J$ samples from $\mathcal{N}(\hat{Q}_{h+1}(s',\hat{a}),\sigma(N_{h+1}(s',\hat{a}))^2)$:

$$\overline{V}_{h+1}(s')=\max_{j\in[J]}\tilde{V}^j,\qquad J=\frac{\log(SAT/\delta)}{\log(4/(4-p_1))},\quad p_1=\frac{\Phi(-1)-\delta}{H-\delta}.$$

Taking the maximum of $J$ samples boosts "constant-probability optimism" to "high-probability optimism" (Lemma 1: $\overline{V}_{k,h}(s_{k,h})\ge V^*_h(s_{k,h})$). Crucially, this optimism **only applies to the target calculation** and does not contaminate the decision in Line 6, preventing errors from multiplying recursively.

Second, it **uses $\hat{a}$ instead of the actually played action $a_{h+1}$**. The difficulty is that $\overline{V}_{h+1}$ is sampled at $\hat{a}$, while the decision uses $a_{h+1}=\arg\max_a\tilde{Q}$. The authors prove (Lemma C.2) that $\hat{a}$ is a "special action": its standard deviation is bounded by the played action's uncertainty as $\sigma(N(s,\hat{a}))^2<2\sigma(N(s,a_h))^2\log(1/\delta)$. Thus, the gap between $\overline{V}_h(s_h)$ and the decision sample $\tilde{Q}_h(s_h,a_h)$ is controlled by $\tilde{O}(\sigma(N_{k,h}(s_{k,h},a_{k,h})))$, which decays as $1/\sqrt{N}$.

### Loss & Training
There is no traditional loss function—PSQL uses online TD updates. The core training involves the posterior mean update $\hat{Q}_h\leftarrow(1-\alpha_n)\hat{Q}_h+\alpha_n z$ with learning rate $\alpha_n=\frac{H+1}{H+n}$ and variance schedule $\sigma(n)^2=64H^3\log(KH/\delta)/(n+1)$. Initialization: $\hat{Q}_h=\hat{V}_h=H$ (optimistic initialization), $\hat{Q}_{H+1}=0$.

## Key Experimental Results

### Main Results: Regret Upper Bound Comparison
This is a theoretical work; the core "experiment" is the comparison of regret magnitudes in tabular episodic RL ($T=KH$).

| Algorithm | Regret | Type |
|-----------|--------|------|
| UCBQL (Jin et al., 2018) | $\tilde{O}(H^{1.5}\sqrt{SAT})$ | Q-learning + UCB |
| Q-EarlySettled-Advantage (Li et al., 2021a) | $\tilde{O}(H\sqrt{SAT})$ | Q-learning + UCB |
| RLSVI (Russo, 2019) | $\tilde{O}(H^3 S^{1.5}\sqrt{AT})$ | Approx. Value Iteration |
| C-RLSVI (Agrawal et al., 2021) | $\tilde{O}(H^2 S\sqrt{AT})$ | Approx. Value Iteration |
| Staged-RandQL (Tiapkin et al., 2023) | $\tilde{O}(H^2\sqrt{SAT})$ | LR Randomization |
| **PSQL (Ours)** | $\tilde{O}(H^2\sqrt{SAT})$ | **Q Gaussian Posterior** |
| Lower Bound (Jin et al., 2018) | $\Omega(H\sqrt{SAT})$ | — |

PSQL's $\tilde{O}(H^2\sqrt{SAT})$ is near-optimal in $S, A, T$, differing from the lower bound only by a factor of $H$. It improves the dependence on $S$ from $S^{1.5}$ (RLSVI) to $\sqrt{S}$, matching the more complex Staged-RandQL but with a simpler algorithm.

### Ablation Study
In a chain MDP environment (Figure 1), a heuristic variant PSQL* was compared with UCBQL, Staged-RandQL, and RLSVI. It verified that the empirical superiority of posterior sampling over UCB holds for the Q-learning path. Note: PSQL* used in experiments is a **vanilla single-sample version** (single sample for both decision and target), which performs better than the provable Algorithm 1 but is harder to analyze.

### Key Findings
- **Location of optimism injection is everything**: Injecting optimism only at the target (not the decision) is key to bypassing exponential decay of optimism probability.
- **Specialized action $\hat{a}$ is the analysis pivot**: Lemma 2 binds the gap between the "target sampled at $\hat{a}$" and the "actually played action" to $\sigma(N)\propto1/\sqrt{N}$.
- **Empirical optimality $\neq$ Ease of analysis**: The vanilla single-sample version (PSQL*) is empirically strongest but currently lacks a provable bound.

## Highlights & Insights
- **Giving a rationale to "hand-tuned" learning rates**: Explaining Jin et al.'s (2018) $\alpha_n=\frac{H+1}{H+n}$ via entropy-regularized ELBO elevates an engineering trick to a principled Bayesian inference result.
- **"Asymmetric sampling" philosophy**: Keeping the decision point natural (single sample) and injecting optimism only into the recursive bottleneck (next-step value in target) is an insightful strategy for analyzing other bootstrap-based algorithms.
- **Clear diagnosis of difficulties**: The authors explicitly present failure modes like "exponential decay of optimism probability" and "exponential error from multi-sample decisions," showing how their design overcomes them.

## Limitations & Future Work
- **Gap between theory and practice**: Algorithm 1 (provable with multi-sample targets) is empirically **weaker** than the single-sample PSQL*, which currently cannot be analyzed.
- **$H$ dependence is not yet optimal**: The upper bound $\tilde{O}(H^2\sqrt{SAT})$ still differs from the lower bound $\Omega(H\sqrt{SAT})$ by a factor of $H$.
- **Limited to the tabular setting**: The analysis relies on finite $S,A$ and explicit visit counts $N$, and has not yet been extended to function approximation or Deep RL.
- **Large constants**: The constant 64 and $H^3$ variance make the exploration strength quite conservative, potentially explaining why the provable version is empirically inferior to the vanilla version.

## Related Work & Insights
- **vs UCBQL (Jin et al., 2018)**: Both are Q-learning based, but UCBQL achieves optimism via explicit bonuses at the **decision point**, whereas PSQL uses **single-sample decisions** and moves optimism to the **target point**. PSQL also provides a Bayesian interpretation for UCBQL's learning rate.
- **vs Staged-RandQL (Tiapkin et al., 2023)**: Both are model-free and achieve $\tilde{O}(H^2\sqrt{SAT})$. However, Staged-RandQL uses Dirichlet posteriors on transitions and LR randomization, which is closer to a model-based concept. PSQL directly uses Gaussian posteriors on Q-values, which is more natural and involves fewer sampling steps.
- **vs RLSVI (Russo, 2019)**: RLSVI injects noise into rewards and re-fits Q-functions using least squares without bootstrapping. It is computationally heavy and has looser $S$ dependence ($S^{1.5}$). PSQL uses incremental TD updates and improves $S$ dependence to $\sqrt{S}$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First provable Q-learning with "Q-value posterior sampling." ELBO re-interpretation of the learning rate is a strong independent contribution.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; empirical validation is limited to small environments like chain MDPs and focuses on a heuristic variant.
- Writing Quality: ⭐⭐⭐⭐⭐ Diagnosis of technical difficulties and design motivation are exceptionally clear.
- Value: ⭐⭐⭐⭐ Provides a starting point and reusable technical insights for the long-standing open problem of analyzing "posterior sampling + TD/DP."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](../../ICML2025/reinforcement_learning/fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[ICLR 2026\] Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning](policy_likelihood-based_query_sampling_and_critic-exploited_reset_for_efficient_.md)
- [\[AAAI 2026\] Speculative Sampling with Reinforcement Learning](../../AAAI2026/reinforcement_learning/speculative_sampling_with_reinforcement_learning.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Toward Efficient Exploration by Large Language Model Agents](toward_efficient_exploration_by_large_language_model_agents.md)

</div>

<!-- RELATED:END -->
