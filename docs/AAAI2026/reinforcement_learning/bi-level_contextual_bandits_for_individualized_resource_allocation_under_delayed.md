---
title: >-
  [Paper Note] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback
description: >-
  [AAAI 2026][Reinforcement Learning][Contextual Bandits] This paper proposes MetaCUB — a bi-level contextual bandit framework for individualized resource allocation under delayed feedback, dynamic cohorts…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Contextual Bandits"
  - "Resource Allocation"
  - "Delayed Feedback"
  - "Fairness"
  - "Bi-Level Optimization"
date: 2026-05-08
content_hash: 3d100075f3a0dc12
---

# Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback

**Conference**: AAAI 2026
**arXiv**: [2511.10572](https://arxiv.org/abs/2511.10572)  
**Code**: [GitHub](https://github.com/sinatorrr/MAB)  
**Area**: Reinforcement Learning
**Keywords**: Contextual Bandits, Resource Allocation, Delayed Feedback, Fairness, Bi-Level Optimization

## TL;DR

This paper proposes MetaCUB — a bi-level contextual bandit framework for individualized resource allocation under delayed feedback, dynamic cohorts, cooldown constraints, and fairness requirements. The meta-level optimizes subgroup budget allocation to ensure fairness, while the base-level applies a UCB strategy to select the most promising individuals within each subgroup.

## Background & Motivation

In high-stakes domains such as education, healthcare, and employment, resource allocation must balance short-term utility against long-term impact while satisfying ethical constraints and fairness requirements. Existing multi-armed bandit (MAB) methods exhibit several critical limitations:

**Instantaneous Feedback Assumption**: Most methods assume rewards are immediately observable, whereas the effects of real-world interventions (e.g., tutoring, vocational training, medical treatment) may take weeks or months to manifest.

**Static Population Assumption**: Existing methods ignore the dynamic nature of real deployments, where participants enter and exit in batches (e.g., semesters, enrollment cycles).

**Single-Level Optimization**: Methods either optimize at the individual level (ignoring group fairness) or focus solely on group fairness (ignoring individual heterogeneity).

**Missing Constraints**: Practical constraints such as cooldown periods (preventing the same resource from being repeatedly allocated to the same individual) are not considered.

## Method

### Overall Architecture

MetaCUB adopts a bi-level optimization architecture:

- **Meta-level**: Distributes the total resource budget proportionally across demographic subgroups (e.g., racial or gender groups) to ensure inter-group fairness.
- **Base-level**: Within each subgroup, applies a contextual bandit strategy to select individuals most likely to benefit.

Core modeling elements:
- $N$ individuals belonging to $K$ subgroups, each with a feature vector $\mathbf{x}^i \in \mathbb{R}^M$
- $R$ resource types, each with budget $b^r$
- Objective: maximize expected cumulative reward over $T$ decision rounds $\max \mathbb{E}[\sum_{t=1}^T y(t)]$

### Key Designs

#### 1. Delayed Feedback Modeling (Delay Kernel)

Each resource $r$ has a dedicated delay kernel $K^r$ characterizing the temporal distribution of rewards:

$$K^r(\tau) = \int_{\tau/T}^{(\tau+1)/T} \text{Beta}(z; \alpha^r, \beta^r) \, dz$$

- $\alpha^r < 1$: immediate feedback type
- $\beta^r < 1$: long-tailed delay type
- $\alpha^r, \beta^r > 1$: unimodal concentrated type

The observed reward aggregates delayed effects from all historical allocations:
$$y(t) = \sum_{u=1}^{t} \sum_{i \in \mathcal{I}_{h(u)}} \sum_{r \in R} K^r(t-u) \cdot f(\mathbf{x}^i(u)) \cdot z_{i,r}(u)$$

#### 2. Dynamic Cohorts and Cooldown Constraints

- **Cohort dynamics**: The time horizon is divided into $H = \lceil T/L \rceil$ blocks of length $L$, each corresponding to a cohort $\mathcal{I}_h$ that is periodically replaced.
- **Cooldown constraints**: After an individual receives resource $r$, they cannot receive the same resource again for the next $c^r$ rounds:
$$\sum_{s=t}^{t+c^r} z_{i,r}(s) \leq 1$$

#### 3. Meta-Level Optimization (Algorithm 1)

The meta-level searches for the optimal subgroup-level budget allocation $\bar{\boldsymbol{z}}^* = \{\bar{z}_r^k\}$ via a simulation-based UCB strategy:

1. Initialize $n_0$ candidate allocation policies.
2. At each round, sample a candidate set and evaluate each candidate over $B$ simulated runs.
3. UCB acquisition function: $\bar{\boldsymbol{z}}(t_m) = \arg\max_{\bar{\boldsymbol{z}}} (\mu(\bar{\boldsymbol{z}}) + \beta_{t_m} \sigma(\bar{\boldsymbol{z}}))$
4. Use a pre-trained reward model $f$ as a simulation surrogate to avoid high-dimensional GP inference.

#### 4. Base-Level Optimization (Algorithm 2)

Given the meta-level allocation $\bar{z}_r^k$, the base-level selects the highest-benefit individuals within each (subgroup, resource) pair:

1. Compute allocation count $n_{k,r} = \lfloor \bar{z}_r^k \cdot |\mathcal{I}_k| \rfloor$.
2. Compute UCB score for each individual: $G_{i,r} = \hat{y}_{i,r} + \beta \cdot u_{i,r}$
3. Select the top-$n_{k,r}$ individuals by score.

### Loss & Training

- Prediction model $f$: ridge regression (linear) and neural network (nonlinear) for the ELS dataset; logistic regression and neural network for the JOBS dataset.
- Prediction accuracy is ensured to be consistent across subgroups (e.g., 86% overall on ELS, 85%–87% across racial groups).
- Cooldown periods are sampled uniformly from $\{1, 2, 3\}$.

## Key Experimental Results

### Main Results (Cumulative Regret on ELS Dataset)

| Algorithm | Instant+Linear | Instant+Nonlinear | Delayed+Linear | Delayed+Nonlinear |
|-----------|---------------|-------------------|---------------|-------------------|
| MetaCUB | **Lowest** | **Lowest** | **Lowest** | **Lowest** |
| UCB | High | Medium | Very High | High |
| LinUCB | Medium | Medium | High | Medium |
| CUCB | Med-High | — | — | — |
| EXP3 | High | High | Very High | Very High |
| DUCB | — | — | Med-High | Med-High |
| SWUCB | — | — | Med-High | Medium |

MetaCUB's advantage is most pronounced under the delayed feedback setting.

### Fairness Analysis (Allocation Fairness Ratio on ELS)

| Algorithm | Asian | White | Black | Hispanic |
|-----------|-------|-------|-------|----------|
| MetaCUB (Instant) | 0.84 | 1.03 | 1.02 | 0.98 |
| MetaCUB (Delayed) | 1.02 | 0.96 | 1.00 | 0.97 |
| UCB (Instant) | 0.62 | 1.29 | 0.48 | 0.57 |
| UCB (Delayed) | 0.41 | 1.42 | 0.33 | 0.51 |

A fairness ratio close to 1.0 indicates balanced allocation. MetaCUB achieves ratios closest to 1.0 across all subgroups, whereas UCB exhibits severe bias toward the White group.

### Ablation Study

**Effect of Delay Kernel Type**:
- Type-I kernel (concentrated, early-to-mid feedback): all algorithms achieve relatively low regret, with MetaCUB showing a clear advantage.
- Type-II kernel (diffuse, extended support): temporally diluted reward signals cause non-delay-aware methods to suffer sharply increased regret, while MetaCUB remains robust.

**Linear vs. Nonlinear**: All algorithms achieve lower regret under nonlinear reward functions, suggesting that more expressive models facilitate learning efficiency.

### Key Findings

1. MetaCUB achieves the lowest cumulative regret across all 8 experimental configurations (2 datasets × 2 kernel types × 2 reward functions).
2. Traditional bandit algorithms (UCB, EXP3) incur significantly higher regret under delayed feedback, validating the necessity of explicit delay modeling.
3. The bi-level structure substantially improves fairness — MetaCUB's subgroup allocation ratios all fall within the range 0.84–1.03.

## Highlights & Insights

1. **Elegant Delay Kernel Design**: The Beta-distribution-based parameterizable delay kernel can flexibly represent diverse feedback patterns — instantaneous, long-tailed, and unimodal — while maintaining normalization guarantees.
2. **Bi-Level Decoupling**: Fairness (meta-level) and personalization (base-level) are elegantly separated, ensuring group-level equity without sacrificing individual-level optimality.
3. **Theoretical Fairness Guarantee** (Lemma 1): It is proven that the subgroup allocation disparity under MetaCUB is strictly smaller than that of single-level bandits, with the gap $\delta(T_m, f)$ increasing with the number of meta-level rounds and prediction accuracy.

## Limitations & Future Work

1. The delay kernel parameters $(\alpha^r, \beta^r)$ must be specified in advance; online learning or adaptive estimation is not supported.
2. The meta-level employs simulation-based UCB instead of GP, yielding a less tight theoretical regret bound compared to classical Bayesian optimization.
3. Experiments are conducted on only two datasets, without coverage of additional high-stakes domains such as healthcare.
4. The fairness notion adopted is relatively simple (subgroup mean disparity) and does not account for more sophisticated concepts such as causal fairness.
5. Overall performance is highly sensitive to the quality of the prediction model $f$, yet robustness to model misspecification is not discussed.

## Related Work & Insights

- **Relation to Classical MAB**: MetaCUB extends standard UCB with delay kernels, cooldown constraints, and a bi-level structure to accommodate real-world deployment conditions.
- **Relation to Fair MAB**: Unlike flat fairness constraints, the bi-level structure achieves a fairness–efficiency balance at both the group and individual levels.
- **Insights**: The delay kernel modeling approach is transferable to other online learning problems with delayed feedback; the bi-level framework can be generalized to multi-level hierarchical resource allocation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of bi-level framework and delay kernel is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive experiments across two real-world datasets, multiple settings, and fairness analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, though the notation is dense and the paper is somewhat lengthy.
- Value: ⭐⭐⭐⭐ — Highly practical, open-sourced, and directly addresses real-world resource allocation pain points.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](../../ICLR2026/reinforcement_learning/virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](../../NeurIPS2025/reinforcement_learning/bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](../../ICML2026/reinforcement_learning/parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](../../NeurIPS2025/reinforcement_learning/variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)

</div>

<!-- RELATED:END -->
