---
title: >-
  [Paper Note] On the Exponential Convergence for Offline RLHF with Pairwise Comparisons
description: >-
  [AAAI 2026][LLM Alignment][Offline RLHF] Under the offline RLHF pairwise comparison setting, this paper proposes the RL-LOW algorithm achieving exponential convergence $\exp(-\Omega(n/H))$ for simple regret, and derives the first instance-dependent lower bound proving this rate is optimal in the exponential sense.
tags:
  - AAAI 2026
  - LLM Alignment
  - Offline RLHF
  - Exponential Convergence
  - Pairwise Comparisons
  - Instance-Dependent Lower Bound
  - Differential Privacy
date: 2026-05-08
content_hash: d1f487dd9319f88e
---

# On the Exponential Convergence for Offline RLHF with Pairwise Comparisons

**Conference**: AAAI 2026
**arXiv**: [2406.12205](https://arxiv.org/abs/2406.12205)
**Code**: None
**Area**: LLM Alignment / Reinforcement Learning Theory
**Keywords**: Offline RLHF, Exponential Convergence, Pairwise Comparisons, Instance-Dependent Lower Bound, Differential Privacy

## TL;DR

Under the offline RLHF pairwise comparison setting, this paper proposes the RL-LOW algorithm achieving exponential convergence $\exp(-\Omega(n/H))$ for simple regret, and derives the first instance-dependent lower bound proving this rate is optimal in the exponential sense.

## Background & Motivation

**State of the Field**: RLHF (Reinforcement Learning from Human Feedback) has become the core paradigm for LLM alignment. In the offline setting, training a reward model from pre-collected human pairwise preference data is a key step in systems such as InstructGPT. On the theoretical side, Zhu et al. (2023) established the minimax optimality of Pessimistic MLE, and Zhan et al. (2024) extended this to general reward functions.

**Limitations of Prior Work**: Existing theoretical analyses—including Zhu et al. (2023), Zhan et al. (2024), Cen et al. (2024), and Liu et al. (2024)—all focus exclusively on worst-case regret bounds, yielding upper bounds of the form $\tilde{O}(n^{-1/2})$, i.e., inverse polynomial convergence rates. Such worst-case analyses fail to reveal how the intrinsic structure of a problem instance influences algorithm performance.

**Root Cause**: Worst-case bounds are independent of instance-specific parameters (e.g., suboptimality gaps of individual actions), and thus cannot distinguish between "easy" and "hard" instances, offering limited insight into the true capabilities and limitations of algorithms.

**Paper Goals**: To establish an instance-dependent analytical framework for offline RLHF with pairwise comparisons, addressing two core questions: (1) What is the fastest achievable convergence rate? (2) Is this rate fundamentally unimprovable?

**Starting Point**: Drawing on instance-dependent optimal analysis from the bandit and hypothesis testing literature, the paper conducts a fine-grained analysis exploiting the structure of the linear reward + BTL preference model.

**Core Idea**: Design *locally optimal weights* to minimize a proxy for the variance of relative reward estimates for each state-action pair, thereby achieving the tightest possible exponential convergence for each instance.

## Method

### Overall Architecture

Under the Bradley-Terry-Luce (BTL) preference model, rewards are linear functions of feature vectors: $r_{k,i} = \langle \phi(k,i), \theta \rangle$. Given an offline dataset $\mathcal{D} = \{(s_i, a_i^{(0)}, a_i^{(1)}, \sigma_i)\}_{i=1}^n$, the goal is to identify the optimal action for each state so as to minimize simple regret $R_n = \mathbb{E}_{k \sim \rho}[r_{k,i_k^*} - r_{k,\hat{i}_k}]$.

The algorithm proceeds in six steps: (1) compute sample proportions; (2) compute empirical success rates; (3) clip outliers; (4) compute locally optimal weights; (5) estimate relative rewards; (6) select the optimal action for each state.

### Key Designs

#### Module 1: Locally Optimal Weights

- **Function**: Tailor a set of optimal linear combination weights for each state-action triplet $(k,i,j)$ to estimate the relative reward between actions $i$ and $j$ from the data.
- **Mechanism**: Define the weight space $\mathcal{U}_{k,i,j}$ as the set of all vectors $u$ satisfying the linear constraint $\phi(k,i)-\phi(k,j) = \sum u_{k',i',j'}(\phi(k',i')-\phi(k',j'))$. The locally optimal weight $w^{(k,i,j)}$ is obtained by minimizing the variance proxy $\sum (u_{k',i',j'})^2 / N_{k',i',j'}$ over this space.
- **Design Motivation**: Global unified estimation (e.g., MLE) imposes the same estimation strategy on all action pairs, whereas locally optimal weights tailor the minimum-variance weight allocation to each pair individually. The key insight is that $(u_{k',i',j'})^2/N_{k',i',j'}$ serves as a variance proxy for success rate estimation, and minimizing it is equivalent to obtaining the tightest sub-Gaussian concentration inequality.

#### Module 2: Clipped Success Rate

- **Function**: Constrain observed empirical success rates to a theoretically valid interval.
- **Mechanism**: Under the bounded reward assumption $|\langle \phi(k,i), \theta \rangle| \leq L$, success rates in the BTL model must lie in $[1/(1+e^{2L}),\, e^{2L}/(1+e^{2L})]$. Empirical success rates outside this interval are clipped back to its boundaries.
- **Design Motivation**: Ensures that estimates are consistent with the model, and prevents extreme empirical values—especially under small sample sizes—from causing numerical instability in subsequent log-odds transformations.

#### Module 3: Instance-Dependent Difficulty Characterization

- **Function**: Define a difficulty parameter $H(v)$ to precisely characterize the intrinsic hardness of each problem instance.
- **Mechanism**: $H(v) = \max_{k,i:\,i \neq i_k^*} \gamma_{k,i}/\Delta_{k,i}^2$, where $\gamma_{k,i}$ is a variance proxy for the relative reward estimate and $\Delta_{k,i}$ is the suboptimality gap. Intuitively, the ratio $\gamma_{k,i}/\Delta_{k,i}^2$ measures the difficulty of misidentifying suboptimal action $i$ as optimal—larger variance and smaller gap both increase difficulty.
- **Design Motivation**: This parameter bridges the upper and lower bound analyses. The upper bound shows that RL-LOW achieves $\exp(-n/(C_{up} \cdot H(v)))$, while the lower bound shows that no algorithm can surpass $\exp(-n/(C_{lo} \cdot H(v)))$, with both matching in the exponential rate.

### Loss & Training

**Relative Reward Estimation**: Relative rewards are estimated using locally optimal weights and the log-odds transformation of clipped success rates:

$$\hat{r}_{k,i,j} = \sum_{k',i',j'} w^{(k,i,j)}_{k',i',j'} \log\frac{B^{CLP}_{k',i',j'}}{1 - B^{CLP}_{k',i',j'}}$$

**Differential Privacy Extension**: RL-LOW is combined with the Gaussian mechanism to achieve $(\varepsilon, \delta)$-differential privacy. The key result is that the difficulty parameter $H$ remains unchanged as $n \to \infty$, meaning privacy constraints do not increase the learning difficulty asymptotically.

**Computational Complexity**: $\mathcal{O}(SAd + nd^2 + d^3)$, on par with linear regression.

## Key Experimental Results

### Main Results

This paper is a purely theoretical contribution; core results are presented as theorems:

| Result Type | Bound Form | Remarks |
|---|---|---|
| Upper Bound (Thm 3.3) | $\exp(-n/(C_{up} \cdot H(v)))$ | First instance-dependent exponential upper bound |
| Lower Bound (Thm 4.3) | $\exp(-n/(C_{lo} \cdot H(v)))$ | First instance-dependent lower bound |
| Worst-case Upper Bound (Prop 3.4) | $O(n^{-1/2})$ | Improves upon Zhu et al.'s $O(\sqrt{n^{-1}\log n})$ |
| Differential Privacy | $H$ asymptotically unchanged | Privacy is "free" |

### Ablation Study

- **Necessity of the Consistency Condition (Prop 2.3)**: Proves that no algorithm can achieve vanishing regret under inconsistent instances—a fundamental limitation of the problem.
- **Well-Definedness of the Set $\hat{\mathcal{I}}_k$ (Prop 3.2)**: Proves that RL-LOW always identifies a unique optimal action estimate without exhaustively enumerating all action pair combinations.

### Key Findings

1. **Exponential convergence is achievable**: Compared to the prior $O(n^{-1/2})$ worst-case bound, instance-dependent bounds reveal that regret actually decays at an exponential rate.
2. **Optimality established**: Upper and lower bounds match in their dependence on $H(v)$ in the exponential rate, proving that RL-LOW is instance-optimal.
3. **"Free lunch" of differential privacy**: The difficulty parameter is unchanged after incorporating privacy protection, implying that privacy constraints do not affect learning efficiency in the large-data regime.
4. **Improvement over Pessimistic MLE**: RL-LOW also marginally improves upon Zhu et al.'s result in the worst case by removing the $\log n$ factor.

## Highlights & Insights

- **Paradigm shift from worst-case to instance-dependent analysis**: This work establishes, for the first time, a complete instance-dependent picture (upper bound + lower bound + matching) for offline RLHF with pairwise comparisons—a significant theoretical advance.
- **Design philosophy of locally optimal weights**: Whereas conventional methods (e.g., MLE) employ a unified estimator for all parameters, locally optimal weights tailor the optimal data utilization strategy to each individual decision problem, fundamentally reducing estimation variance.
- **Technical contributions in lower bound proofs**: Lemmas 4.1 and 4.2 provide precise tools for characterizing KL divergence under the BTL model (via the approximate KL divergence $\tilde{D}$), techniques that may prove valuable for a broader class of preference learning theory problems.

## Limitations & Future Work

1. **Linear reward assumption**: Rewards must be linear functions of features; nonlinear rewards or general function approximation cannot be handled directly.
2. **BTL preference model restriction**: BTL is a specific parametric preference model, whereas actual human preferences may be more complex (e.g., exhibiting intransitivity).
3. **Static offline setting**: Exploration is not considered, and the data distribution is fixed; extensions to online or hybrid settings require new techniques.
4. **Tabular/finite state space**: States and actions are assumed finite; analysis in continuous spaces is an important future direction.
5. **Lack of empirical validation**: As a purely theoretical work, numerical experiments to validate finite-sample behavior are absent.

## Related Work & Insights

- **Zhu et al. (2023)**: Minimax optimality of Pessimistic MLE—the direct predecessor of this work; RL-LOW achieves an instance-dependent leap beyond it.
- **Chowdhury et al. (2024)**: Label differential privacy under preference feedback, but analyzes only reward estimation error without addressing simple regret.
- **Instance-dependent analysis in the bandit literature**: The technical approach draws on the locally optimal allocation idea from the multi-armed bandit literature.
- Insight: Instance-dependent analysis in RLHF theory remains in its early stages; more general settings beyond the linear reward structure merit further exploration.

## Rating

⭐⭐⭐⭐

A solid theoretical contribution that establishes, for the first time, a complete picture of exponential convergence (with matching upper and lower bounds) for offline RLHF with pairwise comparisons. The design of locally optimal weights is insightful. Weaknesses include strong assumptions (linear rewards + BTL model) and the absence of empirical validation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](../../ICLR2026/llm_alignment/beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)
- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](../../ICLR2026/llm_alignment/general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](../../ICLR2026/llm_alignment/unifying_stable_optimization_and_reference_regularization_in_rlhf.md)

<!-- RELATED:END -->
