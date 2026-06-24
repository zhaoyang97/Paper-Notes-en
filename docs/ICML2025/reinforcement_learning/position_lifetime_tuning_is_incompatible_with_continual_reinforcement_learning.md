---
title: >-
  [Paper Note] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning
description: >-
  [ICML 2025][Reinforcement Learning][continual RL] This position paper identifies a critical methodological flaw in continual reinforcement learning (RL) research: "lifetime tuning" (hyperparameter tuning over the entire agent lifetime) masks the true continual learning capability of algorithms. It proposes k%-percent tuning as a more reasonable alternative for evaluation.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "continual RL"
  - "hyperparameter tuning"
  - "lifetime tuning"
  - "evaluation methodology"
  - "loss of plasticity"
date: 2026-05-08
content_hash: eb575e1ee6182be5
---

# Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning

---

**Conference**: ICML 2025  
**arXiv**: [2404.02113](https://arxiv.org/abs/2404.02113)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: continual RL, hyperparameter tuning, lifetime tuning, evaluation methodology, loss of plasticity

## TL;DR

This position paper identifies a critical methodological flaw in continual reinforcement learning (RL) research: "lifetime tuning" (hyperparameter tuning over the entire agent lifetime) masks the true continual learning capability of algorithms. It proposes k%-percent tuning as a more reasonable alternative for evaluation.

---

## Background & Motivation

**Background**: Continual reinforcement learning (Continual RL) aims to build agents capable of lifelong learning under non-stationary environments. Recently, various methods have emerged to improve the continual learning capabilities of standard RL algorithms, falling into three main categories: reset-based methods (periodically resetting network parameters) such as Nikishin et al. (2022), regularization-based methods (parameters kept near initialization) such as Kumar et al. (2024), and normalization-based methods (layer normalization to maintain plasticity) such as Lyle et al. (2023).

**Limitations of Prior Work**: Continual RL research typically follows a fixed template: (1) introduce a non-stationary continual learning benchmark, (2) demonstrate that existing algorithms fail on this new benchmark, and (3) propose a new mitigation algorithm and prove its effectiveness. However, this seemingly reasonable pipeline contains a severe methodological flaw: hyperparameters of all algorithms (including both baselines and new methods) are optimized over the **entire lifetime** of the agent. This implies that researchers effectively "peek" at the test set through repeated trial and error.

**Key Challenge**: Lifetime tuning contradicts the fundamental definition of continual learning. A core assumption of continual RL is that the agent does not know the deployment duration and must cope with lifetimes of unknown lengths. However, lifetime tuning allows researchers to meticulously optimize hyperparameters (e.g., epsilon decay schedules, buffer sizes) for a specific lifetime length (e.g., 200M frames), making the algorithm perform optimally—but if the actual deployment is longer or shorter, performance degrades. Worse, in non-stationary environments, running repeated experiments effectively leaks information about the hidden dynamics to researchers, stripping the benchmark of true partial observability.

**Goal**: (1) Argue and empirically demonstrate why lifetime tuning is detrimental to continual RL research; (2) show how lifetime tuning masks the true advantages of continual learning algorithms; and (3) propose a more reasonable evaluation methodology.

**Key Insight**: Grounded in the most fundamental machine learning principle of "do not peek at the test set," this work draws an analogy to the train/test split in supervised learning, pointing out that lifetime tuning in continual RL essentially constitutes overfitting on the test set.

**Core Idea**: Restrict hyperparameter tuning to only use the interaction data from the first k% of the lifetime, thereby forcing the algorithm to possess genuine continual learning capabilities rather than the ability to overfit to a specific lifetime duration.

## Method

### Overall Architecture

The core thesis of the paper is developed through a series of progressive experiments. First, the issue of lifetime tuning is demonstrated using DQN on a Non-stationary Catch environment: (1) DQN indeed fails under default hyperparameters in non-stationary environments; (2) W0-DQN (weight regularization) appears effective under lifetime tuning; (3) however, if DQN is also subjected to lifetime tuning, DQN becomes equally effective—lifetime tuning makes all algorithms look similarly good; (4) key experiment: keeping the previously found optimal hyperparameters fixed and extending the experiment duration by 20 times—DQN's performance collapses while W0-DQN remains stable, proving that the latter indeed possesses superior continual learning capabilities, which were simply masked by lifetime tuning.

### Key Designs

1. **k%-percent Tuning**:

    - **Function**: Restraints hyperparameter search to only use interactive data from the first k% of the lifetime.
    - **Mechanism**: If the agent is to run for $n$ steps, hyperparameter search (via grid search or Bayesian optimization) is only permitted within the first $j = \lfloor kn \rfloor$ steps. Once the optimal hyperparameters are selected, the agent is deployed and run multiple times over the full $n$ steps under this configuration to report performance. Typical settings for $k$ are 1%, 5%, and 10%.
    - **Design Motivation**: This simulates real-world deployment scenarios—namely, one only has access to a limited amount of early trial data for tuning, after which the hyperparameters must remain fixed for long-term deployment. A smaller $k$ poses a higher demand on the adaptive capabilities of algorithms, aligning better with the philosophy of "lifelong learning".

2. **Demonstration of the Two Pitfalls of Lifetime Tuning**:

    - **Function**: Systematically demonstrates how lifetime tuning misleads research conclusions.
    - **Mechanism**: **Pitfall 1**—If the baseline algorithm is not equally tuned, one will falsely conclude that the baseline is unsuitable for continual learning (when in fact it was merely a hyperparameter mismatch with the new environment). **Pitfall 2**—If all algorithms undergo lifetime tuning, they will perform similarly, making it impossible to identify the truly superior continual learning algorithms. These two pitfalls have led to "mixed progress" in recent continual RL research.
    - **Design Motivation**: Explains why, despite numerous publications in the continual RL field, overall progress remains limited—the evaluation methodology itself suffers from systematic bias.

3. **Multi-Environment and Multi-Algorithm Validation**:

    - **Function**: Validates the effectiveness of k% tuning across diverse setups.
    - **Mechanism**: Tests DQN (discrete actions) and SAC (continuous actions) on multiple continual/non-stationary environments, including Non-stationary Mountain Car, Continuing Mountain Car, Non-stationary CartPole, Non-stationary Acrobot, and modified Catch. It compares performance differences between lifetime tuning and k%-percent tuning across different $k$ values, while testing various mitigation strategies (W0-regularization, layer normalization, periodic resets).
    - **Design Motivation**: Rules out the possibility that the conclusions are only applicable to specific environments or algorithms.

### Evaluation Metrics Considerations

The paper also discusses which metrics should be used to select hyperparameters during the $k\%$ tuning phase. Three metrics are considered: (1) total return during the tuning phase, (2) average return of the last 10% of the tuning phase, and (3) average TD error during the tuning phase. It is found that no single metric is optimal across all environment-algorithm combinations, suggesting that the metric choice itself should be treated as a hyperparameter.

## Key Experimental Results

### Lifetime Tuning vs k%-percent Tuning (DQN)

| Environment | Lifetime Tuning Total Return | k=5% Tuning Total Return | k=1% Tuning Total Return |
|------|----------------------|-------------------|-------------------|
| NS Mountain Car | Highest (≈Optimal) | ~90% of lifetime | ~70% of lifetime |
| NS CartPole | Highest (≈Optimal) | ~85% of lifetime | ~60% of lifetime |
| Continuing MC | Highest (≈Optimal) | ~80% of lifetime | ~55% of lifetime |

*Specific values vary by environment, but the trend remains consistent: smaller $k$ values result in lower performance but reflect the true continual learning capability more accurately.*

### Effectiveness of Mitigation Strategies Under Different Tuning Setups

| Method | Outperforms DQN Under Lifetime Tuning? | Outperforms DQN Under k%-percent Tuning? |
|------|----------------------------|------------------------------|
| W0-DQN (Regularization) | No (Comparable to DQN) | **Yes** (Significantly outperforms) |
| LayerNorm-DQN | No (Comparable to DQN) | **Yes** (Significantly outperforms) |
| Reset-DQN | No (Comparable to DQN) | Outperforms in some environments |

### Key Findings

- **Core Finding**: Under lifetime tuning, all algorithms (including vanilla DQN and various continual learning mitigations) perform similarly—hyperparameter search completely hides differences in algorithmic design.
- Under k%-percent tuning, continual learning mitigation strategies (W0-regularization, layer normalization) show a clear advantage, reinforcing that these methods indeed aid continual learning.
- The optimal $k$ value is agent-environment dependent: some combinations find good hyperparameters at $k=1\%$, while others require $k=10\%$.
- The metric used to select hyperparameters is also crucial: total return, tail average return, and TD error perform differently across different scenarios.
- Keeping hyperparameters fixed while extending the experiment is highly effective for distinguishing algorithms—true continual learning algorithms maintain performance over longer deployments, whereas algorithms overfitted to a specific lifetime degrade.

## Highlights & Insights

- The paper raises a critical but widely overlooked methodological issue in the research community—which may explain the "high volume of papers but slow progress" in the continual RL field.
- The analogy to "do not peek at the test set" is both intuitive and powerful, simplifying a complex methodological argument into one of the most fundamental principles of machine learning.
- The progressive experimental demonstration on Non-stationary Catch (Figures 1-3) is highly educational and clearly illustrates the core of the problem.
- Although simple, k%-percent tuning shifts how we evaluate continual learning algorithms—encouraging the development of genuinely adaptive algorithms rather than those relying on meticulous tuning.
- A profound insight: repeatedly running experiments in non-stationary environments inherently leaks information about the hidden dynamics to researchers, subtly reducing the partial observability of the benchmark.
- The paper suggests that the continual RL community needs a new set of "experimental protocols", analogous to the well-established train/test split convention in supervised learning.
- The precise characterization of the two pitfalls—"false negative" (falsely assuming an algorithm fails) and "false positive" (falsely assuming all algorithms succeed)—is highly instructive.

## Limitations & Future Work

- k%-percent tuning itself still requires choosing a $k$ value, and the optimal $k$ is task-dependent—this introduces a meta-hyperparameter issue, which, though better than lifetime tuning, is not perfect.
- The paper only validates its findings on relatively simple environments (Catch, Mountain Car, CartPole, Acrobot) without covering more complex benchmarks like Atari or MuJoCo, which limits its persuasiveness.
- No concrete algorithmic solution for automatic parameter tuning (meta-learning hyperparameters) is provided; it only points out the direction.
- k%-percent tuning assumes that early experience is representative—if non-stationarity only occurs late in the lifetime or undergoes qualitative domain shifts, the first $k\%$ of of data may be insufficient.
- As a position paper, it raises more questions than it solves and lacks a systematic solution.
- It does not deeply analyze the interactions between hyperparameters—certain hyperparameter combinations might perform well in the short term but become unsustainable in the long run.
- The discussion on which performance metrics to select during the tuning phase is not deep enough, with only three simple metrics being tested.

## Related Work & Insights

- Continual learning advancements such as network resizing/resetting in Nikishin et al. (2022), W0 regularization in Kumar et al. (2024), and layer normalization in Lyle et al. (2023) receive a fairer evaluation under the proposed framework—they clearly outperform vanilla DQN under k%-percent tuning.
- The work of Patterson et al. (2024) on the statistical handling of hyperparameters in RL is complementary to this paper, collectively pointing out systematic issues in current RL experimental methodologies.
- Formal definitions of continual RL in Abel et al. (2023) and Khetarpal et al. (2022) provide a theoretical basis for the problem formulation of this work.
- Implications for the continual learning community: future papers should explicitly declare what percentage of lifetime data was used for hyperparameter searches, and validate algorithmic stability over longer deployment periods.
- Implications for actual deployment: prioritize algorithms with fewer hyperparameters and those with adaptive mechanisms—such as automatic hyperparameter tuning via meta-learning or population-based training.
- The analogy to "cross-validation" in supervised learning is thought-provoking: continual RL requires a "train/test split" along the temporal dimension.

## Rating

⭐⭐⭐⭐ As a position paper, it introduces a highly significant and widely neglected methodological issue with clear and compelling lines of reasoning—the characterization of the two pitfalls is exceptionally precise. The progressive experiments on Non-stationary Catch (Fig 1$\rightarrow$2$\rightarrow$3), while simple, are highly persuasive and educational. The experiments showing that mitigation strategies indeed exhibit advantages under k%-percent tuning provide the strongest evidence. However, the proposed solution (k%-percent tuning) is somewhat simplistic and introduces a new meta-hyperparameter $k$; it also lacks validation on standard, large-scale benchmarks such as Atari or MuJoCo, nor does it provide automated tuning solutions. Nonetheless, it holds deep potential impact on the evaluation standards of the continual RL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Deployed Reinforcement Learning should be Continual](../../ICML2026/reinforcement_learning/position_deployed_reinforcement_learning_should_be_continual.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICML 2025\] Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](mitigating_plasticity_loss_in_continual_reinforcement_learning_by_reducing_churn.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
