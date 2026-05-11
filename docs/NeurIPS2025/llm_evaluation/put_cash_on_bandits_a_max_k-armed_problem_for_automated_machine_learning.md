---
title: >-
  [Paper Note] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning
description: >-
  [NeurIPS 2025][LLM Evaluation][CASH] This paper addresses the Combined Algorithm Selection and Hyperparameter Optimization (CASH) problem in AutoML. Through data-driven analysis…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "CASH"
  - "Max K-Armed Bandit"
  - "Hyperparameter Optimization"
  - "AutoML"
  - "Upper Confidence Bound"
date: 2026-05-08
content_hash: a97b5a569e576d2a
---

# Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.05226](https://arxiv.org/abs/2505.05226)
**Code**: [GitHub](https://github.com/amirbalef/CASH_with_Bandits)
**Area**: AutoML / Multi-Armed Bandits
**Keywords**: CASH, Max K-Armed Bandit, Hyperparameter Optimization, AutoML, Upper Confidence Bound

## TL;DR

This paper addresses the Combined Algorithm Selection and Hyperparameter Optimization (CASH) problem in AutoML. Through data-driven analysis, it reveals that HPO reward distributions are bounded and left-skewed, and proposes MaxUCB—a bandit algorithm specifically tailored to this distributional property—achieving both theoretical and empirical improvements over existing methods.

## Background & Motivation

CASH (Combined Algorithm Selection and Hyperparameter Optimization) is a core problem in AutoML: given a limited budget, simultaneously search for the best model class (e.g., LightGBM, XGBoost, MLP) and its optimal hyperparameter configuration.

**Two existing paradigms**:

**Combined search**: Merges the hyperparameter spaces of all models into a single hierarchical space and applies Bayesian Optimization (BO) directly. Limitation: high-dimensional conditional spaces lead to inefficient HPO.

**Two-level CASH**: Uses a bandit at the upper level to select a model class, and runs HPO on the selected model at the lower level. Limitation: the objective function and distributional assumptions for the upper-level bandit remain unclear.

**Key Challenge**: Existing Max K-Armed Bandit (MKB) algorithms assume heavy-tailed reward distributions (e.g., Pareto), whereas HPO reward distributions in CASH are in fact **bounded, left-skewed, and light-tailed**. This distributional mismatch leads to poor performance of existing extreme bandit algorithms on CASH.

**Key Insight**: The paper first conducts a systematic empirical analysis to characterize the true properties of HPO reward distributions, then designs a new bandit algorithm grounded in the correct distributional assumptions, with proven regret upper bounds.

## Method

### Overall Architecture

MaxUCB is a two-level optimization framework: the upper level is a Max K-Armed Bandit (selecting the model class on which to run HPO), and the lower level is a standard HPO method (e.g., Bayesian Optimization). At each round, the upper level selects a model class, the lower level evaluates one new configuration via an HPO iteration, and the negative loss is returned as the bandit reward signal.

### Key Designs

1. **Data-Driven Distributional Analysis (Section 3)**: The authors analyze the survival functions of HPO reward distributions across four AutoML benchmarks and identify three key properties:

   - **Boundedness**: Rewards (model performance metrics) are inherently bounded, with each arm having a distinct maximum value.
   - **Light-tailed and left-skewed**: Rewards are concentrated near the maximum, and extreme events are not outliers—contrary to the heavy-tailed assumption commonly used in MKB literature.
   - **Approximate stationarity**: The optimal arm does not change over time.

   Based on these observations, Lemma 3.3 introduces two distribution-dependent constants $L$ and $U$ to characterize the shape of a bounded distribution near its maximum: $L\epsilon \leq G(b-\epsilon) \leq U\epsilon$. Empirical evidence shows that $L$ is typically greater than 1 (whereas heavy-tailed distributions yield $L \approx 0$), resolving the theoretical barrier identified in prior work that deemed extreme bandits infeasible.

2. **MaxUCB Algorithm (Algorithm 1)**: The core contribution lies in the design of the exploration bonus. While classical UCB uses $\sqrt{\frac{\alpha\log t}{n}}$, MaxUCB replaces it with $\left(\frac{\alpha\log t}{n}\right)^2$.

   The index update rule is:
   $U_i = \max(r_{i,1}, \ldots, r_{i,n_i}) + \left(\frac{\alpha\log(t)}{n_i}\right)^2$

   This faster concentration rate stems from the special concentration properties of the maximum in bounded distributions. The probability of a bad event can be written as $P(\text{Bad}) \leq O(e^{-n\sqrt{C(n)}} + nC(n))$, which is minimized by setting $C(n) = 1/n^2$.

3. **Theoretical Analysis (Theorem 4.2)**: For any suboptimal arm $i$, the number of suboptimal pulls under MaxUCB satisfies:

   $N_i(T) \leq \frac{T^{1-2L_{i^*}\alpha\sqrt{\Delta_i}}}{1-2L_{i^*}\alpha\sqrt{\Delta_i}} + 2\alpha\sqrt{U_i T}\log(T)$

   Performance is governed by the suboptimality gap $\Delta_i$, the left-tail parameter $L_{i^*}$ of the optimal arm, and the right-tail parameter $U_i$ of the suboptimal arm. When $L_{i^*}$ is large (extreme values of the optimal arm are easily sampled) and $\Delta_i$ is large, the number of suboptimal pulls decays rapidly.

   **Corollary 4.3** (Total Regret Bound): With an appropriate choice of $\alpha$, $R(T) \leq \mathcal{O}\left(\frac{K\log T}{\sqrt{T}}\max_{i} b_i\right)$.

### Role of Hyperparameter $\alpha$

$\alpha$ controls the exploration–exploitation trade-off. Equation 9 provides theoretical guidance: $\alpha = \frac{1}{4L_{i^*}\sqrt{\min\Delta_i}}\left(1 - \frac{2\log(\log T)}{\log T}\right)$. Smaller $L_{i^*}$ or smaller $\Delta_i$ requires larger $\alpha$ to promote exploration. Experiments show that $\alpha \approx 0.5$ is a robust default value.

## Key Experimental Results

### Main Results: Sign Test Across Four AutoML Benchmarks

| Benchmark (HPO Method) | MaxUCB p-value | Quantile BayesUCB p-value | Rising Bandits p-value | MaxUCB w/t/l |
|---|---|---|---|---|
| TabRepo [RS] | **0.00000** | **0.00000** | **0.00000** | 186/4/10 |
| TabRepoRaw [SMAC] | **0.00072** | **0.00261** | 0.42777 | 24/0/6 |
| YaHPOGym [SMAC] | **0.00880** | **0.00503** | **0.00074** | 64/0/39 |
| Reshuffling [HEBO] | — | — | — | Significantly outperforms combined search |

MaxUCB and Quantile Bayes UCB are the only two methods that statistically significantly outperform combined search.

### Ablation Study: Sensitivity to $\alpha$

| $\alpha$ Range | Characteristics | Remarks |
|---|---|---|
| Low (~0.1) | Better performance under small budgets | Quickly commits to promising arms, but may exploit prematurely |
| High (~2.0) | Better performance under large budgets | More thorough exploration, higher final performance |
| Intermediate (~0.5) | Robust anytime performance | Performs well across all budget levels |

### Key Findings

- **Two-level vs. combined search**: Nearly all bandit methods outperform combined search at early stages ($T=50$), confirming that decomposition strategies are particularly effective under limited budgets.
- **MaxUCB vs. other bandits**: MaxUCB achieves the best anytime and final performance. Classical UCB and QoMax-SDA perform worst due to distributional mismatch.
- **Rising Bandits** are competitive on YaHPOGym, where non-stationarity is present, as they explicitly model non-stationary rewards.
- Extreme bandits and classical UCB operate under assumptions that are either too loose or mismatched, resulting in low efficiency on CASH.

## Highlights & Insights

- **Data-driven methodology**: Analyzing true distributional properties before algorithm design avoids the pitfalls of assumption mismatch.
- **Resolves an open problem**: Nishihara et al. (2016) argued that extreme bandits are infeasible under certain distributions; this paper demonstrates that those negative results do not apply to the actual distributions encountered in CASH.
- The squared exploration bonus $(\cdot)^2$ instead of $\sqrt{\cdot}$ is a concise yet profound finding.
- The constants $L$ and $U$ provide a unified framework for characterizing the behavior of different distribution families near their extremes.

## Limitations & Future Work

- The method is highly tailored to bounded left-skewed distributions and may degrade under heavy-tailed or symmetric distributions.
- The stationarity assumption may not hold during the early phase of HPO, necessitating a burn-in period.
- Adaptive tuning of $\alpha$ is theoretically infeasible (Locatelli & Carpentier, 2018), though meta-learning information could be leveraged.
- Lower bounds for bounded extreme bandits remain an open problem; distributional optimality has not been established.
- Computation-cost-aware resource allocation (accounting for varying training times across model classes) is not considered.

## Related Work & Insights

- **Relation to Quantile Bayes UCB**: Both target the upper region of the distribution, but MaxUCB directly optimizes the maximum rather than a quantile.
- The approach can be extended to sub-supernet selection in NAS, as NAS reward distributions exhibit similar properties.
- AutoML systems (e.g., Auto-sklearn, Auto-WEKA) can directly integrate MaxUCB as the upper-level model selection strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ The data-driven distributional analysis establishes the correct application of MKB to CASH.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks, multiple baselines, sensitivity analysis, and statistical tests are all included.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from data analysis → theory → algorithm → experiments is exceptionally clear.
- Value: ⭐⭐⭐⭐ Directly actionable for AutoML practitioners; the theoretical analysis also has independent merit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](learning_generalizable_shape_completion_with_sim3_equivariance.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)

</div>

<!-- RELATED:END -->
