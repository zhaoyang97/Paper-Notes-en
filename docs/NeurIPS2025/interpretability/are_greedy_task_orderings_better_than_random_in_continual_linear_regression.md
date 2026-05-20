---
title: >-
  [Paper Note] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?
description: >-
  [NeurIPS 2025][Interpretability][Continual Learning] This paper systematically analyzes the convergence differences between greedy task orderings (maximizing dissimilarity between consecutive tasks) and random orderings…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Continual Learning"
  - "Task Ordering"
  - "Linear Regression"
  - "Kaczmarz Method"
  - "Greedy Strategy"
date: 2026-05-08
content_hash: 3e82d94364aade98
---

# Are Greedy Task Orderings Better Than Random in Continual Linear Regression?

**Conference**: NeurIPS 2025
**arXiv**: [2510.19941](https://arxiv.org/abs/2510.19941)  
**Code**: None  
**Area**: Continual Learning / Optimization Theory
**Keywords**: Continual Learning, Task Ordering, Linear Regression, Kaczmarz Method, Greedy Strategy

## TL;DR

This paper systematically analyzes the convergence differences between greedy task orderings (maximizing dissimilarity between consecutive tasks) and random orderings in continual linear regression. It reveals that greedy orderings are competitive with random orderings in the full-rank setting, but single-pass greedy ordering can fail catastrophically in the general-rank setting, whereas greedy ordering with repetition achieves a convergence rate of $\mathcal{O}(1/\sqrt[3]{k})$.

## Background & Motivation

### Root Cause

**Background**: In Continual Learning, a model must learn multiple tasks sequentially, and the order in which tasks are presented significantly affects final performance. Prior work has focused primarily on random orderings — both with and without replacement — which have been shown to achieve an average loss convergence rate of $\mathcal{O}(1/\sqrt{k})$ after $k$ iterations. Greedy ordering, which selects the next task to be maximally dissimilar from the current one, is intuitively appealing but lacks rigorous theoretical analysis.

**Goal**: The core motivations are:

- **Is greedy ordering always superior to random ordering?** Intuition suggests that diverse task sequences should be more beneficial, yet theoretical guarantees are absent.
- **How does allowing task repetition affect greedy ordering?** This relates to data availability constraints in practical applications.
- **How can theoretical tools from the Kaczmarz literature transfer to continual learning analysis?** This represents a methodological contribution bridging two research areas.

## Method

### Overall Architecture

The paper formalizes continual linear regression as a variant of the Kaczmarz method. Given $T$ tasks, each task $t$ corresponds to a linear system $A_t x = b_t$, and the model performs gradient descent on each task sequentially according to a prescribed ordering.

**Dissimilarity Measure**: The dissimilarity between two consecutive tasks is defined as a function of the principal angles between their data subspaces. Greedy ordering selects, at each step, the task whose subspace has the largest principal angle with respect to the current task.

### Key Designs

**Formalization of Greedy Orderings**:
- **Single-pass Greedy**: Each task is used exactly once, ordered greedily by maximum dissimilarity.
- **Greedy with Repetition**: Tasks may be selected multiple times; at each step the task most dissimilar to the previous one is chosen greedily.

**Joint Realizability Assumption**: It is assumed that a global optimal solution $x^*$ exists that simultaneously satisfies all tasks, i.e., $A_t x^* = b_t, \forall t$. This is a standard simplifying assumption corresponding to the overparameterized setting in linear regression.

**Geometric Intuition**: The authors exploit the geometric relationships between subspaces (e.g., orthogonality, principal angles) to derive algebraic expressions characterizing greedy selection behavior, with the intuition that greedy ordering tends to alternate updates along orthogonal directions in parameter space.

### Loss & Training

The model uses mean squared error loss and performs one gradient projection step per task:

$$x_{k+1} = x_k - A_{i_k}^\dagger (A_{i_k} x_k - b_{i_k})$$

where $i_k$ is the index of the task selected at step $k$, and $A_{i_k}^\dagger$ denotes the pseudoinverse.

## Key Experimental Results

### Main Results

**Synthetic Data Experiments**: Different ordering strategies are compared on randomly generated linear regression tasks.

| Ordering Strategy | Avg. Loss @ 50 Iters | Avg. Loss @ 100 Iters | Convergence Trend |
|---|---|---|---|
| Random (with replacement) | ~0.15 | ~0.08 | $\mathcal{O}(1/\sqrt{k})$ |
| Random (without replacement) | ~0.12 | ~0.06 | $\mathcal{O}(1/\sqrt{k})$ |
| Greedy (with repetition) | ~0.08 | ~0.04 | Empirically faster |
| Greedy (single-pass) | Diverges/unstable | — | Catastrophic failure possible |

**CIFAR-100 Linear Probe Experiments**: Linear classification using pretrained features, validating greedy ordering on practical tasks.

| Ordering Strategy | Avg. Acc. @ 10 Rounds | Avg. Acc. @ 50 Rounds | Final Acc. |
|---|---|---|---|
| Random | 45.2% | 62.1% | 68.3% |
| Greedy (with repetition) | 52.8% | 67.5% | 72.1% |
| Greedy (single-pass) | 38.6% | Unstable | High variance |

### Ablation Study

- **Effect of Rank**: In the full-rank setting (all task data matrices are full rank), the theoretical upper bounds for greedy and random orderings coincide; differences emerge in the low-rank setting.
- **Effect of Task Count**: The advantage of greedy ordering becomes more pronounced as the number of tasks increases, but the risk of single-pass failure also grows.
- **Effect of Repetition Count**: Allowing each task to repeat 2–3 times is sufficient to recover most of the performance benefit.

### Key Findings

1. **Full-rank Setting**: The loss upper bound for greedy ordering is analogous to that of random ordering, both at $\mathcal{O}(1/\sqrt{k})$.
2. **General-rank Setting**: Single-pass greedy can fail catastrophically (loss does not converge), whereas greedy with repetition achieves a convergence rate of $\mathcal{O}(1/\sqrt[3]{k})$.
3. **Empirical Advantage**: Despite looser theoretical bounds, greedy ordering consistently outperforms random ordering in experiments.

## Highlights & Insights

- **Methodological Innovation**: Tools from the Kaczmarz literature are systematically brought to bear on continual learning theory, establishing a bridge between the two fields.
- **Counterintuitive Finding**: Single-pass greedy ordering can perform worse than random ordering, revealing that "maximum diversity" is not always beneficial.
- **Practical Guidance**: Allowing task repetition is a critical condition for greedy ordering to succeed, with direct implications for curriculum learning and data replay strategies.
- **Theory–Practice Gap**: The theoretical upper bound $\mathcal{O}(1/\sqrt[3]{k})$ is slower than the random bound $\mathcal{O}(1/\sqrt{k})$, yet greedy ordering converges faster empirically, suggesting the bound may not be tight.

## Limitations & Future Work

1. **Restricted to Linear Regression**: Whether the conclusions generalize to nonlinear models (e.g., neural networks) remains an open question.
2. **Joint Realizability Assumption Is Strong**: In practical continual learning, conflicting objectives across tasks are common.
3. **Theoretical Bounds May Not Be Tight**: Empirical performance substantially exceeds theoretical predictions, suggesting room for improvement in the analysis techniques.
4. **Computational Cost of Greedy Ordering**: Computing pairwise dissimilarities across all tasks incurs significant overhead when the number of tasks is large.
5. **Online Task Arrival Not Considered**: In practice, tasks may arrive in a streaming fashion, precluding global greedy selection.

## Related Work & Insights

- **Kaczmarz Method**: A classical iterative solver for linear systems; its randomized variant (Strohmer & Vershynin, 2009) provides an important theoretical foundation for this work.
- **Curriculum Design in Continual Learning**: Selecting task orderings is a central challenge in continual learning; this paper contributes a rigorous theoretical perspective.
- **Catastrophic Forgetting**: Greedy ordering indirectly mitigates forgetting by maximizing task diversity, but the single-pass failure result highlights that diversity must be coupled with a repetition mechanism.

## Rating

- Theoretical Depth: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Novelty: ⭐⭐⭐⭐
- Value: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](towards_scaling_laws_for_symbolic_regression.md)
- [\[AAAI 2026\] SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](../../AAAI2026/interpretability/som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)
- [\[NeurIPS 2025\] Better Estimation of the Kullback-Leibler Divergence Between Language Models](better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)
- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)

</div>

<!-- RELATED:END -->
