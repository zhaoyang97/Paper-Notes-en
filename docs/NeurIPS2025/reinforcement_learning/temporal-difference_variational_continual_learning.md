---
title: >-
  [Paper Note] Temporal-Difference Variational Continual Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Continual Learning] This paper proposes the TD-VCL objective, which reformulates the learning target in Variational Continual Learning (VCL) as a weighted combination of multiple past posterior estimates. This reformulation reveals a deep connection to temporal-difference (TD) methods in reinforcement learning, and effectively mitigates the progressive accumulation of approximation errors by "spreading" regularization pressure across mul…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Continual Learning"
  - "Variational Inference"
  - "Temporal Difference"
  - "Catastrophic Forgetting"
  - "Bayesian Learning"
date: 2026-05-08
content_hash: 1618fc58b18fc4ac
---

# Temporal-Difference Variational Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2410.07812](https://arxiv.org/abs/2410.07812)  
**Code**: [https://github.com/luckeciano/TD-VCL](https://github.com/luckeciano/TD-VCL)  
**Area**: Reinforcement Learning / Continual Learning
**Keywords**: Continual Learning, Variational Inference, Temporal Difference, Catastrophic Forgetting, Bayesian Learning

## TL;DR

This paper proposes the TD-VCL objective, which reformulates the learning target in Variational Continual Learning (VCL) as a weighted combination of multiple past posterior estimates. This reformulation reveals a deep connection to temporal-difference (TD) methods in reinforcement learning, and effectively mitigates the progressive accumulation of approximation errors by "spreading" regularization pressure across multiple historical posteriors.

## Background & Motivation

Continual learning (CL) requires a model to sequentially learn new tasks from a non-stationary data stream while retaining performance on previous tasks. The central challenge lies in balancing plasticity (learning new tasks) and memory stability (retaining old ones); an imbalance leads to catastrophic forgetting.

Within the Bayesian CL framework, Variational Continual Learning (VCL) exploits the recursive relationship of the posterior: $p(\boldsymbol{\theta}|\mathcal{D}_{1:T}) \propto p(\boldsymbol{\theta}|\mathcal{D}_{1:T-1})p(\mathcal{D}_T|\boldsymbol{\theta})$, and incrementally updates the posterior via online variational inference. The VCL objective maximizes the current task likelihood while constraining the new posterior to remain close to the previous one:

$$\mathcal{L}_{VCL}^t = \mathbb{E}[\log p(\mathcal{D}_t|\boldsymbol{\theta})] - D_{KL}(q_t(\boldsymbol{\theta}) \| q_{t-1}(\boldsymbol{\theta}))$$

The **core problem** is that VCL relies solely on the **most recent posterior estimate** $q_{t-1}$ as the regularization target. If the posterior estimate at a given step is of particularly poor quality, its error propagates entirely to the next step and accumulates progressively through recursive updates (compounding error), causing the current estimate to deviate severely from the true posterior.

The central insight of this paper is that **the same optimization objective can be equivalently expressed as a function of multiple past posterior estimates and task likelihoods**. By distributing regularization across multiple historical posteriors, the influence of any single erroneous posterior estimate is diluted, while accurate estimates can exert a corrective effect.

## Method

### Overall Architecture

Starting from the standard KL-minimization objective of VCL, the paper derives two equivalent but practically superior objective functions: $n$-Step KL VCL and TD($\lambda$)-VCL. Both are essentially combinations of multiple $n$-step TD targets and exhibit a profound structural correspondence to TD learning in reinforcement learning.

### Key Designs

1. **$n$-Step KL Regularization Objective (Proposition 4.1)**: The standard VCL objective is equivalently rewritten as:

    $q_t = \arg\max_q \mathbb{E}\left[\sum_{i=0}^{n-1}\frac{n-i}{n}\log p(\mathcal{D}_{t-i}|\boldsymbol{\theta})\right] - \sum_{i=0}^{n-1}\frac{1}{n}D_{KL}(q_t \| q_{t-i-1})$

   The KL regularization term is uniformly distributed across $n$ past posteriors. If a particular $q_{t-i}$ is highly biased, it affects only $1/n$ of the regularization. The likelihood term covers multiple tasks and is weighted by temporal recency, from which **data replay emerges naturally from the objective function**.

2. **TD($\lambda$)-VCL Objective (Proposition 4.2)**: Building on the $n$-Step KL formulation, geometrically decaying weights $\lambda^i$ are introduced so that more recent posterior estimates receive greater weight:

    $\text{KL weight} \propto \frac{\lambda^i(1-\lambda)}{1-\lambda^n}, \quad \text{likelihood weight} \propto \frac{\lambda^i(1-\lambda^{n-i})}{1-\lambda^n}$

   This provides finer-grained control: $\lambda \to 0$ reduces to VCL, and $\lambda = 1$ reduces to $n$-Step KL. The objective is equivalent to a weighted sum of multiple TD targets (Proposition 4.4), establishing a precise correspondence with the $\lambda$-return in reinforcement learning.

3. **$n$-Step TD Objective (Definition 4.3)**: The TD target in CL is defined as $\text{TD}_t(n) = \mathbb{E}[\sum_{i=0}^{n-1}\log p(\mathcal{D}_{t-i}|\boldsymbol{\theta})] - D_{KL}(q_t \| q_{t-n})$, comparing against a more temporally distant posterior. Each TD target is equivalent to the standard VCL objective under exact inference, but offers a different bias–variance trade-off under approximate inference.

### Loss & Training

A mean-field Gaussian approximate posterior and a Gaussian prior $\mathcal{N}(0, \sigma^2\mathbf{I})$ are used. The KL term is computed analytically, and the expected log-likelihood is approximated via Monte Carlo estimation with the reparameterization trick. Likelihood tempering is applied to prevent variational over-pruning. At test time, the posterior predictive distribution is computed by marginalizing over the approximate posterior via MC sampling.

## Key Experimental Results

### Main Results

| Benchmark (t = final) | Metric | TD($\lambda$)-VCL | VCL | VCL CoreSet | Gain |
|---|---|---|---|---|---|
| PermutedMNIST-Hard ($t$=10) | Avg Acc | **0.89** | 0.78 | 0.81 | +0.11 |
| SplitMNIST-Hard ($t$=5) | Avg Acc | **0.66** | 0.64 | 0.62 | +0.02 |
| SplitNotMNIST-Hard ($t$=5) | Avg Acc | **0.58** | 0.51 | 0.51 | +0.07 |
| CIFAR100-10 ($t$=10) | Avg Acc | **0.71** | 0.66 | 0.65 | +0.05 |
| TinyImageNet-10 ($t$=10) | Avg Acc | **0.56** | 0.51 | 0.54 | +0.05 |

### Ablation Study (TD Objective Applied to Other Bayesian CL Methods)

| Method | PermutedMNIST ($t$=10) | SplitMNIST ($t$=5) | Note |
|---|---|---|---|
| VCL | 0.78 | 0.64 | Baseline |
| TD($\lambda$)-VCL | **0.89** | **0.67** | +0.11 / +0.03 |
| UCL | 0.73 | 0.66 | Baseline |
| TD($\lambda$)-UCL | **0.84** | **0.70** | +0.11 / +0.04 |
| UCB | 0.77 | 0.66 | Baseline |
| TD($\lambda$)-UCB | **0.85** | **0.69** | +0.08 / +0.03 |

### Key Findings

- TD-VCL consistently outperforms standard VCL across all benchmarks, with the advantage becoming more pronounced as the number of tasks increases.
- Per-task analysis (Figure 3) reveals that catastrophic forgetting disproportionately affects earlier tasks; TD-VCL is substantially more robust in this regard — Task 1 retains approximately 80–85% accuracy after 10 tasks, compared to 50–60% for VCL.
- The TD objective is equally effective when applied to UCL and UCB, demonstrating its generality — it is orthogonal and complementary to different variational methods.
- On SplitNotMNIST-Hard, TD-VCL is the only method that maintains non-trivial accuracy after all tasks.

## Highlights & Insights

- The theoretical contribution is elegant: the paper proves an equivalent reformulation of the VCL objective and its connection to TD learning through three progressively structured propositions.
- The insight that "data replay emerges naturally from the objective function" is particularly compelling — the re-evaluation of past task likelihoods is an inherent component of the objective rather than a heuristic addition.
- TD-VCL spans a continuous spectrum from VCL ($\lambda \to 0$) to $n$-Step KL ($\lambda = 1$), providing a flexible bias–variance trade-off mechanism.
- The connection to TD learning in neuroscience provides additional motivation and interpretability for the proposed approach.

## Limitations & Future Work

- Storing multiple past posterior estimates incurs memory overhead that grows with $n$.
- Old task data must be stored or replayed to estimate likelihood terms, though this requirement arises naturally from the objective rather than being imposed heuristically.
- Experiments are conducted primarily on relatively small-scale tasks and networks; the behavior of the approach on large-scale pretrained models remains to be validated.
- Theoretical guidance for selecting the optimal hyperparameters $n$ and $\lambda$ is currently lacking.

## Related Work & Insights

- **vs. VCL (Nguyen et al.)**: VCL regularizes only with the most recent posterior, whereas TD-VCL employs multi-step posteriors, and the replay mechanism emerges from the objective function naturally rather than being added heuristically.
- **vs. UCL/UCB**: These methods improve regularization and adaptive learning rate mechanisms; the TD objective is orthogonal and complementary to them.
- **vs. EWC**: EWC applies Fisher information regularization and is a non-Bayesian approach; TD-VCL provides a more principled solution within the Bayesian framework.
- **vs. TD Learning (RL)**: The paper reveals a structural correspondence between VCL posterior updates and value function updates in RL; the $\lambda$-return plays the same role in both domains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The insight connecting VCL to TD learning is highly original, and the theoretical derivations are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, three base methods, and per-task analysis are provided, though large-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, theory and intuition are well balanced, and figures are clean and informative.
- Value: ⭐⭐⭐⭐ The work makes an important contribution to Bayesian CL, and the cross-domain insight connecting to TD learning is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](../../AAAI2026/reinforcement_learning/language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](../../ICML2025/reinforcement_learning/continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)

</div>

<!-- RELATED:END -->
