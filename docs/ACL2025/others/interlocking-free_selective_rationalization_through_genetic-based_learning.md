---
title: >-
  [Paper Note] Interlocking-free Selective Rationalization Through Genetic-based Learning
description: >-
  [ACL 2025][Selective rationalization] This paper proposes GenSPP, the first selective rationalization framework that completely eliminates the interlocking problem. By using genetic algorithms to separately optimize the generator and predictor, it significantly improves rationale quality (Hl-F1 increased by 6.5%–10.3%) on synthetic datasets and hate speech detection tasks, while maintaining comparable classification performance.
tags:
  - "ACL 2025"
  - "Selective rationalization"
  - "genetic algorithms"
  - "interlocking problem"
  - "self-interpreting models"
  - "rationale extraction"
date: 2026-05-08
content_hash: 5e528611a72a6c1d
---

# Interlocking-free Selective Rationalization Through Genetic-based Learning

**Conference**: ACL 2025  
**arXiv**: [2412.10312](https://arxiv.org/abs/2412.10312)  
**Code**: Yes (the paper mentions open-source data and code)  
**Area**: Others  
**Keywords**: Selective rationalization, genetic algorithms, interlocking problem, self-interpreting models, rationale extraction

## TL;DR

This paper proposes GenSPP, the first selective rationalization framework that completely eliminates the interlocking problem. By using genetic algorithms to separately optimize the generator and predictor, it significantly improves rationale quality (Hl-F1 increased by 6.5%–10.3%) on synthetic datasets and hate speech detection tasks, while maintaining comparable classification performance.

## Background & Motivation

**Background**: Selective rationalization is a popular paradigm in explainable AI. Its core architecture is the select-then-predict (SPP) pipeline, where a generator extracts highlights (rationales) from the input text, which are then fed into a predictor for classification. Such models naturally possess the property of faithful self-explanation and are widely applied in high-risk scenarios like fact-checking and legal analysis.

**Limitations of Prior Work**: The SPP architecture suffers from a severe optimization issue known as interlocking: because the generator produces discrete binary masks while the predictor learns through continuous gradient updates, their update rates mismatch. When the generator is stuck with a sub-optimal mask, the predictor overfits to this mask, which in turn reinforces the generator to maintain this selection, forming a vicious cycle. Existing methods (such as Gumbel-Softmax sampling, weight sharing, and soft rationalization guidance) can only mitigate but not eliminate this issue, and introduce extra hyperparameter tuning burdens.

**Key Challenge**: The root cause of interlocking lies in the piece-wise constant policy characteristics caused by the generator's rounding function in SGD joint optimization. Even with smoothing techniques, the generated masks can remain unchanged for several gradient steps, leading to predictor overfitting.

**Goal**: To design a completely interlocking-free selective rationalization framework without requiring additional heuristics, sampling tricks, or architectural modifications.

**Key Insight**: The authors observe that interlocking is essentially caused by the coupling of the two modules in the joint optimization problem $\min_\theta \min_\omega \mathcal{L}$. If the double minimization is decoupled into disjoint optimization—where the generator is defined independently first, and then the predictor is trained from scratch—interlocking can be structurally broken.

**Core Idea**: To replace SGD joint training with the global search of Genetic Algorithms (GAs). Each individual represents a generator parameter configuration, and a predictor is trained independently from scratch for each individual to evaluate its fitness, thereby achieving completely disjoint optimization.

## Method

### Overall Architecture

GenSPP adopts the classical SPP architecture (generator $g_\theta$ + predictor $f_\omega$), but its training method is completely different. First, it initializes a population $\mathcal{P}$ containing $I$ individuals, where each individual represents a set of generator parameters. In each generation: (1) for each individual, a predictor is trained from scratch to minimize the classification loss (while keeping the generator frozen); (2) each individual is evaluated using a fitness function $h$; (3) the next generation is generated through selection, crossover, and mutation. After iterating for $G$ generations, the optimal individual is output.

### Key Designs

1. **Disjoint Training**:

    - Function: Structurally eliminates interlocking
    - Mechanism: Reformulates the joint optimization problem as a constrained optimization $\min_\theta \Omega(m)$ s.t. $\min_\omega \mathcal{L}(f_\omega(g_\theta(x) \odot x), y) \leq l + \epsilon$, where $l$ is the loss of the optimal predictor trained on the full input. This means finding the regularization-optimal highlight such that the predictor can achieve performance close to that on the original input. The dependency between the generator and the predictor is unidirectional: $f_\omega$ relies on $g_\theta$, but not vice versa.
    - Design Motivation: The unequal learning rates (continuous vs. discrete) of the two modules in SGD joint training are the root cause of interlocking. Disjoint optimization fundamentally severs this coupling.

2. **Non-differentiable Fitness Function Design**:

    - Function: Simultaneously evaluates classification performance and highlight quality without the need for weight balancing
    - Mechanism: Defines $\tilde{h} = 1 - \mathcal{L}$ (when $\mathcal{L}_t < l + \epsilon$), where $\mathcal{L} = (1 - \Omega(m)) \times (1 - \min(\mathcal{L}_t, 1))$. When the classification loss does not meet the threshold, it is directly set to 0, which makes the learning process first focus on classification performance and then progressively optimize highlight quality. The final fitness is $h = 1/(\tilde{h} + \hat{\epsilon})$.
    - Design Motivation: Traditional weighting $\mathcal{L}_t + \Omega(m)$ maps $(0.0, 1.0)$ and $(0.5, 0.5)$ to the same cost, even though the former has perfect classification with terrible highlights, while the latter is average in both aspects—they should be treated differently. The non-differentiable fitness function addresses this issue.

3. **Genetic Search Strategy (GA Operations)**:

    - Function: Realizes local and global search in the parameter space
    - Mechanism: Uses roulette-wheel selection to pair individuals, one-point crossover to generate new individuals, Gaussian noise mutation for local exploration, and half-elitism survival selection to retain the best individuals. The population size is $I=50$, evolutionary generations $G=100$.
    - Design Motivation: GA's population search naturally reduces the risk of falling into local optima, while crossover provides global exploration, and mutation provides local exploration. Moreover, GA does not require gradient computation, avoiding the variance issues introduced by sampling.

### Loss & Training

When evaluating each individual, the predictor is trained using classification cross-entropy $\mathcal{L}_{ce}$ for 3 epochs with a learning rate of $10^{-2}$. Highlight regularization employs a sparsity constraint $\mathcal{L}_s$ and (optionally) a continuity constraint $\mathcal{L}_c$. Fitness evaluation is conducted on the validation set.

## Key Experimental Results

### Main Results

Comparison on the synthetic Toy dataset and the real HateXplain hate speech dataset:

| Method | Toy Clf-F1 | Toy Hl-F1 | HateXplain Clf-F1 | HateXplain Hl-F1 |
|------|-----------|-----------|-------------------|-----------------|
| FR | Baseline | Baseline | Baseline | Baseline |
| MGR | Multi-generator | Higher variance | Comparable | Moderate |
| MCD | Causal guidance | Moderate | Comparable | Moderate |
| G-RAT | Attention guidance | Higher | Comparable | Higher |
| **GenSPP** | **Comparable** | **+10.3%** | **Comparable** | **+6.5%** |

GenSPP significantly outperforms all competing methods in highlight quality (Wilcoxon test $p \leq 0.01$) while achieving comparable classification performance.

### Ablation Study (Synthetic Skewing Recovery Experiment)

| Configuration | Toy Hl-F1 | HateXplain Hl-F1 | Description |
|------|-----------|-------------------|------|
| GenSPP (G=100) | High | High | Standard configuration |
| GenSPP (G=150) | Optimal | Optimal | Increased budget, fully recovered |
| GenSPP_sk (skewed initialization) | Close to standard | Close to standard | Recovered from skewed state |
| Baseline models (skew) | High variance | Unstable | Most seeds fail to recover |

### Key Findings

- The variance of GenSPP is significantly lower than all baselines, demonstrating the robustness of genetic search. On the Toy dataset, MGR and G-RAT exhibit significant instability.
- On HateXplain, GenSPP learns to select no highlights for negative instances (normal text) while retaining valuable choices for positive instances—this flexibility is unattainable by baseline models.
- GenSPP is the smallest model (same size as FR, 2-4 times smaller than other baselines), yet it performs the best.
- Computational cost: A single seed run of GenSPP takes about 36 minutes on Toy and about 78 minutes on HateXplain, which is much higher than the baselines (8/4 minutes). However, the low variance implies that multiple runs are not required.

## Highlights & Insights

- **Curing interlocking from an optimization perspective**: Instead of patching symptoms (sampling, guidance, regularization), it structurally eliminates coupling from the optimization process, which is a very clean approach. This "optimizer swapping" mindset can be transferred to other two-component systems with discrete-continuous coupling.
- **Ingenious design of the non-differentiable fitness function**: Using a threshold mechanism to ensure classification meets the standard first, and then optimizing highlights, avoids the hyperparameter tuning issues involved in traditional multi-objective weighting.
- **Design of the synthetic dataset**: Constructing a controllable string-matching task to evaluate the rationalization framework, where each category has a unique highlight pattern accompanied by distractor segments. This is the first evaluation benchmark of its kind in the community.

## Limitations & Future Work

- **Computational Overhead**: Genetic search requires training a predictor for each individual in every generation, which incurs a computational workload about 10-20 times higher than SGD-based methods. The authors point out that this can be mitigated through parallel evaluation and more efficient GAs (e.g., CMA-ES).
- **Limited Model Scale**: Experiments are only verified on lightweight models at the GRU level and are not extended to large models such as Transformers. The parameter space of genetic search grows exponentially with model size.
- **Small Dataset Sizes**: Toy has only 10k samples and HateXplain has about 20k, lacking validation on large-scale NLP tasks.
- **Focusing Only on Unsupervised Rationalization**: The effectiveness under supervised rationalization settings has not been explored.

## Related Work & Insights

- **vs FR (Liu et al., 2022)**: FR uses Gumbel-Softmax + weight sharing to mitigate interlocking, but it remains joint SGD optimization in essence, where interlocking is only alleviated. GenSPP's disjoint training structurally eliminates the problem.
- **vs G-RAT (Hu & Yu, 2024)**: G-RAT introduces an additional attention guidance module to provide soft rationalization signals. Although effective, it increases model complexity (with parameters 4 times larger than GenSPP). GenSPP does not require any extra modules.
- **vs Li et al., 2022 (3-stage)**: The three-stage method attempts to break interlocking through iterative freezing, but interlocking still exists in the first stage. GenSPP completely avoids joint training.

## Rating

- Novelty: ⭐⭐⭐⭐ Replacing SGD joint training with genetic algorithms to solve interlocking is a unique idea, though the genetic algorithm itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes synthetic datasets, real datasets, skew recovery experiments, and comparison with multiple baselines, though the datasets and model scales are limited.
- Writing Quality: ⭐⭐⭐⭐ The derivation of motivation is clear, the methodology is described in detail, and the mathematical formulation is comprehensive.
- Value: ⭐⭐⭐ Possesses theoretical contribution, but its practicality is limited by computational overhead and scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[NeurIPS 2025\] What Does It Take to Build a Performant Selective Classifier?](../../NeurIPS2025/others/what_does_it_take_to_build_a_performant_selective_classifier.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)

</div>

<!-- RELATED:END -->
