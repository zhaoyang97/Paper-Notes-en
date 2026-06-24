---
title: >-
  [Paper Note] Fair Model-Based Clustering
description: >-
  [AAAI 2026 Oral][AI Safety][Fair Clustering] This paper proposes FMC, a fair clustering algorithm based on finite mixture models. By exerting fairness constraints on model parameters (instead of sample-level assignments), it achieves scalable fair clustering where the number of parameters is independent of the sample size. FMC supports mini-batch learning and categorical data, significantly outperforming existing methods on large-scale datasets.
tags:
  - "AAAI 2026 Oral"
  - "AI Safety"
  - "Fair Clustering"
  - "Finite Mixture Models"
  - "EM Algorithm"
  - "Mini-batch Learning"
  - "Scalability"
date: 2026-05-08
content_hash: 8e0f138e38438888
---

# Fair Model-Based Clustering

**Conference**: AAAI 2026 Oral  
**arXiv**: [2602.21509](https://arxiv.org/abs/2602.21509)  
**Code**: None  
**Area**: AI Safety/Fairness  
**Keywords**: Fair Clustering, Finite Mixture Models, EM Algorithm, Mini-batch Learning, Scalability

## TL;DR
This paper proposes FMC, a fair clustering algorithm based on finite mixture models. By exerting fairness constraints on model parameters (instead of sample-level assignments), it achieves scalable fair clustering where the number of parameters is independent of the sample size. FMC supports mini-batch learning and categorical data, significantly outperforming existing methods on large-scale datasets.

## Background & Motivation

**Background**: Fair clustering requires the proportion of sensitive attributes (e.g., gender, race) in each cluster to be similar to that of the entire dataset. Existing methods are mostly based on K-means clustering, optimizing cluster centers and assignment mappings simultaneously under fairness constraints.

**Limitations of Prior Work**:
   - Cluster assignment for each sample must be optimized simultaneously with cluster centers $\rightarrow$ the number of learnable parameters is proportional to the dataset size $N$ $\rightarrow$ hard to scale to large-scale data.
   - Fairness constraints rely on the assignments of the complete dataset $\rightarrow$ mini-batch learning becomes infeasible (traditional speed-up methods fail).
   - K-means-based methods require a metric space $\rightarrow$ unable to handle non-metric data such as categorical data.
   - After training, the assignment mapping is only defined on the training data $\rightarrow$ hard to perform fair assignment on new data.

**Key Challenge**: How to decouple the computational complexity of fair clustering algorithms from the sample size while maintaining fairness guarantees?

**Goal**: To develop a scalable fair clustering algorithm where the number of parameters is independent of the sample size.

**Key Insight**: Replacing geometric distance with probabilistic mixture models, and shifting fairness constraints from assignment mappings to model parameters—parameterized assignment mappings inherently support mini-batch learning and out-of-sample assignment.

**Core Idea**: Fair clustering based on finite mixture models, achieving scalability independent of $N$ by imposing Gap constraints on model parameters.

## Method

### Overall Architecture
The inputs are a dataset with sensitive attributes and the number of clusters $K$, and the output is a fair probabilistic assignment mapping. It is assumed that the data is generated from $K$ mixture components: $X \sim \sum_{k=1}^K \pi_k f(\cdot; \theta_k)$. The parameters $\Theta = (\boldsymbol{\pi}, (\theta_1, ..., \theta_K))$ are estimated by maximizing the log-likelihood subject to fairness constraints.

### Key Designs

1. **Parameterized Assignment Mapping**:

    - **Function**: Uses the posterior probabilities of the mixture model as soft assignment mappings.
    - **Mechanism**: $\psi_k(x_i; \Theta) = \frac{\pi_k f(x_i; \theta_k)}{\sum_l \pi_l f(x_i; \theta_l)}$
    - **Design Motivation**: The assignment mapping is completely determined by the model parameters $\Theta$, and the number of parameters ($K$ means + covariances + mixture weights) is independent of $N$.

2. **Gap Fairness Constraint**:

    - **Function**: Ensures that the proportions of different sensitive groups in each cluster are close.
    - **Mechanism**: Define $\Delta(\Theta) = \max_k |\frac{\sum_{x_i \in \mathcal{D}^{(1)}} \psi_k(x_i; \Theta)}{N_1} - \frac{\sum_{x_j \in \mathcal{D}^{(2)}} \psi_k(x_j; \Theta)}{N_2}|$, and the optimization objective is $\max_\Theta \ell(\Theta | \mathcal{D}) - \lambda \Delta(\Theta)$.
    - **Design Motivation**: The Gap metric is numerically more stable than the Balance metric, making it suitable for gradient optimization.

3. **FMC-GD and FMC-EM Optimization Algorithms**:

    - FMC-GD: Performs gradient descent directly on $\ell(\Theta|\mathcal{D}; \lambda)$.
    - FMC-EM: Modifies the Q-function by adding a fairness penalty $Q_{fair} = \mathbb{E}[\ell_{comp}(\Theta|Y)] - \lambda\Delta(\Theta)$, using GEM (Generalized EM) to guarantee that the Q-function increases monotonically at each step.
    - Experiments show that FMC-EM achieves a better cost-fairness trade-off with lower variance.

4. **Mini-batch Learning and Subsampled $\Delta$**:

    - **Function**: Resolves the computational bottlenecks on large-scale data.
    - **Mechanism**: Applies mini-batching to the likelihood part and subsampling to $\Delta$—theoretically proving that the approximation error of subsampled $\Delta$ is $O(\sqrt{d/n'})$ (Proposition 1).
    - **Design Motivation**: Existing methods cannot utilize mini-batch learning (as fairness relies on complete assignments). The parameterized assignment mapping proposed in this work makes this feasible.

### Loss & Training
$\max_\Theta \ell(\Theta | \mathcal{D}) - \lambda \Delta(\Theta)$. FMC-GD: $T = 10000$, learning rate $10^{-3}$. FMC-EM: $T = 200$, inner loop $R = 10$, learning rate $10^{-2}$.

## Key Experimental Results

### Main Results
The Pareto frontiers of $\Delta$ vs. Cost are compared on three medium-scale UCI datasets: Adult, Bank, and Credit. FMC-EM is competitive with baseline methods such as SFC, VFC, and FCA, showing noticeable improvements over baselines on the Credit dataset.

### Large-Scale Experiment (Census, 2.45M Samples)

| Method | Time (seconds) | Feasibility |
|------|---------|--------|
| SFC | Timeout | ❌ |
| VFC | Timeout | ❌ |
| FCA | ~3000 | ✅ |
| FMC (Subsampling 5%) | ~60 | ✅ |

FMC runs approximately 50 times faster than the fastest baseline FCA on the Census dataset, while maintaining comparable fairness and clustering cost.

### Key Findings
- FMC-EM outperforms FMC-GD, achieving a better Pareto frontier and smaller variance.
- Performance remains almost intact with 5% subsampling—subsampling rates dropping from 100% to 5% result in minimal changes in both $\Delta$ and Cost.
- FMC can be directly extended to categorical data (by replacing Gaussian distribution with multinomial distribution), which cannot be done by existing K-means-based baseline methods.
- Fair assignment for out-of-sample data: Once FMC learns the parameterized assignment mapping, it can be directly applied to testing data, whereas existing methods require re-optimization.

## Highlights & Insights
- The perspective shift **from assignment optimization to parameter optimization** is elegant—addressing scalability, mini-batch training, and out-of-sample assignment simultaneously in one step.
- The **theoretical guarantee for subsampled $\Delta$** (Proposition 1) provides the first theoretical support for using mini-batch learning in fair clustering.
- The **extension to categorical data** is a unique advantage that existing fair clustering methods cannot cover.

## Limitations & Future Work
- Gaussian mixture model assumes a specific data distribution, which may not be suitable for complex non-linear manifold structures.
- The selection of $\lambda$ requires manual parameter tuning to control the fairness-cost trade-off.
- The main analysis only considers binary sensitive attributes; although multi-sensitive extensions are discussed, experimental validation is limited.
- It has not yet been integrated with deep clustering or contrastive learning techniques.

## Related Work & Insights
- **vs SFC (fairlet)**: SFC performs fairlet decomposition in preprocessing, which has high computational complexity and times out on large-scale data; FMC's parameterized approach is naturally scalable.
- **vs VFC**: VFC uses KL divergence for variational fairness constraints, which also does not support mini-batches; FMC's subsampling strategy resolves this issue.
- **vs FCA**: FCA is the latest SOTA, showing competitive performance on medium-scale datasets, but FMC is far faster than FCA on large-scale datasets.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of using mixture models for fair clustering is novel and solves the core challenge of scalability.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets + large-scale validation + subsampling analysis, highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear motivation.
- Value: ⭐⭐⭐⭐ The first scalable probabilistic model-based fair clustering, carrying high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications](generalizing_fair_clustering_to_multiple_groups_algorithms_and_applications.md)
- [\[ICML 2025\] Relative Error Fair Clustering in the Weak-Strong Oracle Model](../../ICML2025/ai_safety/relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks](deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)

</div>

<!-- RELATED:END -->
