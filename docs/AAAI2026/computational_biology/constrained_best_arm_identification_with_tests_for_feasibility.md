---
title: >-
  [Paper Note] Constrained Best Arm Identification with Tests for Feasibility
description: >-
  [AAAI 2026][Computational Biology][Best Arm Identification] This paper proposes a new framework for best arm identification (BAI) with feasibility constraints…
tags:
  - "AAAI 2026"
  - "Computational Biology"
  - "Best Arm Identification"
  - "feasibility constraints"
  - "fixed confidence"
  - "sample complexity"
  - "asymptotic optimality"
date: 2026-05-08
content_hash: b90f800d044c29c3
---

# Constrained Best Arm Identification with Tests for Feasibility

**Conference**: AAAI 2026
**arXiv**: [2511.09808](https://arxiv.org/abs/2511.09808)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: Best Arm Identification, feasibility constraints, fixed confidence, sample complexity, asymptotic optimality

## TL;DR

This paper proposes a new framework for best arm identification (BAI) with feasibility constraints, allowing the decision-maker to test arm performance and feasibility constraints separately. An asymptotically optimal algorithm is designed that adaptively eliminates suboptimal arms via whichever criterion—performance or feasibility—is easier to satisfy.

## Background & Motivation

**Wide applicability of BAI**: BAI aims to identify the arm with the highest expected reward among $K$ arms via random sampling, with broad applications in drug discovery, A/B testing, crowdsourcing, and database tuning.

**Need for constraints in practical settings**: In drug discovery, the best drug must not only be maximally efficacious but also satisfy safety thresholds on toxicity and solubility; in database tuning, latency must be minimized while keeping system failure risk sufficiently low.

**Performance and constraints can be tested independently**: In drug discovery, experiments for efficacy, solubility, and toxicity can be conducted independently without measuring all metrics simultaneously. However, existing constrained BAI work (Katz-Samuels and Scott 2018, 2019) assumes all metrics are observed upon each arm pull, which is unrealistic and sample-inefficient.

**Naïve strategies are suboptimal**: A "feasibility-first, then performance" strategy wastes samples on arms with large performance gaps but hard-to-determine feasibility; a "performance-first, then feasibility" strategy wastes samples on arms that are clearly infeasible but close to optimal in performance.

**Core insight**: Each suboptimal arm should be eliminated via the easier criterion—arms with clearly inferior performance should be eliminated by performance, and arms that are evidently infeasible should be eliminated by feasibility. The algorithm must make this choice adaptively.

## Method

### Overall Architecture

Given $K$ arms, each arm is associated with $N+1$ distributions: $\nu_{i,0}$ for performance and $\nu_{i,\ell}$ ($\ell \in [N]$) for the $\ell$-th feasibility constraint. Arm $i$ is feasible if and only if $\mu_{i,\ell} < \frac{1}{2}$ for all $\ell \in [N]$. At each round, the decision-maker selects $(i, \ell)$ and receives a sample from $\nu_{i,\ell}$. The goal is to identify the feasible arm with the highest performance with probability at least $1-\delta$, while minimizing total sample count.

The algorithm extends the LUCB framework, maintaining the following key sets:

- $S$: the surviving arm set (arms not yet eliminated)
- $P$: the focus set (arms in $S$ with high performance confidence bounds)
- $F$: the set of confirmed feasible arms
- $I$: the set of confirmed infeasible arms
- $H_i$: the subset of constraints for arm $i$ not yet confirmed feasible

### Key Design 1: Two-Dimensional Complexity and Arm Classification

For each arm $i$, a feasibility complexity $\theta_i$ and a performance complexity $\phi_i$ are defined. For an infeasible arm, only **one** constraint violating the threshold needs to be found for elimination, so $\theta_i$ is the inverse square of the maximum deviation from the threshold; for a feasible arm, **all** constraints must be verified, so $\theta_i$ is the sum of inverse squares of all constraint deviations:

$$\theta_i = \begin{cases} (\max_{\ell \in [N]} \mu_{i,\ell} - \frac{1}{2})^{-2}, & i \notin \mathcal{F} \\ \sum_{\ell=1}^{N} (\mu_{i,\ell} - \frac{1}{2})^{-2}, & i \in \mathcal{F} \end{cases}$$

Performance complexity measures the gap relative to the optimal arm $i^\star$—arms that outperform $i^\star$ in expectation but are infeasible cannot be eliminated by performance ($\phi_i = \infty$) and must be eliminated via feasibility:

$$\phi_i = \begin{cases} \infty, & i < i^\star \\ (\mu_{i^\star,0} - \mu_{i,0})^{-2}, & i > i^\star \end{cases}$$

Suboptimal arms are partitioned into $\mathcal{I}$ (easier to eliminate via infeasibility, $\theta_i < \phi_i$) and $\mathcal{W}$ (easier to eliminate via performance comparison). The total problem complexity is $\mathcal{H} = \sum_{i=1}^{K} \mathcal{H}_i$; for the optimal arm, both feasibility and performance superiority must be verified.

### Key Design 2: SAMPLE-FEASIBILITY Subroutine

When testing the feasibility of arm $i$, one constraint must be intelligently selected from $N$ candidates. Inspired by Kano et al. (2019), the constraint with the highest UCB is chosen:

$$\ell_t = \arg\max_{\ell \in H_i} \tilde{\mu}_{i,\ell}(t), \quad \tilde{\mu}_{i,\ell}(t) = \hat{\mu}_{i,\ell}(t) + \sqrt{\frac{2 \log M_i(t)}{N_{i,\ell}(t)}}$$

where $M_i(t)$ is the total number of feasibility tests for arm $i$. Intuitively, the constraint most likely to exceed threshold $\frac{1}{2}$ is prioritized. After sampling: if the lower confidence bound of the constraint exceeds $\frac{1}{2}$, the arm is confirmed infeasible and added to $I$; if the upper confidence bound falls below $\frac{1}{2}$, the constraint is verified and removed from $H_i$; if $H_i$ becomes empty, the arm is confirmed feasible and added to $F$.

### Key Design 3: LUCB-Style Main Loop Sampling

Each epoch selects at most two arms for performance testing:

- $a_t = \arg\max_{i \in P} \hat{\mu}_{i,0}(t)$: the arm with the highest empirical mean in the focus set
- $b_t = \arg\max_{i \in P \setminus \{a_t\}} \bar{\mu}_{i,0}(t)$: the arm with the highest performance upper bound among the remaining focus set arms

After performance testing, if $a_t$ or $b_t$ is not in $F$, one call to SAMPLE-FEASIBILITY is made for each. When $P$ degenerates to a singleton $\{i\}$: if $i \in F$, return $i$; otherwise, continue testing its feasibility. When $S = \emptyset$, return "no feasible arm."

The epoch-end update achieves dual elimination: (1) all infeasible arms in $I$ are removed from $S$; (2) if $F \neq \emptyset$, arms whose performance upper bound falls below the performance lower bound of the best known feasible arm are eliminated. The focus set $P$ is updated to retain arms whose performance upper bound exceeds the maximum performance lower bound in the surviving set.

### Theoretical Results

**Lower bound**: For any $\delta$-correct algorithm, the expected sample complexity satisfies $\mathbb{E}[\tau] \geq 2\mathcal{H} \log \frac{1}{2.4\delta}$, proved by constructing a carefully designed alternative bandit instance.

**Upper bound**: The expected stopping time of the algorithm satisfies $\limsup_{\delta \to 0} \frac{\mathbb{E}[\tau]}{\log(1/\delta)} \in \widetilde{\mathcal{O}}(\mathcal{H})$ as $\delta \to 0$, asymptotically matching the lower bound. The key to proving the upper bound is a multi-level decomposition of the stopping time $\tau$ by elimination type and selection rule.

## Key Experimental Results

### Synthetic Experiments ($K=5, N=3$, Gaussian distributions, $\delta=0.1$)

Using the proposed algorithm's sample count as baseline (1.0), relative sample counts for each algorithm are:

| Algorithm | Type a (many infeasible arms) | Type b (many feasible but low-performance arms) | Type c (mixed) |
|-----------|-------------------------------|--------------------------------------------------|----------------|
| **Ours** | **1.0** | **1.0** | **1.0** |
| F-first | 0.58 | 3.13 | 4.00 |
| P-first | 3.09 | 0.84 | 3.47 |
| TF-LUCB-C | 1.77 | 1.98 | 2.78 |
| Naive | 1.81 | 4.75 | 4.06 |

F-first performs better on Type a (0.58×) and P-first on Type b (0.84×), but only the proposed algorithm performs near-optimally across all types, with a substantial advantage in the mixed Type c setting.

### Scalability Experiments

- **Constraint count $N$ from 2 to 64**: The proposed algorithm's sample count remains nearly constant as $N$ grows, since infeasible arms require only one violated constraint to be found; other methods that sample all constraints simultaneously grow linearly.
- **Arm count $K$ from 4 to 64**: The proposed algorithm consistently outperforms all baselines.
- **$\delta$ from $10^{-1}$ to $10^{-9}$**: Sample count grows linearly with $\log(1/\delta)$, consistent with theoretical predictions.

### Real-World Drug Discovery Data

Data are drawn from a secukinumab clinical trial for rheumatoid arthritis (Genovese et al. 2013), with 5 dosage levels as arms and 3 metrics (efficacy, adverse event probability, infection probability). The task is to identify the most efficacious dose that satisfies safety constraints. The proposed algorithm outperforms all baselines across all values of $\delta$.

## Highlights & Insights

1. **Novel problem formulation**: Separating performance testing from feasibility constraint testing accurately models settings such as drug discovery and database tuning, where experiments can be conducted independently.
2. **Elegant adaptive elimination**: By comparing $\theta_i$ and $\phi_i$, the algorithm automatically selects the optimal elimination strategy for each arm without prior knowledge of the problem structure.
3. **Matching upper and lower bounds**: Both information-theoretic lower bounds and algorithmic upper bounds are established, achieving asymptotic optimality as $\delta \to 0$, constituting a complete theoretical contribution.
4. **Scalability in constraint count**: Sample complexity does not grow linearly with $N$, since determining infeasibility requires only one violated constraint.

## Limitations & Future Work

1. **Non-asymptotic gap**: When $N > 1$ and $\delta \not\to 0$, a gap between the upper and lower bounds remains for arms in $\mathcal{I}$, stemming from exploration costs that cannot be fully eliminated; closing this gap remains an open problem.
2. **Uniform threshold assumption**: All constraint thresholds are assumed to be $\frac{1}{2}$; while different thresholds can be handled via rescaling, a more general non-uniform threshold setting would be more natural.
3. **Strong distributional assumptions**: The 1-sub-Gaussian assumption is required; extending results to heavy-tailed or nonparametric distributions requires additional work.
4. **Limited experimental scale**: Synthetic experiments use $K=5$, which is relatively small; although scalability experiments extend to 64, validation in very large-scale settings is lacking.
5. **Constraint correlations not exploited**: Constraints are tested independently; statistical correlations among constraints are not taken into account.

## Related Work & Insights

1. **LUCB algorithm** (Kalyanakrishnan et al. 2012): Provides the foundation for the sampling strategy in the main loop.
2. **Thresholding bandits** (Locatelli et al. 2016): Identify all arms whose means exceed a threshold; the present work requires only one violated constraint to declare infeasibility.
3. **Good Arm Identification** (Kano et al. 2019): Inspires the constraint selection strategy in SAMPLE-FEASIBILITY.
4. **Bayesian constrained optimization** (Gardner et al. 2014): Also allows separate evaluation of objectives and constraints, but addresses continuous search spaces rather than the $K$-armed setting.
5. **Broader implications**: The idea of separating tests can be extended to multi-stage clinical trials (safety screening followed by efficacy testing), multi-objective A/B testing, and other sequential decision-making scenarios.

## Rating

⭐⭐⭐⭐ (4/5)

**Strengths**: Novel and practically motivated problem formulation, complete theoretical results (asymptotically matching upper and lower bounds), and an elegant adaptive dual-elimination mechanism.

**Weaknesses**: Limited experimental scale, the sub-Gaussian assumption restricts applicability, and optimality in the non-asymptotic regime remains an open problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](../../NeurIPS2025/computational_biology/constrained_discrete_diffusion.md)
- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](../../ICLR2026/computational_biology/verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](../../ICML2026/computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[AAAI 2026\] On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples](on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)
- [\[AAAI 2026\] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)

</div>

<!-- RELATED:END -->
