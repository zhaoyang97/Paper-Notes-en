---
title: >-
  [Paper Note] LCRON: Learning Cascade Ranking as One Network
description: >-
  [ICML 2025][Recommender Systems][Cascade Ranking] This work proposes LCRON, which trains multi-stage cascade ranking systems as a unified network in an end-to-end manner. Specifically, an end-to-end surrogate loss $L_{e2e}$ constructed via differentiable ranking techniques directly optimizes the lower bound of the survival probability of ground truth items through the entire cascade. This is assisted by auxiliary individual stage losses $L_{single}$ derived from the tightness…
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "Cascade Ranking"
  - "End-to-End Training"
  - "Differentiable Ranking"
  - "Survival Probability"
  - "Surrogate Loss"
  - "Multi-stage Collaboration"
date: 2026-05-08
content_hash: 068f111c022097f0
---

# LCRON: Learning Cascade Ranking as One Network

**Conference**: ICML 2025  
**arXiv**: [2503.09492](https://arxiv.org/abs/2503.09492)  
**Code**: None  
**Area**: Recommender Systems / Cascade Ranking  
**Keywords**: Cascade Ranking, End-to-End Training, Differentiable Ranking, Survival Probability, Surrogate Loss, Multi-stage Collaboration

## TL;DR

This work proposes LCRON, which trains multi-stage cascade ranking systems as a unified network in an end-to-end manner. Specifically, an end-to-end surrogate loss $L_{e2e}$ constructed via differentiable ranking techniques directly optimizes the lower bound of the survival probability of ground truth items through the entire cascade. This is assisted by auxiliary individual stage losses $L_{single}$ derived from the tightness of the lower bound to drive collaboration among stages. LCRON achieves significant improvements in both public benchmarks and online A/B tests of industrial advertising systems (Ad Revenue +4.10%, User Conversion +1.60%).

## Background & Motivation

**Background**: Large-scale recommendation/advertising systems widely adopt cascade ranking architectures (Matching $\rightarrow$ Pre-ranking $\rightarrow$ Ranking $\rightarrow$ Mix-ranking) to filter candidates progressively in a multi-stage funnel. Models of varying capacities are used at each stage to balance resource efficiency and performance. The ultimate goal of the system is to ensure that the ground truth items (items truly of interest to the user) survive all stages and are final-selected.

**Limitations of Prior Work**: Traditional methods train each stage independently, which suffers from two core issues: (1) **Objective Misalignment**: each stage is optimized separately using pointwise/pairwise losses. This is stricter than the true cascade objective (collaboratively selecting all relevant items), leading to inefficiency when model capacity is limited. (2) **Lack of Collaborative Learning**: stages trained independently fail to learn interaction patterns (e.g., the matching model preemptively avoiding items that the ranking model would overestimate), making collaboration during online serving purely coincidental.

**Key Challenge**: ICC only allows unidirectional interaction, RankFlow requires iterative training (which is complex and unstable), FS-LTR utilizes all-stage samples but fails to align with the global objective, and ARF only optimizes a single stage assuming the downstream model is optimal. No existing methods resolve both objective alignment and collaborative learning challenges simultaneously.

**Goal**: To design an end-to-end training paradigm that simultaneously addresses objective misalignment and the lack of collaborative learning, allowing all stages to be optimized jointly as a unified network.

**Key Insight**: Formulate the cascade ranking objective as maximizing the survival probability of ground truth items through all stages. Differentiable ranking techniques are leveraged to relax the discrete top-$k$ selection into continuous probabilities, allowing direct optimization after deriving a lower bound.

**Core Idea**: Build the top-$k$ selection probability of each stage using soft permutation matrices from differentiable ranking. The joint survival probability of the cascade is factorized into a lower bound ($\hat{P}_{CS}^{q_2} = \prod_i P_{\mathcal{M}_i}^{q_i}$) of the product of stage-wise probabilities. Directly optimizing this lower bound achieves end-to-end alignment, while auxiliary losses derived from the tightness of the lower bound promote inter-stage consistency.

## Method

### Overall Architecture

Taking a two-stage cascade (matching $\mathcal{M}_1$ + ranking $\mathcal{M}_2$) as an example: construct all-stage training samples (downsampled from each stage), execute differentiable top-$k$ selection to compute the stage-wise selection probability $P_{\mathcal{M}_i}^{q_i}$, construct $L_{e2e}$ based on the lower bound of the product of probabilities, and derive the auxiliary stage losses $L_{single}$ from the tightness of the lower bound. The total loss is adaptively weighted via UWL (Uncertainty-based Weighted Loss) to reduce hyperparameters.

### Key Designs

1. **End-to-End Surrogate Loss $L_{e2e}$**
    - **Function**: Directly optimizes the survival probability of ground truth items through the entire cascade.
    - **Mechanism**: The exact cascade survival probability $P_{CS}^{q_2} = \mathbb{E}_{\pi \sim P_\pi} \frac{P_{\mathcal{M}_2}^{q_2} \odot \pi}{\langle \pi, P_{\mathcal{M}_2}^{q_2} \rangle / \langle \mathbf{1}, P_{\mathcal{M}_2}^{q_2} \rangle}$ is computationally complex, but it can be mathematically proven that $\hat{P}_{CS}^{q_2} = \prod_i P_{\mathcal{M}_i}^{q_i} \leq P_{CS}^{q_2}$. Here, $P_{\mathcal{M}_i}^{q_i} = \frac{\sum_{j=1}^{q_i} (\hat{\mathcal{P}}_{\mathcal{M}_i}^\downarrow)_{j,:}}{\oslash sp(\sum_t (\hat{\mathcal{P}}_{\mathcal{M}_i}^\downarrow)_{t,:})}$, where $\hat{\mathcal{P}}$ is the soft permutation matrix generated by differentiable ranking. The cross-entropy between $\hat{P}_{CS}^{q_2}$ and labels $\mathbf{y}$ is used as $L_{e2e}$.
    - **Design Motivation**: Compared to independent pointwise/pairwise losses per stage, $L_{e2e}$ allows models with insufficient capacity to prioritize crucial rankings while tolerating minor errors. When a certain stage assigns a low score to the ground truth, it not only optimizes that stage but also encourages other stages to compensate, achieving bidirectional collaboration.

2. **Auxiliary Stage Loss $L_{single}$**
    - **Function**: Provides extra supervision to each stage to tighten the lower bound.
    - **Mechanism**: The tightness of the lower bound is related to the consistency of top-$k$ selection across stages. When $\langle \pi, P_{\mathcal{M}_2}^{q_2} \rangle / \langle \mathbf{1}, P_{\mathcal{M}_2}^{q_2} \rangle$ approaches 1 (meaning items selected by $\mathcal{M}_1$ also score high in $\mathcal{M}_2$), the lower bound becomes tighter. Therefore, $L_{single}$ is designed to optimize independent Recall for each stage, forcing each stage to recognize ground truth from the complete candidate pool rather than just the upstream-filtered subset.
    - **Design Motivation**: Resolves the gradient vanishing issue of $L_{e2e}$ when the survival probability of a certain stage is close to 0. It inherits the concept of $L_{Relax}$ from ARF but improves the utilization of information from the soft permutation matrix.

3. **Probability Relaxation based on Differentiable Ranking**
    - **Function**: Converts discrete top-$k$ selection operations into differentiable ones.
    - **Mechanism**: Utilizes differentiable ranking methods (such as NeuralSort or SoftSort) to generate a soft permutation matrix $\hat{\mathcal{P}} \in [0,1]^{N \times N}$, where $(\hat{\mathcal{P}})_{j,k}$ represents the soft probability of item $k$ being ranked at position $j$. As temperature $\tau \to 0$, it converges to a hard permutation. Row normalization is applied to guarantee probability validity.
    - **Design Motivation**: Differentiable ranking is the core technical foundation for elevating learning objectives from pointwise/pairwise to listwise recall optimization.

### Loss & Training

The total loss is $L = L_{e2e} + \sum_i L_{single}^{(i)}$, where each term is adaptively weighted via UWL (Kendall et al., 2018) to minimize hyperparameters. All-stage training samples are constructed following the FS-LTR strategy: downsampling is performed from each stage ($\mathcal{Q}_0, \mathcal{Q}_1, \mathcal{Q}_2, CS_{gt}$), and labels are determined jointly by the stage order and intra-stage ranking. End-to-end training allows parameters of all stages to be updated simultaneously via gradients.

## Key Experimental Results

### Public Benchmark RecFlow (Streaming Evaluation)

| Method | e2e-Recall@50 | e2e-Recall@100 | Training Approach |
|------|-------------|---------------|---------|
| Independent | Baseline | Baseline | Independent per stage |
| ICC | 0.1316 | 0.2253 | Unidirectional score integration |
| RankFlow | 0.1373 | 0.2320 | Iterative training |
| FS-LTR | 0.1392 | 0.2362 | All-stage samples + LambdaRank |
| **LCRON** | **0.1429** | **0.2417** | **End-to-end unified network** |

### Online A/B Test (Real Advertising System)

| Metric | Improvement vs FS-LTR |
|------|-------------|
| Ad Revenue | **+4.10%** |
| User Conversions | **+1.60%** |
| End-to-End Recall | Significant Improvement |

### Ablation Study

| Configuration | Change in e2e-Recall |
|------|-------------|
| $L_{e2e}$ only | +2.3% (vs FS-LTR) |
| $L_{single}$ only | +1.5% |
| $L_{e2e}$ + $L_{single}$ (LCRON) | **+4.2%** |
| Various Differentiable Ranking Methods | NeuralSort and SoftSort perform comparably |
| UWL vs Fixed Weight | UWL is more stable |

### Key Findings

- $L_{e2e}$ and $L_{single}$ are complementary: $L_{e2e}$ provides global optimization directions, while $L_{single}$ offers valid gradients when local survival probability is close to 0.
- Bidirectional collaboration outperforms ICC's unidirectional interaction: LCRON allows improvements in any stage to affect other stages via gradients.
- Compared to the iterative training of RankFlow, the one-time end-to-end training of LCRON is more stable and efficient.
- The 4.10% increase in ad revenue in online A/B testing directly demonstrates industrial value.
- The consistent advantages in Streaming evaluation (testing on any day $t$ with models trained on days $0$ to $t-1$) indicate excellent temporal generalization characteristics.

## Highlights & Insights

- **Mathematically rigorous and elegant derivation of survival probability lower bound**: The proof of $P_{CS}^{q_2} \geq \prod_i P_{\mathcal{M}_i}^{q_i}$ is concise and intuitively clear (normalization factor $\leq 1$), providing a solid theoretical foundation for the end-to-end loss.
- **Auxiliary loss naturally derived from lower bound tightness**: $L_{single}$ is not an ad-hoc regularization term, but is derived from the motivation of tightening the lower bound, showing highly consistent theory and practice.
- **Grand business value of online A/B testing**: The Ad Revenue gain of 4.10% represents a very significant improvement in industry, directly reflecting the practical impact of the method.
- **Addressing the two core challenges of cascade ranking training**: Objective alignment and collaborative learning are tackled simultaneously under a single framework.

## Limitations & Future Work

- End-to-end training poses higher memory and computational demands: differentiable ranking generates $N \times N$ soft permutation matrices, which incurs significant overhead when $N$ is large.
- This study only demonstrates with a two-stage cascade. Although claiming scalability to more stages, no experiments exceeding two stages are provided.
- The downsampling strategy for all-stage training samples (i.e., sample size $n_i$ per stage) impacts performance but has not been analyzed in depth.
- The choice of temperature $\tau$ in differentiable ranking is not discussed in detail: a too-low $\tau$ may cause gradient issues.
- Joint exploration with newer multi-objective optimization methods (e.g., Pareto optimization) is lacking.

## Related Work & Insights

- **ICC (Gallagher et al., 2019)**: The earliest attempt to jointly train cascades, but it only allows unidirectional interaction (ranking $\rightarrow$ matching) with limited sample space.
- **RankFlow (Qin et al., 2022)**: An iterative training paradigm, where upstream decides downstream training samples and downstream knowledge is distilled to upstream; it significantly outperforms ICC but remains unstable.
- **FS-LTR (Zheng et al., 2024)**: All-stage samples + LambdaRank. It serves as the primary baseline for LCRON, upon which LCRON introduces end-to-end objective alignment.
- **ARF (Wang et al., 2024)**: Proposes a surrogate loss of Recall based on differentiable ranking, but only optimizes a single stage assuming the downstream stage is optimal. LCRON extends this to the entire cascade.
- **Insight**: Differentiable ranking techniques enable a leap from pointwise/pairwise to listwise recall optimization, acting as a core component of the next-generation training paradigm for cascade ranking.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The joint design of survival probability lower bound + auxiliary loss is the core contribution, supported by solid theoretical derivations)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Public benchmark + industrial deployment + online A/B testing with comprehensive ablations)
- **Writing Quality**: ⭐⭐⭐⭐ (Mathematical derivations are clear, though the intensive notations require careful reading)
- **Value**: ⭐⭐⭐⭐⭐ (The industrial verification with a 4.10% ad revenue boost holds extremely high value and carries direct commercial impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](../../AAAI2026/recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [\[ICLR 2026\] Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies](../../ICLR2026/recommender/off-policy_evaluation_for_ranking_policies_under_deterministic_logging_policies.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[ICML 2025\] Not All Explanations for Deep Learning Phenomena Are Equally Valuable](not_all_explanations_for_deep_learning_phenomena_are_equally_valuable.md)

</div>

<!-- RELATED:END -->
