---
title: >-
  [Paper Note] Complexity Scaling Laws for Neural Models using Combinatorial Optimization
description: >-
  [NeurIPS 2025][Reinforcement Learning][neural scaling laws] Using the Traveling Salesman Problem (TSP) as a case study, this paper investigates predictable scaling relationships between problem complexity (solution space…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "neural scaling laws"
  - "combinatorial optimization"
  - "TSP"
  - "problem complexity"
date: 2026-05-08
content_hash: 9a7145d7928310fb
---

# Complexity Scaling Laws for Neural Models using Combinatorial Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2506.12932](https://arxiv.org/abs/2506.12932)
**Code**: [GitHub](https://github.com/lowellw6/complexity-scaling-laws)
**Area**: Reinforcement Learning
**Keywords**: neural scaling laws, combinatorial optimization, TSP, problem complexity, reinforcement learning

## TL;DR

Using the Traveling Salesman Problem (TSP) as a case study, this paper investigates predictable scaling relationships between problem complexity (solution space size, representation space dimensionality) and model performance under fixed model capacity, revealing systematic performance trends for RL and SFT in combinatorial optimization.

## Background & Motivation

### State of the Field

Neural scaling laws have demonstrated that model performance improves predictably with computational budget, model size, and dataset size. However, a critical dimension has been overlooked: how does **problem complexity** affect the performance ceiling of fixed-capacity models?

This question poses two key challenges:
1. In domains such as language modeling and vision, it is difficult to decouple problem complexity into independently scalable metrics.
2. RL reward signals are typically non-smooth, making it hard to obtain clean scaling laws.

TSP, as a canonical combinatorial optimization problem, offers unique advantages:

### Limitations of Prior Work

The **solution space** (number of nodes $n$) and the **representation space** (spatial dimensionality $d$) can be scaled independently.

### Root Cause

Combinatorial optimization induces smooth cost trends, enabling meaningful scaling laws even in the absence of an interpretable loss function.

## Method

### Overall Architecture

The paper fixes model parameter count and systematically scales two complexity dimensions of TSP:

1. **Solution space scaling**: Fix $d=2$, vary $n \in \{5, 10, 15, \ldots, 50\}$, with solution space ranging from 12 to $\sim3 \times 10^{62}$.
2. **Representation space scaling**: Fix $n$, vary $d \in \{2, 3, \ldots, 12, 15, 20, 30, 40, 50, 100\}$.

The performance metric is the **suboptimality gap**:

$$s = \mu_{model} - \mu_{opt}$$

where $\mu_{model}$ is the model's mean tour length and $\mu_{opt}$ is the optimal tour length.

### Key Designs

**Model Architecture**: A Transformer-based autoregressive Pointer Network that constructs the tour sequence step by step, selecting the next unvisited node at each step via a softmax policy distribution.

**Two Training Paradigms**:
- **Reinforcement Learning (RL)**: A bandit problem with reward equal to negative tour length, trained with PPO for 1 million steps.
- **Supervised Fine-Tuning (SFT)**: Minimizes the negative log-likelihood of the model's edge selections with respect to the optimal solution.

**Neural Scaling Laws** (parameter/compute scaling): For 2D TSP with 20 nodes, suboptimality decays as a power law with parameter count or compute:

$$s \propto N^{-\alpha} \quad \text{(parameter scaling)}$$
$$s \propto C^{-\alpha} \quad \text{(compute scaling)}$$

**Problem Complexity Scaling Laws**:

- Node scaling: Suboptimality grows as a super-linear power law:
$$s \propto (n - \gamma)^{\alpha}$$

- Dimension scaling: Suboptimality decays exponentially toward an asymptote:
$$s \approx \beta_\psi + a \cdot \psi^{-d}$$

### Loss & Training

- RL: PPO clipped surrogate objective with reward signal $-\text{tour\_length}$.
- SFT: Negative log-likelihood loss $\mathcal{L} = -\log p(\text{optimal edges} | \text{model})$ with teacher forcing.

## Key Experimental Results

### Main Results

**Parameter Scaling** (2D TSP, 20 nodes):

| Training Paradigm | Parameter Scaling Exponent $\alpha$ | Compute Scaling Trend |
|---|---|---|
| RL | Power-law decay | Compute-efficient frontier follows power-law decay |
| SFT | Potentially faster decay | Higher compute efficiency |

The growth rate exponent of optimal model size vs. compute budget is approximately 0.5–0.75, closely consistent with findings in language modeling.

**Node Scaling** (fixed model capacity):

| Training Paradigm | Growth Exponent $\alpha$ | Trend Form |
|---|---|---|
| RL | ≈1.86 | Super-linear power-law growth |
| SFT | ≈1.69 | Super-linear power-law growth |

**Dimension Scaling** (RL, fixed model capacity):

| Node Count | Trend Form | Asymptote $\beta_\psi$ |
|---|---|---|
| 10 | Exponential decay | Higher |
| 20 | Exponential decay | Lower (consistent ordering) |

### Ablation Study

**Comparison with 2-opt Local Search**:

- Dimension scaling: 2-opt trends closely match those of RL, with asymptotes nearly aligned at 10 nodes.
- Node scaling: Unconstrained 2-opt produces irregular trends (with an inflection point), but limiting search depth $M$ recovers super-linear power-law growth.
- A sub-exponential decay form $\psi^{-d^\phi}$ fits 2-opt better, though at the cost of an additional parameter.

**SFT vs. RL Comparison**:
- SFT achieves higher compute efficiency within the tested scales (at the cost of node-level supervision).
- Parameter efficiency is comparable at current scales, but SFT may surpass RL at larger model sizes.

### Key Findings

1. **TSP naturally yields smooth scaling laws**: Using tour length directly as the performance metric produces high-quality power-law relationships without requiring an interpretable loss.
2. **Super-linear node scaling must eventually break down**: Extrapolation indicates that model suboptimality would exceed that of a random tour at approximately 40,000 nodes, implying the true trend will flatten.
3. **Reason for convergence in dimension scaling**: In high dimensions, random and optimal tour lengths diverge at similar rates, causing the suboptimality gap to approach a constant (proven in Theorem 6).
4. **Separability of embedding parameters**: In dimension scaling, the role of embedding layer parameters is fundamentally distinct from model capacity parameters, corroborating the embedding separation principle established in prior neural scaling law literature.
5. **Training paradigm insensitivity**: Complexity scaling under fixed parameter budgets is insensitive to the choice of RL or SFT training.

## Highlights & Insights

- **Core Contribution**: The first systematic study of problem complexity scaling laws under fixed model capacity, filling an important gap in the neural scaling laws literature.
- **Analogy**: Performance degradation of fixed-capacity models as complexity increases is analogous to fixed-depth local search—the model is effectively forced into early termination near local optima.
- **Experimental Design**: The creation of a large-scale dataset of 128M optimal TSP solutions serves both SFT training and precise evaluation.
- **Cross-Paradigm Consistency**: The growth rate of optimal model size with compute budget (0.5–0.75) in TSP is closely consistent with findings in language modeling, game playing, and other domains.

## Limitations & Future Work

1. **Single-problem limitation**: All conclusions are based solely on Euclidean TSP; generalizability to other combinatorial optimization problems such as VRP and QAP remains unverified.
2. The triangle inequality and the existence of a PTAS for Euclidean TSP make it "easier" than most NP-hard problems.
3. Only a single training seed is used per scale, precluding assessment of statistical significance.
4. Limited computational resources (50-node RL experiments require 24 V100-days) prevent verification of trends at larger scales.
5. The theoretical explanation for why specific power-law exponents and asymptotic values are obtained remains unclear.

## Related Work & Insights

- **Connection to Kaplan et al.**: Extends the scaling law paradigm from language modeling—covering model size, data, and compute—to the new dimension of problem complexity.
- **Distinction from Jones (AlphaZero Hex)**: Jones's approach cannot study the bottleneck of fixed model capacity, as the model achieves perfect play at every board size.
- **Future Directions**: Investigating complexity scaling in more complex combinatorial optimization settings such as VRP and knapsack problems; developing theoretical frameworks to predict $\alpha$ and $\beta$ parameters; exploring complexity scaling under limited compute or data budgets.

## Rating

- ⭐ Novelty: 5/5 — Problem complexity scaling laws represent a genuinely new and far-reaching research direction.
- ⭐ Value: 3/5 — Valuable for algorithm benchmarking and performance prediction in edge deployment, though practical application scenarios are currently limited.
- ⭐ Experimental Thoroughness: 4/5 — The three-way comparison across RL, SFT, and local search is comprehensive; the 128M dataset is a significant contribution.
- ⭐ Writing Quality: 5/5 — The paper is exceptionally well organized, with thorough discussion and candid, insightful analysis of limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization](parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)

</div>

<!-- RELATED:END -->
