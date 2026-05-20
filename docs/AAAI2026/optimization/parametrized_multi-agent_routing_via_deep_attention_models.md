---
title: >-
  [Paper Note] Parametrized Multi-Agent Routing via Deep Attention Models
description: >-
  [AAAI2026][Optimization][Combinatorial Optimization] This paper proposes the Deep FLPO framework, which integrates the algebraic structure of the Maximum Entropy Principle (MEP) with a permutation-invariant encoder-decod…
tags:
  - "AAAI2026"
  - "Optimization"
  - "Combinatorial Optimization"
  - "Multi-Agent Path Planning"
  - "Maximum Entropy Principle"
  - "Transformer"
  - "Facility Location"
date: 2026-05-08
content_hash: 7da45edc25d4a47b
---

# Parametrized Multi-Agent Routing via Deep Attention Models

**Conference**: AAAI2026
**arXiv**: [2507.22338](https://arxiv.org/abs/2507.22338)  
**Authors**: Salar Basiri, Dhananjay Tiwari, Srinivasa M. Salapaka (UIUC)
**Code**: [GitHub](https://github.com/salar96/LearningFLPO)  
**Area**: Optimization
**Keywords**: Combinatorial Optimization, Multi-Agent Path Planning, Maximum Entropy Principle, Transformer, Facility Location

## TL;DR

This paper proposes the Deep FLPO framework, which integrates the algebraic structure of the Maximum Entropy Principle (MEP) with a permutation-invariant encoder-decoder neural network (SPN) to address the NP-hard mixed-integer problem of joint facility location and routing optimization. The framework achieves a 100× speedup in policy inference while matching Gurobi's exact solutions at 1500× faster runtime.

## Background & Motivation

### Parametrized Sequential Decision-Making (ParaSDM)
Traditional neural combinatorial optimization (NCO) focuses on discrete optimization over fixed state/action spaces (e.g., TSP, VRP). However, a more complex class of problems exists in practice — **Parametrized Sequential Decision-Making (ParaSDM)**: $N$ agents make discrete decisions over $K$ steps while jointly optimizing a set of shared continuous parameters $\mathcal{Y} \subset \mathbb{R}^d$ (e.g., facility locations). This induces strong coupling and high non-convexity between discrete policies and continuous parameters.

### The FLPO Problem
A canonical instance of ParaSDM is the **Facility Location and Path Optimization (FLPO)** problem: $N$ agents (e.g., drones) must travel from a source to a destination through $M$ facilities (e.g., charging stations), with the objective of jointly optimizing discrete routes $\eta^i(\gamma)$ and continuous node positions $\mathcal{Y}$ to minimize total transportation cost. Even at the small scale of $N=2, M=4$, the solution space contains numerous local optima.

### Scalability Bottleneck of MEP-Based Methods
Prior work based on the Maximum Entropy Principle (MEP) relaxes hard assignments $\eta^i \in \{0,1\}$ to soft probabilities $p^i \in [0,1]$, balancing exploration and exploitation via an annealing parameter $\beta$. The MEP policy variables admit a closed-form Gibbs distribution:

$$p_\beta^i(\gamma) = \frac{\exp[-\beta \cdot d^i(\gamma)]}{\sum_{\gamma' \in \mathcal{G}^i} \exp[-\beta \cdot d^i(\gamma')]}$$

However, the per-step gradient complexity is $O(NM^4)$ (even after Markov DP optimization), rendering it infeasible for medium-scale problems.

## Method

### Shortest Path Network (SPN)

The SPN is a permutation-invariant encoder-decoder model that approximates the MEP Gibbs policy distribution.

**Encoder**: Employs an induced attention mechanism. The source $S$, destination $\Delta$, and facility coordinates $Y^t$ for all agents are concatenated into $E_0 \in \mathbb{R}^{N \times (M+2) \times d_h}$, processed through $L_{\text{enc}}$ attention blocks. Crucially, $M^*$ learnable inducing points $T \in \mathbb{R}^{M^* \times d_h}$ reduce the standard self-attention complexity from $O(NM^2)$ to $O(NM \cdot M^*)$:

$$\hat{E}_{l-1} = \text{MHA}(T, E_{l-1}, E_{l-1})$$
$$I_l = \text{LN}(E_{l-1} + \text{MHA}(E_{l-1}, \hat{E}_{l-1}, \hat{E}_{l-1}))$$

**Decoder**: Fuses current position and destination information via gated interpolation to produce a goal-aware query vector:

$$\alpha = \sigma([h_X, h_\Delta] W), \quad q = \alpha \odot h_X + (1-\alpha) \odot h_\Delta$$
$$\Pi_k = \text{softmax}(E_{L_{\text{enc}}} \cdot q / \sqrt{d_h})$$

Two decoder variants are proposed: DED (Direct Embedding Decoder) and DCAD (Dual Cross-Attention Decoder). Ablation studies show that DED with annealing supervision achieves the best performance, with an average optimality gap of approximately 6%.

### Mixed-Sampling Gradient Estimation

To approximate the annealing behavior of MEP, $L$ paths are sampled per agent: the first $b$ are generated via SPN beam search or sampling (short paths assigned high Gibbs weights for exploitation), while the remaining $L-b$ are drawn from uniform random sampling (promoting exploration at low $\beta$). The free energy gradient is estimated as:

$$\widehat{\nabla_{\mathcal{Y}^t} F_\beta} = \sum_{i=1}^N \rho_i \sum_{q=1}^L \hat{p}_\beta^i(\gamma^q) \nabla_{\mathcal{Y}^t} d^i(\gamma^q)$$

where $\hat{p}_\beta^i$ re-normalizes Gibbs weights over the sample set. All gradients $\nabla_{\mathcal{Y}^t} d^i(\gamma^q)$ are obtained via PyTorch automatic differentiation.

### Curriculum Learning Training

| Phase | Mode | Pretrained | Epochs | Nodes $M$ |
|-------|------|------------|--------|-----------|
| 1 | Supervised (KL divergence) | None | 10k | 10 |
| 2 | Supervised | Phase 1 | 1.5k | 50 |
| 3 | RL (Policy Gradient) | None/Phase 2 | 50k | 50 |
| 4 | RL (Policy Gradient) | Phase 3 | 10k | 100 |
| FT | RL (Fine-Tuning) | Phase 4 | 1k | {10, 100} |

The supervised phase minimizes the KL divergence between the model policy and the Gibbs target distribution; the reinforcement learning phase uses REINFORCE with a POMO-style baseline.

## Key Experimental Results

### Scalability Speedup
- **Gradient computation**: SPN + Sampling is approximately $200\times$ faster than the original ParaSDM solver at $N=200, M=100$
- **Shortest path inference**: SPN greedy inference is approximately $100\times$ faster than ParaSDM DP at $N=M=200$

### Comparison with Baselines

| Method | Cost (vs. Gurobi) | Speed |
|--------|-------------------|-------|
| GA / SA / CEM | $>10\times$ higher than Deep FLPO | Slower |
| Gurobi (exact) | Optimal | Baseline |
| Deep FLPO (high $\beta$) | Only 2% higher | $\sim 1500\times$ faster |
| Deep FLPO (annealing) | **Matches Gurobi** | $\sim 1500\times$ faster |

### Large-Scale FLPO Scenarios
- $N=200, M=40$: SPN-only is $10\times$ faster than the annealing variant, while the annealing variant achieves 17% lower cost
- $N=800, M=200$: The annealing variant achieves 8% lower cost but is $\sim 3\times$ slower — at this scale, both Gurobi and metaheuristics become practically infeasible
- SPN generalization: Trained on 10–100 nodes, the average optimality gap is only 6% when evaluated on 10–300 nodes

### Ablation Study
- DED + annealing supervision achieves an average gap of ~6% on $M=10\sim300$, outperforming DCAD + annealing by ~4%
- Supervised pretraining reduces the gap for DED by ~3%; without supervised pretraining, DCAD's RL training collapses (400–500% gap)

## Highlights & Insights

- **Deep integration of structural priors and neural networks**: Rather than black-box replacement of classical methods, the algebraic structure of MEP's Gibbs distribution is embedded directly into the SPN architecture
- **First scalable neural solver for ParaSDM**: The original $O(NM^4)$ DP complexity is substantially reduced to $O(NM^2)$ with GPU parallelism
- **Elegant annealing + mixed-sampling strategy**: Uniform exploration at low $\beta$ avoids local optima, while SPN focuses on short paths for exploitation at high $\beta$
- **Cross-scale generalization**: Trained on 10–100 nodes, the model maintains a 6% optimality gap at 300-node test instances
- **Practical utility**: Matches Gurobi's exact solutions while running 1500× faster, supporting large-scale scenarios with 800 agents × 200 nodes

## Limitations & Future Work

- **Euclidean distance costs only**: Road networks, obstacles, and other realistic cost functions are not addressed
- **No facility capacity constraints**: Real charging stations have capacity limits; extensions are needed to handle multi-agent competition
- **Primarily 2D settings**: Although the formulation supports $d$-dimensional spaces, experiments are conducted only on $\mathcal{D} = [0,1]^2$
- **SPN training cost**: The four-phase curriculum training likely requires substantial GPU time, which is not reported in detail
- **Fixed Gibbs temperature during training**: SPN is trained and evaluated at high $\beta$, but the choice of $\beta$ significantly affects performance

## Related Work & Insights

- **Attention Model (Kool et al.)**: Designed for TSP/VRP; cannot jointly optimize continuous parameters $\mathcal{Y}$. SPN extends its decoder to accommodate ParaSDM
- **Pointer Network**: Uses LSTM + attention; SPN adopts Transformer + induced attention for greater efficiency
- **SIL (Self-Improved Learning)**: Extends NCO with linear attention but does not handle parametrized action spaces
- **Original MEP (Srivastava & Salapaka)**: Theoretically rigorous but not scalable at $O(NM^4)$; Deep FLPO preserves MEP's structural guarantees while achieving 100× speedup
- **Gurobi**: Exact but not scalable to large instances; Deep FLPO is 1500× faster at matching accuracy

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work to integrate MEP structural priors with NCO transformers for ParaSDM, opening a new research direction
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers scalability, baseline comparisons, and ablations, but lacks validation on real-world application scenarios
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is rigorous and method derivation is clear, though heavy notation increases reading difficulty
- Value: ⭐⭐⭐⭐⭐ — Addresses the core scalability bottleneck of ParaSDM with strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Nonparametric Teaching of Attention Learners](../../ICLR2026/optimization/nonparametric_teaching_of_attention_learners.md)
- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](../../NeurIPS2025/optimization/effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](../../ICLR2026/optimization/neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)

</div>

<!-- RELATED:END -->
