---
title: >-
  [Paper Note] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees
description: >-
  [NeurIPS 2025][Optimization][Neural Combinatorial Optimization] This paper reveals that existing NCO methods severely overfit to fixed constraint tightness (e.g., fixed vehicle capacity $C=50$ in CVRP), and proposes a Variable Constraint Tightness (VCT) training scheme along with a Multiple Expert Module (MEM), enabling models to effectively handle the full spectrum of constraints from extremely tight to extremely loose.
tags:
  - NeurIPS 2025
  - Optimization
  - Neural Combinatorial Optimization
  - Vehicle Routing Problem
  - Constraint Tightness
  - Multiple Expert Module
  - Generalization
date: 2026-05-08
content_hash: 07063d7742dbbcdd
---

# Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees

**Conference**: NeurIPS 2025
**arXiv**: [2505.24627](https://arxiv.org/abs/2505.24627)
**Code**: [GitHub](https://github.com/CIAM-Group/Rethinking_Constraint_Tightness)
**Area**: Optimization
**Keywords**: Neural Combinatorial Optimization, Vehicle Routing Problem, Constraint Tightness, Multiple Expert Module, Generalization

## TL;DR

This paper reveals that existing NCO methods severely overfit to fixed constraint tightness (e.g., fixed vehicle capacity $C=50$ in CVRP), and proposes a Variable Constraint Tightness (VCT) training scheme along with a Multiple Expert Module (MEM), enabling models to effectively handle the full spectrum of constraints from extremely tight to extremely loose.

## Background & Motivation

NCO methods for VRP are typically trained with fixed constraint values (e.g., CVRP100 with fixed capacity $C=50$ and average demand 5), achieving strong performance on in-distribution test data. However, constraint tightness is highly variable in real-world applications.

**Limitations of Prior Work**: When constraint tightness deviates from the training setting, existing NCO models suffer severe performance degradation. POMO achieves a gap of only 3.66% at $C=50$, but this surges to 20.26% (5.5×) at $C=10$ (tight constraints) and 34.12% (9.3×) at $C=500$ (loose constraints). All mainstream NCO methods (AM, POMO, BQ, LEHD, etc.) exhibit this problem, with average gaps expanding by at least 1.7×.

**Root Cause**: Varying constraint tightness fundamentally alters the structure of optimal solutions:
- When $C$ is very small (e.g., $C=10$), each vehicle can serve only 2–3 customers, and the optimal solution structure resembles Open VRP (OVRP).
- When $C$ is very large (e.g., $C=500$), a single vehicle can serve all customers, and the optimal solution structure resembles TSP.

This implies that the model must learn fundamentally different solving strategies. A model trained at $C=50$ learns neither the short-route strategy of OVRP nor the long-tour strategy of TSP.

**Core Idea**: By uniformly sampling different constraint tightness levels during training, the model learns generalizable strategies across constraints. A Multiple Expert Module further allows different expert layers to specialize in different tightness ranges, avoiding conflicting optimization directions.

## Method

### Overall Architecture

Built upon the LEHD (Light Encoder Heavy Decoder) backbone, improvements are made at two levels: (1) Variable Constraint Tightness training (VCT); and (2) Multiple Expert Module (MEM).

### Key Designs

1. **Variable Constraint Tightness Training (VCT)**: During training, constraint tightness is sampled independently for each instance from $[C_{min}, C_{max}]$. For example, capacity is randomly drawn from $\{10, 11, \ldots, 500\}$. A key design choice is **instance-level** sampling rather than batch-level sampling — different instances within the same batch may have different capacities. This outperforms batch-consistent sampling by providing richer gradient signals.

   Training data is annotated by the HGS solver, with each instance having a randomly assigned capacity and corresponding optimal solution. The LEHD model is trained via supervised learning.

2. **Multiple Expert Module (MEM)**: After $L$ stacked attention layers in the LEHD decoder, a Multiple Expert Module is appended. It consists of a gating mechanism $G(\cdot)$ and $m$ expert layers $\{E_1, E_2, \ldots, E_m\}$. The gating mechanism hard-routes each instance to the corresponding expert based on constraint tightness $C_k$:

$$P_t = \text{MEM}(H_t, C_k) = \sum_{i=1}^{m} G(C_k) E_i(H_t)$$

The gate produces a one-hot vector, where the $i$-th expert handles instances with tightness in $[(i-1)\beta, i\beta]$, with $\beta = |C_{max} - C_{min}| / m$. Each expert layer internally contains $m_e$ stacked attention layers and a final probability prediction layer. This hard-routing design is simple and efficient, avoiding gradient interference across different tightness levels that arises in soft routing.

3. **Theoretical Connection Between Constraint Tightness and Problem Similarity**: Experiments validate the hypothesis that CVRP under different constraints is equivalent to different problems. The OVRP-specialized POMO achieves a gap of 1.9% at $C=10$, far outperforming all CVRP models (best: 17.44%); the TSP-specialized POMO achieves a gap of 2.5% at $C=500$, also surpassing CVRP models (best: 6.8%). This provides structural justification for the VCT design.

### Loss & Training

Supervised learning with the Adam optimizer (initial learning rate 1e-4, decay rate 0.9 per epoch) for 40 epochs. Training uses 1 million CVRP100 instances with batch size 512. Each instance's capacity is randomly sampled from $\{10, 11, \ldots, 500\}$, with labels generated by the HGS solver.

Model configuration: embedding dimension 192, FFN hidden size 512, query dimension 16, 12-head attention, $L=6$ decoder attention layers, $m=3$ experts, each with $m_e=3$ attention layers.

## Key Experimental Results

### Main Results (CVRP100 across different capacities)

| Method | C=10 | C=50 | C=100 | C=200 | C=300 | C=400 | C=500 | Avg. Gap |
|--------|------|------|-------|-------|-------|-------|-------|----------|
| POMO | 20.26% | 3.66% | 11.17% | 22.45% | 29.56% | 30.80% | 34.12% | 21.72% |
| BQ | 18.02% | 3.25% | 3.48% | 4.53% | 6.53% | 6.54% | 8.16% | 7.22% |
| LEHD | 36.56% | 4.22% | 4.87% | 4.18% | 5.78% | 4.92% | 6.82% | 9.62% |
| Prev. SOTA (per dataset) | 7.47% | 3.25% | 3.48% | 4.18% | 5.78% | 4.92% | 6.82% | 5.13% |
| **Ours** | **1.54%** | 3.82% | 3.67% | **1.49%** | **0.87%** | **0.75%** | **0.87%** | **1.86%** |

### Ablation Study

| Configuration | C=10 | C=50 | C=500 | Avg. Gap |
|---------------|------|------|-------|----------|
| Base Model (trained at C=50) | 45.64% | 3.72% | 7.29% | 10.78% |
| Base Model + MEM | 37.03% | 3.28% | 6.11% | 8.95% |
| Base Model + VCT | 1.88% | 4.51% | 1.25% | 2.41% |
| Base Model + VCT + MEM | **1.54%** | **3.82%** | **0.87%** | **1.86%** |

### Extension to CVRPTW

| Method | α=0.2 (tight) | α=1.0 (moderate) | α=3.0 (very loose) | Avg. Gap |
|--------|--------------|------------------|-------------------|----------|
| POMO | 14.95% | 11.39% | 33.56% | 19.58% |
| POMO+VCT | 9.80% | 12.37% | 11.23% | 12.14% |
| POMO+VCT+MEM | **9.13%** | 11.91% | **10.37%** | **11.58%** |

### Key Findings

- **VCT is the primary contributor**: VCT alone reduces average gap from 10.78% to 2.41% (a 78% reduction), with MEM further improving it to 1.86%.
- **MEM alone has limited effect** (8.95% vs. 10.78%), but consistently improves performance across all constraint ranges when combined with VCT.
- In the $C=300$–$500$ range, the proposed method achieves gaps below 1%, approaching HGS solver quality.
- **Generality of the approach**: The same methodology successfully applies to POMO + CVRPTW, reducing average gap from 19.58% to 11.58%.
- Overfitting is more severe under tight constraints — AM reaches a gap of 45.16% at $C=10$.

## Highlights & Insights

- **Depth of problem diagnosis**: Beyond identifying the overfitting phenomenon, the paper explains its root cause through controlled experiments with OVRP and TSP — different constraint tightness levels fundamentally correspond to different optimization problems.
- **Simplicity and efficiency of the method**: VCT merely changes the training data sampling strategy (zero additional computation); MEM only appends a few attention layers to the decoder tail (minimal parameter overhead), yet both yield significant improvements.
- **Hard-routing design choice**: Compared to soft mixture-of-experts with its computational overhead and training instability, one-hot hard-routing is straightforward and sufficiently effective.
- **Slight regression at $C=50$/$100$** indicates that VCT trades per-point accuracy for broader strategy coverage, which is a reasonable tradeoff.

## Limitations & Future Work

- VCT employs uniform sampling; adaptive sampling (e.g., adjusting sampling frequency based on current performance at each tightness level) may yield further improvements.
- Hard routing in MEM may cause knowledge isolation between experts; soft routing or shared-representation approaches are worth exploring.
- Performance at $C=50$/$100$ is slightly below dedicated baselines, suggesting the method is best suited for "one model covers all" scenarios.
- Only CVRP and CVRPTW are evaluated; applicability to other constrained combinatorial optimization problems such as scheduling and bin packing remains to be verified.

## Related Work & Insights

- Compared to MVMoE (mixture-of-experts for diverse VRPs), the proposed MEM is simpler (hard routing vs. learnable gating) and specifically targets the constraint tightness dimension.
- Complementary to NCO generalization research (e.g., INViT): INViT addresses generalization across problem scales, while this work addresses generalization across constraint values.
- Insight: The NCO community should pay greater attention to the distributional coverage of training data; fixed constraint settings may constitute a pervasive and underappreciated pitfall.

## Rating

- Novelty: ⭐⭐⭐⭐ — In-depth problem diagnosis; method is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons (9 NCO methods × 7 capacity levels), clear ablations, and CVRPTW extension for validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous experimental design and analysis.
- Value: ⭐⭐⭐⭐ — Exposes a systematically overlooked issue in NCO training; solution is simple and practical.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning to Insert for Constructive Neural Vehicle Routing Solver](learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[AAAI 2026\] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](../../AAAI2026/optimization/bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)

<!-- RELATED:END -->
