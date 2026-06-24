---
title: >-
  [Paper Note] Optimal Auction Design in the Joint Advertising
description: >-
  [ICML2025][Joint Advertising] This paper proposes an optimal auction mechanism for joint advertising scenarios (where retailers and suppliers co-bid for ad slots): for a single slot, a Myerson-style closed-form optimal solution is derived; for multiple slots, a BundleNet neural network is designed to construct IC constraints on a per-bundle basis, maximizing platform revenue while ensuring approximate incentive compatibility.
tags:
  - "ICML2025"
  - "Joint Advertising"
  - "Optimal Auction"
  - "BundleNet"
  - "Incentive Compatibility"
  - "Neural Network Mechanism Design"
date: 2026-05-08
content_hash: b0b16922d479f55b
---

# Optimal Auction Design in the Joint Advertising

**Conference**: ICML2025  
**arXiv**: [2507.07418](https://arxiv.org/abs/2507.07418)  
**Code**: Not released  
**Area**: Auction Design / Automated Mechanism Design  
**Keywords**: Joint Advertising, Optimal Auction, BundleNet, Incentive Compatibility, Neural Network Mechanism Design

## TL;DR

This paper proposes an optimal auction mechanism for joint advertising scenarios (where retailers and suppliers co-bid for ad slots): for a single slot, a Myerson-style closed-form optimal solution is derived; for multiple slots, a BundleNet neural network is designed to construct IC constraints on a per-bundle basis, maximizing platform revenue while ensuring approximate incentive compatibility.

## Background & Motivation

- **Joint Advertising** is an emerging online advertising paradigm (already deployed on platforms like Facebook): while traditional advertising involves only unilateral bidding by retailers, joint advertising allows retailers and suppliers (or brands) to co-bid for an ad slot, with the platform charging both parties.
- Existing methods suffer from two major limitations:
  1. **VCG/JAMA-like methods** establish incentive compatibility (IC) only along a single dimension (either brand or retailer), ignoring the bundle structure, which leaves room for optimization in allocation efficiency.
  2. **JRegNet** (a neural-network-based method built on RegretNet) exhibits poor generalization, with performance collapsing even below VCG in large-scale, complex bipartite graph scenarios.
- Key Challenge: In joint advertising, a bundle is co-determined by the strategies of two separate bidders. Because the bundle itself lacks an independent bidding strategy, directly defining bundle-level IC constraints is highly challenging.

## Method

### Problem Modeling

In a joint advertising system, the set of retailers $R$ and the set of suppliers $S$ are disjoint, and the set of advertisements $E \subseteq R \times S$ forms a bipartite graph $G=(R,S,E)$. Each advertisement $e=(r,s)$ links a retailer and a supplier. The click-through rates (CTRs) for $m$ slots are denoted by $\boldsymbol{\lambda}=(\lambda_1,\dots,\lambda_m)$, satisfying $\lambda_1 \geq \cdots \geq \lambda_m$.

An auction mechanism $\mathcal{M}=(x,p)$ consists of allocation rules and payment rules:

$$x^e(v) = x_r^e(v) = x_s^e(v), \quad p^e(v) = p_r^e(v) + p_s^e(v)$$

It must satisfy both **Incentive Compatibility (IC)** (truthful bidding is a dominant strategy) and **Individual Rationality (IR)** (non-negative utility for participation).

### Optimal Single-Slot Mechanism (Theorem 4.3)

Under regular distribution assumptions, the virtual value function is defined as:

$$c_i(v_i) = v_i - \frac{1-F_i(v_i)}{f_i(v_i)}$$

The virtual value of bundle $e=(r,s)$ is: $c^e(v_r, v_s) = c_r(v_r) + c_s(v_s)$

The optimal mechanism takes the form of a **step function**: allocation occurs when the bid exceeds a critical value $\hat{v}_i(v_{-i})$, and the payment is exactly equal to this critical value. The critical threshold is determined such that the virtual value of the optimal bundle containing bidder $i$ must simultaneously satisfy: (1) being no less than the platform reserve price $v_0$; and (2) being no less than the virtual value of any bundle that does not contain $i$.

### Multi-Slot BundleNet

#### Bundle-level IC Constraints (Core Innovation)

Unlike JRegNet which constructs separate IC constraints for each bidder, this work proposes a **bundle-based** IC constraint:

$$rgt^e(w) = \mathbb{E}\left[\max_{v_r'} \{u_r^e(\text{misreport}) - u_r^e(\text{truth})\} + \max_{v_s'}\{u_s^e(\text{misreport}) - u_s^e(\text{truth})\}\right]$$

**Lemma 5.1** guarantees that $\sum_{i \in R \cup S} rgt_i(w) \leq \sum_{e \in E} rgt^e(w)$, which means individual IC constraints approach zero as the bundle-level constraints approach zero.

#### Network Architecture

1. **Graph Feature Fusion**: Aggregates bipartite node features into edge features
    - Divided Bids: $DB^e = [X_r, X_s] \in \mathbb{R}^{2m}$ (concatenation)
    - Stacked Bids: $SB^e = X_r + X_s \in \mathbb{R}^m$ (summation)
2. **Allocation Network**: MLP processes $SB^E$ $\to$ doubly stochastic matrix approach (taking the minimum of row/column softmax) to ensure allocation feasibility.
3. **Payment Network**: MLP processes $DB^E$ $\to$ Sigmoid structures normalize payment coefficients $\tilde{p} \in [0,1]$, multiplied by the allocation-weighted bid to obtain the actual payment, structurally guaranteeing IR.

#### Optimization and Training

Augmented Lagrangian method is adopted to solve the constrained optimization:

$$\mathcal{L}_\rho(w;\mu) = -\text{rev}(w) + \sum_{e \in E}\mu_e \cdot \widehat{rgt}^e(w) + \frac{\rho}{2}\sum_{e \in E}(\widehat{rgt}^e(w))^2$$

Network parameters $w$, misreported bids $v'$, and Lagrange multipliers $\mu$ are updated alternately.

## Key Experimental Results

| Scenario | Method | U2 | U3 | U4 | U5 | E2 | E3 | E4 | E5 |
|------|------|------|------|------|------|------|------|------|------|
| Single-slot | **BundleNet** | 0.529 | 0.668 | 0.781 | 0.880 | 0.425 | 0.546 | 0.635 | 0.722 |
| Single-slot | Optimal | 0.525 | 0.671 | 0.783 | 0.882 | 0.425 | 0.548 | 0.647 | 0.738 |
| Single-slot | JRegNet | 0.562 | 0.729 | 0.779 | 0.788 | 0.473 | 0.589 | 0.631 | 0.694 |
| Single-slot | RVCG | 0.381 | 0.600 | 0.746 | 0.861 | 0.282 | 0.465 | 0.593 | 0.704 |

| Scenario | Method | U5×5 | U8×5 | U10×5 | U12×5 |
|--------|------|------|------|-------|-------|
| 5-slot | **BundleNet** | 1.498 | 2.089 | 2.405 | 2.650 |
| 5-slot | JRegNet | 1.497 | 1.935 | 1.962 | 1.997 |
| 5-slot | RVCG | 0.842 | 1.883 | 2.280 | 2.587 |

- In single-slot settings, BundleNet almost perfectly approximates the theoretical optimal solution. Although JRegNet sometimes yields higher revenue, it violates the optimal mechanism (cheats on IC constraints).
- In multi-slot settings, BundleNet-5 outperforms RVCG and JRegNet across the board, with the advantage becoming more pronounced as the number of bundles increases.
- All methods maintain an IC violation of < 0.001.

## Highlights & Insights

1. **Solid Theoretical Contribution**: It derives the necessary and sufficient conditions for the optimal single-slot auction under joint advertising for the first time (Theorem 4.3), successfully extending Myerson's lemma to bundle contexts.
2. **Elegant Bundle-level IC Constraints**: Instead of directly defining strategies for bundles (which is impossible since they lack independent strategies), the paper proves via Lemma 5.1 that bundle-level constraints upper-bound individual-level constraints, resolving the core challenge.
3. **Problem-aligned Network Design**: The graph feature fusion leverages the node-edge correspondence in bipartite graphs. Meanwhile, the allocation network employs doubly stochastic matrices to guarantee feasibility, and the payment network uses a Sigmoid structure to structurally ensure IR.
4. **Closed-loop Experimental Validation**: Single-slot experiments validate correctness by comparing with the theoretical optimum, while multi-slot experiments demonstrate practical superiority.

## Limitations & Future Work

1. **Regular Distribution Assumption**: The single-slot optimal results rely on distribution regularity (monotonically increasing virtual values). Non-regular distributions would require ironing processes.
2. **Single-parameter Bidders Only**: Each bidder has only one private value, which has not been extended to multi-item or multi-parameter settings.
3. **Randomly Generated Bipartite Graph Structures**: The retailer-supplier relationship matrices in the experiments are randomly generated, and have not yet been evaluated end-to-end in real-world large-scale advertising systems.
4. **Approximate IC Guarantees**: Although the empirical IC violation is extremely low, the work lacks a theoretical bound on approximate IC.
5. **Scalability**: Computational overhead and performance scaling under larger scenarios (e.g., hundreds of bundles) are not reported.

## Related Work & Insights

- **Myerson (1981)**: The classic single-parameter optimal auction theory. This work extends it to the bundle scenarios of joint advertising.
- **RegretNet (Dütting et al., 2024)**: Pioneering work utilizing neural networks to approximate optimal mechanisms. BundleNet improves upon its architecture and constraints.
- **JRegNet (Zhang et al., 2024)**: A RegretNet variant for joint advertising, representing the direct benchmark for this study.
- **JAMA (Ma et al., 2024)**: An AMA-based approach for joint advertising that established the joint advertising setting.

## Rating

- Novelty: ⭐⭐⭐⭐ (the bundle-level IC constraint and single-slot optimal characterization are highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive coverage of single/multi-slot settings and various distributions, though lacks real-world deployment evaluation)
- Writing Quality: ⭐⭐⭐⭐ (clear structure and robust connection between theory and experiments)
- Value: ⭐⭐⭐⭐ (joint advertising is a newly deployed practical setting, with dual contributions in both theory and algorithms)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[ICML 2025\] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)

</div>

<!-- RELATED:END -->
