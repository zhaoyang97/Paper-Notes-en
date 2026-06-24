---
title: >-
  [Paper Note] Lyapunov Stable Graph Neural Flow
description: >-
  [CVPR 2025][AI Safety][Graph Neural Networks] Integrates Lyapunov stability theory (integer-order and fractional-order) with graph neural flows. By utilizing a learnable Lyapunov function and a projection mechanism, it dynamically constrains GNN feature trajectories within a stable space. This provides the first provable adversarial robustness guarantee for graph neural flows and is orthogonally stackable with adversarial training.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Graph Neural Networks"
  - "Adversarial Robustness"
  - "Lyapunov Stability"
  - "Fractional-order Differential Equations"
  - "Control Theory"
date: 2026-05-08
content_hash: 0f2d19f64afe77a1
---

# Lyapunov Stable Graph Neural Flow

**Conference**: CVPR 2025  
**arXiv**: [2603.12557](https://arxiv.org/abs/2603.12557)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Graph Neural Networks, Adversarial Robustness, Lyapunov Stability, Fractional-order Differential Equations, Control Theory

## TL;DR
Integrates Lyapunov stability theory (integer-order and fractional-order) with graph neural flows. By utilizing a learnable Lyapunov function and a projection mechanism, it dynamically constrains GNN feature trajectories within a stable space. This provides the first provable adversarial robustness guarantee for graph neural flows and is orthogonally stackable with adversarial training.

## Background & Motivation

### Background
**Background**: Dynamical system-based GNNs (such as GRAND, GREAD, FROND) model node feature updates as differential equation evolution, but remain vulnerable to adversarial perturbations.

**Limitations of Prior Work**: Traditional defenses (adversarial training, data purification) incur high computational overhead and lack theoretical guarantees. Existing robustness analyses from a dynamical systems perspective (such as Hamiltonian energy conservation) only provide heuristic stability rather than rigorous proof.

**Key Challenge**: The fundamental reason behind GNN failures on adversarial examples is neglecting to preserve the underlying stability structure, making the model dynamics prone to deviating from stable equilibrium points under perturbations.

**Goal**: Can Lyapunov stability theory be leveraged to provide rigorous theoretical stability guarantees for graph neural flows?

**Key Insight**: Lyapunov's direct method—constructing a positive-definite and decreasing Lyapunov function can establish system stability without explicitly solving the differential equations.

**Core Idea**: Utilize a learnable Lyapunov function paired with a projection mechanism to constrain the dynamics of graph neural flows onto a stable manifold.

## Method

### Overall Architecture
Node feature evolution is modeled as an integer-order or fractional-order ODE: $\frac{d\mathbf{U}(t)}{dt} = \mathcal{F}(t, \mathbf{U}(t); \mathbf{W})$ or $D_t^\beta \mathbf{U}(t) = \mathcal{F}(t, \mathbf{U}(t); \mathbf{W})$. The Lyapunov stability module projects the base dynamics into a stable space satisfying the conditions of Lyapunov's direct method. An equilibrium separation classification layer maximizes the distance between the equilibrium points of different classes.

### Key Designs

1. **Learnable Lyapunov Function**:

    - **Function**: Construct an energy function that satisfies Lyapunov stability conditions.
    - **Mechanism**: Implement the Lyapunov function $V(t, \mathbf{U})$ using an Input Convex Neural Network (ICNN) to guarantee positive definiteness ($V(\mathbf{U}^*) = 0$, $V > 0$ elsewhere) and decay ($\dot{V} \leq -cV$). The ICNN structure naturally guarantees convexity, enforced via non-negative weight projection constraints.
    - **Design Motivation**: Hand-designed Lyapunov functions struggle to adapt to complex graph data, whereas a learnable approach balances theoretical guarantees and flexibility.

2. **Stability Projection Mechanism**:

    - **Function**: Project the dynamics of the base graph neural flow onto a stable manifold satisfying Lyapunov conditions.
    - **Mechanism**: Given the base dynamics $\mathcal{F}$, if the current direction violates the Lyapunov condition ($\dot{V} > -cV$), it is projected and corrected to the nearest compliant direction $\hat{\mathcal{F}}$. The projection guarantees that the system still converges to a stable equilibrium under adversarial perturbations.
    - **Design Motivation**: Hard constraints preserve stability better than soft regularization—regularization can be overwhelmed by other losses, while projection is always satisfied.

3. **Equilibrium Separation Classification Layer**:

    - **Function**: Maximize the distance between the equilibrium points of different classes.
    - **Mechanism**: Design a specific classification layer to ensure that the equilibriums $\mathbf{U}^*$ of different classes are sufficiently separated in the feature space.
    - **Design Motivation**: If equilibrium points of different classes are too close, the system remains prone to being affected by cross-class perturbations, even if stable.

### Loss & Training
- The Lyapunov stability module is embedded as a hard constraint into the ODE solving process and jointly trained with the standard cross-entropy loss.
- The method is orthogonal to adversarial training and can be used conjunctively.
- ICNN weights are guaranteed to be non-negative via ReLU projection and clamped after each forward pass.
- The ODE integration time is $T=1.0$, using the Dormand-Prince (dopri5) adaptive solver.
- The fractional-order version uses Grünwald-Letnikov discretization, with the order $\beta \in (0,1)$ as a learnable parameter.
- The stability constant $c$ is tuned on the validation set, with typical values inside the range of 0.1–1.0.

## Key Experimental Results

### Main Results

| Method | Clean Accuracy | Metattack | DICE | Description |
|------|---------|----------|------|------|
| GCN | Baseline | Significant Drop | Significant Drop | No Defense |
| GRAND | Moderate | Moderate Drop | Moderate Drop | Dynamical System Baseline |
| HANG | Moderate | Slightly Better | Slightly Better | Energy Conservation |
| **IL-GNN (Ours)** | **High** | **Significantly Better** | **Significantly Better** | Integer-order Lyapunov |
| **FL-GNN (Ours)** | **High** | **Best** | **Best** | Fractional-order Lyapunov |

### Ablation Study

| Configuration | Description |
|------|------|
| Base Flow (No Lyapunov) | Degrades significantly under adversarial attacks |
| + Lyapunov Projection | Substantial robustness boost |
| + Equilibrium Separation | Further improves inter-class discrimination |
| + Adversarial Training Stack | Cumulative effect, achieving optimal robustness |

### Key Findings
- Both integer-order and fractional-order Lyapunov stability modules significantly improve robustness, with the fractional-order version performing slightly better (capturing historical states more effectively via the memory effect).
- The method is orthogonal to adversarial training—stacking them yields cumulative rather than redundant improvements, as the Lyapunov constraints and adversarial training operate at different levels.
- Consistently effective across various attack scenarios, including black-box/white-box and poisoning/evasion attacks.
- Clean accuracy is close to or even exceeds the baseline—stability constraints do not compromise standard performance.

## Highlights & Insights
- **The bridge between control theory and graph learning** is elegant: Lyapunov theory is fundamentally a framework concerning whether a system returns to equilibrium under perturbations, making it naturally aligned with adversarial robustness.
- **Hard Constraints vs. Soft Regularization**: The projection mechanism guarantees that stability conditions are strictly met at all times, unlike loss-based regularizations whose weights can be overwhelmed.
- **Fractional-order Extension** holds high theoretical depth: Generalizing from integer-order to fractional-order Mittag-Leffler stability criteria.
- **Orthogonality** is an important practical attribute: It can serve as a plug-and-play component to enhance any existing defense.

## Limitations & Future Work
- The expressive power of ICNNs is constrained by convexity requirements, which might limit their ability to model complex Lyapunov surfaces.
- The projection mechanism introduces additional computational overhead during ODE solving.
- Validated only on node classification tasks; graph classification, link prediction, and other tasks remain unexplored.
- Numerical solvers for fractional-order ODEs are inherently computationally intensive.
- The design of the equilibrium separation classification layer is relatively heuristic and lacks deep theoretical analysis.

## Related Work & Insights
- **vs. HANG**: Both adopt an energy/dynamical-system perspective, but HANG uses Hamiltonian energy conservation while this work employs Lyapunov energy decay—decay corresponds more directly to stability than conservation.
- **vs. Adversarial Training**: Data augmentation level vs. dynamic constraint level, offering orthogonal and complementary benefits.
- **vs. FROND**: A baseline fractional-order flow model, upon which this work builds to provide Lyapunov stability guarantees.
- **Broader Inspiration**: The Lyapunov framework can be generalized to other ODE-based networks (e.g., Neural ODEs, continuous normalizing flows), acting as a general-purpose robustness tool for continuous deep models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Incorporates Lyapunov's direct method into GNNs for the first time, delivering significant theoretical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple attack scenarios with cumulative stacking experiments, though the datasets used are standard.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, but possesses a steep learning curve for readers without a control theory background.
- Value: ⭐⭐⭐⭐ Establishes a novel theoretical foundation for GNN robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](../../NeurIPS2025/ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[NeurIPS 2025\] Robust Graph Condensation via Classification Complexity Mitigation](../../NeurIPS2025/ai_safety/robust_graph_condensation_via_classification_complexity_mitigation.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](../../AAAI2026/ai_safety/fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[NeurIPS 2025\] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization](../../NeurIPS2025/ai_safety/dual-flow_transferable_multi-target_instance-agnostic_attacks_via_in-the-wild_ca.md)

</div>

<!-- RELATED:END -->
