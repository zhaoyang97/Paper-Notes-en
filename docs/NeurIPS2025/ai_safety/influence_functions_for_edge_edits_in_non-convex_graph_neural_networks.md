---
title: >-
  [Paper Note] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks
description: >-
  [NeurIPS 2025][AI Safety][Influence Functions] This paper proposes influence functions for edge edits applicable to non-convex GNNs. By leveraging the proximal Bregman response function (PBRF)…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Influence Functions"
  - "Graph Neural Networks"
  - "Edge Edits"
  - "Non-Convex Optimization"
  - "Graph Rewiring"
date: 2026-05-08
content_hash: b66f7d1a49b54997
---

# Influence Functions for Edge Edits in Non-Convex Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2506.04694](https://arxiv.org/abs/2506.04694)  
**Code**: None  
**Area**: Graph Neural Networks / AI Safety
**Keywords**: Influence Functions, Graph Neural Networks, Edge Edits, Non-Convex Optimization, Graph Rewiring

## TL;DR

This paper proposes influence functions for edge edits applicable to non-convex GNNs. By leveraging the proximal Bregman response function (PBRF), the method relaxes the convexity assumption and jointly accounts for both parameter shift and message propagation effects, supporting both edge deletion and insertion.

## Background & Motivation

**Background**: The influence of individual edges on GNN behavior remains poorly understood; characterizing edge contributions is critical for interpretability and robustness. Influence functions offer an efficient means of estimating the effect of removing training data on model behavior without retraining.

**Limitations of Prior Work**: Existing graph influence functions (GIF) rely on strict convexity assumptions (which real-world GNNs naturally violate), consider only edge deletion while ignoring edge insertion, and fail to capture changes in message propagation paths induced by edge edits.

**Key Challenge**: Standard influence functions require strict loss convexity, yet common GNN architectures are inherently non-convex; furthermore, edge edits alter the GNN's computational graph structure, whereas conventional influence functions model only parameter changes.

**Goal**: To design accurate influence functions for edge edits in non-convex GNNs, supporting both deletion and insertion while explicitly modeling message propagation effects.

**Key Insight**: A combination of the proximal Bregman response function (PBRF) and chain-rule decomposition, factoring the influence into a parameter shift term and a message propagation term.

**Core Idea**: Relax the convexity assumption via an edge-edit PBRF, and decompose the influence function into parameter shift and message propagation components, providing a unified treatment of both edge deletion and insertion.

## Method

### Overall Architecture

Given a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E}, \mathbf{X})$ and an evaluation function $f(\theta, \mathcal{G})$, the method applies the chain rule to decompose the effect of an edge edit into two components:

$$\frac{df(\theta^*_\epsilon, \mathcal{G}^\epsilon)}{d\epsilon}\bigg|_{\epsilon=0} = \underbrace{\nabla_\theta f \cdot \frac{\partial \theta^*_\epsilon}{\partial \epsilon}}_{\text{Parameter Shift}} + \underbrace{\frac{\partial f}{\partial A^\epsilon}\frac{\partial A^\epsilon}{\partial \epsilon}}_{\text{Message Propagation}}$$

### Key Designs

1. **Edge-Edit PBRF (Proximal Bregman Response Function)**: Extends the PBRF of Bae et al. to the graph edge-edit setting:

    $\theta^*_\epsilon := \arg\min_\theta \frac{1}{N}\sum_{v \in \mathcal{V}_{train}} D_\mathcal{L}(h_v^{\mathcal{G},\theta}, h_v^{\mathcal{G},\theta_s}) + \frac{\lambda}{2}\|\theta - \theta_s\|^2 + \sum_v \epsilon(\mathcal{L}(h_v^{\mathcal{G},\theta}) - \mathcal{L}(h_v^{\mathcal{G}^{-1/N},\theta}))$

    - The first two terms constrain the parameters to remain close to the reference point $\theta_s$ in both output and parameter space.
    - The third term encourages the parameters to perform poorly on the original graph but well on the edited graph, thereby responding to the edge edit.
    - Strict convexity of the loss is no longer required; it suffices for the loss to be convex with respect to the output (naturally satisfied by cross-entropy and MSE).

2. **Parameter Shift Term**: Utilizes the generalized Gauss-Newton Hessian $\mathbf{G} = \mathbf{J}_{h\theta_s}^\top \mathbf{H}_{h_s} \mathbf{J}_{h\theta_s} + \lambda\mathbf{I}$:

    $-\nabla_\theta f(\theta_s, \mathcal{G})^\top \mathbf{G}^{-1} \sum_v (\nabla_\theta \mathcal{L}(h_v^{\mathcal{G},\theta_s}) - \nabla_\theta \mathcal{L}(h_v^{\mathcal{G}^{-1/N},\theta_s}))$

3. **Message Propagation Term**: Explicitly computes the direct effect of edge weight changes on the evaluation function (independent of parameter changes):

    $(2\mathbb{I}[\{u,v\} \in \mathcal{E}] - 1) \cdot N \cdot \left(\frac{\partial f(\theta_s, \mathcal{G})}{\partial A_{uv}} + \frac{\partial f(\theta_s, \mathcal{G})}{\partial A_{vu}}\right)$

   This term is entirely neglected by prior methods such as GIF; experiments demonstrate that it exhibits low correlation with the parameter shift term and is of comparable magnitude.

4. **Multiple Evaluation Metrics**:

    - **Over-squashing measure**: Quantifies information propagation influence by masking $L$-hop neighborhood features.
    - **Over-smoothing (Dirichlet energy)**: Measures the dissimilarity between representations of neighboring nodes.
    - **Validation loss**: Standard cross-entropy.

### Loss & Training

No dedicated training is involved. The LiSSA algorithm is used to approximate the product involving $\mathbf{G}^{-1}$. Influence functions are computed on an already-trained GNN, where $\theta_s$ denotes the post-training parameters.

## Key Experimental Results

### Main Results

Influence prediction accuracy (correlation between predicted and actual influence, Cora dataset, 4-layer GCN):

| Method | Over-squashing Corr. | Dirichlet Energy Corr. | Validation Loss Corr. |
|--------|---------------------|----------------------|----------------------|
| GIF (prior work) | 0.09 | 0.14 | — |
| **Ours (deletion)** | **~0.95** | **~0.95** | **~0.95** |
| **Ours (insertion)** | **~0.95** | **~0.95** | **~0.95** |

Effect of influence-function-guided graph rewiring on test accuracy:

| Method | Cora | CiteSeer | PubMed |
|--------|------|----------|--------|
| GCN (original) | 81.0±0.3 | 69.3±0.5 | 75.6±1.0 |
| Random | 81.1±0.4 | 69.2±0.4 | 75.7±0.8 |
| GIF | 80.9±0.5 | 69.2±0.5 | 75.6±0.9 |
| **Ours (VL)** | **82.1±0.5** | **69.6±0.7** | **76.4±1.3** |

### Ablation Study

Influence prediction accuracy under simultaneous multi-edge edits (Cora, GCN):

| Number of Edges Inserted Simultaneously | Predicted vs. Actual Correlation |
|----------------------------------------|----------------------------------|
| 10 | ~0.90 |
| 20 | ~0.87 |
| 100 | 0.84 |

### Key Findings

- **The message propagation term is indispensable**: The parameter shift and message propagation contributions exhibit low mutual correlation and are of comparable magnitude; both must be modeled jointly.
- **Validation loss is the most effective metric for attack and improvement**: Edge edits guided by validation loss improve test accuracy across all three datasets, whereas optimizing Dirichlet energy or over-squashing does not necessarily translate into better classification performance.
- **Homophily analysis**: Regardless of whether the graph is homophilic or heterophilic, inserting homophilic edges (connecting nodes of the same class) is consistently beneficial, consistent with GCN's homophily bias.
- **Side effects of graph rewiring methods**: Edge insertions by BORF and FoSR alleviate over-squashing but exacerbate over-smoothing.

## Highlights & Insights

- **Unified framework**: The first influence function framework to handle both edge deletion and insertion in a unified manner, applicable to non-convex GNNs.
- **Elegant decomposition**: The influence of graph edits is cleanly factored into two independent and complementary dimensions—parameter shift and message propagation.
- **Multi-perspective analysis tool**: Provides a unified analysis across three perspectives—over-squashing, over-smoothing, and validation loss—revealing previously unnoticed side effects of existing graph rewiring methods.
- **Adversarial attack capability**: Influence-function-guided adversarial attacks based on validation loss outperform dedicated attack methods (DICE, PRBCD).

## Limitations & Future Work

- Accuracy degrades when editing a large number of edges simultaneously (first-order approximation error), with correlation dropping to 0.84 at 100 edges.
- Scalability to deeper GNNs remains to be validated.
- LiSSA-based inverse Hessian approximation incurs non-trivial computational overhead, which may become a bottleneck for large-scale graphs.
- Evaluation is currently limited to GCN/GAT/ChebNet; more complex GNN architectures such as Graph Transformers are not covered.

## Related Work & Insights

- **Classical influence functions** (Koh & Liang): Require strict convexity and are unreliable for deep networks.
- **PBRF** (Bae et al.): Relaxes the convexity assumption via Bregman divergence and damping; the present work extends it to edge edits in the graph setting.
- **GIF** (Chen et al., Wu et al.): Pioneering influence functions for graphs, but consider only parameter changes; this work supplements the message propagation component, critically improving prediction accuracy (0.09→0.95).
- Diverse applications of influence functions: model interpretability, data valuation, adversarial analysis, and machine unlearning.

## Rating

- Novelty: ⭐⭐⭐⭐ The edge-edit influence decomposition and edge-edit PBRF design are novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets, three GNN architectures, three evaluation metrics, and multiple application scenarios.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, and experimental design closely mirrors the theoretical framework.
- Value: ⭐⭐⭐⭐ Provides practical tools for interpretability and robustness analysis of GNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[NeurIPS 2025\] Robust Graph Condensation via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Improved Balanced Classification with Theoretically Grounded Loss Functions](improved_balanced_classification_with_theoretically_grounded_loss_functions.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)

</div>

<!-- RELATED:END -->
