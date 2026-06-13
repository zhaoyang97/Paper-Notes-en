---
title: >-
  [Paper Note] URS: A Unified Neural Routing Solver
description: >-
  [ICML 2026][Optimization][Routing Problems] The authors propose the Unified Data Representation (UDR) and Mixed Bias Module (MBM) to replace problem enumeration—enabling a single neural model to achieve zero-shot general…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Routing Problems"
  - "Zero-shot Generalization"
  - "Unified Representation"
  - "Multi-task Learning"
date: 2026-05-08
content_hash: 882a61c5d8c1195c
---

# URS: A Unified Neural Routing Solver

**Conference**: ICML 2026  
**arXiv**: [2509.23413](https://arxiv.org/abs/2509.23413)  
**Code**: https://github.com/CIAM-Group/URS  
**Area**: Combinatorial Optimization / Neural Solvers / Vehicle Routing Problems  
**Keywords**: Routing Problems, Zero-shot Generalization, Unified Representation, Multi-task Learning

## TL;DR
The authors propose the Unified Data Representation (UDR) and Mixed Bias Module (MBM) to replace problem enumeration—enabling a single neural model to achieve zero-shot generalization across 110 VRP variants (99 unseen) without fine-tuning.

## Background & Motivation

**Background**: Vehicle Routing Problems (VRP) are critical combinatorial optimization tasks. Recent Neural Combinatorial Optimization (NCO) methods have shown outstanding performance on specific problems. However, multi-task neural solvers primarily adopt two strategies: (1) constraint combination methods (treating VRP variants as combinations of different constraints, relying on manually predefined problem labels); and (2) adapter fine-tuning (reducing retraining costs but still requiring additional fine-tuning, thus failing to achieve zero-shot generalization).

**Limitations of Prior Work**: Existing problem boundaries are fixed by manually specified constraint sets, which cannot cover the open-ended VRP constraint space. Maintaining a problem taxonomy requires significant domain expertise. Furthermore, the constraint space is combinatorial and open; simply enumerating combinations leads to model bloat.

**Key Challenge**: How can a single model simultaneously solve dozens or even hundreds of VRP variants without relying on problem enumeration, while maintaining generalization capabilities for completely unseen variants?

**Goal**: To build the first neural solver capable of handling 100+ VRP variants—including 99 unseen ones—using a single model without any fine-tuning.

**Key Insight**: Although VRP variants vary significantly, they share a universal structural representation at the data level. We should rethink the problem from the perspective of data representation unification rather than constraint enumeration.

**Core Idea**: Replace discrete problem labels with a Unified Data Representation (UDR) and a multi-hot problem representation ($\bm{\lambda}$), achieving cross-problem generalization through data unification instead of problem enumeration.

## Method

### Overall Architecture
URS is based on an encoder-decoder architecture (using AM). The core innovation lies in unification across three levels: (1) UDR at the data level; (2) MBM at the encoding level to capture multiple geometric priors; and (3) conditional parameter generation based on the multi-hot problem representation $\lambda$ at the decoding level. This allows the model to adapt to different problem constraints rather than explicitly encoding them.

### Key Designs

1.  **Unified Data Representation (UDR)**:
    - **Function**: Unifies inputs of different VRP variants into the same feature space to avoid problem enumeration.
    - **Mechanism**: For each node $\mathbf{u}_i=\{\bm{\rho}_i, \bm{\omega}_i, \bm{\xi}_i\}$—position identity $\bm{\rho}_i=\{\eta_i, x_i, y_i\}$ handles symmetric/asymmetric cases; a unified attribute set $\bm{\omega}_i=\{\delta_i, \epsilon_i, \mu_i, e_i, l_i, s_i\}$ corresponds to demand, reward, penalty, time windows, etc. (zero-padded if missing); and node type identity $\bm{\xi}_i \in \{0,1\}^5$. A multi-hot representation $\bm{\lambda}$ is derived to indicate the active features of the current problem.
    - **Design Motivation**: By replacing "problem enumeration" with "data unification," new constraints can be implicitly enforced via mask functions without modifying the model architecture.

2.  **Mixed Bias Module (MBM)**:
    - **Function**: Efficiently captures multiple geometric priors (distance matrices, relationship matrices) in the encoder to improve node embedding quality.
    - **Mechanism**: Replaces standard attention with three-way parallel attention. It separately calculates attention for outgoing distance $\bm{D}$, incoming distance $\bm{D}^{\mathrm{T}}$, and an optional relationship matrix $\bm{R}$. The three outputs are then concatenated and fused via a learned matrix $W^O$: $\hat{\mathbf{h}}_i^{(\ell)}=[\bar{\mathbf{h}}_i^{(0)}, \bar{\mathbf{h}}_i^{(1)}, \bar{\mathbf{h}}_i^{(2)}]W^O$.
    - **Design Motivation**: Traditional methods (like MatNet) handle asymmetry through parallel layers, leading to high complexity; MBM uses a unified design to handle symmetric, asymmetric, and relationship constraints at a low cost.

3.  **Conditional Parameter Generation**:
    - **Function**: Dynamically generates decoder parameters for each problem based on the multi-hot representation $\bm{\lambda}$, enabling zero-shot adaptation.
    - **Mechanism**: A lightweight bias network $\mathrm{BIAS}(\bm{\lambda})=\max(1, (\bm{\lambda}W_1+\mathbf{b}_1)W_2+\mathbf{b}_2)$ adjusts the fusion degree of multiple priors in MBM; a hyper-network $\mathrm{WEIGHT}(\bm{\lambda})$ generates the decoder projection matrices $W_Q(\bm{\lambda}), W_K(\bm{\lambda}), W_V(\bm{\lambda})$.
    - **Design Motivation**: Compared to adapter fine-tuning which requires subsequent training, conditional generation achieves true zero-shot performance—given $\bm{\lambda}$ for a new problem, the model directly outputs reasonable parameters.

## Key Experimental Results

### Main Results (Seen Problems)

| Dataset | Ours (URS) | Best MTL Baseline | Gain | Solve Time |
| :--- | :--- | :--- | :--- | :--- |
| TSP100 | 0.57% | POMO 0.13% | Comparable | 6s |
| CVRP100 | 1.81% | ReLD-MTL 1.42% | Slight degradation | 6s |
| ATSP100 | 2.26% | Baseline 3.05%+ | Significantly better than GOAL | 1.1m |
| CVRPTW100 | 6.13% | MVMoE 3.14% | Sacrifice for generalization | 8s |

### Zero-shot Generalization (Unseen Problems)

| Problem Type | URS Performance | MVMoE | Gain | Description |
| :--- | :--- | :--- | :--- | :--- |
| CVRPBP (Unseen) | 12.95% | 13.95% | +1.0pp | Complex multi-constraint |
| MDOCVRPBPTW (Unseen) | 26.31% | 63.77% | +37.5pp | Extremely hard: Multi-depot + open + time window |
| APDCVRP (Unseen) | 7.03% | × | No baseline | Asymmetric + Pickup and Delivery |
| SPCTSP (Unseen) | -2.37% | × | Beating optimal | Precedence TSP |

### Key Findings
- URS outperforms existing multi-task methods across an average of 99 unseen VRP variants.
- The advantage is particularly evident in complex multi-constraint combinations (+37pp).
- It even surpasses heuristic algorithms on SPCTSP.
- The model maintains reasonable solution times on large-scale instances with up to 7000 nodes.

## Highlights & Insights
- **Data Unification vs. Problem Enumeration**: A fundamental paradigm shift—moving from "designing labels for every constraint combination" to "representing all constraints in a universal feature space and letting the model learn the trade-offs."
- **Generality of the Mixed Bias Module**: A single attention framework is designed to be compatible with symmetric distances, asymmetric distances, and relationship constraints through a multi-path design.
- **Empirical Evidence of Zero-shot Generalization**: The model works without any fine-tuning on 99 completely unseen problems, outperforming baselines specifically trained for those tasks in most cases.

## Limitations & Future Work
- Precision trade-off on seen problems—URS slightly underperforms task-specific models on standard problems like TSP/CVRP.
- Scalability to ultra-large instances remains unknown—tested up to 7000 nodes, but real-world cases may reach tens of thousands.
- Insufficient analysis of constraint satisfaction—the degree of optimization for boundary constraints has not been analyzed in depth.
- Future directions: Knowledge distillation; integration of search strategies; verifying the method's generality in other combinatorial optimization domains.

## Related Work & Insights
- **vs. Constraint Combination Methods (MVMoE, MTPOMO)**: Exhaustive constraint combination leads to limited problem coverage (≤48); URS's unified representation is open and compatible, covering 110+.
- **vs. Adapter Fine-tuning Methods (GOAL, TSP-FT)**: Requiring fine-tuning of adapter parameters for each new problem is not true zero-shot; URS achieves it in one step through conditional parameter generation.
- **General Inspiration**: Proves that neural networks can handle the open space of "problem families" much like heuristic algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolving cross-problem generalization via data representation unification is a paradigm-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on 110 VRP variants (99 unseen) + ablation + scalability up to 7000 nodes.
- Writing Quality: ⭐⭐⭐⭐ Method is clear, tables are detailed; some details are slightly bloated.
- Value: ⭐⭐⭐⭐⭐ First to unify 100+ VRP variants into a single model, holding significant value for practical applications like logistics scheduling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning to Insert for Constructive Neural Vehicle Routing Solver](../../NeurIPS2025/optimization/learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)
- [\[ICML 2026\] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](probing_neural_tsp_representations_for_prescriptive_decision_support.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
