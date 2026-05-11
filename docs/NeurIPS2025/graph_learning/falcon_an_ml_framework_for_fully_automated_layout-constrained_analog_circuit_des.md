---
title: >-
  [Paper Note] FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design
description: >-
  [NeurIPS 2025][Graph Learning][Analog Circuit Design] FALCON proposes an end-to-end framework for automated analog/RF circuit design via a three-stage pipeline: MLP-based topology selection…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Analog Circuit Design"
  - "Graph Neural Networks"
  - "Inverse Design"
  - "Layout-Aware Optimization"
  - "Millimeter-Wave Circuits"
date: 2026-05-08
content_hash: b8368a0312ca51ff
---

# FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design

**Conference**: NeurIPS 2025
**arXiv**: [2505.21923](https://arxiv.org/abs/2505.21923)
**Code**: [https://github.com/AsalMehradfar/FALCON](https://github.com/AsalMehradfar/FALCON)
**Area**: Graph Learning
**Keywords**: Analog Circuit Design, Graph Neural Networks, Inverse Design, Layout-Aware Optimization, Millimeter-Wave Circuits

## TL;DR
FALCON proposes an end-to-end framework for automated analog/RF circuit design via a three-stage pipeline: MLP-based topology selection, edge-centric GNN performance prediction, and differentiable layout-constrained gradient inference. Trained on a million-scale Cadence simulation dataset, the framework achieves >99% topology selection accuracy, <10% performance prediction error, and sub-second per-instance inference.

## Background & Motivation

**Background**: Analog/RF/millimeter-wave circuit design remains heavily reliant on human expertise, spanning three stages: topology selection, parameter tuning, and layout feasibility verification. Machine learning methods have begun to address this domain, but typically target only one subproblem at a time.

**Limitations of Prior Work**:
- Most methods assume a fixed topology and cannot adapt to new performance specifications.
- Optimization approaches rely on black-box search strategies (RL, Bayesian optimization), incurring high computational costs and poor scalability.
- Performance prediction models do not support inverse design (inferring parameters from target specifications).
- Layout constraints are typically handled as post-processing steps, missing the opportunity to incorporate physical constraints during optimization.
- Training datasets are often based on symbolic/synthetic simulators, lacking commercial-grade fidelity.

**Key Challenge**: The mapping from performance specifications to final layout is a multi-stage coupled problem, and existing methods address each stage in isolation without integration.

**Goal**: To construct a unified, differentiable, end-to-end framework that integrates target performance → topology selection → parameter inference → layout feasibility into a single pipeline.

**Key Insight**: Circuits are represented as graphs (netlists → multi-edge heterogeneous graphs), with a GNN trained as a differentiable surrogate forward model. Inverse design is then achieved via gradient backpropagation, with differentiable layout cost functions embedded in the optimization.

**Core Idea**: An edge-centric GNN learns a differentiable mapping from circuit parameters to performance. Inverse design is performed by gradient-based inference through a frozen GNN, while layout constraints are encoded as differentiable penalty terms enabling layout-aware optimization.

## Method

### Overall Architecture
Input: target performance vector $y_{\text{target}} \in \mathbb{R}^{16}$ (16 analog/RF metrics). Three-stage pipeline:
1. **Stage 1**: An MLP classifier selects the best-matching topology $T^*$ from 20 expert-designed topologies.
2. **Stage 2**: An edge-centric GNN forward model predicts $\hat{y} = f_\theta(T, x)$.
3. **Stage 3**: The GNN is frozen; gradient descent over parameters $x$ solves $x^* = \arg\min_x \mathcal{L}_{\text{perf}}(f_\theta(T^*,x), y_{\text{target}}) + \lambda\mathcal{L}_{\text{layout}}(x)$.

### Key Designs

1. **Circuit Graph Representation (Netlist-to-Graph)**:

    - Function: Converts Cadence netlists into multi-edge heterogeneous graphs, where nodes represent voltage nets (electrical connection points) and edges represent circuit elements.
    - Mechanism: Multi-port devices (e.g., transistors) are decomposed into multiple edges (GS/DS/DG), each annotated with: (i) categorical device type, (ii) fixed numerical attributes (e.g., channel length), (iii) parameterized attributes (e.g., W1, R3), (iv) one-hot categorical features, and (v) computed attributes (e.g., diffusion area). Graph scale ranges from 4–40 nodes and 7–70 edges.
    - Design Motivation: Preserves native alignment with foundry netlists without losing multi-edge or heterogeneous information, enabling generalization across circuit families.

2. **Stage 1: Topology Selection MLP**:

    - Function: Given a 16-dimensional performance vector, classifies and selects the most appropriate topology.
    - Mechanism: A 5-layer MLP (hidden size 256, ReLU activations) takes z-score normalized performance vectors as input and outputs a 20-class probability distribution. Trained with cross-entropy loss and Adam optimizer.
    - Design Motivation: Performance vectors carry rich semantic information (t-SNE visualizations reveal clear topology clustering), making a lightweight MLP sufficient. The >99% accuracy validates this design choice.

3. **Stage 2: Edge-Centric GNN Forward Model**:

    - Function: Learns a differentiable mapping $(T, x) \to \hat{y}$.
    - Mechanism:
        - Type-specific MLP encoders $z_e = \phi^{(t_e)}_{\text{enc}}(x_e)$ handle heterogeneous edge features.
        - 4-layer edge-centric message passing with residual connections: $m^{(\ell)}_u = \sum_{e\in\mathcal{E}_u}\phi_{\text{MSG}}(h^{(\ell)}_{\text{src}(e)}, z_e)$, $h^{(\ell+1)}_u = \text{ReLU}(\phi_{\text{UPD}}(m^{(\ell)}_u) + h^{(\ell)}_u)$.
        - Global mean pooling: $z_{\text{graph}} = \frac{1}{|V|}\sum_{u\in V}h^{(L)}_u$.
        - An output MLP predicts the 16-dimensional performance vector.
    - Training Loss: Masked MSE $\mathcal{L}_{\text{masked}} = \frac{1}{\sum_i m_i}\sum_{i=1}^d m_i(\hat{y}_i - y_i)^2$ (undefined metrics are masked to 0).
    - Design Motivation: The edge-centric design allows parameter information to propagate directly through edge attributes, making it more suitable for circuit topologies than node-centric GNNs, since parameters are attached to components (edges) rather than nodes.

4. **Stage 3: Layout-Aware Gradient Inference**:

    - Function: With the GNN frozen, gradient descent over parameters $x$ jointly satisfies performance targets and layout constraints.
    - Mechanism: Total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{perf}} + \lambda_{\text{area}}\cdot\mathcal{L}_{\text{layout}}\cdot g(\mathcal{L}_{\text{perf}})$, where $g(\mathcal{L}_{\text{perf}})=1-\sigma(\gamma(\mathcal{L}_{\text{perf}}-\tau))$ is a sigmoid gate — when performance error is large, layout penalty is suppressed, prioritizing functional correctness before area minimization.
    - Layout Cost: $\mathcal{L}_{\text{layout}}(x)=\sum_{e\in\mathcal{E}_{\text{passive}}}A_e(x)$, computing area analytically for passive devices (capacitors, inductors, resistors) using expressions such as $A=W\cdot L$. Areas are normalized to 1 mm².
    - Adaptive learning rate with ReduceLROnPlateau scheduling and failure-restart strategy.
    - Design Motivation: Embedding layout constraints within the optimization loop (rather than as post-processing) allows gradient signals to simultaneously encode functional and physical objectives.

### Cross-Topology Generalization
- The RVCO topology is entirely excluded from training.
- Zero-shot prediction yields an average relative error of 30.4%.
- Fine-tuning (updating only the output MLP head) reduces error to 0.9%, demonstrating that structural priors learned by the GNN encoder are highly transferable.

## Key Experimental Results

### Main Results (Forward Prediction, 16 Performance Metrics)

| Metric | R² | RMSE | MAE | Rel. Error |
|------|-----|------|-----|------------|
| DC Power | 1.0 | 0.27 | 0.198 | 11.2% |
| Voltage Gain | 1.0 | 0.101 | 0.072 | 2.6% |
| S11 | 0.93 | 1.515 | 0.554 | 11.4% |
| Noise Figure | 0.99 | 0.534 | 0.2 | 4.5% |
| Osc. Freq. | 0.97 | 0.723 | 0.184 | 0.6% |
| Phase Noise | 0.89 | 2.536 | 1.159 | 1.3% |
| **Mean R²** | **0.972** | — | — | **9.09%** |

### Topology Selection

| Metric | Score |
|------|------|
| Accuracy | 99.57% |
| Balanced Accuracy | 99.33% |
| Macro F1 | 99.30% |

### RVCO Cross-Topology Generalization (Fine-tuning)

| Metric | R² | Rel. Error |
|------|-----|------------|
| DC Power | 1.0 | 0.75% |
| Osc. Freq. | 1.0 | 0.85% |
| Tuning Range | 1.0 | 1.63% |
| Output Power | 0.97 | 0.69% |
| Phase Noise | 0.98 | 0.73% |

### Inverse Design (Stage 3)

| Metric | Value |
|------|------|
| Test instances | 9,500 (500 per topology) |
| Success rate | 78.5% |
| Mean relative error (successful designs) | 17.7% |
| Per-instance inference time | <1 second (MacBook CPU) |

### Key Findings
- Topology selection is near-perfect (>99%); the only confusion occurs between common-gate and common-source voltage amplifiers, where performance overlap arises when gain-bandwidth products are similar.
- The forward model is especially accurate for frequency-related metrics (OscF 0.6%, PN 1.3%), as these share more explicit analytical relationships with circuit parameters.
- The sigmoid gating mechanism $g(\mathcal{L}_{\text{perf}})$ in layout-aware optimization is critical: the strategy of converging on functionality before optimizing area is more stable than simultaneous multi-objective optimization.
- Fine-tuning requires updating only the output head (~30 minutes on MacBook CPU), confirming that GNN encoder layers learn circuit representations that generalize across topologies.

## Highlights & Insights
- **The edge-centric GNN design** is well-motivated: the core information in a circuit (parameters, device types) resides in components (edges) rather than nodes (voltage nets), making edge-centric message passing the most natural modeling choice. This insight transfers to other graph learning problems where parameters are attached to edges (e.g., pipe networks, traffic networks).
- **The differentiable layout cost** integrates physical constraints directly into gradient-based optimization rather than treating them as hard constraints or post-processing steps. The sigmoid-gated "functionality first, layout second" strategy elegantly addresses priority ordering in multi-objective optimization.
- **The million-scale Cadence simulation dataset** is itself a significant contribution — prior analog circuit ML datasets are mostly based on symbolic simulators and lack industrial fidelity.
- The end-to-end pipeline unifies three previously independent research directions (topology selection, performance prediction, and layout optimization) within a single differentiable framework.

## Limitations & Future Work
- **Fixed topology library of 20 designs**: Adding new topologies requires retraining the classifier and forward model; although fine-tuning has been shown feasible, manual curation is still required.
- **Single process node (45nm CMOS)**: Different technology nodes (e.g., 7nm FinFET) exhibit substantially different parameter ranges and parasitic effects, necessitating new data generation.
- **Analytical layout model is an approximation**: Area and parasitics are estimated via simplified formulas, without accurate modeling of EM coupling, electromigration, or other physical effects. The authors acknowledge plans to incorporate learned parasitic models in future work.
- **78.5% success rate** implies that approximately one in five instances fails optimization, likely due to poor initialization or infeasible performance targets; better feasibility detection mechanisms are needed.
- Applicability is limited to mm-wave circuits; suitability for low-frequency analog circuits (op-amps, ADCs) remains unverified.

## Related Work & Insights
- **vs. ALIGN / LayoutCopilot**: These methods generate layout from netlists with already-determined parameters and do not support inverse design. FALCON embeds layout constraints into the parameter optimization process, unifying schematic and physical design.
- **vs. AnalogGym / AutoCkt**: These rely on symbolic simulators with small-scale datasets and lack process fidelity. FALCON's Cadence dataset represents a qualitative improvement in both scale and simulation accuracy.
- **vs. DICE**: DICE explores transistor-level schematic transformations for self-supervised learning but does not perform inverse design. FALCON's netlist-to-graph conversion preserves native parameter information and supports end-to-end inference.

## Rating
- Novelty: ⭐⭐⭐⭐ The end-to-end differentiable pipeline integrating topology selection, forward prediction, and layout-aware inverse design is the first of its kind in the analog circuit domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ One million data points, 20 topologies, 16 performance metrics, cross-topology generalization, and closed-loop Cadence validation.
- Writing Quality: ⭐⭐⭐⭐ Method is described clearly and modularly, with each stage presented as a self-contained section; figures and tables are informative.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in end-to-end automated analog circuit design; both dataset and code are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](../../ACL2026/graph_learning/autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[NeurIPS 2025\] Practical Bayes-Optimal Membership Inference Attacks](practical_bayes-optimal_membership_inference_attacks.md)
- [\[NeurIPS 2025\] When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning](when_no_paths_lead_to_rome_benchmarking_systematic_neural_relational_reasoning.md)

</div>

<!-- RELATED:END -->
