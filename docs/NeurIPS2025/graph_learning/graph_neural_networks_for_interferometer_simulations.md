---
title: >-
  [Paper Note] Graph Neural Networks for Interferometer Simulations
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][Graph Learning][GNN] This work presents the first application of graph neural networks to optical interferometer simulation…
tags:
  - "NeurIPS 2025 (AI for Science Workshop)"
  - "Graph Learning"
  - "GNN"
  - "interferometer simulation"
  - "LIGO"
  - "gravitational waves"
  - "optical simulation"
date: 2026-05-08
content_hash: 9bed7f4eda7c0ed1
---

# Graph Neural Networks for Interferometer Simulations

**Conference**: NeurIPS 2025 (AI for Science Workshop)
**arXiv**: [2512.16051](https://arxiv.org/abs/2512.16051)  
**Code**: [LIGO GitLab](https://git.ligo.org/uc_riverside/gnn-ifosim)  
**Area**: Graph Learning / Scientific Computing
**Keywords**: GNN, interferometer simulation, LIGO, gravitational waves, optical simulation

## TL;DR

This work presents the first application of graph neural networks to optical interferometer simulation, employing a GATv2 + KAN architecture to predict electromagnetic field power and spatial intensity distributions within LIGO interferometers. The approach achieves inference speeds up to 815× faster than the standard simulation software (FINESSE) while maintaining satisfactory physical accuracy.

## Background & Motivation

Gravitational waves—stretching and compressing the fabric of spacetime—were predicted by Einstein in 1916. LIGO (Laser Interferometer Gravitational-Wave Observatory) detects them using dual-recycled Michelson interferometers. To attain the sensitivity required for gravitational-wave detection, LIGO and the next-generation Cosmic Explorer must be robust to realistic manufacturing errors, placing stringent demands on interferometer design.

**Limitations of Prior Work**: Searching for optimally robust design parameters is a computationally formidable problem, requiring thousands of high-fidelity optical simulations and optimization over a high-dimensional, non-convex loss landscape. The primary computational bottlenecks are:
1. Capturing fine spatial features requires higher-order modes, causing scattering matrix dimensions to grow as $\mathcal{O}(n^2)$;
2. Computing individual matrix elements is extremely time-consuming, particularly when thermal lensing and finite-aperture scattering are included;
3. Interferometers must be "locked"—cavity mirror positions must be finely tuned to achieve resonance—and each adjustment requires rerunning the simulation.

**Key Challenge**: Training data collection is itself prohibitively expensive, and limited generalization severely reduces practical utility. Nevertheless, even coarse simulation surrogates are valuable: they enable optimization routines to rapidly prune unfavorable regions of the design parameter space.

**Key Insight**: The interaction of electromagnetic fields with a sequence of optical components in an interferometer is naturally suited to a graph representation. Each optical component can be decomposed into incident/outgoing field nodes, with edges representing spatial connections between fields. GNN message passing effectively captures both local interactions and long-range dependencies.

**Core Idea**: Train a GNN to learn the steady-state electromagnetic fields (power and intensity distributions) of an interferometer as a surrogate model for FINESSE simulations, achieving substantial inference acceleration while preserving physical accuracy.

## Method

### Overall Architecture

**Input**: A graph representation of the interferometer. Each optical component (mirror) is decomposed into 4 nodes (2 per side, corresponding to incident and outgoing fields respectively). Each node carries 3 features: wavefront radius of curvature, optical component reflectivity, and orientation angle. Each edge carries 2 features: length and refractive index.

**Output**: Two tasks—(1) incident/outgoing field power at each node; (2) spatial intensity distribution (2D intensity map) at each node.

**Dataset**: Three interferometer topologies:
- Fabry-Perot (FP) cavity: 30,000 graphs, 10 nodes each
- Simple coupled cavity (Simple CC): 5,000 graphs, 18 nodes each
- Arm-SRC coupled cavity: 30,000 graphs, 74 nodes each

### Key Designs

1. **Power Prediction Model**:

    - *Function*: Predicts incident and outgoing field power at each optical point in the interferometer.
    - *Mechanism*: 20-layer GATv2 + 6-layer feedforward network + LeakyReLU activations + residual connections. Because power spans several orders of magnitude within the interferometer (kW-level intracavity vs. mW-level output), the model predicts $\log P$ rather than raw power.
    - *Design Motivation*: The attention mechanism of GATv2 allows the network to learn priority relationships among different optical components; 20 layers of depth ensures sufficient information propagation across complex interferometer topologies.

2. **Physics-Constrained Loss Function**:

    - *Function*: Augments the standard reconstruction loss with an energy-conservation regularization term.
    - *Core Formula*:
    $$\mathcal{L} = \frac{1}{n} \sum_{n}^{N} \| \mathbf{y}_n - \hat{\mathbf{y}}_n \|_1 + \lambda \| \hat{\mathbf{y}}_n - \mathbf{A}^T \hat{\mathbf{y}}_n \|_1$$
      The first term is the standard MAE loss; the second penalizes predictions that violate energy conservation (the sum of incident powers at a node should equal the node power).
    - *Design Motivation*: Embedding the physical prior of energy conservation into the training objective ensures the physical feasibility of predictions.

3. **Intensity Distribution Prediction Model**:

    - *Function*: Predicts the 2D spatial intensity distribution at each optical point.
    - *Mechanism*: 15-layer GAT → Deep Kolmogorov-Arnold Network (KAN). Exploiting the radial symmetry of the field, a KAN first learns the radial intensity profile $I(r)$, which is then rotated to generate the full 2D intensity map.
    - *Design Motivation*: (1) Exploiting physical symmetry reduces the degrees of freedom from $\mathcal{O}(n^2)$ to $\mathcal{O}(n)$; (2) KANs outperform MLPs in learning physics special functions (e.g., spherical harmonics) with fewer parameters.

4. **Graph Representation Design**:

    - *Function*: Converts the interferometer optical system into a graph structure.
    - *Mechanism*: Each mirror is decomposed into 4 nodes (two sides × incident/outgoing); edges connect incident fields to reflected and transmitted fields, as well as outgoing fields to the incident field of the next optical component.
    - *Design Motivation*: This decomposition accurately reflects the physical propagation path of electromagnetic fields through the interferometer.

### Loss & Training

- **Power prediction**: L1 loss + energy-conservation regularization term
- **Intensity prediction**: L1 loss ($W/m^2$)
- **Data collection**: Starting from an "ideal" interferometer configuration, parameters (radius of curvature, reflectivity, spacing) are perturbed via random walks, with FINESSE simulations run at each step to obtain ground-truth labels.
- **Training set composition**: The Mixed model is trained on 20,000 FP + 4,000 Arm-SRC CC samples.

## Key Experimental Results

### Main Results (Power Prediction L1 Loss)

| Training Data | Test: Arm-SRC CC | Test: Fabry-Perot | Test: Simple CC |
|---|:-:|:-:|:-:|
| GAT+MLP (FP) | ∞ | 0.52 | 2.94 |
| GAT+MLP (Mixed) | **0.25** | **0.54** | **3.01** |
| GAT+MLP (Arm-SRC) | 0.24 | 1.36 | 2.98 |
| GAT+KAN (Mixed) | 0.38 | 0.76 | 1.09 |
| MLP Only (Mixed) | 0.41 | 1.32 | 1380.19 |
| KAN Only (Mixed) | 0.19 | 33.98 | 38.91 |
| GraphTransformer+MLP (Mixed) | 0.34 | 0.65 | 1.71 |

### Ablation Study

**Effect of GAT Depth**:

| GAT Layers | FP | Simple CC | Arm-SRC CC |
|---|:-:|:-:|:-:|
| 1 | 1.06 | 2.86 | 0.43 |
| 3 | 0.86 | 2.20 | 0.42 |
| 8 | 0.54 | 0.83 | 0.39 |
| 15 | 0.53 | 0.70 | 0.39 |
| 20 | 0.53 | 0.71 | 0.38 |

**Intensity Prediction Model Comparison**:

| Model | L1 Loss ($W/m^2$) |
|---|:-:|
| GAT + KAN | **27.2** |
| GAT + MLP (same parameter count) | 58.4 |

### Computational Efficiency Comparison

| Method | Single Simulation (s) | FP Optimization (s) |
|---|:-:|:-:|
| FINESSE | 2.857 | 170.8 |
| SIS | 14.932 | - |
| GNN (Power) | **0.018** | 53.7 |
| GNN (Intensity) | **0.011** | - |

Single-simulation speedup: **159×** over FINESSE and **815×** over SIS.

### Key Findings

1. **GNNs substantially outperform graph-free models**: MLP Only and KAN Only generalize poorly to unseen topologies (losses of 1380 or ∞ on CC datasets), whereas GNN models maintain consistently low losses.
2. **Diminishing returns with GAT depth**: Loss decreases substantially from 1 to 8 layers, with negligible improvement beyond 15–20 layers, indicating sufficient information propagation.
3. **KAN outperforms MLP for intensity prediction**: Learning radial distributions with KAN reduces L1 loss by 53% (27.2 vs. 58.4), as KANs are better suited to learning physics special functions.
4. **Mixed training enhances generalization**: Incorporating as few as 4,000 Arm-SRC CC training samples enables the Mixed model to match the performance of a topology-specific model on that topology.
5. **Cross-topology generalization is limited but useful**: An FP-trained model fails on CC topologies (∞ loss), yet even coarse estimates remain valuable for design space pruning during architecture optimization.
6. **Energy-conservation loss is effective**: The physics-constraint regularization term ensures that predictions satisfy fundamental physical laws.

## Highlights & Insights

1. **Pioneering application**: This is the first application of deep learning to electromagnetic field propagation prediction in optical interferometer simulation, opening a new direction in ML for instrumentation design.
2. **KAN combined with symmetry exploitation**: Reducing 2D prediction to 1D by exploiting radial field symmetry, then leveraging KAN's function approximation advantage, simultaneously reduces computation and improves accuracy—an exemplary integration of physical priors with network architecture design.
3. **Physics-constrained loss**: The energy-conservation regularization term provides a simple yet effective means of embedding physical priors into training.
4. **Practicality-first orientation**: The paper explicitly acknowledges that the model need not achieve extreme precision—even coarse approximations enable optimization routines to prune unfavorable design regions rapidly. This pragmatic positioning is highly valuable.
5. **Comprehensive dataset contribution**: High-fidelity simulation datasets for three interferometer topologies are released as a benchmark.

## Limitations & Future Work

1. **Limited cross-topology generalization**: Generalization to interferometer topologies absent from the training set remains a critical limitation, as exploring novel topologies is central to interferometer design optimization.
2. **Partial physical modeling**: Higher-order effects such as point absorbers, thermal lensing, and astigmatic beam shapes are not included, leaving a gap relative to full-scale interferometer design.
3. **Locking procedure not modeled**: The model is trained on data from already-locked interferometers, whereas the locking process itself requires substantial simulation effort.
4. **Limited practical speedup for FP optimization**: Although single-simulation acceleration reaches 159×, the actual optimization speedup is only 3.2× (170.8 s vs. 53.7 s), bottlenecked by overhead in converting FINESSE outputs to GNN input format.
5. **Insufficient ablation coverage**: The paper is self-described as a proof of concept; many architectural choices (layer count, GATv2 vs. alternatives) lack systematic ablation.
6. **Power scale challenges**: Although predicting $\log P$ mitigates the kW-to-mW dynamic range, prediction accuracy for very low-power nodes remains limited.

## Related Work & Insights

- **Pfaff et al. (2021)**: Mesh-based GNN physical simulation → This work applies analogous ideas to optical simulation without focusing on temporal evolution.
- **Alkin et al. (2024)**: Universal Physics Transformers → Points toward a universal physics foundation model, though that setting involves temporal evolution and full-field prediction rather than the steady-state regime addressed here.
- **Liu et al. (2024)**: KAN advantages for learning physics special functions such as spherical harmonics → Motivates replacing MLP with KAN for radial intensity distribution prediction.
- **Paganini et al. (2018)**: GAN-accelerated particle physics simulation → Generative models could be applied in future work to model the conditional distribution of interferometer parameter perturbations.

**Takeaway**: The primary contribution of this work lies not in GNN architectural innovation but in applying GNNs to an entirely new scientific problem and demonstrating feasibility. The "even coarse estimates are useful" positioning strategy is instructive for other scientific computing acceleration efforts. The design pattern of combining KANs with physical symmetry has broad generalization value.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of GNNs to interferometer simulation; novel and practically motivated problem selection.
- Experimental Thoroughness: ⭐⭐⭐ Three topologies + multi-architecture comparison + ablation + speed benchmarking, but lacks testing on higher-order physical effects.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, physical background is well explained, and the appendix is comprehensive.
- Value: ⭐⭐⭐⭐ Opens a new direction in ML for instrumentation design; dataset contribution has long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[NeurIPS 2025\] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
