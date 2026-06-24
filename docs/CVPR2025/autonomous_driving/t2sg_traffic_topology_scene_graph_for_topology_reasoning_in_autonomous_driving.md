---
title: >-
  [Paper Note] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving
description: >-
  [CVPR 2025][Autonomous Driving][Traffic Topology Reasoning] Defines a unified Traffic Topology Scene Graph (T²SG) to explicitly model lanes, traffic signal control relationships, and topology connections among lanes. It also proposes TopoFormer, which achieves precise topology reasoning using a Lane Aggregation Layer (LAL) and a Counterfactual Intervention Layer (CIL), reaching a SOTA of 46.3 OLS on OpenLane-V2.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Traffic Topology Reasoning"
  - "Scene Graph"
  - "Lane Detection"
  - "Counterfactual Intervention"
  - "HD Maps"
date: 2026-05-08
content_hash: 3e155acb0b26ab9c
---

# T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving

**Conference**: CVPR 2025  
**arXiv**: [2411.18894](https://arxiv.org/abs/2411.18894)  
**Code**: [https://github.com/MICLAB-BUPT/T2SG](https://github.com/MICLAB-BUPT/T2SG)  
**Area**: Autonomous Driving  
**Keywords**: Traffic Topology Reasoning, Scene Graph, Lane Detection, Counterfactual Intervention, HD Maps

## TL;DR

Defines a unified Traffic Topology Scene Graph (T²SG) to explicitly model lanes, traffic signal control relationships, and topology connections among lanes. It also proposes TopoFormer, which achieves precise topology reasoning using a Lane Aggregation Layer (LAL) and a Counterfactual Intervention Layer (CIL), reaching a SOTA of 46.3 OLS on OpenLane-V2.

## Background & Motivation

- Autonomous driving requires understanding the topological relationships between elements in traffic scenes, rather than merely detecting individual elements (e.g., lanes, traffic signals).
- Existing HD mapping methods focus on spatial relationships but neglect the control and guidance of traffic signals on lanes (e.g., a "left-turn" traffic light only controls left-turn lanes).
- Although TopoNet addresses this issue, it neglects the implicit semantic information of control and guidance under traffic rules.
- Local prediction methods ignore coherent road structures (e.g., intersections, straight roads) in traffic scenes; joint reasoning can resolve the ambiguity inherent in local predictions.
- A unified scene graph representation is required to simultaneously model lane categories (incorporating traffic signal semantics), topological connectivity, and road structures.

## Method

### Overall Architecture

Based on a DETR-like lane centerline detector, lane instances are extracted from multi-view images. TopoFormer conducts topology reasoning on the query features output by the detector. Pipeline: Multi-view images → Backbone + BEVFormer → BEV features → Deformable DETR lane detection → Lane Aggregation Layer (LAL) → Counterfactual Intervention Layer (CIL) → Edge prediction head → T²SG.

### Key Designs

**Design 1: Lane Aggregation Layer (LAL)**

- **Function**: Leverages geometric distances of lane centerlines to guide the aggregation of global structural information.
- **Mechanism**: Introduces a spatial proximity matrix $A_{SPM} = \text{Norm}(\frac{1}{d(\hat{v}_{i,l}^p, \hat{v}_{j,0}^p) + \epsilon})$, which computes the normalized inverse distance from the end point of each lane to the start point of another lane. SPM is then added to standard self-attention to construct Geometric-guided Self-Attention (GSA): $A^l = \text{softmax}(\frac{X^l W_Q^l \cdot (X^l W_K^l)^\top}{\sqrt{d}} + A_{SPM})$
- **Design Motivation**: Lane connectivity is highly correlated with spatial distance (lanes with short end-to-start distances are more likely to be connected). Instead of merely enhancing individual lane features (as in TopoMLP), spatial information should be used to aggregate global lane interactions, equipping features with scene-level context.

**Design 2: Counterfactual Intervention Layer (CIL)**

- **Function**: Captures plausible road structures in traffic scenes (e.g., straight roads, intersections) to enhance topological reasoning.
- **Mechanism**: Treats learned attention weights as "factual" road structures and zero attention weights as "counterfactual" structures. Counterfactual Self-Attention is defined as $\text{CSA}(X^l) = \text{softmax}(\overline{A^l} + A_{SPM}) \cdot X^l W_V^l$, where $\overline{A^l}$ is an all-zero matrix. During training, the Total Indirect Effect (TIE) defined as $\hat{\mathcal{E}}_{TIE} = \mathbb{E}[\hat{\mathcal{E}}_A - \hat{\mathcal{E}}_{\overline{A}}]$ is used as the edge prediction output and optimized with focal loss; during inference, only the normal prediction $\hat{\mathcal{E}}_A$ is utilized.
- **Design Motivation**: Purely geometric methods rely heavily on the accuracy of centerline detection, and detection errors will propagate to topology reasoning. By comparing prediction differences between factual and counterfactual structures, the total indirect effect of the learned road structures is maximized, encouraging the model to learn more plausible structural representations.

**Design 3: Unified Scene Graph Representation (T²SG)**

- **Function**: Consistently models lane categories (including traffic signal semantics) and inter-lane connectivity with a graph structure.
- **Mechanism**: $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where each node $v_i = [v_i^c, v_i^p]$ consists of a classification label and centerline coordinates, and an edge $e_{ij} \in \{0,1\}$ represents directional connectivity (the endpoint of lane $i$ connects to the startpoint of lane $j$). Lane category $\mathcal{C}_{lc}$ incorporates traffic signal semantics, enabling lanes and traffic elements with the same category to establish implicit associations automatically.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_\mathcal{V} + \mathcal{L}_\mathcal{E} = (\lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{reg} \cdot \mathcal{L}_{reg}) + \lambda_{cls} \cdot \mathcal{L}_{cls}(\hat{\mathcal{E}}_{TIE}, \mathcal{E}_{GT})$$

Node detection utilizes Focal Loss + L1 regression, while edge prediction employs Focal Loss based on TIE.

## Key Experimental Results

### OpenLane-V2 Results

| Method | OLS ↑ | TOP_ll ↑ | TOP_lt ↑ | DET_l ↑ | DET_t ↑ |
|------|-------|----------|----------|---------|---------|
| TopoNet | Baseline | — | — | — | — |
| TopoMLP | Moderate | — | — | — | — |
| **TopoFormer (T²SG)** | **46.3** | **SOTA** | **SOTA** | **SOTA** | — |

### T²SG Scene Graph Generation Ablation

| Method | Node AP₁.₀ ↑ | Edge mAP₁.₀ ↑ | Edge A@1₁.₀ ↑ |
|------|-------------|---------------|---------------|
| Baseline | 10.4 | 4.1 | 8.0 |
| w/ 3DSSG | 10.7 | 4.4 | 0.4 |
| w/ LAL | Gain | Gain | Significant Gain |
| w/ LAL + CIL | Best | Best | Best |

### Key Findings

- Directly applying the 3D Scene Graph (3DSSG) method degrades the A@1 metric for edge prediction, indicating that generic scene graph methods are not well-suited for traffic scenes.
- The geometric-guided self-attention in LAL exhibits better generalization than the lane feature enhancement via positional encoding in TopoMLP.
- The counterfactual intervention in CIL effectively improves the physical plausibility of the learned road structures.
- Expanding lane categories to incorporate traffic signal semantics enables unified modeling of lane-signal relationships.
- The generated T²SG scene graphs directly boost downstream traffic topology reasoning tasks.

## Highlights & Insights

1. **Unified Scene Graph for Traffic Topology**: First to introduce scene graphs into traffic scene understanding, encoding traffic signal semantics through node categories.
2. **Innovative Application of Counterfactual Intervention**: Introduces causal inference techniques into topology reasoning, enhancing learning by contrasting factual and counterfactual structures.
3. **Geometric-Guided Global Aggregation**: The SPM matrix guides attention using lane endpoint distances, offering better interpretability than simple positional encodings.

## Limitations & Future Work

- The counterfactual intervention relies solely on all-zero attention as the counterfactual, and alternative counterfactual structures can be explored.
- The approach depends heavily on the accuracy of the lane centerline detector; detection errors propagate to the topology reasoning phase.
- Temporal information is currently not incorporated; multi-frame inputs could potentially offer richer topological cues.
- The design of the category taxonomy requires domain expertise, limiting the degree of automation.

## Related Work & Insights

- **TopoNet** [Li et al.] first modeled heterogeneous topology graphs of lanes and traffic signals using GNNs.
- **TopoMLP** [Wu et al.] leveraged spatial lane positions to enhance topology reasoning.
- **Counterfactual Intervention** [Niu et al. VQA] inspired the application of causal inference to structured reasoning.
- **Scene Graph Generation** [Xu et al.] provided the foundational paradigm for mapping visual inputs to structured representations.

## Rating

⭐⭐⭐⭐ — Problem definition is highly valuable (unified modeling of traffic topology), the designs of LAL and CIL demonstrate theoretical depth, and the 46.3 OLS SOTA is compelling. The application of counterfactual intervention to traffic reasoning is highly novel.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [\[CVPR 2025\] A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)
- [\[CVPR 2026\] TopoHR: Hierarchical Centerline Representation for Cyclic Topology Reasoning in Driving Scenes with Point-to-Instance Relations](../../CVPR2026/autonomous_driving/topohr_hierarchical_centerline_representation_for_cyclic_topology_reasoning_in_d.md)
- [\[CVPR 2025\] GLane3D: Detecting Lanes with Graph of 3D Keypoints](glane3d_detecting_lanes_with_graph_of_3d_keypoints.md)

</div>

<!-- RELATED:END -->
