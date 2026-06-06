---
title: >-
  [Paper Note] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation
description: >-
  [ICLR 2026][Graph Domain Adaptation] This paper proposes DiffGDA—the first method to introduce diffusion models into graph domain adaptation (GDA). It formulates the continuous-time joint structure-semantic evolution fro…
tags:
  - "ICLR 2026"
  - "Graph Domain Adaptation"
  - "Diffusion Models"
  - "SDE"
  - "Continuous Evolution"
  - "Domain-Aware Guidance"
date: 2026-05-08
content_hash: 1176045e971f423e
---

# Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation

**Conference**: ICLR 2026
**arXiv**: [2602.10506](https://arxiv.org/abs/2602.10506)  
**Code**: [DiffGDA](https://github.com/chenwei23/DiffGDA)  
**Area**: Other
**Keywords**: Graph Domain Adaptation, Diffusion Models, SDE, Continuous Evolution, Domain-Aware Guidance

## TL;DR

This paper proposes DiffGDA—the first method to introduce diffusion models into graph domain adaptation (GDA). It formulates the continuous-time joint structure-semantic evolution from source graphs to target graphs using stochastic differential equations (SDEs), and employs a density-ratio-based domain-aware guidance network to steer the diffusion trajectory toward the target domain. Theoretical convergence to the optimal adaptation path is proven, and DiffGDA comprehensively outperforms state-of-the-art methods across 14 transfer tasks on 8 real-world datasets.

## Background & Motivation

**Problem Definition**: Graph Domain Adaptation (GDA) aims to transfer knowledge from a labeled source graph to an unlabeled target graph, addressing cross-domain distribution shift. This paper focuses on node-level GDA tasks.

**Existing Paradigms**: GDA encompasses two major paradigms—(1) *model-oriented methods* (learning domain-invariant representations via MMD or adversarial training), which assume limited structural variation and fail under large structural discrepancies; and (2) *data-oriented methods* (constructing intermediate graphs to bridge structural gaps), which rely on discrete alignment steps and are constrained by fixed step counts.

**Fundamental Limitation of Discrete Alignment**: Real-world graph structures are driven by complex processes such as social dynamics, citation growth, and knowledge diffusion, making evolution inherently continuous and nonlinear. Fixed-step discrete alignment cannot approximate the actual transformation process—especially when both structure and semantics lack explicit anchors for alignment in unlabeled target graphs.

**Advantages of Continuous Evolution**: Continuous-time modeling (1) represents structural changes as smooth temporal trajectories that flexibly accommodate nonlinear heterogeneous topologies, and (2) allows semantic information to evolve continuously along the transformation path, enabling the model to automatically learn optimal alignment trajectories.

**Opportunity from Diffusion Models**: Diffusion models have achieved remarkable success in capturing complex distributional transformations. The SDE framework can represent cross-graph transfer as a continuous probability flow, naturally unifying structural and semantic adaptation.

**Research Gap**: Existing graph diffusion models primarily focus on symmetric diffusion processes (generative tasks), whereas GDA requires asymmetric diffusion (directional transfer from source to target domain). No prior work has explored applying diffusion models to GDA.

## Method

### Overall Architecture: DiffGDA

```
Source Graph G^S → [Concatenate features + labels] → Forward diffusion (SDE noise injection) → Gaussian distribution
                                    ↓ Reverse diffusion
         Domain-aware guidance network → Guide diffusion trajectory → Generate intermediate graph G'
                                    ↓
         GNN node classification (Cross-entropy + MMD alignment) → Target graph inference
```

### Component 1: Forward Diffusion Process

The source graph is progressively corrupted toward a Gaussian distribution. A key design choice is to concatenate node features $\mathbf{X}^{\mathcal{S}}$ and labels $\mathbf{Y}^{\mathcal{S}}$ along the channel dimension to form augmented features $\tilde{\mathbf{X}}^{\mathcal{S}} = [\mathbf{X}^{\mathcal{S}} || \mathbf{Y}^{\mathcal{S}}] \in \mathbb{R}^{N_{\mathcal{S}} \times (F+C)}$, injecting label knowledge into the diffusion process. The forward SDE is:

$$\mathrm{d}\mathbf{G}^{\mathcal{S}}_t = \mathbf{f}_t(\mathbf{G}^{\mathcal{S}}_t)\mathrm{d}t + g_t(\mathbf{G}^{\mathcal{S}}_t)\mathrm{d}\mathbf{w}$$

### Component 2: Domain-Guided Reverse Diffusion

The reverse SDE recovers a graph from noise; crucially, however, the goal is not to recover the source graph distribution but to guide generation toward the target domain distribution. The reverse process is:

$$\mathrm{d}\mathbf{G}_t^{\mathcal{S}} = \big[\mathbf{f}_t(\mathbf{G}_t^{\mathcal{S}}) - g_t^2 \nabla_{\mathbf{G}_t^{\mathcal{S}}} \log p_t(\mathbf{G}_t^{\mathcal{S}})\big]\mathrm{d}\bar{t} + g_t \mathrm{d}\bar{\mathbf{w}}$$

Two networks are introduced:
- **Score network** $\mathbb{P}(\boldsymbol{\ell})$: estimates the score function $\nabla \log p_t$, decomposed into separate components for node features and the adjacency matrix, each trained independently.
- **Guidance network** $\mathbb{Q}(\boldsymbol{\delta})$: learns the gradient of the density ratio to steer diffusion toward the target domain.

### Component 3: Domain-Aware Guidance Network (Core Contribution)

**Theorem 1 (Core)**: The optimal diffusion network for the target graph satisfies:

$$\mathbb{P}(\boldsymbol{\ell}^{\star}) = \nabla_{\mathbf{G}_t^{\mathcal{S}}} \log p_t(\mathbf{G}_t^{\mathcal{S}}) + \nabla_{\mathbf{G}_t^{\mathcal{S}}} \log \mathbb{E}_{p(\mathbf{G}_0^{\mathcal{S}}|\mathbf{G}_t^{\mathcal{S}})} \frac{q(\mathbf{G}_0^{\mathcal{T}})}{p(\mathbf{G}_0^{\mathcal{S}})}$$

The first term is the source graph score function; the second term is the **gradient of the target-to-source density ratio**—which constitutes the guidance signal.

**Density Ratio Estimation**: A GNN classifier $\mathcal{C}_{\text{gnn}}$ is trained to distinguish source and target domain nodes, and the density ratio is approximated as $q/p \approx (1-\mathbf{y}(\mathbf{x}))/\mathbf{y}(\mathbf{x})$.

**Key Design—Decomposed Computation**:
- The score network is decomposed into $\mathbb{P}(\boldsymbol{\ell}_1)$ (node feature score, using MLP + GNN) and $\mathbb{P}(\boldsymbol{\ell}_2)$ (adjacency structure score, using MLP + GMH multi-head attention).
- The guidance network is likewise decomposed into $\mathbb{Q}(\boldsymbol{\delta}_1)$ (feature domain estimation) and $\mathbb{Q}(\boldsymbol{\delta}_2)$ (structural domain estimation), both implemented as lightweight MLPs.

**Selective Diffusion Strategy**: A hyperparameter $\alpha$ controls the diffusion ratio, applying diffusion to only a subset of nodes, thereby balancing efficiency and information preservation.

### GNN Training

After generating the labeled intermediate graph $\mathbf{G}' = (\mathbf{X}', \mathbf{A}', \mathbf{Y}')$, the GNN is trained via:

$$\mathcal{L}_{\text{GNN}} = \mathcal{L}_{\text{CE}}(\text{GNN}(\mathbf{X}', \mathbf{A}'), \mathbf{Y}') + \eta \mathcal{L}_{\text{MMD}}(\text{GNN}(\mathbf{X}', \mathbf{A}'), \text{GNN}(\mathbf{X}^{\mathcal{T}}, \mathbf{A}^{\mathcal{T}}))$$

Cross-entropy loss combined with MMD alignment enables end-to-end joint optimization of diffusion and GNN parameters.

## Key Experimental Results

### Table 1: Citation Domain — 6 Transfer Tasks (Mi-F1/Ma-F1)

| Method | A→C | A→D | C→A | C→D | D→A | D→C | Avg |
|--------|-----|-----|-----|-----|-----|-----|-----|
| GCN | 70.82 | 65.05 | 65.44 | 69.46 | 59.92 | 66.83 | 64.83 |
| UDAGCN | 80.68 | 74.66 | 73.46 | 76.97 | 69.36 | 77.81 | 75.03 |
| A2GNN (AAAI'24) | 80.93 | 75.94 | 75.09 | 77.16 | 73.21 | 79.72 | 75.97 |
| TDSS (AAAI'25) | 80.41 | 74.04 | 72.88 | 77.23 | 72.38 | 79.04 | 75.72 |
| GAA (ICLR'25) | 80.03 | 73.32 | 73.15 | 76.04 | 68.32 | 78.27 | 72.65 |
| **DiffGDA** | **82.28** | **76.70** | **75.75** | **78.11** | **74.55** | **80.71** | **77.58** |

### Table 2: Airport Domain — 6 Transfer Tasks (Mi-F1/Ma-F1)

| Method | U→B | U→E | B→U | B→E | E→U | E→B | Avg |
|--------|-----|-----|-----|-----|-----|-----|-----|
| AdaGCN | 65.65 | 50.63 | 46.87 | 54.44 | 48.62 | 73.74 | 56.17 |
| GraphAlign (KDD'24) | 62.54 | 52.18 | 50.33 | 55.23 | 54.35 | 71.02 | 56.39 |
| TDSS (AAAI'25) | 67.43 | 52.05 | 47.62 | 51.80 | 46.08 | 55.73 | 49.24 |
| **DiffGDA** | **71.76** | **54.18** | **54.37** | **57.15** | **56.20** | **74.81** | **60.75** |

### Table 3: Runtime Comparison

| Method | A→D Total Time (s) | A→D Mi-F1 | A→C Total Time (s) | A→C Mi-F1 |
|--------|-------------------|-----------|-------------------|-----------|
| UDAGCN | 83.19 | 67.52 | 104.91 | 72.64 |
| GraphAlign | 269.65 | 70.14 | 297.15 | 76.62 |
| **DiffGDA** | **126.73** | **73.41** | **125.41** | **80.23** |

DiffGDA reduces runtime by over 50% compared to GraphAlign—another graph generative method—while achieving superior performance.

## Key Findings

1. **Continuous outperforms discrete**: DiffGDA consistently surpasses methods based on discrete intermediate graphs (GraphAlign, GGDA) across all 14 transfer tasks, demonstrating the fundamental advantage of continuous-time evolution modeling in capturing nonlinear structural discrepancies.

2. **Guidance network is essential**: Ablation experiments (Figure 2) show that removing the domain-aware guidance network leads to the largest performance drop—particularly on high-difficulty tasks such as B→E—confirming that unguided diffusion alone cannot automatically evolve toward the target domain.

3. **Three components are mutually complementary**: The guidance network stabilizes the diffusion path, MMD promotes cross-domain alignment, and adjacency constraints preserve structural dependencies; all three are indispensable.

4. **Diffusion step count is relatively robust**: Convergence is achieved at $T = 40$–$80$ steps, with diminishing returns beyond that range. The sampling ratio $\alpha$ requires adjustment based on graph scale (large graphs necessitate smaller $\alpha$ due to memory constraints).

5. **Cleaner representation space**: t-SNE visualizations (Figure 5) show that DiffGDA produces more compact and better-separated clusters, effectively eliminating domain-irrelevant noise and enhancing inter-class discriminability.

## Highlights & Insights

- **"First to introduce diffusion into GDA"**: Pioneering use of SDE-based diffusion to jointly model structure and semantic adaptation, establishing an entirely new paradigm for GDA.
- **Elegant density-ratio guidance design**: Rather than generating the target directly from noise, DiffGDA starts from the source graph and uses density ratio gradients to guide the diffusion trajectory—preserving source-domain labels while evolving toward the target domain.
- **Label injection into diffusion**: Concatenating labels with features during diffusion ensures that the generated intermediate graphs carry labels by construction, eliminating the need for separate label propagation.
- **Strong in both theory and practice**: Convergence to the optimal solution is theoretically proven (Theorem 1), and DiffGDA comprehensively outperforms baselines across all 14 tasks.
- **Selective diffusion**: Diffusion is applied to only a subset of nodes (controlled by $\alpha$), balancing efficiency and information preservation.

## Limitations & Future Work

- **Limited scalability**: Memory constraints on large-scale graphs require small sampling ratios ($\alpha$); ratios above 50% cause out-of-memory errors on Airport domain tasks, raising concerns about practical utility on large graphs.
- **Additional cost of MMD alignment**: The need for MMD alignment after diffusion generation suggests that diffusion alone cannot fully eliminate domain gaps, requiring a two-stage pipeline.
- **Sensitivity to loss weight $\eta$**: Performance consistently degrades as $\eta$ increases, indicating that overly strong alignment leads to excessive regularization and necessitating careful hyperparameter tuning in practice.
- **Validation limited to node classification**: Generalizability to other graph tasks such as graph classification and link prediction has not been verified.

## Related Work & Insights

| Dimension | GraphAlign (KDD'24) | DiffGDA |
|-----------|---------------------|---------|
| Alignment | Discrete intermediate graph construction | SDE continuous evolution |
| Adaptation paradigm | Data-oriented (fixed steps) | Generative (continuous time) |
| Typical performance (Citation Avg) | 67.41 | **77.58** |
| Runtime | Slowest (high graph generation overhead) | Reduced by 50%+ |

| Dimension | A2GNN (AAAI'24) | DiffGDA |
|-----------|----------------|---------|
| Core Idea | Adversarial training + attention aggregation | Diffusion + domain guidance |
| Adaptation paradigm | Model-oriented | Generative (data-oriented) |
| Citation Avg | 75.97 | **77.58** |
| Airport Avg | 49.57 | **60.75** |
| Structural discrepancy adaptation | Moderate | Strong (continuous nonlinear modeling) |

| Dimension | GAA (ICLR'25) | DiffGDA |
|-----------|--------------|---------|
| Core Idea | Graph augmentation + alignment | SDE diffusion + domain guidance |
| Citation Avg | 72.65 | **77.58** |
| Airport Avg | 51.98 | **60.75** |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce diffusion models into GDA; replaces discrete alignment with SDE-based continuous evolution—a pioneering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 datasets, 14 tasks, 16 baselines, with ablation studies, parameter analysis, efficiency comparisons, visualizations, and significance testing ($p < 0.05$).
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, the framework diagram is intuitive, and notation is consistent throughout.
- **Value**: ⭐⭐⭐⭐ Opens a new paradigm for continuous graph domain adaptation, though scalability to large graphs remains to be improved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](../../AAAI2026/others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](../../ICML2026/others/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)

</div>

<!-- RELATED:END -->
