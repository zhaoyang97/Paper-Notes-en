---
title: >-
  [Paper Note] DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring
description: >-
  [ICLR 2026][Medical Imaging][cell instance segmentation] This paper formulates densely-overlapping cell instance segmentation as a graph coloring problem and proposes Disco…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "cell instance segmentation"
  - "graph coloring"
  - "dense overlap"
  - "adjacency constraints"
  - "pathology images"
date: 2026-05-08
content_hash: 21597ffedfc21f0c
---

# DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring

**Conference**: ICLR 2026
**arXiv**: [2602.05420](https://arxiv.org/abs/2602.05420)
**Code**: [https://github.com/SR0920/Disco](https://github.com/SR0920/Disco)
**Area**: Medical Imaging / Segmentation
**Keywords**: cell instance segmentation, graph coloring, dense overlap, adjacency constraints, pathology images

## TL;DR

This paper formulates densely-overlapping cell instance segmentation as a graph coloring problem and proposes Disco, a divide-and-conquer framework combining explicit conflict node marking with implicit adjacency-constrained disambiguation. By decomposing cell adjacency graphs via BFS and introducing five collaborative loss functions, Disco achieves a 7.08% PQ improvement on the high-density pathology dataset GBC-FS 2025 while attaining state-of-the-art performance across four heterogeneous datasets.

## Background & Motivation

**Background**: Cell instance segmentation is a core task in digital pathology analysis, essential for cell counting, morphological analysis, and cancer grading. Current mainstream methods fall into four categories: detection-based (Mask R-CNN), relying on bounding boxes and NMS; contour-based (DCAN), sensitive to binarization thresholds; distance/orientation map-based (HoverNet, StarDist), requiring complex post-processing; and graph coloring-based (FCIS), which models the problem from a topological perspective. The shared weakness of the first three categories is their reliance on local pixel-level or geometric information for instance assignment, leading to systematic errors in dense cell clusters.

**Limitations of Prior Work**: Graph coloring methods offer a globally-aware paradigm for dense segmentation, but their core assumptions are critically flawed. The simplest 2-coloring model assumes the cell adjacency graph (CAG) is bipartite (containing no odd cycles), yet through the first systematic analysis across four datasets, this paper finds that real-world CAGs are mostly non-bipartite. In CryoNuSeg, 56.67% of images contain non-bipartite graphs; in the high-density GBC-FS 2025, as many as 30.49% of nodes are conflict nodes. On the other hand, the 4-coloring used by FCIS, while theoretically complete, introduces unnecessary representational redundancy and optimization difficulty—most regions in practice require only 2 colors.

**Key Challenge**: This is a Goldilocks problem: 2-coloring is too simple to handle odd-cycle topological conflicts, while 4-coloring is unnecessarily complex and imposes learning burdens in simple regions. A solution is needed that efficiently handles the dominant bipartite structure while precisely addressing sparse conflict regions.

**Goal**: (1) How to systematically characterize the topological properties of real-world cell adjacency graphs? (2) How to design an adaptive coloring framework that degenerates to efficient 2-coloring for simple topologies while dynamically handling conflicts in complex ones? (3) How to enable the network to implicitly learn discriminative representations in continuous feature space when the label system cannot perfectly encode secondary conflicts?

**Key Insight**: The authors conduct rigorous topological analysis—constructing CAGs for four datasets and measuring node degree distributions, odd-cycle distributions, and conflict node ratios. Key findings are: (1) over 90% of odd cycles in all non-bipartite graphs are 3-cycles (triangles); (2) conflict nodes tend not to be isolated but form dense "conflict clusters" with abundant "secondary conflicts" (conflict nodes adjacent to each other). This analysis directly guides the framework design: BFS efficiently extracts the maximal bipartite subgraph to resolve the majority, while a dedicated mechanism handles the conflict remainder.

**Core Idea**: Divide and conquer the graph coloring problem—BFS separates the bipartite subgraph for 2-coloring, remaining conflict nodes are labeled as a third color, and an adjacency constraint loss implicitly disambiguates them in continuous probability space, realizing a collaborative framework of explicit marking and implicit disambiguation.

## Method

### Overall Architecture

Given a pathology image and its instance annotations, Disco proceeds in three stages: (1) **Topological Analysis and Label Generation**: a CAG is constructed from instance masks using morphological dilation (3×3 kernel) to define 8-connected adjacency; BFS then extracts the maximal bipartite subgraph, partitioning nodes into two independent sets $V_1, V_2$ (colors 1 and 2) and a conflict set $V_{conf}$ (color 3), yielding a 4-class annotation map $Y_{Disco} \in \{0,1,2,3\}^{H \times W}$ including background; (2) **Dual-Branch Segmentation Network**: a semantic branch predicts the foreground/background semantic map $P_{sem}$, and a coloring branch predicts the 4-class coloring map $P_{color}$; (3) **Decoupled Constraint Loss System**: five losses are jointly optimized—semantic loss $\mathcal{L}_{sem}$, coloring loss $\mathcal{L}_{color}$, consistency loss $\mathcal{L}_{cons}$, conflict loss $\mathcal{L}_{conf}$, and the pivotal adjacency constraint loss $\mathcal{L}_{adj}$. At inference, instances are reconstructed via topological decoding from the coloring map.

### Key Designs

1. **Cross-Dataset Topological Analysis and CAG Construction**:

    - Function: Provides a data-driven theoretical foundation for the graph coloring paradigm and guides framework design.
    - Mechanism: Instance masks are converted into a CAG $G=(V,E)$, where each node $v_i$ corresponds to instance $s_i$, and edges are defined as $E = \{(v_i, v_j) \mid \mathcal{N}(s_i) \cap s_j \neq \emptyset\}$, with $\mathcal{N}(s_i)$ being the pixel set of instance $s_i$ after 3×3 dilation. Analysis across four datasets reveals: PanNuke is 100% bipartite (0% conflict rate), DSB2018 has a 1.99% conflict rate, CryoNuSeg 5.64%, and GBC-FS 2025 30.49%. In all non-bipartite datasets, 3-cycles account for over 90% of odd cycles, and 24.64% of nodes in GBC-FS exhibit secondary conflicts.
    - Design Motivation: This analysis forms the cornerstone of the paper. It quantitatively reveals the extent to which the 2-coloring assumption breaks down in practice, while demonstrating that conflicts concentrate in local 3-cycle structures—providing ample justification for the divide-and-conquer strategy without globally upgrading to a higher-chromatic-number model.

2. **Explicit Marking — Conflict-Aware Label Generation**:

    - Function: Transforms the NP-hard optimal graph coloring problem into a learnable classification task.
    - Mechanism: BFS is applied to the CAG to extract the maximal bipartite subgraph, partitioning nodes into $V_1$ (color 1), $V_2$ (color 2), and $V_{conf}$ (color 3). While the residual graph $G_{rem}$ could in principle be recursively decomposed, the authors make a key engineering decision based on the "sufficient representation principle": recursion is terminated and all conflict nodes are uniformly assigned color 3. This is because conflict nodes are few but topologically complex, making further coloring marginally beneficial at increasing computational cost. The resulting annotation map is $Y_{Disco} \in \{0,1,2,3\}^{H \times W}$.
    - Design Motivation: Optimal coloring is NP-hard; BFS provides an efficient approximation. Splitting the problem into a "resolved bipartite subgraph" and a "pending conflict set" achieves correct coloring for the vast majority of nodes at minimal computational cost, providing clear learning targets for subsequent implicit disambiguation.

3. **Implicit Disambiguation — Adjacency Constraint Loss $\mathcal{L}_{adj}$**:

    - Function: Resolves secondary conflicts—which discrete labels cannot encode—in continuous probability space.
    - Mechanism: For each instance $s_i$, the mean color probability vector $\bar{P}(s_i)$ is computed, and the cosine similarity is minimized over all edges $(v_i, v_j) \in E$ in the adjacency graph: $\mathcal{L}_{adj} = \frac{1}{|E|} \sum_{(v_i,v_j) \in E} \frac{\bar{P}(s_i) \cdot \bar{P}(s_j)}{\|\bar{P}(s_i)\| \|\bar{P}(s_j)\|}$. This can be interpreted as a supervised contrastive loss where pixels of the same instance are positive samples (aggregated by $\mathcal{L}_{color}$) and adjacent instances are explicit negative samples (repelled by $\mathcal{L}_{adj}$). In bipartite regions, this reinforces orthogonal 2-color encodings; in secondary conflict regions, it compels the network to learn separable representations along secondary feature dimensions.
    - Design Motivation: Since all conflict nodes share the color-3 label under explicit marking, mutually adjacent conflict nodes cannot be distinguished. $\mathcal{L}_{adj}$ provides repulsive gradients in continuous space, pushing conflict nodes with identical labels but different topological contexts toward distinct probability distributions. Ablation results show that $\mathcal{L}_{adj}$ alone contributes PQ +5.69%, the largest individual contribution among the five losses.

### Loss & Training

The total loss is a linear combination of five components: $\mathcal{L}_{total} = \mathcal{L}_{sem} + \mathcal{L}_{color} + \mathcal{L}_{cons} + \mathcal{L}_{conf} + \mathcal{L}_{adj}$

- $\mathcal{L}_{sem}$: Semantic segmentation loss for foreground/background binary classification.
- $\mathcal{L}_{color}$: Weighted cross-entropy loss for 4-class coloring, with class balancing for the conflict class.
- $\mathcal{L}_{cons}$ (Consistency Loss): Suppresses erroneous conflict-color predictions in bipartite regions: $\mathcal{L}_{cons} = \mathbb{E}_{i \in M_{bip}}[(\sigma(P_{color}(i))_t)^2]$—driving the conflict-color probability of bipartite nodes toward 0.
- $\mathcal{L}_{conf}$ (Conflict Loss): Encourages accurate conflict-color prediction in conflict regions: $\mathcal{L}_{conf} = \mathbb{E}_{i \in M_{conf}}[(1 - \sigma(P_{color}(i))_t)^2]$—driving the conflict-color probability of conflict nodes toward 1.
- $\mathcal{L}_{adj}$ (Adjacency Constraint Loss): Repels the probability representations of adjacent instances in probability space (see above).

$\mathcal{L}_{cons}$ and $\mathcal{L}_{conf}$ form a push-pull mechanism: the former pushes conflict-color predictions away in simple regions, while the latter pulls them in for conflict regions. Training uses the Adam optimizer with an initial learning rate of $1 \times 10^{-4}$, weight decay of $5 \times 10^{-4}$, a 10× learning rate decay at epoch 70, linear warmup for the first 100 iterations, 200 total training epochs, and 8 × RTX 4090 GPUs.

## Key Experimental Results

### Main Results

Comprehensive comparison with 8 mainstream methods across four datasets (reporting AJI and PQ only):

| Dataset | Method | AJI (↑) | PQ (↑) | Note |
|--------|------|---------|--------|------|
| PanNuke | CellPose | 0.6262 | 0.5918 | Distance/orientation map |
| PanNuke | FCIS | 0.6394 | 0.6109 | 4-coloring |
| PanNuke | **Disco** | **0.6566** | **0.6271** | PQ +1.62% |
| DSB2018 | CellPose | 0.8247 | 0.7647 | Distance/orientation map |
| DSB2018 | FCIS | 0.8287 | 0.7739 | 4-coloring |
| DSB2018 | **Disco** | **0.8426** | **0.7781** | PQ +0.42% |
| CryoNuSeg | CellPose | 0.5876 | 0.5724 | Distance/orientation map |
| CryoNuSeg | FCIS | 0.5944 | 0.5793 | 4-coloring |
| CryoNuSeg | **Disco** | **0.6134** | **0.5970** | PQ +1.77% |
| GBC-FS 2025 | CellPose | 0.4376 | 0.4218 | Distance/orientation map |
| GBC-FS 2025 | FCIS | 0.4518 | 0.4379 | 4-coloring |
| GBC-FS 2025 | **Disco** | **0.5209** | **0.5087** | **PQ +7.08%** |

### Ablation Study

Framework-level comparison (GBC-FS 2025):

| Method | Coloring Scheme | Adjacency Constraint | DICE | AJI | PQ |
|------|---------|---------|------|-----|-----|
| Baseline | 2-Color | None | 0.727 | 0.379 | 0.338 |
| FCIS | 4-Color | Feature orthogonality constraint | 0.779 | 0.452 | 0.438 |
| **Disco** | **Explicit Marking** | **Probability-space $\mathcal{L}_{adj}$** | **0.814** | **0.521** | **0.509** |

Loss component ablation (GBC-FS 2025):

| Configuration | $\mathcal{L}_{cons}$ | $\mathcal{L}_{conf}$ | $\mathcal{L}_{adj}$ | AJI | PQ |
|------|:---:|:---:|:---:|-----|-----|
| Explicit marking only | ✗ | ✗ | ✗ | 0.449 | 0.426 |
| + Consistency + Conflict | ✓ | ✓ | ✗ | 0.471 | 0.458 |
| + Adjacency constraint | ✗ | ✗ | ✓ | 0.506 | 0.483 |
| **Disco (Full)** | **✓** | **✓** | **✓** | **0.521** | **0.509** |

### Key Findings

- **Greater conflict node density amplifies Disco's advantage**: With a 30.49% conflict rate in GBC-FS 2025, Disco surpasses FCIS by 7.08% PQ (absolute) / 16.2% (relative); in PanNuke with 0% conflict rate, the gain is only 1.62%, demonstrating that the framework adaptively degenerates to efficient 2-coloring.
- **$\mathcal{L}_{adj}$ is the core engine**: Adding $\mathcal{L}_{adj}$ alone (without $\mathcal{L}_{cons}$ and $\mathcal{L}_{conf}$) raises PQ from 0.426 to 0.483 (+5.69%), far exceeding the contribution of the consistency + conflict losses (+3.26%). The full combination of five losses achieves PQ 0.509.
- **Disco vs. 4-coloring**: Disco (dynamic 2+1 coloring) comprehensively outperforms FCIS (fixed 4-coloring) across all datasets, demonstrating that the divide-and-conquer strategy is superior to brute-force color count expansion—fewer colors yield clearer learning targets, with conflict handling delegated to the loss system rather than the label system.
- **SQ dimension advantage**: On DSB2018, while FCIS is marginally better on DQ, Disco significantly outperforms on SQ, indicating that Disco produces more accurate cell contours.

## Highlights & Insights

- **"Analyze first, then design" paradigm**: Rather than directly proposing a method, the paper first conducts systematic topological analysis across four datasets, quantitatively revealing the failure modes of 2-coloring and the distribution of conflicts, before designing the framework accordingly. This data-structure-driven methodology is highly rigorous and avoids ad hoc design choices.
- **Elegant realization of divide-and-conquer philosophy**: Instead of pursuing a theoretically complete high-chromatic-number model (4-coloring), the paper exploits the empirical observation that most graphs are bipartite, covering the majority of nodes with minimal cost (2-coloring + conflict marking) and delegating exceptions to continuous-space constraints. This "coarse labels + fine constraints" paradigm is transferable to other label-noisy or ambiguous settings.
- **Clever design of the adjacency constraint loss**: Cosine similarity constraints are applied in probability space (rather than feature space). Compared to FCIS's orthogonality constraints in feature space, probability space more directly corresponds to classification decision boundaries, and the contrastive loss definition—"positive samples = same instance, negative samples = adjacent instances"—naturally leverages topological information derived from graph structure.
- **Conflict maps as interpretability tools**: The predicted conflict map itself quantifies the topological complexity of tissue organization, offering a new perspective for data-driven pathology research—not only yielding accurate segmentation but also informing pathologists that "the cellular arrangement in this region is particularly complex."

## Limitations & Future Work

- **Non-uniqueness of BFS decomposition**: Maximal bipartite subgraph extraction depends on BFS traversal order, and different random starting points may yield different $V_1, V_2, V_{conf}$ partitions. The paper does not discuss the effect of this randomness on training stability. Deterministic decomposition algorithms or consensus over multiple BFS runs could be explored.
- **Information loss from assigning all conflict nodes the same label**: When the internal structure of the conflict set is complex (e.g., $\chi(G_{rem}) > 2$), a single conflict color loses potential coloring information. While $\mathcal{L}_{adj}$ partially compensates, it may be insufficient for very large conflict clusters. Recursive decomposition of $V_{conf}$ (at least one level) could be considered.
- **Dataset homogeneity**: All four datasets consist of H&E or fluorescence-stained cell nuclei, and GBC-FS 2025 is a proprietary single-WSI dataset. The generalizability of the method to other dense instance segmentation scenarios (e.g., satellite remote sensing for buildings, crowded pedestrian detection, dense text detection) remains unvalidated.
- **Inference efficiency not discussed**: Does inference require constructing a CAG? If so, the latency of building the graph from predicted masks and decoding instances could become a bottleneck in real-time pathology analysis settings.

## Related Work & Insights

- **vs. FCIS (ICML 2025)**: The first work to apply the four-color theorem to cell segmentation, using fixed 4-coloring with feature orthogonality constraints. Disco's core improvements are twofold: (1) dynamic coloring replaces fixed coloring—using 2 colors to reduce learning burden in simple regions and introducing a third color only for conflict regions; (2) adjacency constraints in probability space replace orthogonality constraints in feature space, more directly corresponding to classification decisions. Disco outperforms FCIS by 7.08% PQ on GBC-FS 2025.
- **vs. HoverNet / StarDist**: Classic distance/orientation map-based methods relying on local pixel-level information and complex post-processing. Disco avoids error propagation from post-processing through global topological modeling, with particularly pronounced advantages in dense clusters.
- **vs. CellPose (Nature Methods 2025)**: A general-purpose gradient-flow-based segmentation method with strong performance but still limited by local information in densely overlapping scenarios. Disco surpasses it on all datasets.
- **Transferable insights**: The framework of "graph-structure-driven divide-and-conquer + continuous-space constraints" is broadly applicable to any dense instance segmentation task with local topological conflicts, such as stuff-thing boundary handling in panoptic segmentation and dense object separation in 3D point cloud instance segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ Combining graph coloring theory with deep learning via a divide-and-conquer strategy is novel, with rigorous topological analysis directly guiding design; however, the core contributions are primarily at the loss function level.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four heterogeneous datasets, 8 comparison methods, framework-level ablation, loss component ablation, and feature space visualization provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ The narrative logic is clear and the derivation from analysis to design is naturally coherent, though some descriptions are slightly verbose.
- Value: ⭐⭐⭐⭐ Fills a critical gap in the graph coloring paradigm for practical use (handling non-bipartite graphs) with direct applicability to dense pathology segmentation.
- **vs. HoVer-Net**: A classic cell segmentation method; DISCO demonstrates clear advantages in dense scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Graph coloring theory-driven divide-and-conquer strategy is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets + detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ Topological analysis is clear.
- Value: ⭐⭐⭐⭐ Practically meaningful for dense cell segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ICCV 2025\] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation](../../ICCV2025/medical_imaging/coin_confidence_score-guided_distillation_for_annotation-free_cell_segmentation.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/medical_imaging/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/medical_imaging/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)

</div>

<!-- RELATED:END -->
