---
title: >-
  [Paper Note] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning
description: >-
  [AAAI 2026][Self-Supervised Learning][Graph Contrastive Learning] This paper proposes EPA-GRL (Explanation-Preserving Augmentation for Graph Representation Learning), which employs a GNN explainer trained with a small number of labels to identify semantic subgraphs (explanation subgraphs). During augmentation, only the non-semantic portions (marginal subgraphs) are perturbed, achieving semantics-preserving graph augmentation. EPA-GRL significantly outperforms semantics-agnostic random augmentation methods across 6 benchmarks.
tags:
  - AAAI 2026
  - Self-Supervised Learning
  - Graph Contrastive Learning
  - Semantics-Preserving Augmentation
  - Explainable AI
  - Semi-Supervised Learning
  - GNN Explainer
date: 2026-05-08
content_hash: 7e0e419682d58886
---

# Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning

**Conference**: AAAI 2026
**arXiv**: [2410.12657](https://arxiv.org/abs/2410.12657)
**Code**: [https://github.com/realMoana/EPA-GRL](https://github.com/realMoana/EPA-GRL)
**Area**: Self-Supervised Learning
**Keywords**: Graph Contrastive Learning, Semantics-Preserving Augmentation, Explainable AI, Semi-Supervised Learning, GNN Explainer

## TL;DR
This paper proposes EPA-GRL (Explanation-Preserving Augmentation for Graph Representation Learning), which employs a GNN explainer trained with a small number of labels to identify semantic subgraphs (explanation subgraphs). During augmentation, only the non-semantic portions (marginal subgraphs) are perturbed, achieving semantics-preserving graph augmentation. EPA-GRL significantly outperforms semantics-agnostic random augmentation methods across 6 benchmarks.

## Background & Motivation
**State of the Field**: Graph Contrastive Learning (GCL) learns invariant representations by generating two augmented views of the same graph. Methods such as GraphCL and JOAO adopt augmentation strategies including random node/edge dropping.

**Limitations of Prior Work**: Existing augmentation strategies are **semantics-agnostic** — random perturbations may destroy key substructures relevant to classification (e.g., benzene rings in molecular graphs), causing augmented graphs to lose core semantics and degrading downstream classification performance.

**Root Cause**: Effective augmentation must simultaneously satisfy (1) semantic preservation and (2) introduction of variance; however, existing methods focus exclusively on the latter.

**Paper Goals**: How can graph augmentation preserve semantics while introducing sufficient variance?

**Starting Point**: Leverage GNN explainability techniques to identify semantic subgraphs, protecting them during augmentation while perturbing only the non-semantic parts.

**Core Idea**: Train an explainer with a small number of labels → identify semantic subgraphs → perturb only the non-semantic portions = semantics-preserving graph augmentation.

## Method

### Overall Architecture
EPA-GRL is a two-stage approach:
1. **Pre-training Stage**: Train a GNN classifier $f_{pt}$ and explainer $\Psi$ using a small number of labeled graphs.
2. **Representation Learning Stage**: Use the frozen explainer to generate semantic subgraphs → perturb only the marginal part → train the encoder via contrastive learning.

### Key Designs

1. **GNN Explainer Pre-training**:

    - **Function**: Learn a parametric explainer that takes a graph $G$ as input and outputs an explanation subgraph $G^{(exp)}$.
    - **Mechanism**: Train under the Graph Information Bottleneck (GIB) principle: $\arg\min_{\Psi} \sum CE(Y; f_{pt}(\Psi(G))) + \lambda|\Psi(G)|$. The first term ensures that the subgraph retains classification-relevant information; the second enforces compactness (preventing inclusion of irrelevant components).
    - **Design Motivation**: With a small number of labels, the explainer learns to identify class-discriminative substructures and generalizes to unlabeled graphs.

2. **Explanation-Preserving Augmentation**:

    - **Function**: Generate augmented graphs that preserve semantics.
    - **Mechanism**: $G^{(exp)} = \Psi(G)$ (kept unchanged); $\Delta G = G \setminus G^{(exp)}$ (randomly perturbed). Perturbation strategies include: node dropping on $\Delta G$, edge dropping on $\Delta G$, attribute masking on $\Delta G$, subgraph sampling from $\Delta G$, and mixup (replacing the current marginal subgraph with that of another graph).
    - **Design Motivation**: The semantic subgraph is protected from corruption, while the marginal part provides sufficient variance.

3. **Theoretical Analysis**:

    - **Function**: Formally establish the superiority of semantics-preserving augmentation.
    - **Mechanism**: On a modified BA-2motifs graph, it is proven that a semantics-preserving encoder $f_{enc}^{sp}$ achieves classification error approaching 0, whereas a semantics-agnostic encoder $f_{enc}^{sa}$ achieves error approaching 1/2 (random guessing) — the gap can be made arbitrarily large.

### Loss & Training
- The explainer is trained with GIB loss combined with cross-entropy (CE) loss.
- Representation learning employs GraphCL (contrastive) or SimSiam loss.
- Semi-supervised setting: a small number of labels are used to train the explainer, while all graphs (including unlabeled ones) participate in contrastive learning.

## Key Experimental Results

### Main Results
Graph classification accuracy on 6 benchmarks (BA-2motifs, MUTAG, PROTEINS, DD, NCI1, COLLAB):

| Method | BA-2motifs | MUTAG | PROTEINS | NCI1 |
|--------|-----------|-------|----------|------|
| GraphCL (random aug) | Baseline | Baseline | Baseline | Baseline |
| JOAO | Marginal gain | Marginal gain | Marginal gain | Marginal gain |
| AD-GCL | Moderate | Moderate | Moderate | Moderate |
| **EPA-GraphCL** | **Significant gain** | **Significant gain** | **Significant gain** | **Significant gain** |
| **EPA-SimSiam** | **Significant gain** | **Significant gain** | **Significant gain** | **Significant gain** |

### Ablation Study

| Configuration | Performance | Remarks |
|---------------|-------------|---------|
| Random-Aug (semantics-agnostic) | Large drop in GNN classification accuracy | Semantics destroyed |
| EPA-Aug (semantics-preserving) | GNN classification accuracy close to original | Semantics preserved |
| No explainer (full-graph random aug) | Poor performance | Lack of semantic protection |
| Varying label ratio (1–10%) | More labels → better explainer | Effective even at 1% |

### Key Findings
- On BA-2motifs, Random-Aug degrades GNN accuracy from near 100% to below 50%, while EPA-Aug maintains accuracy above 90%.
- t-SNE visualization confirms that EPA embedding distributions remain consistent with those of the original graphs, whereas Random-Aug causes significant distributional shift.
- EPA is plug-and-play and yields consistent improvements when combined with different contrastive learning frameworks (GraphCL/SimSiam).
- EPA remains effective even when as few as 1–5% of labeled graphs are used to train the explainer.

## Highlights & Insights
- **Bridging XAI and Data Augmentation**: The connection is elegant — repurposing explainable AI techniques for data augmentation establishes a novel paradigm of "using explanations to protect semantics."
- **Compelling Theoretical Guarantees**: On the analytical model of BA-2motifs, the gap in classification error between semantics-preserving and semantics-agnostic encoders can be made arbitrarily large.
- **Strong Practicality**: EPA is plug-and-play and can be combined with any graph contrastive learning framework.

## Limitations & Future Work
- **Explainer quality depends on label quantity and GNN quality**: If the pre-trained GNN is of poor quality, the explainer may identify incorrect semantic subgraphs.
- **Additional pre-training cost**: Training the GNN and explainer introduces an extra step compared to purely unsupervised methods.
- **Assumption that semantics = subgraph**: In some graphs, semantics may not reduce to simple subgraph structures (e.g., global attribute patterns).
- **Evaluated only on graph-level tasks**: Whether node-level contrastive learning can similarly benefit remains unexplored.

## Related Work & Insights
- **vs. GraphCL/JOAO**: These methods rely on random augmentation without semantic preservation. EPA consistently outperforms them across all benchmarks.
- **vs. AD-GCL**: AD-GCL uses adversarial augmentation to learn *what not to keep*, whereas EPA uses an explainer to learn *what to keep* — the two approaches are complementary in direction.
- **vs. ENGAGE**: ENGAGE identifies important nodes in an unsupervised manner, but without label guidance it may capture structures that are not class-discriminative. EPA uses a small number of labels to ensure that the identified structures are genuinely semantically meaningful.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bridging XAI and GRL is a fully novel direction, well-supported by both theory and experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six datasets, multiple frameworks, ablation studies, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, the method is presented naturally, and the theoretical analysis adds value.
- Value: ⭐⭐⭐⭐ Provides a new perspective and tool for addressing "how to perform effective augmentation" in graph contrastive learning.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[AAAI 2026\] HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association](hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](self-supervised_inductive_logic_programming.md)
- [\[AAAI 2026\] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)

<!-- RELATED:END -->
