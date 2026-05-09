---
title: >-
  [Paper Note] Disentangling Hyperedges through the Lens of Category Theory
description: >-
  [NeurIPS 2025][Self-Supervised Learning][hypergraph] This work is the first to analyze hyperedge disentanglement through the lens of category theory. By deriving a naturality condition, it establishes a "factor representation consistency" criterion (aggregation-then-disentanglement vs. disentanglement-then-aggregation should yield consistent results), and proposes Natural-HNN, which comprehensively outperforms 14 baselines across 6 cancer subtype classification datasets (BRCA F1: 75.7% → 80.4%) while achieving 100% accuracy in capturing the functional context of genetic pathways.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - hypergraph
  - disentangled representation
  - category theory
  - naturality condition
  - genetic pathway
  - cancer subtype classification
date: 2026-05-08
content_hash: 321385ea549d32a9
---

# Disentangling Hyperedges through the Lens of Category Theory

**Conference**: NeurIPS 2025
**arXiv**: [2510.16289](https://arxiv.org/abs/2510.16289)
**Code**: [Natural-HNN](https://github.com/Yoonho-Lee-AI4Science/Natural-HNN)
**Area**: Self-Supervised
**Keywords**: hypergraph, disentangled representation, category theory, naturality condition, genetic pathway, cancer subtype classification

## TL;DR
This work is the first to analyze hyperedge disentanglement through the lens of category theory. By deriving a naturality condition, it establishes a "factor representation consistency" criterion (aggregation-then-disentanglement vs. disentanglement-then-aggregation should yield consistent results), and proposes Natural-HNN, which comprehensively outperforms 14 baselines across 6 cancer subtype classification datasets (BRCA F1: 75.7% → 80.4%) while achieving 100% accuracy in capturing the functional context of genetic pathways.

## Background & Motivation

**State of the Field**: Disentangled representation learning has been successfully applied to graph-structured data for factor capture at the node level (DisenGCN), edge level (DisenHAN), and subgraph level (HSDN). However, hyperedge disentanglement—capturing latent contextual factors in group interactions (hyperedges)—has not been systematically studied.

**Typical Application Scenario**: Genetic pathways are canonical instances of hyperedges: a set of genes (nodes) performs specific biological functions through group interactions, and the functional context of a pathway (e.g., signal transduction, metabolic regulation) determines how the gene group interaction influences disease labels.

**Limitations of Prior Work**: The most prevalent disentanglement criterion assumes "factor representation similarity"—if the $k$-th factor representations of two nodes are similar, then factor $k$ is relevant to that edge. This assumption does not hold for hyperedges: the context of a group interaction need not be reflected in the similarity among participants. For instance, researchers from different domains convening to discuss a complex problem share a common discussion topic that is unrelated to their mutual similarity.

**Root Cause**: A general criterion is needed—one that does not rely on data-specific assumptions and is instead derived from the definition of hyperedge disentanglement itself.

**Starting Point**: The paper employs the compositional structural perspective of category theory to analyze hypergraph message-passing neural networks. It finds that entangled and disentangled representations are mappings of the same partially ordered set (poset) structure under different functors, and the naturality condition between these functors directly yields a disentanglement criterion.

**Core Idea**: For factors relevant to the group interaction of a hyperedge, the paths "aggregate-then-disentangle" and "disentangle-then-aggregate" should produce consistent factor representations. This naturality condition constitutes a general criterion for hyperedge disentanglement.

## Method

### Overall Architecture
Category-theoretic formalization → derivation of the naturality condition as the disentanglement criterion → dual-branch architecture (aggregation-first / disentanglement-first) for computing factor relevance scores → Natural-HNN integrating relevance scores into hypergraph message passing.

### Key Designs

**1. Category-Theoretic Formalization (Sections 3.1–3.2)**

- **PISet Category**: A poset category is defined with node set $\mathcal{V}$ and hyperedge set $\mathcal{E}$ as objects, and set inclusion as morphisms. The hypergraph topology is naturally encoded as compositional relations within the category.
- **DLRep Category**: A deep learning representation category is defined with vector representations (node features $X$, hyperedge representations $H$, updated representations $Y$) as objects and transformation operations (aggregation, encoding, etc.) as morphisms.
- **Functor Mapping**: The entangled representation functor $F: \mathbf{PISet} \to \mathbf{DLRep}$ and the disentangled representation functor $G: \mathbf{PISet} \to \mathbf{DLRep}$ are two distinct numerical realizations of the same poset structure.
- **Naturality Condition**: A natural transformation $\alpha$ exists between $F$ and $G$. For any morphism $f$, the commutative diagram $f^{en} \circ \alpha_{H,k} = \alpha_{X,k} \circ f_k^{dis}$ must hold for each relevant factor $k$. This implies that the hyperedge representation $H_k^{dis}$ for factor $k$ should be consistent regardless of the computational path taken.

**2. Factor Representation Consistency Criterion (Section 4.1)**

A dual-branch architecture implements verification of the naturality condition:

- **Aggregation-First Branch**: All nodes within hyperedge $e_j$ are first mean-aggregated to obtain an entangled hyperedge representation, which is then disentangled by the $k$-th MLP:
$$\tilde{h}_{e_j}^k = \text{MLP}_k(\text{mean}(\{x_{v_i} \mid v_i \in e_j\}))$$

- **Disentanglement-First Branch**: Each node is first disentangled by the $k$-th MLP, and the resulting disentangled node representations are then mean-aggregated:
$$h_{e_j}^k = \text{mean}(\{\text{MLP}_k(x_{v_i}) \mid v_i \in e_j\})$$

- **Relevance Score Computation**: The consistency between the two branches is measured via $L_2$-normalized bilinear similarity:
$$\alpha_i^k = \sigma\left(\frac{h_{e_i}^k}{\|h_{e_i}^k\|_2} W_k \frac{\tilde{h}_{e_i}^{k\top}}{\|\tilde{h}_{e_i}^k\|_2}\right)$$

where $W_k \in \mathbb{R}^{d/K \times d/K}$ is a learnable parameter matrix. A high $\alpha_i^k$ indicates that factor $k$ is highly relevant to the group interaction of hyperedge $e_i$.

**3. Natural-HNN Model Architecture (Sections 4.2–4.3)**

Each layer comprises three steps:
- **Node-to-Hyperedge Propagation**: The dual-branch procedure computes the hyperedge representation and relevance score for each factor; the final factor representation is $\alpha_i^k h_{e_i}^k$.
- **Hyperedge-to-Node Propagation**: Node representations are updated by aggregating hyperedge representations weighted by factor relevance: $y_{v_i}^k = \frac{\sum_{e_j \ni v_i} \alpha_j^k h_{e_j}^k}{\sum_{e_j \ni v_i} \alpha_j^k}$
- **Output Fusion**: The $K$ factor representations are concatenated, linearly interpolated 1:1 with the node's own disentangled representation, and normalized via LayerNorm: $z_{v_i} = \text{LayerNorm}(0.5 \cdot y_{v_i} + 0.5 \cdot h_{v_i})$

### Loss & Training
- Primary loss: classification cross-entropy $\mathcal{L}_{task}$
- Optional factor discrimination loss: $\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{dis}$, encouraging different factors to capture distinct information
- The factor discrimination loss reduces inter-factor Pearson correlation (from ~0.15 to ~0.10) but has minimal impact on performance (−0.3%), and is therefore treated as an optional component

## Key Experimental Results

### Main Results: 6 Cancer Subtype Classification Datasets (Macro F1, 14 Baselines)

| Cancer Type | Natural-HNN | Runner-up | Gain |
|-------------|-------------|-----------|------|
| BRCA (Breast) | **80.4%** | HSDN 75.7% | +4.7% |
| STAD (Gastric) | **65.9%** | HSDN 62.9% | +3.0% |
| SARC (Sarcoma) | **74.5%** | UniGCNII 72.8% | +1.7% |
| LGG (Low-grade Glioma) | **70.7%** | ED-HNN 70.0% | +0.7% |
| HNSC (Head & Neck) | **86.2%** | ED-HNNII 84.5% | +1.7% |
| CESC (Cervical) | **88.1%** | ED-HNNII 89.5% | −1.4% |

The method achieves the best performance on 5 of 6 datasets, comprehensively outperforming the hypergraph disentanglement method HSDN and attention-based methods HyperGAT/SHINE.

### Functional Context Capture Validation (RQ2)

Top-15 pathways are selected via SHAP values, clustered using CliXO, and evaluated against Lin's BMA semantic similarity:
- **Natural-HNN**: 16/16 pathway clusters with correctly captured functional similarity
- **HSDN**: Only 8/16 clusters captured (50%)
- Even when evaluating on pathways selected as important by HSDN itself, Natural-HNN captures functional similarity more effectively

### Ablation Study

| Configuration | BRCA F1 |
|---------------|---------|
| Natural-HNN (full) | 80.4% |
| w/o naturality weighting (w/o $\alpha$) | 75.6% (−4.8%) |
| w/o factor discrimination loss | 80.1% (−0.3%) |

### Generalization & Hyperparameter Sensitivity (RQ3)
- As training set size decreases from 50% to 10%, performance degradation is comparable to convolutional/DeepSet-style methods and better than attention-based methods
- Factor count $K \in \{2, 4, 8\}$: all settings capture core strong similarities; $K=4$ performs best
- Excessively small representation dimensions cause some functional similarity to be lost; excessively large dimensions yield marginally higher overall similarity than ground truth; yet core patterns are preserved in all settings
- The factor discrimination loss weight $\lambda$ has minimal effect on functional context capture

## Highlights & Insights
- **Elegant Translation from Category Theory to Algorithm**: The naturality condition (commutativity of the commutative diagram) maps directly onto the computable "dual-branch consistency" metric, converting theoretical depth into concrete performance gains.
- **Unsupervised Functional Discovery**: Trained solely on cancer subtype labels, the model autonomously discovers functional contexts that align with independently established cancer-relevant pathways.
- **Breaking the Similarity Assumption**: This work is the first to propose a hyperedge disentanglement criterion that does not rely on the assumption "similar factor → similar representation," fundamentally broadening the applicability of disentanglement methods.
- **A Proof-of-Concept Model as a Strong Method**: Natural-HNN, implemented with only mean aggregation and MLP encoders, substantially outperforms complex baselines, underscoring the importance of theory-guided design.

## Limitations & Future Work
- The high notational density and technical prerequisites of the category-theoretic formalization may limit community accessibility and broader adoption.
- Experiments are validated solely on genetic pathway cancer subtype classification; other hyperedge application domains—such as opinion dynamics in social networks or recommender systems—remain unexplored.
- The search space for factor count $K$ is narrow ({2, 4, 8}); ablations over a wider range would provide deeper insight into the method's characteristics.
- Hypergraph-level classification relies on simple concatenation rather than pooling, and topology-aware pooling methods are absent.

## Related Work & Insights
- **vs. DisenGCN/DisenHAN** (node/edge-level disentanglement): These methods are based on a factor similarity criterion and are not directly applicable to hyperedges.
- **vs. HSDN** (hypergraph structure-level disentanglement): Also relies on a similarity criterion, focusing on substructures rather than hyperedge semantics; the proposed criterion is more general and achieves superior performance.
- **vs. ED-HNN** (equivariant hypergraph message passing): A strong architecture lacking theoretical guidance for disentanglement; Natural-HNN provides a theoretical foundation via category theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A rare yet substantive application of category theory in graph learning; the first systematic treatment of hyperedge disentanglement.
- Experimental Thoroughness: ⭐⭐⭐ Solid as a proof-of-concept (6 datasets + 14 baselines + ablation + generalization + sensitivity analysis), but the application domain is narrow.
- Writing Quality: ⭐⭐⭐ Theoretical derivations are rigorous, though readability of the category theory sections is limited by high notational density.
- Value: ⭐⭐⭐⭐ Provides the first theoretical foundation for disentanglement in hypergraph learning; the functional context discovery carries substantive biological significance.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2026\] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](../../CVPR2026/self_supervised/omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)

<!-- RELATED:END -->
