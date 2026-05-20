---
title: >-
  [Paper Note] Relatron: Automating Relational Machine Learning over Relational Databases
description: >-
  [ICLR 2026][Graph Learning][Relational Databases] This work systematically compares relational deep learning (RDL/GNN) and deep feature synthesis (DFS) on predictive tasks over relational databases…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Relational Databases"
  - "Graph Neural Networks"
  - "Deep Feature Synthesis"
  - "Architecture Selection"
  - "Homophily"
date: 2026-05-08
content_hash: c68398b470c475d3
---

# Relatron: Automating Relational Machine Learning over Relational Databases

**Conference**: ICLR 2026
**arXiv**: [2602.22552](https://arxiv.org/abs/2602.22552)  
**Code**: [https://github.com/amazon-science/Automating-Relational-Machine-Learning](https://github.com/amazon-science/Automating-Relational-Machine-Learning)  
**Area**: Graph Learning / AutoML
**Keywords**: Relational Databases, Graph Neural Networks, Deep Feature Synthesis, Architecture Selection, Homophily

## TL;DR
This work systematically compares relational deep learning (RDL/GNN) and deep feature synthesis (DFS) on predictive tasks over relational databases, finding that neither dominates uniformly and performance is highly task-dependent. The authors propose Relatron — a task-embedding-based meta-selector that leverages RDB task homophily and affinity embeddings for automatic architecture selection, achieving up to 18.5% improvement in joint architecture–hyperparameter search.

## Background & Motivation

**Background**: Predictive modeling over relational databases (RDB) follows two main paradigms: DFS (programmatically composing aggregation primitives to generate feature tables, then applying a tabular learner) and RDL (end-to-end training of GNNs on heterogeneous entity–relation graphs). Both outperform relation-agnostic baselines.

**Limitations of Prior Work**: It remains entirely unknown which paradigm is superior in which setting. Practitioners lack principled guidance for choosing between DFS and RDL. Validation performance is often an unreliable proxy for model selection — more extensive search can actually lead to worse test performance (the "over-tuning" effect).

**Key Challenge**: (a) No single architecture dominates across all tasks; (b) there is a substantial gap between the configuration selected by validation and the test-optimal configuration, particularly when temporal splits introduce distribution shift.

**Goal**: Given an RDB task, automatically select between RDL and DFS and determine the specific architecture configuration.

**Key Insight**: A large-scale architecture search is conducted to build a "performance bank," followed by analysis of the factors driving the RDL–DFS performance gap. RDB task homophily and training scale emerge as key predictors.

**Core Idea**: High homophily → linear aggregation in DFS suffices; low homophily → nonlinear aggregation in RDL is advantageous. A meta-classifier is trained on task embeddings (homophily + affinity + scale) to enable automatic macro- and micro-architecture selection.

## Method

### Overall Architecture

Construct a decomposed design space for RDL and DFS → conduct large-scale architecture search to build a performance bank → analyze drivers of the performance gap → design task embeddings → train the meta-selector Relatron → apply loss landscape metrics for post-selection.

### Key Designs

1. **RDB Task Homophily (Definition 1)**:

    - **Function**: Measures label consistency along meta-paths in an RDB task.
    - **Mechanism**: Defines self-loop meta-paths $m$ on an augmented heterogeneous graph and computes $H(\mathcal{G};m) = \frac{1}{|\mathcal{E}_m|}\sum \mathcal{K}(\hat{y}_u, \hat{y}_v)$. Dot-product similarity is used for classification tasks and Pearson correlation for regression. Adjusted homophily is also supported to correct for class imbalance.
    - **Design Motivation**: Spearman $\rho = -0.43$ ($p < 0.05$) indicates a strong correlation between homophily and the RDL–DFS performance gap. Lower homophily corresponds to a larger RDL advantage.

2. **Anchor Affinity Embeddings**:

    - **Function**: Captures structural, feature-based, and temporal properties of a task.
    - **Mechanism**: Path affinity (single forward pass of randomly initialized GraphSAGE/NBFNet + linear fit), feature affinity (zero-training validation performance via TabPFN), temporal affinity (statistics of label evolution over time), and $\log(N_{train})$ training scale.
    - **Design Motivation**: Homophily alone captures message-passing preference but additional signals are needed for path model preference, feature quality, and temporal dynamics.

3. **Loss Landscape Post-Selection**:

    - **Function**: Selects more robust checkpoints from the top candidates identified by validation performance.
    - **Mechanism**: Three metrics — first-order $P_1$ (worst-case finite-difference slope), second-order $P_2$ (largest eigenvalue of the Hessian), and energy barrier $P_{bar}$ (maximum loss ridge along a ray). Preference is given to flatter minima.
    - **Design Motivation**: The validation–test gap is reflected in the loss landscape geometry; flatter minima are more robust to distribution shift.

### Loss & Training

The meta-classifier is trained on the performance bank with leave-one-out (LOO) evaluation, using homophily, statistical, and temporal features. Search efficiency: computational cost is only $1/10$ that of Fisher information matrix-based methods.

## Key Experimental Results

### Main Results

| Method | LOO Accuracy (val selection) | LOO Accuracy (test selection) | Avg. Compute Time |
|--------|------------------------------|-------------------------------|-------------------|
| Model-free (ours) | 87.5% | 79.2% | 0.48 min |
| Training-free model | 66.7% | 66.7% | 5 min |
| Autotransfer (anchor) | 66.7% | 66.7% | 50 min |
| Simple heuristic | 70.8% | 75.0% | 0 min |

Relatron achieves up to 18.5% improvement over strong baselines in joint HPO, at 10× lower computational cost.

### Ablation Study

| Configuration | Kendall corr. (w/o g) | Kendall corr. (w/ g) | Note |
|---------------|-----------------------|----------------------|------|
| Model-free | 0.066 | 0.163 | Best task similarity |
| Training-free | -0.038 | -0.030 | Negative correlation |
| Autotransfer | -0.049 | -0.011 | Expensive and negatively correlated |

### Key Findings
- **RDL does not consistently outperform DFS**: Performance is highly task-dependent, with each paradigm showing clear advantages in distinct settings.
- **Macro-selection resolves most of the problem**: Once the correct paradigm (RDL/DFS) is chosen, the validation–test gap narrows substantially.
- **Homophily is the strongest predictor**: Adjusted homophily yields a Spearman $\rho = -0.43$ with the RDL–DFS performance gap.
- **Over-tuning effect**: Larger search budgets can degrade performance — Relatron's macro-selection effectively mitigates this.
- **Validation is unreliable**: Under temporal splits, the configuration selected by validation diverges significantly from the test-optimal configuration.

## Highlights & Insights
- **The underestimated value of DFS**: On suitable tasks, DFS can fully outperform sophisticated GNNs; the key is matching the method to task properties.
- **Theoretical explanation for the homophily-driven RDL advantage**: Under low homophily, linear aggregation conflates positive and negative signals, whereas RDL can learn relational weights that flip contribution signs.
- **Loss landscape post-selection** serves as a practical generalization metric transferable to other AutoML settings.

## Limitations & Future Work
- Task embedding correlations are overall modest (Kendall $\tau$ at most 0.163), limiting the effectiveness of transfer HPO.
- Foundation models (e.g., KumoRFM) are not included.
- The performance bank is limited in scale (< 20 tasks); meta-learning would benefit from larger coverage.
- Loss landscape metrics are only applicable for within-family comparisons.

## Related Work & Insights
- **vs. KumoRFM**: Relational foundation models achieve strong performance, but implementation details are not publicly available. Relatron targets efficient from-scratch training scenarios.
- **vs. Autotransfer**: Fisher information matrix-based task embeddings are computationally expensive and perform poorly on RDB tasks.
- **vs. Griffin**: Cross-table attention frequently underperforms GNNs.

## Rating
- Novelty: ⭐⭐⭐⭐ The definition of RDB task homophily is novel, though the overall methodological framework follows standard meta-learning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 tasks, large-scale architecture search, performance bank, and multi-faceted ablations — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with in-depth theoretical analysis.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical practical pain point in RDB ML; the performance bank has long-term research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Relational Graph Transformer](relational_graph_transformer.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](../../NeurIPS2025/graph_learning/moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[NeurIPS 2025\] When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning](../../NeurIPS2025/graph_learning/when_no_paths_lead_to_rome_benchmarking_systematic_neural_relational_reasoning.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)
- [\[AAAI 2026\] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](../../AAAI2026/graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)

</div>

<!-- RELATED:END -->
