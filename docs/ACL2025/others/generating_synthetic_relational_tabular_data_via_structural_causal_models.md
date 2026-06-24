---
title: >-
  [Paper Note] Generating Synthetic Relational Tabular Data via Structural Causal Models
description: >-
  [ACL 2025][synthetic data] This paper extends the TabPFN approach for synthetic data generation based on Structural Causal Models (SCMs), proposing a framework capable of generating multi-table relational synthetic tabular data by coupling nodes and latent causal relationships to model cross-table dependencies.
tags:
  - "ACL 2025"
  - "synthetic data"
  - "relational tables"
  - "structural causal models"
  - "DAG"
  - "tabular foundation model"
date: 2026-05-08
content_hash: 1b95272034d9f097
---

# Generating Synthetic Relational Tabular Data via Structural Causal Models

**Conference**: ACL 2025  
**arXiv**: [2507.03528](https://arxiv.org/abs/2507.03528)  
**Code**: None  
**Area**: Data Generation / Tabular Data  
**Keywords**: synthetic data, relational tables, structural causal models, DAG, tabular foundation model

## TL;DR

This paper extends the TabPFN approach for synthetic data generation based on Structural Causal Models (SCMs), proposing a framework capable of generating multi-table relational synthetic tabular data by coupling nodes and latent causal relationships to model cross-table dependencies.

## Background & Motivation

Synthetic tabular data generation has received increasing attention in recent years, especially playing a critical role in training foundation models. TabPFN achieved breakthrough results using large-scale synthetic datasets generated via SCMs, proving the importance of synthetic data to tabular foundation models. However, current generation methods suffer from a core limitation:

**Real-world tabular data is mostly relational**—composed of multiple interconnected tables (such as primary and foreign key tables in a database), but existing methods can only generate single independent tables.

**Learning-based methods** (e.g., Synthetic Data Vault, GNN+diffusion models) require real-world datasets as a basis to extract statistical and relational patterns, while accessible real relational datasets are scarce.

**TabPFN's SCM method**, although independent of real data and naturally capturing causal dependencies, is constrained to single-table scenarios.

The motivation of this paper is clear: to extend the SCM method from single-table to relational multi-table, enabling the unlimited generation of synthetic data with complex cross-table causal dependencies to train relational tabular foundation models.

## Method

### Overall Architecture

The method is based on the Directed Acyclic Graph (DAG) of SCM, executing in three steps: (1) Structure Sampling—building graph topology; (2) Pre-Sampling—calibrating noise scale and classification boundaries; (3) Main Sampling—propagating data and reading out tables. Based on this, it extends to relational data generation by coupling nodes to connect multiple DAGs.

### Key Designs

1. **Structure Sampling**: Barabási-Albert model is used to sample a directed graph. After removing isolated nodes and back edges, a DAG is obtained. The root node data is initialized with a multi-dimensional vector (sampled from Normal/Gamma distributions), and each node defines a propagation function $g_i: \mathbb{R}^{|\text{pa}(i)| \cdot n} \to \mathbb{R}^n$ (a single-layer fully connected neural network + random non-linear activation functions such as ReLU, logabs). **Key difference from TabPFN**: the generation of categorical features is not embedded within the propagation function, but discretized only at the readout stage—this ensures subsequent nodes receive continuous vectors rather than states constrained by the number of classes. Readout projects the $n$-dimensional vector to a scalar through a pooling function (norm, mean, median, etc.).

2. **Pre-Sampling**: To match the noise scale with the data distribution, a noise-free "pre-run" is performed first: root node data is independently sampled and propagated through the entire graph, calculating the 10% and 90% quantiles $q_{0.1}(i), q_{0.9}(i)$ for each node. In the main sampling, noise is scaled by the quantile difference:
    $$x_i = g_i(x_{\text{pa}(i)}) + (q_{0.9}(i) - q_{0.1}(i)) \varepsilon_i$$
   This ensures the propagation signal $g_i$ remains the primary information source, and noise is a moderate perturbation. Used for categorical nodes, k-means is used to cluster pre-sampled data into $K$ classes, defining the categorical pooling function as nearest centroid assignment.

3. **Relational Data Extension (Core Contribution)**: Two DAGs, $\mathcal{G}_{\text{main}}$ and $\mathcal{G}_{\text{add}}$, are sampled independently, and a coupling node $C$ is introduced to connect them ($C$ is caused by the sink node of $\mathcal{G}_{\text{add}}$ and points to the feature node of $\mathcal{G}_{\text{main}}$). To model latent causal influence, the feature nodes of $\mathcal{G}_{\text{add}}$ are additionally connected to the target node of $\mathcal{G}_{\text{main}}$ (yellow edges). Data is sampled twice: once on the merged graph (generating the main table) and once only on $\mathcal{G}_{\text{add}}+C$ (generating the additional table). The two tables are associated through the $C$ column, and their sample sizes can differ.

4. **Design of Categorical Node $C$**: The coupling node uses a relatively large number of classes (e.g., 175 classes) to simulate foreign key columns, making the association between the two tables closer to real database foreign key constraints.

### Loss & Training

This paper proposes a data generation method and does not involve model training or loss functions. The generation process is entirely based on random sampling and deterministic propagation, with all parameters (graph structure, distribution parameters, activation functions) randomly initialized.

## Key Experimental Results

### Main Results

Based on the example graph in Fig. 2, two associated tables (main table 100,000 rows, additional table 500 rows) are generated. Using the EmbDI embedding method + k-NN prediction, we compare using only the main table vs. combining both tables:

| Target Node | Task Type | Main Table Only | Main Table + Additional Table | Trend |
|---|---|---|---|---|
| M4 | Classification (AUC) | ~0.55 | ~0.65 | Combination is Better |
| M6 | Regression (RMSE) | ~0.45 | ~0.35 | Combination is Better |

(Specific values vary with embedding dimensions; under high embedding dimensions, the joint method consistently outperforms the single-table method)

### Ablation Study

| Parameter | Impact |
|---|---|
| Latent dimension $n$ | Larger $n$ makes predicting the target from features harder (larger information compression loss) |
| Activation function type | Affects data complexity and diversity of non-linear relationships |
| Noise ratio (quantile selection) | Controls the balance between signal and noise |
| Number of categories $K$ | Affects the difficulty of classification tasks and separability between classes |

### Key Findings

1. **Validation of cross-table dependency effectiveness**: The latent causal information contained in the additional table indeed affects the target column of the main table, and this information cannot be inferred solely from column $C$ of the main table—which is the core feature of real relational data.
2. When the embedding dimension is sufficiently high, the prediction quality of the combined two tables is stably superior to using only the main table.
3. The latent causal connection (yellow edges) is key: without them, the information propagated by column $C$ would be sufficient to represent all relationships, and the additional table would provide no extra information.

## Highlights & Insights

- **Simple yet goal-oriented method**: Naturally extends the single-table SCM framework to multi-table via coupling nodes and latent causal edges, with a clear concept and straightforward implementation.
- **Notable improvements to the original TabPFN framework**: Separates categorical feature generation from the propagation function and discretizes it only in the readout stage. This prevents the number of categories from restricting the information flow of subsequent nodes—a design modification that is also applicable to single-table scenarios.
- **High scalability**: The same method can chain more DAGs to generate three or more associated tables.

## Limitations & Future Work

1. **Insufficient experimental scale**: Only validated on a single example dataset, lacking large-scale systematic experiments and parameter sensitivity analyses.
2. **Single data type**: Only supports numerical and categorical features, without covering multimodal data such as text and images.
3. **Simple relational structure**: Only validated relations between two tables; complex scenarios such as cross-joins and cyclic dependencies among three or more tables remain unexplored.
4. **Limited evaluation on downstream tasks**: Only validated with simple baselines (EmbDI + k-NN), without testing actual performance in training tabular foundation models.
5. Lacks direct comparison with existing relational data generation methods (e.g., Synthetic Data Vault, Hudovernik 2024).

## Related Work & Insights

- Directly builds upon the SCM data generation framework of TabPFN (Hollmann et al. 2025).
- Synthetic Data Vault (Patki et al. 2016) is the first learning-based relational data generation method, but it relies on real data.
- Hudovernik (2024) combines GNN embeddings and diffusion models to handle relational structures, but also requires real data.
- The advantage of this work lies in being completely independent of real data and capable of large-scale generation, making it suitable for foundation model pre-training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The direction of exploration is clear and reasonable, but the core idea is a natural extension of the TabPFN framework, representing a relatively incremental contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ — Only verified on one example dataset, lacks systematic quantitative metrics, which fits a short-paper level validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Tightly structured, concepts are clearly elaborated, and both the algorithm pseudocode and illustrations are extremely intuitive.
- **Value**: ⭐⭐⭐⭐ — Potentially valuable to the tabular foundation model community, but requires more thorough experiments to prove practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](../../ICLR2026/others/tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](../../NeurIPS2025/others/radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[ACL 2025\] Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)

</div>

<!-- RELATED:END -->
