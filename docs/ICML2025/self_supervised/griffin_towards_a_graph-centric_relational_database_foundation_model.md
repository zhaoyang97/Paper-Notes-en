---
title: >-
  [Paper Note] Griffin: Towards a Graph-Centric Relational Database Foundation Model
description: >-
  [ICML 2025][Self-Supervised Learning][Relational Database Foundation Model] Griffin is the first foundation model designed for Relational Databases (RDBs). By transforming multi-table structures into heterogeneous graphs, and combining a unified encoder/decoder, cross-attention, and a hierarchical aggregation MPNN, it conducts self-supervised masked completion pre-training on over 150M+ rows of data followed by joint SFT, achieving cross-database, cross-domain…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Relational Database Foundation Model"
  - "Graph Neural Networks"
  - "Self-Supervised Pre-training"
  - "Cross-Table Reasoning"
  - "Transfer Learning"
date: 2026-05-08
content_hash: 8f3753421b7ca084
---

# Griffin: Towards a Graph-Centric Relational Database Foundation Model

**Conference**: ICML 2025  
**arXiv**: [2505.05568](https://arxiv.org/abs/2505.05568)  
**Code**: [github.com/yanxwb/Griffin](https://github.com/yanxwb/Griffin)  
**Area**: Self-Supervised Learning  
**Keywords**: Relational Database Foundation Model, Graph Neural Networks, Self-Supervised Pre-training, Cross-Table Reasoning, Transfer Learning

## TL;DR

Griffin is the first foundation model designed for Relational Databases (RDBs). By transforming multi-table structures into heterogeneous graphs, and combining a unified encoder/decoder, cross-attention, and a hierarchical aggregation MPNN, it conducts self-supervised masked completion pre-training on over 150M+ rows of data followed by joint SFT, achieving cross-database, cross-domain, and cross-task generalized predictions.

## Background & Motivation

While foundation models have achieved massive success in NLP, CV, tabular data, and graph data domains, the field of **relational databases (RDBs)** still lacks a generic foundation model. In RDBs, multiple tables are linked through primary key-foreign key (PK-FK) associations, introducing several challenges:

**Structural Complexity**: Relations among multiple tables are highly intertwined. Simple single-table flattening (or joining) easily loses a significant amount of relational topology.

**Heterogeneous Feature Spaces**: Categorical, numerical, and text features across different RDBs have entirely distinct semantics, making them inherently difficult to unify.

**Diverse Tasks**: Classification and regression tasks coexist, each possessing vastly different label spaces and numerical ranges.

**Large-scale Temporal Constraints**: Real-world RDBs often contain timestamps, requiring predictions to strictly satisfy causal temporal constraints (relying exclusively on historical data).

Existing GNN approaches (such as RDBTOGRAPH, ATJ-Net, or GFS) model RDBs by converting them into graphs but are constrained to small, task-specific models that cannot generalize across databases. Griffin aims to build a **general-purpose RDB foundation model that is pre-trained once and generalizable across multiple tasks**.

## Method

### Overall Architecture

The Griffin pipeline consists of three steps:

1. **Graph Construction**: Map each row of the RDB to a node, and PK-FK relations to edges, establishing a heterogeneous temporal graph.
2. **Subgraph Sampling**: Given a target column, sample a temporally-constrained subgraph $\mathcal{T}^{(L)}$ rooted at the target row, which contains only nodes with timestamps earlier than the target row.
3. **Encoding-MPNN-Decoding**: A unified encoder processes heterogeneous features $\rightarrow$ an MPNN aggregates multi-hop neighbor information $\rightarrow$ a unified decoder outputs the prediction.

### Key Designs

#### 1. Unified Data Encoder

**Categorical/Text Features**: Use a pre-trained text encoder (Nomic Embed) to encode all categorical values and texts into $d$-dimensional vectors. Unlike traditional one-hot encodings or task-bound embedding layers, this approach preserves rich semantic information, allowing the semantic distance between different categories to be naturally measured via cosine similarity.

**Numerical Features**: First map numerical values to a normal distribution using Quantile Normalization, and then use a pre-trained MLP encoder to map the scalar into a $d$-dimensional vector:

$$w = \text{ENC}(x) \in \mathbb{R}^d, \quad y = \text{DEC}(w) \in \mathbb{R}$$

During training, the reconstruction error is minimized using L1 loss $|y - x|$, and LayerNorm (without affine parameters) is applied to prevent representation collapse. Once pre-training is complete, ENC and DEC are frozen and do not participate in subsequent training.

**Metadata Encoding**: Metadata such as table names, column names, and edge types are also processed via the text encoder to serve as additional node and edge features.

**Task Representation**: Convert the column name of the target column using the text encoder to generate a task embedding $t \in \mathbb{R}^d$, enabling the model to distinguish different prediction tasks on the same row.

*Encoding output*: Each node $i$ is associated with a feature tensor $x_i \in \mathbb{R}^{L_i \times d}$, a metadata tensor $m_i \in \mathbb{R}^{L_i \times d}$, and each edge carries a relation vector $e_r \in \mathbb{R}^d$.

#### 2. Improved MPNN Architecture

**Cross-Attention Module**: Traditional GNNs aggregate columns within a node using simple mean pooling, which tends to lose feature-specific information. Griffin introduces cross-attention, using the task embedding and node representation to generate the Query, column name metadata as the Key, and column values as the Value:

$$v_i^l = \text{Attention}^l(\text{QMLP}^l(u_i, t),\ m_i,\ x_i)$$

where $u_i$ is the intermediate node representation and $t$ is the task embedding. This allows the model to selectively focus on task-relevant columns instead of treating all columns equally.

**Key Improvement**: The first layer of cross-attention actually degenerates into mean aggregation (due to the lack of sufficient task information at that stage). Therefore, the first layer is modified to **Self-Attention** to allow column names and column values to interact first, while subsequent layers perform task-conditioned aggregation.

**Hierarchical Aggregation**: Unlike standard MPNNs that aggregate all neighbors uniformly, Griffin employs a two-level aggregation:

$$h_i^{r,l} = \text{Mean}^l(\text{AMLP}^l(u_j) \mid (i,j) \in \mathcal{E}^r)$$
$$h_i^l = \text{Max}^l(h_i^{r,l} \odot e_r \mid r \in R)$$

It first performs Mean aggregation **within the same relation type**, and then performs Max aggregation across different relation types. This prevents domain-specific neighbors from overwhelming the information of other relationships when their count is excessively high, preserving the relational structure.

#### 3. Unified Task Decoder

**Classification Tasks**: Directly use the text embeddings of target class labels as the classification head. Given all candidate category embeddings $z_1, \ldots, z_c$ and the model output $z$, the predicted probability is:

$$P = \text{softmax}([\langle z, z_i \rangle \mid i=1,\ldots,c])$$

This eliminates the need to maintain independent classification heads for each task and enables the handling of classification tasks with varying numbers of categories.

**Regression Tasks**: Pass the output vector through the pre-trained numerical decoder DEC to reconstruct the scalar, and then apply inverse normalization to get the final prediction.

### Loss & Training

Griffin adopts a **three-stage training pipeline**:

**Stage 1: Completion Pre-training**  
On 200+ single-table datasets (~10 million rows), randomly mask a column value in a row and predict the masked value using the remaining columns. The loss is the cosine distance between the predicted embedding and the ground-truth embedding:

$$\mathcal{L} = 1 - \cos(\text{Model}_\theta(T_{i,:\setminus j'}^k),\ \text{Encoder}(T_{i,j'}^k))$$

This stage **does not require labeled data** and constitutes self-supervised pre-training.

**Stage 2: Joint Supervised Fine-Tuning (Joint SFT)**  
Perform supervised training using ground-truth labels on both single-table and RDB datasets. Cross-entropy is used for classification tasks, and L2 loss is used for regression tasks. SFT data is strictly isolated from downstream evaluation data to prevent data leakage.

**Stage 3: Downstream Task Fine-Tuning**  
Perform task-specific fine-tuning on specific downstream tasks.

Three model variants:
- **Griffin-unpretrained**: No pre-training, relying solely on architectural advantages.
- **Griffin-pretrained**: Pre-trained only on single-table data.
- **Griffin-RDB-SFT**: Joint SFT on single-table and RDB data.

## Key Experimental Results

### Main Results

Evaluated on two major benchmarks, 4DBInfer and RelBench, covering 24 tasks (classification + regression) across multiple domains such as e-commerce, sports, social networks, and healthcare.

| Model | Avg Rank↓ | ROC-AUC Rep Task | MAE Rep Task | Note |
|------|----------|----------------|------------|------|
| SAGE | 4.79 | 0.792 (Seznam) | 1.357 (Avito) | GNN Baseline |
| GAT | 5.21 | 0.805 | 1.370 | GNN Baseline |
| PNA | 5.33 | 0.800 | 0.894 | GNN Baseline |
| HGT | 5.33 | 0.797 | 0.669 | Heterogeneous Graph Baseline |
| DFS+FTTransformer | 5.92 | 0.747 | 0.634 | Single Table + DFS |
| DFS+XGB | 7.08 | 0.760 | 0.674 | Single Table + DFS |
| **Griffin-unpretrained** | **3.71** | 0.800 | 0.659 | Architectural Advantage Only |
| **Griffin-pretrained** | **3.04** | **0.813** | **0.659** | + Single-Table Pre-training |

Griffin-unpretrained outperforms all baseline average rankings solely through architectural improvements; incorporating pre-training yields further performance gains.

### Ablation Study

| Config | Avg Rank↓ | ROC-AUC (Retailrocket) | MAE (Avito) | Note |
|------|----------|------------------------|-------------|------|
| Griffin (Full) | **1.4** | 0.716 | -0.659 | Cross-Attention + Max Aggregation |
| Griffin-avg-attention | 2.7 | 0.692 | -0.760 | Cross-Attention $\rightarrow$ Mean Aggregation |
| Griffin-mean-GNN | 1.9 | 0.708 | -0.670 | Max Aggregation $\rightarrow$ Mean Aggregation |

The cross-attention module has the greatest impact on performance; removing it drops the average rank from 1.4 to 2.7. Hierarchical Max aggregation also contributes significantly.

### Key Findings

1. **Architecture alone provides benefits**: Even without pre-training, Griffin outperforms all GNN and DFS baselines in average ranking due to cross-attention and hierarchical aggregation.
2. **Single-table pre-training is universally effective**: Pre-training solely on single-table data (without any RDB elements) generally improves performance on downstream RDB tasks.
3. **Transfer is driven by similarity and diversity**:
    - Commerce $\rightarrow$ Commerce transfer works best (similarity hypothesis).
    - Others $\rightarrow$ Commerce also shows positive transfer, sometimes even outperforming intra-domain transfer (diversity hypothesis).
    - Commerce $\rightarrow$ Others transfer performs poorly (lack of similarity).
4. **Significant advantages in low-data scenarios**: In few-shot fine-tuning, the pre-trained Griffin shows a larger margin of improvement compared to the non-pre-trained version.
5. **Complementary to TabPFNv2+DFS**: Griffin performs better on Commerce-2 and Others-2 tasks, while TabPFNv2 shines on Commerce-1 and Others-1.

## Highlights & Insights

1. **Clever Unified Encoder Design**: Mapping heterogeneous features into the same space using pre-trained text/numerical encoders allows the model to generalize across RDBs with different schemas, which is a core requirement for RDB foundation models.
2. **Classification Head using Label Text Embeddings**: Instead of setting a fixed classification layer, using the inner product with label name text embeddings enables the classifier to naturally adapt to tasks with varying numbers of classes.
3. **First-Layer Self-Attention $\rightarrow$ Subsequent Cross-Attention**: Observing that the first layer of cross-attention degenerates into mean aggregation, the authors targeted and modified it to self-attention, demonstrating a deep analysis of model behavior.
4. **Hierarchical Aggregation (Mean Intra-Relation, then Max Inter-Relation)**: This addresses the imbalance in neighbor counts across different relation types in heterogeneous graphs, with Max keeping the salient signal of each relation type.
5. **Pre-training Data Scale Reaches 150M+ Rows**: The scale reaches the level of a foundation model, covering multiple domains and scenarios.

## Limitations & Future Work

1. **Dependence on Pre-training Data Quality**: Some massive single-table data was excluded from pre-training due to severe distribution shifts from downstream RDBs; automatic selection of effective pre-training data is still to be explored.
2. **Asymmetric Transfer Direction**: Others-1 $\rightarrow$ Others-2 is effective, but the reverse is not, indicating that transfer success is highly dependent on the diversity of pre-training data composition.
3. **High Computational Overhead**: The training cost of a 4-layer MPNN + cross-attention module on large-scale graphs (AWS g6.48x instances) cannot be neglected.
4. **Unexplored Text-Intense Scenarios**: When text columns dominate the RDB, truncating to 512-dimensional text embeddings might be insufficient.
5. **Regression Requires Extra Inverse Normalization**: The pipeline of quantile normalization + pre-trained decoder is complex, and its adaptation to extreme values or heavy-tailed distributions needs verification.
6. **Limited to Prediction Tasks**: It does not cover other RDB-related tasks such as SQL generation or table QA.

## Related Work & Insights

- **Single-Table Foundation Models** (TransTab, XTab, UniTabE, TabPFN): These methods are restricted to single tables and cannot model multi-table relationships. Griffin extends this to RDBs.
- **RDB Graph Methods** (RDBTOGRAPH, GFS, 4DBInfer, RelBench): These provide the RDB $\rightarrow$ Graph modeling paradigm but lack pre-training generalization capabilities.
- **Graph Foundation Models** (OFA, UniGraph): Though these are graph foundation models, they focus on generic graphs and are not optimized for the tabular characteristics of RDBs.
- **Insight**: Incorporating structured data (tables/relational databases) into the foundation model paradigm is a promising direction; the design concept of unified encoder + metadata embeddings is generalizable to other heterogeneous data sources.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First RDB foundation model, defining a pioneering problem. |
| Technical Depth | 4 | Elaborated designs in encoder/MPNN/decoder. |
| Experimental Thoroughness | 5 | 24 tasks + ablation + transfer analysis + raw data. |
| Writing Quality | 4 | Clear structure and rich figures/tables. |
| Value | 4 | Direct application value for enterprise-level RDB prediction scenarios. |
| **Overall** | **4.2** | A solid and systematic work, establishing an important baseline for the RDB foundation model direction. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](../../ICML2026/self_supervised/learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[ICML 2025\] A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints](a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](../../ICML2026/self_supervised/flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)

</div>

<!-- RELATED:END -->
