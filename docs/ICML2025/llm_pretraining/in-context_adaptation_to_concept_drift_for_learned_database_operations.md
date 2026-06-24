---
title: >-
  [Paper Note] In-Context Adaptation to Concept Drift for Learned Database Operations
description: >-
  [ICML2025][LLM Pretraining][Concept Drift] Proposes the FLAIR framework, which utilizes database execution results as context to achieve in-context adaptation. This addresses concept drift without runtime parameter updates, achieving a $5.2\times$ speedup and a 22.5% error reduction on tasks such as cardinality estimation.
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Concept Drift"
  - "in-context learning"
  - "Learned Database Operations"
  - "Online Adaptation"
  - "Bayesian Meta-Training"
date: 2026-05-08
content_hash: 0748c111579d4dde
---

# In-Context Adaptation to Concept Drift for Learned Database Operations

**Conference**: ICML2025  
**arXiv**: [2505.04404](https://arxiv.org/abs/2505.04404)  
**Authors**: Jiaqi Zhu, Shaofeng Cai, Yanyan Shen, Gang Chen, Fang Deng, Beng Chin Ooi
**Affiliations**: Beijing Institute of Technology, National University of Singapore, Shanghai Jiao Tong University, Zhejiang University  
**Code**: To be confirmed  
**Area**: LLM Pre-training  
**Keywords**: Concept Drift, in-context learning, Learned Database Operations, Online Adaptation, Bayesian Meta-Training

## TL;DR

Proposes the FLAIR framework, which utilizes database execution results as context to achieve in-context adaptation. This addresses concept drift without runtime parameter updates, achieving a $5.2\times$ speedup and a 22.5% error reduction on tasks such as cardinality estimation.

## Background & Motivation

**Core Problem**: Learned operations in databases (e.g., cardinality estimation, approximate query processing) rely on Machine Learning (ML) models. However, the dynamic nature of databases (frequent inserts/deletes/updates) leads to data distribution shifts (concept drift), which degrades the performance of trained models.

**Limitations of Prior Work**:

- **Transfer Learning / Active Learning / Multi-Task Learning**: These are reactive training methods that require collecting new data and updating model parameters after deployment, bringing significant latency and computational overhead.
- **Neglecting Inter-Query Dependencies**: Traditional methods process each input independently, failing to utilize shared contextual information across database queries.
- **Static Mapping Assumption**: Inability to adapt to continuously changing data distributions.

**Key Observation**: Databases possess a unique property—the execution results (ground-truth labels) of queries can be obtained immediately. For example, executing a `SELECT COUNT` query instantly yields the true row count. This real-time feedback can serve as contextual information for dynamic adaptation.

**Two Key Challenges**:

1. How to achieve instant adaptation to constantly changing data without retraining/fine-tuning?
2. How to dynamically inject contextual information into the modeling process for context-aware prediction?

## Method

### Overall Architecture

FLAIR formalizes the adaptation process as conditional prediction:

$$f: (\mathbf{x} \mid \mathcal{C}_t) \to \mathbf{y}$$

where $\mathbf{x}$ is the input query, $\mathcal{C}_t$ is the dynamic context memory (composed of recent queries and their execution results), and $\mathbf{y}$ is the predicted output. FLAIR consists of two cascaded modules:

$$\mathcal{M}_F(\mathbf{x}; \Theta_\mathcal{T}, \Theta_\mathcal{D}) = \mathcal{M}_{DDE}(\mathcal{M}_{TFM}(\mathbf{x}; \Theta_\mathcal{T}); \Theta_\mathcal{D})$$

### Module 1: Task Featurization Module (TFM)

TFM is responsible for standardizing different database operations into a unified structural representation, consisting of three steps:

**1) Data Encoding**: Each database column uses a histogram to represent its distribution. The attribute value range is discretized using $\delta$ bins, normalized to $[0,1]$, and aggregated to form a data vector $X_D$ with dimension $\delta \times \sum_{i=1}^{N} n_i$.

**2) Query Encoding**:
- Join predicates are encoded as one-hot binary vectors $\mathbf{q}_J$
- Filter predicates (including comparison operators $<, \leq, \geq, >, =$) are encoded as boundary vectors $\mathbf{q}_F$
- The final query vector is $\mathbf{q}_\mathcal{Q} = \langle \mathbf{q}_J, \mathbf{q}_F \rangle$

**3) Task Featurization**: A lightweight Transformer architecture is adopted:
- **Data Modeling Stage**: Data vectors are processed through multiple layers of Multi-head Self-attention (MHSA) + FFN + LayerNorm to capture the implicit joint distribution among attributes:

$$\hat{\mathbf{Z}}^l = \text{MHSA}(\text{LN}(\mathbf{Z}^{l-1})) + \mathbf{Z}^{l-1}$$
$$\mathbf{Z}^l = \text{FFN}(\text{LN}(\hat{\mathbf{Z}}^l)) + \hat{\mathbf{Z}}^l$$

- **Interaction Modeling Stage**: Multi-head Cross-attention (MHCA) is utilized, where the query vector $\mathbf{q}_\mathcal{Q}$ serves as the query, and the output of the data modeling stage, $\mathbf{Z}_\mathcal{O}$, serves simultaneously as the key and value. This allows the TFM to dynamically focus on data features relevant to the current query, outputting a unified task vector.

### Module 2: Dynamic Decision Engine (DDE)

DDE is the core of FLAIR, responsible for performing dynamic predictions based on context.

**Bayesian Meta-Training Mechanism**:

- A large number of "tasks" are sampled from a synthetic prior distribution for pre-training, equipping the DDE with prior knowledge to handle diverse dynamic scenarios.
- Leveraging the concept of Prior-data Fitted Networks (PFN), pre-training on synthetic data allows the model to capture uncertainties and various distribution shifts.
- **Key Advantage**: Post-deployment requires zero parameter updates (no gradient-based optimization), adapting to new concepts solely by updating the context $\mathcal{C}_t$.

**Online Inference Process**:

1. After a query is executed by the database, the (query, result) pair is appended to the context memory $\mathcal{C}_t$.
2. For a new query $\mathbf{x}$, the TFM extracts the task vector.
3. The DDE leverages the current context $\mathcal{C}_t$ to perform conditional prediction, outputting results aligned with the current concept.

### Fundamental Differences from Traditional Methods

| Characteristics | Traditional Reactive Methods | FLAIR |
|------|--------------|-------|
| Adaptation Approach | Parameter Retraining/Fine-tuning | Context update, no parameter updates |
| Adaptation Latency | Requires new data collection + training | Instantaneous (utilizing execution results as feedback) |
| Inter-Query Information | Processed independently | Shared via context |
| Computational Cost | High (gradient optimization) | Low (forward inference only) |

## Key Experimental Results

### Experimental Setup

- **Task Coverage**:
    - System-internal tasks: Cardinality Estimation
    - User-oriented tasks: Approximate Query Processing, In-database Data Analytics

### Main Results

According to the key data reported in the paper's Abstract and Introduction:

| Metric | FLAIR Performance |
|------|-----------|
| Adaptation Speed | **5.2$\times$** faster than SOTA |
| GMQ Error Reduction (Cardinality Estimation) | **22.5%** |
| Query Execution Efficiency Improvement (Integrated into PostgreSQL) | Up to **1.9$\times$** |

### Integration with PostgreSQL

- FLAIR was integrated into PostgreSQL for query optimization.
- Achieved up to a $1.9\times$ speedup in end-to-end query execution.
- Demonstrated the practicality of the framework in real-world database systems.

## Highlights & Insights

1. **Paradigm Innovation**: Initiates the application of in-context learning to learned database operations. Exploiting the inherent immediate feedback property of databases (where execution results act as labels) is a natural and elegant design.
2. **Task Agnosticism**: FLAIR is designed as a task-agnostic framework. By unifying representation across different tasks via TFM, the same framework can be applied to diverse database operations.
3. **Zero Runtime Training Overhead**: Once deployed, it requires no parameter optimization, accomplishing adaptation solely through forward inference. This is crucial for database systems with stringent real-time requirements.
4. **Bayesian Meta-Training**: Pre-training via synthetic prior distributions enables the model to generalize to unseen concept drift scenarios, preventing overfitting to specific datasets.

## Limitations & Future Work

1. **Context Window Limitations**: The size of the context memory $\mathcal{C}_t$ is limited, which might fail to capture sufficient distribution shift information in extremely rapid drift scenarios.
2. **Coverage of Synthetic Priors**: Bayesian meta-training relies on synthetic data distributions. If the concept drift patterns in real-world scenarios deviate significantly from the synthetic priors, the model's generalization capabilities may be compromised.
3. **Task Scope**: Currently, validation is limited to SPJ (Select-Project-Join) query-related tasks. Its applicability to more complex query types (e.g., nested subqueries, recursive queries) remains unexplored.
4. **Granularity of Histogram Encoding**: Data encoding depends on histograms with a fixed number of bins ($\delta$), which may incur significant information loss for high-dimensional or sparse distributions.
5. **Incomplete Content Cache**: The cached file only contains sections up to Section 3.1.2. Complete DDE details and experimental settings are not present in the cache, which may impact the technical depth of the notes.
6. **Scalability with Large-scale Databases**: In scenarios involving a massive number of tables and complex schemas, the encoding dimension of the TFM may scale excessively.

## Related Work & Insights

- **Prior-data Fitted Networks (PFN)**: The concept of pre-training on synthetic priors from PFN is borrowed by FLAIR for Bayesian meta-training.
- **In-Context Learning (ICL)**: The ICL paradigm from NLP is transferred to the database domain, demonstrating the feasibility of cross-domain method migration.
- **Concept Drift Handling**: Traditional drift detection and retraining methods are too expensive in database scenarios. FLAIR provides a detection-free continuous adaptation scheme.
- **Insights**: The idea of leveraging natural feedback signals from systems for in-context adaptation could potentially be extended to other systems with immediate feedback (e.g., recommender systems, network routing optimization).

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduces the in-context adaptation paradigm into database operations for the first time; leveraging execution results as context is an ingenious insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple database tasks, with integration into PostgreSQL validating its practicality. However, the incomplete cache prevents a detailed evaluation of all experiments.
- Writing Quality: ⭐⭐⭐⭐ — The problem-solution presentation is logically clear, with rigorous formal definitions.
- Value: ⭐⭐⭐⭐ — Resolves key practical issues in learned database operations with a practical and elegant framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] When Can In-Context Learning Generalize Out of Task Distribution?](when_can_in-context_learning_generalize_out_of_task_distribution.md)
- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](../../NeurIPS2025/llm_pretraining/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)
- [\[ICLR 2026\] Learned Meta-Tokens for Language Modeling](../../ICLR2026/llm_pretraining/learned_meta-tokens_for_language_modeling.md)
- [\[ACL 2025\] TokAlign: Efficient Vocabulary Adaptation via Token Alignment](../../ACL2025/llm_pretraining/tokalign_vocab_adaptation.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)

</div>

<!-- RELATED:END -->
