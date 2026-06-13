---
title: >-
  [Paper Note] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data
description: >-
  [ICLR 2026][Time Series][Relational Database] This paper proposes the Relational Transformer (RT) architecture, which leverages task table prompting, cell tokenization…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Relational Database"
  - "Zero-Shot Learning"
  - "Transformer"
  - "Foundation Model"
  - "Relational Attention"
date: 2026-05-08
content_hash: 55942959d1eaad09
---

# Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data

**Conference**: ICLR 2026
**arXiv**: [2510.06377](https://arxiv.org/abs/2510.06377)  
**Code**: [snap-stanford/relational-transformer](https://github.com/snap-stanford/relational-transformer)  
**Area**: Relational Data Modeling / Foundation Models
**Keywords**: Relational Database, Zero-Shot Learning, Transformer, Foundation Model, Relational Attention

## TL;DR
This paper proposes the Relational Transformer (RT) architecture, which leverages task table prompting, cell tokenization, and a Relational Attention mechanism to enable zero-shot transfer to unseen datasets and tasks after pretraining on multiple relational databases. The 22M-parameter model achieves 93% of fully supervised AUROC in the zero-shot setting, significantly outperforming a 27B LLM at 84%.

## Background & Motivation
Pretrained Transformers have demonstrated strong zero-shot adaptation to new tasks in sequence modeling, yet the relational data domain still lacks architectures capable of cross-dataset and cross-task transfer. The core challenge lies in the diversity of relational data: heterogeneous schemas, graph structures, and functional dependencies make designing a universal architecture highly non-trivial. Existing methods are typically trained on a single dataset and cannot be directly applied to unseen databases. Large language models offer some generalization capability but fall short in understanding structured relational data (a 27B LLM achieves only 84% AUROC). The central motivation of this work is to build a general-purpose, pretrain-and-transfer architecture for relational data, analogous to foundation models in the text domain.

## Method

### Overall Architecture
RT takes as input a relational database (comprising multiple tables linked via primary and foreign keys) along with a downstream task specified in the form of a "task table." The model converts each cell in the tables into tokens via cell tokenization, models relationships along three dimensions—columns, rows, and primary-foreign key links—through Relational Attention, and outputs the final predictions.

### Key Designs

1. **Task Table Prompting**: Analogous to prompt design in NLP, RT specifies downstream tasks through a "task table." Concretely, the task table contains the IDs of entities to be predicted and a target column to be filled in; the model fills the target column using contextual information from the database. This design allows the same pretrained model to be applied zero-shot to diverse prediction tasks (e.g., customer churn prediction, sales forecasting) without task-specific fine-tuning or retrieval of in-context examples.

2. **Cell Tokenization with Metadata**: Rather than serializing table rows into text, RT treats each cell as an independent token augmented with table/column metadata. This tokenization preserves the structured nature of relational data, enabling the model to be aware of which table and column a cell originates from. Compared to processing serialized text with an LLM, this design exploits the structural information of relational data more efficiently.

3. **Relational Attention**: This is the core innovation of RT. While conventional Transformer attention operates over one-dimensional sequences, RT introduces three attention patterns:

    - **Column Attention**: Computes attention across different rows within the same column, learning intra-column statistical patterns and distributional features.
    - **Row Attention**: Computes attention across different columns within the same row, capturing inter-attribute relationships of the same entity.
    - **Primary-Foreign Key Attention**: Propagates information along primary-foreign key links in the database, modeling relationships between entities across tables.

   Together, these three attention types enable effective information propagation and feature learning over the complex structure of relational databases.

4. **Masked Token Prediction Pretraining**: RT adopts masked token prediction as its pretraining objective, analogous to BERT's MLM, but applied to cell tokens in relational data. By pretraining jointly on multiple heterogeneous RelBench datasets spanning tasks such as customer churn and sales forecasting, the model learns general-purpose representations of relational data.

### Loss & Training
- **Pretraining**: Joint pretraining across multiple RelBench datasets using a leave-one-out strategy (holding out the target dataset).
- **Continued Pretraining**: Further pretraining on the target dataset while holding out the target task.
- **Fine-tuning**: Fine-tuning on the target task, demonstrating high sample efficiency.
- The model contains only 22M parameters, far fewer than the 27B LLM used for comparison.

## Key Experimental Results

### Main Results

| Method | Metric | Zero-Shot Result | Note |
|--------|--------|-----------------|------|
| RT (22M, zero-shot) | Binary AUROC | 93% of fully supervised | Single forward pass |
| 27B LLM (zero-shot) | Binary AUROC | 84% of fully supervised | Much larger model, still below RT |
| RT (fine-tuned) | Binary AUROC | SOTA | High sample efficiency |

### Key Findings
- RT achieves an average zero-shot AUROC of 93% relative to the fully supervised baseline, requiring only a single forward pass.
- The 22M-parameter RT surpasses the 27B-parameter LLM by 9 percentage points in the zero-shot setting.
- Fine-tuned RT achieves state-of-the-art performance with high sample efficiency.
- Ablation analysis indicates that zero-shot transfer in RT depends on the joint contribution of task context, relational attention patterns, and schema semantic information.

### Ablation Study

| Configuration | Description |
|--------------|-------------|
| w/o Relational Attention | Significant performance degradation, confirming the importance of relational attention. |
| w/o Task Table Prompting | Zero-shot inference is no longer feasible. |
| w/o Metadata | Cell tokens lack structural information, leading to performance degradation. |
| Pretraining dataset quantity | More datasets yield better generalization. |

## Highlights & Insights
- **Elegant architectural design**: Relational Attention models relationships along three dimensions—columns, rows, and primary-foreign keys—perfectly aligned with the structural properties of relational databases.
- **Remarkable efficiency**: A 22M-parameter model substantially outperforms a 27B LLM in the zero-shot setting, demonstrating that inductive biases tailored to data characteristics are more effective than brute-force parameter scaling.
- **Task Table Prompting as a key innovation**: Encoding the task itself as a table enables the model to perform diverse tasks without additional task-specific heads or fine-tuning.
- **Inaugurating the foundation model era for relational data**: Analogous to GPT for text and ViT for images, RT provides the first effective foundation model framework for the relational data domain.

## Limitations & Future Work
- Pretraining data are primarily drawn from RelBench, limiting domain coverage; broader generalization requires validation on more diverse heterogeneous data sources.
- Despite strong zero-shot performance, a gap of approximately 7% remains relative to the fully supervised baseline; few-shot settings offer room for further improvement.
- Zero-shot results are currently demonstrated only on binary classification tasks; regression and multi-class classification tasks require further exploration.
- Scalability to highly complex schemas (dozens of tables, intricate many-to-many relationships) remains to be validated.
- Subsequent work PluRel further improves pretraining and architecture through synthetic data.

## Related Work & Insights
- **vs. TabPFN**: TabPFN handles single-table data, whereas RT operates on multi-table relational data.
- **vs. LLMs**: LLMs serialize tables into text, discarding structural information, while RT preserves relational structure.
- **vs. GNNs on relational data**: RT models directly at the cell level and replaces GNN message passing with Relational Attention.
- **Insight**: Designing inductive biases matched to a specific data modality is more efficient than applying general-purpose LLMs; the concept of task prompting can be extended to other structured data domains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Relational Feature Caching for Accelerating Diffusion Transformers](relational_feature_caching_for_accelerating_diffusion_transformers.md)
- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](../../NeurIPS2025/time_series/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)

</div>

<!-- RELATED:END -->
