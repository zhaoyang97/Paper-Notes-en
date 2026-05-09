---
title: >-
  [Paper Note] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models
description: >-
  [AAAI2026][Medical Imaging][open-set classification] This paper proposes the Coarse-to-Fine Classification (CFC) framework, which leverages the zero-shot reasoning capability of LLMs to supply semantically grounded OOD samples and a potential OOD label space for open-set graph node classification, enabling the model not only to detect OOD nodes but also to classify them into specific unknown categories.
tags:
  - AAAI2026
  - Medical Imaging
  - open-set classification
  - OOD detection
  - graph neural networks
  - large language models
  - node classification
date: 2026-05-08
content_hash: b79ef9b1794f93c8
---

# Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models

**Conference**: AAAI2026
**arXiv**: [2512.16244](https://arxiv.org/abs/2512.16244)
**Code**: [sihuo-design/CFC](https://github.com/sihuo-design/CFC)
**Area**: Medical Imaging
**Keywords**: open-set classification, OOD detection, graph neural networks, large language models, node classification

## TL;DR

This paper proposes the Coarse-to-Fine Classification (CFC) framework, which leverages the zero-shot reasoning capability of LLMs to supply semantically grounded OOD samples and a potential OOD label space for open-set graph node classification, enabling the model not only to detect OOD nodes but also to classify them into specific unknown categories.

## Background & Motivation

Graph neural networks (GNNs) perform well in closed-set settings but frequently encounter classes unseen during training (i.e., OOD samples) in real-world deployment. Existing open-set classification methods suffer from the following key limitations:

1. **Reliance on synthetic/auxiliary OOD samples**: Generating large numbers of synthetic samples is computationally expensive and fails to accurately reflect real OOD distributions.
2. **Lack of semantic understanding**: Generated or auxiliary data carries no genuine semantic meaning, leading to overfitting to specific datasets.
3. **Excessively small OOD subspace**: Without semantic OOD samples, the decision boundary for OOD detection becomes overly sharp.
4. **Inability to distinguish different unknown classes**: All unknown classes are collapsed into a single OOD category, severely limiting practical utility in high-stakes scenarios such as medical diagnosis and fraud detection.

The core driving question is: can OOD detection be extended to OOD classification without access to ground-truth OOD labels?

## Core Problem

This paper extends traditional open-set classification from a $(C+1)$-class problem ($C$ in-distribution (ID) classes plus one OOD class) to a $(C+u)$-class problem ($C$ ID classes plus $u$ OOD classes), where $u$ is unknown in the open-set setting. Two critical challenges must be addressed:

- How to approximate the OOD space without any annotation?
- How to derive meaningful labels for unknown classes?

## Method

The CFC framework consists of three core stages:

### 1. LLM Coarse Classifier

Graph data is mapped into the text space, where expert knowledge encoded in the LLM is exploited for preliminary OOD detection and potential label generation. Two detection strategies are designed based on the coverage of the ID label space:

- **Easy-Reject**: Applied when the ID classes cover only a small fraction of their parent category (e.g., Cora, DBLP, WikiCS). A confidence-aware prompt is designed to mark a node as OOD only when the LLM is highly confident, while simultaneously generating anomalous class labels.
- **Hard-Reject**: Applied when the ID classes cover the majority of their parent category (e.g., Citeseer). The LLM first summarizes the parent category of the ID classes to generate a candidate OOD label space, and then performs classification based on the expanded label space.

A confidence threshold of 0.7 is used to filter noisy annotations.

### 2. GNN Fine Classifier

Based on the semantic OOD sample set $\mathcal{V}_{\text{ood}}$ obtained from the coarse classifier, a $(C+1)$-class GNN classifier is constructed:

- **Denoising**: Label propagation $\mathbf{Y}^{l(k)} = \mathbf{D}^{-1}\mathbf{A}\mathbf{Y}^{l(k-1)}$ is applied to correct OOD samples misclassified by the LLM. After each iteration, ID training node labels are reset; OOD samples re-predicted as ID are discarded after $K$ propagation rounds.
- **OOD Data Augmentation**: An improved Manifold Mixup strategy collects $K$ nodes in the training set with low classification confidence (near the decision boundary) and mixes their hidden-layer embeddings with the OOD sample centroid: $\tilde{x}_i = \alpha \boldsymbol{h}_i^k + (1-\alpha)\boldsymbol{h}_c^k$, where hyperparameter $\alpha$ controls the distance between generated samples and OOD samples.
- **Joint Training**: A GCN classifier is trained with cross-entropy loss on $\mathcal{V}_{\text{train}} \cup \mathcal{V}_{\text{ood}}^a$.

### 3. OOD Classification

For OOD samples $\mathcal{V}_{\text{ood}}^f$ detected by the fine classifier, the potential OOD label space generated during the coarse classification stage is utilized. Similar categories are merged via TF-IDF similarity measures, categories with too few samples are filtered out, and a post-processed OOD label space is obtained. LLM prompting is then used to assign final classification labels to the OOD samples.

### Theoretical Analysis

It is proven that by introducing semantic OOD samples, CFC expands the dimensionality of the OOD subspace from $\text{dim}(\mathcal{H}) - (C+1)$ to $\text{dim}(\mathcal{H} + \mathcal{H}') - (C+1)$, thereby producing a smoother and flatter OOD detection decision boundary.

## Key Experimental Results

**Datasets**: Cora, Citeseer, WikiCS, DBLP (text-attributed graphs); Amazon-Computer, Amazon-Photo (non-text graphs). Each dataset designates $u \geq 2$ OOD classes.

**OOD Detection Performance** (two OOD classes, overall accuracy):

| Dataset | NodeSafe (Runner-up) | CFC | Gain |
|---------|---------------------|-----|------|
| Cora | 85.71% | **90.00%** | +4.3% |
| Citeseer | 72.74% | **77.21%** | +4.5% |
| WikiCS | 79.59% | **80.44%** | +0.9% |
| DBLP | 76.21% | **84.03%** | +7.8% |

**OOD Classification Accuracy** (GPT-4o + post-processed OOD label space): Cora 69.76%, Citeseer 70.30%, WikiCS 57.96%, DBLP 48.45%.

**Key Ablation Findings**:
- Even without denoising and Mixup (CFC w/o D/M), the use of semantic OOD samples alone surpasses all baselines.
- OOD detection on Cora improves from 0% (GCN_sigmoid) to 95.74% (CFC).

## Highlights & Insights

1. **Novel Problem Formulation**: This is the first work to extend open-set graph classification from simple OOD detection to OOD classification, defining the $(C+u)$-class problem.
2. **Semantic OOD Samples**: Rather than relying on synthetic or auxiliary data, the framework leverages LLMs to identify samples that are genuinely semantically out-of-distribution, yielding stronger interpretability and practical utility.
3. **Strong Framework Generality**: CFC is not limited to graph data and can be directly extended to the text domain.
4. **Solid Theoretical Grounding**: The advantage of semantic OOD samples is demonstrated through analysis of subspace dimensionality and decision boundary smoothness.
5. **Elegant Dual-Strategy Design**: The Easy-Reject/Hard-Reject design accounts for scenarios with differing degrees of ID label coverage.

## Limitations & Future Work

1. **Dependence on LLM Quality**: The coarse classification stage relies heavily on powerful LLMs such as GPT-4o; open-source models (e.g., Llama2-7b) exhibit notably weaker detection capability.
2. **Constraint on Text-Attributed Graphs**: Non-text graphs require an additional feature encoding step to convert node attributes into textual descriptions, increasing preprocessing complexity.
3. **Limited OOD Classification Accuracy**: The highest accuracy of approximately 70% may be insufficient for high-stakes applications.
4. **Slight Degradation in ID Classification**: Introducing OOD detection causes a 2–5% drop in ID accuracy (e.g., Cora drops from 90.64% to 87.49%).
5. **Computational Cost**: Multiple LLM API calls are required, making real-world deployment non-trivial in terms of cost.
6. **Unknown Number of OOD Classes**: The estimation of $u$ relies entirely on the LLM and post-processing, lacking a more principled method for automatic determination.

## Related Work & Insights

| Method | Requires Synthetic OOD | Supports OOD Classification | Uses LLM | General (Graph + Text) |
|--------|:---:|:---:|:---:|:---:|
| G²Pxy | Yes (proxy unknown nodes) | No | No | No |
| GNNSafe | No (energy propagation) | No | No | No |
| NodeSafe | No (energy propagation) | No | No | No |
| GOLD | No (energy propagation) | No | No | No |
| **CFC** | **No (semantic OOD)** | **Yes** | **Yes** | **Yes** |

The key distinction of CFC lies in its use of LLMs to obtain semantically grounded real OOD samples rather than synthetic ones, and it is the only method capable of multi-class OOD classification.

**Broader Implications**:
- **LLMs as Open-World Sensors**: The zero-shot reasoning capability of LLMs can provide semantic out-of-distribution signals for conventional models, a paradigm extensible to other open-world tasks such as open-vocabulary detection and open-set segmentation.
- **Generality of the Coarse-to-Fine Strategy**: The approach of using a strong but noisy signal for initial screening followed by a structured model for fine-grained discrimination is applicable to annotation-budget-constrained scenarios.
- **Potential in Medical and Security Domains**: The framework has direct applicability in fraud detection, medical diagnosis, and other high-stakes scenarios requiring the discrimination of different types of unknown anomalies.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to define the graph OOD classification problem; the coarse-to-fine LLM+GNN framework design is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six datasets, multiple LLMs, comprehensive ablation studies, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, figures are intuitive, and method description is systematic.
- Value: ⭐⭐⭐⭐ — Addresses the long-neglected OOD sub-classification problem in open-set classification with strong practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](../../ACL2026/medical_imaging/text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)

<!-- RELATED:END -->
