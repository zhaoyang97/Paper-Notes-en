---
title: >-
  [Paper Note] Geometry of Decision Making in Language Models
description: >-
  [NeurIPS 2025][Model Compression][Intrinsic Dimension] By measuring the **Intrinsic Dimension (ID)** of hidden representations across layers in 28 open-source Transformer models at scale, this paper reveals a consistent "low–high–low" pattern: early layers operate on low-dimensional manifolds, middle layers expand the representational space, and later layers re-compress into low-dimensional representations aligned with decision-making.
tags:
  - NeurIPS 2025
  - Model Compression
  - Intrinsic Dimension
  - Hidden Representation Geometry
  - Decision Dynamics
  - Multiple-Choice QA
  - Transformer
date: 2026-05-08
content_hash: c6f6995cae9ae609
---

# Geometry of Decision Making in Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.20315](https://arxiv.org/abs/2511.20315)
**Code**: None
**Area**: Model Compression
**Keywords**: Intrinsic Dimension, Hidden Representation Geometry, Decision Dynamics, Multiple-Choice QA, Transformer

## TL;DR

By measuring the **Intrinsic Dimension (ID)** of hidden representations across layers in 28 open-source Transformer models at scale, this paper reveals a consistent "low–high–low" pattern: early layers operate on low-dimensional manifolds, middle layers expand the representational space, and later layers re-compress into low-dimensional representations aligned with decision-making.

## Background & Motivation

### Core Problem

Large language models (LLMs) exhibit strong generalization across diverse tasks, yet the internal decision-making process—how a model progresses from input to prediction—remains opaque. Prior work has studied internal mechanisms through the lens of attention analysis and probing classifiers, but the **geometric structure** of hidden representations has received comparatively little attention.

### Intrinsic Dimension (ID)

Intrinsic dimension is a statistic that measures the true dimensionality of the manifold on which a set of high-dimensional data points lies. Intuitively, even when a hidden layer has $d = 4096$ dimensions, the representation vectors may effectively concentrate on a submanifold of far lower dimensionality. ID can reveal the degree to which each layer compresses or expands information.

### Why the MCQA Setting

Multiple-choice question answering (MCQA) provides a well-defined decision structure: the model must select the correct answer from a fixed set of options. This enables researchers to:
- Quantify each layer's contribution to the final decision via layer-wise accuracy
- Correlate ID variation with decision quality
- Control experimental variables and avoid the uncertainty inherent in open-ended generation tasks

## Method

### Overall Architecture

The experimental pipeline proceeds as follows:
1. Select 28 open-source Transformer models spanning different architectures and parameter scales
2. Feed test data through each model on MCQA tasks
3. Extract hidden representations at each layer
4. Compute the intrinsic dimension at each layer using multiple ID estimators
5. Simultaneously compute per-layer MCQA accuracy by performing classification directly on each layer's output
6. Analyze the relationship between ID and layer-wise performance

### Key Designs

#### ID Estimation Methods

Multiple ID estimators are employed to ensure the robustness of the conclusions:

| Estimator | Type | Principle |
|:---|:---|:---|
| TwoNN | Local | Based on nearest-neighbor distance ratios |
| MLE (Levina–Bickel) | Local | Maximum likelihood estimation |
| PCA (explained variance) | Global | Proportion of variance explained |
| Other topological methods | Hybrid | Based on persistent homology, etc. |

Using multiple estimators mitigates the bias of any single method and strengthens the credibility of the findings.

#### Layer-wise Performance Quantification

For each layer $l$, the hidden representation $h^{(l)}$ is used directly for prediction:
- Representational similarity between answer options is computed, or a linear probe is applied
- The resulting per-layer MCQA accuracy $\text{Acc}^{(l)}$ is recorded
- The correspondence between $\text{ID}^{(l)}$ and $\text{Acc}^{(l)}$ is established

### Loss & Training

This paper does not involve training new models. All analyses are conducted on existing pretrained models, constituting an **analytical study**.

## Key Experimental Results

### Main Results: ID Variation Pattern

Across all 28 models, the following three-stage pattern is consistently observed:

| Layer Range | ID Behavior | Interpretation |
|:---|:---|:---|
| Early layers (0–20% depth) | Low ID | Input embeddings lie on a low-dimensional manifold; initial encoding is compact |
| Middle layers (20–70% depth) | ID rises to peak | Spatial expansion; model explores rich representations |
| Late layers (70–100% depth) | ID decreases again | Compression into a low-dimensional structure aligned with decision-making |

| Model Category | Representative Models | Early ID | Peak ID | Final ID |
|:---|:---|:---|:---|:---|
| Small (~1B) | Pythia-1B, GPT-Neo-1.3B | ~10–20 | ~40–60 | ~15–25 |
| Medium (~7B) | LLaMA-2-7B, Mistral-7B | ~15–30 | ~80–120 | ~20–40 |
| Large (~13B+) | LLaMA-2-13B, Falcon-40B | ~20–40 | ~100–150 | ~30–50 |

### Ablation Study

#### Relationship Between ID and Layer-wise Performance

| Layer Range | Mean MCQA Accuracy | ID Trend | Relationship |
|:---|:---|:---|:---|
| Early layers | Near random (~25%) | Low ID | Information not yet integrated |
| Middle-to-late layers | Rapid increase | ID begins to decline | Decision formation begins |
| Final layers | Highest | Low ID | Decision compressed into low-dimensional representation |

Key finding: **ID decline and accuracy improvement are highly correlated**, indicating that the model compresses representations onto a low-dimensional manifold precisely when arriving at a decision.

#### Consistency Across Estimators

| Estimator Pair | Spearman Rank Correlation |
|:---|:---|
| TwoNN vs. MLE | > 0.95 |
| TwoNN vs. PCA | > 0.90 |
| MLE vs. PCA | > 0.88 |

The high agreement across estimators validates the robustness of the conclusions.

### Key Findings

1. **Universal "low–high–low" ID pattern**: Consistently observed across all 28 models and multiple ID estimators; this constitutes an architecture- and scale-agnostic property.

2. **ID compression co-occurs with decision formation**: The sharp ID decline in the final layers coincides with a rapid rise in MCQA accuracy, suggesting that late layers project representations onto a structured low-dimensional manifold aligned with task-relevant decisions.

3. **Effect of model scale**: Larger models tend to exhibit higher peak IDs, indicating richer representational spaces in the middle layers, while still ultimately compressing to a relatively low-dimensional decision manifold.

## Highlights & Insights

- **Novelty of the geometric perspective**: Unlike probing or attention analysis, ID analysis provides a more fundamental, task-agnostic geometric measure
- **Large-scale validation across 28 models**: Coverage spans multiple architectures and scales including Pythia, LLaMA, Mistral, Falcon, and GPT-Neo
- **Support for "representation learning as dimensionality selection"**: The results suggest that LLM training can be understood as identifying the correct low-dimensional manifold within a high-dimensional space
- **Implications for layer pruning and early exit**: If late layers are primarily performing dimensionality compression, more efficient alternatives for achieving this step may exist

## Limitations & Future Work

- Validation is limited to the MCQA setting; whether the same ID patterns hold for open-ended generation tasks remains unexamined
- ID estimation under limited sample sizes introduces statistical noise, particularly for extremely high-dimensional representations
- The effect of fine-tuning or RLHF on ID patterns has not been explored
- No quantitative comparison with probing accuracy or information bottleneck theory is provided
- Causal analysis is absent; it remains unclear whether ID variation causes decision formation or is merely a byproduct

## Related Work & Insights

- **Ansuini et al. (2019)**: First systematic study of ID variation patterns in deep networks
- **Cai et al. (2023)**: Analysis of ID in Vision Transformers
- **Information Bottleneck Theory (Shwartz-Ziv & Tishby, 2017)**: Posits that deep learning proceeds through "fitting" and "compression" phases, consistent with the ID patterns reported in this work
- **Mechanistic Interpretability**: Elhage et al., Olsson et al., and others analyze Transformers through the lens of circuits
- This paper complements the mechanistic understanding of LLM internals from a geometric perspective

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Large-scale ID analysis in LLMs represents a new direction
- **Technical Depth**: ⭐⭐⭐ — Experimental design is solid, though the methodology itself is relatively straightforward
- **Practicality**: ⭐⭐⭐ — Analytical in nature, with implications for model compression and interpretability
- **Clarity**: ⭐⭐⭐⭐ — Conclusions are intuitive and clearly presented
- **Overall Score**: 7.5/10

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)

<!-- RELATED:END -->
