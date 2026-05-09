---
title: >-
  [Paper Note] LangBridge: Interpreting Image as a Combination of Language Embeddings
description: >-
   LangBridge achieves interpretable vision-language alignment by explicitly decomposing visual features into linear combinations of LLM vocabulary embeddings, and supports pretraining-free adapter transfer across different LLMs.
tags:

date: 2026-05-08
content_hash: f2b853b5e9bc9486
---

# LangBridge: Interpreting Image as a Combination of Language Embeddings

## Basic Information

- **Conference**: ICCV 2025
- **arXiv**: 2503.19404
- **Code**: [Project Page](https://curryx-001.github.io/LangBridge.github.io/)
- **Area**: Information Retrieval
- **Keywords**: Vision-language alignment, adapter transfer, vocabulary embedding projection, MLP analysis, cross-model reuse

## TL;DR

LangBridge achieves interpretable vision-language alignment by explicitly decomposing visual features into linear combinations of LLM vocabulary embeddings, and supports pretraining-free adapter transfer across different LLMs.

## Background & Motivation

Mainstream large vision-language models (LVLMs) follow the LLaVA paradigm, employing an MLP to project visual features into the text embedding space of an LLM. While effective, this approach has two core issues:

**Opaque MLP alignment mechanism**: The underlying mechanism by which the MLP bridges the modality gap remains poorly understood, with few studies systematically analyzing its fundamental alignment behavior.

**Non-transferable adapters**: Switching the LLM backbone requires retraining the MLP adapter from scratch due to input dimension mismatches and feature distribution discrepancies, incurring substantial computational cost.

The authors first conduct a systematic investigation of MLP adapter behavior and identify two key findings:
- Visual embeddings exhibit strong correlation with semantically related text tokens (e.g., high cosine similarity between apple image patches and the text "Green Apple").
- The projection capability of the MLP develops progressively during training, gradually learning to project visual features into the corresponding text embedding subspace.

## Method

### Overall Architecture

The core idea of LangBridge is **Language Basis Vector Projection**: representing each visual embedding as a weighted linear combination of LLM vocabulary embeddings:

$$\mathbf{v} = \sum_{k=1}^{N} \beta_k \mathbf{t}_k$$

where $\beta_k$ is a probability distribution over vocabulary tokens. The method consists of three stages:

### Key Designs

**Stage 1: Visual Feature Extraction**

A Vision Transformer (CLIP-ViT-L/14@336px) is used to extract patch-level visual features from the input image:

$$\{v_i\}_{i=1}^{N} = \text{ViT}(\mathcal{I}), \quad v_i \in \mathbb{R}^{D}$$

**Stage 2: Probability Computation**

A two-layer MLP projects visual features into the LLM text embedding space, followed by a linear layer that produces a probability distribution over the vocabulary:

$$\mathbf{p} = \mathbf{W} \cdot \text{MLP}(\mathbf{v})$$

where $\mathbf{W} \in \mathbb{R}^{T \times D}$ and $T$ denotes vocabulary size.

**Stage 3: Linear Combination of Text Embeddings**

The probability distribution is used as coefficients to linearly combine the LLM's vocabulary embeddings:

$$\mathbf{v}_{\text{tokens}} = \sum_{i=1}^{T} p_i \mathbf{e}_i$$

### Vocabulary Selection Strategy

Directly using the full vocabulary embedding matrix is computationally prohibitive (~1B parameters). The authors merge the vocabularies of LLaMA and Qwen, retaining only tokens shared by both, then compute token frequencies over the ShareGPT4V and LLaVA-CC3M-Pretrain-595K datasets to select the top-19,200 high-frequency tokens as a compact shared vocabulary.

### Cross-LLM Adapter Transfer

LangBridge learns only the linear combination relationship between visual patches and vocabulary embeddings (i.e., the probability distribution), rather than a direct dimensional mapping. During transfer:

$$P = \text{LangBridge}_{\text{LLM}_1}(I) \in \mathbb{R}^{|V_{\text{shared}}|}$$
$$\text{Visiontoken}_{\text{LLM}_2} = P \cdot V_{\text{shared}}$$

The probability distribution $P$ can be directly applied to weight the vocabulary embeddings of any target LLM without repretraining.

## Key Experimental Results

### Main Results: Same-Architecture Transfer

Results of LangBridge pretrained on Qwen2-0.5B and directly transferred to larger models:

| SFT-LLM | Connector | GQA | TextVQA | MME | MMBench | MMVeT | POPE | SciQA |
|---|---|---|---|---|---|---|---|---|
| Qwen2-7B | 7B-Pretrain-MLPs | 62.92 | 57.24 | 1938 | 72.7 | 35.5 | 87.8 | 79.44 |
| Qwen2-7B | 0.5B-Pretrain-LB | 63.03 | 57.25 | 1886 | 71.7 | 34.1 | 88.2 | 79.23 |
| Qwen2.5-14B | 14B-Pretrain-MLPs | 63.71 | 61.32 | 2038 | 78.2 | 37.7 | 88.1 | 85.59 |
| Qwen2.5-14B | 0.5B-SFT-LB | 63.92 | 62.02 | 1990 | 77.4 | 38.4 | 87.6 | 84.77 |

Key finding: After transferring a LangBridge connector pretrained on a 0.5B model to a 14B model, performance on TextVQA (+1.14%) and MMVeT (+1.86%) even surpasses the baseline.

### Ablation Study: Vocabulary Size

| Vocab Size | GQA | TextVQA | MME | MMBench | MMVeT | POPE | SciQA |
|---|---|---|---|---|---|---|---|
| 19,200 | 63.15 | 57.34 | 1904 | 71.0 | 31.6 | 88.3 | 79.25 |
| 25,600 | 63.13 | 57.58 | 1842 | 71.8 | 32.9 | 87.9 | 79.01 |
| 32,000 | 63.11 | 57.19 | 1832 | 72.7 | 33.2 | 88.6 | 79.11 |

A vocabulary size of 19,200 significantly outperforms larger vocabularies on MME (by 6–8%) and achieves the best overall performance.

### Other Key Findings

- **Cross-architecture transfer**: Transferring a LangBridge connector pretrained on Qwen2-0.5B to LLaMA3-8B yields a +9.68% gain on MMVeT with overall performance improvements.
- **Standard setting comparison**: LangBridge trained directly on LLaMA3-8B outperforms the MLP baseline on GQA, TextVQA, MME, and MMBench.
- **Computational cost**: Training overhead increases by only ~10% (4.273 vs. 3.876 s/iter), while cross-LLM pretraining can be entirely skipped.

## Highlights & Insights

1. **In-depth mechanistic analysis**: Visualization-based analysis reveals the progressive process by which the MLP adapter learns to project visual features into the text embedding subspace, providing theoretical grounding for the proposed design.
2. **Elegant design philosophy**: Transforming "implicit projection" into "explicit linear combination" renders the vision-language alignment process interpretable.
3. **Practical transferability**: A single pretraining run on a small model enables transfer to multiple large models, substantially reducing the cost of multi-model deployment.
4. **Shared vocabulary strategy**: Cross-model compatible tokens are selected via frequency statistics, achieving cross-architecture compatibility with minimal parameter overhead.

## Limitations & Future Work

- Vocabulary selection relies on frequency statistics over specific training datasets and may not generalize to other data distributions.
- Validation is currently limited to the LLaVA-style framework and has not been extended to other architectures such as Flamingo or BLIP-2.
- A vocabulary size of 19,200 may be insufficient to cover all fine-grained visual concepts.
- On certain benchmarks (e.g., MME), transfer still incurs a 2–4% performance drop.

## Related Work & Insights

- **Ovis** employs a visual embedding table for structured alignment but does not support cross-LLM transfer.
- The MLP adapters in the **LLaVA** series, while simple and effective, lack interpretability.
- The "vocabulary embeddings as basis vectors" paradigm proposed in this paper may inspire further research on cross-modal alignment grounded in the intrinsic representations of LLMs.

## Rating

⭐⭐⭐⭐ — Mechanistic analysis is insightful, the method design is elegant, and the cross-LLM transfer offers high practical value; however, some performance degradation persists on certain benchmarks after transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching](aligning_information_capacity_between_vision_and_language_via_dense_to_sparse_feature_distillation.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](../../ACL2026/information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[NeurIPS 2025\] The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](../../NeurIPS2025/information_retrieval/the_narrow_gate_localized_imagetext_communication_in_native.md)
- [\[ICCV 2025\] ViLU: Learning Vision-Language Uncertainties for Failure Prediction](vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [\[ICCV 2025\] aligning information capacity between vision and language via dense-to-sparse fe](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)

</div>

<!-- RELATED:END -->
