---
title: >-
  [Paper Note] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models
description: >-
  [NeurIPS 2025][Model Compression][spatio-temporal foundation models] This paper proposes FactoST-v2, a factorized spatio-temporal foundation model framework that decouples universal temporal pre-training from domain-spec…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "spatio-temporal foundation models"
  - "time series forecasting"
  - "factorization paradigm"
  - "spatio-temporal adaptation"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: fb7ba368ef87c9b5
---

# Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2601.12083](https://arxiv.org/abs/2601.12083)  
**Code**: [GitHub](https://github.com/CityMind-Lab/FactoST)  
**Area**: Model Compression
**Keywords**: spatio-temporal foundation models, time series forecasting, factorization paradigm, spatio-temporal adaptation, zero-shot generalization

## TL;DR

This paper proposes FactoST-v2, a factorized spatio-temporal foundation model framework that decouples universal temporal pre-training from domain-specific spatial adaptation, achieving cross-domain zero-shot/few-shot/full-shot spatio-temporal forecasting with linear complexity.

## Background & Motivation

**Background**: Spatio-temporal (ST) foundation models aim to learn universal representations from multi-domain data to enable generalizable cross-dataset forecasting. Existing approaches such as UniST and OpenCity adopt a joint spatio-temporal pre-training paradigm.

**Limitations of Prior Work**: Joint pre-training faces the challenge of "pattern mismatch" — temporal dynamics exhibit cross-domain invariance (periodicity, trends, etc.), whereas spatial correlations are highly topology-dependent (road networks vs. power grids). Forcing joint modeling incurs quadratic complexity $\mathcal{O}(N^2T^2)$ and negative transfer.

**Key Challenge**: Temporal and spatial patterns are fundamentally different in nature — temporal patterns are domain-invariant while spatial patterns are domain-specific — yet existing methods compel both to be learned jointly.

**Goal**: To design an ST foundation model framework that achieves both generalizability and efficiency.

**Key Insight**: The authors propose the "pattern factorization hypothesis" — effective spatio-temporal generalization requires decoupling domain-invariant temporal dynamics from domain-specific spatial context.

**Core Idea**: Pre-train a universal temporal backbone first, then inject spatial awareness via lightweight adapters, enabling cross-domain transfer with linear complexity.

## Method

### Overall Architecture

FactoST-v2 consists of two stages: **Stage 1 (UTP)** performs universal temporal pre-training on large-scale multi-source time series to learn domain-invariant temporal representations; **Stage 2 (STA)** injects spatio-temporal awareness into the pre-trained backbone via lightweight adapters for downstream adaptation.

### Key Designs

1. **Encoder-Only Backbone + Random Sequence Masking**: The encoder-decoder architecture from v1 is replaced by a minimalist encoder-only design. A learnable [REG] token separates the context and prediction intervals, and random sequence masking $l_{mask} \sim \mathcal{U}(0, L_{max} - L_{min})$ enables the model to learn arbitrary-length mappings. This eliminates fixed input/output length constraints and enables 100% parameter transfer.

2. **Semantics-Aware Positional Encoding (p-RoPE) + Gated Attention**: The embedding space is decomposed into high-frequency and low-frequency subspaces; the rotation matrix is applied only to the high-frequency component to preserve order awareness, while the low-frequency component remains unchanged to retain semantic magnitude: $\text{p-RoPE}(\mathbf{x}, m) = [\mathbf{x}_{high} \otimes \mathbf{R}_{\Theta,m} \| \mathbf{x}_{low}]$. Gated attention filters noise via $\mathbf{O} = \text{Attention}(\mathbf{Q}', \mathbf{K}', \mathbf{V}) \odot \sigma(\mathbf{G})$.

3. **Probabilistic Quantile Prediction**: Upgrading from the deterministic point estimation in v1 to multi-quantile prediction, optimized with Pinball Loss: $\mathcal{L}_{UTP} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}}\max((q-1)(\mathbf{y}-\hat{\mathbf{y}}_q), q(\mathbf{y}-\hat{\mathbf{y}}_q))$. This supports uncertainty quantification.

4. **Spatio-Temporal Metadata Fusion (STMF)**: Spatial node embeddings $\mathbf{E}_n \in \mathbb{R}^{N \times d}$ and multi-scale calendar temporal embeddings are injected and linearly projected to generate a spatio-temporal context $\mathbf{I}_{st}$.

5. **Spatio-Temporal Filtering (STF)**: Three scalar affinities — spatial affinity $\mathbf{S}_s$, temporal affinity $\mathbf{S}_t$, and temporal lag affinity $\mathbf{S}_d$ — are computed, fused via Softmax, and gated by Sigmoid to dynamically reweight the spatio-temporal identifiers.

6. **Domain-Specific Prompt Alignment (DSPA)**: Low-rank learnable prompt tokens $\mathbf{P} = \mathbf{U}\mathbf{V}^\top$ ($r \ll d$) are introduced and prepended to the fused features as prefix tokens, aligning the pre-training and downstream distributions.

### Loss & Training

- **UTP Stage**: Pinball Loss (quantile loss)
- **STA Stage**: L1 Loss (median quantile taken for comparison against deterministic baselines)
- **Continual Memory Replay (CMR)**: Datasets are partitioned into a memory buffer and a current stream; each mini-batch mixes samples from both to mitigate catastrophic forgetting

## Key Experimental Results

### Main Results

**Few-Shot Forecasting (Short-Term 12→12) MAE/RMSE:**

| Model | PEMS-03 | PEMS-04 | PEMS-07 | PEMS-08 |
|-------|---------|---------|---------|---------|
| UniST (STFM) | 40.39/53.44 | 42.76/59.07 | 40.77/54.86 | 35.70/46.74 |
| OpenCity (STFM) | 17.90/28.80 | 24.78/40.41 | 44.43/65.47 | 32.16/48.47 |
| TimesFM (TSFM) | 21.99/35.31 | 27.84/43.15 | 32.61/50.20 | 22.06/33.87 |
| GWNet (STEM) | 17.25/27.79 | 23.27/35.62 | 26.51/41.08 | 18.47/29.04 |
| FactoST (v1) | 17.54/28.10 | 23.93/37.44 | 26.48/41.92 | 18.94/29.59 |
| **FactoST-v2** | **16.75/27.20** | **22.61/35.95** | **24.70/40.79** | **17.65/28.44** |

**Full-Shot Forecasting (Short-Term 12→12) MAE/RMSE:**

| Model | PEMS-03 | PEMS-04 | PEMS-07 | PEMS-08 |
|-------|---------|---------|---------|---------|
| D2STGNN (STEM) | **14.91/25.82** | **18.75/30.12** | **20.19/33.25** | **14.63/23.73** |
| GWNet (STEM) | 15.93/28.11 | 20.93/32.96 | 23.86/37.83 | 16.48/26.19 |
| **FactoST-v2** | 15.65/24.90 | 20.61/32.81 | 21.95/35.39 | 15.80/25.46 |

### Ablation Study

Key upgrade contributions from v1 to v2 (as described in the paper):

| Upgrade | Effect |
|---------|--------|
| Encoder-Only (replacing Enc-Dec) | 100% parameter transfer, arbitrary-length support |
| Random sequence masking | Flexible length generalization |
| Probabilistic quantile prediction | Uncertainty quantification |
| DSPA Prompt (replacing HDA) | Cleaner adaptation, pure forecasting objective |

### Key Findings

- FactoST-v2 substantially outperforms jointly pre-trained STFMs (UniST, OpenCity) in few-shot and zero-shot settings
- Performance is competitive with or superior to domain-expert models (D2STGNN, GWNet) in full-shot settings
- Joint ST models encounter OOM on long-horizon forecasting (GWNet, D2STGNN), while FactoST-v2 maintains linear complexity
- Clear scaling laws are observed across model sizes from Minuscule to Base

## Highlights & Insights

- **The core insight of the factorization paradigm is exceptionally clear**: temporal patterns are cross-domain invariant while spatial patterns are domain-specific; decoupling the two constitutes a well-motivated inductive bias
- A complete theoretical-to-practical justification is provided, including complexity analysis (linear vs. quadratic) and generalization bound analysis
- The encoder-only design with random masking is elegantly simple, achieving arbitrary-length generalization and 100% parameter transfer
- The pre-training corpus covers 11B time points across 8 diverse domains, establishing a large-scale temporal foundation model

## Limitations & Future Work

- Spatial adaptation still relies on lightweight modules, limiting modeling capacity for complex topological structures such as dynamic graphs
- The pre-training stage entirely ignores spatial information, potentially discarding valuable spatio-temporal coupling patterns
- As a journal extension of a conference paper, the work is highly comprehensive but also considerably dense
- The remaining gap with expert models under full-data conditions indicates that the factorization paradigm still has room for improvement

## Related Work & Insights

- FactoST-v2 is distinguished from purely temporal foundation models such as TimesFM and Chronos by the incorporation of spatial adaptation
- It differs from joint ST foundation models such as UniST and OpenCity by decoupling temporal and spatial learning
- The factorization paradigm generalizes naturally to other multi-modal and multi-domain pre-training scenarios

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The factorization paradigm represents a significant conceptual advance in the ST foundation model field
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 datasets, 16 baselines, and five major research questions
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear logic, though the content is overly dense as a journal extension
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and scalable practical path for ST foundation models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](../../ICCV2025/model_compression/b_vllm_a_vision_large_language_model_with_balanced_spatio_temporal_tokens.md)
- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [\[NeurIPS 2025\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](specialization_after_generalization_towards_understanding_test-time_training_in_.md)
- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)

</div>

<!-- RELATED:END -->
