---
title: >-
  [Paper Note] Less is More: Local Intrinsic Dimensions of Contextual Language Models
description: >-
  [NeurIPS 2025][Video Understanding][intrinsic dimension] This paper proposes using the Local Intrinsic Dimension (LID) of contextual token embeddings as an unsupervised signal for monitoring LLM training dynamics — a decrease in LID indicates improved generalization, while an increase signals overfitting. The utility of this geometric signal is validated on tasks including dialogue state tracking, grokking, and sentiment recognition.
tags:
  - NeurIPS 2025
  - Video Understanding
  - intrinsic dimension
  - LLM
  - fine-tuning
  - grokking
  - overfitting detection
  - embedding geometry
date: 2026-05-08
content_hash: 02062b30b434fedd
---

# Less is More: Local Intrinsic Dimensions of Contextual Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.01034](https://arxiv.org/abs/2506.01034)
**Code**: [GitHub](https://github.com/aidos-lab/Topo_LLM_public)
**Area**: Video Understanding
**Keywords**: intrinsic dimension, LLM, fine-tuning, grokking, overfitting detection, embedding geometry

## TL;DR

This paper proposes using the Local Intrinsic Dimension (LID) of contextual token embeddings as an unsupervised signal for monitoring LLM training dynamics — a decrease in LID indicates improved generalization, while an increase signals overfitting. The utility of this geometric signal is validated on tasks including dialogue state tracking, grokking, and sentiment recognition.

## Background & Motivation

- **Understanding LLM internals remains difficult**: Even fundamental questions such as how fine-tuning affects model behavior typically require extensive empirical evaluation.
- **Lack of unsupervised training diagnostic tools**: Most performance diagnostics rely on labeled validation sets or task-specific probes, which are unavailable in low-resource settings.
- **Limitations of existing dimensionality studies**: Tulchinskii et al. found that AI-generated text exhibits lower global intrinsic dimension, but their analysis operates on individual text segments; Aghajanyan et al. defined intrinsic dimension in parameter space rather than embedding space; Valeriani et al. studied how global dimensionality changes as data passes through an LLM, but without localized analysis.
- **Global dimensionality lacks granularity**: The embedding space is not a single manifold of uniform dimensionality, but rather a union of manifolds with varying local dimensions, necessitating local estimation.

## Method

### Latent Space Modeling

Given a text corpus $\mathcal{D} = (s_0, \ldots, s_D)$ and a model $\mathcal{M}$ of depth $l$, each sequence $s_m$ is tokenized by $\mathcal{T}$ and produces contextual embeddings at layer $i$:

$$\mathcal{M}_i(s_m) = (\mathcal{M}_i(t_0^m), \ldots, \mathcal{M}_i(t_{n_m}^m))$$

All token embeddings form a point cloud $\mathbb{T}_i = \{\mathcal{M}_i(t_j^m)\}_{m, j}$, with distances measured in Euclidean space.

### Two-Stage Sampling Strategy

In practice, $\mathbb{T}_i$ can contain millions of vectors, making direct neighborhood computation infeasible. The procedure is:
1. Sample $M$ sequences from $\mathcal{D}$
2. After deduplication, subsample $N$ token vectors
3. Compute the $L$-nearest neighborhood $\mathcal{N}_L(t_j; \mathbb{T})$ for each token

### Local TwoNN Dimension Estimation

The TwoNN estimator is applied, leveraging the ratio $r_2/r_1$ of distances to the nearest and second-nearest neighbors (which follows a Pareto distribution under mild assumptions) to estimate local dimensionality:

$$\text{LID}(v) = \text{TwoNN}(\mathcal{N}_L(v; \mathbb{T}))$$

This yields a dimensionality vector $\in \mathbb{R}_{\geq 0}^N$ over all sampled tokens, aggregated into a mean LID as the overall geometric signature.

### Cross-Model Comparison

A base model $\mathcal{M}$ and its fine-tuned counterpart $\mathcal{M}^\Delta$ share the same architecture and tokenizer, establishing a natural point-wise correspondence between their embedding spaces, enabling direct comparison of dimensionality changes.

## Experiments

### Experiment 1: Fine-Tuning Induces Dataset-Specific Dimensional Shifts

**Setup**: RoBERTa-base is fine-tuned via MLM on MultiWOZ dialogue data for 5 epochs; LID is measured on MultiWOZ, Wikipedia, and Reddit.

**Results**:
- MultiWOZ (fine-tuning data): significant LID decrease (standardized mean difference 1.19)
- Wikipedia/Reddit (out-of-distribution data): LID nearly unchanged (standardized mean difference 0.08/0.10)

**Key Finding**: LID reduction is **dataset-specific** — it occurs only within the fine-tuning data distribution and does not affect unrelated data regions.

### Experiment 2: LID Detects Grokking

**Setup**: A 2-layer decoder-only Transformer is trained on modular addition mod $p=197$, with training data ratios ranging from 10% to 50%.

| Training Data Ratio | Grokking? | LID Trend on Training Set |
|:---:|:---:|:---|
| 10% | No | Rises then plateaus |
| 15% | No | Rises then plateaus |
| ≥20% | Yes | Rises then **significantly decreases** |

**Key Finding**: A pronounced LID decrease on the training set coincides with the onset of rising validation accuracy — grokking can be predicted from training data alone, without requiring validation labels.

### Experiment 3: LID Detects Training Capacity Exhaustion

**Setup**: TripPy-R dialogue state tracking model (RoBERTa encoder) is trained on MultiWOZ for 20 epochs.

**Results**:
- Spearman correlation between mean LID on training set and Joint Goal Accuracy (JGA): **−0.982**
- Validation loss is minimized by step 7,500, yet JGA continues to improve and LID continues to decrease — indicating that validation loss gives a misleading "convergence" signal
- LID stabilizes after approximately 25,000 steps, in synchrony with JGA convergence

**Key Finding**: LID is a more reliable convergence indicator than validation loss.

### Experiment 4: LID Detects Overfitting

**Setup**: BERT-base with a linear classifier is trained on EmoWOZ emotion classification for 8 epochs.

**Results**:
- After epoch 1, LID drops sharply from ~9.94 to ~7.25 (model finds an efficient representation)
- LID subsequently rises gradually to ~8 (dimensional increase suggests memorization)
- Validation loss **rises continuously** after epoch 1 — a clear overfitting signal
- Spearman correlation of LID with training loss: −0.952; with validation loss: +0.952

**Key Finding**: The pattern of LID decreasing then increasing corresponds to the transition from "finding an efficient representation" to "overfitting," and can serve as an unsupervised early-stopping signal.

## Highlights & Insights

- **A unified framework covering diverse training dynamics**: The same LID metric detects fine-tuning effects, grokking, training convergence, and overfitting across four distinct phenomena.
- **Label-free diagnostic signal**: Entirely grounded in the embedding geometry of training data, requiring no labeled validation set.
- **Simple and effective heuristic**: LID decreasing → improved generalization; LID increasing → memorization/overfitting — an intuitive and actionable rule.
- **Carefully designed experiments**: Cover encoders (RoBERTa/BERT) and decoders (GPT-2/tiny Transformer), as well as sequence labeling and classification settings.

## Limitations & Future Work

- **High computational cost**: Requires extensive forward passes to build embeddings plus $O(dN^2)$ nearest-neighbor search, limiting real-time monitoring.
- **Strong TwoNN assumptions**: The estimator requires approximately constant local density and a Poisson process assumption; its applicability to Transformer embeddings is validated only empirically.
- **Absolute values are not cross-architecture comparable**: LID absolute values depend on hyperparameters ($M$, $N$, $L$); only relative changes are meaningful for comparison.
- **Causality not established**: The relationship between LID decrease and generalization improvement is correlational rather than causal; theoretical explanation remains lacking.
- **Validated only on smaller models**: Experiments focus on RoBERTa-base and GPT-2-medium; scalability to models with 7B+ parameters is unknown.

## Related Work & Insights

- **LLM intrinsic dimensionality** (Aghajanyan et al., 2021): Studies dimensionality in parameter space rather than embedding space, finding that larger models have lower parameter-space dimensionality.
- **Global embedding dimensionality** (Valeriani et al., 2023; Tulchinskii et al., 2023): Analyzes global dimensionality differences between AI-generated and human-written text, without localization.
- **Token-level dimensionality** (Viswanathan et al., 2025): Analyzes token dimensionality within individual prompts; this paper subsamples from an entire dataset.
- **Topological deep learning** (Papamarkou et al., 2024): Geometric and topological methods for observational analysis of ML models; this paper extends the direction to training dynamics diagnosis.
- **LoRA rank adaptation** (Ed-dib et al., 2024): Adjusts LoRA rank based on the rank of hidden-state information matrices; the LID proposed here is complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ — Local intrinsic dimension as an unsupervised diagnostic signal for training dynamics is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four independent experiments covering fine-tuning, grokking, convergence, and overfitting across diverse models and tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear correspondence between theory and experiments; each experiment is framed around a well-defined research question.
- Value: ⭐⭐⭐⭐ — Provides a valuable geometric tool for LLM training monitoring, particularly meaningful in low-resource scenarios.

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Insights & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](../../CVPR2026/video_understanding/token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

</div>

<!-- RELATED:END -->
