---
title: >-
  [Paper Note] Token CropR: Faster ViTs for Quite a Few Tasks
description: >-
  [CVPR 2025][Segmentation][ViT acceleration] Proposes Token CropR (Cropr), a cross-attention-based ViT token pruning method that learns to select tokens based on task relevance end-to-end via auxiliary prediction heads. During inference, these auxiliary heads are discarded to achieve a throughput close to that of a random pruner, achieving $1.5\text{-}4\times$ speedups with minimal performance loss across classification, semantic segmentation, object detection…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "ViT acceleration"
  - "token pruning"
  - "cross-attention"
  - "multi-task generalizability"
  - "semantic segmentation"
date: 2026-05-08
content_hash: eace8316cb0dd0e2
---

# Token CropR: Faster ViTs for Quite a Few Tasks

**Conference**: CVPR 2025  
**arXiv**: [2412.00965](https://arxiv.org/abs/2412.00965)  
**Code**: [GitHub](https://github.com/benbergner/cropr)  
**Area**: Segmentation/Efficient Vision  
**Keywords**: ViT acceleration, token pruning, cross-attention, multi-task generalizability, semantic segmentation

## TL;DR

Proposes Token CropR (Cropr), a cross-attention-based ViT token pruning method that learns to select tokens based on task relevance end-to-end via auxiliary prediction heads. During inference, these auxiliary heads are discarded to achieve a throughput close to that of a random pruner, achieving $1.5\text{-}4\times$ speedups with minimal performance loss across classification, semantic segmentation, object detection, and instance segmentation.

## Background & Motivation

- **ViT Efficiency Pain Points**: The $O(n^2)$ complexity of self-attention makes sequence length a major bottleneck, which is further exacerbated by larger model sizes, higher resolutions, and finer patch sizes.
- **Limitations of Prior Token Pruning**: Attention-score-based heuristics do not explicitly model task-specific token importances; attribution-based methods require full forward passes, inducing excessive computation overhead; and nearly all existing approaches are restricted strictly to image classification.
- **Conflict in Dense Prediction**: Semantic segmentation requires pixel-level predictions, which fundamentally conflicts with the concept of token pruning — recovering information of pruned tokens remains a core challenge.
- **Indirect Loss of Inference Efficiency**: Parameterized pruning modules introduce extra layers and auxiliary training losses that distort backbone representations and increase inference overhead.
- **Goal**: Fast (minimal overhead during inference), preserving high performance, and applicable to various vision tasks.

## Method

### Overall Architecture

The Cropr module is integrated between internal ViT layers. Each module consists of: scorer (cross-attention that computes scores via learnable query × token keys) $\rightarrow$ selector (Top-K selection of keep tokens) $\rightarrow$ aggregator (reusing the attention matrix for weighted averaging) $\rightarrow$ auxiliary task head (providing training signals). During inference, the aggregator and auxiliary heads are discarded, and multiple queries are aggregated into a single vector to enable $O(M)$ scoring. Dense tasks leverage Last Layer Fusion (LLF) to recover pruned tokens.

### Key Designs

**Design One: Cross-Attention Scoring + Auxiliary Head Training**
- **Function**: End-to-end learning to score tokens based on task relevance.
- **Mechanism**: Cross-attention is performed between learnable queries $\mathbf{Q} \in \mathbb{R}^{N \times D}$ and token keys as $\mathbf{A} = \mathbf{Q} \times \mathbf{K}(\mathbf{X})^\top$. Token scores $\mathbf{a}$ are obtained by summing over the query dimension. The aggregator reuses $\mathbf{A}$ to calculate weighted averages for the auxiliary task head to generate intermediate predictions, which backpropagates signals to train the scorer. Crucially, a stop-gradient is applied to the input of the scorer to isolate gradients between translation/auxiliary heads and the backbone.
- **Design Motivation**: Top-down task-specific signals (from auxiliary heads) reflect token task relevance better than bottom-up heuristics (like attention scores); stop-gradients prevent auxiliary head gradients from interfering with the main backbone representation learning.

**Design Two: Inference-Time Query Aggregation Optimization**
- **Function**: Reduces the inference overhead from $O(N \times M)$ to $O(M)$.
- **Mechanism**: After discarding the aggregator and auxiliary heads at inference time, only the token scores are required. Exploiting the distributive property: $\mathbf{a} = \sum_{n=1}^N \mathbf{Q}_n \mathbf{K}^\top = (\sum_{n=1}^N \mathbf{Q}_n) \mathbf{K}^\top = \bar{\mathbf{q}} \mathbf{K}^\top$, the aggregated query $\bar{\mathbf{q}}$ is pre-computed, requiring only a single vector-matrix multiplication and a Top-K per Cropr module during inference.
- **Design Motivation**: For semantic segmentation, $N = h \times w$ can be extremely large, making the inference overhead without aggregation close to that of full attention. Changing the order of operations reduces the complexity to a constant factor.

**Design Three: Last Layer Fusion (LLF) — Token Recovery for Dense Tasks**
- **Function**: Recovers pruned tokens in the final ViT block.
- **Mechanism**: Pruned tokens from all previous Cropr modules are gathered and re-inserted by spatial positions right after the second-to-last ViT block, then processed alongside the retained tokens in the final ViT block. This allows pruned tokens to attend to deep-level retained tokens to retrieve contextual representations without adding extra parameters.
- **Design Motivation**: Simply discarding tokens is unacceptable for dense prediction tasks; directly concatenating pruned tokens in the final layer so that they can "see" deep features is simpler and more effective than alternative techniques like token summarization or featuring projections.

### Loss & Training

Standard main task loss + task-specific auxiliary losses for each Cropr module (softmax cross-entropy for classification, per-patch cross-entropy with downsampled labels for segmentation, and multi-label binary cross-entropy for detection). The auxiliary losses do not affect backbone gradients due to the stop-gradient mechanism.

## Key Experimental Results

### ImageNet-1k Classification (EVA-02 backbone)

| Method | Top-1 Acc | Speedup |
|------|----------|-------|
| No pruning | 85.8% | 1.0× |
| Random pruning + LLF | 83.8% | ~2× |
| **Cropr** | **89.7%** (ViT-L) | **2.1×** |

### ADE20k Semantic Segmentation

| Method | mIoU | Speedup |
|------|------|-------|
| No pruning | baseline | 1.0× |
| **Cropr** | **-0.1 mIoU** | **2.0×** |

### COCO Detection & Instance Segmentation

| Method | AP_box | Speedup |
|------|--------|-------|
| Liu et al. | ~34% speedup | small model |
| **Cropr** | **63.0** | **1.9×** |

### Key Findings

1. Achieving a $2\times$ speedup on ADE20k semantic segmentation with only a loss of $0.1$ mIoU (median over 5 seeds) is virtually a "free lunch."
2. The throughput of Cropr during inference is close to that of a random pruner, as auxiliary heads are fully omitted.
3. Cropr achieves more pronounced speedups under larger models (ViT-L) and higher input resolutions (512+).
4. LLF introduces no additional parameters but yields more effective results than alternatives like token summarization and feature projection.

## Highlights & Insights

- **Decoupled Training & Inference Design**: The auxiliary heads learn the scoring during training and are completely removed during inference, elegantly resolving the trade-off between "learning and inference overhead".
- **Mathematical Trick for Query Aggregation**: Leveraging the distributive property of linear algebra reduces the scoring complexity to $O(M)$, which is a simple yet crucial engineering optimization.
- **Multi-Task Universality**: The first token pruning method to achieve general applicability and effectiveness across all four major tasks (classification, segmentation, detection, and instance segmentation).
- The LLF mechanism is exceptionally clean — pruned tokens simply bypass intermediate layers and merge directly into the final layer, introducing zero extra parameters.

## Limitations & Future Work

- Top-K selection represents hard pruning and does not support image-adaptive pruning rates (which maintains batch processing efficiency but sacrifices flexibility for easy/hard inputs).
- The auxiliary head design still requires task-specific customization (classification vs segmentations vs detection), adding engineering overhead for adaptation to new tasks.
- The performance recovery of LLF under extreme pruning rates is limited, as the final layer cannot fully compensate for information loss from bypassing multiple skipped intermediate layers.
- Evaluation has not been conducted on video Transformers or NLP Transformers, leaving cross-domain generalizability unverified.

## Related Work & Insights

- The paradigm of Cropr's auxiliary-head training with inference-time removal can be generalized to other scenarios requiring learning-based token selection.
- The core concept behind LLF — "bypassing intermediate layers and fusion at the final layer" — is applicable to any other token pruning method.

## Rating

⭐⭐⭐⭐ — Highly clean and elegant design with minimal inference overhead. As the first truly multi-task general token pruning method, its $2\times$ speedup on ADE20k with only a $0.1$ mIoU drop is extremely convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[ECCV 2024\] DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs](../../ECCV2024/segmentation/densenets_reloaded_paradigm_shift_beyond_resnets_and_vits.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](../../NeurIPS2025/segmentation/mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)

</div>

<!-- RELATED:END -->
