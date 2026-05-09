---
title: >-
  [Paper Note] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference
description: >-
  [ACL 2026][Model Compression][KV cache compression] This paper proposes ASL (Adaptive Selection Layer), which monitors the variance of token attention score rankings to adaptively determine the layer at which KV cache pruning is performed. ASL significantly outperforms fixed-layer selection methods on difficult tasks while remaining training-free.
tags:
  - ACL 2026
  - Model Compression
  - KV cache compression
  - adaptive layer selection
  - attention pruning
  - long-context inference
  - training-free method
date: 2026-05-08
content_hash: 7623dba9396b779d
---

# Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference

**Conference**: ACL 2026
**arXiv**: [2601.07667](https://arxiv.org/abs/2601.07667)
**Code**: [GitHub](https://github.com/TANIGUCHIREI/ASL)
**Area**: Model Compression / KV Cache Optimization
**Keywords**: KV cache compression, adaptive layer selection, attention pruning, long-context inference, training-free method

## TL;DR

This paper proposes ASL (Adaptive Selection Layer), which monitors the variance of token attention score rankings to adaptively determine the layer at which KV cache pruning is performed. ASL significantly outperforms fixed-layer selection methods on difficult tasks while remaining training-free.

## Background & Motivation

**Background**: The KV cache is the primary memory bottleneck in LLM inference. Layer-wise token pruning—selecting an important subset of tokens at a designated layer and discarding the rest—is a mainstream compression approach.

**Limitations of Prior Work**: Existing layer-wise pruning methods (e.g., FastKV, GemFilter) rely on predefined fixed selection layers. This design performs adequately on simple tasks (e.g., QA) but degrades severely on difficult tasks (e.g., KV retrieval), where high semantic similarity between queries and contexts prevents early layers from distinguishing relevant tokens.

**Key Challenge**: Fixed selection layers face a fundamental trade-off—early selection saves computation but sacrifices accuracy, while late selection preserves accuracy but reduces memory savings. The optimal selection layer varies substantially across tasks.

**Goal**: Design an adaptive method that automatically determines the optimal token selection layer based on task difficulty.

**Key Insight**: Attention score rankings converge to a stable subset at different rates across tasks—simple tasks stabilize at intermediate layers, while difficult tasks require deeper layers before stabilization.

**Core Idea**: Monitor the variance of token rankings as an indicator of "attention focus." Token selection is triggered when the variance falls below a specified threshold.

## Method

### Overall Architecture

ASL operates during the prefilling stage: starting from layer $L_{min}$, it computes the ranking variance of pooled attention scores over every $L_{obs}$ consecutive layers. When the relative variance drops below a user-specified threshold, one-shot token selection is performed at that layer. ASL can subsequently be combined with methods such as SnapKV to further optimize the decoding stage.

### Key Designs

1. **Rank-Variance-Based Adaptive Selection**

    - **Function**: Automatically determines the optimal layer for token pruning based on task difficulty.
    - **Mechanism**: Computes pooled attention scores $PA = \text{pool}(\text{softmax}(\frac{\mathbf{q}_w \mathbf{k}_c + \mathbf{m}_w}{\sqrt{d}}))$, then calculates the variance of token rankings over $L_{obs}$ consecutive layers. Low variance indicates that attention has stably focused on a fixed subset of tokens.
    - **Design Motivation**: Rank variance is more robust than raw attention scores—it is agnostic to specific score magnitudes and focuses solely on whether the set of attended tokens has stabilized.

2. **Threshold-Controlled Adaptive Trade-off**

    - **Function**: Allows users to control the accuracy–efficiency trade-off via a single parameter.
    - **Mechanism**: The user specifies a threshold $\theta$; selection is triggered once variance falls below $\theta$. A higher $\theta$ leads to earlier selection (faster but potentially less accurate), while a lower $\theta$ leads to later selection (more accurate but slower).
    - **Design Motivation**: Different application scenarios have varying requirements for accuracy and speed; a single tunable parameter is more practical than manually adjusting the selection layer.

3. **Seamless Integration with Existing Methods**

    - **Function**: Jointly optimizes the full inference pipeline in combination with methods such as SnapKV.
    - **Mechanism**: ASL optimizes the prefilling stage (determining the selection layer), while SnapKV optimizes the decoding stage (compressing the KV cache prior to the selection layer). ASL can also be combined with GemFilter using a two-pass strategy.
    - **Design Motivation**: ASL is an orthogonal improvement that can directly replace the fixed-layer selection component in existing methods.

### Loss & Training

ASL requires no training whatsoever and operates entirely at inference time. Two hyperparameters, $L_{min}$ and $L_{obs}$, control the starting monitoring layer and the observation window size, respectively.

## Key Experimental Results

### Main Results

| Method | KV Retrieval (Hard) | QA (Easy) | NIAH | Memory Usage |
|--------|---------------------|-----------|------|--------------|
| FastKV (fixed layer) | Severe degradation | Strong | Moderate | Low |
| GemFilter (fixed layer) | Degradation | Strong | Moderate | Low |
| ASL (adaptive) | Significant improvement | Maintained | Improved | Comparable |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Threshold sensitivity | Smooth transition | Different thresholds yield a continuous accuracy–speed trade-off |
| Cross-task adaptability | InfiniteBench, 10 tasks | Different tasks automatically select layers of different depths |
| 256K context | Effective | Also applicable to long-context scenarios |

### Key Findings

- For simple tasks (QA), attention stabilizes at intermediate layers (~layer 15); for difficult tasks (KV retrieval), stabilization requires deeper layers (~layer 25 or beyond).
- ASL substantially outperforms fixed-layer methods on difficult tasks while maintaining comparable performance on simple tasks.
- Relative variance serves as an effective "task difficulty probe"—enabling adaptive behavior without requiring prior knowledge of the task type.

## Highlights & Insights

- Reframes the question of "when to select" from a hyperparameter-tuning problem to an automatic detection problem, significantly improving practical usability.
- Observation-driven design—grounded in the cross-layer evolution of attention patterns, with a clear and coherent logical chain.
- Entirely training-free, ready to use out of the box, and orthogonally composable with existing methods.

## Limitations & Future Work

- Currently validated only on Llama 3.1 8B; evaluation on a broader range of model architectures is needed.
- Monitoring rank variance incurs a small computational overhead, which may require optimization in extreme low-latency settings.
- The optimal threshold value still requires user selection based on the target scenario.
- Future work may explore a progressive variant that performs token pruning gradually across multiple adaptively selected layers.

## Related Work & Insights

- **vs. FastKV/GemFilter**: ASL replaces fixed layer selection with adaptive selection, fundamentally addressing the issue of task sensitivity.
- **vs. PyramidKV/DynamicKV**: These methods adaptively allocate budgets but do not adaptively select layers; the two approaches are complementary.
- **vs. SnapKV**: ASL optimizes layer selection during prefilling, while SnapKV optimizes token retention during decoding; the two can be used in combination.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of using rank variance as a task difficulty probe is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks and context lengths.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from observation → motivation → method → validation is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ Directly applicable to optimizing LLM long-context inference.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[CVPR 2026\] FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](../../CVPR2026/model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)
- [\[NeurIPS 2025\] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](../../NeurIPS2025/model_compression/dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)

<!-- RELATED:END -->
