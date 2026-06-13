---
title: >-
  [Paper Note] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference
description: >-
  [ACL 2026][Model Compression][KV Cache Compression] ASL (Adaptive Selection Layer) is proposed to adaptively determine the layer position for KV cache pruning by monitoring the ranking variance of token attention scores.…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "KV Cache Compression"
  - "Adaptive Layer Selection"
  - "Attention Pruning"
  - "Long-Context Inference"
  - "Training-Free Method"
date: 2026-05-08
content_hash: 36633ab102d864cb
---

# Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.07667](https://arxiv.org/abs/2601.07667)  
**Code**: [GitHub](https://github.com/TANIGUCHIREI/ASL)  
**Area**: Model Compression / KV Cache Optimization  
**Keywords**: KV Cache Compression, Adaptive Layer Selection, Attention Pruning, Long-Context Inference, Training-Free Method

## TL;DR

ASL (Adaptive Selection Layer) is proposed to adaptively determine the layer position for KV cache pruning by monitoring the ranking variance of token attention scores. It significantly outperforms fixed-layer selection methods on difficult tasks while maintaining a training-free nature.

## Background & Motivation

**Background**: KV cache is the primary memory bottleneck in LLM inference. Layer-wise token pruning (selecting a subset of important tokens at a specific layer and pruning others) is a mainstream compression scheme.

**Limitations of Prior Work**: Existing layer-wise pruning methods (e.g., FastKV, GemFilter) use predefined fixed selection layers. This design is effective for simple tasks (e.g., QA) but degrades severely on difficult tasks (e.g., KV retrieval). The cause is the high semantic similarity between the query and context in difficult tasks, making it hard for early layers to distinguish relevant tokens.

**Key Challenge**: Fixed selection layers face a fundamental trade-off—early selection saves computation but loses accuracy, while late selection maintains accuracy but reduces memory savings. The optimal selection layer varies significantly across different tasks.

**Goal**: Design an adaptive method to automatically determine the optimal token selection layer based on task difficulty.

**Key Insight**: It is observed that the speed at which attention score rankings converge to a stable subset varies across tasks—simple tasks stabilize at middle layers, while difficult tasks require deeper layers for stability.

**Core Idea**: Monitor the ranking variance of tokens as an indicator of "attention focus." Token selection is triggered when the variance drops below a certain threshold.

## Method

### Overall Architecture

ASL operates during the prefilling stage: starting from layer $L_{min}$, it calculates the ranking variance of pooled attention scores over every $L_{obs}$ consecutive layers. When the relative variance is lower than a user-specified threshold, one-shot token selection is executed at that layer. This can be jointly optimized with methods like SnapKV for the decoding stage.

### Key Designs

1. **Adaptive Selection Based on Rank Variance**:

    - **Function**: Automatically determines the optimal layer for token pruning based on task difficulty.
    - **Mechanism**: Calculates pooled attention scores $PA = \text{pool}(\text{softmax}(\frac{\mathbf{q}_w \mathbf{k}_c + \mathbf{m}_w}{\sqrt{d}}))$ and computes the ranking variance of tokens across $L_{obs}$ consecutive layers. Low variance indicates that attention has stably focused on a fixed subset.
    - **Design Motivation**: Rank variance is more robust than raw attention scores—it focuses on whether "which tokens are being attended to" is stable, rather than the specific score values.

2. **Threshold-Controlled Adaptive Trade-off**:

    - **Function**: Allows users to control the accuracy-efficiency balance via a single parameter.
    - **Mechanism**: The user specifies a threshold $\theta$; selection is triggered once variance falls below $\theta$. A higher $\theta$ leads to earlier selection (faster but potentially less accurate), while a lower $\theta$ leads to later selection (more accurate but slower).
    - **Design Motivation**: Different applications have varying requirements for accuracy and speed; a single parameter control is more practical than manual layer adjustment.

3. **Seamless Integration with Existing Methods**:

    - **Function**: Jointly optimizes the entire inference pipeline with methods like SnapKV.
    - **Mechanism**: ASL optimizes the prefilling stage (determining the selection layer), while SnapKV optimizes the decoding stage (compressing the KV cache before the selection layer). It can also be combined with GemFilter using a two-pass strategy.
    - **Design Motivation**: ASL is an orthogonal improvement that can directly replace the fixed-layer selection components in existing methods.

### Loss & Training

ASL is entirely training-free and only runs during inference. Two hyperparameters $L_{min}$ and $L_{obs}$ control the starting monitor layer and the observation window size, respectively.

## Key Experimental Results

### Main Results

| Method | KV Retrieval (Hard) | QA (Simple) | NIAH | Memory Usage |
|------|------------|---------|------|--------|
| FastKV (Fixed Layer) | Severe Degradation | Strong | Medium | Low |
| GemFilter (Fixed Layer) | Degradation | Strong | Medium | Low |
| ASL (Adaptive) | Significant Gain | Maintained | Improved | Comparable |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| Threshold Sensitivity | Smooth Transition | Different thresholds produce a continuous accuracy-speed trade-off |
| Cross-task Adaptability | InfiniteBench 10 tasks | Different tasks automatically select layers of varying depths |
| 256K Context | Effective | Works equally well in long-context scenarios |

### Key Findings
- Attention stabilizes at middle layers (~layer 15) for simple tasks (QA) but requires deeper layers (~layer 25+) for difficult tasks (KV retrieval).
- ASL significantly outperforms fixed-layer methods on difficult tasks while maintaining comparable performance on simple tasks.
- Relative variance serves as an effective "task difficulty probe"—enabling adaptation without prior knowledge of the task type.

## Highlights & Insights
- Transforms the "when to select" problem from hyperparameter tuning to automatic detection, significantly enhancing practicality.
- Observation-driven design—derived from the cross-layer evolution patterns of attention masks, with a clear logical chain.
- Completely training-free and plug-and-play, while being orthogonal and combinable with existing methods.

## Limitations & Future Work
- Currently validated only on Llama 3.1 8B; needs testing on more model architectures.
- Monitoring rank variance incurs some computational overhead (albeit small), which may require optimization in extreme low-latency scenarios.
- The optimal threshold value still requires user selection based on the specific scenario.
- Future work could explore a progressive version—gradual pruning across multiple adaptively selected layers.

## Related Work & Insights
- **vs FastKV/GemFilter**: Replaces fixed layers with adaptive selection to fundamentally solve the task-sensitivity issue.
- **vs PyramidKV/DynamicKV**: These methods adaptively allocate budgets but do not adaptively select layers; they are complementary to ASL.
- **vs SnapKV**: ASL optimizes layer selection in the prefilling stage, while SnapKV optimizes token retention in the decoding stage; they can be used in combination.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using rank variance as a task difficulty probe is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks and context lengths.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic chain from observation $\rightarrow$ motivation $\rightarrow$ method $\rightarrow$ validation is very clear.
- Value: ⭐⭐⭐⭐ Provides direct practical value for the optimization of LLM long-context inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ACL 2026\] LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference](leap_layer-wise_exit-aware_pretraining_for_efficient_transformer_inference.md)
- [\[ACL 2026\] A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification](a_bertology_view_of_llm_orchestrations_token-_and_layer-selective_probes_for_eff.md)
- [\[ICML 2026\] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](../../ICML2026/model_compression/respinquant_efficient_layer-wise_llm_quantization_via_subspace_residual_rotation.md)
- [\[CVPR 2026\] FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](../../CVPR2026/model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)

</div>

<!-- RELATED:END -->
