---
title: >-
  [Paper Note] BlockPruner: Fine-grained Pruning for Large Language Models
description: >-
  [ACL 2025][Model Compression][Structured Pruning] Proposes BlockPruner, which decomposes Transformer layers into two minimal residual blocks (MHA and MLP), evaluates block importance based on perplexity, and performs fine-grained pruning through iterative search, achieving superior compression performance compared to layer-level pruning.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Structured Pruning"
  - "LLM Compression"
  - "Block-level Redundancy"
  - "Perplexity"
  - "Iterative Search"
date: 2026-05-08
content_hash: 1acdac42973a831e
---

# BlockPruner: Fine-grained Pruning for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2406.10594](https://arxiv.org/abs/2406.10594)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Structured Pruning, LLM Compression, Block-level Redundancy, Perplexity, Iterative Search  

## TL;DR

Proposes BlockPruner, which decomposes Transformer layers into two minimal residual blocks (MHA and MLP), evaluates block importance based on perplexity, and performs fine-grained pruning through iterative search, achieving superior compression performance compared to layer-level pruning.

## Background & Motivation

### 1. Background

The size of Large Language Models (LLMs) continues to grow, leading to high deployment costs. Model compression techniques (knowledge distillation, quantization, pruning) have become essential for practical deployment. Recent studies reveal substantial layer redundancy in LLMs, where removing these layers has a limited impact on overall performance.

### 2. Limitations of Prior Work

- Current layer-level pruning methods (e.g., ShortGPT, LaCo) use the entire Transformer layer as the minimum pruning unit, which is too coarse-grained.
- Although unstructured pruning can maintain performance at high compression ratios, it requires specialized hardware support, making real-world speedup difficult.
- Structured pruning methods (e.g., LLM-Pruner, SliceGPT) typically require model retraining after pruning.

### 3. Key Challenge

Layer-level pruning overlooks finer-grained redundancy inside the layers. MHA and MLP blocks exhibit different levels of redundancy; removing an entire layer may discard critical sub-modules simultaneously.

### 4. Goal

How to perform structural pruning of LLMs at a granularity finer than the layer level without requiring retraining?

### 5. Key Insight

Leveraging the residual connection structure of Transformer layers, each layer is decomposed into two minimal residual blocks (MHA and MLP) to independently evaluate and remove redundant blocks.

### 6. Core Idea

Decompose Transformer layers into MHA/MLP blocks, measure block importance using perplexity, and progressively remove the least important blocks via iterative greedy search.

## Method

### Overall Architecture

BlockPruner comprises three core steps:

1. **Minimal Residual Block Decomposition**: Decomposing each Transformer layer into two independent residual blocks, MHA and MLP.
2. **Block Importance Evaluation**: Measuring the importance of each block through perplexity.
3. **Iterative Search Pruning**: Progressively removing the least important blocks.

### Key Designs

#### Minimal Residual Block

The computation of each Transformer layer can be decomposed into two residual connections:

$$X_i' = \text{MHA}(\text{LN}(X_{i-1})) + X_{i-1}$$

$$X_i = \text{MLP}(\text{LN}(X_i')) + X_i'$$

Both sub-modules follow the residual form $f(x) + x$, allowing them to be independently "masked" without disrupting the information flow. For a model with $L$ layers, there are $2L$ prunable blocks in total.

#### Block Importance Metric — Perplexity

Unlike **local** metrics such as Block Influence (BI) used in ShortGPT, BlockPruner adopts a **global** metric — perplexity:

$$\text{PPL} = \exp\left(-\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(w_i|w_{<i})\right)$$

Specifically, each block $B_i$ is masked one by one, and the perplexity $P_i$ of the masked model is computed on a calibration dataset. A lower perplexity indicates that the block is more redundant.

#### Iterative Search Algorithm

Instead of removing all low-importance blocks at once, BlockPruner employs an **iterative greedy search**:

1. Mask and calculate the perplexity for each of the $2L - j + 1$ remaining blocks individually.
2. Remove the block that yields the lowest perplexity.
3. Repeat the process until the target number of pruned blocks $K$ is reached.

This iterative approach accounts for block interactions. Ablation studies demonstrate that removing the iterative search significantly degrades performance (e.g., a 30.80% drop on Baichuan2-7B).

### Loss & Training

BlockPruner is a **training-free** method that requires no fine-tuning or retraining. It only utilizes 256 samples from the Alpaca dataset as the calibration set.

## Key Experimental Results

### Main Results

| Model | Method | Pruning Ratio | PPL↓ | Avg Score |
|------|------|--------|------|-----------|
| Llama2-7B | Dense | 0% | 5.47 | 68.96 |
| Llama2-7B | ShortGPT | 21.02% | 18.45 | 58.18 |
| Llama2-7B | SliceGPT | 21.45% | 30.74 | 57.83 |
| Llama2-7B | **BlockPruner** | 21.99% | **11.51** | **60.17** |
| Llama2-13B | Dense | 0% | 4.89 | 71.72 |
| Llama2-13B | ShortGPT | 24.37% | 20.06 | 62.60 |
| Llama2-13B | **BlockPruner** | 25.12% | **8.16** | **64.53** |
| Qwen1.5-14B | Dense | 0% | 7.44 | 69.07 |
| Qwen1.5-14B | ShortGPT | 22.25% | 1237.21 | 44.72 |
| Qwen1.5-14B | **BlockPruner** | 23.72% | **15.67** | **60.45** |

### Ablation Study

| Model | Method | Pruning Ratio | Avg Score |
|------|------|--------|-----------|
| Llama2-7B | BlockPruner | 21.99% | 60.17 |
| Llama2-7B | - Iterative Search | 20.95% | 55.89 (-7.11%) |
| Llama2-7B | - Block-to-Layer | 21.02% | 58.63 (-2.56%) |
| Baichuan2-7B | BlockPruner | 22.45% | 56.08 |
| Baichuan2-7B | - Iterative Search | 22.39% | 38.81 (-30.80%) |
| Qwen1.5-14B | BlockPruner | 23.72% | 60.45 |
| Qwen1.5-14B | - Iterative Search | 22.98% | 40.80 (-32.51%) |

### Key Findings

1. **MHA is more redundant than MLP**: When the pruning ratio is <17%, pruning only the MHA blocks results in less performance loss; beyond this point, MHA performance drops sharply.
2. **Larger models exhibit higher redundancy**: Llama2-13B maintains better performance than Llama2-7B at the same pruning ratio.
3. **Alpaca is more suitable as a calibration set than Wikitext2**: Instruction-following data aligns more closely with the downstream task distribution.
4. **256 samples are sufficient**: Increasing the number of samples does not significantly improve pruning performance.
5. The perplexity after BlockPruner pruning correlates positively with downstream task performance.

## Highlights & Insights

- **Simple and Effective Design**: Decomposes layers naturally into two independently removable blocks using the residual structure of Transformers, without introducing extra architectural complexity.
- **Combination of Global Metric and Iterative Search**: Although perplexity is not a perfect local metric, combining it with iterative search captures interactions between blocks, heavily outperforming one-shot pruning.
- **Asymmetric Redundancy in MHA/MLP**: Reveals that the MHA module possesses more redundancy, offering insights for both model design and compression strategies.
- **Training-Free**: Compared to methods like LLM-Pruner that require fine-tuning, BlockPruner is completely training-free, offering greater practicality.

## Limitations & Future Work

1. The iterative search for perplexity calculation incurs high computational overhead—each round requires iterating through all remaining blocks, resulting in a time complexity of $O(K \cdot 2L)$.
2. The combination of this method with other compression techniques, such as quantization, has not been explored.
3. Experiments only cover 7B-14B models; the performance on larger-scale models (70B+) remains unknown.
4. The block importance metric still has room for optimization—perplexity is a global indicator, and more precise block-level metrics might exist.

## Related Work & Insights

- **ShortGPT / LaCo**: Layer-level pruning methods; BlockPruner refines the granularity on top of these.
- **SliceGPT**: Prunes columns/rows of the weight matrix by computing invariance, offering a different but complementary approach.
- **FINERCUT** (concurrent work): A similar block-level pruning method, but uses logits similarity rather than perplexity as the importance metric.
- **Insight**: Residual connections are key structural characteristics enabling fine-grained pruning. Future work can explore even finer granularity (e.g., attention head level) pruning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of block-level pruning is intuitive and clear. While not the first of its kind, the combination of perplexity and iterative search yields an effective solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 6 models, 5 benchmarks, with comprehensive ablation studies and analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly logical writing, naturally deriving the method from preliminary empirical observations.
- **Value**: ⭐⭐⭐⭐ — Highly practical with a straightforward approach, providing significant practical reference value to the LLM compression community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](../../ICML2025/model_compression/slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](../../ICML2025/model_compression/instruction-following_pruning_for_large_language_models.md)
- [\[ACL 2025\] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](drpruning_robust_pruning.md)

</div>

<!-- RELATED:END -->
