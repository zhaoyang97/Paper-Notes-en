---
title: >-
  [Paper Note] Wanda++: Pruning Large Language Models via Regional Gradients
description: >-
  [ACL 2025][Model Compression] Wanda++ is proposed: a lightweight LLM pruning framework based on decoder block-level regional gradients. It improves the pruning criterion with Regional Gradient Score (RGS) and minimizes the output discrepancies between dense and sparse blocks via Regional Optimization (RO). Under 2:4 sparsity, it reduces WikiText perplexity by up to 32% compared to Wanda, while pruning a 7B model within 10 minutes on a single H100 GPU.
tags:
  - "ACL 2025"
  - "Model Compression"
date: 2026-05-08
content_hash: 15f6c27d9ebd5805
---

# Wanda++: Pruning Large Language Models via Regional Gradients

**Conference**: ACL 2025  
**arXiv**: [2503.04992](https://arxiv.org/abs/2503.04992)  
**Code**: None  
**Area**: Model Compression  

## TL;DR

Wanda++ is proposed: a lightweight LLM pruning framework based on decoder block-level regional gradients. It improves the pruning criterion with Regional Gradient Score (RGS) and minimizes the output discrepancies between dense and sparse blocks via Regional Optimization (RO). Under 2:4 sparsity, it reduces WikiText perplexity by up to 32% compared to Wanda, while pruning a 7B model within 10 minutes on a single H100 GPU.

## Background & Motivation

1. **LLM Inference Bottleneck**: Large language models have huge parameters (e.g., LLaMA-2 70B requires 140GB of VRAM), leading to high inference latency and an urgent need for model compression.
2. **High Accuracy Loss of Existing Pruning Methods**: Although post-training pruning methods (SparseGPT, Wanda) are efficient, they suffer from severe performance degradation under 2:4 semi-structured sparsity, lagging far behind quantization methods (such as AWQ which achieves nearly lossless 4x compression).
3. **Gradient Information is Valuable but Expensive to Obtain**: GBLM and Pruner-Zero have demonstrated that gradient information can significantly improve pruning results. However, they rely on full-model backpropagation, which demands unacceptable GPU time and memory (GBLM takes 5800 seconds to prune a 7B model, compared to 55 seconds for Wanda).
4. **Flaws in the Linear Assumption of Layer-by-Layer Pruning**: Wanda evaluates weight importance independently at each layer, ignoring the cumulative error propagation effects across layers.
5. **Core Problem**: Can gradient information be utilized effectively while remaining lightweight?

## Method

### Overall Architecture

Wanda++ performs two-stage processing sequentially for each decoder block: **Regional Gradient Score (RGS) Pruning** + **Regional Optimization (RO) Weight Restoration**, iteratively for K rounds.

### 1. Regional Gradient Score (RGS)

Core Idea: Replace full-model gradients with decoder block-level "regional gradients" to substantially reduce computational overhead.

**Regional Loss Definition** (without labels):
$$\mathcal{L}_{RGS}^l(\mathbf{X}_n^l) = \|f^l(\mathbf{X}_n^l)\|_2$$

i.e., the L2 norm of the output of the $l$-th decoder block. A single backpropagation pass on this loss yields the gradients of all weights within that block.

**RGS Pruning Criterion**:
$$S_{ij} = (\alpha \cdot G_{ij} + \|\mathbf{X}_j\|_2) \cdot |W_{ij}|$$

- $G_{ij}$: Regional gradient magnitude (RMS over N samples)
- $\|\mathbf{X}_j\|_2$: Original input activation norm from Wanda
- $\alpha = 100$: Scaling factor balancing gradient and activation
- The regional gradient is computed only once per block, and is fused with the layer-wise updated activation norm, balancing efficiency and accuracy.

### 2. Regional Optimization (RO)

After each round of RGS pruning, weights within the block are fine-tuned to repair the errors introduced by pruning:

$$\mathcal{L}_{ro}^{l,k}(\hat{\mathbf{X}}_m^l) = (f^l(\hat{\mathbf{X}}_m^l) - \hat{f}_k^l(\hat{\mathbf{X}}_m^l))^2$$

- MSE loss between dense block output and pruned block output
- 32 samples are randomly selected from 128 calibration samples for RO
- Uses RMSprop optimizer, learning rate of 3e-7
- Iterated for K=5 rounds per block (RGS pruning → RO optimization → re-pruning to restore sparsity)

### 3. Algorithm Pipeline

The L decoder blocks are processed sequentially:
1. Compute regional gradient G (single backpropagation)
2. K iterations of: RGS pruning → RO weight update
3. Final RGS pruning to restore sparsity constraints
4. Update the input hidden states of the next block

## Key Experimental Results

### Table 1: WikiText Perplexity (↓ lower is better)

| Method | LLaMA-1 7B (2:4) | LLaMA-1 13B (2:4) | OpenLLaMA 3B (2:4) | LLaMA-3.1 8B (2:4) |
|:---|:---:|:---:|:---:|:---:|
| Dense Baseline | 5.68 | 5.09 | 7.27 | 6.39 |
| Wanda | 11.53 | 9.58 | 28.04 | 24.83 |
| GBLM | 11.33 | 9.16 | 24.75 | 24.34 |
| SparseGPT | 11.00 | 9.11 | 15.91 | - |
| **Wanda++** | **9.43 (-19%)** | **7.75 (-20%)** | **19.03 (-32%)** | **18.32 (-26%)** |

The improvement is most significant on small models (3B/7B); 2:4 sparsity benefits more than unstructured and 4:8 sparsity.

### Table 2: Zero-Shot Downstream Task Accuracy (LLaMA-1 7B, 2:4 Sparsity)

| Method | MRPC | HellaSwag | ARC-e | RTE | MMLU |
|:---|:---:|:---:|:---:|:---:|:---:|
| Dense | 69.12 | 56.96 | 75.29 | 66.43 | 35.10 |
| Wanda | 46.81 | 41.66 | 59.34 | 49.82 | 25.85 |
| **Wanda++** | **68.38 (+46%)** | **45.31 (+8%)** | **63.72 (+7%)** | **62.09 (+24%)** | **27.52 (+6%)** |

Performance on MRPC and RTE tasks is almost restored to the dense baseline level.

### Pruning Efficiency (LLaMA-1 7B)

| Method | Time | VRAM |
|:---|:---:|:---:|
| GBLM | 5801s | 26 GB |
| SparseGPT | 322s | 23 GB |
| Wanda | 55s | 22 GB |
| **Wanda++** | **290s** | **25 GB** |

VRAM usage is comparable to Wanda, while the execution time is only 1/20 of GBLM.

## Highlights & Insights

- **Ingenious Concept of Regional Gradients**: Avoids full-model backpropagation (BP) through block-level BP, reducing gradient computation complexity from O(full model) to O(single block), with VRAM footprint independent of the total number of layers.
- **Complementary Two-Stage Design (RGS + RO)**: RGS improves pruning decisions while RO repairs pruning errors. The combined effect exceeds either component applied individually.
- **Orthogonality to LoRA Fine-Tuning**: Running LoRA fine-tuning after Wanda++ pruning yields further improvements, showing that the two techniques are complementary.
- **Scalability to Large Models**: Theoretical analysis suggests that single-block optimization on a 530B model requires only ~40GB VRAM, making it feasible on a single GPU.

## Limitations & Future Work

- **Notable Accuracy Loss in 2:4 Sparsity**: Even with Wanda++, the perplexity of LLaMA-1 7B under 2:4 sparsity increases from 5.68 to 9.43, which is far from the "near-lossless" level achieved by quantization.
- **Sensitivity to Calibration Data Volume**: The effect of RO is unstable when using small calibration sets (<64 samples).
- **Increased Time due to RO Iterations**: Takes 290s compared to Wanda's 55s (approx. 5x slower), although still 20x faster than GBLM.
- **Evaluated Only on the LLaMA Family**: Generality has not been validated on other architectures such as Qwen or Mistral.

## Related Work & Insights

| Dimension | Wanda++ | Wanda | GBLM | SparseGPT |
|:---|:---|:---|:---|:---|
| Gradient Information | Regional gradients (block-level) | None | Full-model gradients | Second-order Hessian approximation |
| Weight Update | RO (block-level MSE) | None | None | Column-wise OBS update |
| 7B Pruning Time | ~5 min | ~1 min | ~97 min | ~5 min |
| VRAM Requirement | Comparable to Wanda | Lowest | Full model loading | Medium |
| 2:4 Perplexity Improvement | Best (-19 to -32%) | Baseline | Minor | Moderate |

## Rating

- ⭐⭐⭐⭐ Novelty: Replacing full-model gradients with regional gradients is simple yet effective, and block-level MSE optimization reduces error propagation.
- ⭐⭐⭐⭐ Practicality: 10 mins for 7B on a single GPU, orthogonal to LoRA, scalable to ultra-large models, and engineering-friendly deployment.
- ⭐⭐⭐⭐ Experimental Thoroughness: Comprehensive coverage of 4 model families, 3 sparsity patterns, perplexity, downstream tasks, efficiency, latency, and ablation studies.
- ⭐⭐⭐ Writing Quality: The method description is clear, but some experimental figures and tables are not intuitive, and the performance drop in certain zero-shot tasks is not fully explained.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ACL 2025\] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](language_specific_features.md)

</div>

<!-- RELATED:END -->
