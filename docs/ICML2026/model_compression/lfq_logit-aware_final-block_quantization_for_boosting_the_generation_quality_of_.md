---
title: >-
  [Paper Note] LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs
description: >-
  [ICML 2026][Model Compression][Low-bit quantization] Addressing the quality degradation of block-wise PTQ in generation tasks, LFQ replaces the quantization objective of the final Transformer block from MSE with logit-le…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Low-bit quantization"
  - "Post-training quantization"
  - "Cross-entropy alignment"
  - "Generation quality"
  - "block-wise PTQ"
date: 2026-05-08
content_hash: 3e918a7e2b61b877
---

# LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.29756](https://arxiv.org/abs/2605.29756)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Low-bit quantization, Post-training quantization, Cross-entropy alignment, Generation quality, block-wise PTQ  

## TL;DR
Addressing the quality degradation of block-wise PTQ in generation tasks, LFQ replaces the quantization objective of the final Transformer block from MSE with logit-level cross-entropy loss. This aligns the token distribution of the quantized model with the full-precision model, consistently improving accuracy across generation benchmarks such as IFEval, GSM8K, MATH500, and AIME.

## Background & Motivation

**Background**: Low-bit weight-only PTQ (e.g., GPTQ, FlexRound, OmniQuant, Block-AP) is the mainstream approach for LLM memory compression. Block-wise PTQ minimizes the MSE between the quantized block and the full-precision block outputs iteratively, approaching full-precision baselines in language modeling (WikiText2) and understanding (MMLU) tasks.

**Limitations of Prior Work**: When evaluated on long-text generation (IFEval) and complex reasoning (MATH500, AIME), the accuracy of block-wise PTQ drops significantly. Large reasoning models (DeepSeek-R1, L1-Max) particularly rely on long chains of thought to improve task accuracy, and quantized generation trajectories tend to deviate from the correct path.

**Key Challenge**: There are two root causes: (1) block-wise PTQ completely ignores the unembedding layer (LM head), while the final block output must pass through the LM head to generate token distributions; (2) even when considering the LM head, MSE minimization does not guarantee consistent token ranking. The authors provide a 2-token example clearly demonstrating that a quantization scheme with smaller MSE can predict the wrong top-1 token, while one with larger MSE preserves the correct prediction.

**Goal**: To bring the generation quality of block-wise PTQ close to the full-precision baseline without changing the quantization scheme or inference kernels.

**Key Insight**: Minimizing cross-entropy is equivalent to minimizing KL divergence, which is zero if and only if two distributions are identical. Therefore, cross-entropy aligns token probability distributions more directly than MSE.

**Core Idea**: Modify only the quantization loss of the final Transformer block—replacing MSE with logit-level cross-entropy—to align the next-token distribution of the quantized model with the full-precision model.

## Method

### Overall Architecture
LFQ is a plug-and-play enhancement module applicable to any block-wise PTQ method. The overall pipeline remains unchanged: MSE is minimized block-by-block from the 1st to the penultimate Transformer block. The only change occurs in the final block, where the LM head is included in the forward path and cross-entropy loss replaces MSE to optimize quantization parameters. The quantized model uses the exact same packing/unpacking kernels as the original method, requiring no additional engineering adaptation.

### Key Designs

1. **Logit-level Cross-Entropy Objective**:

    - **Function**: Transitions the quantization optimization objective of the final Transformer block from activation-space MSE to logit-space cross-entropy.
    - **Mechanism**: For the final block, the cross-entropy $\mathcal{L}_{\text{CE}} = -\frac{1}{L}\sum_{i,j} \sigma(X W_{\text{FP}} W_{\text{Head}})_{i,j} \log \sigma(X W_q W_{\text{Head}})_{i,j}$ between full-precision logits $\sigma(X W_{\text{FP}} W_{\text{Head}})$ and quantized logits $\sigma(X W_q W_{\text{Head}})$ is calculated to directly optimize the consistency of token probability distributions.
    - **Design Motivation**: Small MSE errors in activation space can flip top-1 token rankings after LM head projection; cross-entropy is sensitive to probability ranking and naturally protects top-k order.

2. **LM Head Inclusion in Final Block Forward Path**:

    - **Function**: Allows the quantization optimizer to "see" the impact of the unembedding layer on the output distribution.
    - **Mechanism**: Standard block-wise PTQ only minimizes the MSE of block output activations, completely ignoring subsequent LM head projections. LFQ fixes the LM head weights $W_{\text{Head}}$ during forward propagation (without quantizing them), making gradients aware of changes in the final token distribution.
    - **Design Motivation**: The authors prove via counterexamples that even if the MSE of block outputs is minimized, completely opposite token predictions can occur after LM head projection.

3. **Method-Agnostic Plug-and-Play Design**:

    - **Function**: Compatible with any block-wise PTQ methods such as FlexRound, OmniQuant, and Block-AP.
    - **Mechanism**: LFQ only replaces the loss function of the final block without changing the quantization parameterization. Whether optimizing $(s_1, S_2, s_3)$ for FlexRound, $(\gamma, \beta)$ for OmniQuant, or $(s, W_{\text{FP}})$ for Block-AP, one only needs to replace MSE with $\mathcal{L}_{\text{CE}}$.
    - **Design Motivation**: Maintains full compatibility with the existing quantization ecosystem—same memory overhead, same inference kernels, and single-GPU execution capability.

## Key Experimental Results

### Main Results (Qwen2.5-7B-Instruct)

| Method | Bits | WikiText2↓ | MMLU↑ | IFEval↑ | MATH500↑ |
|------|------|-----------|-------|---------|----------|
| BF16 Baseline | 16 | 6.85 | 73.49 | 70.79 | 74.2 |
| FlexRound | W4 | 7.23 | 72.50 | 69.50 | 72.6 |
| FlexRound+LFQ | W4 | **7.21** | 72.48 | **71.35** | **73.4** |
| FlexRound | W3g128 | 7.63 | 70.13 | 66.54 | 65.6 |
| FlexRound+LFQ | W3g128 | **7.58** | **70.26** | **67.84** | **68.0** |
| OmniQuant | W4 | 7.73 | **71.00** | 68.21 | 69.8 |
| OmniQuant+LFQ | W4 | **7.53** | 70.99 | **69.50** | **71.6** |

### Ablation Study (Llama 3.1 8B Instruct, W4)

| Method | LM Head | Cross-Entropy | WikiText2↓ | IFEval↑ | GSM8K↑ |
|------|---------|--------|-----------|---------|--------|
| FlexRound | ✗ | ✗ | 7.06 | 70.24 | 81.35 |
| +LM Head | ✓ | ✗ | 7.08 | 71.53 | 81.58 |
| +LM Head+CE (LFQ) | ✓ | ✓ | **7.06** | **72.09** | **81.80** |
| OmniQuant | ✗ | ✗ | 7.49 | 70.61 | 78.17 |
| +LM Head | ✓ | ✗ | 7.48 | 71.35 | 78.32 |
| +LM Head+CE (LFQ) | ✓ | ✓ | **7.47** | **71.35** | **79.76** |

### Key Findings
- LFQ consistently improves performance on generation tasks (IFEval, MATH500, AIME) without degrading performance on understanding tasks (WikiText2, MMLU).
- Ablations show that both components (LM Head inclusion and cross-entropy loss) contribute incremental gains, with the combination providing the best effect.
- Under W3g128 settings for the reasoning model L1-Qwen-7B-Max, LFQ recovers greedy accuracy on AIME'24 from 23.33% to 30.00% (BF16 is 46.67%), recovering nearly half of the quantization degradation.

## Highlights & Insights
- Using a 2-token constructive counterexample intuitively proves the disconnection between MSE and token prediction consistency. This analytical approach is simple yet powerful and can be generalized to other compression scenarios requiring discrete decision consistency.
- Changing the loss function of only the final block comprehensively improves generation quality. This minimal yet effective modification embodies the "applying the right constraints in the right place" design philosophy.
- Analysis of the probability distribution for "aha moment" tokens (e.g., "Wait"/"But") in reasoning models reveals the overconfidence issue of quantized models at critical thinking junctures, providing a new perspective on how quantization affects chain-of-thought reasoning.
- Validation on the MoE model Qwen3-30B-A3B shows that LFQ is equally effective for sparsely activated architectures, with AIME'25 greedy accuracy improving from 53.33% to 60.00%.

## Limitations & Future Work
- Cross-entropy calculation requires softmax over the entire vocabulary, which may increase optimization overhead when the vocabulary is very large (e.g., 150K+); the paper does not discuss this overhead.
- Focuses only on weight-only quantization and does not explore logit alignment in activation quantization scenarios.
- Future work could consider extending the cross-entropy objective to multiple trailing blocks (rather than just the last one) or combining it with temperature scaling to further control distribution alignment precision.

## Related Work & Insights
- **vs FlexRound/OmniQuant/Block-AP**: These methods are sufficient using MSE for intermediate blocks; LFQ proves that simply switching the loss function in the last block can compensate for generation quality shortfalls.
- **vs Knowledge Distillation QAT**: QAT performs end-to-end KL alignment across the entire model but requires massive computation. LFQ achieves similar probability alignment effects in the final block at extremely low cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Precise observation, simple but effective method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 model families, 3 PTQ methods, two bit-widths, and multiple generation benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Constructive counterexamples and reasoning trajectory visualizations are highly persuasive.
- Value: ⭐⭐⭐⭐ Plug-and-play, zero additional inference cost, direct practical value for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](../../AAAI2026/model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)
- [\[ICLR 2026\] Boosting Entropy with Bell Box Quantization](../../ICLR2026/model_compression/boosting_entropy_with_bell_box_quantization.md)

</div>

<!-- RELATED:END -->
