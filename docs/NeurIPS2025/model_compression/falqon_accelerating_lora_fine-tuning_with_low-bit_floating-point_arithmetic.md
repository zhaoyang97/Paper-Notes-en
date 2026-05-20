---
title: >-
  [Paper Note] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic
description: >-
  [NeurIPS 2025][Model Compression][LoRA acceleration] FALQON eliminates the small-matrix quantization overhead introduced by standalone LoRA paths by directly melding LoRA adapters into FP8-quantized backbone weights. Com…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "LoRA acceleration"
  - "FP8 quantization"
  - "low-precision training"
  - "LLM fine-tuning"
  - "quantization overhead"
date: 2026-05-08
content_hash: e8129f896b65e314
---

# FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic

**Conference**: NeurIPS 2025
**arXiv**: [2510.24061](https://arxiv.org/abs/2510.24061)  
**Code**: [https://github.com/iamkanghyunchoi/falqon](https://github.com/iamkanghyunchoi/falqon)  
**Area**: Model Compression
**Keywords**: LoRA acceleration, FP8 quantization, low-precision training, LLM fine-tuning, quantization overhead

## TL;DR
FALQON eliminates the small-matrix quantization overhead introduced by standalone LoRA paths by directly melding LoRA adapters into FP8-quantized backbone weights. Combined with efficient gradient computation and a row-wise proxy update mechanism, it achieves approximately 3× training speedup over existing quantized LoRA methods.

## Background & Motivation

**Background**: The FP8 low-precision format is natively supported on modern GPUs (NVIDIA Hopper/Blackwell), theoretically doubling matrix multiplication throughput. LoRA is the dominant PEFT method for LLM fine-tuning, reducing trainable parameters via low-rank decomposition.

**Limitations of Prior Work**: FP8 quantization is effective for large matrices, but LoRA introduces small-dimensional matrices (rank typically 16–128). Applying FP8 quantization to these small matrices incurs quantization overhead (max reduction, scaling) that far outweighs the computational gain, making FP8 LoRA slower than FP16 LoRA. Empirically, FP8 LoRA achieves only about half the throughput of FP16 LoRA.

**Key Challenge**: The standalone LoRA forward/backward path requires separate quantization of three small tensors — $\tilde{A}$, $\tilde{B}$, and $O_A$ — adding 3 extra quantization operations per iteration. When matrix dimensions fall below 4K, the $O(n^2)$ quantization overhead overwhelms the $O(n^3)$ gains from faster matrix multiplication.

**Goal**: How to genuinely exploit FP8 hardware to accelerate LoRA fine-tuning, rather than merely applying weight-only quantization for memory savings?

**Key Insight**: Since the standalone LoRA path is the source of overhead, eliminating it entirely — by directly melding LoRA into the quantized backbone — is the natural solution. The quantization error itself can be interpreted as an implicit low-rank adapter.

**Core Idea**: Meld LoRA into the FP8 backbone to eliminate small-matrix quantization overhead; use concatenation to enable single-pass forward computation; maintain the fused weights via a row-wise proxy update mechanism.

## Method

### Overall Architecture
FALQON eliminates the standalone LoRA path. At initialization, SVD of the quantization error serves as an implicit LoRA initialization. Matrix $A$ is concatenated into the backbone to enable single-pass forward computation; only the gradient of $B$ is computed; a proxy buffer selectively writes large updates back to the backbone.

### Key Designs

1. **Melded LoRA: Fusing LoRA into the Quantized Backbone** (Section 5.1):

    - **Function**: Utilizes FP8 quantization error as an implicit LoRA initialization.
    - **Mechanism**: $DQ_{fp8}(\tilde{W}) = W - \Delta_Q W \approx W + \hat{B}\hat{A}$, where $\hat{A}, \hat{B}$ are obtained from a rank-$r$ SVD of $-\Delta_Q W$. The quantized backbone $\tilde{W}$ thus implicitly encodes LoRA, and a single $DQ_{fp8}(\tilde{W})x$ simultaneously produces both the backbone output and the LoRA adaptation output.
    - **Design Motivation**: Eliminates all small-matrix quantization operations from the standalone LoRA path. Unlike methods such as IR-QLoRA that also exploit quantization error, FALQON genuinely removes computational overhead rather than merely using it for parameter initialization.

2. **Efficient Gradient Computation** (Section 5.2):

    - **Function**: Re-derives the gradient for melded LoRA to further reduce quantization operations.
    - **Mechanism**: Rewrites $\partial\mathcal{L}/\partial B = (\partial\mathcal{L}/\partial O) \cdot x^\top A^\top$ as $(\partial\mathcal{L}/\partial O) \cdot (Ax)^\top$. Since $Ax$ is already computed during the forward pass via the concatenated weight $\tilde{W}' = [\tilde{W}; \tilde{A}]$ (i.e., $O_{merged} = \tilde{W}' \tilde{x} / (s_W s_x) = [O; O_{\hat{A}}]$), no additional computation is required.
    - **Design Motivation**: Only $B$ is trained ($A$ is frozen and embedded in the backbone), following the spirit of LoRA-FA.

3. **Row-wise Proxy Update Mechanism** (Section 5.3):

    - **Function**: Efficiently propagates gradient updates of $B$ back to the fused backbone weights.
    - **Mechanism**: A proxy buffer $\Delta B$uffer accumulates updates to $B$. Since the low-precision backbone $\tilde{W}$ may not reflect small updates after requantization, only the top-$k$ rows with the largest accumulated magnitudes are selectively written back: $\mathbf{k} = \text{topk}(\sum_j |\Delta B_{i,j}|; k)$, $\tilde{W}[\mathbf{k}] = \tilde{W}[\mathbf{k}] + \Delta\text{Buffer}[\mathbf{k}] A$.
    - **Design Motivation**: Since $k \ll m$ (number of output channels), this avoids ineffective full-weight updates while preserving LoRA's memory advantage — only $\Delta B$uffer needs to be stored rather than a full-precision copy of $W$.

### Loss & Training
- Paged AdamW optimizer; batch size 16; learning rate $2 \times 10^{-5}$; 1875 training steps.
- Matrix $A$ shares the backbone scaling factor $s_W$ during FP8 quantization, avoiding additional quantization operations.
- All experiments conducted on a single RTX 4090 24 GB GPU.

## Key Experimental Results

### Main Results
LLaMA-7B + Alpaca dataset, evaluated on MMLU:

| Method | Time/Step (s) | Speedup | MMLU Avg |
|--------|--------------|---------|----------|
| QLoRA | 5.45 | 1.0× | 0.3272 |
| QA-LoRA | 9.44 | — | 0.3548 |
| IR-QLoRA | 8.27 | — | 0.3388 |
| FALQON | **1.80** | **3.02×** | 0.3491 |

LLaMA-13B + Alpaca dataset:

| Method | Time/Step (s) | Speedup | MMLU Avg |
|--------|--------------|---------|----------|
| QLoRA | 9.37 | 1.0× | 0.4443 |
| QA-LoRA | 18.02 | — | 0.4729 |
| IR-QLoRA | 14.46 | — | 0.4349 |
| FALQON | **3.26** | **2.87×** | 0.4644 |

### Ablation Study
FP8 LoRA overhead analysis (LLaMA-7B, rank=64):

| Component | FP16 Latency | FP8 Latency | Note |
|-----------|-------------|-------------|------|
| Computation | High | Low (speedup) | FP8 MatMul is faster |
| Quantization overhead | 0 | Very high (~4×) | Quantizing A/B/OA is the bottleneck |
| Total | Faster | Slower | Applying FP8 directly to LoRA is counterproductive |

### Key Findings
- Applying FP8 directly to LoRA yields only ~50% the throughput of FP16, with no improvement even at rank=512.
- FALQON's speedup primarily stems from eliminating 3 small-matrix quantization operations, which account for the majority of total runtime.
- The concatenation in melded LoRA adds only $r$ rows, introducing negligible forward-pass overhead.
- The top-$k$ row selection strategy in the proxy update effectively avoids ineffective updates under low precision.
- At the end of FALQON training, weights are already in FP8 format, eliminating the need for post-training quantization.

## Highlights & Insights
- The **in-depth analysis of FP8 + LoRA** is particularly valuable: it clearly identifies that the bottleneck lies in quantization overhead rather than numerical precision — an insight with broad implications for the low-precision training community.
- **Melded LoRA initialization** cleverly repurposes quantization error as an implicit adapter, simultaneously compensating for quantization loss without introducing additional parameters.
- **Gradient reuse via $A$ concatenation** is a highly engineering-friendly trick: a single forward pass yields both the output and the intermediate results needed for gradient computation.
- The **end-to-end FP8 workflow** eliminates the extra quantization step at inference deployment, providing direct practical value.

## Limitations & Future Work
- Freezing $A$ and updating only $B$ may underperform full $A$+$B$ updates on certain tasks.
- The top-$k$ row selection strategy is relatively simple and may miss important but spatially diffuse updates.
- Validation is currently limited to LLaMA-7B/13B; effectiveness on larger models (70B+) remains to be confirmed.
- Requires FP8 hardware support (Hopper architecture or newer); incompatible with older GPUs.
- The melded LoRA initialization may be less numerically stable than Kaiming initialization in theory.

## Related Work & Insights
- **vs. QLoRA**: QLoRA uses NF4 weight-only quantization for memory savings without acceleration; FALQON uses FP8 weight+activation quantization for both memory savings and speedup.
- **vs. IR-QLoRA**: IR-QLoRA similarly uses SVD of quantization error for LoRA initialization, but retains the standalone path and does not address the speed problem.
- **vs. FP8-LM/TorchAO**: These methods target large-matrix optimizations in pretraining (achieving ~1.38× speedup) and do not address the quantization overhead of LoRA's small matrices.
- **vs. LoRA-FA**: FALQON borrows the idea of training only $B$ from LoRA-FA, but reframes it within an FP8 melding framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combined design of melded LoRA and concatenation-based gradient reuse is novel, with original insights into FP8 quantization overhead.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Detailed latency analysis, coverage across multiple datasets and models, and thorough breakdown analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is exceptionally clear; figures and tables effectively convey core information.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the practical bottleneck of LoRA fine-tuning slowdown on FP8 hardware; the 3× speedup has significant implications for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)

</div>

<!-- RELATED:END -->
