---
title: >-
  [Paper Note] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache
description: >-
  [AAAI 2026][Model Compression][KV Cache Quantization] This paper proposes KVmix, which evaluates the importance of each layer's KV Cache by computing the $L_2$ norm of gradients with respect to Key/Value projection weigh…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "KV Cache Quantization"
  - "Mixed-Precision"
  - "Layer Importance"
  - "Gradient Analysis"
  - "Dynamic Context Selection"
  - "CUDA Optimization"
date: 2026-05-08
content_hash: d9040de0515a2e27
---

# KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache

**Conference**: AAAI 2026
**arXiv**: [2506.08018](https://arxiv.org/abs/2506.08018)  
**Code**: [LfLab-AI/KVmix](https://github.com/LfLab-AI/KVmix)  
**Area**: Model Compression
**Keywords**: KV Cache Quantization, Mixed-Precision, Layer Importance, Gradient Analysis, Dynamic Context Selection, CUDA Optimization

## TL;DR

This paper proposes KVmix, which evaluates the importance of each layer's KV Cache by computing the $L_2$ norm of gradients with respect to Key/Value projection weights, enabling layer-wise mixed-precision quantization (Key avg. 2.19-bit, Value avg. 2.38-bit). Combined with a dynamic Recent Pivotal Context (RPC) selection strategy, KVmix achieves near-lossless inference, 4.9× memory compression, and 5.3× throughput acceleration on models such as Llama and Mistral.

## Background & Motivation

**KV Cache is the memory bottleneck of LLM inference**: For a 70B model with a 20k-token sequence, KV Cache can exceed 50 GB, far beyond single-GPU memory capacity. Under concurrent requests, KV Cache cannot be shared, causing rapid memory saturation, frequent HBM swapping, and surging latency.

**Quantization is the dominant compression approach, but existing methods are insufficient**: Methods such as KIVI apply uniform precision across all layers (one-size-fits-all), lacking flexibility. Dynamic methods such as QAQ incur high computational overhead and struggle to adaptively prioritize critical KV pairs in long-context settings.

**KV contributions to model output vary significantly across layers**: Experiments on Llama 2-7B show that applying 2-bit quantization independently to different layers produces drastically different impacts on GSM8K/TruthfulQA — some layers incur negligible degradation while others cause catastrophic accuracy drops.

**Projection weights encode layer importance information**: $K_{i,t} = W_{k_i} \cdot H_{i-1,t}$. Since $W_{k_i}$ and $W_{v_i}$ are fixed after training, weight heatmaps reveal significant differences in magnitude and distribution patterns across layers.

**A lightweight and effective layer importance metric is needed**: Inspecting weight magnitudes alone is insufficient, as it ignores the relationship with the loss function. A metric that quantifies "the impact of KV perturbation on model output" is required.

**Long-context scenarios require dynamic optimization**: Retaining a fixed number of full-precision residuals (e.g., KIVI's r64) prevents dynamic reduction of full-precision KV entries during inference, resulting in memory waste.

## Method

### Overall Architecture

KVmix comprises three core components: (1) **KVmix Profiler** — offline gradient analysis that determines the importance of Key/Value in each layer and generates a mixed-precision configuration; (2) **Asymmetric low-bit quantization** — mixed-precision quantization with per-channel grouping for Keys and per-token grouping for Values; (3) **Dynamic Recent Pivotal Context selection (RPC)** — adaptively retaining full-precision KV for recent critical tokens based on layer importance. The entire profiling requires only a single offline execution (10–15 minutes) and does not affect inference efficiency.

### Key Design 1: Gradient-Based KV Importance Analysis (KVmix Profiler)

- **Function**: Computes the $L_2$ norm of the gradient of the model loss $L$ with respect to each layer's Key/Value projection weights $W_{k_i}$ and $W_{v_i}$ as the layer-wise KV importance score.
- **Mechanism**: By the chain rule, $\|\frac{\partial L}{\partial K_i}\|_2 = \frac{\|\nabla_{W_{k_i}} L\|_2}{\|H_{i-1}\|_2}$. After quantization introduces perturbation $\Delta K$, the loss change satisfies $\Delta L \approx \langle \frac{\partial L}{\partial K_i}, \Delta K \rangle$: a larger gradient norm implies greater loss sensitivity under equivalent quantization error. Importance scores are defined as $s_{k_i} = \|\nabla_{W_{k_i}} L\|_2$ and $s_{v_i} = \|\nabla_{W_{v_i}} L\|_2$, averaged over $P$ prompts: $\bar{s}_{k_i} = \frac{1}{P}\sum_{p=1}^{P} s_{k_i}^{(p)}$.
- **Design Motivation**: Unlike approaches based on raw weight magnitudes or attention score statistics, gradient norms directly reflect quantization sensitivity — grounded in the theoretical guarantee of a first-order Taylor expansion. The offline analysis (30 prompts, one forward + backward pass) is extremely low-cost.

### Key Design 2: Asymmetric Low-Bit Quantization

- **Function**: Keys use per-channel quantization; Values use per-token quantization. By default, important layers are quantized to 3-bit or 4-bit and the remaining layers to 2-bit (top 20% high-precision, 80% low-precision).
- **Mechanism**: Key Cache exhibits significant outliers along the channel dimension, so per-channel quantization isolates errors; Value Cache lacks prominent outliers but is critical to attention output, so per-token quantization preserves individual token integrity. The quantization formula is $q = \text{round}(\frac{x - \text{min\_val}}{s})$, $s = \frac{\text{max\_val} - \text{min\_val}}{q_{\max}}$, with results packed into int32 storage. A novel 3-bit packing scheme is introduced: groups of 11 elements (first 10 at 3-bit + the 11th at 2-bit) achieve 10% higher packing density than uniform 3-bit packing.
- **Design Motivation**: Differentiating quantization strategies according to the distinct distributional characteristics of Keys and Values minimizes quantization error.

### Key Design 3: Dynamic Recent Pivotal Context Selection (RPC)

- **Function**: Assigns an RPC ratio $r$ to each layer based on its importance score, computes $\text{num\_RPC} = \lfloor r \times \text{current\_RPC} \rfloor$, retains full-precision KV for recent critical tokens, and compresses older tokens via quantization.
- **Mechanism**: Layers with higher importance scores (larger $\bar{s}_{k_i}$/$\bar{s}_{v_i}$) are assigned a larger RPC ratio. High-precision layers use RPC=20%; low-precision layers use RPC=10%. As decoding progresses, the number of full-precision KV entries decreases dynamically.
- **Design Motivation**: Unlike KIVI's fixed r64 full-precision residual or StreamingLLM's attention-sink-only retention, RPC dynamically adjusts based on layer importance analysis, progressively compressing older KV entries during long-context inference to avoid the memory pressure caused by linearly growing full-precision KV counts.

### Key Design 4: Efficient CUDA Implementation

- **Function**: Three types of fused kernels are designed — quantization + concatenation fusion, dequantization + matrix-vector multiplication fusion, and dedicated kernels for multiple bit widths.
- **Mechanism**: Intermediate memory allocations are eliminated; quantized values are directly appended to the historical KV Cache, and dequantization is performed on-the-fly with immediate accumulation against the Query.
- **Design Motivation**: Eliminating the additional memory accesses and storage overhead introduced by quantization/dequantization translates the theoretical compression advantage of low-bit quantization into practical inference acceleration.

## Key Experimental Results

### Table 1: LongBench Long-Context Evaluation (5 models × 8 datasets; Llama 2-7B shown)

| Method | TriviaQA | Qasper | MF-en | QMSum | 2WikiMQA | Rbench-P | TREC | PsgRetr | Avg. |
|--------|---------|--------|-------|-------|----------|----------|------|---------|------|
| FP16 | 78.89 | 9.55 | 22.86 | 21.19 | 9.94 | 55.64 | 66.00 | 6.64 | 33.84 |
| KVmix-2bit | 77.57 | 9.58 | 22.47 | 20.45 | 9.15 | 56.34 | 66.00 | 5.29 | 33.36 |
| random-k2.19v2.38 | 78.30 | 9.39 | 22.54 | 20.41 | 9.46 | 56.36 | 66.00 | 5.49 | 33.49 |
| KVmix-k2.19v2.38 w/o RPC | 77.95 | 9.19 | 21.03 | 19.98 | 9.05 | 56.13 | 65.50 | 5.61 | 33.06 |
| **KVmix-k2.19v2.38** | **78.78** | **9.59** | **22.82** | **20.49** | **9.77** | **56.54** | **66.00** | **5.72** | **33.71** |

### Table 2: GSM8K / Wikitext-2 Comparison with SOTA (Llama 2-7B)

| Method | GSM8K acc↑ | Wikitext-2 ppl↓ |
|--------|-----------|-----------------|
| FP16 | 13.52 | 8.71 |
| 2bit (k-T, v-T) | 0.83 | 11089 |
| KIVI-2bit-r64 | 12.75 | 8.80 |
| QJL-3bit | 13.11 | 8.75 |
| KVQuant-3bit-1% | 13.23 | 8.71 |
| **KVmix-k2.19v2.38** | **13.25** | **8.71** |

### Efficiency Results

- **Memory compression**: KVmix-k2.19v2.38 achieves **4.9× memory compression** over the FP16 baseline, surpassing KIVI-2bit due to RPC dynamically reducing full-precision KV entries.
- **Throughput acceleration**: Maximum batch size reaches 30 (vs. 4 for FP16); inference throughput is **1032 tokens/s**, yielding **5.3× acceleration**.
- **Profiling cost**: Only 30 prompts and 10–15 minutes are required; a single profiling run can be reused indefinitely.

## Key Findings

- Uniform 2-bit quantization (KVmix-2bit) incurs an average 4.53% degradation on LongBench, whereas gradient-guided mixed-precision (KVmix-k2.19v2.38) incurs only 1.67% — **importance-aware mixed-precision significantly outperforms uniform precision**.
- Randomly selecting high-precision layers (random-k2.19v2.38) incurs 4.06% degradation, demonstrating that **gradient analysis accurately identifies critical layers** and cannot be replaced by random selection.
- Removing RPC (w/o RPC) incurs 3.28% degradation, confirming that **dynamic context selection is essential for long-context quality**.
- Symmetric 2-bit quantization (key per-token, value per-token) causes GSM8K accuracy to collapse from 13.52 to 0.83, highlighting that **the choice of quantization granularity (per-channel vs. per-token) is critically important**.
- KVmix-k2.19v2.38 achieves ppl=8.71 on Wikitext-2, perfectly matching FP16, with only 0.27% degradation on mathematical reasoning (GSM8K).

## Highlights & Insights

- Using **gradient norms as KV importance indicators** is theoretically grounded (first-order Taylor expansion) and computationally cheap, making it more lightweight than attention score statistics or search-based optimization (KVTuner).
- **Independent analysis of Keys and Values**: within the same layer, Key and Value importances can differ (e.g., a layer where Key is important but Value is not), enabling finer-grained precision and RPC ratio assignment.
- The **3-bit packing scheme** (10×3-bit + 1×2-bit → 32-bit) improves packing density by 10%, demonstrating solid engineering rigor.
- The method is highly **tunable**: users can flexibly balance accuracy, memory, and throughput by adjusting the high-precision layer ratio (10%/20%/30%).
- KVmix is orthogonal to weight quantization methods (GPTQ/AWQ) and KV sparsification methods, and can be composed with them.

## Limitations & Future Work

1. **Profiling requires full model loading and backpropagation**: Although it is performed only once, it still demands substantial computational resources for very large models (70B+).
2. **Fixed 20/80 split**: The high/low precision layer ratio is fixed after profiling and cannot be dynamically adjusted per input — the authors themselves identify real-time bit adjustment as a direction for future work.
3. **Validation limited to 7B–8B scale models**: Larger models (13B/70B) and VLM/multimodal models have not been evaluated.
4. **RPC ratios are hyperparameters**: The RPC ratios of 20% for high-precision layers and 10% for low-precision layers require manual specification, with no adaptive mechanism.
5. **No joint evaluation with KV eviction/merging methods**: Complementary methods such as SnapKV and H2O are not explored in combination with KVmix.
6. **Long-context evaluation capped at 4096 tokens**: Due to GPU memory constraints, LongBench experiments use sequences of at most 4096 tokens; validation on truly long contexts (32k/128k) is absent.

## Related Work & Insights

- **vs. KIVI**: KIVI applies uniform 2-bit quantization across all layers with a fixed r64 full-precision residual → KVmix uses higher precision for important layers with dynamic RPC, achieving better accuracy and lower memory usage.
- **vs. KVQuant**: KVQuant's 3-bit + 1% outlier handling yields accuracy close to KVmix, but incurs higher preprocessing overhead and lower inference efficiency.
- **vs. QAQ**: QAQ dynamically computes per-token quantization bits online, incurring high overhead; KVmix performs a one-time offline analysis with zero additional overhead at inference time.
- **vs. KVTuner**: KVTuner formulates mixed-precision assignment as a search optimization problem over a large search space; KVmix directly ranks layers by gradient norm, which is more efficient.
- **vs. StreamingLLM/PyramidInfer**: These methods focus on the attention sink phenomenon for KV pruning; KVmix's RPC strategy is more principled as it is grounded in layer importance analysis.
- **Insight**: Gradient norms can be generalized as a universal "component importance" metric — applicable not only to quantization precision allocation, but also to token retention decisions in KV eviction and cross-layer token budget allocation.

## Rating

- Novelty: ⭐⭐⭐⭐ The pipeline from gradient norms → layer importance → mixed-precision quantization is clear and theoretically supported; the RPC dynamic strategy is an effective contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five models, three evaluation categories (long-context / language modeling / mathematical reasoning), multiple SOTA comparisons, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Highly practical — open-source with a CUDA implementation, tunable, and orthogonal to other compression methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](../../ICML2026/model_compression/gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICML 2026\] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](../../ICML2026/model_compression/xkv_cross-layer_kv-cache_compression_via_aligned_singular_vector_extraction.md)
- [\[ACL 2026\] IMPACT: Importance-Aware Activation Space Reconstruction](../../ACL2026/model_compression/impact_importance-aware_activation_space_reconstruction.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)

</div>

<!-- RELATED:END -->
