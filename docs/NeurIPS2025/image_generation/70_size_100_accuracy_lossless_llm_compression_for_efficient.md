---
title: >-
  [Paper Note] 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)
description: >-
  [NeurIPS 2025][Image Generation][Lossless Compression] DFloat11 exploits the low-entropy property of exponent bits in BFloat16 weights to losslessly compress LLMs and diffusion models to approximately 70% of their original size (equivalent to ~11 bits) via Huffman coding. It further introduces hierarchical lookup tables and a two-phase GPU kernel for efficient online decompression, enabling lossless inference of Llama 3.1 405B on a single node with 8×80GB GPUs.
tags:
  - NeurIPS 2025
  - Image Generation
  - Lossless Compression
  - Huffman Coding
  - BFloat16
  - GPU Inference
  - Entropy Coding
date: 2026-05-08
content_hash: d346c0b3dcf1f3c9
---

# 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)

**Conference**: NeurIPS 2025
**arXiv**: [2504.11651](https://arxiv.org/abs/2504.11651)
**Code**: [https://github.com/LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)
**Area**: Image Generation
**Keywords**: Lossless Compression, Huffman Coding, BFloat16, GPU Inference, Entropy Coding

## TL;DR

DFloat11 exploits the low-entropy property of exponent bits in BFloat16 weights to losslessly compress LLMs and diffusion models to approximately 70% of their original size (equivalent to ~11 bits) via Huffman coding. It further introduces hierarchical lookup tables and a two-phase GPU kernel for efficient online decompression, enabling lossless inference of Llama 3.1 405B on a single node with 8×80GB GPUs.

## Background & Motivation

The rapid growth in large model (LLM and diffusion model) parameter counts poses severe memory bottlenecks for deployment. For instance, the BF16 weights of Llama 3.1 405B alone require approximately 810 GB of GPU memory, far exceeding the 640 GB capacity of a single 8×80GB GPU node, necessitating costly multi-node deployment.

The dominant compression paradigm is **quantization** (lossy compression), which suffers from three key limitations:
1. **Uncontrollable accuracy degradation**: Quantization error interacts in complex ways with the model, method, benchmark, and target bit-width, making accuracy loss difficult to predict in advance. Even 8-bit quantization can cause significant performance drops on certain tasks (e.g., 8-bit SmoothQuant on DeepSeek-R1-Distill-Qwen-1.5B yields an average 9.09% drop on reasoning benchmarks).
2. **Behavioral drift**: Even when overall accuracy is largely preserved, quantized models exhibit an "answer flip" phenomenon—previously correct answers become incorrect and vice versa—indicating that the underlying model behavior has changed.
3. **Compliance risk**: In sensitive domains such as finance and healthcare, the behavioral divergence of quantized models from their originals may fail to satisfy regulatory requirements.

Existing **lossless compression** methods (Deep Compression, ZipNN) primarily target storage and checkpoint scenarios, offering no efficiency benefit during GPU inference. The only GPU-inference-capable alternative, NeuZip, relies on the proprietary nvCOMP library and exhibits substantially lower decompression throughput and higher latency compared to DFloat11.

## Core Problem

**Can LLM memory footprint be substantially reduced without sacrificing any accuracy, while still enabling efficient GPU inference?** The fundamental challenge is that entropy-coded variable-length representations are inherently at odds with the massively parallel GPU architecture—traditional Huffman decoding requires sequential bit-by-bit traversal of the code tree, which cannot be parallelized.

## Method

### Overall Architecture

The core observation of DFloat11 is that **the 8-bit exponent field of BFloat16 is highly redundant**. Analysis of weight distributions across diverse mainstream LLMs reveals that the sign bit (1 bit) and mantissa (7 bits) have entropies close to their bit-width upper bounds, leaving little room for compression. In contrast, the Shannon entropy of the exponent (8 bits) is only approximately **2.6 bits**—only about 40 of the 256 possible exponent values actually appear in practice, and their frequency decays rapidly.

DFloat11 therefore applies Huffman coding exclusively to the exponent field, leaving the sign and mantissa bits uncompressed. The encoded exponents are tightly packed into a byte array `EncodedExponent`, while the sign and mantissa are stored in a separate byte array `PackedSignMantissa`. This yields an average of approximately 11 bits per weight (1 sign + ~2.6 encoded exponent + 7 mantissa), giving rise to the name DFloat11.

During inference, weights reside in GPU HBM in compressed form. When a weight matrix is required for matrix multiplication, it is decompressed online to BF16, used for computation, and the BF16 copy is immediately discarded to conserve memory.

### Key Designs

1. **Hierarchical Lookup Tables (LUTs) for Decoding**: Traditional Huffman decoding traverses the tree bit by bit, with frequent branching and low parallelism. DFloat11 decomposes the Huffman tree into a set of non-overlapping subtrees of height 8, each corresponding to a compact 256-entry LUT. Decoding proceeds by reading one byte at a time and performing an array lookup—if the current subtree can directly decode a symbol, it returns the symbol; otherwise, it points to the next-level LUT. The sparsity of exponent values in LLM weights (extreme exponent values in the range 240–255 never appear) is exploited by reusing these vacant entries as subtable pointers. The total memory for all LUTs and CodeLengths tables does not exceed $(8+1) \times 256$ bytes, fitting entirely in GPU SRAM (shared memory) for high-speed repeated lookups.

2. **Two-Phase GPU Kernel with Lightweight Auxiliary Variables**: The encoded exponent byte stream is partitioned into fixed-size chunks of $n=8$ bytes, with each GPU thread responsible for decoding one chunk. Variable-length coding introduces two parallelism challenges: (a) the starting bit offset for each thread is unknown; (b) the output position of decoded results is unknown for all but the first thread. The solutions are:

    - **Gaps array**: Each thread stores 5 bits recording the bit offset of the first valid Huffman code within its chunk relative to the chunk's starting byte (range [0, 31]).
    - **BlockOutputPos array**: Each thread block stores a single 32-bit integer representing the output index of the first element in that block. Compared to storing an output position per thread, this reduces auxiliary memory overhead by several hundred to thousands of times.
    - **Two-phase execution**: Phase 1—each thread decodes its chunk and counts the number of decoded elements (without writing to HBM), followed by an intra-block prefix sum (Blelloch algorithm) to determine the exact output position for each thread. Phase 2—threads re-decode their chunks, write results into an SRAM buffer, and finally flush to HBM via a single coalesced write. Encoded data is loaded into SRAM before Phase 1 to avoid redundant HBM reads.

3. **Transformer Block-Level Batched Decompression**: Individual weight matrices are often too small to fully saturate GPU resources. DFloat11 batches all weight matrices within a single Transformer block for joint decompression immediately before that block's forward pass. Token embeddings and the LM head are large enough to saturate GPU resources individually and are decompressed separately.

### Loss & Training

DFloat11 is a purely lossless compression scheme with **no training involved**. Compression is performed as a one-time offline preprocessing step: frequency statistics are collected over the BF16 exponents of each weight matrix, a Huffman tree is constructed, and the weights are encoded. Transformer blocks are independent and can be compressed in parallel across multiple threads. Compression time per block is approximately 191 ms for Llama 3.1 8B and 2133 ms for 405B.

## Key Experimental Results

| Model | Original Size | Compressed | Ratio | Effective Bit-Width |
|-------|--------------|------------|-------|---------------------|
| Llama 3.1 8B Instruct | 16.06 GB | 10.90 GB | 67.84% | 10.85 bit |
| Llama 3.3 70B Instruct | 141.11 GB | 95.40 GB | 67.61% | 10.82 bit |
| Llama 3.1 405B Instruct | 811.71 GB | 551.22 GB | 67.91% | 10.87 bit |
| Qwen 3 14B | 29.54 GB | 20.14 GB | 68.17% | 10.91 bit |
| FLUX.1 dev | 23.80 GB | 16.33 GB | 68.61% | 10.98 bit |
| Stable Diffusion 3.5 Large | 16.29 GB | 11.33 GB | 69.52% | 11.12 bit |

**Lossless verification**: DFloat11 achieves accuracy and perplexity on MMLU, TruthfulQA, WikiText, and C4 that are bit-for-bit identical to the BF16 baseline. SD 3.5 Large produces pixel-identical images under the same random seed.

**Inference efficiency vs. CPU offloading**:

| Scenario | DFloat11 Advantage |
|----------|--------------------|
| Throughput / Latency | 2.31–46.24× faster than CPU offloading |
| Context length | Supports 5.70–14.86× longer generation under the same memory budget |
| 405B single-node | DFloat11 enables the 810 GB model to run on 8×80GB GPUs |

**Diffusion models**: SD 3.5 achieves a 28.3% memory reduction with only a 4.1% latency increase; FLUX.1 achieves a 27.8% memory reduction with a 5.5% latency increase.

### Ablation Study

- **Latency breakdown**: Decompression overhead is constant and does not grow with batch size. The overhead is amortized more effectively at larger batch sizes.
- **DFloat11 vs. nvCOMP ANS**: Across varying matrix sizes and GPU models, DFloat11 decompression throughput is up to 20.97× higher than nvCOMP and up to 34.95× higher than CPU-to-GPU transfer. DFloat11 also achieves better compression ratios (68% vs. 79%).
- **Larger matrices yield higher GPU utilization and better decompression throughput**, motivating the block-level batched decompression design.

## Highlights & Insights

- **Distinctive problem framing**: Rather than pursuing lossy quantization, the paper identifies an information-theoretic vulnerability in BFloat16 exponents—8 bits carry only 2.6 bits of information—an elegant and well-motivated observation.
- **Sophisticated engineering**: The hierarchical LUT design cleverly repurposes the sparse region of the exponent value space (values 240–255 never appear in LLM weights) as subtable pointers; the Gaps array requires only 5 bits per thread; and BlockOutputPos requires only 32 bits per block—collectively reducing auxiliary variable memory overhead to near zero.
- **The decode-count-then-decode-write two-phase kernel pattern** is a general solution to the parallel write conflict inherent in variable-length coding, and is transferable to other parallel variable-length decoding scenarios.
- **High practical utility**: A pip package `dfloat11` is available, pre-compressed models are hosted on HuggingFace, and integration with the HuggingFace Transformers framework enables out-of-the-box deployment.

## Limitations & Future Work

- **BF16-only**: The method does not support FP32, FP16, FP8, or other formats; exponent distributions differ across formats and would require distinct compression strategies.
- **Non-negligible latency overhead at small batch sizes**: Inference is approximately 2× slower at batch size 1 (as noted in the GitHub README), which is problematic for latency-sensitive online serving scenarios.
- **GPU-only evaluation**: The method has not been evaluated on CPUs, TPUs, or specialized accelerators, where platform-specific optimizations may be required.
- **Fixed compression ceiling**: The compression ratio is fundamentally bounded by the entropy of the exponent distribution (~2.6 bits), locking the achievable size reduction at approximately 30%. Further compression might be possible by applying near-lossless entropy coding to the mantissa (e.g., entropy-coding high-order mantissa bits while allowing minimal error in low-order bits).
- **Orthogonality with quantization unexplored**: The paper does not investigate combining DFloat11 with quantization (e.g., INT8 weights followed by DF11 compression of residual precision bits), which could yield greater compression while better preserving accuracy.

## Related Work & Insights

| Method | Type | Compression Ratio | GPU Inference | Lossless |
|--------|------|------------------|---------------|----------|
| GPTQ / AWQ | Lossy quantization | ~25% (4-bit) | ✅ Fast | ❌ |
| Deep Compression | Lossless (Huffman on quantized CNNs) | ~22% additional | ❌ Storage only | ✅ |
| ZipNN | Lossless (storage compression) | > zlib/zstd | ❌ Storage only | ✅ |
| NeuZip | Lossless (ANS + nvCOMP) | ~21% | ✅ But slow | ✅ |
| **DFloat11** | **Lossless (Huffman + custom kernel)** | **~30%** | **✅ Fast** | **✅** |

DFloat11 is currently the only compression scheme that simultaneously achieves **losslessness, GPU-inference friendliness, and open-source availability**. Compared to NeuZip, it achieves over 20× faster decompression without relying on the proprietary nvCOMP library. Compared to quantization methods, while the compression ratio is less aggressive than 4-bit quantization (DFloat11 ~11 bits vs. INT4's 4 bits), it completely eliminates concerns over accuracy loss and behavioral drift.

**Further connections and implications**:
- **Complementarity with quantization**: DFloat11 and quantization are not mutually exclusive. One could first quantize a model to INT8/FP8, then apply DFloat11-style entropy coding to the redundant bits of the quantized weights. Such cascaded "lossy then lossless" compression could yield additional size reductions.
- **Format-level optimization perspective**: The central insight of this paper—that standard data formats are not information-optimal for model weights—generalizes broadly. Analogous adaptive compression formats may be applicable to KV caches, activations, and gradients.
- **GPU kernel design patterns**: The two-phase decoding kernel and hierarchical LUT design are transferable to other scenarios requiring efficient GPU decoding of variable-length data, such as video decoding and sparse matrix decompression.

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea of Huffman coding over exponents is not entirely novel, but the information-theoretic motivation and GPU kernel design are excellent.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple mainstream LLM and diffusion model architectures, with latency breakdowns, ablations, and comparisons against nvCOMP and CPU offloading.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly structured, with compelling motivation, precise technical descriptions, and detailed illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly practical, with an open-source implementation and pre-compressed models readily available, offering direct value for industrial deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[NeurIPS 2025\] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation](llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](../../ICLR2026/image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[CVPR 2026\] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation](../../CVPR2026/image_generation/evatok_adaptive_length_video_tokenization_for_eff.md)

<!-- RELATED:END -->
