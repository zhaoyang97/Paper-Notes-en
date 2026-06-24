---
title: >-
  [Paper Note] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks
description: >-
  [ICML 2025][Model Compression][Binary/Ternary Networks] Proposes the RSR/RSR++ algorithm—by preprocessing fixed binary/ternary weight matrices to build bucketed permutation indices, it achieves vector-matrix multiplication with $O(n^2/\log n)$ complexity, achieving up to 29× faster matrix multiplication and 6× memory savings compared to the standard $O(n^2)$ method, as well as a 5.24× speedup in 1.58-bit LLM inference.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Binary/Ternary Networks"
  - "Matrix Multiplication"
  - "Inference Acceleration"
  - "Logarithmic Factor Improvement"
  - "1.58-bit LLM"
date: 2026-05-08
content_hash: 12db5b0b14b632f5
---

# An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2411.06360](https://arxiv.org/abs/2411.06360)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Binary/Ternary Networks, Matrix Multiplication, Inference Acceleration, Logarithmic Factor Improvement, 1.58-bit LLM

## TL;DR
Proposes the RSR/RSR++ algorithm—by preprocessing fixed binary/ternary weight matrices to build bucketed permutation indices, it achieves vector-matrix multiplication with $O(n^2/\log n)$ complexity, achieving up to 29× faster matrix multiplication and 6× memory savings compared to the standard $O(n^2)$ method, as well as a 5.24× speedup in 1.58-bit LLM inference.

## Background & Motivation

**Background**: DNN and LLM inference efficiency is a key bottleneck for deployment. Weight quantization (especially binary/ternary quantization) has been shown to retain good accuracy—for example, 1.58-bit LLMs (BitNet b1.58) use ternary weights $\{-1, 0, 1\}$ while achieving accuracy close to full-precision models.

**Limitations of Prior Work**: Even when weights are quantized to 1-2 bits, the speedup during inference mainly comes from hardware-level optimizations (such as XNOR operations), while optimization at the algorithmic level is overlooked. Matrix multiplication remains the major bottleneck with $O(n^2)$ complexity.

**Key Challenge**: The weight matrix remains unchanged after training—this is an overlooked structural property. Acceleration during inference (online) can be obtained in exchange for preprocessing (offline).

**Goal**: Leverage the static nature of the weight matrix to design an inference algorithm with sub-quadratic time complexity.

**Key Insight**: Partition the columns of the binary weight matrix into blocks and sort the rows lexicographically $\rightarrow$ rows with the same pattern in the same block share computations $\rightarrow$ update the result vector in batches using aggregate values.

**Core Idea**: Preprocess the weight matrix into "bucketed permutation indices" $\rightarrow$ utilize these indices during inference to share computations among rows within each lexicographical group, achieving a logarithmic-factor speedup.

## Method

### Overall Architecture
Split into two phases:
1. **Preprocessing (Offline)**: Convert ternary matrices to binary matrix pairs $\rightarrow$ partition columns into blocks $\rightarrow$ sort rows lexicographically $\rightarrow$ construct bucketed indices.
2. **Inference (Online)**: Perform highly efficient vector-matrix multiplication using the constructed indices.

### Key Designs

1. **Ternary-to-Binary Decomposition**:

    - **Function**: Decompose the ternary matrix $W \in \{-1, 0, 1\}^{n \times n}$ into two binary matrices $W^+, W^- \in \{0, 1\}^{n \times n}$.
    - **Mechanism**: $W = W^+ - W^-$, where $W^+_{ij} = \mathbb{1}[W_{ij} = 1]$ and $W^-_{ij} = \mathbb{1}[W_{ij} = -1]$.
    - **Design Motivation**: The binary matrix structure is simpler, making it easier to leverage the redundancy of 0/1 patterns.

2. **RSR Algorithm (Reduction-Sharing-Reconstruction)**:

    - **Function**: Partition columns into blocks of width $b$, and sort the rows lexicographically within each block.
    - **Mechanism**: Each block has $n$ rows, but there are only $2^b$ possible 0/1 patterns (for a block of width $b$). Compute an aggregate value (the sum of corresponding positions in the input vector) for each pattern, then distribute the aggregate values to all matching rows.
    - **Complexity**: When choosing $b = \lfloor \log n \rfloor$, each block has at most $n$ distinct patterns that can share computation, yielding a total complexity of $O(n^2 / (\log n - \log \log n))$.
    - **Design Motivation**: Logarithmic column-block widths ensure that the number of patterns ($2^b \approx n$) is on the same order of magnitude as the number of rows ($n$), maximizing sharing benefits.

3. **RSR++ Algorithm (Improved Version)**:

    - **Function**: Further optimize the process of writing aggregate values back to the result vector.
    - **Mechanism**: Exploit continuity after sorting to optimize the distribution of aggregate values from $O(n)$ to $O(n/\log n)$.
    - **Complexity**: $O(n^2 / \log n)$—yielding a strict logarithmic-factor improvement.
    - **Design Motivation**: "Last-mile" optimization, eliminating the $\log \log n$ term in RSR.

4. **Memory Compression (Logarithmic Factor)**:

    - **Function**: Replace original weight matrices with indices.
    - **Mechanism**: The original storage for an $n \times n$ binary matrix is $O(n^2)$ bits, whereas storing it with permutation indices requires $O(n^2 / \log n)$ bits.
    - **Design Motivation**: The indices encode the lexicographical grouping of the rows, implicitly capturing the matrix content.

### Loss & Training
- Pure inference optimization that does not affect the training processes.
- Preprocessing is performed once during model loading.
- Compatible with any training methods that produce binary/ternary weights.

## Key Experimental Results

### Basic Matrix Multiplication ($n \times n$ Matrix)

| Matrix Size | Standard Multiplication | RSR++ | Speedup | Memory Savings |
|---------|---------|-------|--------|---------|
| 1024 | 1.0× | 0.18× | 5.6× | 4.2× |
| 4096 | 1.0× | 0.07× | 14.3× | 5.1× |
| 8192 | 1.0× | 0.05× | 20× | 5.8× |
| 16384 | 1.0× | 0.034× | **29×** | **6×** |

### 1.58-bit LLM Inference (BitNet b1.58)

| Model | Standard Inference Latency | RSR++ Latency | Speedup |
|------|-----------|-----------|--------|
| BitNet-700M | 1.0× | 0.31× | 3.2× |
| BitNet-3B | 1.0× | 0.19× | **5.24×** |

### Ablation Study

| Configuration | Speedup (n=8192) | Description |
|------|---------------|------|
| RSR (Basic Version) | 15× | $O(n^2/(\log n - \log \log n))$ |
| **RSR++** | **20×** | $O(n^2/\log n)$ |
| Tuning Block Width $b$ | 22× | Near theoretical optimal $b = \log n$ |
| Binary Matrix (Direct) | 25× | Slightly faster than ternary (no decomposition required) |
| Ternary Matrix | 20× | Requires decomposition into binary pairs |

### Key Findings
- The speedup grows with matrix size—as the logarithmic-factor improvement becomes more prominent in larger matrices.
- Achieves over 10× speedup on LLM-scale matrices ($n \geq 4096$).
- Memory savings also follow a logarithmic factor (4-6× for investigated matrix sizes).
- Preprocessing time occupies only a small fraction of inference time (completed during model loading).
- End-to-end acceleration of 1.58-bit LLMs validates the practical value of the proposed approach.

## Highlights & Insights
- The simple observation that the **"weight matrix is invariant after training"** holds powerful optimization potential—post-training preprocessing is virtually free.
- Elegant theoretical results: $O(n^2/\log n)$ is near-optimal complexity for binary matrix-vector multiplication.
- Integration with 1.58-bit LLMs (BitNet b1.58) has broad practical significance, enabling these models to run faster on consumer-grade devices.
- The method is completely decoupled from training—any existing binary/ternary training pipeline can directly benefit from it.
- Joint improvements in memory footprint and execution speed make the approach highly appealing.

## Limitations & Future Work
- Only applicable to strictly binary/ternary weights—not generalizing to higher-precision (e.g., 4-bit) quantization.
- Preprocessing indices requires a one-time overhead calculation.
- Potential hardware-level implementations (e.g., on FPGA/ASIC) could further amplify advantages but are left uninvestigated.
- Real-world speedups on GPUs might be limited in massive parallel scenarios (where GPUs already have highly optimized matrix multiplication).
- The optimal choice of block width $b$ depends heavily on the hardware cache size.

## Related Work & Insights
- **vs XNOR-Net**: XNOR optimizes binary multiplication at the hardware level, whereas RSR++ optimizes at the algorithmic level—the two are complementary.
- **vs BitNet b1.58**: BitNet defines training methods for ternary weights; RSR++ uncovers a new mechanism to accelerate inference for these models.
- **vs any4/GPTQ**: These are 4-bit+ quantization methods, whereas RSR++ focuses on the more extreme 1-2 bit scenario.
- **Insights**: The invariance of static weight matrices might contain unexploited structures in other quantization levels—such as row repetition patterns.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Elegant theory (logarithmic factor improvement); simple observations yield profound insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-scale validation spanning basic matrix multiplications to end-to-end LLMs.
- **Writing Quality**: ⭐⭐⭐⭐ Algorithms are clearly described.
- **Value**: ⭐⭐⭐⭐⭐ Holds major practical significance for deploying binary/ternary models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training](../../ICLR2026/model_compression/bep_a_binary_error_propagation_algorithm_for_binary_neural_networks_training.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ACL 2025\] Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](../../ACL2025/model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)
- [\[ICML 2025\] Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](sparse_spectral_training_and_inference_on_euclidean_and_hyperbolic_neural_networ.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)

</div>

<!-- RELATED:END -->
