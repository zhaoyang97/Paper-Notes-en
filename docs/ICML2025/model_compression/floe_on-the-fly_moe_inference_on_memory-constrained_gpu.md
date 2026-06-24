---
title: >-
  [Paper Note] FloE: On-the-Fly MoE Inference on Memory-constrained GPU
description: >-
  [ICML2025][Model Compression][MoE Inference] Proposes FloE, an on-the-fly MoE inference system tailored for consumer-grade GPUs. By design of specialized intra-expert hybrid compression (contextual sparsification + ultra-low-bit quantization) and dual predictors, FloE pipelines computation and transmission. It successfully deploys Mixtral-8×7B on a single RTX 3090 with only 11GB VRAM, achieving a 48.7x speedup compared to DeepSpeed-MII with only a 4.4%--7.6% performance degra…
tags:
  - "ICML2025"
  - "Model Compression"
  - "MoE Inference"
  - "Expert Offloading"
  - "Activation Sparsity"
  - "Ultra-low-bit Quantization"
  - "Memory-constrained GPU"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 09340cee4abccd54
---

# FloE: On-the-Fly MoE Inference on Memory-constrained GPU

**Conference**: ICML2025  
**arXiv**: [2505.05950](https://arxiv.org/abs/2505.05950)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: MoE Inference, Expert Offloading, Activation Sparsity, Ultra-low-bit Quantization, Memory-constrained GPU, Inference Acceleration

## TL;DR
Proposes FloE, an on-the-fly MoE inference system tailored for consumer-grade GPUs. By design of specialized intra-expert hybrid compression (contextual sparsification + ultra-low-bit quantization) and dual predictors, FloE pipelines computation and transmission. It successfully deploys Mixtral-8×7B on a single RTX 3090 with only 11GB VRAM, achieving a 48.7x speedup compared to DeepSpeed-MII with only a 4.4%--7.6% performance degradation.

## Background & Motivation
MoE models (e.g., DeepSeek-R1, Mixtral) reduce inference computation through sparse activation, but a large number of inactive experts consume immense amounts of memory. Mixtral-8×7B in FP16 format requires 94GB VRAM, out of which 70% is "wasted" on inactive experts.

**Expert Offloading**, which stores expert parameters in CPU memory and loads them to the GPU on demand, is a natural solution. However, the core bottlenecks are:
- PCIe 4.0 bandwidth is only 32GB/s, far below the GPU's internal HBM bandwidth (300GB/s).
- A single Mixtral expert has 300MB of FP16 parameters, taking ~15ms to transfer but only ~5ms to compute.
- Prior works adopt uniform ultra-low-bit quantization to reduce transmission overhead, which severely degrades generation quality.

**Core Problem**: How to hide expert I/O overhead within model computation to achieve on-the-fly inference without significantly degrading performance?

## Method

### Overall Architecture
FloE consists of three core components: hybrid expert compression, dual sparsity predictors, and system-level co-optimization.

### 1. Hybrid Expert Compression
**Key Discovery**: The three projection matrices of an MoE expert (gate, up, and down) show different sensitivities to compression.

#### Contextual Sparsification — for gate and down projections
- **Observation**: MOE experts exhibit significant inner activation sparsity (with the absolute values of up projection outputs heavily concentrated near zero).
- **Theoretical Proof**: Pruning according to the magnitude of the down projection input yields the smallest error, followed by the up projection, whereas the gate projection is the most sensitive.
- **Design Choice**: Generate a mask based on the output magnitude of the up projection, applying channel-level sparsification to the gate and down projections.
- At 90% sparsity, the perplexity of Mixtral increases by only about 0.5%.

Sparsified forward propagation: $\mathbf{a}^S(x) = (\text{SiLU}(\mathbf{x}\mathbf{W}^{\text{gate}}) \odot S_t(\mathbf{x}\mathbf{W}^{\text{up}})) \mathbf{W}^{\text{down}}$

#### Ultra-low-bit Quantization — for up projection
- **Observation**: The up projection is the least sensitive to quantization (under INT2, its perplexity degradation is only 46% of gate's and 27% of down's).
- **Analysis**: The MLP can be viewed as a key-value memory model, where up/gate acts as keys to select activated values (down). As a key, up can tolerate more quantization noise.
- Quantize the up projection using HQQ INT2.

**Final Compression Effect**: 9.3x parameter compression per expert.

### 2. Dual Sparsity Predictors

#### Inter-expert Predictor (learning-based)
- Uses the current layer's hidden states to predict which experts should be activated in the next layer.
- The cosine similarity between hidden states of adjacent layers holds > 0.95 (except for the first layer).
- Employs a single-layer MLP (32K parameters) for shallow layers, and a two-layer MLP (2M parameters) for deep layers.
- Average accuracy: 0.88.

#### Intra-expert Predictor (reuse-based)
- Multiplies the current layer's hidden states with the next layer's up projection matrix (reused) to estimate the sparsity distribution.
- No extra parameters, zero additional memory overhead.
- Average recall: 0.95.

### 3. System Co-optimization
- **Efficient Sparse GEMV Kernel**: Column-major storage + selective column loading implemented via Triton.
- **Compact Asynchronous Transmission**: Column-aligned storage of gate and down projections, AVX-512 instructions + multi-threaded packing + multi-stream asynchronous transmission.
- Achieves 88% of peak transmission bandwidth, which is 12.6x faster than PyTorch's native implementation.

## Key Experimental Results

### End-to-End Inference Speed (RTX 3090, 12GB VRAM)

| Method | Relative Mixtral-GPU Speed | Speedup |
|------|----------------------|--------|
| DeepSpeed-MII | Extremely slow (FP16 Offloading) | 1x |
| Mixtral-Offloading | - | 18.7x |
| Fiddler | - | 15.5x |
| **FloE** | **91% of Mixtral-GPU** | **48.7x** |

### Deployment Capabilities

| Metric | Value |
|------|------|
| Minimum VRAM Requirement | 11GB |
| Memory Footprint Compression | 8.5x |
| Per-expert Parameter Compression | 9.3x |
| Performance Degradation | 4.4%--7.6% |

### Downstream Tasks (Average Accuracy across 7 Zero/Few-shot Tasks)
- FloE-W^up (sparsification only) outperforms CATS by 2.8% at 80% sparsity and by 9.8% at 90% sparsity.
- FloE (sparsified + quantized) still outperforms HQQ INT3 and CHESS.
- Sparsification and quantization errors are mostly independent and additive.

### Sparse Kernel Speedup (RTX 3090 Single Expert)

| Sparsity | Speedup |
|--------|--------|
| 50% | 1.43x |
| 70% | 1.72x |
| 90% | 1.92x |

## Highlights & Insights
1. **Matrix-level differentiated compression** is the core innovation: up projection quantization + gate/down projection sparsification preserves quality significantly better than uniform quantization.
2. **The dual-predictor design** is elegant: the learning-based inter-expert predictor is parameterized but lightweight, while the reuse-based intra-expert predictor is parameter-free with zero overhead, collectively enabling a seamless computation-transmission pipeline.
3. **Excellent coordination of theory and practice**: The authors theoretically prove that pruning based on down-projection input is optimal, and experimentally verify that using up-projection output is near-optimal while also enabling transfer predictability.
4. **Comprehensive system engineering**: Full-stack optimization from algorithms and kernels to transmission protocols renders the end-to-end acceleration highly tangible.
5. **Deployability on consumer-grade GPUs**: Running Mixtral-8x7B on just 11GB of VRAM drastically lowers the barrier to using MoE models.

## Limitations & Future Work
1. Complete evaluation is only conducted on Mixtral-8x7B; generalization to newer MoE architectures like DeepSeek-V2/V3 remains to be validated.
2. Currently only supports single-batch inference (latency-sensitive scenarios), while high-throughput batch inference is not yet covered.
3. The inter-expert predictor requires training, inducing additional costs when scaling to new models.
4. The trade-off between sparsity and performance is more pronounced in knowledge-intensive tasks like MMLU.

## Related Work & Insights
- **Mixtral-Offloading (Eliseev & Mazur, 2023)**: A pioneering scheme featuring unified quantization, prediction, and caching.
- **CATS (Lee et al., 2024a)**: An activation sparsification method for dense LLMs.
- **Fiddler (Kamahori et al., 2024)**: A CPU-GPU co-execution scheme.
- Insight: MoE inference optimization should exploit redundancy "within experts" rather than focusing solely on routing between experts.

## Rating
- Novelty: ⭐⭐⭐⭐ (System-level innovation combining hybrid compression and dual predictors)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Dual dimensions of efficiency and effectiveness, verified across multiple GPUs and tasks)
- Writing Quality: ⭐⭐⭐⭐ (Rich illustrations, clear logical flow from motivation to observation and design)
- Value: ⭐⭐⭐⭐⭐ (Enables on-the-fly MoE execution on consumer-grade GPUs, offering extremely high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds](../../ICML2026/model_compression/memory-efficient_partitioned_dnn_inference_on_resource-constrained_android_crowd.md)
- [\[CVPR 2026\] Otil: Accelerating Diffusion Model Inference via Communication-Efficient Multi-GPU Parallelism](../../CVPR2026/model_compression/otil_accelerating_diffusion_model_inference_via_communication-efficient_multi-gp.md)
- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](../../NeurIPS2025/model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [\[ICLR 2026\] Study of Training Dynamics for Memory-Constrained Fine-Tuning (TraDy)](../../ICLR2026/model_compression/study_of_training_dynamics_for_memory-constrained_fine-tuning.md)

</div>

<!-- RELATED:END -->
