---
title: >-
  [Paper Note] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving
description: >-
  [NeurIPS 2025][Model Compression][LoRA] Loquetier is a framework that unifies the fine-tuning and inference of multiple LoRA adapters within a single runtime via a Virtualized Module and a Segmented Multi-LoRA Multiplication (SMLM) kernel, achieving a 3.0× throughput improvement for inference-only tasks and a 46.4× higher SLO attainment rate for unified tasks.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "LoRA"
  - "multi-adapter serving"
  - "unified fine-tuning and inference"
  - "virtualized module"
  - "kernel optimization"
date: 2026-05-08
content_hash: e79226c7579d4bd6
---

# Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving

**Conference**: NeurIPS 2025

**arXiv**: [2511.00101](https://arxiv.org/abs/2511.00101)

**Code**: [Available (GitHub)](https://github.com/NJUDeepEngine/Loquetier)

**Area**: Model Compression / LLM Systems Optimization

**Keywords**: LoRA, multi-adapter serving, unified fine-tuning and inference, virtualized module, kernel optimization

## TL;DR

Loquetier is a framework that unifies the fine-tuning and inference of multiple LoRA adapters within a single runtime via a Virtualized Module and a Segmented Multi-LoRA Multiplication (SMLM) kernel, achieving a 3.0× throughput improvement for inference-only tasks and a 46.4× higher SLO attainment rate for unified tasks.

## Background & Motivation

LoRA (Low-Rank Adaptation) has become the predominant technique for parameter-efficient fine-tuning (PEFT) of LLMs. In practical deployments, a single base model typically serves multiple LoRA adapters corresponding to different downstream tasks.

**Limitations of existing systems**:

**Separation of fine-tuning and inference**: Training and inference rely on distinct systems, leading to low resource utilization.

**High adapter-switching overhead**: Loading and unloading different LoRA adapters incurs substantial I/O and memory operations.

**Inefficient batching**: Requests from different LoRA adapters are difficult to batch efficiently.

**Fragmented kernel invocations**: Each adapter's forward pass requires independent kernel calls.

Prior works such as S-LoRA and dLoRA have addressed aspects of multi-LoRA serving, but:
- S-LoRA supports inference only and does not support unified fine-tuning and inference;
- Conventional PEFT frameworks are insufficient for mixed multi-task scenarios;
- No existing solution merges the computation paths of fine-tuning and inference at the kernel level.

## Method

### Overall Architecture

Loquetier consists of two core components:

1. **Virtualized Module**: Isolates PEFT modifications on top of a shared base model.
2. **Optimized Computation Flow + SMLM Kernel**: Merges fine-tuning and inference paths during the forward pass.

### Key Designs

#### Virtualized Module

**Design Motivation**: Drawing an analogy to OS-level virtualization, each LoRA adapter's modifications are encapsulated as an independent "virtual instance."

Core features:
- **Instance isolation**: Each LoRA adapter runs as an independent virtual instance without interference.
- **Shared base model**: All instances share a single copy of the base model weights, reducing memory consumption.
- **Flexible migration**: Supports dynamic loading and migration at the instance level.
- **Multi-PEFT support**: Not limited to LoRA; extensible to other PEFT methods.

**Architectural overview**:

```
Base Model (shared)
├── Virtual Instance 1 (LoRA-A, inference)
├── Virtual Instance 2 (LoRA-B, inference)
├── Virtual Instance 3 (LoRA-C, fine-tuning)
└── Virtual Instance 4 (LoRA-D, fine-tuning)
```

#### Segmented Multi-LoRA Multiplication Kernel (SMLM)

**Core Idea**: Consolidates the computation of multiple LoRA adapters into a single kernel invocation.

In conventional approaches, $m$ LoRA adapters require $m$ independent matrix multiplications:

$$y_i = x_i W + x_i A_i B_i, \quad i = 1, \ldots, m$$

SMLM merges these operations by:

1. **Input segmentation**: Arranging inputs from different adapters into contiguous segments.
2. **Segmented matrix multiplication**: Processing all segments within a single GPU kernel call.
3. **Shared base model computation**: Computing $xW$ only once.

**Computation flow comparison**:

| Operation | Conventional | SMLM |
|-----------|:-----------:|:----:|
| Base model forward | $m$ times | 1 (batched) |
| LoRA forward | $m$ independent calls | 1 segmented call |
| Total kernel calls | $O(m \cdot L)$ | $O(L)$ |

where $L$ denotes the number of layers and $m$ the number of active adapters.

#### Unified Fine-tuning and Inference

**Unified scheduling**:

- Inference and fine-tuning requests are processed within the same batch.
- Inference requests: forward pass only.
- Fine-tuning requests: forward and backward pass.
- Computation paths are distinguished via flags; gradients are applied only to the corresponding LoRA parameters.

**Gradient isolation**: The Virtualized Module ensures that gradient updates from fine-tuning affect only the corresponding LoRA adapter.

### Loss & Training

Loquetier, as a system framework, supports arbitrary user-defined loss functions. Internal optimizations include:

- **Memory management**: A unified memory pool for LoRA weights.
- **Scheduling policy**: Priority-aware request scheduling that balances fine-tuning and inference demands.
- **CUDA kernels**: Custom SMLM kernels implemented using the CUTLASS library.

## Key Experimental Results

### Main Results

Evaluations are conducted under three task settings using LLaMA-series models.

#### Setting 1: Inference-only Multi-LoRA Serving

| System | Throughput (tokens/s) | Avg. Latency (ms) | P99 Latency (ms) | Relative Performance |
|--------|-----------------------|-------------------|------------------|:--------------------:|
| PEFT (baseline) | 1× | baseline | baseline | 1.0× |
| S-LoRA | 2.1× | lower | lower | 2.1× |
| **Loquetier** | **3.0×** | **lowest** | **lowest** | **3.0×** |

Loquetier achieves **3.0×** throughput over state-of-the-art joint serving systems in the inference-only setting.

#### Setting 2: Fine-tuning-only Multi-LoRA Training

| System | Training Throughput | Memory Usage | Multi-task Support |
|--------|:------------------:|:------------:|:-----------------:|
| PEFT (sequential) | 1× | $m \times$ base | serial |
| PEFT (time-sliced) | 0.8× | 1× base | interleaved |
| **Loquetier** | **higher than PEFT** | **1× base + LoRA pool** | **parallel** |

#### Setting 3: Unified Fine-tuning + Inference

| System | Inference Throughput | Fine-tuning Progress | SLO Attainment | Relative SLO |
|--------|:-------------------:|:-------------------:|:--------------:|:------------:|
| PEFT (separated) | low | normal | low | 1× |
| Naive switching | medium | slow | medium | ~5× |
| **Loquetier** | **high** | **normal** | **high** | **46.4×** |

In the unified setting, Loquetier achieves **46.4×** higher SLO attainment.

### Ablation Study

#### Contribution of Kernel Optimizations

| Component | Inference Throughput Gain | Description |
|-----------|:------------------------:|-------------|
| No optimization (baseline) | 1.0× | Naïve implementation |
| + Batched base model | 1.8× | Shared base computation |
| + SMLM kernel | 2.5× | Merged LoRA computation |
| + Virtualized Module | **3.0×** | Eliminated switching overhead |

#### Scalability with Number of Adapters

| Active Adapters $m$ | Conventional Latency | Loquetier Latency | Speedup |
|:-------------------:|:-------------------:|:-----------------:|:-------:|
| 2 | 1.5× | 1.05× | 1.4× |
| 4 | 2.8× | 1.10× | 2.5× |
| 8 | 5.2× | 1.18× | 4.4× |
| 16 | 10.1× | 1.25× | 8.1× |

Loquetier's latency grows slowly with the number of adapters, whereas conventional approaches scale linearly.

### Key Findings

1. The SMLM kernel is the largest contributor to performance gains, significantly reducing GPU overhead by eliminating redundant kernel invocations.
2. The isolation design of the Virtualized Module enables genuine parallel execution of fine-tuning and inference without locking the base model.
3. Loquetier's advantage becomes more pronounced as the number of active adapters increases—from 1.4× speedup with 2 adapters to 8.1× with 16.
4. The 46.4× improvement in SLO attainment is primarily attributable to the elimination of long-tail latency caused by fine-tuning–inference switching.

## Highlights & Insights

1. **Systems-level innovation**: The contribution is a systems architecture innovation rather than an algorithmic one, applying OS virtualization principles to ML serving.
2. **Kernel-level optimization**: The SMLM kernel eliminates multi-adapter computational fragmentation at the lowest level and is the key to performance gains.
3. **Open-source implementation**: Code is publicly available, with custom CUDA kernels implemented via CUTLASS.
4. **Unified paradigm**: Loquetier is the first framework to genuinely unify LoRA fine-tuning and inference, addressing a practical pain point in production deployment.

## Limitations & Future Work

1. **Single-GPU scope**: The current implementation primarily targets single-GPU scenarios; distributed multi-GPU scaling is not discussed.
2. **LoRA-centric validation**: Although multi-PEFT support is claimed, experiments are conducted exclusively with LoRA.
3. **Simple scheduling policy**: The current request scheduling may be suboptimal under high load.
4. **Large-scale models**: Performance on very large models (100B+) remains unverified.
5. **Potential directions**:
    - Integration with inference optimization techniques such as PagedAttention/vLLM.
    - Mixed deployment supporting heterogeneous PEFT methods.
    - QoS guarantees for multi-tenant scenarios.

## Related Work & Insights

- **S-LoRA** (Sheng et al., 2024): A pioneering work on multi-LoRA inference serving; Loquetier extends it to include fine-tuning.
- **PEFT** (Hugging Face): The mainstream parameter-efficient fine-tuning library, used as the primary comparison baseline.
- **vLLM**: An efficient LLM inference engine; Loquetier's kernel optimizations are complementary to it.
- **dLoRA**: A dynamic LoRA loading scheme focused on adapter-switching efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The virtualized approach to unified fine-tuning and inference represents a novel systems design.
- **Technical Depth**: ⭐⭐⭐⭐ — CUDA kernel-level optimizations demonstrate strong systems expertise.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across three settings with complete ablations.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly addresses real-world pain points in multi-LoRA production deployment.
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[ICML 2025\] LoRA Fine-Tuning without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs](../../ICML2025/model_compression/lora_fine-tuning_without_gpus_a_cpu-efficient_meta-generation_framework_for_llms.md)
- [\[NeurIPS 2025\] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](../../ACL2025/model_compression/domix_an_efficient_framework_for_exploiting.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)

</div>

<!-- RELATED:END -->
