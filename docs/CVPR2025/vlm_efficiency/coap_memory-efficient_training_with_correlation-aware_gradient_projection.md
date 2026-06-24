---
title: >-
  [Paper Note] "COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection"
description: >-
  [CVPR2025][VLM Efficiency][Paper Note] Academic paper note for "COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection".
tags:
  - CVPR2025
  - VLM Efficiency
date: 2024-11-30
content_hash: 2bc9e9d40f6841a4
---
# COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection

## Background & Motivation

Training large language models (LLMs) faces severe memory bottlenecks. Taking LLaMA-7B as an example, the model parameters themselves occupy approximately 14GB (FP16), but the Adam optimizer needs to maintain two sets of state variables of the same size as the parameters (momentum and variance), resulting in an additional 28GB of VRAM being consumed by optimizer states. This makes training medium-to-large models on a single consumer-grade GPU (such as RTX 4090, 24GB) almost impossible.

Existing memory-efficient training methods mainly fall into two categories:

**Low-Rank Optimizers** (e.g., GaLore, Flora): Project gradients into a low-rank subspace to reduce the dimensions of optimizer states. However, the update of the projection matrix relies on SVD, which incurs extremely high computational overhead.

**Quantized Optimizers** (e.g., Q-Adam): Quantize and compress optimizer states. However, quantization errors accumulate, affecting training quality.

The core problem of low-rank projection methods lies in the **update strategy of the projection matrix $P_t$**. GaLore performs a full SVD on the current gradient every $T$ steps to update $P_t$. This operation is extremely time-consuming for large matrices (approximately 540 seconds per full SVD for a 7B model), severely slowing down the training speed.

The core insight of this paper: **High correlation exists between projection matrices of adjacent update cycles**. Leveraging this property, expensive full SVD can be replaced by incremental updates at an extremely low cost.

## Method

### Problem Formulation

Standard low-rank gradient projection projects the $m 	imes n$ gradient matrix $G_t$ into a rank-$r$ subspace:

$$	ilde{G}_t = P_t P_t^T G_t$$

where $P_t \in \mathbb{R}^{m 	imes r}$ is the projection matrix. Optimizer states (momentum, variance) are maintained in the low-rank space $\mathbb{R}^{r 	imes n}$, reducing the memory from $O(mn)$ to $O(rn + mr)$.

### Correlation-Aware Projection Update

The core innovation of COAP is dividing the update of the projection matrix into two phases:

#### Phase 1: SGD Incremental Update (Executed Every Step)

Utilizing the correlation between projections, the projection matrix $P_t$ is incrementally updated via a simple SGD step:

$$P_{t+1} = P_t - \eta_P 
abla_{P} \mathcal{L}$$

The computational complexity of this update is only $O(mr)$, which is far smaller than the $O(m^2n)$ of SVD.

#### Phase 2: Occasional Low-Cost SVD (Executed Every $T$ Steps)

Every $T$ steps, a **warm-start SVD** is executed: using the current $P_t$ as initialization, a partial SVD is performed on the gradient. Since the initialization is already close to the optimal solution, convergence requires very few iterations.

| Operation | GaLore SVD | COAP Warm-Start SVD | Speedup |
|------|-----------|----------------|-------|
| Execution Time per Step (LLaMA-7B) | ~540s | ~23s | **~20×** |
| Update Frequency | Every 200 steps | Every 200 steps | - |
| Amortized Overhead per Step | 2.7s | 0.12s | **~23×** |

### Inter-Projection Correlation Analysis

This paper experimentally verifies the high correlation between adjacent projection matrices:

$$\text{sim}(P_t, P_{t+T}) = rac{\|P_t^T P_{t+T}\|_F}{\|P_t\|_F \|P_{t+T}\|_F} > 0.95$$

This observation provides a theoretical foundation for the SGD incremental updates: the projection space changes slowly, allowing a small-step incremental update to track the optimal subspace.

### Memory Analysis

| Method | Optimizer Memory (LLaMA-1B) | Relative to Standard Adam |
|------|---------------------|-------------|
| Adam (FP16) | 4.0 GB | 100% |
| Adam (BF16) | 4.0 GB | 100% |
| GaLore (r=256) | 1.8 GB | 45% |
| Flora (r=256) | 1.6 GB | 40% |
| **COAP (r=256)** | **1.56 GB** | **39%** |

COAP achieves an **-61%** reduction in optimizer memory.

## Experimental Results

### LLaMA Pre-training

| Method | LLaMA-1B PPL↓ | LLaMA-7B PPL↓ | Training Speed (tokens/s) |
|------|--------------|--------------|-------------------|
| Adam | 14.89 | 12.31 | 1× |
| GaLore | 16.12 | 13.05 | 0.72× |
| Flora | 15.98 | 12.87 | 0.81× |
| **COAP** | **15.56** | **12.58** | **0.93×** |

### LLaVA-7B Fine-Tuning

| Method | Training Time | Accuracy | GPU Memory |
|------|---------|--------|---------|
| LoRA | 12.3h | 88.1% | 18GB |
| Full fine-tuning (Adam) | 47.1h | 82.4% | 62GB |
| GaLore | 15.2h | 87.3% | 24GB |
| **COAP** | **7.6h** | **92.3%** | **22GB** |

COAP achieves a **6.2× speedup** (7.6h vs. 47.1h) on LLaVA-7B fine-tuning, while improving accuracy from 82.4% to 92.3%.

### Downstream Task Evaluation

| Task | Adam | GaLore | COAP |
|------|------|--------|------|
| MMLU (5-shot) | 46.2 | 43.8 | 45.7 |
| HellaSwag | 72.1 | 69.4 | 71.5 |
| ARC-Challenge | 41.3 | 38.9 | 40.8 |
| WinoGrande | 67.4 | 65.1 | 66.9 |

## Summary & Outlook

By observing the high correlation between projection matrices, COAP designs an efficient two-phase projection update strategy: SGD incremental updates + occasional warm-start SVD. This design reduces the computational overhead of SVD by approximately 20× while maintaining projection quality comparable to full SVD. It achieves a PPL of 15.56 and a 61% reduction in optimizer memory during LLaMA-1B pre-training, and realizes a 6.2× speedup and a 9.9% accuracy improvement in LLaVA-7B fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Curvature-Aware Zeroth-Order Optimization for Memory-Efficient Test-Time Adaptation](../../CVPR2026/vlm_efficiency/curvature-aware_zeroth-order_optimization_for_memory-efficient_test-time_adaptat.md)
- [\[CVPR 2026\] Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding](../../CVPR2026/vlm_efficiency/attention-aware_inference_optimizations_for_large_vision-language_models_with_me.md)
- [\[CVPR 2026\] ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation in Vision-Language Models](../../CVPR2026/vlm_efficiency/zoo-prune_training-free_token_pruning_via_zeroth-order_gradient_estimation_in_vi.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](../../ACL2026/vlm_efficiency/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/vlm_efficiency/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)

</div>

<!-- RELATED:END -->
