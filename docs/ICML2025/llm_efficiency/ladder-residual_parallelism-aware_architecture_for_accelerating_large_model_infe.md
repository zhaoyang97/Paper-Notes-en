---
title: >-
  [Paper Note] Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference
description: >-
  [ICML 2025][LLM Efficiency][Tensor Parallelism] This paper proposes Ladder Residual, a simple architectural modification that shifts the input of each block from the output of the previous layer to the output of the layer before the previous one (staggered residual connection). This design decouples module computation from AllReduce communication, enabling overlap between communication and computation. It achieves a 29% end-to-end acceleration in 8-GPU Tensor Parallelism (TP)…
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Tensor Parallelism"
  - "Communication Hiding"
  - "Residual Connection"
  - "Inference Acceleration"
  - "Model Architecture"
date: 2026-05-08
content_hash: 8543b814b40b56f7
---

# Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference

**Conference**: ICML 2025  
**arXiv**: [2501.06589](https://arxiv.org/abs/2501.06589)  
**Code**: None (Implemented based on gpt-fast)  
**Area**: LLM Efficiency  
**Keywords**: Tensor Parallelism, Communication Hiding, Residual Connection, Inference Acceleration, Model Architecture

## TL;DR
This paper proposes Ladder Residual, a simple architectural modification that shifts the input of each block from the output of the previous layer to the output of the layer before the previous one (staggered residual connection). This design decouples module computation from AllReduce communication, enabling overlap between communication and computation. It achieves a 29% end-to-end acceleration in 8-GPU Tensor Parallelism (TP) inference on a 70B model with performance comparable to standard Transformers.

## Background & Motivation
**Background**: As the scale of LLMs grows, Tensor Parallelism (TP) remains a critical technique for multi-GPU inference, partitioning weights and activations across multiple devices for parallel execution.

**Limitations of Prior Work**: TP requires blocking AllReduce synchronization at each layer. In a 70B model setup with TP=8, AllReduce communication accounts for 38% of the inference latency, and even exceeds 50% when peer-to-peer (P2P) transfers are disabled.

**Key Challenge**: Existing communication optimization solutions (e.g., Flux, CoCoNet) often rely on low-level kernel fusion or custom compilers, which are difficult to port across hardware platforms and restricted by the inherent sequential dependency of the model architecture ($h_{i+1}$ depends on the communication results of $x_i$).

**Goal**: Decouple communication and computation through model-level architectural changes (instead of low-level system optimizations) to hide communication latency.

**Key Insight**: Based on the observation that activations in Transformers change slowly (the norm of the update $h_i(x)$ per layer is relatively small compared to the input residual $x$), using "stale" inputs instead of the latest inputs does not significantly degrade model performance.

**Core Idea**: Change the standard residual connection $x_{i+1} = h_{i+1}(x_i) + x_i$ to $x_{i+1} = h_{i+1}(x_{i-1}) + x_i$, so that the computation of $h_{i+1}$ does not depend on the AllReduce result of $x_i$, allowing them to be executed in parallel.

## Method

### Overall Architecture
Ladder Residual modifies the residual connection pattern of Transformers: the input of each module (Attention or MLP) is no longer the immediate preceding residual stream output, but rather the residual stream output from two steps prior. The residual stream itself is still accumulated in the standard manner to ensure no loss of information. Consequently, the computation of each module can be parallelized with the AllReduce of the preceding module.

### Key Designs

1. **Ladder Residual Connection**:

    - **Function**: Modifies the standard Transformer residual connection $x_{i+1} = h_{i+1}(x_i) + x_i$ to $x_{i+1} = h_{i+1}(x_{i-1}) + x_i$.
    - **Mechanism**: Module $h_{i+1}$ utilizes $x_{i-1}$ (which has completed its AllReduce in the previous step), eliminating the wait for the AllReduce of $x_i$. Therefore, AllReduce($x_i^*$) can run in parallel with $h_{i+1}(x_{i-1})$.
    - **Design Motivation**: In a Transformer, the norm of the update at each layer is small relative to the residual stream. Using an input "one step stale" has a limited impact on the final representation.
    - **Difference from standard Transformer**: The residual stream is still accumulated normally (ensuring that block $i$ can process all information up to block $i-2$), and only the module inputs utilize the stale values.

2. **Asynchronous AllReduce Implementation**:

    - **Function**: Employs NCCL asynchronous communication combined with handle passing to achieve non-blocking AllReduce.
    - **Mechanism**: Starts an asynchronous AllReduce and returns a handle immediately after the Attention computation is complete; it then synchronizes using the previous layer's MLP handle and executes the current MLP computation.
    - Implementation in PyTorch is highly concise, requiring no custom kernels.

3. **Hybrid Ladder Adaptation (Post-training Adaptation)**:

    - **Function**: Converts the upper half layers of a pre-trained Llama-3.1-8B-Instruct model into Ladder Residuals, followed by lightweight fine-tuning on only 3B tokens.
    - **Mechanism**: Keeps the lower half layers unchanged to prevent disrupting fundamental knowledge that is hard to recover, and only modifies the upper half (layers 16–32).
    - Fine-tuning with 3B tokens is sufficient to recover performance comparable to the original model.

### Loss & Training
- From-scratch training: Standard language modeling loss, cosine scheduler, peak LR 3e-4, warmup 8B tokens.
- Post-training adaptation: Supervised Fine-Tuning (SFT) on the Infinity-Instruct dataset (3B tokens), LR 5e-6, 200 warmup steps.
- Training configuration: DDP/HSDP, BF16 mixed precision, batch size 4M tokens.

## Key Experimental Results

### Main Results (Inference Acceleration across Different Model Scales)

| Model Scale | Speedup (P2P Disabled) | Speedup (P2P Enabled) |
|-------------|-----------------------|-----------------------|
| 1B          | 1.39x                 | 1.56x                 |
| 3B          | 1.50x                 | 1.57x                 |
| 8B          | 1.40x                 | 1.46x                 |
| 34B         | 1.47x                 | 1.44x                 |
| 70B         | 1.59x                 | 1.29x                 |
| 176B        | 1.54x                 | 1.35x                 |
| 405B        | 1.57x                 | 1.31x                 |

### Detailed Latency Breakdown for 70B Model (TP=8, batch=1)

| Model | Prefill Improvement | Decode Improvement | Token/sec Improvement |
|------|------------|------------|--------------|
| Ladder-70B (P2P=1) | 5.78% | 23.71% | **30.79%** |
| Ladder-70B (P2P=0) | 6.94% | 37.71% | **59.87%** |
| Parallel-70B (P2P=1) | 5.42% | 18.04% | 21.75% |

### Pre-training Quality Comparison (100B tokens, FineWeb-edu)

| Model | ARC-C | HellaSwag | PIQA | SciQ | Winograde | Average | WikiText PPL |
|------|-------|----------|------|------|-----------|------|-------------|
| Standard-1.2B | 34.22 | 41.10 | 71.49 | 87.30 | 55.41 | 59.98 | 18.54 |
| Ladder-1.2B | 31.31 | 41.18 | 71.49 | 86.60 | 55.17 | 58.92 | 18.42 |
| Standard-3.5B | 38.99 | 46.48 | 74.59 | 92.00 | 58.48 | 64.11 | 14.48 |
| Ladder-3.5B | 36.77 | 45.66 | 73.72 | 89.90 | 58.96 | 62.91 | 14.90 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Hybrid-Ladder-8B-16L (After Fine-tuning) | Average 57.61 vs Original 56.11 | Only 16 upper layers modified + 3B token fine-tuning; performance matches or slightly exceeds standard. |
| Hybrid-Ladder-8B-20L (After Fine-tuning) | Average 53.86 | Modifying 20 layers is too aggressive, leading to some performance degradation. |
| 30% Larger Ladder | Better than Standard of the same size | Scaling up the model using the computational budget saved from speedups is a superior strategy. |

### Key Findings
- Slower communication yields greater speedups: scale-up on 70B increases from 29% to 60% when P2P is disabled.
- The Decode stage benefits the most (due to its low compute intensity and high communication ratio).
- Ladder Residual is fully compatible with Pipeline Parallelism.
- Post-training adaptation requires only 3B tokens, which is significantly less than other architecture conversion methods (e.g., Llama-to-Mamba requires 50B tokens).

## Highlights & Insights
- Significantly accelerates inference via extremely simple architectural modifications without requiring low-level kernel engineering.
- Exploits the intrinsic property of "slow-changing activations" in Transformers to trade stale inputs for hidden communication latency.
- The post-training adaptation path is highly practical—existing models only need minor adjustments to achieve speedups.
- A 30% larger Ladder Transformer outperforms a standard Transformer with the same FLOPs, indicating that trading speedup for model size scaling is a highly effective strategy.

## Limitations & Future Work
- At the 3.5B scale, Ladder Transformer is slightly inferior to Standard (average accuracy difference of 1.2 points).
- More aggressive Ladder adaptation (e.g., 20 layers) leads to performance degradation; the optimal number of layers to adapt warrants further exploration.
- Speedup in the Prefill stage is limited (since Prefill is highly compute-bound with a small communication footprint).
- Currently only validated on language models; applicability to vision or multimodal models remains unexplored.

## Related Work & Insights
- Difference from Parallel Attention+MLP (PaLM): PaLM fuses Attn and MLP to reduce communication by half but sacrifices expressiveness; Ladder preserves the original computational graph and only modifies input routing.
- Difference from system-level optimizations like Flux/CoCoNet: These methods require re-writing low-level kernels, whereas Ladder can be implemented entirely in the top-level PyTorch APIs.
- Inspirations for future model design: "Parallelism-aware architecture" is a direction worthy of deeper exploration.
- Complementary efficiency optimizations such as Cross-Layer Attention and parameter sharing are compatible.

## Rating
- Novelty: ⭐⭐⭐⭐ (Simple but effective idea exploiting the intrinsic properties of Transformers)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers multiple scales from 1B to 405B, evaluate both pre-training and post-adaptation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, comprehensive analysis, and well-designed benchmarks)
- Value: ⭐⭐⭐⭐⭐ (Highly practical, applicable to any residual-based architectures)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](../../ICLR2026/llm_efficiency/scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)
- [\[ICLR 2026\] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](../../ICLR2026/llm_efficiency/semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)
- [\[ICLR 2026\] SpecBranch: Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism](../../ICLR2026/llm_efficiency/specbranch_speculative_decoding_via_hybrid_drafting_and_rollback-aware_branch_pa.md)
- [\[NeurIPS 2025\] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](../../NeurIPS2025/llm_efficiency/jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](../../ACL2025/llm_efficiency/accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)

</div>

<!-- RELATED:END -->
