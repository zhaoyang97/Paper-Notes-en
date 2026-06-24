---
title: >-
  [Paper Note] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models
description: >-
  [ACL 2025][Model Compression][Quantization-Aware Training] L4Q is proposed to deeply integrate Quantization-Aware Training (QAT) with LoRA: it first merges weights with LoRA parameters and then applies unified quantization. By customizing the backpropagation path, it eliminates the memory overhead of storing weight gradients, enabling joint optimization of quantization and fine-tuning parameters, which significantly outperforms existing methods under 4-bit and 3-bit quantizat…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Quantization-Aware Training"
  - "LoRA"
  - "Low-Precision Inference"
  - "Memory Optimization"
  - "LLM Compression"
date: 2026-05-08
content_hash: 165c515d0a56468c
---

# L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2402.04902](https://arxiv.org/abs/2402.04902)  
**Area**: Model Compression / Large Language Model Quantization / Parameter-Efficient Fine-Tuning  
**Keywords**: Quantization-Aware Training, LoRA, Low-Precision Inference, Memory Optimization, LLM Compression

## TL;DR

L4Q is proposed to deeply integrate Quantization-Aware Training (QAT) with LoRA: it first merges weights with LoRA parameters and then applies unified quantization. By customizing the backpropagation path, it eliminates the memory overhead of storing weight gradients, enabling joint optimization of quantization and fine-tuning parameters, which significantly outperforms existing methods under 4-bit and 3-bit quantization.

## Background & Motivation

The deployment of LLMs faces high memory and computational cost challenges. Quantization (reducing parameter bit-width) and parameter-efficient fine-tuning (such as LoRA) alleviate this issue from the inference and training perspectives, respectively. Existing methods combining the two suffer from fundamental limitations:

**Three Major Limitations of Prior Work**:

**QLoRA/LoftQ Schemes**: They first apply PTQ quantization and then perform LoRA fine-tuning. The high-precision LoRA parameters cannot be merged with the low-precision quantized weights, introducing additional mixed-precision overhead during inference.

**QA-LoRA Scheme**: It constrains the LoRA matrix structure so that it can be integrated into the quantization scale/bias, but this constraint limits the fine-tuning capability.

**Two-Stage Disconnected Problem**: Quantization and fine-tuning are optimized separately. Updates to quantization parameters do not affect LoRA parameters and vice versa, failing to achieve a global optimum.

**Key Challenge**: QAT can effectively reduce quantization errors but suffers from massive memory overhead (requiring ~80GB for a 7B model), while LoRA reduces training costs but direct integration diminishes their respective strengths.

## Method

### Overall Architecture

The core idea of L4Q is **"merge first, then quantize"**:

1. Merge the original frozen weights $W_0$ with the LoRA parameters $\alpha BA$ into $W_{comb} = W_0 + \alpha BA$
2. Quantize the merged weights uniformly: $\tilde{w} = \text{round}(\text{clamp}(\frac{W_{comb}-b}{s}, Q_N, Q_P))$
3. During inference, only the fully quantized weights $W_q$ are used: $Y = W_q X$

This is fundamentally different from QAT-LoRA (which separately maintains quantized weights and LoRA parameters, requiring two-path computation during inference).

### Key Designs

**1. Fully Quantized Linear Layer**: Unlike QA-LoRA, L4Q does not impose any constraints on the structure of the LoRA matrices. Since quantization is applied after merging, inference only requires a single fully quantized forward path $Y = W_q X$, eliminating the extra overhead of mixed-precision inference.

**2. Memory-Efficient QAT Backpropagation**:
- Traditional QAT requires storing the full weight gradient $\frac{\partial L}{\partial W_q}$ to update the quantization parameters $s$ and $b$.
- In backpropagation, L4Q **locally computes** the weight gradient $\frac{\partial L}{\partial W_q} = \frac{\partial L}{\partial Y} X^\top$, which is **immediately released** after being used to update the quantization parameters.
- The weight gradient is simultaneously **reused** for computing the LoRA parameter gradients, avoiding redundant calculation.

**3. LoRA Gradient Propagation through the Quantization Function**:
- Since quantization is applied after LoRA, the gradients of LoRA need to backpropagate through the non-linear quantization function.
- The Straight-Through Estimator (STE) is used to approximate the derivative of the rounding function.
- Gradient formula: $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial W_q} \cdot \frac{\partial W_q}{\partial A}$, where $\frac{\partial W_q}{\partial A} = \alpha B^\top$ (within the non-clamping interval).

**4. Joint Optimization**: Since $\frac{\partial L}{\partial W_q}$ participates in the gradient computation for both quantization parameters and LoRA parameters, variations in LoRA parameters naturally affect quantization adjustments and vice versa, achieving true joint optimization.

**5. L4Qinit Quantization Parameter Initialization**:
- Addressing the issue of activation outliers in LLMs, a conservative symmetric quantization initialization scheme is designed.
- $s = \text{Max}(|\frac{\text{Min}(W)}{Q_N}|, |\frac{\text{Max}(W)}{Q_P}|)$
- Compared with LSQ+init (based on standard deviation) and asymmetric initialization (min-to-max range), L4Qinit effectively reduces truncation errors during the training process.

## Key Experimental Results

### Main Results

**4-bit Quantization Accuracy Comparison (CSQA Accuracy %)**:

| Model | Pre-trained (16-bit) | LoRA (16-bit) | GPTQ(4) | OmniQ(4) | QLoRA*(4&16) | QA-LoRA(4) | L4Q(4) |
|------|----------------|--------------|---------|----------|--------------|------------|--------|
| OpenLLaMA 3B | 54.8 | 55.9 | 50.7 | 54.1 | 54.4 | 54.5 | **55.0** |
| LLaMA-3 8B | 65.6 | 67.2 | 57.9 | 64.9 | 58.6 | 58.0 | **66.8** |
| LLaMA-1 7B | 61.7 | 63.4 | 59.4 | 58.1 | 61.3 | 61.3 | **62.7** |
| LLaMA-2 7B | 61.9 | 63.3 | 60.7 | 59.5 | 61.3 | 61.0 | **63.6** |
| LLaMA-2 13B | 65.0 | 66.5 | 64.4 | 59.9 | 64.0 | 64.5 | **65.8** |
| Mistral 7B | 66.2 | 66.4 | 65.3 | 64.7 | 65.8 | 65.4 | **66.1** |

**3-bit Quantization Accuracy Comparison (CSQA)**:

| Model | GPTQ(3) | OmniQ(3) | QLoRA*(3&16) | QA-LoRA(3) | L4Q(3) |
|------|---------|----------|--------------|------------|--------|
| OpenLLaMA 3B | 52.2 | 50.0 | 51.0 | 51.5 | **54.0** |
| LLaMA-3 8B | 53.5 | 58.7 | — | — | **Significantly Leads** |

**Training Memory Overhead Comparison (GB, NVIDIA A100)**:

| Model | LoRA | QAT (Traditional) | QAT-LoRA | L4Q |
|------|------|-----------|----------|-----|
| LLaMA-1 7B | 25.1 | 79.5 | 41.9 | **25.4** |
| LLaMA-1 13B | 43.8 | OOM | 70.6 | **44.3** |
| LLaMA-1 33B | 71.9 | OOM | OOM | **73.2** |

### Key Findings

1. **L4Q training memory is almost matching LoRA**: For the 7B model, it only requires an additional 0.3GB (25.4 vs 25.1), whereas traditional QAT requires 79.5GB and QAT-LoRA requires 41.9GB.
2. **More pronounced advantage in 3-bit quantization**: In more extreme low-precision settings, the advantage of L4Q's joint optimization becomes even more prominent, widening the gap with other methods.
3. **No compromise on inference speed**: The fully quantized model enjoys inference acceleration comparable to state-of-the-art quantization methods, without the additional overhead of mixed precision.
4. **Scalability**: Traditional QAT encounters OOM at 13B, and QAT-LoRA encounters OOM at 33B, whereas L4Q can scale up to 33B.
5. **LLaMA-1 7B 4-bit MMLU 0-shot**: L4Q reaches 34.9%, outperforming QLoRA (32.8%), QA-LoRA (34.5%), and QAT-LoRA (33.8%).

## Highlights & Insights

- **Elegant design of "merge first, then quantize"**: A seemingly simple adjustment in the execution order (applying quantization on merged weights rather than separately) addresses three major problems: mixed-precision inference, disconnected optimization, and structural constraints on LoRA.
- **Engineering wisdom of gradient reuse**: The weight gradient is reused for computing LoRA parameter gradients immediately after updating the quantization parameters and then released. This preserves the optimization capability of QAT without increasing memory overhead.
- **Practicality of L4Qinit**: It identifies the specific impact of outliers on quantization initialization in LLMs. The standard deviation-based method in LSQ+ is effective on CNNs but fails on LLMs.
- **Deployment-friendly**: The resulting fully quantized model does not require additional LoRA paths and can be directly accelerated using standard quantized inference kernels.

## Limitations & Future Work

1. **Evaluation limited to instruction tuning**: Only evaluated on the Stanford-Alpaca dataset, leaving other fine-tuning scenarios (e.g., code generation, mathematical reasoning) unexplored.
2. **Weight-only quantization support only**: Activation quantization is not covered, which is equally crucial for actual deployment.
3. **Small LoRA rank** (default $r=4$): This may limit performance on tasks requiring substantial adaptation.
4. **Requirement to expand the full weight matrix during training**: Although weight gradients are not stored, the full merged matrix $W_0 + \alpha BA$ must be computed at each step.

## Related Work & Insights

- **PTQ Methods**: GPTQ, OmniQuant, SmoothQuant—require no training but suffer from hard-to-recover accuracy loss.
- **QAT Methods**: LSQ, LSQ+—exhibit excellent accuracy but incur excessive memory overhead, preventing direct application to LLMs.
- **Quantization-Aware PEFT**: QLoRA (PTQ+LoRA, mixed-precision inference), QA-LoRA (constrained LoRA structure), LoftQ (SVD approximation of quantization errors).
- **Parameter-Efficient Fine-Tuning**: LoRA, Prefix Tuning, Adapter—L4Q integrates QAT's optimization capabilities on top of LoRA.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "merge first, then quantize" design and the meticulously crafted gradient paths demonstrate a deep understanding of the problem's essence.
- **Technical Depth**: ⭐⭐⭐⭐ — The derivation of the backpropagation path is complete, and the application of STE to composite function chains shows technical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons covering multiple models (3B-33B), multiple bit-widths (3/4-bit), multiple benchmarks (CSQA/MMLU), as well as memory and inference speed.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical for LLM quantization and deployment, addressing the core pain points of QLoRA-family methods.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A work of outstanding engineering taste, presenting a truly practical and unified solution at the intersection of quantization and fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ACL 2025\] One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)

</div>

<!-- RELATED:END -->
