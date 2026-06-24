---
title: >-
  [Paper Note] GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration
description: >-
  [ICML 2025][Model Compression][Post-Training Quantization] GPTAQ proposes a tuning-free quantization method featuring asymmetric calibration. By aligning the output of quantized layers with the exact output of the full-precision model (instead of just the current layer's output) and deriving a closed-form solution based on the Optimal Brain Compression (OBC) framework to jointly minimize both quantization and cumulative asymmetric errors, GPTAQ significantly improves the perf…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Asymmetric Calibration"
  - "Optimal Brain Compression"
  - "GPTQ Improvement"
  - "Low-bit Quantization"
date: 2026-05-08
content_hash: 5e70e3f6a6a72e92
---

# GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration

**Conference**: ICML 2025  
**arXiv**: [2504.02692](https://arxiv.org/abs/2504.02692)  
**Code**: [https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ)  
**Area**: Model Compression / Quantization  
**Keywords**: Post-Training Quantization, Asymmetric Calibration, Optimal Brain Compression, GPTQ Improvement, Low-bit Quantization

## TL;DR
GPTAQ proposes a tuning-free quantization method featuring asymmetric calibration. By aligning the output of quantized layers with the exact output of the full-precision model (instead of just the current layer's output) and deriving a closed-form solution based on the Optimal Brain Compression (OBC) framework to jointly minimize both quantization and cumulative asymmetric errors, GPTAQ significantly improves the performance of GPTQ in low-bit quantization while adding only about 20 lines of code.

## Background & Motivation

**Background**: Post-Training Quantization (PTQ) is one of the mainstream methods for compressing large-scale Transformer models. GPTQ is the most widely used method in this area, which calibrates quantization parameters layer-by-layer and independently based on the Optimal Brain Compression (OBC) framework.

**Limitations of Prior Work**: The core assumption of GPTQ is layer-wise independent calibration—the quantization of each layer only considers minimizing its own output error. However, this strategy overlooks a critical issue: quantization error from preceding layers continuously accumulates and propagates to subsequent layers. This cumulative error becomes particularly severe as the quantization bit-width decreases (e.g., 3-bit or 2-bit).

**Key Challenge**: Layer-wise independent calibration is oblivious to the changes in error distribution introduced by previously quantized layers. In particular, when the input distribution of a quantized layer has already shifted, merely matching the original input-output relationship of that layer may conversely amplify the cumulative error.

**Goal**: How to enable the quantization calibration of each layer to perceive and compensate for the cumulative quantization error of preceding layers, without introducing finetuning overhead?

**Key Insight**: Change the output target of the quantized layer from "matching the current full-precision layer output" to "matching the corresponding exact output of that layer in the full-precision model," thereby establishing asymmetric calibration—the quantized layer receives quantized upstream inputs (asymmetric input) but is required to produce the output of the full-precision model.

**Core Idea**: Derive a closed-form solution for asymmetric calibration within the Optimal Brain Compression framework, explicitly minimizing a joint objective of quantization error and asymmetric cumulative error, and achieving high computational efficiency through channel parallelization, neuron decomposition, and Cholesky reformulation.

## Method

### Overall Architecture
The overall workflow of GPTAQ is similar to GPTQ, performing post-training quantization sequentially layer by layer. The key difference lies in the change of the calibration target:
- **Input side**: Employs the actual output from the quantized model (with preceding layers already quantized) as the input for the current layer.
- **Output side**: Targets the exact output of the corresponding layer in the full-precision model.
- This setup where "inputs originate from the quantized model and outputs align with the full-precision model" is termed "asymmetric calibration."

### Key Designs

1. **Asymmetric Calibration**:

    - In traditional GPTQ, the quantization objective for the $l$-th layer is to minimize $\|W_l X_l - Q(W_l) X_l\|^2$, where $X_l$ is the full-precision input.
    - GPTAQ modifies the objective to minimize $\|W_l X_l - Q(W_l) \hat{X}_l\|^2$, where $\hat{X}_l$ represents the actual input after passing through preceding quantized layers.
    - Thus, the quantization of each layer not only minimizes its own quantization error but also explicitly compensates for cumulative upstream errors.
    - Design Motivation: In low-bit quantization (2-3 bits), cumulative errors across layers are often larger than the quantization error of a single layer, making explicit modeling during calibration crucial.

2. **OBC-based Closed-form Solution**:

    - Reformulates the asymmetric calibration problem into a constrained quadratic optimization problem.
    - Derives a new closed-form update formula using the OBC framework, which simultaneously covers both the quantization error term and the asymmetric error term.
    - The key equation format is: $$\delta_q = \arg\min_\delta \left[\delta^T H \delta + \lambda \cdot \text{AsymErr}\right]$$
    - where $H$ is the Hessian matrix and $\text{AsymErr}$ reflects the extra error induced by input distribution shift.
    - Design Motivation: The closed-form solution avoids the high computational overhead of iterative optimization, maintaining GPTQ-level computational efficiency.

3. **High-efficiency Parallelization**:

    - **Channel Parallelization**: Decouples the quantization computation of different output channels, supporting GPU parallelization.
    - **Neuron Decomposition**: Decomposes the matrix operations of fully connected layers into smaller independent blocks to mitigate memory bottlenecks.
    - **Cholesky Reformulation for Matrix Fusion**: Utilizes Cholesky factorization to fuse multiple matrix operations into a single computation, reducing redundant operations.
    - Design Motivation: Although asymmetric calibration introduces optional error terms, the actual computational overhead can be maintained at a level comparable to GPTQ through the XML-level parallelization techniques.

### Loss & Training
GPTAQ is a finetuning-free method that does not involve gradient updates. Its "training" process is the sequential, layer-by-layer execution of quantization calibration: for each layer, it collects the actual outputs of preceding quantized layers and the target outputs of the full-precision model, and then computes the quantized weights of that layer in a single step using the closed-form solution. The entire process requires only a small amount of calibration data (typically 128-256 samples).

## Key Experimental Results

### Main Results

| Model | Method | Bit-width | GSM8K (flex-extract) | ARC-Challenge |
|------|------|--------|---------------------|---------------|
| LLaMA-3.1-8B-Instruct | GPTQ v1 | 4-bit | 39.95% | 50.00% |
| LLaMA-3.1-8B-Instruct | GPTAQ v2 | 4-bit | **76.01%** | **50.34%** |
| EVA-02 (ViT) | GPTQ | 4-bit | - | See paper |
| EVA-02 (ViT) | GPTAQ | 4-bit | - | Significant improvement |
| LLaMA-405B | GPTAQ | 4-bit | Quantizable on a single GPU | - |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Symmetric Calibration (GPTQ) | GSM8K 39.95% | Baseline, layer-wise independent |
| Asymmetric Calibration (GPTAQ) | GSM8K 76.01% | ~36% absolute gain |
| + Channel Parallelization | Speed on par with GPTQ | No accuracy loss |
| + Cholesky Fusion | Memory further reduced | Supports ultra-large models |

### Key Findings
- Asymmetric calibration exhibits particularly significant gains at low bit-widths (3-bit, 2-bit), because cumulative error across layers is more severe in low-bit scenarios.
- The performance gain of GPTAQ on GSM8K is highly remarkable (from ~40% to ~76% ), indicating that mathematical reasoning tasks are extremely sensitive to cumulative quantization errors.
- The method is equally effective on Vision Transformers (EVA-02), demonstrating it is not restricted to language models.
- A 405B parameter model can be quantized on a single GPU, showcasing outstanding scalability.

## Highlights & Insights
- **Minimal Implementation**: It adds only about 20 lines of code over GPTQ. Extremely minimal engineering changes yield huge performance gains, representing an exceptionally strong academic contribution.
- **Dual Excellence in Theory and Engineering**: It features both rigorous theoretical derivation within the OBC framework and highly efficient, practical parallelization schemes.
- **Core Insight**: The accumulation of quantization error is far more severe than single-layer errors. This indicates that in any step-by-step model compression method, close attention must be paid to error propagation.
- It has already been integrated into the GPTQModel library and is ready for use.

## Limitations & Future Work
- Asymmetric calibration requires maintaining intermediate outputs of both the full-precision model and the quantized model simultaneously, leading to slightly higher memory consumption than GPTQ.
- The paper primarily focuses on weight quantization, with limited discussion on activation quantization.
- The impact of calibration data selection on performance lacks in-depth analysis.
- Future work could explore closer integration with rotation-based quantization (such as QuaRot, SpinQuant) to investigate prospective combined effects.

## Related Work & Insights
- **vs GPTQ**: GPTAQ serves as a direct extension/upgrade of GPTQ. The core improvement is the transition from symmetric calibration to asymmetric calibration, with both theory and experiments demonstrating substantial advantages.
- **vs QuaRot/SpinQuant**: These methods reduce quantization difficulty using rotation transformations, which represents an orthogonal technical route to GPTAQ. The paper also showcases performance when combined as QuaRot+GPTAQ and SpinQuant+GPTAQ.
- **vs QLoRA/LoftQ**: These methods compensate for quantization losses by finetuning adapters, whereas GPTAQ requires absolutely no finetuning.

## Rating
- Novelty: ⭐⭐⭐⭐ While the idea of asymmetric calibration is natural, deriving a closed-form solution from the OBC framework and implementing it efficiently is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various bit-widths across both LLMs and ViTs, though more detailed ablation studies could be included.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the selling point of "only 20 lines of code" is highly compelling.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value. Having been integrated into industrial-grade libraries, it serves as a definitive upgrade and replacement for GPTQ.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gradient-Aligned Calibration for Post-Training Quantization of Diffusion Models](../../ICLR2026/model_compression/gradient-aligned_calibration_for_post-training_quantization_of_diffusion_models.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](../../ICCV2025/model_compression/ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[ACL 2026\] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](../../ACL2026/model_compression/improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)

</div>

<!-- RELATED:END -->
