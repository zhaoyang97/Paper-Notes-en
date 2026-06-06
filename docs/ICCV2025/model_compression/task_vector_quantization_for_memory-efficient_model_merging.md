---
title: >-
  [Paper Note] Task Vector Quantization for Memory-Efficient Model Merging
description: >-
  [ICCV 2025][Model Compression][Model Merging] This paper proposes quantizing task vectors (the difference between fine-tuned and pre-trained weights) rather than the fine-tuned weights themselves. By exploiting the narro…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Model Merging"
  - "Task Vector Quantization"
  - "Low-Bit Storage"
  - "Residual Quantization"
  - "Multi-Task Learning"
date: 2026-05-08
content_hash: 678b00baed8c6e18
---

# Task Vector Quantization for Memory-Efficient Model Merging

**Conference**: ICCV 2025
**arXiv**: [2503.06921](https://arxiv.org/abs/2503.06921)  
**Code**: [https://aim-skku.github.io/TVQ/](https://aim-skku.github.io/TVQ/)  
**Area**: Model Compression
**Keywords**: Model Merging, Task Vector Quantization, Low-Bit Storage, Residual Quantization, Multi-Task Learning

## TL;DR
This paper proposes quantizing task vectors (the difference between fine-tuned and pre-trained weights) rather than the fine-tuned weights themselves. By exploiting the narrower numerical range of task vectors, the method achieves quantization down to 3-bit without accuracy loss. The paper further proposes Residual Task Vector Quantization (RTVQ), which decomposes task vectors into a shared high-precision base vector and low-precision per-task offsets, maintaining or even improving model merging performance while using only 8% of the original storage.

## Background & Motivation
Model merging constructs efficient multi-task models by combining multiple single-task fine-tuned models, avoiding the inference overhead of ensemble methods. Task Arithmetic defines task vectors as $\tau_t = \theta_{ft}^t - \theta_{pre}$ and achieves multi-task merging via linear combination.

However, storing multiple fine-tuned checkpoints incurs substantial storage costs. For example, a single ViT-L/14 checkpoint occupies 1.14 GB, totaling 22.8 GB for 20 tasks. On edge devices such as the NVIDIA Jetson Nano (16 GB storage), this overhead becomes a bottleneck for scaling to larger models and more tasks.

A naive approach is to directly quantize the fine-tuned weights. However, the authors observe a critical phenomenon: **the numerical range of task vectors is an order of magnitude smaller than that of fine-tuned weights**. Since the upper bound of quantization error is $|\epsilon| \leq \frac{\theta_{max} - \theta_{min}}{2(2^b - 1)}$, a narrower range implies smaller quantization error at the same bit-width.

**Core Idea**: Quantize task vectors rather than full weights, exploiting their naturally narrow dynamic range.

## Method

### Overall Architecture
The system comprises two quantization schemes: (1) TVQ — directly quantizing task vectors $\tau_t$; and (2) RTVQ — decomposing task vectors into a shared base vector and task-specific offset vectors, quantized separately. Both are post-training quantization (PTQ) approaches that do not modify any merging method, operating solely on checkpoint storage.

### Key Designs
1. **Task Vector Quantization (TVQ)**

    - **Function**: Replaces stored fine-tuned weights $\theta_{ft}^t$ with task vectors $\tau_t = \theta_{ft}^t - \theta_{pre}$, which are then quantized.
    - **Mechanism**: Applies asymmetric quantization $\tau^q = \text{Round}(\tau / \Delta) + z$, where $\Delta = \frac{\tau_{max} - \tau_{min}}{2^b - 1}$. Since the dynamic range of $\tau$ is far smaller than that of $\theta_{ft}$, the step size $\Delta$ is smaller at the same bit-width, yielding lower quantization error.
    - **Design Motivation**: Experiments confirm that the L2 quantization error of task vectors is significantly lower than that of directly quantized fine-tuned weights across all precisions, with the gap widening below 4-bit.
    - **Key Advantage**: The method only modifies checkpoint storage format and integrates seamlessly into all existing task vector merging frameworks.

2. **Residual Task Vector Quantization (RTVQ)**

    - **Function**: Addresses the sharp performance degradation of TVQ at ultra-low precision (2-bit).
    - **Mechanism**: Decomposes task vectors into two components:
    $\tau_t = \underbrace{(\theta_{ft}^t - \theta_{ft\_avg})}_{\text{Offset Vector}} + \underbrace{(\theta_{ft\_avg} - \theta_{pre})}_{\text{Base Vector}}$
      where $\theta_{ft\_avg} = \frac{1}{T}\sum_t \theta_{ft}^t$ is the mean of all fine-tuned weights. The base vector is stored at higher precision (e.g., 3-bit) and the offset at ultra-low precision (e.g., 2-bit), yielding an effective precision of approximately $2 + 4/8 = 2.5$ bits/task.
    - **Quantization Error Correction**: The base vector is first quantized to obtain $\theta_{ft\_avg\_ec} = Q(\theta_{ft\_avg} - \theta_{pre}) + \theta_{pre}$, and the corrected reference is then used to compute offset vectors, reducing accumulated error.
    - **Design Motivation**: The base vector is shared across all tasks, so its amortized overhead is negligible; the offset vectors, representing deviations from the mean, have smaller magnitudes and are thus well-suited for low-precision quantization.

3. **Quantization Regularization Effect**

    - **Function**: The paper finds that 3-bit quantization sometimes improves merging performance.
    - **Mechanism**: Quantization noise acts as a regularizer, reducing overfitting in a manner analogous to dropout.
    - This phenomenon — where TVQ-INT3 outperforms FP32 — is consistently observed across Task Arithmetic, Ties Merging, and other merging methods.

### Loss & Training
- No additional training is required; the method is a pure PTQ approach.
- Quantization is performed offline at the checkpoint stage, prior to model merging.
- Layer-wise quantization is supported, with scale and zero-point computed independently per layer.

## Key Experimental Results

### Main Results
8-task image classification (ViT-B/32):

| Method | FP32 | FQ-INT4 | TVQ-INT4 | TVQ-INT3 | TVQ-INT2 | RTVQ(3+2) |
|--------|------|---------|----------|----------|----------|-----------|
| Task Arithmetic | 69.2 | 4.2 | 69.1 | **71.2** | 62.1 | 70.2 |
| AdaMerging | 81.8 | 4.5 | 81.5 | **82.0** | 78.1 | **82.8** |
| EMR-Merging | 88.3 | 3.9 | **89.8** | **90.0** | 77.2 | 83.2 |

8-task image classification (ViT-L/14):

| Method | FP32 | TVQ-INT4 | TVQ-INT3 | RTVQ(3+2) |
|--------|------|----------|----------|-----------|
| Task Arithmetic | 84.3 | 84.4 | **84.8** | **84.8** |
| AdaMerging | 90.8 | 90.9 | **91.0** | **90.9** |

### Ablation Study
Dense prediction tasks (NYUv2, ResNet-50):

| Method | Seg(mIoU)↑ | Depth(RelErr)↓ | Normal(MAE)↓ |
|--------|-----------|----------------|--------------|
| Task Arith. FP32 | 31.6 | 24.0 | 30.6 |
| Task Arith. TVQ-4 | 31.5 | 24.0 | 30.6 |
| Task Arith. TVQ-2 | 36.4 | 26.2 | 36.1 |
| Task Arith. RTVQ | 36.1 | 24.6 | 32.6 |

### Key Findings
- **FQ-INT4 fails completely**: Directly quantizing fine-tuned weights to 4-bit reduces accuracy to approximately 4% (random-chance level), demonstrating the necessity of task vector quantization.
- **TVQ-INT3 consistently outperforms FP32**: The regularization effect of quantization noise appears consistently across multiple merging methods.
- **RTVQ effectively mitigates 2-bit degradation**: When TVQ-INT2 suffers significant performance drops, RTVQ (effective 2.375 bit) maintains performance close to FP32.
- **RTVQ becomes more advantageous as task count increases**: At 20 tasks, the effective bit-width reduces to only 2.15 bit/task, with stronger amortization of the base vector overhead.
- Storage requires only 8% of the original FP32 checkpoint size.

## Highlights & Insights
- **Simple yet powerful observation**: The narrow dynamic range of task vectors relative to fine-tuned weights, though straightforward to observe, underpins the entire method's effectiveness.
- **Seamless compatibility with existing methods**: As a purely checkpoint-side operation, TVQ integrates into Task Arithmetic, Ties Merging, AdaMerging, and all other merging frameworks at zero modification cost.
- **Elegant residual decomposition**: The shared base vector plus task-specific offset decomposition naturally exploits inter-task redundancy.
- **Unexpected regularization finding**: The phenomenon of 3-bit quantization improving performance carries theoretical research value.

## Limitations & Future Work
- The RTVQ base vector depends on the mean of all tasks; adding new tasks requires recomputation.
- Dense prediction tasks are more sensitive to quantization, particularly methods relying on task-specific masks such as EMR-Merging.
- Only the PTQ paradigm is evaluated; whether quantization-aware training (QAT) could further improve performance remains unexplored.
- Validation on NLP tasks is limited, with the experiments primarily focused on vision domains.

## Related Work & Insights
- Unlike storage optimization methods such as TALL Mask and TSV-C, TVQ is more general and requires no additional task-specific structures.
- The discovery of quantization regularization effects may inspire new regularization strategies for model merging.
- The residual decomposition idea is extensible to other scenarios requiring storage of multiple model variants (e.g., storing multiple LoRA adapters).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core observation is simple but profound; the RTVQ design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers classification, dense prediction, and NLP; evaluates 8/14/20-task scales across multiple merging methods.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear with persuasive figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical — preserving performance at 8% storage has significant implications for resource-constrained deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](msq_memory-efficient_bit_sparsification_quantization.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](../../NeurIPS2025/model_compression/accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)

</div>

<!-- RELATED:END -->
