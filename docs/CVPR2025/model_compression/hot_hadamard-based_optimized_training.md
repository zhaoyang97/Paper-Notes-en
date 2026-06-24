---
title: >-
  [Paper Note] HOT: Hadamard-based Optimized Training
description: >-
  [CVPR 2025][Model Compression][Training Acceleration] The HOT method is proposed. By analyzing the differentiated sensitivity of different gradient paths ($g_x$ for activation gradients and $g_m$ for weight gradients) in backpropagation, Hadamard transform and quantization are selectively applied: $g_x$ uses HT + INT4 to accelerate computation, while $g_m$ uses HLA + INT8 to save activation memory. This achieves a 75% activation memory saving and 2.6x GPU acceleration…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Training Acceleration"
  - "Hadamard Transform"
  - "Gradient Quantization"
  - "Activation Compression"
  - "Mixed Precision"
date: 2026-05-08
content_hash: a0f04bc44b8f6bc9
---

# HOT: Hadamard-based Optimized Training

**Conference**: CVPR 2025  
**arXiv**: [2503.21261](https://arxiv.org/abs/2503.21261)  
**Code**: [https://github.com/sungonuni/HOT](https://github.com/sungonuni/HOT)  
**Area**: Model Compression  
**Keywords**: Training Acceleration, Hadamard Transform, Gradient Quantization, Activation Compression, Mixed Precision

## TL;DR
The HOT method is proposed. By analyzing the differentiated sensitivity of different gradient paths ($g_x$ for activation gradients and $g_m$ for weight gradients) in backpropagation, Hadamard transform and quantization are selectively applied: $g_x$ uses HT + INT4 to accelerate computation, while $g_m$ uses HLA + INT8 to save activation memory. This achieves a 75% activation memory saving and 2.6x GPU acceleration, with only a 0.17% accuracy drop on ImageNet for ViT-B.

## Background & Motivation

**Background**: Large-scale model training faces a triple bottleneck: large weight/optimizer memory, large activation memory, and slow backpropagation computation. Existing methods solve only one of these: LoRA reduces weight memory, LBP-WHT accelerates backpropagation, and gradient quantization reduces memory, but no method addresses all three simultaneously.

**Limitations of Prior Work**: Directly quantizing training gradients severely degrades accuracy—especially the gradient accumulation steps in backpropagation, which are extremely sensitive to precision. Existing methods do not distinguish the sensitivity of different gradient paths; a "one-size-fits-all" quantization strategy forces a compromise between accuracy and efficiency.

**Key Challenge**: The two gradient paths during training ($g_x$ for activation gradient propagation and $g_m$ for weight updates) have completely different tolerances to quantization errors. $g_x$ can tolerate low-precision acceleration due to batch-averaged noise reduction, whereas $g_m$ involves accumulated updates and is highly sensitive to precision. However, existing methods do not differentiate between them.

**Goal**: Design a differentiated gradient optimization strategy that simultaneously achieves memory savings, computation acceleration, and accuracy preservation.

**Key Insight**: Apply differentiated Hadamard transform and quantization processes to the two gradient paths: $g_x$ uses HT + INT4 (primarily for acceleration), and $g_m$ uses HLA (Hadamard Low-rank Approximation) + INT8 (primarily for memory saving).

**Core Idea**: Distinguish the sensitivities of the two gradient paths in backpropagation. Apply INT4 + Hadamard to $g_x$ for accelerated computation, and HLA + INT8 to $g_m$ to compress activation storage, simultaneously saving memory and accelerating training with minimal accuracy overhead.

## Method

### Overall Architecture
In the backpropagation of standard training, two gradient paths are identified: $g_x$ (activation gradient, used to propagate gradients to the previous layer) and $g_m$ (weight gradient, used to update current layer weights). For $g_x$: Hadamard transform followed by INT4 quantization is applied, leveraging batch averaging to reduce quantization noise (prioritizing acceleration). For $g_m$: HLA (50% rank reduction) + INT8 is used to store forward activations (prioritizing memory). It is integrated with LoRA to handle frozen weights.

### Key Designs

1. **$g_x$ Gradient Path: HT + INT4 Quantization**:

    - **Function**: Accelerate matrix multiplications in backpropagation.
    - **Mechanism**: Apply Hadamard transform first to distribute signals in the frequency domain, reducing outlier concentration, followed by INT4 quantization. $g_x$ can be averaged across the batch dimension, meaning that quantization noise naturally diminishes during averaging, thereby tolerating extremely low precision (INT4). A custom CUDA kernel is used to leverage TensorCore's fused INT4$\times$INT8 operations.
    - **Design Motivation**: $g_x$ is a "computation-intensive, precision-insensitive" path; the statistical effect of batch averaging naturally smooths out quantization noise.

2. **$g_m$ Gradient Path: HLA + INT8 Activation Compression (ABC)**:

    - **Function**: Reduce storage memory for forward activations.
    - **Mechanism**: HLA (Hadamard Low-rank Approximation) performs a 50% rank reduction on the activation matrix (reducing the dimension from $d$ to $d/2$), followed by INT8 quantization for storage. Compressed activations are stored during the forward pass and decompressed during the backward pass to calculate $g_m$. Memory compression ratio: $d/2 \times$ INT8 vs $d \times$ FP32 = $1/8$ (only 12.5% of the original memory).
    - **Design Motivation**: $g_m$ involves gradient accumulation (summation over multiple steps), making it highly sensitive to the cumulative effects of quantization errors. HLA retains principal components during rank reduction, and INT8 provides better accumulation precision than INT4.

3. **Layer-wise Quantizer Selection (LQS)**:

    - **Function**: Adaptively select the quantization strategy for each layer.
    - **Mechanism**: Select per-token or per-tensor quantization based on gradient outlier patterns. The MSE outlier ratio of each layer's gradient is calculated; layers exceeding a 50% threshold use per-token quantization (finer-grained), while the rest use per-tensor (faster). In ViTs, attention and fc2 layers tend to require per-token quantization.
    - **Design Motivation**: Different layers exhibit distinct gradient statistical characteristics; a uniform quantization strategy is either too conservative or too aggressive.

### Loss & Training
The standard training loss remains unchanged; HOT only affects the backpropagation computation. When integrated with LoRA: LoRA handles parameter-efficient fine-tuning of frozen weights, while HOT manages computation- and memory-efficient backpropagation.

## Key Experimental Results

### Main Results

| Metric | HOT | FP32 Baseline | Savings |
|------|-----|---------|------|
| Activation Memory | 25% | 100% | **75%** |
| GPU Speed | 2.6× | 1× | **2.6x Speedup** |
| ViT-B Top-1 (ImageNet) | 76.29% | 76.46% | Only 0.17% drop |
| ViT-B Usable Batch Size | 1024 | 256 | **4x** |

### Ablation Study

| Configuration | ResNet50 CIFAR-100 | Description |
|------|-------------------|------|
| FP32 Baseline | 76.46% | Full Precision |
| HT+4bit Q (on $g_x$) | 76.16% | Effective acceleration |
| HLA (on $g_m$) | 76.29% | Effective memory saving |
| Internal-HLA (inside $g_m$) | 76.29% | Optimal $g_m$ strategy |

### Key Findings
- The sensitivity of $g_x$ and $g_m$ to quantization indeed differs: employing INT4 for $g_x$ is feasible, whereas using INT4 for $g_m$ triggers accuracy collapse, validating the necessity of a differentiated strategy.
- The error introduced by HLA's 50% rank reduction can be partially compensated by the accumulation effect of $g_m$.
- LQS reveals that approximately 50% of the layers in ViTs require per-token quantization (attention/fc2), while the remaining 50% can utilize the faster per-tensor quantization.
- When paired with LoRA, HOT further eliminates backpropagation overheads that LoRA alone does not address.

## Highlights & Insights
- **Differentiated Gradient Path Optimization**: Recognizing the differences in sensitivity of $g_x$ and $g_m$ and optimizing them separately is a simple yet profound engineering insight.
- **75% Activation Memory Compression**: Achieves a 4x increase in usable batch size, boosting GPU utilization immensely.
- **Full Training Stack Coverage**: Simultaneously addresses weight memory (LoRA), activation memory (ABC), and computation speed (INT4 TensorCore), representing the first full-stack solution.

## Limitations & Future Work
- Requires custom CUDA kernels, which elevates the deployment barrier.
- Evaluated only on ViTs and ResNets, with LLM training left unaddressed.
- The 50% rank reduction rate of HLA is fixed; adaptive rank selection might yield better results.

## Related Work & Insights
- **vs LoRA**: LoRA reduces weight memory but does not accelerate backpropagation; HOT accelerates backpropagation and reduces activation memory, making them complementary.
- **vs LBP-WHT**: LBP-WHT accelerates backpropagation but does not apply differentiated processing; HOT's divide-and-conquer strategy for $g_x$/$g_m$ is much finer-grained.
- **vs Automatic Mixed Precision (AMP)**: AMP switches between FP16 and FP32; HOT pushes compression down to the INT4/INT8 level.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight into differentiated gradient path optimization is valuable, and the combination of Hadamard and quantization is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ The three-dimensional evaluation across memory, speed, and accuracy is comprehensive, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The sensitivity analysis diagrams and charts are intuitive.
- Value: ⭐⭐⭐⭐ Provides direct value for accelerating practical large-scale model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Expressive yet Efficient Feature Expansion with Adaptive Cross-Hadamard Products](../../ICLR2026/model_compression/expressive_yet_efficient_feature_expansion_with_adaptive_cross-hadamard_products.md)
- [\[CVPR 2025\] Style Quantization for Data-Efficient GAN Training](style_quantization_for_data-efficient_gan_training.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[CVPR 2025\] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)

</div>

<!-- RELATED:END -->
