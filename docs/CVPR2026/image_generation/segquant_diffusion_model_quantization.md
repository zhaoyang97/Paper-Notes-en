---
title: >-
  [Paper Note] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Post-training quantization] This paper proposes SegQuant, a deployment-oriented post-training quantization (PTQ) framework for diffusion models. It achieves cross-architecture, high-fidelity W8A8/W4A8 quantization on SD3.5, FLUX, and SDXL via semantics-aware segmented quantization (SegLinear) based on static computational graph analysis and hardware-native dual-scale polarity-preserving quantization (DualScale), while maintaining compatibility with industrial inference engines such as TensorRT.
tags:
  - CVPR 2026
  - Image Generation
  - Post-training quantization
  - semantics-aware segmentation
  - polarity preservation
  - deployment-friendly
  - DiT quantization
date: 2026-05-08
content_hash: 9297b94058ce4f2f
---

# SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models

**Conference**: CVPR 2026  
**arXiv**: [2507.14811](https://arxiv.org/abs/2507.14811)  
**Code**: [https://github.com/OptiSys-ZJU/segquant](https://github.com/OptiSys-ZJU/segquant)  
**Area**: Image Generation  
**Keywords**: Post-training quantization, semantics-aware segmentation, polarity preservation, deployment-friendly, DiT quantization  

## TL;DR
This paper proposes SegQuant, a deployment-oriented post-training quantization (PTQ) framework for diffusion models. It achieves cross-architecture, high-fidelity W8A8/W4A8 quantization on SD3.5, FLUX, and SDXL via semantics-aware segmented quantization (SegLinear) based on static computational graph analysis and hardware-native dual-scale polarity-preserving quantization (DualScale), while maintaining compatibility with industrial inference engines such as TensorRT.

## Background & Motivation
Diffusion models are computationally intensive, and quantization is an effective means of reducing inference overhead. PTQ is the most practical approach as it requires no retraining; however, existing PTQ methods for diffusion models suffer from a **Compiler Gap**:

1. **Architecture-specific heuristics**: Methods such as Q-Diffusion manually craft splitting rules for UNet skip-connections, which do not generalize to DiT architectures.
2. **Runtime dynamic dependencies**: Methods such as PTQ4DiT rely on timestep-varying activation distributions (salient channels), which are incompatible with modern static-graph compilers.
3. **Polarity asymmetry in activations**: DiT architectures employ SiLU/GELU activation functions whose output distributions are asymmetric—the negative range is narrow yet semantically rich—causing standard quantization to over-compress negative values and lose high-frequency detail.

## Core Problem
How to design a high-performance, compiler-native, and generalizable quantization framework that addresses semantic heterogeneity and activation polarity asymmetry in diffusion models?

## Method

### Overall Architecture
SegQuant is a modular, top-down framework comprising four interchangeable components: Optimizer (e.g., SmoothQuant/SVDQuant), Calibrator (e.g., GPTQ/AMax), SegLinear (semantic segmentation), and DualScale (polarity preservation). Users may freely substitute the Optimizer and Calibrator, while SegLinear and DualScale serve as core enhancement modules.

### Key Designs

1. **SegLinear (Semantics-Aware Segmented Quantization)**: The core observation is that linear layers in modules such as AdaNorm and TimeEmbedding in DiT process semantically heterogeneous inputs—e.g., distinct semantic branches after chunk/split operations (temporal information vs. latent representations). SegLinear employs `torch.fx` static computational graph analysis to automatically detect chunk/split (output segmentation) and concat/stack/reshape (input segmentation) patterns, then splits weight matrices and activations along semantic boundaries for independent quantization. For example, the DiT.0.norm1 layer in SD3.5 is automatically inferred to have 6 chunks, reducing the F-norm quantization error from 0.70 to 0.54. A key advantage is that the approach relies entirely on static graph analysis without runtime data, making it natively compatible with AI compilers.

2. **DualScale (Dual-Scale Polarity-Preserving Quantization)**: This module specifically addresses the positive–negative polarity asymmetry of SiLU/GELU activations. The activation matrix is decomposed into a positive part $X^+ = \max(X, 0)$ and a negative part $X^- = \min(X, 0)$, each quantized with an independent scale $s^+$ or $s^-$, followed by separate matrix multiplications that are linearly recombined: $Y \approx s^+ s_w (X^+ W) + s^- s_w (X^- W)$. Although this appears to require two GEMMs, both computations are executed in parallel within a single kernel launch via CUTLASS BatchedGEMM, with combination performed in a fused epilogue. Compared to asymmetric quantization, DualScale requires no zero-point correction, making it simpler and more efficient.

3. **Static Graph Pattern Detection Algorithm**: `torch.fx` symbolic tracing combined with shape propagation is used to obtain the full computational graph. The algorithm automatically traverses the graph to identify weight-segmentation patterns (Linear → chunk/split) and input-segmentation patterns (concat/stack/reshape → Linear), inferring segment sizes directly from operator arguments.

### Loss & Training
- Entirely training-free PTQ method.
- SmoothQuant migration strength $\alpha$ is searched per layer over [0.0, 1.0] with step 0.1, selecting the value that minimizes MSE.
- Calibration sets: 256 images (SD3/SDXL), 64 images (FLUX 8-bit), 32 images (FLUX 4-bit).
- 8-bit: per-tensor scheme; 4-bit: per-channel weights + per-token dynamic activations.

## Key Experimental Results

### MJHQ-30K Main Results (SD3.5 DiT W8A8)

| Method | FID ↓ | Image Reward ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|--------|-------|---------------|---------|--------|--------|
| PTQD | Poor | Poor | Poor | Poor | Poor |
| PTQ4DiT | Moderate | Moderate | Moderate | Moderate | Moderate |
| Smooth+ | Moderate | Moderate | Moderate | Moderate | Moderate |
| **SegQuant-G** | **Best** | **Best** | **Best** | **Best** | **Best** |

SegQuant consistently outperforms baselines across three architectures: SD3.5, FLUX-DiT, and SDXL-UNet.

### Ablation Study (SD3.5 W8A8, MJHQ)

| Method | FID ↓ | IR ↑ | LPIPS ↓ | PSNR ↑ |
|--------|-------|------|---------|--------|
| Baseline (SmoothQuant) | 23.35 | 0.877 | 0.419 | 11.93 |
| + SegLinear | 23.36 | 0.899 | 0.395 | 12.03 |
| + DualScale | 22.61 | 0.909 | 0.401 | 12.14 |
| + Both | **22.54** | **0.952** | **0.377** | **12.50** |

The two modules are complementary, and their combination yields the best performance.

### Per-Layer Error Analysis

| Layer | Method | w/o SegLinear | w/ SegLinear |
|-------|--------|--------------|-------------|
| DiT.0.norm1 | GPTQ | 0.835 | **0.444** |
| DiT.11.norm1_context | GPTQ | 3.017 | **1.761** |

SegLinear substantially reduces quantization error in sensitive layers.

### Efficiency Analysis
- SegQuant applies DualScale to only 12%–29% of layers, incurring a memory overhead of merely 3.4–5.6 MB.
- The DualScale kernel achieves faster inference than the AdaNorm quantization scheme in PTQ4ViT.
- Calibration time: approximately 2.5 hours for SD3 (single L20 GPU, 256 calibration images).

## Highlights & Insights
- **Compiler-friendly**: Based entirely on static graph analysis with no runtime data dependency, enabling direct integration into deployment pipelines such as TensorRT.
- **Cross-architecture generalizability**: A single framework supports both DiT (SD3.5, FLUX) and UNet (SDXL) without manual rules.
- **Automated semantic segmentation**: While Q-Diffusion requires manually crafted rules for UNet skip-connections, SegLinear automatically discovers all layers requiring segmentation via graph analysis.
- **Hardware-native DualScale**: BatchedGEMM fuses two GEMMs into a single kernel launch, avoiding custom data formats or non-standard kernels.

## Limitations & Future Work
- **Calibration cost**: Hyperparameter search (α step 0.1) requires ~25 hours, which, although a one-time cost, remains substantial.
- **FLUX memory constraints**: A 48 GB GPU requires a swap-in/swap-out strategy during calibration.
- **Video diffusion models not evaluated**: Although the method is theoretically general, it has not been validated on video generation models.
- **Remaining quality gap under 4-bit**: W4A8 outperforms baselines but still exhibits a noticeable gap relative to FP16.
- Specific numerical values in some quantitative results are partially unreadable due to HTML rendering issues (`\cellcolor{lightgray}` annotations).

## Related Work & Insights
- **vs. Q-Diffusion**: Q-Diffusion manually designs UNet skip-connection splitting rules, constituting architecture-specific heuristics; SegLinear generalizes this via automatic computational graph analysis.
- **vs. PTQ4DiT**: PTQ4DiT relies on runtime timestep-varying activation statistics and is incompatible with static-graph compilers; SegQuant operates entirely on static structure.
- **vs. SVDQuant**: SVDQuant uses low-rank decomposition to handle outliers; SegQuant can integrate it as an Optimizer component (SVDQuant is indeed used as the Optimizer in the 4-bit experiments).

The semantic graph analysis approach of SegLinear is potentially extensible to VLM quantization, where text and visual tokens exhibit analogous semantic heterogeneity. The polarity decomposition idea in DualScale may also benefit other large models using SiLU/GELU, such as LLMs. The overall modular design (Optimizer + Calibrator + enhancement modules) provides a valuable reference for future quantization framework design.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The designs of semantic segmentation and polarity preservation are insightful, though the overall contribution is engineering-oriented framework innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three architectures, three datasets, multiple precision settings, detailed ablations, and practical efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated, and consistent terminology.
- **Value**: ⭐⭐⭐⭐ — Addresses practical deployment pain points in diffusion model quantization with strong applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_score_function_generalization_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)

<!-- RELATED:END -->
