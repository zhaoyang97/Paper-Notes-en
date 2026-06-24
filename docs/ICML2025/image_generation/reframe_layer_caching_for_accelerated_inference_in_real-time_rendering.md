---
title: >-
  [Paper Note] ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering
description: >-
  [ICML 2025][Image Generation][Layer Caching] Extends the intermediate layer caching technique (DeepCache) from diffusion models to U-Net/U-Net++ networks in real-time rendering pipelines, achieving an average of 1.4× inference speedup with negligible image quality degradation through a frame-difference adaptive caching strategy.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Layer Caching"
  - "Real-Time Rendering"
  - "U-Net"
  - "Temporal Consistency"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 4a003e8a1f3a0784
---

# ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering

**Conference**: ICML 2025  
**arXiv**: [2506.13814](https://arxiv.org/abs/2506.13814)  
**Code**: [https://ubc-aamodt-group.github.io/reframe-layer-caching/](https://ubc-aamodt-group.github.io/reframe-layer-caching/)  
**Area**: Image Generation  
**Keywords**: Layer Caching, Real-Time Rendering, U-Net, Temporal Consistency, Inference Acceleration

## TL;DR
Extends the intermediate layer caching technique (DeepCache) from diffusion models to U-Net/U-Net++ networks in real-time rendering pipelines, achieving an average of 1.4× inference speedup with negligible image quality degradation through a frame-difference adaptive caching strategy.

## Background & Motivation
**Background**: Real-time rendering (e.g., DLSS 4.0) heavily relies on U-Net-style neural networks for tasks such as denoising, super-resolution, and frame extrapolation, with network inference accounting for a significant portion of the rendering pipeline.

**Limitations of Prior Work**: (a) High temporal correlation exists between rendered frames, yet full inference is performed for every frame; (b) methods utilizing inter-frame differences such as DeltaCNN are difficult to accelerate on current GPUs due to sparse computation; (c) DeepCache is designed solely for multi-step inference in diffusion models.

**Key Challenge**: In rendering, every frame must individually yield high-quality outputs (unlike diffusion models, which can tolerate approximations in intermediate steps), even though inter-frame features change slowly.

**Goal**: How to leverage temporal redundancy under the strict quality constraints of real-time rendering?

**Key Insight**: Cache deep intermediate features of U-Net to skip most of the encoder-decoder computation, recomputing only the shallow layers (which are sensitive to changes in new inputs).

**Core Idea**: Slow temporal variations in features during real-time rendering $\rightarrow$ cache deep features $\rightarrow$ adaptive refresh strategy $\rightarrow$ training-free acceleration.

## Method

### Overall Architecture
In U-Net, the deep feature $C_t$ is cached during full inference; subsequent frames only compute the first layer $X^0$ and the final layer $X^n$, replacing intermediate layers with $C_t$. For U-Net++, all skip connection branches except the first layer are cached.

### Key Designs

1. **Layer Caching Mechanism**:

    - **Function**: Skip deep computation of the encoder-decoder
    - **Mechanism**: Cache $C_t = X^{n-1}(\text{concat}(\ldots))$, and for subsequent frames, compute $O = X^n(\text{concat}(C_t, X^0(I)))$
    - **Design Motivation**: Deep features capture high-level semantics and vary the slowest across frames, while shallow features capture low-level details and are most sensitive to new inputs

2. **Frame-Difference Adaptive Strategy (Frame Deltas)**:

    - **Function**: Determine whether to refresh the cache based on the degree of change in the input
    - **Mechanism**: Calculate the SMAPE between the current input and the cached frame, refreshing the cache if it exceeds a threshold $\tau$. Two tiers are provided: high sensitivity (Delta_H) and low sensitivity (Delta_L)
    - **Design Motivation**: A fixed Every-N strategy cannot adapt to unpredictable scene changes in rendering (e.g., rapid camera movement vs. static scenes)

3. **Motion Vector Threshold Strategy**:

    - **Function**: Utilize existing motion vectors from the rendering pipeline to determine whether to refresh
    - **Mechanism**: Refresh the cache when the average motion exceeds a threshold $\tau$
    - **Design Motivation**: No additional storage overhead (as motion vectors are already pre-existing bi-products of the rendering pipeline)

### Loss & Training
- **Fully training-free**: Network weights are not modified; caching logic is added solely during inference.
- Can be combined with orthogonal techniques such as quantization and pruning.

## Key Experimental Results

### Main Results

| Task | Scene | Strategy | Speedup ↑ | FLIP ↓ | SSIM ↑ |
|------|------|------|---------|--------|--------|
| Frame Extrapolation | Sun Temple | Delta_H | 1.42× | 0.017 | 0.994 |
| Frame Extrapolation | Sun Temple | Delta_L | 1.72× | 0.033 | 0.984 |
| Super-Resolution | Sun Temple | Delta_H | 1.30× | 0.049 | 0.970 |
| Super-Resolution | Sun Temple | Delta_L | 1.85× | 0.118 | 0.930 |
| Image Synthesis | Garden Chair | Delta_H | 1.05× | 0.001 | 1.000 |

### Ablation Study

| Strategy | Avg. Frame Skip Rate | Avg. Speedup | FLIP | Description |
|------|-----------|---------|------|------|
| Every-2 | 50% | ~1.4× | Medium | Fixed interval |
| Every-4 | 75% | ~1.7× | High | Quality drops drastically during fast motion |
| Delta_H | 30-50% | 1.1-1.4× | **Lowest** | Adaptive, preserving quality |
| Delta_L | 60-80% | 1.5-1.9× | Low | Adaptive balance |

### Key Findings
- FLIP < 0.05 is considered an acceptable quality loss in the rendering domain (reference values range from 0.05 to 0.28).
- Frame extrapolation tasks benefit the most (as continuous frame transitions are smoothest), followed by super-resolution.
- Adaptive strategies prevent the drastic image quality drop that fixed strategies experience during fast camera movements.

## Highlights & Insights
- **Training-free + Universality**: No network retraining is required; it can be applied to any encoder-decoder network possessing skip connections.
- **Extension to U-Net++**: For the first time, the caching technology is expanded from U-Net to U-Net++.
- **Reinvesting Saved Computation into Rendering**: Saved inference time can be utilized to increase the ray-tracing sample rate, which may overall improve rendering quality.

## Related Work & Insights
- **vs DeepCache**: DeepCache targets the fixed-step inference of diffusion models, while ReFrame addresses the single-frame output requirement of rendering; adaptive refreshing is the key difference.
- **vs DeltaCNN**: DeltaCNN utilizes pixel-level differences to sparsify computations, but existing GPU hardware struggles to accelerate sparse operations; the layer caching in ReFrame is fully compatible with existing hardware.
- **vs DLSS**: DLSS natively utilizes U-Net, and ReFrame can serve as a component to further accelerate it.
- The caching strategy and DeltaCNN can theoretically be combined: cached frames only compute shallow layers, which are then sparsified internally using deltas.

## Limitations & Future Work
- Evaluated on an RTX 2080 Ti; verification of benefits on the latest GPUs (e.g., RTX 4090/5090) is lacking.
- The thresholds for the adaptive strategy require task-specific tuning, lacking an automated determination method.
- Only three rendering tasks and five scenes were evaluated, which limits the scope.
- Handling of rapid scene cuts (e.g., teleportation in games) is not discussed.
- Challenges regarding cache consistency in multiplayer synchronized rendering are not considered.

## Rating
- Novelty: ⭐⭐⭐ The transfer of DeepCache to rendering is relatively direct; the adaptive strategy is an incremental contribution.
- Experimental Thoroughness: ⭐⭐⭐ The number of tasks and scenes is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear and systematic, with rich illustrations.
- Value: ⭐⭐⭐⭐ Holds practical significance for optimizing rendering pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[NeurIPS 2025\] Real-Time Execution of Action Chunking Flow Policies](../../NeurIPS2025/image_generation/real-time_execution_of_action_chunking_flow_policies.md)
- [\[CVPR 2026\] SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching](../../CVPR2026/image_generation/sencache_accelerating_diffusion_model_inference_via_sensitivity-aware_caching.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](../../CVPR2025/image_generation/semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)

</div>

<!-- RELATED:END -->
