---
title: >-
  [Paper Note] RenderFlow: Single-Step Neural Rendering via Flow Matching
description: >-
  [CVPR 2026][Image Generation][Neural rendering] RenderFlow recasts neural rendering as a single-step conditional flow matching problem from albedo to full-illumination images. Using G-buffers as conditions and a pretrained video DiT as the backbone, it achieves deterministic rendering more than 10× faster than diffusion-based methods (~0.19 s/frame). An optional sparse keyframe guidance module further improves physical accuracy, and inverse rendering is supported via a frozen backbone with lightweight adapters.
tags:
  - CVPR 2026
  - Image Generation
  - Neural rendering
  - flow matching
  - single-step inference
  - G-buffer
  - keyframe guidance
date: 2026-05-08
content_hash: dea3c03c1c4f0f4c
---

# RenderFlow: Single-Step Neural Rendering via Flow Matching

**Conference**: CVPR 2026
**arXiv**: [2601.06928](https://arxiv.org/abs/2601.06928)
**Code**: Unavailable (Disney Research internal project)
**Area**: Diffusion Models / Image Generation / 3D Vision
**Keywords**: Neural rendering, flow matching, single-step inference, G-buffer, keyframe guidance

## TL;DR
RenderFlow recasts neural rendering as a single-step conditional flow matching problem from albedo to full-illumination images. Using G-buffers as conditions and a pretrained video DiT as the backbone, it achieves deterministic rendering more than 10× faster than diffusion-based methods (~0.19 s/frame). An optional sparse keyframe guidance module further improves physical accuracy, and inverse rendering is supported via a frozen backbone with lightweight adapters.

## Background & Motivation

1. **Background**: Physically-based rendering (PBR) via Monte Carlo path tracing is the gold standard for offline rendering but incurs prohibitive computational cost. Recent diffusion-based neural rendering methods (e.g., RGB-X, DiffusionRenderer) leverage G-buffers as conditions to synthesize photorealistic images and have demonstrated strong visual quality.

2. **Limitations of Prior Work**: (a) The iterative denoising of diffusion models requires 20–50 network evaluations, introducing latency incompatible with interactive applications; (b) the stochasticity of diffusion sampling yields insufficient physical accuracy and temporal flickering, failing to meet production-grade rendering standards. Both issues stem fundamentally from the "generate from noise" paradigm of diffusion models.

3. **Key Challenge**: There exists a fundamental conflict between the generative capacity of diffusion models and the need for real-time, deterministic rendering — high-frequency detail synthesis is required, yet iterative sampling and stochasticity are unacceptable.

4. **Goal**: (a) Achieve single-step, deterministic neural rendering; (b) improve physical accuracy without relying on explicit scene geometry or light-transport simulation; (c) reuse the same backbone for both forward and inverse rendering.

5. **Key Insight**: The key insight is that rendering can be understood as a *residual flow* learning problem from albedo (diffuse base color) to full-illumination images. Since albedo already encodes low-frequency color information, the model only needs to learn to add high-frequency effects such as lighting, shadows, and reflections. Replacing noise with albedo as the flow's starting point preserves geometric integrity.

6. **Core Idea**: Flow matching is used to learn a deterministic velocity field from albedo to full-illumination images, conditioned on G-buffers and built on a pretrained video DiT backbone. Single-step, high-fidelity rendering is achieved within a bridge matching framework.

## Method

### Overall Architecture
The inputs consist of a set of G-buffer attributes — albedo (base color), normal, depth, material (roughness / metalness / specular), and an environment map (global illumination). Albedo replaces noise as the flow's starting point and is encoded into a latent $\mathbf{z}_0$ via a VAE; the target is the ground-truth path-traced image $\mathbf{z}_1$. The model learns a velocity field $v_\theta$ that directly maps $\mathbf{z}_0$ to $\mathbf{z}_1$, producing a complete rendering result in a single inference step. Optional sparse keyframes provide physical-accuracy anchors via a cross-attention adapter.

### Key Designs

1. **Albedo-to-Render Flow Matching**
    - **Function**: Models rendering as deterministic single-step flow generation.
    - **Mechanism**: Trained under a bridge matching framework. During training, a timestep $t \in [0,1]$ is sampled and the interpolation $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1 + \sigma\sqrt{t(1-t)}\epsilon$ (with $\sigma=0.005$) is computed. The model learns to predict the velocity field $v_\theta(\mathbf{z}_t, t)$ to approximate the target direction $\frac{\mathbf{z}_1 - \mathbf{z}_t}{1-t}$. At inference, a single step $\hat{\mathbf{z}}_1 = \mathbf{z}_t + v_\theta(\mathbf{z}_t, t)(1-t)$ completes the rendering. Training uses a 4-step SDE schedule while inference uses a single step, empirically avoiding multi-step error accumulation.
    - **Design Motivation**: Replacing Gaussian noise with albedo as the flow's starting point offers two key advantages: (a) low-frequency color and geometry information is preserved, so the model only needs to synthesize high-frequency illumination details; (b) the flow begins from an informative point, making the ODE trajectory shorter and straighter, enabling high-accuracy single-step arrival at the target. The small noise perturbation in SDE training improves robustness.

2. **G-Buffer Condition Injection**
    - **Function**: Supplies scene geometry and material information to the rendering network.
    - **Mechanism**: The backbone is based on the Wan2.1 video DiT. The albedo latent is converted to render tokens via an input embedder. The remaining G-buffers (normal, depth, material) are encoded by the same VAE and processed through a dedicated attribute embedder; because they are spatially aligned, they are added element-wise to the render tokens (following the VACE architecture). The environment map is first rotated into camera view space and Reinhard tone-mapped to produce an LDR image, then encoded by the VAE and injected into each Transformer block via adaptive instance normalization (AdaIN), which predicts scale $\gamma$ and shift $\beta$ to modulate the render features.
    - **Design Motivation**: G-buffers and render tokens are spatially aligned, making element-wise addition the most efficient injection strategy. The environment map is global and not spatially aligned, making AdaIN modulation more appropriate. Rotating the environment map into camera space allows the network to implicitly learn directional illumination without explicit directional encoding.

3. **Sparse Keyframe Guidance (Keyframe Adapter)**
    - **Function**: Uses a small number of offline path-traced reference frames to improve physical accuracy and temporal stability.
    - **Mechanism**: A cross-attention branch is added in parallel to each self-attention layer, with keyframe tokens serving as keys/values and render tokens as queries. Rotary Position Embeddings (RoPE) encode the temporal distance between the current frame and keyframes for both keys and queries. LoRA modules are also added to the FFN layers. Two-stage training is used: Stage 1 trains the base model for core rendering; Stage 2 freezes the base model and trains only the Keyframe Adapter.
    - **Design Motivation**: Keyframes provide strong conditional anchoring that grounds the generation process in real light transport. Two-stage training ensures that the model functions correctly even in the absence of keyframes. RoPE temporal distance encoding allows the model to weight the influence of reference frames by proximity — closer frames have greater influence, distant frames less.

4. **Inverse Rendering Adapter**
    - **Function**: Reuses the frozen forward rendering backbone to decompose images into G-buffer intrinsics.
    - **Mechanism**: The forward rendering backbone is frozen, and trainable components are added: an inverse embedder (encoding RGB to tokens), LoRA on self-attention projections, prompt-conditioned cross-attention (text prompt selects the target intrinsic), and a lightweight MLP head for each intrinsic. Only adapter parameters are optimized during training, using modality-specific reconstruction losses (L1+LPIPS for albedo, cosine similarity for normals, scale-and-shift-invariant loss for depth, L1 for material).
    - **Design Motivation**: Demonstrates the generality of the framework — the same backbone can switch between forward and inverse rendering via text prompts, with parameter-efficient adaptation.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{latent}} + \lambda \mathcal{L}_{\text{pixel}}$
- Latent loss: bridge matching velocity prediction loss.
- Pixel loss: $\mathcal{L}_{\text{LPIPS}} + \mathcal{L}_{\text{grad}}$ (LPIPS perceptual loss + gradient loss to recover high-frequency details such as contact shadows).
- Training is performed on short clips (5 frames); long-video inference uses a progressive overlapping-chunk strategy.
- Dataset: Custom-built in Unreal Engine 5, comprising ~4,000 unique meshes and 30 HDR environment maps, approximately 130K frames (30K artistic scenes + 100K procedural scenes), at 512×512 resolution, 256 SPP rendered and denoised with Intel OIDN.

## Key Experimental Results

### Main Results

| Method | Paradigm | Params | PSNR↑ | SSIM↑ | LPIPS↓ | Inference Time (s)↓ |
|---|---|---|---|---|---|---|
| Path Tracing | Traditional | — | — | — | — | >10 |
| Deferred Rendering | Traditional | — | 24.649 | 0.927 | 0.097 | — |
| RGB-X | Diffusion | 950M | 20.984 | 0.793 | 0.165 | ~2.19 |
| DiffusionRenderer | Diffusion | 1.7B | 23.758 | 0.863 | 0.128 | ~1.40 |
| **RenderFlow (w/o key)** | **Flow** | **1.4B** | **24.214** | **0.874** | **0.113** | **~0.19** |
| **RenderFlow (w/ key)** | **Flow** | **1.7B** | **26.663** | **0.883** | **0.101** | **~0.24** |

### Ablation Study

| Training Strategy | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Uniform SDE (4-step) | 22.192 | 0.858 | 0.120 |
| 4-step ODE (4-step inference) | 23.089 | 0.865 | 0.110 |
| 4-step ODE (1-step inference) | 23.304 | 0.867 | 0.108 |
| 4-step SDE (4-step inference) | 23.384 | 0.865 | 0.111 |
| **4-step SDE (1-step inference)** | **23.590** | **0.868** | **0.107** |

| Loss Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Latent loss only | 21.588 | 0.840 | 0.148 |
| + LPIPS | 23.538 | 0.867 | 0.105 |
| + LPIPS + gradient | 23.590 | 0.868 | 0.107 |

### Key Findings
- **Single-step inference outperforms multi-step inference**: Training with a 4-step SDE schedule and running 1-step inference (23.590) surpasses 4-step inference (23.384), as multi-step error accumulation is avoided — a counterintuitive finding.
- **SDE training outperforms ODE training**: The small noise perturbation ($\sigma=0.005$) encourages more diverse outputs and improves robustness.
- **Keyframe guidance yields substantial gains**: PSNR improves from 24.214 to 26.663 (+2.449) and LPIPS drops from 0.113 to 0.101, with only ~0.05 s added to inference time.
- **Zero-variance deterministic output**: Unlike diffusion methods, which exhibit significant variance across repeated inferences, RenderFlow is fully deterministic, achieving zero variance over 100-frame sequences — critical for production environments.
- **VAE is the performance bottleneck**: Of the ~0.19 s inference time, G-buffer encoding accounts for ~0.12 s and image decoding for ~0.04 s, with the VAE consuming ~90% of total time.
- **Competitive inverse rendering quality**: Normal angular error of 16.2° substantially outperforms RGB-X (46.5°) and DiffusionRenderer (47.6°).

## Highlights & Insights
- **The albedo-as-flow-start design achieves three goals simultaneously**: (a) preserving low-frequency color shortens the flow trajectory, enabling single-step high accuracy; (b) geometric integrity is maintained; (c) the design is semantically natural — rendering is conceptually the addition of illumination effects onto a base color. This "meaningful starting point" paradigm for flow matching is transferable to any image-to-image task with a structured correspondence between input and output (e.g., the inverse process of depth estimation or semantic segmentation).
- **The "multi-step training, single-step inference" finding is highly practical**: SDE training introduces micro-noise to improve robustness, yet no multi-step inference is needed at test time to achieve optimal performance — an efficient training/inference asymmetry strategy.
- **Unified framework for forward and inverse rendering**: By switching between forward and inverse rendering tasks within the same model via a frozen backbone, lightweight adapters, and prompt switching, the framework demonstrates the reusability of large-scale pretrained models.

## Limitations & Future Work
- **VAE encoding/decoding accounts for ~90% of inference time**: The model itself is fast, but the VAE is the bottleneck. Lightweight VAEs or end-to-end pixel-space approaches may enable further speedups.
- **Reliance on UE5 synthetic training data**: Generalization to real photographic scenes has not been thoroughly validated; domain gap may limit practical deployment.
- **Strong environment map assumption**: Real-world rendering scenarios may involve more complex combinations of direct and indirect light sources not fully representable by a single environment map.
- **512×512 resolution constraint**: All experiments are conducted at 512×512; scalability to higher resolutions (e.g., 4K) has not been verified.
- **Not a replacement for explicit geometry rendering**: The authors acknowledge that the method is not intended to replace highly optimized industrial real-time rendering pipelines, but rather to provide high-quality approximations in the absence of explicit geometry.

## Related Work & Insights
- **vs. DiffusionRenderer**: DiffusionRenderer is based on a video diffusion model but requires 30 inference steps (~1.40 s), achieving PSNR 23.758; RenderFlow reaches 24.214 in a single step (~0.19 s), being 7× faster at higher quality.
- **vs. RGB-X**: RGB-X is an image-level diffusion model requiring 50 steps (~2.19 s) and achieves only PSNR 20.984; RenderFlow is 10× faster with substantially superior quality.
- **vs. LBM (Latent Bridge Matching)**: RenderFlow adopts LBM's bridge matching training strategy ($\sigma=0.005$) but introduces rendering-specific designs: albedo-as-start, G-buffer condition injection, and keyframe guidance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The albedo-to-render flow matching formulation offers a fresh perspective, though the overall framework builds on established components (bridge matching, Wan2.1, adapters).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative and qualitative comparisons are comprehensive and ablations are detailed, but evaluation is limited to synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Method motivation is clear, technical details are complete, and figures are well designed.
- **Value**: ⭐⭐⭐⭐ The approach has practical applicability for interactive rendering and virtual production; deterministic output is a hard requirement in production environments.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](../../ICLR2026/image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)

<!-- RELATED:END -->
