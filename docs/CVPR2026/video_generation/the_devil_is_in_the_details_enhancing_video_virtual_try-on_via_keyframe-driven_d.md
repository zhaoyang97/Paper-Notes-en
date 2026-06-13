---
title: >-
  [Paper Note] The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection
description: >-
  [CVPR2026][Video Generation][video virtual try-on] This paper proposes KeyTailor, a framework that employs a keyframe-driven details injection strategy—comprising garment dynamic enhancement and collaborative background…
tags:
  - "CVPR2026"
  - "Video Generation"
  - "video virtual try-on"
  - "diffusion transformer"
  - "keyframe injection"
  - "garment fidelity"
  - "background integrity"
date: 2026-05-08
content_hash: df196704888401e9
---

# The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection

**Conference**: CVPR2026  
**arXiv**: [2512.20340](https://arxiv.org/abs/2512.20340)  
**Code**: [ViT-HD Dataset](https://huggingface.co/datasets/zijiyingcai/ViT-HD)  
**Area**: Video Generation  
**Keywords**: video virtual try-on, diffusion transformer, keyframe injection, garment fidelity, background integrity

## TL;DR

This paper proposes KeyTailor, a framework that employs a keyframe-driven details injection strategy—comprising garment dynamic enhancement and collaborative background optimization—to substantially improve garment fidelity and background consistency in video virtual try-on without modifying the DiT architecture. A 15K high-resolution dataset, ViT-HD, is also released.

## Background & Motivation

1. **Strong demand for video virtual try-on (VVT)**: E-commerce and short-video platforms require high-fidelity garment replacement videos with cross-frame motion consistency and visual realism.
2. **Trend of replacing U-Net with DiT**: U-Net-based methods struggle with complex textures and body motion; recent research has shifted toward large-scale video DiTs (e.g., Wan2.1-14B) for joint spatiotemporal modeling.
3. **Insufficient garment dynamic details**: Existing DiT methods introduce additional encoding components to learn garment appearance, yet still fail to capture fine-grained dynamics across consecutive frames (back-side textures, motion-induced wrinkles, lighting variations), resulting in over-smoothed outputs.
4. **Background region inconsistency**: Existing methods rely solely on garment-agnostic video for background conditioning, which frequently causes detail loss (texture blurring), temporal inconsistency (inter-frame artifacts), and environmental structure drift.
5. **High model complexity**: To enrich generation conditions, existing methods insert additional interaction modules inside the DiT, drastically increasing parameter count and computational overhead (e.g., MagicTryOn introduces 15.11% extra parameters).
6. **Limited dataset scale and quality**: Public datasets VVT (791 samples, 192×256) and ViViD (9,700 samples, 632×824) suffer from low resolution, limited scene diversity, and insufficient scale, constraining DiT generalization.

## Method

### Overall Architecture: KeyTailor

Built upon Wan2.1-I2V-14B pretrained weights, the core idea is **keyframe-driven details injection**—leveraging the multi-view garment dynamics and background details naturally present in keyframes to enhance generation quality, rather than inserting interaction modules inside the DiT. Inputs include: reference garment image $I_{ref}$, source video $V_{in}$, agnostic video $V_{agn}$, agnostic masks $M_{agn}$, and pose representation $P$.

### Instruction-guided Keyframe Sampling (IKS)

- A large vision-language model (QWen) parses predefined view-action instructions to extract target viewpoint $\mathcal{V}_{tar}$ and action $\mathcal{A}_{tar}$.
- HumanParsing generates normalized multi-anchor pose frames $F_{anc}$.
- A motion difference score $S_m(f)$ and a garment area ratio score $S_r(f)$ are computed per frame, yielding a composite score $S_f(f) = 1 - S_m(f) + \lambda \cdot S_r(f)$.
- A **dual-threshold selection strategy** constrains score differences and temporal intervals to avoid redundancy and ensure temporal uniformity.

### Garment Dynamic Details Enhancement Module (GDDE)

- A pretrained single-image try-on model with LoRA injects the garment into the agnostic first frame, producing a try-on first frame.
- The VAE encoder encodes this into a garment latent representation $L_g$.
- Multi-view features $L_{key}^{gar}$ are extracted from keyframe garment regions (e.g., back-side textures, wrinkles caused by raised arms).
- A lightweight distillation component $\mathcal{D}$ (two-layer 1×1 convolution + LayerNorm) injects keyframe garment variations into $L_g$:
  $\bar{L}_g = \mathcal{D}(\text{Concat}(L_g, \frac{1}{|F_{key}|}\sum L_k))$

### Collaborative Background Details Optimization Module (CBDO)

- **Coarse-grained global branch**: Mask Guider $\mathcal{E}_{BG}$ (four-layer 3D convolution, channels 32→96→192→256, with zero-initialized linear layer) encodes $V_{agn}$ into a global background latent representation $L_{bg}$.
- **Fine-grained keyframe branch**: Inverse human body masks crop background regions from keyframes; VAE encodes these into $L_{key}^{bg}$; the frame with the highest background completeness is selected as a supplement.
- Fusion: $\bar{L}_{bg} = \alpha \cdot L_{bg} + (1-\alpha) L_{key}^{max}$, with $\alpha=0.3$.

### Three-Step Fusion and Generation

1. Pose latent $L_p$ and mask $L_m$ are concatenated, patchified, and fused with $\bar{L}_g$ via a projection layer to obtain $L$.
2. $L$ is concatenated with patchified noise $\epsilon$ to obtain $\bar{L}$.
3. $\bar{L}_{bg}$ is injected via an "addto" operation, ultimately guiding DiT denoising.
4. $\bar{L}_g$ replaces text tokens in cross-attention to preserve garment details.
5. Only LoRA fine-tuning is applied to DiT attention modules; **the DiT architecture is not modified**.

### Loss & Training

Standard diffusion training loss (denoising objective). LoRA is applied to the self-attention and cross-attention of the DiT backbone. Learning rate: 1e-4; optimizer: AdamW; 14,500 iterations; batch size = 1; 81 frames per sample.

## Key Experimental Results

### Video Virtual Try-On — ViT-HD Dataset

| Method | VFID_I^p ↓ | VFID_R^p ↓ | SSIM ↑ | LPIPS ↓ | VFID_I^u ↓ | VFID_R^u ↓ |
|--------|-----------|-----------|--------|---------|-----------|-----------|
| MagicTryOn | 14.06 | 0.246 | 0.862 | 0.083 | 19.23 | 0.559 |
| CatV2TON | 15.87 | 0.290 | 0.855 | 0.098 | 20.02 | 0.576 |
| **KeyTailor** | **7.53** | **0.163** | **0.907** | **0.040** | **13.66** | **0.352** |

### Video Virtual Try-On — VVT Dataset

| Method | VFID_I^p ↓ | SSIM ↑ | LPIPS ↓ |
|--------|-----------|--------|---------|
| MagicTryOn | 1.991 | 0.958 | 0.024 |
| CatV2TON | 1.778 | 0.900 | 0.039 |
| **KeyTailor** | **1.226** | **0.968** | **0.016** |

### Image Virtual Try-On — VITON-HD

| Method | FID_p ↓ | SSIM ↑ | LPIPS ↓ |
|--------|---------|--------|---------|
| CatVTON | 6.139 | 0.869 | 0.097 |
| MagicTryOn | 8.036 | 0.894 | 0.048 |
| **KeyTailor** | **5.293** | **0.920** | **0.057** |

### Computational Efficiency Comparison

| Method | Trainable Params (B) | FLOPs (G) | Inference Time (s) |
|--------|---------------------|-----------|-------------------|
| MagicTryOn | 16.446 | 206,935 | 345.27 |
| **KeyTailor** | **0.206** | **194,607** | **281.65** |

KeyTailor adds only 2.10% parameters relative to the backbone, far fewer than MagicTryOn's 15.11%.

### Ablation Study

| Variant | VFID_I^p ↓ | SSIM ↑ | LPIPS ↓ |
|---------|-----------|--------|---------|
| w/o GDDE | 19.90 | 0.843 | 0.114 |
| w/o CBDO | 17.21 | 0.852 | 0.098 |
| w/o distillation $\mathcal{D}$ | 22.52 | 0.766 | 0.211 |
| w/o IKS | 16.26 | 0.804 | 0.102 |
| $F_{key}=1$ | 16.39 | 0.817 | 0.099 |
| **KeyTailor (full)** | **7.53** | **0.907** | **0.040** |

Removing the distillation component $\mathcal{D}$ has the greatest impact (VFID: 7.53 → 22.52), validating its central role in injecting keyframe garment dynamics.

## Highlights & Insights

- **No DiT architecture modification**: High-quality video try-on is achieved solely through external details injection and LoRA fine-tuning, with only 0.2B trainable parameters—highly practical for engineering deployment.
- **Novel keyframe-driven paradigm**: IKS leverages a vision-language model to parse viewpoint/action instructions for keyframe selection, simultaneously capturing multi-view garment dynamics and background consistency.
- **Dual-module design (GDDE + CBDO)**: Each module targets a distinct aspect—garment detail fidelity and background integrity—and ablation studies confirm both are indispensable.
- **Large-scale high-resolution dataset ViT-HD**: 15,070 samples at 810×1080, covering upper/lower/full-body garments, substantially surpassing existing public datasets.
- **Comprehensive experiments**: Covering 3 video datasets, 2 image datasets, a user study, extensive ablations, and efficiency analysis.

## Limitations & Future Work

- Keyframe sampling relies on large language model inference (e.g., QWen), increasing deployment complexity and latency.
- Background optimization selects only a single highest-completeness keyframe, which may be insufficient for complex dynamic backgrounds (e.g., moving cameras).
- Inference still requires 281.65s (25 denoising steps, 81 frames), far from real-time applicability.
- The dataset originates from e-commerce platforms with predominantly indoor display scenes; generalization to in-the-wild scenarios remains to be validated.
- The first frame depends on the quality of the single-image try-on model FiTDiT; errors in the first frame propagate to subsequent frames.

## Related Work & Insights

| Method | Backbone | Extra Params | Characteristics |
|--------|----------|-------------|-----------------|
| ViViD | SD1.5 | +157% | Reference branch + temporal attention; high complexity |
| CatV2TON | SD1.5 (modified) | — | Removes text attention layers; concatenated conditioning |
| MagicTryOn | Wan2.1 | +15.1% | Interaction modules inside DiT; large parameter overhead |
| DreamVVT | DiT | — | Fine-tuned on proprietary data; limited gains on ViViD |
| **KeyTailor** | Wan2.1 | **+2.1%** | External injection + LoRA; lightweight and efficient |

## Rating

- Novelty: ⭐⭐⭐⭐ — The keyframe-driven injection strategy and the "no DiT modification" design philosophy are novel; GDDE/CBDO modules are concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, a user study, nine ablation groups, and efficiency analysis; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and figures are abundant, though some mathematical notation could be more consistent.
- Value: ⭐⭐⭐⭐ — The ViT-HD dataset provides significant value to the community; the lightweight design is well-suited for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)
- [\[ICML 2026\] iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](../../ICML2026/video_generation/itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](../../ICML2026/video_generation/enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
