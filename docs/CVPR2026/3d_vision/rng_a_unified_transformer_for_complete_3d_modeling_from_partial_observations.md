---
title: >-
  [Paper Note] RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations
description: >-
  [CVPR 2026][3D Vision][3D reconstruction] RnG proposes Reconstruction-Guided Causal Attention, which reinterprets the Transformer's KV-Cache as an implicit 3D representation…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D reconstruction"
  - "novel view synthesis"
  - "transformer"
  - "KV-Cache"
  - "feed-forward"
date: 2026-05-08
content_hash: 4700be2820439c9b
---

# RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

**Conference**: CVPR 2026
**arXiv**: [2603.01194](https://arxiv.org/abs/2603.01194)  
**Code**: [https://npucvr.github.io/RnG](https://npucvr.github.io/RnG)  
**Area**: 3D Vision
**Keywords**: 3D reconstruction, novel view synthesis, transformer, KV-Cache, feed-forward

## TL;DR

RnG proposes Reconstruction-Guided Causal Attention, which reinterprets the Transformer's KV-Cache as an implicit 3D representation, enabling a single feed-forward Transformer to jointly perform reconstruction and generation—recovering complete 3D geometry and appearance from sparse, pose-free images—at over 100× the speed of diffusion-based methods.

## Background & Motivation

### Core Problem
Current 3D reconstruction foundation models (e.g., VGGT, DUSt3R) can recover the geometry of visible regions from a small number of images, but **cannot model unobserved regions**. Novel view synthesis (NVS) methods can render unseen viewpoints but typically lack consistent 3D structure, or rely on known camera poses or diffusion models that incur slow inference.

### Limitations of Prior Work

| Method | Pose-free Inference | Camera Control | Generate Unseen Regions | Explicit 3D | Real-time Inference |
|--------|:-------------------:|:--------------:|:-----------------------:|:-----------:|:-------------------:|
| VGGT | ✓ | N/A | ✗ | ✓ | ✓ |
| DUSt3R | ✓ | N/A | ✗ | ✓ | ✓ |
| LVSM | ✗ | ✓ | ✓ | ✗ | ✓ |
| LGM | ✗ | ✓ | ✗ | ✓ | ✓ |
| Matrix3D | ✓ | ✓ | ✓ | ✓ | ✗ |
| **RnG (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

Although Matrix3D achieves unified reconstruction and generation, its diffusion-based design requires 27 seconds per novel view, making it unsuitable for real-time interactive applications.

### Core Insight
The latent space of 3D reconstruction foundation models may already encode a more complete 3D understanding than what is visible in the observed geometry. If view-conditioned neural rendering can be formulated as a query against the model's latent space, both reconstruction and generation capabilities can be simultaneously activated. In contrast to the prevailing direction of using generative priors to assist reconstruction, RnG demonstrates that **driving generation with reconstruction priors** is equally feasible and highly efficient.

## Method

### Overall Architecture

RnG builds upon the architecture and pretrained weights of VGGT as a feed-forward Transformer. The processing pipeline is as follows:

1. **Input Encoding**: Source-view images $\{\mathbf{I}_s\}$ are encoded into tokens via a DINOv2 Vision Transformer; target views are encoded as Plücker ray maps and mapped to tokens via a linear layer.
2. **Joint Processing**: Source- and target-view tokens are concatenated and processed through $L=24$ alternating layers of Global Attention and Frame Attention.
3. **Multi-head Decoding**:
    - Source-view tokens → Camera Head → estimated camera poses $\{\hat{\mathbf{g}}_s\}$
    - Target-view tokens → RGB Head $\mathcal{D}_\text{RGB}$ → novel view images $\hat{\mathbf{I}}_t$
    - Target-view tokens → Point Head $\mathcal{D}_\text{pmap}$ → point maps $\hat{\mathbf{p}}_t$ (explicit geometry)

To preserve the knowledge acquired by VGGT, the first source view uses dedicated camera and register tokens, while the remaining source views and target views share the same token type. During training, the pose of the first view is fixed to:

$$\hat{\mathbf{g}}_{s=1} = \left[I_{3\times3} \mid [0, 0, -1]^\top\right]$$

This implicitly defines the world coordinate frame for reconstruction.

### Key Design 1: Reconstruction-Guided Causal Attention

This is the central innovation of RnG. The design motivation is: **reconstruction should guide generation, but generation should not interfere with reconstruction**.

Specifically, a binary mask $M$ is introduced into the global attention block to control information flow:

$$M_{i,j} = \begin{cases} 0 & \text{if } i \in \{s\} \text{ and } j \in \{t\} \\ 1 & \text{elsewhere} \end{cases}$$

where $\{s\}$ and $\{t\}$ denote the indices of source- and target-view tokens, respectively. The attention computation becomes:

$$\text{Out} = \text{softmax}\left(\frac{M \odot QK^\top}{\sqrt{d_k}}\right)V$$

**Information flow**:
- Source-view queries can only attend to source-view keys → reconstruction is shielded from target-view influence.
- Target-view queries attend to both source- and target-view keys → generation leverages reconstruction information.

The elegance of this design lies in the use of **shared parameters** for both source and target views, while the attention mask functionally decouples the two roles: source-view tokens handle perception and pose estimation (reconstruction), and target-view tokens handle appearance and geometry synthesis (generation). The model is thus parameter-efficient and amenable to joint training.

### Key Design 2: KV-Cache as Implicit 3D Representation

A further key property of causal attention is that it allows the cached key/value tokens to be reinterpreted as an **implicit 3D representation**—a latent memory encoding scene geometry and appearance independently of the viewing direction. This enables inference to be decomposed into two stages:

**Stage 1: Reconstruction and Caching** (~0.2 s)

Since source-view token attention is entirely independent of target-view tokens, the model can perform reconstruction using only source views and cache the key and value tokens from each global attention layer:

$$K_s' = \text{Cache}(K_s), \quad V_s' = \text{Cache}(V_s)$$

**Stage 2: Generation and Querying** (<0.1 s/view)

When synthesizing novel views, global and frame attention over source views need not be recomputed; the model reads directly from the cache:

$$\text{Out}_t = \text{softmax}\left(\frac{Q_t \cdot [K_s'; K_t]^\top}{\sqrt{d_k}}\right)[V_s'; V_t]$$

After $L$ layers of processing, the target-view tokens are decoded by two DPT Heads:

$$\hat{\mathbf{I}} = \mathcal{D}_\text{RGB}(\text{Out}_t), \quad \hat{\mathbf{P}} = \mathcal{D}_\text{pmap}(\text{Out}_t)$$

Complete 3D structure is recovered by accumulating point maps across multiple queried viewpoints, effectively acting as a **virtual 3D scanner**.

### Loss & Training

The multi-task loss consists of three components:

$$\mathcal{L} = \mathcal{L}_\text{RGB} + \lambda_\text{pmap}\mathcal{L}_\text{pmap} + \lambda_c\mathcal{L}_\text{cam}$$

**Novel-view image loss**—MSE + perceptual loss:

$$\mathcal{L}_\text{RGB} = |\mathbf{I}_t - \hat{\mathbf{I}}_t|_2 + \lambda_p \cdot \text{Perceptual}(\mathbf{I}_t, \hat{\mathbf{I}}_t)$$

**Point map loss**—uncertainty-weighted aleatoric uncertainty loss, where the Point Head outputs four channels (xyz + uncertainty $\Sigma_t$):

$$\mathcal{L}_\text{pmap} = \|\Sigma_t \odot (\mathbf{P}_t - \hat{\mathbf{P}}_t)\| + \|\Sigma_t \odot (\nabla\mathbf{P}_t - \nabla\hat{\mathbf{P}}_t)\| - \alpha \cdot \log\Sigma_t$$

**Camera pose loss**—Huber loss:

$$\mathcal{L}_\text{cam} = \sum_s |\mathbf{g}_s - \hat{\mathbf{g}}_s|_\epsilon$$

Hyperparameters: $\lambda_\text{pmap}=0.2$, $\lambda_c=1$, $\lambda_p=0.5$, $\alpha=0.2$.

**Training details**:
- Training data: Objaverse dataset (LVIS subset + LGM-filtered list, totaling 113.5K objects)
- Resolution: $256 \times 256$, patch size = 8
- Hardware: 8 × A800 GPUs, total batch size = 96
- Training steps: 40K steps
- Precision: bfloat16 + gradient checkpointing

## Key Experimental Results

### Main Results (GSO Dataset)

| Metric Category | Metric | Matrix3D (unposed) | VGGT | LVSM (posed) | **RnG (Ours)** |
|-----------------|--------|:------------------:|:----:|:------------:|:--------------:|
| Pose | RA@5↑ | 43.77 | 74.24 | — | **85.15** |
| Pose | RT@5↑ | 65.92 | 65.68 | — | **86.02** |
| Pose | AUC@30↑ | 66.39 | 77.23 | — | **86.94** |
| Source-view Depth | Rel↓ | 9.43 | 5.96 | — | **0.584** |
| Source-view Depth | a1↑ | 92.26 | 97.72 | — | **99.93** |
| Novel-view Depth | Rel↓ | 9.96 | — | — | **0.717** |
| Novel-view Depth | a1↑ | 90.28 | — | — | **99.85** |
| Novel View Synthesis | PSNR↑ | 18.74 | — | 27.52 | **26.28** |
| Novel View Synthesis | SSIM↑ | 0.786 | — | 0.902 | 0.891 |
| Novel View Synthesis | LPIPS↓ | 0.193 | — | 0.090 | 0.098 |
| Complete 3D | CD↓ | 0.067 | 0.026 | — | **0.0067** |

**Key Findings**:
- RnG substantially outperforms VGGT and Matrix3D on all reconstruction metrics; pose estimation RA@5 improves from 74.24 to 85.15.
- Source-view depth Rel error (0.584) is reduced by an order of magnitude compared to VGGT (5.96).
- As a pose-free method, RnG's novel view synthesis quality (PSNR 26.28) approaches that of LVSM (27.52), which requires known camera poses.
- Chamfer Distance (0.0067) is markedly superior to all baselines, demonstrating high geometric consistency in multi-view 3D fusion.

### Ablation Study

| Model Variant | RA@5↑ | PSNR↑ | LPIPS↓ | Note |
|---------------|:-----:|:-----:|:------:|------|
| LVSM-100K | — | 27.52 | 0.090 | Best LVSM performance (requires poses) |
| LVSM-40K | — | 24.62 | 0.154 | Equivalent training steps |
| **Ours-40K** | **85.15** | **26.28** | **0.098** | Full model |
| Ours-15K | 81.65 | 24.86 | 0.124 | Smaller dataset |
| Ours-15K-scratch | 8.25 | 20.78 | 0.204 | No pretrained weights |
| Ours-15K-w/o cam | — | 24.85 | 0.124 | Without camera pose supervision |
| Ours-15K-FullAttn | 82.72 | 24.86 | 0.119 | Full bidirectional attention |

**Key Findings**:
1. **Reconstruction priors are critical**: Training from scratch causes substantial performance degradation (PSNR drops by 4 points), confirming that VGGT's pretrained weights are a key driver.
2. **Training efficiency advantage**: Ours-15K already surpasses LVSM-40K, demonstrating improved data efficiency from reconstruction priors.
3. **Causal vs. full attention**: Replacing causal attention with bidirectional attention (FullAttn) yields nearly identical performance, confirming that the causal design achieves architectural benefits without sacrificing accuracy.
4. **Pose supervision compatibility**: Removing the Camera Head does not affect generation quality, indicating that reconstruction and generation are compatible within multi-task learning.

### Efficiency Comparison

KV-Cache caching substantially accelerates inference: latency is reduced from 213 ms to 85 ms, and FLOPs drop from 12.26T to 2.29T; this is **300×+ faster** than Matrix3D's 27 s per view.

### Generalization

Although trained with 4 input images, RnG generalizes directly to an arbitrary number of inputs. Synthesis quality improves consistently as the number of source views increases; for objects with symmetric structure, even a single input image yields reasonable results.

## Highlights & Insights

- **Unified framework**: The first feed-forward Transformer model to jointly achieve pose-free 3D reconstruction and novel-view synthesis of both geometry and appearance.
- **Causal attention**: Task decoupling is achieved via attention masking rather than separate modules, yielding a parameter-efficient and elegant design.
- **KV-Cache reuse**: The KV-Cache mechanism from NLP is endowed with a novel semantic role—implicit 3D representation—enabling efficient multi-query synthesis after a single caching pass.
- **Reverse knowledge transfer**: The direction of knowledge transfer from reconstruction to generation is complementary to the mainstream paradigm of using generative priors to assist reconstruction.

## Limitations & Future Work

1. **Insufficient texture detail**: As a deterministic feed-forward model, RnG cannot generate highly fine-grained textures in the manner of diffusion-based methods.
2. **World coordinate frame assumption**: Data preparation assumes all cameras face the world origin; handheld capture in real-world applications must satisfy this assumption.
3. **Accumulated noise in multi-view fusion**: Complete 3D reconstruction requires accumulating point maps across multiple queried views, which may introduce noise and geometric conflicts during multi-view fusion.

## Rating

| Dimension | Score |
|-----------|:-----:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐⭐ |

> Reinterpreting the KV-Cache as an implicit 3D representation is an elegant design choice. The reconstruction-driven generation paradigm offers a real-time-viable new path toward unified 3D understanding. The experiments are comprehensive, achieving state-of-the-art performance across multiple tasks at inference speeds two orders of magnitude faster than diffusion-based approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[CVPR 2026\] NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather](nimbusgs_unified_3d_scene_reconstruction_under_hybrid_weather.md)
- [\[CVPR 2026\] PoseMaster: A Unified 3D Native Framework for Stylized Pose Generation](posemaster_a_unified_3d_native_framework_for_stylized_pose_generation.md)
- [\[CVPR 2026\] LitePT: Lighter Yet Stronger Point Transformer](litept_lighter_yet_stronger_point_transformer.md)

</div>

<!-- RELATED:END -->
