---
title: >-
  [Paper Note] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][3D Reconstruction] tttLRM is the first work to introduce Test-Time Training (TTT) into large-scale 3D reconstruction models. It leverages LaCT layers to achieve long-context and autoregressive 3D Gaussian reconstruction at linear complexity. Multi-view observations are compressed into TTT fast weights to form an implicit 3D representation, which is then decoded into explicit formats such as 3DGS, achieving state-of-the-art performance on both object-level and scene-level benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Reconstruction
  - Test-Time Training
  - Large Reconstruction Model
  - Gaussian Splatting
  - Autoregressive Reconstruction
date: 2026-05-08
content_hash: c2fc95046bf8f046
---

# tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2602.20160](https://arxiv.org/abs/2602.20160)
**Area**: 3D Vision
**Keywords**: 3D Reconstruction, Test-Time Training, Large Reconstruction Model, Gaussian Splatting, Autoregressive Reconstruction

## TL;DR

tttLRM is the first work to introduce Test-Time Training (TTT) into large-scale 3D reconstruction models. It leverages LaCT layers to achieve long-context and autoregressive 3D Gaussian reconstruction at linear complexity. Multi-view observations are compressed into TTT fast weights to form an implicit 3D representation, which is then decoded into explicit formats such as 3DGS, achieving state-of-the-art performance on both object-level and scene-level benchmarks.

## Background & Motivation

Reconstructing explicit 3D representations from streaming visual input is a central goal of 3D vision, yet existing approaches exhibit clear bottlenecks:

**Traditional optimization methods** (NeRF, 3DGS): require per-scene optimization, taking minutes to hours.

**Feed-forward large reconstruction models** (LRM, GS-LRM): attention-based architectures that support only a limited number of input views (typically ≤4), due to $O(N^2)$ attention complexity.

**Long-LRM**: extends to 32 views, but bidirectional attention still hinders further scaling and precludes streaming input.

**Implicit latent 3D representations** (LVSM, etc.): achieve high novel-view synthesis quality but suffer from slow rendering speed, limited controllability, and poor interpretability.

The root cause lies in the need for a **linear-complexity alternative to attention for long-context modeling that simultaneously supports streaming/autoregressive inference**.

tttLRM is inspired by an analogy to human perception: humans observe a continuous visual stream → build an abstract internal representation → decode it on demand into an explicit 3D structure. The fast weights of the TTT framework naturally correspond to this "internal memory" mechanism.

## Method

### Overall Architecture

tttLRM consists of three core components:

1. **Image encoding**: input images are patchified and projected into token sequences.
2. **LaCT layers updating fast weights**: tokens iteratively update the TTT fast weights $W$, forming an implicit 3D representation.
3. **Virtual token query and decoding**: virtual view tokens query the fast weights, and a linear decoder outputs explicit 3D representations (e.g., 3DGS parameters).

### TTT and LaCT Principles

**TTT (Test-Time Training)** reformulates sequence modeling as an online learning problem:

$$W \leftarrow W - \eta \nabla \mathcal{L}_{\text{MSE}}(f_W(k), v)$$

The fast weights $W$ are updated at inference time based on input key-value pairs, encoding the KV cache as a fixed-size neural memory.

**LaCT (Large Chunk TTT)** performs large-chunk updates (up to 1M tokens) with intra-chunk gradient accumulation to achieve high GPU utilization. Each LaCT layer comprises:

- A window attention module (capturing intra-view local relationships)
- Fast weight update (linear complexity)
- Fast weight application (linear complexity)

### Model Architecture Details

The model consists of **24 LaCT blocks** with hidden dimension 768 and patch size $8 \times 8$.

For each input image $\mathbf{I}_i \in \mathbb{R}^{H \times W \times 3}$, it is concatenated with ray embeddings $\mathbf{R}_i \in \mathbb{R}^{H \times W \times 9}$, then patchified and tokenized. The processing pipeline is:

$$\mathbf{T}_i = \mathbf{T}_i + \text{WinAttn}(\mathbf{T}_i)$$
$$W = \text{Update}(\{\mathbf{T}_i\}_{i=1}^N)$$
$$\mathbf{T}_i^v = \text{Apply}(W, \mathbf{T}_i^v)$$

Virtual tokens $\mathbf{T}^v$ participate only in the Apply operation without updating the fast weights. A decoder transforms them into per-patch Gaussian parameters (color, scale, rotation, opacity, depth).

### Autoregressive Reconstruction

The autoregressive mode converts the model into an RNN-like inference process:

- Initialize $W \leftarrow W_0$
- For each batch $b$: update $W \leftarrow \mathcal{F}(W, \mathcal{I}_{(b)})$, predict $G_{(b)} \leftarrow \mathcal{F}(W, \mathcal{I}^v_{(b)})$
- Return the final Gaussians $G_{(B)}$

Upon arrival of each batch (e.g., 4 images), the fast weights are incrementally updated and 3D Gaussians are immediately predicted, enabling online progressive reconstruction.

### Distributed Feed-Forward Reconstruction

To support large numbers of input views and high-resolution images, sequence parallelism is introduced:

1. Shard the sequence dimension across multiple GPUs.
2. After fast weight synchronization, each GPU independently predicts Gaussians for its assigned views.
3. Gaussians are aggregated to form the complete scene.
4. Each GPU renders a subset of novel views, computes the loss, and gradients are synchronized via All-Reduce.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{opacity}} \mathcal{L}_{\text{opacity}}$$

- **Rendering loss**: MSE + VGG-19 perceptual loss
- **Depth regularization**: scale-invariant depth loss using pseudo-GT from a monocular depth estimator
- **Opacity regularization**: penalizes the number of opaque Gaussians

### Multi-Format Output

Beyond 3DGS, the architecture can flexibly decode into other 3D formats such as triplane NeRF — requiring only the replacement of virtual tokens with triplane tokens to query the fast weights.

## Key Experimental Results

### Object-Level Reconstruction (GSO Dataset, Tab. 1)

| Method | Resolution | Views | Time (s) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------------|-------|----------|--------|--------|---------|
| GS-LRM | 256² | 8 | 0.1 | 31.55 | 0.964 | 0.028 |
| **Ours** | 256² | 8 | 0.1 | **33.14** | **0.972** | **0.024** |
| GS-LRM | 512² | 8 | 0.7 | 32.83 | 0.969 | 0.029 |
| **Ours** | 512² | 8 | **0.3** | **34.02** | **0.974** | **0.025** |
| GS-LRM | 512² | 16 | 2.5 | 33.55 | 0.976 | 0.023 |
| **Ours** | 512² | 16 | **0.8** | **34.67** | **0.978** | **0.022** |
| GS-LRM | 512² | 24 | 5.5 | 33.26 | 0.976 | 0.022 |
| **Ours** | 512² | 24 | **1.1** | **34.80** | **0.979** | **0.022** |

At 512² resolution, inference is 2× faster than attention-based models, with PSNR improvements of **>1 dB**.

### Scene-Level Reconstruction (DL3DV-140 + Tanks&Temples, Tab. 2)

| Views | Method | Time | DL3DV PSNR ↑ | T&T PSNR ↑ |
|-------|--------|------|-------------|------------|
| 16 | Long-LRM | 0.4s | 22.66 | 17.51 |
| 16 | **Ours** | 3.6s | **23.60** | **18.15** |
| 32 | Long-LRM | 1s | 24.10 | 18.38 |
| 32 | Long-LRM + optim | 12s | 24.99 | 18.69 |
| 32 | **Ours** | 7.2s | **25.07** | **19.22** |
| 64 | Long-LRM | 3.7s | 24.63 | 19.11 |
| 64 | **Ours** | 14.8s | **25.95** | **20.31** |

A single model generalizes across varying numbers of input views and consistently outperforms Long-LRM with post-optimization.

### Ablation Study

**Effect of Pre-training (Tab. 3)**:

| 3D Representation | Pre-training | PSNR ↑ | LPIPS ↓ |
|-------------------|-------------|--------|---------|
| GS | None | 32.77 | 0.026 |
| GS | **With pre-training** | **33.14** | **0.024** |
| Triplane | None | 26.40 | 0.093 |
| Triplane | **With pre-training** | **27.87** | **0.075** |

Initializing from TTT-LVSM pre-training significantly accelerates convergence and improves final quality, demonstrating effective knowledge transfer from novel-view synthesis to explicit 3D reconstruction.

**Autoregressive Strategy (Tab. 4)**:

| Strategy | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|--------|--------|---------|
| Predict & Merge | 21.50 | 0.891 | 0.318 |
| **Full reconstruction (Ours)** | **23.63** | **0.904** | **0.259** |

While "Predict & Merge" is computationally efficient, it degrades quality due to accumulated errors (2.13 dB PSNR gap).

## Highlights & Insights

1. **TTT fast weights as implicit 3D memory**: This is an elegant analogy — fast weights updated dynamically at inference time naturally correspond to an "internal 3D representation refined with increasing observations," offering greater expressivity than fixed-size KV caches.
2. **Practical significance of linear complexity**: Beyond supporting more input views, it crucially enables **autoregressive/streaming reconstruction**, a capability fundamentally inaccessible to attention-based models.
3. **Effectiveness of pre-training transfer**: The transfer learning strategy from NVS to explicit 3D is concise and effective, demonstrating that implicit 3D understanding can transfer across representation formats.
4. **Unified multi-format output**: The same framework outputs either 3DGS or triplane NeRF by simply swapping virtual tokens, showcasing the generality of the architecture.
5. **Sequence-parallel training**: Exploiting the linear structure of LaCT fast weight updates, gradients can be straightforwardly synchronized via All-Reduce, enabling linear multi-GPU scaling for both training and inference.

## Limitations & Future Work

1. **Fixed-size fast weights**: Neural memory capacity is bounded; extremely complex scenes with very large numbers of input views may exceed the encoding capacity.
2. **Quality–speed trade-off**: Compared to pre-trained implicit LVSM models, explicit 3D output quality is slightly lower, though this is offset by real-time rendering and controllability.
3. **Dependence on pseudo depth GT**: Scene-level training relies on a monocular depth estimator for pseudo supervision, and estimation errors propagate into the 3D reconstruction.
4. **Non-real-time inference**: Although orders of magnitude faster than optimization-based methods, scene-level reconstruction with 64 views still requires approximately 15 seconds, falling short of real-time streaming reconstruction.

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is a highly forward-looking work. Introducing the TTT mechanism into 3D reconstruction is a natural yet profound innovation; linear complexity makes long-context and autoregressive modeling feasible. The experiments are comprehensive and convincing, achieving state-of-the-art performance on both object-level and scene-level benchmarks. The strategy of transferring NVS pre-training to explicit 3D reconstruction is elegant and practical. The unified architecture design and scalability lay a strong foundation for future real-time 3D perception systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning 3D Reconstruction with Priors in Test Time](tco_learning_3d_reconstruction_with_priors_in_test_time.md)
- [\[CVPR 2026\] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)

</div>

<!-- RELATED:END -->
