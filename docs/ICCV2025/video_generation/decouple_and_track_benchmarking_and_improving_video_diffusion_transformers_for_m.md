---
title: >-
  [Paper Note] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer
description: >-
  [ICCV 2025][Video Generation][Diffusion Transformer] To address the difficulty of decoupling motion from appearance in DiT models with 3D full-attention, this paper proposes Shared Temporal Kernels and a Dense Point Tracking Loss, along with a comprehensive motion transfer benchmark MTBench and a hybrid motion fidelity metric.
tags:
  - ICCV 2025
  - Video Generation
  - Diffusion Transformer
  - motion transfer
  - temporal kernel
  - trajectory tracking
  - benchmark
date: 2026-05-08
content_hash: cda3e18c3fd21d83
---

# Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer

**Conference**: ICCV 2025  
**arXiv**: [2503.17350](https://arxiv.org/abs/2503.17350)  
**Code**: [Project Page](https://shi-qingyu.github.io/DeT.github.io)  
**Area**: Video Generation / Motion Transfer  
**Keywords**: Diffusion Transformer, motion transfer, temporal kernel, trajectory tracking, benchmark

## TL;DR

To address the difficulty of decoupling motion from appearance in DiT models with 3D full-attention, this paper proposes Shared Temporal Kernels and a Dense Point Tracking Loss, along with a comprehensive motion transfer benchmark MTBench and a hybrid motion fidelity metric.

## Background & Motivation

Motion transfer aims to transfer motion patterns from a source video to newly generated videos while allowing text-controlled appearance of foreground and background. The core challenge lies in decoupling motion from appearance.

Limitations of prior work:
- **3D U-Net-based methods** (e.g., MotionDirector, SMA) leverage separate spatial/temporal self-attention for decoupling—freezing spatial attention and training only temporal attention. However, these methods are incompatible with modern DiT models.
- **DiT models** (e.g., CogVideoX, HunyuanVideo) employ 3D full-attention mechanisms that **do not explicitly separate spatial and temporal information**, making motion-appearance decoupling extremely difficult.
- Existing benchmarks (e.g., DMT, MotionDirector benchmark) are small-scale with limited motion diversity, making comprehensive evaluation difficult.

Through visualization of DiT features, the authors identify a key issue: **foreground and background features are difficult to distinguish in 3D full-attention**, and background features exhibit temporal inconsistency during denoising, causing foreground-background confusion in certain frames and entangling background appearance with foreground motion.

## Method

### Overall Architecture

DeT fine-tunes a pretrained DiT model with two core components: Shared Temporal Kernels for decoupling motion/appearance and learning motion patterns, and a Dense Point Tracking Loss for enhancing foreground motion consistency. During inference, temporal kernels in the last 65% of DiT blocks are removed to improve editing fidelity.

### Key Designs

1. **Shared Temporal Kernel**:

   Core observation: Analysis of 3D attention maps reveals that **significant attention scores appear only on the diagonal of adjacent frames**, indicating that modeling temporal changes in DiT features requires only a local temporal receptive field rather than a spatial one.

   Design rationale: A temporal convolution smoothing operation is applied to DiT features $\mathcal{I}' \in \mathbb{R}^{hw \times t \times c}$ along the temporal dimension. From a manifold learning perspective, the temporal kernel is equivalent to a Laplacian smoothing operator along the temporal axis:

    $\hat{\mathcal{I}}_{xy,i} = \mathcal{I}'_{xy,i} + \sum_{j=-\frac{k-1}{2}}^{\frac{k-1}{2}} \mathcal{I}'_{xy,i+j} \times \mathcal{K}^{t}_j$

   A down-and-up architecture is adopted to reduce parameters and memory: $\mathcal{K}^{t}_{down} \in \mathbb{R}^{k \times c \times m}$, $\mathcal{K}^{t}_{up} \in \mathbb{R}^{k \times m \times c}$, with GELU activation in between:

    $\tilde{\mathcal{I}} = \mathcal{K}^{t}_{up} * \sigma(\mathcal{K}^{t}_{down} * \mathcal{I}) + \text{Attention}(\mathcal{I}, \mathcal{E}_{text})$

   **Dual role**: (1) Temporal smoothing makes foreground/background features temporally consistent, facilitating their distinction; (2) Temporal 1D convolution effectively captures inter-frame changes (i.e., motion information) without introducing spatial information, thereby avoiding appearance memorization.

2. **Dense Point Tracking Loss**:

   Based on the observation that foreground DiT features should remain temporally consistent, explicit supervision is introduced to enhance foreground motion consistency:
    - CoTracker is used to track foreground in the source video, generating trajectory set $\mathcal{T} \in \mathbb{R}^{N \times T \times 2}$ and visibility matrix $\mathcal{V} \in \{0,1\}^{N \times T \times 1}$
    - Features are aligned along trajectories on the predicted latent $\hat{\mathcal{E}}(\mathcal{S})$ using an occlusion-aware L2 distance:
    $\mathcal{L}_{TL} = \|\min(\mathcal{V}(t+1), \mathcal{V}(t)) \times [\hat{\mathcal{E}}(\mathcal{S})[\mathcal{T}(t+1)] - \hat{\mathcal{E}}(\mathcal{S})[\mathcal{T}(t)]]\|_2^2$
    - Final loss: $\mathcal{L} = \lambda_{DL} \mathcal{L}_{DL} + \lambda_{TL} \mathcal{L}_{TL}$ ($\lambda_{DL}=1.0$, $\lambda_{TL}=0.1$)

3. **MTBench Benchmark Construction**:

    - Sourced from DAVIS and YouTube-VOS, comprising 100 high-quality videos and 500 evaluation prompts
    - Qwen2.5-VL-7B is used to generate descriptions; Qwen2.5-14B generates evaluation prompts
    - SAM + CoTracker annotate foreground trajectories, with distance-weighted sampling of initial points (ensuring coverage of narrow regions such as limbs)
    - K-means clustering categorizes motion into three difficulty levels: Easy/Medium/Hard

### Hybrid Motion Fidelity Metric

Combining Fréchet distance (global shape similarity) and velocity direction cosine similarity (local motion direction consistency):

$$\mathcal{M}(\mathcal{T}_i, \mathcal{T}_j) = \frac{1}{N} \sum_{n=1}^{N} [\alpha \cdot e^{-d_F(\mathcal{T}_i^n, \mathcal{T}_j^n)} + (1-\alpha) \cdot \bar{c}_n]$$

where $\alpha = 0.5$ balances the two components.

### Loss & Training

The model is trained on a single source video for 500 steps using AdamW optimizer with learning rate 1e-5 and weight decay 1e-2. The intermediate dimension is 128 and kernel size is 3. During inference, temporal kernels in the last 27 blocks (65%) are removed to improve editing fidelity. DDIM scheduler with 50 denoising steps and CFG scale 6.0 are used. Training takes approximately 1 hour on a single A100 GPU.

## Key Experimental Results

### Main Results: Full MTBench Evaluation

| Method | Edit Fidelity | Temporal Consistency | Motion Fidelity |
|------|--------------|---------------------|----------------|
| MotionDirector (U-Net) | 31.9 | 91.7 | 67.7 |
| SMA (U-Net) | 31.6 | 82.9 | 55.1 |
| MotionClone (U-Net) | 30.8 | 80.9 | 78.9 |
| MOFT (U-Net) | 33.0 | 91.1 | 52.5 |
| DreamBooth (CogVideoX) | 28.4 | 85.6 | 80.4 |
| MotionInversion (CogVideoX) | 26.6 | 85.4 | 85.0 |
| **DeT (CogVideoX)** | **31.2** | **89.6** | **85.8** |
| **DeT (HunyuanVideo)** | **31.9** | **91.9** | **85.9** |

### Ablation Study

| Motion Learning Approach | Edit Fidelity | Motion Fidelity | Note |
|-------------|--------------|----------------|------|
| LoRA | 28.4 | 80.4 | Full parameter adaptation; appearance overfitting |
| Conv3D | 27.1 | 84.9 | 3D convolution introduces spatial information |
| Local Attention | 31.3 | 73.1 | Poor motion fidelity with local attention |
| **Temporal Conv1D** | **31.6** | **85.6** | Best balance |

| Hyperparameter | Optimal Value | Note |
|--------|-------|------|
| Layer drop ratio | 65% | Remove temporal kernels from last 65% of blocks during inference |
| $\lambda_{TL}$ | 1e-1 | Tracking loss weight |
| Kernel size $k$ | 3 | Temporal convolution kernel size |
| Intermediate dimension $m$ | 128 | Intermediate dimension in down-up architecture |

### Key Findings

- DeT achieves the best balance between motion fidelity and editing fidelity; MotionInversion achieves high motion fidelity but severely overfits appearance.
- Cross-category motion transfer (human → panda, train → boat) performs well, demonstrating that temporal kernels effectively extract category-agnostic motion patterns.
- The difficulty stratification of MTBench validates the expected trend: motion fidelity decreases and editing fidelity increases as motion complexity grows.

## Highlights & Insights

- The temporal 1D convolution design is theoretically grounded in the diagonal structure of 3D attention maps and is concise and efficient to implement.
- The dual functionality of temporal kernels—smoothing and learning—elegantly resolves the tension between decoupling and motion capture.
- Selectively removing temporal kernels during inference (retaining the first 35%) is an elegant engineering choice that balances motion transfer and text-guided editing.
- The MTBench construction methodology (distance-weighted sampling + clustering-based difficulty stratification) offers a valuable reference for future benchmark development.

## Limitations & Future Work

- Per-source-video fine-tuning of 500 steps (~1 hour) is required, precluding zero-shot motion transfer.
- Validation is limited to 49-frame videos; the effectiveness of motion transfer on longer videos remains unexplored.
- Foreground trajectory quality depends on CoTracker, which may be limited in complex occlusion scenarios.
- Although larger than prior benchmarks, MTBench's 100 videos may still be insufficient for exhaustive evaluation.

## Related Work & Insights

- **MotionDirector** and **SMA** represent the traditional paradigm of motion-appearance decoupling via U-Net temporal attention; DeT is the first to extend motion transfer to DiT architectures.
- The temporal kernel design can be generalized to other video editing tasks requiring spatiotemporal decoupling.
- The Dense Point Tracking Loss concept—correspondence between optical flow in pixel and feature space—can be extended to domains such as video style transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ The use of temporal kernels for simultaneous motion decoupling and learning is novel; benchmark construction methodology is rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validation across three DiT base models, comprehensive ablation studies, and new benchmark with new metrics.
- Writing Quality: ⭐⭐⭐⭐ Method design logic driven by visualization analysis is clear and well-motivated.
- Value: ⭐⭐⭐⭐ Provides foundational methods and an evaluation framework for motion transfer in the DiT era.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers](magicmirror_id-preserved_video_generation_in_video_diffusion_transformers.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)
- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)

<!-- RELATED:END -->
