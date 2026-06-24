---
title: >-
  [Paper Note] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming
description: >-
  [NeurIPS 2025][Model Compression][4D Gaussian Splatting] This paper proposes 4DGCPro, a hierarchical 4D Gaussian compression framework that achieves multi-bitrate progressive volumetric video streaming within a single model, via perception-weighted hierarchical Gaussian representation, motion-aware adaptive grouping, and end-to-end entropy-optimized training. The framework supports real-time decoding and rendering on mobile devices and surpasses existing SOTA in rate-distorti…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "4D Gaussian Splatting"
  - "Progressive Streaming"
  - "Volumetric Video Compression"
  - "Hierarchical Representation"
  - "Entropy Coding"
date: 2026-05-08
content_hash: cb6c0d3b5a67fdf2
---

# 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming

**Conference**: NeurIPS 2025
**arXiv**: [2509.17513](https://arxiv.org/abs/2509.17513)  
**Code**: [Project Page](https://mediax-sjtu.github.io/4DGCPro) (code to be released)  
**Area**: Model Compression
**Keywords**: 4D Gaussian Splatting, Progressive Streaming, Volumetric Video Compression, Hierarchical Representation, Entropy Coding

## TL;DR

This paper proposes 4DGCPro, a hierarchical 4D Gaussian compression framework that achieves multi-bitrate progressive volumetric video streaming within a single model, via perception-weighted hierarchical Gaussian representation, motion-aware adaptive grouping, and end-to-end entropy-optimized training. The framework supports real-time decoding and rendering on mobile devices and surpasses existing SOTA in rate-distortion performance.

## Background & Motivation

Volumetric video enables immersive free-viewpoint navigation but imposes enormous demands on bandwidth, storage, and real-time decoding—far exceeding those of 2D video. Existing methods face two core limitations:

1. **Lack of flexibility**: Most existing methods train separate models for each bitrate, incurring high storage costs and inability to adapt quality to dynamic network conditions—in stark contrast to scalable coding in 2D video (e.g., H.264/SVC).
2. **Insufficient decoding efficiency**: NeRF-based methods (e.g., HPC) offer compression capability but suffer from high decoding latency (121 ms), precluding real-time playback on mobile platforms. 3DGS-based methods render faster, but existing dynamic 3DGS compression approaches (e.g., 4DGC) fail in large-motion scenes due to rigid-body modeling constraints, and likewise do not support multi-bitrate delivery.

## Core Problem

How to achieve **multi-bitrate progressive streaming** of volumetric video within a **single model**, while guaranteeing real-time decoding and rendering on mobile devices and maintaining high-fidelity reconstruction in complex large-motion scenes?

The challenge lies in three levels of tension: (1) high compression ratio vs. high reconstruction quality; (2) single model vs. multi-bitrate support; (3) temporal modeling stability vs. compactness in large-motion scenes.

## Method

The core mechanism of 4DGCPro is to organize 3D Gaussians into a hierarchical structure according to visual importance, model inter-frame motion via rigid transformation plus residual deformation, and end-to-end optimize the rate-distortion performance of each layer during training, ultimately generating a progressive bitstream via the H.264 encoder.

### Overall Architecture

Input: multi-view video sequences → Output: progressive compressed bitstream, decoded layer by layer on the client side.

The pipeline consists of three stages:
1. **Keyframe hierarchical Gaussian representation** (Sec 3.1): Initial meshes are obtained via NeuS2; high-quality 3DGS is trained and then partitioned into $L=6$ layers by a perceptual importance metric $\Psi$, with lower layers retaining core structures and higher layers supplementing fine details.
2. **Hierarchical motion modeling + adaptive grouping** (Sec 3.2): Inter-frame motion is decomposed into rigid transformation (position + rotation) and residual deformation (scale + opacity + SH), with grouping boundaries determined adaptively based on motion magnitude.
3. **End-to-end entropy-optimized training + progressive coding** (Sec 3.3): Per-layer RD supervision and attribute-specific entropy estimation are introduced; after training, Gaussian attributes are flattened into 2D image sequences and encoded with H.264.

### Key Designs

1. **Perception-weighted importance metric $\Psi$**: $\Psi = \alpha + \lambda_\Psi \cdot S$, where $\alpha$ is opacity and $S = \frac{4}{3}\pi abc$ is the spatial volume of the Gaussian. This metric orthogonally combines geometric contribution (large volume → structurally important) and visual contribution (high opacity → rendering important). Experiments show that simple multiplication is inferior to weighted addition, and $\lambda_\Psi = 1 \times 10^5$ is the optimal balancing weight. Gaussians are sorted by $\Psi$ in descending order and partitioned into $L$ layers: the base layer $\mathbf{G}_1$ retains the most important Gaussians, and clients decode up to layer $l$ according to available bandwidth.

2. **Hierarchical motion modeling**:

    - **Rigid transformation**: A multi-resolution hash grid encodes the previous-frame Gaussian positions; two lightweight MLPs predict translation $\Delta\boldsymbol{\mu}_t$ and rotation $\Delta\mathbf{R}_t$ respectively. Position and rotation are updated cumulatively.
    - **Residual deformation**: Residuals $(\Delta\mathbf{s}_t, \Delta\alpha_t, \Delta\mathbf{f}_t)$ for scale, opacity, and SH coefficients are further learned on top of the rigid transformation, resolving artifacts caused by existing methods that model only rigid-body motion.

3. **Motion-aware adaptive grouping**: A new reference frame is initiated when the mean inter-frame Gaussian translation $\overline{\Delta\boldsymbol{\mu}_t}$ exceeds threshold $\tau_\mu$. This avoids the dilemma of fixed group lengths—long groups lead to error accumulation while short groups introduce data redundancy. Different thresholds are applied to different datasets (0.0025 for the self-collected dataset, 0.001 for HiFi4G, 0.01 for N3DV), automatically adapting to motion intensity.

4. **Attribute-specific entropy modeling**: Keyframe Gaussian attribute distributions are irregular (Fig. 3b), so FFT-accelerated KDE is used to estimate the PMF; inter-frame residual attributes naturally follow a Gaussian distribution (Fig. 3c), requiring only the mean and variance—an observation that substantially simplifies the training of inter-frame coding.

### Loss & Training

**Keyframe training** proceeds in two stages: pre-training for 12,000 steps with $\mathcal{L}_{color}$ → pruning low-opacity Gaussians (40%) → hierarchical RD optimization for 1,500 steps. The hierarchical RD loss is:

$$\mathcal{L}_{key} = \sum_{l=1}^{L} \lambda_l \left( \mathcal{L}_{color}^l + \lambda_{rate\_key} \mathcal{L}_{rate\_key}^l \right)$$

where $\lambda_l = 0.5/l$ (for $l < L$) or $1$ (for $l = L$), imposing stricter quality constraints on higher (detail) layers.

**Inter-frame training** also proceeds in two stages:
- Rigid transformation stage (800 steps): supervised solely by $\mathcal{L}_{color}$, with simulated quantization but no entropy constraint, to ensure position and rotation accuracy.
- Residual deformation stage (2,000 steps): entropy loss $\mathcal{L}_{rate\_inter}$ and temporal consistency regularization $\mathcal{L}_{reg}$ are added; the latter explicitly constrains temporal smoothness of residual attributes across frames.

Uniform noise $u \sim U(-\frac{q}{2}, \frac{q}{2})$ is injected during training to simulate quantization, ensuring gradient propagation.

**Encoding stage**: Positions are quantized with uint16/uint32 (precision-sensitive); other attributes use uint8. Attributes are flattened channel-wise into 2D image sequences and encoded with H.264 x264 (no B-frames, 3 reference frames, YUV 4:4:4, qp=10/20).

## Key Experimental Results

| Dataset | Metric | Ours (High) | Prev. SOTA | Gain |
|---------|--------|-------------|------------|------|
| 4DGCPro | PSNR/Size | 29.47 dB/1.31 MB | V³: 28.11 dB/1.60 MB | +1.36 dB, −18% size |
| HiFi4G | PSNR/Size | 36.38 dB/0.75 MB | V³: 36.26 dB/0.92 MB | +0.12 dB, −18% size |
| N3DV | PSNR/Size | 31.64 dB/0.64 MB | 4DGC: 31.58 dB/0.50 MB | +0.06 dB |
| 4DGCPro | BD-PSNR (vs. ReRF) | **4.20 dB** | HPC: 3.42 dB, V³: 1.90 dB | Surpasses all methods |
| HiFi4G | BD-PSNR (vs. ReRF) | **7.87 dB** | HPC: 5.84 dB, V³: 7.19 dB | Surpasses all methods |

**Efficiency comparison** (4DGCPro dataset):

| Metric | HPC | V³ | Ours (Mid) |
|--------|-----|----|------------|
| Decoding (ms) | 121 | 20 | **19** |
| Rendering (ms) | 231 | 2.8 | **2.5** |
| Training (min) | 93 | 0.97 | **4.3** |

**Mobile performance**: Full pipeline on iPad M2 runs at 43 ms (≈23 FPS) at high quality and 39 ms (≈26 FPS) at mid quality; iPhone A15 achieves 34 ms at high quality.

Key highlight: a single model supports High/Mid/Low three quality tiers (or arbitrary combinations of 6 layers), whereas HPC requires three separate models. Compared to HPC, 4DGCPro achieves a 3× compression ratio improvement at equivalent quality.

### Ablation Study

- **Importance metric $\Psi$**: Removing opacity → −0.98 dB; removing volume → −1.86 dB; multiplicative combination → −1.33 dB. Weighted addition (proposed) is optimal.
- **Adaptive vs. fixed grouping**: The best fixed group length (5 frames) still incurs +8.11% BD-BR and −0.25 dB BD-PSNR; fixed 1-frame (per-frame independent) → +48.37% BD-BR.
- **Number of layers $L$**: $L=4$ → −0.87 dB BD-PSNR; $L=6$ achieves the best balance (4.3 min training); $L=8$ yields only +0.09 dB with 28% longer training.
- **Entropy modeling**: Removing hierarchical supervision (H-S) → −2.89 dB BD-PSNR (largest contributor); removing simulated quantization (S-Q) → +4.36% BD-BR; using KDE for all attributes is feasible but adds 1.2 min per frame.
- **Motion decomposition**: Removing motion decomposition → Mid quality drops from 28.68 to 28.17 dB (−0.51 dB).
- **Hierarchical supervision**: Removing it has negligible effect on High quality (29.53 vs. 29.47 dB) but causes severe degradation at Mid/Low (26.49 vs. 28.68 / 24.98 vs. 27.69 dB), confirming that hierarchical supervision is critical for lower-layer Gaussians.

## Highlights & Insights

- **"One model for all" progressive streaming architecture**: This is the central contribution—the first single-model multi-bitrate solution in 3DGS-based volumetric video compression, genuinely addressing seamless quality switching under fluctuating bandwidth, with significant practical deployment value.
- **Concise yet effective perceptual importance metric**: $\Psi = \alpha + \lambda_\Psi S$ relies solely on two fundamental geometric attributes without any learned parameters or complex computation, yet ablations demonstrate its superiority over opacity-only, volume-only, or multiplicative alternatives.
- **Insightful exploitation of attribute distribution characteristics**: The observation that keyframe Gaussian attributes have irregular distributions (→ KDE) while inter-frame residuals are naturally Gaussian (→ mean/variance parameterization) enables a tailored entropy modeling strategy that is both efficient and intuitive.
- **Leveraging standard video codecs for last-mile encoding**: Flattening attributes into 2D image sequences for H.264 encoding directly exploits hardware codec acceleration, avoiding the difficulties of deploying custom decoders on mobile platforms.
- **Adaptive grouping** is simple and practical: using a motion magnitude threshold to automatically switch reference frames comprehensively outperforms fixed-length grouping.

## Limitations & Future Work

- **Non-trivial training time**: Hierarchical supervision requires rendering at each layer, making training 4.4× longer than V³ (4.3 min vs. 0.97 min), though still far shorter than HPC (93 min); further optimization is possible.
- **Dependence on dense multi-view input**: Dense synchronized camera arrays (e.g., 81 cameras) are required; performance degrades under sparse-view settings, limiting applicability to consumer-grade scenarios.
- **Relatively simple importance metric**: Only volume and opacity are considered, without accounting for view-dependent visibility (how many views observe a given Gaussian?) or semantic importance (foreground subjects vs. background), which may yield suboptimal results in scenes with large semantic saliency differences.
- **Limited evaluation on large-scale scenes**: The framework is primarily validated on person-centric scenes; scalability to large-scale indoor/outdoor environments remains uncertain.
- **Manual threshold selection for grouping**: The motion threshold $\tau_\mu$ differs across datasets (0.001–0.01) without an automatic determination mechanism.

## Related Work & Insights

Compared to **V³** (the most direct baseline): V³ uses fixed group lengths, leading to error accumulation in large groups and redundancy in small groups, and does not support progressive bitrate control. 4DGCPro addresses both issues via adaptive grouping and hierarchical representation. V³ fails on the Coser2 sequence due to NeuS2 initialization failure, which 4DGCPro resolves through a residual NeuS2 strategy. RD performance comprehensively surpasses V³ (BD-PSNR +2.3 dB on the 4DGCPro dataset).

Compared to **HPC**: HPC is a progressive encoding framework for NeRF-based methods with scalable coding capability, but suffers from high decoding latency (121 ms vs. 19 ms), precluding real-time operation, and cannot handle N3DV scenes with backgrounds. 4DGCPro decodes more than 6× faster than HPC.

Compared to **4DGC**: 4DGC is a predecessor from the same group (CVPR 2025) that also performs end-to-end RD optimization but supports only a single bitrate; its rigid-body modeling severely fails in large-motion scenes (PSNR of only 21.48 dB vs. 29.47 dB on the 4DGCPro dataset).

**Broader implications**:
- **Generality of progressive representation**: The "sort by importance → layer-wise enhancement" paradigm is not limited to streaming; it is also applicable to LOD rendering, interactive editing (coarse-layer edits propagated to fine layers), and even 3D generation (coarse-to-fine generation strategies).
- **Exploiting attribute distribution characteristics**: The use of KDE for irregular keyframe distributions and parametric modeling for approximately Gaussian residuals can generalize to other 3DGS tasks requiring probabilistic modeling (e.g., 3DGS uncertainty estimation).
- **Using H.264 as a last-mile encoder**: Directly leveraging the existing hardware codec ecosystem rather than developing custom decoders is an engineering-wise astute choice, and the same philosophy can be applied to other 3D representations targeting edge deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of progressive hierarchical Gaussians, adaptive grouping, and attribute-specific entropy modeling is novel, though individual components (hierarchical representation, motion decomposition, KDE entropy estimation) are not new in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks plus a self-collected dataset, six baselines, four ablation groups, multi-platform efficiency tests, and robustness validation—comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, clearly motivated, and richly illustrated; notation is occasionally dense and some symbols are not immediately intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses core deployment pain points (multi-bitrate + mobile real-time) in volumetric video streaming, with high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[NeurIPS 2025\] Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)
- [\[NeurIPS 2025\] Smooth Regularization for Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)
- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](../../ICML2025/model_compression/fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)

</div>

<!-- RELATED:END -->
