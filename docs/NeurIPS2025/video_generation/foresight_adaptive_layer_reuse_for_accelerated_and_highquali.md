---
title: >-
  [Paper Note] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation
description: >-
  [NeurIPS 2025][Video Generation][Adaptive Caching] This paper proposes Foresight, a training-free adaptive layer reuse framework that establishes per-layer MSE thresholds during a warmup phase and dynamically decides at inference time whether to reuse cached features or recompute each layer. Evaluated on 5 video generation models, Foresight achieves superior quality and speed trade-offs compared to static methods, with up to 2.23× acceleration.
tags:
  - NeurIPS 2025
  - Video Generation
  - Adaptive Caching
  - DiT Acceleration
  - Feature Reuse
  - Text-to-Video Generation
  - Training-Free
date: 2026-05-08
content_hash: 87d6ddf499fee694
---

# Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation

**Conference**: NeurIPS 2025
**arXiv**: [2506.00329](https://arxiv.org/abs/2506.00329)
**Code**: [https://github.com/STAR-Laboratory/foresight](https://github.com/STAR-Laboratory/foresight)
**Area**: Video Generation
**Keywords**: Adaptive Caching, DiT Acceleration, Feature Reuse, Text-to-Video Generation, Training-Free

## TL;DR

This paper proposes Foresight, a training-free adaptive layer reuse framework that establishes per-layer MSE thresholds during a warmup phase and dynamically decides at inference time whether to reuse cached features or recompute each layer. Evaluated on 5 video generation models, Foresight achieves superior quality and speed trade-offs compared to static methods, with up to 2.23× acceleration.

## Background & Motivation

**State of the Field**: Diffusion Transformers (DiTs) have become the dominant architecture for text-to-video generation, with models such as OpenSora, CogVideoX, and HunyuanVideo all built upon spatiotemporal DiT backbones. However, the $O(L^2)$ complexity of self-attention grows rapidly with resolution and frame count, and the typical requirement of 30–50 denoising steps results in prohibitively high inference latency.

**Limitations of Prior Work**: Feature caching is a training-free acceleration technique that reduces computation by reusing intermediate features across adjacent denoising steps. However, existing methods (Static, PAB, Δ-DiT, T-GATE, TeaCache) all adopt **static strategies**—applying uniform reuse across all layers at fixed intervals—ignoring the substantial variation in reuse potential across layers, prompts, and configurations.

**Root Cause**: By analyzing spatial feature MSE heatmaps across 28 DiT layers in OpenSora, the authors identify three critical observations: (1) early layers exhibit small feature changes and are safe to reuse, while later layers exhibit large changes and suffer significant quality degradation under aggressive reuse; (2) prompts depicting fast-changing scenes have substantially lower reuse potential than those depicting static scenes; (3) changing resolution from 240p to 720p noticeably alters the MSE patterns of the same layers. Static methods cannot adapt to any of these variations.

**Paper Goals**: To make adaptive per-step, per-layer decisions of "reuse or recompute" so as to achieve a superior Pareto frontier between speed and quality.

**Starting Point**: Use statistical measures of feature MSE as the reuse criterion. Per-layer thresholds are automatically learned during a warmup phase and compared against at runtime, requiring no training or architectural modification.

**Core Idea**: Replace static reuse intervals with dynamic per-layer MSE thresholds, enabling each layer at each step to independently decide whether to reuse cached features.

## Method

### Overall Architecture

Foresight divides the denoising process into two phases: a **Warmup Phase** and a **Reuse Phase**. During the Warmup Phase, all layers are computed normally to build the cache and establish per-layer thresholds. During the Reuse Phase, reuse and recomputation steps alternate; each recomputation step updates the reuse metric, and the next step makes independent per-layer decisions by comparing the metric against the threshold. The entire process only requires storing two outputs per DiT block (spatial + temporal) and does not modify model weights.

### Key Designs

1. **Warmup Phase and Adaptive Threshold Initialization**:

    - Function: Perform full computation for the first $W$ steps (default: 15%) to allow features to stabilize and establish per-layer reuse thresholds.
    - Mechanism: The threshold for each layer is computed as a geometrically weighted average of MSE over the last three warmup steps: $\lambda_x^l = \sum_{t=W-2}^{W} \frac{1}{10^{W-t}} \cdot \text{MSE}^l(t, t-1)$, with greater weight assigned to more recent steps. Thresholds naturally vary across layers, prompts, and resolutions.
    - Design Motivation: Static methods require manual tuning of reuse intervals. Foresight derives thresholds automatically from data. Layers with large MSE receive high thresholds, while layers with small MSE receive low thresholds, automatically enforcing frequent reuse in early layers and conservative reuse in later layers.

2. **Dynamic Decision Mechanism in the Reuse Phase**:

    - Function: Update the reuse metric $\delta$ at each recomputation step, and use it in the following step to make per-layer decisions by comparing against the threshold.
    - Mechanism: At each recomputation step, the MSE between the current feature and the cached feature is computed as the metric $\delta_x^l(t)$. If $\delta \leq \gamma \cdot \lambda_x^l$, the cache is reused in the next step; otherwise, the layer is recomputed. The scaling factor $\gamma \in (0, 2]$ controls the speed–quality trade-off.
    - Design Motivation: $\gamma$ provides a simple control knob—$\gamma=0.25$ minimizes reuse and yields high quality (PSNR 38), while $\gamma=2.0$ maximizes reuse and speed.

3. **Coarse-Grained Block-Level Caching**:

    - Function: Cache the output of entire DiT blocks rather than separately caching fine-grained attention/MLP outputs.
    - Mechanism: Only two block outputs per layer (spatial and temporal) are cached, resulting in a cache size of $2L \cdot H \cdot W \cdot F$—3× smaller than PAB's $6L \cdot H \cdot W \cdot F$.
    - Design Motivation: Block-level features across adjacent steps are already highly similar (cosine similarity >0.99). Fine-grained caching offers greater theoretical flexibility but incurs storage and management overhead that is not justified.

### Convergence Analysis

The authors prove that the error introduced by Foresight's reuse is bounded and controllable. At each reused layer, the error satisfies $\varepsilon_t^l \leq \gamma \cdot \lambda_x^l$, and the accumulated error over the entire denoising chain satisfies $\|\hat{x}_t - x_t^*\| \leq \varepsilon_{\text{tot}} / (1 - \rho)$, where $\rho = \max_s \sqrt{1 - \beta_s} < 1$. Tightening $\gamma$ allows the output to arbitrarily approach the baseline.

## Key Experimental Results

### Main Results (VBench, 550 prompts)

| Model | Method | VBench Acc | PSNR↑ | SSIM↑ | FVD↓ | Speedup |
|-------|--------|------------|-------|-------|------|---------|
| OpenSora | PAB | 75.32 | 25.67 | 0.85 | 541.53 | 1.26× |
| OpenSora | **Foresight** (N=1,R=2) | **75.90** | **29.67** | **0.90** | **306.66** | **1.28×** |
| OpenSora | **Foresight** (N=2,R=3) | 75.62 | 27.49 | 0.87 | 457.69 | **1.44×** |
| CogVideoX | PAB | 77.89 | 29.04 | 0.91 | 340.24 | 1.37× |
| CogVideoX | **Foresight** (N=1,R=2) | **77.94** | **34.75** | **0.95** | **130.65** | **1.46×** |
| CogVideoX | **Foresight** (N=2,R=3) | 77.84 | 28.45 | 0.87 | 531.99 | **1.63×** |

### Ablation Study

| Configuration | Latency (s) | PSNR↑ | Notes |
|---------------|-------------|-------|-------|
| PAB baseline | 19.88 | 28.12 | Static baseline |
| γ=0.25 | 20.50 (+0.62) | 38.09 (+9.97) | Minimal reuse, highest quality |
| γ=0.5 | 18.70 (−1.17) | 32.38 (+4.26) | Default configuration |
| γ=2.0 | 16.02 (−3.85) | 29.51 (+1.39) | Maximum reuse |
| N=3, R=4 | 14.79 (−5.08) | 29.03 (+0.91) | More aggressive reuse, still outperforms PAB |

### Key Findings

- **Late layers are the quality bottleneck**: Dividing layers into early/middle/late groups, static reuse of late layers causes the largest quality degradation; Foresight automatically schedules more frequent recomputation for late layers.
- **Scales to recent models**: On HunyuanVideo, Foresight achieves 1.62× speedup with PSNR 41.79, substantially outperforming TeaCache's 37.31; on Wan-2.1, it achieves 2.23× speedup.
- **Quality-matched comparison**: When constrained to match PAB's output quality, Foresight achieves 1.68×, 1.58×, and 1.95× speedups on OpenSora, Latte, and CogVideoX, respectively.
- **Cross-task generalization**: Applied to the FLUX text-to-image model, Foresight also achieves approximately 2× acceleration.

## Highlights & Insights

- **Elegant adaptive threshold design**: Thresholds are automatically derived from MSE statistics during the warmup phase, requiring no manual tuning. Different prompts, resolutions, and layers naturally yield different thresholds—a design principle transferable to any scenario involving "cache or not" decisions.
- **Counterintuitive advantage of coarse-grained caching**: PAB carefully designs hierarchical broadcast strategies for spatial, temporal, and cross-attention separately, yet Foresight's block-level coarse-grained reuse outperforms it—because the adaptive decision mechanism compensates for the coarser granularity.
- **Theoretical and empirical coherence**: Convergence guarantees provide theoretical grounding, while extensive validation across 5 models demonstrates strong practical deployability.

## Limitations & Future Work

- The maximum achievable speedup is bounded by the $N$ and $W$ configuration, reaching at most approximately 2.23×—far below the 10–50× of step compression or distillation methods (though the two approaches are orthogonal and can be combined).
- The current block-level reuse granularity could potentially be extended to attention-head or token level for further gains.
- Threshold initialization relies on the quality of MSE estimates during the warmup phase; very short videos with few warmup steps may yield unstable thresholds.
- The combination with step compression methods such as consistency distillation has not been explored.

## Related Work & Insights

- **vs. PAB**: PAB empirically fixes broadcast ranges per attention type and requires model-specific tuning; Foresight is data-driven and adaptive, operating with a single set of parameters across all models.
- **vs. TeaCache**: TeaCache uses the change in timestep embeddings as the caching criterion, whereas Foresight uses actual feature MSE, which more directly reflects layer-wise feature variation.
- **vs. Δ-DiT**: Δ-DiT caches residual offsets rather than full features but remains a static scheme unable to adapt to variations in prompt or configuration.

## Rating

- Novelty: ⭐⭐⭐ The idea of adaptive caching is not entirely new, but the threshold design and the two-phase warmup-reuse framework offer genuine contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five video models + one image model × three benchmarks × multiple configurations × ablations—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear and systematic; the convergence analysis is a notable addition.
- Value: ⭐⭐⭐⭐ A plug-and-play, training-free acceleration solution with high practical engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[ICCV 2025\] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](../../ICCV2025/video_generation/dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](../../ICCV2025/video_generation/motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](../../ICCV2025/video_generation/magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)

<!-- RELATED:END -->
