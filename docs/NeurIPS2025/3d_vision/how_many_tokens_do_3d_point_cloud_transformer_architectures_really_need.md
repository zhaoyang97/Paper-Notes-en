---
title: >-
  [Paper Note] How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?
description: >-
  [NeurIPS 2025][3D Vision][Point Cloud Transformer] This paper systematically demonstrates that 90–95% of tokens in 3D point cloud Transformers (e.g., PTv3, Sonata) are redundant, and proposes gitmerge3D — a globally informed graph-based token merging method that achieves up to 5.3× FLOPs reduction and 6.4× memory savings with negligible accuracy loss, via an energy-score-driven adaptive merging strategy.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Point Cloud Transformer
  - Token Redundancy
  - Token Merging
  - 3D Semantic Segmentation
  - Computational Efficiency
date: 2026-05-08
content_hash: 78d42f2212149378
---

# How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?

**Conference**: NeurIPS 2025
**arXiv**: [2511.05449](https://arxiv.org/abs/2511.05449)
**Code**: [https://gitmerge3d.github.io](https://gitmerge3d.github.io)
**Area**: 3D Vision
**Keywords**: Point Cloud Transformer, Token Redundancy, Token Merging, 3D Semantic Segmentation, Computational Efficiency

## TL;DR
This paper systematically demonstrates that 90–95% of tokens in 3D point cloud Transformers (e.g., PTv3, Sonata) are redundant, and proposes gitmerge3D — a globally informed graph-based token merging method that achieves up to 5.3× FLOPs reduction and 6.4× memory savings with negligible accuracy loss, via an energy-score-driven adaptive merging strategy.

## Background & Motivation

**Background**: Transformer architectures — particularly Point Transformer v3 (PTv3) — have become the dominant backbone for 3D point cloud understanding, achieving state-of-the-art performance on semantic segmentation, object detection, and 3D reconstruction. PTv3 serializes point clouds into 1D sequences via space-filling curves and performs self-attention within local partitions.

**Limitations of Prior Work**: Despite architectural optimizations in PTv3 (serialization replacing KNN to reduce inference time by 28%, removal of relative positional encoding to save 26% compute), the model still processes a large number of redundant tokens in self-attention. Each partition uses 1024 tokens by default, and attention complexity scales as $\mathcal{O}(N^2)$, resulting in high FLOPs and memory consumption.

**Key Challenge**: It is widely assumed that dense tokenization is critical for 3D Transformer performance. In reality, 3D point cloud data is spatially sparse and fine-grained — neighboring points on the same object surface have highly similar features, leading to substantial redundancy.

**Goal**: (1) Quantify the degree of token redundancy in 3D point cloud Transformers; (2) Design efficient token merging strategies tailored to 3D data.

**Key Insight**: The paper adapts token merging/pruning techniques from 2D vision Transformers to 3D point clouds, finding through experiments that generic methods can already merge 50% of tokens without accuracy drop. The authors then design a dedicated merging strategy exploiting 3D spatial locality and attention saliency to enable more aggressive compression (95–99%).

**Core Idea**: 3D point cloud Transformers are severely over-tokenized; retaining only 5–10% of the most informative tokens is sufficient to maintain near-identical performance.

## Method

### Overall Architecture
Token merging operations are inserted before each attention layer in PTv3. Merged tokens participate in attention computation, after which an unmerge operation restores the original resolution for compatibility with subsequent MLP layers. The method comprises two components: (1) global-informed energy score computation; and (2) energy-based adaptive merging strategy.

### Key Designs

1. **Global-Informed Energy Score**:

    - Function: Evaluates the "information content" of each token to guide which tokens should be merged.
    - Mechanism: A bipartite graph $G = (\mathcal{V}, \mathcal{E})$ is constructed, where the vertex set consists of all tokens $x_i$ and all partition centroids $\bar{P}_j = \frac{1}{|\mathcal{P}_j|} \sum_{x_k \in \mathcal{P}_j} x_k$. The energy score of each token is defined as the negation of its average cosine similarity to all partition centroids: $E(x_i) = -\frac{1}{|\mathcal{N}(x_i)|} \sum_{\bar{P}_j \in \mathcal{N}(x_i)} \cos(x_i, \bar{P}_j)$. Tokens with low energy (more consistent with the global structure) are considered redundant and mergeable; tokens with high energy (inconsistent with global structure) carry more unique information and should be retained.
    - Design Motivation: Generic merging methods (e.g., ToMe) rely solely on local token similarity without considering global structure. The energy score captures global token distinctiveness through its relationship with all partition centroids, making it better suited to 3D scenes with non-uniform density.

2. **Adaptive Partition-Level Merging Strategy**:

    - Function: Adaptively selects the merging ratio per partition based on its average energy — high-energy partitions are merged conservatively, low-energy partitions aggressively.
    - Mechanism: Partition importance is defined as $E(\mathcal{P}) = \frac{1}{|\mathcal{P}|} \sum_{x \in \mathcal{P}} E(x)$. If $E(\mathcal{P}) > \tau$, a moderate merging ratio $r$ is applied; otherwise, an aggressive ratio $r^+$ ($r^+ \gg r$) is used. The threshold is fixed at $\tau = 0.2$ and generalizes across all datasets without task-specific tuning.
    - Design Motivation: Information density varies drastically across 3D scenes (e.g., object boundaries vs. large planar surfaces). A uniform merging ratio either loses detail in high-information regions or is insufficiently aggressive in low-information regions. The adaptive strategy strikes an effective balance.

3. **Spatially Aware Token Merging Execution**:

    - Function: Executes the actual token merging within each partition.
    - Mechanism: Each partition is evenly divided into bins; destination tokens are randomly selected within each bin, and source tokens are merged into destinations via feature averaging. Using Value features as the merging metric and performing merging independently per attention head yields the best results. An unmerge function $f^{-1}$ is then applied to restore the original resolution.
    - Design Motivation: Random destination selection within bins enforces spatial locality — merged tokens are proximate in 3D space — preventing erroneous merging of spatially distant tokens.

### Loss & Training
The method is primarily designed as an off-the-shelf approach applicable at inference time without retraining. For scenarios requiring maximum accuracy, only 10% of the original training epochs of fine-tuning are needed (updating only MLP layers adjacent to the attention layers), after which the model not only recovers but may exceed baseline performance.

## Key Experimental Results

### Main Results — 3D Semantic Segmentation (mIoU)

| Method | ScanNet Val | ScanNet200 Val | S3DIS Area5 |
|--------|-------------|----------------|-------------|
| PTv3 (original) | 77.6 | 35.2 | 74.7 |
| - Random Drop (80%) | 70.1 | 31.1 | 73.4 |
| - FPS (80%) | 71.2 | 32.4 | 70.9 |
| - VoxelGrid Down. (80%) | 72.1 | 32.2 | 69.1 |
| - **Ours (off-the-shelf, 80%)** | **77.0** | **34.4** | **72.3** |
| - **Ours (fine-tuned, 80%)** | **77.4** | **35.2** | **74.3** |
| PTv3-Sonata (original) | 79.0 | 30.4 | 72.2 |
| - **Ours (off-the-shelf, 80%)** | **77.5** | **28.8** | **72.8** |
| - **Ours (fine-tuned, 80%)** | **78.9** | **30.9** | **73.5** |

### Efficiency Gains (PTv3, 90% merging)

| Metric | PTv3 Original | PTv3 + Ours | Improvement |
|--------|--------------|-------------|-------------|
| FLOPs | 107.5 GFLOPs | 19.9 GFLOPs | **5.3×** |
| Memory | 10.12 GB | 1.6 GB | **6.4×** |
| NuScenes Memory | 6.20 GB | 0.92 GB | **6.7×** |
| NuScenes FLOPs | 101.68 | 32.45 | **3.1×** |

### 3D Reconstruction (SplatFormer + Ours, 90% merging)

| Dataset | SplatFormer Orig. PSNR | + Ours PSNR | SSIM Diff. |
|---------|----------------------|-------------|-----------|
| GSO-OOD | 24.71 | ~24.6 | < 0.01 |
| Objaverse-OOD | 22.43 | ~22.3 | < 0.01 |
| RealWorld-OOD | 24.33 | ~24.2 | < 0.01 |

Merging 90% of tokens results in virtually unchanged 3D reconstruction quality, while still substantially outperforming methods such as 3DGS and 2DGS.

### Key Findings
- **Core Finding**: Even after merging 90–95% of tokens, the PCA feature representations of PTv3 remain almost identical. Most prediction changes are confined to a small number of points at object boundaries.
- Using Value features as the merging metric combined with per-head independent merging yields the best performance (ScanNet mIoU 76.98 vs. 76.27 for the Q-based variant).
- The energy threshold $\tau = 0.2$ generalizes across all tasks and datasets without requiring task-specific tuning.
- On SpatialLM object detection, merging 80% of tokens not only avoids accuracy degradation (F1 O50 improves from 0.1894 to 0.2006) but also reduces memory from 12.36 GB to 2.53 GB.

## Highlights & Insights
- **Challenging a Core Assumption**: The widely accepted assumption that dense tokenization is essential for 3D Transformers is systematically refuted through controlled experiments. The finding that 90% of tokens can be discarded has far-reaching implications for the design of 3D foundation models.
- **Global Energy Score Design**: Measuring token informativeness via its correlation with all partition centroids more faithfully reflects the global structural characteristics of 3D scenes than local similarity metrics. This idea is generalizable to other modalities with spatial redundancy, such as video Transformers.
- **Off-the-Shelf Applicability**: The method can be directly applied to any PTv3-based model (Sonata, SplatFormer, SpatialLM) without retraining, offering substantial practical value.

## Limitations & Future Work
- The method is primarily validated on PTv3 and its variants; token redundancy in other 3D Transformer architectures (e.g., OctFormer, Swin3D) remains unexplored.
- Although $\tau = 0.2$ is claimed to generalize across datasets, extreme scenarios (e.g., very dense industrial point clouds or highly sparse aerial LiDAR) may require adjustment.
- The unmerge step employs a simple copy strategy to restore original resolution, which may lose local detail at fine boundaries.
- While a 99% merging ratio is feasible, accuracy begins to degrade noticeably; a range of 80–95% is a more prudent choice for practical deployment.

## Related Work & Insights
- **vs. ToMe (Token Merging)**: ToMe uses bipartite soft matching to merge 2D tokens and is a general-purpose method. This paper improves upon ToMe for 3D settings through a better merging metric (global energy score) and an adaptive strategy (partition-level), achieving clearly superior results at 95% merging ratio.
- **vs. Traditional Downsampling (Random, FPS, VoxelGrid)**: These input-level methods suffer significant accuracy drops at high compression ratios (ScanNet mIoU 70.1–72.1 vs. 77.0 for the proposed method), as they discard entire points along with their geometric information. The proposed method operates at the feature level, preserving more semantic information.
- **vs. PiToMe/ALGM**: These advanced token merging methods were also evaluated but, being designed for 2D classification, do not support feature restoration for dense prediction tasks and underperform the proposed method on 3D segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study revealing token redundancy in 3D Transformers with a dedicated merging solution; the finding itself is of significant value
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Spans segmentation, reconstruction, and detection tasks across multiple backbones and datasets with detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Observations are clear and experiments are rich, though figure references in the method description are occasionally inconsistent
- Value: ⭐⭐⭐⭐⭐ Directly and significantly informs efficiency optimization for 3D foundation models; off-the-shelf applicability greatly enhances practical utility

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[CVPR 2026\] LitePT: Lighter Yet Stronger Point Transformer](../../CVPR2026/3d_vision/litept_lighter_yet_stronger_point_transformer.md)
- [\[NeurIPS 2025\] Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)

<!-- RELATED:END -->
