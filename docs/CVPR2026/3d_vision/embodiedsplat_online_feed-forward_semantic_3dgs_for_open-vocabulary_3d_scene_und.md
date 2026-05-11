---
title: >-
  [Paper Note] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes EmbodiedSplat, the first online feed-forward semantic 3DGS framework. It achieves memory-efficient per-Gaussian semantic representation via a sparse coeff…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "open-vocabulary scene understanding"
  - "online reconstruction"
  - "feed-forward 3DGS"
  - "semantic embedding"
date: 2026-05-08
content_hash: 19df85714fcfd3d8
---

# EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.04254](https://arxiv.org/abs/2603.04254)
**Code**: Available (project page: EmbodiedSplat.io)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, open-vocabulary scene understanding, online reconstruction, feed-forward 3DGS, semantic embedding

## TL;DR

This paper proposes EmbodiedSplat, the first online feed-forward semantic 3DGS framework. It achieves memory-efficient per-Gaussian semantic representation via a sparse coefficient field and a CLIP global codebook, and integrates 3D geometry-aware features to enable full-scene open-vocabulary 3D understanding at 5–6 FPS over 300+ streaming frames.

## Background & Motivation

### 1. State of the Field

Embodied AI tasks such as robotic manipulation and navigation require agents to understand 3D scenes in real time during exploration. 3D Gaussian Splatting (3DGS), owing to its explicit structure and real-time rendering capability, has become a dominant scene representation paradigm. A large body of recent work distills semantic knowledge from 2D foundation models such as CLIP into 3DGS to enable open-vocabulary 3D scene understanding.

### 2. Limitations of Prior Work

Existing semantic 3DGS methods suffer from two fundamental limitations:

- **Per-scene optimization**: Methods such as LangSplat, LEGaussians, OpenGaussian, and Dr. Splat all require hours of per-scene optimization (2–6 hours) and cannot generalize to unseen scenes.
- **Offline setting**: These methods require a complete set of images collected in advance and cannot handle streaming input, making them unsuitable for online exploration.
- The few online methods (e.g., Online-LangSplat) support streaming input but still rely on heavy per-scene SLAM optimization, achieving only 1.12 FPS.
- Feed-forward methods (e.g., LSM, SIU3R) generalize across scenes but support only 2–3 input views, preventing full-scene reconstruction.

### 3. Root Cause

Embodied scene understanding demands that a 3D perception model simultaneously satisfy five requirements: **online**, **real-time**, **high generalizability**, **full-scene understanding**, and **open-vocabulary understanding**. Existing methods satisfy at most 2–3 of these. In particular, storing full CLIP features for every Gaussian (often exceeding 1 million) incurs enormous memory overhead, while existing compression approaches (autoencoders, product quantization) require pretraining and introduce information loss.

### 4. Paper Goals

To design an online feed-forward semantic 3DGS framework that reconstructs open-vocabulary semantic 3D Gaussian fields at near-real-time speed from 300+ streaming frames, while maintaining memory efficiency and preserving the full semantic capacity of CLIP.

### 5. Starting Point

Rather than following the conventional "3D rendering to 2D" distillation pipeline, the paper adopts a direct "2D to 3D" lifting approach: pixel-level CLIP features are back-projected into 3D space, dense per-Gaussian CLIP vector storage is replaced by a **sparse coefficient field with a global codebook**, and a 3D U-Net injects geometric priors as a complement.

### 6. Core Idea

The number of semantically distinct entities in a scene is far smaller than the number of Gaussians. Consequently, instance-level CLIP features can be used to construct a global codebook, and each Gaussian need only store a small number of codebook indices and sparse weights to reconstruct its full semantic representation.

## Method

### Overall Architecture

EmbodiedSplat is built on the pretrained FreeSplat++ (a feed-forward 3DGS model), with its inference pipeline adapted for online operation. At each time step, the framework receives the current frame and $N$ reference frames, encodes them via a CNN encoder into per-pixel Gaussian triplets (position, confidence, latent), and integrates the new Gaussians into the global Gaussian set via an online fusion strategy. The core contribution lies in binding two types of CLIP features to each Gaussian:

- **2D semantic features**: stored using a sparse coefficient field with a CLIP global codebook, preserving the full open-vocabulary capability of CLIP.
- **3D geometry-aware features**: aggregated from point cloud features via a 3D sparse U-Net, injecting 3D geometric priors.

During inference, the two feature types are integrated via a geometric mean to yield the final classification probabilities.

### Key Designs

#### Module 1: CLIP Global Codebook

- **Function**: Accumulates instance-level CLIP features from all frames into a global codebook that serves as a set of semantic basis functions.
- **Mechanism**: FastSAM is applied to each frame to extract instance masks; pixel-level CLIP features within each instance are average-pooled to obtain instance-level features, which are appended to the codebook across time steps. The number of codebook entries $K$ is far smaller than the number of Gaussians $M$ (in experiments, $K \approx 8.7\text{K}$ vs. $M \approx 3.2\text{M}$).
- **Design Motivation**: The number of semantically distinct entities in a scene is limited, so storing 512/768-dimensional CLIP vectors for each Gaussian is unnecessary. Unlike per-scene optimized codebooks, this codebook requires no pretraining and uses raw CLIP features directly, with zero information loss.

#### Module 2: Online Sparse Coefficient Field

- **Function**: Maintains an index cache and a weight cache of length $L$ for each Gaussian, replacing dense CLIP vectors.
- **Mechanism**: Each Gaussian stores $L-1$ codebook indices and corresponding weights ($L=6$); its semantic feature is reconstructed via sparse linear combination: $\mathbf{s}_g^T(i) = \sum_{j=1}^{L-1} \Omega_g^T(i,j) \cdot \mathbf{C}^T(\mathbf{I}_g^T(i,j))$
- **Online update strategy** (Algorithm 1): During fusion, local Gaussian codebook indices are appended to the global cache, weights are updated via confidence-weighted averaging, and only the top $L-1$ entries by weight are retained at each step. This removes low-confidence noisy indices while keeping cache size fixed (only 10 scalars per Gaussian).
- **Design Motivation**: Each Gaussian requires only $2(L-1)=10$ numbers rather than 512/768 dimensions, reducing memory from 2295 MB to 148 MB while supporting incremental online updates.

#### Module 3: Geometry-Aware 3D Semantic Features

- **Function**: Constructs CLIP features enriched with 3D geometric priors to compensate for the limited spatial awareness of 2D features.
- **Mechanism**: Gaussian latents are summed with projected CLIP features and passed through a 3D sparse U-Net with a memory-based adapter, producing compact 64-dimensional 3D features. A GRU network aggregates temporal information during fusion.
- **Design Motivation**: While 2D CLIP features are semantically rich, they lack explicit 3D priors (as they originate from 2D images), whereas 3D features encode the geometric structure of the point cloud. Their complementarity improves overall performance (Table 3 shows consistent gains across all metrics when combined).

#### Module 4: Codebook-Based Cosine Similarity

- **Function**: Accelerates text-to-Gaussian matching during inference.
- **Mechanism**: Exploiting the linearity of sparse linear combinations and inner products, cosine similarities between the text query and all $K$ codebook entries are precomputed in $O(KD)$, after which each Gaussian requires only $L-1$ weighted additions. Total complexity is reduced from $O(MD)$ to $O(KD + M(L-1))$.
- **Design Motivation**: Per-Gaussian cosine similarity computation over millions of Gaussians takes 14.35 ms; the codebook-based approach requires only 1.18 ms, yielding approximately a **14× speedup**.

### Loss & Training

- **Loss function**: Only a 2D–3D cosine similarity loss is used: $\mathcal{L}_{cos} = 1 - \cos(\mathbf{s}_g^T, \mathcal{D}^{sem}(\hat{\mathbf{g}}_g^T))$, without any label supervision.
- **Training strategy**: Two-stage training — (1) *Warm-up*: single-view perception model trained for 100K iterations without the memory adapter; (2) *Fine-tuning*: streaming multi-frame input (8–10 randomly sampled consecutive frames) with the memory adapter, trained for 300K iterations.
- FreeSplat++ parameters are frozen; only the 3D U-Net and memory adapter are optimized.
- During inference, 2D and 3D features are integrated via a geometric mean: $\mathbf{P} = \max(\mathbf{P}^{2D}, \mathbf{P}^{3D})^\tau \cdot \min(\mathbf{P}^{2D}, \mathbf{P}^{3D})^{1-\tau}$

## Key Experimental Results

### Main Results

**Table 1: 3D Semantic Segmentation Performance Comparison** (ScanNet / ScanNet200 / ScanNet++)

| Method | Type | ScanNet-10 mIoU | ScanNet-19 mIoU | ScanNet200-70 mIoU | ScanNet++ mIoU | Recon. Time | Setting |
|--------|------|---------|---------|----------|---------|----------|------|
| LangSplat | 2D | 6.52 | 1.34 | 0.72 | 2.21 | ~6 hr | Per-scene / Offline |
| Online-LangSplat | 2D | 7.13 | 3.45 | 2.45 | 4.51 | 5.4 min | Per-scene / Online |
| OpenGaussian | 3D | 29.50 | 22.52 | 15.15 | 25.65 | ~2.5 hr | Per-scene / Offline |
| Dr. Splat | 3D | 39.21 | 28.38 | 19.29 | 39.85 | ~2 hr | Per-scene / Offline |
| Occam's LGS | 3D | 42.14 | 30.49 | 20.32 | 34.08 | ~2 hr | Per-scene / Offline |
| **EmbodiedSplat (RGB)** | 3D | **49.81** | **46.22** | **31.16** | 41.93 | **8 min** | Generalizable / Online |
| **EmbodiedSplat-fast** | 3D | 47.86 | 41.03 | 30.46 | 45.53 | **1 min 10 s** | Generalizable / Online |
| EmbodiedSplat (RGB-D) | 3D | 57.41 | 52.12 | 34.75 | 44.03 | 8 min | Generalizable / Online |

**Table 2: Cross-Domain 3D Semantic Segmentation**

| Method | ScanNet++ → ScanNet (19-cls) mIoU | ScanNet → ScanNet++ (20-cls) mIoU | ScanNet → Replica (48-cls) mIoU |
|--------|---------|---------|---------|
| Dr. Splat | 28.38 | 39.85 | 14.47 |
| Occam's LGS | 30.49 | 34.08 | 16.19 |
| EmbodiedSplat (RGB) | **45.32** | 30.65 | 9.88 |
| EmbodiedSplat (RGB-D) | **50.80** | **44.14** | 11.42 |

### Ablation Study

**2D–3D feature complementarity** (Table 3):

- 2D features only: ScanNet-19 mIoU 45.09
- 3D features only: ScanNet-19 mIoU 45.39
- 2D + 3D combined: ScanNet-19 mIoU **46.22** (+1.13 gain)

**Codebook cosine similarity speedup** (Table 4):

- Per-Gaussian computation: 14.35 ms
- Codebook-accelerated: 1.18 ms (**14× speedup**)

**Memory efficiency comparison** (Table 5):

- Occam's LGS (raw 512-dim): 2295 MB, no information loss
- Dr. Splat (PQ quantization): 173 MB, information loss, requires pretraining
- LangSplat (autoencoder compressed to 3-dim): 30 MB, severe information loss
- **EmbodiedSplat** (sparse coefficient field): **148 MB, no information loss, no pretraining required**

**Effect of cache size $L$** (Table 6): $L=2$→44.38, $L=4$→45.01, $L=6$→45.09, $L=11$→45.08; $L=6$ provides the optimal trade-off.

### Key Findings

1. 2D-based methods (e.g., LangSplat) perform poorly under direct 3D query evaluation, because linear interpolation during rendering severely degrades the transfer of CLIP semantics to individual Gaussians.
2. The feed-forward design reduces reconstruction time from hours to minutes (8 min vs. 2–6 hr); the fast variant further achieves 1 min 10 s (5.18 FPS).
3. Cross-domain experiments reveal the critical role of depth estimation: ScanNet→ScanNet++ drops by 11.28 mIoU due to difficult regions such as ceilings, and this is largely recovered with a depth sensor (44.14 vs. 44.03).
4. A large domain gap exists in the real-to-synthetic transfer (ScanNet→Replica), where the feed-forward approach underperforms per-scene optimization methods.

## Highlights & Insights

1. **Elegant sparse coefficient field design**: Each Gaussian requires only 10 numbers (5 indices + 5 weights) in place of a 512-dimensional CLIP vector, achieving a 15× memory reduction while being mathematically equivalent to a sparse reconstruction of the original feature — with no pretraining and no information loss.
2. **Elegant derivation of codebook acceleration**: Exploiting the linearity of sparse combinations and inner products, per-Gaussian search is transformed into per-codebook precomputation plus lightweight weighted summation, yielding a 14× speedup at virtually no cost.
3. **Instructive 2D–3D complementarity**: 2D features are semantically rich while 3D features carry geometric priors. Combining cosine similarity distillation with geometric mean integration is both simple and effective.
4. **Well-designed online fusion algorithm**: Confidence-weighted updates combined with top-$k$ pruning ensure semantic accuracy while maintaining a fixed cache size, making the approach well-suited for streaming scenarios.

## Limitations & Future Work

1. **Limited cross-domain generalization**: Performance drops substantially in real-to-synthetic transfer (ScanNet→Replica), indicating that the feed-forward model is sensitive to domain shift.
2. **Depth estimation as a bottleneck**: In RGB mode, cross-dataset depth estimation discrepancies cause significant performance degradation (−11 mIoU for ScanNet→ScanNet++), suggesting practical deployments may depend on depth sensors.
3. **Continuously growing codebook**: The global codebook grows by appending entries at each time step; in long-horizon exploration (well beyond 300 frames), codebook compression or deduplication strategies may be necessary.
4. **Validation limited to indoor scenes**: Experiments are confined to indoor datasets such as ScanNet and Replica; performance in large-scale outdoor settings remains unknown.
5. **Accuracy trade-off in the fast variant**: Removing the 3D U-Net achieves 5–6 FPS but incurs a 2–4 mIoU drop on some metrics.

## Related Work & Insights

- **FreeSplat++**: The backbone feed-forward 3DGS model; its online fusion design (GRU + confidence-weighted updates) is generalizable to other 3DGS enhancement tasks.
- **Dr. Splat / Occam's LGS**: Representative methods following the direct feature lifting paradigm, consistent with the 2D→3D back-projection approach in this paper, but constrained to per-scene optimization.
- **OpenScene / PLA**: Point cloud + foundation model distillation methods; the 3D U-Net module in this paper draws on their 3D backbone designs.
- **Inspiration for future work**: Codebook + sparse coefficients can be extended to efficient storage of other high-dimensional features (e.g., DINOv2, SAM features).

## Rating

- Novelty: ⭐⭐⭐⭐ — The sparse coefficient field + global codebook compression scheme is elegant, though the overall framework leans toward integrative innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three major datasets, cross-domain evaluation, and comprehensive memory/speed/ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; the online update algorithm is described rigorously.
- Value: ⭐⭐⭐⭐ — The practical positioning of online feed-forward semantic 3DGS is well-defined, with direct implications for embodied scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp3d_joint_open_vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
