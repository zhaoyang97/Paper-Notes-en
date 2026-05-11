---
title: >-
  [Paper Note] Class-Partitioned VQ-VAE and Latent Flow Matching for Point Cloud Scene Generation
description: >-
  [AAAI2026][3D Vision][point cloud generation] This paper proposes a Class-Partitioned VQ-VAE (CPVQ-VAE) and a Latent Flow Matching Model (LFMM)…
tags:
  - "AAAI2026"
  - "3D Vision"
  - "point cloud generation"
  - "scene generation"
  - "VQ-VAE"
  - "flow matching"
  - "codebook partitioning"
date: 2026-05-08
content_hash: 33bf60e5e38e0106
---

# Class-Partitioned VQ-VAE and Latent Flow Matching for Point Cloud Scene Generation

**Conference**: AAAI2026
**arXiv**: [2601.12391](https://arxiv.org/abs/2601.12391)
**Code**: [ddsediri/CPVQ-VAE-LFMM](https://github.com/ddsediri/CPVQ-VAE-LFMM)
**Area**: 3D Vision
**Keywords**: point cloud generation, scene generation, VQ-VAE, flow matching, codebook partitioning

## TL;DR

This paper proposes a Class-Partitioned VQ-VAE (CPVQ-VAE) and a Latent Flow Matching Model (LFMM), achieving the first purely generative point cloud scene generation method that requires no external database retrieval, reducing Chamfer Distance by 70.4% on complex living room scenes.

## Background & Motivation

Existing 3D scene generation methods (e.g., Diffuscene) typically generate only bounding box parameters and latent features for each object in multi-class, multi-object complex scenes, and then retrieve the nearest object mesh from a predefined database using L2 distance. This pipeline has two critical bottlenecks:

1. **Unreliable generated latent codes**: Object latents produced directly by diffusion models frequently diverge from the target class, causing the VAE decoder to reconstruct point clouds that are severely inconsistent with the intended category (e.g., decoding a chair as a table).
2. **Dependency on external databases**: The retrieval paradigm inherently limits generation diversity and cannot produce novel object shapes not seen in the training set.

The authors observe that Diffuscene exhibits an extremely high decoding failure rate on complex scenes (e.g., living rooms), while methods such as ATISS abandon object latent generation entirely and rely solely on object size for retrieval. This motivates the need for a new architecture that guarantees both class consistency and reliable decoding.

## Core Problem

How can complete 3D scenes be generated directly as point clouds—with correct category labels and shapes—without relying on an external object database? Specifically, three challenges must be addressed:

- How to ensure generated latents are consistent with target object classes
- How to mitigate codebook collapse in VQ-VAE under a class-partitioned setting
- How to efficiently sample complete scene layouts from noise

## Method

### Overall Architecture

The system operates in two stages: (1) LFMM generates bounding box parameters (translation, rotation, size), class vectors, and 32-dimensional latent features for each object in the scene; (2) CPVQ-VAE maps the generated class and features to codebook entries via class-aware inverse lookup, and decodes them into point clouds.

### CPVQ-VAE: Class-Partitioned Vector-Quantized VAE

In standard VQ-VAE, the codebook is label-free and quantization searches for the nearest neighbor over the entire codebook. The key innovation of CPVQ-VAE is **partitioning the codebook by class**:

- Total codevector count $N_K = N_c \times N_q$, where $N_c$ is the number of classes and $N_q$ is the number of codevectors allocated per class.
- An indicator function $\mathbf{1}(c,k)$ is introduced during quantization to restrict nearest-neighbor search to the corresponding class partition.
- This guarantees that each quantized feature $z^{q_c}$ belongs to the correct class.

The training loss comprises three terms: Chamfer distance reconstruction loss (weight $\lambda_{CD}=10$), codebook alignment loss, and commitment loss—identical in form to standard VQ-VAE but operating over the class-partitioned codebook.

### Class-Aware Running Average Update

To address codebook collapse (many codevectors becoming "dead" entries during training), the authors propose a class-aware re-initialization strategy:

1. **Usage tracking**: The usage rate of each codevector is tracked via exponential moving average $U_s^k = \gamma U_{s-1}^k + \frac{1-\gamma}{B} u_s^k$ ($\gamma=0.99$).
2. **Class-aware anchor selection**: For each codevector, the nearest same-class encoding within the mini-batch is selected as an anchor.
3. **Decay-based update**: A decay coefficient $\alpha_s^k = \exp(-\frac{10 U_s^k N_q}{1-\gamma} - \epsilon)$ is computed, assigning larger re-initialization weights to low-usage codevectors.

The key distinction from the class-agnostic approach of Zheng & Vedaldi (2023) is that both anchor selection and updates are confined within class partitions.

### LFMM: Latent Flow Matching Model

Optimal-transport-based flow matching is employed to transport Gaussian noise to clean scene layouts in the latent space:

- Intermediate states follow linear interpolation: $x_t = (1-t)x_0 + tx_1$
- The model learns a constant velocity field $v_\theta(x_t; t, \mathcal{F}_p) = x_1 - x_0$
- The loss applies separate weights to different attributes (translation, rotation, size, class, features)
- The network architecture is a U-Net conditioned on the room floor plan $\mathcal{F}_p$
- Inference uses the Euler method with $N_{\hat{t}}=100$ steps

### Class-Aware Inverse Lookup

At inference, the generated 32-dimensional feature $\hat{F}$ must be mapped back to a 128-dimensional codebook entry. This is achieved by maximizing cosine similarity between the truncated codevector (first 32 dimensions) and $\hat{F}$, with the indicator function restricting search to the target class partition. The best-matching codebook entry is then passed to the decoder to generate the final point cloud.

## Key Experimental Results

Evaluation is conducted on the 3D-FRONT dataset across three scene types: living room (2338/587), dining room (2071/516), and bedroom (5668/224).

**Point Cloud Generation Quality** (CD/P2M $\times 10^3$, lower is better):

| Method | LR CD | LR P2M | DR CD | DR P2M | BR CD | BR P2M |
|--------|-------|--------|-------|--------|-------|--------|
| Diffuscene | 30.63 | 29.87 | 30.60 | 29.49 | 45.01 | 44.88 |
| LFMM + VAE | 24.65 | 23.41 | 2.66 | 2.62 | 4.24 | 3.63 |
| LFMM + CPVQ-VAE | **9.06** | **8.27** | **2.38** | **2.17** | **2.46** | **2.06** |

- Living room: CD reduced by 70.4% and P2M by 72.3% compared to Diffuscene.
- Compared to the LFMM+VAE variant: CD reduced by 63.2% and P2M by 64.7%.

**Runtime Efficiency**: Inference time is 0.892s, 90.3% faster than Diffuscene (9.153s).

**Ablation Study** (Bedroom):

| Variant | VAE | VQ-VAE | Class Partition | Running Avg. Update | CD | P2M |
|---------|-----|--------|-----------------|---------------------|----|-----|
| V1 | ✓ | | | | 4.24 | 3.63 |
| V2 | | ✓ | | | 36.27 | 33.93 |
| V3 | | ✓ | ✓ | | 5.00 | 4.17 |
| V4 | | ✓ | ✓ | ✓ | **2.46** | **2.06** |

V2 (VQ-VAE without partitioning) suffers severe codebook collapse and performs worst. Adding partitioning (V3) yields a substantial improvement but remains below the VAE baseline. Incorporating the running average update (V4) surpasses all variants.

## Highlights & Insights

1. **First purely generative point cloud scene generation method**: Complete point cloud objects are decoded directly without requiring an external object database.
2. **Elegant class-partitioned codebook design**: A clean partitioning mechanism implemented via indicator functions ensures class consistency in decoding.
3. **Class-aware codebook maintenance**: The running average update effectively mitigates codebook collapse while preserving class awareness.
4. **Flow matching over diffusion**: Sampling steps are reduced from thousands (diffusion) to 100, achieving an order-of-magnitude speedup.
5. **Well-structured ablation study**: The V1–V4 ablation chain clearly demonstrates the contribution of each component.

## Limitations & Future Work

1. **Limited point cloud density**: The current autoencoder processes only 2025 points per object, constraining mesh reconstruction quality.
2. **Quantization error**: Vector quantization inevitably introduces error, slightly reducing the diversity of generated objects.
3. **Scene-type-specific training**: A separate model must be trained for each scene type (living room, dining room, bedroom).
4. **Incomplete baseline comparison**: DeBaRa and CASAGPT are excluded from comparison due to unavailable open-source code.
5. **Latent feature truncation**: The effect of truncating 128-dimensional codebook entries to 32 dimensions for LFMM training has not been thoroughly analyzed.

## Related Work & Insights

| Method | Generation Paradigm | Generates Point Clouds | Requires External DB | Requires Class Condition |
|--------|--------------------|-----------------------|----------------------|--------------------------|
| ATISS | Autoregressive | No | Yes | No |
| Diffuscene | Diffusion | Yes (poor quality) | Yes | No |
| DeBaRa | EDM Diffusion | No | Yes | Yes |
| SDF-based | Diffusion | No (voxel) | No | Yes (+ scene graph) |
| **Ours** | Flow matching | **Yes (high quality)** | **No** | **Auto-generated** |

The core advantage of the proposed method lies in jointly generating class labels and volumetric features, with CPVQ-VAE leveraging the generated class labels for conditional decoding, forming a closed-loop pipeline.

The class-partitioned codebook concept is transferable to any scenario requiring multi-class discrete representations (e.g., image tokenizers, speech synthesis). Flow matching demonstrates a favorable efficiency–quality trade-off on structured 3D data compared to diffusion. Class consistency is identified as a fundamental challenge in scene generation, a finding equally applicable to other modalities such as text-to-3D and interior design generation.

## Rating

- **Novelty**: 8/10 — The class-partitioned codebook and class-aware update mechanism are original contributions; applying flow matching to scene generation is also relatively novel.
- **Experimental Thoroughness**: 7/10 — Comprehensive evaluation across three scene types with clear ablation, though comparison with DeBaRa and similar methods is missing.
- **Writing Quality**: 8/10 — Method description is clear, mathematical derivations are complete, and figures are intuitive.
- **Value**: 7/10 — Advances point cloud scene generation from a retrieval-based to a purely generative paradigm, though constrained by point cloud density.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[CVPR 2026\] PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences](../../CVPR2026/3d_vision/pcstracker_long-term_scene_flow_estimation_for_point_cloud_sequences.md)
- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](../../CVPR2026/3d_vision/geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](../../NeurIPS2025/3d_vision/rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)

</div>

<!-- RELATED:END -->
