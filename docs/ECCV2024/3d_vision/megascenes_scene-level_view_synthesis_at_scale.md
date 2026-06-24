---
title: >-
  [Paper Note] MegaScenes: Scene-Level View Synthesis at Scale
description: >-
  [ECCV 2024][3D Vision][Novel View Synthesis] Constructing MegaScenes, a large-scale scene-level 3D dataset containing over 100k SfM reconstructions from Wikimedia Commons internet photos, and combining warp conditioning with pose conditioning to improve pose consistency in scene-level novel view synthesis.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Large-Scale Dataset"
  - "Diffusion Models"
  - "Pose-Conditioned Generation"
  - "SfM Reconstruction"
date: 2026-05-08
content_hash: 2c08556b655e3640
---

# MegaScenes: Scene-Level View Synthesis at Scale

**Conference**: ECCV 2024  
**arXiv**: [2406.11819](https://arxiv.org/abs/2406.11819)  
**Code**: [https://megascenes.github.io](https://megascenes.github.io)  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Large-Scale Dataset, Diffusion Models, Pose-Conditioned Generation, SfM Reconstruction

## TL;DR

Constructing MegaScenes, a large-scale scene-level 3D dataset containing over 100k SfM reconstructions from Wikimedia Commons internet photos, and combining warp conditioning with pose conditioning to improve pose consistency in scene-level novel view synthesis.

## Background & Motivation

**Background**: Pose-conditioned diffusion models (such as Zero-1-to-3, ZeroNVS) have made progress in novel view synthesis (NVS), but are primarily trained on object-level (Objaverse) or object-centric small-scale scene datasets (DTU, CO3D).

**Limitations of Prior Work**: There is a lack of large-scale, diverse scene-level training data, making it difficult for existing methods to generalize to real in-the-wild scenes; MegaDepth contains only 196 landmarks, and CO3D/ACID/RealEstate10K have limited category coverage.

**Key Challenge**: While object-level datasets are abundant (Objaverse-XL has millions of models), scene-level datasets are highly insufficient in scale, diversity, and pose distribution, creating a bottleneck for scene-level NVS.

**Goal**: 
   - Build a large-scale scene-level 3D dataset to bridge the data gap.
   - Solve the issue of inaccurate poses in existing NVS methods when applied to scene-level contexts.

**Key Insight**: Leveraging over 8 million openly licensed images from Wikimedia Commons to obtain scene-level 3D information via SfM reconstruction; using warped images as an additional pose condition to improve consistency.

**Core Idea**: Large-scale internet photo SfM reconstruction + warp image conditioning = high-quality scene-level NVS.

## Method

### Overall Architecture

The MegaScenes framework consists of two parts: (1) Dataset construction — collecting images from Wikimedia Commons and performing SfM reconstruction using COLMAP; (2) NVS method improvement — adding warp image conditioning and extrinsic matrix conditioning on top of ZeroNVS.

### Key Designs

1. **Dataset Construction Pipeline**:

    - **Scene Identification**: Utilizing the Wikidata category hierarchy (e.g., "bridges", "religious buildings") to identify scene categories in Wikimedia Commons in a top-down manner.
    - **Image Downloading**: Downloading all images under each scene category and removing irrelevant images through subcategory filtering.
    - **SfM Reconstruction & Cleaning**: Extracting SIFT features and performing vocabulary tree matching + sparse reconstruction using COLMAP for each scene; using the Doppelgangers pipeline to handle visually ambiguous scenes.
    - Final scale: approximately 430k scenes, 9 million images, 100k+ SfM reconstructions, and 2 million registered images.

2. **Training Data Mining**:

    - Illumination consistency: Filtering image pairs with a shooting time difference of < 3 hours using metadata.
    - Visual overlap: Requiring at least 50 common SfM 3D points.
    - Aspect ratio preservation: Resizing the long edge to 256 and padding the short edge to prevent information loss from center cropping.
    - Manual inspection to remove 298 heavily occluded scenes.
    - Yielding a final set of $2,086,036$ training pairs from 32,259 scenes.

3. **Warp Conditioning**:

    - **Key Insight**: Numerical encoding of pose matrices is non-intuitive, forcing the model to learn spatial transformations on its own. Conversely, a warped image directly encodes how pixels should move, naturally aligning with the scene scale.
    - **Mechanism**: Estimating monocular depth with Depth-Anything $\rightarrow$ aligning it with COLMAP sparse point clouds $\rightarrow$ back-projecting the reference RGBD image into a mesh $\rightarrow$ rendering from the target pose to obtain the warped image.
    - Concatenating the warped image with the target image and reference image as input to the diffusion model.
    - The warp can be computed during both training and inference; during single-image inference, monocular depth is used to determine the scene scale.

4. **Joint Warp + Pose Conditioning**:

    - Using warp conditioning alone has two issues: (a) poor warp quality when depth estimation is inaccurate; (b) the model struggles to understand 3D structures with only 2D pixel motion cues.
    - Therefore, the extrinsic matrix condition from ZeroNVS is additionally retained (flattened and passed through cross-attention) to achieve complementarity between warp and pose.
    - Extrinsics ensure 3D geometric consistency (e.g., generating dividing walls, complete buildings), while the warp ensures precise pixel alignment.

### Loss & Training

- Finetuned based on the Stable Diffusion model using standard diffusion training loss.
- Pose encoding: The flattened extrinsic matrix and FOV are together used as key/value for cross-attention.
- The CLIP embedding of the reference image is also injected via cross-attention to maintain consistency between the generated results and the reference.
- The translation scale is determined by the 20th percentile of the reference image depth.

### Evaluation Metrics Design

Proposing "Masked" versions of metrics (Masked LPIPS/PSNR/SSIM): assessing only the regions with pixel coverage after warping the reference image to the target viewpoint, which more reasonably measures generative consistency rather than creative generation in unmapped areas.

## Key Experimental Results

### Main Results — MegaScenes Test Set

| Method | LPIPS↓ | PSNR↑ | SSIM↑ | FID↓ | KID↓ |
|------|--------|-------|-------|------|------|
| Zero-1-to-3 (released) | 0.548 | 9.09 | 0.241 | 86.9 | 0.063 |
| ZeroNVS (released) | 0.616 | 7.47 | 0.151 | 69.1 | 0.049 |
| Zero-1-to-3 (MS) | 0.429 | 12.16 | 0.367 | 9.78 | 0.002 |
| ZeroNVS (MS) | 0.386 | 12.90 | 0.401 | 9.84 | 0.002 |
| SD-inpainting | 0.425 | 12.36 | 0.392 | 38.5 | 0.024 |
| **Ours** | **0.344** | **13.40** | **0.445** | **11.6** | **0.004** |

### Cross-Domain Generalization Results — RealEstate10K

| Method | LPIPS↓ | PSNR↑ | SSIM↑ | FID↓ |
|------|--------|-------|-------|------|
| ZeroNVS (released) | 0.456 | 9.49 | 0.353 | 123.0 |
| ZeroNVS (MS) | 0.205 | 16.02 | 0.630 | 61.1 |
| **Ours** | **0.177** | **17.22** | **0.666** | **60.0** |

### Key Findings

- Simply finetuning on MegaScenes significantly improves performance: ZeroNVS FID drops from 69.1 to 9.84.
- Zero-1-to-3 (MS) already outperforms ZeroNVS (released), demonstrating the intrinsic value of the MegaScenes dataset itself.
- Joint warp + pose conditioning achieves the best performance across all four test sets (MegaScenes/DTU/MipNeRF360/RE10K).
- While SD-inpainting performs well on masked reconstruction metrics (by directly copying warp pixels), its 3D understanding is poor, with FID/KID metrics falling far behind those of the finetuned models.

## Highlights & Insights

- **Data-Driven Insight**: The diversity of internet photos (illumination, weather, devices, pose distribution) is key to improving generalization, far outperforming datasets from controlled environments.
- **Clever Design of Warp Conditioning**: Visualizing geometric info as a warped image allows the diffusion model to learn 3D transformations through image comprehension, which is more effective than directly encoding pose matrices.
- **Masked Metrics**: This resolves the issue where free generation in non-overlapping regions by generative models distorts reconstruction metrics.
- **Engineering Scalability**: Based on openly licensed data from Wikimedia Commons, supporting continuous expansion in the future.

## Limitations & Future Work

1. Only a small fraction of the dataset is utilized (475k out of 2 million images), and metadata such as text captions is not leveraged.
2. It relies on monocular depth estimation for warping, and performance degrades when depth is inaccurate.
3. It cannot handle large-angle viewpoint changes (e.g., viewing behind the scene).
4. Illumination condition modeling is not considered; the illumination issue is merely bypassed by metadata temporal filtering.
5. Scene scale inference relies on the absolute scale of monocular depth, which may be inaccurate.

## Related Work & Insights

- **MegaDepth** $\rightarrow$ MegaScenes: Also leverages internet photos + SfM, but MegaScenes is over 100x larger in scale.
- **Zero-1-to-3/ZeroNVS**: Serve as baselines, proving that pose-conditioned diffusion models require scene-level data.
- **DL3DV-10K**: Concurrent work that builds a scene dataset from videos, but has limited pose diversity.
- **Insights**: The importance of dataset scale and diversity for 3D generative models might be underestimated; internet photos are an undervalued source of large-scale 3D data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (Solid dataset construction methodology, simple yet effective warp conditioning design)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Comprehensive evaluation with 4 test sets, multiple baselines, and both qualitative and quantitative analyses)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and fully articulated motivation)
- **Value**: ⭐⭐⭐⭐⭐ (The dataset holds long-term value for the community, with open-sourced code, data, and models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)
- [\[ECCV 2024\] Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [\[ECCV 2024\] Efficient Depth-Guided Urban View Synthesis (EDUS)](efficient_depth-guided_urban_view_synthesis.md)
- [\[ECCV 2024\] Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis](forest2seq_revitalizing_order_prior_for_sequential_indoor_scene_synthesis.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)

</div>

<!-- RELATED:END -->
