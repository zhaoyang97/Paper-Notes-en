---
title: >-
  [Paper Note] Novel View Synthesis with Pixel-Space Diffusion Models
description: >-
  [CVPR 2025][3D Vision][Novel View Synthesis] VIVID achieves end-to-end novel view synthesis using the EDM2 pixel-space diffusion model. By employing a dual U-Net encoder-decoder with cross-attention to transfer geometric information, a simple camera pose embedding (instead of complex geometric encodings), and single-view data augmentation based on homography, it achieves an FID of 2.89 (51% lower than GenWarp) and a PSNR of 17.36 (+29%) on RealEstate10K.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Pixel-space Diffusion"
  - "Dual U-Net"
  - "Single-view Augmentation"
  - "Homographic Transformation"
date: 2026-05-08
content_hash: d28686bc2080df99
---

# Novel View Synthesis with Pixel-Space Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2411.07765](https://arxiv.org/abs/2411.07765)  
**Code**: Project Page  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Pixel-space Diffusion, Dual U-Net, Single-view Augmentation, Homographic Transformation

## TL;DR

VIVID achieves end-to-end novel view synthesis using the EDM2 pixel-space diffusion model. By employing a dual U-Net encoder-decoder with cross-attention to transfer geometric information, a simple camera pose embedding (instead of complex geometric encodings), and single-view data augmentation based on homography, it achieves an FID of 2.89 (51% lower than GenWarp) and a PSNR of 17.36 (+29%) on RealEstate10K.

## Background & Motivation

1. **Background**: Novel view synthesis (NVS) methods are categorized into 3D representation-based (NeRF/3DGS, requiring multi-view training) and generative-based (diffusion models, enabling single-view inference). Existing diffusion methods mostly operate in latent space (e.g., Zero-1-to-3, GeoGPT).
2. **Limitations of Prior Work**: (1) Latent-space diffusion introduces reconstruction loss via the VAE encoder-decoder, leading to severe detail loss, particularly under large view changes; (2) Existing methods require complex geometric encodings (epipolar features, depth warping), increasing engineering complexity; (3) Training relies solely on multi-view video data, leading to poor generalization on out-of-domain scenes.
3. **Key Challenge**: Latent space is efficient but suffers from reconstruction bottlenecks; pixel space provides high fidelity but is difficult to train (high resolution and slow convergence).
4. **Goal**: Perform NVS directly in pixel space to bypass the latent-space reconstruction loss, while reducing complexity through simple pose embeddings and single-view augmentation.
5. **Key Insight**: Advances in the EDM2 architecture dramatically improve the training efficiency of pixel-space diffusion, making pixel-space NVS feasible.
6. **Core Idea**: Dual U-Net + cross-attention geometric transfer + pose embedding + homographic rotation augmentation.

## Method

### Overall Architecture

Source view image $\rightarrow$ Encoder U-Net extracts features $\rightarrow$ Joint self-attention + cross-attention transfers source view features to the target view $\rightarrow$ Target pose embedding (flattened extrinsics + intrinsics) $\rightarrow$ Decoder U-Net denoises to generate the target view image. Cascaded design: low-resolution base model + super-resolution model.

### Key Designs

1. **Simple Pose Embedding**

    - **Function**: Inject camera pose information into the diffusion process.
    - **Mechanism**: Directly flatten and normalize ($\mu=0, \sigma=1$) the extrinsic matrix, focal length, and principal point to use as embedding. Ablation studies show that a simple pose embedding (FID 3.00) performs close to complex pose+epipolar encodings (FID 2.87).
    - **Design Motivation**: Complex geometric encodings (epipolar features, depth warping) increase engineering complexity with marginal gains. Cross-attention is already capable of learning implicit geometric correspondences.

2. **Cross-Attention Geometric Transfer**

    - **Function**: Transfer geometric and appearance information from the source view to the target view.
    - **Mechanism**: Joint self-attention + cross-attention, where queries come from the target view while keys/values come from the joint features of both source and target views.
    - **Design Motivation**: More flexible than warping operations. Warping fails in occluded regions, whereas attention can "borrow" information from visible areas.

3. **Single-View Homographic Augmentation**

    - **Function**: Simulate multi-view data augmentation using single-view images.
    - **Mechanism**: Apply random rotational homography $H_{rot} = K_{dst} R_{dst} R_{src}^{-1} K_{src}^{-1}$ to single-view images to generate source-target image pairs. A 10% mixing ratio is optimal.
    - **Design Motivation**: The training data (RealEstate10K) primarily consists of indoor house tour videos, yielding poor generalization to out-of-domain environments (e.g., outdoor natural scenes). Single-view augmentation introduces appearance diversity from out-of-domain images.

### Loss & Training

Standard EDM2 diffusion loss. CFG scale of 1.5 (in-domain) / 2.0 (out-of-domain). Cascaded two-stage: base model + super-resolution.

## Key Experimental Results

### Main Results

| Method | Mid-range FID↓ | Mid-range PSNR↑ | Long-range FID↓ | Long-range PSNR↑ |
|------|-----------|-------------|-----------|-------------|
| GeoGPT | 6.43 | 14.06 | 7.22 | 13.13 |
| GenWarp | 5.91 | 13.43 | 7.38 | 12.10 |
| PhotoNVS | 7.12 | 13.32 | 9.22 | 12.05 |
| **VIVID** | **2.89** | **17.36** | **3.89** | **15.21** |

### Ablation Study

| Geometric Encoding | FID↓ | PSNR↑ | Description |
|---------|------|-------|------|
| No Encoding | 5.75 | 13.39 | Baseline |
| Epipolar Encoding | 4.14 | 17.43 | Geometry helps |
| **Pose Embedding** | **3.00** | **21.11** | Simpler is better |
| Pose + Epipolar | 2.87 | 21.15 | Marginal gain |

### Key Findings

- FID is 51% lower than GenWarp (5.91 $\rightarrow$ 2.89) and PSNR is 29% higher, highlighting the significant fidelity advantage of pixel space.
- The performance gap between simple pose embedding (FID 3.00) and pose+epipolar encoding (FID 2.87) is minimal, suggesting that complex geometric encodings are unnecessary.
- 10% single-view augmentation registration reduces out-of-domain FID from 36.14 to 31.98 (-11.4%), but excessive augmentation (25%) degrades performance.
- Out-of-domain generalization remains the main bottleneck (in-domain FID 2.89 vs. out-of-domain >30).

## Highlights & Insights

- **The "Simple Pose is Enough" Discovery**: Challenges the consensus that "NVS must use complex geometric encoding"—attention mechanisms can implicitly learn geometric correspondence.
- **Empirical Comparison of Pixel Space vs. Latent Space**: The first systematic study to compare them in NVS, demonstrating that pixel space has an inherent advantage in fidelity.
- **Clever use of Homographic Augmentation**: Although only handling rotations (not translations), it is sufficient to introduce valuable out-of-domain diversity.

## Limitations & Future Work

- Homographic augmentation only models rotations and lacks translation-based depth changes.
- Pixel-space diffusion requires more computational resources compared to latent-space methods.
- Out-of-domain generalization still incurs significant performance drops (FID >30).
- RealEstate10K consists mainly of indoor scenes; outdoor scenes require more training data.

## Related Work & Insights

- **vs GenWarp**: Employs warping operations for geometric alignment, which fail in occluded regions. VIVID replaces warping with attention, making it more robust.
- **vs GeoGPT**: A latent-space method (FID 6.43). Pixel-space VIVID yields an FID of 2.89. The disparity in fidelity is mainly due to VAE reconstruction loss.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of pixel-space NVS and simplified pose embedding is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple distances + out-of-domain + detailed ablation + multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Offers a new architectural alternative for NVS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2025\] MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)
- [\[CVPR 2026\] SmokeSVD: Smoke Reconstruction from A Single View via Progressive Novel View Synthesis and Refinement with Diffusion Models](../../CVPR2026/3d_vision/smokesvd_smoke_reconstruction_from_a_single_view_via_progressive_novel_view_synt.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)

</div>

<!-- RELATED:END -->
