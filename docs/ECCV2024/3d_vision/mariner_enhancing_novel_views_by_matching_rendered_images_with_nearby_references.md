---
title: >-
  [Paper Note] MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References
description: >-
  [ECCV 2024][3D Vision][Novel View Enhancement] The MaRINeR method is proposed to enhance the rendered image quality of 3D reconstructions using nearby reference images via deep feature matching and hierarchical detail transfer. It is applicable to post-processing rendering for various 3D representations, including explicit (mesh) and implicit (NeRF) representations.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Novel View Enhancement"
  - "Reference Image Super-Resolution"
  - "Feature Matching"
  - "Render Optimization"
  - "3D Reconstruction"
date: 2026-05-08
content_hash: 14d9c324a902c308
---

# MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References

**Conference**: ECCV 2024  
**arXiv**: [2407.13745](https://arxiv.org/abs/2407.13745)  
**Code**: [Project Page](https://boelukas.github.io/mariner/)  
**Area**: 3D Vision  
**Keywords**: Novel View Enhancement, Reference Image Super-Resolution, Feature Matching, Render Optimization, 3D Reconstruction

## TL;DR

The MaRINeR method is proposed to enhance the rendered image quality of 3D reconstructions using nearby reference images via deep feature matching and hierarchical detail transfer. It is applicable to post-processing rendering for various 3D representations, including explicit (mesh) and implicit (NeRF) representations.

## Background & Motivation

**Background**: 3D reconstruction methods (point clouds, meshes, NeRF, Gaussian Splatting) all have limitations, producing rendered images that contain geometric noise, appearance artifacts, and inconsistencies, which directly affect downstream tasks.

**Limitations of Prior Work**: 
   - Existing artifact removal methods (such as NeRFLiX) are usually designed for specific pipelines and cannot generalize across different pipelines.
   - Reference-based Super-Resolution (RefSR) methods are designed to solve resolution differences and cannot handle geometric artifacts and appearance inconsistencies in rendered images.
   - Style transfer methods either destroy content structure or fail to remove artifacts.

**Key Challenge**: A general post-processing method is needed that can both remove rendering artifacts from various 3D representations and transfer realistic details from reference images.

**Goal**: Propose a 3D reconstruction pipeline-agnostic rendering enhancement method that recovers rendering quality using existing reference images.

**Key Insight**: Model the problem as reference-image-guided rendering enhancement—borrowing feature matching and transfer concepts from RefSR but adapting them to the render-to-real domain gap.

**Core Idea**: Extract multi-level features of rendered and reference images using a shared encoder, establish correspondences through coarse-to-fine matching, and hierarchically fuse reference details into the rendered images.

## Method

### Overall Architecture

MaRINeR is adapted from the MASA RefSR pipeline and consists of four stages: (1) extracting multi-layer features of the rendered image $I$ and the reference image $R$ using a shared convolutional encoder; (2) performing coarse-to-fine matching on the deepest features; (3) warping reference features to the rendered view based on the matching correspondences; and (4) dynamically fusing the rendered and warped reference features in a hierarchical decoder. It supports iterative refinement.

### Key Designs

1. **Shared Multi-layer Convolutional Encoder**: The encoder extracts three levels of features, with the resolution halved at each level:

$$\{\mathcal{F}_1^I \in \mathbb{R}^{H \times W \times F_1}, \mathcal{F}_2^I \in \mathbb{R}^{H/2 \times W/2 \times F_2}, \mathcal{F}_3^I \in \mathbb{R}^{H/4 \times W/4 \times F_3}\}$$

The rendered and reference images share encoder weights. Key modification: Unlike RefSR, which performs matching on low-resolution shallow features, MaRINeR matches features at the deepest layer $\mathcal{F}_3$ to leverage the stronger robustness of deep features against rendering artifacts. Each encoder layer contains 1 convolutional layer and 4 residual blocks, with the number of channels fixed at $F_i = 64$.

2. **Matching and Extraction Module (MEM)**: Uses the coarse-to-fine matching strategy of MASA: first, cosine similarity matching is conducted on a coarse grid (with stride), followed by dense matching within a fixed window around the matched points. This generates a mapping and matching scores:

$$m_{I \to R}: (x,y) \in \mathcal{F}_3^I \to \{(u,v) \in \mathcal{F}_3^R, s \in \mathbb{R}\}$$

The matching score $s$ is used to weight the warped reference features: regions with high-confidence matching utilize more reference information, while low-confidence regions retain more rendered features. This allows the model to adaptively decide when to rely on the reference.

3. **Hierarchical Fusion Decoder**: Employs SAM (Spatial Adaptation Module) and DRAM (Dual Residual Aggregation Module) for rendering-reference feature fusion:

$$\mathcal{O}_3 = P_3(\text{SAM}(\mathcal{F}_3^I, \mathcal{F}_3^{R \to I}) \oplus \mathcal{F}_3^I)$$
$$\mathcal{O}_2 = P_2(\text{DRAM}(\text{SAM}(\mathcal{O}_3^\uparrow, \mathcal{F}_2^{R \to I}), \mathcal{O}_3))$$
$$\mathcal{O}_1 = P_1(\text{DRAM}(\text{SAM}(\mathcal{O}_2^\uparrow, \mathcal{F}_1^{R \to I}), \mathcal{O}_2))$$

SAM learns to remap the distribution of reference features to match the distribution of rendered features (aligning the domain gap), while DRAM fuses features across different resolutions and upsamples them using transposed convolutions. Unlike RefSR, MaRINeR merges features at the same spatial resolution, allowing the acceptance of structural details from the reference to patch rendering artifacts.

4. **Iterative Refinement**: Since rendering artifacts can obscure real geometry and lead to poor initial matching, multiple iterations are utilized. The first iteration removes artifacts and provides a coarse enhancement, while subsequent iterations establish better correspondences on cleaner images. Every iteration receives supervision signals. Experiments indicate that 2 iterations yield the most significant improvement.

### Loss & Training

$$\mathcal{L} = \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\text{per}}\mathcal{L}_{\text{per}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}}$$

- **Reconstruction Loss**: $\mathcal{L}_{\text{rec}} = \|I_{\text{GT}} - I_{\text{ER}}\|_1$, L1 norm.
- **Perception Loss**: $\mathcal{L}_{\text{per}} = \frac{1}{3}\sum_{i=1}^{3}\|\phi_i(I_{\text{GT}}) - \phi_i(I_{\text{ER}})\|_2^2$, using the shallow relu1_1/relu2_2/relu3_3 VGG16 features (shallower than standard RefSR, because deep features might introduce artifacts due to large render-real domain gaps).
- **Adversarial Loss**: Relativistic GAN formulation with $\lambda_{\text{adv}} = 0.001$.

Training Strategy: The model is first trained using reconstruction + perceptual loss for 60 epochs, followed by fine-tuning with adversarial loss for another 20 epochs.

**Data Augmentation**: 
- Mesh downsampling augmentation: During training, low-quality mesh renderings keeping only 10% of the triangular faces are mixed in to improve robustness to various mesh qualities.
- Random reference augmentation: Reference images are randomly selected within a 5-second time window to simulate situations where the reference and target viewpoints differ significantly.

## Key Experimental Results

### Main Results - Comparison with RefSR and Style Transfer Methods

| Method | Category | CAB PSNR↑ | CAB LPIPS↓ | LIN PSNR↑ | LIN LPIPS↓ | HGE PSNR↑ | HGE LPIPS↓ |
|------|------|-----------|-----------|-----------|-----------|-----------|-----------|
| Render (Baseline) | - | 15.60 | 0.380 | 14.39 | 0.392 | 15.84 | 0.364 |
| MASA | RSR | 15.47 | 0.367 | 14.17 | 0.397 | 15.62 | 0.360 |
| DATSR | RSR | 15.63 | 0.364 | 14.34 | 0.438 | 15.80 | 0.376 |
| NNST | ST | 16.33 | 0.370 | 18.53 | 0.315 | 18.48 | 0.315 |
| **MaRINeR** | RE | **20.03** | **0.180** | **21.73** | **0.155** | **20.96** | **0.176** |

MaRINeR outperforms RefSR and ST methods by a large margin across all metrics. Performance on unseen HGE scenes is comparable to training scenes, showing strong generalization capabilities.

### Ablation Study - Effect of Data Augmentation

| Mesh Size | W/ Aug. PSNR | W/o Aug. PSNR | W/ Aug. SSIM | W/o Aug. SSIM |
|----------|-----------|-----------|-----------|-----------|
| 100% | 19.80 | 19.45 | 0.687 | 0.686 |
| 50% | 19.41 | 18.83 | 0.677 | 0.673 |
| 10% | **19.10** | 17.98 | **0.650** | 0.626 |

Mesh downsampling augmentation brings a significant improvement on low-quality meshes (1.12 dB PSNR difference at 10%), proving the effectiveness of the augmentation strategy.

### Application Experiments - Homography Estimation Improvement

| Method | Matches Count | Inlier Ratio | Homography Error |
|------|---------|--------|---------|
| Original Render ↔ Image | 39.21 | 61.24% | 4.86 |
| +MaRINeR ↔ Image | **58.89** | **78.16%** | **1.88** |

### Key Findings

- RefSR methods preserve the structure and color of the low-resolution input but fail to remove rendering artifacts.
- Style transfer methods can adjust color distributions but fail to distinguish real content from artifacts, introducing distortions.
- MaRINeR successfully achieves a three-in-one effect: color transfer, artifact removal, and missing content inpainting.
- Shallow VGG features (relu1_1/2_2/3_3) are more suitable for perceptual loss across the render-to-real domain gap than deeper features.
- The model generalizes well to grayscale rendering with color reference, and different 3D reconstruction methods (such as LiDAR/RGB-D SLAM).
- 2-iteration refinement yields the best results; more iterations show diminishing returns while linearly increasing inference time.

## Highlights & Insights

- **Pipeline Agnosticity**: Applicable to various 3D representations/rendering pipelines (such as mesh rendering, NeRF rendering, IBR rendering), serving as a general post-processing tool.
- **Adaptation from RefSR to Rendering Enhancement**: Key modifications (deep matching, same-resolution merging, shallow perceptual loss, iterative refinement) are simple yet highly effective.
- **Practical Downstream Applications**: Displays clear application value through automated pseudo-GT validation (replacing manual inspection), synthetic trajectory enhancement, and NeRF post-processing.
- **Robustness Against Weak References**: The matching-score weighting mechanism allows the model to adaptively degrade to relying solely on rendering features when references are lacking.

## Limitations & Future Work

- Currently trained on $160 \times 160$ resolution; higher resolutions require a two-step process (downsampling followed by super-resolution).
- May incorrectly identify real content as artifacts and remove it.
- Leverages texture-level matching instead of semantic-level matching, requiring objects to have similar textures.
- May introduce flickering between sequential frames, lacking temporal consistency constraints.
- Can consider replacing the matching module with more advanced pipelines (LoFTR/CroCo), though this would increase inference time.

## Related Work & Insights

- **MASA** [CVPR 2021]: A coarse-to-fine correspondence matching RefSR method $\to$ direct foundation of MaRINeR.
- **DATSR** [ECCV 2022]: Swin-Transformer-enhanced matching $\to$ comparison baseline in experiments.
- **NeRFLiX** [ICCV 2023]: Rendering enhancement for NeRF $\to$ restricted to NeRF, cannot generalize across pipelines.
- **WCT2** [CVPR 2019]: Wavelet-based photorealistic style transfer $\to$ preserves structure but does not remove artifacts.

## Rating

- **Novelty**: ⭐⭐⭐ The core idea is adapting RefSR to rendering enhancement, where innovations lie mainly at the application level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid comparisons, detailed ablation studies, validation on multiple downstream applications, and cross-scene generalization testing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definitions and proper elucidation of differences between RefSR and ST methods.
- **Value**: ⭐⭐⭐⭐ Practical value as a general post-processing tool, especially for automated pseudo-GT validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Flying with Photons: Rendering Novel Views of Propagating Light](flying_with_photons_rendering_novel_views_of_propagating_light.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[ICLR 2026\] The Less You Depend, the More You Learn: Synthesizing Novel Views from Sparse, Unposed Images with Minimal 3D Knowledge](../../ICLR2026/3d_vision/the_less_you_depend_the_more_you_learn_synthesizing_novel_views_from_sparse_unpo.md)
- [\[ECCV 2024\] GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views](gaura_generalizable_approach_for_unified_restoration_and_rendering_of_arbitrary_.md)
- [\[ECCV 2024\] TC-Stereo: Temporally Consistent Stereo Matching](temporally_consistent_stereo_matching.md)

</div>

<!-- RELATED:END -->
