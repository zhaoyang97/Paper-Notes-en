---
title: >-
  [Paper Note] ERUPT: Efficient Rendering with Unposed Patch Transformer
description: >-
  [CVPR 2025][3D Vision][Novel View Synthesis] ERUPT proposes an efficient latent view synthesis model. By replacing pixel-level decoding with a patch-based decoder, incorporating learnable latent camera poses, and utilizing a frozen DINOv2 feature extractor, it achieves novel view synthesis at 600 fps using only 5 unposed images without requiring precise camera poses, reaching SOTA performance on the MSN dataset.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Scene Representation"
  - "Unposed Rendering"
  - "Patch Decoder"
  - "Latent View Synthesis"
date: 2026-05-08
content_hash: 3034de4c8cf08dc7
---

# ERUPT: Efficient Rendering with Unposed Patch Transformer

**Conference**: CVPR 2025  
**arXiv**: [2503.24374](https://arxiv.org/abs/2503.24374)  
**Code**: None (Dataset MSVS-1M provided)  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Scene Representation, Unposed Rendering, Patch Decoder, Latent View Synthesis

## TL;DR
ERUPT proposes an efficient latent view synthesis model. By replacing pixel-level decoding with a patch-based decoder, incorporating learnable latent camera poses, and utilizing a frozen DINOv2 feature extractor, it achieves novel view synthesis at 600 fps using only 5 unposed images without requiring precise camera poses, reaching SOTA performance on the MSN dataset.

## Background & Motivation

1. **Background**: Historically, the field of novel view synthesis has mostly relied on two major frameworks, NeRF and 3D Gaussian Splatting, which achieve high-quality rendering by training scene-specific models, but require dense images and precise camera poses. Recently, methods like SRT and RUST have explored feed-forward generalization schemes based on latent scene representations.

2. **Limitations of Prior Work**: NeRF/3DGS requires re-training for each new scene and depends on a large number of input images with accurate poses. SRT requires precise camera parameters for all images; while RUST supports unposed training, it still needs part of the target image during inference to query the model, preventing direct camera control. Both employ pixel-by-pixel decoding, which is computationally expensive.

3. **Key Challenge**: Existing latent scene representation methods face bottlenecks along three dimensions: (1) inability to train effectively on unposed data, (2) inability to directly control the camera at inference time, and (3) extremely low computational efficiency due to pixel-by-pixel decoding.

4. **Goal**: Design a generalized novel view synthesis model that supports both posed and unposed training, allows direct camera pose specification at inference time, and improves computational efficiency by an order of magnitude.

5. **Key Insight**: Alternate self-attention and cross-attention in the encoder to distinguish tokens from different images, and introduce patch-based decoding to replace pixel-level decoding.

6. **Core Idea**: Solve unposed training, direct camera control, and computational efficiency simultaneously through a tripartite architectural design incorporating "patch ray queries + learnable latent poses + alternating-attention scene Transformer".

## Method

### Overall Architecture
The input to ERUPT is a set of unordered (potentially unposed) scene images (typically 5), and the output is a novel view image rendered from any specified camera pose. The entire pipeline consists of three stages: (1) extracting feature tokens from each image using a frozen DINOv2; (2) generating a compact scene representation and estimating the camera pose for each image via a scene Transformer that alternates self-attention and cross-attention; (3) efficiently rendering the target image from the scene representation using a patch-based decoder.

### Key Designs

1. **Alternating-Attention Scene Transformer**:

    - **Function**: Extract a compact scene representation from unposed input images, while estimating relative camera poses for each image.
    - **Mechanism**: Unlike SRT which directly concatenates all tokens for global attention, ERUPT alternates intra-image self-attention (mixing within tokens of the same image) and scene-wide cross-attention (mixing each image's tokens with all scene tokens). A learnable camera token is appended to each image, naturally aggregating camera information during the alternating attention process. The scene Transformer consists of 6 blocks.
    - **Design Motivation**: Simple token mixing in SRT cannot distinguish tokens from different images, and RUST only uses simple tags to differentiate reference and non-reference frames. The alternating attention design allows the model to distinguish different images and construct robust scene representations even without knowing camera parameters.

2. **Target Camera Switching**:

    - **Function**: Support unposed data during training while enabling direct camera control at inference time.
    - **Mechanism**: Concatenate the estimated latent pose with the sinusoidal encoding of the ground-truth pose. During training, three modes are randomly sampled with 1/3 probability: latent pose only, ground-truth pose only, or both provided simultaneously. During inference, only the encoded target pose is utilized, enabling direct camera control. Ablation studies show minimal performance degradation when using only 5% ground-truth poses.
    - **Design Motivation**: RUST requires half of the target image to estimate poses, limiting the generate-able views. This switching mechanism allows the model to generate correct outputs via the latent channel when poses are imprecise, while maintaining accurate control during inference.

3. **Patch-Based Decoder**:

    - **Function**: Improve rendering efficiency by an order of magnitude.
    - **Mechanism**: Use $8 \times 8$ patch rays instead of pixel-by-pixel rays to query the scene representation, reconstructing 64 pixels per query instead of 1. The decoder consists of 4 standard Transformer decoder blocks that alternate self-attention and cross-attention with the scene representation, followed by 3 convolutional pixel shuffle upsampling blocks. An additional token decoder matches the semantic embeddings of the DINOv2 backbone.
    - **Design Motivation**: Pixel-by-pixel decoding in SRT and RUST leads to out-of-memory (OOM) issues on a 48GB A6000 at 224 resolution (RUST). Patch decoding reduces VRAM requirements by $64\times$.

### Loss & Training
- The image decoder uses $L_2$ pixel loss; the token decoder utilizes ArcGeo auxiliary loss to match DINOv2 semantic features; camera poses are trained with $L_2$ position loss + negative cosine view loss. Five target images per scene are trained to reuse the scene representation. The model is optimized using AdamW for 160 epochs with a batch size of 128. For GAN fine-tuning, $L_2$ is replaced with $L_1$ + perceptual + GAN loss. For Stable Diffusion rendering, the token decoder output is used as a prompt, fine-tuning the SD U-Net for 20 epochs.

## Key Experimental Results

### Main Results

| Method | Input Poses | Target Poses | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|---------|---------|-------|-------|--------|------|
| SRT | ✓ | ✓ | 23.41 | 0.697 | 0.369 | - |
| SRT* | ✓ | ✓ | 25.93 | - | 0.237 | 67.29 |
| RUST | ✗ | ✗ | 23.49 | 0.703 | 0.351 | - |
| DORSal | ✗ | ✗ | 18.99 | - | 0.265 | 9.00 |
| ERUPT L+LORA | ✗ | Partial | **25.26** | **0.769** | 0.340 | 91.1 |
| ERUPT B+GAN | ✗ | Partial | 23.38 | 0.713 | **0.204** | 7.45 |
| ERUPT B+SD | ✗ | Partial | 21.06 | 0.637 | 0.234 | **6.89** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Description |
|------|-------|-------|------|
| ERUPT B (baseline) | 23.85 | 0.718 | Full model |
| Single target training | 23.43 | 0.700 | Reusing scene representation for multiple targets is beneficial |
| Patch RUST | 23.20 | 0.690 | Simple token mixing is inferior to alternating attention |
| ERUPT B+LORA | 24.69 | 0.749 | LORA fine-tuning of the backbone yields significant improvement |
| ERUPT L+LORA | 25.26 | 0.769 | Scaling up the model further improves performance |
| 5% known poses | 23.55 | 0.706 | Requires very few ground-truth poses |

### Key Findings
- LoRA fine-tuning of the DINOv2 backbone contributes the most (PSNR +0.84), indicating that even robust foundation models need adaptation for 3D scene synthesis tasks.
- Using only 5% of the ground-truth target poses results in a performance drop of only 0.3 PSNR, demonstrating the robustness of the pose-switching strategy.
- The patch decoder makes training 5 times faster than RUST (at 224 resolution) and reduces VRAM footprint by $64\times$; RUST directly runs out of memory (OOM) on a 48GB GPU at 224 resolution.
- GAN and SD fine-tuning significantly improve perceptual quality (FID drops from ~100 to 7-9), although multi-view consistency for SD remains challenging.

## Highlights & Insights
- **Patch ray queries** is the core efficiency innovation—decoding 64 pixels instead of 1 pixel per query, yielding almost no degradation in quality while improving performance by an order of magnitude. This design concept can be transferred to any ray-query-based scene representation method.
- **Random pose switching training** is highly ingenious—switching between three modes with a 1/3 probability lets the model learn to leverage both ground-truth and latent poses simultaneously, with selective utilization during inference. This strategy of "mixing multiple input modes during training" can be transferred to other multimodal tasks.
- The introduction of the MSVS-1M real-world dataset (1 million images from Mapillary street views) fills the void of lacking large-scale real-world datasets in this field.

## Limitations & Future Work
- $L_2$ loss produces blurry outputs in scenes with high uncertainty; although GAN/SD fine-tuning offers improvements, they introduce new artifacts (GAN) or multi-view inconsistency (SD).
- SD rendering speed is only about 1 fps (at 512 resolution), which prevents real-time utilization.
- Each frame is rendered independently, lacking temporal consistency; incorporating multi-view diffusion models (such as the approach used in DORSal) is a promising future direction.
- Performance drops significantly on the real-world MSVS-1M dataset (PSNR 20.64 vs. MSN 24.69), indicating that the model's generalization capabilities in complex real-world scenes still require enhancement.

## Related Work & Insights
- **vs SRT**: SRT requires precise poses for all images, whereas ERUPT requires no input poses and only a fraction of target poses. ERUPT naturally resolves the pose issue using alternating attention and camera tokens.
- **vs RUST**: RUST requires half of the target image during inference and cannot directly control the camera. ERUPT's pose-switching mechanism achieves both unposed training and camera control at inference time.
- **vs DORSal**: DORSal uses multi-view diffusion to guarantee consistency, yielding better FID but significantly worse PSNR. ERUPT+SD outperforms DORSal in FID while maintaining a higher PSNR.

## Rating
- Novelty: ⭐⭐⭐⭐ Patch decoding and pose-switching training are highly innovative, though the overall architecture still follows the encoder-decoder paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments on MSN and real-world datasets are comprehensive, ablations are thorough, and computational efficiency comparisons are detailed.
- Writing Quality: ⭐⭐⭐⭐ Structurally clear with complete technical details, though it contains quite a few equations and notations.
- Value: ⭐⭐⭐⭐ The efficiency gains from patch decoding and the contribution of the real-world dataset are of practical value, and the pose-switching training paradigm is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](../../ICCV2025/3d_vision/spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [\[CVPR 2025\] Learnable Infinite Taylor Gaussian for Dynamic View Rendering](learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)

</div>

<!-- RELATED:END -->
