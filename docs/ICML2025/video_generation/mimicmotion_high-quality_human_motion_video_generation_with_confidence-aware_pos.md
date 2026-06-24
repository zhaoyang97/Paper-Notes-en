---
title: >-
  [Paper Note] MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance
description: >-
  [ICML 2025][Video Generation][pose-guided video generation] Built upon Stable Video Diffusion, this pose-guided human video generation framework achieves a FID-VID of 9.3 (prev. best 12.4) on the TikTok dataset by encoding pose estimation confidence into guidance signals, amplifying training loss for high-confidence hand regions, and employing position-aware progressive latent fusion. It also natively supports the generation of smooth videos of arbitrary length.
tags:
  - "ICML 2025"
  - "Video Generation"
  - "pose-guided video generation"
  - "confidence-aware"
  - "progressive latent fusion"
  - "hand region enhancement"
  - "SVD"
date: 2026-05-08
content_hash: f8f684cef363c2df
---

# MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance

**Conference**: ICML 2025  
**arXiv**: [2406.19680](https://arxiv.org/abs/2406.19680)  
**Code**: [https://tencent.github.io/MimicMotion](https://tencent.github.io/MimicMotion)  
**Area**: Video/Image Generation  
**Keywords**: pose-guided video generation, confidence-aware, progressive latent fusion, hand region enhancement, SVD

## TL;DR

Built upon Stable Video Diffusion, this pose-guided human video generation framework achieves a FID-VID of 9.3 (prev. best 12.4) on the TikTok dataset by encoding pose estimation confidence into guidance signals, amplifying training loss for high-confidence hand regions, and employing position-aware progressive latent fusion. It also natively supports the generation of smooth videos of arbitrary length.

## Background & Motivation

**Background**: Pose-guided human motion video generation is a crucial subfield of video generation. Existing methods such as AnimateAnyone, MagicAnimate, and MagicPose leverage diffusion models conditioned on a reference image and a pose sequence to generate corresponding human motion videos. These base models are typically built upon pre-trained models like Stable Video Diffusion (SVD).

**Limitations of Prior Work**: Existing approaches suffer from three core issues: (1) Severe hand distortion—finger deformation and incorrect positioning are highly common, especially in large-motion scenes; (2) Trade-off between quality and temporal smoothness—to ensure inter-frame smoothness, frame details are often sacrificed, leading to blurry frames; (3) Noisy pose estimation—detectors like DWPose have limited accuracy in dynamic scenes, resulting in duplicate detections and false positives under occlusion, which causes models to overfit to noisy training samples.

**Key Challenge**: The uncertainty of pose estimation impairs both the training and inference stages. During training, noisy poses lead the model to learn incorrect motion mappings; during inference, inaccurate pose guidance directly generates distorted outputs. Furthermore, due to computational constraints, existing methods can only generate a dozen frames at a time, failing to natively support long videos.

**Goal**: (1) How to maintain high generation quality when pose estimation is inaccurate? (2) How to targetedly improve the generation quality of critical regions like hands? (3) How to generate smooth videos of arbitrary length?

**Key Insight**: The authors observe that pose estimators (e.g., DWPose) naturally output confidence scores for each keypoint. However, prior methods completely ignore this signal, applying only simple filtering with hard thresholds. If the confidence scores are directly encoded into the pose guidance signals, the model can automatically differentiate between reliable and unreliable poses, thereby mitigating the negative impact of noisy poses in both training and inference stages.

**Core Idea**: Encode pose estimation confidence as the luminance of guidance signals to let the model adaptively trust high-confidence pose keypoints, while utilizing regional loss amplification and progressive latent fusion to resolve hand distortion and long-video stitching issues, respectively.

## Method

### Overall Architecture

MimicMotion is built upon the pre-trained Stable Video Diffusion (SVD). The input consists of a reference image $I_{\text{ref}}$ and a pose sequence. The reference image takes two paths: cross-attention features are extracted via CLIP and injected into each layer of the U-Net; meanwhile, the latent representation is encoded through a frozen VAE encoder, duplicated along the temporal dimension, and concatenated with video frame features in the channel dimension. The pose sequence is extracted as features by a PoseNet consisting of multi-layer convolutions and is element-wise added to the output of the first convolutional layer of the U-Net (instead of every layer, to avoid disturbing the spatio-temporal interaction layers of the pre-trained model). The entire denoising process is performed in the latent space, and the video frames are finally reconstructed via a VAE decoder with temporal layers.

### Key Designs

1. **Confidence-Aware Pose Guidance**:

    - **Function**: Encodes the uncertainty of pose estimation into the guidance signals, enabling the model to distinguish between reliable and unreliable poses.
    - **Mechanism**: Instead of using traditional fixed-threshold filtering for keypoints, the color values of each keypoint and limb connection are multiplied by their confidence scores. High-confidence keypoints appear brighter (more salient) in the pose guidance map, while low-confidence keypoints appear darker (approaching black). Consequently, the model automatically reduces its focus on low-confidence poses during training and can handle uncertain pose inputs during inference. For instance, when DWPose generates duplicate detections or false positives under occlusion, the low confidence makes the erroneous keypoints nearly invisible in the guidance map.
    - **Design Motivation**: Pose estimation naturally carries uncertainty in dynamic videos (due to self-occlusion, motion blur, etc.), and confidence scores are readily available yet overlooked signals. Continuous confidence weighting, rather than binary filtering, preserves the gradation of information and is more robust than hard thresholding.

2. **Hand Region Enhancement**:

    - **Function**: Targetedly improves hand generation quality and reduces finger distortion.
    - **Mechanism**: Reliable hand regions are identified based on the confidence scores of hand keypoints. When the confidence of all hand keypoints exceeds a threshold, a bounding box enclosing the hand keypoints is constructed, and the training loss for this region is scaling-amplified by 10 times ($w_{\text{hand}}=10$). This biases the model towards focusing on high-quality hand samples during training, learning more accurate hand generation.
    - **Design Motivation**: Hands are the most distortion-prone areas in video generation (due to fine details and high degrees of freedom) and are also focal points for human observers. Combining this with confidence ensures that weighting is only applied to high-quality hand regions, avoiding the reinforcement of erroneous samples.

3. **Progressive Latent Fusion**:

    - **Function**: Generates smooth videos of arbitrary length,消除 segment boundary flickering and sudden transitions.
    - **Mechanism**: A long pose sequence is segmented into fixed-length video segments of length $N$, with an overlap of $C$ frames between adjacent segments ($C \ll N$). During each denoising step, each segment is denoised independently, and a position-aware weighted fusion is performed on the overlapping regions. The fusion weight is defined as $\lambda_{\text{fusion}} = 1/(C+1)$, where frames closer to the center of the current segment receive higher weights, and frames closer to the boundaries receive lower weights. Specifically, for the $j$-th frame of the $i$-th segment ($j \leq C$), the fusion formula is:
      $$\mathbf{z}_i^j \leftarrow j\lambda_{\text{fusion}}\mathbf{z}_i^j + (1 - j\lambda_{\text{fusion}})\mathbf{z}_{i-1}^{N-C+j}$$
      This method is training-free and is only applied during inference.
    - **Design Motivation**: Direct MultiDiffusion-style average fusion uses equal weights for all overlapping frames, which leads to sudden transitions at segment boundaries (e.g., background becoming sharp $\to$ blurry $\to$ sharp). Progressive fusion ensures smooth transitions between segments through position-aware weight changes.

### Loss & Training

The baseline loss follows the standard MSE denoising loss of diffusion models, $\mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t; \mathbf{c}, t)\|_2^2]$. Building upon this, hand region enhancement amplifies the loss value within reliable hand regions by a factor of 10, making this region contribute more to gradient updates. Training is conducted on 8 A100 GPUs for 20 epochs with a learning rate of $10^{-5}$, using 16-frame clips, with 4,436 dance videos collected as training data.

## Key Experimental Results

### Main Results: Quantitative Comparison on the TikTok Test Set

| Method | FID-VID↓ | FVD↓ | SSIM↑ | PSNR↑ |
|------|----------|------|-------|-------|
| MagicAnimate | 16.2 | 848 | 0.740 | 17.5 |
| MagicPose | 13.3 | 916 | 0.776 | 18.8 |
| Moore-AnimateAnyone | 12.4 | 728 | 0.758 | 18.7 |
| MuseV | 14.6 | 754 | 0.766 | 17.6 |
| **MimicMotion (Ours)** | **9.3** | **594** | **0.795** | **20.1** |

MimicMotion comprehensively outperforms existing methods across all four metrics, with FID-VID dropping from 12.4 to 9.3 (a 25% reduction) and FVD decreasing from 728 to 594 (an 18% reduction).

### Ablation Study: Contribution of Each Component

| Hand Enhancement | Confidence-Aware | Progressive Fusion | FID-VID↓ | FVD↓ | SSIM↑ | PSNR↑ |
|---------|----------|---------|----------|------|-------|-------|
| ✗ | ✗ | ✗ | 14.6 | 776 | 0.760 | 18.0 |
| ✓ | ✗ | ✗ | 15.0 | 678 | 0.758 | 17.9 |
| ✓ | ✓ | ✗ | 12.2 | 623 | 0.787 | 18.4 |
| ✓ | ✓ | ✓ | **9.3** | **594** | **0.795** | **20.1** |

### Key Findings

- The addition of confidence-aware pose guidance reduces the FVD from 678 to 623, confirming its pivotal role in temporal smoothness.
- Hand region enhancement alone exhibits limited efficacy (FID-VID even slightly increases to 15.0), but when combined with confidence awareness, the performance improves significantly (FID-VID drops from 14.6 to 12.2). This indicates a synergistic effect between the two: hand loss should only be amplified when confidence is reliable.
- Progressive fusion further reduces FVD from 623 to 594 while boosting PSNR from 18.4 to 20.1, proving its dual contribution to both temporal coherence and frame quality in long videos.
- In the user study, 75.5%–100% of the participants preferred the results of MimicMotion. Even compared to MuseV, which generates high image quality, it still achieved a 75.5% preference rate.

## Highlights & Insights

- Clever utilization of confidence signals: Pose estimators naturally output confidence scores, but previous methods failed to encode them into guidance signals. This represents a "free lunch."
- The hand region loss amplification design is simple and engineering-friendly, requiring only two hyperparameters: a confidence threshold and an amplification factor.
- Progressive fusion is a training-free inference strategy that does not affect model training and can be applied plug-and-play.
- The three contributions precisely map to three practical pain points (pose noise $\to$ confidence awareness, hand distortion $\to$ regional enhancement, long video $\to$ progressive fusion), representing a problem-driven rather than technology-driven approach.

## Limitations & Future Work

- Reliance on the DWPose detector limits performance in non-human characters or extreme motion scenarios.
- It only supports 2D pose skeleton guidance, lacking support for 3D parametric representations like SMPL/DensePose.
- The training data consists of only 4,436 dance videos, which limits scene diversity.
- Progressive fusion may still exhibit transition artifacts when the style shifts abruptly between adjacent segments.

## Related Work & Insights

- **vs AnimateAnyone/MagicAnimate**: These methods ignore the uncertainty of pose estimation, making MimicMotion's confidence-aware design a key differentiator.
- **vs MultiDiffusion/Lumiere**: MultiDiffusion uses equal-weight average fusion for overlapping frames. Lumiere inherits this strategy and still suffers from segment boundary mutations. MimicMotion's position-aware progressive fusion solves this problem fundamentally.
- **Insight**: The concept of using detector byproducts (confidence scores) as conditioning signals can be generalized to other conditional generation tasks (e.g., depth-guided or segmentation-guided generation).

## Rating

⭐⭐⭐⭐ Problem-driven, precisely designed, and highly practical in engineering, with clear ablation validation for every component. Confidence-aware guidance is a simple yet effective innovation. However, the evaluation is limited to dance video scenarios, leaving generalizability to be proven.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](../../CVPR2025/video_generation/osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[NeurIPS 2025\] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](../../NeurIPS2025/video_generation/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](../../CVPR2025/video_generation/posetraj_pose-aware_trajectory_control_in_video_diffusion.md)
- [\[CVPR 2026\] 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation](../../CVPR2026/video_generation/3d-aware_implicit_motion_control_for_view-adaptive_human_video_generation.md)

</div>

<!-- RELATED:END -->
