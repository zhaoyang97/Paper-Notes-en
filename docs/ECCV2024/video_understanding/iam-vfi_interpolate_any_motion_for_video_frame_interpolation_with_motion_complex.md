---
title: >-
  [Paper Note] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map
description: >-
  [ECCV 2024][Video Understanding][Video Frame Interpolation] The IAM-VFI framework is proposed, which introduces a Motion Complexity Map (MCM) to perceive the difficulty levels of local motion. By adaptively allocating computational resources and processing strategies to regions with varying complexities, it achieves robust video frame interpolation for arbitrary motion patterns.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Video Frame Interpolation"
  - "Motion Complexity Map"
  - "Optical Flow Estimation"
  - "Adaptive Processing"
  - "Multi-scale Feature Fusion"
date: 2026-05-08
content_hash: 0973a8358e3cb897
---

# IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Frame Interpolation, Motion Complexity Map, Optical Flow Estimation, Adaptive Processing, Multi-scale Feature Fusion

## TL;DR
The IAM-VFI framework is proposed, which introduces a Motion Complexity Map (MCM) to perceive the difficulty levels of local motion. By adaptively allocating computational resources and processing strategies to regions with varying complexities, it achieves robust video frame interpolation for arbitrary motion patterns.

## Background & Motivation

**Background**: Video Frame Interpolation (VFI) aims to synthesize intermediate frames between two existing video frames. It is a fundamental task in video processing, applied in scenarios such as slow-motion generation, video compression, and frame rate up-conversion. Prevailing methods are divided into optical flow-based approaches (such as RIFE, IFRNet, AMT) and kernel-based approaches. Optical flow-based methods first estimate bidirectional optical flows and then utilize them to warp and fuse input frames to generate intermediate frames.

**Limitations of Prior Work**: Existing VFI methods typically employ a uniform processing strategy across the entire image, overlooking the critical fact that motion complexity varies drastically across different regions. Static background areas require almost no complex processing, where simple linear interpolation suffices, whereas regions with occlusions, large motion, or non-linear motion (such as rotating objects or elastic deformation) require more elaborate modeling. A uniform strategy leads to two issues: (1) computational resources are wasted on simple regions; (2) handling capacity is insufficient for complex regions.

**Key Challenge**: The key challenge of VFI lies in the diversity of motion — a single frame can simultaneously contain static background, objects with uniform linear motion, rotating objects, and appearing/disappearing occluded regions. Existing methods lack the capability to "perceive motion difficulty," making them unable to dynamically adjust strategies based on local motion characteristics.

**Goal**: How to enable VFI models to perceive and adapt to the motion complexity of different regions in an image, adopting lightweight processing for simple motions and elaborate processing for complex motions (large displacements, occlusions, non-linearities).

**Key Insight**: The authors propose estimating a "Motion Complexity Map" (MCM) from the input frame pair to explicitly quantify the interpolation difficulty at each pixel location. The MCM can then guide subsequent optical flow refinement and frame synthesis modules, achieving differentiated processing for regions of varying complexities.

**Core Idea**: Explicitly perceive local motion difficulty using a motion complexity map to guide the VFI model in adaptively allocating processing precision and computational resources across different regions.

## Method

### Overall Architecture
The pipeline of IAM-VFI consists of four stages: (1) extracting multi-scale features from input frame pairs; (2) estimating initial bidirectional optical flows and a Motion Complexity Map (MCM); (3) MCM-guided optical flow refinement — performing more elaborate iterative optimization on regions with complex motion; (4) MCM-aware frame synthesis — adaptively fusing warped features based on motion complexity to generate the final intermediate frame.

### Key Designs

1. **Motion Complexity Map (MCM) Estimation**:

    - **Function**: Generates a scalar value for each pixel location to quantify the difficulty of frame interpolation at that location.
    - **Mechanism**: The estimation of the MCM is based on a comprehensive analysis of various motion cues: local variance of optical flow (reflecting motion non-uniformity), forward-backward consistency checks (detecting occluded regions), optical flow magnitude (larger motion is more difficult), and local texture information (optical flow is unreliable in low-texture regions). The network learns to predict the MCM from the input frame pair and multi-scale features, outputting a single-channel map via a lightweight encoder-decoder head. High values indicate complex motion in the region (large displacements, occlusions, or non-linear motions), while low values indicate simple motion (static or small, uniform motions).
    - **Design Motivation**: Motion complexity is a concept that has long been implicitly handled in VFI (occlusion detection, large motion handling, etc., are all attempts to deal with "difficult regions"). The MCM unifies these heterogeneous signals into an explicit spatial prior, enabling subsequent modules to directly utilize this information.

2. **MCM-guided Optical Flow Refinement**:

    - **Function**: Differentiantly refines optical flow estimation based on motion complexity.
    - **Mechanism**: Initial optical flows are obtained through standard correlation volumes or feature matching. In the refinement stage, regions are classified into different complexity levels based on the MCM: for low-complexity regions (low MCM values), the optical flow is directly used or processed with only a few iterations; for high-complexity regions (high MCM values), a refinement loop with more iterations is applied, introducing a wider search range and higher-resolution correlation features. This adaptive iterative strategy concentrates computational resources on regions where they are truly needed. During refinement, the MCM is also used as attention weights to modulate the contributions of features at different scales.
    - **Design Motivation**: Traditional methods apply the same number of refinement steps to all pixels, which is both wasteful and insufficient. Regions with large motion require wider search ranges and more iterations to converge to correct optical flows, whereas stable backgrounds only require a single step.

3. **Complexity-aware Frame Synthesis Network**:

    - **Function**: Adaptively fuses warped features and generates the final intermediate frame based on the MCM.
    - **Mechanism**: Two frames are warped to the target time step using the refined optical flows, then fused through a synthesis network. The MCM plays two roles in the synthesis stage: (1) acting as a soft prior for occlusion masks, where high-MCM regions rely more on the visible frame's information rather than the occluded frame; (2) modulating the intensity of residual learning, where high-complexity regions allow larger residual corrections to handle non-linear motions. The synthesis network adopts a multi-scale architecture, with the MCM guiding feature fusion weights at each scale.
    - **Design Motivation**: Standard linear blending (alpha blending) is sufficient for simple motion, but occluded regions and large-motion areas require a smarter fusion strategy. The MCM provides a unified prior that guides when to trust warping results and when to rely on the predictions of the synthesis network.

### Loss & Training
Training utilizes a weighted combination of multiple losses: (1) reconstruction $L_1$ loss — pixel-level error between the generated frame and the ground truth; (2) perceptual loss — calculating perceptual similarity using features extracted from a pre-trained VGG network; (3) optical flow supervision loss — using optical flow EPE loss on datasets with optical flow ground truths; (4) MCM supervision — weakly supervising the MCM by using the spatial distribution of reconstruction errors as pseudo-labels. Training is conducted in stages: the optical flow estimation and MCM modules are trained first, followed by end-to-end fine-tuning of the entire pipeline.

## Key Experimental Results

### Main Results

| Dataset | Metric | IAM-VFI | RIFE | IFRNet | AMT-S |
|--------|------|---------|------|--------|-------|
| Vimeo90K | PSNR | **Best** | Baseline | Moderate | Second Best |
| UCF101 | PSNR | **Best** | Baseline | Moderate | Second Best |
| SNU-FILM Hard | PSNR | **Significantly Best** | Significant degradation | Moderate | Second Best |
| SNU-FILM Extreme | PSNR | **Significantly Best** | Severe degradation | Significant degradation | Moderate |

### Ablation Study

| Configuration | PSNR | Description |
|------|------|------|
| Full model | Highest | Complete IAM-VFI |
| w/o MCM | Significant Drop | Significant degradation after removing complexity awareness |
| w/o MCM-guided refinement | Drop | Poorer performance when using uniform iteration steps |
| w/o MCM-guided synthesis | Drop | Loss of adaptive capability during the synthesis stage |
| MCM + Simple backbone | Still improves | MCM is generalizable to different backbones |

### Key Findings
- The contribution of the MCM is most prominent in "difficult" scenarios — with gains on the Hard and Extreme splits of SNU-FILM being much larger than those on the Easy and Medium splits.
- The Motion Complexity Map correlates highly with the actual distribution of interpolation errors, indicating that the MCM indeed learns a meaningful prior of motion difficulty.
- Adaptive optical flow refinement is particularly effective in large-motion scenarios, as large motions require more iterations to converge.
- The computational overhead of the MCM is minimal (utilizing a lightweight encoder-decoder head), but it yields significant gains, offering high cost-effectiveness.
- In occluded regions, MCM values are noticeably high, which is highly consistent with the results of forward-backward consistency checks.

## Highlights & Insights
- **Motion Complexity as a Unified Prior**: Occlusion detection, large motion handling, and non-linear motion compensation were previously tackled as separate sub-problems. The MCM unifies them into a single spatial complexity map to guide the entire pipeline. This "one-map-fits-all" strategy is both simple and effective.
- **Adaptive Computation Allocation Concept**: Different regions employ varying numbers of refinement steps, echoing the concept of adaptive computation. This design can be transferred to other dense prediction tasks — such as optical flow estimation and stereo matching — to adaptively allocate computational loads to regions of different difficulties.
- **Weakly Supervised MCM Learning**: Utilizing reconstruction errors as pseudo-labels for the MCM is a clever bootstrapping strategy that bypasses the need for explicit motion complexity annotations.

## Limitations & Future Work
- The definition and learning target of the MCM are somewhat heuristic, lacking a rigorous theoretical formulation — what is the optimal way to quantify "motion complexity"?
- Adaptive iterations can be difficult to parallelize efficiently in practice (varying steps for different regions), which may lead to suboptimal GPU utilization.
- For extreme large-scale motion (e.g., abrupt changes across the entire frame), the MCM might label the entire map as high complexity, degrading the system back to uniform processing.
- Combining the MCM with temporal information remains unexplored — leveraging the temporal trends of MCMs across multiple frames could yield better motion predictions.
- Comparisons with recent diffusion-based VFI methods are missing.

## Related Work & Insights
- **vs RIFE**: A lightweight method using coarse-to-fine bidirectional optical flow and simple linear blending; it is fast but performs poorly in large-motion and occluded regions.
- **vs AMT**: Introduces a transition framework from optical flow to all-pairs matching but still employs a uniform processing strategy for all regions.
- **vs IFRNet**: Performs frame interpolation using intermediate feature flow, achieving a good balance between speed and quality, but lacks motion complexity awareness.
- **vs EMA-VFI**: Uses motion-aware adaptive fusion, which shares some similarity with the MCM concept of IAM-VFI but follows a different implementation path.

## Rating
- Novelty: ⭐⭐⭐⭐ The Motion Complexity Map is a novel concept in VFI, unifying multiple motion cues.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation on standard datasets is comprehensive, but analyses on extreme scenarios could be deeper.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, and the visualization of the MCM is intuitive.
- Value: ⭐⭐⭐⭐ The adaptive processing concept holds practical value in the VFI domain and can be generalized to other video tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions](../../CVPR2025/video_understanding/bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)
- [\[ECCV 2024\] Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation](motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)
- [\[ECCV 2024\] UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation](uniinr_event-guided_unified_rolling_shutter_correction_deblurring_and_interpolat.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](self-supervised_any-point_tracking_by_contrastive_random_walks.md)

</div>

<!-- RELATED:END -->
