---
title: >-
  [Paper Note] HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation
description: >-
  [CVPR 2025][Video Generation][Portrait Animation] HunyuanPortrait proposes the first implicit condition portrait animation framework based on Stable Video Diffusion, achieving high-fidelity control of fine facial dynamics and robust identity consistency through an intensity-aware motion encoder and an ID-aware multi-scale adapter.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Portrait Animation"
  - "Implicit Motion Representation"
  - "Video Diffusion Models"
  - "Identity Preservation"
  - "Facial Dynamics"
date: 2026-05-08
content_hash: 31c785a08758b861
---

# HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation

**Conference**: CVPR 2025  
**arXiv**: [2503.18860](https://arxiv.org/abs/2503.18860)  
**Code**: Yes (provided on project page)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Portrait Animation, Implicit Motion Representation, Video Diffusion Models, Identity Preservation, Facial Dynamics

## TL;DR
HunyuanPortrait proposes the first implicit condition portrait animation framework based on Stable Video Diffusion, achieving high-fidelity control of fine facial dynamics and robust identity consistency through an intensity-aware motion encoder and an ID-aware multi-scale adapter.

## Background & Motivation

**Background**: The field of portrait animation has made significant progress in recent years. Methods of the GAN era (e.g., HeadGAN, LivePortrait) achieve facial animation through warping strategies but have limited generalization capabilities. Methods of the diffusion model era (e.g., AniPortrait, EchoMimic) fine-tune SD models and incorporate motion modules, yet they still face challenges such as temporal smoothness and fine-grained facial detail control.

**Limitations of Prior Work**: Existing methods exhibit three key deficiencies: (1) Limitations of explicit keypoint control—facial keypoints shift during cross-identity driving due to significant variations in facial shapes, leading to identity distortion and inaccurate facial control; (2) Temporal instability—methods based on image diffusion models + separately trained motion modules lack motion pre-training, resulting in poor frame rate adaptability; (3) Loss of fine facial dynamics—explicit keypoints fail to capture fine details such as micro-expressions and gaze direction.

**Key Challenge**: The diversity of facial shapes makes it difficult for explicit keypoints to accurately transfer motion information across different identities; when there is a significant geometric discrepancy between the driving video and the source image, the keypoint alignment strategy fails.

**Goal**: (1) How to accurately transfer motion information across different facial geometries? (2) How to maintain high temporal consistency? (3) How to capture fine-grained facial dynamics (micro-expressions, gaze direction, lip-sync)?

**Key Insight**: Replace explicit keypoints with implicit motion representations (feature vectors extracted from cropped facial regions by a pre-trained encoder) to encode motion information. Implicit representations are naturally unaffected by variations in facial shapes and can encode fine dynamics that keypoints cannot capture. Stable Video Diffusion (SVD) is used as the base model to leverage pre-trained temporal modeling capabilities.

**Core Idea**: Utilize implicit motion representation + intensity-aware encoder + motion memory bank to achieve fine dynamic control across diverse facial geometries, combined with an ID-aware multi-scale adapter for strong identity preservation.

## Method

### Overall Architecture
Based on Stable Video Diffusion (SVD), the input consists of a single portrait image as the appearance reference and a driving video. The framework comprises two core components: an appearance extractor (handling identity and background) and a motion extractor (handling facial dynamics). The appearance and motion features are injected into the denoising UNet through meticulously designed attention layers. Simultaneously, spatial conditions (DWPose skeletons) are integrated to ensure stability in non-facial regions.

### Key Designs

1. **Intensity-Aware Motion Extractor**:

    - **Function**: Extracts identity-agnostic fine motion features from the driving video, adapting to different motion intensities.
    - **Mechanism**: First, the facial center region (from eyebrows to the bottom of the mouth) is cropped to reduce interference from background noise and facial shapes. The cropped pixels are input into a pre-trained motion encoder (MegaPortraits) to obtain coarse implicit motion features $F_m$. Subsequently, two dimensions of motion intensity are calculated: expression intensity $I_e$ (standard deviation of keypoints relative to the mean, normalized by facial scale) and head pose intensity $I_h$ (standard deviation of facial center displacement). The continuous intensity values are discretized into 64 levels and mapped into embedding vectors $E_s$, which are injected into the motion features via AdaLN. Finally, a motion memory bank (64 learnable memory vectors of dimension 768) is introduced to interact with motion features through cross-attention, supplementing contextual information to enhance temporal modeling.
    - **Design Motivation**: Different motion intensities (e.g., slight blinking vs. large head rotation) affect the generated pixels differently; the intensity-aware encoder allows the model to adapt to a wide dynamic range. The motion memory bank compensates for the lack of cross-frame context in pixel-level extracted motion features.

2. **ID-Aware Multi-Scale Adapter (IMAdapter)**:

    - **Function**: Enhances the identity consistency of the generated videos, preserving facial textures and geometric details.
    - **Mechanism**: Building upon the DiNOv2 backbone (a patch-level image encoder with frozen parameters), IMAdapter is introduced to enhance identity capabilities. The workflow of IMAdapter is as follows: (1) perform linear projection to reduce dimensions and obtain low-rank features $\hat{f}_a$; (2) apply multi-scale convolution (MConv) in parallel and concatenate along the channel dimension; (3) perform cross-attention using ArcFace's ID features $f_{id}$ as the query and the multi-scale convolutional features as key/value; (4) linearly project back to the original dimension with a residual connection. During training, the ID information is randomly sampled from frames of the video sequence, while the reference image is used during inference.
    - **Design Motivation**: Existing video diffusion models are relatively weak in identity preservation. DiNOv2 provides rich visual details (clothing, background) and ArcFace provides precise ID features; IMAdapter fuses the two, enhancing identity awareness while retaining the pre-trained knowledge of DiNOv2 (parameters frozen).

3. **Training and Inference Enhancement Strategies**:

    - **Function**: Enhances the generalization capability and cross-identity driving ability of the model.
    - **Mechanism**: (1) Use AnimeGANv3 for style-transfer data augmentation to adapt the model to diverse image styles; (2) apply color jitter to the facial crop input of the motion encoder to reduce the influence of skin tone on motion information; (3) use DWPose as spatial conditions, randomly deleting skeleton edges during training to enhance robustness; (4) perform skeleton translation and scaling adaptation based on the nose tip offset during inference, avoiding the use of eye keypoints to prevent facial shape drift.
    - **Design Motivation**: Large differences in skeleton positions and proportions during cross-identity driving introduce distortions if used directly. The random-edge-deletion augmentation strategy makes the model less sensitive to detector accuracy.

### Loss & Training
The standard diffusion training objective (noise prediction MSE loss) is utilized. Optimized using AdamW with a learning rate of $1\times10^{-5}$ and gradient clipping of 0.99. Trained on 128 A100 GPUs for 3 days. Inference employs the DDIM sampler with a classifier-free guidance scale of 2.0. The ID encoder, VAE, and DiNOv2 parameters are frozen.

## Key Experimental Results

### Main Results

| Method | LMD↓ | FID-VID↓ | FVD↓ | PSNR↑ | SSIM↑ | LPIPS↓ | ID Similarity↑ |
|------|------|----------|------|-------|-------|--------|----------------|
| LivePortrait | 9.14 | 82.71 | 483.38 | 31.41 | 0.72 | 0.22 | 8.71 |
| AniPortrait | 6.67 | 81.90 | 430.24 | 30.54 | 0.67 | 0.27 | 7.95 |
| X-Portrait | 6.23 | 82.93 | 416.41 | 30.81 | 0.71 | 0.19 | 8.03 |
| **HunyuanPortrait** | **2.02** | **75.81** | **333.48** | **32.98** | **0.81** | **0.11** | **8.87** |

In cross-reenactment user studies, HunyuanPortrait leads comprehensively across three dimensions: facial motion (4.55), video quality (4.69), and temporal smoothness (4.61).

### Ablation Study

| Configuration | ID Similarity↑ | FID-VID↓ | FVD↓ | LMD↓ |
|------|----------------|----------|------|------|
| Full model | **8.87** | **75.81** | **333.48** | **2.02** |
| - Memory Bank | 8.75 | 78.43 | 361.94 | 2.78 |
| - IAME | 8.63 | 80.79 | 385.43 | 4.01 |
| - ID Features | 8.21 | 75.03 | 330.12 | 1.98 |
| - IMAdapter | 8.09 | 77.14 | 352.67 | 2.54 |

### Key Findings
- LMD (Landmark Distance) leads by a large margin: 2.02 vs. 5.63 (second best), demonstrating that implicit motion representations far outperform explicit keypoint methods in facial dynamic control.
- Removing IAME has the most significant impact: FVD increases by 15.6%, indicating that the intensity-aware encoder is critical for video quality.
- Removing ID Features slightly reduces LMD (1.98), but ID similarity drops by 7.4%, showing that while ID constraints slightly limit motion freedom, they are crucial for fidelity.
- The motion memory bank improves fine facial details (such as more prominent forehead wrinkles when eyebrows are raised) and temporal smoothness.
- The pre-trained temporal modeling capability of SVD enables HunyuanPortrait to remain smooth under varying frame rates, an advantage that SD-based methods lack.

## Highlights & Insights
- **Paradigm Shift of Implicit Motion Representation**: The transition from explicit keypoints to implicit features is a key innovation. Implicit representations naturally bypass keypoint distribution discrepancies caused by varying facial geometries, and they can encode fine-grained dynamics (such as micro-expressions and gaze direction) that explicit keypoints fail to capture. This paradigm can be extended to fields such as full-body animation and gesture-driven synthesis.
- **Utility of Intensity-Aware Design**: Discretizing and embedding motion intensity (degree of expression deformation + head pose amplitude) into the feature space allows the model to adaptively adjust generation based on the scale of motion—focusing on structures for large movements, and details for small ones.
- **Selecting SVD over SD**: Leveraging SVD's pre-trained temporal prior bypasses the need for training separate motion modules, fundamentally resolving frame-rate adaptation and temporal smoothness challenges.

## Limitations & Future Work
- The pre-trained motion encoder (MegaPortraits) has not completely decoupled identity and motion information, requiring advanced training strategies to compensate.
- The facial cropping region is fixed (from eyebrows to the bottom of the mouth), which may truncate valid information during large head movements.
- The training resource requirement is extremely high (128 A100 GPUs for 3 days), making replication challenging.
- Spatial conditions (DWPose skeletons) remain explicit, meaning that full-body motion control is still limited by detector accuracy.
- Future directions: Explore fully end-to-end motion decoupling, support audio-driven animation (currently only video-driven is supported), and reduce model size and training costs.

## Related Work & Insights
- **vs. LivePortrait**: Warping-based methods cannot handle large head rotations and tend to jitter in complex backgrounds. HunyuanPortrait is diffusion-based, offering robust generalization.
- **vs. AniPortrait / FollowYE**: Explicit keypoint methods are highly affected by disputes in facial geometries, suffering from severe identity shift during cross-identity driving; HunyuanPortrait completely circumvents this issue using implicit representations.
- **vs. X-Portrait**: Also a diffusion-based method but built on Stable Diffusion (SD), suffering from poor temporal smoothness and insufficient facial details. HunyuanPortrait is built on SVD and additionally incorporates motion enhancement modules.
- It can serve as a foundation framework for digital human video generation, combined with TTS to achieve end-to-end talking head synthesis.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of implicit conditioning and SVD is clear and effective; the design of each component is exquisite, though not a revolutionary breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering self-reconstruction, cross-reconstruction, ablation studies, and user studies, evaluated on multiple datasets and compared with multiple SOTA baselines.
- Writing Quality: ⭐⭐⭐⭐ Detailed methodology and sufficient experimental analysis, but the paper is relatively long.
- Value: ⭐⭐⭐⭐⭐ Outperforms existing methods by a significant margin in practical effects, with extremely high industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](spatiotemporal_skip_guidance_for_enhanced_video_diffusion_sampling.md)
- [\[CVPR 2026\] FlashPortrait: 6× Faster Infinite Portrait Animation with Adaptive Latent Prediction](../../CVPR2026/video_generation/flashportrait_6x_faster_infinite_portrait_animation_with_adaptive_latent_predict.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2026\] PersonaLive! Expressive Portrait Image Animation for Live Streaming](../../CVPR2026/video_generation/personalive_expressive_portrait_image_animation_for_live_streaming.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](../../CVPR2026/video_generation/facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)

</div>

<!-- RELATED:END -->
