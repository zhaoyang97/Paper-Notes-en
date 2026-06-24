---
title: >-
  [Paper Note] You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale
description: >-
  [CVPR 2025][3D Vision][Multi-view Diffusion] This paper presents See3D, a pose-free visually conditioned multi-view diffusion model trained on large-scale internet videos (320M frames / 16M video clips). Through an automated data filtering pipeline and a time-dependent visual conditioning design, it achieves zero-shot open-world 3D generation capabilities.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-view Diffusion"
  - "Pose-Free Training"
  - "Video Data"
  - "3D Generation"
  - "Large-Scale Learning"
date: 2026-05-08
content_hash: 95213799a2a25d92
---

# You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale

**Conference**: CVPR 2025  
**arXiv**: [2412.06699](https://arxiv.org/abs/2412.06699)  
**Code**: [Project Page](https://vision.baai.ac.cn/see3d)  
**Area**: 3D Vision  
**Keywords**: Multi-view Diffusion, Pose-Free Training, Video Data, 3D Generation, Large-Scale Learning

## TL;DR

This paper presents See3D, a pose-free visually conditioned multi-view diffusion model trained on large-scale internet videos (320M frames / 16M video clips). Through an automated data filtering pipeline and a time-dependent visual conditioning design, it achieves zero-shot open-world 3D generation capabilities.

## Background & Motivation

- **Scarcity of 3D Data**: Existing 3D generative models rely on expensive 3D "gold labels" (such as Objaverse's 0.8M objects) or 2D diffusion priors, and their performance is constrained by the limited scale of 3D data. Constructing large-scale 3D datasets remains a burden for academia.
- **Videos as 3D Data Sources**: Human 3D perception stems from multi-view observations rather than specific 3D representations. Internet videos provide a vast, diverse, and low-cost source of multi-view images.
- **Key Challenge**: (1) To identify and filter out 3D-aware video clips (static scenes + large view changes) from raw videos, (2) to learn general 3D priors without 3D geometric or camera pose annotations.
- **Pose Annotation Bottleneck**: Existing multi-view diffusion models usually require precise camera poses as conditional inputs, but annotating poses for web-scale videos is extremely expensive or unfeasible.
- **Core Idea**: "You See it, You Got it" — acquiring 3D knowledge solely by watching video content. The authors propose a purely 2D-inductive visual condition (instead of 3D-inductive pose condition), enabling the model to be trained at scale on videos without pose annotations.

## Method

### Overall Architecture

See3D consists of three core components:
1. **Data Filtering Pipeline**: Automatically filters 15.99M 3D-aware video clips from 25.48M raw videos.
2. **Visually Conditioned Multi-View Diffusion Model**: Learns camera control through time-dependent visual conditions without pose annotations.
3. **Warping-Based 3D Generation Framework**: Leverages See3D for long-sequence novel view synthesis.

### Key Designs

**1. Automated Video Data Filtering Pipeline**
- **Function**: Identifies and extracts 3D-aware data (static scenes + large view changes) from massive internet videos.
- **Mechanism**: A four-step pipeline — (a) spatial-temporal downsampling for efficiency, (b) Mask R-CNN semantic recognition for dynamic objects (humans, animals, etc.), (c) optical flow estimation for precise filtering of dynamic regions, and (d) keypoint trajectory tracking to filter out videos with small view changes. Ultimately, the WebVi3D dataset containing 320M frames was collected from four sources: Pexels, Artgrid, Airvuz, and Skypixel.
- **Design Motivation**: Dynamic content distorts scene geometry, and small view changes do not provide sufficient 3D observations. Manual annotation verification shows the pipeline achieves an accuracy of 88.6%.

**2. Time-Dependent Visual Conditions**
- **Function**: Implicitly controls camera motion without pose annotations, and generalizes to downstream task-specific visual conditions.
- **Mechanism**: A three-step processing is applied to the target image $G$: (1) random irregular masking to reduce direct reliance on pixel signals, (2) adding time-dependent noise $C_t = \sqrt{\bar{\alpha}_{t'}} (1-M)X_0 + \sqrt{1-\bar{\alpha}_{t'}} \epsilon$, controlling the level of signal leakage via the function $t' = f(t)$, and (3) time-dependent blending $V_t = W_t \cdot C_t + (1-W_t) \cdot X_t$, ensuring the model relies on visual cues at large timesteps and on $X_t$ at small timesteps, which mitigates domain gap.
- **Design Motivation**: Directly using video frames as conditions leads to overfitting on video data (signal leakage), while pure noise loses camera control. The time-dependent noise + blending mechanism balances the two, turning the condition into a general "visual hint."

**3. Iterative Sparse Pixel-Level Depth Alignment**
- **Function**: Corrects geometry estimation errors in warping-based 3D generation to prevent cumulative degradation.
- **Mechanism**: During iterative generation, correspondences between anchor views are established via keypoint matching. Monocular depth estimation and sparse alignment are utilized to recover dense depth maps for warping subsequent views. This includes two steps: pixel-level depth scale alignment and global metric depth recovery.
- **Design Motivation**: Directly using monocular depth results in distortion and stretching of the warped image due to scale ambiguity and estimation errors, which accumulates severely during iteration.

### Loss & Training

Standard conditional diffusion training objective:
$$\mathbb{E}_{X_0, Y_0, \epsilon, t}\left[\|\epsilon_\theta(X_t, Y_0, V_t, t) - \epsilon\|_2^2\right]$$

where $V_t$ is the visual condition, $Y_0$ is the reference view, and the loss is computed only on the target image.

## Key Experimental Results

### Main Results: Single-View 3D Reconstruction (GSO Dataset)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Data |
|------|:---:|:---:|:---:|:---:|
| Zero123++ | Medium | Medium | Medium | Objaverse (0.8M) |
| SV3D | Mid-High | Mid-High | Mid-Low | Objaverse + Video |
| **Ours** | **Highest** | **Highest** | **Lowest** | **WebVi3D (16M)** |

### Sparse-View Reconstruction (DTU Dataset)

| Method | PSNR ↑ | SSIM ↑ | Training Data |
|------|:---:|:---:|:---:|
| PixelNeRF | Low | Low | 3D Data |
| ZeroNVS | Medium | Medium | 3D + pose |
| **Ours** | **Highest** | **Highest** | **Video (Pose-Free)** |

### Key Findings

- See3D is trained solely on video data yet outperforms models trained on expensive 3D datasets (which require pose annotations) across multiple benchmarks.
- The WebVi3D dataset is 20 times larger than Objaverse and 200 times larger than RealEstate10K.
- The visual conditions do not require pose annotations and can be transferred zero-shot to warping-based generation and 3D editing tasks.
- Data scale is critical — ablation studies show that more data consistently improves performance.
- The model is effective for both object-level and scene-level 3D generation.

## Highlights & Insights

1. **Data-Driven 3D Learning Paradigm**: Demonstrates that large-scale video data can replace expensive 3D annotation data, opening a new path for scalable training in 3D generation.
2. **Universality of Visual Conditions**: The time-dependent noise + blending mechanism makes the same condition format applicable to different domains in training (videos) and inference (warped images).
3. **Practical Value of the Automated Data Pipeline**: The pipeline with 88.6% accuracy can sustainably acquire 3D data from the continuously growing pool of internet videos.
4. **Breakthrough in "Pose-Free" Training**: Completely eliminates the need for pose annotations, allowing training to scale to any size.

## Limitations & Future Work

- Video data primarily comes from natural scenes, which may lack sufficient coverage of man-made objects and indoor scenes.
- Camera control accuracy in pose-free training may not match that of pose-conditioned methods.
- Iteratively generated long sequences may still accumulate errors.
- Future research can combine video pre-training with a small amount of 3D fine-tuning to achieve better geometric accuracy.

## Related Work & Insights

- **Zero123/Zero123++**: Multi-view generation trained on Objaverse, but limited by data scale.
- **SV3D**: Combines video and 3D data, but still requires camera poses.
- **SDS Series**: Optimizes 3D using 2D diffusion priors, but results in poor 2D-to-3D transition quality.
- **Emu3**: Scene segmentation methods in video understanding; this paper uses a similar pipeline for data filtering.
- **Insight**: AI systems can "learn 3D by watching" similar to humans; large-scale multi-view observations are the key to scalable 3D understanding.

## Rating

⭐⭐⭐⭐⭐ — This is a work with the potential to shift the paradigm of 3D generation. The combination of data scale (16M video clips / 320M frames) and pose-free training makes it highly scalable. The visual conditioning design is elegant and general. It outperforms methods trained on expensive 3D data across multiple benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] Perceptual Inductive Bias is What You Need Before Contrastive Learning](perceptual_inductive_bias_is_what_you_need_before_contrastive_learning.md)
- [\[ICCV 2025\] Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels](../../ICCV2025/3d_vision/do_it_yourself_learning_semantic_correspondence_from_pseudo-labels.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](../../NeurIPS2025/3d_vision/you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)

</div>

<!-- RELATED:END -->
