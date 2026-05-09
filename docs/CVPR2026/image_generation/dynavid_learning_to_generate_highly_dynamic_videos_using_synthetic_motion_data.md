---
title: >-
  [Paper Note] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data
description: >-
  [CVPR 2026][Image Generation][Video Diffusion Models] DynaVid proposes utilizing synthetic optical flow rendered via computer graphics (rather than synthetic videos) to train video diffusion models. Through a two-stage framework consisting of a motion generator and a motion-guided video generator, it achieves realistic video synthesis of highly dynamic motions and fine-grained camera control.
tags:
  - CVPR 2026
  - Image Generation
  - Video Diffusion Models
  - Synthetic Motion Data
  - Optical Flow
  - Dynamic Motion Generation
  - Camera Control
date: 2026-05-08
content_hash: c4a8ebdfb696679b
---

# DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data

**Conference**: CVPR 2026  
**arXiv**: [2604.01666](https://arxiv.org/abs/2604.01666)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Video Diffusion Models, Synthetic Motion Data, Optical Flow, Dynamic Motion Generation, Camera Control

## TL;DR
DynaVid proposes utilizing synthetic optical flow rendered via computer graphics (rather than synthetic videos) to train video diffusion models. Through a two-stage framework consisting of a motion generator and a motion-guided video generator, it achieves realistic video synthesis of highly dynamic motions and fine-grained camera control.

## Background & Motivation

1. **Background**: Current video diffusion models (e.g., Wan2.2, CogVideoX) can generate high-quality videos but still rely heavily on the distribution of motion types in large-scale training data. In mainstream training datasets, highly dynamic motion scenes (e.g., street dance, fast camera rotations) are extremely scarce.

2. **Limitations of Prior Work**:
    - Models struggle to synthesize realistic videos containing highly dynamic motions due to insufficient samples in training sets.
    - Camera control models (e.g., CameraCtrl, AC3D) require accurate 3D camera pose annotations, but pose estimation is highly unreliable in extreme motion scenarios.
    - Directly using synthetic rendered videos for training introduces a severe appearance domain gap—models replicate the artificial textures and lighting of synthetic data.

3. **Key Challenge**: Synthetic data can provide rich dynamic motion and precise control signals, but its non-realistic appearance features pollute the visual quality of generative models. How to "keep the essence (motion information) and discard the dross (artificial appearance)" is a key challenge.

4. **Goal**
    - How can models learn highly dynamic motion patterns without sacrificing visual realism?
    - How to achieve precise camera trajectory control under extreme camera movement?

5. **Key Insight**: Optical flow naturally encodes only motion information and is decoupled from appearance. Therefore, replacing rendered videos with rendered optical flow can eliminate the appearance domain gap.

6. **Core Idea**: Use synthetic optical flow (instead of synthetic videos) to train a motion generator to learn dynamic motion patterns, then use real videos to train a motion-guided video generator to maintain realistic appearance. This two-stage decoupling achieves "motion from synthetic, appearance from real."

## Method

### Overall Architecture
DynaVid is a two-stage video generation framework. In the first stage, the **Motion Generator** accepts text conditions to generate motion sequences represented as optical flow. In the second stage, the **Motion-Guided Video Generator** synthesizes RGB video frames conditioned on the generated optical flow. Both stages are based on the Wan2.2-5B video diffusion model and use the VACE architecture to inject control signals. For camera control scenarios, the motion generator additionally receives Plücker embeddings as camera parameter inputs.

### Key Designs

1. **Synthetic Motion Dataset Construction (DynaVid-Human & DynaVid-Camera)**:
    - Function: Provides highly dynamic motion supervision signals and precise camera control annotations.
    - Mechanism: Uses the Blender Cycles renderer to construct 3D scenes. DynaVid-Human integrates Mixamo motion sequences and animatable human models to render optical flow of dynamic human movement. DynaVid-Camera interpolates key camera positions using NURBS curves in complex 3D environments to generate trajectories and optical flow of rapid viewpoint changes. The core output is synthetic optical flow $\mathcal{F}^{syn}$ rather than synthetic videos.
    - Design Motivation: Optical flow only encodes motion information and is completely decoupled from appearance, avoiding the non-realistic appearance issues of synthetic videos.

2. **HSV-RGB Conversion of Optical Flow and VAE Encoding**:
    - Function: Maps optical flow to the RGB domain to reuse a pre-trained VAE.
    - Mechanism: First normalizes the optical flow vectors using the 99th percentile as a scaling factor $s_f$, then maps the normalized magnitude and direction to the Value (V) and Hue (H) channels of the HSV color space, respectively, and finally converts to RGB for encoding into the latent space.
    - Design Motivation: Avoids training a separate VAE for optical flow and directly reuses existing video VAEs to reduce training costs.

3. **Two-Stage Decoupled Training Strategy**:
    - Function: Achieves the decoupling of "learning motion from synthetic data and appearance from real data."
    - Mechanism: The motion generator is first pre-trained on real optical flow $\mathcal{F}^{real}$ (estimated from real videos using WAFT) to learn general motion statistics, then fine-tuned on synthetic optical flow $\mathcal{F}^{syn}$ to learn dynamic motions. During fine-tuning, each batch mixes real and synthetic optical flow to prevent forgetting. The motion-guided video generator is trained only on real video-optical flow pairs to learn how to translate optical flow conditions into realistic appearances. Both stages use the Flow Matching objective function.
    - Design Motivation: Mixed training prevents the model from overfitting to synthetic motion patterns while losing natural motion priors; training the video generator only on real data ensures appearance realism.

### Loss & Training
- The motion generator adopts the Flow Matching objective: $\mathbb{E}[\|\hat{u}^{\mathcal{F}}(\mathcal{F}_{t_f}; c_{txt}, C, t_f) - v^{\mathcal{F}}\|_2^2]$
- The motion-guided video generator also adopts Flow Matching: $\mathbb{E}[\|\hat{u}^{\mathcal{I}}(\mathcal{I}_{t_I}; c_{txt}, \mathcal{F}, t_I) - v^{\mathcal{I}}\|_2^2]$
- Data Filtering: Inaccurate real optical flow-video pairs are filtered via optical flow cycle-consistency error (threshold 1.19 pixels, 90th percentile) to improve the motion fidelity of the motion-guided video generator.

## Key Experimental Results

### Main Results — Dynamic Object Motion Generation

| Method | Dataset | FVD↓ | A-Qual↑ | I-Qual↑ | M-Smooth↑ | T-Flick↑ |
|------|--------|------|---------|---------|-----------|----------|
| CogVideoX-5B | Pexels | 1519.54 | 0.5646 | 0.6613 | 0.9844 | 0.9673 |
| Wan2.2-5B | Pexels | 1172.02 | 0.5779 | 0.7235 | 0.9928 | 0.9883 |
| **Ours** | **Pexels** | **1126.38** | **0.5807** | **0.7342** | 0.9900 | 0.9748 |
| CogVideoX-5B | DynaVid-Human | 2238.68 | 0.5071 | 0.5562 | 0.9779 | 0.9565 |
| Wan2.2-5B | DynaVid-Human | 1775.99 | 0.5389 | 0.6974 | 0.9904 | 0.9791 |
| **Ours** | **DynaVid-Human** | **1351.94** | 0.5312 | **0.7352** | 0.9931 | 0.9864 |

### Main Results — Extreme Camera Control

| Method | Dataset | mRotErr↓ | FVD↓ | A-Qual↑ | I-Qual↑ |
|------|--------|----------|------|---------|---------|
| AC3D | DynaVid-Camera | 1.1529 | 782.01 | 0.4483 | 0.5407 |
| GEN3C | DynaVid-Camera | 1.1852 | 237.15* | 0.3889 | 0.5659 |
| **Ours** | **DynaVid-Camera** | **0.9289** | 674.72 | **0.4501** | **0.6713** |

### Ablation Study

| Configuration | Pexels FVD↓ | DynaVid-Human FVD↓ | Description |
|------|-------------|---------------------|------|
| Full model | 1126.38 | 1351.94 | Complete model |
| w/o Synthetic Motion Data | 1076.53 | 1878.98 | Severe degradation in dynamic scenes (+527 FVD) |
| w/o batch mixture | 1885.74 | 1229.70 | Severe degradation on Pexels, overfits synthetic patterns |
| w/ Synthetic Video (not flow) | 1230.81 | 698.0* | Produces artificial appearance, Pexels degrades |

### Key Findings
- **Synthetic motion data is the key to dynamic scene performance**: Without it, DynaVid-Human FVD surges from 1352 to 1879, with little change on standard Pexels scenes.
- **Mixed-batch training is crucial**: Fine-tuning only on synthetic data leads to severe overfitting, with Pexels FVD increasing to 1886.
- **Correctness of using optical flow instead of rendered video**: Training with synthetic videos yields low FVD on synthetic test sets but produces an artificial appearance, causing FVD on real-world Pexels scenes to rise significantly.
- The motion-guided video generator remains robust at noise levels above 20dB.

## Highlights & Insights
- **Using optical flow instead of rendered video to eliminate domain gap**: This idea is ingenious—optical flow naturally encodes motion without appearance, perfectly solving the contradiction of synthetic data being "useful but looking fake." This strategy of "selecting the right intermediate representation to bridge synthetic and real" is transferable to other cross-domain learning scenarios.
- **Two-stage decoupled framework**: The decoupled design of motion and appearance allows both to be trained with the most appropriate data sources. This decoupling logic can be used for any generative task requiring separate modeling of content and motion.
- **Data filtering based on cycle consistency**: A simple but effective quality control method; a 90th percentile threshold significantly enhances motion fidelity.

## Limitations & Future Work
- The synthetic dataset primarily focuses on single-person scenes, and the synthesis of multi-person dynamic scenes is less effective.
- Relying on the quality of the optical flow estimator; estimation errors affect the accuracy of training data.
- More complex motion representations (e.g., scene flow, 3D motion fields) have not been explored; using only 2D optical flow may limit 3D consistency.
- The diversity of the synthetic dataset can be expanded (multi-person interactions, object motions, etc.).

## Related Work & Insights
- **vs Wan2.2-5B / CogVideoX**: These general video generation models are limited by the lack of dynamic motion in training data. DynaVid fills this data gap by introducing synthetic motion data.
- **vs HyperMotion**: HyperMotion relies on the first frame as input and easily produces an artificial appearance; DynaVid is pure text-to-video and requires no additional input.
- **vs AC3D / GEN3C**: Both perform poorly under extreme camera motion; DynaVid learns patterns of rapid viewpoint changes through synthetic motion data.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using optical flow instead of rendered video to eliminate domain gaps is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations covering both dynamic motion and camera control scenarios, with noise robustness verification.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and well-articulated motivation.
- Value: ⭐⭐⭐⭐ Provides a general framework for utilizing synthetic data to enhance video generation capabilities.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](ahs_adaptive_head_synthesis.md)
- [\[CVPR 2026\] Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training](beyond_the_golden_data_resolving_the_motion-vision_quality_dilemma_via_timestep_.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[NeurIPS 2025\] OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data](../../NeurIPS2025/image_generation/osmgen_highly_controllable_satellite_image_synthesis_using_openstreetmap_data.md)

<!-- RELATED:END -->
