---
title: >-
  [Paper Note] 3D Visual Illusion Depth Estimation
description: >-
  [NeurIPS 2025][3D Vision][3D visual illusion] This paper reveals that 3D visual illusions (e.g., wall paintings, screen replays, mirror reflections) severely mislead existing state-of-the-art monocular and stereo depth estimation methods. The authors construct a large-scale dataset comprising approximately 3k scenes and 200k images, and propose a VLM-driven monocular-stereo adaptive fusion framework that achieves state-of-the-art performance across diverse illusion scenarios.
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D visual illusion"
  - "depth estimation"
  - "monocular-stereo fusion"
  - "vision-language model"
  - "Flow Matching"
date: 2026-05-08
content_hash: 0b77dda6d9cb80c5
---

# 3D Visual Illusion Depth Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2505.13061](https://arxiv.org/abs/2505.13061)  
**Code**: [GitHub](https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation)  
**Area**: 3D Vision / Depth Estimation
**Keywords**: 3D visual illusion, depth estimation, monocular-stereo fusion, vision-language model, Flow Matching

## TL;DR
This paper reveals that 3D visual illusions (e.g., wall paintings, screen replays, mirror reflections) severely mislead existing state-of-the-art monocular and stereo depth estimation methods. The authors construct a large-scale dataset comprising approximately 3k scenes and 200k images, and propose a VLM-driven monocular-stereo adaptive fusion framework that achieves state-of-the-art performance across diverse illusion scenarios.

## Background & Motivation
Depth estimation is critical for downstream applications such as AR/VR and robotics. Current monocular methods (e.g., DepthAnything V2, Marigold) and stereo methods (e.g., RAFT-Stereo, IGEV) have approached human-level performance on ordinary scenes, yet they fail severely in **3D visual illusion scenarios**—paintings on flat surfaces, printed images, screen content, holographic projections, and mirror/transparent objects all induce erroneous depth predictions. Such illusions are pervasive in the real world and pose threats to safety-critical applications such as autonomous driving and robot navigation, yet systematic research and evaluation benchmarks have been lacking.

## Core Problem
1. **Quantitatively revealing** the impact of 3D visual illusions on depth estimation—how different illusion types separately mislead monocular and stereo methods;
2. **Constructing a large-scale benchmark** for systematic evaluation;
3. **Designing a fusion framework** that exploits the complementarity of monocular and stereo methods to resist illusions.

Key insight: monocular methods rely on texture cues (shape, perspective, shading) and are easily deceived by 3D textures simulated on flat surfaces, yet can handle mirrors through learned priors; stereo methods rely on pixel matching and are immune to texture illusions, but fail on mirror/transparent surfaces due to overlapping reflections. The two approaches are thus **strongly complementary**.

## Method

### Overall Architecture
The proposed VLM-Driven Monocular-Stereo Fusion Model consists of two core components:
1. **Dual-branch prediction network**: simultaneously outputs a stereo disparity map and a monocular depth map.
2. **VLM fusion network**: leverages the commonsense reasoning capability of a vision-language model to assess the reliability of each depth cue across different regions and generates a confidence map to guide fusion.

### Key Designs
1. **Dual-branch prediction network**:
    - Stereo branch: based on a GRU iterative refinement framework; extracts features from rectified image pairs, constructs a cost volume, and iteratively refines disparity from a zero initialization.
    - Monocular branch: a frozen DepthAnything V2 predicts affine-invariant inverse depth; monocular features are simultaneously extracted as left-view context to assist stereo disparity refinement.
    - Elegant design: monocular features not only produce an independent depth prediction but also feed back into the stereo branch.

2. **VLM prediction stage**:
    - A pretrained QwenVL2-7B is employed; the visual prompt comprises the left image, the stereo disparity map, and the monocular disparity map.
    - The language prompt is designed from the perspective of "which materials interfere with stereo matching" (e.g., transparent/reflective objects) rather than directly describing complex illusion textures.
    - The last layer of the VLM is fine-tuned with LoRA.

3. **Confidence map generation (Flow Matching)**:
    - Inspired by FLUX, flow matching is used to learn a guided path flow from Gaussian noise to the confidence distribution.
    - Image-text embeddings from the VLM serve as conditioning information, injected via Transformer with cross-attention.
    - A VAE decoder maps the final state back to image space; the result is concatenated with the cost volume and processed by a convolutional layer to predict the confidence map.

4. **Global fusion stage**:
    - An affine transformation aligns monocular disparity to absolute/metric space: $\tilde{D}_m = s_m \cdot D_m + t_m$.
    - Affine parameters are learned via convolution over the concatenation of monocular and stereo disparities.
    - Parameters in low-confidence regions are refined through pooling from high-confidence neighboring regions.
    - The aligned monocular disparity, stereo disparity, and confidence map are concatenated and processed through convolution and upsampling to yield the final high-resolution disparity map.

### Loss & Training
- **Disparity loss** $\mathcal{L}_d$: L1 loss supervising the GRU disparity at each iteration, the aligned monocular disparity, and the final predicted disparity.
- **Confidence map loss** $\mathcal{L}_c$: Focal Loss, with ground truth derived from the discrepancy between the final stereo prediction and the ground truth.
- Training strategy: pre-trained on SceneFlow, then fine-tuned on the synthetic 3D-Visual-Illusion data.
- Trained on 4×H100 GPUs for approximately 20 days with a batch size of 6 per GPU.

## Dataset Construction (3D-Visual-Illusion Dataset)
Five illusion categories are covered:
- **Inpainting illusion** (paintings on walls/floors)
- **Picture illusion** (images printed or drawn on paper)
- **Replay illusion** (content replayed on screens)
- **Holography illusion** (holographic projections)
- **Mirror illusion** (mirror/transparent surfaces)

Synthetic data: 5,226 videos (52M frames) crawled from the web, automatically filtered by Qwen2-VL-72B and manually curated to 1,384 videos/236k frames; mirror-type data generated by Sora/Kling/HunyuanVideo yielding 234 videos/2,382 frames. Depth ground truth is produced via DepthAnything V2 + SAM2 segmentation + RANSAC plane fitting correction.

Real data: ZED Mini stereo camera + RealSense L515 LiDAR, 72 scenes/617 frames, with GT depth obtained through calibration, Z-buffering, and back-projection validation.

## Key Experimental Results

### Illusion Region Evaluation on Real Data

| Method | Type | EPE↓ | bad2↓ | AbsRel↓ | δ1↑ |
|--------|------|------|-------|---------|-----|
| DA V2 | Mono | 5.81 | 61.45 | 0.14 | 92.86 |
| DepthPro+align | Mono | 4.36 | 44.98 | 0.09 | 93.83 |
| RAFT-Stereo | Stereo | 1.62 | 24.32 | 0.04 | 99.18 |
| Ours | Fusion | 1.77 | 26.72 | 0.03 | 99.60 |

### Zero-Shot Generalization on Booster Dataset (Mirror/Transparent Regions)

| Method | All EPE↓ | All bad2↓ | Trans EPE↓ | Trans bad2↓ |
|--------|----------|-----------|------------|-------------|
| RAFT-Stereo | 4.08 | 17.61 | 9.55 | 67.84 |
| MochaStereo | 3.79 | 16.77 | 9.18 | 66.64 |
| **Ours** | **2.43** | **13.84** | **7.32** | **56.77** |

### Zero-Shot Generalization on Middlebury

| Metric | RAFT-Stereo | Selective-IGEV | MochaStereo | Ours |
|--------|-------------|----------------|-------------|------|
| EPE | 2.34 | 2.66 | 2.89 | **1.50** |
| Bad-2 | 12.04 | 10.18 | 11.93 | 11.79 |

### Ablation Study
- The pure stereo baseline achieves bad2=80.38% on Booster; incorporating VLM fusion reduces this to 56.77% (↓24 pp).
- Introducing monocular features (MF) improves the bad metric but slightly degrades EPE—better overall geometry at the cost of some extreme offsets.
- Adaptive post-fusion (APF, dual GRU) introduces noise due to inconsistent updates between the two branches.
- The VLM confidence estimator achieves an error rate of approximately 20% in zero-shot scenarios, demonstrating strong generalization.

## Highlights & Insights
- **Novel problem formulation**: the first systematic study of 3D visual illusions in depth estimation, defining five illusion categories.
- **Insightful complementarity analysis**: clearly articulates the failure modes of monocular (generative/texture-to-geometry mapping) versus stereo (discriminative/pixel matching) methods under different illusions and their complementary relationship.
- **VLM commonsense-driven fusion**: leverages LLM commonsense knowledge (e.g., "mirrors/glass cause matching failures") to guide confidence estimation, achieving better generalization than purely data-driven approaches.
- **Comprehensive dataset construction**: combines synthetic and real data using web-crawled videos, generative models, and LiDAR ground truth in a systematic pipeline.
- **No degradation on ordinary scenes**: EPE improves on Middlebury, demonstrating the generality of the fusion framework.

## Limitations & Future Work
- **High computational cost**: the VLM component requires approximately 54 GB VRAM and 4.77 s/sample (H100), far exceeding pure stereo methods (5.6 GB, 0.87 s).
- **Manual annotation dependency**: synthetic data generation relies on SAM2 manual annotation of illusion regions and supporting regions.
- **Limited real data coverage**: only three illusion categories (inpainting, picture, replay) are covered in real data; holography and mirror types are absent.
- **Single-illusion assumption**: complex scenarios involving multiple superimposed illusion types are not studied.
- **Planarity assumption**: disparity correction assumes that illusion regions are coplanar with supporting regions, which may fail for non-planar cases.

## Related Work & Insights
- Compared with monocular methods (DepthAnything V2, Marigold, etc.): this paper reveals that, as "generative models" mapping texture cues to geometry, these methods are inherently susceptible to simulated textures; fine-tuning does not fundamentally resolve the issue—performance improves on illusion regions but degrades on ordinary regions.
- Compared with stereo methods (RAFT-Stereo, IGEV, etc.): reflections on mirror/transparent surfaces cause matching failures; fine-tuning is similarly limited because the learning signals from different illusion types conflict with each other.
- Compared with multi-view methods (DUSt3R, VGGT): these exhibit a strong monocular bias in illusion scenarios.
- Compared with the Booster dataset: Booster focuses on transparent/reflective surfaces, whereas this work covers a broader scope (five illusion categories).

The core insight—complementary monocular-stereo fusion driven by VLM commonsense reasoning—is generalizable to other 3D tasks that require handling "counter-intuitive" scenes.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The problem formulation is highly novel (the first systematic study of 3D visual illusions), though the fusion approach itself (dual-branch + confidence-based fusion) follows a relatively standard technical paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The dataset construction is comprehensive, the baseline comparisons are extensive, the ablation study is thorough, and fine-tuning analysis as well as downstream task (3D detection) visualization are included.
- **Writing Quality**: ⭐⭐⭐⭐ The writing is clear, with in-depth analysis of problem motivation and complementarity; however, certain dataset construction details (e.g., mathematical derivation of plane fitting) occupy disproportionate space.
- **Value**: ⭐⭐⭐⭐ The paper surfaces an overlooked yet important problem, and the dataset and benchmark offer long-term value to the community; however, the high computational cost of the VLM fusion component limits practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[CVPR 2026\] Mirror Illusion Art](../../CVPR2026/3d_vision/mirror_illusion_art.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](../../ICCV2025/3d_vision/simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)

</div>

<!-- RELATED:END -->
