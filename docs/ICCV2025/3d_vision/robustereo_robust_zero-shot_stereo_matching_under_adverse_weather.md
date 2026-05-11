---
title: >-
  [Paper Note] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather
description: >-
  [ICCV 2025][3D Vision][stereo matching] This paper proposes RobuSTereo, a framework that significantly improves the zero-shot generalization of stereo matching models under adverse weather conditions (rain, fog…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "stereo matching"
  - "adverse weather"
  - "zero-shot generalization"
  - "diffusion-based data generation"
  - "robust feature encoder"
  - "depth estimation"
date: 2026-05-08
content_hash: 1d8d7d1b1d08eb25
---

# RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather

**Conference**: ICCV 2025
**arXiv**: [2507.01653](https://arxiv.org/abs/2507.01653)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: stereo matching, adverse weather, zero-shot generalization, diffusion-based data generation, robust feature encoder, depth estimation

## TL;DR

This paper proposes RobuSTereo, a framework that significantly improves the zero-shot generalization of stereo matching models under adverse weather conditions (rain, fog, snow) via a diffusion-based stereo data generation pipeline and a robust feature encoder combining a denoising Vision Transformer (DVT) with VGG19.

## Background & Motivation

Stereo matching is a fundamental computer vision task that estimates depth by computing disparity between left and right images, with broad applications in autonomous driving, robotics, and augmented reality. Existing methods (IGEV, StereoBase, StereoAnything, etc.) perform well under normal conditions but suffer severe performance degradation in adverse weather such as rain, fog, and snow.

Two core challenges arise:

**Scarcity of training data**: Existing stereo datasets are predominantly captured under normal weather. Traditional approaches simulate weather via graphics rendering (e.g., vKITTI), but fail to capture complex optical phenomena (e.g., specular reflections on wet pavements), introducing a domain gap relative to real-world scenes. Collecting real adverse-weather data is constrained by the poor accuracy of sensors such as LiDAR under such conditions and high acquisition costs.

**Difficulty in feature extraction**: Encoders pretrained on normal conditions (e.g., MobileNetV2, ImageNet-pretrained models) produce unstable, noise-contaminated features when confronted with degraded images exhibiting low visibility and high noise levels, directly impacting matching accuracy.

Both challenges are particularly pronounced in zero-shot settings, where models are deployed without fine-tuning on adverse-weather data, leading to greater performance loss. The core motivation of this paper is to simultaneously address the robustness of adverse-weather stereo matching from both the data and model perspectives.

## Method

### Overall Architecture

RobuSTereo comprises three core components:

1. **Prompts Generation**: Uses an LLM (GPT-4) and a depth estimation network to generate weather-descriptive text prompts and depth maps.
2. **Data Generation**: A diffusion-based data generation pipeline built on ControlNet + Stable Diffusion, with a coherence enhancement module.
3. **Robust Stereo Matching Model**: A stereo matching network integrated with a robust feature encoder.

### Key Design 1: Diffusion-Based Stereo Data Generation

**Objective**: Transform stereo data from the normal-weather domain $(I_R, I_L, D)_{norm}$ into the adverse-weather domain $(I'_R, I'_L, D)_{adv}$, while preserving the disparity annotation $D$.

**Pipeline**:

- **Prompt generation**: The source image is fed into an LLM to automatically generate text prompts corresponding to target weather conditions (e.g., *"Rainy, dark clouds, wet pavement, raindrops, reflections, and misty air"*).
- **Depth-conditioned control**: DepthAnythingV2 predicts a depth map $D_{pred}$, which serves as the conditioning input to a Depth2Image ControlNet to ensure geometric consistency with the original disparity ground truth.
- **Image generation**: ControlNet provides conditional features $c$ to guide Stable Diffusion 1.5 in generating images stylized to the target weather. A DDIM scheduler with 50 sampling steps is used.
- **Data sources**: KITTI and vKITTI serve as source data; the resulting synthetic dataset is named RST-Dataset.

This pipeline can in principle generate unlimited training data covering diverse weather conditions (rain, fog, snow, etc.) while retaining accurate disparity annotations.

### Key Design 2: Coherence-Enhanced Consistency Module

Diffusion models generate highly diverse content, but left and right images may exhibit inconsistencies that render them unsuitable for stereo network training. The paper proposes a Disparity Fusion Method (DFM):

- Feature patches from the left and right images are partitioned into src and dst sets.
- Patch correspondences are computed based on disparity similarity and image similarity to identify the top-$n$ similar patch pairs.
- A patch fusion operation $\mathcal{M}$ is applied, refining consistent features via a self-attention mechanism $Attn(\cdot)$.
- An un-fusion step $\mathcal{U}$ restores the fused representation back to independent left and right images.

This module is embedded directly within the diffusion generation process to ensure geometric and content alignment in the generated stereo image pairs.

### Key Design 3: Robust Feature Encoder

The conventional single-branch CNN encoder is replaced with a dual-branch architecture:

- **VGG19 branch**: Extracts multi-scale pyramid features $f_c^{(i)}$ (at $i \in \{4, 8, 16\}$ downsampling factors), capturing fine-grained local structures.
- **DVT (Denoising Vision Transformer) branch**: Produces high-dimensional robust features $f_c^{(32)}$ at $1/32$ resolution, performing feature-level denoising and capturing semantic and contextual information.
- **Disparity refinement network**: Follows the iterative refinement strategy of StereoBase.

The key advantage of DVT lies in performing denoising at the feature level, addressing feature instability caused by degraded image quality under adverse weather.

### Loss & Training

Training employs a standard L1 loss, supervised with generated synthetic data paired with ground-truth disparity annotations.

## Key Experimental Results

### Main Results: DrivingStereo Dataset (Table 1)

| Method | Dataset | Rainy EPE↓ | Rainy D1↓ | Foggy EPE↓ | Foggy D1↓ | Overall EPE↓ | Overall D1↓ |
|--------|---------|-----------|----------|-----------|----------|-------------|------------|
| StereoAnything | MIX | 1.144 | 5.395 | 1.134 | 4.821 | 1.042 | 3.865 |
| MonSter | MIX | 1.153 | 5.335 | 1.152 | 5.275 | 1.081 | 4.325 |
| LightStereo | MIX | 1.105 | 4.846 | 1.155 | 4.927 | 1.088 | 4.107 |
| StereoBase | SceneFlow | 1.695 | 8.610 | 1.224 | 5.980 | 1.302 | 5.974 |
| **RobuSTereo** | **RST** | **0.973** | **1.939** | **0.853** | **1.610** | **0.836** | **1.598** |

**RobuSTereo achieves an Overall D1 of only 1.598%, representing a 58.6% reduction over the second-best method, StereoAnything (3.865%).**

### SeeingThroughFog Dataset (Table 2)

| Method | Snow EPE↓ | Rain EPE↓ | Dense Fog EPE↓ | Overall EPE↓ | Overall D1↓ |
|--------|----------|----------|---------------|-------------|------------|
| StereoAnything | 4.265 | 3.440 | 6.055 | 4.204 | 26.526 |
| LightStereo | 3.894 | 3.034 | 5.904 | 3.853 | 28.431 |
| **RobuSTereo** | **3.409** | **2.577** | **5.317** | **3.359** | **20.881** |

State-of-the-art performance is maintained on this extreme-weather benchmark as well.

### Dataset Effectiveness Validation (Table 3)

StereoBase trained solely on RST-Dataset (without model-level enhancements):

| Training Data | Overall EPE↓ | Overall D1↓ |
|--------------|-------------|------------|
| SceneFlow | 1.302 | 5.974 |
| KITTI | 0.927 | 2.279 |
| vKITTI | 1.159 | 5.157 |
| RST-Dataset | **0.875** | **2.050** |

**RST-Dataset outperforms all existing datasets**, demonstrating the intrinsic value of the data generation pipeline.

### Ablation Study (Table 4)

| Component | Setting | Overall EPE↓ | Overall D1↓ |
|-----------|---------|-------------|------------|
| Consistency module | Off | 1.308 | 5.997 |
| Consistency module | **On** | **0.875** | **2.050** |
| Source data | vKITTI | 1.039 | 4.689 |
| Source data | **KITTI** | **0.875** | **2.050** |
| Encoder | MobileNetV2 | 0.875 | 2.050 |
| Encoder | DINOv2 | 0.852 | 1.793 |
| Encoder | **Robust Encoder** | **0.836** | **1.598** |

**Key Findings**:
- The consistency module contributes the most: disabling it causes D1 to surge from 2.050 to 5.997 (+192%).
- Real data source (KITTI) outperforms synthetic source (vKITTI), presumably because vKITTI's overly simple textures introduce a larger domain gap.
- The robust encoder further reduces D1 from 2.050 to 1.598.

## Highlights & Insights

1. **Elegant data generation strategy**: Leveraging diffusion models for style transfer rather than rendering from scratch preserves original disparity annotations and fundamentally addresses the labeling difficulty of adverse-weather data. The pipeline can generate data at essentially unlimited scale with quality surpassing traditional CG methods.
2. **The consistency module is critical**: DFM draws on token-merging ideas from video editing and employs disparity-similarity-driven patch fusion to enforce left-right consistency—the key enabler for using generated data in stereo network training.
3. **Feature-level denoising in the robust encoder**: DVT performs denoising in feature space rather than image space, a noteworthy design choice that directly addresses degraded features rather than restoring degraded images prior to feature extraction.
4. **Data quality alone is sufficient**: Even without the robust encoder, training an existing model solely on RST-Dataset surpasses prior state-of-the-art, indicating that data quality is the primary bottleneck in adverse-weather stereo matching.
5. **High practical value**: Specular reflections on wet pavements are a genuine pain point in autonomous driving. The point cloud visualizations (Figure 6) demonstrate that competing methods produce severe artifacts on wet surfaces, which the proposed method effectively suppresses.

## Limitations & Future Work

1. **Reliance on Stable Diffusion 1.5**: The use of an older SD version limits generation quality and diversity; upgrading to more recent base models (e.g., SDXL, SD3) could yield further improvements.
2. **Insufficient validation of extreme weather generalization**: Evaluation is primarily conducted on DrivingStereo and SeeingThroughFog; assessment under more extreme conditions such as sandstorms, hail, and strong backlighting is lacking.
3. **Inference efficiency not discussed**: The dual-branch DVT + VGG19 encoder incurs additional computational overhead compared to a single encoder, yet no inference speed comparisons are provided.
4. **Slow data generation pipeline**: The full pipeline—50-step DDIM sampling, ControlNet, and LLM-based prompt generation—is likely to exhibit low generation throughput.
5. **Dependence on depth estimation quality**: The ControlNet is conditioned on depth maps predicted by DepthAnythingV2, meaning depth estimation errors propagate into the generated images.
6. **Limited validation across base model architectures**: The ablation study evaluates only PSMNet, IGEV, and StereoBase, leaving generalizability to a broader set of recent architectures unverified.

## Related Work & Insights

- **Data generation paradigm**: Unlike direct data collection or CG rendering, the proposed "diffusion-based style transfer with preserved annotations" paradigm provides a generalizable methodology for other tasks requiring special-condition data (e.g., nighttime detection, underwater perception).
- **Consistency constraints**: The DFM module adapts ideas from VidToMe (CVPR 2024) in video editing, transferring multi-frame consistency techniques to the stereo consistency problem—a compelling example of cross-task method transfer.
- **Feature-level vs. image-level denoising**: The feature-space denoising strategy of DVT suggests a promising direction: under degraded conditions, operating directly in feature space may be more efficient than first restoring images and then extracting features.
- **Comparison with MonSter (CVPR 2025)**: MonSter integrates monocular depth and stereo matching but is not specifically optimized for adverse weather; the proposed method achieves substantially superior weather robustness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual strategy of data generation and robust encoding is original; the consistency module design is particularly elegant.
- **Technical Depth**: ⭐⭐⭐⭐ — The integration of the diffusion generation pipeline with the stereo matching network is thorough, supported by comprehensive ablation analysis.
- **Experimental Persuasiveness**: ⭐⭐⭐⭐⭐ — Achieves comprehensive state-of-the-art on two adverse-weather benchmarks with substantial performance margins (D1 reduced by 58%+).
- **Practical Value**: ⭐⭐⭐⭐ — Directly targets adverse-weather autonomous driving scenarios; the data generation pipeline is broadly reusable.
- **Overall**: ⭐⭐⭐⭐ — A solid contribution with clear problem formulation, well-motivated design choices, and compelling experimental validation.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[ICCV 2025\] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts](learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)
- [\[CVPR 2026\] What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](../../CVPR2026/3d_vision/what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [\[ICCV 2025\] BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)

</div>

<!-- RELATED:END -->
