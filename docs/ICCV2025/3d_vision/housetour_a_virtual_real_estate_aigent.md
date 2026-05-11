---
title: >-
  [Paper Note] HouseTour: A Virtual Real Estate A(I)gent
description: >-
  [ICCV 2025][3D Vision][Camera Trajectory Generation] HouseTour is proposed to jointly generate human-like 3D camera trajectories and real estate textual descriptions given a set of indoor images with known poses. The sys…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Camera Trajectory Generation"
  - "Real Estate"
  - "Diffusion"
  - "VLM"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: df50ede9ba77540d
---

# HouseTour: A Virtual Real Estate A(I)gent

**Conference**: ICCV 2025
**arXiv**: [2510.18054](https://arxiv.org/abs/2510.18054)
**Code**: [https://house-tour.github.io/](https://house-tour.github.io/)
**Area**: 3D Vision / Vision-Language Models / Trajectory Generation
**Keywords**: Camera Trajectory Generation, Real Estate, Diffusion, VLM, 3D Gaussian Splatting

## TL;DR
HouseTour is proposed to jointly generate human-like 3D camera trajectories and real estate textual descriptions given a set of indoor images with known poses. The system employs a Residual Diffuser for diffusion-based trajectory planning and integrates spatial features into Qwen2-VL-3D to produce 3D-grounded text summaries.

## Background & Motivation

Property video tours are a critical tool in the U.S. real estate market (valued at $3.43 trillion). However, producing professional tour videos requires: (1) on-site filming by agents equipped with high-end photography equipment; and (2) carefully crafted spatial descriptions. This process is labor-intensive and costly.

**Limitations of Prior Work**:
- Vision-language models (VLMs) lack sufficient geometric reasoning capability to understand 3D spatial layouts.
- Camera trajectories in existing 3D datasets are designed for reconstruction tasks (close-up surface scanning, jittery motion) and are ill-suited for showcasing overall spaces.
- Scene description datasets enumerate furniture and their spatial relations but lack professional descriptions of spatial layout, architectural features, materials, and ambiance.

**Goal**: To enable ordinary users to automatically generate professional-grade property tour videos and textual descriptions by simply uploading a set of smartphone photos, without requiring specialized equipment.

## Method

### Overall Architecture
Given a sparse ordered set of camera poses $\mathcal{C}=[c_1,...,c_{N_c}]$ and corresponding RGB images $\mathcal{I}$:
1. **Residual Diffuser**: Generates a smooth, human-like tour trajectory $\tau$ ($N > N_c$ frames).
2. **Qwen2-VL-3D**: Produces a real estate textual summary $\Sigma$ using trajectory spatial features and visual tokens.
3. **3DGS Rendering**: Renders the final tour video along the generated trajectory.

### Residual Diffuser — Diffusion-Based Trajectory Planning

**Core Idea**: Rather than directly learning absolute trajectories (which vary greatly across different property layouts), the model learns **residuals relative to a spline interpolation**.

**Formulation**: $\tilde{p} = \mathcal{S} + \Delta p$, where $\mathcal{S}$ is the spline interpolation between known poses and $\Delta p$ is the predicted residual. At time steps corresponding to known poses, the residual is a zero vector.

**Reverse Diffusion Process**:
$$\begin{cases} \vec{0} = \delta(p^i) & \text{if } i \in t_\tau \\ p_\theta(\Delta p_{t-1}^i | \Delta p_t^i, \mathcal{S}) = \mathcal{N}(\Delta p_{t-1}^i; \mu_\theta, \Sigma_\theta) & \text{else} \end{cases}$$

**Trajectory Loss**:
- Translation: L2 norm computed over uniformly sampled dense spline points.
- Rotation: Geodesic loss on the SO(3) manifold.
$$\mathcal{L}_\theta = \mathbb{E}_{t,\tau,\epsilon}\left[\|\epsilon_{pos} - \epsilon_\theta(pos_t,t)\|^2 + d_{geo}(\epsilon_{rot}, \epsilon_\theta(rot_t,t))\right]$$

**Key Design**: Uniform sample points on spline segments between consecutive camera poses are evaluated efficiently using Horner's method, modeling the trajectory as a continuous function rather than a discrete point sequence.

### Qwen2-VL-3D — Spatially Aware Text Generation

**Two-Stage Training**:
1. **LoRA Fine-tuning**: Qwen2-VL is fine-tuned on 96-frame multi-image inputs to learn the linguistic style and architectural terminology of real estate descriptions.
2. **Spatial Feature Integration**:
   - Special tokens `<|traj_start|>`, `<|traj_pad|>`, and `<|traj_end|>` are introduced.
   - The denoised poses $p_0^i$ and bottleneck features $f_0^i$ from the Residual Diffuser are concatenated and projected into the VLM's language embedding space via a linear layer.
   - Spatial information for each frame is encoded using a single token.

### HouseTour Dataset
- 1,639 tour videos covering diverse properties ranging from apartments to multi-story villas.
- 1,298 videos with textual descriptions (half with timestamps); 878 scenes with 3D reconstructions.
- Provides scene-level human trajectories, dense point clouds, and professional real estate descriptions.

## Key Experimental Results

### End-to-End Performance

| Method | R@75cm ↑ | Rot. Score ↑ | BT ↑ | SLS ↑ |
|--------|----------|-------------|------|-------|
| Baseline (Catmull-Rom + Qwen2-VL SFT) | 57.1 | 96.8 | 71.4 | 71.7 |
| **HouseTour** | **60.2** | **97.1** | **79.5** | **76.0** |

HouseTour outperforms the baseline on all metrics, with particularly notable gains in text generation (BT +8.1).

### Ablation Study on Trajectory Generation

| Method | R@50cm ↑ | R@1m ↑ | Euclidean ↓ | DTW ↓ | Geodesic ↓ |
|--------|----------|--------|-------------|-------|-----------|
| Linear Interp. | 41.2% | 59.8% | 145.8 | 192.1 | 0.20 |
| Catmull-Rom | 45.9% | 64.7% | 106.2 | 146.3 | 0.10 |
| **Residual Diffuser** | **46.2%** | **69.4%** | **73.9** | **128.8** | **0.09** |

**Key Findings**:
- The Residual Diffuser substantially outperforms interpolation baselines at R@1m (69.4% vs. 64.7%), indicating fewer large errors.
- Euclidean distance is reduced by over 30%, demonstrating that residual learning significantly outperforms direct trajectory learning.
- The advantage of the Residual Diffuser is more pronounced in high-uncertainty regions (i.e., locations far from known poses).

## Highlights & Insights
1. **Residual Diffusion Modeling**: Reformulating trajectory generation from learning absolute positions to learning residuals relative to a spline elegantly addresses the challenge of varying layouts across scenes.
2. **Novel Joint Evaluation Metric SLS**: The paper introduces a Spatial-Language Score (harmonic mean of translation recall, rotation score, and Bradley-Terry score) as the first joint spatial-language evaluation metric.
3. **Tri-modal VLM**: Language, vision, and 3D localization are integrated into a single VLM, enabling spatially grounded text generation.
4. **Strong Practical Value**: The end-to-end system is directly applicable to the real estate and tourism industries.

## Limitations & Future Work
- Performance depends on the quality and coverage of smartphone-captured images.
- 3DGS rendering quality is limited under sparse-view conditions (explicitly noted by the authors as out of scope).
- The dataset covers only real estate scenes and has not been extended to other tour scenarios (e.g., museums, tourist attractions).
- Text generation still relies on LoRA fine-tuning; generalization to new domains requires further investigation.

## Related Work & Insights
- **Long Video Understanding**: Methods such as TimeChat handle long sequences but lack domain knowledge of architectural environments.
- **Trajectory Planning**: The Diffuser family is applied to robot decision-making in fixed environments, not in varying layouts.
- **3D Vision-Language**: ScanRefer and DenseCap describe object relations but overlook spatial layout and architectural features.

## Rating
- **Novelty**: ★★★★☆ — The combination of residual diffusion trajectory planning and a tri-modal VLM is novel.
- **Practical Value**: ★★★★★ — Directly targets a trillion-dollar real estate market.
- **Experimental Thoroughness**: ★★★★☆ — Introduces a new dataset and evaluation metrics with thorough ablations.
- **Writing Quality**: ★★★★☆ — Well-structured with a clearly defined problem formulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)

</div>

<!-- RELATED:END -->
