---
title: >-
  [Paper Note] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation
description: >-
  [CVPR 2025][Autonomous Driving][Depth Foundation Models] Prompt Depth Anything introduces the "prompting" paradigm to depth foundation models for the first time. Using low-cost LiDAR (such as iPhone LiDAR) as metric prompts, it guides the Depth Anything model to output accurate metric depth through a concise multi-scale prompt fusion architecture, achieving high-quality depth estimation at up to 4K resolution.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Depth Foundation Models"
  - "Metric Depth Estimation"
  - "LiDAR Prompts"
  - "Depth Completion"
  - "4K Resolution"
date: 2026-05-08
content_hash: cc76b1962b062186
---

# Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation

**Conference**: CVPR 2025  
**arXiv**: [2412.14015](https://arxiv.org/abs/2412.14015)  
**Code**: [https://PromptDA.github.io/](https://PromptDA.github.io/)  
**Area**: Autonomous Driving  
**Keywords**: Depth Foundation Models, Metric Depth Estimation, LiDAR Prompts, Depth Completion, 4K Resolution

## TL;DR

Prompt Depth Anything introduces the "prompting" paradigm to depth foundation models for the first time. Using low-cost LiDAR (such as iPhone LiDAR) as metric prompts, it guides the Depth Anything model to output accurate metric depth through a concise multi-scale prompt fusion architecture, achieving high-quality depth estimation at up to 4K resolution.

## Background & Motivation

**Background**: Depth foundation models (e.g., Depth Anything v2, DepthPro, Metric3D v2) have achieved strong generalization capabilities through large-scale training, enabling them to generate high-quality relative depth maps. However, these models still suffer from scale ambiguity in metric depth estimation, meaning they cannot provide accurate absolute distances.

**Limitations of Prior Work**: Existing solutions for metric depth estimation primarily follow two approaches: (1) fine-tuning depth foundation models on metric datasets, which degrades their generalization capability; (2) training metric depth models with camera intrinsics as extra inputs, which still yields suboptimal results. As shown in Figure 1(b), Metric3D v2 exhibits inaccurate scale and frame-to-frame inconsistency.

**Key Challenge**: Depth foundation models possess strong local shape understanding capability (inheriting rich geometric priors) but lack absolute scale information. Monocular images inherently cannot provide metric information, and existing metric prompting methods (e.g., camera intrinsics) do not offer sufficiently strong constraints.

**Goal**: To design a method to "inject" metric information into depth foundation models, enabling them to output accurate absolute depth while maintaining high resolution and generalization capability.

**Key Insight**: The authors draw an analogy to the "pre-training + prompt tuning" paradigm in NLP and vision foundation models—a well-designed prompt can unlock the potential of foundation models for downstream tasks. Low-cost LiDAR (such as the built-in dToF sensor on iPhones) provides accurate but low-resolution (240×320) and noisy depth, which serves perfectly as a "metric prompt."

**Core Idea**: Reformulate metric depth estimation as a conditional generation task where metric prompts guide a depth foundation model. The low-cost LiDAR provides absolute scale information, while the foundation model provides high-resolution local geometric details, making them complementary.

## Method

### Overall Architecture

Prompt Depth Anything is built upon the ViT-Large + DPT decoder architecture of Depth Anything v2. The inputs are an RGB image and a low-resolution LiDAR depth map, and the output is a high-resolution (up to 4K) accurate metric depth map. The core modification is the addition of a prompt fusion module at each scale of the DPT decoder to inject LiDAR depth information multi-scally into the decoding process. The training data is constructed through a hybrid pipeline of synthetic data LiDAR simulation and real data pseudo-GT depth generation.

### Key Designs

1. **多尺度提示融合架构（Prompt Fusion Architecture）**:

    - **Function**: Integrate low-resolution LiDAR depth as conditional information into the decoding process of the depth foundation model.
    - **Mechanism**: At each scale $S_i$ of the DPT decoder, the LiDAR depth map is first bilinearly interpolated to the spatial resolution of the current scale. Then, a shallow convolutional network (two 3×3 convolutions) extracts depth features. These features are projected to the same dimension as the image features using a zero-initialized convolutional layer, and finally added to the intermediate features of the DPT for depth decoding. The entire fusion module adds only 5.7% computational overhead (1.789 vs 1.691 TFLOPs).
    - **Design Motivation**: The zero-initialization design ensures that the initial output is completely consistent with the original foundation model, fully inheriting the capability of the pre-trained model. Multi-scale fusion makes full use of spatial distance information at different granularities provided by the LiDAR. Compared to other conditional injection methods such as ControlNet, CrossAttention, and Adaptive LayerNorm, a simple additive fusion better utilizes the pixel-alignment characteristics between the LiDAR and the output depth.

2. **合成数据 LiDAR 仿真（Sparse Anchor Interpolation）**:

    - **Function**: Generate realistic LiDAR depth inputs for synthetic data (e.g., Hypersim).
    - **Mechanism**: Simple downsampling of GT depth fails to simulate the noise characteristics of real LiDAR, causing the model to degrade into learning depth super-resolution. To address this, the authors propose Sparse Anchor Interpolation: the GT depth is first downsampled to the LiDAR resolution (192×256), sparse anchors are sampled on it using a perturbed grid (step size 7), and the remaining points are interpolated via RGB-similarity KNN. This simulates LiDAR noise and interpolation artifacts.
    - **Design Motivation**: If the simulated LiDAR is too "clean," the model will only learn to perform super-resolution rather than correcting noise, compromising performance on real LiDAR. Sparse anchor interpolation introduces noise patterns similar to real iPhone LiDAR.

3. **伪 GT 深度生成 + 边缘感知损失**:

    - **Function**: Generate high-quality training supervision signals for real data (e.g., ScanNet++) that only has coarse GT depth.
    - **Mechanism**: The FARO laser scan depth of ScanNet++ has poor quality at edges of texture-dense regions (due to holes and artifacts caused by occlusion), while Zip-NeRF reconstruction yields high-quality edges but is inaccurate in textureless areas. The edge-aware loss combines the advantages of both: $\mathcal{L}_{edge} = L_1(\mathbf{D}_{gt}, \hat{\mathbf{D}}) + \lambda \cdot \mathcal{L}_{grad}(\mathbf{D}_{pseudo}, \hat{\mathbf{D}})$, where the $L_1$ term supervises the overall depth values using FARO depth (accurate in textureless planar regions), and the gradient term supervises the gradient of the output depth using the gradient of the Zip-NeRF pseudo-GT (accurate at edges).
    - **Design Motivation**: Since any single data source has defects, the loss function design cleverly combines the complementary advantages of both depth types, avoiding the negative impacts of unreliable regions in the pseudo-GT on training.

### Loss & Training

- Depth Normalization: Linearly scale using the min/max of the LiDAR depth to [0,1], and the output is normalized synchronously to ensure scale consistency.
- Initialized from the metric model of Depth Anything v2, starting with 10K warmup steps (fine-tuning to output normalized depth), followed by 200K training steps.
- AdamW optimizer, ViT backbone lr=5e-6, other parameters lr=5e-5, batch size=2, 8 GPUs.
- $\lambda=0.5$

## Key Experimental Results

### Main Results

ARKitScenes Dataset (768×1024 resolution):

| Method | Type | L1↓ | RMSE↓ |
|------|------|-----|-------|
| Depth Anything v2 (aligned) | Post | 0.0771 | 0.0647 |
| Metric3D v2 (aligned) | Post | 0.0524 | 0.1721 |
| DepthPro (fine-tuned) | w/o LiDAR | 0.0435 | 0.0665 |
| Depth Prompting | Net | 0.0253 | 0.0422 |
| ARKit Depth | - | 0.0250 | 0.0423 |
| **Ours (zero-shot)** | **Net** | **0.0163** | **0.0371** |
| **Ours** | **Net** | **0.0132** | **0.0315** |

ScanNet++ Dataset (Depth Estimation + TSDF Reconstruction):

| Method | L1↓ | RMSE↓ | F-score↑ |
|------|-----|-------|----------|
| Depth Anything v2 (fine-tuned) | 0.0510 | 0.1010 | 0.6595 |
| **Ours** | **0.0250** | **0.0829** | **0.7619** |
| Ours (zero-shot, synthetic only) | 0.0327 | 0.0966 | 0.7307 |

### Ablation Study

| Configuration | L1↓ | RMSE↓ | Description |
|------|-----|-------|------|
| Full model (prompt fusion) | 0.0135 | 0.0326 | Full model |
| Adaptive LayerNorm | - | - | Suboptimal, not suitable for pixel-aligned conditions |
| CrossAttention | - | - | High computational overhead, inferior to direct fusion |
| ControlNet | - | - | Too many copied encoder parameters |
| w/o foundation model initialization | Significant drop | - | Proves that inheriting pre-trained weights is crucial |
| w/o edge-aware loss | Degraded edge quality | - | Edge-aware loss is crucial for real-world data training |
| Simple downsampling simulated LiDAR | Degenerates to super-resolution | - | Proves the necessity of LiDAR noise simulation |

### Key Findings

- Even the zero-shot model (trained only on synthetic data) outperforms other methods trained/fine-tuned on target datasets, demonstrating the strong generalization capability of the "prompting-foundation-model" paradigm.
- Simple additive fusion is more effective than complex designs like ControlNet and CrossAttention, because the LiDAR and depth outputs are pixel-aligned, eliminating the need for cross-modal attention mechanisms.
- The depth foundation model as a "local shape learner" paired with LiDAR acting as "global scale anchors" establishes clear labor division, outperforming any single-modality approach.
- The method is highly extensible, allowing replacement of the foundation model (e.g., with DepthPro) and the LiDAR type (e.g., automotive LiDAR).

## Highlights & Insights

- **Introducing the prompting paradigm to depth estimation**: Analogous to prompt tuning in NLP, using LiDAR as a "prompt" to activate the foundation model's capability in metric depth estimation. This concept can be generalized to other foundation model tasks that require auxiliary signal guidance.
- **Zero-initialized fusion module**: Ensures that the newly added module does not alter the original model's output at initialization, leading to more stable training while fully preserving the generalization capability of the pre-trained model.
- **Loss function design for complementary data sources**: The methodology of using gradient loss solely to supervise edges and L1 loss solely to supervise flat regions is widely applicable to any scenario requiring the combination of multiple imperfect supervision signals.

## Limitations & Future Work

- Relying on LiDAR sensor input, making it unusable in purely monocular scenarios (although devices like iPhones commonly feature LiDAR, many scenarios still lack such sensors).
- Currently evaluated mainly on indoor scenes; the generalization to large-scale outdoor environments requires further assessment.
- The LiDAR simulation strategy is relatively simplified; more sophisticated LiDAR noise models (e.g., considering material reflectivity, multi-path interference) could potentially improve performance further.
- Training requires Zip-NeRF to generate pseudo-GTs for real-world data, which is time-consuming and may introduce systematic errors.
- Future work can explore other forms of "metric prompts," such as sparse SfM points, IMU data, etc.

## Related Work & Insights

- **vs Depth Anything v2**: This work adds LiDAR prompts on top of DAv2, upgrading it from a relative depth model to an accurate metric depth model, proving that the foundation model + prompt paradigm is more effective than direct fine-tuning.
- **vs Depth Prompting**: Depth Prompting performs post-processing fusion with sparse depth after the foundation model outputs, which is not a true "prompt." This work performs multi-scale fusion of LiDAR information during decoding, representing a deeper conditional integration.
- **vs Traditional Depth Completion Methods**: Traditional methods (e.g., NLSPN, BPNet) learn sparse-to-dense mapping as an independent task without utilizing the strong priors of pre-trained depth foundation models. This work leverages the foundation model as a strong regularizer, significantly enhancing generalization capability.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing the concept of prompting to depth foundation models is novel, though the technical implementation (multi-scale feature fusion) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across two datasets, multiple resolutions, multiple baseline comparisons, along with rich ablations and downstream application verifications.
- Writing Quality: ⭐⭐⭐⭐⭐ Smooth narrative, clear motivation, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ High practical value; combining ubiquitous iPhone LiDAR with a foundation model yields 4K high-quality depth, directly benefiting downstream applications such as 3D reconstruction and robotic grasping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](../../CVPR2026/autonomous_driving/ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[CVPR 2025\] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting](pansplat_4k_panorama_synthesis_with_feed-forward_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
