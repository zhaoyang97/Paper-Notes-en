---
title: >-
  [Paper Note] 4D Visual Pre-training for Robot Learning
description: >-
  [ICCV 2025][3D Vision][3D point cloud pre-training] FVP proposes a visual pre-training framework based on 4D (3D spatial + temporal) point cloud prediction. By formulating the pre-training objective as "next-frame point cloud prediction" and implementing it via a diffusion model, FVP significantly improves the success rate of multiple 3D imitation learning methods on real-robot manipulation tasks (average +28% on DP3).
tags:
  - ICCV 2025
  - 3D Vision
  - 3D point cloud pre-training
  - diffusion models
  - robot manipulation
  - imitation learning
  - visual representation
date: 2026-05-08
content_hash: 01bf652970155244
---

# 4D Visual Pre-training for Robot Learning

**Conference**: ICCV 2025
**arXiv**: [2508.17230](https://arxiv.org/abs/2508.17230)
**Code**: [https://4d-visual-pretraining.github.io/](https://4d-visual-pretraining.github.io/)
**Area**: 3D Vision / Robot Learning
**Keywords**: 3D point cloud pre-training, diffusion models, robot manipulation, imitation learning, visual representation

## TL;DR
FVP proposes a visual pre-training framework based on 4D (3D spatial + temporal) point cloud prediction. By formulating the pre-training objective as "next-frame point cloud prediction" and implementing it via a diffusion model, FVP significantly improves the success rate of multiple 3D imitation learning methods on real-robot manipulation tasks (average +28% on DP3).

## Background & Motivation
- **Background**: Current robot visual representation learning is dominated by 2D image pre-training (R3M, MVP, VC-1, etc.), which achieves strong results on large-scale datasets such as ImageNet and Ego4D, but neglects the inherently 3D nature of the physical world.
- **Advantages of 3D Methods**: Imitation learning methods based on 3D point clouds (DP3, RISE, etc.) demonstrate stronger generalization and efficiency on real-robot tasks, yet a universal 3D pre-training paradigm remains absent.
- **Key Challenge**: Large-scale 3D data is scarce, making it infeasible to learn general 3D representations from massive web-scale data as in the 2D setting.
- **Key Insight**: Rather than training a single universal 3D representation, the paper proposes designing a pre-training framework capable of enhancing arbitrary 3D encoders. Leveraging the generative capacity of diffusion models, the pre-training objective is defined as "predicting the next-frame point cloud from the current frame," thereby learning visual features that capture the dynamics of the physical environment.
- **Core Idea**: 4D pre-training is reformulated as a conditional diffusion generation problem — historical frame point clouds serve as conditions, and a diffusion model denoises them to generate future frame point clouds, compelling the 3D encoder to learn meaningful spatiotemporal representations.

## Method

### Overall Architecture
The FVP pipeline consists of two stages: (1) **Pre-training**: robot trajectory data (observation–action pair sequences) are collected, and a diffusion model is trained with a "next-frame point cloud prediction" self-supervised objective; (2) **Downstream Fine-tuning**: the pre-trained 3D encoder replaces the original encoder in policies such as DP3 and RISE, followed by end-to-end fine-tuning.

### Key Designs

1. **Next-Frame Point Cloud Prediction Objective**

    - **Function**: The visual representation encoded from the previous frame $o^{t-1}$ is used together with a diffusion model to predict the current frame point cloud $o^t$.
    - **Mechanism**: Unlike contrastive learning (positive/negative sample construction) or masked reconstruction, FVP learns physical dynamics by predicting future states. The encoder maps $o^{t-1}$ to a latent representation $\mathbf{z} \in \mathbb{R}^{N \times C_v}$, which is then concatenated with the noised target point cloud to form the input $o_{T,+}^t \in \mathbb{R}^{N \times (C_v+3)}$.
    - **Design Motivation**: Predicting future point clouds forces the encoder to understand robot motion characteristics and environmental dynamics. Such "predictive" learning captures temporally relevant information critical for robot control more effectively than "reconstructive" learning.

2. **Conditional Diffusion Generative Model**

    - **Function**: The denoising process is modeled as a conditional diffusion probabilistic model that progressively recovers the target point cloud from Gaussian noise.
    - **Mechanism**: The diffusion process follows the Markov chain $q(X_t|X_{t-1}) = \mathcal{N}(X_t; \sqrt{1-\beta_t}X_{t-1}, \beta_t\mathbf{I})$, with training loss $\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\mathbf{I})}[\|\epsilon - \epsilon_\theta(o_{+,T}^t, T)\|_2^2]$.
    - **Implementation**: A modified Point-Voxel Diffusion network is adopted, with architecture $\epsilon_\theta: \mathbb{R}^{N\times(C_v+3)} \to \mathbb{R}^{N\times 3}$, augmented with robot action information to assist generation.
    - **Design Motivation**: Diffusion models have demonstrated the ability to produce high-quality representations in visual tasks, and their iterative denoising process is naturally suited to learning the spatial structure of point clouds.

3. **Universal Encoder Compatibility**

    - **Function**: FVP is designed as a plug-and-play pre-training module compatible with multiple 3D encoders.
    - **Supported Encoders**: PointNet++, Point Transformer, DP3 Encoder, RISE Encoder.
    - **Design Motivation**: By decoupling the encoder from the pre-training objective, FVP becomes a general-purpose pre-training framework rather than an architecture-specific solution.

### Loss & Training
- **Pre-training Loss**: Standard noise-prediction L2 loss of the diffusion model; conditional inputs include historical frame encodings and action information.
- **Downstream Fine-tuning**: After replacing the original encoder with the pre-trained one, the visual encoder and policy backbone are fine-tuned end-to-end.
- **Data Requirements**: Either in-domain or cross-domain data (e.g., the RoboMind public dataset) can be used; 50 expert demonstrations are collected per task.

## Key Experimental Results

### Main Results

| Task | Method | Success Rate | Prev. SOTA | Gain |
|------|--------|-------------|------------|------|
| PickSquare | DP3+FVP | 20/20 | MAE (18/20) | +10% |
| PlaceBottle | DP3+FVP | 20/20 | MVP (15/20) | +25% |
| PickPlace | DP3+FVP | 17/20 | MVP (16/20) | +5% |
| FlipCup | DP3+FVP | 16/20 | MVP (17/20) | ≈0% |
| Assembly | DP3+FVP | 13/20 | MAE (11/20) | +10% |
| ArtiManip | DP3+FVP | 16/20 | R3M/MVP (14/20) | +10% |
| **Average** | **DP3+FVP** | **16.4/20** | **MAE (15.3/20)** | **+5.7%** |

### Ablation Study

| Configuration | PickSquare | PlaceBottle | PushDraw | ToastBread | Notes |
|---------------|-----------|-------------|----------|------------|-------|
| DP3+FVP (full) | 20/20 | 20/20 | 20/20 | 16/20 | Historical frame as condition |
| Replace historical frame with current frame | 15/20 | 14/20 | 13/20 | 13/20 | Temporal history is critical |
| Freeze visual encoder | 11/20 | 9/20 | 10/20 | 7/20 | End-to-end fine-tuning is essential |

### Key Findings
- FVP yields an average improvement of 16.9% on simulation tasks when pre-trained on in-domain data, and 24.7% when using cross-domain data.
- Gains are especially pronounced on dexterous hand tasks, where FVP leverages temporal frames to understand complex motion trajectories.
- Applying FVP to the VLA model RDT-1B improves spatial understanding from 8/20 to 14/20, and knowledge transfer from 10/20 to 16/20.
- A historical frame stride of 1 yields the best performance (20/20); performance degrades as stride increases (stride 4: 14–15/20).
- 2D pre-training methods (R3M/MVP/MAE), even when pre-trained on 300M+ data, underperform FVP's 3D pre-training.

## Highlights & Insights
- **Pre-training Paradigm Innovation**: This is the first work to introduce "next-frame prediction" into 3D point cloud pre-training, departing from conventional contrastive learning and masked modeling, and more closely aligning with the sequential decision-making nature of robotics.
- **Plug-and-Play Design**: As a universal pre-training module, FVP enhances multiple 3D encoders (PointNet++, Point Transformer, etc.) and policy methods (DP3, RISE).
- **Effective Cross-Domain Pre-training**: Encoders pre-trained on public datasets such as RoboMind transfer successfully to different robot platforms.
- **VLA Model Enhancement**: The paper demonstrates that 3D point cloud inputs combined with FVP pre-training effectively improve spatial perception in large-scale VLA models.

## Limitations & Future Work
- Poor performance when freezing the pre-trained encoder indicates a gap between in-domain and out-of-domain data, suggesting that pre-trained representations are not yet sufficiently general.
- Each task requires independently pre-training a visual encoder; true "pre-train once, apply to multiple tasks" capability has not been achieved.
- Performance degrades as historical frame stride increases (stride 4: PickSquare drops from 20/20 to 15/20), revealing limited capacity for modeling long-range temporal dependencies.
- Experimental validation on larger-scale 3D pre-training datasets is lacking.
- The computational efficiency of the diffusion-based pre-training process is not discussed — it remains unclear whether the additional overhead is justified compared to simpler reconstruction objectives.
- Only manipulation tasks are evaluated; the effectiveness on other robot tasks such as navigation and motion planning is unknown.
- Point cloud acquisition relies on RGB-D cameras; the impact of sensor noise characteristics on pre-training quality warrants further investigation.

## Related Work & Insights
- **vs. R3M/MVP**: 2D pre-training models (trained on 300M+ datasets) are clearly outperformed by FVP within the same policy framework, even when FVP uses only small in-domain or cross-domain datasets, validating the superiority of 3D representations.
- **vs. PointMAE/STRL/C2P**: FVP comprehensively outperforms these 3D/4D pre-training methods in both in-domain and cross-domain settings. PointMAE uses masked reconstruction, STRL uses temporal contrastive learning, and C2P uses cross-modal prediction; FVP's "next-frame prediction" objective more directly captures physical dynamics.
- **vs. ACT/Diffusion Policy**: 2D imitation learning methods are sensitive to camera placement and struggle to capture 3D spatial information.
- **Integration with VLA Models**: FVP directly enhances large-scale VLA models such as RDT-1B, yielding improvements in spatial perception, task transfer, and long-horizon tasks, demonstrating the potential of combining 3D pre-training with large-scale models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introduces the next-frame prediction paradigm from NLP/2D vision into 3D point cloud pre-training in a concise and effective manner.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 24 tasks across simulation and real-world settings, four robot platforms (single-arm / dexterous hand / dual-arm / humanoid), and VLA model integration.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, well-organized experiments, and strong visualizations.
- **Value**: ⭐⭐⭐⭐ Provides a practical pre-training solution for 3D robot learning with solid engineering potential.
- **Overall**: ⭐⭐⭐⭐ Comprehensive experiments, simple methodology, and significant performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](faster_and_better_3d_splatting_via_group_training.md)

</div>

<!-- RELATED:END -->
