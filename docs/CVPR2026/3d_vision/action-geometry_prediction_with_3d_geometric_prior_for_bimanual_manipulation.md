---
title: >-
  [Paper Note] GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation
description: >-
  [CVPR2026][3D Vision][Bimanual Manipulation] GAP leverages a pretrained 3D geometric foundation model (π³) to extract 3D features, fuses them with 2D semantic features and proprioception, and jointly predicts future action sequences and future 3D point maps via conditional diffusion, achieving state-of-the-art performance on RoboTwin 2.0 and real-world bimanual manipulation benchmarks.
tags:
  - CVPR2026
  - 3D Vision
  - Bimanual Manipulation
  - 3D Geometric Prior
  - Diffusion Policy
  - Point Map Prediction
  - Imitation Learning
date: 2026-05-08
content_hash: 06e59a70da0cf04b
---

# GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation

**Conference**: CVPR2026
**arXiv**: [2602.23814](https://arxiv.org/abs/2602.23814)
**Code**: [https://github.com/Chongyang-99/GAP.git](https://github.com/Chongyang-99/GAP.git)
**Area**: 3D Vision
**Keywords**: Bimanual Manipulation, 3D Geometric Prior, Diffusion Policy, Point Map Prediction, Imitation Learning

## TL;DR
GAP leverages a pretrained 3D geometric foundation model (π³) to extract 3D features, fuses them with 2D semantic features and proprioception, and jointly predicts future action sequences and future 3D point maps via conditional diffusion, achieving state-of-the-art performance on RoboTwin 2.0 and real-world bimanual manipulation benchmarks.

## Background & Motivation

**Background**: Bimanual manipulation requires a policy to simultaneously generate coordinated actions for two robot arms, involving precision assembly, deformable object handling, and interaction in cluttered environments. Dominant approaches include 2D-based ACT (action chunking + DETR Transformer), diffusion policy (DP), and 3D-aware DP3 (point cloud input).

**Limitations of Prior Work**:
- **2D methods lack spatial awareness**: ACT and DP rely on 2D features and cannot explicitly reason about 3D spatial relationships, occlusions, or contacts, leading to poor performance on bimanual tasks requiring precise spatial reasoning.
- **3D methods depend on explicit point clouds**: DP3 and similar approaches require depth cameras and point cloud pipelines, which are sensitive to calibration errors, noise, and occlusion in real-world settings. 2D-to-3D lifting methods (e.g., back-projection) suffer from low resolution and significant engineering overhead.
- **No predictive 3D reasoning**: Existing methods only perceive the current 3D state without predicting how 3D geometry evolves after action execution, limiting long-horizon planning.

**Key Challenge**: Bimanual manipulation requires 3D spatial reasoning, yet reliably acquiring explicit 3D information (point clouds) in real-world settings remains challenging. Furthermore, perceiving only the current state is insufficient for complex manipulation tasks that demand anticipation of future geometric changes.

**Goal**: Can a 3D geometric foundation model be used to obtain implicit 3D features directly from RGB images, bypassing explicit point cloud pipelines? Can jointly predicting future 3D structure enhance the policy's spatial understanding and long-horizon planning?

**Key Insight**: Recent 3D geometric foundation models (e.g., DUSt3R, VGGT, π³) can robustly reconstruct dense 3D structures from RGB images in a feed-forward manner. The authors adopt π³ as the perception backbone, whose latent features inherently encode rich 3D geometric information—eliminating the need for explicit point cloud generation and directly conditioning the policy on these latents. Furthermore, predicting "future 3D latents" encourages the model to learn 3D-aware anticipatory reasoning.

**Core Idea**: Use the latents of a pretrained 3D geometric foundation model as 3D priors, and jointly denoise future actions and future 3D point maps to realize an RGB-only, 3D-aware bimanual manipulation policy.

## Method

### Overall Architecture
**Inputs**: 5 historical RGB frames $V$, current frame $I_t$, current proprioception $p_t \in \mathbb{R}^{14}$ (6 joint angles + 1 gripper state per arm × 2 arms). **Outputs**: Future $N$-step bimanual action sequence $a_{t:t+N} \in \mathbb{R}^{N \times 14}$ and the future 3D point map at step $N$: $P_{t+N} \in \mathbb{R}^{H \times W \times 4}$.

The pipeline consists of four stages: (1) three parallel encoders extract modality-specific features → (2) a Transformer fuses them into a unified context → (3) a conditional diffusion decoder jointly denoises → (4) separate heads decode actions and point maps.

### Key Designs

1. **Geometry 3D Encoder (π³ Encoder)**

    - **Function**: Extracts features encoding 3D geometric information from sequential RGB frames.
    - **Mechanism**: Five frames are uniformly sampled from the history $V$ and concatenated with the current frame $I_t$ to form a 6-frame sequence, which is fed into the π³ encoder (a multi-view geometry model). Each frame is patchified into $14 \times 14$ patches. Features from the last two backbone layers are concatenated to produce 1024-dimensional 3D geometric features $\mathbf{f}_{3d}$. Only the encoder of π³ is used; decoding heads are discarded.
    - **Design Motivation**: As a pretrained 3D geometric foundation model, π³'s latents inherently encode multi-view and multi-frame 3D geometric relationships. Compared to explicit point clouds, latent features are more robust (unaffected by calibration errors or depth noise) and are computed in a single feed-forward pass.

2. **Semantics 2D Encoder (DINOv3 Encoder)**

    - **Function**: Extracts high-level semantic features from the current frame.
    - **Mechanism**: The current frame $I_t$ is encoded by DINOv3 and divided into $16 \times 16$ patches, yielding 1024-dimensional semantic features $\mathbf{f}_{2d}$.
    - **Design Motivation**: While 3D geometric features capture spatial structure, they lack task-relevant high-level semantics. DINOv3 provides object-level semantic priors (e.g., identifying which object to manipulate), complementing the geometric features.

3. **State Encoder (MLP Encoder)**

    - **Function**: Encodes the robot's proprioceptive state.
    - **Mechanism**: A simple MLP maps $p_t \in \mathbb{R}^{14}$ to a 1024-dimensional embedding $\mathbf{f}_p$.

4. **Semantic-Geometric Fusion**

    - **Function**: Fuses three heterogeneous feature streams into a unified context representation.
    - **Mechanism**: The three 1024-dimensional features $[\mathbf{f}_{3d}, \mathbf{f}_{2d}, \mathbf{f}_p]$ are concatenated along the token dimension and passed through a **4-layer DETR encoder** for deep fusion, yielding the Semantic-Geometric Fused Context $\mathbf{f}_c$.
    - **Design Motivation**: The self-attention mechanism of the DETR encoder enables full cross-modal interaction: 3D geometric features inform semantic features about "where objects are," semantic features inform geometric features about "which objects matter," and proprioception constrains "what the robot can currently do."

5. **Joint Diffusion Decoder**

    - **Function**: Conditioned on $\mathbf{f}_c$, jointly denoises to generate the future action sequence and future 3D latent.
    - **Mechanism**: A DETR decoder is adopted for conditional diffusion. During training, Gaussian noise is added to the clean target $x_0 = \{a_{t:t+N}, \mathbf{f}_{t+N}, P_{t+N}\}$ to obtain $x_k$; the reverse process uses the decoder to predict the clean target $\hat{x}_0$. The loss is an L1 objective:
    $$\mathcal{L} = \mathbb{E}_{k, x_0, \epsilon}\left[\|{\hat{a}_{t:t+N}} - a_{t:t+N}\|_1 + \lambda\|\hat{\mathbf{f}}_{t+N} - \mathbf{f}_{t+N}\|_1 + \gamma\|\hat{P}_{t+N} - P_{t+N}\|_1\right]$$
    - **Two prediction targets**:
        - Future Action Chunk: $\mathbb{R}^{N \times 14}$, representing $N$-step bimanual actions (6-DoF joints + 1-DoF gripper × 2 arms).
        - Future 3D Point Map Latent: $\mathbf{f}_{t+N} \in \mathbb{R}^{H/14 \times W/14 \times 1024}$, decoded by π³'s dense head into $P_{t+N} \in \mathbb{R}^{H \times W \times 4}$ ($x, y, z$ + confidence).
    - **Design Motivation**: Jointly predicting future 3D structure compels the model to reason about "what the 3D scene will look like after executing these actions"—an implicit world model. Supervising only the final state at horizon $N$ (rather than step-by-step) forces the model to reason about the cumulative effect of the entire action sequence, enhancing long-horizon planning.

6. **Pseudo-GT Generation Strategy**

    - **Function**: Generates stable supervision signals for 3D latent targets.
    - **Mechanism**: Rather than naively running π³ on single frames (which yields noisy, unstable outputs), for each frame $s$ in the dataset, $n$ historical frames are uniformly sampled to form a temporal window $\{V, I_s\}$, which is fed into the π³ encoder; only the latent $\mathbf{f}_s$ corresponding to $I_s$ is retained. The training target is set to $\mathbf{f}_{t+N}$.
    - **Design Motivation**: Joint processing over a temporal window substantially stabilizes the quality of 3D latent features.

### Loss & Training
- Standard diffusion training with ACT-style action chunking.
- 2D-based methods (including the proposed method) are trained for 200–600 epochs; 3D-based methods for 3000 epochs; batch size 32.
- 100 expert demonstrations (simulation) or 50 teleoperation demonstrations (real world).
- At inference, denoising is performed iteratively for $K$ steps from Gaussian noise; point map decoding can be optionally skipped for efficiency.

## Key Experimental Results

### Main Results — RoboTwin 2.0 Simulation (Average Success Rate % across Three Task Categories)

| Method | Dominant-select (16 tasks) | Sync-bimanual (8 tasks) | Seq-coordinate (8 tasks) |
|--------|---------------------------|------------------------|--------------------------|
| ACT (2D) | 34.1 | 32.4 | 29.4 |
| DP (2D) | 44.4 | 37.1 | 33.6 |
| DP3 (3D point cloud) | 61.2 | 42.0 | 42.0 |
| G3Flow (3D + semantics) | 54.3 | 43.2 | 40.5 |
| RDT (1.2B parameters) | 49.5 | 44.6 | 41.2 |
| Xu et al. (2D + prediction) | 55.1 | 47.5 | 44.9 |
| **GAP (Ours)** | **63.2** | **51.3** | **50.4** |

### Ablation Study (Average Success Rate % over 4 Tasks)

| 2D Semantic | 3D Geometric | Geometric Imagination | Avg. Success Rate |
|:-----------:|:------------:|:---------------------:|:-----------------:|
| ✓ | ✓ | ✓ | **25.1** |
| ✗ | ✓ | ✓ | 24.4 |
| ✓ | ✓ | ✗ | 23.6 |
| ✓ | ✗ | ✗ | 21.0 |

### Real-World Experiments (Success Rate %, 20 trials/task)

| Task | ACT | DP | Xu et al. | **Ours** |
|------|-----|----|-----------|----------|
| Place Empty Cup | 70 | 70 | 75 | **80** |
| Place Dual Shoes | 0 | 10 | 15 | **20** |
| Hanging Mug | 0 | 0 | 5 | **20** |
| Scan Object | 25 | 20 | 35 | **40** |
| **Average** | 23.8 | 25 | 32.5 | **40** |

### Key Findings
- **3D geometric perception is critical**: Removing both the 3D Geometric Module and Geometric Imagination reduces success rate from 25.1% to 21.0% (a relative drop of 16.3%), making it the most impactful component.
- **Geometric Imagination (predicting future 3D) is the core innovation**: Removing it alone drops success rate from 25.1% to 23.6% (−6.0%), demonstrating that predicting future 3D structure genuinely improves the policy's 3D understanding.
- **RGB-only input surpasses explicit point cloud methods**: GAP (RGB only) outperforms DP3 on Dominant-select tasks (63.2% vs. 61.2%), validating that latents from pretrained 3D foundation models can substitute explicit point clouds.
- **Strong advantage on synchronized bimanual tasks**: GAP achieves 43.3% on Place Dual Shoes, compared to only 17.7% for DP3, indicating superior bimanual coordination reasoning.
- **High data efficiency**: With as few as 10–20 demonstrations, GAP already shows meaningful learning signal owing to pretrained features, while DP completely fails (0% success rate).
- **Real-world Hanging Mug**: ACT and DP both fail entirely (0%), while GAP achieves 20%—a task requiring precise 3D geometric reasoning to localize the spatial relationship between the mug handle and hook.

## Highlights & Insights
- **Using 3D foundation model latents directly as policy conditions is an elegant paradigm**: It bypasses the engineering complexity of point cloud acquisition; π³'s latents inherently encode rich 3D geometry. This idea generalizes to any robotic task requiring 3D spatial awareness.
- **Joint prediction of actions and future 3D structure constitutes an implicit world model**: No separately trained world model is needed; 3D prediction capability is naturally injected into the policy network through joint denoising in the diffusion process. Predicting only the 3D state at the end of the horizon—rather than at each step—elegantly reduces computational and supervisory complexity.
- **DETR encoder for semantic-geometric fusion**: Simple yet effective; self-attention enables full cross-modal interaction among 3D geometric features, 2D semantic features, and proprioception, avoiding hand-crafted fusion design.
- **The pseudo-GT generation strategy** (stabilizing π³ outputs via temporal windows) is transferable to any downstream task that uses 3D foundation models for supervision.

## Limitations & Future Work
- **Single-step horizon prediction**: Only the 3D state at horizon $N$ is predicted, without multi-step 3D trajectory prediction. This may be insufficient for very long-horizon tasks—future work could extend the approach to predict 3D structures at multiple future time steps.
- **No persistent 3D memory**: The model cannot accumulate 3D knowledge across episodes; each inference independently processes the current temporal window. Incorporating persistent 3D memory (inspired by Wang et al.'s continuous 3D perception model) is a promising direction.
- **Real-world success rates remain modest**: The best result on Hanging Mug is only 20%, indicating substantial room for improvement on complex, precision manipulation tasks. More demonstration data or better sim-to-real transfer strategies may be needed.
- **Inference efficiency**: Two large backbones (π³ and DINOv3) combined with iterative diffusion denoising incur considerable computational cost. Inference latency is not reported in the paper; model distillation or acceleration may be required for practical deployment.
- **Generalization to unseen tasks/objects**: Experiments are conducted only on known tasks within the RoboTwin benchmark; zero-shot generalization has not been evaluated.

## Related Work & Insights
- **vs. DP3**: DP3 takes point clouds as direct input, requiring depth cameras and point cloud preprocessing pipelines. GAP requires only RGB input with pretrained 3D model latents and outperforms DP3 on most tasks, demonstrating that implicit 3D representations are more robust than explicit point clouds.
- **vs. G3Flow**: G3Flow projects 2D semantic features onto 3D point clouds. GAP performs semantic-geometric fusion in latent space, avoiding quantization errors and calibration dependencies introduced by 3D projection.
- **vs. Xu et al.**: Xu et al. jointly predict actions and future 2D frames. GAP elevates the prediction target from 2D frames to 3D point maps, better aligning with the inherently 3D nature of manipulation tasks—ablation experiments confirm that 3D prediction outperforms 2D prediction.
- **vs. ACT**: ACT serves as the architectural foundation for GAP (DETR architecture + action chunking). GAP augments ACT with 3D foundation model features and geometric imagination, improving the average success rate from approximately 32% to approximately 55%.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to adopt 3D geometric foundation model latents as the core perceptual prior for a manipulation policy, combined with joint prediction of future 3D structure.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Simulation across 32 tasks in three categories, 6 baselines, ablation studies, data efficiency analysis, and real-world validation on 4 tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and framework figures are intuitive, though some experimental details require the supplementary material.
- **Value**: ⭐⭐⭐⭐ Incorporating 3D foundation models into bimanual manipulation is an important research direction; the joint 3D prediction paradigm has broad implications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[CVPR 2026\] Action-guided Generation of 3D Functionality Segmentation Data](action-guided_generation_of_3d_functionality_segmentation_data.md)
- [\[ICLR 2026\] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](../../ICLR2026/3d_vision/ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[CVPR 2026\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)

<!-- RELATED:END -->
