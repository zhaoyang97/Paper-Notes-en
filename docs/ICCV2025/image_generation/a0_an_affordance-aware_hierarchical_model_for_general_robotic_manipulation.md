---
title: >-
  [Paper Note] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation
description: >-
  [ICCV 2025][Image Generation][robotic manipulation] This paper proposes A0, a hierarchical affordance-aware diffusion model that decomposes manipulation tasks into high-level spatial understanding and low-level action ex…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "robotic manipulation"
  - "spatial affordance"
  - "hierarchical model"
  - "diffusion model"
  - "cross-platform generalization"
date: 2026-05-08
content_hash: 831f9801318a0d4f
---

# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation

**Conference**: ICCV 2025
**Code**: [https://a-embodied.github.io/A0/](https://a-embodied.github.io/A0/)  
**Area**: Image Generation
**Keywords**: robotic manipulation, spatial affordance, hierarchical model, diffusion model, cross-platform generalization

## TL;DR

This paper proposes A0, a hierarchical affordance-aware diffusion model that decomposes manipulation tasks into high-level spatial understanding and low-level action execution by predicting object-centric contact points and post-contact trajectories via an Embodiment-Agnostic Affordance Representation. Pre-trained on 1 million contact-point annotations, A0 generalizes across four robot platforms: Franka, Kinova, Realman, and Dobot.

## Background & Motivation

**Why do existing methods fall short on spatial affordance?** The core challenge in robotic manipulation is understanding *where* and *how* to interact with objects — i.e., spatial affordance. Existing approaches fall into two broad categories:

**Modular methods** (MOKA, ReKep): leverage large vision models for spatial reasoning, but lack deep understanding of object spatial and physical properties, particularly failing to capture **manipulability**.

**End-to-end VLA methods** (π0, RDT): directly generate action sequences without sufficient spatial understanding, leading to poor performance on complex tasks such as wiping a whiteboard or stacking objects.

**Why object-centric representations?** Existing affordance methods typically produce heatmaps or dense point flows, both of which are computationally expensive and tightly coupled to specific robot morphologies. An object-centric contact point + trajectory representation is inherently **embodiment-agnostic**: it only requires predicting keypoints on the object and does not depend on the kinematics of any particular robot.

**Why hierarchical?** Direct end-to-end mapping from perception to action is overly difficult. Decomposing the problem into "understanding where and how to manipulate" and "actually executing the manipulation" yields simpler, more transferable learning objectives at each level.

## Method

### Overall Architecture

A0 decomposes robotic manipulation into two levels:

1. **High-level spatial affordance understanding**: predicts contact points and post-contact trajectories (the core of the A0 model).
2. **Low-level action execution**: projects 2D predictions into 3D space, estimates grasp poses, and executes motion.

### Key Designs

#### Embodiment-Agnostic Affordance Representation

The unified representation is defined as: $R = R_R \cup R_H \cup R_C = \{(I, L, C, T) | C = (c^{2D}_0), T = (t^{2D}_0, t^{2D}_1, t^{2D}_2, \cdots)\}$

- $I$: object-centric RGB image
- $L$: natural language manipulation instruction
- $C$: contact point (2D coordinate)
- $T$: post-contact trajectory (sequence of 2D keypath points)

**Why unify heterogeneous data sources?** Affordance knowledge is distributed across multiple data modalities: real robot data $R_R$ (precise but scarce), hand-object interaction data $R_H$ (rich interaction knowledge), and custom/simulation data $R_C$ (controllable but with a sim-to-real gap). The unified representation consolidates all sources into a common format.

Dataset composition:
- **PixMo-One-Point**: 1 million single contact-point annotations (internet images)
- **HOI4D-22k**: 22,000 human-object interaction trajectories
- **DROID-3k**: 3,056 real robot manipulation trajectories
- **ManiSkill-5k**: 4,965 simulation trajectories

#### A0 Model Architecture

Built on a Diffusion Transformer (DiT) backbone with $N=28$ layers and 1B parameters:

**Input**: diffusion timestep $k$ + noisy waypoints $x^k_{t:t+T}$
**Conditioning**: observation images $I_{t-1:t}$ (current and previous frames) + language instruction $\ell$

Waypoints are defined as the affordance representation: $x_{t:t+T}$, where $x_t = (u, v) \in [0,1]^2$ and $T=5$ is the chunk size.

**Position Offset Attention (POA)**:
Motion information is critical for understanding manipulation progress. POA computes motion tokens by subtracting adjacent-frame visual tokens: $I^i_m = I^i_t - I^i_{t-1}$, then concatenates them with the current frame: $o_t = \text{concat}([I^i_t, I^i_m], \text{dim}=1)$.

**Spatial Information Aggregation Layer (SIAL)**:
A nonlinear MLP decoder appended as the final layer, mapping from latent space to physical coordinate space. **Why is an additional projection layer needed?** DiT outputs reside in latent space; direct decoding may fail to precisely map to pixel coordinates, so SIAL provides an accurate coordinate transformation from latent to physical space.

**Encoders**:
- Visual encoder: pre-trained SigLiP (400M)
- Language encoder: pre-trained Qwen2.5-7B
- Image and text tokens condition the diffusion process via an interleaved cross-attention mechanism

#### Action Execution Module

1. **2D-to-3D projection**: $X_i = D(x_i) K^{-1} \tilde{x}_i$, using the depth map and camera intrinsic matrix.
2. **Grasp pose estimation**: GraspNet is queried to generate candidates; the candidate closest to the contact point is selected: $G^* = \arg\min_{G \in \mathcal{G}} \|G - X_t\|$.
3. **Waypoint-guided execution**: a VLM selects height categories, and a smooth trajectory is generated in $SE(3)$ space.

### Loss & Training

**Pre-training phase** (80K steps, 5 days, 4×A100):
Uses single-frame images and the first waypoint (contact point) only; MSE loss:
$$L_p(\theta) = \frac{1}{n}\sum_{i=1}^n ((x^0_t)_i - (f_\theta(k, x^k_t, I_t, \ell))_i)^2$$

**Supervised fine-tuning phase** (30K steps, 50 hours):
Extended to $T$ waypoints, incorporating motion information; forward diffusion noise is added before predicting the original waypoints:
$$L_s(\theta) = \frac{1}{n}\sum_{i=1}^n ((x^0_{t:t+T})_i - (f_\theta(k, x^k_{t:t+T}, I_{t-1:t}, \ell))_i)^2$$

At inference, a fast ODE solver is used, requiring only $K_D = 5$ denoising steps (vs. $K_F = 1000$ during training).

## Key Experimental Results

### Main Results

**Multi-platform real-world performance comparison (Table 2, 20 trials per task):**

| Robot | Method | Place Object | Open Drawer | Press Button | Wipe Board | Avg. Success Rate |
|-------|--------|-------------|-------------|-------------|------------|-------------------|
| Kinova | MOKA | 70 | 50 | 30 | 30 | 45.00 |
| Kinova | ReKep | 75 | 55 | 5 | 0 | 33.75 |
| Kinova | **A0-1B** | 60 | 65 | 40 | **50** | **53.75** |
| Franka | Magma | 25 | 10 | 30 | 0 | 16.25 |
| Franka | Molmo | 60 | 40 | 55 | 20 | 43.75 |
| Franka | **A0-1B** | 60 | **75** | **70** | **45** | **62.50** |

**vs. VLA methods (Table 3, Kinova platform):**

| Method | Place Object | Open Drawer | Press Button | Wipe Board | Avg. | Steps |
|--------|-------------|-------------|-------------|------------|------|-------|
| RDT-1B | 20 | 0 | 25 | 0 | 11.25 | 25–50 |
| π0 | 40 | 20 | 10 | 10 | 20.00 | 25–50 |
| π0 + FAST | 35 | 10 | 30 | 0 | 18.75 | 25–50 |
| **A0-1B** | **60** | **65** | **40** | **50** | **53.75** | **4–5** |

On the Wipe Board task, A0 outperforms π0 by **40 percentage points** while requiring only 4–5 execution steps (vs. 25–50).

### Ablation Study

**Architecture ablation (Table 1, MAE↓ after pre-training):**

| Configuration | HOI4D-22k | ManiSkill-5k | DROID-3k |
|--------------|-----------|-------------|---------|
| A0-1B | 47.5 | 5.5 | 17.5 |
| w/o POA | 47.9 | 6.3 (+0.8) | 18.5 |
| w/o SIAL | 61.1 (+13.6) | 10.2 (+4.7) | 19.6 |

The impact of SIAL is most pronounced: removing it increases MAE on HOI4D by 13.6 pixels, demonstrating that precise mapping from latent space to coordinate space is indispensable.

**Pre-training effect (Figure 4):**

| Transfer Paradigm | Dataset | MAE w/o Pre-training | MAE w/ Pre-training | Reduction |
|-------------------|---------|---------------------|---------------------|-----------|
| Real-to-Sim | ManiSkill-5k | 50.4 | 43.9 | −13% |
| Sim-to-Real | HOI4D-22k | 172.2 | 35.1 | −80% |
| Sim-to-Real | DROID-3k | 125.2 | 29.1 | −77% |

Pre-training is particularly impactful in Sim-to-Real settings, reducing MAE by 77–80%.

### Key Findings

1. **Hierarchical > end-to-end**: A0 outperforms π0 by 33.75% on average success rate.
2. **Single inference vs. multi-step inference**: A0 requires only 4–5 keypath waypoints, while VLA methods require 25–50 steps.
3. **Trajectory tasks show the largest gains**: tasks requiring precise trajectory following, such as Wipe Board, exhibit the greatest improvement.
4. **Pre-training is essential**: large-scale contact-point localization pre-training on 1 million samples significantly boosts downstream performance.

## Highlights & Insights

- **Minimalist object-centric affordance representation**: predicting only contact points and subsequent trajectory points substantially reduces complexity.
- **Empirical validation of embodiment-agnosticism**: demonstrated across four distinct robot platforms, which is highly convincing.
- **Successful application of the pre-train → fine-tune paradigm in robotics**: 1 million internet-sourced contact-point annotations establish a strong localization foundation.
- **Efficiency advantage**: single-inference 4–5 steps vs. 25–50 steps is practically significant for real-world deployment.
- **Data fusion strategy**: internet data, HOI data, and robot data are unified into a shared representation space.

## Limitations & Future Work

- A0 underperforms MOKA and ReKep on Place Object for Kinova, likely because the latter leverage SAM/GPT-4 which have been exposed to a broader range of real-world objects.
- Long-horizon planning relies on an external VLM for task decomposition, which is not end-to-end.
- Fine-grained, orientation-sensitive manipulation is not supported without additional VLM prompting.
- Validation is limited to four simple household tasks; more complex scenarios such as assembly or tool use are not addressed.
- Depth map quality substantially affects 2D-to-3D projection, yet robustness to depth noise is not discussed.

## Related Work & Insights

A0 differs from Helix (hierarchical reinforcement learning) in that it employs an explicit spatial affordance representation, whereas Helix learns implicit semantic representations. Compared to MOKA/ReKep, which directly leverage large vision models, A0 acquires deeper spatial understanding through large-scale pre-training. This work suggests that **decoupling "understanding" from "execution," combined with large-scale pre-training, constitutes an effective paradigm for embodied AI**.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](aether_geometric-aware_unified_world_modeling.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](diffusion_image_prior.md)

</div>

<!-- RELATED:END -->
