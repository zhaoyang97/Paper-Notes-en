---
title: >-
  [Paper Note] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation
description: >-
  [CVPR 2026][Image Generation][First-person video] EgoFlow proposes a generative framework based on Flow Matching that integrates multimodal scene conditions through a Mamba-Transformer-Perceiver hybrid architecture. During inference, it applies differentiable physical constraints (collision avoidance, motion smoothness) via gradient-guided sampling to generate physically plausible 6DoF object motion trajectories from first-person videos, reducing collision rates by up to 79%.
tags:
  - CVPR 2026
  - Image Generation
  - First-person video
  - 6DoF trajectory generation
  - Flow Matching
  - Gradient-guided sampling
  - Mamba-Transformer hybrid architecture
date: 2026-05-08
content_hash: 14a60ef0b1b62595
---

# EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation

**Conference**: CVPR 2026  
**arXiv**: [2604.01421](https://arxiv.org/abs/2604.01421)  
**Code**: [https://abhi-rf.github.io/egoflow/](https://abhi-rf.github.io/egoflow/)  
**Area**: Image Generation  
**Keywords**: First-person video, 6DoF trajectory generation, Flow Matching, Gradient-guided sampling, Mamba-Transformer hybrid architecture

## TL;DR
EgoFlow proposes a generative framework based on Flow Matching that integrates multimodal scene conditions through a Mamba-Transformer-Perceiver hybrid architecture. During inference, it applies differentiable physical constraints (collision avoidance, motion smoothness) via gradient-guided sampling to generate physically plausible 6DoF object motion trajectories from first-person videos, reducing collision rates by up to 79%.

## Background & Motivation

1.  **Background**: With the popularity of AR devices and the emergence of large-scale first-person datasets (EgoExo4D, HOT3D, HD-EPIC), understanding and predicting object motion from a first-person perspective has become a core capability for embodied perception and robotic interaction. Existing methods primarily rely on diffusion models or autoregressive prediction to generate object trajectories.

2.  **Limitations of Prior Work**:
    *   First-person scenes are highly diverse and cluttered; objects are frequently occluded; and limited fields of view combined with fast camera motion lead to blurring.
    *   In long-term prediction, tiny spatial errors accumulate over time, leading to unrealistic motion patterns.
    *   Existing generative models lack explicit physical reasoning capabilities, and generated trajectories cannot guarantee collision-free paths or dynamic smoothness.

3.  **Key Challenge**: Generative models must simultaneously satisfy two seemingly contradictory requirements—motion diversity (learning rich motion distributions) and physical consistency (collision avoidance, motion smoothness)—whereas pure data-driven methods cannot explicitly reason about physical constraints in new scene configurations.

4.  **Goal**
    *   How to generate long-term, physically plausible 6DoF object trajectories?
    *   How to ensure collision avoidance and motion smoothness in generated trajectories without physical supervision?
    *   How to effectively fuse multimodal conditions such as scene geometry, semantics, and goals?

5.  **Key Insight**: Using Flow Matching instead of diffusion models to learn a deterministic transport field for efficient trajectory synthesis; using gradient-guided sampling to inject physical constraints during inference (decoupling training and testing); and using a Mamba+Transformer+Perceiver hybrid architecture to handle long-sequence multimodal fusion.

6.  **Core Idea**: The combination of learning motion distributions with Flow Matching and imposing physical constraints via gradient guidance achieves an elegant decoupling where the "model learns the data distribution, and constraints are injected on-demand during inference."

## Method

### Overall Architecture
Given a 3D scene (point clouds, fixture bounding boxes), a task text prompt, and a goal pose, EgoFlow first encodes the scene context into a unified condition vector $\mathbf{u}$ via a multimodal condition fusion module. Then, a Flow Matching model (Mamba-Transformer hybrid architecture) generates a 6DoF trajectory $\mathbf{x}_{H+1:T} \in \mathbb{R}^{(T-H) \times 9}$ (position $\mathbb{R}^3$ + continuous 6D rotation $\mathbb{R}^6$) from noise. During inference, the velocity field is optimized via gradient-guided sampling to satisfy physical constraints. Given 30% of the observed trajectory, the model predicts the remaining 70%.

### Key Designs

1.  **Multimodal Scene Condition Fusion**:
    *   **Function**: Encodes multimodal information (geometry, semantics, goals) into a unified condition vector.
    *   **Mechanism**: Merges five modalities—(i) Trajectory dynamics: Observed history $\mathbf{x}_{1:H}$ is linearly projected into temporal embeddings $\mathbf{F}_{traj}$; (ii) Local geometry: PointNet++ encodes point clouds, with features propagated to object centers via inverse distance weighting; (iii) Fixture layout: Bounding box geometric embeddings capture spatial relationships (e.g., "tabletop is above the cabinet") via self-attention; (iv) Semantic prompts: CLIP encodes object categories and task descriptions; (v) Goal description: The target pose $\mathbf{x}_T$ is mapped to a goal token via an MLP. Finally, these are concatenated and projected as $\mathbf{u} = \text{MLP}([\mathbf{F}_{traj}, \mathbf{F}_p, \mathbf{F}_g, \mathbf{F}_b, \mathbf{F}_s, \mathbf{F}_{goal}])$.
    *   **Design Motivation**: A single modality cannot fully describe a scene—geometry provides collision info, semantics provide behavior priors, and goals constrain long-term planning direction.

2.  **Mamba-Transformer-Perceiver Hybrid Architecture**:
    *   **Function**: Efficiently models long-sequence temporal dependencies and implements multimodal reasoning.
    *   **Mechanism**: A three-stage design—Stage 1 (Temporal Encoding): 3-layer bidirectional Mamba, injecting conditions via FiLM modulation ($\mathbf{h}_t' = \gamma(\mathbf{u}_t) \odot \mathbf{h}_t + \beta(\mathbf{u}_t)$), capturing long-range dependencies with linear complexity. Bidirectional processing allows the model to utilize both past and future contexts; Stage 2 (Cross-modal Attention): 6-layer Perceiver-style Transformer, using self-attention to model temporal relationships between trajectory tokens, followed by cross-attention to query multimodal conditions $\mathbf{u}_t$, and finally an FFN to refine fused representations; Stage 3 (Trajectory Refining): Another 3-layer bidirectional Mamba + FiLM, refining the representations before a linear head outputs $\mathbb{R}^9$ velocity predictions.
    *   **Design Motivation**: Mamba’s linear complexity suits long trajectory sequences, Transformer’s cross-attention is ideal for multimodal reasoning, and Perceiver reduces computational overhead. The three stages compress time, fuse modalities, and refine output sequentially.

3.  **Gradient-Guided Sampling (Inference-time Physical Constraints)**:
    *   **Function**: Dynamically injects collision avoidance and motion smoothness constraints during inference without additional training.
    *   **Mechanism**: Before each Euler integration step, the predicted velocity field is optimized for $K=50$ gradient descent steps: $\mathbf{v}_\theta^{(k+1)} = \mathbf{v}_\theta^{(k)} - \alpha \nabla_\mathbf{v} \mathcal{J}$. The total cost function $\mathcal{J} = \mathcal{J}_{coll} + \lambda_{rot}\mathcal{J}_{rot} + \lambda_{vel}\mathcal{J}_{vel}$ includes: collision avoidance using SDF to penalize distance violations between trajectory points and fixtures (safety margin $\epsilon=5$cm); rotational consistency using cosine similarity of continuous rotation changes to penalize sudden jumps; and translational smoothness to penalize linear acceleration. All cost functions are fully differentiable, supporting backpropagation.
    *   **Design Motivation**: Pure data-driven models cannot explicitly avoid obstacles in new scenes. Decoupling constraints from generation—where the model learns the distribution and constraints are injected as needed—avoids training-testing distribution mismatch and allows for flexible adjustment of constraint strength.

### Loss & Training
*   Flow Matching Loss: $\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}[\|\mathbf{v}_\theta(\mathbf{x}_t, t, \mathcal{S}) - (\mathbf{x}_1 - \mathbf{x}_0)\|_1]$
*   Linear interpolation path: $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$
*   Equal-weighted L1 loss for position and rotation components.
*   AdamW, lr=$10^{-4}$, batch 32, 100 epochs, 20-step Euler sampling.

## Key Experimental Results

### Main Results — HD-EPIC Dataset

| Method | ADE↓ | FDE↓ | Frechet↓ | Geodesic↓ | Collision↓ |
|------|------|------|----------|-----------|------------|
| GIMO | 0.285 | 0.509 | 0.210 | **0.725** | 23.5% |
| CHOIS | 0.471 | 0.755 | 0.262 | 1.255 | 18.7% |
| M2Diffuser | 0.601 | 0.442 | 0.476 | 1.788 | 8.5% |
| EgoScaler | 1.330 | 1.494 | 0.315 | 1.614 | 35.8% |
| **Ours** | **0.279** | **0.102** | **0.197** | 1.141 | **2.5%** |

### Zero-Shot Transfer — Ego-Exo4D→HOT3D

| Method | ADE↓ | FDE↓ | GD↓ |
|------|------|------|-----|
| GIMO | 0.299 | 0.436 | 2.06 |
| EgoScaler | 0.351 | 0.540 | **0.856** |
| **Ours** | **0.265** | **0.027** | 1.49 |

### Ablation Study — HD-EPIC Input and Guidance Ablation

| Configuration | ADE↓ | FDE↓ | Frechet↓ | Coll.↓ |
|------|------|------|----------|--------|
| Full model | 0.279 | 0.102 | 0.197 | 2.5% |
| w/o Point Cloud $\mathcal{P}$ | 0.305 | 0.110 | 0.205 | 2.9% |
| w/o Action Text | 0.330 | 0.147 | 0.213 | 3.1% |
| w/o Goal Pose $\mathbf{x}_T$ | 0.386 | 0.619 | 0.239 | 3.1% |
| w/o Obs. History | 0.405 | 0.207 | 0.275 | - |

### Key Findings
*   **Massive Reduction in Collision Rate**: EgoFlow achieves only 2.5%, a 71% reduction compared to the runner-up M2Diffuser (8.5%) and an 89% reduction compared to GIMO (23.5%). Gradient-guided sampling is the key contributor.
*   **Extremely Low Endpoint Error**: FDE=0.102 (HD-EPIC) and 0.027 (HOT3D), significantly outperforming all baselines. The goal pose condition $\mathbf{x}_T$ contributes most (removing it increases FDE from 0.102 to 0.619).
*   **Excellent Cross-dataset Generalization**: Leads significantly even in the Ego-Exo4D training and HOT3D zero-shot testing setup.
*   **Goal Pose as Most Critical Input**: Removing $\mathbf{x}_T$ causes the worst degradation in FDE (6x), while removing point clouds or action text has a relatively smaller impact.

## Highlights & Insights
*   **Decoupling of Training-Inference Constraints**: The model is trained only on collision-free demonstrations (learning the motion distribution), while physical constraints are injected via gradient guidance during inference. This design elegantly avoids the distribution mismatch where a "model that hasn't seen collisions during training doesn't know how to avoid them during inference."
*   **Meticulous Hybrid Architecture**: FiLM provides global modulation, while cross-attention performs selective fusion. These two injection methods serve different roles—FiLM is efficient but less granular, while cross-attention is fine-grained but computationally heavier. The three-stage architecture allows both to perform at their best.
*   **Flow Matching over Diffusion**: Deterministic transport fields are better suited for trajectory generation than stochastic diffusion—providing smoother probability paths and requiring fewer sampling steps (only 20), resulting in more coherent trajectories.

## Limitations & Future Work
*   The 50 optimization steps for gradient guidance increase inference time overhead; more efficient injection methods (e.g., classifier-free guidance or learned constraint networks) could be explored.
*   SDF collision detection is based on simple OBBs (Oriented Bounding Boxes); objects with complex shapes may require more precise collision detection.
*   Training data trajectories are derived from hand motion proxies (rigid-body coupling assumption), introducing noise and approximation errors.
*   Dynamic obstacles are not considered (only static scene fixtures are handled); real-world interactions may involve other moving objects.

## Related Work & Insights
*   **vs EgoScaler**: EgoScaler is a vision-language generation framework based on PointLLM, but it suffers from high position errors and collision rates (35.8%); EgoFlow surpasses it in both position accuracy and physical plausibility.
*   **vs M2Diffuser**: M2Diffuser also employs inference-time physical constraints but is based on diffusion models and still has an 8.5% collision rate; EgoFlow reduces this to 2.5% using Flow Matching and more refined gradient guidance.
*   **vs GMT**: GMT is a precursor to EgoFlow, sharing the multimodal condition design. EgoFlow builds upon this by introducing Flow Matching + Mamba hybrid architecture + gradient guidance.

## Rating
*   Novelty: ⭐⭐⭐⭐ The combination of Flow Matching, gradient-guided physical constraints, and Mamba-Transformer hybrid architecture is novel in the field of trajectory generation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, seven baselines, comprehensive ablations, zero-shot transfer, and qualitative visualizations.
*   Writing Quality: ⭐⭐⭐⭐ The method description is well-organized, with clear design motivations for each module.
*   Value: ⭐⭐⭐⭐ Physics-guided trajectory generation methods have broad potential applications in robotics and AR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](../../NeurIPS2025/image_generation/value_gradient_guidance_for_flow_matching_alignment.md)
- [\[CVPR 2026\] HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching](hazematching_dehazing_light_microscopy_images_with_guided_conditional_flow_match.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)

</div>

<!-- RELATED:END -->
