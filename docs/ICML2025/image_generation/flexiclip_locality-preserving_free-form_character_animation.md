---
title: >-
  [Paper Note] FlexiClip: Locality-Preserving Free-Form Character Animation
description: >-
  [Image Generation] FlexiClip proposes a clipart animation framework based on temporal Jacobian correction, probability flow ODE continuous-time modeling, and GFlowNet flow matching loss. It significantly improves temporal smoothness and geometric integrity of animations while maintaining visual consistency.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 6a4acd0d4ee9ecbd
---

# FlexiClip: Locality-Preserving Free-Form Character Animation

| Information | Content |
|------|------|
| Conference | ICML 2025 |
| arXiv | [2501.08676](https://arxiv.org/abs/2501.08676) |
| Code | [Project Page](https://creative-gen.github.io/flexiclip.github.io/) |
| Area | Image Generation/Animation |
| Keywords | clipart animation, temporal consistency, Bezier curves, probability flow ODE, Flow Matching, GFlowNet, Video SDS |

## TL;DR

FlexiClip proposes a clipart animation framework based on temporal Jacobian correction, probability flow ODE continuous-time modeling, and GFlowNet flow matching loss. It significantly improves temporal smoothness and geometric integrity of animations while maintaining visual consistency.

## Background & Motivation

Translating static clipart into smooth animations is a classic challenge in computer graphics. Existing methods face two core challenges:

**Temporal Inconsistency**: Methods like AniClipart model keypoint trajectories through cubic Bezier curves and use ARAP (As Rigid As Possible) deformation to preserve geometric consistency. However, they easily suffer from abrupt movements and geometric distortions during frame transitions.

**Domain Gap Issue**: Text-to-Video (T2V) and Image-to-Video (I2V) models perform poorly on clipart because natural videos and clipart have significant differences in statistical properties.

Specifically, AniClipart predicts motion dynamics for each frame independently, lacking a correction mechanism for noise accumulation across frames, which leads to unnatural motion artifacts during fast pose transitions. Gal23, though similarly learning neural displacement fields, is also unable to resolve temporal noise issues.

## Method

### Overall Architecture

The core idea of FlexiClip is to decompose animation generation into two stages: **spatial posing** and **temporal smoothing correction**:

1. Use UniPose to detect keypoints and construct the skeleton.
2. Define spatial motion trajectories via cubic Bezier curves.
3. Introduce temporal Jacobian and probability flow ODE to handle temporal noise.
4. Utilize GFlowNet flow matching loss to reduce temporal noise.
5. Distill knowledge from pre-trained video diffusion models through Video SDS loss.

### Spatial Posing

Given an initial mesh $\mathcal{M}_0 = (\mathbf{V}_0, \mathbf{F}_0)$, where $\mathbf{V}_0 \in \mathbb{R}^{V \times 2}$ is the vertex positions, and $\mathbf{F}_0$ is the triangular faces. Keypoints are defined through an indicator matrix $\mathbf{K}_c$ with target positions $\mathbf{T}_c = \mathbf{V}_c + \mathbf{D}_c$.

Mesh deformation is described by a Jacobian field, solving the optimization problem:

$$\mathbf{V}^* = \arg\min_{\mathbf{V}} \|\mathbf{L}\mathbf{V} - \nabla^T \mathcal{A} \mathbf{J}\|^2 + \lambda \|\mathbf{K}_c \mathbf{V} - \mathbf{T}_c\|^2$$

where $\mathbf{L}$ is the cotangent Laplacian operator, and $\mathcal{A}$ is the mass matrix. Keypoints evolve along cubic Bezier curves:

$$p_t(i) = \sum_{j=0}^{3} B_j(u_t) c_j(i)$$

where $u_t \in [0,1]$ is the normalized time, and $B_j$ is the Bernstein basis function.

### Temporal Smoothing

This is the core contribution of Ours. The total Jacobian is decomposed into spatial Jacobian $\mathbf{J}_t^P$ and temporal Jacobian $\mathbf{J}_t^R$ (correction term):

$$\mathbf{J}_t = \mathbf{J}_t^P + \mathbf{J}_t^R$$

The temporal Jacobian models its continuous-time evolution through an ODE:

$$\frac{d\mathbf{J}_t^R}{dt} = f_R(\mathbf{J}_0^P, C_W^P, C_{W-1}^R, t; \theta_R)$$

where $C_W^P$ is the attention-encoded feature of the spatial Jacobian in the current window, and $C_{W-1}^R$ is the attention-encoded feature of the temporal Jacobian from the past window. It is solved by integration:

$$\mathbf{J}_t^R = \mathbf{J}_0^R + \int_0^t f_R(\mathbf{J}_0^P, C_W^P, C_{W-1}^R, \tau; \theta_R) d\tau$$

The initial condition is set to $\mathbf{J}_0^R = \mathbf{0}$, ensuring no correction on the first frame. This design maps the noise term $C(t)$ in pfODE to $C_W^P$, and the scaling term $A(t)$ to $C_{W-1}^R$.

### Loss & Training

**Video SDS Loss**: Distills knowledge from a pre-trained video diffusion model:

$$\nabla_\theta \mathcal{L}_{\text{SDS}}(\phi, \mathbf{X}) = \mathbb{E}_{t', \epsilon}\left[w(t')(\epsilon_\phi(\mathbf{z}_{t'}; \mathbf{y}, t') - \epsilon) \frac{\partial \mathbf{X}}{\partial \theta}\right]$$

**Flow Matching Loss** (inspired by GFlowNet detailed balance conditions):

$$L_{flow} = \mathbb{E}_{t',t} \|\nabla_\mathbf{X} \log p_{t'}(\mathbf{X}, \mathbf{J}_t) - \nabla_\mathbf{X} \log p_{t'}(\mathbf{X}, \mathbf{J}_t^P)\|^2 + \mathbb{E}_t \|\mathbf{J}_t - \mathbf{J}_t^P\|^2$$

The second term is a correction minimization term that encourages the temporal Jacobian to be as small as possible. The total loss is: $L_{SDS} + \lambda \cdot L_{flow}$, where $\lambda = 15$ in experiments.

### Network Architecture

- Spatial posing: 4-layer MLP + LeakyReLU with a linear final layer
- Temporal Jacobian (pfODE): 3-layer MLP
- Attention Network: 2 networks, 32-dimensional key/value, 2 attention heads

## Key Experimental Results

### Main Results

| Method | CLIP Score ↑ | X-CLIP Score ↑ |
|------|-------------|----------------|
| DynamiCrafter | 0.8031 | 0.1732 |
| Gal23 | 0.8395 | 0.1865 |
| VideoCrafter2 | 0.8410 | 0.1988 |
| AniClipart | 0.9401 | 0.2075 |
| **FlexiClip** | **0.9563** | **0.2102** |

| Method | MV ↑ | TC ↓ | GD ↓ | DS ↓ | AE (×10³) ↑ |
|------|------|------|------|------|-------------|
| AniClipart | 20.87 | 8.51 | 50.98 | 18.49 | 75.23 |
| **FlexiClip** | **25.33** | **8.14** | 52.34 | **13.76** | **113.44** |

FlexiClip outperforms AniClipart in both visual fidelity (CLIP 0.9563 vs 0.9401) and text-video alignment (X-CLIP 0.2102 vs 0.2075). In terms of animation metrics, motion vitality is improved by 21%, deformation smoothness is improved by 26%, and animation energy is improved by 51%.

### Ablation Study

| Variant | MV ↑ | TC ↓ | GD ↓ | DS ↓ | AE ↑ |
|------|------|------|------|------|------|
| w/o Temporal Jacobian | 23.00 | 8.80 | 51.50 | 14.00 | 105.00 |
| w/o Flow Matching Loss | 24.50 | 8.40 | 53.00 | 14.20 | 95.00 |
| **Full Model** | **25.33** | **8.14** | 52.34 | **13.76** | **113.44** |

- w/o Temporal Jacobian: motion vitality decreases, temporal consistency degrades, and rigid deformations appear.
- w/o Flow Matching Loss: geometric distortion increases, animation energy decreases, and limb movements become unstable.

### User Study

55 cliparts, 30 participants, 6 methods compared. FlexiClip leads in all dimensions: identity preservation 94.9%, text alignment 94.5%, and smoothness 93.8%, far exceeding AniClipart (83.6%, 80.7%, and 76.4%, respectively).

## Highlights & Insights

1. **Jacobian Decomposition Strategy**: Decomposes the total Jacobian into spatial and temporal correction terms to achieve fine-grained motion control.
2. **pfODE Continuous-Time Modeling**: Better handles cross-frame noise accumulation compared to discrete-time methods.
3. **GFlowNet-Inspired Flow Matching Loss**: Artfully leverages detailed balance conditions to make the forward process eliminate the temporal noise introduced by the backward process.
4. **Multi-Functional Support**: Supports rotation, multi-text conditions, multi-object interaction, and layered animation.
5. **End-to-End Differentiable**: The entire pipeline is differentiable, allowing joint optimization of Bezier parameters and temporal parameters.

## Limitations & Future Work

1. **Slightly Higher GD Metric**: Since ARAP deformation is not used, the geometric deviation is slightly higher than AniClipart (52.34 vs 50.98).
2. **Computational Overhead**: Generating a 24-frame animation takes about 40 minutes on a V100, consuming 26GB of VRAM.
3. **Dependency on Pre-trained Models**: Video SDS loss relies on the quality of the ModelScope T2V model.
4. **Sensitivity to Hyperparameters**: The choice of $\lambda$ has a significant impact on motion quality and convergence speed (too low leads to slow convergence, too high leads to unnatural motion).
5. **Limited to 2D**: The current framework is limited to 2D clipart and has not been extended to 3D animations.

## Rating

⭐⭐⭐⭐ (4/5)

Highly innovative, introducing pfODE and GFlowNet to clipart animation is an interesting cross-domain fusion. The experiments are comprehensive, including quantitative evaluation, ablation study, and user study. However, the computational cost is high, the GD metric is slightly inferior, and it is limited to 2D scenarios, which makes practical application scenarios relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](../../ICCV2025/image_generation/emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)

</div>

<!-- RELATED:END -->
