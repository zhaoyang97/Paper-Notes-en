---
title: >-
  [Paper Note] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control
description: >-
  [ICLR 2026][Video Generation] This paper proposes RoboMaster, a framework that decomposes the robot–object interaction process into three temporal stages—pre-interaction, in-interaction, and post-interaction—via a collaborative trajectory representation, combined with appearance- and shape-aware object embeddings, to achieve high-quality video generation for robotic manipulation.
tags:
  - ICLR 2026
  - Video Generation
  - robotic manipulation
  - collaborative trajectory
  - diffusion model
  - interaction modeling
date: 2026-05-08
content_hash: 23c0c451184ebac2
---

# Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control

**Conference**: ICLR 2026
**arXiv**: [2506.01943](https://arxiv.org/abs/2506.01943)
**Code**: [Project Page](https://fuxiao0719.github.io/projects/robomaster/)
**Area**: Video Generation
**Keywords**: video generation, robotic manipulation, collaborative trajectory, diffusion model, interaction modeling

## TL;DR

This paper proposes RoboMaster, a framework that decomposes the robot–object interaction process into three temporal stages—pre-interaction, in-interaction, and post-interaction—via a collaborative trajectory representation, combined with appearance- and shape-aware object embeddings, to achieve high-quality video generation for robotic manipulation.

## Background & Motivation

1. **Background**: Video diffusion models have demonstrated significant potential for generating robot decision-making data, and trajectory-conditioned control enables fine-grained control over robot motion.

2. **Limitations of Prior Work**: Existing trajectory control methods (e.g., Tora, DragAnything) focus primarily on the independent motion of individual objects, using separate trajectories to control the robotic arm and the manipulated object. This leads to feature entanglement in interaction regions (overlapping areas), degrading generation quality.

3. **Key Challenge**: Robotic manipulation is inherently a multi-object interaction process, yet prior methods reduce it to independent motion control, failing to capture physically plausible interactions. If synthesized videos cannot accurately represent interaction phases, inverse dynamics models will extract unreliable action labels.

4. **Goal**: To design a video generation framework that accurately models robot–object interaction dynamics, such that the generated videos can serve as high-quality demonstration data for robot learning.

5. **Key Insight**: Rather than decomposing objects, the paper decomposes the interaction process—dividing manipulation into three sub-stages, each guided by the dominant agent, and unifying them into a single collaborative trajectory.

6. **Core Idea**: By decomposing the interaction process rather than individual objects, multi-object trajectories are unified into a collaborative trajectory representation, fundamentally avoiding feature entanglement in overlapping regions.

## Method

### Overall Architecture

RoboMaster builds upon the pretrained CogVideoX-5B architecture. Given an initial frame $\mathbf{I}$, a text prompt $\mathbf{c}$, object masks $\mathbf{M}_d, \mathbf{M}_s$, and a collaborative trajectory $\mathcal{C}$, the model generates a manipulation video $\mathbf{X}$. The pipeline proceeds as follows: (1) object representations $\mathbf{v}_d, \mathbf{v}_s$ are encoded via appearance- and shape-aware embeddings; (2) the trajectory is decomposed into three sub-stages and associated with corresponding object features; (3) a motion injection module injects the collaborative trajectory embeddings into the DiT blocks.

### Key Designs

**1. Coupled Appearance and Shape Embedding**

- **Function**: Maintains semantic consistency of objects across the video sequence.
- **Mechanism**: The initial frame is projected into latent features $\mathbf{z}$ via the VAE encoder; downsampled object masks are used to extract masked latent features, which are then pooled to obtain $\tilde{\mathbf{v}}$. At each timestep, a circular volumetric representation $\mathbf{v} \in \mathbb{R}^{c \times h \times w}$ is constructed centered at the trajectory point with a radius proportional to the mask area.
- **Design Motivation**: Compared to point-based representations (e.g., Tora), mask-based representations encode both object appearance and spatial shape information, accelerating training convergence and improving cross-frame identity consistency.

**2. Collaborative Trajectory Representation**

- **Function**: Unifies multi-object interaction dynamics modeling and avoids feature entanglement.
- **Mechanism**: The trajectory is decomposed into three temporal stages: pre-interaction $\mathcal{C}_1$ (arm-dominant), in-interaction $\mathcal{C}_2$ (object-dominant), and post-interaction $\mathcal{C}_3$ (arm-dominant). A causal representation propagates the latent from the previous timestep to subsequent frames. The final distribution is factorized as a product of three object-aware sub-distributions.
- **Design Motivation**: During the interaction stage, the manipulated object's motion implicitly guides the arm trajectory (their relative dynamics are constrained). The temporal variation in feature representation $\mathbf{v}_d \rightarrow \mathbf{v}_s \rightarrow \mathbf{v}_d$ provides cues for modeling behavioral transitions.

**3. Motion Injection Module**

- **Function**: Injects collaborative trajectory information into the video DiT generation process.
- **Mechanism**: The collaborative trajectory latent $\mathbf{V} \in \mathbb{R}^{f \times c \times h \times w}$ is patchified and encoded via zero-initialized 2D spatial convolution and 1D temporal convolution, then added to the hidden states of the DiT blocks: $\mathbf{h} = \mathbf{h} + \text{norm}(\tilde{\mathbf{V}}) + \tilde{\mathbf{V}}$
- **Design Motivation**: Zero initialization ensures the pretrained model's generative capability is not disrupted at the start of training; the plug-and-play design facilitates integration.

### Loss & Training

Standard diffusion model denoising loss: $\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}[\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{x}_t, \mathbf{c}, \mathbf{M}_d, \mathbf{M}_s, \mathcal{C}, t)\|_2^2]$

Training configuration: 8× A800 GPUs, AdamW optimizer, DiT learning rate $2 \times 10^{-5}$, motion injector learning rate $1 \times 10^{-4}$, batch size 16, trained for 30K steps. Inference uses 50-step DDIM with CFG scale 6.0.

## Key Experimental Results

### Main Results

Video generation quality and trajectory accuracy comparison (Bridge dataset; all baselines retrained on identical data):

| Method | FVD↓ | PSNR↑ | SSIM↑ | TrajError_robot↓ | TrajError_obj↓ |
|--------|------|-------|-------|------------------|----------------|
| TesserAct | 261.84 | 18.99 | 0.778 | 37.34 | 54.64 |
| IRASim | 159.04 | 20.88 | 0.782 | 19.25 | 34.39 |
| DragAnything | 158.42 | 21.13 | 0.792 | 18.97 | 27.41 |
| Tora | 152.28 | 21.24 | 0.788 | 18.14 | 26.43 |
| **RoboMaster** | **147.31** | **21.55** | **0.803** | **16.47** | **24.16** |

Robot action planning success rates (RLBench + SIMPLER; average success rate over 100 trials each):

| Method | pick up cup | put knife | open microwave | close box | pick coke can |
|--------|-------------|-----------|----------------|-----------|---------------|
| OpenVLA | 0.55 | 0.46 | 0.35 | 0.45 | 0.59 |
| Tora | 0.79 | 0.82 | 0.61 | 0.72 | 0.89 |
| RoboMaster | **0.83** | 0.76 | 0.54 | **0.79** | **0.91** |

### Ablation Study

| Configuration | FVD↓ | PSNR↑ | TrajError_obj↓ | Note |
|---------------|------|-------|----------------|------|
| w/o causal embedding | 151.62 | 21.30 | 27.15 | Reduced temporal coherence |
| Point repr. replacing mask | 157.49 | 20.87 | 31.41 | Significant drop in object identity consistency |
| Separate trajectories | 152.01 | 21.08 | 25.84 | Feature entanglement in interaction regions |
| Cross-attention injection | 163.56 | 19.38 | 29.16 | Inferior to additive injection |
| **Full model** | **147.31** | **21.55** | **24.16** | All components synergistically optimal |

### Key Findings

- The collaborative trajectory design outperforms Tora on 8 out of 10 robotic tasks, demonstrating the importance of precise interaction modeling for downstream policy learning.
- The mask representation retains 99.81% PSNR under 90% sparsity, exhibiting strong robustness to coarse user inputs.
- Even when 40% of prompts are replaced with imprecise descriptions, the model maintains over 96% PSNR performance.
- A user preference rate of 45.16% substantially surpasses all baselines (the second-ranked Tora achieves only 17.74%).

## Highlights & Insights

- The core insight of **"decomposing the interaction rather than the objects"** is both concise and effective, fundamentally resolving the feature entanglement problem.
- The collaborative trajectory design also simplifies user interaction—users need only define piecewise trajectories rather than complete multi-object trajectories.
- The circular volumetric representation captures both appearance and shape information, constituting an elegant design choice.
- Downstream robot planning experiments validate a positive correlation between video generation quality and policy learning performance.

## Limitations & Future Work

- The method operates purely in 2D pixel space; integrating depth cues could enable more precise 3D control.
- Out-of-domain inputs may produce incomplete or distorted objects.
- Generalization to different robot morphologies still requires expanded training data.
- The collaborative trajectory requires prior knowledge of the temporal boundary points between interaction stages; automatic detection would be more practical.

## Related Work & Insights

- This work presents a clear contrast with separate-trajectory methods such as Tora, highlighting the importance of an "interaction modeling" perspective in video generation.
- Compared to TesserAct's 4D approach, the 2D method offers advantages in data efficiency.
- The paradigm of video generation as a world simulator demonstrates considerable promise for robot learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The interaction decomposition via collaborative trajectory is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Encompasses video quality, trajectory accuracy, robot planning, ablations, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with high-quality figures.
- Value: ⭐⭐⭐⭐ Provides practical guidance for robotic video generation and data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2026/video_generation/flashmotion_fewstep_controllable_video_generation.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](../../CVPR2026/video_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
