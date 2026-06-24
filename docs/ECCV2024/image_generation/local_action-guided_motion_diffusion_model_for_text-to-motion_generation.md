---
title: >-
  [Paper Note] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation
description: >-
  [ECCV 2024][Image Generation] GuidedMotion is proposed to guide global motion diffusion generation using local actions as fine-grained control signals. By estimating guidance weights through semantic graph parsing and Graph Attention Networks, it supports continuously adjustable motion control, demonstrating significant advantages in generating complex multi-action motions.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: f0d5b10da125ffc6
---

# Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation

**Conference**: ECCV 2024  
**arXiv**: [2407.10528](https://arxiv.org/abs/2407.10528)  
**Area**: Image Generation

## TL;DR

GuidedMotion is proposed to guide global motion diffusion generation using local actions as fine-grained control signals. By estimating guidance weights through semantic graph parsing and Graph Attention Networks, it supports continuously adjustable motion control, demonstrating significant advantages in generating complex multi-action motions.

## Background & Motivation

Text-to-motion generation requires grounding local actions from language (e.g., "walking", "raising hands") and seamlessly merging them into diverse and realistic global motions. However, existing methods mainly focus on directly synthesizing global motion, ignoring the generation and control of local actions. Expressing complex trajectories, poses, and long motion sequences containing multiple actions precisely with text is highly challenging, often requiring iterative prompt editing.

This paper proposes a new paradigm of **local action guidance**: using reference local actions as control signals to provide conditional guidance during the global motion diffusion process, thereby achieving motion generation from local to global.

## Method

### Overall Architecture

GuidedMotion consists of the following core modules:
1. **Automatic Local Action Sampling**: Semantic graph parsing $\rightarrow$ local action descriptions $\rightarrow$ text-to-motion model (MLD) to generate local actions.
2. **Local Action Diffusion Guidance**: Utilizing an energy function to compute gradients of local actions, providing conditional guidance in the reverse diffusion process.
3. **Hierarchical Motion Diffusion Model**: Dividing the diffusion process into three semantic levels: motion-level, action-level, and detail-level.

### Key Designs

**Semantic graph parsing** maps a motion description to a hierarchical graph structure, containing three types of nodes (motion/action/detail) and twelve types of edges. For example, "a person jogs and looks around" is parsed into a global motion node, two action nodes ("jogs" and "looks"), and their respective attribute nodes.

**Local action guidance** is based on energy functions and score-matching theory. The reverse diffusion process is modified as:

$$\mathbf{z}_{t-1} = \tilde{\mathbf{z}}_{t-1} - \sum_{k=1}^K \lambda_t^k \nabla_{\mathbf{z}_t} \mathcal{E}(\mathbf{c}^k, \mathbf{z}_t)$$

where $\mathcal{E}$ is the energy function (using L2 distance in the latent space), and $\lambda_t^k$ is the guidance weight.

**Guidance weight estimation**: Uses GAT (Graph Attention Networks) to model the attention coefficients between action nodes and the global motion node in the semantic graph as guidance weights; users can also manually adjust the $\rho$ parameter to scale up or down the guidance intensity.

**Hierarchical diffusion**: Divides diffusion into three levels—the motion level provides a coarse initial value, the action level exerts local action guidance, and the detail level further refines the generation to conform to the original description.

### Loss & Training

The three levels are trained independently, and the total loss is: $\mathcal{L} = \mathcal{L}_M + \mathcal{L}_A + \mathcal{L}_S$

Each level is a standard diffusion denoising loss (MSE), accelerated with DDIM (50 steps).

## Key Experimental Results

### Main Results

Comparison with SOTA methods on the HumanML3D dataset:

| Method | R-Top3 ↑ | FID ↓ | MM-Dist ↓ | Diversity → | MModality ↑ |
|------|:-:|:-:|:-:|:-:|:-:|
| MDM | 0.611 | 0.544 | 5.566 | 9.559 | **2.799** |
| MLD | 0.772 | 0.473 | 3.196 | 9.724 | 2.413 |
| T2M-GPT | 0.775 | 0.116 | 3.118 | 9.761 | 1.856 |
| ReMoDiffuse | **0.795** | 0.103 | **2.974** | 9.018 | 1.795 |
| **GuidedMotion** | 0.788 | **0.057** | 3.040 | 9.864 | 2.473 |

GuidedMotion significantly leads on the FID metric with 0.057 (the second best being 0.103), and its multi-modality outperforms most methods.

### Ablation Study

Ablation of each module on HumanML3D:

| Motion Level | Action Level | Detail Level | Local Action Guidance | R-Top3 ↑ | FID ↓ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | | | | 0.760 | 0.186 |
| ✓ | ✓ | | | 0.771 | 0.133 |
| ✓ | ✓ | | ✓ | 0.778 | 0.119 |
| ✓ | ✓ | ✓ | | 0.769 | 0.107 |
| ✓ | ✓ | ✓ | ✓ | **0.788** | **0.057** |

Local action guidance brings significant improvements with or without the detail level.

Comparison on the complex motion subset ($\ge 3$ local actions and $\ge 150$ frames):

| Method | R-Top3 ↑ | FID ↓ |
|------|:-:|:-:|
| MLD | 0.710 | 0.783 |
| T2M-GPT | 0.712 | 0.314 |
| **GuidedMotion** | **0.732** | **0.144** |

The advantage on complex motions is more pronounced.

### Key Findings

- The local-to-global paradigm reduces the difficulty of directly generating complex global motions.
- By sampling different combinations of local actions, diverse motions that satisfy different user preferences can be generated.
- The guidance weight supports continuous adjustment, enabling fine-grained control over motion trajectories and poses.
- The method also achieves SOTA on the KIT dataset (FID 0.213, MModality 4.138).

## Highlights & Insights

- The **local-to-global** paradigm is more controllable than direct global generation, showing a particularly outstanding advantage for complex motions.
- The design of semantic graph parsing + GAT weight estimation provides the method with strong interpretability.
- Users can flexibly combine preferred local actions without repeatedly adjusting the text prompt.
- The three-stage strategy of hierarchical diffusion effectively balances generation stability and detail quality.

## Limitations & Future Work

- Semantic graph parsing relies on semantic role labeling tools, which may fail for non-standard or ambiguous descriptions.
- Local action sampling uses MLD to generate individually, whose upper performance bound is restricted by the base model.
- Guidance weight adjustment requires a certain level of user understanding, and the degree of automation could be further improved.

## Rating

⭐⭐⭐⭐ Innovative paradigm, elegant local-to-global concept, clear advantages in complex motion scenarios, and solid experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

</div>

<!-- RELATED:END -->
