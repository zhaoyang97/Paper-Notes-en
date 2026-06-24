---
title: >-
  [Paper Note] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models
description: >-
  [ECCV 2024][Image Generation] Proposes M2D2M, which generates multi-segment continuous human motion sequences based on discrete diffusion models, achieving smooth transitions between actions through dynamic transition probabilities and a Two-Phase Sampling (TPS) strategy without requiring additional multi-motion training data.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 4d8dc0428a18bd1a
---

# M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2407.14502](https://arxiv.org/abs/2407.14502)  
**Area**: Image Generation

## TL;DR

Proposes M2D2M, which generates multi-segment continuous human motion sequences based on discrete diffusion models, achieving smooth transitions between actions through dynamic transition probabilities and a Two-Phase Sampling (TPS) strategy without requiring additional multi-motion training data.

## Background & Motivation

Existing text-to-motion methods mainly focus on the generation of single-action sequences, but practical applications (such as storytelling, gaming, and simulation training) require generating multi-motion sequences containing a series of continuous actions. Existing multi-motion methods (such as the Handshake algorithm of PriorMDM, and the SLERP interpolation of TEACH) first generate each action independently and then connect them in post-processing, which often leads to:
- Abrupt transitions at action boundaries
- Degradation in the fidelity of individual actions
- Requirements for additional transition length hyperparameters

This paper proposes a unified generation scheme based on discrete diffusion models, **directly generating multi-motion sequences using a model trained on single motions**, without additional training or post-processing.

## Method

### Overall Architecture

M2D2M consists of three modules:
1. **Motion VQ-VAE**: Encodes motion sequences into discrete tokens
2. **Denoising Transformer**: Learns conditional denoising under a discrete diffusion framework
3. **Two-Phase Sampling (TPS)**: Joint sampling to establish a rough outline $\rightarrow$ independent sampling to fine-tune each motion segment

### Key Designs

**Dynamic Transition Probabilities**: Improves the uniform transition probability in VQ-Diffusion by dynamically adjusting it according to the distance between codebook tokens:

$$\beta(t, d) = (1 - \gamma_t - \alpha_t) \cdot \text{softmax}_d\left(\eta \cdot \frac{t}{T} \cdot \frac{d}{K}\right)$$

In the early stages of diffusion (large $t$), distant tokens are preferentially explored to promote diversity, and in later stages, it gradually converges to a uniform distribution for precise convergence. This exploration-exploitation strategy is crucial for pattern blending at multi-motion boundaries.

**Two-Phase Sampling (TPS)**:
- **Joint Sampling Phase** (steps $T \rightarrow T_s+1$): Merges mask tokens of all actions and jointly denoises them using the denoising Transformer, allowing tokens of different actions to influence each other through the self-attention mechanism, ensuring smooth transitions.
- **Independent Sampling Phase** (steps $T_s \rightarrow 1$): Denoises each action independently to align with the corresponding text description, maintaining individual fidelity.

Key advantage: No multi-motion training data is required; sequences can be generated directly using a model trained on single motions.

**New Evaluation Metric Jerk**: Measures the smoothness of multi-motion sequences at action boundaries:

$$Jerk = \sum_p \ln \frac{1}{v_{p,\text{peak}}^2} \int_{t_1}^{t_2} \left\| \frac{d}{dt} \mathbf{a}_p(t) \right\|_2^2 dt$$

Introduces Jerk to multi-motion generation evaluation for the first time.

### Loss & Training

Standard discrete diffusion objective: Variational Lower Bound (VLB) + denoising cross-entropy loss:

$$\mathcal{L} = \mathcal{L}_{\text{vlb}} + \lambda \mathbb{E}_{z_t \sim q(z_t|z_0)} [-\log p_\theta(z_0 | z_t, y)]$$

Utilizes a CLIP text encoder, relative position encoding, and classifier-free guidance (10% unconditional dropout rate).

## Key Experimental Results

### Main Results

HumanML3D multi-motion generation (N=4 actions):

| Method | R-Top3 ↑ | FID ↓ | MMdist ↓ | Jerk → |
|------|:-:|:-:|:-:|:-:|
| GT (Single) | 0.791 | 0.002 | 2.707 | 1.192 |
| GT (Concat) | — | — | — | 1.371 |
| PriorMDM | 0.586 | 0.832 | 5.901 | 0.476 |
| T2M-GPT | 0.719 | 0.342 | 3.512 | 1.321 |
| **M2D2M** | **0.733** | **0.253** | **3.165** | **1.238** |

M2D2M leads significantly in all individual motion metrics, and its Jerk value is close to real single motions (1.238 vs 1.192), which is far superior to simple concatenation (1.371). The Jerk of PriorMDM is only 0.476, indicating that over-smoothing leads to a lack of realism in motion.

HumanML3D single-motion generation comparison (compared with 13 methods, partial results):

| Method | R-Top3 ↑ | FID ↓ | MM-Dist ↓ | MModality ↑ |
|------|:-:|:-:|:-:|:-:|
| MotionGPT | 0.778 | 0.232 | 3.096 | 2.008 |
| ReMoDiffuse | **0.795** | 0.103 | **2.974** | 1.795 |
| **M2D2M** | 0.788 | **0.057** | 3.040 | 2.473 |

### Ablation Study

KIT-ML multi-motion generation (N=4):

| Method | R-Top3 ↑ | FID ↓ | Jerk → |
|------|:-:|:-:|:-:|
| PriorMDM | 0.292 | 3.311 | 0.594 |
| T2M-GPT | 0.667 | 0.907 | 1.388 |
| **M2D2M** | **0.711** | **0.817** | **1.351** |

Joint effect of TPS and dynamic transition probabilities (ablation studies show that the two work best in synergy, with limited effect when used alone).

### Key Findings

- TPS is a single-stage multi-motion generation algorithm that does not require completed independent motions or transition length hyperparameters.
- Dynamic transition probabilities promoting pattern mixture in the early stages of diffusion are crucial for transitions at multi-motion boundaries.
- PriorMDM's Handshake algorithm over-smooths boundaries (Jerk too low), losing detailed motion characteristics.
- Relative position encoding allows the model to extrapolate to long sequences unseen during training.

## Highlights & Insights

- **Zero additional training cost for multi-motion generation**: Directly generates multi-motions using models trained on single motions, solving the scarcity of multi-motion labeled data.
- **Introduction of the Jerk metric**: Fills the gap in evaluating transition smoothness in multi-motion generation.
- **Exploration-exploitation strategy of dynamic transition probabilities**: Encouraging distant token mixtures in the early stages of diffusion provides theoretical support for the fusion of different action patterns at multi-motion boundaries.
- The joint $\rightarrow$ independent two-phase design of TPS is clean and elegant.

## Limitations & Future Work

- The number of steps $T_s$ in the joint sampling phase is a hyperparameter and needs to be adjusted manually.
- Discrete representation based on VQ-VAE may introduce quantization errors.
- Lacks sufficient comparison with concurrent work such as FineMoGen.
- Evaluation of diversity in multi-motion transition areas depends on randomly combined test sets, which may introduce bias.

## Rating

⭐⭐⭐⭐ A new solution for multi-motion generation from the perspective of discrete diffusion, with novel designs for dynamic transition probabilities and TPS, as well as a significant contribution from the Jerk metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
