---
title: >-
  [Paper Note] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning
description: >-
  [ECCV 2024][Image Generation][personalized T2I] This work models personalized T2I generation as a Deterministic Policy Gradient (DPG) framework—with the diffusion model acting as the policy and the denoising steps as actions. By introducing a "look forward" mechanism to capture long-term visual consistency and a DINO similarity reward, it improves the DINO score from 0.694 to 0.738 (+6.3%) and CLIP-I from 0.762 to 0.797 (+4.6%) on the DreamBooth benchmark.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "personalized T2I"
  - "reinforcement learning"
  - "deterministic policy gradient"
  - "look forward"
  - "DINO reward"
date: 2026-05-08
content_hash: 8024037c5fd208ef
---

# Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning

**Conference**: ECCV 2024  
**arXiv**: [2407.06642](https://arxiv.org/abs/2407.06642)  
**Code**: [GitHub](https://github.com/wfanyue/DPG-T2I-Personalization)  
**Area**: Image Generation / Personalized Text-to-Image  
**Keywords**: personalized T2I, reinforcement learning, deterministic policy gradient, look forward, DINO reward

## TL;DR

This work models personalized T2I generation as a Deterministic Policy Gradient (DPG) framework—with the diffusion model acting as the policy and the denoising steps as actions. By introducing a "look forward" mechanism to capture long-term visual consistency and a DINO similarity reward, it improves the DINO score from 0.694 to 0.738 (+6.3%) and CLIP-I from 0.762 to 0.797 (+4.6%) on the DreamBooth benchmark.

## Background & Motivation

**Background**: Personalized T2I (Textual Inversion, DreamBooth, Custom Diffusion) embeds personal concepts (pets, friends, etc.) by fine-tuning diffusion models, but generally suffers from the **loss of visual details**—the color, texture, and structure of generated objects are often inconsistent with the reference images.

**Limitations of Prior Work**: (1) Existing methods employ simple **step-by-step reconstruction loss** ($\epsilon$-prediction), which cannot directly optimize the visual consistency of the final generation output; (2) Different denoising timesteps focus on different features (early stages on structure, later stages on details), but step-by-step reconstruction loss is blind to this; (3) RL methods for general T2I (DPOK, DRaFT) rely on human preference or aesthetic rewards, whereas personalized scenarios typically have only 4~6 reference images, making it difficult to train a dedicated reward model.

**Key Challenge**: Step-by-step reconstruction loss cannot capture the long-term visual consistency of the diffusion process, especially the correspondence in structure and detail between the final generated image and the reference images.

**Goal**: To design a flexible RL framework utilizing various differentiable/non-differentiable objective functions to improve the visual fidelity of personalized T2I.

**Key Insight**: Treating the diffusion model as a deterministic policy and introducing a Q-function to learn cumulative rewards, enabling a "look forward" mechanism to the final generated output.

**Core Idea**: Learn the cumulative reward of looking forward from the current timestep to $\hat{x}_{0,t}$: $\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\|\hat{z}_t - z_t\|^2$ through the Q-function in the DPG framework, combined with DINO similarity rewards to directly optimize visual consistency.

## Method

### Overall Architecture

Reference image set → Diffusion process noise injection → U-Net policy noise prediction → "Look forward" to obtain $\hat{x}_{0,t}$ → Decode to image → DINO encoder feature extraction → Reward calculation → Q-function cumulative reward learning → Gradient backpropagation to optimize U-Net policy.

### Key Designs

1. **Deterministic Policy Gradient (DPG) Framework**

    - State: $\{x_t, t, \tau(y)\}$ (latent state + timestep + text condition)
    - Action: $\hat{z}_t = \epsilon_\theta(x_t, t, \tau(y))$ (predicted noise)
    - Policy: Diffusion model $\epsilon_\theta$
    - Q-function $Q_\phi$ estimates cumulative rewards, with the optimization objective: $\max_\theta \mathbb{E}[Q_\phi(x_t, \epsilon_\theta(x_t, t, \tau(y)))]$
    - Design Motivation: Q-learning in RL naturally supports long-term cumulative rewards, offsetting the myopic nature of step-by-step reconstruction loss.

2. **"Look Forward" Mechanism**

    - Predict the final output at any timestep $t$: $\hat{x}_{0,t} = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\hat{z}_t)$
    - Rewrite reward as $\|\hat{x}_{0,t} - x_0\|^2 = \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\|\hat{z}_t - z_t\|^2$—a reconstruction loss with timestep-dependent weighting.
    - Q-function learns the cumulative reward: $Q_\phi(x_t, \cdot) = \mathcal{L}(x_t, \cdot) + \gamma Q_\phi(x_{t-1}, \cdot)$
    - Design Motivation: The "look forward" results at different timesteps reflect different levels of features (early steps = structure, later steps = details). The Q-function implicitly learns to focus on different levels.

3. **DINO Reward**

    - Decode $\hat{x}_{0,t}$ into image $\hat{I} = \mathcal{D}(\hat{x}_{0,t})$ and extract feature $\hat{\kappa}$ using a DINO encoder.
    - Reward $r(x_t) = -(1 - \hat{\kappa} \cdot \kappa)$ (cosine distance from the DINO features of the reference image).
    - Combined with reconstruction reward: $\nabla_\theta \frac{1}{B}\sum_B(\lambda Q_\phi + (-\|\epsilon - \epsilon_\theta\|^2))$
    - Design Motivation: DINO excels at capturing unique visual characteristics of objects, making it a more suitable personalization reward signal than human preferences.

### Loss & Training

Alternating optimization of the Q-function and U-Net (Algorithm 1). Based on the DreamBooth baseline, using Stable Diffusion V1.4, the parameter size of the Q-function is only 0.26M (vs. U-Net's 859.4M). Training was performed on a 32G V100 GPU.

## Key Experimental Results

### Main Results

Comparison on the DreamBooth benchmark (30 concepts, 25 prompts):

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|-------|---------|---------|
| Custom Diffusion | 0.649 | 0.712 | 0.321 |
| Custom Diffusion + DINO reward | 0.640 | 0.715 | 0.320 |
| Custom Diffusion + Look Forward | 0.669 | 0.728 | 0.322 |
| DreamBooth | 0.694 | 0.762 | 0.282 |
| DreamBooth + DINO reward | 0.723 | 0.783 | 0.270 |
| **DreamBooth + Look Forward** | **0.738** | **0.797** | 0.269 |

Comparison on the Custom benchmark:

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|-------|---------|---------|
| DreamBooth | 0.640 | 0.737 | 0.309 |
| DreamBooth + Look Forward | **0.680** | **0.773** | 0.303 |
| DreamBooth + DINO reward | 0.653 | 0.753 | 0.310 |

### Ablation Study

| Ablation Item | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|--------|-------|---------|---------|
| DreamBooth Baseline | 0.644 | 0.707 | 0.239 |
| w/o discount factor $\gamma$ | 0.727 | 0.761 | 0.209 |
| $\gamma = 0.9986$ | 0.704 | 0.743 | 0.213 |
| $\lambda = 0.1$ (DINO weight) | 0.704 | 0.743 | 0.213 |
| $\lambda = 1$ (DINO weight) | 0.727 | 0.746 | 0.211 |

User Study:

| Preference | Ours | DreamBooth | Similar |
|------|------|------------|------|
| Image Fidelity | **55.1%** | 12.0% | 32.9% |
| Text Fidelity | 19.6% | 20.4% | 60.0% |

### Key Findings

- Look Forward brings the most significant improvement: DINO increases from 0.694 to 0.738 (+6.3%), and CLIP-I from 0.762 to 0.797 (+4.6%).
- DINO reward improves DINO score by 4.2% (0.694 to 0.723) over the DreamBooth baseline.
- An inherent trade-off exists between image fidelity and text fidelity, but the drop in text fidelity is marginal (0.282 to 0.269).
- The Q-network parameter size is extremely small (0.26M), adding almost no computational overhead.
- 55.1% of users prefer the image fidelity of this method (vs. 12.0% for DreamBooth).

## Highlights & Insights

- The DPG framework elegantly maps the diffusion process into an RL problem, where the Q-function learns long-term cumulative rewards.
- The derivation of the "look forward" mechanism is concise: it is equivalent to a timestep-weighted reconstruction loss, but achieved cumulatively via the Q-function.
- The framework is highly flexible: any differentiable or non-differentiable reward function can be plugged in (DINO is just one realization).
- The Q-network has only 0.26M parameters, leading to a lightweight, low-cost implementation.

## Limitations & Future Work

- In some scenarios, overemphasizing visual fidelity might lead to a decline in text alignment.
- Only DINO is used as a reward example, leaving other stronger visual similarity metrics (such as DINOv2, SSIM, etc.) unexplored.
- Based on the DreamBooth baseline, performance is bound by the generation capabilities and text encoder of Stable Diffusion V1.4.
- Direct comparison with contemporaneous RL-based T2I methods (like DRaFT) in personalized scenarios is missing.

## Related Work & Insights

- **vs. DreamBooth**: DreamBooth only uses step-by-step reconstruction loss, whereas this work introduces long-term visual consistency via the DPG framework.
- **vs. DRaFT**: DRaFT propagates gradients directly based on differentiable rewards, whereas this work supports more flexible rewards via the Q-function.
- **vs. DPOK**: DPOK uses stochastic policy gradient with KL regularization for general T2I, whereas this work uses deterministic policy gradient for personalization.
- **vs. Custom Diffusion**: Custom Diffusion only fine-tunes cross-attention, leading to weaker visual fidelity; incorporating Look Forward can improve this.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling the diffusion process as DPG provides an elegant theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ DreamBooth + Custom benchmarks + user study + ablation studies.
- Writing Quality: ⭐⭐⭐ Extensive mathematical derivations in the methodology section, but overall clear.
- Value: ⭐⭐⭐⭐ DINO improved by 6.3%, 55.1% user preference, showing significant practical efficacy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)
- [\[ECCV 2024\] Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models](lego_learning_to_disentangle_and_invert_personalized_concepts_beyond_object_appe.md)
- [\[ICLR 2026\] RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning](../../ICLR2026/image_generation/reprompt_reasoning-augmented_reprompting_for_text-to-image_generation_via_reinfo.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)

</div>

<!-- RELATED:END -->
