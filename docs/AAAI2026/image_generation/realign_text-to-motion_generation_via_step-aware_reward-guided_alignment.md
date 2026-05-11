---
title: >-
  [Paper Note] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment
description: >-
  [AAAI 2026][Image Generation][Text-to-Motion Generation] This paper proposes ReAlign (Reward-guided sampling Alignment), which employs a step-aware reward model and a reward-guided sampling strategy to dynamically steer…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Text-to-Motion Generation"
  - "Diffusion Models"
  - "Reward-Guided Sampling"
  - "Text-Motion Alignment"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: d4d379a4bad428fd
---

# ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.19217](https://arxiv.org/abs/2511.19217)
**Code**: [https://wengwanjiang.github.io/ReAlign-page](https://wengwanjiang.github.io/ReAlign-page)
**Area**: Image Generation
**Keywords**: Text-to-Motion Generation, Diffusion Models, Reward-Guided Sampling, Text-Motion Alignment, Plug-and-Play

## TL;DR

This paper proposes ReAlign (Reward-guided sampling Alignment), which employs a step-aware reward model and a reward-guided sampling strategy to dynamically steer sampling trajectories toward distributions with high text-motion alignment during diffusion inference, significantly improving the generation quality of various motion generation methods without fine-tuning any diffusion model. Using MLD as a baseline, R@1 improves by 17.9% and FID improves by 58.8%.

## Background & Motivation

### State of the Field

Text-to-Motion Generation aims to synthesize 3D human motions from natural language descriptions, with broad applications in gaming, film, and robotics. Diffusion models (e.g., MDM, MLD, MotionDiffuse) have become the dominant paradigm, capable of producing diverse and realistic motions.

### Limitations of Prior Work

**Insufficient text-motion alignment**: Diffusion models rely on CLIP to encode text embeddings, but CLIP is trained on text-image pairs and is inherently ill-suited for capturing semantic alignment between text and motion. As a result, generated motions frequently fail to match the input description (e.g., "walk to the right" produces "walk to the left").

**High probability density ≠ semantic consistency**: During diffusion sampling, the score function $\nabla \log p_t(\mathbf{x})$ guides samples toward high-density regions, which are not necessarily semantically consistent with the text. Models tend to generate motions that are "probabilistically likely but semantically off."

**Limitations of prior alignment methods**: RL-based approaches (ReinDiffuse, MotionRL, SoPo) focus on fine-tuning generative models to align with preferences or improve quality, but they cannot handle noisy motion inputs, and the alignment problem should be addressed during denoising rather than post-hoc correction.

**Disconnection between motion generation and retrieval**: The two related tasks are typically studied independently, with little exploration of unified models or mutual enhancement.

### Root Cause

The diffusion sampling distribution $p_t(\mathbf{x})$ prioritizes high probability density at the expense of semantic fidelity. The core challenge is how to make sampling trajectories simultaneously respect both probability density and semantic alignment without modifying the diffusion model itself.

### Starting Point

**Core Idea**: Construct an ideal sampling distribution $p_t^I(\mathbf{x}|c) = p_t(\mathbf{x}|c) \cdot p_t^r(\mathbf{x}|c) / Z(c)$ by multiplying the original sampling distribution with a reward-based alignment distribution. A step-aware reward model is trained to estimate the reward distribution, and reward gradients are injected into the reverse SDE at inference time to guide sampling toward joint optimality in both semantics and motion quality. The method is fully plug-and-play and requires no fine-tuning of any diffusion model.

## Method

### Overall Architecture

ReAlign consists of two core components:

1. **Step-Aware Reward Model**: Evaluates text-motion alignment under varying noise levels across different denoising steps.
2. **Reward-Guided Sampling Strategy**: Injects reward gradients into the diffusion denoising process.

### Key Designs

#### 1. **Step-Aware Reward Model**

**Function**: Accurately evaluates the alignment between noisy motions and text at different denoising steps (different noise levels).

**Problem**: Existing alignment models (e.g., TMR, LaMP) assume clean, noise-free motion inputs and cannot handle noisy motions during denoising—precisely the stage where guidance is most needed.

**Mechanism**: A timestep token $[e_t]$ is introduced and concatenated with the motion frame sequence $[x_t^1, x_t^2, \ldots, x_t^N]$ to form $[e_t, x_t^1, x_t^2, \ldots, x_t^N]$, which is fed into a Transformer encoder. This enables the model to adapt to varying noise levels while processing motion dynamics.

During training, noise corresponding to different steps $t$ is added to motions, and two complementary losses are used:
$$\mathcal{L}_{RM}(\varphi; \mathbf{x}_t, c) = \mathcal{L}_C(\varphi; \mathbf{x}_t, c) + \mathcal{L}_R(\varphi; \mathbf{x}_t, c)$$

- $\mathcal{L}_C$: Contrastive loss, ensuring accurate text-motion retrieval.
- $\mathcal{L}_R$: Representation loss, learning meaningful motion embeddings.

After training, the reward is defined as cosine similarity: $R_\varphi(\mathbf{x}, c) = \cos(\mathbf{z_x}, \mathbf{z}_c)$.

**Design Motivation**: The denoising process transitions from pure noise to clean motion with continuously changing noise levels. Conventional alignment models fail entirely at high-noise steps. The timestep token informs the model of the current noise level, enabling accurate alignment evaluation at every step.

#### 2. **Motion-to-Motion Reward**

**Function**: Ensures that generated motions conform to real-world motion patterns, compensating for the ambiguity of text descriptions.

**Mechanism**: The step-aware reward model retrieves a reference motion from the training set that best matches the text condition:
$$\mathbf{x}^c = \arg\max_{\mathbf{x} \in \mathcal{D}_{tr}} R_\varphi(\mathbf{x}, c)$$

The cosine similarity between the generated motion's embedding and the reference motion's embedding is then computed:
$$R_m(\mathbf{x}_t, c) = \cos(\mathbf{z_x}, \mathbf{z_{x^c}})$$

The reference motion $\mathbf{x}^c$ serves as a dynamic anchor, ensuring the generated motion remains faithful to the real motion pattern implied by the text.

**Dual alignment reward**:
$$R(\mathbf{x}_t, c) = \mu R_\varphi(\mathbf{x}_t, c) + \eta R_m(\mathbf{x}_t, c)$$

**Reward distribution**: $p_t^r(\mathbf{x}_t|c) = \exp(R(\mathbf{x}_t, c)) / Z^r(c)$

#### 3. **Reward-Guided Sampling**

**Function**: Integrates the reward distribution into the diffusion reverse process to redirect the sampling trajectory toward the ideal distribution.

**Theoretical derivation** (Theorems 1–3):

Defining the ideal distribution $p_t^I(\mathbf{x}|c) = p_t(\mathbf{x}|c) \cdot p_t^r(\mathbf{x}|c) / Z(c)$ and substituting into the reverse SDE yields:

$$\mathbf{dx} = [\mathbf{f}(\mathbf{x},t) - g(t)^2 \nabla(\log p_t(\mathbf{x}|c) + \log p_t^r(\mathbf{x}|c))] dt + g(t) d\mathbf{w}$$

Discretized under the DDPM framework (Theorem 3):
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\bar{\mathbf{x}}_{t-1} + \sqrt{\beta_t}\epsilon\right) + \frac{\beta_t}{\sqrt{\alpha_t}} \nabla R(\mathbf{x}_t, c)$$

For sampling stability, the weight $\frac{\beta_t}{\sqrt{\alpha_t}}$ is removed, yielding the final practical formula:
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\bar{\mathbf{x}}_{t-1} + \sqrt{\beta_t}\epsilon\right) + \nabla R(\mathbf{x}_t, c)$$

The reward gradient guides samples simultaneously toward high-density and high-alignment regions at each denoising step.

### Loss & Training

**Reward model training**:
- Architecture: SkipTransformer (9 layers, 4 heads, latent dimension 256)
- Maximum timestep 1000, noisy motion probability 0.5
- AdamW, lr $10^{-4}$, batch size 512
- Training follows the TMR framework

**Inference**: The reward model is plug-and-play and used in conjunction with CFG, requiring zero additional training.

## Key Experimental Results

### Main Results

**Text-to-motion generation on HumanML3D** (ReAlign applied to MLD and MLD++):

| Method | R@1 ↑ | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity → |
|--------|-------|-------|-------|-----------|-------------|
| Real | 0.511 | 0.797 | 0.002 | 2.974 | 9.503 |
| MLD | 0.481 | 0.772 | 0.473 | 3.196 | 9.724 |
| **MLD + ReAlign** | **0.567 (+17.9%)** | **0.848 (+9.8%)** | **0.195 (+58.8%)** | **2.704 (+15.4%)** | 9.474 |
| MLD++ | 0.548 | 0.829 | 0.073 | 2.810 | 9.658 |
| **MLD++ + ReAlign** | **0.572 (+4.4%)** | **0.852 (+2.8%)** | **0.055 (+24.7%)** | **2.648 (+5.8%)** | 9.478 |

**KIT-ML dataset** (ReAlign applied to MDM):

| Method | R@1 ↑ | R@3 ↑ | FID ↓ | MM Dist ↓ |
|--------|-------|-------|-------|-----------|
| MDM | 0.403 | 0.731 | 0.497 | 3.096 |
| **MDM + ReAlign** | **0.451 (+11.9%)** | **0.784 (+7.3%)** | **0.276 (+44.5%)** | **2.775 (+10.4%)** |

### Ablation Study

**Component ablation on HumanML3D (MLD baseline)**:

| T2M Reward | M2M Reward | Step-Aware | R@1 ↑ | FID ↓ | MM Dist ↓ |
|------------|------------|------------|-------|-------|-----------|
| ✗ | ✗ | ✗ | 0.481 | 0.473 | 3.196 |
| ✓ | ✗ | ✗ | 0.556 | 0.213 | 2.761 |
| ✗ | ✓ | ✗ | 0.517 | 0.205 | 2.932 |
| ✓ | ✓ | ✗ | 0.556 | 0.199 | 2.750 |
| ✓ | ✗ | ✓ | 0.568 | 0.212 | 2.714 |
| **✓** | **✓** | **✓** | **0.567** | **0.195** | **2.704** |

**Plug-and-play capability validation** (applied to 5 different diffusion models):

| Baseline | R@1 Gain | FID Gain | MM Dist Gain |
|----------|----------|----------|--------------|
| Mo.Diffuse | +8.8% | +41.3% | +9.9% |
| MDM | +3.3% | +33.5% | +6.0% |
| MLD | +17.9% | +58.8% | +15.4% |
| MotionLCM | +7.6% | +10.2% | +7.1% |
| MLD++ | +4.4% | +24.7% | +5.8% |

### Key Findings

1. **T2M reward is the primary driver**: Using only the T2M reward improves R@1 from 0.481 to 0.556 (+15.6%) and reduces FID from 0.473 to 0.213 (+54.9%).
2. **Step-aware training is indispensable**: The comparison in Figure 3 shows that a reward model without step-awareness fails at high-noise steps, while the step-aware version significantly outperforms at all denoising steps.
3. **M2M reward provides additional realism gain**: Its standalone effect is limited (constrained by text-motion retrieval accuracy), but it further reduces FID when combined with T2M reward and step-awareness.
4. **Generalizability confirmed**: Consistent improvements are observed across 5 different baselines, with FID improvements ranging from 10.2% to 58.8%, demonstrating the universality of the plug-and-play design.
5. **Improvements also observed on motion retrieval**: R@1 reaches 67.59% (T2M retrieval) and 68.94% (M2T retrieval) on HumanML3D, surpassing TMR and LaMP.

## Highlights & Insights

1. **Theoretical elegance**: The mathematical derivation proceeds cleanly from the definition of $p_t^I$ through SDE derivation to DDPM discretization (Theorems 1–3), with reward gradients naturally integrated into the denoising process.
2. **Core value of plug-and-play design**: No diffusion model is modified; reward guidance is added solely at inference time. Substantial improvements are achieved across all baseline methods, validating the approach's generality.
3. **Unified generation and retrieval**: The step-aware reward model simultaneously serves both generation guidance and motion retrieval, establishing a bridge between the two tasks.
4. **Effective use of noise augmentation**: Incorporating noisy motions during reward model training not only adapts the model to the noisy intermediate states of denoising, but also acts as data augmentation, enhancing the retrieval model's ability to discriminate subtle motion differences.
5. **First step-aware design in motion generation**: Injecting timestep information as a token into the alignment model is conceptually simple yet empirically effective.

## Limitations & Future Work

1. **Increased inference overhead**: Each denoising step requires an additional forward pass through the reward model plus gradient computation, multiplying inference time.
2. **Quality bottleneck of retrieved motions**: The M2M reward depends on reference motions retrieved from the training set; its effectiveness is limited when the training set lacks diversity.
3. **Diversity reduction**: Diversity slightly decreases in some settings, reflecting the constraint imposed by reward guidance on variability. While the authors argue this reflects the elimination of incorrect motions through better alignment, excessively strong guidance may lead to mode collapse.
4. **Selection of reward weights $\mu$ and $\eta$**: The ablation does not thoroughly explore optimal hyperparameter search, and manual tuning may not generalize broadly.
5. **Validation limited to the DDPM framework**: Performance under DDIM and ODE samplers has not been verified, and the theoretical derivation's compatibility with these samplers remains to be explored.

## Related Work & Insights

- **TMR/LaMP**: Foundational text-motion alignment models that assume clean inputs; ReAlign's step-aware design extends their applicability to noisy intermediate states.
- **Classifier Guidance** (Dhariwal et al.): Uses classifier gradients to guide generation in image diffusion; ReAlign generalizes this idea to alignment rewards in the motion domain.
- **ReinDiffuse/MotionRL/SoPo**: Fine-tune diffusion models via RL to align with preferences, but require training. ReAlign's training-free approach offers greater flexibility.
- **EnergyMoGen**: A concurrent work also addressing alignment in motion generation; ReAlign substantially outperforms it on R@1 (0.567 vs. 0.526).
- The reward-guided sampling paradigm is generalizable to other text-conditioned generation tasks (3D generation, music generation, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of step-aware reward and inference-time guidance is original, with a complete theoretical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two datasets, five baselines, detailed ablations, and retrieval task validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivation is clear; motivation is well articulated.
- **Value**: ⭐⭐⭐⭐⭐ — The plug-and-play design is highly practical with significant performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](../../CVPR2026/image_generation/bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[AAAI 2026\] MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [\[ICLR 2026\] Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution](../../ICLR2026/image_generation/step-aware_residual-guided_diffusion_for_eeg_spatial_super-resolution.md)
- [\[ICLR 2026\] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](../../ICLR2026/image_generation/event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](../../ICCV2025/image_generation/enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)

</div>

<!-- RELATED:END -->
