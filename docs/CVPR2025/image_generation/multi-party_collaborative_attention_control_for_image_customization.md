---
title: >-
  [Paper Note] Multi-party Collaborative Attention Control for Image Customization
description: >-
  [Image Generation] Proposes MCA-Ctrl, a tuning-free image customization method that achieves high-quality text/image-conditioned subject editing and generation through collaborative self-attention control (SAGI + SALQ) over a three-way parallel diffusion process. A Subject Localization Module is also introduced to address subject leakage and confusion in complex scenarios.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: bed1615f05a8156b
---

# Multi-party Collaborative Attention Control for Image Customization

## TL;DR

Proposes MCA-Ctrl, a tuning-free image customization method that achieves high-quality text/image-conditioned subject editing and generation through collaborative self-attention control (SAGI + SALQ) over a three-way parallel diffusion process. A Subject Localization Module is also introduced to address subject leakage and confusion in complex scenarios.

## Background & Motivation

Existing image customization methods face four major limitations:
1. **Single-modality constraints**: Most methods only accept either image or text conditions and cannot support both simultaneously.
2. **Subject leakage/confusion**: In complex visual scenarios (e.g., occlusion, multiple objects, and foreground-background similarities), inaccurate high-response areas in the model lead to feature leakage.
3. **Background inconsistency**: Under image-conditioned customization, there is often a large discrepancy between the generated background and the source image.
4. **High computational cost**: Inversion-based methods (e.g., DreamBooth, Textual Inversion) require expensive fine-tuning for each subject.

Existing zero-shot methods (e.g., IP-Adapter, BLIP-Diffusion) reduce costs by training multimodal encoders with alignment projection layers, but they still incur significant storage and training overheads and underperform in complex scenarios.

**Core Motivation**: Explore a training-free customization method that is compatible with both text and image conditions, incurs low computational cost, and delivers high quality.

## Method

### Overall Architecture

Based on Stable Diffusion v1.5, MCA-Ctrl manipulates the self-attention layers of a three-way parallel diffusion process to control target image generation:
- **Subject diffusion process** $\mathcal{B}_{sub}$: Performs DDIM inversion on the subject image to obtain initial features.
- **Condition diffusion process** $\mathcal{B}_{con}$: Receives either a conditioning image (processed via DDIM inversion) or text (initialized with random Gaussian noise).
- **Target diffusion process** $\mathcal{B}_{tgt}$: Shares the initial features of $\mathcal{B}_{con}$ to generate the target image.

The three-way parallel process is implemented via a single inference pass with a batch size of 3, **introducing no additional computational overhead**. It supports three tasks: subject generation (text-driven), subject replacement, and subject addition (image-driven).

### Key Designs

#### 1. Self-Attention Local Query (SALQ)

The target image uses its query $Q_T$ to query the key/value pairs of the subject and condition separately:
- For the subject, it queries only the **foreground region** (filtered by mask $M_S$) to acquire appearance features.
- For the condition, it queries only the **background region** (filtered by mask $M_C$) to acquire layout and background content.
- The two types of features are fused via mask-weighted summation: $\mathcal{F}^*_{T} = M_C \cdot \mathcal{F}^Q_{T,C} + (1-M_C) \cdot \mathcal{F}^Q_{T,S}$

**Execution is recommended to start from the early steps of the UNet decoder**, where the layout has already been initially formed.

#### 2. Self-Attention Global Injection (SAGI)

Directly **injects** the respective attention features of the subject and condition into the target process:
- The subject's original self-attention features are filtered via mask to extract the subject foreground features $\mathcal{F}^I_S$.
- The condition's original self-attention features are filtered via mask to extract the background features $\mathcal{F}^I_C$.
- Global injection is achieved by replacing the feature output of the target process: $\mathcal{F}^*_T = M_C \cdot \mathcal{F}^I_C + (1-M_C) \cdot \mathcal{F}^I_S$

**SAGI is executed during the early denoising steps** (the stage dominated by semantic information) and alternates with SALQ, ensuring their execution intervals do not overlap.

#### 3. Subject Localization Module (SLM)

Composed of Grounding DINO (detection) + SAM (segmentation), this module processes multimodal instructions:
- Inputs the subject image + text description, and outputs the subject binary mask $M^s_C$.
- Inputs the condition image + text description, and outputs the editable region mask $M_S$.
- Dilates $M^s_C$ with a 3×3 kernel to obtain $M_C$, ensuring sufficient transition space for the edited region.

### Loss & Training

MCA-Ctrl is an **inference-time method** (tuning-free) and does not involve training loss. The underlying model utilizes the standard diffusion model objective:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\epsilon} \| \epsilon_t - \epsilon_\theta(z_t, t, P) \|^2$$

## Key Experimental Results

### Main Results

**Subject Swapping（DreamEditBench）**:

| Method | DINOsub↑ | DINOback↑ | CLIP-Isub↑ | CLIP-Iback↑ | ImageReward↑ |
|------|----------|-----------|------------|-------------|-------------|
| DreamBooth | 0.640 | 0.427 | 0.811 | 0.736 | -1.171 |
| BLIP-Diffusion | 0.616 | 0.639 | 0.801 | 0.825 | 0.219 |
| PHOTOSWAP | 0.631 | 0.607 | 0.789 | 0.798 | -0.198 |
| **Ours (Specified)** | **0.643** | **0.678** | **0.811** | **0.868** | **0.321** |

**Subject Generation（DreamBench）**:

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ | ImageReward↑ |
|------|-------|---------|---------|-------------|
| DreamBooth | 0.668 | 0.843 | 0.306 | 0.384 |
| BLIP-Diffusion | 0.670 | 0.825 | 0.302 | 0.183 |
| **Ours (Specified)** | **0.672** | **0.844** | **0.306** | **0.413** |

### Ablation Study

| Configuration | DINOsub↑ | DINOback↑ | ImageReward↑ |
|------|----------|-----------|-------------|
| Full (Uniform) | 0.633 | 0.668 | 0.273 |
| w/o SALQ | 0.424↓ | 0.749↑ | 0.245↓ |
| w/o SAGI | 0.590↓ | 0.685↑ | 0.272↓ |
| w/o SLM | 0.491↓ | 0.824↑ | 0.191↓ |
| Reverse execution order | 0.459↓ | 0.555↓ | 0.108↓ |

### Key Findings

- **SALQ is the core component**: Removing it leads to a 21 percentage point drop in DINOsub, emphasizing its critical role in ensuring subject consistency.
- **SAGI enhances detailed realism**: It corrects feature confusion caused by SALQ (e.g., orange-colored confusion around a cat's mouth).
- **SLM is indispensable in complex scenarios**: It addresses four types of complex visual scenarios, including object interactions, occlusions, and similarities among multiple objects.
- **Execution order is crucial**: Reversing the order of SAGI/SALQ leads to a substantial decline in all metrics.
- **Human evaluation total score reaches 2.73**, outperforming BLIP-Diffusion (2.60) and IP-Adapter (2.63).

## Highlights & Insights

1. **Training-free, high-quality customization**: Handled via attention manipulation with no fine-tuning required, executable in a single inference pass with a batch size of 3.
2. **Elegant complementary design of SAGI+SALQ**: SALQ conducts local content querying (to extract appearance), while SAGI performs global feature injection (to enhance details and mitigate confusion). Their execution intervals do not overlap.
3. **Universal and plug-and-play SLM module**: Leverages the open-world capability of Grounding DINO and SAM without being restricted to specific datasets.
4. **Unified framework for three customization tasks**: Subject generation, swapping, and addition are integrated into a single framework, requiring adjustments to only a few hyperparameters (execution steps and layers).

## Limitations & Future Work

1. **Constrained by base model capability**: When the subject contains fine-grained features (such as text), Stable Diffusion v1.5 cannot accurately reproduce them.
2. **Limitations in color modification**: Color edits might only affect localized areas of the subject rather than the whole.
3. **Requirement for manual masks or text descriptions**: The SLM relies on user-provided text instructions to locate the subject.
4. **Hyperparameter sensitivity**: Although uniform parameters perform reasonably well, achieving optimal results still requires category-specific fine-tuning of four parameters: $S_{GI}$, $E_{GI}$, $Layer_{LQ}$, and $E_{LQ}$ .

## Related Work & Insights

- **MasaCtrl**: Revealed that K/V features within self-attention layers contain rich semantic representations, which inspired SALQ.
- **Prompt-to-Prompt**: Demonstrated the feasibility of image editing via cross-attention control.
- **PHOTOSWAP / TIGIC**: Customization methods targeting single tasks (replacement or addition). Contrastingly, this work unifies three customization tasks.
- **Insight**: Attention manipulation serves as a core lever for controllable generation in diffusion models, and multi-process collaboration is more efficient than single-process control.

## Rating

⭐⭐⭐⭐ — The method is elegantly designed, and the complementary mechanism of SAGI+SALQ is novel. The absence of training overhead is a major advantage. However, relying on Stable Diffusion v1.5 limits its performance cap, and it requires tuning four hyperparameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)

</div>

<!-- RELATED:END -->
