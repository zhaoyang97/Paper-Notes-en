---
title: >-
  [Paper Note] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation
description: >-
  [ECCV 2024][Image Generation][Image-to-Image Translation] This paper proposes PIC (Prompt Interpolation-based Correction), a training-free image-to-image translation method for diffusion models. By constructing a noise correction term through progressive prompt embedding interpolation and linearly combining it with the noise prediction of the source image, PIC achieves structure-preserving, high-fidelity image editing with an inference speed (18.1s) outperforming all baseline…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image-to-Image Translation"
  - "Training-free Method"
  - "Noise Correction"
  - "Prompt Interpolation"
  - "Diffusion Model Editing"
date: 2026-05-08
content_hash: 8e63ac1fb3a88419
---

# Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation

**Conference**: ECCV 2024  
**arXiv**: [2409.08077](https://arxiv.org/abs/2409.08077)  
**Code**: None (implemented based on the Pix2Pix-Zero code framework)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Image-to-Image Translation, Training-free Method, Noise Correction, Prompt Interpolation, Diffusion Model Editing

## TL;DR

This paper proposes PIC (Prompt Interpolation-based Correction), a training-free image-to-image translation method for diffusion models. By constructing a noise correction term through progressive prompt embedding interpolation and linearly combining it with the noise prediction of the source image, PIC achieves structure-preserving, high-fidelity image editing with an inference speed (18.1s) outperforming all baseline methods.

## Background & Motivation

**Background**: Text-driven image-to-image translation based on diffusion models has become a popular research direction. The goal of this task is to modify local areas of a source image according to a target prompt while preserving the background and the overall structure. Existing methods fall into two categories:
1. Fine-tuning-based methods (DiffusionCLIP, Imagic, etc.): These customize pre-trained models via fine-tuning, resulting in high computational costs.
2. Training-free methods (Prompt-to-Prompt, Plug-and-Play, Pix2Pix-Zero, etc.): These manipulate the denoising strategy during the reverse process.

**Limitations of Prior Work**:

**Inaccurate starting point**: The starting point of the DDIM reverse process $\mathbf{x}_T^{\text{tgt}}$ is directly set to $\mathbf{x}_T^{\text{src}}$. However, the actual target starting point $\mathbf{x}_T^{\text{tgt*}}$ deviates from it, which prevents the naive reverse process from generating the ideal target image.

**Abrupt text embedding transition**: The sudden switch from the source prompt embedding $\mathbf{y}^{\text{src}}$ to the target prompt embedding $\mathbf{y}^{\text{tgt}}$ leads to instability in the generation process.

**Difficulty in preserving structure**: Existing methods struggle to perfectly retain the background and overall structure while editing target regions.

**Key Challenge**: How to compensate for the deviation caused by the incorrect starting point of the reverse process without any training, while precisely controlling the boundaries between the edited and preserved regions?

**Key Insight**: Instead of directly performing denoising using the target prompt, this method progressively interpolates from the source prompt to the target prompt, using the noise prediction difference during the interpolation process to construct a correction term that guides the reverse process toward the correct target.

**Core Idea**: Modified noise prediction = source prompt noise prediction of the source image (to preserve structure) + correction term (for selective editing), where the correction term is naturally localized to the regions requiring editing through progressive prompt interpolation.

## Method

### Overall Architecture

PIC modifies the noise prediction network of the standard DDIM reverse process, decomposing it into two parts:
1. **Structure preservation term**: $\epsilon_\theta(\mathbf{x}_t^{\text{src}}, t, \mathbf{y}^{\text{src}})$ — reconstructs the source image using the source latent and the source prompt.
2. **Noise correction term**: $\gamma \Delta\epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t)$ — guides target regions to align with the target domain.

The complete modified noise prediction is formulated as:
$$\hat{\epsilon}_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}^{\text{tgt}}) := \epsilon_\theta(\mathbf{x}_t^{\text{src}}, t, \mathbf{y}^{\text{src}}) + \gamma \Delta\epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t)$$

The noise correction is only applied during the first $\tau$ steps (default 25 steps) of the reverse process, after which standard DDIM is used to complete the remaining denoising steps.

### Key Designs

1. **Noise Correction Term**:

    - **Function**: Captures the difference in noise space between the target prompt and the source prompt, only affecting the regions that require editing.
    - **Mechanism**: 
    $\Delta\epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t) := \epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t) - \epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}^{\text{src}})$
      namely, the difference in noise prediction of the same target latent under the interpolated prompt and the source prompt.
    - **Design Motivation**: When $\mathbf{y}_t$ matches $\mathbf{y}^{\text{src}}$, the difference is zero (no editing); as $\mathbf{y}_t$ approaches $\mathbf{y}^{\text{tgt}}$, the difference concentrates on regions related to the target prompt. Visualization confirms that the correction term automatically "highlights" the regions to be edited, while the background values approach zero. This effectively compensates for the gap between $\mathbf{x}_T^{\text{tgt*}}$ and $\mathbf{x}_T^{\text{src}}$.

2. **Progressive Prompt Interpolation**:

    - **Function**: Smoothly transitions the text embedding from the source prompt to the target prompt during the reverse process.
    - **Mechanism (Word Swap Task)**: Token-by-token linear interpolation:
    $\mathbf{y}_t[\ell] = \beta_t \mathbf{y}^{\text{tgt}}[\ell] + (1 - \beta_t) \mathbf{y}^{\text{src}}[\ell]$
      where the mixing coefficient is $\beta_t = \beta + (1-\beta) \times \frac{T-t}{T}$, gradually transitioning from the source embedding to the target embedding as the denoising step progresses.
    - **Mechanism (Phrase Addition Task)**: Directly uses the target embedding for newly added tokens and performs interpolation on the subsequent shared tokens:
    $\mathbf{y}_t[\ell] = \begin{cases} \mathbf{y}^{\text{src}}[\ell], & \text{if } \ell < \ell_s \\ \mathbf{y}^{\text{tgt}}[\ell], & \text{if } \ell_s \leq \ell \leq \ell_f \\ \beta_t \mathbf{y}^{\text{tgt}}[\ell] + (1-\beta_t)\mathbf{y}^{\text{src}}[\ell - \ell_f + \ell_s], & \text{if } \ell > \ell_f \end{cases}$
    - **Design Motivation**: Avoids sudden mutations in text embeddings, allowing the model to adapt progressively to the target domain, which preserves more source information during the early stages of the reverse process (the low-frequency/structural generation phase).

3. **Integration with Existing Methods**:

    - **Function**: Extends PIC's noise correction framework to other methods like Prompt-to-Prompt, Plug-and-Play, and Pix2Pix-Zero.
    - **Mechanism**: For each method, the target prompt embedding $\mathbf{y}^{\text{tgt}}$ of the original method is replaced with $\mathbf{y}_t$ (the interpolated embedding), and the method-specific noise correction is wrapped into the PIC framework.
    - **Design Motivation**: The formulation of PIC is orthogonal to existing methods, allowing it to serve as a plug-and-play performance-boosting module.

### Loss & Training

- **Training-free method**: Involves no loss functions or training process.
- Hyperparameter settings: $\tau = 25$ (correction steps), $\gamma = 1.0$, $\beta = 0.3$ for word swap tasks, and $\beta = 0.8$ for phrase addition tasks.
- Both forward and reverse processes use 50 steps, with Stable Diffusion v1.4 as the backbone.
- Uses BLIP to automatically generate source prompts, combined with classifier-free guidance.

## Key Experimental Results

### Main Results (with SOTA comparison, LAION-5B dataset)

| Method | CS (CLIP Similarity) ↑ | BD (Background Distance) ↓ | SD (Structure Distance) ↓ | Inference Time (s) |
|------|---------------|-------------|-------------|-----------|
| Prompt-to-Prompt | 0.302 | 0.113 | 0.040 | 31.2 |
| Plug-and-Play | 0.305 | 0.120 | 0.036 | 24.4 |
| Pix2Pix-Zero | 0.301 | 0.136 | 0.066 | 52.2 |
| **PIC (Ours)** | **0.304** | **0.071** | **0.034** | **18.1** |

PIC leads significantly in background preservation (BD improved by 37% compared to PtP) and structure preservation (best SD), while achieving the fastest inference speed.

**Performance of PIC integrated into existing methods**:

| Method | Original BD ↓ | BD ↓ +PIC | Original SD ↓ | SD ↓ +PIC |
|------|---------|---------|---------|---------|
| PtP | 0.113 | **0.069** | 0.040 | **0.023** |
| PnP | 0.120 | **0.098** | 0.036 | **0.029** |
| P2P | 0.136 | **0.066** | 0.066 | **0.015** |

As a plug-in, PIC consistently improves indices across all methods. Notably, for P2P + PIC, the SD drops from 0.066 to 0.015 (a 77% improvement).

### Ablation Study

| Configuration | CS ↑ | BD ↓ | SD ↓ | Description |
|------|-----|-----|-----|------|
| Naive DDIM | 0.302 | 0.216 | 0.094 | Baseline, structure is severely corrupted |
| DDIM+PI (prompt interpolation only) | 0.302 | 0.184 | 0.081 | Interpolation brings improvement but is insufficient |
| DDIM+NC (noise correction only, no interpolation) | 0.306 | 0.081 | 0.044 | Noise correction substantially reduces BD/SD |
| **PIC (Interpolation + Correction)** | **0.304** | **0.071** | **0.034** | The combination of both achieves the optimal performance |

### Key Findings

- The noise correction term is the primary source of performance gains (BD drops from 0.216 to 0.081), with prompt interpolation providing further optimization on top of it (BD drops from 0.081 to 0.071).
- Visualization of the correction term demonstrates that it automatically focuses on the target editing regions during the reverse process, while background values remain near zero.
- $\gamma$ controls the trade-off between editing strength and structure preservation: a lower $\gamma$ preserves structure, while a higher $\gamma$ enhances fidelity.
- Correction is only necessary in the first $\tau$ steps; subsequent stages focusing on detail generation do not require correction.
- The inference time of PIC is only 18.1s because the correction term has a limited active duration and does not introduce additional gradient computations.

## Highlights & Insights

- **Extremely simple yet effective**: The core formulation (Eq. 5-7) is elegant and concise, constructing the correction term using only the difference between two noise predictions without introducing any trainable parameters.
- **Orthogonality**: PIC is fully orthogonal to existing approaches and can be integrated into other methods in a plug-and-play manner to consistently boost performance.
- **Efficiency advantage**: Compared to methods requiring attention map injection (PtP/PnP) or gradient computations (P2P), PIC incurs negligible extra overhead.
- The progressive prompt interpolation strategy is highly intuitive: the reverse process handles low-frequency global structures during early stages, where source semantics should be preserved more, and switches fully to target semantics in later stages for detail generation.

## Limitations & Future Work

- It still fails on certain complex tasks (e.g., large posture or texture changes); overall, keeping fine-grained details of the source image remains a common challenge for all methods (including PIC).
- $\beta$ requires manual tuning for different task types (e.g., word swap vs. phrase addition), leaving room for improvement in automated hyperparameter selection.
- The experiments were only conducted on Stable Diffusion v1.4, leaving performance on newer models like SD 2.x, SDXL, or SD3 unverified.
- The prompt interpolation strategy assumes that tokens from the source and target prompts can be easily aligned, which might not hold for more complex semantic editing scenarios.
- A user study is lacking to complement the automated quantitative metrics.

## Related Work & Insights

- **Prompt-to-Prompt**: Preserves structure by injecting cross-attention and self-attention maps of the source image; the proposed method is complementary to it.
- **Plug-and-Play**: Replaces intermediate-layer self-attention maps and feature maps, suffering from high computational cost.
- **Pix2Pix-Zero**: Aligns cross-attention maps via gradient optimization, resulting in the longest inference time (52.2s).
- **Null-text/Negative-prompt Inversion**: Improves reconstruction accuracy in the reverse process, which can be orthogonally integrated with PIC.
- **Insight**: In diffusion model editing, a "gradual transition" is superior to an "abrupt switch," aligning with the progressive nature of the diffusion process itself.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of noise correction and prompt interpolation is simple yet novel, showing clear theoretical intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 translation tasks, 3 baseline methods, 3 integrated validations, and detailed ablations, though it lacks a user study.
- Writing Quality: ⭐⭐⭐⭐ Features a clear structure, easy-to-understand derivations, and detailed descriptions of integration with prior works.
- Value: ⭐⭐⭐⭐ The plug-and-play nature offers high practical value, with a clear advantage in its 18.1s inference speed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)
- [\[ECCV 2024\] EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models](ebdm_exemplar-guided_image_translation_with_brownian-bridge_diffusion_models.md)
- [\[ECCV 2024\] FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction](finematch_aspect-based_fine-grained_image_and_text_mismatch_detection_and_correc.md)
- [\[ECCV 2024\] 2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction](2s-odis_two-stage_omni-directional_image_synthesis_by_geometric_distortion_corre.md)

</div>

<!-- RELATED:END -->
