---
title: >-
  [Paper Note] LUSD: Localized Update Score Distillation for Text-Guided Image Editing
description: >-
  [ICCV 2025][Image Generation][Score Distillation] LUSD addresses the failure modes of existing score distillation methods in image editing (particularly object insertion) through two simple modifications—attention-based…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Score Distillation"
  - "Text-Guided Image Editing"
  - "Attention Regularization"
  - "Gradient Normalization"
  - "Diffusion Prior"
date: 2026-05-08
content_hash: 253b5479ae2224c5
---

# LUSD: Localized Update Score Distillation for Text-Guided Image Editing

**Conference**: ICCV 2025
**arXiv**: [2503.11054](https://arxiv.org/abs/2503.11054)
**Code**: None
**Area**: Diffusion Models
**Keywords**: Score Distillation, Text-Guided Image Editing, Attention Regularization, Gradient Normalization, Diffusion Prior

## TL;DR

LUSD addresses the failure modes of existing score distillation methods in image editing (particularly object insertion) through two simple modifications—attention-based spatial regularization and gradient filtering-normalization—which resolve instabilities caused by large disparities in gradient magnitude and spatial distribution, achieving a better balance between prompt fidelity and background preservation.

## Background & Motivation

**Background**: Diffusion models have demonstrated strong generative prior capabilities for text-guided image editing. Recent work has introduced score distillation techniques that leverage pretrained text-to-image diffusion models for training-free image editing. Representative methods include DDS (Delta Denoising Score) and PDS (Posterior Distillation Sampling).

**Limitations of Prior Work**: Score distillation-based methods frequently fail on tasks such as object insertion. Specifically, edited results either fail to correctly insert the target object (low prompt fidelity) or introduce excessive modifications to background regions (poor background preservation). It is observed that gradient magnitudes and spatial distributions vary significantly across different input images and editing targets, making hyperparameter tuning highly input-dependent or even infeasible.

**Key Challenge**: During score distillation, gradient updates are spatially non-uniform—the gradient signals received by editing regions and background regions differ substantially, and this disparity pattern is inconsistent across samples. This input-specificity prevents a single hyperparameter configuration from generalizing across different editing scenarios.

**Goal**: To design a robust score distillation method that operates stably across diverse editing tasks and input images without requiring input-specific hyperparameter tuning.

**Key Insight**: Through a thorough analysis of gradient properties in existing methods, the authors identify that the root cause lies in instability along two dimensions: spatial distribution and magnitude of gradients. The proposed modifications are therefore motivated from a gradient processing perspective.

**Core Idea**: Constrain the spatial distribution of gradients via attention-based spatial regularization, and stabilize gradient magnitudes via gradient filtering-normalization, enabling score distillation to operate reliably across diverse editing scenarios.

## Method

### Overall Architecture

LUSD is built upon the score distillation framework. Given a source image and a target editing prompt, the method iteratively optimizes the latent representation of the source image. At each iteration, the current image is fed into a pretrained text-to-image diffusion model to compute the score distillation gradient, which is then processed by two modules—attention-based spatial regularization and gradient filtering-normalization—before updating the image. No additional model fine-tuning is required.

### Key Designs

1. **Attention-based Spatial Regularization**:

    - **Function**: Constrains gradient updates to focus on editing-relevant spatial regions, preventing unnecessary modification of background areas.
    - **Mechanism**: Cross-attention maps from the diffusion model are used to identify spatial regions relevant to the target prompt. The attention maps are thresholded to produce soft masks, which are then used to spatially weight the score distillation gradients so that updates are concentrated in the regions corresponding to the editing target. This implicitly encodes spatial information about "where to edit" into the gradient.
    - **Design Motivation**: Score distillation gradients are spatially non-uniform, while cross-attention maps naturally capture the correspondence between textual conditions and spatial locations. Leveraging this information enables localized updates and prevents global gradients from interfering with background regions.

2. **Gradient Filtering-Normalization**:

    - **Function**: Stabilizes the magnitude distribution of gradients and eliminates inter-sample scale disparities.
    - **Mechanism**: The computed score distillation gradients are first filtered to remove outliers and noise components, then normalized to map gradient magnitudes to a consistent scale. These two steps ensure that the update step size remains stable and controllable, preventing editing failures caused by excessively large or small gradients for certain inputs.
    - **Design Motivation**: The authors find that gradient magnitudes across different input images can differ by several orders of magnitude, which is the fundamental reason why hyperparameters fail to generalize. Normalization allows a single set of hyperparameters to be applied across diverse editing scenarios.

3. **Integration into the Score Distillation Framework**:

    - **Function**: Seamlessly integrates the two modifications into the standard score distillation pipeline.
    - **Mechanism**: At each diffusion denoising iteration, standard score distillation gradients (in the style of DDS/PDS) are computed first, followed by sequential application of spatial regularization and gradient normalization. Both modules are plug-and-play and do not alter the parameters of the underlying diffusion model.
    - **Design Motivation**: Maintains the simplicity and generality of the method, facilitating compatibility with different score distillation variants.

### Loss & Training

LUSD requires no additional training and relies entirely on the generative prior of the pretrained diffusion model. The optimization objective is the score distillation loss after applying spatial regularization and gradient normalization.

## Key Experimental Results

### Main Results

LUSD is compared against several state-of-the-art score distillation-based image editing methods, evaluated on prompt fidelity and background preservation.

| Method | Prompt Fidelity (CLIP-T) ↑ | Background Preservation (LPIPS) ↓ | User Preference ↑ |
|--------|----------------------------|------------------------------------|-------------------|
| DDS | Low | Moderate | ~20% |
| PDS | Moderate | Low | ~20% |
| LUSD (Ours) | **Highest** | **Lowest** | **58–64%** |

### Ablation Study

| Configuration | Prompt Fidelity | Background Preservation | Notes |
|---------------|-----------------|------------------------|-------|
| Full model (LUSD) | Best | Best | Complete model |
| w/o Spatial Regularization | Degraded | Significantly degraded | Edits leak into background |
| w/o Gradient Normalization | Significantly degraded | Moderate | Editing fails on some inputs |
| Standard SDS only | Poor | Poor | Baseline method |

### Key Findings

- Attention-based spatial regularization contributes most to background preservation; removing it frequently causes edits to bleed into background regions.
- Gradient normalization is critical for improving editing success rate, particularly on challenging tasks such as object insertion, where its removal causes complete editing failure on some inputs.
- In user studies, 58–64% of participants preferred LUSD over competing methods across three dimensions: prompt fidelity, background preservation, and overall quality.
- The method demonstrates consistent performance across diverse editing tasks, including object insertion, attribute modification, and style transfer.

## Highlights & Insights

- **Gradient-analysis-driven method design**: Rather than blindly modifying loss functions or network architectures, the authors identify the root cause through careful analysis of gradient properties in failure cases and address it with two remarkably concise operations. This "diagnose-then-fix" research paradigm is instructive.
- **Plug-and-play improvements**: Both modification modules can be directly applied to any score distillation-based method without retraining, offering strong generality.
- **Attention maps as spatial guidance**: Using the cross-attention maps of the diffusion model itself to localize editing regions is a transferable idea applicable to other diffusion model applications requiring spatial control.

## Limitations & Future Work

- The approach relies on the quality of cross-attention maps; spatial regularization may be less effective when prompts are complex or attention is diffuse.
- Validation is primarily conducted within the score distillation framework; integration with other editing paradigms (e.g., DDIM inversion with guided sampling) remains unexplored.
- Editing speed is constrained by the iterative optimization process inherent to score distillation, posing challenges for real-time applications.
- The capability to handle simultaneous multi-object editing warrants further investigation.

## Related Work & Insights

- **vs. DDS (Delta Denoising Score)**: DDS directly uses the raw score difference as the gradient without spatial constraints or magnitude normalization, leading to instability in complex editing scenarios. The two modifications in LUSD directly address the core limitations of DDS.
- **vs. PDS (Posterior Distillation Sampling)**: PDS improves score distillation quality through posterior estimation but remains affected by spatially uneven gradient distributions. LUSD's spatial regularization is complementary to PDS.
- **vs. Prompt-to-Prompt / Null-text Inversion**: These methods control editing by modifying attention maps or optimizing null-text embeddings. While the underlying approach differs from LUSD, the motivation is similar—both leverage attention mechanisms for editing control.

## Rating

- Novelty: ⭐⭐⭐ The core contributions are two simple gradient processing techniques; effective but of limited conceptual novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes both automatic metrics and user studies, with comprehensive baselines and complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear and the motivation is logically well-grounded.
- Value: ⭐⭐⭐⭐ Offers plug-and-play improvements for score distillation-based editing methods with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-based Image Editing](addressing_text_embedding_leakage_in_diffusion_based_image_editing.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)

</div>

<!-- RELATED:END -->
