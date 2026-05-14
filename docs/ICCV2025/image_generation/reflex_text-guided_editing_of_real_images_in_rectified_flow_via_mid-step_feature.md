---
title: >-
  [Paper Note] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation
description: >-
  [ICCV 2025][Image Generation][Image Editing] To address the challenge of real image editing in Rectified Flow (ReFlow) models, this paper systematically analyzes intermediate representations in MM-DiT…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image Editing"
  - "Rectified Flow"
  - "FLUX"
  - "Feature Injection"
  - "Attention Adaptation"
date: 2026-05-08
content_hash: a380285e03cefd49
---

# ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation

**Conference**: ICCV 2025
**arXiv**: [2507.01496](https://arxiv.org/abs/2507.01496)
**Code**: None
**Area**: Image Generation
**Keywords**: Image Editing, Rectified Flow, FLUX, Feature Injection, Attention Adaptation

## TL;DR

To address the challenge of real image editing in Rectified Flow (ReFlow) models, this paper systematically analyzes intermediate representations in MM-DiT, identifies three key features (I2I-SA, I2T-CA, and residual features), and proposes mid-step feature extraction along with two attention adaptation techniques. The resulting training-free, user-mask-free method achieves high-quality real image editing on the FLUX model, attaining a 68.2% human preference rate that substantially outperforms competing approaches.

## Background & Motivation

Text-guided real image editing is a critical application of image generation. U-Net-based diffusion models (e.g., Stable Diffusion) have established mature editing pipelines (P2P, PnP) that exploit cross-attention and self-attention features for structure preservation. However, the latest generation of Rectified Flow models (e.g., FLUX) adopts a fundamentally different architecture and training paradigm, making direct transfer of existing methods non-trivial.

**Limitations of Prior Work in ReFlow Editing**:
- **RF-Inversion**: Constructs a controlled ODE by interpolating between the source image and noise, resulting in over-reliance on the source image and limited editing flexibility.
- **RF-Edit / FireFlow**: Reduces reconstruction error via second-order Taylor expansion but relies on value features, which can cause appearance leakage.
- **FlowEdit**: Builds a source-to-target ODE that minimizes edit magnitude, but struggles with large-scale modifications such as color changes.
- **Common Issue 1**: MM-DiT entangles text and image tokens in a joint self-attention mechanism, which differs fundamentally from the decoupled cross/self-attention in U-Net architectures; it remains unclear which features are effective for editing.
- **Common Issue 2**: ReFlow models often fail to accurately reconstruct the source image structure when starting from a fully inverted latent.

**Core Contributions**:
1. A systematic decomposition of the joint self-attention map in MM-DiT into four components—I2I-SA, I2T-CA, T2I-CA, and T2T-SA—revealing that only the first two are effective for editing.
2. A finding that extracting features from a mid-step latent rather than the fully inverted latent substantially improves structure preservation.

## Method

### Overall Architecture

ReFlex operates in three stages:
1. **Inversion**: The real image is inverted via the ReFlow forward process to obtain both $z_T$ (fully noised) and $z_{t'=T/2}$ (mid-step latent).
2. **Feature Extraction**: Starting from $z_{t'}$, a single inference step is performed to extract the three key features.
3. **Target Generation**: Starting from $z_T$, the target image is generated using the target prompt, with the extracted features injected during early timesteps and attention adaptation applied throughout.

### Key Designs

1. **Identification of Key MM-DiT Features**:

    - **Function**: Determine which intermediate representations in MM-DiT are effective for image editing.
    - **Mechanism**: The joint self-attention matrix $Q \cdot K^T$ is decomposed into four components by query-key modality. PCA visualization (Fig. 4a) shows that I2I-SA ($Q_{image} \cdot K_{image}^T$) encodes structural information and I2T-CA ($Q_{image} \cdot K_{text}^T$) encodes text-image relationships. Injection experiments confirm that only I2I-SA and I2T-CA preserve source image structure, as they are the only components that directly influence image token embeddings. For residual connections, analysis of the residual feature $f(x)_{image}$ versus the identity feature $x_{image}$ shows that the latter retains excessive appearance detail and limits editability; thus only the residual feature is adopted.
    - **Design Motivation**: T2I-CA and T2T-SA directly influence text token embeddings, whereas only image token embeddings are fed to the decoder for image synthesis, rendering these components ineffective for structural preservation.

2. **Mid-Step Feature Extraction**:

    - **Function**: Extract features from a mid-step latent rather than the fully inverted latent to ensure structural fidelity.
    - **Mechanism**: Instead of reconstructing from the fully noised $z_T$ (which accumulates large reconstruction errors), features are extracted from $z_{t'=T/2}$. Fig. 5 illustrates the reconstruction quality at different timesteps: the latent at $t=T/2$ can nearly perfectly recover the source image, whereas the latent at $t=T$ loses substantial structural information.
    - **Design Motivation**: The inversion–reconstruction pipeline of ReFlow models exhibits significant error accumulation. Features extracted from the mid-step latent are more faithful to the source image. Although this reduces editability by over-preserving source information, the subsequent attention adaptation mechanisms compensate for this effect.

3. **I2T-CA Adaptation**:

    - **Function**: Enhance the response to novel tokens in the target prompt when injecting I2T-CA.
    - **Mechanism**: A token mapping function $f$ is defined between the source and target prompts. For tokens with correspondences, the source I2T-CA is injected directly; for new tokens without correspondences (e.g., "horse" when editing "goat" → "horse"), the target I2T-CA is scaled by an amplification factor $\alpha > 1$: $CA'[:,i] = \alpha \times CA_T[:,i]$ (when $f(i) = \emptyset$).
    - **Design Motivation**: Directly injecting the source I2T-CA suppresses semantic expression of novel tokens. Amplifying the attention weights of new tokens enhances editability and compensates for the reduced editability introduced by mid-step feature extraction.

4. **I2I-SA Adaptation**:

    - **Function**: Prevent the injected source I2I-SA from overly preserving local structure.
    - **Mechanism**: Visualization (Fig. 6) reveals that each query in I2I-SA concentrates excessively on a small number of nearby pixels. The top-$k$ attention values are replaced with the corresponding values from the target I2I-SA (after normalization), preserving global structure while permitting local modifications.
    - **Design Motivation**: In the linear scale, I2I-SA concentrates heavily on local regions, yet in the log scale it effectively captures global structure. Replacing the top-$k$ highest values frees the model for local edits without disrupting the global layout.

### Loss & Training

- Entirely training-free; all operations are performed at inference time via feature manipulation.
- Hyperparameters: sampling steps $T=28$, feature extraction step $t'=14$, I2T-CA amplification factor $\alpha=4$, I2I-SA $k=20$.
- With source prompt: I2T-CA injected for the first $0.4T$ steps, I2I-SA for the first $0.25T$ steps, residual features for the first $0.15T$ steps.
- Without source prompt: I2T-CA injection is omitted; I2I-SA injection is extended to $0.4T$ steps.
- Optional mask generation: an editing region mask is extracted from I2T-CA and used with latent blending to enable local editing.

## Key Experimental Results

### Main Results (Text Alignment and Structure Preservation)

| Method | Base Model | PIE-Bench CLIP-T↑ | Wild-TI2I CLIP-T↑ | User Preference % |
|--------|-----------|------------------|-------------------|------------------|
| RF-Inversion | FLUX | −1.69% vs. ReFlex | −3.21% vs. ReFlex | 7.8% |
| RF-Edit | FLUX | — | — | 7.3% |
| FireFlow | FLUX | — | — | 5.5% |
| FlowEdit | FLUX | — | — | 11.2% |
| DI+PnP | SD | — | — | 11.4% |
| DDIM+PnP | SD | — | — | 7.5% |
| SDEdit | SD | — | — | 9.4% |
| P2P-Zero | SD | — | — | 10.6% |
| **ReFlex** | **FLUX** | **Best** | **Best (+3.21–16.46%)** | **68.2% / 61.0%** |

### Ablation Study (Component Contributions on Wild-TI2I-Real)

| Configuration | Text Alignment | Structure Preservation (IoU) | Notes |
|---------------|---------------|------------------------------|-------|
| Without mid-step extraction (full inversion) | Higher | Significant drop | Source structure largely lost |
| Without I2T-CA adaptation | Drop | Higher | Editing fails to fully follow target prompt |
| Without I2I-SA adaptation | Drop | Higher | Large-scale edits (e.g., color) are difficult |
| **Full ReFlex** | **Best** | **Pareto-optimal** | **Balanced editability and preservation** |

### Key Findings
- ReFlex achieves overwhelming user preference: 68.2% (vs. FLUX-based methods) and 61.0% (vs. SD-based methods) in human evaluation.
- Mid-step feature extraction is the most critical component; removing it causes a substantial drop in IoU.
- $t' = T/2 = 14$ is the optimal extraction point (Fig. 10): extracting too close to the fully noised end loses structure, while extracting too close to the image end degrades output quality.
- $k=20$ provides a favorable trade-off for I2I-SA adaptation (Fig. 11): larger $k$ leads to loss of source structure.
- The method operates without a source prompt (Wild-TI2I-Real), demonstrating practical applicability.

## Highlights & Insights
- The systematic analysis of MM-DiT features constitutes the most significant contribution of this work, establishing a theoretical foundation for editing with ReFlow models.
- Mid-step feature extraction is conceptually simple yet highly effective, exploiting the property that mid-step latents in ReFlow models can be reconstructed far more faithfully than fully inverted latents.
- The design of replacing the top-$k$ highest values in I2I-SA adaptation cleverly leverages the different behaviors of attention distributions in linear versus log scales.
- The method is entirely training-free and integrates seamlessly into the FLUX inference pipeline, offering high practical value.

## Limitations & Future Work
- When the editing region overlaps with the subject, unintended attributes of the subject may be altered (e.g., hairstyle changes when removing glasses, Fig. 12a).
- Editing masks derived from I2T-CA are insufficiently precise and may introduce artifacts (Fig. 12b).
- Edited results exhibit some stochastic variation.
- Blended words are required to define editing regions, currently relying on manual selection or predefined choices.
- The effectiveness of the approach on other ReFlow models (e.g., SD3.5) has not been explored.

## Related Work & Insights
- P2P and PnP are seminal editing methods for diffusion models; ReFlex transfers their core ideas (attention/feature injection) to the MM-DiT architecture.
- The strategy of starting from a mid-step rather than fully noised latent bears conceptual similarity to SDEdit, though in ReFlex it is applied for feature extraction rather than direct denoising.
- The analysis of inversion errors in ReFlow models is noteworthy, as it explains why naïvely applying DM editing methods to FLUX yields poor results.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)

</div>

<!-- RELATED:END -->
