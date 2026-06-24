---
title: >-
  [Paper Note] One Wave To Explain Them All: A Unifying Perspective On Feature Attribution
description: >-
  [ICML2025][Audio & Speech][Wavelet Transform] Proposes the Wavelet Attribution Method (WAM), which shifts feature attribution from the pixel domain to the wavelet domain, leveraging the spatial-scale locality of wavelet coefficients to provide unified and more structurally informative model explanations for audio, image, and volumetric data.
tags:
  - "ICML2025"
  - "Audio & Speech"
  - "Wavelet Transform"
  - "Feature Attribution"
  - "Explainable AI"
  - "Multimodal"
  - "SmoothGrad"
  - "Integrated Gradients"
date: 2026-05-08
content_hash: 87233fb147512f3a
---

# One Wave To Explain Them All: A Unifying Perspective On Feature Attribution

**Conference**: ICML2025  
**arXiv**: [2410.01482](https://arxiv.org/abs/2410.01482)  
**Code**: [Project Page](https://gabrielkasmi.github.io/wam/)  
**Area**: Audio & Speech  
**Keywords**: Wavelet Transform, Feature Attribution, Explainable AI, Multimodal, SmoothGrad, Integrated Gradients

## TL;DR

Proposes the Wavelet Attribution Method (WAM), which shifts feature attribution from the pixel domain to the wavelet domain, leveraging the spatial-scale locality of wavelet coefficients to provide unified and more structurally informative model explanations for audio, image, and volumetric data.

## Background & Motivation

- **Core Problem of Feature Attribution**: Existing methods (Saliency, SmoothGrad, Integrated Gradients, GradCAM, etc.) compute gradients in the pixel domain to generate heatmaps indicating "where the model looks," but the pixel domain lacks scale/frequency structural information of the signal.
- **Limitations of Prior Work**:
    - Only spatial translation relationships exist between pixels, failing to reveal multi-scale features such as textures, edges, and shapes that the model relies on.
    - For 1D audio and 3D volumetric data, they are usually projected to 2D before attribution, further losing structural information.
- **Key Insight**: The wavelet transform simultaneously preserves spatial location and frequency/scale information. The decomposed coefficients naturally correspond to interpretable low-level features such as edges, textures, and transients, and are applicable to any square-integrable signal (1D/2D/3D).
- **Background**: Only a few works (CartoonX, ShearletX, WCAM) have attempted attribution in the wavelet/Shearlet domain, and all are restricted to images and require perturbation sampling, lacking a unified cross-modal framework.

## Method

### Core Idea

Key operation of WAM: Applies the discrete wavelet transform to the input $\boldsymbol{x}$ to obtain coefficients $\boldsymbol{z} = \mathcal{W}(\boldsymbol{x})$, and then computes the gradient of the classifier output with respect to the wavelet coefficients, instead of the original pixels.

### Wavelet-domain Saliency Map

The saliency of the classifier $\boldsymbol{f}_c$ in the wavelet domain is defined as:

$$\boldsymbol{\gamma}_{\text{Sa}}(\boldsymbol{z}) = \left| \frac{\partial \boldsymbol{f}_c(\boldsymbol{x})}{\partial \boldsymbol{z}} \right| = \left| \frac{\partial \boldsymbol{f}_c(\boldsymbol{x})}{\partial \boldsymbol{x}} \cdot \frac{\partial \mathcal{W}^{-1}(\boldsymbol{z})}{\partial \boldsymbol{z}} \right|$$

where $\mathcal{W}^{-1}$ is the Jacobian of the inverse wavelet transform. In practice, the wavelet transform is first applied to the input, and then gradients are computed with respect to $\boldsymbol{z}$ via automatic differentiation, requiring only a single backward pass.

### WAM$_{\text{SG}}$ (SmoothGrad Variant)

Adds noise and smooths in the wavelet domain, averaging over $n$ noisy samples:

$$\boldsymbol{\gamma}_{\text{SG}}(\boldsymbol{z}) = \frac{1}{n} \sum_{i=1}^{n} \nabla_{\tilde{\boldsymbol{z}}} \boldsymbol{f}(\mathcal{W}^{-1}(\tilde{\boldsymbol{z}})), \quad \tilde{\boldsymbol{z}} = \mathcal{W}(\boldsymbol{x} + \boldsymbol{\delta}), \; \boldsymbol{\delta} \sim \mathcal{N}(0, I\sigma^2)$$

### WAM$_{\text{IG}}$ (Integrated Gradients Variant)

Performs path integration in the wavelet domain, from a baseline $\boldsymbol{z}_0$ to the current $\boldsymbol{z}$:

$$\boldsymbol{\gamma}_{\text{IG}} = (\boldsymbol{z} - \boldsymbol{z}_0) \cdot \int_0^1 \frac{\partial \boldsymbol{f}_c(\mathcal{W}^{-1}(\boldsymbol{z}_0 + \alpha(\boldsymbol{z} - \boldsymbol{z}_0)))}{\partial \boldsymbol{z}} \, d\alpha$$

WAM$_{\text{IG}}$ inherits the Sensitivity and Implementation Invariance axioms of Integrated Gradients.

### Multi-scale Interpretation

- Wavelet coefficients are hierarchical across scales (scale level $j$) and directions (horizontal/vertical/diagonal). The attribution results naturally decompose into "where" (spatial localization) and "what is seen" (scale/texture/edge).
- Summing the attributions at each scale allows for the quantification of the model's reliance on different frequency bands.

### Modality Agnosticism

- The wavelet transform is defined for 1D (audio waveforms), 2D (images), and 3D (volumetric/medical imaging) data. Thus, WAM can be applied across modalities without any modality-specific adaptation.

## Key Experimental Results

### Evaluation Metrics

- **Faithfulness** = Insertion − Deletion (higher is better)
- **Insertion** (↑): Gradually inserts features in descending order of attribution scores and observes the rate of increase in prediction probability.
- **Deletion** (↓): Gradually removes features in descending order of attribution scores and observes the rate of decrease in prediction probability.

### Main Results

| Modality | Model | Dataset | Method | Ins ↑ | Del ↓ | Faith ↑ |
|------|------|--------|------|-------|-------|---------|
| Audio | ResNet | ESC-50 | Integrated Gradients | 0.267 | 0.047 | 0.264 |
| Audio | ResNet | ESC-50 | SmoothGrad | 0.251 | 0.067 | 0.184 |
| Audio | ResNet | ESC-50 | GradCAM | 0.274 | 0.201 | 0.072 |
| Audio | ResNet | ESC-50 | **WAM$_{\text{IG}}$** | **0.436** | 0.260 | 0.176 |
| Audio | ResNet | ESC-50 | **WAM$_{\text{SG}}$** | **0.449** | 0.252 | **0.197** |
| Image | EfficientNet | ImageNet | GradCAM | 0.364 | 0.303 | 0.061 |
| Image | EfficientNet | ImageNet | **WAM$_{\text{IG}}$** | **0.447** | **0.049** | **0.370** |
| Volumetric | 3D Former | AdrenalMNIST3D | Saliency | 0.751 | 0.742 | 0.009 |
| Volumetric | 3D Former | AdrenalMNIST3D | **WAM$_{\text{IG}}$** | 0.719 | **0.621** | **0.098** |

**Key Findings**:

- On the image modality, WAM$_{\text{IG}}$ achieves a Faithfulness of 0.370, far exceeding GradCAM's 0.061 and SmoothGrad's 0.010.
- On volumetric data, WAM is the first method to achieve positive Faithfulness (other methods are either negative or close to zero).
- On audio, WAM leads significantly in terms of the Insertion score (0.449 vs 0.274), although its Deletion is slightly higher than the best-performing baseline (a trade-off).

## Highlights & Insights

1. **Unified Framework**: For the first time, a single method (wavelet-domain gradients) is used to explain three modalities (audio, image, and volumetric data) in a unified manner, without requiring modality-specific designs.
2. **A New Perspective of Multi-scale Decomposition**:
    - For images, the attribution at each spatial location can be decomposed into fine textures (blue/high frequency), mid-scale contours (red/medium frequency), and coarse edges (yellow/low frequency), which is impossible for pixel-domain attribution.
    - For 3D volumetric data, it presents the first demonstration of multi-scale attribution decomposition—coarse scales capture organ/lesion contours, while fine scales capture texture details.
3. **Bridge for Robustness Evaluation**: By summing the attributions at each scale, the model's reliance on low/high-frequency features can be directly quantified. Experiments demonstrate that adversarially trained models rely more heavily on coarse-scale (low-frequency) features, which aligns with robustness literature, and bypasses the need for complex frequency perturbation experiments.
4. **Application in Audio Denoising**: In 0 dB white noise experiments, WAM effectively filters out noise and retains the target sound components by reconstructing the audio from selected important wavelet coefficients. This is done in an entirely post-hoc manner without training models like NMF.
5. **Theoretical Guarantees**: WAM$_{\text{IG}}$ preserves the Sensitivity and Implementation Invariance axioms.

## Limitations & Future Work

- **No Support for Point Clouds**: Wavelet transform requires a structured grid and cannot directly process unstructured point cloud data; future work could explore graph wavelet transforms.
- **Limited Quality of Aural Explanations**: The greedy extraction of important coefficients is not smooth enough, which may introduce artifacts in the generated aural explanations.
- **Not Applicable to Text**: The wavelet transform cannot be mathematically applied to discrete text data, limiting its application in NLP and multimodal VLMs.
- **Choice of Wavelet Bases**: Different mother wavelets (Haar, Daubechies, etc.) affect the interpretation of attribution results; the paper does not fully discuss how to automatically select the optimal wavelet base.
- **Computational Overhead**: WAM$_{\text{SG}}$ requires multiple samplings, and WAM$_{\text{IG}}$ requires path integration; the computational costs on high-resolution inputs are not reported in detail.

## Related Work & Insights

- **CartoonX / ShearletX** (Kolek et al., 2022/2023): Perform Meaningful Perturbation in the wavelet/Shearlet domain, but these are perturbation-based methods and are computationally expensive.
- **WCAM** (Kasmi et al., 2023): Computes Sobol attribution in the wavelet domain; WAM is a gradient-based generalization of it.
- **Scattering Transform** (Bruna & Mallat, 2013): Fixed feature extractors built from wavelet coefficients, which are naturally interpretable and form part of WAM's theoretical foundation.
- **Frequency Bias Literature** (Zhang et al., 2022; Wang et al., 2020): Studies the relationship between model robustness and frequency dependence; WAM provides a more concise means of quantification.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of a unified framework for wavelet-domain attribution is novel, with valuable cross-modal generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three modalities, multiple datasets, and various baselines/metrics, with rich supplementary experiments in the appendix.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations, excellent visualizations, and a complete structure.
- Value: ⭐⭐⭐⭐ — Provides a new domain selection perspective for explainable AI with high practicality (robustness evaluation, audio denoising).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Transcripts: A Renewed Perspective on Audio Chaptering](../../ACL2026/audio_speech/beyond_transcripts_a_renewed_perspective_on_audio_chaptering.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](../../NeurIPS2025/audio_speech/unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[NeurIPS 2025\] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](../../NeurIPS2025/audio_speech/from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)
- [\[ICML 2026\] Multimodal Fact-Level Attribution for Verifiable Reasoning](../../ICML2026/audio_speech/multimodal_fact-level_attribution_for_verifiable_reasoning.md)

</div>

<!-- RELATED:END -->
