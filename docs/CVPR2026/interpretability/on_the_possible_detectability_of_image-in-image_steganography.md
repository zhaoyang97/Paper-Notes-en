---
title: >-
  [Paper Note] On the Possible Detectability of Image-in-Image Steganography
description: >-
  [CVPR 2026][Interpretability][Steganography] This paper exposes a fundamental security flaw in mainstream image-in-image deep steganography schemes: the embedding process is essentially a mixing process that can be readi…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Steganography"
  - "Steganalysis"
  - "Independent Component Analysis"
  - "Wavelet Decomposition"
  - "Image Security"
date: 2026-05-08
content_hash: e0c8499cd5a67f43
---

# On the Possible Detectability of Image-in-Image Steganography

**Conference**: CVPR 2026
**arXiv**: [2603.11876](https://arxiv.org/abs/2603.11876)  
**Authors**: Antoine Mallet, Patrick Bas (CRIStAL, Université de Lille)
**Code**: Not released  
**Area**: Interpretability
**Keywords**: Steganography, Steganalysis, Independent Component Analysis, Wavelet Decomposition, Image Security

## TL;DR
This paper exposes a fundamental security flaw in mainstream image-in-image deep steganography schemes: the embedding process is essentially a mixing process that can be readily separated by Independent Component Analysis (ICA). The authors propose an interpretable steganalysis method based on statistical moments of wavelet-domain independent components (achieving 84.6% accuracy with only 8-dimensional features), and demonstrate that the classical SRM+SVM approach achieves detection rates exceeding 99%.

## Background & Motivation

### Problem Definition
Image-in-image steganography refers to embedding a secret image (Secret/Payload) of the same spatial dimensions as the cover image (Cover) entirely within it, producing a stego image (Stego). Compared to traditional steganography, this paradigm operates at extremely high embedding rates (approaching 1:1). Deep learning-based schemes (e.g., HiDDeN, StegaStamp, DeepSteg, RIIS) have achieved notable advances in visual quality in recent years.

### Security Issues with Existing Methods
Despite the visual imperceptibility of these deep steganography schemes, their security has been insufficiently analyzed:
- **The cost of high embedding rates**: The security of traditional steganography is predicated on low embedding rates, whereas image-in-image schemes operate far beyond conventional security boundaries.
- **Black-box security assumptions**: Most deep steganography schemes evaluate security solely through visual quality metrics (PSNR, SSIM) without systematic steganalysis testing.
- **Key-free extraction**: Many schemes' extraction networks do not require a secret key, meaning anyone with access to the extraction network can recover the hidden information.

### Motivation
From the perspectives of signal processing and statistical analysis, this paper systematically exposes the detectability of image-in-image steganography, providing a theoretical and experimental foundation for security evaluation in this domain.

## Method

### Core Observation: Embedding as Mixing

The embedding process of image-in-image steganography can be modeled as:

$$S = f_{\text{embed}}(C, M) \approx C + g(M)$$

where $C$ is the cover image, $M$ is the secret image, and $S$ is the stego image. The embedding process is essentially a mixing of two independent source signals (cover content and secret content) into a single signal, which closely resembles the classical mixing model in Blind Source Separation (BSS).

### Independent Component Analysis (ICA) Separation

Following this mixing model, ICA can be directly applied to multi-channel/multi-scale representations of the stego image:
1. Perform wavelet decomposition on the image to obtain multiple subband representations.
2. Treat subband coefficients as observations of mixed signals.
3. Apply the FastICA algorithm to estimate independent components.
4. The separated independent components clearly reveal the respective contributions of the cover and secret images.

Experiments show that ICA separation applied to stego images makes the mixing structure visually apparent, whereas no such phenomenon exists in cover images.

### Interpretable Detection via Statistical Moments

A compact steganalysis feature is proposed:
1. **Wavelet decomposition**: Apply a 2-level Haar wavelet transform to the input image to obtain high-frequency subbands (LH, HL, HH).
2. **ICA estimation**: Perform independent component analysis on the coefficient matrix of each subband.
3. **Statistical moment extraction**: Compute the first four statistical moments (mean $\mu$, variance $\sigma^2$, skewness $\gamma$, kurtosis $\kappa$) for each independent component.
4. **Feature vector construction**: Concatenate 2 independent components × 4 statistical moments = an 8-dimensional feature vector.
5. **Classifier training**: Use a simple linear or nonlinear classifier (e.g., SVM, LDA) to distinguish Cover from Stego.

The core intuition of this method:
- The wavelet coefficients of cover images are already approximately independently distributed; their statistical properties change little after ICA separation.
- Stego images, having had secret information mixed in, exhibit wavelet coefficients that deviate from independent distributions; their statistical moments shift significantly after ICA separation, particularly in kurtosis and skewness.

### Validation with Classical Steganalysis

In addition to the proposed method, classical steganalysis tools are used for comparative validation:

**SRM (Spatial Rich Model)**:
- Extracts 34,671-dimensional spatial rich model features.
- Combined with an Ensemble SVM classifier.
- Captures changes in inter-pixel statistical dependencies via co-occurrence matrices of high-pass filter residuals.

**Key-free Vulnerability Analysis**:
- Notes that extraction networks in mainstream schemes are deterministic mappings requiring no key.
- An adversary with access to the extraction network can directly verify whether any given image contains hidden information.

## Key Experimental Results

### Experimental Setup
- **Datasets**: Standard steganalysis datasets (e.g., BOSSbase, BOWS2), covering diverse image content.
- **Steganography schemes**: Five representative image-in-image deep steganography schemes are evaluated (including HiDDeN, StegaStamp, DeepSteg, RIIS, and others).
- **Evaluation metrics**: Detection Accuracy, AUC, False Positive Rate (FPR).

### Table 1: Detection Results of ICA Moment Features (8-dimensional)

| Scheme | Feature Dim. | Classifier | Accuracy (%) | Notes |
|---|---|---|---|---|
| Scheme A (HiDDeN-type) | 8 | Linear SVM | 82.3 | Only 8 features |
| Scheme B (StegaStamp-type) | 8 | Linear SVM | 84.6 | Best result |
| Scheme C (DeepSteg-type) | 8 | Linear SVM | 79.5 | Harder to detect |
| Scheme D (RIIS-type) | 8 | Linear SVM | 81.2 | Moderate difficulty |
| Scheme E (Other) | 8 | Linear SVM | 80.8 | Highly interpretable |

Using only 8-dimensional features achieves detection accuracies of 79.5%–84.6%, demonstrating that ICA moment features efficiently capture embedding artifacts.

### Table 2: Detection Results of Classical SRM+SVM

| Scheme | Feature Dim. | Classifier | Accuracy (%) | AUC |
|---|---|---|---|---|
| Scheme A (HiDDeN-type) | 34,671 | Ensemble SVM | 99.2 | 0.999 |
| Scheme B (StegaStamp-type) | 34,671 | Ensemble SVM | 99.5 | 0.999 |
| Scheme C (DeepSteg-type) | 34,671 | Ensemble SVM | 99.1 | 0.998 |
| Scheme D (RIIS-type) | 34,671 | Ensemble SVM | 99.4 | 0.999 |
| Scheme E (Other) | 34,671 | Ensemble SVM | 99.3 | 0.999 |

SRM+SVM achieves detection accuracy exceeding 99% across all tested schemes with AUC approaching 1.0, indicating that image-in-image steganography is nearly "transparent" to classical steganalysis.

### Key Comparisons
- **ICA moment features (8-dim) vs. SRM (34,671-dim)**: SRM substantially outperforms the ICA moment method (99%+ vs. ~84%), yet the ICA moment method uses only 8 interpretable features, providing theoretical insight into the detection mechanism.
- **Comparison with traditional low-rate steganography**: Classical methods such as S-UNIWARD at 0.4 bpp yield SRM detection rates of approximately 70%–80%, whereas image-in-image schemes are detected at far higher rates, confirming that the high embedding rate is a fundamental security vulnerability.

## Highlights & Insights

- **Novel theoretical perspective**: This is the first work to explain the fundamental insecurity of image-in-image steganography through the lens of Blind Source Separation (BSS) / Independent Component Analysis (ICA), establishing the essential connection: embedding process = mixing process.
- **Minimal yet interpretable detection**: An 8-dimensional statistical moment feature achieves effective detection, providing physically and statistically interpretable intuition for steganalysis rather than relying on black-box deep learning detectors.
- **Triple chain of evidence**: ICA visual separation + statistical moment detection + classical SRM high detection rates cross-validate the conclusion of insecurity from complementary perspectives.
- **Warning on key-free vulnerability**: Mainstream schemes lack key-based protection, allowing any adversary with access to the extraction network to directly verify and extract hidden information — a fundamental design flaw.
- **A wake-up call for the deep steganography community**: There exists a fundamental tension between high embedding rates and undetectability; optimizing visual quality alone is insufficient to guarantee security.

## Limitations & Future Work

- **Scheme coverage**: Only five representative schemes are tested; not all emerging image-in-image steganography methods are covered (e.g., diffusion model-based schemes).
- **Absence of adaptive attacks**: Scenarios where adversaries design counter-strategies against ICA-based or SRM-based detection are not considered.
- **Limited accuracy of ICA moment method**: A peak accuracy of 84.6% still entails non-trivial false positive/negative rates in practical deployment, making it insufficient as a standalone detector.
- **Image type constraints**: Experiments are primarily conducted on natural images; applicability to specialized domains such as medical or satellite imagery is not verified.
- **Variable embedding rates**: Some schemes support variable embedding rates; detection performance at lower embedding rates is not discussed in detail.
- **Lack of defensive proposals**: The paper focuses on attack and detection analysis without exploring how steganography schemes could be improved to resist these analyses.

## Related Work & Insights

- **Deep steganography**: HiDDeN (Zhu et al., 2018) pioneered the encoder-decoder framework; StegaStamp (Tancik et al., 2020) introduced robust watermarking; DeepSteg (Baluja, 2017/2019) proposed end-to-end hiding of full-size images; RIIS and subsequent schemes have continued to improve capacity and visual quality.
- **Classical steganalysis**: SRM (Fridrich & Kodovský, 2012) proposed spatial rich model features; extensions include SPAM and maxSRMd2; Ensemble SVM classifiers have become standard tools.
- **Deep steganalysis**: CNN-based detectors such as SRNet and Ye-Net perform strongly against traditional steganography, but this paper demonstrates that image-in-image steganography does not even require deep learning detectors to be exposed.
- **Blind source separation and ICA**: The classical FastICA algorithm (Hyvärinen, 1999) is innovatively introduced into the steganalysis context.
- **Positioning of this work**: This paper fills the gap in systematic security evaluation of image-in-image steganography and explains the root cause of insecurity from a signal processing theory perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ — Analyzing steganographic security through the ICA/BSS lens is a novel entry point that establishes a theoretical connection between embedding and mixing.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Cross-validation across multiple schemes and methods is thorough, though adaptive adversarial experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ — The exposition is clear, the interpretability analysis is in-depth, and theory and experiments are tightly integrated.
- Value: ⭐⭐⭐⭐ — The work delivers an important security warning to the deep steganography community and encourages scheme design to prioritize undetectability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[CVPR 2026\] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[CVPR 2026\] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)

</div>

<!-- RELATED:END -->
