---
title: >-
  [Paper Note] On the Possible Detectability of Image-in-Image Steganography
description: >-
  [CVPR 2026][Steganalysis] This paper exposes a fundamental security vulnerability in invertible neural network (INN)-based image-in-image steganography: the embedding process is intrinsically a mixing process identifiable via independent component analysis (ICA). Using only 8-dimensional statistical features with an SVM achieves a detection rate of 84.6%, while the classical SRM+SVM baseline exceeds 99%.
tags:
  - CVPR 2026
  - Steganalysis
  - ICA
  - Wavelet Transform
  - Invertible Neural Networks
  - Blind Source Separation
date: 2026-05-08
content_hash: 5792d68a77781578
---

# On the Possible Detectability of Image-in-Image Steganography

**Conference**: CVPR 2026
**arXiv**: [2603.11876](https://arxiv.org/abs/2603.11876)
**Code**: None (uses scikit-learn FastICA + SVM; highly reproducible)
**Area**: Steganalysis / Information Security
**Keywords**: Steganalysis, ICA, Wavelet Transform, Invertible Neural Networks, Blind Source Separation

## TL;DR

This paper exposes a fundamental security vulnerability in invertible neural network (INN)-based image-in-image steganography: the embedding process is intrinsically a mixing process identifiable via independent component analysis (ICA). Using only 8-dimensional statistical features with an SVM achieves a detection rate of 84.6%, while the classical SRM+SVM baseline exceeds 99%.

## Background & Motivation

**Background**: A new class of image-in-image steganographic schemes has emerged in recent years. Unlike traditional steganography—which hides bitstreams at low embedding rates—these methods leverage invertible neural networks (INNs) to embed an entire same-size image within a cover image, resulting in extremely high embedding capacity. Representative schemes include HiNet, PRIS, and DeepMIH.

**Limitations of Prior Work**: Despite claims of high security, two fundamental issues have been overlooked: (1) most schemes perform keyless extraction, meaning anyone with knowledge of the network architecture can recover the secret image, violating Kerckhoffs's principle; (2) the coupling layers of INNs are affine transformations, and the embedding process likely leaves detectable statistical traces in the frequency domain. Security has rarely been systematically analyzed.

**Key Challenge**: A fundamental tension exists between extremely high embedding capacity (an entire image) and steganographic security (undetectability)—the greater the amount of embedded information, the higher the likelihood of leaving statistical traces.

**Goal**: To systematically analyze the detectability of INN-based image steganography and demonstrate, using simple and interpretable methods, that its security claims are unreliable.

**Key Insight**: The INN embedding process is viewed through a signal-processing lens as a mixing process, which is then analyzed using classical blind source separation (ICA).

**Core Idea**: The INN steganographic embedding mixes cover and payload signals in the wavelet domain; ICA can separate the resulting independent components, whose statistical moment differences distinguish cover from stego images.

## Method

### Overall Architecture

DWT (wavelet transform) → PCA (weak component selection) → ICA (blind source separation) → statistical feature extraction (first four moments) → SVM classifier. The entire pipeline is minimal: a three-level Haar wavelet transform yields 12 subbands (4 subbands × 3 color channels); PCA selects weak components likely to encode embedding modifications; FastICA extracts two independent components; and a Gaussian-kernel SVM is trained on 8-dimensional features comprising the mean $\mu$, standard deviation $\sigma$, skewness $\gamma$, and kurtosis $\kappa$ of each component.

### Key Designs

1. **Mixing Process Analysis and ICA Blind Source Separation**:
    - Function: Demonstrates that the INN embedding process in the DWT domain constitutes a separable mixing process, and employs ICA to extract independent components corresponding to the cover and payload.
    - Mechanism: The correlation matrix between stego embedding changes (DWT differences between stego and cover) and payload subbands is computed, showing that embedding changes are strongly correlated with low-frequency payload components—embedding is not random noise but a structured modification carrying payload semantic information. Although only a single stego image (one mixture) is available, the mixing ratios of cover and payload differ across the 12 DWT subbands, enabling these subbands to serve as the multiple mixed observations required by ICA. FastICA extracts two independent components by maximizing non-Gaussianity.
    - Design Motivation: Recasting steganalysis as a classical blind source separation problem provides a theoretically grounded and interpretable framework.

2. **PCA Weak Component Selection and Minimal Statistical Features**:
    - Function: PCA preprocessing selects components most likely to contain embedding traces; minimal features are then extracted from the ICA output.
    - Mechanism: PCA is applied to the 12 DWT subbands; principal components capture the structural content of the cover image, while weak components (e.g., the 9th and 11th, explaining only 0.03% and 0.01% of variance, respectively) are more likely to encode embedding modifications. Only these weak components are passed to ICA for more meaningful independent component extraction. From the two ICA components $c_1, c_2$, only the first four statistical moments are computed, yielding 8 features in total.
    - Design Motivation: This minimal feature design maximizes interpretability—cover and stego images exhibit systematic differences in the statistical moments of their ICA components, and each feature carries a clear signal-processing meaning.

### Loss & Training

No deep learning training is required. The entire detection pipeline relies solely on classical signal processing (DWT + PCA + ICA) combined with a Gaussian-kernel SVM. Five-fold cross-validation is used with balanced cover and stego samples. Experimental data consist of 2,500 512×512 stego images generated from the COCO dataset. SRM+SVM (34,671-dimensional features) is also evaluated as a classical steganalysis baseline.

## Key Experimental Results

### Main Results

| Scheme | Architecture | Ours (ICA) | SRM+SVM |
|--------|-------------|-----------|---------|
| PRIS | INN | **84.62%** | **99.96%** |
| DeepMIH | INN | 82.58% | 99.92% |
| HiNet | INN | 80.31% | 99.02% |
| Weng | CNN | 74.96% | 99.64% |
| Baluja | CNN | 61.83% | 80.06% |

### Ablation Study

| Dimension | Finding | Remark |
|-----------|---------|--------|
| INN vs. non-INN schemes | INN schemes are more detectable by ICA | INNs operate directly in the DWT domain, yielding more regular mixing structures |
| PCA component selection | Components 9 and 11 are optimal | Weak components encode embedding modifications; principal components are dominated by cover structure |
| Robustness of Baluja to ICA | Only 61.83% | Pixel-domain schemes are comparatively more secure (though SRM still reaches 80%) |
| Keyless extraction vulnerability | PSNR drops only 0.08 dB | Setting the extraction network's noise input to a zero vector suffices to recover the secret image |

### Key Findings

- INN-based schemes (HiNet, PRIS, DeepMIH) are highly detectable under both methods, rendering their security claims unreliable.
- Even the 8-dimensional minimal feature set achieves 84.6% accuracy, clearly demonstrating that the embedding process leaves significant statistical traces.
- Classical SRM+SVM easily exceeds 99%, indicating that these schemes fall far short of the security standards achieved by traditional steganography.
- Keyless extraction constitutes an additional fundamental vulnerability: PSNR for PRIS degrades by only 0.08 dB under zero-noise input.

## Highlights & Insights

- Theoretically precise framing: Identifying INN steganography as a linear mixing process reveals its fundamental fragility from a signal-processing perspective, rather than relying on brute-force deep learning detection.
- Minimal yet effective: 8 features + SVM achieving 84.6% is sufficient to invalidate the "security" claims of these schemes.
- Strong interpretability: DWT separates frequencies → PCA selects weak signals → ICA separates mixed sources → statistical moments capture distributional differences; each step has clear intuition.
- Multi-level validation: The dedicated ICA method (high interpretability) and classical SRM+SVM (higher detection rate) together provide complementary evidence of the vulnerability.

## Limitations & Future Work

- Only five steganographic schemes are evaluated, all using publicly available models and weights; generalization to unknown architectures remains unverified.
- The PCA component selection (components 9 and 11) results from a grid search on specific schemes; optimal components may vary across methods.
- Adversarial settings are not considered—it is unclear whether the ICA method remains effective if an undetectability loss is incorporated into the steganographic training objective.
- Experiments are limited to a fixed resolution of 512×512; the effect of varying resolutions is unexplored.
- The ICA method shows limited effectiveness against pixel-domain schemes such as Baluja (61.83%), indicating greater sensitivity to DWT-domain operations.

## Related Work & Insights

- **vs. Peng et al. (ICASSP 2024)**: The only prior work in this area, which trains a surrogate model to extract the payload via supervised learning; it requires substantial training data and lacks interpretability. The proposed method uses unsupervised ICA with a simple SVM, offering far superior interpretability.
- **vs. Security claims of HiNet/PRIS**: These papers report low detection rates using specifically configured classical steganalysis methods, yet SRM+SVM achieves 99%+—their security claims are not credible.
- **vs. Traditional steganography (e.g., LSB)**: Traditional schemes operate at much lower embedding rates and benefit from mature security designs such as adaptive embedding. This paper demonstrates that high-capacity schemes must pay a fundamental security cost.

## Rating

- Novelty: ⭐⭐⭐⭐ Analyzing INN steganographic security from an ICA/blind source separation perspective is a genuinely novel angle; the tools are classical, but their combination is innovative.
- Experimental Thoroughness: ⭐⭐⭐ Comparison across five schemes is adequate, but generalization experiments across different resolutions and datasets are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is concise and well-structured; the motivation at each step is clearly articulated and the overall readability is excellent.
- Value: ⭐⭐⭐⭐ The work carries an important cautionary message for the steganographic security community and is likely to motivate the design of more secure steganographic schemes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[CVPR 2026\] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[CVPR 2026\] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)
- [\[CVPR 2026\] SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)

</div>

<!-- RELATED:END -->
