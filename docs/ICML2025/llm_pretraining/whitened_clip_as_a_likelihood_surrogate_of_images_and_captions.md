---
title: >-
  [Paper Note] Whitened CLIP as a Likelihood Surrogate of Images and Captions
description: >-
  [ICML 2025][LLM Pretraining][CLIP] Proposes Whitened CLIP (W-CLIP), which applies an invertible PCA whitening transformation to CLIP embeddings to approximate an i.i.d. standard normal distribution. This allows for direct estimation of image and caption log-likelihoods using the squared Euclidean norm, demonstrating effectiveness in artifact detection, domain shift analysis, and full-circle SLERP image manipulation.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "CLIP"
  - "Whitening Transformation"
  - "Likelihood Surrogate"
  - "Isotropy"
  - "OOD Detection"
date: 2026-05-08
content_hash: 606097b78d5392a2
---

# Whitened CLIP as a Likelihood Surrogate of Images and Captions

**Conference**: ICML 2025  
**arXiv**: [2505.06934](https://arxiv.org/abs/2505.06934)  
**Code**: Yes (Link provided in the paper)  
**Area**: LLM Pre-training  
**Keywords**: CLIP, Whitening Transformation, Likelihood Surrogate, Isotropy, OOD Detection

## TL;DR

Proposes Whitened CLIP (W-CLIP), which applies an invertible PCA whitening transformation to CLIP embeddings to approximate an i.i.d. standard normal distribution. This allows for direct estimation of image and caption log-likelihoods using the squared Euclidean norm, demonstrating effectiveness in artifact detection, domain shift analysis, and full-circle SLERP image manipulation.

## Background & Motivation

**Difficulty of Image Likelihood Estimation**: Computing image likelihood $P(X)$ is a fundamental problem in computer vision. However, existing methods (such as diffusion models) can only approximate the score function $\nabla_x \log P(X)$, and generative models like GANs, VAEs, and EBMs only implicitly estimate the distribution, making it impossible to directly obtain $P(X)$.

**Structural Problems in CLIP Space**: CLIP maps image and text embeddings into a shared space, widely used for image-text matching. However, its embedding space suffers from two known defects: the **Narrow Cone Effect** (embeddings are concentrated in a narrow angular range) and the **Modality Gap** (the distributions of image and text embeddings do not intersect), limiting its utility as a probability estimator.

**Core Idea**: Apply a whitening transformation (zero mean + identity variance) to CLIP embeddings to transform the original ellipsoidal space into a hypersphere. Under the standard normal assumption, the log-likelihood can be directly estimated by the squared Euclidean norm in the whitened space: $\ell(x) = -\frac{1}{2}(d\log(2\pi) + \|x\|^2)$. This transformation requires no training at all and relies solely on a precomputed whitening matrix. This represents the first method to provide direct probability computation for images based on high-level semantics.

## Method

### Overall Architecture

The pipeline of W-CLIP is extremely simple: (1) Compute the mean $\mu$ and covariance matrix $\Sigma$ of CLIP embeddings on a representative dataset (such as 5000 images from the MS-COCO validation set); (2) Obtain the whitening matrix $W = \Lambda^{-1/2}V^\top$ via PCA decomposition $\Sigma = V\Lambda V^\top$; (3) Compute the whitened embedding $y = W(x - \mu)$ for any new sample's CLIP embedding $x$; (4) Estimate the likelihood using $\ell(x) = -\frac{1}{2}(d\log(2\pi) + \|y\|^2)$. The image and text modalities are whitened independently, and the whitening matrix only needs to be precomputed once before reuse. The CLIP ViT-L/14 model ($d=768$) is utilized.

### Key Designs

1. **PCA Whitening Transformation**:
    - **Function**: Transforms CLIP embeddings from an anisotropic ellipsoidal distribution into an isotropic hyperspherical distribution.
    - **Mechanism**: Given the covariance matrix $\Sigma = V\Lambda V^\top$, the whitening matrix is $W = \Lambda^{-1/2}V^\top$, and the whitened embedding $y = W\hat{x}$ satisfies $\mu_Y = 0, \Sigma_Y = I$. The transformation is invertible, allowing the original space to be recovered via $x = W^{-1}y + \mu$. The Diagonal Score verifies that the post-whitened covariance is almost perfectly diagonalized, meaning uncorrelation is equivalent to independence under the normality assumption.
    - **Design Motivation**: Whitening is the only linear transformation that simultaneously achieves zero mean, identity variance, and decorrelation. It is entirely data-driven with zero hyperparameters, incurs extremely low computational overhead, and its invertibility ensures that original CLIP functionalities remain unaffected.

2. **Norm-Likelihood Mapping and Normality Validation**:
    - **Function**: Statistically validates that whitened embeddings approximate a standard normal distribution, establishing an exact mapping between the norm and likelihood.
    - **Mechanism**: Uses both Anderson-Darling (focusing on tail deviation) and D'Agostino-Pearson (combining skewness and kurtosis) tests. More than 98% of image embedding features pass the normality test, and over 90% of text features pass. The norm follows a chi distribution $\chi_d$ with an expectation of $\mathbb{E}[S] = \sqrt{2}\frac{\Gamma(\frac{d+1}{2})}{\Gamma(\frac{d}{2})} \approx \sqrt{d - 1/2}$. When $d=768$, the theoretical value is 27.7, and the empirically measured mean of image embeddings is 27.43 (a deviation of only 0.98%).
    - **Design Motivation**: Only after validating the correctness of the normality assumption can the norm serve as a reliable likelihood surrogate. The high alignment between empirical and theoretical values confirms the statistical foundation of the method.

3. **Full-Circle Spherical Linear Interpolation (Full-Circle SLERP)**:
    - **Function**: Extends standard SLERP from $t \in [0,1]$ to a full $360°$, enabling interpolation and extrapolation between images.
    - **Mechanism**: Given the interpolation angle $\omega$, let $t = \omega/\theta$, and substitute into the SLERP formula $\text{SLERP}(t; E_1, E_2) = \frac{\sin((1-t)\theta)}{\sin\theta}E_1 + \frac{\sin(t\theta)}{\sin\theta}E_2$. While original CLIP space generates noise at $180°$, W-CLIP generates natural images across all angles. The "antipodal embedding" at $180°$ is determined solely by the source image, serving as its fixed symmetric counterpart.
    - **Design Motivation**: The Narrow Cone effect in CLIP causes embeddings to deviate from the hypersphere, causing SLERP to fail when exceeding the interpolation interval. Whitening distributes embeddings uniformly over the hypersphere, ensuring all directions remain in-distribution.

### Loss & Training

W-CLIP is entirely training-free. The whitening matrix $W$ and mean $\mu$ are precomputed once on a representative dataset. Cross-dataset generalization validation shows that after swapping the whitening and testing roles of MS-COCO and Flickr8k, the likelihood correlation still reaches 0.69-0.88.

## Key Experimental Results

### Main Results: Normality Distribution Tests

| Test Method | Modality | Mean Score | Passing Ratio | Threshold |
|---------|------|---------|---------|------|
| Anderson-Darling | Image | 0.489| 98.3% | < 0.752 |
| Anderson-Darling | Text | 0.593 | 90.1% | < 0.752 |
| D'Agostino-Pearson | Image | 0.362 | 99.3% | > 0.05 |
| D'Agostino-Pearson | Text | 0.257 | 99.2% | > 0.05 |

### Empirical vs. Theoretical Comparison ($d=768$)

| Modality | Mean (Empirical / Theoretical) | Std (Empirical / Theoretical) |
|------|----------------|-------------------|
| Image | 27.43 / 27.7 (Deviation 0.98%) | 3.94 / 3.96 (Deviation 0.55%) |
| Text | 28.49 / 27.7 (Deviation 2.85%) | 5.72 / 6.60 (Deviation 13.24%) |

### Full-Circle SLERP Antipodal Image Quality

| Method | Total Variation | Entropy | Saturated Pixel Ratio |
|------|----------------|---------|-------------|
| Real MS-COCO Images | 222.3 | 7.3 | 4.2% |
| CLIP Antipodal Images | 156.7 | 4.8 | 55.5% |
| W-CLIP Antipodal Images | 215.9 | 7.2 | 6.4% |

### Likelihood Separation Comparison (AUC)

| Model | Domain Separation (Caption vs. General Text) | Noun-Stripped Separation |
|------|---------------------------|-----------|
| GPT-2 (LLM) | 0.80 | 0.43 |
| OPT (LLM) | 0.80 | 0.58 |
| NEO (LLM) | 0.77 | 0.58 |
| BLIP (VLM) | 0.92 | 0.66 |
| GIT (VLM) | 0.97 | 0.69 |
| **W-CLIP (Ours)** | **0.999** | **0.94** |

### Ablation Study: Cross-Dataset Generalization

| Test Set | Whitened Dataset | Avg. AD | Likelihood Correlation (Image/Text) |
|--------|----------|---------|---------------------|
| COCO | COCO | 0.489 | Baseline |
| COCO | Flickr8k | 0.466 | 0.69 / 0.74 |
| Flickr8k | COCO | 0.641 | 0.77 / 0.88 |
| Flickr8k | Flickr8k | 0.522 | Baseline |

### Key Findings

- The W-CLIP norm can effectively distinguish real images from artifact-containing AI-generated images (all generated images in the SynArtifact dataset show a lower likelihood than their real counterparts).
- ImageNet-C noise level is monotonically positively correlated with the W-CLIP norm (stronger noise leading to lower likelihood), and the style shifts in ImageNet-R are well-ordered: graffiti is closest to real, while video game rendering shows the largest shift.
- W-CLIP is extremely sensitive to grammatical errors (noun-stripping) (AUC=0.94), far surpassing all LLMs ($\le 0.58$) and VLMs ($\le 0.69$).
- Text complexity is negatively correlated with likelihood: removing specific words (names/places) $\rightarrow$ likelihood increases, while adding specific words $\rightarrow$ likelihood decreases.
- Generative models (UnCLIP) exhibit a systematic likelihood bias, where embedding norms progressively increase during iterative generation, causing degradation. This can be mitigated by normalizing to $\sqrt{d}$.

## Highlights & Insights

- **Zero-Cost Post-Processing**: The whitening matrix is precomputed once, requiring only a single matrix multiplication during inference, with extremely low memory and computational footprints.
- **Complementarity with Language Models**: W-CLIP is insensitive to text length but highly sensitive to semantic changes (grammar, captions vs. general text), whereas language models behave conversely.
- **Uniformity Improvement**: After whitening, cosine similarity concentrates around zero (with extremely small standard deviation), resolving the issue in the original CLIP where similarity concentrated around 0.5.
- **Invertibility Guarantees Compatibility**: All downstream applications of CLIP can be seamlessly integrated with W-CLIP.

## Limitations & Future Work

- The normal approximation for the text modality is less accurate than that of images (standard deviation deviation is 13.24%), limiting the accuracy of likelihood estimation on the text side.
- Whitening relies on a representative dataset to calculate the covariance, making it sensitive to domain shifts.
- Only the CLIP ViT-L/14 model has been validated; the applicability of other architectures (such as ViT-B/32, OpenCLIP) has not been systematically evaluated.
- The correlation with language model likelihood is only 0.33-0.48, indicating that W-CLIP captures a different dimension of "likelihood".
- The detection of generated images is only analyzed qualitatively in a preliminary manner, lacking large-scale quantitative evaluation and comparison with specialized detectors.

## Related Work & Insights

- Liang et al. (2022) discovered the Modality Gap, and Schrodi et al. (2024) identified the Narrow Cone Effect; whitening simultaneously resolves both of these issues.
- The dual-ellipsoid geometric analysis of Levi & Gilboa (2025) complements this work from a probabilistic perspective.
- Iterative experiments with UnCLIP reveal a systematic likelihood bias in generative models, inspiring the use of W-CLIP for monitoring generation quality.
- Extensible directions: Using W-CLIP likelihood as a quality metric for image generation, an OOD detection baseline, or a sampling guidance signal in conditional generation.

## Rating

- Novelty: ⭐⭐⭐⭐ Uniquely combines the classic whitening operation with CLIP's probabilistic interpretation, presenting a distinct perspective and self-consistent theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Encompasses normality tests, cross-domain generalization ablations, multi-application scenarios, and comparisons with various LLMs/VLMs.
- Writing Quality: ⭐⭐⭐⭐ Features rigorous mathematical derivations, rich tables and figures, and a clear structure.
- Value: ⭐⭐⭐ The method is extremely simple and practical, but the depth of application is limited, making it closer to a preliminary exploration than a fully matured tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Double-Ellipsoid Geometry of CLIP](the_double-ellipsoid_geometry_of_clip.md)
- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](../../CVPR2025/llm_pretraining/seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](../../ICLR2026/llm_pretraining/chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)

</div>

<!-- RELATED:END -->
