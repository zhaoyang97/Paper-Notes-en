---
title: >-
  [Paper Note] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction
description: >-
  [CVPR 2026][Radio interferometric imaging] POLISH++ extends the POLISH framework by introducing a patch-wise training-and-stitching strategy and an arcsinh nonlinear transformation…
tags:
  - "CVPR 2026"
  - "Radio interferometric imaging"
  - "deep learning deconvolution"
  - "super-resolution"
  - "high dynamic range"
  - "strong gravitational lensing"
date: 2026-05-08
content_hash: c02cfa87dd0c9ef8
---

# POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.09162](https://arxiv.org/abs/2603.09162)  
**Code**: None (extended from [POLISH](https://github.com/liamconnor/polish-pub))  
**Area**: Other
**Keywords**: Radio interferometric imaging, deep learning deconvolution, super-resolution, high dynamic range, strong gravitational lensing

## TL;DR
POLISH++ extends the POLISH framework by introducing a patch-wise training-and-stitching strategy and an arcsinh nonlinear transformation, addressing two major practical deployment challenges in radio interferometric imaging: wide-field imaging (images exceeding ten thousand pixels) and high dynamic range ($10^4$–$10^6$). On T-RECS simulated data, POLISH++ substantially outperforms CLEAN in source detection accuracy, recovers strong gravitational lens systems near the PSF scale through super-resolution, and is projected to increase the number of gravitational lens discoveries in DSA surveys by approximately one order of magnitude.

## Background & Motivation
Radio interferometric imaging synthesizes a large aperture from antenna arrays to achieve high angular resolution, which is fundamentally an image deconvolution problem (recovering the true sky from a dirty image). The classical CLEAN method assumes a point-source model and iteratively subtracts the PSF; its resolution is bounded by the PSF and it cannot handle sources with complex morphologies. Deep learning methods such as the original POLISH have demonstrated efficient inference and super-resolution capabilities, yet existing DL approaches leave three practical deployment problems unsolved: (1) test images are small (<1000 pixels), whereas next-generation telescopes such as DSA must handle images exceeding $10000 \times 10000$ pixels; (2) validation has been limited to low dynamic range (<$10^3$), while real-world dynamic ranges can reach $10^6$; (3) the assumption that training and testing conditions are matched does not hold in practice due to atmospheric effects, calibration errors, and PSF variation.

## Core Problem
How can deep learning methods for radio interferometric imaging be made truly deployable on next-generation survey telescopes such as DSA? Three specific challenges must be resolved: (1) GPU memory cannot accommodate training and inference on extremely large images; (2) extreme dynamic range prevents networks from simultaneously learning to reconstruct bright and faint sources; (3) robustness to PSF train–test mismatch. Together, these issues prevent DL methods from transitioning from the laboratory to real astronomical observations.

## Method

### Overall Architecture
POLISH++ is an end-to-end CNN model based on the WDSR architecture. It takes a low-resolution dirty image as input and outputs a high-resolution clean sky, achieving 2× super-resolution. The overall pipeline consists of three steps: (1) apply an arcsinh transformation to the input dirty image to compress the dynamic range; (2) perform CNN forward inference to obtain the reconstruction in the transformed space; (3) apply the inverse arcsinh transformation to recover the original intensity space. Both training and inference operate at the patch level; at inference time, patches are stitched back into the full field of view. The model is progressively extended from POLISH to POLISH+ (adding patch-wise training) and POLISH++ (adding the arcsinh transformation).

### Key Designs
1. **Patch-Wise Processing**: The full-field image of $12960 \times 12960$ pixels is divided into $J$ non-overlapping patches of $324 \times 324$ pixels, forming training pairs $\{(I_{\text{dirty}}^{[j]}, I_{\text{true}}^{[j]})\}$. A key insight is that a patch extracted from a full-field dirty image is not equivalent to directly applying the forward model to the ground-truth patch of that region — bright sources outside the patch boundary "contaminate" the patch's dirty image through PSF sidelobes, introducing cross-patch artifacts. POLISH++ lets the network implicitly learn to handle this cross-patch contamination rather than relying on explicit physical modeling. Eighteen full-field images yield 28,800 training patches containing approximately six million detectable galaxy samples.

2. **Arcsinh Dynamic Range Transformation**: The nonlinear transformation is defined as $\text{AsinhStretch}(x; a) = \frac{\text{arcsinh}(x/a)}{\text{arcsinh}(1/a)}$, whose logarithmic-like behavior compresses pixel values spanning multiple orders of magnitude into a common scale. Compared with photographic gamma encoding, arcsinh handles both positive and negative values (dirty images contain negative values), making it particularly suitable for radio images. Training is performed in the transformed space, and intensities are recovered after inference via the inverse transform $a \cdot \sinh(x \cdot \text{arcsinh}(1/a))$. The parameters are set to $a_{\text{dirty}} = a_{\text{true}} = 0.1$.

3. **Robustness and Adaptability Under PSF Mismatch**: The model is trained exclusively on ideal PSFs, yet it maintains visually stable reconstructions when tested on randomly distorted PSFs (perturbation strength $\gamma \in [0, 30]$). Although PSNR degrades as the perturbation increases, practical source detection quality remains acceptable. Moreover, fine-tuning to a new PSF distribution reaches the optimum in only 11 epochs, compared with 57 epochs when training from scratch — a speedup exceeding 5×, which is critical for rapid deployment across different pointing directions of DSA.

### Loss & Training
- $\ell_1$ loss computed in the arcsinh-transformed space: $\theta^* = \arg\min_\theta \frac{1}{NJ}\sum_{i,j}\|G_\theta(\text{AsinhStretch}(I_{\text{dirty}}^{[j]}; a_d)) - \text{AsinhStretch}(I_{\text{true}}^{[j]}; a_t)\|_1$
- Adam optimizer, learning rate $10^{-4}$, batch size 12
- 18 full-field images for training, 5 for testing, yielding 28,800 patches
- Training data generated by the T-RECS simulator, containing AGN (point sources) and SFRGs (elliptical Sérsic profiles), with noise standard deviation 1 μJy

## Key Experimental Results

| Method | Precision | Recall | F1 Score | Major Axis RMSE (″) | Minor Axis RMSE (″) |
|--------|-----------|--------|----------|---------------------|---------------------|
| CLEAN | 0.3612 | 0.2220 | 0.2750 | 1.0046 | 0.7862 |
| POLISH | 0.5560 | 0.4612 | 0.5042 | 0.9642 | 0.3219 |
| POLISH+ | 0.8744 | 0.5751 | 0.6938 | 0.4335 | 0.1889 |
| POLISH++ | 0.8433 | 0.6142 | 0.7107 | 0.4654 | 0.2056 |

**Strong Gravitational Lens Discovery**: At FPR=$10^{-3}$, the CNN lens finder based on POLISH/POLISH++ recovers lens systems with Einstein radii close to the PSF scale (CLEAN can only recover systems more than 3× the PSF size), and is projected to increase the number of gravitational lens discoveries by DSA by approximately one order of magnitude. The TPR of POLISH on lenses approaches the ground-truth upper bound.

**PSF Robustness**: Even under extreme PSF perturbation ($\gamma=30$), reconstruction visual quality remains stable. Fine-tuning to a new PSF requires only 11 epochs vs. 57 epochs from scratch (5× speedup).

### Ablation Study
- **Patch-wise training (POLISH→POLISH+)**: Precision increases from 0.56 to 0.87 and F1 from 0.50 to 0.69, demonstrating that learning at the patch level — including cross-patch artifacts — substantially outperforms training directly on small images.
- **Arcsinh transformation (POLISH+→POLISH++)**: Recall improves from 0.58 to 0.61 (+4%) and F1 from 0.69 to 0.71, with the primary contribution being the recovery of more faint sources and an improved precision–recall balance.
- **CLEAN provides superior flux estimation**: This is a known limitation of DL methods — nonlinear reconstruction lacks an explicit flux calibration mechanism, whereas CLEAN maintains absolute flux accuracy within its model-based framework.

## Highlights & Insights
- **The "cross-patch contamination" insight**: The paper is the first to clearly articulate that, when training radio imaging networks at the patch level, the dirty-image patch contains PSF sidelobe artifacts from bright sources in neighboring patches — this is fundamentally different from training on small standalone images. Teaching the network to handle signals that "do not belong" to a given patch constitutes an elegant form of implicit physical modeling.
- **Astronomical suitability of the arcsinh transformation**: Compared with log transformation, arcsinh handles negative values; compared with gamma encoding, it has a well-defined inverse and clear physical motivation. A single simple nonlinear transformation resolves dynamic range challenges of up to $10^6$.
- **Task-relevant evaluation**: The paper evaluates not only image quality metrics such as PSNR/SSIM but also source detection accuracy (precision/recall/F1) and shape parameter estimation errors, aligning more closely with real astronomical requirements.
- **Rapid adaptation via fine-tuning**: Adapting to different PSF conditions does not require training from scratch; the 5× speedup of fine-tuning makes per-pointing deployment feasible.

## Limitations & Future Work
- **Flux estimation still inferior to CLEAN**: DL methods lag behind CLEAN in absolute flux accuracy, necessitating dedicated flux calibration steps or post-processing strategies.
- **Evaluation restricted to the image plane**: The CASA deconvolve task (minor cycle) is used, without end-to-end validation in the visibility domain.
- **Limited training data volume**: Only 18 full-field images are used for training; although patch-wise expansion yields 28,800 samples, the diversity of sky models may be insufficient.
- **Spectral dimension not exploited**: Radio interferometric imaging is inherently multi-band; the current method processes only single-band images.
- **Real systematic effects not fully validated**: Although PSF mismatch is tested, the combined impact of real ionospheric disturbances, gain variations, RFI contamination, and similar effects has not been verified.

## Related Work & Insights
- **vs. CLEAN**: POLISH++ surpasses CLEAN by 159% in source detection F1 (0.27→0.71) and reduces shape estimation RMSE by 54%, though flux estimation remains inferior. The fundamental difference is that CLEAN is limited to PSF resolution, whereas POLISH leverages learned priors to achieve super-resolution.
- **vs. R2D2 (Aghabiglou et al., 2024)**: R2D2 is an unrolled iterative optimization network with a maximum image size of 512 pixels and dynamic range of $5 \times 10^5$. POLISH++ scales to 12,960 pixels and $10^6$ dynamic range, and performs single-pass forward inference, making it suitable for high-throughput scenarios.
- **vs. GU-Net/RI-GAN**: These methods are tested on synthetic data with image sizes below 360/512 pixels and dynamic ranges below 600, and do not account for train–test PSF mismatch, leaving a substantial gap from real deployment conditions.

## Related Papers

- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](../../AAAI2026/others/semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[CVPR 2026\] SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)

## Rating
- Novelty: ⭐⭐⭐ — Patch-wise training and the arcsinh transformation are not technically novel in themselves, but their application and engineering-scale deployment in radio astronomical imaging are meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across four dimensions — source detection, shape estimation, gravitational lens discovery, and PSF robustness — with experimental design closely aligned with real astronomical requirements.
- Writing Quality: ⭐⭐⭐⭐ — Background and motivation are clearly and thoroughly presented, but some astronomical terminology may be inaccessible to CV readers.
- Value: ⭐⭐⭐⭐ — Directly applicable to next-generation radio telescopes such as DSA; the projected tenfold increase in gravitational lens discoveries represents a substantive contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](../../AAAI2026/others/semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[CVPR 2026\] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion](omnifood8k_nutrition_estimation.md)

</div>

<!-- RELATED:END -->
