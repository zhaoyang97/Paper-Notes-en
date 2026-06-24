---
title: >-
  [Paper Note] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis
description: >-
  [ECCV 2024][Image Generation] Identifies the misalignment issue between training and testing latent code distributions in IMLE. Proposes RS-IMLE, which alters the training prior distribution via rejection sampling, achieving an average FID reduction of 45.9% across nine few-shot image datasets.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: f99788eff9555b25
---

# Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2409.17439](https://arxiv.org/abs/2409.17439)  
**Area**: Image Generation

## TL;DR

Identifies the misalignment issue between training and testing latent code distributions in IMLE. Proposes RS-IMLE, which alters the training prior distribution via rejection sampling, achieving an average FID reduction of 45.9% across nine few-shot image datasets.

## Background & Motivation

- Few-shot image synthesis aims to learn deep generative models from extremely limited training samples (e.g., 10-100 images).
- GANs suffer severely from mode collapse in few-shot scenarios, while diffusion models perform poorly under few-shot settings due to learning an isotropic Gaussian mixture.
- IMLE (Implicit Maximum Likelihood Estimation) avoids mode collapse by ensuring every training sample has a generated sample close to it.
- **Key Discovery**: In existing IMLE methods, there is a significant misalignment between the distribution of latent codes selected via the min operation during training and the distribution sampled from a standard Gaussian during testing (misalignment issue).
- This leads to large "blank regions" in the latent space that are uncovered by training, resulting in low-quality samples when sampled during testing.

## Method

### Overall Architecture

The core idea of RS-IMLE is to change the training prior distribution (from a standard Gaussian to a new distribution $\mathcal{P}$ via rejection sampling) to align the latent code distributions between training and testing.

### Key Designs

**1. Theoretical analysis of the misalignment issue**

In IMLE, the CDF of the distance after selecting the nearest sample is:

$$F_{D_i^*}(t) = 1 - (1 - F_{D_{i1}}(t))^m$$

where $m$ is the number of samples. This indicates that the distance distribution of selected codes is biased towards smaller values compared to random codes, and the bias becomes more severe as $m$ increases.

Corresponding PDF:

$$f_{D_i^*}(t) = m(1-F_{D_{i1}}(t))^{m-1} f_{D_{i1}}(t)$$

**2. Ideal prior design**

The goal is to find a prior $\mathcal{P}$ such that the distribution selected via the min operation after sampling $m$ samples from $\mathcal{P}$ is identical to the selection distribution after sampling $n$ samples ($n$ is the number of data points) from a standard Gaussian. Deriving this yields the PDF of the target prior:

$$f_{\tilde{D}_{i1}}(t) = \phi(t) \cdot f_{D_{i1}}(t)$$

where $\phi(t) = \frac{n}{m} \frac{(1-F_{D_{i1}}(t))^{n-1}}{(1-F_{\tilde{D}_{i1}}(t))^{m-1}}$

**3. Rejection sampling implementation**

Simplifying the above theory into intuitive rejection sampling steps:
- Sample $\mathbf{z} \sim \mathcal{N}(0, I)$ from a standard Gaussian.
- If $d(\mathbf{x}_i, T_\theta(\mathbf{z})) < \epsilon$ (for all data points), reject the sample.
- Otherwise, accept the sample.

In essence, this excludes samples that are already close to any data point, forcing the model to train on more challenging, non-trivial samples.

**4. Intuitive explanation from a gradient perspective**

- Traditional IMLE: As training converges, the nearest samples become extremely close to the data points, resulting in smaller losses and gradients, which stagnates learning.
- RS-IMLE: By excluding easy samples via a radius $\epsilon$, the loss for each data point is kept $\geq \epsilon$, maintaining meaningful parameter updates.

### Loss & Training

$$\theta_{\text{RS-IMLE}} = \arg\min_\theta \mathbb{E}_{z_1,...,z_m \sim \mathcal{P}} \left[\sum_{i=1}^n \min_{j \in [m]} d(\mathbf{x}_i, T_\theta(\mathbf{z}_j))\right]$$

Constraints: $d(\mathbf{x}_i, T_\theta(\tilde{\mathbf{z}}_j)) \geq \epsilon, \forall \tilde{\mathbf{z}}_j \in \tilde{Z}, i \in [n]$

Uses DCI fast nearest neighbor search to accelerate distance computation, and reduces search complexity via random projection dimensionality reduction.

## Key Experimental Results

### Main Results

FID comparison on nine few-shot datasets (256x256 resolution):

| Dataset | FastGAN | FakeCLR | FreGAN | ReGAN | AdaIMLE | **RS-IMLE** | Gain% |
|--------|---------|---------|--------|-------|---------|-------------|-------|
| Obama | 41.1 | 29.9 | 33.4 | 45.7 | 25.0 | **14.0** | 44.0% |
| Grumpy Cat | 26.6 | 20.6 | 24.9 | 27.3 | 19.1 | **11.5** | 39.8% |
| Panda | 10.0 | 8.8 | 9.0 | 12.6 | 7.6 | **3.5** | 54.0% |
| FFHQ-100 | 54.2 | 62.1 | 50.5 | 87.4 | 33.2 | **12.9** | 61.1% |
| Cat | 35.1 | 27.4 | 31.0 | 42.1 | 24.9 | **15.9** | 36.1% |
| Dog | 50.7 | 44.4 | 47.9 | 57.2 | 43.0 | **23.1** | 46.3% |
| Anime | 69.8 | 77.7 | 59.8 | 110.8 | 65.8 | **35.8** | 45.6% |
| Skulls | 109.6 | 106.5 | 163.3 | 130.7 | 81.9 | **51.1** | 37.6% |
| Shells | 120.9 | 148.4 | 169.3 | 236.1 | 108.5 | **55.4** | 48.9% |

### Ablation Study

Precision and Recall comparison (selected datasets):

| Dataset | Metric | FastGAN | FakeCLR | FreGAN | ReGAN | AdaIMLE | **RS-IMLE** |
|--------|------|---------|---------|--------|-------|---------|-------------|
| Obama | Prec | 0.92 | 0.96 | 0.82 | 0.62 | 0.99 | 0.98 |
| Obama | Rec | 0.09 | 0.30 | 0.33 | 0.01 | 0.68 | **High** |
| Grumpy Cat | Prec | — | — | — | — | — | Close to 1.0 |
| FFHQ-100 | Prec | — | — | — | — | — | Close to 1.0 |

**Core Conclusion**: RS-IMLE maintains near-perfect precision while achieving significantly higher recall than all baseline methods.

Convergence comparisons on 2D toy problems confirm:
- In IMLE, selected latent codes during training form tight, narrow bands with substantial uncovered gaps in between.
- In RS-IMLE, selected latent codes during training cover the prior distribution more uniformly, aligning with the test-time distribution.

### Key Findings

1. **Average FID decreases by 45.9%**, achieving SOTA on all nine datasets.
2. The most significant improvement is observed on FFHQ-100 (61.1%), dropping from 33.2 to 12.9.
3. Precision is close to 1.0 (high generation quality), while Recall far exceeds GAN-based methods (good mode coverage).
4. The alignment issue is intuitively validated through 2D toy experiments: diffusion models fail to learn the data manifold in few-shot settings (Gaussian mixture failure), whereas RS-IMLE resolves it effectively.
5. The hyperparameter $\epsilon$ is selected via cross-validation, and the method is not overly sensitive to it.

## Highlights & Insights

- **Theory-driven methodological improvement**: Rigorously derives the root cause of training/testing distribution misalignment from the perspective of order statistics, rather than heuristic modifications.
- **Elegant simplification of rejection sampling**: Complex theoretical derivations ultimately simplify into a simple and intuitive $\epsilon$-ball exclusion rule.
- **Orthogonal to prior IMLE improvements**: RS-IMLE is complementary to Adaptive IMLE and can be combined.
- **Failure cases of diffusion models in few-shot settings**: Clearly demonstrates the fundamental flaws of the isotropic Gaussian assumption in diffusion models when applied to extremely few samples.

## Limitations & Future Work

- Rejection sampling reduces sampling efficiency, requiring extra sampling to compensate for rejected samples.
- The selection of $\epsilon$ requires cross-validation; different datasets may require different values.
- The generator architecture is based on the decoder of VDVAE, leaving more modern architectures unexplored.
- Validated only on 256x256 resolution, without extension to high-resolution generation.
- No comparison with few-shot methods based on fine-tuning pre-trained models (transfer learning).

## Rating

- Innovation: ⭐⭐⭐⭐⭐ — Breakthroughs in both theory and methodology
- Practicality: ⭐⭐⭐⭐ — Highly valuable for few-shot scenarios
- Performance: ⭐⭐⭐⭐⭐ — SOTA on all datasets, with an average improvement of 45.9%
- Overall Score: 9/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)

</div>

<!-- RELATED:END -->
