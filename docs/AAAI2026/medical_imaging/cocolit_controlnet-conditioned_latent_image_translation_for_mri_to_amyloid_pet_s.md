---
title: >-
  [Paper Note] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis
description: >-
  [AAAI 2026][Medical Imaging][MRI-to-PET synthesis] This paper proposes CoCoLIT, a ControlNet-conditioned latent diffusion framework for synthesizing amyloid PET images from structural MRI. Through a Weighted Image Space Loss (WISL) and Latent Averaging Stabilization (LAS), CoCoLIT substantially outperforms existing methods.
tags:
  - AAAI 2026
  - Medical Imaging
  - MRI-to-PET synthesis
  - latent diffusion model
  - ControlNet
  - Alzheimer's disease
  - amyloid
date: 2026-05-08
content_hash: 04e5733cbd58cc9a
---

# CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis

**Conference**: AAAI 2026
**arXiv**: [2508.01292](https://arxiv.org/abs/2508.01292)
**Code**: [GitHub](https://github.com/brAIn-science/CoCoLIT)
**Area**: Medical Imaging
**Keywords**: MRI-to-PET synthesis, latent diffusion model, ControlNet, Alzheimer's disease, amyloid

## TL;DR

This paper proposes CoCoLIT, a ControlNet-conditioned latent diffusion framework for synthesizing amyloid PET images from structural MRI. Through a Weighted Image Space Loss (WISL) and Latent Averaging Stabilization (LAS), CoCoLIT substantially outperforms existing methods.

## Background & Motivation

- Early diagnosis of Alzheimer's disease (AD) relies on amyloid PET (Aβ PET), yet PET imaging is costly, involves radiation exposure, and has limited accessibility.
- Structural MRI is inexpensive and non-invasive but cannot directly detect Aβ deposition; evidence suggests MRI may encode information correlated with amyloid accumulation.
- Synthesizing PET from MRI would enable large-scale, low-cost AD screening, but the high dimensionality and structural complexity of 3D neuroimaging data pose significant challenges.
- Limitations of prior work:
    - GAN-based methods (3D-cGAN, pix2pix) suffer from training instability and mode collapse.
    - FICD performs diffusion in 3D image space, incurring prohibitive computational cost.
    - PASTA operates on 2D slices only, failing to capture inter-slice dependencies.
    - IL-CLDM embeds Aβ-positive labels during training but these labels are unavailable at inference, creating a train–inference mismatch.

## Method

### Overall Architecture

CoCoLIT employs a staged training pipeline consisting of five building blocks:

1. **Blocks A & B — Representation Learning**: MRI VAE and PET VAE are trained independently (fine-tuned from the MAISI VAE).
    - MRI VAE: encoder $\mathcal{E}^{(x)}$ maps MRI volume $x \in \mathbb{R}^D$ to $z^{(x)} \in \mathbb{R}^d$; decoder $\mathcal{D}^{(x)}$ reconstructs the input.
    - PET VAE: encoder $\mathcal{E}^{(y)}$ encodes PET to $z^{(y)} \in \mathbb{R}^d$; decoder $\mathcal{D}^{(y)}$ reconstructs the input.
    - Training losses include reconstruction, perceptual, adversarial terms, and KL regularization.

2. **Block C — Unconditional LDM**: An unconditional latent diffusion model is trained on the PET latent space to learn the distribution of $z^{(y)}$.

3. **Block D — ControlNet-Conditioned Generation**:
    - The LDM backbone is frozen; a ControlNet module is trained to learn the conditional distribution $p(z^{(y)}|z^{(x)})$.
    - Zero-initialized convolutional layers inject the MRI conditioning signal into each layer of the U-Net.
    - The PET VAE decoder is simultaneously fine-tuned using WISL.

4. **Block E — Inference**: Final PET predictions are produced via DDIM sampling (50 steps) combined with LAS.

### Key Designs

**Weighted Image Space Loss (WISL)**:

A core contribution that introduces image-space supervision beyond the latent-space objective:

$$\mathcal{L}_{WISL} = \mathbb{E}_{t, z_t^{(y)}, \epsilon} [\lambda_t \| y - \mathcal{D}^{(y)}(\hat{z}_0^{(y)}) \|_1]$$

- $\hat{z}_0^{(y)}$ is the fully denoised latent estimated from the noisy latent $z_t^{(y)}$.
- $\lambda_t = (T-t)/T$ is a linear timestep weight: at high $t$, the loss prioritizes low-frequency synthesis; at low $t$, it emphasizes high-frequency detail reconstruction.
- Compared to a constant-weight ISL, WISL aligns supervision with the progressive refinement process of diffusion denoising, avoiding the premature imposition of fine-detail constraints.

**Latent Averaging Stabilization (LAS)**:

$m$ latent vectors are sampled from the conditional distribution, averaged, and decoded in a single forward pass through the decoder:

$$\hat{y} = \mathcal{D}^{(y)}(\bar{z}^{(y)}), \quad \bar{z}^{(y)} = \frac{1}{m}\sum_{j=1}^m z^{(y,j)}$$

- The paper provides the first theoretical analysis of LAS: a second-order Taylor expansion demonstrates that LAS is a biased estimator.
- The bias is approximately $(1/m - 1) \cdot \frac{1}{2} \text{Tr}(H_{\mathcal{D}^{(y)}} \Sigma_{z^{(y)}})$.
- The key assumption is that the latent distribution is sufficiently concentrated in a well-trained generative model, making the decoder approximately linear and the bias negligible.
- Experimental validation: PCC = 0.9994 ± 0.0015, confirming the local linearity assumption of the decoder.

### Loss & Training

- Total loss for the ControlNet stage: $\mathcal{L}_{WCN} = \mathcal{L}_{WISL} + \mathcal{L}_{CN}$
- The PET VAE decoder weights are allowed to be fine-tuned during this stage, as WISL is decoder-dependent.
- Inference uses DDIM sampling with 50 steps and LAS with $m=64$.
- All training and experiments are conducted on NVIDIA A100 GPUs.

## Key Experimental Results

### Main Results (Comparison with SOTA)

**Datasets**: ADNI (1,515 pairs, 787 subjects) + A4 external test set (350 pairs)

| Method | SSIM↑ | PSNR↑ | MSE↓ | CABC↑ | HABC↑ | BA↑ |
|--------|-------|-------|------|-------|-------|-----|
| pix2pix | 0.693 | 13.97 | 0.0416 | 0.178 | 0.363 | 51.8% |
| FICD | 0.678 | 12.66 | 0.0549 | 0.049 | 0.193 | 48.2% |
| IL-CLDM | 0.718 | 18.99 | 0.0131 | -0.062 | 0.280 | 46.0% |
| PASTA | 0.860 | 21.63 | 0.0076 | -0.006 | 0.378 | 51.6% |
| **CoCoLIT** | **0.896** | **24.14** | **0.0050** | **0.328** | **0.522** | **62.3%** |

CoCoLIT achieves even stronger performance on the external test set: SSIM = 0.940, BA = 79.8% (+23.7%).

### Ablation Study

**Effect of LAS parameter $m$** (internal test set):

| m | SSIM | PSNR | BA |
|---|------|------|----|
| 1 | 0.865 | 22.57 | 57.4% |
| 8 | 0.892 | 23.94 | 57.1% |
| 64 | **0.896** | **24.14** | **62.3%** |

**Component ablation**:

| Configuration | SSIM | PSNR | BA |
|---------------|------|------|----|
| Base | 0.841 | 21.25 | 43.9% |
| + ISL | 0.870 | 22.45 | 58.5% |
| + WISL | 0.865 | 22.57 | 57.4% |
| + LAS + ISL | 0.896 | 24.03 | 56.7% |
| + LAS + WISL | **0.896** | **24.14** | **62.3%** |

### Key Findings

- The timestep-adaptive weighting in WISL outperforms constant-weight ISL on Aβ-related metrics, owing to its alignment with the denoising process.
- LAS achieves near-identical performance to an unbiased estimator (SSIM difference < 0.001) while being computationally far more efficient.
- Performance on the external dataset is higher than on the internal set, possibly due to smoother SUVR signals in the A4 data.

## Highlights & Insights

1. **First application of ControlNet to MRI-to-PET translation**: The frozen-backbone plus trainable-copy paradigm efficiently injects conditioning signals.
2. **Design philosophy of WISL**: Aligning image-space supervision with the progressive nature of the diffusion process proves more effective than a naive image-space loss.
3. **Theoretical contribution of LAS**: The first statistical analysis of LAS demonstrates that its bias is negligible in well-trained models.
4. **Substantial performance gains**: CoCoLIT leads the second-best method by +10.5% in Aβ-positive classification BA on the internal set and +23.7% on the external set.

## Limitations & Future Work

- The best reported BA of 62.3% (internal) remains insufficient for reliable clinical deployment.
- LAS requires $m$ sampling passes, which still incurs computational cost without GPU parallelism.
- Validation is limited to Florbetapir PET; generalization to other PET tracers has not been tested.
- The framework is extensible to other conditional generation tasks, including disease progression modeling and image quality transfer.

## Related Work & Insights

- This work provides a successful case study of applying latent diffusion models to 3D medical image synthesis.
- The strategy for adapting ControlNet to the medical domain is noteworthy: it requires no modification to the pre-trained model architecture.
- The timestep-adaptive weighting concept underlying WISL is generalizable to other conditional diffusion generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — ControlNet + LDM for 3D cross-modal synthesis; WISL is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Internal and external datasets, comprehensive ablations, and theoretical validation.
- Writing Quality: ⭐⭐⭐⭐ — Theory and experiments are tightly integrated.
- Value: ⭐⭐⭐⭐ — Offers a viable pathway toward low-cost AD screening.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](../../CVPR2026/medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)
- [\[AAAI 2026\] Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model](virtual_multiplex_staining_for_histological_images_using_a_marker-wise_condition.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)

</div>

<!-- RELATED:END -->
