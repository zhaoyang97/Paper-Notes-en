---
title: >-
  [Paper Note] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising
description: >-
  [ECCV 2024][Image Restoration][Image decomposition] This paper proposes DenoiSplit, the first method to jointly address semantic image splitting and unsupervised denoising. By integrating pixel noise models and an improved KL divergence loss weighting strategy into a hierarchical VAE, the method achieves end-to-end denoising and splitting on fluorescence microscopy images, significantly outperforming serial pipelines that perform denoising prior to splitting.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Image decomposition"
  - "unsupervised denoising"
  - "fluorescence microscopy"
  - "variational autoencoder"
  - "noise model"
date: 2026-05-08
content_hash: 34249fc557d909fd
---

# DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising

**Conference**: ECCV 2024  
**arXiv**: [2403.11854](https://arxiv.org/abs/2403.11854)  
**Code**: [Yes (GitHub)](https://github.com/juglab/denoiSplit)  
**Area**: Image Restoration  
**Keywords**: Image decomposition, unsupervised denoising, fluorescence microscopy, variational autoencoder, noise model

## TL;DR

This paper proposes DenoiSplit, the first method to jointly address semantic image splitting and unsupervised denoising. By integrating pixel noise models and an improved KL divergence loss weighting strategy into a hierarchical VAE, the method achieves end-to-end denoising and splitting on fluorescence microscopy images, significantly outperforming serial pipelines that perform denoising prior to splitting.

## Background & Motivation

### Image Splitting in Fluorescence Microscopy

Fluorescence microscopy is a fundamental tool for studying cellular and subcellular structures. However, simultaneously observing multiple structures requires **multiplexed imaging protocols**, which are time-consuming and labor-intensive. **Semantic image splitting** offers an alternative: decomposing a single image containing a superposition of multiple structures into individual, separate channel images.

Formal definition: Given a superimposed image $x_j = c_{1,j} + c_{2,j}$ (the pixel-wise sum of two channels), the goal is to recover $(c_{1,j}, c_{2,j})$ from $x_j$. Although the sum of two numbers cannot be uniquely decomposed, decomposition becomes feasible by learning structural priors if each component possesses distinct structural characteristics (e.g., morphological differences between microtubules and vesicles).

### The Noise Problem: A Critical Weakness of Existing Methods

Prior work, such as μSplit, demonstrated impressive splitting performance on relatively noiseless data. However, in practical microscopy scenarios, images are **invariably corrupted by significant noise** (Poisson-Gaussian noise) due to physical limitations of imaging hardware and constraints related to sample photosensitivity.

When trained on noisy data, μSplit inevitably attributes noise to the predicted outputs, resulting in a phenomenon known as **noise breakthrough**. This occurs because the bottom-most levels of the hierarchical latent space in μSplit (which encode fine, pixel-level structures) lack sufficient constraints to suppress noise.

### Why Not Simply "Denoise First, Then Split"?

An intuitive approach is to first denoise the data using methods like HDN, and then apply μSplit for decomposition. However, this sequential pipeline suffers from several issues:
- It requires training **three independent HDN models** (to denoise the input mixture and the two individual channels) plus one μSplit model, resulting in high computational overhead (~11 hours compared to 1.5 hours for DenoiSplit).
- Errors introduced by the denoising models propagate to the subsequent splitting step.
- Fine structures may be lost during the denoising process, adversely affecting downstream biological analysis.

The core insight of DenoiSplit is that **denoising and splitting should be performed jointly**, achieving both objectives simultaneously by explicitly modeling noise within the training objective.

## Method

### Overall Architecture

DenoiSplit is based on the **Variational Splitting Encoder-Decoder (VSE) Network**, adapted from a hierarchical VAE (HVAE). Distinct from standard autoencoders, its output consists of two split and denoised channel images $(\hat{c}_1, \hat{c}_2)$, rather than the reconstruction of the input.

The training objective is to maximize the likelihood of the noisy dual-channel data:

$$\boldsymbol{\theta} = \arg\max_{\boldsymbol{\theta}} \sum_{1 \leq j \leq n} \log P(c_{1,j}^N, c_{2,j}^N; \boldsymbol{\theta})$$

This objective utilizes a modified ELBO, assuming that the predictions for the two channels are conditionally independent given the latent variable $z$.

### Key Designs

1. **Hierarchical KL Loss Weighting**: A key innovation to address the noise breakthrough problem.

   The latent space of HVAE is hierarchical: bottom layers $Z[i]$ (low index $i$) encode fine pixel-level structures (high resolution), while top layers encode large-scale structures. In μSplit, the KL divergence of each layer is scaled as:
    $\text{kl}_i = \alpha \cdot \sum_{j,h,w} \frac{\text{KL}_i[j,h,w]}{h_i \cdot w_i}$
   which normalizes by the spatial dimensions. Consequently, the constraint weight of the KL divergence at each pixel location in the bottom layers is small, allowing noise to easily bypass constraints and leak into the output through the bottom latent space.

   DenoiSplit modifies this to a **direct summation** (without spatial normalization):
    $\text{kl}_i = \alpha \cdot \sum_{j,h,w} \text{KL}_i[j,h,w]$

   This enforces a stronger Gaussian prior constraint on the bottom latent space, preventing noise from being encoded. This seemingly minor modification yields significant performance gains—using this modification alone (referred to as Altered μSplit) substantially improves splitting quality on noisy data.

2. **Pixel Noise Models**: The core component enabling unsupervised denoising.

   The log-likelihood terms for individual channels are replaced with the likelihoods defined by the noise models:

    $E_{q(z|x;\phi)}[\log P^{nm}(c_1^N|\hat{c}_1) + \log P^{nm}(c_2^N|\hat{c}_2)] - KL(q(z|x;\phi), P(z))$

   The noise model $P_i^{nm}(c_i^N|c_i)$ characterizes the mapping from clean pixel intensities to noisy observations (and vice versa), which is pixel-wise independent:
    $P_i^{nm}(c_i^N|c_i) = \prod_k P_i^{nm}(c_i^N[k]|c_i[k])$

   This independence corresponds closely to the Poisson and Gaussian noise distributions found in microscopy. Instead of predicting Gaussian distribution parameters, the network directly predicts clean pixel values $(\hat{c}_1, \hat{c}_2)$ and optimizes the probability of generating the observed noisy data using these predictions as input to the noise model. The noise model can be calibrated directly from physical microscopy setup measurements or estimated from the training data.

3. **Calibrated Data Uncertainties**: Providing reliable prediction error estimation.

   Leveraging the variational nature of the VSE network, the method estimates uncertainty by sampling predictions $k=50$ times for each input to compute pixel-wise standard deviations $\sigma_1, \sigma_2$ as uncertainty estimation. Two scalar parameters, $\alpha_1, \alpha_2$, are learned to calibrate the uncertainty, establishing an approximately linear relationship between predicted uncertainty and the actual root-mean-squared error (RMSE).

   Calibration evaluation: Pixels are sorted by scaled standard deviation and divided into 30 bins. The Root Mean Variance (RMV) and RMSE are computed for each bin; ideal calibration yields RMSE ≈ RMV.

### Loss & Training

The total loss is the modified ELBO:
$$\mathcal{L} = -E_{q(z|x;\phi)}[\log P^{nm}(c_1^N|\hat{c}_1) + \log P^{nm}(c_2^N|\hat{c}_2)] + \sum_i \text{kl}_i$$

where the KL terms are summed hierarchically without spatial normalization, and the noise model terms encourage the network to output clean predictions consistent with the noisy observations. Uncertainty calibration uses the validation set to learn scaling scalars, which are then evaluated on the test set. Training takes approximately 1.5 hours on a single Tesla V100.

## Key Experimental Results

### Main Results

**Four splitting tasks on the BioSR dataset (PSNR / MS-SSIM, noise level $\sigma=1$, without Poisson noise):**

| Task | Method | Training Time (h) | PSNR | MS-SSIM |
|------|------|-------------|------|---------|
| T1: ER vs CCPs | μSplit | 7 | 30.3 | 0.853 |
| | HDN⊕μSplit | 11 | 37.3 | 0.982 |
| | Altered μSplit | 1.3 | 38.9 | 0.988 |
| | **DenoiSplit** | **1.5** | **39.7** | **0.989** |
| T3: CCPs vs MT | μSplit | 7.2 | 30.5 | 0.880 |
| | HDN⊕μSplit | 11 | 38.4 | 0.981 |
| | Altered μSplit | 1.4 | 38.9 | 0.985 |
| | **DenoiSplit** | **1.6** | **40.1** | **0.986** |

**Performance comparison at a high noise level ($\sigma=4$):**

| Task | μSplit | HDN⊕μSplit | DenoiSplit |
|------|--------|------------|-----------|
| T1: ER vs CCPs | 25.9 / 0.42 | 29.4 / 0.872 | **31.1 / 0.912** |
| T3: CCPs vs MT | 25.6 / 0.46 | 29.3 / 0.844 | **30.6 / 0.872** |
| T4: F-actin vs ER | 22.4 / 0.331 | 25.8 / 0.725 | **26.0 / 0.725** |

**Real noisy data (Hagen et al. Actin-Mito dataset):**

| Method | PSNR | MS-SSIM |
|------|------|---------|
| μSplit | 26.5 | 0.872 |
| HDN⊕μSplit | 28.1 | 0.887 |
| Altered μSplit | **31.1** | **0.936** |
| DenoiSplit | 31.0 | 0.935 |

### Ablation Study

**Ablation of KL loss weighting strategies (T1: ER vs CCPs, $\sigma=1$ without Poisson noise):**

| Configuration | PSNR | MS-SSIM | Description |
|------|------|---------|------|
| μSplit (Spatially normalized KL) | 30.3 | 0.853 | Severe noise breakthrough |
| Altered μSplit (Directly summed KL) | 38.9 | 0.988 | +8.6 dB, noise breakthrough is suppressed |
| DenoiSplit (Directly summed KL + Noise Model) | **39.7** | **0.989** | +0.8 dB, performance further improved by noise model |

**Systematic comparison across all noise levels (T1, 8 noise configurations):**

| Method | $\sigma=1,\lambda=0$ | $\sigma=2,\lambda=0$ | $\sigma=4,\lambda=0$ | $\sigma=1,\lambda=1000$ | $\sigma=4,\lambda=1000$ |
|------|---------|---------|---------|------------|------------|
| μSplit | 30.3 | 27.4 | 25.9 | 29.4 | 25.9 |
| HDN⊕μSplit | 37.3 | 33.8 | 29.4 | 36.3 | 29.4 |
| **DenoiSplit** | **39.7** | **35.4** | **31.1** | **37.9** | **31.2** |

### Key Findings

1. **KL weighting is the key improvement**: Merely modifying the normalization of the KL loss (Altered μSplit) increases PSNR by 8.6 dB, indicating that the root cause of noise breakthrough is insufficient constraints on the bottom latent space.
2. **End-to-end outperforms sequential pipelines**: DenoiSplit achieves better performance (39.7 vs 37.3 PSNR) with significantly less training time (1.5h vs 11h), demonstrating that joint optimization of denoising and splitting is more efficient than separating them.
3. **More pronounced advantage at high noise levels**: At $\sigma=4$, DenoiSplit improves on μSplit by 5.2 dB and outperforms HDN⊕μSplit by 1.7 dB.
4. **Calibrated uncertainties are effective**: The RMSE vs. RMV curve is close to $y=x$, showing that the model can reliably estimate prediction errors.
5. **Effective on real noisy data**: On the Hagen et al. dataset, both Altered μSplit and DenoiSplit significantly outperform the two baselines.

## Highlights & Insights

- **Elegant probabilistic framework**: By naturally integrating the noise model into a variational inference framework and replacing the MSE loss with a likelihood function, the proposed approach is theoretically rigorous. The noise models can be tailored to specific microscopy configurations, offering high versatility.
- **Deep insight into KL loss normalization**: The spatial normalization of the KL loss in μSplit makes the KL constraint weight for each pixel in the bottom latent layers very small, allowing noise to leak through the bottom layers. Summing the KL loss directly assigns greater total weight to the bottom layers (due to the larger pixel count), strictly regularizing the bottom-level information encoding. This generic finding is broadly applicable to any scenario using HVAEs with noisy data.
- **Practical value of uncertainty quantification**: In biological applications, knowing "how reliable a prediction is" is just as important as the prediction itself. DenoiSplit's calibrated uncertainty estimation allows researchers to identify low-confidence regions, enabling more reliable biological assessments.
- **Training efficiency**: The single end-to-end model trained in 1.5 hours, which is far less than the 11 hours required for HDN⊕μSplit.

## Limitations & Future Work

- Currently only handles two-channel splitting; extending to multi-channel splitting requires additional noise models.
- The noise models must be predefined or estimated in advance, which may require extra efforts for data with unknown noise characteristics.
- Evaluations are predominantly based on synthetic (manually added) noise, with limited assessment on naturally noisy real data.
- Failure cases are mentioned in the supplementary material but are not fully discussed in the main text.
- Domain adaptation remains limited; generalization across different microscopes or imaging modalities warrants further investigation.

## Related Work & Insights

- **Relationship to μSplit**: DenoiSplit introduces two key modifications (KL weighting and noise models) over the HVAEs + Regular-LC architecture of μSplit, yielding substantial performance gains with minimal architectural changes.
- **Relationship to HDN**: The noise modeling concept of HDN is integrated more elegantly into the splitting task, achieving two goals with a single architecture.
- **Practicality of variational inference**: The model demonstrates the real-world utility of HVAE posterior sampling and uncertainty calibration in scientific computing contexts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Proposes the first framework for joint image splitting and unsupervised denoising; the insight behind the KL weighting improvement is profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Highly systematic evaluation including 4 splitting tasks across 8 noise configurations, real noisy datasets, and uncertainty calibration assessments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly defined problem setup, mathematically rigorous derivations, and polished illustrations.
- **Value**: ⭐⭐⭐⭐ — Holds direct practical value for fluorescence microscopy image analysis; the methodological framework is generalizable to other image decomposition scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[ECCV 2024\] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [\[ECCV 2024\] Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography](joint_rgb-spectral_decomposition_model_guided_image_enhancement_in_mobile_photog.md)
- [\[NeurIPS 2025\] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy](../../NeurIPS2025/image_restoration/scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)
- [\[ECCV 2024\] TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)

</div>

<!-- RELATED:END -->
