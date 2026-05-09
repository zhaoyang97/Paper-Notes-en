---
title: >-
  [Paper Note] Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data
description: >-
  [ICCV 2025][SAR image generation] This paper proposes Φ-GAN, which integrates the ideal Point Scattering Center (PSC) electromagnetic scattering physical model into GAN training as a differentiable neural module. Through a dual physics loss (generator physical consistency constraint + discriminator electromagnetic feature distillation), the method significantly improves the quality and stability of SAR image generation under data-scarce conditions.
tags:
  - ICCV 2025
  - SAR image generation
  - GAN regularization
  - point scattering center model
  - physical constraints
  - few-shot learning
date: 2026-05-08
content_hash: 2319c1629cb5eb0e
---

# Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data

**Conference**: ICCV 2025
**arXiv**: [2503.02242](https://arxiv.org/abs/2503.02242)
**Code**: N/A
**Area**: Other
**Keywords**: SAR image generation, GAN regularization, point scattering center model, physical constraints, few-shot learning

## TL;DR

This paper proposes Φ-GAN, which integrates the ideal Point Scattering Center (PSC) electromagnetic scattering physical model into GAN training as a differentiable neural module. Through a dual physics loss (generator physical consistency constraint + discriminator electromagnetic feature distillation), the method significantly improves the quality and stability of SAR image generation under data-scarce conditions.

## Background & Motivation

Synthetic Aperture Radar (SAR) is critical in remote sensing due to its all-weather, all-day imaging capability. However, SAR images exhibit unique electromagnetic scattering characteristics and are costly to annotate, making large-scale datasets scarce. This motivates the use of GANs for SAR image generation, which faces **three major challenges**:

**Extreme data scarcity**: In practice, each class may contain only dozens of images (e.g., 5% of MSTAR yields only 121 images; 1% of OpenSARShip yields only 46 images), far fewer than natural image datasets.

**Failure of conventional augmentation**: The electromagnetic scattering characteristics of SAR targets vary significantly with azimuth angle ("target rotation" ≠ "image rotation"), making standard augmentations such as rotation ineffective or even harmful for SAR images.

**Lack of physical consistency**: Existing data-driven generative models have no knowledge of SAR imaging physics, and the generated images may visually resemble SAR imagery while being physically inconsistent.

These challenges are empirically confirmed: DiffAugment degrades significantly on SAR (FID from 290 to 1089), and ADA provides almost no improvement. The core insight is that **the electromagnetic scattering model (PSC) for SAR images provides domain prior knowledge unavailable to natural image augmentation methods**, and integrating it into GAN training can simultaneously constrain the generator to produce physically consistent images and prevent the discriminator from overfitting to speckle noise.

## Method

### Overall Architecture

Φ-GAN extends a standard conditional GAN (cGAN) with three new components: (1) a physics-inspired neural module $\mathcal{F}_{\text{est}}$ that estimates PSC physical parameters; (2) a PSC physical model $\mathcal{F}_{\text{phy}}$ that reconstructs electromagnetic scattering features from physical parameters; and (3) a dual-discriminator structure ($\mathcal{D}_{\text{img}}$ evaluating raw images + $\mathcal{D}_{\text{phy}}$ evaluating physically reconstructed results). The outputs of both the generator and discriminator jointly consider judgments in the image domain and the physical domain:

$$d_{out}(u, s) = \alpha \mathcal{D}_{\text{img}}(u) + (1-\alpha) \mathcal{D}_{\text{phy}}(s)$$

Conditional inputs include target class (one-hot encoding) and azimuth angle (encoded via Cyclic High-frequency Embedding, CHE):

$$r(\theta) = [\sin\theta, \cos\theta, \sin 2\theta, ..., \sin 5\theta, \cos 5\theta]$$

### Key Designs

1. **Ideal Point Scattering Center (PSC) Model**: At high frequencies, a SAR target can be approximated as the superposition of $N$ independent point scatterers, where each PSC response is a function of radar frequency $f$ and azimuth angle $\phi$:

$$E(f, \phi) = \sum_{i=1}^{N} A_i \cdot \exp\left(-j\frac{4\pi f}{c}(x_i \cos\phi + y_i \sin\phi)\right)$$

where $A_i$ is the scattering amplitude and $(x_i, y_i)$ determines the PSC location. This model carries explicit physical meaning: $A_i$ reflects the scattering characteristics of the target geometry, and the positional parameters reflect the spatial distribution of the target.

2. **Physics-Inspired Neural Module $\mathcal{F}_{\text{est}}$**: Efficiently estimating PSC parameters from SAR images is the key bottleneck for integrating the physical model into GAN training. The paper formulates PSC parameter estimation as a sparse reconstruction problem:

$$\hat{\mathbf{o}} = \arg\min_{\mathbf{o}} \|\mathbf{\Psi}\mathbf{o} - \mathbf{r}\|_2 + \lambda\|\mathbf{o}\|_1$$

where $\mathbf{\Psi}$ is a dictionary matrix encoding positional information, $\mathbf{r}$ is the input image, and $\mathbf{o}$ is the sparse PSC response vector. The **Half-Quadratic Splitting (HQS) algorithm is unrolled** into a two-stage neural network, with each stage admitting a closed-form solution:

$$\mathbf{o}^{(k)} = \mathbf{p}^{(k-1)} + \mu^{(k)} \mathbf{\Psi}^H (\mathbf{\Psi}\mathbf{\Psi}^H)^{-1} (\mathbf{r} - \mathbf{\Psi}\mathbf{p}^{(k-1)})$$

$$\mathbf{p}^{(k)} = S_{\rho^{(k)}}(\mathbf{p}^{(k-1)} + t^{(k)} \mathbf{\Psi}^H (\mathbf{\Psi}\mathbf{p}^{(k-1)} - \mathbf{o}^{(k)}))$$

where $S_\rho$ denotes the soft-thresholding function. The fixed parameters $t, \rho, \mu$ of the conventional HQS algorithm are replaced by **learnable parameters** at each stage, resulting in only 6 trainable parameters in total across both stages, enabling effective optimization even under extreme data scarcity.

3. **Dual Physics Loss**:

   **Generator physics loss** $\mathcal{L}_{\text{phy}}^G$: Constrains the generated images to have physical parameters consistent with real images, including both image-level and feature-level terms:
    $\mathcal{L}_{\text{phy}}^G = \beta \cdot \text{MSE}(s, \tilde{s}) + \gamma \sum_{i=1}^M \frac{1}{C^i H^i W^i} \|F_{\text{phy}}^i(s) - F_{\text{img}}^i(\tilde{u})\|_2$

   **Discriminator physics loss** $\mathcal{L}_{\text{phy}}^D$: By distilling the electromagnetic features of $\mathcal{D}_{\text{phy}}$ into $\mathcal{D}_{\text{img}}$, this loss forces the image discriminator to **make decisions based on electromagnetic scattering features rather than speckle noise patterns**:
    $\mathcal{L}_{\text{phy}}^D = \gamma \sum_{i=1}^M \frac{1}{C^i H^i W^i} (\|F_{\text{img}}^i(\tilde{u}) - F_{\text{phy}}^i(\tilde{s})\|_2 + \|F_{\text{img}}^i(u) - F_{\text{phy}}^i(s)\|_2)$

### Loss & Training

- Total loss: $\mathcal{L}^G = \mathcal{L}_{\text{ori}}^G + \mathcal{L}_{\text{phy}}^G$, $\mathcal{L}^D = \mathcal{L}_{\text{ori}}^D + \mathcal{L}_{\text{phy}}^D$
- Hyperparameters: $\alpha=0.6$, $\beta=1$, $\gamma=10$
- Optimizer: Adam with learning rate 0.0002; $\mathcal{F}_{\text{est}}$ uses AdamW with learning rate 0.002
- Training strategy: In the early stage, the discriminator is trained 5 times for every 1 generator update
- $\mathcal{F}_{\text{est}}$ is pretrained independently before GAN training, after which its parameters are frozen
- PSC module loss: $\mathcal{L}_{\text{PSC}} = \|\mathbf{r} - \mathbf{\Psi}\mathbf{o}^{(K)}\|_2^2 + \lambda_1\|\mathbf{o}^{(K)} - \mathbf{p}^{(K)}\|_2^2 + \lambda_2\|\mathbf{p}^{(K)}\|_1$

## Key Experimental Results

### Main Results

Comparison based on ACGAN on 10% MSTAR dataset (237 images, 10 classes):

| Method | SSIM↑ | VIF↑ | FSIM↑ | GMSD↓ | FID↓ | KID↓ |
|--------|-------|------|-------|-------|------|------|
| ACGAN | 0.3224 | 0.0386 | 0.7432 | 0.1510 | 290.05 | 0.4548 |
| +ADA | 0.2606 | 0.0243 | 0.7171 | 0.1643 | 320.59 | 0.4240 |
| +DA(DiffAug) | 0.2168 | 0.0188 | 0.6570 | 0.2018 | 1089.47 | 1.8056 |
| +DIG | 0.3279 | 0.0311 | 0.7283 | 0.1570 | 373.07 | 0.6243 |
| **+Ours** | **0.3583** | **0.0781** | **0.7622** | **0.1385** | **87.27** | **0.0414** |

### Ablation Study

Incrementally adding each loss component on 10% MSTAR:

| Configuration | SSIM↑ | FSIM↑ | GMSD↓ | FID↓ | KID↓ |
|---------------|-------|-------|-------|------|------|
| ACGAN (baseline) | 0.3224 | 0.7432 | 0.1510 | 290.05 | 0.4548 |
| +$\mathcal{L}_{\text{phy/s}}^G$ | 0.3514 | 0.7611 | 0.1392 | 97.45 | 0.0671 |
| +$\mathcal{L}_{\text{phy/f}}^G$ | 0.3477 | 0.7525 | 0.1414 | 99.93 | 0.0568 |
| +$\mathcal{L}_{\text{phy}}^D$ | 0.3467 | 0.7526 | 0.1422 | 92.76 | 0.0471 |
| **Ours (Full)** | **0.3583** | **0.7622** | **0.1385** | **87.27** | **0.0414** |

Generalization across backbone models (StyleGAN as baseline):

| Baseline + Method | 10% MSTAR FID↓ | 5% MSTAR FID↓ | 1% OpenSARShip FID↓ |
|-------------------|---------------|---------------|-------------------|
| StyleGAN | 290.64 | 339.28 | 45.10 |
| StyleGAN+Ours | **174.83** | **305.05** | **44.78** |

### Key Findings

- DiffAugment severely degrades performance on SAR (FID from 290 to 1089), validating the argument that natural image methods are not suitable for SAR
- Φ-GAN reduces FID from 290 to 87.27 (approximately 70% reduction) and KID from 0.4548 to 0.0414 (approximately 91% reduction)
- Each physics loss component is independently effective: $\mathcal{L}_{\text{phy/s}}^G$, $\mathcal{L}_{\text{phy/f}}^G$, and $\mathcal{L}_{\text{phy}}^D$ each reduce FID to below approximately 100
- The physics-inspired neural module has only 6 trainable parameters, incurring negligible computational overhead
- The method consistently improves performance across three baseline architectures (CVAE-GAN, ACGAN, StyleGAN), demonstrating its generality
- The method performs well under extreme few-shot conditions such as 1% OpenSARShip (46 images) and 14% SAR-Airplane (20 images)

## Highlights & Insights

- **Exemplary fusion of physical modeling and deep learning**: Rather than using the physical model as pre- or post-processing, the paper transforms the physical solving process into a differentiable neural module via algorithm unrolling, enabling end-to-end training.
- **Symmetric elegance of the dual-loss design**: The generator-side loss enforces physical consistency, while the discriminator-side loss prevents overfitting to speckle noise; the two are complementary.
- The physical module with only 6 trainable parameters embodies a "less is more" design philosophy — under extreme data scarcity, introducing strong priors is more effective than increasing parameter count.
- The PSC model is a well-established physical tool in the SAR domain; its integration approach carries methodological generality — any domain with a mature physical model can adopt a similar strategy to incorporate it into generative models.

## Limitations & Future Work

- The PSC model only describes the idealized case of point scatterers, with limited ability to model distributed scatterers such as ground clutter.
- The current work is validated only on conditional GANs; extension to diffusion models is a natural direction for future work.
- The physical module requires pre-constructing the dictionary matrix $\mathbf{\Psi}$, which must be reconfigured for different SAR system parameters.
- The method is primarily evaluated on military target recognition scenarios (e.g., MSTAR); its effectiveness in civilian SAR scenes (e.g., ships, buildings) warrants further investigation.

## Related Work & Insights

- The proposed method is orthogonal to data augmentation (ADA, DiffAugment) and regularization (DIG, RLC) approaches and can be combined with them.
- CAE and CVAE-GAN focus on SAR-specific generative architecture design, while Φ-GAN focuses on physical constraints; the two directions are complementary.
- The algorithm unrolling idea originates from signal processing works such as LISTA; this paper represents a novel application of that idea in the context of generative models.
- The work provides a successful example of Physics-Informed AI in the GAN domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First end-to-end integration of an electromagnetic scattering physical model into GAN training, with an elegant physical module design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, two backbone architectures, comprehensive ablation, and comparison with SAR-specific methods.
- **Writing Quality**: ⭐⭐⭐⭐ Physical background is clearly introduced, and the methodological derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ Directly valuable to the SAR image generation community; the physical integration paradigm is transferable to other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)

</div>

<!-- RELATED:END -->
