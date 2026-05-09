---
title: >-
  [Paper Note] From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE–Normalizing Flows
description: >-
  [NeurIPS 2025][Scientific Computing][Normalizing Flows] A two-stage VAE–Normalizing Flow probabilistic inference framework is proposed that infers stellar mass, SFR, redshift, black hole mass, metallicity, and emission line fluxes directly from SDSS galaxy images and photometric data, surpassing existing non-spectroscopic methods in accuracy while being over 100× faster than SED fitting.
tags:
  - NeurIPS 2025
  - Scientific Computing
  - Normalizing Flows
  - VAE
  - galaxy parameter inference
  - emission line prediction
  - probabilistic inference
date: 2026-05-08
content_hash: 337bfe71ba1324a6
---

# From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE–Normalizing Flows

**Conference**: NeurIPS 2025
**arXiv**: [2511.12737](https://arxiv.org/abs/2511.12737)
**Code**: Not released
**Area**: Scientific Computing
**Keywords**: Normalizing Flows, VAE, galaxy parameter inference, emission line prediction, probabilistic inference

## TL;DR

A two-stage VAE–Normalizing Flow probabilistic inference framework is proposed that infers stellar mass, SFR, redshift, black hole mass, metallicity, and emission line fluxes directly from SDSS galaxy images and photometric data, surpassing existing non-spectroscopic methods in accuracy while being over 100× faster than SED fitting.

## Background & Motivation

- **Inferring galaxy physical parameters is a core task in astrophysics**: Parameters such as stellar mass, star formation rate (SFR), redshift, gas-phase metallicity, and central black hole mass are essential for understanding galaxy formation and evolution.
- **Emission line measurements require expensive spectroscopic observations**: Lines such as Hα, Hβ, [N II], and [O III] are fundamental diagnostics for SFR, metallicity, dust content, and AGN activity, but spectroscopic observations are time-consuming and cannot scale to billions of galaxies in future large surveys.
- **Traditional SED fitting methods are computationally costly**: Methods such as Prospector, Bagpipes, and CIGALE are physically well-grounded but computationally intensive, making them impractical for large-scale surveys.
- **Existing deep learning methods largely produce point estimates**: Methods such as AstroCLIP leverage contrastive learning for cross-modal prediction but mostly yield point estimates without calibrated uncertainty quantification.
- **Complex degeneracies exist among physical parameters**: Degeneracies among stellar mass, SFR, and redshift make it difficult for deterministic regressors to accurately model the joint distribution.
- **No probabilistic non-spectroscopic estimator exists for black hole mass**: Prior to this work, no method provided probabilistic estimates of central black hole mass from images and photometry alone.

## Method

### Overall Architecture

A two-stage architecture is adopted. In the first stage, a VAE encodes 160×160 gri three-channel galaxy images into a 32-dimensional latent representation. In the second stage, the VAE latent variables are concatenated with photometric color indices and fed into a conditional Normalizing Flow to model the joint posterior distribution of physical parameters and emission line fluxes. The overall pipeline is: Image → VAE Encoder → Latent $z$ + Photometry → MLP Encoder → Conditional RealNVP → Joint Posterior.

### Key Design 1: VAE Image Encoder

- **Function**: Compresses SDSS galaxy gri images into a 32-dimensional latent vector as feature input for downstream inference.
- **Mechanism**: The encoder consists of three convolutional layers (kernel 4, stride 2, padding 1) followed by fully connected layers, outputting mean $\mu \in \mathbb{R}^{32}$ and log-variance $\log \sigma^2 \in \mathbb{R}^{32}$; latent variables are sampled via the reparameterization trick as $z \sim \mathcal{N}(\mu, \sigma^2)$. The decoder reconstructs images via transposed convolutions.
- **Design Motivation**: Feeding high-dimensional images directly into the NF is impractical; the VAE provides a compact yet informative image representation while preserving galaxy morphological information.

### Key Design 2: Two-Branch Conditional RealNVP Flow

- **Function**: Models the joint posterior distribution of physical parameters and emission line fluxes in separate branches.
- **Mechanism**: An MLP first predicts mean estimates, after which a 12-layer affine-coupling RealNVP models the residual distribution. Physical parameters are divided into two sub-branches: (1) a "core parameters" branch using a 4D flow to model the joint residual distribution of $M_\star$, SFR, $z$, and $M_\mathrm{BH}$; (2) a "metallicity" branch using a 1D conditional affine flow conditioned on samples from the core parameters to model O/H. Inference decomposes via the chain rule as: $p(y_\mathrm{core}, \mathrm{O/H} \mid x) = p(y_\mathrm{core} \mid x) \cdot p(\mathrm{O/H} \mid y_\mathrm{core}, x)$.
- **Design Motivation**: The chain rule decomposition explicitly models the physical dependence of metallicity on other parameters, avoiding the oversight of inter-parameter correlations inherent in independent regression.

### Key Design 3: Metallicity Detectability Classifier

- **Function**: An MLP with sigmoid output predicts whether a galaxy has a measurable metallicity value.
- **Mechanism**: Jointly trained with the regression task (MSE + BCE joint loss), leveraging the shared encoded representation for binary classification.
- **Design Motivation**: Not all galaxies have reliable metallicity measurements; the classifier first determines whether the parameter is meaningful before proceeding with inference, achieving approximately 84% accuracy.

### Key Design 4: Photometric Feature Augmentation

- **Function**: Concatenates four color indices ($u{-}g$, $g{-}r$, $r{-}i$, $i{-}z$) and apparent magnitude with the VAE latent variables as input to the NF.
- **Mechanism**: An MLP maps the 64-dimensional VAE features (32-dimensional mean + 32-dimensional standard deviation) together with photometric features into a unified 256-dimensional representation.
- **Design Motivation**: UMAP visualizations show that high- and low-mass galaxies are difficult to separate using image latent variables alone; adding photometric information substantially improves separation (Figure 4).

## Loss & Training

- **VAE stage**: MSE reconstruction loss + KL divergence regularization; Adam optimizer, learning rate $10^{-4}$; trained for 1.5 hours on an A100 GPU.
- **NF stage**: Negative log-likelihood loss (maximizing data likelihood under the flow model); trained for approximately 30 minutes on a T4 GPU.
- **Data scale**: Approximately 250K SDSS galaxies ($z \leq 0.3$); 100K used for VAE training, ~125K for NF training (70/15/15 split).
- All physical parameters and emission lines are normalized to zero mean and unit variance.

## Key Experimental Results

### Physical Parameter Prediction (Table 1: $R^2$ Comparison)

| Method | Redshift $z$ | Stellar Mass | SFR | BH Mass | Metallicity |
|--------|-------------|-------------|-----|---------|------------|
| (r,g,z) Photometry + MLP | 0.68 | 0.67 | 0.34 | N/A | 0.41 |
| Image Embedding + MLP | 0.78 | 0.73 | 0.42 | N/A | 0.43 |
| Image Embedding + kNN | 0.79 | 0.74 | 0.44 | N/A | 0.44 |
| Image Embedding [Gagliano] | 0.83 | 0.75 | N/A | N/A | N/A |
| **Image + Phot + NF (Ours)** | **0.80** | **0.85** | **0.76** | **0.67** | **0.76** |
| Photometry + NF (Ours) | 0.72 | 0.80 | 0.75 | 0.62 | 0.65 |

The proposed method substantially outperforms the previous best baseline in stellar mass (+0.10), SFR (+0.32), and metallicity (+0.32). Even when using only photometric data, SFR and metallicity predictions remain superior to image-embedding-based methods.

### Uncertainty Decomposition (Table 2: Validation Set)

| Uncertainty | $M_\mathrm{BH}$ | $\log M_\star$ | $12{+}\log(\mathrm{O/H})$ | $\log \mathrm{SFR}$ | $z$ | Hα | Hβ | [N II] | [O III] |
|------------|-----------------|---------------|--------------------------|---------------------|-----|----|----|--------|---------|
| $\sigma_\mathrm{aleatoric}$ | 0.589 | 0.191 | 0.134 | 0.327 | 0.018 | 0.427 | 0.381 | 0.427 | 0.611 |
| $\sigma_\mathrm{epistemic}$ | 0.034 | 0.012 | 0.010 | 0.019 | 0.001 | 0.027 | 0.026 | 0.027 | 0.045 |

Aleatoric uncertainty dominates epistemic uncertainty across all parameters, indicating that the model has converged well and that the primary source of uncertainty is intrinsic scatter in the data. Redshift and metallicity are most tightly constrained, while black hole mass and [O III] exhibit the largest uncertainties.

### Emission Line Prediction

- Balmer lines (Hα, Hβ): $R^2 = 0.79$–$0.80$, high prediction accuracy.
- [N II] λ6584: $R^2 = 0.70$, moderate accuracy.
- [O III] λ5007: $R^2 = 0.50$, more difficult to predict, reflecting its strong dependence on ionization conditions.

## Highlights & Insights

- First probabilistic estimation of central black hole mass from images and photometric data alone.
- SFR prediction $R^2$ improves substantially from the previous best of 0.44 to 0.76.
- The chain rule decomposition elegantly captures physical inter-parameter dependencies; the resulting posterior distributions exhibit expected astrophysical structures such as the star-forming main sequence.
- Inference speed exceeds SED fitting by over 100×, making the method suitable for large-scale surveys such as Roman and Rubin LSST.
- Latent space interpretability analyses (perturbation decoding, UMAP embeddings) enhance physical credibility.

## Limitations & Future Work

- Validation is performed only on SDSS DR1 data, which is relatively shallow and noisy; the higher-quality DR17 is not utilized.
- The redshift range is restricted to $z \leq 0.3$, precluding coverage of earlier cosmic epochs.
- [O III] prediction achieves only $R^2 = 0.50$; the model struggles to capture emission lines strongly dependent on ionization conditions.
- The VAE smooths small-scale structures under noisy inputs, potentially losing morphological detail (the authors suggest replacing the VAE with a diffusion model).
- Black hole mass labels are derived from the empirical $M_\mathrm{BH}$–$\sigma$ relation rather than direct measurements, introducing indirect systematic errors.

## Related Work & Insights

- **SED fitting methods**: Physics-driven pipelines such as Prospector, Bagpipes, and CIGALE are physically well-grounded but computationally expensive and difficult to scale.
- **AstroCLIP**: A multimodal contrastive learning framework that aligns image–spectrum shared latent spaces for cross-modal prediction, but produces only point estimates.
- **Gagliano et al.**: A conditional VAE for inferring stellar mass and redshift with relatively high $R^2$, but without uncertainty quantification and without predicting emission lines.
- **Positioning of this work**: The proposed method unifies probabilistic joint inference, calibrated uncertainty quantification, and emission line prediction, filling the capability gaps of the above approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of VAE+NF to joint probabilistic inference of galaxy physical parameters and emission lines; the chain rule decomposition design is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-parameter, multi-baseline comparisons, uncertainty decomposition, and latent space interpretability analyses.
- Writing Quality: ⭐⭐⭐ Structure is clear but contains occasional typographical errors (e.g., *distrbution*, *accruacy*).
- Value: ⭐⭐⭐⭐ Provides a practical and efficient probabilistic inference tool for large-scale astronomical surveys, directly serving Roman/Rubin LSST.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE & Normalizing Flows](from_images_to_physics_probabilistic_inference_of_galaxy_parameters_and_emission.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)

</div>

<!-- RELATED:END -->
