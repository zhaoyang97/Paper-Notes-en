---
title: >-
  [Paper Note] Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders
description: >-
  [NEURIPS2025][Physics & Scientific Computing][high-redshift galaxies] A variational autoencoder (VAE) is applied to unsupervised clustering of 2,743 JWST high-redshift ($z>4$) galaxy spectra, uncovering 12 distinct astrophysical categories and more than doubling the known sample sizes of rare populations including post-starburst galaxies, Lyman-α emitters, extreme emission line galaxies, and Little Red Dots.
tags:
  - "NEURIPS2025"
  - "Physics & Scientific Computing"
  - "high-redshift galaxies"
  - "VAE"
  - "JWST"
  - "unsupervised discovery"
  - "spectral clustering"
date: 2026-05-08
content_hash: a4880ce9334c22ca
---

# Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders

**Conference**: NEURIPS2025  
**arXiv**: [2511.05439](https://arxiv.org/abs/2511.05439)  
**Code**: [https://github.com/aayush3009/learnspec](https://github.com/aayush3009/learnspec)  
**Area**: Physics  
**Keywords**: high-redshift galaxies, VAE, JWST, unsupervised discovery, spectral clustering  

## TL;DR
A variational autoencoder (VAE) is applied to unsupervised clustering of 2,743 JWST high-redshift ($z>4$) galaxy spectra, uncovering 12 distinct astrophysical categories and more than doubling the known sample sizes of rare populations including post-starburst galaxies, Lyman-α emitters, extreme emission line galaxies, and Little Red Dots.

## Background & Motivation

**Background**: Since its launch in 2021, JWST has fundamentally transformed our understanding of galaxy formation in the early universe. Its near-infrared spectroscopic capability captures redshifted UV/optical light from galaxies formed within the first few hundred million years after the Big Bang. A large volume of public spectroscopic data is now available, yet analysis still relies heavily on manual visual inspection to identify galaxies of interest.

**Limitations of Prior Work**:
   - Manual classification does not scale as the volume of public data grows rapidly;
   - Supervised methods require large annotated datasets, whereas labeled examples of rare high-redshift galaxies are extremely scarce;
   - VAEs had previously only been applied to low-redshift galaxy spectra from ground-based telescopes and had never been used on JWST high-redshift spectra.

**Key Challenge**: It is necessary to automatically discover rare yet astrophysically important galaxy populations from large volumes of high-dimensional spectral data in the absence of prior labels.

**Goal**:
   - Establish an unsupervised pipeline to automatically discover and classify high-redshift galaxy populations from public JWST spectra;
   - Substantially expand the known sample sizes of rare galaxy types.

**Key Insight**: VAE learning of compact and interpretable latent representations, combined with UMAP dimensionality reduction and GMM clustering, forms an end-to-end unsupervised discovery pipeline.

**Core Idea**: An unsupervised VAE-plus-clustering pipeline is applied for the first time to JWST high-redshift spectra, automatically discovering and substantially expanding the sample sizes of five astrophysically critical galaxy populations.

## Method

### Overall Architecture

- **Input**: 2,743 JWST/NIRSpec spectra ($z > 4$, corresponding to the first 1.5 billion years of cosmic history), preprocessed and normalized
- **Encoding**: A VAE encoder compresses high-dimensional spectra into a 16-dimensional latent space
- **Reconstruction**: A VAE decoder reconstructs the original spectra from latent vectors
- **Clustering**: UMAP reduces the 16D latent space to 2D → GMM identifies 12 clusters → clusters are manually labeled with domain knowledge

### Key Designs

1. **VAE Architecture**:

    - Function: Learn compact latent representations of spectra
    - Mechanism: A symmetric four-layer fully connected network; encoder $d \to 512 \to 256 \to 128 \to 64 \to 16$, decoder in reverse. Optimizes ELBO = reconstruction fidelity − KL regularization. Latent dimensionality $k=16$ balances expressiveness and computational efficiency.
    - Design Motivation: VAEs support both reconstruction and generation, and their latent space admits continuous interpolation.

2. **Spectral Preprocessing**:

    - Function: Convert spectra at different redshifts into a unified rest-frame representation
    - Mechanism: (1) De-redshifting: $\lambda_{\text{rest}} = \lambda_{\text{obs}} / (1+z)$; (2) Normalization: continuum at 1500 Å scaled to 1.0; (3) **arcsinh transformation**: $\text{arcsinh}(x) = \ln(x + \sqrt{x^2+1})$, which is approximately linear for small values (continuum) and approximately logarithmic for large values (emission lines), preserving information from both regimes.
    - Design Motivation: JWST spectra contain both low-flux continuum and high-flux emission lines; naive normalization sacrifices information from one component. The arcsinh transformation is a key innovation.

3. **Masked Reconstruction Loss**:

    - Function: Handle missing data in spectra
    - Mechanism: $L_{\text{rec}} = \frac{1}{N} \sum_i \sum_j M_{ij}(x_{ij} - \hat{x}_{ij})^2$, where $M_{ij}$ is a binary mask that excludes missing wavelength bins. The VAE can predict and infill masked regions.
    - Design Motivation: Galaxies at different redshifts have different rest-frame wavelength coverages, making missing regions unavoidable.

4. **Two-Stage Clustering**:

    - Function: Identify distinct galaxy populations in the latent space
    - Mechanism: UMAP first reduces 16D to 2D (mitigating the curse of dimensionality), then GMM (5–15 components, tested 100 times) selects the optimal grouping by Silhouette score. The final solution yields 12 clusters with a Silhouette score of 0.44.
    - Design Motivation: Direct clustering in high-dimensional latent space is unstable; the UMAP + GMM combination has been demonstrated effective in astronomical applications.

### Loss & Training

- Learning rate: Exponential decay, initial $10^{-4}$, decay rate 0.95 per 500 steps
- Regularization: L2 weight regularization $\lambda = 0.001$, batch normalization, dropout (0.2 → 0.1)
- Early stopping: Training halts if validation reconstruction loss does not improve for 50 steps
- Train/validation split: 85%/15%

## Key Experimental Results

### Main Results — Galaxy Population Discovery

| Galaxy Class | Discovered | Prev. Known | Astrophysical Significance |
|---------|---------|------------|----------|
| Post-starburst/Quenched | 326 | ~170 | More than doubles known count at $z>4$; traces galaxy quenching |
| Lyman-α Emitters (LAEs) | 213 | ~100 | Doubled; traces cosmic reionization |
| Extreme Emission Line Galaxies | 180 | ~80 | Doubled; highest star-formation rates in the universe |
| High-z (highest redshift) | 320 | Small sample | Traces earliest galaxy formation after the Big Bang |
| Little Red Dots (LRDs) | 142 | Few | Compact V-shaped continuum + strong emission lines; physical mechanism unclear |

### Ablation Study — Reconstruction Quality

| Metric | Value | Notes |
|------|-----|------|
| Median MSE | 0.122 | Excellent reconstruction for the majority of spectra |
| MSE std. dev. | 0.124 | One-sided long-tail distribution |
| High-error causes | Noise + artifacts | MSE > 0.1 typically arises from low-SNR spectra |
| Masked region prediction | VAE can infill missing bands | Increases reconstruction error but carries astrophysical significance |

### Key Findings

- **12 clusters span diverse astrophysical phenomena**: Each cluster contains 63–334 galaxies with no single dominant group, indicating the model captures a diverse range of galaxy populations.
- **Natural redshift–class association**: Without explicitly providing redshift as input, the VAE naturally separates galaxies at different redshifts into distinct clusters (Figure 2, right), demonstrating that the latent space encodes physically meaningful structure.
- **Composite spectra validation**: The median spectrum of each cluster exhibits highly consistent continuum shapes and emission line features, confirming the astrophysical coherence of the clustering.

## Highlights & Insights

- **arcsinh transformation for dynamic range**: This simple yet elegant preprocessing step simultaneously preserves continuum (linear at low values) and emission line (logarithmic at high values) information, resolving the intrinsic dynamic range problem of spectral data. The approach generalizes to other signal data with extreme dynamic ranges.
- **VAE's natural handling of missing data**: Through masked loss combined with latent space regularization, the VAE produces physically plausible predictions for missing wavelength bands, which is more astrophysically meaningful than conventional interpolation.
- **Unsupervised discovery → substantial expansion of rare samples**: Known sample sizes for five key galaxy populations are more than doubled, which is highly significant for high-redshift galaxy statistical studies and demonstrates the transformative value of machine learning in astronomical discovery.
- **Pipeline integration**: The method can be directly integrated into JWST spectroscopic data repositories for automatic classification and anomaly detection at ingestion time.

## Limitations & Future Work

- **Clustering performed in 2D UMAP space**: UMAP dimensionality reduction may discard high-dimensional structural information; clustering directly in the 16D latent space or employing hierarchical clustering may be preferable.
- **Class labeling still requires human expertise**: Astrophysical interpretation of clustering results depends on domain experts comparing outputs against known spectral features, and is therefore not fully automated.
- **Class degeneracy**: A single spectrum may belong to multiple known categories (e.g., EELG + LAE simultaneously); the current hard GMM assignment cannot accommodate this.
- **Limited sample size**: The dataset comprises 2,743 spectra; the approach is scalable to larger datasets as JWST data continues to grow.
- **Future directions**:
    - Incorporate JWST imaging (multimodal VAE) and high-resolution spectra;
    - Replace GMM with DBSCAN/OPTICS to handle non-convex cluster structures;
    - Explore $\beta$-VAE or conditional VAE to improve latent space interpretability.

## Related Work & Insights

- **vs. Portillo et al. / Bohm et al.**: Prior VAE applications targeted low-redshift spectra from ground-based surveys such as SDSS. This work is the first to extend the approach to JWST high-redshift spectra, opening a substantially larger discovery space.
- **vs. supervised classification**: Supervised methods require labeled data, yet high-redshift rare galaxies are precisely those lacking annotations. Unsupervised methods are naturally suited to this "known unknown" discovery scenario.
- **vs. traditional visual classification**: Visual inspection does not scale and introduces subjective bias; the proposed method processes thousands of spectra consistently.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of VAEs to unsupervised discovery in JWST high-redshift spectra; the arcsinh transformation is an elegant design choice.
- Experimental Thoroughness: ⭐⭐⭐ Reconstruction quality and clustering results are well-documented, but ablation comparisons against alternative unsupervised methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-balanced presentation of astronomical background and methodology.
- Value: ⭐⭐⭐⭐ Significant discovery value for high-redshift astronomy; more than doubling the sample sizes of five rare populations constitutes a substantive contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[ICML 2025\] Mixture-of-Expert Variational Autoencoders for Cross-Modality Embedding of Type Ia Supernova Data](../../ICML2025/physics/mixture-of-expert_variational_autoencoders_for_cross-modality_embedding_of_type_.md)
- [\[NeurIPS 2025\] From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE & Normalizing Flows](from_images_to_physics_probabilistic_inference_of_galaxy_parameters_and_emission.md)
- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)
- [\[NeurIPS 2025\] From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)

</div>

<!-- RELATED:END -->
