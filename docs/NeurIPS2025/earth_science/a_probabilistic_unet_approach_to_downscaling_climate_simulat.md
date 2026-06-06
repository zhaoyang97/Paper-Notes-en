---
title: >-
  [Paper Note] A Probabilistic U-Net Approach to Downscaling Climate Simulations
description: >-
  [NeurIPS 2025][Earth Science][Probabilistic U-Net] This work presents the first application of a probabilistic U-Net to statistical climate downscaling (16× super-resolution). By sampling from a variational latent space…
tags:
  - "NeurIPS 2025"
  - "Earth Science"
  - "Probabilistic U-Net"
  - "climate downscaling"
  - "variational inference"
  - "extreme events"
  - "training objectives"
date: 2026-05-08
content_hash: 4866229d7f32cfab
---

# A Probabilistic U-Net Approach to Downscaling Climate Simulations

**Conference**: NeurIPS 2025
**arXiv**: [2511.03197](https://arxiv.org/abs/2511.03197)  
**Code**: [https://github.com/MaryamAlipourH/prob-unet-climate-downscaling](https://github.com/MaryamAlipourH/prob-unet-climate-downscaling)  
**Area**: AI for Science / Climate Science
**Keywords**: Probabilistic U-Net, climate downscaling, variational inference, extreme events, training objectives

## TL;DR

This work presents the first application of a probabilistic U-Net to statistical climate downscaling (16× super-resolution). By sampling from a variational latent space, the model generates ensemble forecasts for uncertainty quantification. The paper systematically compares four training objectives — WMSE, MS-SSIM, WMSE-MS-SSIM, and afCRPS — revealing complementary trade-offs between extreme event capture and fine-scale spatial variability preservation.

## Background & Motivation

**Background**: Global climate models are computationally constrained to run at coarse spatial resolutions (tens of kilometers). Yet downstream impact studies — such as hydrological risk assessment and urban heat island analysis — require fine-scale (~10 km) meteorological fields. Dynamic downscaling (nesting regional climate models within global models) can produce fine-scale outputs but is prohibitively expensive for generating large ensembles. Statistical downscaling — learning a coarse-to-fine mapping via machine learning — offers an efficient alternative.

**Limitations of Prior Work**: Deterministic deep learning approaches (e.g., standard U-Net trained with MSE) suffer from two fundamental shortcomings: (1) the MSE-optimal solution is the conditional mean, which inherently produces over-smoothed fields that erase fine-scale structure and spatial texture; (2) extreme events (heavy precipitation, extreme temperatures) are rare in training data, providing almost no learning incentive under MSE, causing systematic underestimation. Yet in climate impact research, it is precisely extreme events that drive flood, drought, and heat wave risks.

**Key Challenge**: Climate downscaling is intrinsically a one-to-many mapping — a single coarse-resolution field can correspond to many physically plausible fine-scale realizations. Deterministic models produce only a single "average" prediction, unable to express this inherent uncertainty. Furthermore, extreme event capture and spatial variability preservation exhibit inherent tension under different training objectives.

**Goal**: (1) Introduce the probabilistic U-Net to provide uncertainty quantification for climate downscaling; (2) Systematically evaluate which training objective best captures extreme events and spatial detail.

**Key Insight**: The probabilistic U-Net was originally designed for medical image segmentation — where a single medical image may admit multiple valid annotations. The analogy to climate downscaling is direct: a single coarse field can correspond to multiple plausible fine-scale realizations, making the variational latent sampling mechanism a natural fit.

**Core Idea**: Leverage the variational latent space of the probabilistic U-Net to generate downscaling ensembles, and reveal the extreme-event vs. spatial-variability trade-off through systematic comparison of four training objectives.

## Method

### Overall Architecture

The input is a 16× coarse-resolution meteorological field ($8 \times 8$ grid points), bilinearly upsampled via nearest-neighbor interpolation to match the spatial dimensions of the high-resolution target ($128 \times 128$ grid points, ~12 km resolution). The model predicts the **residual** between the interpolated field and the true high-resolution field. The probabilistic U-Net framework wraps a deterministic U-Net backbone; at inference time, multiple latent variables $z$ are sampled from the prior network to produce an ensemble of high-resolution realizations, thereby quantifying downscaling uncertainty.

### Key Designs

1. **U-Net Backbone**:

    - **Function**: Deterministic feature extraction and upsampling for large-resolution meteorological fields.
    - **Mechanism**: A four-level U-Net following the StyleGAN/EDM architecture. The encoder halves spatial resolution and doubles channels from 64 to 256 at each level; the decoder mirrors this symmetrically. The encoder employs 2 residual blocks per level and the decoder 3, with skip connections concatenating features at corresponding scales. Upsampling uses nearest-neighbor interpolation followed by $3 \times 3$ convolution to avoid the checkerboard artifacts of transposed convolutions.
    - **Design Motivation**: Residual learning (predicting the difference between the interpolated and true fields) reduces learning difficulty, allowing the network to focus on modeling fine-scale structure rather than large-scale patterns.

2. **Probabilistic U-Net Variational Framework**:

    - **Function**: Introduces uncertainty representation into downscaling — generating multiple plausible realizations from the same input.
    - **Mechanism**: The prior network $P(z|X)$ conditions only on the coarse-resolution input $X$; the posterior network $Q(z|X,Y)$ additionally conditions on the true high-resolution target $Y$. Both output axis-aligned Gaussian distributions over a 16-dimensional latent space. During training, $z$ is sampled from the posterior, broadcast as a feature map, concatenated to the final U-Net activation, and decoded through three $1 \times 1$ convolutions. The total loss is a reconstruction term plus $\gamma \cdot \text{KL}(Q \| P)$, where $\gamma$ is gradually increased after a warm-up period. At inference, multiple $z$ samples are drawn from the prior $P(z|X)$, each producing one high-resolution realization.
    - **Design Motivation**: The variational framework explicitly models downscaling uncertainty as a distribution over the latent space — sampling multiple $z$ values yields an ensemble forecast, with inter-member variability serving as a measure of uncertainty.

3. **Physics-Constrained Reparameterization**:

    - **Function**: Ensures network outputs satisfy basic physical laws.
    - **Mechanism**: Precipitation is passed through softplus $\log(1 + e^{x+c})$ (with $c=10^{-7}$) to guarantee non-negativity; for temperature, softplus is applied to $T_{\max} - T_{\min}$ to enforce $T_{\max} \geq T_{\min}$.
    - **Design Motivation**: Enforcing physical constraints directly at the output layer — rather than via post-hoc clipping — is both elegant and guarantees physical consistency across all ensemble members.

### Loss & Training

A central contribution of this paper is a systematic comparison of four training objectives as alternatives to cross-entropy loss:

1. **WMSE ($\lambda=1$)**: Weighted MSE with weights $w(Y_i) = \min\{\alpha e^{\beta Y_i}, 1\}$ assigning higher weight to large precipitation values ($\alpha=0.007, \beta=0.048$). However, it remains a pixel-wise loss and is susceptible to spectral smoothing.

2. **MS-SSIM ($\lambda=0$)**: Multi-Scale Structural Similarity Index, which focuses on pattern matching of local luminance, contrast, and structure rather than pixel-wise error, better preserving spatial texture and fine detail.

3. **WMSE-MS-SSIM ($\lambda=0.158$)**: A tuned weighted combination of the above two, with $\lambda$ determined via hyperparameter search, aiming to balance pixel-wise accuracy and structural fidelity.

4. **afCRPS ($\eta=0.95$)**: Almost Fair Continuous Ranked Probability Score, designed for training generative models. It penalizes deviations of ensemble members from the true value while rewarding inter-member spread, encouraging ensemble diversity and distributional calibration.

Training runs for 10 epochs with batch size 32 and latent dimensionality 16.

## Key Experimental Results

### Main Results

**Dataset**: ClimEx climate simulation ensemble (southern Quebec and the Maritime Provinces, Canada), at 0.11° (~12 km) resolution. Training: 1960–1990; validation: 1990–1997; test: 1998–2005. Variables: daily total precipitation (mm), daily minimum/maximum temperature (°C).

| Loss | CRPS Precip. | CRPS $T_{\min}$ | CRPS $T_{\max}$ | MAE Precip. | MAE $T_{\min}$ | MAE $T_{\max}$ |
|------|-------------|----------------|----------------|------------|---------------|---------------|
| afCRPS | **0.94** | **0.68** | **0.62** | 1.35 | **0.90** | **0.75** |
| MS-SSIM ($\lambda=0$) | 1.07 | 0.86 | 0.68 | 1.29 | 1.06 | 0.88 |
| WMSE ($\lambda=1$) | 1.13 | 0.78 | 0.59 | **1.19** | 0.94 | 0.74 |
| WMSE-MS-SSIM ($\lambda=0.158$) | 1.06 | 0.85 | 0.66 | 1.27 | 1.05 | 0.85 |
| Nearest-neighbor baseline | — | — | — | 1.51 | 1.76 | 1.30 |

### Ablation Study

| Evaluation Dimension | afCRPS | WMSE ($\lambda=1$) | MS-SSIM ($\lambda=0$) | WMSE-MS-SSIM ($\lambda=0.158$) |
|---------------------|--------|-------------------|----------------------|-------------------------------|
| Extreme events (return level) | Tends to overestimate | Underestimates | Reasonable | **Best** |
| Spectral fidelity (PSD) | **Best** | Severely underestimates high freq. | Reasonable | Reasonable |
| Distribution tail (log histogram) | Overestimates tail | Severely underestimates | Reasonable | Reasonable |
| Temperature fields | All variants perform well | All variants perform well | All variants perform well | All variants perform well |

### Key Findings

- **No single loss function dominates across all evaluation dimensions** — this is the most important empirical finding of the paper.
- **afCRPS excels at spectral fidelity and CRPS metrics**, but tends to overestimate extreme precipitation events; the spread-rewarding term encourages ensemble diversity, potentially causing excessive dispersion in the tail.
- **WMSE-MS-SSIM ($\lambda=0.158$) achieves the best extreme event capture**, with predicted empirical return levels falling within the 95% confidence band of the true values.
- **Pure WMSE ($\lambda=1$) fails comprehensively on spectral fidelity** — high-wavenumber variance is severely underestimated, effectively producing spatially smooth outputs just as standard MSE does.
- **Temperature fields are insensitive to the choice of loss function**; all variants reproduce temperature distributions well — extreme precipitation values represent the core challenge.
- The authors suggest that combining afCRPS with MS-SSIM may yield a more balanced solution, though this is not experimentally validated.

## Highlights & Insights

- **Cross-domain transfer from medicine to climate**: The transfer of the probabilistic U-Net from segmentation to downscaling is highly natural — both are one-to-many mapping problems. This cross-domain transfer strategy offers a reference for other uncertainty modeling tasks such as multi-modal prediction and super-resolution.
- **Residual learning reduces problem difficulty**: Having the network predict the difference between the coarse interpolation and the true field — rather than the target field directly — means large-scale patterns are handled by interpolation, and the network focuses solely on modeling fine-scale details. This is a simple but effective engineering decision.
- **Three-dimensional qualitative evaluation framework**: Beyond CRPS/MAE metrics, the paper evaluates models along three complementary dimensions: return levels (extreme value statistics), power spectral density (spatial frequency content), and log-frequency histograms (distributional shape). This analysis framework is broadly applicable to meteorological downscaling and super-resolution work.
- **Physics constraints embedded in the network**: Physical laws (precipitation $\geq 0$, $T_{\max} \geq T_{\min}$) are enforced directly at the output layer via reparameterization, which is more principled than post-hoc clipping and guarantees physical consistency across all ensemble members.

## Limitations & Future Work

- **Single ensemble member, single region**: Only one ClimEx member is used, covering a single region in eastern Canada. Generalizability to different climate regimes (tropical, arid) and different climate model outputs remains unknown.
- **Temporal independence assumption**: Each time step is downscaled independently, without modeling the temporal continuity and autocorrelation of weather events. This is a critical limitation for assessing extreme event duration (e.g., multi-day heavy rainfall).
- **No integration of extreme event and spatial variability objectives**: The authors identify afCRPS and MS-SSIM as complementary but do not experimentally validate their combination. A hybrid afCRPS + MS-SSIM training objective is an explicit direction for future work.
- **Fixed downscaling factor**: Only 16× downscaling ($8 \times 8 \to 128 \times 128$) is evaluated; other factors and multi-scale cascading are unexplored.
- **Absence of comparisons with concurrent methods**: Contemporary approaches such as diffusion model-based downscaling (Watt & Mansfield 2024) and conditional normalizing flows (Winkler et al. 2024) are not directly benchmarked against.
- **Limited return level evaluation**: Thirty years of test data provide insufficient statistical robustness for estimating high return period events (e.g., 100-year return levels).

## Related Work & Insights

- **vs. Deterministic U-Net + MSE**: The probabilistic U-Net generates ensembles via latent space sampling, providing uncertainty quantification and avoiding the over-smoothing inherent to MSE. However, inference cost is higher (one forward pass per $z$ sample).
- **vs. GAN-based downscaling (Annau et al. 2023)**: GANs can produce visually sharp outputs, but adversarial training is unstable and "hallucination" effects may generate physically implausible extreme values. The probabilistic U-Net trains stably and offers a clear probabilistic interpretation.
- **vs. Diffusion model downscaling (Bassetti et al. 2024)**: Diffusion models generally achieve superior spectral fidelity, but inference requires many denoising steps (tens to hundreds), incurring far greater computational cost than the probabilistic U-Net's single-sample forward pass.
- **Insight**: The choice of training objective profoundly shapes generative model behavior — different losses exhibit systematic differences in distributional tail behavior, spectral characteristics, and calibration. This insight generalizes beyond climate science to medical image super-resolution, weather forecasting, and related domains.

## Rating

- **Novelty**: ⭐⭐⭐ — The probabilistic U-Net itself is not new; its transfer to climate downscaling is a reasonable but incremental contribution. The systematic comparison of four loss functions constitutes the core academic value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The combination of three-dimensional qualitative analysis and quantitative metrics is comprehensive, though comparisons with other generative models are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, systematic analysis, high-quality figures, and a substantive discussion of the extreme event vs. spatial variability trade-off.
- **Value**: ⭐⭐⭐ — Practically useful to the climate downscaling community; the empirical guidance on loss function selection is directly adoptable. Methodological novelty is, however, limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive Online Emulation for Accelerating Complex Physical Simulations](adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)
- [\[NeurIPS 2025\] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)
- [\[NeurIPS 2025\] Predicting Public Health Impacts of Electricity Usage](predicting_public_health_impacts_of_electricity_usage.md)
- [\[NeurIPS 2025\] Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)
- [\[ICML 2026\] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](../../ICML2026/earth_science/sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)

</div>

<!-- RELATED:END -->
