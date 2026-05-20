---
title: >-
  [Paper Note] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking
description: >-
  [NeurIPS 2025][Image Generation][irregular time series] This paper proposes a two-stage framework for generating regular time series from irregularly sampled data: (1) a TST autoencoder completes missing values to constr…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "irregular time series"
  - "diffusion model"
  - "image representation"
  - "completion"
  - "masking"
  - "ImagenTime"
date: 2026-05-08
content_hash: 1427eb72e6c86416
---

# A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking

**Conference**: NeurIPS 2025
**arXiv**: [2510.06699](https://arxiv.org/abs/2510.06699)  
**Code**: [GitHub](https://github.com/azencot-group/ImagenI2R)  
**Area**: Diffusion Models / Time Series Generation
**Keywords**: irregular time series, diffusion model, image representation, completion, masking, ImagenTime

## TL;DR
This paper proposes a two-stage framework for generating regular time series from irregularly sampled data: (1) a TST autoencoder completes missing values to construct a "natural neighborhood," and (2) a masking strategy applied during visual diffusion model training computes loss only on observed pixels, avoiding over-reliance on completed values. The approach achieves an average 70% improvement in discriminative score and a 6.5× training speedup.

## Background & Motivation

**Background**: Synthetic time series generation has important applications in healthcare, finance, and science. Real-world time series are often irregularly sampled (non-uniform intervals, missing values). Recent work ImagenTime achieves state-of-the-art performance on regular time series generation by mapping sequences to images and applying visual diffusion models, but it cannot handle irregular data.

**Limitations of Prior Work**: (a) Existing irregular time series generation methods (GT-GAN, KoVAE) are based on GAN/VAE architectures, which have been surpassed by diffusion models; (b) they rely on computationally expensive NCDE preprocessing and fail to scale to long sequences (KoVAE training is 6.5× slower than the proposed method); (c) these methods fully trust NCDE-completed data as samples from the true distribution—an overly strong assumption—resulting in performance far inferior to models trained on regular data (on average 540% worse).

**Key Challenge**: Naively extending ImagenTime to irregular data (with simple masking and zero-filling of missing values) produces "unnatural neighborhoods"—zero values mixed with valid values—causing convolutional kernels to learn incorrect local patterns. However, fully trusting completed values without masking introduces completion errors.

**Goal**
- How can visual diffusion models (ImagenTime) efficiently handle irregular time series?
- How can the framework balance exploiting completion information (to construct natural neighborhoods) against avoiding completion errors (via masking)?

**Key Insight**: Through carefully designed toy experiments, the authors find that zero-padding creates unnatural neighborhoods that distract convolutional kernels toward invalid pixels, degrading score estimation. The solution is to first complete (constructing natural neighborhoods) and then apply masking (preventing over-reliance on completed values), with the combination yielding optimal performance.

**Core Idea**: Completion creates natural neighborhoods so that convolutional kernels learn correctly; masking prevents over-reliance on completed values. The two mechanisms are complementary.

## Method

### Overall Architecture
The method consists of two stages: (1) **Completion**: a TST (Time Series Transformer) autoencoder completes the irregularly sampled sequence into a regular sequence; (2) **Diffusion Generation**: the completed regular sequence is transformed into an image via delay embedding and used to train a visual diffusion model (U-Net), with the loss masked to only the pixels corresponding to actual observations. At inference time, images are generated directly from noise and converted back to regular time series via inverse delay embedding.

### Key Designs

1. **Discovery and Analysis of the Unnatural Neighborhood Problem**

    - **Function**: Identify and quantify the failure mode of simple zero-filling combined with masking.
    - **Mechanism**: In a toy experiment, 2D Gaussian mixture points are mapped to 3×4 images with only the two center pixels populated and the rest set to zero. Three diffusion models are trained: (a) full-image denoising (score = 0.71), (b) masked loss only on center pixels with zero-filling (score = 0.67), (c) neighborhood filled via completion followed by masking (score = 0.32). Convolutional kernel visualizations show that zero-filling disperses kernel attention onto invalid zero pixels, while completion + masking focuses kernels on meaningful pixels.
    - **Design Motivation**: This analysis constitutes the core motivation of the paper—explaining why naive masking is insufficient and why completion is necessary.

2. **TST Autoencoder Completion**

    - **Function**: Efficiently complete irregular sequences into regular sequences.
    - **Mechanism**: A Time Series Transformer leverages self-attention to model temporal dependencies. The encoder receives irregular inputs (with positional encodings marking observation times), and a GRU decoder outputs regularly sampled sequences. Compared to NCDE, TST requires no differential equation solving and is substantially more computationally efficient.
    - **Design Motivation**: TST is lightweight and parallelizable, capable of handling long sequences (up to 65K points). Critically, completion need not be perfect—subsequent masking provides protection, so completion only needs to supply a "reasonable neighborhood."

3. **Masked Diffusion Training**

    - **Function**: Exclude completed pixels from the diffusion training loss, optimizing only on pixels corresponding to actual observations.
    - **Mechanism**: The completed regular sequence is mapped to an image $x_{\text{img}}$ via delay embedding, and a binary mask $m$ is generated to indicate which pixels originate from true observations. The denoising objective is computed only at positions where $m=1$. During inference, no mask is needed; complete images are generated directly.
    - **Design Motivation**: Masking prevents the model from memorizing specific completed values, while still utilizing the neighborhood structure that completion provides. This is the key to the complementarity between completion and masking.

4. **Improved Inverse Delay Embedding**

    - **Function**: Replace the original ImagenTime first-pixel selection with average aggregation.
    - **Mechanism**: Original ImagenTime selects only the value of the first pixel mapped to each time point during the inverse transformation. This work instead averages all pixels corresponding to each time point, exploiting redundant information in the image.
    - **Design Motivation**: A simple yet effective improvement that leverages multiple predictions made by the diffusion model at different image positions for the same time point.

### Loss & Training
- TST autoencoder: pretrained on irregular data to reconstruct regular sequences.
- Diffusion model: standard denoising score matching with a mask applied to the loss.
- Inference is identical to standard ImagenTime—no mask or completion required.

## Key Experimental Results

### Main Results: Discriminative Score (Length=24, averaged over 30%/50%/70% missing rates)

| Method | ETTh1 | ETTh2 | ETTm1 | Stock | Avg. |
|--------|-------|-------|-------|-------|------|
| TimeGAN-Δt | 0.499 | 0.499 | 0.499 | 0.479 | ~0.49 |
| GT-GAN | 0.471 | 0.369 | 0.412 | 0.249 | ~0.38 |
| KoVAE | 0.197 | 0.081 | 0.050 | 0.118 | ~0.11 |
| **Ours** | **0.037** | **0.009** | **0.012** | **0.008** | **~0.02** |

Lower discriminative score is better (0.5 = random chance). The proposed method improves over KoVAE by an average of **70%**.

### Training Efficiency

| Method | Avg. Training Time, Length=24 (h) | Avg. Training Time, Length=768 (h) |
|--------|----------------------------------|-----------------------------------|
| KoVAE | ~7.5 | ~39.4 |
| **Ours** | **~1.5** | **~6.6** |

Average speedup of **6.5×**, with greater advantage on longer sequences (6× speedup at length 768).

### Ablation Study

| Configuration | Discriminative Score (Stocks, L=24) |
|---------------|-------------------------------------|
| Zero-filling + no masking | 0.71 |
| Zero-filling + masking | 0.67 |
| **Completion + masking** | **0.32** |

### Key Findings
- **Both completion and masking are necessary**: masking alone (with zero-filling) yields limited gains; completion alone without masking over-relies on completed values; their combination achieves the best results.
- **TST vs. NCDE**: replacing NCDE with TST alone (without masking) performs poorly, but TST + masking substantially outperforms the NCDE-based approach, as masking compensates for TST's lower completion fidelity.
- **Scalability to very long sequences**: delay embedding compresses long sequences into images (65K → 256×256), enabling the method to handle extremely long time series—beyond the capability of NCDE-based methods.
- **Improved inverse transform helps**: average aggregation outperforms first-pixel selection on most datasets.

## Highlights & Insights
- **Precise diagnosis of the "unnatural neighborhood" problem** is the central contribution: through toy experiments and convolutional kernel visualizations, the paper clearly demonstrates why zero-filling + masking is insufficient—kernels cannot distinguish "true zeros" from "missing zeros," causing attention to be dispersed. The analysis is intuitive and persuasive.
- **The complementary completion + masking framework** is elegantly designed: completion provides a reasonable local context for convolutional kernels (without modifying the convolutional architecture), while masking prevents the model from memorizing completed values (without assuming perfect completion). Each component addresses what the other cannot.
- **Architecture-agnostic masking**: the masking strategy requires no modification to the diffusion model architecture, making it compatible with any backbone including U-Net and DiT.
- **The paradigm of reformulating time series as images** (via delay embedding) remains effective in the irregular setting, extending the applicability of ImagenTime.

## Limitations & Future Work
- **Requires known observation timestamps**: the method assumes knowledge of which time points are observed and which are missing; it is not applicable to fully unstructured missing data scenarios.
- **Linear mixing matrix assumption**: delay embedding is a linear transformation, which may limit the representation of nonlinear temporal structures.
- **Sensitivity to completion quality**: although masking mitigates dependence on completion, severely poor completion still yields unnatural neighborhoods.
- **Low-dimensional multivariate settings only**: experiments involve low-dimensional time series; performance on high-dimensional multivariate data remains unexplored.
- **High missing rate regimes**: at missing rates exceeding 90%, completion quality and model performance may degrade sharply.

## Related Work & Insights
- **vs. ImagenTime**: ImagenTime handles only regular time series; this work extends it to irregular settings via completion + masking while keeping the inference pipeline unchanged.
- **vs. GT-GAN / KoVAE**: both methods adopt a two-stage pipeline (NCDE completion + generation), but NCDE is computationally expensive and assumes perfect completion; the proposed method uses TST for efficiency and masking to relax the perfect-completion assumption.
- **vs. time series diffusion methods (CSDI, TimeGrad)**: these methods apply diffusion directly in the temporal domain without exploiting the locality advantage of image representations; the proposed image representation + visual U-Net yields superior generation quality.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The discovery of the "unnatural neighborhood" problem and the completion + masking solution are genuine contributions, though each individual component (TST, masking, ImagenTime) is drawn from prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 10 datasets, 3 missing rates, 3 sequence lengths, and 12 evaluation tasks—highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is communicated clearly through toy experiments; method description is well-structured.
- **Value**: ⭐⭐⭐⭐ High practical value: 70% improvement in discriminative score and 85% reduction in training time make the approach directly deployable for healthcare and financial time series generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising](tidmad_time_series_dataset_for_discovering_dark_matter_with_ai_denoising.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[NeurIPS 2025\] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[AAAI 2026\] SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](../../AAAI2026/image_generation/simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)

</div>

<!-- RELATED:END -->
