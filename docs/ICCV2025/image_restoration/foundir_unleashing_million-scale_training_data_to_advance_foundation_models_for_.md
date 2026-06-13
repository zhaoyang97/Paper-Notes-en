---
title: >-
  [Paper Note] FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration
description: >-
  [ICCV 2025][Image Restoration][Foundation Model] This work constructs the first million-scale real-world paired image restoration dataset covering 20 degradation types, and proposes the FoundIR framework…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Foundation Model"
  - "Universal Image Restoration"
  - "Million-scale Dataset"
  - "Incremental Learning"
  - "Diffusion Model"
date: 2026-05-08
content_hash: b8e69a5c814a874f
---

# FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration

**Conference**: ICCV 2025
**arXiv**: [2412.01427](https://arxiv.org/abs/2412.01427)  
**Code**: [Project Page](https://www.foundir.net)  
**Area**: Image Restoration
**Keywords**: Foundation Model, Universal Image Restoration, Million-scale Dataset, Incremental Learning, Diffusion Model

## TL;DR

This work constructs the first million-scale real-world paired image restoration dataset covering 20 degradation types, and proposes the FoundIR framework, which combines a degradation-agnostic generalist model with degradation-aware expert models to surpass existing performance ceilings across 24 benchmarks.

## Background & Motivation

The development of foundation models for image restoration lags far behind NLP (GPT-4) and high-level CV (CLIP, SAM), primarily due to insufficient scale and quality of training data:

**Domain gap of synthetic data**: Existing universal restoration methods (AirNet, PromptIR, etc.) are trained by concatenating multiple small-scale synthetic datasets (e.g., BSD400, Rain100L, RESIDE), yet a significant domain gap exists between synthetic and real degradations. Experiments show that increasing synthetic data volume leads to a performance bottleneck.

**Scarcity of real data**: Existing real-world datasets are extremely small (SIDD ~160 pairs, RealRain ~1k pairs), far insufficient for training foundation models. Continuously adding available real data yields notable improvements, but the total volume is limited.

**Limited degradation diversity**: Existing training sets cover only a narrow set of isolated degradation types (denoising, deraining, dehazing, etc.), whereas real-world degradations are typically coupled combinations of multiple factors (e.g., low-light + noise + blur).

**Immature training strategies**: When training data scale increases dramatically and degradation types diversify, naive mixed training is prone to catastrophic forgetting and optimization difficulties.

This work advances the field simultaneously along two axes—**data** and **model**—with the aim of establishing a true foundation model for image restoration.

## Method

### Overall Architecture

FoundIR adopts a generalist–expert joint architecture: (1) a diffusion-based degradation-agnostic generalist model learns universal representations to remove degradations from diverse degraded inputs; (2) degradation-aware expert models refine quality for specific scenarios. Training employs an incremental learning strategy to overcome catastrophic forgetting in large-scale data training.

### Key Designs

1. **Million-scale Data Acquisition System**:

    - Function: Designs an electromechanical capture system to automatically collect large-scale real paired data.
    - Mechanism: A SONY ILCE-7M3 camera mounted on a motorized slider (GVM Slider 120) captures data in three rounds: the first round acquires GT images with fixed exposure (ISO ≤ 300); the second round adjusts internal camera parameters (ISO: 800–20000, shutter: 1/40–1/1000, focus mode, etc.) to obtain noise/blur/low-light LQ images; the third round modifies the external environment (turning off lights, motorized sprinklers simulating rain, water droplets on glass, etc.) to obtain weather-degraded images. Temporal alignment is ensured via a reference-marker interval matching strategy with errors < 0.2 s.
    - Design Motivation: Real paired data is the fundamental solution to breaking the synthetic data domain gap bottleneck. The three rounds respectively cover internal-parameter degradations and external-environment degradations, yielding approximately 8,500 scenes and 1.01 million high-resolution image pairs (average 2514×1516), spanning 7 isolated and 13 coupled degradation types.

2. **Degradation-Agnostic Generalist Model**:

    - Function: Employs a residual diffusion model to learn a degradation-agnostic universal representation space.
    - Mechanism: The degraded image $I_{LQ}$ is introduced as an explicit condition in the forward diffusion process: $I_t = (\bar{\alpha}_t - \bar{\beta}_t)I_{LQ} + (1-\bar{\alpha}_t)I_{HQ} + \bar{\delta}_t\epsilon$. As $t \to T$, images with different degradations progressively converge to a shared distribution $I_T = \bar{\gamma}_T I_{LQ} + \bar{\delta}_T\epsilon$, where $\bar{\gamma}_T = 1 - \bar{\beta}_T$ controls the degree of degradation-agnostic learning. The reverse process predicts the residual $R = I_{LQ} - I_{HQ}$, with training loss $\|R - R_\theta(I_t, I_{HQ}, t)\|_1$.
    - Design Motivation: Learning degradation-agnostic universal representations prevents the model from acquiring degradation-specific representations for each degradation type, thereby reducing the learning burden and inter-task competition.

3. **Incremental Learning Training Strategy**:

    - Function: Overcomes catastrophic forgetting through a two-stage task-incremental and class-incremental training procedure.
    - Mechanism: In stage one, tasks are incrementally added from isolated degradation classes $\mathcal{D}_i$ over $n$ training iterations to obtain $\theta^{id}$; in stage two, joint training is performed by sampling from both isolated and coupled degradation classes $\mathcal{D}_c$ for $2n$ iterations. This ensures that knowledge of isolated degradations provides a solid starting point for coupled degradations.
    - Design Motivation: Directly mixing all degradation data leads to severe catastrophic forgetting. Staged incremental training allows the model to progressively adapt to increasingly complex degradation combinations.

4. **Degradation-Aware Expert Model Pool**:

    - Function: Introduces expert models for quality refinement in specific degradation scenarios.
    - Mechanism: An expert model pool $E_i$ is constructed, including text restoration experts, weather experts, illumination experts, etc. The most suitable expert model is automatically selected based on the degradation pattern of the input image. Expert models adapt quickly under the guidance of generalist outputs.
    - Design Motivation: The generalist model underperforms task-specific models on individual tasks. Through expert collaboration rather than cascading, each expert shares knowledge while avoiding multi-stage error accumulation.

### Loss & Training

- **Generalist loss**: $L_1$ residual prediction loss $\|R - R_\theta(I_t, I_{HQ}, t)\|_1$
- **Incremental training**: Isolated degradations for $n$ rounds → load weights → joint training on isolated + coupled degradations for $2n$ rounds
- **Expert fine-tuning**: Each expert rapidly adapts to its specific task at low training cost, building upon generalist outputs

## Key Experimental Results

### Main Results

**PSNR comparison on 20 real-world degradation test sets (representative degradations)**

| Method | Blur | Noise | Rain | Low-light | Low-light+Noise | Low-light+Blur+Noise | Average |
|--------|------|-------|------|-----------|-----------------|----------------------|---------|
| Real-ESRGAN | 25.20 | 34.46 | 28.95 | 19.26 | 17.49 | 21.43 | 24.26 |
| AirNet | 18.07 | 19.61 | 17.61 | 6.60 | 7.75 | 8.76 | 14.03 |
| PromptIR | 21.91 | 36.57 | 27.59 | 16.33 | 10.83 | 22.63 | 22.49 |
| DiffIR | 21.88 | 37.69 | 25.77 | 16.90 | 14.81 | 22.00 | 24.68 |
| **FoundIR** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

**Effect of training data scale on performance**

| Training Data Configuration | Real-world Test PSNR Trend | Notes |
|----------------------------|---------------------------|-------|
| Synthetic data only (scaled up) | Performance saturates/degrades | Domain gap causes bottleneck |
| Synthetic + small-scale real data | Significant improvement | Real data breaks the bottleneck |
| Synthetic + large-scale real data | Continued improvement | Million-scale real data unleashes potential |

**Incremental learning vs. conventional training**

| Training Strategy | Outcome |
|------------------|---------|
| Full mixed training | Catastrophic forgetting, unstable performance |
| Batch-wise mixing | Partially alleviates the issue |
| Incremental learning (Ours) | Stable convergence, best performance |

### Key Findings

1. Data quality matters more than data quantity: increasing synthetic data volume cannot compensate for the domain gap, whereas the marginal benefit of real data far exceeds that of synthetic data.
2. Coupled degradations are the core challenge in real-world scenarios: existing methods perform reasonably on isolated degradations but suffer dramatic performance drops on coupled degradations such as low-light + noise and low-light + blur + noise.
3. Incremental learning is critical for large-scale multi-degradation training: the curriculum learning paradigm of simple-to-complex significantly improves convergence stability.
4. FoundIR achieves state-of-the-art results on 24 benchmarks, including 4 publicly available real-world benchmarks (4KRD, RealRain-1K, HazeRD, UHD-LL).

## Highlights & Insights

- **Data-driven paradigm**: Unlike most methodological innovations, this work focuses on data infrastructure, demonstrating the decisive role of high-quality large-scale data for foundation models.
- **Engineered data acquisition**: The industrial pipeline of motorized slider + three-round capture + marker-based alignment is reproducible and scalable, serving as a methodological reference.
- **Degradation-agnostic representation learning**: Rather than teaching the model to distinguish "this is noise / this is blur," the model learns universal restoration representations, reducing inter-task interference.
- **Generalist + expert hybrid architecture**: Inspired by MoE in large language models, the design balances generalization and specialization.

## Limitations & Future Work

1. The cost of collecting a million-scale dataset is extremely high (electromechanical systems, multi-round capture, manual curation), making replication by other teams difficult.
2. The dataset is predominantly composed of static scenes (camera moving on a slider), with insufficient coverage of degradations involving moving objects.
3. The automatic expert selection mechanism is not described in detail and may fail when degradation types are ambiguous.
4. Diffusion model inference is slow, making it unsuitable for real-time applications.
5. The dataset currently covers only indoor and outdoor static scenes; generalization to specialized domains (medical, remote sensing, underwater) remains to be validated.

## Related Work & Insights

- **AirNet / PromptIR**: Representative universal restoration methods, but constrained by small-scale synthetic training data.
- **Residual diffusion models**: FoundIR extends this paradigm by introducing degraded image conditioning to enable degradation-agnostic learning.
- **SIDD / FMD**: Classic real-world denoising datasets, but too small in scale.
- Insight: The next breakthrough in image restoration may lie not in model architecture but in data scale and diversity—analogous to the success of the GPT series.

## Rating

- Novelty: ⭐⭐⭐⭐ The million-scale real dataset is a significant contribution; the combination of degradation-agnostic learning and incremental training is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 24 benchmarks and 20 degradation types is exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ Systematically organized with detailed descriptions of the data acquisition pipeline.
- Value: ⭐⭐⭐⭐⭐ The dataset alone constitutes a major contribution and is poised to advance the development of foundation models for image restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](../../ICLR2026/image_restoration/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](unires_universal_image_restoration_for_complex_degradations.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)

</div>

<!-- RELATED:END -->
