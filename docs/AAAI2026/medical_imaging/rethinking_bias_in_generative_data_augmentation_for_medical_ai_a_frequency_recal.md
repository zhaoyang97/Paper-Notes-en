---
title: >-
  [Paper Note] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach
description: >-
  [AAAI 2026][Medical Imaging][Generative data augmentation] This paper identifies high-frequency distribution discrepancies between AI-generated and real medical images as the root cause of unreliable generative data augm…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Generative data augmentation"
  - "frequency bias"
  - "medical image classification"
  - "denoising autoencoder"
  - "frequency recalibration"
date: 2026-05-08
content_hash: 1a58087081b1c071
---

# Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach

**Conference**: AAAI 2026
**arXiv**: [2511.12301](https://arxiv.org/abs/2511.12301)  
**Code**: None  
**Area**: Medical Imaging / Data Augmentation / Frequency Domain Analysis
**Keywords**: Generative data augmentation, frequency bias, medical image classification, denoising autoencoder, frequency recalibration

## TL;DR

This paper identifies high-frequency distribution discrepancies between AI-generated and real medical images as the root cause of unreliable generative data augmentation (GDA), and proposes FreRec (Frequency Recalibration), a coarse-to-fine post-processing module comprising Statistical High-frequency Replacement (SHR) and Reconstructive High-frequency Mapping (RHM) to align frequency distributions, consistently improving downstream medical image classification performance as a plug-and-play component.

## Background & Motivation

Medical AI relies on large-scale datasets, yet data scarcity is common due to privacy concerns, acquisition costs, and class imbalance. GDA leverages GANs and diffusion models to synthesize realistic medical images for training set augmentation, and has been widely applied across brain MRI, chest X-ray, fundus imaging, and other modalities.

Nevertheless, the reliability of GDA has been substantially underestimated:

**Bias observed in general AI**: Language models collapse after iterative training on generated content; visual models using GDA fail to yield consistent improvements.

**Overlooked in medical settings**: Although many studies report positive outcomes from GDA, whether AI-synthesized samples are consistently beneficial or may introduce harmful features remains unclear.

**Frequency discrepancy as the key clue**: Recent studies reveal significant differences in high-frequency components between AI-generated and real images. Medical images (MRI, X-ray, etc.) are particularly sensitive to high-frequency variation due to their imaging characteristics and reliance on subtle pathological details.

The central hypothesis of this paper is that frequency distribution discrepancies between real and synthetic images—as evidenced by spectral differences in Figure 1—are the primary driver of GDA instability. Unlike approaches that require retraining generative models, FreRec operates as an independent post-processing step compatible with any generative model, incurring low overhead and requiring no modification of the upstream pipeline.

## Method

### Overall Architecture

FreRec adopts a coarse-to-fine two-stage strategy to align the frequency distributions of synthetic and real images:

- **Stage 1 — SHR (Statistical High-frequency Replacement)**: Replaces the high-frequency components of synthetic images with statistically derived high-frequency representations from real images, achieving coarse alignment.
- **Stage 2 — RHM (Reconstructive High-frequency Mapping)**: A denoising autoencoder trained exclusively on real images maps the SHR-perturbed images onto the natural frequency manifold, restoring image quality and refining high-frequency details.

The authors recommend deploying FreRec as a unified preprocessing module applied to both training and inference stages of the downstream classifier—all samples (synthetic and real) are processed through FreRec to ensure frequency distribution consistency.

### Key Designs

1. **SHR (Statistical High-frequency Replacement)**: After applying the Fourier transform to a synthetic image $x_i^S$, a binary mask $\mathcal{M}$ with a fixed ratio $r$ partitions the spectrum into low-frequency $\mathcal{F}^l$ and high-frequency $\mathcal{F}^h$ components. The key innovation is that replacement is not performed one-to-one; instead, the Top-$K$ real images $\{x_k^R\}_{k=1}^K$ most similar in SSIM are retrieved, and their high-frequency components' per-channel mean/standard deviation statistics (modeled as Gaussian) are used to sample new mean $\hat{\mu}$ and standard deviation $\hat{\sigma}$, which recalibrate the high-frequency components of the synthetic image: $\hat{\mathcal{F}}_i^{Sh} = \hat{\sigma} \cdot \frac{\mathcal{F}_i^{Sh} - \mu}{\sigma} + \hat{\mu}$. **Design Motivation**: One-to-one replacement is highly stochastic and cannot guarantee distribution-level alignment; statistics-based batch replacement is more stable. SSIM-guided retrieval minimizes replacement perturbation while preserving semantic content.

2. **RHM (Reconstructive High-frequency Mapping)**: Although SHR achieves coarse frequency alignment, it degrades image quality. Directly reconstructing $\hat{x}_i^S \to x_i^S$ would revert to the original frequency-mismatched image; therefore, a one-directional manifold mapping is designed. **Mechanism**: A denoising autoencoder $\mathcal{A}: \hat{x}_i^R \to x_i^R$ is trained solely on real images (with SHR-perturbed real images as input), learning the natural frequency manifold $\mathbf{z}_{\mathcal{F}}^R$. After training, $\mathcal{A}^*$ is applied to synthetic images—since SHR has mapped both synthetic and real images to the same starting space, the autoencoder projects synthetic images onto the natural frequency manifold along the reconstruction trajectory learned from real images. **Design Motivation**: By sharing the starting space (via SHR alignment) and constraining reconstruction direction (learned from real images only), the method avoids the degenerate case where naive denoising reverts to a frequency-mismatched state.

3. **FET-Block and FESA Module**: The autoencoder $\mathcal{A}$ is based on the Restormer architecture, with Transformer blocks replaced by Frequency-enhanced Transformer blocks (FET-blocks), comprising: (a) a global spatial self-attention branch processing RGB features with transposed attention $\hat{\mathbf{F}}_{rgb} = C_1(\text{softmax}(Q \cdot K / a) \cdot V)$; (b) a local frequency self-attention branch (FESA) that transforms RGB features via FFT to obtain the amplitude spectrum, divides it into annular regions along the radial direction—each ring corresponding to a channel—and derives frequency-enhanced features via sigmoid-gated local attention and iFFT; (c) the two branches are concatenated, fused, and added with a residual connection. The network comprises 4 levels with 2, 4, 6, and 8 FET-blocks respectively, with an additional 2 blocks in the refinement stage.

### Loss & Training

The autoencoder training loss combines pixel reconstruction loss and spectral similarity loss:

$$\min_{\mathcal{A}} \mathcal{L} = \underbrace{\|x_i^R - \mathcal{A}(\hat{x}_i^R)\|^2}_{\text{pixel similarity}} + \underbrace{\|\mathcal{F}(x_i^R) - \mathcal{F}(\mathcal{A}(\hat{x}_i^R))\|^2}_{\text{frequency similarity}}$$

The spectral loss ensures accurate frequency-domain reconstruction, preventing the autoencoder from introducing additional frequency distortions. SHR hyperparameters: mask ratio $r = 0.5$, number of retrieved samples $K = 200$.

## Key Experimental Results

### Main Results

Classification results on three medical datasets (brain tumor MRI / cardiomegaly chest X-ray / diabetic retinopathy fundus images) across three classifiers (ResNet50 / DenseNet / ViT-B-16):

| Dataset | Method | DenseNet AUC | ResNet50 AUC | ViT AUC |
|---------|--------|-------------|-------------|---------|
| Cardiomegaly | RAW | 0.842 | 0.834 | 0.832 |
| Cardiomegaly | GDA (no calibration) | 0.871 | 0.834 | 0.848 |
| Cardiomegaly | **GDA+FreRec** | **0.899** | **0.888** | **0.888** |
| Brain Tumor | RAW | 0.840 | 0.793 | 0.753 |
| Brain Tumor | GDA (no calibration) | 0.794 | 0.783 | 0.758 |
| Brain Tumor | **GDA+FreRec** | **0.855** | **0.843** | **0.787** |
| Diabetic Retinopathy | RAW | 0.840 | 0.843 | 0.834 |
| Diabetic Retinopathy | GDA (no calibration) | 0.863 | 0.848 | 0.834 |
| Diabetic Retinopathy | **GDA+FreRec** | **0.879** | **0.878** | **0.852** |

**Key Finding**: **GDA without FreRec can be harmful**—on the brain tumor dataset, all classifiers under GDA underperform RAW (e.g., DenseNet AUC: 0.794 vs. 0.840), confirming GDA unreliability. With FreRec, GDA consistently improves performance across all evaluations, becoming a reliable augmentation strategy.

### Ablation Study

| Configuration | Cardiomegaly AUC | Cardiomegaly Acc | PSNR | SSIM |
|---------------|-----------------|-----------------|------|------|
| SHR only | 0.81 | 0.79 | 25.10 | 0.76 |
| RHM only (w/o FESA) | 0.85 | 0.79 | **36.44** | **0.98** |
| RHM only (w/ FESA) | 0.87 | 0.82 | 35.51 | 0.96 |
| **Full FreRec** | **0.89** | **0.84** | 35.62 | 0.95 |

### Key Findings

1. **SHR alone is insufficient**: Although it achieves coarse frequency alignment, it severely degrades image quality (PSNR of only 25.10), yielding no improvement—or even degradation—in classification. This indicates that frequency alignment must be coupled with quality restoration.
2. **RHM is the critical component**: Introducing RHM on top of SHR yields substantial gains (AUC: 0.81→0.85+), serving as the key step for fine-grained calibration and detail recovery.
3. **FESA improves classification but slightly reduces reconstruction quality**: Incorporating frequency-enhanced attention raises classification AUC from 0.85 to 0.87, while PSNR marginally decreases (36.44→35.51). Full FreRec achieves the best trade-off between classification performance and reconstruction quality.
4. **Grayscale images are more thoroughly calibrated than color images**: T-SNE visualizations show complete overlap between synthetic and real feature distributions on brain MRI and X-ray datasets after calibration, whereas overlap is incomplete for color fundus images, as richer pixel information in color images makes high-frequency reconstruction more challenging.
5. **Inference time is acceptable**: FreRec processes each image in approximately 15–18 ms (GTX 4090), which is lower than or comparable to the DoGE method (~33 ms), satisfying clinical deployment requirements.

## Highlights & Insights

- **Identifying the root cause of GDA bias from a frequency-domain perspective**: Rather than attributing GDA bias generically to "domain shift," this paper pinpoints high-frequency distribution discrepancies as the key factor, offering a new lens for understanding systematic bias in AI-generated content.
- **Clear empirical demonstration of GDA's dual nature**: GDA underperforming the no-augmentation baseline on the brain tumor dataset provides direct evidence that synthetic data can introduce harmful features.
- **Practical plug-and-play design**: FreRec requires no retraining of the generative model—it operates purely as post-processing on generated images and is compatible with any GAN or diffusion model, incurring minimal engineering overhead.
- **Elegant statistical replacement in SHR**: Rather than one-to-one replacement, Top-K statistical sampling ensures distribution-level alignment while introducing controlled stochasticity.
- **Unified FreRec processing across training and inference is practically motivated**: Given that the autoencoder is imperfect, applying it uniformly to both real and synthetic images eliminates residual discrepancies.

## Limitations & Future Work

- Validation is limited to classification tasks; extension to more complex downstream tasks such as segmentation and detection remains unexplored.
- Calibration effectiveness on color medical images (e.g., fundus photographs) is inferior to that on grayscale images, indicating limited capacity for handling high-dimensional pixel information.
- Only three generative models (FastGAN, StyleGAN3, and VC-Diffusion) are evaluated; the method has not been tested on images generated by recent large-scale diffusion models (e.g., Stable Diffusion 3, DALL-E 3).
- The autoencoder must be trained on real images from the target domain, limiting applicability when real data are extremely scarce (e.g., rare diseases).
- The computational cost of SSIM-based Top-K retrieval in SHR scales with dataset size, requiring efficiency optimization for large-scale scenarios.

## Related Work & Insights

- Compared to methods that incorporate frequency regularization during generative model training (e.g., Durall et al.), FreRec's post-processing paradigm is more practical—it requires no access to or modification of the generative model and is applicable to synthetic images from any source.
- The frequency-domain bias analysis framework can generalize to quality assessment and calibration of other AI-generated modalities (text, audio).
- The coarse-to-fine two-stage calibration strategy—statistical coarse alignment followed by learned fine mapping—constitutes a general distribution alignment paradigm transferable to other domain adaptation scenarios.
- Practical implication for medical AI: the frequency distribution characteristics of synthetic samples should be verified before applying GDA, to avoid blindly increasing synthetic data volume.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Explaining GDA bias from a frequency-domain perspective is original; the two-stage calibration design is well-motivated
- **Technical Depth**: ⭐⭐⭐⭐ — The FESA module and joint loss design demonstrate depth; theoretical assumptions are empirically validated
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, three classifiers, multiple baselines, ablations, and visualizations provide comprehensive coverage
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, compatible with any generative model, low inference overhead; highly practical
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach](../../ACL2026/medical_imaging/eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[AAAI 2026\] MPA: Multimodal Prototype Augmentation for Few-Shot Learning](mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)
- [\[NeurIPS 2025\] Demo: Generative AI helps Radiotherapy Planning with User Preference](../../NeurIPS2025/medical_imaging/demo_generative_ai_helps_radiotherapy_planning_with_user_preference.md)

</div>

<!-- RELATED:END -->
