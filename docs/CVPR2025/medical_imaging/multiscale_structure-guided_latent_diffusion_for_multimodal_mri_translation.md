---
title: >-
  [Paper Note] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation
description: >-
  [CVPR2025][Medical Imaging][MRI Synthesis] This paper proposes the MSG-LDM framework, which explicitly decouples style and structure information in the latent space. It extracts modality-invariant multiscale structural priors to guide the diffusion process through the High-Frequency Injection Block (HFIB), Multimodal Structural Feature Fusion (MMSF), and Multiscale Structural Feature Enhancement (MSSE), thereby addressing anatomical inconsistency and texture degradation in MR…
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "MRI Synthesis"
  - "Latent Diffusion Models"
  - "Multimodal Translation"
  - "Structure Guidance"
  - "Missing Modalities"
date: 2026-05-08
content_hash: 74ff883af7bd2c8b
---

# Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation

**Conference**: CVPR2025  
**arXiv**: [2603.12581](https://arxiv.org/abs/2603.12581)  
**Code**: [GitHub](https://github.com/ziyi-start/MSG-LDM)  
**Area**: Medical Imaging  
**Keywords**: MRI Synthesis, Latent Diffusion Models, Multimodal Translation, Structure Guidance, Missing Modalities

## TL;DR

This paper proposes the MSG-LDM framework, which explicitly decouples style and structure information in the latent space. It extracts modality-invariant multiscale structural priors to guide the diffusion process through the High-Frequency Injection Block (HFIB), Multimodal Structural Feature Fusion (MMSF), and Multiscale Structural Feature Enhancement (MSSE), thereby addressing anatomical inconsistency and texture degradation in MRI translation under arbitrary missing modalities.

## Background & Motivation

- Multimodal MRI (T1, T2, T1CE, FLAIR) provides complementary anatomical and pathological information, which is widely used in brain tumor segmentation and lesion analysis.
- In clinical practice, missing modalities frequently occur due to long acquisition times, poor patient tolerance, and equipment/cost limitations, severely degrading the performance of multimodal analysis algorithms.
- Diffusion models have made remarkable progress in image generation and have been applied to medical image synthesis, outperforming GANs in structural fidelity and visual quality.
- However, existing diffusion methods still suffer from three issues under arbitrary missing modalities: (1) anatomical distortion, (2) degradation of high-frequency details, and (3) entanglement of structural information and modality-specific style.
- **Key Insight**: Diffusion models are inherently insensitive to structural information in medical images. Explicitly introducing structural priors can accelerate generation and improve anatomical fidelity.

## Method

### Overall Architecture

The diffusion process is conducted in the VAE latent space. Each modality is equipped with an independent structural encoder $E_j^{str}$ (containing HFIB), a style encoder $E_j^{sty}$, and a reconstruction decoder $D_j^{rec}$, while all modalities share a segmentation decoder $D_{seg}$ to ensure the modality invariance of structural features.

### High-Frequency Injection Block (HFIB)

- At each scale of the structural encoder, features are decomposed using a learnable dynamic Gaussian filter.
- $C_{high}^l = C^l - G_{\theta}(C^l)$ extracts high-frequency residuals (edges and textures), which are then reinjected into the original features.
- $S_j^l = C^l + C_{high}^l$, enhancing structural details without altering the global anatomical layout.
- The parameters of the Gaussian filter are input-adaptive and can be dynamically adjusted based on the content.

### Multimodal Structural Feature Fusion (MMSF)

- At each scale $l$, the attention weight $w_j$ of each modality is computed through a Sigmoid gating network.
- After weighted summation, they are fused via a learnable convolution: $F_l = \text{Fusion}(\sum w_j \cdot S_j^{(l)})$.
- This emphasizes information-rich structures while suppressing irrelevant modality-specific variations.

### Multiscale Structural Feature Enhancement (MSSE)

- Fused features from lower scales ($1$ to $L-1$) are projected via $1\times 1$ convolutions and upsampled to the highest scale using bilinear interpolation.
- The highest scale representation is enhanced via cross-attention from the lower-scale structure-guided features.
- $F_s = F_L + \alpha \cdot \text{Attn}(F_L, \sum \text{Up}(\text{Proj}(F_l)))$
- The unified structural representation $F_s$ simultaneously integrates low-frequency global anatomical layouts and high-frequency detailed structures.

### Style Consistency Loss ($L_{sc}$)

- Similar to contrastive learning: style features of the same modality are pulled closer, while those of different modalities are pushed apart.
- All style features within a mini-batch are $L_2$-normalized, and their temperature-scaled dot-product similarity is computed.
- Optimized through a binary cross-entropy objective with a learnable temperature parameter.
- This encourages the style encoder to suppress modality-specific style variations.

### Structure-Aware Loss ($L_{sa}$)

- **Reconstruction part**: Reconstruct images using the decoupled structural features $F_s$ and style features $S_j$, with $L_1$ loss constraining pixel-level fidelity.
- **Frequency domain part**: Apply 2D DCT to reconstructed and ground-truth images, and compare the SSIM of their magnitude spectra.
- $L_{sa} = L_{rec} + L_{freq}$, jointly constraining the overall anatomical structure and fine-grained details.

### Total Training Objective

$L_{total} = L_{seg} + \lambda_1 \cdot L_{sc} + \lambda_2 \cdot L_{sa} + \lambda_3 \cdot L_{ldm}$, where $L_{seg}$ is the auxiliary segmentation loss and $L_{ldm}$ is the standard diffusion denoising loss.

## Key Experimental Results

### Datasets & Settings

- **BraTS2020**: 369 cases of multimodal brain MRI (T1/T2/T1CE/FLAIR) with tumor segmentation annotations.
- **WMH**: Multi-domain T1 and FLAIR images with white matter hyperintensity annotations.
- **Preprocessing**: Axial slices cropped to $192\times 192$ 2D images.
- **Training**: PyTorch 2.1.0, Adam ($lr=1e-4$), batch size = 9, $3\times$ NVIDIA 4090, 100 epochs.

### BraTS2020 Quantitative Results (Table 1, 3 available modalities $\rightarrow$ synthesizing the 4th modality)

| Method | T1 PSNR | T1 SSIM% | T1CE PSNR | T1CE SSIM% |
|------|---------|----------|-----------|------------|
| MM-GAN | 27.35 | 92.32 | 28.65 | 94.19 |
| SynDiff | 28.95 | 93.34 | 30.65 | 94.86 |
| MISA-LDM | 29.01 | 93.86 | 30.68 | 95.62 |
| **MSG-LDM** | **30.26** | **94.37** | **31.35** | **96.29** |

### WMH Dataset Results (Table 2)

| Method | FLAIR$\rightarrow$T1 PSNR | FLAIR$\rightarrow$T1 SSIM% | T1$\rightarrow$FLAIR PSNR |
|------|---------------|----------------|---------------|
| MISA-LDM | 28.86 | 95.23 | 28.10 |
| **MSG-LDM** | **29.16** | **96.80** | **28.38** |

### Ablation Study (Table 3, FLAIR reconstruction)

| Configuration | PSNR | SSIM% | Dice% |
|------|------|-------|-------|
| w/o Decoupling+MMSF | 27.92 | 92.41 | 85.03 |
| w/o HFIB | 28.17 | 92.68 | 85.41 |
| w/o MSSE | 29.04 | 93.28 | 86.55 |
| w/o $L_{sa}$ | 27.36 | 91.82 | 84.27 |
| w/o $L_{sc}$ | 27.11 | 91.54 | 83.89 |
| **Full model** | **29.68** | **93.62** | **87.60** |

- Removing $L_{sc}$ has the most significant impact (PSNR -2.57), indicating that style consistency is crucial for structural decoupling.

## Highlights & Insights

1. **Structure Priors Accelerating Diffusion**: Experiments demonstrate that diffusion models are inherently insensitive to medical image structures. Explicitly injecting structural priors not only improves quality but also accelerates the generation process.
2. **Elegant Style-Structure Decoupling**: Sharing the segmentation decoder forces structural features to be modality-invariant, and the style consistency loss further suppresses style interference.
3. **Multiscale High-Frequency Preservation**: HFIB uses learnable dynamic Gaussian filters to inject high-frequency details at each scale, which is more flexible than fixed frequency decomposition.
4. **Cross-Scale Structural Feature Enhancement**: MSSE utilizes cross-attention to allow high-level features to focus on fine-grained structural cues from lower levels.
5. **Handling Arbitrary Missing Modalities**: The framework naturally supports inputs of arbitrary modality combinations, with generation quality progressively improving as more modalities become available.

## Limitations & Future Work

1. The experiments are conducted only on 2D slices, and the 3D volume synthesis performance has not been validated.
2. The method is only validated on brain MRI and has not been extended to other anatomical regions or imaging modalities (e.g., CT, PET).
3. Each modality requires an independent encoder and decoder, resulting in a linear growth in parameter size as the number of modalities increases.
4. Training requires paired multimodal data and segmentation annotations, which places high demands on data availability.
5. Quantitative evaluation primarily relies on PSNR/SSIM/Dice, lacking perceptual quality metrics (e.g., FID) and downstream task validation.

## Rating

- **Novelty**: ⭐⭐⭐ — Each component (HFIB, MMSF, MSSE) is reasonably designed but not particularly novel when viewed individually; however, their combination yields significant improvements.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, multiple modality combinations, and detailed ablation studies, which are sufficiently comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive diagrams, and concise formulations.
- **Value**: ⭐⭐⭐⭐ — Synthesizing missing MRI modalities is a critical clinical need, making this method practical. The code is also open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](../../AAAI2026/medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[CVPR 2025\] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [\[CVPR 2026\] Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation](../../CVPR2026/medical_imaging/sketch2ct_multimodal_diffusion_for_structure-aware_3d_medical_volume_generation.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[CVPR 2025\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)

</div>

<!-- RELATED:END -->
