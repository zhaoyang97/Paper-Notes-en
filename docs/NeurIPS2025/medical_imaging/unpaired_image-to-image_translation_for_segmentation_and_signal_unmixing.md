---
title: >-
  [Paper Note] Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing
description: >-
  [NeurIPS 2025][Medical Imaging][Unpaired image-to-image translation] This paper proposes Ui2i, a model built upon CycleGAN that achieves high content-fidelity unpaired image-to-image translation through four key innovations: a UNet-based generator, approximate bidirectional spectral normalization (ABSN) as a replacement for feature normalization, channel-spatial attention, and scale augmentation. The model is successfully applied to two biomedical tasks: IHC→H&E domain adapta…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Unpaired image-to-image translation"
  - "nucleus segmentation"
  - "immunofluorescence unmixing"
  - "CycleGAN"
  - "spectral normalization"
date: 2026-05-08
content_hash: 0b96fa7499f69499
---

# Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing

**Conference**: NeurIPS 2025
**arXiv**: [2505.20746](https://arxiv.org/abs/2505.20746)  
**Code**: Not available  
**Area**: Medical Imaging
**Keywords**: Unpaired image-to-image translation, nucleus segmentation, immunofluorescence unmixing, CycleGAN, spectral normalization

## TL;DR

This paper proposes Ui2i, a model built upon CycleGAN that achieves high content-fidelity unpaired image-to-image translation through four key innovations: a UNet-based generator, approximate bidirectional spectral normalization (ABSN) as a replacement for feature normalization, channel-spatial attention, and scale augmentation. The model is successfully applied to two biomedical tasks: IHC→H&E domain adaptation for nucleus segmentation and single-channel immunofluorescence signal unmixing.

## Background & Motivation

- **Background**: In digital pathology and spatial proteomics, significant domain gaps exist across different staining protocols (H&E, IHC, IF), preventing segmentation models (e.g., StarDist) trained on one domain from generalizing directly to others. Acquiring annotations in new domains is prohibitively expensive, motivating the use of **image-to-image translation** for cross-domain adaptation.

- **Limitations of Prior Work**: Biomedical images impose strict requirements on content fidelity — nuclear shape, boundaries, cellular morphology, and spatial relationships must be precisely preserved, unlike natural scene translation where moderate distortion is acceptable. Although CycleGAN achieves unpaired translation via cycle-consistency loss, it suffers from two fundamental issues:

    - **Non-uniqueness of cycle consistency**: The authors theoretically demonstrate that for a perfectly disentangled generator pair $(\hat{G}_{AB}, \hat{G}_{BA})$, applying any invertible transforms $T_A, T_B$ still satisfies cycle consistency, meaning infinitely many generator pairs minimize $\mathcal{L}_{\text{cyc}}$ — content preservation is thus not uniquely guaranteed.

    - **Artifacts from feature normalization**: Instance Normalization causes the same local object to produce substantially different signal responses depending on global context, leading to "droplet-like" artifacts in translated images that are misidentified as false-positive nuclei.

- **Goal**: Eliminate feature normalization layers entirely, replacing them with **parameter-level approximate bidirectional spectral normalization**, combined with UNet skip connections and attention mechanisms to better preserve spatially local content features.

## Method

### Overall Architecture

Ui2i follows the CycleGAN framework, comprising two generators $G_{AB}$ and $G_{BA}$, a domain discriminator, and a content discriminator. Key improvements include: a UNet-structured generator, bidirectional spectral normalization, channel-spatial attention, and scale augmentation.

### Key Designs

1. **Approximate Bidirectional Spectral Normalization (ABSN)**: All feature normalization (BN, IN, etc.) is removed. Each weight tensor $\mathbf{w}$ is instead normalized using bidirectional spectral normalization. The weight matrix is reshaped in both the forward direction $W^{FW}$ (forward information flow) and the backward direction $W^{BW}$ (gradient propagation). A differentiable lower bound estimates the spectral norm:

$$\|W\| \geq \sigma(W) = \frac{\left\|\left(\sum_{j=1}^n w_j w_j^\top\right)r\right\|}{\|(w_1^\top r, \ldots, w_n^\top r)^\top\|}$$

The forward and backward spectral norms are then aggregated via RMS: $\sigma_{\text{rms}} = \sqrt{(\sigma^2(W^{FW}) + \sigma^2(W^{BW}))/2}$. This eliminates context-dependent response artifacts caused by feature normalization while maintaining training stability.

2. **UNet Generator with Skip Connections**: Replaces CycleGAN's ResNet generator. Skip connections propagate shallow, spatially local encoder features to deep decoder layers, preserving fine-grained structural information. The two generators share weights at the bottleneck. The decoder uses $4\times4$ Lanczos2 upsampling kernels to avoid checkerboard artifacts.

3. **Channel-Spatial Attention Module**: An ESCA (Efficient Symmetric Spatial and Channel Attention) module combined with spatial attention is integrated in series within encoder blocks to refine feature maps. Residual connections enhance content preservation. Attention modules are omitted in the decoder for efficiency.

4. **Stacked Domain Discriminator + Content Discriminator**: The domain discriminator adopts a single PatchGAN structure that concatenates images from both domains along the channel dimension and classifies them as real/fake/identity (three-class), reducing overfitting. The content discriminator classifies bottleneck features by domain (A or B), encouraging domain-invariant content representations.

### Loss & Training

The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{adv}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}} + \lambda_{\text{id}}\mathcal{L}_{\text{id}} + \lambda_{\text{cl}}\mathcal{L}_{\text{cl}}$$

- $\mathcal{L}_{\text{adv}}$: Adversarial loss supervising both the domain discriminator and content discriminator
- $\mathcal{L}_{\text{cyc}}$: Cycle-consistency loss, $\lambda_{\text{cyc}}=10$
- $\mathcal{L}_{\text{id}}$: Identity mapping loss, $\lambda_{\text{id}}=1$
- $\mathcal{L}_{\text{cl}}$: N-pair contrastive loss pulling together bottleneck features of original–translated image pairs, $\lambda_{\text{cl}}=0.1$

Training uses differentiable data augmentation with random scale factors in $[0.75, 1.5]$ to encourage scale-invariant feature learning. Adam optimizer with lr=0.0002, 50K iterations; a buffer of the 50 most recently generated images is maintained to stabilize discriminator training.

## Key Experimental Results

### Main Results: IHC→H&E Nucleus Segmentation

| Method | Instance Precision | Instance Recall | Segm. Quality | Panoptic Quality |
|------|-------------------|----------------|---------------|-----------------|
| No translation (StarDist) | 0.92±0.13 | 0.51±0.21 | 0.78±0.08 | 0.50±0.17 |
| InstanSeg (IHC pretrained) | 0.76±0.16 | 0.70±0.17 | 0.75±0.08 | 0.55±0.12 |
| CycleGAN | 0.72±0.18 | 0.76±0.16 | 0.80±0.05 | 0.59±0.14 |
| **full Ui2i** | **0.87±0.11** | **0.77±0.14** | 0.80±0.05 | **0.65±0.10** |

### Ablation Study

| Configuration | Instance Precision | Instance Recall | Panoptic Quality | Note |
|------|-------------------|----------------|-----------------|------|
| Ui2i w/o augment. | 0.83±0.14 | 0.72±0.14 | 0.60±0.11 | Scale augmentation removed |
| Ui2i w/ feature norm. | 0.75±0.18 | 0.74±0.16 | 0.60±0.14 | Feature normalization restored |
| Ui2i w/o attention | 0.83±0.14 | 0.73±0.15 | 0.63±0.12 | Attention removed |
| full Ui2i | **0.87±0.11** | **0.77±0.14** | **0.65±0.10** | Full model |

### IF Signal Unmixing Quantitative Results

| Metric | SOX2 Channel | Grasp65 Channel |
|------|----------|-------------|
| MicroMS-SSIM | 0.96±0.03 | 0.96±0.02 |
| PSNR | 38±3 | 32±2 |

Compared to MicroSplit (which requires paired data) — MicroMS-SSIM 0.978/0.951, PSNR 40.3/32.8 — Ui2i approaches paired-method performance using only unpaired data.

### Key Findings

- Ui2i significantly outperforms CycleGAN on all StarDist segmentation metrics, and even surpasses InstanSeg pretrained on the IHC domain.
- Feature normalization is the primary source of artifacts; replacing it with spectral normalization improves instance precision from 0.75 to 0.87.
- This work represents the first demonstration of IF signal unmixing learned from truly unpaired data, addressing the practical need of labeling two biomarkers with a single fluorophore.

## Highlights & Insights

- The paper provides a theoretical analysis of the non-uniqueness of cycle-consistency loss, explaining why $\mathcal{L}_{\text{cyc}}$ alone is insufficient to guarantee content preservation.
- Addressing context-dependent artifacts through parameter-level normalization rather than architectural modifications is a novel and elegant approach.
- The IF signal unmixing application is of high practical value — it effectively doubles the labeling capacity of mIF experiments.

## Limitations & Future Work

- Evaluation is limited to two specific tasks (nucleus segmentation and IF unmixing); generalizability requires further validation.
- The HT-T24 dataset used for quantitative evaluation of IF unmixing does not fully simulate a real single-fluorophore multiplexing scenario.
- No comparison is made against recent diffusion model-based image-to-image translation methods.

## Related Work & Insights

- Compared to MicroSplit, Ui2i does not require paired data, making it better suited for practical deployment.
- The ABSN design principle can be extended to other medical image generation tasks requiring high content fidelity.

## Rating

- Novelty: ⭐⭐⭐⭐ Theoretical analysis and ABSN design are original; IF unmixing is a novel application
- Experimental Thoroughness: ⭐⭐⭐⭐ Two application scenarios with full ablations, though more baselines would strengthen the study
- Writing Quality: ⭐⭐⭐⭐ Logic is clear with tight integration of theory, method, and experiments
- Value: ⭐⭐⭐⭐ High practical value for IF unmixing; domain adaptation method is broadly applicable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)
- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](../../ECCV2024/medical_imaging/unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [\[NeurIPS 2025\] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)

</div>

<!-- RELATED:END -->
