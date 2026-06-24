---
title: >-
  [Paper Note] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation
description: >-
  [CVPR 2025][Image Generation][Multi-scale generation] ZoomLDM proposes a scale-conditioned latent diffusion model that constructs a cross-magnification latent space through a trainable Summarizer module, achieving high-quality pathology image generation across multiple scales. It represents the first work to support globally consistent large image synthesis up to $4096 \times 4096$ pixels as well as training-free super-resolution.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-scale generation"
  - "Latent Diffusion Models"
  - "Pathology images"
  - "Self-supervised learning"
  - "Large image synthesis"
date: 2026-05-08
content_hash: d5994590097391d1
---

# ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.16969](https://arxiv.org/abs/2411.16969)  
**Code**: [https://github.com/cvlab-stonybrook/ZoomLDM](https://github.com/cvlab-stonybrook/ZoomLDM)  
**Area**: Medical Images/Diffusion Models  
**Keywords**: Multi-scale generation, Latent Diffusion Models, Pathology images, Self-supervised learning, Large image synthesis

## TL;DR

ZoomLDM proposes a scale-conditioned latent diffusion model that constructs a cross-magnification latent space through a trainable Summarizer module, achieving high-quality pathology image generation across multiple scales. It represents the first work to support globally consistent large image synthesis up to $4096 \times 4096$ pixels as well as training-free super-resolution.

## Background & Motivation

**Background**: Diffusion models have achieved immense success in natural image generation, but their application to large-scale images (such as digital pathology and satellite imagery, opening up to gigapixel levels) remains restricted. Existing methods mostly train diffusion models on fixed-size patches, failing to capture the global structure of large images.

**Limitations of Prior Work**: (1) Direct training on full-resolution gigapixel images is computationally infeasible; (2) Single-scale patch models can only generate local details and lack global context—causing quality to degrade drastically when low-magnification (e.g., $0.15625\times$) data is extremely scarce; (3) The large-image domain lacks paired image-text annotations, precluding the use of text conditioning as in Stable Diffusion; (4) Existing large-image generation methods either produce blurry results ($\infty$-Brush) or lack global consistency (Graikos et al.).

**Key Challenge**: The global structure of large images requires understanding low-magnification information, while local details require high-magnification rendering. However, a single model cannot simultaneously cover a 128-fold magnification range from $20\times$ to $0.15625\times$, and low-magnification data is severely scarce.

**Goal**: To train a unified multi-scale diffusion model capable of generating high-quality patches at all magnifications and leveraging multi-scale joint sampling for globally consistent large image synthesis.

**Key Insight**: Although pathology patches from different scales share the same resolution ($256 \times 256$), each pixel encodes a distinct "zoom level". If a scale-aware model can be trained to share weights across scales, data-rich magnifications can assist data-scarce ones, addressing the data shortage. Self-supervised learning (SSL) encoders (such as UNI) can substitute missing text annotations to provide conditional signals.

**Core Idea**: Train a scale-conditioned diffusion model with shared weights, mapping SSL embeddings of various scales into a unified cross-magnification latent space using a trainable Summarizer, thereby enabling multi-scale collaborative training and joint sampling.

## Method

### Overall Architecture

The training pipeline of ZoomLDM consists of: (1) extracting $256 \times 256$ patches from the initial scale ($20\times$) of a large image and utilizing a UNI encoder to extract SSL embeddings; (2) gradually downsampling the large image by factors of 2, where patches at each scale are paired with the corresponding SSL embedding matrix from the initial scale region; (3) using a Summarizer transformer to compress the variable-length SSL embedding matrix (augmented with magnification embeddings) into a fixed-size conditioning vector; (4) performing unified multi-scale training on the LDM (VQ-f4 autoencoder + U-Net denoiser) conditioned on the Summarizer's output. During inference, a Conditional Diffusion Model (CDM) is used to directly sample conditioning vectors in the latent space, eliminating the need for real images.

### Key Designs

1. **Cross-Magnification Latent Space and Summarizer Module**:

    - **Function**: Maps SSL embeddings from different scales to a shared conditioning space, enabling collaborative multi-scale training.
    - **Mechanism**: Pathology patches at different magnifications correspond to varying numbers of initial-scale SSL embeddings (e.g., a $5\times$ patch corresponds to $4 \times 4 = 16$ embeddings at $20\times$, and a $1.25\times$ patch corresponds to $16 \times 16 = 256$ embeddings). The Summarizer is a 12-layer ViT-Base transformer that ingested variable-length sequences of embeddings summed with magnification positional embeddings. It produces fixed-size conditioning tokens through padding and pooling. The magnification embeddings allow the Summarizer to perceive the current scale and extract the corresponding scale-specific details. To manage computational load, embedding matrices larger than $8 \times 8$ are pre-pooled to $8 \times 8$.
    - **Design Motivation**: SSL encoders (like UNI) are pre-trained only at the initial scale ($20\times$) and cannot directly extract meaningful features for images at other scales. By passing the initial-scale embeddings as a "content description" to all scales, the Summarizer is tasked with extracting information corresponding to the current magnification. This is significantly more efficient than training separate SSL encoders for each individual scale.

2. **Conditional Diffusion Model (CDM)**:

    - **Function**: Samples conditioning vectors in the absence of real reference images, enabling fully unconditional novel image generation.
    - **Mechanism**: After ZoomLDM training is complete, a Diffusion Transformer (DiT) is trained to model the distribution of the Summarizer's output space. Conditioned on magnification, the CDM can directly sample new conditioning vectors to drive the LDM generation. This is far simpler than modeling the raw SSL embedding distribution because the Summarizer output is a compressed, task-centric representation.
    - **Design Motivation**: Real-world scenarios do not always provide access to raw SSL embeddings of real images (e.g., when generating fully synthetic images). The CDM decouples generation from the dependency on real reference data. Crucially, at extremely low magnifications where data is scarce (e.g., only 2,500 samples at $0.15625\times$), the CDM can still generate non-memorized, novel images.

3. **Joint Multi-Scale Sampling**:

    - **Function**: Generates globally consistent large images across multiple scales.
    - **Mechanism**: Leveraging the linear downsampling relationship between different image scales, i.e., $\mathbf{x}^{s+1} = \mathbf{A}(\mathbf{x}_1^s, \mathbf{x}_2^s, \mathbf{x}_3^s, \mathbf{x}_4^s)$, the "clean image" at each scale is first estimated during each denoising step of the diffusion process. The higher-resolution (lower-magnification context) scale acts as a self-guidance signal for the lower-resolution scale. The gradient direction is efficiently approximated using finite differences to update the noisy latents of lower scales so that they maintain consistency with the higher scales. Compared to the original algorithm, this avoids expensive backpropagation through numerical approximation.
    - **Design Motivation**: Sampling each scale independently leads to global inconsistencies (e.g., a higher scale shows a forest, while a lower scale generates details of a desert). Joint sampling ensures semantic consistency through "higher-scale guidance of lower-scales", where the higher scale provides global context (e.g., tissue types) and the lower scale ensures consistent local details (e.g., cell morphology).

### Loss & Training

The LDM utilizes standard latent diffusion training objectives. The Summarizer and LDM are trained jointly end-to-end. The CDM is trained using a DiT architecture. DDIM sampling with 50 steps is employed, with a classifier-free guidance scale of 2.0. The VQ-f4 autoencoder and U-Net are initialized from ImageNet pre-trained checkpoints.

## Key Experimental Results

### Main Results (FID Across Magnifications)

| Magnification | Training Samples | ZoomLDM | Single-Scale SoTA | CDM |
|------|----------|---------|-----------|-----|
| 20× | 12M | **6.77** | 6.98 | 9.04 |
| 10× | 3M | **7.60** | 7.64 | 10.05 |
| 5× | 750K | **7.98** | 9.74 | 14.36 |
| 2.5× | 186K | **10.73** | 20.45 | 19.68 |
| 1.25× | 57K | **8.74** | 39.72 | 14.06 |
| 0.625× | 20K | **7.99** | 58.98 | 13.46 |
| 0.3125× | 7K | **8.34** | 66.28 | 14.40 |
| 0.15625× | 2.5K | **13.42** | 106.14 | 26.09 |

### Ablation Study (Large Image Generation $1024 \times 1024$)

| Method | CLIP FID↓ | Crop FID↓ | Inference Time |
|------|----------|----------|---------|
| Graikos et al. | 7.43 | 15.51 | 60s |
| ∞-Brush | 3.74 | 17.87 | 30s |
| **ZoomLDM** | **1.23** | **14.94** | 28s |

### Key Findings

- **Astonishing Effect of Cross-Scale Weight Sharing**: At the $0.15625\times$ magnification (which only has 2,500 training samples), the single-scale model's FID is as high as 106.14, whereas ZoomLDM reduces it to 13.42 (a 7.9-fold improvement). Multi-scale collaborative training allows data-rich high magnifications to assist data-scarce lower magnifications.
- Globally consistent synthesis of $4096 \times 4096$ pathology images was achieved for the first time, maintaining an inference time of only 8 minutes (vs. 12 hours for $\infty$-Brush).
- Super-resolution ($4\times$) outperforms CompVis and ControlNet baselines in SSIM, PSNR, and LPIPS, requiring no additional training.
- ZoomLDM's intermediate features surpass SoTA SSL encoders in Multiple Instance Learning (MIL) experiments, indicating that representation learning driven by multi-scale generation yields more expressive features than discriminative encoders.

## Highlights & Insights

- The discovery of using **multi-scale generation for representation learning** is highly valuable: ZoomLDM outperforms UNI (the state-of-the-art pathology SSL encoder at $20\times$) using only $20\times$ features, demonstrating that learning to generate multi-scale images forces the model to acquire richer internal multi-scale representations.
- The design of **Summarizer's cross-magnification latent space** elegantly solves the single-scale limitation of SSL encoders: instead of training a new encoder for each scale, it learns a mapping layer to extract scale-adapted information from pre-existing embeddings.
- The "self-guidance" approach of joint sampling (using the model's own higher-scale predictions to guide lower-scale generation) requires no external information, representing an effective test-time trade-off between computational cost and generation quality.

## Limitations & Future Work

- Joint sampling at $4096 \times 4096$ requires managing $16 \times 16 + 1 = 257$ denoising processes simultaneously, which still incurs a heavy computational footprint (8 minutes per image).
- The CLIP FID and Crop FID at $4096 \times 4096$ are relatively inferior to $\infty$-Brush, indicating that global consistency comes at the cost of sacrificing some local detail diversity.
- Detailed evaluation is currently limited to the TCGA-BRCA pathology dataset, with satellite image results available only in the supplementary materials.
- Future work could explore more efficient joint sampling algorithms and extend the method to 3-D medical images (such as multi-resolution CT generation).

## Related Work & Insights

- **vs. Graikos et al.**: This method only trains single-scale patch models and relies on transitions in the SSL embedding space to generate large images, which lacks a native grasp of global structure. ZoomLDM achieves intrinsic global consistency through joint multi-scale training and sampling.
- **vs. $\infty$-Brush**: While using infinite-latitude diffusion models theoretically allows handling arbitrary resolutions, practical results are blurry, and inference is extremely slow (12 hours per 4K image). ZoomLDM is 90x faster and delivers superior local details.
- **vs. Harb et al.**: This work also trains a multi-scale pathology diffusion model but lacks a conditioning mechanism, severely limiting generation quality and controllability. The Summarizer conditioning design in ZoomLDM stands out as a key innovation.

## Rating

- Novelty: ⭐⭐⭐⭐ The concepts of cross-magnification latent space and joint sampling are novel, and multi-scale generation in pathology has pioneer significance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering FID across 8 magnifications, large image generation, super-resolution, and MIL downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Highly clear methodological explanations with sound mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ This is the first practical multi-scale generation approach for gigapixel pathology images, presenting substantial value to the medical AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[CVPR 2025\] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)
- [\[CVPR 2025\] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model](sir-diff_sparse_image_sets_restoration_with_multi-view_diffusion_model.md)
- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)

</div>

<!-- RELATED:END -->
