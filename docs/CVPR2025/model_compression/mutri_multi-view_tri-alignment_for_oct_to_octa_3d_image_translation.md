---
title: >-
  [Paper Note] MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation
description: >-
  [CVPR 2025][Model Compression][OCT] This paper proposes MuTri, which introduces vector quantization (VQ) to the 3D volumetric translation task from OCT to OCTA for the first time. Through a two-stage training process—first pre-training VQ-VAEs for OCT and OCTA reconstruction to provide multi-view priors, and then utilizing multi-view guidance, namely contrastive semantic alignment (3D OCT/OCTA views) and vessel structure alignment (2D OCTA projection view)…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "OCT"
  - "OCTA"
  - "Vector Quantization"
  - "Contrastive Learning"
  - "3D Image Translation"
  - "Retinal Vasculature"
date: 2026-05-08
content_hash: c2d71dfff9c966e9
---

# MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation

**Conference**: CVPR 2025  
**arXiv**: [2504.01428](https://arxiv.org/abs/2504.01428)  
**Code**: [https://github.com/xmed-lab/MuTri](https://github.com/xmed-lab/MuTri)  
**Area**: Model Compression  
**Keywords**: OCT, OCTA, Vector Quantization, Contrastive Learning, 3D Image Translation, Retinal Vasculature

## TL;DR

This paper proposes MuTri, which introduces vector quantization (VQ) to the 3D volumetric translation task from OCT to OCTA for the first time. Through a two-stage training process—first pre-training VQ-VAEs for OCT and OCTA reconstruction to provide multi-view priors, and then utilizing multi-view guidance, namely contrastive semantic alignment (3D OCT/OCTA views) and vessel structure alignment (2D OCTA projection view), to guide the codebook learning of the translation VQ-VAE—the proposed method comprehensively outperforms state-of-the-art (SOTA) methods across three datasets.

## Background & Motivation

**Background**: Optical coherence tomography angiography (OCTA) can provide 3D imaging of retinal microvasculature, which is crucial for the diagnosis of diseases such as diabetic retinopathy and glaucoma. However, OCTA requires specialized sensors and expensive equipment, making it far less prevalent than OCT.

**Limitations of Prior Work**: (1) Early methods (AdjacentGAN, MultiGAN) only process 2D B-scans or projection maps, losing 3D depth information; (2) The latest 3D method TransPro directly learns the OCT $\to$ OCTA mapping in a continuous infinite space, leading to high mapping uncertainty; (3) TransPro relies on a pre-trained vessel segmentation model, whereas high-quality vessel segmentation annotations are extremely costly to obtain; (4) Naive VQ-VAE suffers from low codebook utilization and poor translation quality due to the vast domain gap between OCT and OCTA.

**Key Challenge**: There is a massive domain gap between OCT (structural information) and OCTA (blood flow information)—when directly using VQ-VAE to learn the mapping, the unquantized features output by the encoder are difficult to effectively quantize to the OCTA codebook, leading to codebook collapse and leaving a large number of codewords idle.

**Goal**: (1) Learn the OCT $\to$ OCTA mapping in a discrete finite space (rather than a continuous infinite space) to reduce uncertainty; (2) Solve the problem of low codebook utilization in VQ-VAE during cross-domain translation; (3) Avoid reliance on extra vessel segmentation annotations.

**Key Insight**: Leveraging pre-trained OCT and OCTA reconstruction models to provide high-quality domain features as multi-view guidance signals, and maximizing the mutual information between the translation model and the pre-trained models via contrastive learning to drive the codebook to explore more codewords.

**Core Idea**: Pre-train OCT and OCTA reconstruction VQ-VAEs to provide tri-view priors (3D OCT + 3D OCTA + 2D OCTA projection map), and utilize contrastive semantic alignment and vessel structure alignment to guide the codebook learning of the translation VQ-VAE from both semantic and structural levels, respectively.

## Method

### Overall Architecture

**Stage 1**: Train VQ-VAE reconstruction models for OCT and OCTA separately. The OCT VQ-VAE encodes, quantizes, and decodes the OCT volume to reconstruct it, and the OCTA VQ-VAE does the same. They possess codebooks $\mathcal{Z}_{oct}$ and $\mathcal{Z}_{octa}$, respectively.

**Stage 2**: Train the translation VQ-VAE, which takes the OCT volume as input and outputs the OCTA volume. Three alignment losses leverage the features of the Stage 1 pre-trained models as guidance: (a) contrastive semantic alignment aligns unquantized features from the 3D OCT view; (b) contrastive semantic alignment aligns quantized features from the 3D OCTA view; (c) vessel structure alignment aligns the vessel structure from the 2D OCTA projection map view.

### Key Designs

1. **Contrastive-inspired Semantic Alignment (CSA)**:
    - **Function**: Maximize the mutual information between the translation model and the pre-trained OCT/OCTA models to drive codebook exploration.
    - Key Operation: Segment the feature maps of the translation model and pre-trained models into non-overlapping patches, and perform contrastive learning after projection—patches at the same spatial position serve as positive samples, while patches at different positions serve as negative samples.
    - OCT view contrast: $\mathcal{L}_{OCT}$ aligns the unquantized features of the translation model with the unquantized features of the pre-trained OCT model.
    - OCTA view contrast: $\mathcal{L}_{OCTA}$ aligns the quantized features of the translation model with the quantized features of the pre-trained OCTA model.
    - Theoretical Guarantee: Minimizing $\mathcal{L}_{OCT}$ is equivalent to maximizing the lower bound of the mutual information $I(\mathbf{P}, \mathbf{Q}) \geq \log(\frac{W}{S} \cdot \frac{H}{S} - 1) - \mathbb{E}\mathcal{L}_{OCT}$.

2. **Vessel Structure Alignment (VSA)**:
    - **Function**: Utilize the vessel structure prior of the 2D OCTA projection map to supplement the fine details missing in the 3D alignment.
    - Key Operation: Crop the 2D OCTA projection maps from the pre-trained OCTA model and the translation model into patches, calculate the cosine similarity matrices between patches $\mathcal{C}^{octa}$ and $\mathcal{C}^{oct2octa}$, and minimize the difference: $\mathcal{L}_{proj} = \sum \|\mathcal{C}^{octa} - \mathcal{C}^{oct2octa}\|$.
    - **Design Motivation**: Real OCTA projection maps suffer from vessel discontinuity artifacts caused by scanning instability, whereas the reconstructed projection maps from the pre-trained OCTA model are smoother and more continuous. Using the latter as guidance prevents the translation model from overfitting to these artifacts.
    - Compared with the vessel segmentation auxiliary task in TransPro, VSA does not require any additional annotations.

3. **Two-Stage VQ-VAE Training Strategy**:
    - **Function**: First learn the discrete representation space of each domain, and then utilize them to guide cross-domain translation.
    - Stage 1 Loss: Standard VQ-VAE loss = reconstruction loss + codebook loss + commitment loss.
    - Codebook Utilization Boost: Naive VQ-VAE achieves only ~30% utilization, whereas MuTri's CSA improves the utilization to ~80%+.

### Loss & Training

$$\mathcal{L}_{stage2} = \mathcal{L}_{VQVAE}^{oct2octa}(\mathcal{Z}, E_{oct2octa}, D_{oct2octa}) + \lambda(\mathcal{L}_{OCT} + \mathcal{L}_{OCTA} + \mathcal{L}_{proj})$$

$\lambda=0.5$, $\tau=0.1$ (contrastive temperature), both determined through sensitivity analysis; the model is insensitive to both.

## Key Experimental Results

### Main Results: Three Datasets (Tables 1 & 2)

| Method | OCTA-3M PSNR(dB)↑ | OCTA-6M PSNR(dB)↑ | OCTA2024 PSNR(dB)↑ |
|------|-------------------|-------------------|-------------------|
| Pix2Pix3D | 31.58 | 30.66 | 41.87 |
| VQ-I2I | 31.72 | 29.54 | 41.25 |
| Palette (Diffusion) | 32.42 | 30.02 | 41.40 |
| TransPro | 32.56 | 30.53 | 42.69 |
| **MuTri** | **34.10** | **33.08** | **43.38** |

### Ablation Study (Table 3, OCTA-6M)

| CSA | VSA | MAE↓ | PSNR(dB)↑ | SSIM(%)↑ |
|-----|-----|------|-----------|---------|
| ✗ | ✓ | 0.0765 | 32.31 | 89.53 |
| ✓ | ✗ | 0.0748 | 32.87 | 89.19 |
| ✓ | ✓ | **0.0741** | **33.08** | **90.04** |

### OCTA2024 Projection Metrics

| Method | MAE↓ | PSNR(dB)↑ | SSIM(%)↑ |
|------|------|-----------|---------|
| TransPro | 0.01056 | 37.85 | 87.61 |
| **MuTri** | **0.00870** | **39.65** | **90.31** |

### Key Findings

- Achieves comprehensive SOTA performance across all metrics on all three datasets.
- Improves by **1.54 dB PSNR** on OCTA-3M and **2.55 dB** on OCTA-6M compared to TransPro.
- CSA is the most critical component, individually contributing 0.56 dB; VSA contributes an additional 0.21 dB on top of it.
- The improvement is more significant on the large-scale OCTA2024 dataset (as more data enables training better pre-trained models).
- Ophthalmologist diagnostic analysis confirms that the OCTA translated by MuTri is closer to the pathological patterns of ground-truth OCTA.

## Highlights & Insights

1. **Discrete vs. Continuous Space**: Shifting cross-modal translation from continuous space to discrete codebook space, and using the finiteness of the codebook to constrain mapping uncertainty, represents an important methodological shift for medical image translation.
2. **Pre-trained Model as Teacher**: Instead of direct distillation, contrastive learning is used to maximize the mutual information between the features of the translation model and the pre-trained models, thereby driving the codebook to explore new codewords.
3. **Clever Use of Projection Maps**: The 2D projection map is generated by averaging the 3D volume along the depth direction, which naturally highlights vessel structures. Performing structural alignment using a patch-level similarity matrix is more robust than pixel-wise losses.
4. **First Large-Scale Dataset OCTA2024**: Consisting of 846 OCT-OCTA volume pairs, accelerating the progress of the field.

## Limitations & Future Work

1. The codebook size and patch size of the VQ-VAE, acting as hyperparameters, require manual tuning.
2. The two-stage training increases overall training time and complexity.
3. The encoder/decoder is stacked only pathwise with ResBlocks, leaving more powerful backbones (such as Swin Transformer) unexplored.
4. Validated only on retinal OCT/OCTA; generalization to other 3D medical translation tasks (such as CT $\to$ MRI) requires further validation.

## Related Work & Insights

- **TransPro** [Sun et al., MIA]: Direct predecessor of 3D OCT $\to$ OCTA, which utilizes a pre-trained vessel segmentation model for single-view guidance. MuTri replaces it with tri-view guidance and does not rely on segmentation annotations.
- **VQ-I2I** [ECCV]: Introduces VQ to 2D image-to-image translation. MuTri extends VQ to 3D medical translation and resolves codebook collapse.
- **CUT** [Park et al.]: A pioneer in patch-wise contrastive learning for unpaired translation. MuTri draws inspiration from it but applies it to feature alignment across different tasks (reconstruction vs. translation).
- **Multimodal Pre-training Paradigm of SynthSeg**: The strategy of pre-training models for each domain to provide priors can be generalized to other cross-modal tasks.

## Rating

- ⭐ Novelty: 8/10 — The VQ + tri-view alignment framework is novel, and driving codebook exploration via contrastive learning is theoretically supported.
- ⭐ Experimental Thoroughness: 8/10 — Comprehensive with three datasets + ablation + sensitivity + clinical diagnostic analysis.
- ⭐ Value: 7/10 — Addresses actual clinical needs (low-cost OCT equipment upgrading), but the two-stage training is somewhat heavy.
- ⭐ Overall: 8/10 — A solid work in cross-modal medical image translation, with highly generalizable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-View Distillation and Adaptive Masking for Incomplete Multi-View Multi-Label Classification](../../CVPR2026/model_compression/cross-view_distillation_and_adaptive_masking_for_incomplete_multi-view_multi-lab.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](../../CVPR2026/model_compression/parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)
- [\[ACL 2025\] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](../../ACL2025/model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)
- [\[CVPR 2025\] InsTaG: Learning Personalized 3D Talking Head from Few-Second Video](instag_learning_personalized_3d_talking_head_from_few-second_video.md)
- [\[CVPR 2025\] IterIS: Iterative Inference-Solving Alignment for LoRA Merging](iteris_iterative_inference-solving_alignment_for_lora_merging.md)

</div>

<!-- RELATED:END -->
