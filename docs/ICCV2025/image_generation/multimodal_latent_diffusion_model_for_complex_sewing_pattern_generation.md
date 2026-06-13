---
title: >-
  [Paper Note] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation
description: >-
  [ICCV 2025][Image Generation][Sewing pattern generation] This paper proposes SewingLDM, a multimodal conditional latent diffusion model that generates complex sewing patterns under text, sketch…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Sewing pattern generation"
  - "latent diffusion model"
  - "multimodal conditioning"
  - "body-shape awareness"
  - "CG pipeline"
date: 2026-05-08
content_hash: 427ac831e751145b
---

# Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation

**Conference**: ICCV 2025
**arXiv**: [2412.14453](https://arxiv.org/abs/2412.14453)  
**Code**: [Project Page](https://shengqiliu1.github.io/SewingLDM)  
**Area**: Diffusion Models / Garment Generation
**Keywords**: Sewing pattern generation, latent diffusion model, multimodal conditioning, body-shape awareness, CG pipeline

## TL;DR

This paper proposes SewingLDM, a multimodal conditional latent diffusion model that generates complex sewing patterns under text, sketch, and body-shape conditions via an extended sewing pattern representation and a two-stage training strategy, with seamless integration into CG simulation pipelines.

## Background & Motivation

Sewing patterns are a widely adopted garment representation in industry, offering natural advantages in CG pipelines due to their compatibility with physical simulation and animation. However, existing sewing pattern generation methods face the following challenges:

**Insufficient representation of complex patterns**: Prior methods (e.g., NeuralTailor) support only straight lines and quadratic curves, failing to represent cubic curves, circular arcs, and other complex geometries common in modern garment design, and lack modeling of attachment constraints (e.g., anti-slip constraints for necklines and waistbands).

**Lack of fine-grained control**: Methods such as DressCode support text-driven generation but exhibit high failure rates on complex garment descriptions (e.g., off-shoulder gowns, square-neck shirts). Parametric approaches (e.g., GarmentCodeData) support rich control but require predefined templates and domain expertise.

**Neglect of body-shape adaptation**: Most methods are trained on standard body shapes and cannot generate garments tailored to diverse body types, leading to issues such as cloth penetration and slippage when patterns are dressed onto different bodies.

**Incompatibility of 3D mesh methods with CG pipelines**: 3D mesh generation methods such as Wonder3D and RichDreamer produce visually appealing garments, but their closed-surface meshes cannot be integrated into modern CG production workflows and exhibit severe penetration artifacts when worn.

## Method

### Overall Architecture

SewingLDM consists of three core components:
1. **Extended pattern representation**: Each edge's feature vector is expanded from the original low-dimensional form to 29 dimensions, covering four edge types, attachment constraints, stitch reversal flags, and more.
2. **Compact latent space compression**: An autoencoder compresses the high-dimensional pattern representation into a bounded, compact latent space.
3. **Multimodal conditional diffusion model**: Built on a DiT architecture, text, sketch, and body-shape conditions are injected via a two-stage training strategy.

### Key Designs

1. **Extended sewing pattern representation**: The original representation is augmented with: cubic curve control points $C^b_{i,j} \in \mathbb{R}^4$, circular arc parameters $C^r_{i,j} \in \mathbb{R}^3$, edge-type flags $E^t_{i,j,k}$ (2-bit encoding for 4 edge types), attachment-type flags $A_{i,j,k}$ (3-bit encoding for neckline, waistband, and other constraints), and stitch-direction reversal flags (to prevent stitch crossing during simulation). Each edge is ultimately represented as a 29-dimensional vector:

    $E^f_{i,j} = V_{i,j} \oplus C_{i,j} \oplus C^b_{i,j} \oplus C^r_{i,j} \oplus S_{i,j} \oplus R_i \oplus T_i \oplus E^t_{i,j} \oplus E^m_{i,j} \oplus A_{i,j} \oplus M'_{i,j}$

   All patterns are zero-padded to a fixed size of $(max(N_p) \times max(N_i), 29)$.

2. **Compact latent space compression**: An autoencoder is trained to compress pattern $F$ into a bounded latent space $[-1, 1]$. After encoding, quantization $\hat{z} = \frac{round(n \times tanh(z))}{n}$ maps each dimension uniformly onto $\{-1, -0.5, 0, 0.5, 1\}$ (with $n=2$), facilitating distribution learning by the diffusion model. The training loss combines reconstruction MSE loss $\mathcal{L}_{rec}$, panel integrity loss $\mathcal{L}_{panel}$, stitch accuracy loss $\mathcal{L}_{stitch}$, and a newly introduced binary cross-entropy loss $\mathcal{L}_{BCE}$:

    $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{panel} + \lambda_3 \mathcal{L}_{stitch} + \lambda_4 \mathcal{L}_{BCE}$

3. **Two-stage multimodal condition injection**:

    - **Stage 1**: The latent diffusion model (DiT architecture + T5 tokenizer) is trained with text conditioning only using the IDDPM loss, establishing a foundational generation capability.
    - **Stage 2**: Sketch and body-shape conditions are injected. Features extracted from sketch and body-shape embedders are concatenated and fused via a lightweight Transformer layer (since sketches vary with body shape), then statistically normalized to align the fused feature $\bm{F}_{bs}$ with the distribution of latent features $\bm{F}_z$:

    $\hat{\bm{F}}_z = \frac{(\bm{F}_{bs} - \bm{\mu}_{bs}) \times \bm{\sigma}_{bs}}{\bm{\sigma}_z + \epsilon} + \bm{\mu}_z + \bm{F}_z$

   Only the output layers of attention modules are fine-tuned to preserve responsiveness to text guidance. Sketch/text conditions are zeroed out with 25% probability to support single-modality generation.

### Loss & Training

- Autoencoder: $\lambda_1=5, \lambda_2=1, \lambda_3=1, \lambda_4=1$; trained for 12 hours.
- Stage 1 text-guided LDM: IDDPM loss; trained for 2 days.
- Stage 2 multimodal conditioning: converges with an additional 10 hours of training.
- Dataset: GarmentCode dataset with 120,000 patterns; text annotations assisted by GPT-4; sketches extracted via PiDiNet.

## Key Experimental Results

### Main Results

**Quantitative comparison (generation efficiency, body-shape fit, user evaluation)**:

| Method | Runtime↓ | Cloth-body Distance↓ (cm) | User Score↑ |
|--------|----------|--------------------------|-------------|
| RichDreamer | ~4 hours | 6.19 | 1.89 |
| Wonder3D | ~4 mins | 6.54 | 1.88 |
| Sewformer | ~3 mins | 5.45 | 2.10 |
| DressCode | ~3 mins | 3.69 | 3.56 |
| **SewingLDM (Ours)** | **~3 mins** | **2.20** | **4.60** |

**Reconstruction accuracy comparison**:

| Method | Panel L2↓ | Panel Acc↑ | Edge Acc↑ | Stitch Acc↑ | Failure Rate↓ |
|--------|----------|-----------|----------|------------|--------------|
| SewFormer* | 12.3 | 79.4 | 44.7 | 2.8 | 4.3% |
| AE (Ours) | 0.64 | 99.8 | 88.5 | 90.8 | 0 |
| SewingLDM | 3.13 | 97.8 | 82.7 | 84.2 | 0 |

### Ablation Study

**Latent space compression shape ablation**:

| Compression Shape | Reconstruct | Generate | Cloth-body Dist.↓ | Codebook Usage |
|------------------|-------------|----------|-------------------|----------------|
| No compression | ✓ | ✗ | - | - |
| 256×32, n=32 | ✓ | ✗ | - | 0% |
| 256×8, n=2 | ✓ | ✓ | 2.87 | 91% |
| **256×6, n=2** | **✓** | **✓** | **2.20** | **100%** |
| 256×4, n=2 | ✗ | - | - | - |

**Multimodal condition injection position ablation**: Injection at shallow layers (after block 0) yields the best results; injection at deeper layers causes the loss of critical garment components (sleeves, waistbands). Fine-tuning the output layers of both self-attention and cross-attention outperforms fine-tuning either one alone.

### Key Findings

- SewingLDM achieves a cloth-body distance of only 2.20 cm, substantially outperforming 3D mesh methods (>6 cm) and other pattern-based methods.
- The proposed method leads by a wide margin in user evaluation, scoring 4.60/5.0.
- The compact latent space (256×6, n=2) achieves 100% codebook usage, which is critical to generation quality.
- The autoencoder achieves centimeter-level reconstruction accuracy (Panel L2 = 0.64 cm), meeting industrial standards.

## Highlights & Insights

- **First body-shape-aware sewing pattern generation**: By conditioning the diffusion model on body shape, the generated garments can directly fit diverse body types without manual adjustment.
- **CG pipeline compatibility**: Generated patterns can be directly used for physical simulation and animation, resolving the penetration artifacts of 3D mesh methods.
- **Elegant compact quantized latent space design**: Five-value uniform quantization ($\{-1,-0.5,0,0.5,1\}$) simultaneously ensures reconstruction accuracy and eases distribution learning for the diffusion model.
- **Pragmatic two-stage training strategy**: Foundational text-based capability is established first, followed by fine-tuning to inject additional modalities, avoiding multimodal interference.

## Limitations & Future Work

- Special design details such as zippers and pockets cannot be handled.
- Alignment remains difficult for complex sketches (e.g., wedding dresses).
- The current dataset is sourced from GarmentCodeData, covering a limited range of design styles.
- Future work may explore more comprehensive everyday garment representations and additional conditioning inputs.

## Related Work & Insights

- The pattern vectorization approach of NeuralTailor serves as the foundation of this work, though its representational capacity limits the generation of complex garments.
- The scalability of the DiT architecture makes it well-suited for pattern generation at varying scales.
- The latent space compression strategy proposed in this paper (bounded quantization with high codebook usage) offers useful reference for generative tasks on other structured data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying LDM to sewing pattern generation is a novel cross-domain endeavor; body-shape awareness is a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative and qualitative comparisons are comprehensive, including user studies and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Offers substantial practical value for the digital garment design industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)
- [\[ICCV 2025\] TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation](taxadiffusion_progressively_trained_diffusion_model_for_fine-grained_species_gen.md)

</div>

<!-- RELATED:END -->
