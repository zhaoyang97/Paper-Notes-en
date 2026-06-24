---
title: >-
  [Paper Note] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3DGS] This paper proposes GuardSplat, which achieves high-capacity, high-fidelity, and robust copyright protection for 3DGS assets with a total optimization time of only 15 minutes. This is accomplished via CLIP-guided message decoupling optimization (training the decoder in only 5 minutes) and SH-aware watermark embedding (modifying only spherical harmonics offsets).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3DGS"
  - "watermarking"
  - "CLIP"
  - "spherical harmonics"
  - "copyright protection"
  - "anti-distortion"
date: 2026-05-08
content_hash: 415f4a85847a730a
---

# GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2411.19895](https://arxiv.org/abs/2411.19895)  
**Code**: [GitHub](https://github.com/NarcissusEx/GuardSplat)  
**Area**: 3D Vision  
**Keywords**: 3DGS, watermarking, CLIP, spherical harmonics, copyright protection, anti-distortion

## TL;DR

This paper proposes GuardSplat, which achieves high-capacity, high-fidelity, and robust copyright protection for 3DGS assets with a total optimization time of only 15 minutes. This is accomplished via CLIP-guided message decoupling optimization (training the decoder in only 5 minutes) and SH-aware watermark embedding (modifying only spherical harmonics offsets).

## Background & Motivation

**Background**: 3DGS has been widely applied in films, games, VR, and other fields due to its high fidelity and real-time rendering speed. However, copyright protection for 3DGS assets has not yet been effectively resolved.

**Limitations of Prior Work**:
- **Scheme (a) Direct 3DGS training on watermarked images**: Novel views cannot guarantee consistent watermarks, resulting in low extraction accuracy.
- **Scheme (b) Scene-by-scene decoder training (e.g., CopyRNeRF)**: Training a decoder from scratch for each scene is extremely time-consuming.
- **Scheme (c) Using 2D pre-trained decoders (e.g., WateRF)**: 2D watermarking networks suffer from a fidelity-capacity trade-off, and joint encoder-decoder training is time-consuming; directly using their decoders to optimize 3D models yields sub-optimal results.
- **General limitation**: Modifying all 3DGS attributes (position, covariance, opacity, etc.) disrupts the 3D structures, leading to low rendering fidelity.

**Key Challenge**: How to efficiently embed high-capacity watermarks into 3DGS while maintaining rendering quality and ensuring robustness against various distortion attacks?

**Key Insight**: Leverage the text-image alignment capability of CLIP to establish a bridge from the text domain to the image domain: train the decoder in the text domain (without requiring any images), and then directly apply it to the image domain for watermark extraction.

## Method

### Overall Architecture

Three phases:
1. **Message Decoder Training** (5 min): Encode binary messages into CLIP text tokens $\rightarrow$ CLIP text encoder $\rightarrow$ MLP decoder to extract messages.
2. **SH-aware Watermarking Embedding** (10 min): Freeze all attributes of 3DGS, and only optimize the spherical harmonics (SH) offsets $\mathbf{h}_i^o$.
3. **Message Extraction**: Rendered views $\rightarrow$ CLIP image encoder $\rightarrow$ pre-trained decoder $\rightarrow$ extracted messages.

### Key Designs

**1. CLIP-Guided Message Decoupling Optimization**
- **Function**: Optimize only the message decoder $\mathcal{D}_M$ (3-layer MLP, 512 $\rightarrow$ L) without training an encoder or utilizing any images.
- **Mechanism**:
    - Convert the binary message $M \in \{0,1\}^L$ into a CLIP token sequence $T$ via a mapping function $\Phi$.
    - Extract text features $F_T \in \mathbb{R}^{512}$ using the CLIP text encoder $\mathcal{E}_T$.
    - Recover the message $\hat{M}$ from $F_T$ using the MLP decoder and optimize it with binary cross-entropy (BCE) loss.
    - Leverage CLIP's text-image alignment during inference: the decoder can directly extract the message from CLIP visual features $F_V$.
- **Design Motivation**: Decouple the training of the encoder and decoder. The decoder training is free from fidelity constraints and takes only 5 minutes. Additionally, CLIP's 400M text-image pairs provide rich cross-modal representations.

**2. SH-aware Message Embedding**
- **Function**: Construct a learnable SH offset $\mathbf{h}_i^o \in \mathbb{R}^{48}$ for each 3D Gaussian while freezing all other attributes.
- **Mechanism**: 
    - Modify only the spherical harmonics (SH) coefficients (which control view-dependent appearance) without altering the positions $\mu$, covariances $\Sigma$, or opacities $\alpha$.
    - SH parameters control specular/reflection effects, which are only sensitive in local regions. Therefore, minor offsets have minimal impact on the overall rendering fidelity.
    - Offset regularization: $\mathcal{L}_{off} = -\frac{1}{N}\sum_{i=1}^{N}\|\mathbf{h}_i^o\|_2^2$
- **Design Motivation**: Modifying only the color representations preserves the integrity of the 3D structures, preventing malicious users from removing watermarks by manipulating model files.

**3. Anti-distortion Message Extraction**
- **Function**: Introduce a differentiable distortion layer during optimization to randomly simulate cropping, scaling, rotation, JPEG compression, and brightness jittering.
- **Mechanism**: Enable the SH offsets to learn to resist various distortions during the training process.
- **Design Motivation**: CLIP inherently exhibits robustness against Gaussian blur and noise, but is vulnerable to rotation and JPEG compression, necessitating explicit enhancement.

### Loss & Training

$$\mathcal{L} = \lambda_{recon}(\mathcal{L}_{rgb} + \mathcal{L}_{lpips}) + \lambda_{msg}\mathcal{L}_{msg} + \lambda_{off}\mathcal{L}_{off}$$

- $\lambda_{recon}=1$, $\lambda_{msg}=0.03$, $\lambda_{off}=10$
- $\mathcal{L}_{rgb}$: SSIM + L1 reconstruction loss
- $\mathcal{L}_{lpips}$: LPIPS perceptual loss
- $\mathcal{L}_{msg}$: Message extraction BCE loss

## Key Experimental Results

### Main Results (Blender + LLFF, 32-bit)

| Method | Bit Acc | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| CopyRNeRF | 78.08 | 26.13 | 0.896 | 0.041 |
| WateRF | 88.58 | 31.19 | 0.936 | 0.040 |
| GaussianMarker | 98.85 | 33.98 | 0.979 | 0.016 |
| **GuardSplat (Ours)** | **99.04** | **39.40** | **0.994** | **0.002** |

The PSNR is 5.4 dB higher than GaussianMarker, and LPIPS is 87% lower.

### Robustness (16-bit, Various Distortions)

| Distortion | GuardSplat | GaussianMarker | WateRF |
|---|---|---|---|
| None | 99.64 | 99.36 | 95.67 |
| Rotation (±π/6) | **94.56** | 70.84 | 93.13 |
| JPEG (10%) | **94.70** | 86.22 | 86.99 |
| VAE Attack | **82.35** | 52.00 | 51.73 |
| Combined | **93.38** | 83.49 | 84.12 |

### Efficiency Comparison

| Method | Decoder Training | Watermark Embedding | Total Time |
|---|---|---|---|
| CopyRNeRF | - | ~hours | hours |
| WateRF | ~hours | ~30min | hours |
| GaussianMarker | - | ~30min | ~30min |
| **GuardSplat** | **5min** | **10min** | **15min** |

### Key Findings

1. **Effectiveness of the CLIP Bridge**: The decoder trained in the text domain can directly transfer to the visual domain, and its extraction accuracy is higher than that of pre-trained 2D decoders.
2. **Crucial Role of SH-only Modification**: Compared to modifying all attributes (Offset_all), modifying only the SH offsets significantly boosts rendering fidelity (PSNR +5 dB) and makes the watermark more resilient against removal.
3. **StegExpose Security Detection**: The ROC curve is close to the reference line, meaning the watermark is undetectable by steganic analysis.
4. **48-bit High Capacity**: Even when embedding a 48-bit message, the Bit Acc still reaches 98.29% with a PSNR of 38.90.

## Highlights & Insights

- Creative utilization of CLIP's text-image alignment: Training the decoder in the text domain offers an elegant zero-shot migration solution.
- Exploiting domain knowledge of 3DGS via SH-aware embedding: Since SH controls view-dependent effects, it serves as the optimal location for embedding.
- The total optimization time of 15 minutes makes practical commercial deployment feasible.
- The design of the anti-distortion module enables watermark extraction even under VAE attacks (82.35%), outperforming competing methods by a large margin.

## Limitations & Future Work

- The SH offsets are stored directly in the model file, which could theoretically be vulnerable to targeted attacks (e.g., resetting SH followed by fine-tuning).
- The limited resolution of the CLIP ViT-B/32 visual encoder might affect the embedding of watermarks in high-frequency details.
- Compatibility with other variants like 2DGS or 3DGS++ has not been evaluated.
- Standard CLIP might be replaced by fine-tuned versions, potentially leading to decoding failures.
- Experiments were only conducted on synthetic (Blender) and relatively simple real-world (LLFF) scenes.

## Related Work & Insights

- CopyRNeRF pioneered NeRF watermarking, but scene-by-scene training is impractical. This paper resolves the efficiency bottleneck through decoupled optimization.
- GaussianMarker modifies all attributes, leading to a decrease in rendering fidelity. The SH-only strategy proposed in this paper is a superior approach.
- Insight: The alignment property of CLIP can act as a cross-modal "translator", paving the way for more 3D security and privacy tasks.

## Rating

⭐⭐⭐⭐ — Elegant methodology. Both core designs, i.e., CLIP decoupling and SH-only embedding, are supported by solid insights. The experiments are comprehensive, covering five key dimensions: capacity, fidelity, robustness, security, and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)
- [\[ICLR 2026\] CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting](../../ICLR2026/3d_vision/compmarkgs_robust_watermarking_for_compressed_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](../../ICCV2025/3d_vision/robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)

</div>

<!-- RELATED:END -->
