---
title: >-
  [Paper Note] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes "Splats in Splats," the first steganography framework that embeds 3D hidden content into 3DGS assets without modifying any vanilla 3DGS attributes. It ach…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "steganography"
  - "copyright protection"
  - "spherical harmonics"
  - "information embedding"
date: 2026-05-08
content_hash: 981b578eaf0e76dc
---

# Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2412.03121](https://arxiv.org/abs/2412.03121)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, steganography, copyright protection, spherical harmonics, information embedding

## TL;DR

This paper proposes "Splats in Splats," the first steganography framework that embeds 3D hidden content into 3DGS assets without modifying any vanilla 3DGS attributes. It achieves secure, robust, and efficient copyright protection through importance-graded spherical harmonic (SH) coefficient encryption and autoencoder-assisted opacity mapping.

## Background & Motivation

### Problem Definition

With the widespread adoption of 3D Gaussian Splatting in 3D reconstruction and generation, protecting the copyright of 3DGS assets has become increasingly urgent. The goal of 3DGS steganography is to embed hidden information (e.g., another 3D scene) into a 3DGS asset so that ownership can be verified when needed.

### Core Motivation

Existing 3DGS copyright protection techniques suffer from two critical flaws:

**1. Protecting only rendered images rather than the 3DGS asset itself**: Methods such as GaussianMarker embed information into rendered images, but malicious users can generate new samples using alternative rendering strategies, bypassing watermark protection. What truly needs to be protected is the **3DGS asset itself**.

**2. Compromising the usability of 3DGS assets**: Existing methods (GS-Hider, SecureGS) modify vanilla 3DGS attributes and rendering pipelines:
- GS-Hider introduces a coupled feature field and neural decoder, making the asset incompatible with standard 3DGS rendering engines.
- SecureGS is built on Scaffold-GS rather than vanilla 3DGS, facing the same compatibility issues.
- These modifications make 3DGS assets difficult to deploy in standard toolchains.

### Core Problem

**Does there exist an approach that can embed hidden information into 3DGS itself without modifying any vanilla 3DGS attributes?**

### Key Insight

The authors conduct an in-depth analysis of spherical harmonics (SH) and find that:
- Low-order SH coefficients capture the primary appearance of the scene (low-frequency information), while high-order SH coefficients contain only a small amount of high-frequency reflection information.
- **High-order SH coefficients exhibit information redundancy**, making them suitable for embedding information without significantly affecting rendering quality.
- Meanwhile, the opacity attribute can leverage an autoencoder to establish a mapping between the original and hidden scenes.

## Method

### Overall Architecture

The embedding and extraction pipeline of Splats in Splats consists of three steps:
1. **Hidden attribute training**: Two sets of SH coefficients and opacities are trained using views from both the original and hidden scenes, sharing the same Gaussian primitive positions.
2. **Importance-graded SH coefficient encryption/decryption**: Hidden SH coefficients are embedded into the high-order components of the original SH in a graded manner according to SH band order.
3. **Autoencoder-assisted opacity mapping**: A lightweight convolutional autoencoder is trained to learn the mapping from original to hidden opacity.

### Key Designs

#### 1. In-depth Analysis of Spherical Harmonics

**Function**: Reveals the varying importance of SH coefficients across bands, providing a theoretical basis for the embedding strategy.

**SH basis function definition**:

$$F(s) \approx \sum_{l=0}^{q-1} \sum_{m=-l}^{l} f_l^m Y_l^m(s)$$

where $l$ is the band index; low-order bands correspond to low-frequency basis functions, and high-order bands correspond to high-frequency basis functions.

**Key finding**: By rendering images while retaining only specific-order SH coefficients (setting others to zero), experiments show that:
- The 0th-order SH captures the dominant color and structure of the scene.
- High-order SH contains only minor high-frequency specular details.
- The redundancy in high-order SH coefficients makes information embedding imperceptible.

#### 2. Importance-graded SH Coefficient Encryption

**Function**: Embeds hidden SH coefficients, graded by importance, into the high-order components of the original SH.

**Mechanism**: The low-order (most important) hidden SH coefficients are embedded into the high-order (least important) components of the original SH to maximize fidelity.

**Embedding process**:

Step 1 — Zero out the low bits of the original coefficients according to graded importance:

$$\tilde{c}_{i,j} = c_{i,j} \& \sim((1 << (k + \lfloor\sqrt{j}\rfloor)) - 1)$$

Step 2 — Shift the hidden coefficients to the corresponding positions and apply XOR:

$$c_{i,j}^w = \tilde{c}_{i,j} \oplus (c_{i,n-1-j}' >> (\gamma - (k + \lfloor\sqrt{j}\rfloor)))$$

where:
- $k=17$ is the shift length for 0th-order coefficients;
- $\gamma$ is the maximum bit length;
- $n-1-j$ indicates that the hidden coefficients are reversed to match the graded selection (low-order hidden → high-order original).

**Extraction process**: Recovered via bitwise AND:

$$c_{i,j}' = c_{i,n-1-j}^w \& (1 << (k + \lfloor\sqrt{n-1-j}\rfloor))$$

**Design motivation**:
- Graded embedding minimizes impact on the original scene (high-order coefficients already contribute little).
- The low-order (most important appearance) coefficients of the hidden scene are allocated more bits for storage.
- This strategy is more robust to noise attacks than uniform encryption.

#### 3. Autoencoder-assisted Opacity Mapping

**Function**: Learns a mapping from original opacity to hidden opacity, avoiding direct storage of hidden opacity values.

**Mechanism**:
1. A threshold $\tau = 0.25$ is used to filter unimportant hidden opacity values: $\mathcal{I} = \{i \mid \alpha_i' > \tau\}$.
2. The 3D coordinates $x_\mathcal{I}$ of these positions are stored.
3. It is observed that $\alpha_i$ and $\alpha_i'$ exhibit a complementary relationship at many positions.
4. A convolutional autoencoder learns the mapping $1 - \alpha_\mathcal{I} \to \alpha_\mathcal{I}'$:

$$W_p^* = \arg\min_{\mathcal{E}, \mathcal{D}} \ell_{mse}(\mathcal{D}(\mathcal{E}(1 - \alpha_\mathcal{I})), \alpha_\mathcal{I}')$$

**Extraction**: $\alpha_\mathcal{I}' = \mathcal{D}_p(\mathcal{E}_p(1 - \alpha_\mathcal{I}))$

**Design motivation**:
- The original opacity is not modified; the autoencoder parameters $W_p^*$ serve as the private key.
- Threshold filtering reduces storage requirements (unimportant hidden opacities are set to zero).
- The observed complementary relationship motivates using $1 - \alpha$ as input rather than $\alpha$ directly.

### Loss & Training

- Standard 3DGS training procedure, 30,000 iterations, NVIDIA A800 GPU.
- Hyperparameters: $\tau = 0.25$, $k = 17$.
- Private key stored by the asset owner: autoencoder parameters $W_p^*$ + important position coordinates $x_\mathcal{I}$.

## Key Experimental Results

### Main Results

**Fidelity comparison on the Mip-NeRF360 dataset (PSNR↑)**:

| Method | Original PSNR | Hidden PSNR | FPS | SIBR Compatible |
|---|---|---|---|---|
| 3DGS+StegaNeRF | 24.12 | 16.68 | 22 | ✗ |
| GS-Hider | 25.82 | 25.18 | 44 | ✗ |
| SecureGS | 26.57 | 23.68 | 36 | ✗ |
| **Ours** | **26.75** | **26.52** | **118** | **✓** |

**Key metrics**:
- Highest original scene fidelity (26.75 PSNR), surpassing GS-Hider by 5.31%.
- Best hidden scene quality (26.52 PSNR).
- Rendering speed **3× faster** than GS-Hider (118 vs. 44 FPS).
- The **only method** compatible with the vanilla 3DGS SIBR rendering engine.
- Training time of only 47 minutes, less than half that of GS-Hider.

### Ablation Study

**Necessity of SH/opacity encryption**:

| Configuration | Original PSNR↑ | Hidden PSNR↑ | Note |
|---|---|---|---|
| w/o Opacity | 24.21 | 23.35 | Both degrade without opacity mapping |
| w/o SH | 26.80 | 11.09 | Hidden scene collapses without SH encryption |
| **SH+Opacity (Full)** | **26.75** | **26.52** | Two components are complementary |

**Robustness under 25% sequential pruning**:

| Method | PSNR↑ | SSIM↑ | Note |
|---|---|---|---|
| SecureGS | 16.96 | 0.577 | PSNR drops sharply by 5.17 |
| GS-Hider | 25.17 | 0.780 | PSNR drops by 0.01 |
| **Ours** | **26.52** | **0.797** | PSNR drops by only 0.002 |

**Noise robustness (importance-graded vs. uniform encryption)**:

| Method | Noise 0.0005 | Noise 0.001 | Noise 0.005 | Noise 0.01 | Average |
|---|---|---|---|---|---|
| Uniform (AVG) | 24.17 | 21.99 | 11.44 | 7.47 | 16.27 |
| **Importance-graded (Ours)** | **24.58** | **24.51** | **22.80** | **20.03** | **22.98** |

### Key Findings

1. **Opacity mapping protects geometry; SH encryption protects appearance**: The two components correspond to the two essential constituents of 3D assets.
2. **Extreme robustness under pruning attacks**: 25% sequential pruning causes only a 0.002 PSNR drop.
3. **Importance-graded encryption substantially improves noise robustness**: Under strong noise ($\sigma=0.01$), the uniform scheme collapses to 7.47 PSNR while the proposed method maintains 20.03.
4. **Shift length $k$ has negligible impact on quality**: PSNR for both the original and hidden scenes remains nearly unchanged as $k$ varies from 10 to 22.
5. **The only method that preserves the complete vanilla 3DGS pipeline and attributes.**

## Highlights & Insights

1. **Precise problem formulation**: The paper clearly distinguishes between "protecting rendered images" and "protecting the 3DGS asset itself," and highlights the importance of usability.
2. **In-depth analysis of SH coefficient redundancy** is highly instructive—the information redundancy in high-order SH is a broadly exploitable property.
3. **Bit-level operations for information embedding**: Pure arithmetic operations introduce no neural networks, thereby fully preserving the vanilla 3DGS pipeline.
4. **Discovery of complementary relationship**: The complementarity between $\alpha$ and $\alpha'$ is an interesting empirical observation.
5. **Strong practicality**: 100+ FPS, standard engine compatibility, and fast training.

## Limitations & Future Work

1. View-dependent details (e.g., specular reflections) are somewhat affected, as high-order SH coefficients are partially occupied by the hidden content.
2. Storing the private key (autoencoder parameters + position coordinates) increases storage overhead on the owner's side.
3. Hidden scene quality (26.52 PSNR) remains slightly below that of a natively trained 3DGS, leaving room for improvement.
4. Only a single hidden scene can currently be embedded; multi-layer embedding has not been explored.
5. Fixed-point integer representation is used in the bit-shift operations, limiting precision due to quantization.

## Related Work & Insights

- **GS-Hider**: The most direct competitor; uses a coupled feature field and neural decoder but breaks the vanilla pipeline.
- **SecureGS**: A Scaffold-GS-based steganography method that similarly modifies attributes and the pipeline.
- **StegaNeRF**: A representative NeRF steganography method that embeds secret images by fine-tuning NeRF weights.
- **WaterRF**: A NeRF watermarking method that embeds binary messages using discrete wavelet transforms.
- **3DGS**: This work is built entirely on the vanilla 3DGS framework to ensure compatibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First 3DGS steganography method that modifies no vanilla attributes; importance-graded SH encryption is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across fidelity, efficiency, robustness, security, and usability.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear and SH analysis is thorough, though notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical; fully compatible with the vanilla 3DGS ecosystem.

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](../../ICCV2025/3d_vision/a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[AAAI 2026\] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?](can_protective_watermarking_safeguard_the_copyright_of_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](../../ICCV2025/3d_vision/cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
