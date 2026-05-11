---
title: >-
  [Paper Note] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D steganography] This paper proposes Splats in Splats, the first steganography framework that embeds 3D content into 3DGS assets without modifying any vanilla 3DGS attributes. Through importance-graded spherical harmonic coefficient encryption and autoencoder-assisted opacity mapping, it achieves 5.31% higher scene fidelity and 3× faster rendering speed compared to prior methods.
tags:
  - AAAI 2026
  - 3D Vision
  - 3D steganography
  - Gaussian Splatting
  - spherical harmonics
  - copyright protection
  - information hiding
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

# Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2412.03121](https://arxiv.org/abs/2412.03121)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D steganography, Gaussian Splatting, spherical harmonics, copyright protection, information hiding

## TL;DR

This paper proposes Splats in Splats, the first steganography framework that embeds 3D content into 3DGS assets without modifying any vanilla 3DGS attributes. Through importance-graded spherical harmonic coefficient encryption and autoencoder-assisted opacity mapping, it achieves 5.31% higher scene fidelity and 3× faster rendering speed compared to prior methods.

## Background & Motivation

3D Gaussian Splatting has become the dominant 3D asset representation, widely applied in 3D reconstruction and generation. However, copyright protection for 3DGS assets is increasingly urgent. Existing 3DGS steganography methods suffer from critical flaws:

**Core Problem: Usability is Neglected**
- **GS-Hider**: Introduces a coupled feature field and neural decoder to simultaneously render the original and hidden scenes, but modifies the 3DGS rendering pipeline and attribute structure.
- **SecureGS**: Built on Scaffold-GS, similarly modifying the vanilla 3DGS architecture.
- These methods render the modified 3DGS assets incompatible with standard 3DGS rendering engines (e.g., SIBR Viewer).
- Special rendering tools are required, severely hindering practical deployment.

The authors pose the core question: "Can hidden information be embedded into 3DGS itself without modifying any vanilla 3DGS attributes?" The answer is affirmative, and the key lies in a deep understanding of spherical harmonic (SH) properties.

## Method

### Overall Architecture

The Splats in Splats pipeline consists of three steps:
1. **Hidden attribute training**: Two sets of SH coefficients and opacities are trained separately using views from the original and hidden scenes, sharing Gaussian primitive positions.
2. **Importance-graded SH coefficient encryption**: Hidden SH coefficients are embedded into the high-order components of the original SH in a graded manner.
3. **Autoencoder-assisted opacity mapping**: A convolutional autoencoder establishes a mapping from original to hidden opacity.

### Key Designs

#### 1. **Insight into Spherical Harmonics**: Discovering Information Redundancy

Spherical harmonics represent view-dependent color:
$$F(s) \approx \sum_{l=0}^{q-1} \sum_{m=-l}^{l} f_l^m Y_l^m(s)$$

Key findings:
- Low-order (small band index $l$) SH basis functions represent low-frequency information, capturing the primary appearance of the scene.
- High-order SH basis functions represent high-frequency information, but contribute minimally in most scenes.
- High-order SH coefficients exhibit substantial **information redundancy**, making information embedding in these coefficients difficult to detect and capable of maintaining high fidelity.

Experimental validation: Images rendered using only the 0th-order SH differ minimally from those rendered with all orders, confirming the information redundancy of high-order SH.

#### 2. **Importance-graded SH Coefficient Encryption**: Secure and Robust Information Embedding

The core idea is to embed the more important low-order hidden SH coefficients into the less important high-order components of the original SH.

**Zeroing operation**: Low bits of the original coefficient $c_{i,j}$ are cleared according to the graded importance of the SH band:
$$\tilde{c}_{i,j} = c_{i,j} \& \sim((1 << (k + \lfloor\sqrt{j}\rfloor)) - 1)$$

**Embedding operation**: Hidden coefficients in reversed order are embedded via bit-shifting and XOR:
$$c_{i,j}^w = \tilde{c}_{i,j} \oplus (c_{i,n-1-j}' >> (\gamma - (k + \lfloor\sqrt{j}\rfloor)))$$

where $n-1-j$ indicates that the order of hidden coefficients is reversed—i.e., the low-order (important) hidden coefficients are embedded into the high-order (unimportant) original components. This ensures:
- High fidelity for the original scene (only low bits of high-order coefficients are modified).
- Recoverability of the hidden scene (important information is placed in positions less susceptible to noise).
- Robustness against noise attacks (the graded strategy distributes critical information into more secure bit positions).

#### 3. **Autoencoder-assisted Opacity Mapping**: Hiding Geometric Information

SH coefficients hide appearance information, while opacity carries geometric structure.

**Threshold filtering**: A threshold $\tau$ filters unimportant hidden opacities:
$$\mathcal{I} = \{i \mid \alpha_i' > \tau, i \in \{1,2,...,N\}\}$$

**Complementarity observation**: Original and hidden opacities exhibit a complementary relationship at many positions; thus $1 - \alpha_\mathcal{I}$ is used as the autoencoder input.

**Mapping learning**:
$$W_p^* = \arg\min_{\mathcal{E},\mathcal{D}} \ell_{mse}(\mathcal{D}(\mathcal{E}(1-\alpha_\mathcal{I})), \alpha_\mathcal{I}')$$

The autoencoder consists of simple convolutional/deconvolutional layers to guarantee real-time rendering. The trained model parameters $W_p^*$ are stored as the private key.

**Extraction**:
$$c_{i,j}' = c_{i,n-1-j}^w \& (1 << (k + \lfloor\sqrt{n-1-j}\rfloor))$$
$$\alpha_\mathcal{I}' = \mathcal{D}_p(\mathcal{E}_p(1-\alpha_\mathcal{I}))$$

### Loss & Training

- Standard 3DGS training procedure, 30,000 iterations.
- Two sets of SH coefficients and opacities are trained separately while sharing Gaussian primitive positions.
- The autoencoder is trained with MSE loss.
- Threshold $\tau = 0.25$, shift length $k = 17$.

## Key Experimental Results

### Main Results (Mip-NeRF360 Dataset, PSNR↑)

| Method | Original PSNR | Hidden PSNR | FPS | Vanilla Pipeline | Vanilla Attributes |
|------|-------------|-------------|---------|----------------|----------------|
| 3DGS+StegaNeRF | 24.120 | 16.681 | 22 | ✗ | ✓ |
| GS-Hider | 25.817 | 25.179 | 44 | ✗ | ✗ |
| SecureGS | 26.574 | 23.679 | 36 | ✗ | ✗ |
| **Ours** | **26.749** | **26.517** | **118** | ✓ | ✓ |

- Highest original scene fidelity (surpassing SecureGS by 0.175 and GS-Hider by 0.932 PSNR).
- Best hidden scene quality (surpassing GS-Hider by 1.338 and SecureGS by 2.838 PSNR).
- Rendering speed 3× faster than GS-Hider.
- Training time of only 47 minutes, approximately 40% that of GS-Hider.

### Robustness (Random Pruning Attacks)

| Pruning Ratio | SecureGS PSNR | GS-Hider PSNR | Ours PSNR | Note |
|---------|--------------|--------------|-----------|------|
| 5% | 22.920 | 24.923 | **26.415** | Substantial advantage |
| 10% | 22.596 | 24.673 | **26.375** | Only −0.04 |
| 15% | 22.280 | 24.371 | **26.346** | Continues to remain stable |
| 25% | 21.485 | 23.661 | **26.320** | Only −0.095, far better than GS-Hider (−1.260) |

Performance is even more remarkable under sequential pruning: only 0.002 PSNR drop after 25% pruning.

### Ablation Study

| Configuration | Original PSNR | Hidden PSNR | Note |
|------|-------------|-------------|------|
| w/o opacity mapping | 24.209 | 23.346 | Opacity mapping is critical for both scenes |
| w/o SH encryption | 26.795 | 11.092 | SH encryption is essential for the hidden scene |
| **SH + opacity (Full)** | **26.749** | **26.517** | Both components are indispensable |

**Noise robustness (hidden scene PSNR under different Gaussian noise levels)**:

| Encryption | σ=0.0005 | σ=0.001 | σ=0.005 | σ=0.01 | Average |
|---------|---------|---------|---------|--------|------|
| Uniform (AVG) | 24.167 | 21.991 | 11.442 | 7.471 | 16.267 |
| **Graded (Ours)** | **24.577** | **24.509** | **22.797** | **20.032** | **22.979** |

The advantage of graded encryption is particularly pronounced under high noise (gap of 12.56 PSNR at $\sigma=0.01$).

### Key Findings

1. High-order SH coefficients indeed exhibit substantial information redundancy and can be used for secure embedding.
2. The complementarity observation ($1-\alpha \leftrightarrow \alpha'$) facilitates easier mapping learning.
3. Importance-graded encryption improves robustness against noise attacks by 40%+ over uniform encryption.
4. Maintaining the vanilla 3DGS pipeline is the only approach directly compatible with SIBR Viewer.
5. The threshold $\tau$ has a significant impact on quality; $\tau = 0.25$ is optimal.

## Highlights & Insights

1. **Precise problem formulation**: The paper is the first to articulate the requirement that steganography must preserve vanilla 3DGS attributes, directly addressing practical deployment pain points.
2. **In-depth exploitation of SH properties**: Discovering and leveraging information redundancy in high-order SH reflects a deep understanding of the 3DGS data structure.
3. **Elegant graded encryption design**: Low-order hidden coefficients → high-order original coefficients; this simultaneously protects important information and minimizes impact on the original scene.
4. **Exceptional robustness**: Only 0.095 PSNR drop after 25% random pruning, making the scheme practically impervious to such attacks.
5. **Best-in-class practicality**: The first steganography solution directly deployable in standard 3DGS rendering engines.

## Limitations & Future Work

1. View-dependent details (e.g., specular reflections) are somewhat affected as high-order SH coefficients are partially occupied.
2. Hidden scene quality remains slightly below that of the original scene.
3. Only a single hidden 3D scene can be embedded; multi-content embedding has not been explored.
4. Autoencoder parameters must be securely stored and transmitted.
5. Performance under 3DGS compression scenarios has not been discussed.

## Related Work & Insights

- **GS-Hider**: The most direct competitor; breaks the pipeline by introducing a coupled feature field and neural decoder.
- **StegaNeRF/WaterRF**: Pioneering NeRF steganography methods, but not suited to the explicit representation of 3DGS.
- **3DGS compression**: SH coefficient quantization in methods such as CompGS may conflict with steganographic embedding.
- **Inspiration**: The methodology for analyzing SH coefficient redundancy can be extended to other SH-based 3D representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First steganography method preserving vanilla 3DGS attributes; importance-graded SH encryption is ingeniously designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across fidelity, efficiency, robustness, security, and usability.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear and the method is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — A directly deployable copyright protection solution for the 3DGS ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](../../ICCV2025/3d_vision/cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[CVPR 2026\] Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture](../../CVPR2026/3d_vision/using_gaussian_splats_to_create_high-fidelity_facial_geometry_and_texture.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](../../ICCV2025/3d_vision/a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
