---
title: >-
  [Paper Note] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Steganography] Introduces Splats in Splats, the first steganography framework that embeds 3D content into 3DGS assets without modifying any vanilla 3DGS attributes. Through importance-graded spherical harmonics (SH) coefficient encryption and autoencoder-assisted opacity mapping, it achieves 5.31% higher scene fidelity and 3x faster rendering speeds.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Steganography"
  - "Gaussian Splatting"
  - "Spherical Harmonics"
  - "Copyright Protection"
  - "Information Hiding"
date: 2026-05-08
content_hash: 81f8856e4ac8c005
---

# Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2412.03121](https://arxiv.org/abs/2412.03121)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Steganography, Gaussian Splatting, Spherical Harmonics, Copyright Protection, Information Hiding

## TL;DR

Introduces Splats in Splats, the first steganography framework that embeds 3D content into 3DGS assets without modifying any vanilla 3DGS attributes. Through importance-graded spherical harmonics (SH) coefficient encryption and autoencoder-assisted opacity mapping, it achieves 5.31% higher scene fidelity and 3x faster rendering speeds.

## Background & Motivation

3D Gaussian Splatting (3DGS) has emerged as a mainstream representation for 3D assets, widely applied in 3D reconstruction and generation. However, copyright protection for 3DGS assets has become increasingly urgent. Existing 3DGS steganography methods suffer from severe drawbacks:

**Core Problem: Neglected Usability**
- **GS-Hider**: Introduces a coupled feature field and a neural decoder to simultaneously render the original and hidden scenes, which modifies the 3DGS rendering pipeline and attribute structure.
- **SecureGS**: Based on Scaffold-GS, it also modifies the vanilla 3DGS architecture.
- These methods render the modified 3DGS assets incompatible with standard 3DGS rendering engines (such as SIBR Viewer).
- Users require special rendering tools, which severely hinders practical deployment.

The authors propose the core question: "Can hidden information be embedded within 3DGS itself without modifying any attributes of vanilla 3DGS?" The answer is affirmative, and the key lies in an in-depth insight into the characteristics of Spherical Harmonics (SH).

## Method

### Overall Architecture

The pipeline of Splats in Splats consists of three stages:
1. **Hidden Attribute Training**: Trains two sets of SH coefficients and opacities using views of the original and hidden scenes respectively, while sharing the positions of Gaussian primitives.
2. **Importance-graded SH Coefficient Encryption**: Embeds the hidden SH coefficients into the high-order components of the original SH based on importance grading.
3. **Autoencoder-assisted Opacity Mapping**: Employs a convolutional autoencoder to map the original opacity to the hidden opacity.

### Key Designs

#### 1. **Deep Insight into Spherical Harmonics (Insight in SH)**: Discovery of Information Redundancy

Spherical harmonics are used to represent view-dependent colors:
$$F(s) \approx \sum_{l=0}^{q-1} \sum_{m=-l}^{l} f_l^m Y_l^m(s)$$

Key discoveries:
- Low-order SH basis functions (small band index $l$) represent low-frequency information, capturing the main appearance of the scene.
- High-order SH basis functions represent high-frequency information, which contributes minimally in most scenes.
- High-order SH coefficients exhibit substantial **information redundancy**, allowing hidden information to be embedded with minimal detection risk while maintaining high fidelity.

Experimental verification: The difference between images rendered using only degree-0 SH and those using all degrees is negligible, confirming the information redundancy in high-order SH.

#### 2. **Importance-graded SH Coefficient Encryption**: Secure and Robust Information Embedding

The core idea is to embed the more important hidden low-order SH coefficients into the high-order (less important) components of the original SH.

**Clearing operation**: Clearing the low-order bits of the original coefficient $c_{i,j}$ according to the graded importance of SH degrees:
$$\tilde{c}_{i,j} = c_{i,j} \& \sim((1 << (k + \lfloor\sqrt{j}\rfloor)) - 1)$$

**Embedding operation**: Embedding the reversed-order hidden coefficients through bit-shifting and XOR:
$$c_{i,j}^w = \tilde{c}_{i,j} \oplus (c_{i,n-1-j}' >> (\gamma - (k + \lfloor\sqrt{j}\rfloor)))$$

Here, $n-1-j$ indicates that the order of the hidden coefficients is reversed—meaning the hidden low-order (important) coefficients are embedded into the original high-order (unimportant) components. Consequently:
- Maintains high fidelity for the original scene (modifying only the low-order bits of high-order coefficients).
- Allows recovery of the hidden scene (important information is protected in positions less susceptible to noise).
- Exhibits robustness against noise attacks (the grading strategy distributes key information across safer positions).

#### 3. **Autoencoder-assisted Opacity Mapping**: Hiding of Geometric Information

SH coefficients hide visual appearance, while opacity carries geometric structures.

**Threshold filtering**: Setting a threshold $\tau$ to filter out insignificant hidden opacities:
$$\mathcal{I} = \{i \mid \alpha_i' > \tau, i \in \{1,2,...,N\}\}$$

**Complementarity observation**: Original and hidden opacities exhibit a complementary relationship at many locations; thus, $1-\alpha_\mathcal{I}$ is utilized as the autoencoder input.

**Mapping learning**:
$$W_p^* = \arg\min_{\mathcal{E},\mathcal{D}} \ell_{mse}(\mathcal{D}(\mathcal{E}(1-\alpha_\mathcal{I})), \alpha_\mathcal{I}')$$

The autoencoder consists of simple convolutional/deconvolutional layers to guarantee real-time rendering. The trained model parameters $W_p^*$ are stored as a private key.

**Extraction process**:
$$c_{i,j}' = c_{i,n-1-j}^w \& (1 << (k + \lfloor\sqrt{n-1-j}\rfloor))$$
$$\alpha_\mathcal{I}' = \mathcal{D}_p(\mathcal{E}_p(1-\alpha_\mathcal{I}))$$

### Loss & Training

- Uses the standard 3DGS training pipeline with 30,000 iterations.
- Two sets of SH coefficients and opacities are trained independently while sharing Gaussian primitive locations.
- The autoencoder is trained using MSE loss.
- Threshold is set to $\tau=0.25$, and shift length is set to $k=17$.

## Key Experimental Results

### Main Results (Mip-NeRF360 Dataset, PSNR↑)

| Method | Original Scene PSNR | Hidden Scene PSNR | Rendering FPS | Keeps Vanilla Pipeline | Keeps Vanilla Attributes |
|------|-------------|-------------|---------|----------------|----------------|
| 3DGS+StegaNeRF | 24.120 | 16.681 | 22 | ✗ | ✓ |
| GS-Hider | 25.817 | 25.179 | 44 | ✗ | ✗ |
| SecureGS | 26.574 | 23.679 | 36 | ✗ | ✗ |
| **Ours** | **26.749** | **26.517** | **118** | ✓ | ✓ |

- Achieves the highest original scene fidelity (exceeding SecureGS by 0.175 dB and GS-Hider by 0.932 dB PSNR).
- Achieves optimal hidden scene quality (surpassing GS-Hider by 1.338 dB and SecureGS by 2.838 dB PSNR).
- Rendering speed is 3x faster than GS-Hider.
- Training time is only 47 minutes, which is approximately 40% of GS-Hider.

### Robustness Experiment (Random Pruning Attack)

| Pruning Ratio | SecureGS PSNR | GS-Hider PSNR | Ours PSNR | Description |
|---------|--------------|--------------|-----------|------|
| 5% | 22.920 | 24.923 | **26.415** | Significant advantage |
| 10% | 22.596 | 24.673 | **26.375** | Slight decrease of only 0.04 |
| 15% | 22.280 | 24.371 | **26.346** | Continues to remain stable |
| 25% | 21.485 | 23.661 | **26.320** | Decreases by only 0.095, far superior to GS-Hider (decreases by 1.260) |

Even more outstanding under sequential pruning: only decreases by 0.002 dB PSNR after 25% pruning.

### Ablation Study

| Configuration | Original Scene PSNR | Hidden Scene PSNR | Description |
|------|-------------|-------------|------|
| w/o opacity mapping | 24.209 | 23.346 | Opacity mapping is critical for both scenes |
| w/o SH encryption | 26.795 | 11.092 | SH encryption is vital for the hidden scene |
| **SH + opacity (Full)** | **26.749** | **26.517** | Both components are indispensable |

**Noise Robustness (Hidden scene PSNR under various Gaussian noise levels)**:

| Encryption Scheme | σ=0.0005 | σ=0.001 | σ=0.005 | σ=0.01 | Average |
|---------|---------|---------|---------|--------|------|
| Uniform Encryption (AVG) | 24.167 | 21.991 | 11.442 | 7.471 | 16.267 |
| **Graded Encryption (Ours)** | **24.577** | **24.509** | **22.797** | **20.032** | **22.979** |

Graded encryption demonstrates a particularly pronounced advantage under high noise levels (with a gap of 12.56 dB PSNR at σ=0.01).

### Key Findings

1. High-order SH coefficients indeed exhibit substantial information redundancy, rendering them suitable for secure embedding.
2. The complementarity observation of opacity ($1-\alpha \leftrightarrow \alpha'$) simplifies the mapping learning.
3. Importance-graded encryption improves robustness against noise attacks by more than 40% compared to uniform encryption.
4. Preserving the vanilla 3DGS pipeline is the only way to ensure direct compatibility with the SIBR Viewer.
5. The threshold $\tau$ significantly impacts quality, with $\tau=0.25$ being optimal.

## Highlights & Insights

1. **Unique Problem Formulation**: Pioneers the requirement of maintaining vanilla 3DGS attributes in 3D steganography, directly addressing the pain points of practical deployment.
2. **Deep Exploration of SH Characteristics**: Discovers and cleverly leverages the information redundancy in high-order SH, representing an in-depth academic understanding of the 3DGS data structure.
3. **Exquisite Graded Encryption Design**: Maps low-order hidden coefficients to high-order original coefficients, which safeguards key information while minimizing the impact on the original scene.
4. **Extreme Robustness**: Experiences only a 0.095 dB PSNR drop under 25% random pruning, making it virtually immune to disruption from attacks in practical applications.
5. **Optimal Practicality**: Represents the first steganography scheme that can be directly deployed in standard 3DGS rendering engines.

## Limitations & Future Work

1. Slightly impacts view-dependent details (as high-order SH coefficients are partially occupied).
2. The quality of the hidden scene remains slightly lower than that of the original scene.
3. Only supports embedding a single hidden 3D scene; multi-content embedding has not been explored.
4. The autoencoder parameters require secure storage and transmission.
5. Has not discussed the performance under 3DGS compression scenarios.

## Related Work & Insights

- **GS-Hider**: The most direct competitor, but it is unusable due to pipeline modifications.
- **StegaNeRF/WaterRF**: Pioneering works in NeRF steganography, but inapplicable to the explicit representation of 3DGS.
- **3DGS Compression**: Quantization of SH coefficients in methods like CompGS may conflict with steganography.
- **Insight**: The methodology of analyzing SH coefficient redundancy can be generalized to other SH-based 3D representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First steganography method to maintain vanilla 3DGS attributes, featuring an ingenious SH graded encryption design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across fidelity, efficiency, robustness, security, and usability.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem formulation and intuitive methodologies.
- **Value**: ⭐⭐⭐⭐⭐ — A directly deployable copyright protection scheme for 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](../../ICCV2025/3d_vision/cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](../../ICCV2025/3d_vision/a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[ECCV 2024\] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](../../ECCV2024/3d_vision/splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)
- [\[ICLR 2026\] CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting](../../ICLR2026/3d_vision/compmarkgs_robust_watermarking_for_compressed_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
