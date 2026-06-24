---
title: >-
  [Paper Note] Rotation-Equivariant Self-Supervised Method in Image Denoising
description: >-
  [CVPR 2025][Image Restoration][Self-supervised denoising] This work introduces rotation-equivariant convolutions to self-supervised image denoising for the first time. It rigorously analyzes the impact of up/downsampling operators on equivariance, provides the equivariant error bounds of the complete U-Net architecture, and proposes an adaptive rotation-equivariant network, AdaReNet. Through a learning-based mask fusion module, AdaReNet automatically determines which regions…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Self-supervised denoising"
  - "rotation equivariance"
  - "equivariant convolution"
  - "U-Net"
  - "adaptive fusion"
date: 2026-05-08
content_hash: 6ed4671ff3525163
---

# Rotation-Equivariant Self-Supervised Method in Image Denoising

**Conference**: CVPR 2025  
**arXiv**: [2505.19618](https://arxiv.org/abs/2505.19618)  
**Code**: [https://github.com/liuhanze623/AdaReNet](https://github.com/liuhanze623/AdaReNet)  
**Area**: Image Restoration  
**Keywords**: Self-supervised denoising, rotation equivariance, equivariant convolution, U-Net, adaptive fusion

## TL;DR

This work introduces rotation-equivariant convolutions to self-supervised image denoising for the first time. It rigorously analyzes the impact of up/downsampling operators on equivariance, provides the equivariant error bounds of the complete U-Net architecture, and proposes an adaptive rotation-equivariant network, AdaReNet. Through a learning-based mask fusion module, AdaReNet automatically determines which regions of an image are better suited for the rotation-equivariant network, achieving consistent performance improvements across three classic self-supervised methods: N2N, N2V, and R2R.

## Background & Motivation

**Background**: Self-supervised image denoising methods (Noise2Noise, Noise2Void, R2R, etc.) have attracted widespread attention because they do not require clean-noisy paired datasets. These methods heavily rely on the interior prior information of deep networks to compensate for the lack of supervision signals. Currently, almost all self-supervised denoising methods are based on CNN architectures, as CNNs naturally capture translation equivariance—one of the most critical priors of images.

**Limitations of Prior Work**: The introduction of translation equivariance has brought huge success to CNNs, but natural images also possess another crucial prior: rotation equivariance. Rotation-equivariant convolutions have demonstrated performance improvements in supervised tasks like super-resolution, but have not yet been introduced to self-supervised denoising, a field that depends even more heavily on network priors. Introducing rotation equivariance to self-supervised denoising faces two key challenges: (1) self-supervised denoising commonly uses the U-Net architecture, where the impact of up/downsampling modules on equivariance lacks theoretical analysis; (2) parameter sharing and kernel parameterization in rotation-equivariant designs reduce representation precision, whereas denoising tasks demand high-frequency detail reconstruction.

**Key Challenge**: While the rotation-equivariant prior provides better inductive bias, there is a contradiction between the rigid constraints of equivariant design and the fact that local regions of natural images do not strictly satisfy rotational symmetry. Applying a rotation-equivariant network indiscriminately to the entire image can degrade reconstruction performance.

**Goal**: (1) Provide rigorous theoretical guarantees for rotation equivariance within the U-Net architecture; (2) design an adaptive framework that allows the network to automatically decide which regions should utilize the rotation-equivariant prior.

**Key Insight**: Starting from the continuous domain, the equivariant errors are derived module-by-module to obtain the error bounds of the complete U-Net. A dual-path fusion architecture is then designed to address the flexibility issues of rigid equivariant constraints.

**Core Idea**: Replace all standard convolutions in U-Net with Fconv (equivariant convolution) to obtain an approximately rotation-equivariant network with theoretical guarantees, and then adaptively fuse the outputs of the equivariant network and standard CNN using a mask-based approach to combine their respective strengths.

## Method

### Overall Architecture

The method operates at two levels: (1) Construction of Equivariant U-Net—replacing all convolutional layers in the self-supervised denoising network with rotation-equivariant convolutions (Fconv) and theoretically proving that the equivariant error of the entire U-Net is bounded; (2) Adaptive Framework AdaReNet—consisting of a Vanilla Module (standard CNN), an EQ Module (equivariant CNN), a Fusion Module (Mask network), and a Self-correcting Module (ResNet blocks), which adaptively fuses the outputs of the two networks using a learned pixel-level mask.

### Key Designs

1. **Down/Upsampling Equivariant Error Analysis (Theorem 1 & 2)**:

    - **Function**: Provides theoretical limits for the equivariant error of the unavoidable up/downsampling operations in U-Net.
    - **Mechanism**: Under the condition that the gradient of the continuous function of the feature map is bounded ($\|\nabla e\| \leq G$), the upper bounds of the equivariant errors for Maxpooling and Stride downsampling, as well as Nearest Neighbor and Bilinear upsampling, are proven to be $O(h)$ (where $h$ is the grid size). This implies that higher image resolution leads to smaller equivariant errors. Although this is slower than the $O(h^2)$ convergence rate of the equivariant convolutional layers themselves, it still tends to zero.
    - **Design Motivation**: Prior works only analyzed the equivariance of convolutional layers, leaving open the critical question of whether up/downsampling in U-Net breaks equivariance. This paper provides the first rigorous proof.

2. **Complete U-Net Equivariant Error Bound (Theorem 3)**:

    - **Function**: Proves that replacing all convolutions in U-Net with E-Conv yields an approximately rotation-equivariant network.
    - **Mechanism**: Decomposes U-Net into multiple downsampling blocks (E-Conv + downsampling operator) and upsampling blocks (upsampling operator + 2 layers of E-Conv). By deriving module-by-module errors and combining them, the equivariant error bound of the complete network is verified to be $\leq R_1 h + R_2 h^2$, where $R_1, R_2$ are constants related to network depth, number of channels, and kernel properties. For any rotation angle $\theta$, the bound increases by an additional term $R_3 t^{-1} h$ (where $t$ is the order of the equivariant subgroup).
    - **Design Motivation**: This theorem provides theoretical support for the strategy of "simply replacing all convolutional layers to obtain an equivariant network," significantly lowering engineering and implementation difficulties.

3. **Adaptive Rotation-Equivariant Network AdaReNet**:

    - **Function**: Automatically determines whether standard CNNs or equivariant networks should be prioritized for different regions of an image.
    - **Mechanism**: AdaReNet comprises four modules: the Vanilla Module $f_c = \text{VM}(I)$ using a standard CNN, the EQ Module $f_e = \text{EQ}(I)$ using a rotation-equivariant CNN, the Fusion Module which learns pixel-level fusion weights $M_f = \text{Mask}(I)$ through a MaskNetwork, and the Self-correcting Module which refines the fused results using ResNet blocks. The final output is formulated as $\bar{I} = S_c(M_f \odot f_c + (1-M_f) \odot f_e)$. Observations show that the weight $M_f$ is larger at texture edges (relying more on the Vanilla module) and smaller in low-frequency regions (relying more on the EQ module), which aligns with the common understanding that standard convolutions excel at fitting high-frequency details.
    - **Design Motivation**: Natural images do not strictly satisfy rotation equivariance everywhere, especially along edges and texture regions. Rigid equivariant constraints can also degrade representation accuracy due to parameter sharing. Adaptive fusion preserves the advantages of equivariant priors in low-frequency regions while safeguarding the reconstruction quality of high-frequency details.

### Loss & Training

- Loss Function: $L = \|\bar{I} - \text{target}\|_2 + \alpha_1 \|f_c - \text{target}\|_2 + \alpha_2 \|f_e - \text{target}\|_2$
- The losses of subnetworks are added to the main loss as regularization terms, where $\alpha_1 = \alpha_2 = 0.1$.
- Training Strategy: Varies depending on the baseline method; N2N is trained with noisy pairs, N2V uses blind-spot networks trained on single noisy images, and R2R generates training pairs from noisy images.

## Key Experimental Results

### Main Results

| Method | Dataset | σ=25 (PSNR/SSIM) | σ=50 (PSNR/SSIM) |
|------|--------|------------------|------------------|
| N2N | Kodak | 31.47/0.874 | 28.29/0.778 |
| N2N-EQ | Kodak | 31.60/0.878 | 28.58/0.790 |
| N2N-EQ⁺ (AdaReNet) | Kodak | **31.72/0.880** | **28.69/0.791** |
| N2V | BSD500 | 28.17/0.820 | 26.07/0.725 |
| N2V-EQ | BSD500 | 29.05/0.834 | 26.38/0.735 |
| N2V-EQ⁺ (AdaReNet) | BSD500 | **29.12/0.845** | **26.82/0.755** |

### Ablation Study

| Configuration | Equivariant Error | Explanation |
|------|---------|------|
| N2V (Original) | 0.233 | Standard CNNs completely lack rotation equivariance |
| N2V-EQ | 0.068 | Equivariant convolutions significantly reduce equivariant error |
| N2V-EQ⁺ (AdaReNet) | 0.076 | Adaptive fusion slightly increases equivariant error, but yields better reconstruction quality |

### Key Findings

- The performance improvement of the rotation-equivariant prior in self-supervised denoising is consistent across N2N, N2V, and R2R.
- The learned mask of AdaReNet reveals an interesting pattern: texture and edge regions tend to utilize standard CNNs, while smooth low-frequency regions lean towards the equivariant network.
- The improvement of N2V-EQ⁺ is particularly significant (from 28.17 to 29.12 dB, a gain of nearly 1 dB), which makes sense since N2V is the method most heavily dependent on network priors (trained using only a single noisy image).
- It is also effective across various noise types, including Poisson and salt-and-pepper noise, demonstrating the generalizability of the rotation-equivariant prior.

## Highlights & Insights

- **Solid Theoretical Contribution**: Rigorous analysis of the impact of up/downsampling on equivariant networks is performed for the first time, providing the equivariant error bounds of the complete U-Net. This theoretical outcome is itself of significant reference value to the equivariant network design community.
- **Adaptive Fusion in AdaReNet**: Leveraging a mask network to let the model decide when to utilize the equivariant prior aligns better with the practical characteristics of images compared to a naive, global equivariant design. This concept can be adapted to other scenarios that require incorporating priors but risk over-constraining the model.
- **Plug-and-Play Design**: The proposed method can be directly applied to existing self-supervised denoising frameworks without modifying the training pipeline; only the network architecture needs to be replaced.

## Limitations & Future Work

- Rotation-equivariant convolutions increase the parameter size and computational cost; the paper does not discuss the efficiency overhead in detail.
- The adaptive framework includes two independent networks (Vanilla + EQ), nearly doubling the parameters, which may not be friendly to resource-constrained scenarios.
- The theoretical analysis assumes that the feature functions are smooth and have bounded gradients, which does not necessarily hold strictly in practical deep networks.
- Only rotation equivariance for the $O(2)$ group was verified. Whether this can be extended to 3D rotations ($SO(3)$) for volumetric data denoising is worth exploring.

## Related Work & Insights

- **vs Fconv (Xie et al.)**: Fconv introduces high-precision rotation-equivariant convolutions for image processing, but only for ResNet architectures and is limited to supervised learning. This work extends it to U-Net architectures and self-supervised learning for the first time.
- **vs Data Augmentation**: Rotational data augmentation is a simple way to introduce rotation invariance but does not guarantee the network's internal equivariance. Equivariant convolutions guarantee equivariance from the architectural level, providing better interpretability and generalization.
- **vs DnCNN/FFDNet**: These supervised methods suffer from a dependency on large-scale paired datasets, while AdaReNet's self-supervised and equivariant design exhibits a clearer advantage in data-scarce scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing rotation equivariance to self-supervised denoising for the first time with rigorous theoretical analysis is a major contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on three baseline methods, multiple noise types, and various datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear motivation, and sound experimental design.
- Value: ⭐⭐⭐⭐ Provides a new perspective for self-supervised denoising; the theoretical results hold generalized significance for equivariant network design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2025\] Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise](generalized_recorrupted-to-recorrupted_self-supervised_learning_beyond_gaussian_.md)
- [\[ECCV 2024\] Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)
- [\[CVPR 2026\] Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising](../../CVPR2026/image_restoration/next-scale_prediction_a_self-supervised_approach_for_real-world_image_denoising.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)

</div>

<!-- RELATED:END -->
