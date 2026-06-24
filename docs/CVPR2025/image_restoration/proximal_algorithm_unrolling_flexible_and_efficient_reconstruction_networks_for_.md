---
title: >-
  [Paper Note] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging
description: >-
  [CVPR 2025][Image Restoration][Single-Pixel Imaging] The authors propose ProxUnroll, which trains HQS/ADMM unrolling networks by designing a proximal trajectory (PT) loss function. This forces the deep image restorer (DIR) within the network to approximate the proximal operator of an ideal regularization, thereby equipping the unrolling network with both the flexibility of PnP algorithms (a single model for arbitrary compression ratios) and the high accuracy and fast speed of…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Single-Pixel Imaging"
  - "Algorithm Unrolling"
  - "Proximal Operator"
  - "Plug-and-Play"
  - "Compressed Sensing Reconstruction"
date: 2026-05-08
content_hash: 2dabf30e11035aa7
---

# Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging

**Conference**: CVPR 2025  
**arXiv**: [2505.23180](https://arxiv.org/abs/2505.23180)  
**Code**: [https://github.com/pwangcs/ProxUnroll](https://github.com/pwangcs/ProxUnroll)  
**Area**: Image Restoration / Compressed Sensing  
**Keywords**: Single-Pixel Imaging, Algorithm Unrolling, Proximal Operator, Plug-and-Play, Compressed Sensing Reconstruction

## TL;DR

The authors propose ProxUnroll, which trains HQS/ADMM unrolling networks by designing a proximal trajectory (PT) loss function. This forces the deep image restorer (DIR) within the network to approximate the proximal operator of an ideal regularization, thereby equipping the unrolling network with both the flexibility of PnP algorithms (a single model for arbitrary compression ratios) and the high accuracy and fast speed of unrolling networks.

## Background & Motivation

**Background**: Single-pixel imaging (SPI) is a compressed sensing technology that recovers images using a single light-sensitive detector at sub-Nyquist sampling rates, offering unique advantages in terahertz imaging and non-visible light 3D imaging. Currently, there are two mainstream types of solvers: Plug-and-Play (PnP) algorithms and deep unrolling networks.

**Limitations of Prior Work**: PnP algorithms reconstruct images by iteratively alternating between a data fidelity proximal operator and a pre-trained denoiser. They offer high flexibility (a single model applies to different compression ratios, CRs) but suffer from mediocre accuracy, slow inference speed, and tedious parameter tuning. Unrolling networks convert a truncated iterative optimization process into an end-to-end trainable network, yielding high accuracy and fast inference, but they must be trained for a specific CR and require retraining when the CR changes.

**Key Challenge**: The trade-off between flexibility and accuracy/speed. The flexibility of PnP stems from the pre-trained denoiser approximating the proximal operator of a regularization independent of the degradation level. Conversely, the high accuracy of unrolling networks arises from end-to-end optimization, but there is a lack of interpretability regarding what the trained subnetworks learn, and progressive improvement of intermediate results is not guaranteed.

**Goal**: To merge the advantages of both approaches—enabling unrolling networks to possess the flexibility and interpretability of PnP while maintaining or even surpassing the accuracy and speed of unrolling networks.

**Key Insight**: The authors observe that the flexibility and interpretability of PnP essentially stem from the deep denoiser approximating the proximal operator of a regularization function. If the neural network in the unrolling network can also be driven to approximate the proximal operator of an ideal regularization, both flexibility and high accuracy can be achieved simultaneously.

**Core Idea**: Define an ideal proximal operator (with an analytical solution) where the regularization function is the distance between the degraded image and the clean image. Use this operator to generate a "proximal optimization trajectory" to supervise the training of the unrolling network, forcing the DIR module within the network to approximate the ideal proximal operator.

## Method

### Overall Architecture

The inputs are the compressed measurements $\mathbf{Y}$ and measurement matrices $(\mathbf{H}, \mathbf{W})$, with the initialization set as $\mathbf{X}^0 = \mathbf{H}^\top \mathbf{Y} \mathbf{W}$. The network consists of $K=6$ iterations, where each iteration performs: (1) an explicit proximal operator $\text{Prox}_f$ (a data fidelity step with a closed-form solution); (2) a deep image restorer (DIR) $\mathcal{R}_\theta$ (an implicit regularization step). The final output is the reconstructed image $\mathbf{X}^K$. During training, the proposed proximal trajectory (PT) loss is used to simultaneously supervise the intermediate outputs of each iteration.

### Key Designs

1. **Proximal Trajectory (PT) Loss Function**:

    - **Function**: Forcing the DIR in the unrolling network to approximate the proximal operator of an ideal regularization, thereby gaining flexibility and interpretability.
    - **Mechanism**: An ideal explicit regularization function $\bar{g}(\mathbf{X}') = \frac{1}{2}\|\mathbf{X}' - \mathbf{X}\|_F^2$ is defined, whose proximal operator has a closed-form solution $\text{Prox}_{\bar{g}}(\mathbf{Q}) = \frac{\mu\mathbf{Q} + \lambda\mathbf{X}}{\mu + \lambda}$, which essentially performs a weighted average between the input and the clean image. During training, for each pair of $(\mathbf{X}^0, \mathbf{X})$, an ideal proximal algorithm is utilized to iteratively generate a "perfect trajectory" $\mathbf{X}^0 \to \bar{\mathbf{X}}^1 \to \cdots \to \mathbf{X}$. The output $\mathbf{X}^{k+1}$ of each stage in the unrolling network is then driven to approach the corresponding $\bar{\mathbf{X}}^{k+1}$ on this trajectory. The loss is computed as $\text{PL} = \sum_{k=0}^{K-1} \alpha_k \|\mathbf{X}^{k+1} - \bar{\mathbf{X}}^{k+1}\|_F^2$.
    - **Design Motivation**: End-to-end training only ensures that the final output is close to the ground truth (GT), failing to guarantee progressive improvement of intermediate results or the interpretability of the learned subnetworks. The PT loss leverages explicit trajectory supervision at each step to force the DIR to approximate the proximal operator, simultaneously yielding convergence guarantees, interpretability, and flexibility.

2. **Deep Image Restorer (DIR) Architecture**:

    - **Function**: Serving as the shared image restoration module across all iterations in the unrolling network.
    - **Mechanism**: A 4-layer asymmetric encoder-decoder architecture utilizing CNN-Transformer hybrid Blocks (CTB). CTB splits the features equally into two halves, processing them in parallel via a Swin Transformer (SwinSA to capture low-frequency global information) and a Gated Dynamic CNN (GD-CNN to capture high-frequency details using AdaConv, which generates input-dependent convolutional kernels), followed by feature fusion. AdaConv implements dynamic convolution by predicting coefficients from the input to perform a linear combination of a set of learned static convolutional kernels.
    - **Design Motivation**: Combining the high-frequency modeling capability of CNNs with the low-frequency/global modeling strength of Transformers, while maintaining high dependency on the input through AdaConv.

3. **Memory Block**:

    - **Function**: Propagating useful feature representations across iterations to prevent information loss.
    - **Mechanism**: Channel cross-attention (ChanCA) is employed at each level of the encoder, using the features of the current iteration as query and the features from the previous iteration as key/value to adaptively aggregate prior useful representations. In the first iteration, key/value are set to zero, and the MB is automatically deactivated.
    - **Design Motivation**: In end-to-end trained unrolling networks, intermediate results are not guaranteed to improve progressively, partly because useful representations cannot propagate across iterations. The MB propagates features via side paths to prevent the degradation of intermediate artifacts.

### Loss & Training

Total Loss = standard pixel loss + PT loss. During training, the CR varies within the range of [0.01, 0.50], and the resolution ranges from 256×256 to 512×512 to cover a wide spectrum of degradation levels. 20,000 training samples are generated using 400 images from BSD500, with an initial learning rate of $1\times10^{-3}$ decaying to $1\times10^{-4}$. The measurement matrix is configured as a row-orthogonal, trainable floating-point matrix.

## Key Experimental Results

### Main Results

Evaluated on the Set11 and CBSD68 datasets, with CR ranging from 0.01 to 0.50:

| Method | Flexibility | CR=0.01 | CR=0.04 | CR=0.10 | CR=0.25 | CR=0.50 | Average PSNR |
|------|--------|---------|---------|---------|---------|---------|----------|
| SAUNet | Single CR | 22.43 | 27.80 | 32.15 | 37.11 | 41.91 | 32.28 |
| HATNet | Single CR | 22.54 | 27.98 | 32.26 | 37.24 | 42.05 | 32.41 |
| PnP-DRUNet | Multi-CR | 21.75 | 26.81 | 30.16 | 35.00 | 40.54 | 30.85 |
| **ADMM-ProxUnroll** | **Arbitrary CR** | **22.76** | **28.30** | **32.55** | **37.35** | **41.97** | **32.59** |

### Ablation Study

| Configuration | Average PSNR | Description |
|------|---------|------|
| Unrolling (SAUNet, Single CR) | 32.28 | Requires separate training for each CR |
| PnP-DRUNet (Multi-CR) | 30.85 | Flexible but poor accuracy |
| ADMM-ProxUnroll (w/o PT loss) | ~31.5 | Removing PT loss results in decreased flexibility |
| ADMM-ProxUnroll (full) | 32.59 | Flexibility + SOTA accuracy |

### Key Findings

- ProxUnroll, using a single model, outperforms SAUNet and HATNet (unrolling networks that require specialized training for each CR) across all CR levels.
- The PT loss ensures progressive improvement of intermediate results (verified through visualization of convergence trajectories), whereas the quality of intermediate results in traditional unrolling networks often fluctuates.
- The Memory Block is critical for information propagation across iterations; removing it leads to a notable performance decline.
- It performs exceptionally well on real SPI data, demonstrating practical application potential.

## Highlights & Insights

- **The Ingenuity of PT Loss**: It utilizes the analytical solution of the ideal regularization proximal operator to generate supervised trajectories without requiring auxiliary data or pre-training, serving as a "free" supervisory signal. This concept can be transferred to any unrolling network framework.
- **Breaking the Dichotomy between PnP and Unrolling**: It proves that flexibility and accuracy are not irreconcilable, and the key lies in endowing the network modules with the mathematical properties of proximal operators.
- **Elegant Design of Dynamic Convolution AdaConv**: By performing input-dependent linear combinations of static convolutional kernels, it achieves adaptive capabilities similar to attention mechanisms with minimal overhead.

## Limitations & Future Work

- Currently, the measurement matrix is assumed to be row-orthogonal, which may not hold perfectly in physical SPI cameras.
- The PT loss requires access to ground truth (GT) images during training to calculate the ideal trajectory. How to design similar mechanisms under semi-supervised or unsupervised settings is worth exploring.
- The DIR architecture is not sufficiently lightweight, leaving margin for optimization in extreme real-time scenarios.
- At present, only SPI reconstruction has been validated. Extending the PT loss to other inverse problems (e.g., CT reconstruction, MRI reconstruction) is a natural future direction.

## Related Work & Insights

- **vs PnP-DRUNet**: PnP methods implicitly approximate the proximal operator using a pre-trained denoiser, which is flexible but limited in accuracy. ProxUnroll explicitly trains the DIR to approximate the proximal operator via the PT loss, significantly surpassing its accuracy.
- **vs SAUNet/HATNet**: Traditional unrolling networks offer high accuracy but require separate training for each CR. ProxUnroll outperforms their dedicated models for each CR using only a single model.
- **vs Gradient Denoisers (GS-DRUNet, etc.)**: Prior proximal learning methods constrained networks to satisfy non-expansiveness or input convexity, sacrificing representational capacity. PT loss is free of such constraints, indirectly achieving the properties of proximal operators through trajectory supervision.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of PT loss is highly ingenious, though the overall framework still follows the paradigm of unrolling networks and well-designed subnetworks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluations covering simulation and real data, multiple CRs, various baselines, and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations with well-articulated motivations.
- **Value**: ⭐⭐⭐⭐ Makes a significant contribution to the SPI domain, with the potential of extending the PT loss concept to broader inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Bit is All You Need! Efficient Video Capture via Single Bit Imaging](../../CVPR2026/image_restoration/a_bit_is_all_you_need_efficient_video_capture_via_single_bit_imaging.md)
- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](../../ECCV2024/image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[ECCV 2024\] Intrinsic Single-Image HDR Reconstruction](../../ECCV2024/image_restoration/intrinsic_single-image_hdr_reconstruction.md)
- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)
- [\[CVPR 2025\] Gyro-based Neural Single Image Deblurring](gyro-based_neural_single_image_deblurring.md)

</div>

<!-- RELATED:END -->
