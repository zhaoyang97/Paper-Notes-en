---
title: >-
  [Paper Note] Bias for Action: Video Implicit Neural Representations with Bias Modulation
description: >-
  [CVPR 2025][Image Generation][Implicit Neural Representation] Proposes ActINR, which achieves continuous video representation by sharing weights across frames and modeling motion solely through biases in INR. Under 10× slow-motion, 4× spatial + 2× temporal super-resolution, video denoising, and inpainting tasks, it significantly outperforms existing methods (with average improvements of 3-6dB).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Implicit Neural Representation"
  - "Video Modeling"
  - "Bias Modulation"
  - "Slow-Motion Generation"
  - "Video Inpainting"
date: 2026-05-08
content_hash: 9d1abf24b13994ca
---

# Bias for Action: Video Implicit Neural Representations with Bias Modulation

**Conference**: CVPR 2025  
**arXiv**: [2501.09277](https://arxiv.org/abs/2501.09277)  
**Code**: None (not mentioned)  
**Area**: Image Generation  
**Keywords**: Implicit Neural Representation, Video Modeling, Bias Modulation, Slow-Motion Generation, Video Inpainting

## TL;DR
Proposes ActINR, which achieves continuous video representation by sharing weights across frames and modeling motion solely through biases in INR. Under 10× slow-motion, 4× spatial + 2× temporal super-resolution, video denoising, and inpainting tasks, it significantly outperforms existing methods (with average improvements of 3-6dB).

## Background & Motivation

**Background**: Implicit Neural Representations (INRs) fit continuous signals using MLPs and have been widely applied to image, 3D, and video representations. Representative video INR works, such as the NeRV family, generate frames from frame indices using convolutional decoders, but these methods perform poorly in handling large motions, especially in extreme frame interpolation tasks.

**Limitations of Prior Work**: Existing video INR methods each have their own limitations. FF-NeRV relies on optical flow to estimate motion, which is inaccurate under large motions; H-NeRV Boost modulates feature maps via affine transformations, leading to ghosting artifacts during frame interpolation; ResField replaces temporal coordinates with residual weight matrices, making the parameter space too large and difficult to optimize. A more fundamental issue is that NeRV-like methods are based on convolutional decoding and cannot query arbitrary spatial coordinates, thus they do not support spatial super-resolution.

**Key Challenge**: Motion in videos is essentially the translation of signal local basis functions. However, existing methods model temporal changes either in an overly simplistic manner (single phase shifts) or an over-parameterized manner (residual weight matrices), failing to find an accurate and efficient middle ground for motion modeling.

**Goal**: How to design a continuous video representation that can accurately model both local and global motion while supporting spatio-temporal super-resolution, denoising, and inpainting.

**Key Insight**: The authors start from the perspective of basis function expansion in INR—INR can be viewed as a learnable dictionary, where weights determine the shape and scale of the basis functions, and biases control the positions of the basis functions. For compactly supported activation functions (such as wavelets or Gaussians), local motion is simply the shift of basis function positions, i.e., changes in bias values. Therefore, sharing weights across frames (keeping appearance constant) while only varying biases (modeling motion) is a natural and compact choice.

**Core Idea**: Bind INR biases to motion, achieving compact and precise continuous video representation through shared weights across frames + frame-specific biases (predicted by a temporally continuous bias-INR).

## Method

### Overall Architecture
ActINR consists of two networks. **Frame INR** takes spatial coordinates $(x,y)$ as input and outputs the RGB value of the corresponding pixel. Its weights are shared across all frames of the video, while its biases vary across frames. **Bias-INR** takes continuous time index $t$ as input and outputs the bias vectors required by each layer of the Frame INR, ensuring that biases change smoothly over time to support frame interpolation. The entire video is partitioned into spatial patches of equal size (96×96 pixels, grouped in 10 frames), and each patch is fitted independently with a small INR to achieve spatial divide-and-conquer acceleration.

### Key Designs

1. **Bias-Motion Interplay**

    - **Function**: Modeling local motion in videos through changes in INR bias parameters.
    - **Mechanism**: For compactly supported activation functions (such as WIRE wavelet activation), INR can be viewed as a basis function expansion: the weight $W$ controls the shape and scale of the basis functions, and the bias $b$ controls their spatial centers. When motion occurs in a local region of the scene, the corresponding basis functions only need to shift their positions (changing the bias) without altering their shapes (changing the weights). Therefore, the structure of Frame INR is $y_i^{(l)} = \sigma(W^{(l)} y_i^{(l-1)} + b_i^{(l)})$, where $W^{(l)}$ is shared across frames, and $b_i^{(l)}$ is frame-specific.
    - **Design Motivation**: Compact support implies that basis functions only affect local regions without interfering with each other across different regions, enabling local motion modeling. This is verified by a toy experiment: given two Gaussian blobs where the left one moves to the right, the corresponding basis function #1 shifts its position solely via changing bias values, while basis function #2 remains stationary.

2. **Bias-INR Continuous Bias Prediction**

    - **Function**: Modeling frame-specific biases as a temporally continuous function to support frame interpolation at arbitrary time steps.
    - **Mechanism**: Another MLP (with GeLU activation) is used as a hypernetwork $\psi$. It takes the random Fourier features $\gamma(t) = [\sin 2\pi B t, \cos 2\pi B t]^\top$ of continuous time index $t$ and a patch-level learnable latent vector $z$ as input, and outputs the bias vectors for each layer of the Frame INR. $z$ encodes the stationarity of each patch, allowing the shared bias-INR to adapt to different patches and avoiding the need to train a separate bias-INR for each patch. The key point is the unified optimization of bias-INR during both training and inference, avoiding the training/testing inconsistency of linear interpolation schemes (where linear interpolation testing PSNR drops heavily by 24dB in ablation studies).
    - **Design Motivation**: Independently optimizing the bias of each frame cannot guarantee temporal smoothness and makes interpolation on unseen frames impossible during testing. Bias-INR constrains the biases on a continuous manifold while providing implicit regularization.

3. **WIRE Activation Function + Spatial Partitioning**

    - **Function**: Providing compactly supported and highly expressive basis functions, cooperating with spatial partitioning for efficient local motion modeling.
    - **Mechanism**: WIRE (Wavelet Implicit Representation) is used as the activation function, featuring both compact support (locality) and oscillation (high expressiveness), which is superior to SIREN (no compact support, global interference) and Gauss (no oscillation, weak expressiveness). The video is divided into patches of 96×96 pixels, and each patch is modeled by a 3-layer MLP (hidden dimension 36), totaling about 3 million parameters. Overlapping windows + bilinear blending are applied between patches to eliminate boundary artifacts.
    - **Design Motivation**: The lack of compact support in SIREN's basis functions causes motion in distant areas to interfere with stationary regions (producing artifacts in the background in ablation studies). Compactly supported activation ensures local motion only affects local areas. The partitioning strategy refers to KiloNeRF to reduce the area each INR needs to model.

### Loss & Training
A simple MSE loss is used to minimize the mean squared error between the RGB values predicted by the Frame INR and the ground truth frames. Adam optimizer is employed with a learning rate of $5 \times 10^{-3}$ and step decay (decay ratio 0.1), training each MLP for 2000 iterations. The weights of the next set are initialized with those of the previous set to accelerate convergence.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | Ours | Prev. SOTA | Gain |
|-------------|------|--------|-------------|------|
| 2× Interpolation / DAVIS | PSNR/SSIM | **22.9/0.69** | 22.2/0.65 (H-NeRV Boost) | +0.7dB |
| 2× Interpolation / UVG | PSNR/SSIM | **31.0/0.90** | 30.6/0.90 (H-NeRV Boost) | +0.4dB |
| Video Denoising / DAVIS | PSNR/SSIM | **29.0/0.88** | 25.5/0.78 (H-NeRV Boost) | +3.5dB |
| Video Inpainting / DAVIS | PSNR | **34.7** (avg 9 videos) | 33.1 (H-NeRV Boost) | +1.6dB |
| Spatio-Temporal SR (4× Spatial + 2× Temporal) / UVG | PSNR | **~5.7dB better than compared** | H-NeRV Boost | +5.7dB |

### Ablation Study

| Configuration | Train PSNR | Test PSNR | Description |
|------|---------|---------|------|
| Oracle (independent bias for all frames) | 46.3 | 46.3 | Upper bound |
| Bias-INR (Ours) | 46.0 | **45.8** | Close to oracle, good test generalization |
| Linearly interpolated bias | 44.5 | 20.2 | Train/test inconsistency, drops by 24dB |

### Key Findings
- The largest improvement is observed in the denoising task (+3.5dB), indicating that the continuity constraint of bias-INR provides a strong implicit regularization, effectively discarding noise.
- Under extreme 10× interpolation, the advantage is even more pronounced (more than 5dB higher than compared methods), as optical flow methods fail severely under large frame intervals.
- WIRE activation significantly outperforms SIREN: SIREN's non-compact support leads to artifacts in the stationary background (due to basis function interference), Gauss is second, and WIRE is the best.
- NeRV-like methods cannot perform spatial super-resolution (since they use convolutional decoders instead of coordinate queries); this paper is the first to demonstrate this limitation.
- There is an optimal patch size: too large and the single INR capacity is insufficient, too small and objects easily cross boundaries.

## Highlights & Insights
- The **insights of Bias = Motion** is extremely elegant: mapping the mathematical structure of INR (basis function expansion) to physical intuition (motion = displacement). The observation that bias controls basis function position is simple yet highly inspiring, and can be transferred to any INR application requiring physical local changes in signals.
- **Denoising requires no extra design**: the continuity prior of bias-INR alone achieves outstanding denoising results on noisy data, indicating that a good representation is itself the best prior.
- **Unique capability of handling both spatial and temporal super-resolution**: it retains the capability of INR to query arbitrary coordinates while achieving the high efficiency of NeRV, which was unattainable by prior methods.

## Limitations & Future Work
- Assuming motion is confined within patches, reconstruction quality degrades when objects cross patch boundaries (although an overlapping window solution is proposed, it increases computational cost).
- Encoding takes a long time (about 5 hours per video), making it unsuitable for real-time applications.
- Compression performance is slightly inferior to H-NeRV; its primary advantage lies in inverse problems (interpolation/denoising/inpainting) rather than compression.
- No comparison is made with diffusion-based video interpolation/super-resolution methods, which might achieve better perceptual quality.

## Related Work & Insights
- **vs FF-NeRV**: FF-NeRV models inter-frame motion with optical flow, which fails under large motions. ActINR directly models motion via biases without explicit optical flow, exhibiting a significant advantage in large-displacement scenarios.
- **vs H-NeRV Boost**: H-NeRV modulates feature maps using affine transformations, but the inductive bias of convolutional decoders leads to detail smoothing and ghosting. ActINR has no such inductive bias and can query arbitrary coordinates.
- **vs Phase-INR**: Phase-INR only injects temporal phase shifts in coordinates inside the positional encoding layer, which is too simplistic. ActINR models motion via biases in all layers, offering stronger expressiveness.
- **vs ResField**: ResField models temporal variations using residual weight matrices, resulting in an overly large parameter space. ActINR only changes biases (which are far fewer than weights), being more efficient and stable during optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight pointing to the mapping of bias-motion is highly original, with clear theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four major tasks + multiple datasets + comprehensive ablations, but lacks comparison with diffusion-model-based methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Progresses layer by layer from intuition to theory to experiments, with the toy experiment being extremely helpful for understanding.
- Value: ⭐⭐⭐⭐ Opens up a new paradigm for video INR, and the +3-5dB improvement in denoising/inpainting offers practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](../../ICLR2026/image_generation/cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[AAAI 2026\] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)

</div>

<!-- RELATED:END -->
