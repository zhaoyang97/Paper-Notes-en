---
title: >-
  [Paper Note] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts
description: >-
  [ICCV 2025][Autonomous Driving][Low-light enhancement] This paper is the first to introduce Mixture-of-Experts (MoE) networks into low-light image enhancement (LLIE), employing three specialized sub-expert networks to handle color restoration, detail enhancement, and high-level feature enhancement respectively. A dynamic gating mechanism adaptively adjusts the contribution of each expert, achieving state-of-the-art PSNR performance on five benchmark datasets.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Low-light enhancement
  - Mixture-of-Experts
  - gating mechanism
  - U-Net
  - multi-scale feature fusion
date: 2026-05-08
content_hash: 2e638a5efd8921d6
---

# GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts

**Conference**: ICCV 2025
**arXiv**: [2503.07417](https://arxiv.org/abs/2503.07417)
**Code**: [https://github.com/Sameenok/gm-moe-lowlight-enhancement.git](https://github.com/Sameenok/gm-moe-lowlight-enhancement.git)
**Area**: Autonomous Driving
**Keywords**: Low-light enhancement, Mixture-of-Experts, gating mechanism, U-Net, multi-scale feature fusion

## TL;DR

This paper is the first to introduce Mixture-of-Experts (MoE) networks into low-light image enhancement (LLIE), employing three specialized sub-expert networks to handle color restoration, detail enhancement, and high-level feature enhancement respectively. A dynamic gating mechanism adaptively adjusts the contribution of each expert, achieving state-of-the-art PSNR performance on five benchmark datasets.

## Background & Motivation

Low-light image enhancement (LLIE) has broad applications in autonomous driving, 3D reconstruction, remote sensing, and surveillance. Existing methods suffer from three major limitations:

**Global–local information imbalance**: CNN-based methods struggle to learn global illumination distributions, while Transformers over-emphasize global information, leading to color distortion.

**Insufficient cross-domain generalization**: Existing methods are typically trained on specific datasets and experience significant performance degradation under unseen illumination conditions.

**Difficulty in joint optimization of coupled degradations**: Noise, color distortion, and detail blurring are mutually coupled, making it hard for a single model to address them jointly — suppressing noise may sacrifice fine details, while brightening dark regions may amplify color distortion.

## Method

### Overall Architecture

GM-MoE is built upon an improved U-Net architecture. A low-light input image $I \in \mathbb{R}^{H \times W \times 3}$ first passes through a Shallow Feature Extraction Block (SFEB) to obtain low-level features $X_0$. The encoder progressively downsamples to extract deep features, while the decoder upsamples via pixel-shuffle to restore resolution. GM-MoE modules are embedded at each level of both the encoder and decoder, responsible for fusing low-level encoder features with high-level decoder features. The final output is a residual image $R$, and the enhanced image is $\hat{I} = I + R$.

### Key Designs

1. **Dynamic Gating Weight Generation Network**: The input image is transformed into a feature vector via adaptive average pooling, then passed through a two-layer fully connected network to produce weights $S = [s_1, s_2, s_3]$ for the three expert networks, where $s_1 + s_2 + s_3 = 1$. This enables the network to dynamically adjust parameters according to images from different data domains (i.e., varying scene and illumination characteristics). The final output is the weighted sum: $\tilde{X}_i = s_1 X_{i-1}^1 + s_2 X_{i-1}^2 + s_3 X_{i-1}^3$.

2. **Color Restoration Expert Network (Expert1/Net1)**: Pooling operations are applied to focus on key color features; deconvolution is used to recover image details; and nonlinear interpolation ensures smooth, natural color transitions. Residual connections preserve original image features, and a Sigmoid activation constrains the output to $[0,1]$, reducing color artifacts and oversaturation.

3. **Detail Enhancement Expert Network (Expert2/Net2)**: Channel attention and spatial attention mechanisms are combined. Channel attention extracts salient channel features, while spatial attention leverages both max pooling and average pooling to focus on key spatial locations. The outputs of both attention modules are fused via concatenation and residual connections to improve detail recovery.

4. **High-Level Feature Enhancement Expert Network (Expert3/Net3)**: Multi-scale convolutions are used to extract and fuse features, which are then processed by a gating network (SG) and a channel attention mechanism (SCA). The result is added back to the input via a residual connection to improve overall image quality.

5. **Shallow Feature Extraction Block (SFEB)**: $3 \times 3$ depthwise separable convolutions produce $F_1$, and dilated convolutions with varying dilation rates produce $F_2$ to capture multi-scale spatial information. Channel-weighted features $A_{avg}$ and $A_{max}$ are generated via global pooling, and a $7 \times 7$ convolution produces an attention map: $F_w = F_1' \odot A_{avg} + F_2' \odot A_{max}$, with final output $Y = X \odot F_w$.

### Loss & Training

PSNR Loss is adopted as the training objective, defined as:

$$\text{PSNR loss} = -\frac{10}{\log(10)} \cdot \log(\text{MSE} + \epsilon)$$

where $\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{I}(i) - I_{gt}(i))^2$ and $\epsilon$ is a small positive constant to prevent division by zero.

Training details: PyTorch framework, NVIDIA 4090 GPU, initial learning rate $1.0 \times 10^{-3}$, Adam optimizer (momentum=0.9), inputs resized to $256 \times 256$, batch size=4, total $2.0 \times 10^6$ iterations.

## Key Experimental Results

### Main Results

Comparison against 25+ methods on LOL-v1, LOLv2-Real, and LOLv2-Synthetic:

| Method | LOL-v1 PSNR | LOL-v1 SSIM | LOLv2-Real PSNR | LOLv2-Real SSIM | LOLv2-Syn PSNR | LOLv2-Syn SSIM | Params (M) |
|--------|-------------|-------------|-----------------|-----------------|----------------|----------------|------------|
| Retinexformer | 25.16 | 0.845 | 22.80 | 0.840 | 25.67 | 0.930 | 1.61 |
| DPEC | 24.80 | 0.855 | 22.89 | 0.863 | 26.19 | 0.939 | 2.58 |
| LLFormer | 25.76 | 0.823 | 20.06 | 0.792 | 24.04 | 0.909 | 24.55 |
| SNR-Net | 24.61 | 0.842 | 21.48 | 0.849 | 24.14 | 0.928 | 39.12 |
| **GM-MoE (Ours)** | **26.66** | **0.857** | **23.65** | 0.806 | **26.30** | **0.937** | 19.99 |

Results on LSRW-Huawei/Nikon datasets:

| Method | LSRW-Huawei PSNR | LSRW-Huawei SSIM | LSRW-Nikon PSNR | LSRW-Nikon SSIM |
|--------|-------------------|-------------------|------------------|------------------|
| Restormer | 22.61 | 0.725 | 21.20 | 0.677 |
| DRBN | 20.61 | 0.710 | 21.07 | 0.670 |
| **GM-MoE (Ours)** | **23.55** | **0.741** | **22.62** | **0.700** |

### Ablation Study

Incremental module addition on LOLv2-Real and LOLv2-Synthetic:

| Configuration | LOLv2-Real PSNR | LOLv2-Real SSIM | LOLv2-Syn PSNR | LOLv2-Syn SSIM |
|---------------|-----------------|-----------------|----------------|----------------|
| Baseline | 19.45 | 0.7079 | 20.35 | 0.7431 |
| +SFEB | 20.27 | 0.7236 | 23.44 | 0.7646 |
| +SFEB+Net1 | 21.35 | 0.7446 | 24.35 | 0.8436 |
| +SFEB+Net1+Net2 | 22.11 | 0.8021 | 25.14 | 0.9327 |
| +SFEB+Net1+Net2+Net3 | 23.35 | 0.8055 | 26.15 | 0.9366 |
| **Full Model (+GM)** | **23.65** | **0.8060** | **26.29** | **0.9371** |

### Key Findings

- SFEB alone yields a 3.09 dB PSNR gain on LOLv2-Syn, highlighting the importance of shallow feature extraction.
- The three expert networks provide complementary contributions; removing any one leads to performance degradation.
- The gating mechanism contributes an additional ~0.3 dB improvement in the full model, validating the effectiveness of dynamic weight adjustment for cross-domain generalization.
- On the high-noise LSRW dataset, GM-MoE surpasses Restormer by 0.94 dB and 1.42 dB respectively, demonstrating its advantage under heavy noise.

## Highlights & Insights

- **First application of MoE to LLIE**: Decomposing the multiple sub-problems of low-light enhancement (color restoration, detail recovery, feature enhancement) into independent experts is a natural and effective design choice.
- The **dynamic gating mechanism** enables the model to adaptively adjust across data domains, avoiding the suboptimal solutions imposed by fixed weights.
- Achieves top PSNR on all 5 benchmarks and top SSIM on 4, demonstrating strong generalization.
- With 19.99M parameters, the model strikes a balance between lightweight and heavyweight designs.

## Limitations & Future Work

- SSIM on LOLv2-Real (0.806) is notably lower than DPEC (0.863) and SNR-Net (0.849), indicating room for improvement in structural preservation.
- The gating mechanism relies solely on Softmax to generate three scalar weights, lacking spatial adaptivity at the pixel or region level.
- Training with PSNR Loss alone, without perceptual loss, SSIM loss, or adversarial loss, limits the upper bound of perceptual quality.
- The method has not been validated in video or real-time settings; further verification of inference latency is needed for practical autonomous driving deployment.

## Related Work & Insights

- Compared to lightweight models such as Retinexformer (1.61M) and DPEC (2.58M), GM-MoE has a larger parameter count but achieves superior performance.
- The MoE paradigm can be extended to other image restoration tasks (dehazing, deraining, super-resolution) by assigning different degradation types to different experts.
- The gating mechanism design could draw inspiration from Sparse MoE approaches (e.g., Switch Transformer), activating only a subset of experts to reduce computational cost.

## Rating

- **Novelty**: ⭐⭐⭐ — Introducing MoE to LLIE is a notable contribution, though the individual expert designs are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five datasets, 25+ competing methods, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐ — Structure is clear, but some formulations and descriptions are redundant.
- **Value**: ⭐⭐⭐⭐ — Demonstrates the potential of MoE for low-level vision tasks with convincing experimental results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](../../AAAI2026/autonomous_driving/expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [\[ICCV 2025\] OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving](od-rase_ontology-driven_risk_assessment_and_safety_enhancement_for_autonomous_dr.md)
- [\[ICCV 2025\] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)

</div>

<!-- RELATED:END -->
