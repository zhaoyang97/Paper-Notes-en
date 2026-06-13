---
title: >-
  [Paper Note] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking
description: >-
  [AAAI 2026][HDR reconstruction] This paper proposes a semi-supervised HDR reconstruction framework that evaluates pseudo HDR label quality via an **uncertainty estimation branch**…
tags:
  - "AAAI 2026"
  - "HDR reconstruction"
  - "semi-supervised learning"
  - "uncertainty estimation"
  - "pseudo labels"
  - "bi-level masking"
date: 2026-05-08
content_hash: 1272cd381d5792ec
---

# Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking

**Conference**: AAAI 2026
**arXiv**: [2511.12939](https://arxiv.org/abs/2511.12939)  
**Code**: [https://github.com/JW20211/SmartHDR](https://github.com/JW20211/SmartHDR)  
**Area**: Computational Photography / Semi-Supervised Learning
**Keywords**: HDR reconstruction, semi-supervised learning, uncertainty estimation, pseudo labels, bi-level masking

## TL;DR

This paper proposes a semi-supervised HDR reconstruction framework that evaluates pseudo HDR label quality via an **uncertainty estimation branch**, masking unreliable regions at both the patch and pixel levels. Using only 6.7% of HDR ground-truth annotations, the method achieves performance comparable to fully supervised state-of-the-art.

## Background & Motivation

1. **Background**: Reconstructing ghost-free HDR images from multi-exposure LDR image sets is a key task in computational photography. Learning-based methods (e.g., GFHDR, SAFNet) have achieved notable progress but require paired LDR–HDR data.
2. **Limitations of Prior Work**: High-quality HDR ground truth requires expensive professional equipment or strictly controlled scene motion, making large-scale collection difficult. Annotation-efficient methods such as FSHDR suffer from domain gaps due to synthetic LDR generation; SMAE's adaptive pseudo-label selection relies on reference image similarity and ignores saturated regions, and the teacher–student gap is insufficient.
3. **Key Challenge**: Pseudo labels inevitably contain ghosting or noise artifacts. If the student learns from these errors, **confirmation bias** arises; yet aggressively discarding pseudo labels leads to insufficient training data.
4. **Goal**: How to train high-quality HDR reconstruction models with limited HDR ground truth? How to effectively identify reliable regions within pseudo HDR labels?
5. **Key Insight**: An uncertainty estimation branch is introduced that models predictions as Gaussian distributions and ground truth as Dirac delta functions. Per-pixel uncertainty scores are learned via KL divergence and used to filter unreliable regions at two levels of granularity.
6. **Core Idea**: A learnable uncertainty map evaluates per-pixel reliability of pseudo HDR labels, enabling unreliable regions to be masked at both the patch and pixel levels.

## Method

### Overall Architecture

A teacher–student pseudo-label paradigm is adopted. The student is updated via gradient descent; the teacher is updated via EMA ($\alpha=0.999$). The teacher generates pseudo HDR labels along with corresponding uncertainty maps. Based on these maps, unreliable regions are masked at the patch and pixel levels, and the student learns only from trustworthy regions. Training proceeds in two stages: 30 warm-up epochs (labeled data only) followed by 170 semi-supervised epochs.

### Key Designs

1. **Uncertainty Estimation Branch (Judge Network)**

    - **Function**: Generates a per-pixel reliability score (uncertainty map) for the predicted HDR image.
    - **Mechanism**: Three 3×3 convolution layers with skip connections and sigmoid activation are appended to the spatial attention module output features $F_{att}$ of GFHDR, predicting uncertainty map $\sigma$. The loss is based on the KL divergence under a Gaussian assumption: $\mathcal{L}^k = \frac{1}{n_h} \sum_i \frac{(\bar{h}_i - h_i)^2}{2\sigma_i^2} + \frac{1}{2}\log(\sigma_i^2)$. When the prediction $\bar{h}$ is inaccurate, the network learns to predict a larger $\sigma$ to reduce the loss, thereby automatically identifying low-quality regions.
    - **Design Motivation**: HDR reconstruction is a dense regression task with no confidence scores analogous to those in classification for evaluating pseudo-label quality. Modeling predictions as Gaussian distributions and learning the variance as uncertainty is a natural and effective solution.

2. **Bi-Level Uncertain Area Masking**

    - **Function**: Filters unreliable regions in pseudo labels at patch and pixel granularities, ensuring the student learns only from trustworthy signals.
    - **Mechanism**: **Patch level**: The mean uncertainty of each 64×64 patch is computed and globally normalized to obtain $S_{pa}$; a binary mask $M_{pa}$ is generated via threshold $\tau_{pa}=0.4$, discarding entire high-uncertainty patches. **Pixel level**: For pixels within retained patches, the single-channel uncertainty map is normalized and a pixel-level mask $M_{pi}$ is generated via threshold $\tau_{pi}=0.4$ to precisely filter residual unreliable pixels.
    - **Design Motivation**: Patch-level masking alone may miss localized small-region artifacts; pixel-level masking alone is computationally inefficient and may retain patches containing large-area artifacts. The two-level combination balances efficiency and precision.

3. **Data Augmentation Strategy**

    - **Function**: Enlarges the teacher–student gap to provide better learning signals.
    - **Mechanism**: Inspired by FixMatch, unlabeled data undergoes strong augmentation (random RGB channel shuffling + horizontal flip + 90° rotation), while labeled data uses weak augmentation (vertical flip + 90° rotation). Pseudo labels for strongly augmented unlabeled data are generated by the weakly augmented teacher.
    - **Design Motivation**: An insufficient teacher–student gap (as in SMAE) prevents the student from learning new knowledge. FixMatch has demonstrated that consistency learning with an appropriate gap combined with pseudo labels is highly effective.

### Loss & Training

$\mathcal{L} = \mathcal{L}_s^r + \lambda_v \mathcal{L}_s^v + \mathcal{L}_s^k + \lambda_u(\mathcal{L}_u^r + \lambda_v \mathcal{L}_u^v + \mathcal{L}_u^k)$, where $\mathcal{L}^r$ is the tone-mapped L1 reconstruction loss, $\mathcal{L}^v$ is the VGG perceptual loss, and $\mathcal{L}^k$ is the uncertainty loss. $\lambda_u=1$. Tone mapping uses $\mu$-law ($\mu=5000$). Adam optimizer, learning rate $2 \times 10^{-4}$, trained for 200 epochs, using only $N^l=5$ labeled samples.

## Key Experimental Results

### Main Results

PSNR results on the Kalantari and Hu datasets (semi-supervised vs. fully supervised):

| Method | Annotation | Kalantari PSNR-μ | Kalantari PSNR-l | Hu PSNR-μ | Hu PSNR-l |
|--------|------------|-----------------|-----------------|----------|----------|
| SAFNet | 100% GT | 44.66 | 43.18 | - | - |
| GFHDR | 100% GT | 44.32 | 42.18 | - | - |
| FSHDR | 6.7% GT | 41.94 | 40.80 | 43.98 | 47.13 |
| SMAE | 6.7% GT | 41.61 | 41.54 | 44.24 | 47.41 |
| **Ours** | **6.7% GT** | **44.04** | **41.67** | **45.10** | **47.93** |

### Ablation Study

| Configuration | PSNR-μ | Change | Note |
|---------------|--------|--------|------|
| Full model | 44.04 | — | Complete model |
| w/o patch masking | decrease | −significant | Patch-level filtering is important |
| w/o pixel masking | decrease | −moderate | Pixel-level fine filtering is beneficial |
| w/o uncertainty | decrease | −largest | No filtering leads to confirmation bias |
| w/o strong augmentation | decrease | −moderate | Insufficient teacher–student gap |
| Labeled data only | ~42 | −notable | Validates effectiveness of semi-supervision |

### Key Findings

- Using only 6.7% HDR ground truth (5 scenes), PSNR-μ reaches 44.04, approaching fully supervised SOTA SAFNet (44.66) and far surpassing the prior semi-supervised method SMAE (41.61).
- Uncertainty-driven masking is the key contributor to performance gains—removal causes a substantial drop.
- Patch-level masking contributes more than pixel-level masking, as artifacts tend to be distributed in block patterns.
- The EMA teacher is significantly more stable than a directly copied teacher; $\alpha=0.999$ yields the best results.
- Bi-level thresholds $\tau_{pa}=\tau_{pi}=0.4$ achieve the best balance between accuracy and data utilization.

## Highlights & Insights

- **Elegant combination of uncertainty estimation and pseudo-label filtering**: Modeling regression predictions as Gaussian distributions and learning variance as uncertainty is directly transferable to other dense regression semi-supervised tasks (depth estimation, optical flow, etc.).
- **Positive feedback loop**: a better student → a better EMA teacher → more reliable pseudo labels → more accurate uncertainty estimates → previously uncertain regions can be unlocked.
- **Extreme annotation efficiency**: Only 5 samples suffice to approach fully supervised performance, which has significant practical implications for HDR data acquisition.

## Limitations & Future Work

- Only the 6.7% annotation ratio (5/75) is evaluated; the performance curve under varying annotation quantities is not systematically studied.
- Uncertainty thresholds $\tau_{pa}, \tau_{pi}$ are fixed; adaptive thresholding strategies may yield better results.
- The method is built on the GFHDR backbone; effectiveness on more recent backbones (e.g., SAFNet, diffusion models) remains unverified.
- Pseudo labels are updated once per epoch; more frequent updates may improve quality at the cost of increased computational overhead.

## Related Work & Insights

- **vs. FSHDR**: FSHDR suffers from domain gaps due to synthetic LDR generation, causing artifacts; this work directly uses the EMA teacher to generate pseudo HDR labels, avoiding domain gap issues.
- **vs. SMAE**: SMAE's pseudo-label selection is based on similarity to a reference image (ignoring saturated regions); the uncertainty estimation proposed here is more general and adaptive.
- **vs. FixMatch/Mean Teacher**: Successfully transfers the semi-supervised classification paradigm to the HDR dense regression task.

## Rating

- Novelty: ⭐⭐⭐⭐ Uncertainty-driven bi-level masking is applied to HDR semi-supervised learning for the first time
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two standard datasets; ablation over annotation ratios is insufficient
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; pipeline diagrams are intuitive
- Value: ⭐⭐⭐⭐ Substantially reduces HDR data annotation requirements; has practical deployment value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](../../CVPR2026/others/polishing_the_sky_widefield_and_highdynamic_range.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->
