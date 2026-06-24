---
title: >-
  [Paper Note] MambaIR: A Simple Baseline for Image Restoration with State-Space Model
description: >-
  [ECCV 2024][Image Restoration][State-Space Model] This paper introduces Mamba (Selective State-Space Model) to low-level image restoration tasks for the first time. By designing local convolution enhancement and channel attention mechanisms within the Residual State-Space Block (RSSB), the proposed method addresses the issues of local pixel forgetting and channel redundancy in vanilla Mamba on 2D images. It achieves comparable or even superior performance to Transformer-based…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "State-Space Model"
  - "Mamba"
  - "Super-Resolution"
  - "Image Denoising"
date: 2026-05-08
content_hash: b6e6753da458839a
---

# MambaIR: A Simple Baseline for Image Restoration with State-Space Model

**Conference**: ECCV 2024  
**arXiv**: [2402.15648](https://arxiv.org/abs/2402.15648)  
**Code**: [https://github.com/csguoh/MambaIR](https://github.com/csguoh/MambaIR)  
**Area**: Image Restoration  
**Keywords**: Image Restoration, State-Space Model, Mamba, Super-Resolution, Image Denoising

## TL;DR

This paper introduces Mamba (Selective State-Space Model) to low-level image restoration tasks for the first time. By designing local convolution enhancement and channel attention mechanisms within the Residual State-Space Block (RSSB), the proposed method addresses the issues of local pixel forgetting and channel redundancy in vanilla Mamba on 2D images. It achieves comparable or even superior performance to Transformer-based methods with linear complexity on image super-resolution and denoising tasks (outperforming SwinIR by 0.45dB on SR).

## Background & Motivation

A long-standing core contradiction exists in the field of image restoration: **global receptive field vs. computational efficiency**. CNN methods (e.g., EDSR, RCAN) are computationally efficient but have limited receptive fields. Standard Transformer methods (e.g., SwinIR, HAT) can model global dependencies, but the quadratic complexity of self-attention makes the computational cost extremely high at a full-image scale. Even when utilizing efficient variants such as window attention, they only represent a compromise between global modeling and computational efficiency.

Mamba, as an improved selective structured state-space model, possesses long-range dependency modeling capabilities with linear complexity, making it naturally suited to address this dilemma. However, standard Mamba is designed for 1D NLP sequences, and directly applying it to 2D image restoration faces two main challenges:

**Local pixel forgetting**: After flattening 2D images into 1D sequences, spatially adjacent pixels might be far apart in the sequence, making them prone to being forgotten during recursive processing.

**Channel redundancy**: To memorize long-sequence dependencies, the number of hidden states is typically large, leading to channel feature redundancy.

**Core Idea**: Design a specialized restoration block (RSSB) to compensate for the limitations of Mamba in low-level vision via local convolution enhancement and channel attention, establishing it as a third class of restoration backbone alongside CNNs and Transformers.

## Method

### Overall Architecture

MambaIR adopts a three-stage architecture:
1. **Shallow Feature Extraction**: A 3×3 convolutional layer extracts shallow features $F_S \in \mathbb{R}^{H \times W \times C}$.
2. **Deep Feature Extraction**: Multiple stacked Residual State-Space Groups (RSSG), with each group containing several Residual State-Space Blocks (RSSB).
3. **High-Quality Image Reconstruction**: The shallow and deep features are element-wise added to reconstruct the output.

### Key Designs

1. **Residual State-Space Block (RSSB)**: RSSB is the core module of MambaIR, breaking the fixed paradigm of "Norm → Attention → Norm → MLP" in Transformers. It consists of two sub-blocks:

    - Part 1: LayerNorm → VSSM (Visual State-Space Module) + shortcut connection with a learnable scaling factor $s$, i.e., $Z^l = \text{VSSM}(\text{LN}(F_D^l)) + s \cdot F_D^l$.
    - Part 2: LayerNorm → bottleneck local convolution → Channel Attention (CA) + shortcut connection with a learnable scaling factor $s'$, i.e., $F_D^{l+1} = \text{CA}(\text{Conv}(\text{LN}(Z^l))) + s' \cdot Z^l$.
   
   The local convolution is utilized to restore neighborhood similarity destroyed by 1D flattening, and channel attention is used to alleviate channel redundancy caused by a large number of hidden states. The learnable scaling factors control the information flow of shortcut connections.

2. **Visual State Space Module (VSSM)**: Adopts a dual-branch structure. The first branch extracts global features through Linear → DWConv → SiLU → 2D-SSM → LN; the second branch acts as a gating mechanism via Linear → SiLU. The two branches are fused via Hadamard product and then projected back to the original channel dimension. The formula is:
    $X_1 = \text{LN}(\text{2D-SSM}(\text{SiLU}(\text{DWConv}(\text{Linear}(X)))))$
    $X_2 = \text{SiLU}(\text{Linear}(X))$
    $X_{out} = \text{Linear}(X_1 \odot X_2)$

3. **2D Selective Scanning Module (2D-SSM)**: To enable Mamba to process non-causal data of 2D images, a four-direction scanning strategy is adopted (top-left to bottom-right, bottom-right to top-left, top-right to bottom-left, bottom-left to top-right). The flattened sequence of each direction is modeled for long-range dependencies using discrete state-space equations. Finally, the results of the four directions are summed and reshaped back to 2D.

### Loss & Training

- Image Super-Resolution: L1 loss $\mathcal{L} = \|I_{HQ} - I_{LQ}\|_1$
- Image Denoising: Charbonnier loss $\mathcal{L} = \sqrt{\|I_{HQ} - I_{LQ}\|^2 + \epsilon^2}$, where $\epsilon = 10^{-3}$
- Training patches: 64×64 for SR, 128×128 for denoising
- Data augmentation: horizontal flip + random rotation of 90°/180°/270°
- Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$), initial learning rate $2 \times 10^{-4}$
- SR ×3 and ×4 models are initialized with the pretrained weights of the ×2 model
- Trained on 8× NVIDIA V100 GPUs

## Key Experimental Results

### Main Results

Classical Image Super-Resolution (×2 scale, PSNR/dB):

| Dataset | Metric | MambaIR | SwinIR | SRFormer | Gain |
|--------|------|---------|--------|----------|------|
| Set5 | PSNR | 38.57 | 38.42 | 38.51 | +0.15/+0.06 |
| Set14 | PSNR | 34.67 | 34.46 | 34.44 | +0.21/+0.23 |
| Urban100 | PSNR | 34.15 | 33.81 | 34.09 | +0.34/+0.06 |
| Manga109 | PSNR | 40.28 | 39.92 | 40.07 | +0.36/+0.21 |

Classical Image Super-Resolution (×4 scale, PSNR/dB):

| Dataset | Metric | MambaIR | SwinIR | SRFormer | Gain |
|--------|------|---------|--------|----------|------|
| Set5 | PSNR | 33.03 | 32.92 | 32.93 | +0.11/+0.10 |
| Urban100 | PSNR | 27.68 | 27.45 | 27.68 | +0.23/0.00 |
| Manga109 | PSNR | 32.32 | 32.03 | 32.21 | +0.29/+0.11 |

### Ablation Study

RSSB design ablation (Set5/Set14/Urban100 PSNR):

| Configuration | Set5 | Set14 | Urban100 | Description |
|------|------|-------|----------|------|
| Remove Conv | 38.48 | 34.54 | 34.04 | Local enhancement contributes greatly to Urban100 |
| Remove Conv+CA | 38.55 | 34.64 | 34.06 | Directly using vanilla Mamba yields suboptimal results |
| Replace Conv+CA with MLP | 38.55 | 34.68 | 34.22 | Transformer-like structures are unsuitable for SSMs |
| Full MambaIR | 38.57 | 34.67 | 34.15 | - |

Scanning direction ablation:

| Configuration | Set5 | Urban100 | Description |
|------|------|----------|------|
| Unidirectional | 38.53 | 34.06 | Smallest information perception range |
| Bidirectional | 38.56 | 33.96 | - |
| Four-directional (baseline) | 38.57 | 34.15 | Optimal |

### Key Findings

- MambaIR possesses a true global Effective Receptive Field (ERF) while maintaining computational complexity comparable to SwinIR (verified by ERF visualization).
- Computational complexity grows linearly with input resolution, similar to window attention but with a larger receptive field.
- Although both SSM and Attention can model global dependencies, their behavioral patterns differ and they cannot be simply substituted.
- The lightweight version, MambaIR-light, still outperforms SwinIR-light by 0.34dB with similar parameters and MACs (×4 Manga109).

## Highlights & Insights

1. **Systematically introducing Mamba to low-level vision for the first time**: Although works like VMamba have explored Mamba in high-level vision tasks, MambaIR is the first Mamba-based method tailored for image restoration.
2. **Simple and effective design philosophy**: The two improvements in RSSB (local convolution + channel attention) are extremely simple but directly address the two actual pain points of applying Mamba to 2D images.
3. **Insights on global receptive field**: Through ERF visualization, it is clearly demonstrated that MambaIR achieves a global receptive field similar to full attention, but with linear complexity.
4. **Learnable scaling factors**: Controlling shortcut connections, allowing the network to adaptively adjust the residual information flow.

## Limitations & Future Work

1. The four-directional scanning remains a heuristic 2D adaptation scheme and may not be the optimal spatial modeling method.
2. Ablation studies show that the improvement of MambaIR on some datasets is not particularly significant (e.g., only +0.06dB vs. SRFormer on Set5).
3. Extensive experiments have not been conducted on real-world degradation scenarios (Real-world SR).
4. For different restoration tasks (SR, denoising, JPEG artifact removal), the model configurations need to be adjusted separately.
5. Although the inference speed is better than the vanilla Transformer, detailed latency comparison data is not provided in the paper.

## Related Work & Insights

- **VMamba**: The VSSM and 2D-SSM in MambaIR directly inherit the design from VMamba, indicating that Mamba adaptation schemes in high-level vision can be migrated to low-level vision.
- **SwinIR**: As the main baseline, the limitations of window attention (non-global receptive fields) serve as the starting point for MambaIR.
- **HAT**: The observation that activating more pixels improves restoration performance provides theoretical support for Mamba's global modeling.
- **Insights for future work**: Better 2D scanning strategies can be explored, hybrid architectures combining Mamba and attention can be developed, and the application of Mamba to longer sequence tasks such as video restoration can be investigated.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces Mamba to image restoration for the first time, although the specific module design is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks such as SR and denoising with comprehensive ablation studies, but lacks real-world scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, problem definition is accurate, and experimental arrangement is reasonable.
- Value: ⭐⭐⭐⭐⭐ As a baseline work, it lays the foundation for subsequent Mamba-based restoration research with high impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[CVPR 2025\] MambaIRv2: Attentive State Space Restoration](../../CVPR2025/image_restoration/mambairv2_attentive_state_space_restoration.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](../../CVPR2025/image_restoration/qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)

</div>

<!-- RELATED:END -->
