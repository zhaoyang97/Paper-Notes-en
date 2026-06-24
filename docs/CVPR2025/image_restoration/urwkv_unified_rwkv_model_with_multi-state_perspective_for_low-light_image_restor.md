---
title: >-
  [Paper Note] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration
description: >-
  [CVPR 2025][Image Restoration][Low-light image enhancement] This paper proposes the URWKV model, which introduces a multi-state (intra-stage and inter-stage) perspective into the RWKV architecture. Through Lightness-Adaptive Normalization (LAN), State-aware Quad-directional Token Shift (SQ-Shift), and State-aware Selective Fusion (SSF) modules, a unified model is developed to handle the dynamically coupled degradations (noise, low-light distortion…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Low-light image enhancement"
  - "RWKV"
  - "Multi-state mechanism"
  - "Unified model"
  - "Image deblurring"
date: 2026-05-08
content_hash: ce2f1abe0fc4f5a8
---

# URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2505.23068](https://arxiv.org/abs/2505.23068)  
**Code**: [https://github.com/FZU-N/URWKV](https://github.com/FZU-N/URWKV)  
**Area**: Image Restoration / Low-Light Image Enhancement  
**Keywords**: Low-light image enhancement, RWKV, Multi-state mechanism, Unified model, Image deblurring

## TL;DR

This paper proposes the URWKV model, which introduces a multi-state (intra-stage and inter-stage) perspective into the RWKV architecture. Through Lightness-Adaptive Normalization (LAN), State-aware Quad-directional Token Shift (SQ-Shift), and State-aware Selective Fusion (SSF) modules, a unified model is developed to handle the dynamically coupled degradations (noise, low-light distortion, and motion blur) in low-light images. With only 2.25M parameters, the proposed model comprehensively outperforms existing methods across 8 benchmark datasets.

## Background & Motivation

**Background**: Low-light environments exhibit multiple degradations, including increased noise, loss of detail, reduced contrast, and color distortion. Existing methods can be categorized into three groups: Low-Light Image Enhancement (LLIE) models focusing on brightness elevation and denoising, joint LLIE-deblurring models addressing the coupled degradation of low-light and motion blur, and unified image restoration models such as Restormer and MambaIR.

**Limitations of Prior Work**: (1) LLIE models fail to handle the motion blur degradation frequently encountered in low-light scenes; (2) joint LLIE-deblurring models (e.g., LEDNet, PDHAT) are restricted to predefined degradation categories and cannot adapt to dynamically coupled degradation combinations in real-world scenarios; (3) general unified models (e.g., Restormer) lack adaptive mechanisms for low-light environments, often amplifying degradations or introducing new artifacts; (4) large parameter sizes and high computational costs pose key bottlenecks for practical deployment.

**Key Challenge**: How to flexibly handle dynamically changing coupled degradations in low-light scenes using a parameter-efficient unified model?

**Goal**: To construct a unified model capable of sensing and analyzing complex degradations, which does not rely on predefined degradation types but dynamically adapts to different degradation combinations through multi-state representations.

**Key Insight**: Taking inspiration from the human pupil adaptation mechanism under varying brightness, as well as the linear complexity and sequential modeling capability of the RWKV architecture, this work approaches the problem from a "multi-state" perspective—using inter-stage states for lightness adaptation and intra-stage states to capture long-range degradation dependencies.

**Core Idea**: To extend the single-state token shift in standard RWKV to a multi-state mechanism, aggregating historical states via EMA to capture long-range dependencies, and dynamically modulating normalization parameters through cross-stage states to achieve scene-aware lightness adaptation.

## Method

### Overall Architecture

A standard encoder-decoder architecture is adopted, consisting of 3 stages each for the encoder and decoder, with the URWKV block as the core processing unit. Each stage of the encoder contains $N_1=3$ URWKV blocks followed by downsampling layers, while each stage of the decoder contains $N_2=2$ URWKV blocks followed by upsampling layers. Each URWKV block consists of two sub-blocks: a multi-state spatial mixing sub-block and a multi-state channel mixing sub-block. Features are transmitted from the encoder to the decoder using the SSF module instead of naive skip connections.

### Key Designs

1. **Lightness-Adaptive Normalization (LAN)**:

    - **Function**: Replaces the standard LayerNorm, dynamically adjusting normalization parameters based on multiple historical stage states across the restoration pipeline to achieve scene-aware lightness modulation.
    - **Mechanism**: Collects the global lightness vectors (via global average pooling, GAP) of the current input $X_t$ and all historical stage outputs $M_i$, zero-pads them to a unified dimension, and stacks them into a 2D lightness map. Multi-kernel (1×T, 3×T, 5×T) 1D convolutions are used to aggregate lightness variation patterns across stages. These aggregated patterns are concatenated and passed through an MLP with a tanh activation to predict the lightness modulation parameter $\Delta\gamma_t$. The scaling parameter of LayerNorm is then updated as $\hat{\gamma_t} = \gamma + \Delta\gamma_t$.
    - **Design Motivation**: Fixed parameters in standard LayerNorm cannot adapt to complex lightness variations in low-light scenes. This design is inspired by the adaptive adjustment mechanism of the human pupil to environmental light. Leveraging inter-stage states allows normalization parameters to dynamically perceive lightness evolution throughout the entire restoration process.

2. **State-aware Quad-directional Token Shift (SQ-Shift)**:

    - **Function**: Extends the single-state Q-Shift of original RWKV to a multi-state mechanism, capturing long-range degradation dependencies across multiple states.
    - **Mechanism**: Prior to performing the standard Q-Shift (quad-directional spatial token shifting), an exponential moving average (EMA) is used to aggregate the current state and all preceding states within the same stage: $\text{MSA}(X_t^{LAN}) = \alpha \odot X_t^{LAN} + (1-\alpha) \odot \text{MSA}(H_{t-1})$, where $\alpha=0.5$ is the decay factor. Consequently, each block not only senses the spatial relationships of neighboring tokens but also infuses restoration information from historical states.
    - **Design Motivation**: The single-state mechanism of the original RWKV causes early information to gradually fade, failing to capture the complex dependencies between coupled degradations.

3. **State-aware Selective Fusion (SSF) Module**:

    - **Function**: Replaces naive skip connections, dynamically aligning and selectively fusing multi-state features across encoder stages.
    - **Mechanism**: Channel-wise mean compression is applied to the three encoder outputs to minimize semantic distractions, followed by adaptively aligning them to the target resolution of the decoder. After stacking, inception-style multi-scale convolutions (1×1, 3×3, 5×5) are employed to aggregate degradation patterns. Finally, a sigmoid function generates spatial weights $W_s$, allowing encoder features to be selectively transmitted to the decoder via $D_1' = ([W_s \odot E_3, D_1])W_p$.
    - **Design Motivation**: Naive skip connections (add/concat) easily propagate noise and irrelevant information in low-light environments, and suffer from semantic gaps across stages. SSF selectively filters useful information by predicting spatially guided weights.

### Loss & Training

Unified loss function: $L_1$ Loss + SSIM Loss + Perceptual Loss. The Adam optimizer is utilized ($\beta_1=0.9, \beta_2=0.99$), with an initial learning rate of $2\times10^{-4}$ decaying to $10^{-6}$ via cosine annealing. Data augmentation techniques such as flipping and rotation are applied during training. During testing, the model directly processes inputs of arbitrary shapes without cropping or scaling. Input channels $C=32$, trained on NVIDIA Tesla A40 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | URWKV | Retinexformer | Restormer | PDHAT | Params |
|--------|------|-------|---------------|-----------|-------|--------|
| LOL-v2-real | PSNR/SSIM | **23.11/0.874** | 22.79/0.839 | 18.60/0.789 | 20.16/0.841 | **2.25M** |
| LOL-v2-syn | PSNR/SSIM | **26.36/0.944** | 25.67/0.928 | 21.41/0.831 | 24.94/0.937 | - |
| SDSD-indoor | PSNR/SSIM | **31.24/0.911** | 29.78/0.895 | 28.49/0.892 | 26.37/0.884 | - |
| LOL-blur | PSNR/SSIM | **27.27/0.890** | 25.25/0.821 | 26.38/0.860 | 26.71/0.879 | - |

URWKV requires only 2.25M parameters and 18.34G FLOPs, which is significantly lower than Restormer (26.11M/140.99G) and MIRNet (31.76M/785G).

### Ablation Study

| Configuration | PSNR | SSIM | Params | FLOPs |
|------|------|------|--------|-------|
| Baseline (w/o LAN w/o SSF) | 21.33 | 0.856 | 1.64M | 18.25G |
| + SSF | 21.40 | 0.861 | 1.65M | 18.29G |
| + LAN | 22.71 | 0.869 | 2.25M | 18.30G |
| + LAN + SSF (Full) | **23.11** | **0.874** | 2.25M | 18.34G |

Ablation of multi-state aggregation: The combination of multi-state EMA + Q-Shift outperforms single-state or standard Q-Shift by 0.5–0.9 dB.

### Key Findings

- LAN provides the largest performance boost (+1.38 dB), demonstrating that lightness-adaptive normalization is crucial in low-light scenarios.
- Compared to naive skip connections (Add/Cat), SSF not only achieves better performance but also avoids noise propagation caused by direct multi-state feature fusion.
- Multi-state EMA aggregation significantly outperforms single-state and standard Q-Shift, validating the importance of cross-state long-range dependencies.
- The model surpasses specialized LLIE-deblurring models (PDHAT) on the LOL-blur dataset, proving that a unified model can effectively handle coupled degradations.

## Highlights & Insights

- **Extreme Parameter Efficiency**: Achieving superior performance over the 26M Restormer and 31M MIRNet with only 2.25M parameters, where the linear complexity of the RWKV architecture is key. This is highly valuable for edge-device deployment.
- **Novelty of Multi-State Perspective**: Extending the state concept of RWKV from simple sequential dependency to multi-level context awareness in the restoration process, incorporating both intra-stage (within the degradation restoration process) and inter-stage (across processing stages) states. This represents an elegant design.
- **Biological Inspiration of LAN**: The design of LAN, mimicking the adaptive regulations of the pupil, offers clean physical intuition, and its multi-kernel aggregation strategy effectively captures both local and global lightness variation patterns.

## Limitations & Future Work

- Currently, the evaluation is primarily focused on low-light contexts; whether it can seamlessly extend to wider degradation types (e.g., rain, haze, compression artifacts) remains unverified.
- The decay factor $\alpha$ in EMA is fixed at 0.5, which may not be optimal for varying degrees of degradation; exploring adaptive $\alpha$ could be a worthwhile avenue.
- The SSF module applies channel-wise mean compression to encoder outputs to reduce interference, which might also discard some useful information.
- The model performs slightly worse than Retinexformer on the SID dataset, indicating room for improvement in extremely dark scenes.

## Related Work & Insights

- **vs Retinexformer**: Retinexformer is a specialized LLIE model based on Retinex theory. It shows strong competitiveness on pure enhancement tasks but cannot handle coupled degradations like motion blur. The unified capability of URWKV is a major advantage.
- **vs Restormer**: Restormer is a general unified model but lacks low-light adaptation mechanisms, resulting in poor performance on low-light datasets. URWKV compensates for this through LAN and its multi-state mechanism.
- **vs MambaIR**: Although both leverage efficient linear-complexity architectures, MambaIR lacks components specialized for low-light conditions, underperforming URWKV on multiple low-light datasets.
- **vs PDHAT**: PDHAT is a specialized LLIE-deblurring model that remains highly competitive on LOL-blur but suffers from training instability on pure LLIE tasks. URWKV performs superiorly in both scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-state RWKV perspective is novel, and the designs of LAN and SSF are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets, multiple degradation types, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with strong overall motivation.
- Value: ⭐⭐⭐⭐ A unified model with 2.25M parameters is highly attractive for practical deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [\[CVPR 2025\] Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)

</div>

<!-- RELATED:END -->
