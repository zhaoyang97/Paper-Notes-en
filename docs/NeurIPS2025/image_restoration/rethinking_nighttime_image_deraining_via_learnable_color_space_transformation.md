---
title: >-
  [Paper Note] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation
description: >-
  [NeurIPS 2025][Image Restoration][nighttime image deraining] Motivated by the statistical finding that nighttime rain exhibits far greater contrast in the Y channel (luminance) of YCbCr than in RGB…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "nighttime image deraining"
  - "learnable color space transformation"
  - "YCbCr"
  - "implicit illumination guidance"
  - "HQ-NightRain dataset"
date: 2026-05-08
content_hash: 4809b124a9f29a9b
---

# Rethinking Nighttime Image Deraining via Learnable Color Space Transformation

**Conference**: NeurIPS 2025
**arXiv**: [2510.17440](https://arxiv.org/abs/2510.17440)  
**Code**: [guanqiyuan/CST-Net](https://github.com/guanqiyuan/CST-Net)  
**Institution**: Dalian Polytechnic University / Nanjing University of Science and Technology / Dalian Maritime University
**Area**: Image Restoration
**Keywords**: nighttime image deraining, learnable color space transformation, YCbCr, implicit illumination guidance, HQ-NightRain dataset

## TL;DR

Motivated by the statistical finding that nighttime rain exhibits far greater contrast in the Y channel (luminance) of YCbCr than in RGB, this work proposes a learnable Color Space Converter (CSC) that performs deraining in the Y channel, an Implicit Illumination Guidance (IIG) module that encodes non-uniform nighttime illumination, and a photorealistic dataset HQ-NightRain constructed via illumination-aware synthesis. The three components jointly yield substantial improvements in nighttime deraining performance.

## Background & Motivation

**Background**: Deep learning methods for image deraining (PReNet, Restormer, DRSformer, etc.) are abundant, yet the vast majority focus on daytime scenes. Nighttime scenarios are considerably more challenging due to low ambient light, spatially non-uniform artificial light sources, and the coupled degradation of rain and illumination.

**Limitations of Prior Work**:
1. **Unrealistic datasets**: Existing nighttime rain datasets (GTAV-NightRain, RoadScene, etc.) linearly superimpose globally uniform rain masks onto background images, completely ignoring the physical property that nighttime rain is visible only near light sources, resulting in a significant domain gap between synthetic and real images.
2. **Lack of task-specific design**: Existing nighttime deraining methods still operate in RGB space and do not exploit the intrinsic properties of nighttime rain in specific color channels.

**Key Challenge**: A fundamental mismatch between the physical formation mechanism of nighttime rain degradation (visible near light sources, non-uniformly distributed, with pronounced luminance-channel contrast) and the "daytime assumption" embedded in existing methods and datasets (uniform rain distribution, RGB-space processing).

**Goal**: To leverage the physical characteristics of nighttime rain—luminance-channel saliency and illumination-dependent non-uniform distribution—in order to design more effective deraining methods and datasets.

**Key Insight**: Statistical analysis of pixel values—comparing per-channel histograms across color spaces reveals that the Y channel of YCbCr exhibits the largest discrepancy between rainy and rain-free images (because rain reflects artificial light sources under low-light conditions, producing high-contrast luminance patterns). This observation motivates the design of performing deraining specifically in the Y channel.

**Core Idea**: A learnable color space transformation converts images from RGB to YCbCr; rain degradation is removed in the Y channel where the luminance contrast is greatest; implicit neural representations encode spatially non-uniform illumination information to guide the deraining process.

## Method

### Overall Architecture

CST-Net is a two-stage end-to-end network:

1. **Degradation Removal Stage (Stage 1)**: A learnable CSC transforms the RGB image into YCbCr space; the Y channel (luminance) is extracted and fed into a Transformer encoder-decoder for rain degradation removal, while the Cb/Cr channels are retained and passed forward.
2. **Color Refinement Stage (Stage 2)**: The derained Y channel is concatenated with Cb/Cr, transformed back to RGB via CSC, added to the original input, and fed into a second stage for color restoration.

Both stages adopt a 4-level Transformer encoder-decoder structure based on the Restormer architecture. An Implicit Illumination Guidance (IIG) branch connects the two stages and provides illumination-aware feature guidance.

### Key Designs

1. **Learnable Color Space Converter (CSC)**

    **Function**: Implements bidirectional RGB ↔ YCbCr color space transformation, replacing the conventional fixed-matrix conversion.

    **Mechanism**: The standard RGB→YCbCr conversion uses a fixed 3×3 weight matrix $W$ (e.g., Y=0.299R+0.587G+0.114B). CSC replaces this with a learnable parameter matrix $\Phi = \{\varphi_{i,j}\}$, where each element is a one-dimensional learnable variable. A nonlinear transformation is applied via an MLP before the matrix multiplication: $[Y, Cb, Cr]^T = \text{MLP}(\Phi) \circ [R, G, B]^T$.

    **Design Motivation**: The fixed transformation matrix is a standard paradigm designed for general scenes and cannot adapt to the complex luminance distributions induced by artificial light sources in nighttime settings. Learnable parameters can adaptively adjust channel weight assignments according to varying illumination conditions and scene types, avoiding pixel loss in high-brightness regions and improving robustness to complex, stochastic nighttime rain patterns. Visualization of the fixed-transform results shows pixel loss in highlight regions, whereas the learnable CSC dynamically adjusts weights to avoid this issue.

2. **Implicit Illumination Guidance Module (IIG)**

    **Function**: Propagates illumination information between the two stages, guiding the model to focus on rain degradation in illuminated regions.

    **Mechanism**: Implicit neural representations (INR) are used to encode illumination information. Pixel coordinates of nighttime image patches are stored in a coordinate set $P \in \mathbb{R}^{H \times W \times 2}$. For each pixel, a dynamic weight $w(x', y')$ inversely proportional to the distance from the neighborhood center is applied to extract local illumination context $P_{(x,y)}$. The encoded feature $E$ and positional illumination information are concatenated and decoded by an MLP to produce the derained Y-channel value: $I_{\hat{Y}} = \text{MLP}(\text{cat}[E, P_{(x,y)}])$.

    **Design Motivation**: Nighttime illumination is highly non-uniform (bright near light sources, dark elsewhere), and rain degradation severity varies across illumination regions. Compared to explicit illumination estimation (e.g., Retinexformer), implicit representation is more flexible—different illumination distributions yield different MLP parameter configurations, naturally accommodating complex nighttime scenes. Feature visualizations confirm that IIG accurately focuses on illuminated regions to guide deraining.

3. **HQ-NightRain Dataset Construction**

    **Function**: Provides high-fidelity synthetic training data for nighttime rain.

    **Mechanism**: An illumination-aware rain synthesis pipeline is proposed. A luminance coefficient matrix $\mathbf{I}$ is first extracted from the V channel of the background image in HSV space. High and low thresholds $\tau_1=0.2$ and $\tau_2=0.8$ are set to halve rain visibility in extremely dark and extremely bright regions. The rain mask is then element-wise multiplied by the luminance coefficients: $\sigma(S) = S \odot \mathbf{I}$, generating non-uniformly distributed rain. Additionally, defocus blur is applied to raindrop scenes to simulate real-world refraction effects. The new rain model is $R_s = f[B, \sigma(S)]$, using 3×3 convolution for fusion instead of simple linear addition.

    **Design Motivation**: According to the first law of illuminance, illuminance decreases with distance from the light source; real nighttime rain is therefore visible only near light sources. All existing datasets apply globally uniform rain overlays, violating this physical prior. t-SNE analysis confirms that the feature distribution of HQ-NightRain is substantially closer to that of real rainy images than existing datasets. The dataset contains 11,200 pairs (10K training / 900 validation / 300 test), divided into three subsets—rain streaks (RS), raindrops (RD), and mixed (SD)—along with 512 real captured images.

### Loss & Training

- Both stages use 4-level Transformer encoder-decoders (Restormer architecture)
- Optimizer: Adam (default parameters), initial learning rate $2 \times 10^{-4}$, cosine annealing to $1 \times 10^{-6}$
- 500 training epochs, patch size 128×128, batch size 4
- Training on a single RTX 3090
- Illumination thresholds $\tau_1 = 0.2$, $\tau_2 = 0.8$

## Key Experimental Results

### Main Results

**Average performance on HQ-NightRain + GTAV-NightRain:**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| PReNet | 35.51 | 0.9669 | 0.0821 |
| Restormer | 38.50 | 0.9767 | 0.0526 |
| DRSformer | 38.74 | 0.9766 | 0.0535 |
| NeRD-Rain | 38.51 | 0.9754 | 0.0588 |
| **CST-Net** | **39.07** | **0.9778** | **0.0477** |

**Performance on real dataset RealRain-1k:**

| Method | PSNR (1k-L)↑ | PSNR (1k-H)↑ |
|--------|-------------|-------------|
| DRSformer | 27.21 | 23.73 |
| NeRD-Rain | 27.16 | 23.65 |
| **CST-Net** | **27.31** | **23.81** |

**Multi-weather restoration on Multi-Weather6K (extended experiment):**

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| Restormer | 31.80 | 0.9228 |
| PromptIR | 31.69 | 0.9169 |
| **CST-Net** | **33.82** | **0.9642** |

### Ablation Study

**Module ablation (HQ-NightRain SD subset):**

| Configuration | PSNR↑ | SSIM↑ |
|---------------|-------|-------|
| Stage 1 only + YCbCr + CSC | 35.09 | 0.9650 |
| Stage 2 only + YCbCr + CSC | 36.44 | 0.9740 |
| Stage 1+2 + RGB (no CSC) | 38.75 | 0.9838 |
| Stage 1+2 + YCbCr + fixed transform | 39.60 | 0.9857 |
| Stage 1+2 + YCbCr + learnable CSC | 39.88 | 0.9866 |
| **Full model (+ IIG)** | **40.50** | **0.9881** |

**Color space comparison (two-stage + fixed transform):**

| Color Space | PSNR↑ | SSIM↑ |
|-------------|-------|-------|
| RGB | 38.75 | 0.9838 |
| HSV | 39.03 | 0.9843 |
| HSL | 39.16 | 0.9844 |
| YUV | 39.19 | 0.9846 |
| **YCbCr** | **39.60** | **0.9857** |

### Key Findings

1. **Clear advantage of the Y channel**: YCbCr outperforms RGB by 0.85 dB and HSV by 0.57 dB, validating the statistical observation that nighttime rain is most salient in the Y channel.
2. **Gain from learnable CSC**: Within YCbCr space, the learnable CSC improves over the fixed transform by 0.28 dB, particularly by avoiding pixel loss in highlight regions.
3. **Contribution of IIG**: Adding IIG on top of the learnable CSC yields a further gain of 0.62 dB (39.88→40.50), demonstrating the necessity of illumination guidance.
4. **Indispensability of both stages**: Stage 1 alone achieves only 35.09 dB and Stage 2 alone achieves 36.44 dB, while the joint two-stage design reaches 39.88 dB, confirming the rationality of the architectural design.
5. **Dataset generalization**: Training IDT on HQ-NightRain achieves 26.94 dB on RealRain-1k-L, surpassing training on GTAV-NightRain by 0.47 dB, demonstrating that illumination-aware synthesis effectively reduces the domain gap.
6. **Transfer to multi-weather**: CST-Net outperforms Restormer by 2.02 dB on Multi-Weather6K, indicating that the Y-channel strategy is not limited to rain removal.

## Highlights & Insights

1. **Statistically grounded physical insight**: Discovering via histogram comparison that nighttime rain is most distinctive in the Y channel is a concise, intuitive, and compelling observation that directly motivates the method design—a textbook example of the "good observation → good method" paradigm.
2. **Minimalist learnable design**: Replacing the fixed 3×3 transformation matrix with learnable parameters and an MLP is extremely simple yet effective. This approach of making standard operations learnable is broadly applicable.
3. **Coherent data–method co-design**: Both the method and the dataset are proposed together, and both are grounded in the same physical prior (illumination-dependent rain visibility), forming a complete and self-consistent solution.
4. **t-SNE domain gap analysis**: Using ResNet50 features with t-SNE visualization to quantitatively assess the realism of synthetic data is a methodological practice worth adopting.
5. **Systematic ablation over color spaces**: Comparisons across five color spaces (RGB/HSV/HSL/YUV/YCbCr) provide rigorous empirical support for the design choices.

## Limitations & Future Work

1. **CSC remains a global transform**: The learnable CSC is a global 3×3 matrix applied uniformly to all pixels. Given the strong spatial heterogeneity of nighttime illumination, pixel-wise or region-wise adaptive transformations could yield further improvements.
2. **Modest PSNR gains**: On the average HQ-NightRain metrics, CST-Net improves over DRSformer by approximately 0.33 dB, which is not a decisive margin.
3. **Inherent limitations of synthetic data**: Although t-SNE analysis shows a reduced domain gap, HQ-NightRain remains a synthetic dataset; complex optical phenomena in real rain (scattering, diffraction, hazing) may not be fully captured.
4. **Sensitivity of illumination thresholds**: $\tau_1=0.2$ and $\tau_2=0.8$ are manually set; their applicability to different light source types (LED vs. sodium lamp) is not discussed.
5. **Computational efficiency not reported**: Inference speed and model size for the two-stage Transformer + IIG design are not provided, leaving the feasibility of real-time deployment (e.g., autonomous driving) unclear.

## Related Work & Insights

- **vs. RLP (Zhang et al. 2023)**: RLP learns rain location priors via a recurrent residual network but still operates in RGB space. CST-Net's exploitation of Y-channel luminance properties is more physically grounded.
- **vs. NeRD-Rain (Chen et al. 2024)**: NeRD-Rain also employs implicit neural representations, but for bidirectional deraining in RGB space. CST-Net restricts INR to illumination encoding within the Y channel, resulting in a more targeted use.
- **vs. Restormer/DRSformer**: These general-purpose deraining Transformers apply no customization for color space or illumination. CST-Net demonstrates the value of incorporating task-specific domain knowledge.
- **Color space inspiration**: The advantage of YCbCr for deraining parallels the role of the V channel in HSV for low-light enhancement (e.g., RetinexFormer), suggesting that selecting the right channel is more important than scaling up the model.
- **Transferable ideas**: The learnable color space transformation can be extended to other tasks such as dehazing (where degradation has characteristic channel distributions) and underwater image restoration; the illumination-aware synthesis pipeline is also applicable to data generation for other nighttime degradation scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The integration of Y-channel statistical insight, learnable CSC, and illumination-aware dataset construction is cohesive; while no individual component is technically complex, the problem analysis is thorough and the design choices are tightly coupled.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comparisons across 5 color spaces, full module ablation, evaluation on 3 synthetic datasets + 2 real datasets + multi-weather extension + downstream detection application + t-SNE domain analysis + feature visualization; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is introduced clearly and compellingly through statistical observations; figures are rich (histograms, t-SNE, feature visualizations, qualitative comparisons); the logical chain is complete.
- **Value**: ⭐⭐⭐⭐ — The method is simple, effective, and reproducible; the HQ-NightRain dataset provides open-source value to the nighttime deraining community; practical utility is validated on an autonomous driving detection task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [\[AAAI 2026\] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](../../AAAI2026/image_restoration/clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)

</div>

<!-- RELATED:END -->
