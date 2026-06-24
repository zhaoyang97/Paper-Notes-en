---
title: >-
  [Paper Note] Restoring Images in Adverse Weather Conditions via Histogram Transformer
description: >-
  [ECCV 2024][Image Restoration][adverse weather removal] Proposed Histoformer, an efficient Transformer based on histogram self-attention. By sorting and binning spatial features according to pixel intensity, it performs self-attention within and across bins to establish dynamic-range spatial attention for efficiently processing weather-degraded pixels. Combined with dynamic-range convolution and Pearson correlation loss, it achieves a unified modeling and reaches SOTA perform…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "adverse weather removal"
  - "histogram self-attention"
  - "dynamic-range convolution"
  - "Pearson correlation loss"
date: 2026-05-08
content_hash: 6c467dfb166b6a76
---

# Restoring Images in Adverse Weather Conditions via Histogram Transformer

**Conference**: ECCV 2024  
**arXiv**: [2407.10172](https://arxiv.org/abs/2407.10172)  
**Code**: [GitHub (Open Source)](https://github.com/sunshangquan/Histoformer)  
**Area**: Image Restoration / Adverse Weather Removal  
**Keywords**: image restoration, adverse weather removal, histogram self-attention, dynamic-range convolution, Pearson correlation loss

## TL;DR

Proposed Histoformer, an efficient Transformer based on histogram self-attention. By sorting and binning spatial features according to pixel intensity, it performs self-attention within and across bins to establish dynamic-range spatial attention for efficiently processing weather-degraded pixels. Combined with dynamic-range convolution and Pearson correlation loss, it achieves a unified modeling and reaches SOTA performance on three major tasks: desnowing, deraining/dehazing, and deraindropping.

## Background & Motivation

**Background**: Image restoration under adverse weather conditions such as rain, haze, and snow has been widely studied, with significant progress made from early CNN methods to recent Transformer approaches. All-in-One weather removal (handling multiple weather degradations with a single model) has become an important direction.

**Limitations of Prior Work**: To reduce computational overhead, existing Transformer methods either restrict self-attention to fixed spatial windows (e.g., Swin Transformer) or perform attention only along the channel dimension (e.g., Restormer), sacrificing the ability to capture long-range spatial features.

**Key Challenge**: The spatial distribution of weather-degraded pixels (raindrops, snowflakes, haze) is dynamic and globally scattered, yet they exhibit similar intensity patterns. Fixed-window attention fails to associate these scattered but similar degraded pixels.

**Goal**: To design an efficient attention mechanism that can adaptively focus on spatially scattered but intensity-similar weather-degraded pixels.

**Key Insight**: It is observed that weather degradation typically leads to similar occlusion and brightness patterns. Therefore, pixels can be sorted and binned by intensity to group similarly degraded pixels together for attention.

**Core Idea**: Sort spatial features by intensity and bin them into histogram bins. Self-attention is then performed within bins (fine-grained) and across bins (global), achieving dynamic-range efficient spatial attention.

## Method

### Overall Architecture

Histoformer utilizes an encoder-decoder architecture in a U-Net style. The low-quality input image $I^{lq} \in \mathbb{R}^{3 \times H \times W}$ undergoes patch embedding using a $3 \times 3$ convolution, followed by multi-stage feature extraction via **Histogram Transformer Blocks (HTBs)**. Skip connections link the encoder and decoder, while pixel-unshuffle/shuffle operations are used for downsampling/upsampling between stages. The encoder also contains crude skip connections (avg pooling + pointwise conv + depthwise conv) to supplement original features from the input, causing the encoder to focus more on learning weather-degrading residuals.

Each HTB consists of two core modules:
- **Dynamic-range Histogram Self-Attention (DHSA)**
- **Dual-scale Gated Feed-Forward (DGFF)**

### Key Designs

#### 1. Dynamic-range Histogram Self-Attention (DHSA)

DHSA is the core innovation of this work, consisting of two parts: dynamic-range convolution and dual-pathway histogram self-attention.

**Dynamic-range Convolution**: Prior to conventional convolution, features are sorted and reorganized so that convolution can operate on a dynamic range. The input feature $F \in \mathbb{R}^{C \times H \times W}$ is split along the channel dimension into $F_1, F_2$. $F_1$ is sorted horizontally and then vertically, and then concatenated with $F_2$ followed by a $1 \times 1$ pointwise convolution + $3 \times 3$ depthwise separable convolution:

$$F_1 = \text{Sort}_v(\text{Sort}_h(F_1))$$
$$F = \text{Conv}^d_{3 \times 3}(\text{Conv}_{1 \times 1}(\text{Concat}(F_1, F_2)))$$

**Design Motivation**: After sorting, high/low-intensity pixels are concentrated near the matrix diagonals, naturally clustering weather-degraded pixels (which share high intensity similarities). This allows small convolution kernels to capture feature correlations across dynamic ranges.

**Histogram Self-Attention**: From the output of the dynamic-range convolution, Value $V$ and two pairs of Query-Key $(Q_1, K_1), (Q_2, K_2)$ are extracted. Q and K are rearranged according to the sorted pixel intensity of $V$. Two reshaping approaches are then defined:

- **Bin-wise Histogram Reshaping (BHR)**: Defines $B$ bins, with each bin containing $HW/B$ pixels. Each bin covers a wide range of intensities $\rightarrow$ global feature aggregation.
- **Frequency-wise Histogram Reshaping (FHR)**: Defines $B$ frequencies per bin, resulting in $HW/B$ bins. Each bin contains only a small number of pixels with close intensities $\rightarrow$ fine-grained feature extraction.

Self-attentions are performed along both pathways, followed by element-wise multiplication for fusion:

$$A_B = \text{softmax}\left(\frac{\mathbf{R}_B(Q_1) \cdot \mathbf{R}_B(K_1)^\top}{\sqrt{k}}\right) \mathbf{R}_B(V)$$

$$A_F = \text{softmax}\left(\frac{\mathbf{R}_F(Q_2) \cdot \mathbf{R}_F(K_2)^\top}{\sqrt{k}}\right) \mathbf{R}_F(V)$$

$$A = A_B \odot A_F$$

Finally, the features are restored to their original spatial positions and output via a $1 \times 1$ convolution.

#### 2. Dual-scale Gated Feed-Forward (DGFF)

Replacing the standard FFN, this module introduces dual-scale, dual-range depthwise convolutional pathways. The input first goes through a $1 \times 1$ convolution to expand channels (factor $r=2.667$), and after pixel-shuffle, is split into two branches:
- Branch 1: $5 \times 5$ depthwise convolution (multi-scale)
- Branch 2: $3 \times 3$ dilated depthwise convolution (multi-range)

Branch 2, activated by Mish, serves as a gating map and is element-wise multiplied with Branch 1, then output via pixel-unshuffle + $1 \times 1$ convolution:

$$F_{l+1} = \text{Conv}_{1 \times 1}(\text{Unshuffle}(\text{Mish}(F_{l,2}) \odot F_{l,1}))$$

**Design Motivation**: The multi-scale characteristics of weather degradation require convolutions with different receptive fields to collaborate, and the gating mechanism adaptively selects effective features.

#### 3. Pearson Correlation Loss

It is observed that pixel-level L1 loss ignores the overall linear relationship between the output and the ground truth. The Pearson correlation coefficient is introduced as an auxiliary loss:

$$\rho(I^{hq}, I^{gt}) = \frac{\sum_{i=1}^{3HW}(I^{hq}_i - \bar{I}^{hq})(I^{gt}_i - \bar{I}^{gt})}{3HW \cdot \sigma(I^{hq}) \cdot \sigma(I^{gt})}$$

$$\mathcal{L}_{cor} = \frac{1}{2}(1 - \rho(I^{hq}, I^{gt}))$$

Total loss: $\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{cor}$, with $\alpha = 1$ as default.

**Design Motivation**: Weather degradation disrupts the intensity sorting relationship among pixels within the image. The Pearson loss forces the reconstructed pixels to follow the same sorting as the ground truth, complementing the patch-level structural information not covered by the L1 loss.

### Loss & Training

- **Training Data**: Snow100K (9,000 images) + Raindrop (1,069 images) + Outdoor-Rain (9,000 images), trained jointly
- **Optimizer**: AdamW, initial lr $3 \times 10^{-4}$, cosine annealed to $1 \times 10^{-6}$
- **Training**: 300K iterations, progressive learning (starting with batch size 8, patch size 128)
- **Network Configurations**: Number of blocks per stage {4,4,6,8}, channels C=36, attention heads {1,2,4,8}
- **Data Augmentation**: Random horizontal/vertical flips
- **Hardware**: NVIDIA V100

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | Histoformer | Restormer | TransWeather | WeatherDiff64 | AWRCP |
|------|--------|------|-------------|-----------|-------------|-------------|-------|
| Desnowing | Snow100K-S | PSNR | **37.41** | 36.01 | 32.51 | 35.83 | 36.92 |
| Desnowing | Snow100K-L | PSNR | **32.16** | 30.36 | 29.31 | 30.09 | 31.92 |
| Deraining/Dehazing | Outdoor-Rain | PSNR | **32.08** | 30.03 | 28.83 | 29.64 | 31.39 |
| Deraindropping | RainDrop | PSNR | **33.06** | 32.18 | 30.17 | 30.71 | 31.93 |

Note: Restormer, TransWeather, etc., are trained individually on each task, whereas Histoformer handles all tasks to surpass them using a unified model.

### Ablation Study

**Comparison of Self-Attention Types** (Outdoor-Rain):

| Self-Attention | PSNR | SSIM |
|----------|------|------|
| MDTA (Restormer)| 30.94 | 0.9278 |
| TKSA (Sparse Attention) | 31.12 | 0.9295 |
| w/o BHR | 31.05 | 0.9301 |
| w/o FHR | 31.79 | 0.9364 |
| **DHSA (Full)** | **32.08** | **0.9389** |

**Comparison of Feed-Forward Networks**:

| FFN Type | PSNR | SSIM |
|----------|------|------|
| Vanilla FFN | 31.32 | 0.9313 |
| GDFN (Restormer) | 31.42 | 0.9347 |
| MSFN | 31.78 | 0.9367 |
| **DGFF** | **32.08** | **0.9389** |

**Correlation Loss Weight**:

| $\alpha$ | PSNR | SSIM |
|----------|------|------|
| 0 (w/o $\mathcal{L}_{cor}$) | 31.77 | 0.9358 |
| 0.1 | 32.01 | 0.9369 |
| **1** | **32.08** | **0.9389** |
| 5 | 32.03 | 0.9392 |
| 10 | 31.96 | 0.9375 |

### Key Findings

1. **DHSA contributes the most**: Full DHSA gains 0.96 dB PSNR over TKSA. Both BHR and FHR are indispensable, but BHR contributes more (removing BHR drops PSNR by 1.03 dB, while removing FHR drops it by 0.29 dB).
2. **Dynamic-range convolution is effective but gains are moderate**: Sorting before convolution improves performance by 0.14 dB over conventional convolution, and the sorting order has little impact.
3. **DGFF outperforms all compared FFNs**: Improving on MSFN by 0.3 dB, indicating the effectiveness of combining pixel-shuffle and dilated convolutions.
4. **Pearson loss is consistently effective**: Its addition yields a 0.31 dB gain, and it remains effective within the range of 0.1 to 5, showing insensitivity to hyperparameters.
5. **Increasing $C \times B$ continuously improves performance**, but out-of-memory (OOM) occurs at 44. $C=36$ is chosen as the optimal available configuration.

## Highlights & Insights

- **Histogram binning for attention** is a clever design: light of weather degradation's nature of "scattered distribution of similar pixels" is elegantly resolved by the sorting-binning operation, maintaining linear complexity while achieving long-range spatial attention.
- **The dual-pathway of BHR + FHR** balances global and fine-grained aspects: BHR handles global aggregation across intensity ranges, while FHR manages fine-grained processing within similar intensities, with the two complementing each other.
- **Pearson correlation loss** offers a general insight: pixel-level losses ignore the relative sorting relationships, whereas correlation loss can be extended to any image restoration task.
- **A unified model outperforms specialized models**: Histoformer surpasses specialized models like Restormer across all weather tasks with a single model, demonstrating the strong generalization of histogram attention to diverse weather degradation patterns.

## Limitations & Future Work

1. The computational overhead of the sorting operation is not analyzed in detail and may become a bottleneck on ultra-high-resolution images.
2. Only three weather types (rain, snow, and haze) are validated, without addressing extreme conditions such as sandstorms or freezing weather.
3. The channel dimension $C=36$ is constrained by OOM, suggesting that the model capacity may not be fully unleashed.
4. The dataset is relatively small (~19k for training), and performance on larger datasets remains to be verified.
5. The number of histogram bins $B$ is fixed; an adaptive number of bins might further improve performance.

## Related Work & Insights

- **Restormer** [Zamir et al.] is the strongest baseline method, which employs channel-wise self-attention. Histoformer demonstrates that spatial-wise attention (via the binning strategy) is more effective than channel attention.
- **TransWeather** [Valanarasu et al.] is a pioneer in All-in-One weather removal, but its performance is far inferior to Histoformer.
- **WeatherDiff** [Özdenizci & Bhatt] introduces diffusion models to weather removal, but falls short of Histoformer in both quantitative and visual results.
- **Insight**: Attention mechanisms based on content sorting could be equally effective in other degradation patterns (such as noise or blur), which is worth further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Histogram self-attention is a entirely new attention paradigm, offering a novel concept and natural motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across three major tasks + detailed ablation studies, but lacks complexity analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology, complete equations, and intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐ Achieves a unified SOTA model + provides a highly versatile attention mechanism, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)
- [\[ICCV 2025\] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)](../../ICCV2025/image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)

</div>

<!-- RELATED:END -->
