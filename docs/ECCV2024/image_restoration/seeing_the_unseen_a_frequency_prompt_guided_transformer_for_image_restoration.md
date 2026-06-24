---
title: >-
  [Paper Note] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration
description: >-
  [ECCV 2024][Image Restoration][Frequency Prompt] This paper proposes FPro, guiding image restoration via prompting from a frequency-domain perspective. Employing a Gated Dynamic Decoupler to decouple features into low-frequency and high-frequency components, the method injects learnable prompts into both bands using a Dual Prompt Block (HPM + LPM) to interact with decoder features. It outperforms state-of-the-art methods comprehensively across five tasks: deraining…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Frequency Prompt"
  - "Dual Prompt Block"
  - "Gated Dynamic Decoupler"
  - "Transformer"
date: 2026-05-08
content_hash: 1d8dcf9ef556c233
---

# Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration

**Conference**: ECCV 2024  
**arXiv**: [2404.00288](https://arxiv.org/abs/2404.00288)  
**Code**: [https://github.com/joshyZhou/FPro](https://github.com/joshyZhou/FPro)  
**Area**: Image Restoration / Low-level Vision  
**Keywords**: image restoration, Frequency Prompt, Dual Prompt Block, Gated Dynamic Decoupler, Transformer

## TL;DR

This paper proposes FPro, guiding image restoration via prompting from a frequency-domain perspective. Employing a Gated Dynamic Decoupler to decouple features into low-frequency and high-frequency components, the method injects learnable prompts into both bands using a Dual Prompt Block (HPM + LPM) to interact with decoder features. It outperforms state-of-the-art methods comprehensively across five tasks: deraining, raindrop removal, demoireing, deblurring, and dehazing.

## Background & Motivation

**Background**: Deep learning-based image restoration methods (such as Restormer, Uformer, SwinIR) have yielded significant progress, with Transformer architectures becoming dominant due to their ability to capture global features. Recently, Prompt Learning has been introduced to the image restoration field (e.g., PromptIR, DA-CLIP) to modulate networks by encoding degradation information.

**Limitations of Prior Work**:
1. Existing prompt-based methods (PromptIR, DA-CLIP) focus on extracting degradation information in the spatial domain and ignore frequency-domain clues — where different degradations (rain streaks, raindrops, moiré patterns) affect distinct frequency bands.
2. Self-attention is fundamentally a low-pass filter, which dilutes high-frequency information (textures, edges). Existing Transformer-based restoration models struggle to exploit high-frequency details.
3. Degradation types exhibit diverse manifestations in the frequency domain: rain streaks partially block backgrounds (high frequency), while raindrops cause wide-area occlusions (low frequency). Existing methods do not handle them in a targeted manner.

**Key Challenge**: Spatial-domain prompts cannot capture the frequency characteristic discrepancies of degradations, resulting in residual subtle or imperceptive artifacts in the frequency domain of restored images.

**Key Insight**: Designing a prompt mechanism from a frequency-domain perspective — decoupling features into different frequency bands and injecting prompt modules respectively to encode degradation-specific information.

**Core Idea**: Frequency-domain prompt learning — generating and modulating prompts separately on low-frequency and high-frequency components, using a dual-branch architecture to handle both global structure and local details.

## Method

### Overall Architecture

FPro consists of two branches: (1) the upper Restoration Branch — a standard encoder-decoder structure responsible for image restoration; (2) the lower Prompt Branch — which extracts frequency information from input shallow features to generate prompts and interact with decoder features.

**Input**: Degradation image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$  
**Mechanism**: A 3×3 Conv extracts shallow features $\mathbf{F}_s$ $\rightarrow$ which are simultaneously fed into the Restoration Branch (N1=3 stages of encoder-decoder) and the Prompt Branch (GDD + DPB) $\rightarrow$ the output of the Prompt Branch is fused with decoder features at each stage via 1×1 Conv $\rightarrow$ generating a residual map $\mathbf{R}$ $\rightarrow$ outputting $\hat{\mathbf{I}} = \mathbf{I} + \mathbf{R}$.

**Key Architectural Design**: Attention mechanisms are removed from the encoder (retaining only FFNs) because early layers focus on local patterns and self-attention acts as a low-pass filter that dilutes high frequencies — representing an important design trade-off.

### Key Designs

1. **Gated Dynamic Decoupler (GDD)**:

    - **Function**: Dynamically decouples input features into low-frequency and high-frequency components.
    - **Mechanism**: Generates channel descriptors for the input features $\mathbf{F}_s$ via GAP + Conv. After suppressing unimportant elements through a gating mechanism (sigmoid gating), it uses Softmax normalization to guarantee the generation of a low-pass filter $\mathbf{F}^L$. Applying $\mathbf{F}^L$ to grouped input features yields the low-frequency component:
    $$\mathbf{F}^{lo}_{i,c,h,w} = \sum_{p,q} \mathbf{F}^L_{i,p,q} \mathbf{F}_{i,c,h+p,w+q}$$
    - The high-pass filter is obtained by subtracting the low-pass filter from the identity kernel, yielding the high-frequency component $\mathbf{F}_{hi}$.
    - **Design Motivation**: Different degradations affect different frequency bands, necessitating the adaptive separation of frequency components. Softmax ensures the filter is low-pass, while gating suppresses redundant elements. Filters are dynamically learned for each spatial location and channel group.

2. **High-frequency Prompt Modulator (HPM)**:

    - **Function**: Injects prompt modules into high-frequency features and performs local cross-attention with decoder features.
    - **Mechanism**: (a) Generation: Enhances high-frequency features using depth-wise convolution + GELU gating: $\hat{\mathbf{F}}_{hi} = \tilde{\mathbf{F}}_{hi} \odot \sigma(\text{DConv}_{3\times3}(\tilde{\mathbf{F}}_{hi}))$, then performs element-wise multiplication with the learnable prompt $\mathbf{P}_{hi}$: $\mathbf{F}^{prompt}_{hi} = \hat{\mathbf{F}}_{hi} \odot \mathbf{P}_{hi}$.
    - (b) Modulation: Further enhances high-frequency components via depth-wise conv, and then interacts with decoder features using local window cross-attention (window size $M=8$):
    $$\mathbf{F}^{out}_{hi} = \mathbf{V}_{hi} \cdot \text{Softmax}(\mathbf{K}_{hi} \cdot \mathbf{Q}_{hi} / \sqrt{d})$$
    - **Design Motivation**: High-frequency information corresponds to local details; using window attention is sufficient and computationally resource-efficient. Depth-wise convolution naturally acts as a high-pass filter.

3. **Low-frequency Prompt Modulator (LPM)**:

    - **Function**: Injects prompts into low-frequency features in the Fourier domain and performs global cross-attention with decoder features.
    - **Mechanism**: (a) Generation: Computes FFT of low-frequency features to transition to the frequency domain, filters useful components via gating: $\hat{\mathbf{F}}_{lo} = \mathcal{F}(\tilde{\mathbf{F}}_{lo}) \odot \sigma(\text{Conv}_{1\times1}(\mathcal{F}(\tilde{\mathbf{F}}_{lo})))$, applies element-wise multiplication with a learnable frequency-domain prompt $\mathbf{P}_{lo}$, and maps back to the spatial domain via IFFT:
    $\mathbf{F}^{prompt}_{lo} = \mathcal{F}^{-1}(\hat{\mathbf{F}}_{lo} \odot \mathbf{P}_{lo})$
    - (b) Modulation: Employs adaptive average pooling to enhance low-frequency components, which are then modulated via global cross-attention (where Q comes from decoder features, and K/V from pooled prompt features):
    $$\mathbf{F}^{out}_{lo} = \mathbf{V}_{lo} \cdot \text{Softmax}(\mathbf{K}_{lo} \cdot \mathbf{Q}_{lo} / \alpha)$$
    - **Key Theoretical Insight**: According to the Convolution Theorem, Hadamard products in the frequency domain are equivalent to spatial convolutions — Eq. (8) proves that the entire LPM is equivalent to a dynamic large-kernel depth-wise convolution, which is computationally more efficient when implemented in the frequency domain.
    - **Design Motivation**: Low-frequency information corresponds to global structures, requiring global (non-window) attention. Operating in the frequency domain naturally achieves a global receptive field.

### Loss & Training

- Uses the same widely adopted restoration loss function as PromptIR [60].
- AdamW optimizer, with an initial learning rate of $3 \times 10^{-4}$ and cosine annealing decay down to $1 \times 10^{-6}$.
- Architectural parameters: 3-stage encoder-decoder, block counts of [2,3,6], embedding dimension $C=48$, attention heads [2,4,8], and FFN expansion factor of 3.
- Pixel-unshuffle/pixel-shuffle are utilized for down/up-sampling.

## Key Experimental Results

### Main Results

**Deraining (SPAD)**:

| Method | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Restormer | 47.98 | 0.9921 |
| DRSformer | 48.53 | 0.9924 |
| **FPro** | **48.99** | **0.9936** |

+0.46 dB over DRSformer, and +2.1 dB over SCD-Former.

**Raindrop Removal (AGAN-Data)**:

| Method | PSNR↑ | SSIM↑ |
|------|-------|-------|
| IDT | 31.63 | 0.936 |
| Restormer | 31.68 | 0.934 |
| **FPro** | **31.96** | **0.937** |

+0.28 dB over Restormer.

**Demoireing (TIP-2018)**:

| Method | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Wang et al. | 28.87 | 0.894 |
| **FPro** | **29.25** | **0.879** |

+0.38 dB over the previous best.

### Ablation Study

**GDD Effectiveness**:

| Configuration | PSNR | SSIM | Description |
|------|------|------|------|
| Multi Dynamic Conv | 48.52 | 0.9926 | Replaced with multi-group dynamic convolutions |
| Multi GDD | 48.91 | 0.9934 | One GDD for each DPB |
| Single GDD (Shared) | **48.99** | **0.9936** | Shared single GDD (Optimal) |

**DPB Effectiveness**:

| Configuration | PSNR | SSIM | Description |
|------|------|------|------|
| w/o HPM | 48.77 | 0.9931 | Without high-frequency prompt, -0.22 dB |
| w/o LPM | 48.89 | 0.9933 | Without low-frequency prompt, -0.10 dB |
| Full | **48.99** | **0.9936** | Full model |

**Model Efficiency**:

| Method | FLOPs (G) | Params (M) |
|------|-----------|------------|
| Restormer | 174.7 | 26.1 |
| DRSformer | 242.9 | 33.7 |
| **FPro** | **81.9** | **22.3** |

FPro requires only approximately 47% of Restormer's FLOPs.

### Key Findings
- The contribution of HPM is larger than LPM (excluding HPM drops PSNR by 0.22 dB vs 0.10 dB for LPM), demonstrating that high-frequency details are more critical for image restoration.
- Replacing standard Dynamic Conv with GDD brings a +0.39 dB improvement, confirming that the gating mechanism effectively suppresses redundant filter elements.
- Sharing a single GDD outperforms using individual ones for each DPB (saving 0.02M parameters while gaining 0.08 dB), which suggests that sharing frequency decomposition across different scales is reasonable.
- FPro also achieves the best perceptual quality in real-world scenarios (NIQE 5.30 vs DRSformer 5.59).
- With only 81.9G FLOPs (vs DRSformer's 242.9G), the method is highly computationally efficient.

## Highlights & Insights

- **Theoretical elegance of frequency-domain prompts**: LPM operates Hadamard products + gating in the frequency domain, which is theoretically equivalent to spatial-domain dynamic large-kernel depth-wise convolutions but with much lower computational complexity. This theoretical derivation is a highlight of the paper.
- **Frequency decoupling + dual-branch processing**: Local window attention is used for high frequencies, while global attention is utilized for low frequencies — a design philosophy where frequency characteristics match attention spans naturally, providing valuable insights.
- **Removing attention from the Encoder**: Driven by the understanding that early layers focus on local patterns and self-attention acts as a low-pass filter — this approach of guiding architectural design through a deep understanding of Transformer properties is highly instructive.
- **Learnable frequency-domain prompt components**: Directly injecting $\mathbf{P}_{hi}$ and $\mathbf{P}_{lo}$ as learnable parameters into frequency features is simple and effective.
- **Unified framework for 5 degradations**: Although not an all-in-one model, the same architecture achieves state-of-the-art or near-state-of-the-art performance across 5 different restoration tasks.

## Limitations & Future Work

- The paper does not investigate the all-in-one setting; each degradation task must be trained separately.
- The frequency decomposition uses only 3×3 filters, which may not capture specific frequency patterns with high precision.
- The gating mechanism of GDD relies on global information from GAP, which might lack precision in local degradation scenarios (e.g., small-area occlusions).
- Detailed experiments on dehazing and deblurring are placed in the supplementary material, leaving the presentation in the main text somewhat incomplete.
- The prompt component $\mathbf{P}$ comprises learnable parameters of a fixed size; its adaptability to diverse input resolutions remains to be investigated.

## Related Work & Insights

- **vs Restormer**: Restormer reduces computational cost using channel-wise attention but ignores frequency-domain information. FPro achieves superior performance with fewer FLOPs.
- **vs PromptIR/DA-CLIP**: These methods extract degradation prompts from the spatial domain, ignoring frequency cues. FPro explicitly extracts prompts from a frequency-domain perspective.
- **vs DRSformer**: DRSformer is the SOTA in deraining. FPro outperforms it on SPAD by +0.46 dB with only 1/3 of its FLOPs.
- **vs SwinIR**: Based on window attention but without distinguishing frequency components. FPro's HPM also uses window attention but specifically processes high-frequency information.

## Rating

- Novelty: ⭐⭐⭐⭐ First to design prompt learning from a frequency perspective for image restoration, with solid theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of 5 tasks, thorough ablation studies, and intuitive visual analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ High writing quality, rigorous mathematical derivations, with great alignment among formulas, text, and diagrams.
- Value: ⭐⭐⭐⭐ Generalizable frequency prompt concept, low FLOPs with excellent performance, and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)
- [\[ECCV 2024\] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)
- [\[ECCV 2024\] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)

</div>

<!-- RELATED:END -->
