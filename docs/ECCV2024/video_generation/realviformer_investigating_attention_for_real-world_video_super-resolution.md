---
title: >-
  [Paper Note] RealViformer: Investigating Attention for Real-World Video Super-Resolution
description: >-
  [ECCV 2024][Video Generation][real-world video super-resolution] This paper systematically investigates the behavioral differences between spatial and channel attention in real-world video super-resolution (RWVSR). It is found that channel attention is more robust to degradation artifacts but leads to feature redundancy. Based on this, RealViformer is proposed with Improved Channel Attention (ICA) and Channel Attention Fusion (CAF) modules, achieving SOTA performance with few…
tags:
  - "ECCV 2024"
  - "Video Generation"
  - "real-world video super-resolution"
  - "channel attention"
  - "artifact propagation"
  - "Transformer"
  - "covariance"
date: 2026-05-08
content_hash: d2232b11aacfd048
---

# RealViformer: Investigating Attention for Real-World Video Super-Resolution

**Conference**: ECCV 2024  
**arXiv**: [2407.13987](https://arxiv.org/abs/2407.13987)  
**Code**: [https://github.com/Yuehan717/RealViformer](https://github.com/Yuehan717/RealViformer)  
**Area**: Video Generation  
**Keywords**: real-world video super-resolution, channel attention, artifact propagation, Transformer, covariance

## TL;DR

This paper systematically investigates the behavioral differences between spatial and channel attention in real-world video super-resolution (RWVSR). It is found that channel attention is more robust to degradation artifacts but leads to feature redundancy. Based on this, RealViformer is proposed with Improved Channel Attention (ICA) and Channel Attention Fusion (CAF) modules, achieving SOTA performance with fewer parameters and faster speed.

## Background & Motivation

**Background**: Video super-resolution (VSR) is a core task in low-level vision. Standard VSR assumes LR (low-resolution) frames are downsampled from HR (high-resolution) frames using a known kernel. In recent years, Transformer architectures (such as Swin-based methods) have replaced CNNs as SOTA in standard VSR. Real-world VSR faces complex degradations from camera imaging systems, compression, network transmission, etc., lacking closed-form LR/HR correspondences.

**Limitations of Prior Work**: (a) Recurrent VSR models propagate artifacts temporally through hidden states, which is particularly severe under real-world degradation; (b) spatial attention Transformers (such as Swin-based ones), which perform excellently in standard VSR, produce more artifacts in real-world scenarios (see Fig. 1), underperforming compared to the convolutional model RealBasicVSR; (c) existing RWVSR methods (e.g., RealBasicVSR, FastRealVSR) are mainly designed based on CNNs, lacking a systematic analysis of attention mechanism behavior under degradation conditions.

**Key Challenge**: Spatial attention is adept at spatial matching but highly sensitive to local degradation; channel attention is more robust to degradation but leads to high cross-channel covariance (feature redundancy), limiting reconstruction capacity. Both mechanisms have pros and cons, and how to leverage their strengths while mitigating their weaknesses is critical.

**Goal**: (a) Answer why Transformers are effective in standard VSR but perform poorly in RWVSR; (b) reveal the robustness of channel attention to degradation and its redundancy issues; (c) design an effective channel-attention-based RWVSR Transformer.

**Key Insight**: Starting from the covariance calculation nature of the attention mechanism, this work compares the output stability of spatial and channel attention under degraded queries through experiments, quantifies the channel covariance metric, and resolves the discovered issues using squeeze-excite and covariance recalibration.

**Core Idea**: Channel attention is more robust to degradation due to covariance calculation over a large spatial range. Its feature redundancy can be alleviated through squeeze-excite and attention map-based channel weight recalibration, thereby constructing an efficient RWVSR Transformer.

## Method

### Overall Architecture

RealViformer adopts a unidirectional recurrent Transformer framework:
- **Optical Flow Estimation**: Uses SPyNet to estimate $s^f_{(t-1)\to t}$ and warps the previous hidden state $h_{t-1}$ to the current time step.
- **Reconstruction Module $\mathcal{R}$**: Takes the current frame $I^L_t$ and the aligned hidden state $\hat{h}_{t-1}$ as input, fuses temporal information via the CAF module, and reconstructs utilizing a three-stage encoder-decoder Transformer block with the ICA module.
- **Upsampling Module $\mathcal{U}$**: Upsamples the reconstructed features to output the HR frame.
- Encoder-decoder three-stage structure: Levels 1/2/3 contain [2,3,4] Transformer blocks, [48,96,192] channels, [1,2,4] attention heads, respectively, with a squeeze factor of 4.

### Key Designs

1. **Channel Attention Fusion (CAF) Module**:

    - **Function**: Fuses the shallow features $f_t$ of the current frame and the aligned hidden state $\hat{h}_{t-1}$ using channel attention, constraining the propagation of artifacts in the hidden state.
    - **Mechanism**: The Query is generated from $f_t$ via LayerNorm + 3×3 convolution; the Key/Value is generated from $\hat{h}_{t-1}$ via LayerNorm + 1×1 convolution + 3×3 depthwise convolution, followed by chunk splitting. The attention map $A_t \in \mathbb{R}^{C \times C}$ is calculated using the channel attention formula: $A_t = \text{softmax}(Q_t K_t^T / \alpha)$. The final output is $O_t = K_{1\times1} * K^d_{3\times3} * K_{1\times1} * \mathbf{C}[A_t V_t; f_t]$.
    - **Design Motivation**: Channel attention calculates covariance over a large spatial range (feature size of $\mathbb{R}^{1 \times HW}$), making it insensitive to local degradation. Experiments demonstrate that under blur, noise, and compression degradation, the cosine similarity of channel attention outputs reaches 0.98-0.99 (compared to only 0.75-0.92 for spatial attention).

2. **Improved Channel Attention (ICA) Module**:

    - **Function**: Replaces the original channel attention in Transformer blocks for self-attention feature reconstruction, alleviating channel redundancy.
    - **Mechanism**: (a) Squeeze-and-Excite: A squeeze convolution first compresses input channels by $r$ times. Channel attention is performed in this compressed space (yielding an attention map of size $\mathbb{R}^{C/r \times C/r}$), and an excite convolution restores the channel count, generating new non-redundant information. (b) Covariance-based channel recalibration: Average and maximum values are computed along the row direction of the attention map $A_r$, followed by a linear layer + sigmoid to predict scalar weights $\in \mathbb{R}^{C/r \times 1}$ for each channel, weighting the attention outputs channel-wise.
    - **Design Motivation**: Each channel of the channel attention output is a weighted sum of Value channels, significantly increasing cross-channel covariance ($ac(O) = 0.87$ vs. input $\approx 0.15$). High covariance implies feature redundancy, which is unfavorable for learning. Squeeze-excite performs attention in compressed space to generate low-redundancy features, while attention map-based weight prediction utilizes channel relation information, being more precise than the naive pooling of SE-Net.

3. **Exploratory Experimental System**:

    - **Function**: Systematically validates the sensitivity differences between spatial and channel attention and the channel covariance issue.
    - **Mechanism**: (a) Constructed sensitivity comparison experiments (Fig. 2): performed attention on clean frames to get $O$, and on degraded frames to get $O_{D_i}$, then compared their cosine similarity; (b) inserted attention modules at the temporal aggregation position of the recurrent VSR baseline (Fig. 3), training and testing under synthetic degradations to compare PSNR/LPIPS improvements; (c) quantified channel covariance metric $ac(Z) = \frac{1}{d}\sum_{i\neq j}|Cov(Z)|_{i,j}$.
    - **Design Motivation**: To provide experimental justification for the design choices of RealViformer rather than relying on intuition.

### Loss & Training

- **Two-Stage Training** (following the RealBasicVSR strategy):
    - **Stage 1** (300K iterations): Charbonnier loss + SSIM loss
    - **Stage 2** (130K iterations): Charbonnier loss + SSIM loss + Perceptual loss + GAN loss, with weights 1, 0.001, 1, and 0.005, respectively.
- **Degradation Synthesis**: Follows the random degradation pipeline of Real-ESRGAN (random combinations of blur, noise, JPEG compression, and video compression).
- **Training Details**: REDS dataset, 15-frame sequences, 64×64 cropping, batch size 16, 4× Quadro RTX 8000 GPUs, SPyNet frozen for the first 5K iterations.

## Key Experimental Results

### Main Results

| Method | Params(M) | Runtime(ms) | VideoLQ ILNIQE↓ | VideoLQ NRQM↑ | RealVSR ILNIQE↓ | RealVSR NRQM↑ | REDS4 PSNR↑ | REDS4 LPIPS↓ | UDM10 PSNR↑ | UDM10 LPIPS↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RealSR | 16.7 | 180 | 26.63 | 6.054 | 32.81 | 5.610 | 22.02 | 0.5991 | 25.37 | 0.4811 |
| Real-ESRGAN| 16.7 | 196 | 27.97 | 6.057 | 31.93 | 6.245 | 21.56 | 0.3533 | 24.96 | 0.3395 |
| BSRGAN | 16.7 | 180 | 27.49 | 6.156 | 32.65 | 6.152 | 22.94 | 0.3766 | 25.97 | 0.3388 |
| RealBasicVSR| 6.3 | 73 | 25.98 | 6.306 | 30.37 | 6.582 | 23.09 | 0.2991 | 25.96 | 0.3209 |
| **RealViformer** | **5.3** | **49** | **25.94** | **6.338** | **28.61** | **6.588** | **23.34** | **0.2877** | **26.42** | **0.3063** |

### Ablation Study

| Method | CAF | ICA | VideoLQ NRQM↑ | UDM10 LPIPS↓ |
|:---|:---:|:---:|:---:|:---:|
| Sp-baseline (Spatial Attention) | - | - | 6.061 | 0.3482 |
| Ch-baseline (Channel Attention) | ✗ | ✗ | 6.181 | 0.3085 |
| RealViformer⁻ | ✓ | ✗ | 6.196 | 0.2933 |
| **RealViformer** | **✓** | **✓** | **6.338** | **0.2877** |

### Key Findings

- **Channel attention is more robust to degradation**: Cosine similarity experiments show that channel attention outputs change minimally under blur/noise/compression (0.98-0.99), whereas spatial attention outputs change significantly (0.75-0.92). This stems from channel attention computing covariance over a global spatial range $\mathbb{R}^{1 \times HW}$.
- **Channel attention causes feature redundancy**: The cross-channel covariance of channel attention outputs is $ac(O) = 0.87$, which is much higher than the input ($\approx 0.15$) and spatial attention outputs. In standard VSR, the SSIM of channel attention is lower than that of spatial attention (0.8338 vs. 0.8432).
- **ICA effectively reduces redundancy**: CAF+ICA reduces the channel correlation of propagated information from 0.436 to 0.422, with enhanced power in high-frequency components (RPS analysis).
- **Significant advantages in parameters and speed**: 5.3M parameters (vs. RealBasicVSR's 6.3M), 49ms runtime (vs. RealBasicVSR's 73ms), being overall faster and lighter.
- **User study validation**: In ratings by 30 evaluators across 85 frames, RealViformer outperforms all compared methods in MOS.

## Highlights & Insights

1. **Systematic analytical paradigm**: Instead of directly proposing a new architecture, this work first thoroughly analyzes the behavioral differences between spatial and channel attention under degradation conditions with controlled experiments as quantitative evidence, before designing the model based on these findings. This approach is highly persuasive.
2. **Revealing the redundant nature of channel attention**: This finding has a broad impact on the low-level vision Transformer community, given the widespread use of channel attention in methods like Restormer.
3. **Simple yet effective solution**: ICA introduces only two simple modifications (squeeze-excite and attention map weight prediction) without requiring complex structural designs.
4. **Fewer parameters + faster speed + better performance**: Achieving all three simultaneously is relatively rare in the SR field.

## Limitations & Future Work

1. **Unidirectional recurrent framework**: Only forward propagation is utilized, leaving backward information unused. A bidirectional framework may further enhance performance.
2. **Flow estimation dependency on SPyNet**: SPyNet is lightweight but limited in accuracy. Better optical flow or deformable alignment could improve temporal aggregation.
3. **Fixed squeeze factor of 4 in ICA**: Different levels may require different compression ratios; adaptive strategies could be explored.
4. **Validation limited to ×4 SR**: Results for ×2 or ×8 are not reported.
5. **Limited real-world no-reference evaluation metrics**: Although ILNIQE/NRQM are better than NIQE, they are still imperfect, lacking a more robust perceptual quality assessment.

## Related Work & Insights

- **Restormer [Zamir et al., CVPR 2022]**: Proposed channel attention for image restoration; this work identifies its channel redundancy issue and presents an improvement.
- **RealBasicVSR [Chan et al., CVPR 2022]**: CNN SOTA for RWVSR; this work adopts its training strategy but replaces the CNN with a Transformer.
- **VICReg [Bardes et al., 2021]**: Variance-invariance-covariance regularization in self-supervised learning, inspiring the quantitative analysis of channel covariance in this paper.
- **SE-Net [Hu et al., CVPR 2018]**: The original proponent of the squeeze-and-excite mechanism; this work makes distinctive improvements in two key aspects.
- **Inspirations for attention in video super-resolution**: The complementary concept of using channel attention for temporal aggregation (limiting artifact propagation) and spatial attention for spatial detail reconstruction can be further explored.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core contribution lies in systematic analytical findings rather than architectural innovation, revealing the advantages and redundancy issues of channel attention in RWVSR. The analytical paradigm is highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The experimental system is highly complete, including sensitivity analysis, covariance quantification, multi-dataset comparisons, ablation studies, user studies, and RPS analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Fluently written with a clear logical flow from identifying problems to solving them, featuring a well-structured progression: exploration $\to$ finding $\to$ verification $\to$ model design.
- **Value**: ⭐⭐⭐⭐ Inspiring for both RWVSR and low-level vision Transformer design; the findings on channel attention redundancy can be generalized to more tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Kalman-Inspired Feature Propagation for Video Face Super-Resolution](kalman-inspired_feature_propagation_for_video_face_super-resolution.md)
- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](../../CVPR2025/video_generation/videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](../../CVPR2025/video_generation/patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2026\] Compressed-Domain-Aware Online Video Super-Resolution](../../CVPR2026/video_generation/compressed-domain-aware_online_video_super-resolution.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)

</div>

<!-- RELATED:END -->
