---
title: >-
  [Paper Note] FreeInit: Bridging Initialization Gap in Video Diffusion Models
description: >-
  [ECCV 2024][Video Generation][Video Diffusion Models] This work identifies a training-inference initialization discrepancy in video diffusion models (where low-frequency information leakage during training leads to temporally correlated initial noise, whereas uncorrelated Gaussian noise is used during inference). It proposes FreeInit, which bridges this gap by iteratively refining the spatiotemporal low-frequency components of the initial noise…
tags:
  - "ECCV 2024"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Noise Initialization"
  - "Temporal Consistency"
  - "Frequency Domain"
  - "Inference Strategy"
date: 2026-05-08
content_hash: b5e1698bbce31e98
---

# FreeInit: Bridging Initialization Gap in Video Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2312.07537](https://arxiv.org/abs/2312.07537)  
**Code**: [Project Page](https://tianxingwu.github.io/pages/FreeInit/)  
**Area**: Video Generation  
**Keywords**: Video Diffusion Models, Noise Initialization, Temporal Consistency, Frequency Domain, Inference Strategy

## TL;DR

This work identifies a training-inference initialization discrepancy in video diffusion models (where low-frequency information leakage during training leads to temporally correlated initial noise, whereas uncorrelated Gaussian noise is used during inference). It proposes FreeInit, which bridges this gap by iteratively refining the spatiotemporal low-frequency components of the initial noise, thereby significantly improving the temporal consistency of generated videos.

## Background & Motivation

Video generation based on diffusion models has achieved rapid progress. Most models are built on top of pre-trained image diffusion models (such as Stable Diffusion) and achieve video generation by incorporating temporal layers and training on large-scale video data. However, the generated videos still commonly suffer from **temporal inconsistency** and **unnatural dynamics**.

The authors conduct an in-depth study on the **noise initialization** of video diffusion models and discover a previously overlooked implicit training-inference discrepancy:

**Key Discovery 1: Information Leakage**
- The diffusion process cannot completely corrupt the clean latent into pure Gaussian noise, especially in the **low-frequency band**.
- The corruption rate of low-frequency components is much slower than that of high-frequency components (verifiable from both visualization and SNR analysis).
- At the final diffusion step $t=1000$, the noisy latent still contains a significant amount of low-frequency information from the input video.

**Key Discovery 2: Low-Frequency Dominates Inference Quality**
- Even if 80% of the high-frequency components are replaced, the generation results remain largely unchanged.
- This indicates that the low-frequency components of the initial noise **determine the overall distribution of the generated results**.

**Consequences of the Discrepancy**:
- During training: The initial noise is corrupted from real videos, where the low-frequency band retains temporal correlation.
- During inference: i.i.d. Gaussian noise is used, which has absolutely no temporal correlation.
- Result: Degraded inference quality, temporal inconsistency, and unnatural motion.

## Method

### Overall Architecture

FreeInit is an **inference-time sampling strategy** that requires no extra training or fine-tuning. The core pipeline is as follows:

1. Initialize Gaussian noise $\epsilon$.
2. Perform DDIM denoising sampling to obtain the clean latent $z_0$.
3. Re-diffuse using the original noise $\epsilon$ through the forward diffusion process to obtain $z_T$ (retaining low-frequency information).
4. Noise reinitialization: Fuse the low frequencies of $z_T$ with the high frequencies of a new random noise $\eta$.
5. Use the reinitialized noise $z_T'$ as the starting point for the next round of DDIM sampling.
6. Iterate the above process.

### Key Designs

#### 1. Frequency Analysis of Information Leakage

**SNR Frequency Distribution Analysis**: During the forward diffusion process, band-wise SNR analysis is conducted on the noisy latent $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$. The results show that:

- The SNR of low-frequency components (blue-green curve) decays extremely slowly.
- The SNR of high-frequency components (red curve) decreases rapidly.
- At $t=1000$, the SNR of the low-frequency band is even greater than 0 dB, implying severe information leakage.

**Verification of Low-Frequency Component Dominance**: Diffuse from a real video to obtain $z_T$ and gradually replace its high-frequency components with random Gaussian noise:
- Even when keeping only 20% of the original low-frequency information, the generated results are still close to the results of the complete $z_T$.
- A noticeable degradation in quality only occurs when all components are replaced with Gaussian noise.

#### 2. Denoise and Diffuse Loop

Starting from the initial Gaussian noise $\epsilon$:
1. Obtain an initial clean latent $z_0$ through DDIM sampling.
2. Re-diffuse through the DDPM forward process using the **same noise** $\epsilon$ (key detail!):

$$z_T = \sqrt{\bar{\alpha}_T} z_0 + \sqrt{1-\bar{\alpha}_T} \epsilon = \sqrt{\bar{\alpha}_T}(\text{DDIM}_{sample}(\epsilon)) + \sqrt{1-\bar{\alpha}_T} \epsilon$$

Why use the same noise $\epsilon$? Sampling a new random noise would introduce significant uncertainty in the mid-frequency band, disrupting the existing spatiotemporal correlation.

#### 3. Noise Reinitialization

Fuse the low frequencies of $z_T$ and the high frequencies of a random noise $\eta$ using a spatiotemporal frequency filter:

$$\mathcal{F}_{z_T}^L = \text{FFT}_{3D}(z_T) \odot \mathcal{H}$$
$$\mathcal{F}_\eta^H = \text{FFT}_{3D}(\eta) \odot (1 - \mathcal{H})$$
$$z_T' = \text{IFFT}_{3D}(\mathcal{F}_{z_T}^L + \mathcal{F}_\eta^H)$$

Where:
- $\text{FFT}_{3D}$ operates on both the temporal and spatial dimensions to capture spatiotemporal frequency information.
- $\mathcal{H}$ is a Gaussian Low-Pass Filter (GLPF) with a normalized cutoff frequency of $D_0 = 0.25$.
- Retains the low frequencies of $z_T$ (containing temporal correlation).
- Introduces the high frequencies of $\eta$ (providing randomness for visual details).

#### 4. Iterative Refinement

The above process can be executed iteratively multiple times:
- Each iteration: Low-frequency components gain improved spatiotemporal consistency through denoising.
- Simultaneously: High-frequency components gain flexibility through reinitialization.
- The initial noise progressively converges toward the training distribution.

By default, 4 FreeInit iterations are used.

### Loss & Training

FreeInit is a **purely inference-time method** and does not involve any training or fine-tuning. Key configurations:
- Low-pass filter type: Gaussian Low-Pass Filter (GLPF)
- Cutoff frequency: $D_0 = 0.25$ (normalized spatiotemporal frequency)
- Number of iterations: 4 by default
- Parameters of each model remain unchanged

## Key Experimental Results

### Main Results

Evaluated on three open-source models using UCF-101 and MSR-VTT prompts:

**Temporal Consistency** (DINO ↑):

| Method | UCF-101 | MSR-VTT |
|------|---------|---------|
| AnimateDiff | 85.24 | 83.24 |
| AnimateDiff+FreeInit | **92.01** | **91.86** |
| ModelScope | 88.16 | 88.95 |
| ModelScope+FreeInit | **91.11** | **93.28** |
| VideoCrafter | 85.62 | 84.68 |
| VideoCrafter+FreeInit | **89.27** | **88.72** |

FreeInit improves temporal consistency by **2.92 to 8.62** points across all models.

**Motion Quality**:

| Method | FVD ↓ | MS Diff. ↓ | DD Diff. ↓ |
|------|-------|---------|---------|
| AnimateDiff | 1340.96 | 7.33 | 20.2 |
| AnimateDiff+FreeInit | **1032.47** | **0.04** | **1.53** |
| ModelScope | 785.30 | 1.64 | 3.71 |
| ModelScope+FreeInit | **702.15** | **0.35** | 8.22 |
| VideoCrafter | 730.04 | 6.14 | 15.79 |
| VideoCrafter+FreeInit | **675.39** | **3.19** | **6.44** |

FVD is comprehensively improved, and the motion smoothness and dynamic levels are closer to real videos.

### Ablation Study

1. **Noise Reinitialization and Filter Selection**:

| Method | UCF-101 DINO ↑ | MSR-VTT DINO ↑ |
|------|---------------|----------------|
| AnimateDiff (w/o NR) | 86.77 | 85.18 |
| AnimateDiff (ILPF) | 87.53 | 86.17 |
| AnimateDiff (GLPF) | **92.01** | **91.86** |
| ModelScope (w/o NR) | 88.20 | 90.90 |
| ModelScope (GLPF) | **91.11** | **93.28** |
| VideoCrafter (w/o NR) | 86.09 | 87.11 |
| VideoCrafter (GLPF) | **89.27** | **89.33** |

**Key Conclusions**:
- Noise reinitialization is crucial.
- GLPF with a soft transition significantly outperforms ILPF (Ideal Low-Pass Filter) with a hard cutoff.
- Introducing moderate randomness in mid-to-low frequencies is important for quality improvement.

2. **Impact of Number of Iterations**:
    - Temporal consistency continuously improves with the number of iterations.
    - **The improvement is most significant in the 1st iteration** (which injects low-frequency information for the first time, substantially narrowing the training-inference gap).
    - Subsequent iterations yield diminishing improvements.

3. **Comparison with Equivalent Inference Steps**:
    - The temporal consistency of just 1 FreeInit iteration (totaling $2n$ steps) already exceeds that of a standard DDIM with $5n$ steps (2.5x time overhead).
    - This proves that FreeInit is not equivalent to simply increasing sampling steps—**a good start is more important than struggling against a bad initial state**.

### Key Findings

1. **The training-inference initialization gap is one of the fundamental causes affecting video generation quality.**
2. **Low-frequency components determine the global distribution**: The generation results of video diffusion models are primarily determined by the low-frequency components of the initial noise.
3. **Using the original noise for re-diffusion is critical**: New noise would introduce uncertainty in the mid-frequencies.
4. **GLPF outperforms ILPF**: Soft-transition filters allow moderate variations in mid-to-low frequencies, avoiding excessive restriction.
5. **The method has broad applicability**: It is not limited to video diffusion models, but also applicable to image diffusion models such as SDXL.

## Highlights & Insights

1. **Precise Problem Definition**: This work is the first to systematically study the initial noise of video diffusion models in the frequency domain, attributing the vague "poor generation quality" to a concrete "training-inference initialization gap."
2. **Extremely Simple Method**: The entire algorithm only involves FFT, DDIM sampling, and forward diffusion. It requires no modification to the model or training, and can be integrated with just a few lines of code.
3. **"A Good Start" Philosophy**: Unlike many works focusing on improving the denoising process, this paper emphasizes the importance of the initial state—making the initial noise closer to the training distribution is more effective than struggling during the denoising process.
4. **Highly Universal**: No hyperparameter tuning is required for different models (GLPF with $D_0=0.25$ works universally), making it truly plug-and-play.

## Limitations & Future Work

1. **Increased Inference Time**: The default 4 iterations mean a 5x increase in sampling time. Although more efficient than simply increasing steps, it remains a bottleneck.
2. **Coarse-to-Fine Sampling Strategy**: The authors discuss in the supplementary materials that the efficiency issue can be mitigated using a coarse-to-fine strategy (low resolution first, then high resolution).
3. **Rigorous Low-Frequency Band Definition**: Currently, a fixed $D_0=0.25$ is used, and adaptive frequency partitioning strategies have not been explored.
4. **Reliance on DDIM Determinism**: The method is built on the foundation of deterministic sampling, and its applicability to stochastic sampling has not been fully discussed.

## Related Work & Insights

- **PYoCo (ICCV 2023)**: Also focuses on video noise priors but requires large-scale fine-tuning; FreeInit requires no training.
- **Common Diffusion Noise Schedules (Lin et al.)**: Points out that noise schedules cannot completely corrupt information (in the image domain); FreeInit extends this finding to the video domain.
- **Resampled Diffusion (Lugmayr et al.)**: The resampling operation in inpainting tasks inspired the iterative refinement strategy.
- **Inspiration**: The initialization gap may also exist in other conditional generation tasks (e.g., 3D, audio), and similar frequency-domain refinement strategies are worth exploring.

## Rating

- **Novelty**: ★★★★☆ — Deep insights into the training-inference initialization gap; the frequency-domain refinement scheme is novel.
- **Value**: ★★★★★ — Zero-modification, plug-and-play, with significant and stable improvements, offering high deployment value.
- **Experimental Thoroughness**: ★★★★★ — Three models, two datasets, multidimensional metrics, and detailed ablations; extremely comprehensive.
- **Writing Quality**: ★★★★★ — Clear visualizations, rigorous frequency analysis, and fluent narrative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [\[NeurIPS 2025\] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](../../NeurIPS2025/video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)
- [\[ECCV 2024\] Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)
- [\[ECCV 2024\] MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [\[ECCV 2024\] Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)

</div>

<!-- RELATED:END -->
