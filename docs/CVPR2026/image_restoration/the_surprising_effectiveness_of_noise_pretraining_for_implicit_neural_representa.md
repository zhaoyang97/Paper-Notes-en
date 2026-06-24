---
title: >-
  [Paper Note] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations
description: >-
  [CVPR 2026][Image Restoration][Implicit Neural Representations] This paper discovers through systematic experimental analysis that pretraining INRs with unstructured noise (Uniform/Gaussian distributions) achieves a surprising ~80dB PSNR in image fitting, significantly outperforming all data-driven initialization methods. Meanwhile, noise with a spectral structure of $1/|f^\alpha|$, matching natural images, achieves the best balance between signal fitting and denoising…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Implicit Neural Representations"
  - "Noise Pretraining"
  - "Parameter Initialization"
  - "Signal Fitting"
  - "Denoising"
date: 2026-05-08
content_hash: e27923010a7ac798
---

# The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations

**Conference**: CVPR 2026  
**arXiv**: [2603.29034](https://arxiv.org/abs/2603.29034)  
**Code**: Yes (Project page public)  
**Area**: Image Restoration / Implicit Neural Representations  
**Keywords**: Implicit Neural Representations, Noise Pretraining, Parameter Initialization, Signal Fitting, Denoising

## TL;DR

This paper discovers through systematic experimental analysis that pretraining INRs with unstructured noise (Uniform/Gaussian distributions) achieves a surprising ~80dB PSNR in image fitting, significantly outperforming all data-driven initialization methods. Meanwhile, noise with a spectral structure of $1/|f^\alpha|$, matching natural images, achieves the best balance between signal fitting and denoising, matching SOTA data-driven initialization performance without requiring any real data.

## Background & Motivation

**Background**: Implicit Neural Representations (INRs) use MLPs to map spatial coordinates to signal values and are widely used in compression, inverse imaging, and neural rendering. The convergence performance of INRs depends heavily on parameter initialization strategies—data-driven methods (e.g., meta-learning, Strainer) have proven far superior to standard random initialization.

**Limitations of Prior Work**: Despite the significant effectiveness of data-driven initialization, the fundamental reason for its success remains unclear—is it encoding classic statistical signal priors, or more complex data-specific features? This lack of clarity limits applications in scenarios such as scientific imaging where domain data is scarce.

**Key Challenge**: Data-driven initialization requires prior ground-truth signals, but many application domains lack sufficient domain data. Understanding the fundamental mechanism behind initialization success might lead to efficient alternatives that do not rely on real data.

**Goal**: (1) Identify what level of signal characteristics drive the performance gains in data-driven INR initialization. (2) Determine if noise containing no real data can replace data-driven initialization. (3) Analyze how pretraining on different types of noise affects signal fitting and inverse problem solving differently.

**Key Insight**: Inspired by work on noise pretraining for visual classification networks, this paper pretrains INRs with different categories of noise. Comparative experiments reveal the underlying mechanisms of initialization success, as each noise category has precisely defined properties that serve as controlled variables for analysis.

**Core Idea**: By replacing real data with noise for INR pretraining, the paper finds that unstructured noise is the "king of signal fitting," while spectral noise is an "all-rounder," enabling efficient INR initialization without real data.

## Method

### Overall Architecture

The paper addresses "why data-driven initialization is useful and whether real data can be avoided" rather than "how to train INRs better." The experimental vehicle follows the Strainer INR framework: a 6-layer MLP (sine activation, $\omega=30$), where the first 5 layers are an encoder shared across all signals, and the last layer is an individual decoder head for each signal. During pretraining, the encoder is jointly trained on $N=10$ samples for 5000 steps to create a good "starting point." During testing, the encoder is frozen, a new decoder head is randomly initialized for a new signal, and it is trained for 2000 steps for fitting. The pipeline remains unchanged; only the pretraining data is replaced—real images are substituted with various types of noise. This variant is named Snp (Strainer Noise Pretraining).

### Key Designs

**1. Multiple Noise Categories as Controlled Variables: Decoupling "Initialization Priors"**

The reason data-driven initialization is hard to explain is that real images contain a mixture of low-level statistics, spectral structures, and semantic features. This paper uses noise with precisely known statistical properties as controlled variables: at one end is **unstructured noise** (Uniform $\mathcal{U}(0,1)$, Gaussian $\mathcal{N}(0.5,0.2)$), containing almost no spatial structure; at the other end is **structured noise**, including the Dead Leaves series and statistical models (Spectral $1/|f^\alpha|$, Spectral+Color, Wavelet Edge Models). The latter are generated by imposing natural image spectra or edge statistics on random noise. Since the "priors" in each noise type are predefined, comparing their effects allows for the attribution of convergence gains to specific signal characteristics.

**2. Simultaneous Assessment of Signal Fitting and Denoising: The Trade-off Between Capacity and Priors**

If only fitting PSNR is considered, one might conclude that "less structure in noise is better." However, this overlooks the INR's role as an inverse problem solver. The paper evaluates both dimensions, revealing a clear trade-off curve: unstructured noise pushes fitting to a shocking ~80dB PSNR but performs worst in denoising (23–24dB). Conversely, structured spectral noise $1/|f^\alpha|$ occupies the "sweet spot"—fitting performance approaches Strainer (56.4 vs 57.8 dB), and denoising approaches Siren (27.6 vs 28.3 dB). This suggests a fundamental trade-off in INR initialization: **functional capacity** (how fast/accurately it fits targets) and **deep prior strength** (inductive preference for clean signals) are inversely related.

**3. NTK and Local Complexity: Explaining Divergent Behaviors**

Neural Tangent Kernel (NTK) analysis shows that Snp:Uniform covers most of the target signal energy using very few eigenvalues, explaining its fast and accurate convergence. Local complexity analysis reveals that the initial layers of Snp:Uniform partition the input space into highly non-linear, nearly pseudo-random regions. Functionally, this is similar to the hash encoding in Instant-NGP: extremely high capacity but no structural prior, leading to unmatched fitting but poor denoising. In contrast, Snp:Spectrum and real-data-trained Strainer show nearly identical loss landscapes, explaining why spectral noise can replicate data-driven initialization without real data.

### Loss & Training

Both pretraining and fitting phases use L2 loss and the Adam optimizer ($lr=1e-4$). Pretraining lasts 5000 steps, and test fitting lasts 2000 steps. For denoising tasks, early stopping is used to select the optimal iteration and prevent the INR from overfitting to noise. Video experiments utilize the ResFields framework with learnable residual updates per frame for 100k steps; structured spectral noise remains effective here.

## Key Experimental Results

### Main Results

Image Fitting PSNR (dB), T=2000 steps:

| Method | CelebA-HQ | AFHQ | OASIS-MRI |
|------|----------|------|-----------|
| Siren | 44.9 | 45.1 | 53.0 |
| Strainer | 57.8 | 58.0 | 62.8 |
| TransINR | 51.9 | 49.0 | 55.5 |
| IPC | 49.7 | 47.2 | 51.4 |
| **Snp: Uniform** | **85.7** | **79.9** | **79.3** |
| Snp: Gaussian | 80.0 | 77.0 | 79.1 |
| Snp: Spectrum | 56.4 | 56.2 | 60.0 |

### Ablation Study

Denoising PSNR (dB) and Best Steps (CelebA-HQ):

| Method | PSNR | Best Steps |
|------|----------|---------|
| Siren | **28.3** | 139 |
| Strainer | 27.3 | 70 |
| Snp: Spectrum | 27.6 | 78 |
| Snp: Uniform | 23.0 | 73 |
| Snp: Gaussian | 23.8 | 70 |

Video Fitting and Denoising (Pexels, 100k steps):

| Task | Method | Mean PSNR |
|------|------|----------|
| Fitting | Vanilla ResFields | 29.5 |
| Fitting | **ResFields + Snp:Spectrum** | **31.1** |
| Denoising | Vanilla ResFields | 27.0 |
| Denoising | **ResFields + Snp:Spectrum** | **28.0** |

### Key Findings

- **Surprising Fitting Capacity of Unstructured Noise**: Snp:Uniform reaches 85.7dB, nearly 28dB higher than the strongest data-driven method, Strainer (57.8dB).
- **Fitting vs. Denoising Trade-off**: Unstructured noise is best for fitting but worst for denoising; spectral noise offers the best balance.
- **Spectral Noise ≈ Real Data**: $1/|f^\alpha|$ noise pretraining performance almost matches Strainer pretrained on real face data.
- **Analogy to Hash Encoding**: The functional geometry of Snp:Uniform resembles Instant-NGP's hash encoding, where high capacity stems from pseudo-random input space partitioning.
- **Different Trends in Video**: Spectral noise slightly outperforms uniform noise in video fitting, likely because spatio-temporal continuity benefits more from structured priors.

## Highlights & Insights

- **Strong Discovery-Driven Contribution**: Reveals the core mechanism of INR initialization through meticulously designed controlled experiments.
- **Clear Practical Value**: $1/|f^\alpha|$ noise is easy to generate and can completely replace real data for INR initialization—especially valuable for data-scarce scientific imaging.
- **Deep Trade-off Insight**: The inverse relationship between signal fitting capacity and deep prior strength is a critical takeaway for the INR community.
- **Cross-Domain Generalization**: Findings hold across images to videos and faces to MRI, indicating a general property of INRs rather than data-specific phenomena.
- **NTK/Loss Landscape Analysis**: Provides a theoretical explanatory framework rather than staying at empirical observation.

## Limitations & Future Work

- Only investigated sine activation functions and a fixed number of layers; conclusions might vary with other activations (ReLU, Gaussian, Wavelet).
- Did not explore the interaction between network depth and noise types.
- Denoising experiments were simple (Gaussian noise only); complex inverse problems (e.g., Super-Resolution, CT reconstruction) warrant verification.
- Could explore adaptive mixing of different noise types during pretraining to optimize both fitting and priors.

## Related Work & Insights

- Strainer proved the effectiveness of cross-domain INR initialization (Face → Animal → MRI); this paper proves real data is not even necessary.
- The "Looking at Noise" dataset provided a systematic classification of structured noise, forming the basis for the controlled experiments.
- The analogy between Snp:Uniform and Instant-NGP's hash encoding provides a unified perspective for understanding both methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The counter-intuitive discovery of noise pretraining is highly inspiring and challenges conventional wisdom in the INR community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple noise types, datasets, tasks, and interpretability analyses; the design is exceptionally systematic.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Smooth narrative, intuitive visualizations, and a complete logical chain from experiment to mechanism.
- **Value**: ⭐⭐⭐⭐ — Significant theoretical guidance for the INR community, with practical utility in data-scarce scientific settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SuperF: Neural Implicit Fields for Multi-Image Super-Resolution](../../ICLR2026/image_restoration/superf_neural_implicit_fields_for_multi-image_super-resolution.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] Convexity-Aware Noise Calibration: A Self-Supervised Framework for Noise-Level-Unknown Image Denoising](convexity-aware_noise_calibration_a_self-supervised_framework_for_noise-level-un.md)
- [\[CVPR 2026\] Event-Based Motion Deblurring Using Task-Oriented 3D Gaussian Event Representations](event-based_motion_deblurring_with_unpaired_data.md)
- [\[CVPR 2026\] Towards Generalized Representations for Low-Light Understanding: When Signal Constancy Meets Semantic Enrichment](towards_generalized_representations_for_low-light_understanding_when_signal_cons.md)

</div>

<!-- RELATED:END -->
