---
title: >-
  [Paper Note] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport
description: >-
  [ICLR 2026 (Oral)][Model Compression][optimal transport] This paper formalizes cross-domain lossy compression — where the encoder observes a degraded source and the decoder reconstructs samples from a different target distribution — as an optimal transport problem subject to dual constraints on rate and classification loss. Closed-form DRC/RDC and DRPC tradeoff functions are derived for Bernoulli sources (Hamming distortion) and Gaussian sources (MSE). The theoretical predictions are validated against the empirical behavior of deep end-to-end compression models on super-resolution, denoising, and inpainting tasks.
tags:
  - ICLR 2026 (Oral)
  - Model Compression
  - optimal transport
  - rate-distortion theory
  - lossy compression
  - cross-domain
  - DRC tradeoff
  - DRPC
date: 2026-05-08
content_hash: 50d9ed96e87bbbb7
---

# Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport

**Conference**: ICLR 2026 (Oral)  
**OpenReview**: [mUIGdUTtk2](https://openreview.net/forum?id=mUIGdUTtk2)  
**Code**: Available  
**Area**: Information Theory / Lossy Compression  
**Keywords**: optimal transport, rate-distortion theory, lossy compression, cross-domain, DRC tradeoff, DRPC

## TL;DR

This paper formalizes cross-domain lossy compression — where the encoder observes a degraded source and the decoder reconstructs samples from a different target distribution — as an optimal transport problem subject to dual constraints on rate and classification loss. Closed-form DRC/RDC and DRPC tradeoff functions are derived for Bernoulli sources (Hamming distortion) and Gaussian sources (MSE). The theoretical predictions are validated against the empirical behavior of deep end-to-end compression models on super-resolution, denoising, and inpainting tasks.

## Background & Motivation

**Background**: Classical rate-distortion theory (Shannon 1959) assumes that the encoder and decoder operate within the same distribution domain. In practice, however, the encoder observes degraded inputs (noisy images, low-resolution images, corrupted images), while the decoder must reconstruct samples from a clean target distribution.

**Limitations of Prior Work**:
- Classical RD theory does not address cross-domain settings — the rate-distortion characteristics when source and target distributions differ lack a theoretical foundation.
- The Rate-Distortion-Perception (RDP) framework (Blau & Michaeli 2019) incorporates perceptual constraints but does not explicitly model downstream classification tasks.
- Entropy-constrained optimal transport for cross-domain compression (Liu et al. 2022) does not incorporate classification or perceptual constraints and admits no closed-form solution.
- Existing task-aware compression methods (Zhang 2023) analyze RDC only in the single-domain setting.

**Key Challenge**: Compressed representations must simultaneously serve multiple objectives — (1) low-distortion reconstruction, (2) rate constraints, (3) preservation of downstream classification information, and (4) perceptual quality — yet these objectives involve fundamental tradeoffs that lack a unified theoretical analysis framework.

**Goal**: Establish a theoretical framework for cross-domain lossy compression and derive closed-form expressions for the fundamental tradeoffs among rate, distortion, classification, and perception.

**Key Insight**: The problem is formalized as an optimal transport problem with dual constraints (rate + classification). Shared common randomness is leveraged to eliminate stochasticity in the one-shot setting, and closed-form solutions are derived for classical tractable distribution families.

**Core Idea**: A unified framework combining optimal transport, rate constraints, and classification constraints — providing, for the first time, analytic expressions for DRC/DRPC tradeoffs in the cross-domain setting.

## Method

### Overall Architecture

Given a degraded source $X$, target distribution $Y$, and class label $S$, the framework operates through the Markov chain $S \to X \to Z \to Y$, where $Z$ is the compressed representation. The objective is to jointly minimize distortion $E[d(X,Y)]$ subject to rate constraint $H(Z) \leq R$ and classification constraint $H(S|Y) \leq C$. Under shared common randomness, the one-shot setting reduces to a deterministic transport plan.

### Key Designs

1. **Rate- and Classification-Constrained Optimal Transport**:
    - **Function**: Formalizes cross-domain lossy compression as a constrained optimal transport problem.
    - **Mechanism**: Minimizes distortion $D(R,C) = \min_{P_{Z|X}, P_{Y|Z}} E[d(X,Y)]$ subject to $I(X;Z) \leq R$ (rate constraint) and $H(S|Y) \leq C$ (classification constraint). The term $H(S|Y)$ is linked to a lower bound on classification error via Fano's inequality: $\Pr(S \neq \hat{S}) \geq \frac{H(S|Y)-1}{\log(M-1)}$.
    - **Design Motivation**: $H(S|Y)$ is the information-theoretic natural measure of classification information — small $H(S|Y)$ guarantees the existence of a high-accuracy classifier. The rate constraint limits the information content of the compressed representation, and together these form a three-way tradeoff with distortion.

2. **Closed-Form Solutions for Bernoulli and Gaussian Sources**:
    - **Function**: Derives analytic expressions for DRC/RDC on two classical tractable distribution families.
    - **Mechanism**: For Bernoulli sources with Hamming distortion, the binary symmetric channel structure and shared randomness are exploited to simplify the transport plan. For Gaussian sources with MSE, an orthogonal decomposition separates the rate-distortion-classification problem into independent subproblems, yielding expressions of the form $D(R,C) = \sigma_X^2 \cdot 2^{-2R} + f(C)$.
    - **Design Motivation**: Bernoulli and Gaussian sources are the "hydrogen atom" models of rate-distortion theory. Their closed-form solutions reveal the qualitative structure of the tradeoffs and guide algorithmic design for more complex practical distributions.

3. **DRPC Extension (Adding a Perceptual Constraint)**:
    - **Function**: Augments the DRC framework with a perceptual divergence constraint (KL divergence or Wasserstein distance) to obtain the four-dimensional DRPC tradeoff function.
    - **Mechanism**: An additional constraint $D_\text{perc}(P_Y \| P_{Y^*}) \leq P$ is imposed, where $P_{Y^*}$ is the target perceptual distribution, yielding $D(R,C,P) = \min_{P_{Z|X},P_{Y|Z}} E[d(X,Y)]$ subject to the triple constraints on rate, classification, and perception.
    - **Design Motivation**: In practice, perceptual quality and per-pixel distortion are in tension — low distortion does not imply high perceptual quality. The DRPC framework handles this tradeoff in a unified manner.

### Loss & Training

The deep implementation adopts a Lagrangian objective: $L = \text{MSE} + \lambda_r R + \lambda_p \text{Perception} + \lambda_c \text{CE}(S, \hat{S})$, where $R$ is estimated by an entropy model, perception is implemented via a WGAN-GP discriminator, and CE denotes classification loss. A grid sweep over $(\lambda_r, \lambda_p, \lambda_c)$ is performed, and empirical $(R, C)$ pairs are measured on the validation set to trace the empirical DRC surface. The architecture consists of a convolutional autoencoder, an entropy model, a WGAN-GP discriminator, and a classifier, trained on two RTX 3090 GPUs.

## Key Experimental Results

### Main Results: KODAK Denoising ($\sigma=25$ Gaussian Noise)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | PI↓ |
|------|-------|-------|--------|--------|-----|
| JPEG-2K (non-learning) | 26.44 | 0.736 | 0.402 | 0.242 | 7.479 |
| BM3D (non-learning) | **31.88** | **0.869** | 0.224 | 0.164 | 2.650 |
| DeCompress (unsupervised) | 27.83 | 0.752 | 0.263 | 0.197 | 2.798 |
| OTDenoising (unsupervised) | 31.29 | 0.868 | **0.115** | **0.103** | **2.010** |
| **Ours** (unsupervised) | 27.90 | 0.804 | 0.199 | 0.164 | 2.167 |

### Ablation Study: Multi-Task, Multi-Dataset Validation

| Task | Dataset | Key Metric | Notes |
|------|--------|---------|------|
| Super-resolution (4×) | MNIST | DRC curve | Theoretical predictions qualitatively consistent with experiments |
| Denoising ($\sigma$=10) | Mouse Nuclei | PSNR=33.03, SSIM=0.81 | Validated on microscopy images |
| Denoising (real) | SIDD | PSNR=33.61, SSIM=0.90 | Real smartphone noise |
| Denoising ($\sigma$=20) | SVHN/CIFAR-10/ImageNet | DRC/RDC surfaces | Consistent across datasets |
| Inpainting | SVHN | Supervised + unsupervised | Validates generality of framework |

### Key Findings

- **Theory–Experiment Consistency**: Empirical DRC curves across all datasets exhibit the predicted qualitative behavior — distortion decreases monotonically with rate, and classification accuracy improves monotonically with rate.
- **Empirical Perception–Distortion Tradeoff**: The WGAN-GP discriminator enables the model to outperform BM3D and DeCompress on perceptual metrics (LPIPS, PI), while PSNR remains below BM3D — consistent with the theoretically predicted perception–distortion tradeoff.
- **Effect of Classification Constraint**: At a fixed rate, tightening the classification constraint (requiring higher accuracy) increases achievable distortion — validated by both theory and experiment.
- **Practical Feasibility of Shared Randomness**: Implemented via a public PRNG seed, compatible with broadcast and single-write scenarios.

## Highlights & Insights

- **Elegant Unification of Information Theory, Optimal Transport, and Classification**: Three distinct theoretical domains are integrated within a single framework. The closed-form solutions are not only theoretically elegant but also provide fundamental performance limits.
- **Naturalness of the Cross-Domain Setting**: Nearly all practical image processing tasks (denoising, super-resolution, inpainting) are inherently cross-domain — the source and target distributions differ. This framework provides, for the first time, a unified rate-distortion theory for these tasks.
- **Reviewer F3r6 Awarded a Score of 10**: Soundness 4 / Presentation 4 / Contribution 4, all rated Excellent, with a recommendation to accept as a highlight.
- **Bridging Role of Fano's Inequality**: $H(S|Y)$ directly lower-bounds classification error via Fano's inequality — an elegant connection between an information-theoretic quantity and classification performance.

## Limitations & Future Work

- Closed-form solutions are restricted to Bernoulli and Gaussian distributions; natural images are far more complex, and additional numerical methods are needed to bridge the gap between theory and practice.
- PSNR performance does not match dedicated denoising methods such as BM3D, as the proposed framework simultaneously optimizes for rate, perception, and other objectives.
- Reviewer AfGP initially assigned a score of 2 (subsequently revised to 6), with the core concern being the relationship between $H(S|Y)$ and the CE loss. Although the issue was ultimately resolved via experiments during rebuttal, the behavior of $H(S|Y)$ in certain degenerate corner cases warrants further clarification.
- A systematic comparison with state-of-the-art learned compression methods is absent.

## Related Work & Insights

- **vs. Blau & Michaeli (2019)**: Their RDP framework addresses the rate–distortion–perception tradeoff but excludes classification constraints and operates in the single-domain setting. This paper extends the framework to cross-domain settings with a four-dimensional rate–distortion–classification–perception tradeoff.
- **vs. Liu et al. (2022)**: Their work applies entropy-constrained OT to cross-domain compression but incorporates neither classification nor perceptual constraints and admits no closed-form solution.
- **vs. Zhang (2023)**: Analyzes RDC in the single-domain setting without addressing cross-domain scenarios, shared randomness, or perceptual divergence.
- **vs. OTDenoising (Wang et al. 2023)**: That work employs OT for unsupervised denoising but without rate or classification constraints. This paper provides a unified theoretical framework encompassing those settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic closed-form framework for cross-domain rate-distortion theory; four-dimensional unification of optimal transport, rate, classification, and perception.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Theory supported by 5 datasets, 3 task types (super-resolution / denoising / inpainting), quantitative comparison with baselines, and additional microscopy and SIDD results provided during rebuttal.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous, though the presentation is dense and accessibility is limited.
- **Value**: ⭐⭐⭐⭐⭐ — An important theoretical contribution to information theory, establishing fundamental performance limits for cross-domain compression and image restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](../../AAAI2026/model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)

</div>

<!-- RELATED:END -->
