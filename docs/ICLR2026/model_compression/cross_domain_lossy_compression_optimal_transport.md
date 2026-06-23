---
title: >-
  [Paper Note] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport
description: >-
  [ICLR 2026][Model Compression][optimal transport] This work formalizes cross-domain lossy compression—where the encoder observes a degraded source and the decoder reconstructs a sample from a different target distribution—as an optimal transport problem under dual constraints of compression rate and classification loss. It derives closed-form DRC/RDC and DRPC tradeoff
tags:
  - ICLR 2026
  - Model Compression
  - optimal transport
  - rate-distortion theory
  - lossy compression
  - cross-domain
  - DRC tradeoff
  - DRPC
date: 2026-05-08
content_hash: 8f01b0afb9f78708
---
# Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport

**Conference**: ICLR 2026 (Oral)  
**OpenReview**: [https://openreview.net/forum?id=mUIGdUTtk2](https://openreview.net/forum?id=mUIGdUTtk2)  
**Code**: Available  
**Area**: Information Theory / Lossy Compression  
**Keywords**: optimal transport, rate-distortion theory, lossy compression, cross-domain, DRC tradeoff, DRPC

## TL;DR

This work formalizes cross-domain lossy compression—where the encoder observes a degraded source and the decoder reconstructs a sample from a different target distribution—as an optimal transport problem under dual constraints of compression rate and classification loss. It derives closed-form DRC/RDC and DRPC tradeoff functions for Bernoulli sources (Hamming distortion) and Gaussian sources (MSE). The theoretical predictions are validated through deep end-to-end compression models on super-resolution, denoising, and inpainting tasks, showing consistency between theory and experimental behavior.

## Background & Motivation

**Background**: Classical rate-distortion theory (Shannon 1959) assumes that the encoder and decoder operate within the same distribution domain. However, in practical scenarios, encoders often observe degraded inputs (noisy images, low-resolution images, or corrupted images), while decoders need to reconstruct samples from a clean target distribution.

**Limitations of Prior Work**:
- Classical RD theory does not handle cross-domain settings; the rate-distortion characteristics when source and target distributions differ lack a theoretical foundation.
- The Rate-Distortion-Perception (RDP) framework (Blau & Michaeli 2019) only considers perceptual constraints and does not explicitly model downstream classification tasks.
- Entropy-constrained optimal transport for cross-domain compression (Liu et al. 2022) does not incorporate classification or perceptual constraints and lacks closed-form solutions.
- Existing task-aware compression methods (Zhang 2023) only analyze RDC under single-domain settings.

**Key Challenge**: Compressed representations must simultaneously serve multiple objectives: (1) maintaining low-distortion reconstruction, (2) satisfying rate constraints, (3) preserving downstream classification information, and (4) maintaining perceptual quality. There is a fundamental tradeoff between these goals, and a unified theoretical analysis framework is lacking.

**Goal**: To establish a theoretical framework for cross-domain lossy compression and derive closed-form expressions for the fundamental tradeoff relationships between rate, distortion, classification, and perception.

**Key Insight**: The problem is formalized as an optimal transport problem with dual constraints (rate + classification). By utilizing shared common randomness, the stochasticity in one-shot settings is eliminated, allowing for the derivation of closed-form solutions on classically solvable distribution families.

**Core Idea**: A unified framework combining Optimal Transport + Rate Constraints + Classification Constraints, providing the first analytical expressions for DRC/DRPC tradeoffs in cross-domain settings.

## Method

### Overall Architecture

The paper addresses the theoretical characterization of "cross-domain lossy compression": the encoder observes a degraded input (noisy, low-resolution, or corrupted image $X$), while the decoder reconstructs a clean target $Y$ from another distribution. Classical rate-distortion theory is ill-equipped for this as it assumes identical distributions. The approach models the entire compression-reconstruction chain $S \to X \to Z \to Y$ (where $S$ is the class label and $Z$ is the compressed representation) as a **dual-constrained optimal transport problem**. While minimizing reconstruction distortion $E[d(X,Y)]$, it imposes a rate constraint $I(X;Z)\le R$ (limiting the information carried by $Z$) and a classification constraint $H(S|Y)\le C$ (limiting the loss of categorical information in the reconstruction). This integrates "small size, accurate recognition, and precise reconstruction" into a single analyzable optimization.

The work proceeds in three steps: First, the dual-constrained OT framework is established, using Fano's inequality to explain why $H(S|Y)$ measures classification performance. Second, closed-form solutions for the tradeoff functions are derived for two classically solvable distributions—Bernoulli sources (Hamming distortion) and Gaussian sources (MSE)—to reveal how distortion changes with rate and classification constraints. Finally, a perceptual divergence constraint is introduced to extend the framework into a four-dimensional Rate-Distortion-Classification-Perception (DRPC) tradeoff. By leveraging shared common randomness (shared between encoder and decoder), the stochastic transport plan in a one-shot setting can be reduced to a deterministic one, making closed-form derivation possible.

### Key Designs

**1. Rate-Classification Constrained Optimal Transport: Translating "Small and Accurate" into Solvable OT**

Classical RD theory fails in cross-domain scenarios. This work rewrites the problem as constrained optimal transport: finding a transport plan that minimizes reconstruction distortion $D(R,C) = \min_{P_{Z|X}, P_{Y|Z}} E[d(X,Y)]$ under two constraints—a rate constraint $I(X;Z) \leq R$ and a classification constraint $H(S|Y) \leq C$. Using $H(S|Y)$ to measure "recognition accuracy" is justified by Fano's inequality $\Pr(S \neq \hat{S}) \geq \frac{H(S|Y)-1}{\log(M-1)}$, which provides a lower bound on any classifier error. Thus, minimizing $H(S|Y)$ guarantees the existence of a high-accuracy classifier. Consequently, rate, distortion, and classification are locked into a unified optimization.

**2. Closed-form Solutions for Bernoulli and Gaussian Sources: Mapping the Tradeoff "Atomic" Models**

To ground the abstract framework, closed-form solutions are derived for classic solvable models. For Bernoulli sources with Hamming distortion, the binary symmetric channel structure with shared randomness simplifies the transport plan. For Gaussian sources with MSE, orthogonal decomposition separates rate, distortion, and classification into independent sub-optimizations. This yields expressions like $D(R,C) = \sigma_X^2 \cdot 2^{-2R} + f(C)$, where distortion is the sum of an exponentially decaying term relative to rate and a term $f(C)$ determined solely by classification constraints. These analytical solutions qualitatively reveal the tradeoff structure, providing a guide for algorithm design for complex natural images.

**3. DRPC Extension: Adding Perceptual Constraints for a Four-Dimensional Tradeoff**

In real image tasks, low pixel-wise distortion does not necessarily mean higher visual quality. Beyond DRC, a perceptual divergence constraint $D_\text{perc}(P_Y || P_{Y^*}) \leq P$ is added, requiring the KL divergence (or Wasserstein distance) between the reconstructed distribution $P_Y$ and target perceptual distribution $P_{Y^*}$ to be below $P$. This yields $D(R,C,P) = \min_{P_{Z|X},P_{Y|Z}} E[d(X,Y)]$ under triple constraints. This integrates rate, distortion, classification, and perception into a single tradeoff function, providing the first analytical characterization of the inverse relationship between perceptual quality and pixel-wise distortion.

### Loss & Training

The deep implementation utilizes a Lagrangian objective: $L = \text{MSE} + \lambda_r R + \lambda_p \text{Perception} + \lambda_c \text{CE}(S, \hat{S})$, where $R$ is estimated by an entropy model, Perception is implemented via a WGAN-GP discriminator, and CE represents classification loss. By sweeping the $(\lambda_r, \lambda_p, \lambda_c)$ grid, empirical $(R, C)$ pairs are measured on a validation set to trace the empirical DRC surface. Architecture: Convolutional Autoencoder + entropy model + WGAN-GP discriminator + classifier, trained on two RTX 3090 GPUs.

## Key Experimental Results

### Main Results: KODAK Denoising Comparison ($\sigma=25$ Gaussian noise)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | PI↓ |
|------|-------|-------|--------|--------|-----|
| JPEG-2K (non-learning) | 26.44 | 0.736 | 0.402 | 0.242 | 7.479 |
| BM3D (non-learning) | **31.88** | **0.869** | 0.224 | 0.164 | 2.650 |
| DeCompress (unsupervised) | 27.83 | 0.752 | 0.263 | 0.197 | 2.798 |
| OTDenoising (unsupervised) | 31.29 | 0.868 | **0.115** | **0.103** | **2.010** |
| **Ours** (unsupervised) | 27.90 | 0.804 | 0.199 | 0.164 | 2.167 |

### Ablation Study: Multi-task and Multi-dataset Validation

| Task | Dataset | Key Metric | Description |
|------|--------|---------|------|
| Super-resolution (4×) | MNIST | DRC Curve | Qualitative agreement with theory |
| Denoising ($\sigma=10$) | Mouse Nuclei | PSNR=33.03, SSIM=0.81 | Microscope image validation |
| Denoising (real) | SIDD | PSNR=33.61, SSIM=0.90 | Real smartphone noise |
| Denoising ($\sigma=20$) | SVHN/CIFAR-10/ImageNet | DRC/RDC Surface | Cross-dataset consistency |
| Inpainting | SVHN | Supervised+Unsupervised | Framework versatility validation |

### Key Findings

- **Theory-Experiment Alignment**: Empirical DRC curves across all datasets demonstrate the predicted qualitative behavior—distortion decreases monotonically with rate, and classification accuracy improves monotonically with rate.
- **Perception-Distortion Tradeoff Evidence**: The WGAN-GP discriminator allows the model to outperform BM3D and DeCompress on perceptual metrics (LPIPS, PI), though PSNR remains lower than BM3D—consistent with the theoretical perception-distortion tradeoff.
- **Impact of Classification Constraint**: Tightening the classification constraint (requiring higher accuracy) at a fixed rate leads to increased achievable distortion—validated by both theory and experiments.
- **Feasibility of Shared Randomness**: Implemented via a public PRNG seed, ensuring compatibility with broadcasting and write-once scenarios.

## Highlights & Insights

- **Elegant Unification**: Merges information theory, optimal transport, and classification into a single framework. The closed-form solutions provide both theoretical beauty and fundamental performance limits.
- **Naturalness of Cross-Domain Setting**: Most practical image processing tasks (denoising, SR, inpainting) are inherently cross-domain—with differing source and target distributions. This framework provides the first unified rate-distortion theory for these tasks.
- **Reviewer F3r6 Rating (10/10)**: Soundness 4/Presentation 4/Contribution 4 (all Excellent); recommended for acceptance as a highlight.
- **Bridging Role of Fano’s Inequality**: $H(S|Y)$ leverages Fano's' inequality to directly lower-bound classification error, creating an elegant link between information-theoretic quantities and classification performance.

## Limitations & Future Work

- Closed-form solutions are limited to Bernoulli/Gaussian distributions, whereas natural images are significantly more complex; the theory-practice gap requires further numerical exploration.
- PSNR metrics are lower than specialized denoisers like BM3D because the framework optimizes multiple objectives including rate and perception.
- Addressing initial reviewer concerns (Reviewer AfGP) regarding the relationship between $H(S|Y)$ and CE loss; while resolved experimentally, the behavior of $H(S|Y)$ in specific degradation corner cases warrants further clarification.
- Systematic comparison with the latest learned compression methods is currently lacking.

## Related Work & Insights

- **vs. Blau & Michaeli (2019)**: Their RDP framework considers the rate-distortion-perception tradeoff but lacks classification constraints and is limited to single-domain settings. This work extends it to a four-dimensional cross-domain tradeoff.
- **vs. Liu et al. (2022)**: Their work explores entropy-constrained OT for cross-domain compression but lacks classification/perception constraints and analytical solutions.
- **vs. Zhang (2023)**: Provides single-domain RDC analysis but does not handle cross-domain settings, shared randomness, or perceptual divergence.
- **vs. OTDenoising (Wang et al. 2023)**: Uses OT for unsupervised denoising but excludes rate and classification constraints. This work provides a more unified theoretical framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic closed-form framework for cross-domain rate-distortion theory; four-way unification of OT, rate, classification, and perception.
- Experimental Thoroughness: ⭐⭐⭐⭐ Combines theory with 5 datasets and 3 tasks (SR/Denoising/Inpainting), offering quantitative comparisons and additional SIDD/microscope data in rebuttal.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations, though high density affects accessibility.
- Value: ⭐⭐⭐⭐⭐ Significant theoretical contribution to information theory, establishing fundamental performance limits for cross-domain compression and image restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](../../AAAI2026/model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [\[ICLR 2026\] TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](turboquant_online_vector_quantization_with_near-optimal_distortion_rate.md)
- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)

</div>

<!-- RELATED:END -->
