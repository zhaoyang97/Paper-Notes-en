---
title: >-
  [Paper Note] FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases
description: >-
  [ICCV 2025][talking head generation] FixTalk is proposed as a framework that addresses identity leakage in GAN-based talking head generation through two lightweight plug-and-play modules — the Enhanced Motion Indicator (EMI) and the Enhanced Detail Indicator (EDI). EMI eliminates identity information from motion features to suppress identity leakage, while EDI repurposes the leaked identity information to compensate for missing details under extreme poses…
tags:
  - "ICCV 2025"
  - "talking head generation"
  - "identity leakage"
  - "rendering artifacts"
  - "GAN"
  - "motion decoupling"
date: 2026-05-08
content_hash: 6770405b9bd3adcf
---

# FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases

**Conference**: ICCV 2025
**arXiv**: [2507.01390](https://arxiv.org/abs/2507.01390)  
**Code**: None  
**Area**: Others (Talking Head Generation)
**Keywords**: talking head generation, identity leakage, rendering artifacts, GAN, motion decoupling

## TL;DR

FixTalk is proposed as a framework that addresses identity leakage in GAN-based talking head generation through two lightweight plug-and-play modules — the Enhanced Motion Indicator (EMI) and the Enhanced Detail Indicator (EDI). EMI eliminates identity information from motion features to suppress identity leakage, while EDI repurposes the leaked identity information to compensate for missing details under extreme poses, thereby removing rendering artifacts.

## Background & Motivation

Modern talking head generation must satisfy three objectives: system efficiency (real-time inference), decoupled control (independent control of lip sync, head pose, and emotional expression), and high-quality rendering. Meeting the first two objectives necessitates the use of GANs rather than diffusion models; however, GAN-based approaches suffer from two persistent problems:

**Identity Leakage (IL)**: Identity information from the driving image (e.g., face shape) bleeds into the source image, altering the appearance of the source subject.

**Rendering Artifacts (RA)**: Visible artifacts appear under extreme poses and exaggerated expressions.

Two key observations motivate this work: (1) identity leakage originates from identity information embedded within motion features; and (2) in self-driven scenarios, the leaked identity information can actually aid in recovering missing details. This insight inspires the core mechanism of "taming identity leakage."

## Method

### Overall Architecture

FixTalk is built upon EDTalk (a state-of-the-art GAN-based talking head model), preserving its encoder–decoder structure and Face Decoupling Module (FDM). EMI and EDI are introduced as additions to address identity leakage and rendering artifacts, respectively. The generation process is formulated as:

$$I^g = G(f^s + z^d_{\text{FDM}}, \tilde{f^s_\pi})$$

where $z^d_{\text{FDM}}$ denotes the EMI-enhanced motion features and $\tilde{f^s_\pi}$ denotes the EDI-enhanced multi-scale features.

### Key Designs

1. **Enhanced Motion Indicator (EMI)**: The primary objective is to disentangle identity information from the encoder's fourth-layer features $f_4^d$ — identified as carrying the most identity information — while retaining pure motion information. The design comprises:

    - A lightweight extractor $P$: composed of $N$ cross-attention layers and FFN layers, using learnable query vectors $q_l$ (inspired by Q-Former) to select motion-relevant features from $f_4^d$.
    - Multi-scale feature fusion: encoder features at each layer are processed via average pooling and aggregated through a weighted sum.
    - Decoupling loss: minimizes the cosine similarity between motion features of the source and driving images:

    $\mathcal{L}_{\text{dis}} = \max(0, \cos(z^s, z^d) - \xi)$

2. **Enhanced Detail Indicator (EDI)**: Leverages leaked identity information at inference time to compensate for missing details under extreme poses. The core design is a dual-memory network:

    - **Feature compressor $\Pi$**: compresses $f_4^{s,d} \in \mathbb{R}^{512 \times 32 \times 32}$ into compact tokens $f_\pi^{s,d} \in \mathbb{R}^{1 \times 512}$.
    - **Driving identity memory $M_d$**: stores compact tokens $f_\pi^d$ from driving images during training.
    - **Motion-source memory $M_{m-s}$**: stores combinations of the motion difference $z_s^d = z^d - z^s$ and source features $f_\pi^s$, enabling cross-identity queries at inference time.
    - KL divergence is applied to align the address distributions of the two memories: $\mathcal{L}_{\text{align}} = KL(\Omega_{m-s} \| \Omega_d)$.
    - At inference, source features combined with motion differences are used to query the driving identity memory, retrieving matched detail features.
    - A **decompressor $\Lambda$** and multi-head cross-attention (MHCA) fuse the retrieved features with the source feature space.

3. **Collaborative mechanism of the two modules**: EMI serves a preventive role by eliminating identity leakage in cross-driving scenarios, while EDI serves an exploitative role by extending the benefits of identity leakage observed in self-driven scenarios to cross-driving scenarios. The two modules are complementary and form a complete solution.

### Loss & Training

The total training loss is: $\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{per}} + \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{dis}} + \mathcal{L}_{\text{d-mem}} + \mathcal{L}_{\text{align}}$

Training data: VFHQ (16K+ high-quality speech clips) and MEAD (60 subjects × 8 emotional expressions), at 512×512 resolution and 25 fps. Both EMI and EDI are lightweight and plug-and-play in design.

## Key Experimental Results

### Main Results

**Video-driven comparison (HDTF dataset)**:

| Method | PSNR↑ | F-LMD↓ | FID↓ | CSIM↑ | NIQE↓ | CPBD↑ |
|--------|-------|--------|------|-------|-------|-------|
| DPE (GAN) | 26.078 | 1.232 | 23.126 | 0.567 | 42.96 | 0.183 |
| EmoPor (GAN) | 26.827 | 1.413 | 26.329 | 0.493 | 29.88 | 0.178 |
| EDTalk (GAN) | 26.504 | 1.111 | 13.172 | 0.594 | 42.41 | 0.221 |
| LivePor (GAN) | 27.054 | 1.119 | 12.883 | 0.568 | 15.93 | 0.244 |
| X-Por (Diffusion) | 22.884 | 1.498 | 46.552 | 0.505 | 28.61 | 0.236 |
| FYE (Diffusion) | 23.441 | 1.513 | 42.681 | 0.544 | 18.36 | 0.247 |
| **FixTalk** | **27.164** | **1.093** | **12.715** | **0.613** | **13.44** | **0.282** |

**Audio-driven comparison (MEAD dataset)**:

| Method | PSNR↑ | SSIM↑ | M-LMD↓ | F-LMD↓ | Acc_emo↑ | Sync_conf↑ |
|--------|-------|-------|--------|--------|----------|------------|
| SadTalker | 19.042 | 0.606 | 2.038 | 2.335 | 14.25 | 7.065 |
| AniTalker | 19.714 | 0.614 | 1.903 | 2.277 | 15.62 | 6.638 |
| Hallo (Diffusion) | 19.061 | 0.598 | 1.874 | 2.294 | 18.69 | 6.993 |
| EDTalk | 21.628 | 0.722 | 1.537 | 1.290 | 67.32 | 8.115 |
| **FixTalk** | **22.382** | **0.743** | **1.314** | **1.215** | **68.25** | 8.009 |

### Ablation Study

| Configuration | Identity Preservation | Artifacts | Analysis |
|---------------|-----------------------|-----------|----------|
| Baseline (no EMI+EDI) | ✗ Severe identity change | ✗ Visible artifacts | Validates problem existence |
| w/o EMI (EDI only) | ✗ Face shape biased toward driver | ✓ Artifacts reduced | EDI supplements details but does not resolve identity leakage |
| w/o EDI (EMI only) | ✓ Identity well preserved | ✗ Artifacts reappear | EMI resolves identity but does not supplement details |
| **Full FixTalk** | **✓ Identity well preserved** | **✓ Artifacts eliminated** | **Two modules are complementary** |

- FixTalk requires only 3.65 GB GPU memory and runs at 27.6 FPS, surpassing the real-time threshold of 25 FPS.
- User study (20 participants): motion consistency 4.21, identity preservation 4.47, image quality 4.19 — leading across all metrics.

### Key Findings

- The root cause of identity leakage lies in the encoder's fourth-layer features $f_4^d$, which jointly encode motion and identity information.
- In self-driven scenarios, identity leakage is in fact beneficial (effectively reducing the task to self-reconstruction); this counterintuitive finding serves as the design inspiration for EDI.
- Although diffusion models achieve high generation quality, they are substantially inferior to optimized GAN-based approaches in terms of system efficiency and controllability.
- The trade-off between the number of memory slots and storage efficiency warrants attention.

## Highlights & Insights

- **The "turning a liability into an asset" paradigm**: transforming identity leakage from a pure defect into an exploitable advantage represents a significant methodological innovation.
- **Rigorous feature-level analysis**: identity leakage is localized to $f_4^d$ through systematic intermediate variable substitution experiments, rather than attributed to vague causes.
- **Plug-and-play design**: EMI and EDI can be adapted to other GAN frameworks (e.g., FOMM, AniTalker).
- **Unified achievement of three objectives**: real-time efficiency (27.6 FPS), decoupled control, and high-quality rendering are simultaneously attained.

## Limitations & Future Work

- The capacity of the memory network (number of slots $S$) bounds performance; very large-scale identity repositories may require greater capacity.
- An additional Audio-to-Motion module is required for audio-driven generation.
- Validation is conducted only at 512×512 resolution; higher resolutions remain to be explored.
- Training on public datasets may limit competitiveness against diffusion models trained on large-scale proprietary data.

## Related Work & Insights

- EDTalk achieves facial dynamics decoupling via FDM but suffers from leakage; FixTalk builds upon it with targeted fixes.
- The learnable query mechanism of Q-Former is adopted in EMI to extract motion information from entangled features.
- The memory network mechanism is generalizable to other tasks requiring cross-identity feature transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The concept of "taming leakage" is highly creative, with rigorous logic from problem analysis to solution design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validation across both video-driven and audio-driven scenarios, with intuitive ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is well-structured; the variable substitution experiment in Fig. 3 is particularly convincing.
- **Value**: ⭐⭐⭐⭐ Real-time, high-quality talking head generation has direct commercial value for virtual avatar and digital human applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/others/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ECCV 2024\] HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images](../../ECCV2024/others/hiei_a_universal_framework_for_generating_high-quality_emerging_images_from_natu.md)
- [\[CVPR 2025\] Zero-Shot Head Swapping in Real-World Scenarios](../../CVPR2025/others/zero-shot_head_swapping_in_real-world_scenarios.md)
- [\[ICLR 2026\] Identity-Free Deferral For Unseen Experts](../../ICLR2026/others/identity-free_deferral_for_unseen_experts.md)
- [\[ICCV 2025\] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation](stroke2sketch_harnessing_stroke_attributes_for_training-free_sketch_generation.md)

</div>

<!-- RELATED:END -->
