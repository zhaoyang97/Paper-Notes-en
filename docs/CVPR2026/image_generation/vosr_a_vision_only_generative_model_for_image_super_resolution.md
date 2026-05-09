---
title: >-
  [Paper Note] VOSR: A Vision-Only Generative Model for Image Super-Resolution
description: >-
  [CVPR 2026][Image Generation][super-resolution] This paper proposes VOSR, the first work to demonstrate that a purely vision-trained generative super-resolution model can match or even surpass T2I pretrained-based methods. By leveraging visual semantic conditioning and a restoration-oriented guidance strategy, VOSR achieves high-quality SR at approximately 1/10 the training cost of T2I-based approaches.
tags:
  - CVPR 2026
  - Image Generation
  - super-resolution
  - vision-only
  - diffusion model
  - classifier-free guidance
  - one-step distillation
date: 2026-05-08
content_hash: 2346a97656936e99
---

# VOSR: A Vision-Only Generative Model for Image Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2604.03225](https://arxiv.org/abs/2604.03225)
**Code**: [https://github.com/cswry/VOSR](https://github.com/cswry/VOSR)
**Area**: Image Generation
**Keywords**: super-resolution, vision-only, diffusion model, classifier-free guidance, one-step distillation

## TL;DR

This paper proposes VOSR, the first work to demonstrate that a purely vision-trained generative super-resolution model can match or even surpass T2I pretrained-based methods. By leveraging visual semantic conditioning and a restoration-oriented guidance strategy, VOSR achieves high-quality SR at approximately 1/10 the training cost of T2I-based approaches.

## Background & Motivation

The generative image super-resolution field is currently dominated by methods built upon Text-to-Image (T2I) diffusion models (e.g., Stable Diffusion), which adapt pretrained T2I generators for restoration tasks. The authors argue that this paradigm harbors a fundamental contradiction: SR is an image restoration task conditioned on low-resolution inputs, whereas T2I methods originate from general-purpose generators and introduce semantics via text or text-aligned representations, thereby increasing the risk of detail hallucination.

**Core Problem**: Can a vision-only generative model, without relying on multimodal pretraining, rival T2I-based SR methods?

The authors answer affirmatively through VOSR. With roughly 1/10 the training cost of representative T2I-based SR methods, VOSR achieves competitive or superior results in both perceptual quality and fidelity.

## Method

### Overall Architecture

VOSR is built on the LightningDiT backbone and trained with flow matching in latent space. Given an LR image, two complementary conditions are constructed: a structural condition (VAE-encoded LR latent representation) and a visual semantic condition (high-level features extracted by a DINO encoder). Both conditions are injected into a DiT for HR prediction.

### Key Designs

1. **Visual Semantic Conditioning**: Unlike prior vision-only SR methods that rely solely on LR structural conditions, VOSR introduces a DINO-pretrained visual encoder to extract semantic features. The structural condition is injected via spatially aligned latent conditioning to preserve fidelity, while the semantic condition provides high-level context through cross-attention. The two are complementary—structure ensures fidelity, semantics resolve ambiguity. Critically, the semantics remain entirely within the visual domain, avoiding the spatial coarse-graining problem inherent in text-aligned conditions.

2. **Restoration-Oriented Guidance**: The authors revisit the application of classifier-free guidance (CFG) in vision-only restoration and find that the standard unconditional branch is unsuitable for SR models trained from scratch. They propose replacing the unconditional branch with a partial-condition branch—retaining weakened LR structural cues while removing the semantic condition. This keeps both branches anchored to the input, with the guidance direction moving from weak to strong anchoring. A notable behavioral inversion emerges: increasing the guidance scale leads to higher fidelity (approaching the fully conditioned branch), while decreasing it enhances generative capacity (approaching the partial-condition branch).

3. **One-Step Distillation**: The multi-step VOSR teacher is distilled into a single-step student model, preserving the same conditioning interface and restoration-oriented guidance while improving sampling efficiency. A recursive consistency distillation variant is employed to achieve the optimal balance between perceptual quality and structural fidelity.

### Loss & Training

The multi-step model uses a standard velocity-parameterized diffusion training objective. During training, the model randomly switches between fully conditioned and partially conditioned modes. Approximately 100 million web images are used for training, with LR–HR pairs synthesized via the Real-ESRGAN degradation pipeline. Two model size variants are provided: 0.5B and 1.4B parameters.

## Key Experimental Results

### Main Results

| Dataset | Setting | Ours (VOSR-1.4B-ms) | Prev. SOTA (SeeSR) | Notes |
|--------|------|-------------------|-----------------|------|
| RealSR | Multi-step | Strongly competitive on perceptual metrics | One of the compared methods | VOSR outperforms on fidelity metrics |
| ScreenSR | Multi-step | Best on multiple metrics | — | Newly constructed real-world test set |
| LSDIR | Single-step | Surpasses OSEDiff and others | — | Single-step efficiency comparable to T2I single-step methods |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Without visual semantic condition | Perceptual quality degrades | Semantic conditioning is critical for disambiguation |
| Standard CFG (fully unconditional) | Poor performance | Unconditional branch is difficult to learn; guidance direction is unsuitable for restoration |
| Restoration-oriented guidance | Best overall | Partial-condition branch maintains input anchoring |

### Key Findings

- A vision-only framework for the first time achieves perceptual quality competitive with T2I-based SR, while exhibiting superior fidelity and fewer hallucinations.
- The multi-step model is substantially more efficient than existing T2I-based SR systems; the single-step model matches the efficiency of the latest single-step T2I systems.
- Training cost is approximately 1/10 that of representative T2I-based methods.

## Highlights & Insights

- Fundamentally challenges the necessity of T2I pretraining for SR, providing compelling counter-evidence.
- The restoration-oriented guidance strategy is elegantly designed; the behavioral inversion of the guidance scale (large scale → fidelity, small scale → generation) is a particularly intriguing finding.
- The newly constructed ScreenSR real-world paired test set offers higher-quality references for SR evaluation.
- Demonstrates that strong semantics can be obtained entirely within the visual domain without textual mediation.

## Limitations & Future Work

- Large-scale data and compute are still required for training, albeit significantly less than T2I-based methods.
- Under certain extreme degradations, the method may still fall short of the strong priors provided by T2I approaches.
- The visual encoder (DINO) itself also requires large-scale data for pretraining.

## Related Work & Insights

- Compared to prior vision-only SR methods such as ResShift and SinSR, VOSR achieves substantially higher perceptual quality.
- The restoration-oriented guidance strategy is generalizable to other image restoration tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to demonstrate that vision-only models can rival T2I-based SR.
- Technical Depth: ⭐⭐⭐⭐⭐ — Guidance strategy is carefully designed with thorough theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-scale evaluation, multi-step/single-step comparison, and a new test set; highly comprehensive.
- Value: ⭐⭐⭐⭐⭐ — Low training cost and high efficiency make it highly practical.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_processaware_online_alignment_for_generative.md)
- [\[CVPR 2026\] AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution](alignvar_towards_globally_consistent_visual_autoregression_for_image_super-resol.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)
- [\[AAAI 2026\] GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution](../../AAAI2026/image_generation/gewdiff_geometric_enhanced_wavelet-based_diffusion_model_for_hyperspectral_image.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)

<!-- RELATED:END -->
