---
title: >-
  [Paper Note] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking
description: >-
  [CVPR 2026][Image Generation][image watermarking] SLICE decomposes image semantics into four factors (subject / environment / action / detail), anchors each factor to a distinct spatial partition of the diffusion model's initial noise, and thereby enables fine-grained, semantic-aware watermarking—capable of not only detecting tampering but also precisely localizing which semantic factor has been altered, entirely without training.
tags:
  - CVPR 2026
  - Image Generation
  - image watermarking
  - diffusion model
  - semantic-aware
  - tamper localization
  - training-free
date: 2026-05-08
content_hash: e6a2de25ba6c04fc
---

# SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking

**Conference**: CVPR 2026
**arXiv**: [2603.12749](https://arxiv.org/abs/2603.12749)
**Code**: N/A
**Area**: Image Security / Digital Watermarking
**Keywords**: image watermarking, diffusion model, semantic-aware, tamper localization, training-free

## TL;DR
SLICE decomposes image semantics into four factors (subject / environment / action / detail), anchors each factor to a distinct spatial partition of the diffusion model's initial noise, and thereby enables fine-grained, semantic-aware watermarking—capable of not only detecting tampering but also precisely localizing which semantic factor has been altered, entirely without training.

## Background & Motivation
Initial-noise watermarking for diffusion models (e.g., Tree-Ring, Gaussian Shading, WIND) is the dominant training-free paradigm for image provenance, yet these methods embed noise patterns that are content-agnostic and can be bypassed by inversion-then-regeneration forgery attacks (LFA, RPM). SEAL improves resilience against such attacks by binding watermarks to image semantics via a single global semantic descriptor. However, the recent CSI attack demonstrates that an adversary can perform locally coherent semantic edits—altering only the action or subject—without disrupting global semantic consistency, thereby circumventing SEAL's verification. The root cause is that image semantics are not monolithic; they consist of multiple partially independent factors.

## Core Problem
How to design a fine-grained semantic watermarking scheme such that localized semantic edits (e.g., changing the subject or action) are not only detectable but also precisely attributable to a specific semantic dimension?

## Method

### Overall Architecture
The SLICE pipeline comprises three stages: (1) extracting four semantic factor descriptors from a reference image; (2) mapping each factor to a distinct spatial partition of the initial noise via a keyed hash; and (3) re-extracting semantics at detection time and performing partition-wise verification. SD V2 serves as the generative backbone, and Qwen3-VL is used as the VLM for semantic extraction.

### Key Designs
1. **Factorized Semantic Extraction**: A VLM with a carefully designed meta-prompt extracts four factors from the image: subject, environment, action, and detail. These four factors provide comprehensive and complementary coverage of image semantics. Empirically, Chinese-language prompts yield the highest semantic stability (cosine similarity approaching 1.0) and are adopted as the default.

2. **Spatially-Partitioned Semantic Injection**: The $h \times w$ latent grid is divided into four non-overlapping regions $\Omega_k$, each corresponding to one semantic factor. For each position $p \in \Omega_k$, the noise value is generated as $\mathbf{z}_T(p) = H(E(s_k), p, \sigma)$, where $E$ is the text encoder, $H$ is a keyed hash synthesis function, and $\sigma$ is the secret key. Consequently, the noise in each partition depends exclusively on its corresponding semantic descriptor, spatial position, and key.

3. **Three-State Detection**: At detection time, DDIM inversion is applied to the suspect image to obtain $\mathbf{z}_{inv}$; semantics are re-extracted and used to reconstruct the reference noise $\hat{\mathbf{z}}_T$; pixel-wise distances are then computed. The partition match rate $m_k$ and global match rate $m_g$ jointly determine one of three output states:

    - **State I** (intact and trustworthy): $m_g \geq \tau_g$ and all $m_k \geq \tau_k$
    - **State II** (localized semantic tampering): $m_g \geq \tau_g$ but some $m_k < \tau_k$; failing partitions directly indicate the tampered factors
    - **State III** (no watermark or severe corruption): $m_g < \tau_g$

### Theoretical Guarantees
- **Theorem 4.3**: Under bounded DDIM inversion error and semantic stability assumptions, untampered factors maintain high match rates $m_k \geq 1 - \beta_k - \gamma_k$, while tampered factors are upper-bounded by $m_k \leq 1 - (\rho_k - \beta_k)^+$.
- **Theorem 4.4**: For unwatermarked or keyless forged inputs, the false-acceptance probability decays exponentially: $P(\text{State I or II}) \leq \exp(-hw \cdot D_{KL}(\tau_g \| q))$.

## Key Experimental Results

| Detector (ASR%) | LFA | RPM | CSI |
|---|---|---|---|
| Gaussian Shading | 100 | 100 | 100 |
| Tree-Ring | 93.81 | 100 | 100 |
| WIND | 100 | 100 | 100 |
| SEAL | 0 | 7 | 81 |
| **SLICE** | **0** | **5** | **19** |

| Perturbation Type | Clean | Rotate | JPEG | Blur | Noise | Bright |
|---|---|---|---|---|---|---|
| Detection Accuracy | 1.000 | 1.000 | 0.990 | 0.988 | 0.993 | 0.941 |

- CLIP score on the SDP dataset: 33.034 (pre-injection) → 32.789 (post-injection), indicating negligible image quality degradation.
- Limitation: vulnerable to extreme crop-and-scale (C&S) attacks (AUC drops to 0.054), though combining with passive forensics mitigates this (AUC 0.85–0.99).

### Ablation Study
- A case study demonstrates the accuracy of three-state detection: all factor match counts for the original image exceed threshold 35; after replacing "girl" with "boy staring ahead," subject (55) and environment (75) remain above threshold, while action (26) and detail (33) fall below, correctly triggering State II.
- Chinese prompts achieve the highest stability across all four semantic dimensions; other languages (English, French, Spanish, Portuguese, German, Japanese) exhibit noticeable drift on the detail and action dimensions.

## Highlights & Insights
- The four-factor spatial partitioning design is particularly elegant: it transforms the global semantic binding problem into independent partition-level verification, converting localized edits into partition-level mismatches.
- The three-state detection mechanism goes beyond binary "watermark present/absent" verification to precisely localize "which semantic factor was tampered with"—a novel verification granularity in the watermarking literature.
- The method is entirely training-free and is supported by rigorous theoretical guarantees (exponential decay of false-acceptance probability).
- The finding that Chinese prompts yield the best semantic stability is notable, likely reflecting superior alignment with Qwen3-VL's pretraining.

## Limitations & Future Work
- Vulnerable to extreme geometric transformations (large-scale cropping/resizing); passive forensics are required as a complementary defense.
- Whether four semantic factors provide sufficient granularity remains an open question; scenarios involving multi-object interactions may require finer-grained partitioning.
- Relies on the stability of VLM-based semantic extraction—different VLM versions may introduce semantic drift.
- Designing geometry-invariant disentangled semantic embeddings is a promising direction for future work.

## Related Work & Insights
- **vs. SEAL**: SEAL employs a single global semantic binding and is defeated by the CSI attack (ASR 81%); SLICE's partition-wise binding reduces CSI's ASR to 19%.
- **vs. Tree-Ring / Gaussian Shading / WIND**: These content-agnostic methods are defeated by all forgery attacks (ASR ≈ 100%); SLICE's semantic binding provides fundamental protection.
- **vs. post-processing / fine-tuning watermarks**: SLICE requires no training and no modification of model parameters, offering flexible deployment.
- The partitioned verification paradigm is transferable to other fine-grained provenance scenarios (e.g., video watermarking, multimodal content authentication).
- The semantic factor disentanglement + spatial partitioning framework is extensible to more factors or hierarchical partitioning schemes.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Semantic partition embedding combined with three-state detection establishes a fundamentally new verification granularity in the watermarking field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated against three forgery attacks and five standard perturbations with clear qualitative case studies, though dataset scale is not fully detailed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical proofs are complete, the framework is described clearly, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Addresses a core vulnerability of semantic-aware watermarking, though susceptibility to extreme geometric transformations limits standalone deployment reliability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](morphany3d_unleashing_the_power_of_structured_latent_in_3d_morphing.md)
- [\[CVPR 2026\] gQIR: Generative Quanta Image Reconstruction](gqir_generative_quanta_image_reconstruc_tion.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
