---
title: >-
  [Paper Note] Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal
description: >-
  [CVPR 2026][Image Generation][veiling glare removal] This paper proposes the VeilGen + DeVeiler framework, which employs a physics-guided Stable Diffusion generative model to learn latent transmission and glare maps for…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "veiling glare removal"
  - "simplified optical systems"
  - "Stable Diffusion"
  - "physics-guided generation"
  - "invertible restoration"
date: 2026-05-08
content_hash: 19a6f6b43903b440
---

# Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal

**Conference**: CVPR 2026
**arXiv**: [2511.17353](https://arxiv.org/abs/2511.17353)
**Code**: [GitHub](https://github.com/XiaolongQian/DeVeiler)
**Area**: Image Generation
**Keywords**: veiling glare removal, simplified optical systems, Stable Diffusion, physics-guided generation, invertible restoration

## TL;DR

This paper proposes the VeilGen + DeVeiler framework, which employs a physics-guided Stable Diffusion generative model to learn latent transmission and glare maps for synthesizing realistic compound-degradation training data. A restoration network trained under invertible constraints jointly removes aberrations and veiling glare in simplified optical systems.

## Background & Motivation

1. **Compound degradation in simplified optical systems**: Compact optical systems such as singlet lenses and metalens not only suffer from aberration blur, but also produce veiling glare due to non-ideal optical surfaces and coatings, leading to global contrast reduction. Existing computational aberration correction (CAC) methods cannot address this compound degradation.
2. **Lack of real paired data**: Physically accurate glare simulation requires complete opto-mechanical models and non-sequential ray tracing, which are computationally prohibitive for large-scale training data generation.
3. **Incompatibility of existing methods**: Dehazing methods are based on the atmospheric scattering model (depth-dependent), which is incompatible with intra-lens scattering (depth-independent); lens flare removal methods target structured artifacts (bright spots/streaks) and cannot handle diffuse veiling glare.
4. **Black-box issue of generative models**: Existing SD-based degradation generation methods lack physical constraints, resulting in unstable generation quality and insufficient meaningful guidance signals for restoration networks.

## Method

### Problem Formulation

The degradation model decomposes the observed image into two sequential steps — aberration and glare:

$$I_{de}^{p} = \underbrace{(I_c^{p} \otimes K^{p})}_{I_{ab}^{p}} \cdot T^{p} + I_g^{p}$$

where $K^p$ is the spatially-varying point spread function (PSF), $T^p$ is the transmission map (describing contrast attenuation), and $I_g^p$ is the glare map. The restoration objective is to recover $I_c$ from $I_{de}$, which constitutes a highly ill-posed blind inverse problem.

### Overall Architecture: Three-Stage Physics-Guided Pipeline

#### Stage I: VeilGen — Physics-Guided Degradation Generation

Built upon Stable Diffusion v2-1, the core innovation is the embedding of physical priors into the diffusion generation process:

- **LOTGMP (Latent Optical Transmission and Glare Map Predictor)**: Predicts latent maps $c_{vg} = (z_{trans}, z_{glare})$ from the noisy latent $z_t$, target degradation latent $z_{de}^{\mathcal{T}}$, and timestep $t$.
- **VGIM (Veiling Glare Imposition Module)**: Modulates features using the predicted latent maps to simulate the forward degradation process.

A hybrid training strategy is adopted: the source domain (paired, glare-free) uses the fixed mapping $(\mathbf{1}, \mathbf{0})$, while the target domain (unpaired, compound-degraded) uses LOTGMP-predicted latent maps. The total loss is:

$$\mathcal{L}_{gen} = p \, \mathcal{L}_{\mathcal{S}} + (1-p) \, \mathcal{L}_{\mathcal{T}}$$

where $p = 0.3$ balances contributions from the two domains. After training, VeilGen generates paired datasets offline for restoration network training.

#### Stage II: DDN — Forward Model Distillation

Since the multi-step diffusion sampling of VeilGen is computationally prohibitive during training, it is distilled into a lightweight Distilled Degradation Network (DDN):

$$\mathcal{L}_{distill} = \|DDN(I_c, c_{vg}) - VeilGen(I_c, c_{vg})\|_1$$

The frozen DDN is employed in Stage III as a forward degradation model to provide invertibility constraint supervision.

#### Stage III: DeVeiler — Invertible Restoration Network

The core idea is to train the restoration network to learn the inverse mapping of degradation, rather than a black-box statistical mapping:

- **VG-Enc (Glare Encoder)**: Predicts latent maps $\hat{c}_{vg}$ from the degraded image.
- **VGCM (Veiling Glare Compensation Module)**: Performs inverse feature modulation using the predicted latent maps, forming a structurally symmetric counterpart to VGIM.

**Invertibility Constraint**: The frozen DDN applies degradation forward using the predicted maps $\hat{c}_{vg}$, requiring reconstruction of the observed degraded image:

$$\mathcal{L}_{rev} = \|DDN(I_c, \hat{c}_{vg}) - I_{de}\|_1$$

The total training objective is:

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_{rev} \mathcal{L}_{rev}$$

**Two-phase training**: Phase I pretrains on the source domain to establish an aberration correction baseline; Phase II fine-tunes on a mixed dataset of source domain and VeilGen-synthesized pairs to prevent overfitting.

### Key Designs

- The latent maps predicted by LOTGMP must participate in both the forward (VGIM) and inverse (VGCM) pathways to be effective. Unidirectional usage degrades performance due to domain adaptation issues.
- The invertibility constraint forces the latent maps to acquire physical interpretability — low-transmission and high-glare regions can be verified through visualization of feature maps.

## Key Experimental Results

### Screen-Compound Domain: Full-Reference Evaluation

| Method | Type | Screen-SL PSNR↑ | SSIM↑ | LPIPS↓ | Screen-MRL PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|---------|-------|--------|----------|-------|--------|
| SwinIR | Single-degradation | 18.18 | 0.686 | 0.298 | 19.34 | 0.722 | 0.354 |
| NAFNet | Single-degradation | 18.75 | 0.684 | 0.363 | 18.91 | 0.723 | 0.377 |
| DiffBIR | Single-degradation | 17.95 | 0.621 | 0.398 | 18.70 | 0.625 | 0.412 |
| SwinIR+Flare7K++ | Cascaded | 21.67 | 0.723 | 0.297 | 20.74 | 0.745 | 0.336 |
| QDMR | Domain adaptation | 18.45 | 0.681 | 0.291 | 20.67 | 0.725 | 0.315 |
| **DeVeiler (Ours)** | **Domain adaptation** | **22.38** | **0.729** | **0.261** | **21.57** | **0.746** | **0.301** |

DeVeiler achieves a PSNR gain of +0.71 dB and LPIPS improvement of 12.1% on Screen-SL compared to the strongest competing method (SwinIR+Flare7K++).

### Realworld-Compound Domain: No-Reference Evaluation

| Method | Real-SL (CLIPIQA↑/Q-Align↑/NIQE↓) | Real-MRL (CLIPIQA↑/Q-Align↑/NIQE↓) |
|------|-------------------------------------|--------------------------------------|
| SwinIR | 0.424 / 3.518 / 5.710 | 0.374 / 3.191 / 6.696 |
| SwinIR+DiffDehaze | 0.573 / 3.679 / 6.141 | 0.437 / 3.542 / 5.230 |
| QDMR | 0.405 / 3.864 / 4.773 | 0.376 / 3.337 / 5.509 |
| **DeVeiler (Ours)** | **0.607** / **3.987** / **4.448** | **0.440** / **3.586** / **5.296** |

In real-world scenarios without ground truth, DeVeiler maintains leading performance on most metrics, validating its generalization capability.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|------|-------|-------|--------|------|
| w/o LOTGMP | 20.82 | 0.708 | 0.273 | No physical prior guidance |
| LOTGMP w/o SD prior | 21.39 | 0.708 | 0.268 | Missing diffusion model context |
| Full VeilGen | 21.56 | 0.712 | 0.264 | SD prior critical for generation quality |
| Unidirectional VGCM | 20.83 | 0.712 | 0.264 | Domain mismatch yields no gain |
| **Bidirectional VGIM/VGCM** | **22.38** | **0.729** | **0.261** | Invertibility constraint yields +0.82 dB |

Key finding: Unidirectional use of latent maps (concatenation or VGCM alone) is not only unhelpful but potentially harmful. Only the bidirectional consistency constraint via VGIM/VGCM effectively exploits physical priors.

## Highlights & Insights

- **Physics-guided rather than black-box generation**: Transmission and glare maps from the scattering model are explicitly embedded into the SD generation process, endowing synthesized data with physical meaning.
- **Elegant design of invertibility constraints**: Structural symmetry between VGIM/VGCM and cycle consistency via the frozen DDN unifies restoration and degradation modeling as mutually inverse operations.
- **Strong interpretability**: Internal feature maps of VGCM can be visualized to verify localization and suppression of low-transmission and high-glare regions.
- **High practical relevance**: Validated on two real optical systems — singlet lens and metalens — covering AR/VR and mobile photography application scenarios.
- **Efficient inference**: Inference latency is an order of magnitude lower than DiffBIR (66s) or DiffDehaze (92s).

## Limitations & Future Work

- **Dependence on source-domain paired data**: The framework still requires screen-captured aberration–clean paired data as the source domain; fully unsupervised settings are not applicable.
- **Sequential degradation assumption**: Aberrations and glare are modeled as sequential degradations, whereas the actual physical process may be more complex (e.g., optical crosstalk).
- **Limited data scale**: The Screen-Compound test sets contain only 42 and 25 images respectively; statistical significance remains to be verified at larger scale.
- **Distillation loss**: DDN distillation of VeilGen inevitably introduces approximation errors, raising concerns about coverage of extreme glare scenarios.

## Rating

- ⭐⭐⭐⭐ Novelty: The bidirectional design of embedding physical scattering models into diffusion generation combined with invertible restoration is innovative.
- ⭐⭐⭐⭐ Value: Addresses real pain points in AR/VR and mobile photography with acceptable inference speed.
- ⭐⭐⭐ Experimental Thoroughness: Comparison across two optical systems and multiple baselines is thorough, but test set size is limited.
- ⭐⭐⭐⭐ Writing Quality: Problem formulation is clear, the three-stage framework is logically coherent, and ablation studies are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Latent Proxies for Controllable Single-Image Relighting](learning_latent_proxies_for_controllable_single-image_relighting.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[CVPR 2026\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)
- [\[NeurIPS 2025\] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](../../NeurIPS2025/image_generation/how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)

</div>

<!-- RELATED:END -->
