---
title: >-
  [Paper Note] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement
description: >-
  [NeurIPS 2025][Image Restoration][UHD Image Restoration] This paper proposes Latent Harmony, a two-stage framework that constructs a generalizable VAE (LH-VAE) via latent space regularization, and introduces a high-frequency-guided controllable LoRA fine-tuning mechanism, achieving flexible fidelity-perceptual quality trade-offs in unified multi-degradation UHD image restoration while preserving structural integrity.
tags:
  - NeurIPS 2025
  - Image Restoration
  - UHD Image Restoration
  - VAE Regularization
  - High-Frequency LoRA
  - Fidelity-Perceptual Quality Trade-off
  - All-in-One
date: 2026-05-08
content_hash: b50707a111e2575b
---

# Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement

**Conference**: NeurIPS 2025
**arXiv**: [2510.07961](https://arxiv.org/abs/2510.07961)
**Code**: [GitHub](https://github.com/lyd-2022/Latent-Harmony)
**Area**: Image Restoration / Ultra-High-Definition Image Processing
**Keywords**: UHD Image Restoration, VAE Regularization, High-Frequency LoRA, Fidelity-Perceptual Quality Trade-off, All-in-One

## TL;DR

This paper proposes Latent Harmony, a two-stage framework that constructs a generalizable VAE (LH-VAE) via latent space regularization, and introduces a high-frequency-guided controllable LoRA fine-tuning mechanism, achieving flexible fidelity-perceptual quality trade-offs in unified multi-degradation UHD image restoration while preserving structural integrity.

## Background & Motivation

**State of the Field**: Ultra-high-definition (UHD/4K) image restoration must handle large data volumes and preserve fine details. All-in-One methods aim to process multiple degradation types with a single model. VAE-based latent space approaches can substantially improve efficiency, but they suffer from fundamental limitations.

**Limitations of Prior Work**:
   - Gaussian variational inference in VAEs favors semantic preservation at the cost of high-frequency detail, degrading reconstruction fidelity
   - All-in-One methods typically rely on degradation-aware branches (MoE/Prompt), increasing computational overhead and limiting generalization
   - Existing methods exhibit a fundamental tension between generalizability (across degradation types) and reconstruction capability
   - Jointly optimizing VAE and downstream restoration networks directly disrupts the structure of the pretrained latent space

**Root Cause**: A triple trade-off: (1) latent space generalizability vs. reconstruction fidelity, (2) VAE joint optimization vs. structural preservation, and (3) perceptual quality vs. fidelity in the output.

**Paper Goals**: To systematically address the multiple trade-off challenges posed by VAEs in UHD All-in-One restoration.

**Starting Point**: Frequency-domain analysis reveals that high-frequency information is central to the generalizability–reconstruction tension, motivating targeted regularization and LoRA fine-tuning strategies.

**Core Idea**: Build degradation-invariant generalizable representations through latent space regularization, and separately optimize encoder fidelity and decoder perceptual quality via differentiated high-frequency LoRA modules.

## Method

### Overall Architecture

**Two-Stage Synergistic Framework**:

- **Stage 1 (LH-VAE Training)**: Progressive degradation perturbation + visual semantic constraint + latent equivariance constraint → construction of a generalizable latent space
- **Stage 2 (Restoration + Controllable Refinement)**: Fix VAE to train the restoration network → high-frequency-guided controllable LoRA fine-tuning (encoder fidelity + decoder perception) → inference-time $\alpha$ parameter for trade-off control

### Key Designs

1. **Progressive Degradation Perturbation Strategy (PDPS)**:

    - Gradually intensified degradation perturbations are applied to clean images during training, with three modes weighted by probability:
    $I'_{deg} = \begin{cases} I_{clean} & \text{probability } p_0 \\ \text{SynthDeg}(I_{clean}, \text{sev}(t)) & \text{probability } p_1 \\ (1-\beta(t))I_{clean} + \beta(t)I_{deg} & \text{probability } p_2 \end{cases}$
    - $\text{sev}(t)$ and $\beta(t)$ increase monotonically with training time to ensure learning stability
    - Objective: enable the encoder to progressively learn degradation-insensitive representations

2. **Degradation-Invariant Visual Semantic Loss $L_{INV}$**:

    - Semantic features $f_{VFM}$ of clean images are extracted using a pretrained DINOv2 model
    - The encoder is constrained to align its encoding of perturbed images with these semantic references: $L_{Inv} = d(z'_{deg}, f_{VFM})$
    - This encourages the latent space to be organized by content rather than degradation type (verified via t-SNE)

3. **Latent Space Equivariance Constraint $L_{Eqv}$**:

    - Constrains the decoded result of a randomly downsampled latent code to be consistent with the corresponding downsampled image:
    $L_{Eqv} = \|D_\psi(z_{down}) - I_{down}\|_1$
    - Enhances scale robustness and reduces over-reliance on high-frequency components

4. **High-Frequency-Guided LoRA Fine-Tuning (HF-LoRA)**:

    - **FHF-LoRA (Fidelity-oriented, encoder side)**: Guided by a high-frequency alignment loss to accurately extract high-frequency structures from degraded inputs consistent with ground truth:
    $L_{HF_{Fid}} = \|\text{HF}(D_{\psi^*}(E_{\phi^* + \Delta\phi_{LoRA}}(I_{deg}))) - \text{HF}(I_{clean})\|_1$
    - **PHF-LoRA (Perception-oriented, decoder side)**: GAN-based high-frequency adversarial loss to generate visually natural high-frequency textures:
    $L_{HF_{GAN}} = -\mathbb{E}[\log D_{HF}(\text{HF}(D_{\psi^* + \Delta\psi_{LoRA}}(R_\theta(E_{\phi^*}(I_{deg})))))]$
    - The two LoRA parameter sets are trained via alternating optimization and selective gradient propagation to protect the pretrained latent structure

5. **Inference-Time Controllable Trade-off**: Parameter $\alpha \in [0,1]$ controls the fidelity-perceptual quality balance:
    $\phi = \phi^* + \alpha \cdot \Delta\phi_{LoRA}, \quad \psi = \psi^* + (1-\alpha) \cdot \Delta\psi_{LoRA}$

### Loss & Training

**Stage 1** joint optimization objective:
$$L_{Stage1} = L_{VAE} + \lambda_{Inv} L_{Inv} + \lambda_{Eqv} L_{Eqv}$$

where $L_{VAE}$ includes an L1 reconstruction loss and a KL divergence regularization term.

**Stage 2** step-by-step training:
1. Fix the VAE and train the restoration network $R_\theta$: $L_{Res} = \|D_{\psi^*}(z_{res}) - I_{clean}\|_1$
2. Alternately optimize FHF-LoRA and PHF-LoRA, each responding only to its corresponding high-frequency loss

## Key Experimental Results

### Main Results

UHD All-in-One restoration comparison across four degradation types:

| Method | Full-Res Inference | FLOPs | Params | Low-Light PSNR | Deblur PSNR | Dehaze PSNR | Denoise Avg. PSNR | Avg. PSNR | LPIPS ↓ |
|------|-----------|-------|------|------------|------------|----------|-------------|----------|---------|
| PromptIR | ✗ | 158G | 33M | 23.44 | 25.77 | 19.97 | 26.30 | 24.68 | .2571 |
| UHDprocesser | ✓ | 4G | 1.6M | 27.11 | 26.48 | 20.94 | 33.38 | 29.23 | .2541 |
| **Ours** | ✓ | **3.6G** | **1.2M** | **27.32** | **26.98** | **21.21** | **34.24** | **29.70** | **.2502** |

UHD All-in-One restoration across six degradation types: average PSNR **29.24 dB** (surpassing UHDprocesser's 28.67 dB) with only 1.2M parameters and 3.6G FLOPs.

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Full Latent Harmony | **29.77** | **0.88** | **0.250** |
| w/o $L_{Inv}$ | 24.28 | 0.79 | 0.292 |
| w/o $L_{Eqv}$ | 25.68 | 0.82 | 0.302 |
| w/o PDPS | 27.82 | 0.84 | 0.287 |
| w/o FHF-LoRA | 28.12 | 0.86 | 0.286 |
| w/o PHF-LoRA | 29.02 | 0.84 | 0.306 |
| w/o All LoRA | 28.68 | 0.85 | 0.298 |

Adaptation results with different restoration backbone networks:

| Backbone | Baseline PSNR | +Ours PSNR | Param Reduction | FLOPs Reduction |
|---------|----------|-----------|---------|-----------|
| Restormer | 24.22 | 29.73 (+5.51) | -85% | -95% |
| NAFNet | 24.63 | 29.68 (+5.05) | -93% | -71% |
| SFHformer | 24.54 | 29.70 (+5.16) | -84% | -93% |

### Key Findings

- $L_{Inv}$ is the most critical component; its removal causes a 5.49 dB drop in PSNR, validating the central role of degradation-invariant semantic alignment
- FHF-LoRA primarily improves fidelity metrics (PSNR/SSIM), while PHF-LoRA primarily improves perceptual metrics (LPIPS), validating the rationale of the differentiated design
- Inference time is only 0.43 seconds, 28× faster than DreamUIR (12.3s)
- As $\alpha$ increases from 0.2 to 0.8, PSNR rises from 28.94 to 29.74 and LPIPS rises from 0.2218 to 0.2904, confirming continuous controllability
- In generalization tests (unseen degradations + composite degradations), the proposed method substantially outperforms existing methods (e.g., +5.41 dB on rain vs. UHDprocesser)

## Highlights & Insights

- **Precise problem formulation**: The identification of three core trade-offs and the frequency-domain analysis (Figure 2) are highly convincing
- **Effective use of DINOv2 semantic alignment**: Leveraging the semantic space of a pretrained vision model to guide the VAE in encoding degradation-invariant representations is an elegant design choice
- **Divide-and-conquer strategy for encoder and decoder**: Assigning FHF-LoRA and PHF-LoRA to optimize distinct objectives is a clever solution for high-frequency information processing
- **Extreme efficiency**: 1.2M parameters and 3.6G FLOPs support full-resolution 4K inference
- **General VAE framework**: The proposed VAE can be substituted into other standard-resolution restoration methods with significant performance gains

## Limitations & Future Work

- Training LH-VAE requires paired clean/degraded images, which are not always available in real-world scenarios
- Training the high-frequency discriminator may introduce the instability typical of adversarial training
- The $\alpha$ parameter requires manual selection; automatic or adaptive trade-off strategies remain unexplored
- Performance under extreme degradation conditions (e.g., severe motion blur combined with low light) is not discussed in detail
- The perceptual quality comparison with diffusion-based restoration methods (e.g., DiffUIR) is insufficiently thorough

## Related Work & Insights

- **UHDprocesser** (Liu et al., 2025): The current SOTA for UHD All-in-One restoration; directly compared and surpassed in this work
- **DreamUHD** (Liu et al., 2025): A UHD restoration approach based on VAE with frequency enhancement
- **PromptIR** (Potlapalli et al., 2023): A prompt-based degradation-aware method
- **REPA-E**: A source of inspiration for end-to-end VAE-LDM joint training
- **DINOv2**: A key component serving as a semantic prior from a pretrained vision model
- The proposed approach has direct implications for other VAE-based generation and restoration tasks (e.g., latent space optimization in video restoration and 3D reconstruction)

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A systematic solution to the triple trade-off; the differentiated HF-LoRA design is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple degradations, multiple backbones, standard/UHD resolutions, generalization tests, and $\alpha$ analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation analysis (Sec. 3) is exceptionally thorough; frequency-domain analysis in Figure 2 is highly convincing
- Value: ⭐⭐⭐⭐⭐ Provides a paradigm-level methodology for applying VAEs to restoration tasks with strong practical utility

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)
- [\[NeurIPS 2025\] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[NeurIPS 2025\] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation](ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)

<!-- RELATED:END -->
