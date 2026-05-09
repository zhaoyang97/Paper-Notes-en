---
title: >-
  [Paper Note] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation
description: >-
  [NeurIPS 2025][Image Generation][Source-Free Domain Adaptation] This paper proposes the DPTM framework, which leverages a latent diffusion model to perform semantic transformation on unreliable target samples, generating a pseudo-target domain and iteratively narrowing the gap with the real target domain via a progressive reconstruction mechanism. DPTM achieves up to 18.6% improvement over existing SFDA state-of-the-art methods under large domain shift scenarios.
tags:
  - NeurIPS 2025
  - Image Generation
  - Source-Free Domain Adaptation
  - Diffusion Models
  - Pseudo-Target Domain Generation
  - Progressive Optimization
  - Semantic Transformation
date: 2026-05-08
content_hash: 6b5c9e7b276f18fd
---

# Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2510.25279](https://arxiv.org/abs/2510.25279)
**Code**: N/A
**Area**: Image Generation / Diffusion Models / Domain Adaptation
**Keywords**: Source-Free Domain Adaptation, Diffusion Models, Pseudo-Target Domain Generation, Progressive Optimization, Semantic Transformation

## TL;DR

This paper proposes the DPTM framework, which leverages a latent diffusion model to perform semantic transformation on unreliable target samples, generating a pseudo-target domain and iteratively narrowing the gap with the real target domain via a progressive reconstruction mechanism. DPTM achieves up to 18.6% improvement over existing SFDA state-of-the-art methods under large domain shift scenarios.

## Background & Motivation

Source-Free Domain Adaptation (SFDA) requires domain adaptation using only a pretrained source model and unlabeled target data. Existing methods fall into two categories, both constrained by the source–target domain discrepancy:

- **Non-generative methods**: Rely on pseudo-labels produced by the source model, which are highly unreliable under large domain shifts (e.g., only ~60% accuracy on Ar→Pr in Office-Home), leading to unstable performance.
- **Generative methods**: Generate a pseudo-source domain and convert the problem into a standard UDA setting, but the generation process introduces irrelevant domain features, inadvertently widening the source–target gap.

Key insight: The fundamental bottleneck of both paradigms is the source–target domain shift. The authors propose a novel paradigm—directly generating a pseudo-target domain rather than a pseudo-source domain—to eliminate this bottleneck at its root.

## Method

### Overall Architecture

DPTM consists of three core components executed over $R$ progressive iterations:

1. **Trust/Untrust Set Partitioning**: Target data is divided into a reliable subset $V$ and an unreliable subset $U$ based on prediction uncertainty.
2. **Untrust Set Manipulation Strategy**: Samples in $U$ are semantically transformed to newly assigned class labels via a diffusion model, while preserving target domain distribution characteristics.
3. **Progressive Reconstruction Mechanism**: The gap between the pseudo-target domain and the real target domain is iteratively reduced.

### Key Designs

#### Trust/Untrust Set Partitioning

Entropy $H$ of target model predictions serves as the uncertainty measure, with threshold $E$ used for partitioning:
- Samples with $H \leq E$ are assigned to the trust set $V$ and supervised with pseudo-labels directly.
- Samples with $H > E$ are assigned to the untrust set $U$ for diffusion-based manipulation.

New class labels are uniformly assigned to each sample in the untrust set ($\hat{y}_l = l \bmod \lfloor|U|/C\rfloor$) to ensure class balance; residual tail samples are discarded.

#### Target-guided Initialization

Motivated by the finding that the starting point of diffusion sampling significantly influences generation results, target-domain-guided initialization is constructed via FFT-based frequency decomposition:
- **Low-frequency components** $F_x^L = \text{FFT}(x) \odot H$ are extracted from the original untrust sample $x_l^u$ (capturing domain characteristics such as style and texture).
- **High-frequency components** $F_{I_G}^H = \text{FFT}(I_G) \odot (1 - H)$ are extracted from semantically neutral Gaussian noise $I_G$ (to prevent semantic leakage from the original sample).
- These are combined via IFFT to produce a pseudo-image $\tilde{x}$ that is semantically neutral yet retains target domain characteristics.
- $\tilde{x}$ is encoded into latent space as $\hat{z}_0 = \mathcal{E}(\tilde{x})$, and $T$-step DDPM forward noise is applied to obtain the sampling starting point $z_T$.

#### Semantic Feature Injection

At each denoising timestep $t$, a zigzag self-reflection operation is performed:
- Denoise $z_t \to z_{t-1}$.
- Inject the semantics of the assigned label $\hat{y}_l$ via DDIM inversion: $z_{t-1} \to \tilde{z}_t$.
- Apply classifier-free guidance with guidance scale $\gamma_2$ to ensure semantic alignment.
- **Only the high-frequency components** of $\tilde{z}_t$ (containing semantic information) are retained; low-frequency components (potentially containing domain noise artifacts) are discarded.

#### Domain-specific Feature Preservation

At each timestep, two frequency components are combined:
- **High frequency**: $F_{\tilde{z}_t}^H$ from semantic injection (target class semantics).
- **Low frequency**: $F_{\hat{z}_{0,t}}^L$ extracted from $\hat{z}_0$ after $t$-step noising (target domain distribution characteristics).
- An enhanced latent $\tilde{z}'_t$ is synthesized via IFFT, jointly preserving semantic content and domain features.

#### Progressive Reconstruction Mechanism

$R$ rounds of iterative optimization are performed:
- After round $r$, the updated target model repartitions $V^{(r+1)}$ and $U^{(r+1)}$.
- As the model improves, $|V^{(r+1)}| > |V^{(r)}|$, so the untrust set gradually shrinks.
- Reduced manipulation → reduced domain gap → further model improvement, forming a positive feedback loop.

### Loss & Training

- The trust set and manipulated untrust set are merged to form the pseudo-target domain $\mathcal{D}_p = V \cup U^m$.
- The target model is fine-tuned via standard cross-entropy supervision on $\mathcal{D}_p$.
- Diffusion model: Pretrained SD v1.5, resolution 512×512, 20 denoising steps.
- Hyperparameters: $\gamma_1 = 5.5$, $\gamma_2 = 0$, $E = 0.01$, $R = 10$.
- ResNet-50/101 as the adaptation backbone; SGD optimizer; 15K–20K training steps.

## Key Experimental Results

### Main Results

**Office-31 (ResNet-50)**

| Method | D→A | W→A | Avg |
|--------|------|------|------|
| ProDe (ICLR25) | 79.8 | 79.0 | 89.9 |
| DM-SFDA | 82.7 | 83.5 | 93.7 |
| **DPTM (Ours)** | **92.0** | **91.7** | **95.8** |

On the challenging tasks D→A and W→A, improvements of **+9.3%** and **+8.2%** are achieved, respectively.

**Office-Home (ResNet-50, 12 DA tasks)**

| Method | Ar→Cl | Pr→Cl | Rw→Cl | Avg |
|--------|-------|-------|-------|------|
| ProDe (ICLR25) | 64.0 | 65.4 | 65.5 | 81.1 |
| DM-SFDA | 68.5 | 69.6 | 68.5 | 79.5 |
| **DPTM** | **86.7** | **86.4** | **87.1** | **91.2** |

Average improvement of **+10.1%** over ProDe; over 20% gain on the challenging →Cl tasks.

**DomainNet-126 (ResNet-50, 12 tasks)**

| Method | C→P | Avg |
|--------|------|------|
| CPGA | 61.2 | 67.6 |
| ProDe | 79.3 | 81.5 |
| **DPTM** | **85.6** | **85.2** |

### Ablation Study

**Effect of Threshold $E$ (Office-Home, $R=10$)**

| $E$ | Avg |
|-----|------|
| 0.001 | 80.7 |
| 0.005 | 86.7 |
| 0.01 | 91.2 |

A larger $E$ assigns more samples to the untrust set for manipulation, yielding better performance.

**Component Ablation on Manipulation Strategy**: Removing any single component (Target-guided Initialization / Semantic Injection / Domain Preservation) leads to failure in either semantic alignment or domain preservation. SD v1.5 and SDXL achieve comparable performance (both 75.6% Avg), with SD v1.5 being more efficient.

### Key Findings

- The largest gains occur under large domain shifts (e.g., Rw→Cl), validating the pseudo-target domain strategy as a fundamental solution to the domain gap.
- The trust set expands progressively during reconstruction, demonstrating the self-improving positive feedback loop.
- Frequency-domain decomposition effectively decouples semantic and domain features; all three manipulation components are indispensable.

## Highlights & Insights

1. **Paradigm Innovation**: The first SFDA strategy to generate a pseudo-target domain, fundamentally overcoming the domain shift bottleneck.
2. **Elegant Frequency-Domain Design**: The separation of low-frequency (domain features) and high-frequency (semantic features) via FFT permeates the entire method.
3. **Progressive Self-Improvement**: The positive feedback loop of a shrinking untrust set enables continuous optimization without additional data.
4. **Strong Performance under Challenging Conditions**: Maximum gains (18.6%) are achieved precisely in large domain shift scenarios where other methods typically fail.

## Limitations & Future Work

- Relies on a pretrained diffusion model (SD v1.5), incurring substantial computational and memory overhead.
- Threshold $E$ and iteration count $R$ require tuning, with optimal settings varying across datasets.
- Validated only on classification tasks; extension to detection, segmentation, and other downstream tasks remains unexplored.
- The 512×512 generation resolution of the diffusion model may limit applicability in higher-resolution settings.

## Related Work & Insights

- Core distinction from DM-SFDA: DM-SFDA generates a pseudo-source domain, whereas DPTM manipulates target data to generate a pseudo-target domain.
- Frequency-domain operations are inspired by FreeInit/FlexiEdit research on the influence of low-frequency components at diffusion starting points.
- Zigzag inversion (bai2024zigzag) is adopted for injecting semantics along the sampling trajectory.
- Progressive training is related to curriculum learning but differs in direction—here the dataset shrinks rather than task difficulty increasing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Pseudo-target domain generation paradigm is entirely novel; frequency-domain separation is highly original)
- Technical Depth: ⭐⭐⭐⭐⭐ (Deep integration of diffusion models, frequency-domain operations, and progressive optimization)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 benchmarks, 21 comparison methods, comprehensive ablations)
- Practicality: ⭐⭐⭐⭐ (Significant improvements but with relatively high computational cost)
- Writing Quality: ⭐⭐⭐⭐ (Complex methodology presented clearly in a modular fashion)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[NeurIPS 2025\] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)

<!-- RELATED:END -->
