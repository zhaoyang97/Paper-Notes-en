---
title: >-
  [Paper Note] Scalable Group Choreography via Variational Phase Manifold Learning
description: >-
  [ECCV 2024][Image Generation][Group Choreography] This paper proposes PDVAE (Phase-conditioned Dance VAE), a phase-conditioned variational generative model for scalable group choreography. By learning the phase manifold (amplitude, frequency, offset, phase shift) of dance motion in the frequency domain, it achieves high-quality group dance generation for **an arbitrary number** of dancers with constant memory consumption, comprehensively outperforming existing methods on the…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Group Choreography"
  - "Phase Manifold"
  - "Variational Autoencoder"
  - "Music-driven Dance"
  - "Scalable Generation"
date: 2026-05-08
content_hash: c8ff98968a0d4523
---

# Scalable Group Choreography via Variational Phase Manifold Learning

**Conference**: ECCV 2024  
**arXiv**: [2407.18839](https://arxiv.org/abs/2407.18839)  
**Code**: None (by AIOZ)  
**Area**: Image Generation  
**Keywords**: Group Choreography, Phase Manifold, Variational Autoencoder, Music-driven Dance, Scalable Generation

## TL;DR
This paper proposes PDVAE (Phase-conditioned Dance VAE), a phase-conditioned variational generative model for scalable group choreography. By learning the phase manifold (amplitude, frequency, offset, phase shift) of dance motion in the frequency domain, it achieves high-quality group dance generation for **an arbitrary number** of dancers with constant memory consumption, comprehensively outperforming existing methods on the AIOZ-GDance and AIST-M datasets.

## Background & Motivation

**Background**: Extensive research has been conducted on music-driven dance generation for individual dancers. Group dance generation requires maintaining synchronization and coordination among dancers while generating diverse individual movements. Existing group methods (GDanceR, GCD) utilize collaboration mechanisms such as cross-entity attention or global attention, which require processing the movements of all dancers simultaneously.

**Limitations of Prior Work**: (a) **Poor scalability**: Existing methods are limited by the maximum number of dancers in the dataset (typically 2-5) and cannot scale to more dancers; (b) **Memory explosion**: The memory consumption of mechanisms like cross-entity attention grows linearly or even quadratically with the number of dancers, leading to out-of-memory (OOM) errors in GCD with 10 dancers; (c) **Bottleneck of diffusion models**: Diffusion models operate in the original high-dimensional data space, making them harder to scale.

**Key Challenge**: Existing architectures require **simultaneously processing all dancers'** motions to ensure coordination, which leads to uncontrollable computation/memory scaling with the number of group members. A mechanism is needed where each dancer can be generated independently while preserving group consistency.

**Goal**: Design a group dance generation method that (a) scales to an arbitrary number of dancers; (b) maintains constant memory; (c) preserves group synchronization and individual diversity.

**Key Insight**: Inspired by the success of phase representation in motion synthesis—although different dancers' movements under the same music appear different, their temporal characteristics (beats, periodicity, temporal alignment) are inherently similar. Frequency-domain phase parameters can be used to characterize this shared attribute.

**Core Idea**: Parameterize the latent space of a VAE using frequency-domain phase parameters (amplitude A, frequency F, offset B, phase shift S) to learn a group-consistent phase manifold. During inference, encoding the music once is sufficient to obtain the manifold distribution, which is then sampled infinitely to generate different dancers.

## Method

### Overall Architecture
PDVAE consists of three networks:
- **Encoder $\mathcal{E}$**: Takes motion + music features as input and outputs the parameters of the posterior distribution $q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{a})$.
- **Prior Network $\mathcal{P}$**: Takes only music features as input and learns the conditional prior $p_\theta(\mathbf{z}|\mathbf{a})$.
- **Decoder $\mathcal{D}$**: Reconstructs the motion sequence from the sampled latent phase curve + music features.

During training, the encoder and decoder are used; during inference, only the prior network and decoder are used, sampling the phase parameters once from the prior distribution for each new dancer.

### Key Designs

1. **Variational Phase Manifold (Core Innovation)**:

    - Function: Replaces the conventional Gaussian latent vector in a traditional VAE with frequency-domain phase parameters, giving the latent space a temporal structure.
    - Mechanism: Applies FFT to the latent curve $\mathbf{L} \in \mathbb{R}^{D \times T}$ output by the encoder to calculate the power spectrum, and then extracts the distribution means of the four phase parameters:
        - Amplitude: $\mu_i^A = \sqrt{\frac{2}{T} \sum_j \mathbf{p}_{i,j}}$
        - Frequency: $\mu_i^F = \frac{\sum_j \mathbf{f}_j \cdot \mathbf{p}_{i,j}}{\sum_j \mathbf{p}_{i,j}}$ (power spectrum-weighted average frequency)
        - Offset: $\mu_i^B = \frac{\mathbf{c}_{i,0}}{T}$ (DC component)
        - Phase Shift: $\mu_i^S = \arctan(s_y, s_x)$ (FC layer prediction + two-argument arctan activation)
    - Variational sampling is only applied to amplitude A and phase shift S ($\sigma^A, \sigma^S$ are predicted by an MLP), while frequency F and offset B are set as deterministic (otherwise, group choreography would be uncoordinated).
    - Reconstruct the parameterized latent curve after sampling: $\hat{\mathbf{L}} = \mathbf{A} \cdot \sin(2\pi(\mathbf{F} \cdot \mathcal{T} - \mathbf{S})) + \mathbf{B}$
    - Design Motivation: The single Gaussian vector in traditional VAEs "compresses" the temporal dimension information, failing to represent the temporal dynamics of motion. Phase parameters naturally capture the temporal characteristics of motion (periodicity, beat alignment, start/end timing), and different dancers share frequency and offset to guarantee consistent group beats.

2. **Group Consistency Loss $\mathcal{L}_{csc}$**:

    - Function: Constrains different dancers within the same group to encode into the same phase manifold.
    - Mechanism: $\mathcal{L}_{csc} = D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}^m, \mathbf{a}) \| q_\phi(\mathbf{z}|\mathbf{x}^n, \mathbf{a})) + \|\mathbf{P}^m - \mathbf{P}^n\|_2^2$
    - Where $\mathbf{P}_{2i-1} = \mathbf{A}_i \sin(2\pi \cdot \mathbf{S}_i)$ and $\mathbf{P}_{2i} = \mathbf{A}_i \cos(2\pi \cdot \mathbf{S}_i)$ are phase manifold features.
    - Design Motivation: The CVAE objective is computed independently for each dancer, failing to capture inter-dancer correlations. This loss ensures all dancers map to the same unified manifold.

3. **Transformer Architecture + Siren Activation**:

    - Encoder: Transformer decoder architecture, using cross-attention to learn motion-music relations.
    - Decoder: Transformer decoder, performing cross-attention between parameterized latent curves (query) and music features (key/value).
    - Prior Network: Transformer encoder, capturing global music context with self-attention.
    - Employs the Siren (sinusoidal) activation function to better model the periodicity of phase features.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{rec} + \lambda_{KL} \mathcal{L}_{KL} + \lambda_{csc} \mathcal{L}_{csc}$

$\lambda_{KL} = 5 \times 10^{-4}$, $\lambda_{csc} = 10^{-4}$. Reconstructions use the smooth-L1 loss.

## Key Experimental Results

### Main Results

| Dataset | Method | FID↓ | MMC↑ | GenDiv↑ | PFC↓ | GMR↓ | GMC↑ | TIF↓ |
|--------|------|------|------|---------|------|------|------|------|
| AIOZ-GDance | FACT | 56.20 | 0.222 | 8.64 | 3.52 | 101.52 | 62.68 | 0.321 |
| AIOZ-GDance | EDGE | 31.40 | 0.264 | 9.57 | 2.63 | 63.35 | 61.72 | 0.356 |
| AIOZ-GDance | GCD | 31.16 | 0.261 | 10.87 | 2.53 | 31.47 | 80.97 | 0.167 |
| AIOZ-GDance | **PDVAE** | **31.01** | **0.271** | **10.98** | **2.33** | **30.08** | **84.52** | **0.163** |
| AIST-M | GCD | 35.36 | 0.245 | 10.97 | 1.52 | 42.52 | 72.15 | 0.083 |
| AIST-M | DanY | 40.25 | 0.240 | 11.40 | 1.65 | 50.29 | 63.53 | 0.137 |
| AIST-M | **PDVAE** | **31.49** | **0.257** | **11.81** | **1.42** | 41.24 | **78.64** | **0.076** |

PDVAE achieves the best performance across almost all metrics, especially leading by a large margin in group dance metrics (GMR, GMC, TIF).

### Scalability Experiments (4GB Consumer GPU)

| Number of Dancers | Method | FID↓ | GMR↓ | GMC↑ | TIF↓ |
|--------|------|------|------|------|------|
| 5 | GCD | 35.08 | 38.43 | 81.44 | 0.168 |
| 5 | **PDVAE** | **31.35** | **32.58** | **84.56** | **0.161** |
| 10 | GCD | N/A (OOM) | N/A | N/A | N/A |
| 10 | **PDVAE** | **32.19** | **34.32** | **86.96** | 0.193 |
| 100 | GDanceR | N/A (OOM) | N/A | N/A | N/A |
| 100 | **PDVAE** | **30.97** | **38.13** | **85.73** | 0.222 |

PDVAE can generate 100 dancers on a 4GB GPU, whereas GCD encounters an OOM error with 10 dancers, and GDanceR OOMs with 100 dancers. The memory consumption of PDVAE remains constant.

### Ablation Study

| Configuration | FID↓ | GMR↓ | GMC↑ |
|------|------|------|------|
| PDVAE (Full) | 31.01 | 30.08 | 84.52 |
| w/o Consistency Loss | 35.35 | 57.63 | 66.72 |
| w/o Phase Manifold | 41.78 | 45.32 | 77.93 |
| Replace with LSTM Backbone | 41.29 | 47.47 | 71.82 |
| Replace with CNN Backbone | 36.99 | 44.94 | 75.77 |

### Key Findings
- **Phase manifold contributes the most**: Removing it increases the FID from 31.01 to 41.78 and the GMR from 30.08 to 45.32, which demonstrates that frequency-domain parameterization is the core of the model's success.
- **Consistency loss is vital for group choreography quality**: Removing it causes the GMC to plunge from 84.52 to 66.72, leading to a severe decline in group coordination.
- **Transformer backbone significantly outperforms LSTM and CNN**: The LSTM backbone results in an FID of 41.29, which indicates that long-range dependency modeling is highly crucial for dance generation.
- **Scalability**: PDVAE is the only method capable of generating 100 dancers on a consumer-grade GPU without performance degradation as the number of dancers increases.
- **User study (approx. 70 participants)**: As the number of dancers increases, realism scores for all methods decrease, but PDVAE exhibits the smallest decline.

## Highlights & Insights
- **VAE latent space parameterized by frequency-domain phase**: A highly innovative design. While traditional VAEs throw away temporal information by using Gaussian vectors, phase parameters naturally encode the temporal features of motion ($A$=amplitude, $F$=frequency, $S$=phase shift, $B$=offset), making the latent space structured and interpretable. This direction can be generalized to any VAE-based task requiring temporal structures (e.g., speech or music generation).
- **Scalable generation with constant memory**: During inference, only running the prior network once to obtain the distribution is needed, followed by sampling and decoding for each new dancer. This "encode once, sample infinitely" design is an elegant solution to scalability.
- **Deterministic frequency and offset, variational amplitude and phase shift**: This is a clever design choice—frequency and offset relate to the beat and must be group-consistent, whereas amplitude and phase shift correspond to motion intensity and timing, which can vary individually.

## Limitations & Future Work
- A global trajectory predictor is utilized to avoid dancer collisions, but TIF (0.222) remains relatively high when generating 100 dancers.
- The phase manifold assumes that motion is quasi-periodic, which may lead to suboptimal performance for non-periodic movements (such as poses at the start/end of a dance).
- Only keypoint movements of the SMPL body model are evaluated, omitting finger and facial details.
- The maximum group size in the AIOZ-GDance dataset is limited, making ground-truth data for 100-dancer group choreography unavailable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of the frequency-domain phase-parameterized VAE latent space is highly novel, and the constant-memory scalable design is very clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across two datasets, scalability tests, ablations, and a user study.
- Writing Quality: ⭐⭐⭐⭐ Math derivations are clear but slightly verbose.
- Value: ⭐⭐⭐⭐⭐ Achieves arbitrary-scale group dance generation with constant memory for the first time, significantly advancing the motion synthesis field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ECCV 2024\] ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](scaledreamer_scalable_text-to-3d_synthesis_with_asynchronous_score_distillation.md)
- [\[ICML 2025\] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](../../ICML2025/image_generation/local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)
- [\[CVPR 2026\] Learning Straight Flows: Variational Flow Matching for Efficient Generation](../../CVPR2026/image_generation/learning_straight_flows_variational_flow_matching_for_efficient_generation.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
