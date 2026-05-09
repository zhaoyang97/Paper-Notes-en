---
title: >-
  [Paper Note] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles
description: >-
  [NeurIPS 2025][Medical Imaging][Conformational ensembles] This paper proposes JAMUN, a conformational ensemble generation method built on the Walk-Jump Sampling (WJS) framework. By performing Langevin dynamics on a noise-smoothed manifold and using an SE(3)-equivariant denoiser to jump back to the original distribution, JAMUN achieves peptide conformational sampling an order of magnitude faster than conventional molecular dynamics while retaining transferability to out-of-training systems.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Conformational ensembles
  - Walk-Jump Sampling
  - denoising
  - SE(3) equivariance
  - protein dynamics
date: 2026-05-08
content_hash: 5599e83a9fcf6d06
---

# JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles

**Conference**: NeurIPS 2025

**arXiv**: [2410.14621](https://arxiv.org/abs/2410.14621)

**Code**: [Available](https://github.com/prescient-design/jamun)

**Area**: Molecular Dynamics / Generative Models

**Keywords**: Conformational ensembles, Walk-Jump Sampling, denoising, SE(3) equivariance, protein dynamics

## TL;DR

This paper proposes JAMUN, a conformational ensemble generation method built on the Walk-Jump Sampling (WJS) framework. By performing Langevin dynamics on a noise-smoothed manifold and using an SE(3)-equivariant denoiser to jump back to the original distribution, JAMUN achieves peptide conformational sampling an order of magnitude faster than conventional molecular dynamics while retaining transferability to out-of-training systems.

## Background & Motivation

Proteins are not static structures but exist as conformational ensembles; their dynamic properties are critical for understanding protein function and for drug discovery (e.g., cryptic pocket identification). However:

**Molecular Dynamics (MD) is inefficient**: MD requires extremely short time steps of 1–2 femtoseconds, whereas many important protein dynamic phenomena occur at the millisecond scale — Borhani & Shaw liken this to "tracking glacial advance and retreat over tens of thousands of years by recording observations once per second."

**Limitations of enhanced sampling methods**: These methods typically require domain knowledge about relevant collective variables and do not address the fundamental time-step problem.

**Lack of transferability in existing ML methods**: Flow matching and diffusion models can generate conformations but cannot generalize to systems outside the training set. The sole exception, Transferable Boltzmann Generators (TBG), suffers from limited efficiency.

Core motivation: to develop a conformational ensemble generative model that is simultaneously fast and transferable.

## Method

### Overall Architecture

The core of JAMUN is Walk-Jump Sampling (WJS):

1. **Noise**: Gaussian noise is added to an initial conformation $x^{(0)} \sim p_X$:
   $$y^{(0)} = x^{(0)} + \sigma \varepsilon^{(0)}, \quad \varepsilon^{(0)} \sim \mathcal{N}(0, \mathbb{I}_{N \times 3})$$

2. **Walk**: Langevin dynamics are run on the smoothed noisy manifold using the BAOAB integrator to solve the SDE:
   $$dy = v_y dt, \quad dv_y = \nabla_y \log p_Y(y) dt - \gamma v_y dt + M^{-1/2} \sqrt{2} dB_t$$

3. **Jump**: A denoiser projects samples back to the original distribution:
   $$\hat{x}^{(i)} = \hat{x}(y^{(i)}) = \mathbb{E}[X | Y = y^{(i)}]$$

Key connection: the Walk and Jump steps are linked via the score function: $\hat{x}(y) = y + \sigma^2 \nabla_y \log p_Y(y)$.

### Key Designs

**1. Point Cloud Representation**

Each peptide is represented as a point cloud of $N$ atoms $(x, h)$, where $x \in \mathbb{R}^{N \times 3}$ denotes 3D coordinates and $h$ encodes atom and bond information. JAMUN operates directly in all-atom 3D coordinate space without hand-crafted featurization (e.g., backbone dihedral angles).

**2. SE(3)-Equivariant Denoiser**

A NequIP-based geometric graph neural network is employed, with a critical distinction:
- **SE(3) equivariance** rather than E(3) equivariance: this avoids the symmetric Ramachandran plot artifacts caused by the parity symmetry of E(3) models.
- No post-hoc "chirality checker" (as required by TBG) is needed.
- The model has only **8.2M parameters**.

The denoiser parameterization follows Karras et al. (2022):
$$\hat{x}_\theta(y, \sigma) = c_{\text{skip}}(\sigma) y + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma) y, c_{\text{noise}}(\sigma))$$

**3. Fixed Noise Level**

Unlike diffusion or flow matching models requiring multiple noise levels, WJS requires training at only a **single fixed noise level** $\sigma$. The choice of $\sigma$ balances sampling efficiency (larger is better) against denoising difficulty (excessively large values destroy topology).

**4. Input/Output Normalization**

Normalization coefficients are re-derived for point cloud data, based on pairwise edge distances rather than pixel values:
$$c_{\text{in}}(\sigma) = \frac{1}{\sqrt{C + 6\sigma^2}}, \quad c_{\text{skip}}(\sigma) = \frac{C}{C + 6\sigma^2}$$

where $C = \mathbb{E}_{(i,j)} \|x_i - x_j\|^2$ is estimated from real data.

### Loss & Training

- **Loss function**: Standard L2 denoising loss:
  $$\theta^* = \arg\min_\theta \mathbb{E}_{X \sim p_X, \varepsilon \sim \mathcal{N}} \|\hat{x}_\theta(Y, \sigma) - X\|^2$$
- **Optimizer**: Adam, learning rate 0.002
- **Batch size**: 42 × 6 GPUs (A100)
- **Noise level**: $\sigma = 0.4$ Å for capped-2AA; $\sigma = 0.6$ Å for uncapped-2AA
- **Sampling parameters**: friction $\gamma = 0.1$, step size $\Delta t = \sigma$, $M = 1$

## Key Experimental Results

### Main Results

**Datasets**:
- uncapped-2AA (Timewarp): 380 dipeptides (200 train / 80 validation / 100 test)
- capped-2AA: peptides capped with ACE and NME groups, more complex and closer to real proteins

**Table 1: Time for JAMUN to sample 640,000 conformations (NVIDIA A100 GPU)**

| Capped-2AA Peptide | Total Sampling Time (min) | Time per Sample (ms) |
|:---|:---:|:---:|
| ASP-TRP | 38 | 3.56 |
| GLU-THR | 27 | 2.53 |
| PHE-ALA | 28 | 2.62 |
| ASN-GLU | 29 | 2.71 |
| CYS-TRP | 34 | 3.18 |
| GLY-ASN | 22 | 2.06 |
| HIS-PRO | 30 | 2.81 |
| ILE-GLY | 22 | 2.06 |

These correspond to 100–300 ps of MD simulation time; JAMUN completes the equivalent sampling in approximately one hour.

**Qualitative Ramachandran Plot Comparison** (Figures 8–9):
- JAMUN accurately samples the vast majority of low-energy basins for test peptides.
- In equal GPU time, MD typically remains trapped in a single basin.
- Compared to TBG (Figure 10), JAMUN samples all states within the same compute budget, whereas TBG misses several basins.

### Ablation Study

**Table 2: Jensen-Shannon Divergence Convergence Analysis (Figure 11)**

| Sampling Method | Convergence Characteristics | Final JSD |
|:---|:---|:---|
| JAMUN | Rapid, smooth convergence | Slightly above converged MD |
| MD (equal GPU time) | Stepwise (kinetic traps present) | Far from converged |
| MD (fully converged) | Eventually reaches reference | Reference baseline |

JAMUN converges faster than MD but does not fully match the same precision, partly due to oversampling of rare/transition regions.

**Markov State Model Analysis** (Appendix A, Figures 12–19):
- JAMUN accurately captures the metastable state geometries of specific dipeptide species.
- Subtle states not apparent in simple Ramachandran plots are correctly sampled.

### Key Findings

1. **Speed advantage**: JAMUN is more than an order of magnitude faster than conventional MD for conformational ensemble sampling.
2. **Transferability**: JAMUN accurately generates conformational ensembles for peptides outside the training set.
3. **SE(3) > E(3)**: SE(3)-equivariant networks handle chirality naturally without post-processing.
4. **Single noise level training**: Unlike diffusion models, WJS requires only one noise level, simplifying training.
5. **Preservation of physical priors**: The intrinsic physical priors encoded in smoothed MD data are key to transferability.

## Highlights & Insights

1. **Elegant physical intuition**: The core idea of WJS is that noise smooths the energy landscape; Langevin dynamics on the smoothed manifold efficiently traverses barriers, and denoising jumps samples back to the true distribution.
2. **Decoupling of Walk and Jump**: Sampling proceeds via trajectories that efficiently traverse the smoothed space, rather than restarting from an uninformative Gaussian prior at each step — a fundamental distinction from diffusion/flow matching models.
3. **First application of WJS to point clouds and molecular dynamics.**
4. **Practical value**: In drug discovery, diverse sampling of metastable states is often more important than exact kinetics, and JAMUN excels in this regard.

## Limitations & Future Work

1. **Not tested on large proteins**: Validation is currently limited to dipeptides; transferability to larger proteins remains to be explored.
2. **Approximate sampler**: JAMUN is a Boltzmann emulator rather than a generator and cannot perform exact reweighting.
3. **Oversampling of transition regions**: The stochastic process on the smoothed surface leads to non-physical overrepresentation of transition paths.
4. **Compatibility with enhanced sampling**: Methods such as metadynamics could be applied within the noised space.
5. **Denoiser improvements**: Better architectures could accelerate sampling; multi-step denoising (à la diffusion) could sharpen generation.

## Related Work & Insights

- **Walk-Jump Sampling** (Saremi & Hyvärinen, 2019): Neural Empirical Bayes framework
- **Transferable Boltzmann Generators** (Klein & Noé, 2024): The only prior transferable method
- **Timewarp** (Klein et al., 2024a): Provides the uncapped-2AA benchmark dataset
- **NequIP** (Batzner et al., 2022): Foundation E(3)-equivariant GNN architecture
- **Karras normalization** (Karras et al., 2022): Design principles for denoiser parameterization
- Insight: Adapting diffusion model design principles from image generation to molecular point clouds; physical priors are essential for achieving transferability.

## Rating

| Dimension | Score (1–5) | Notes |
|:---|:---:|:---|
| Novelty | 5 | First application of WJS to point clouds and MD; elegant formulation |
| Technical Quality | 4 | Rigorous theoretical derivation; complete normalization derivation |
| Experimental Thoroughness | 4 | Two datasets (capped/uncapped); in-depth MSM analysis |
| Practicality | 4 | Conformational sampling for drug discovery has real-world value |
| Writing Quality | 4 | Physical intuition clearly articulated; detailed appendix |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Conformational Ensembles of Proteins Based on Backbone Geometry](learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
