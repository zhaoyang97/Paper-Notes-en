---
title: >-
  [Paper Note] Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity
description: >-
  [ICLR 2026][Medical Imaging][3D medical reconstruction] This paper proposes Inter-Slice Consistent Stochasticity (ISCS), which generates inter-slice correlated noise via spherical linear interpolation (Slerp) during the re-noising step of diffusion sampling, eliminating inter-slice discontinuity artifacts in 3D medical reconstruction with 2D diffusion priors at their root cause — with zero additional computation, hyperparameters, or training overhead, and plug-and-play compatibility with any 2D diffusion inverse problem solver, yielding consistent improvements on sparse-view CT, limited-angle CT, and MRI super-resolution.
tags:
  - ICLR 2026
  - Medical Imaging
  - 3D medical reconstruction
  - 2D diffusion models
  - inter-slice consistency
  - spherical linear interpolation
  - plug-and-play
date: 2026-05-08
content_hash: e75fb79cb6e8af03
---

# Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity

**Conference**: ICLR 2026
**arXiv**: [2602.04162](https://arxiv.org/abs/2602.04162)
**Code**: [GitHub](https://github.com/duchenhe/ISCS)
**Area**: Medical Imaging / Diffusion Models
**Keywords**: 3D medical reconstruction, 2D diffusion models, inter-slice consistency, spherical linear interpolation, plug-and-play

## TL;DR

This paper proposes Inter-Slice Consistent Stochasticity (ISCS), which generates inter-slice correlated noise via spherical linear interpolation (Slerp) during the re-noising step of diffusion sampling, eliminating inter-slice discontinuity artifacts in 3D medical reconstruction with 2D diffusion priors at their root cause — with zero additional computation, hyperparameters, or training overhead, and plug-and-play compatibility with any 2D diffusion inverse problem solver, yielding consistent improvements on sparse-view CT, limited-angle CT, and MRI super-resolution.

## Background & Motivation

**Clinical demand for 3D medical imaging**: Clinical diagnosis (e.g., tumor volume assessment, surgical planning, disease progression tracking) relies on complete and accurate 3D volumetric reconstruction rather than individual 2D slices.

**Infeasibility of 3D diffusion models**: Training diffusion models directly on high-dimensional volumetric data suffers from the curse of dimensionality — memory, computation, and data requirements far exceed what most academic and industrial groups can afford (Pinaya et al., 2022; Guo et al., 2025; Wang et al., 2025).

**Practical compromise with 2D priors**: The prevailing approach trains diffusion models on 2D slices and reconstructs 3D volumes slice by slice — computationally feasible, but introducing new problems.

**Root cause of inter-slice discontinuity**: Each 2D slice is sampled independently during reverse diffusion, and the inherent stochastic noise injection renders the sampling trajectories of adjacent slices entirely uncorrelated, producing severe structural discontinuities and artifacts along the z-axis upon stacking.

**Limitations of existing methods**: (a) TV regularization — introduces sensitive hyperparameters and over-smooths fine details; (b) 3D patch training / biplanar priors — increase training/inference complexity and impose additional data constraints (e.g., requiring cubic volumes); (c) these methods fundamentally treat symptoms rather than the root cause.

**Inspiration from video restoration**: Kwon & Ye (2025) identify that temporal flickering in video restoration likewise stems from incoherent diffusion sampling stochasticity, and propose Batch-Consistent Sampling (BCS) as a remedy. This paper systematically transfers that insight to 3D medical reconstruction and proposes a superior solution.

## Method

### Overall Architecture

The 3D medical inverse problem solving framework based on 2D diffusion models consists of three iterative steps: (1) denoising prediction $\hat{x}_{0|t}$, (2) data fidelity update, and (3) re-noising to timestep $t-1$. ISCS modifies only the stochastic noise injection in step (3), leaving all other steps unchanged.

### Key Design 1: Root Cause Analysis of Inter-Slice Inconsistency

- **Function**: Systematically analyzes the fundamental cause of inter-slice discontinuity when using 2D diffusion priors for 3D reconstruction.
- **Mechanism**: The re-noising step of the DDIM sampler can be decomposed into a deterministic component and a stochastic component:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_{0|t} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}-\eta^2\tilde{\beta}_t^2}\epsilon_{\theta^*}^{(t)}(x_t)}_{\text{deterministic noise}} + \underbrace{\eta\tilde{\beta}_t\epsilon}_{\text{stochastic noise}}$$

The deterministic component is governed by network predictions (yielding similar outputs for similar inputs), whereas the stochastic noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is sampled independently for each slice — this is the root cause of inter-slice inconsistency. When the inverse problem is severely underdetermined (e.g., sparse-view CT), the data fidelity term provides weak constraints, granting the independent noise excessive degrees of freedom to drive adjacent slices toward entirely different trajectories.

- **Design Motivation**: Only by identifying the root cause (incoherent stochasticity) can a principled solution be designed, rather than post-hoc treatment of symptoms.

### Key Design 2: Slerp-Based Inter-Slice Correlated Noise Generation

- **Function**: Replaces independent noise sampling with Spherical Linear Interpolation (Slerp) to generate smoothly correlated noise volumes across slices.
- **Mechanism**: For a volume of $S$ slices, two anchor noise vectors $\mathbf{z}_1, \mathbf{z}_S \sim \mathcal{N}(0, \mathbf{I})$ are sampled, and intermediate-slice noise is generated by interpolating along the geodesic on the high-dimensional hypersphere:

$$\epsilon_i^{\text{ISCS}} = \text{slerp}(\mathbf{z}_1, \mathbf{z}_S; \alpha_i) = \frac{\sin((1-\alpha_i)\Omega)}{\sin(\Omega)}\mathbf{z}_1 + \frac{\sin(\alpha_i\Omega)}{\sin(\Omega)}\mathbf{z}_S$$

where $\alpha_i = (i-1)/(S-1)$ is the normalized position and $\Omega = \arccos(\langle \mathbf{z}_1, \mathbf{z}_S \rangle / (\|\mathbf{z}_1\| \cdot \|\mathbf{z}_S\|))$ is the angle between the anchors.

- **Design Motivation**: Why Slerp rather than linear interpolation? By the Gaussian Annulus Theorem, the probability mass of a high-dimensional isotropic Gaussian concentrates on a thin spherical shell of radius $\sqrt{d}$. Linear interpolation traverses a chord — intermediate points satisfy $\|z\| < \sqrt{d}$, deviating from the typical set. Slerp traverses the geodesic, preserving vector norms and distributional statistics so that each slice's noise remains distributed as $\mathcal{N}(0, \mathbf{I})$.

### Key Design 3: Why Slerp Outperforms BCS (Identical Noise)

- **Function**: Constructs a noise structure with "strong local correlation, weak long-range correlation," replacing BCS's identical-noise scheme.
- **Mechanism**: BCS applies identical noise to all slices, which is viable in video restoration (<16 frames, small inter-frame variation) but is overly rigid for medical volumes (>300 slices, significant inter-slice anatomical variation) — suppressing anatomical change and inducing "copy artifacts" (features inappropriately replicated across anatomically distinct slices). ISCS Slerp noise naturally satisfies the desired properties: (i) adjacent slices have highly correlated noise → local consistency; (ii) correlation decays with distance → global structural variation is permitted.
- **Design Motivation**: The intrinsic characteristic of medical volumetric data is "locally continuous but globally varying"; the correlation structure of the noise should match this property rather than being uniformly identical or uniformly independent.

### Key Design 4: Plug-and-Play Integration

- **Function**: The ISCS noise volume directly replaces independent noise in the re-noising step of any diffusion sampler.
- **Mechanism**: The modified update rule is:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_{0|t} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t) + \sigma_t \cdot \epsilon^{\text{ISCS}}$$

No changes to network architecture, loss function, training procedure, or inference optimization steps are required.

- **Design Motivation**: Minimizing integration cost enables broad applicability to any existing 2D diffusion inverse problem solver (DDNM, DDS, etc.), which is especially valuable in resource-constrained medical imaging settings.

## Key Experimental Results

### Experimental Setup

- **CT dataset**: AAPM 2016 low-dose CT (10 patients, 5936 slices, 256×256); evaluated on 256×256×300 volumes.
- **MRI dataset**: IXI T1 brain scans; evaluated on 256×256×150 volumes with 5× downsampling along the z-axis to simulate anisotropy.
- **Baselines**: FDK, ADMM-TV (traditional); DDNM, DDS (2D diffusion solvers); DDS+TV regularization.
- **Evaluation metrics**: PSNR, SSIM, LPIPS; independently evaluated on three views (axial / coronal / sagittal).

### Table 1: Sparse-View CT (30 views) Main Results

| Method | Axial PSNR | Coronal PSNR | Sagittal PSNR | Coronal SSIM | Sagittal LPIPS |
|--------|-----------|-------------|-------------|-------------|---------------|
| FDK | 23.91 | 23.92 | 23.79 | 0.414 | 0.310 |
| ADMM-TV | 32.94 | 33.67 | 33.72 | 0.895 | 0.107 |
| DDS | 34.76 | 35.12 | 35.33 | 0.906 | 0.141 |
| DDS+TV | 36.26 | 37.08 | 37.50 | 0.938 | 0.088 |
| **DDS+ISCS** | **36.97** | **37.75** | **38.16** | **0.944** | **0.065** |

### Table 2: Limited-Angle CT ([0°, 100°]) Main Results

| Method | Axial PSNR | Coronal PSNR | Sagittal PSNR | Coronal SSIM | \|Δ\| |
|--------|-----------|-------------|-------------|-------------|-----|
| DDNM | 28.40 | 28.75 | 28.22 | 0.774 | 0.016443 |
| DDNM+ISCS | 30.89 | 31.88 | 31.59 | 0.906 | 0.001899 |
| DDS+TV | 31.40 | 33.33 | 32.83 | 0.906 | 0.002566 |
| **DDS+ISCS** | **31.65** | **32.90** | **32.49** | **0.917** | **0.001966** |

### Table 3: BCS vs. ISCS Ablation (SVCT)

| Noise Type | Coronal PSNR | Coronal SSIM | Sagittal PSNR | Sagittal LPIPS |
|-----------|-------------|-------------|-------------|---------------|
| BCS (identical noise) | 38.00 | 0.937 | 38.24 | 0.081 |
| **ISCS (Slerp noise)** | **38.16** | **0.941** | **38.78** | **0.073** |

## Key Findings

1. **ISCS consistently improves across all tasks and baselines**: Regardless of solver (DDNM or DDS) or modality (CT or MRI), adding ISCS yields improvements, and in most metrics surpasses TV regularization, which requires additional optimization.
2. **Gains are most pronounced on coronal and sagittal views**: These two views directly reflect inter-slice consistency along the z-axis — ISCS reduces Sagittal LPIPS on SVCT from 0.141 (DDS) to 0.065 (a 54% reduction), and on LACT from 0.193 to 0.077.
3. **Slerp outperforms BCS**: Identical noise produces conspicuous stripe-like "copy artifacts" in medical volumes; the gradually varying correlation structure of Slerp better accommodates inter-slice anatomical variation.
4. **Inter-slice difference metric |Δ| converges earlier**: ISCS brings the inter-slice difference close to the ground-truth reference early in the sampling process, whereas baselines maintain a large gap until late stages — ISCS narrows the effective search space, enabling more reliable convergence.
5. **DDNM benefits more substantially**: On the LACT task, DDNM+ISCS improves over DDNM by +2.49/+3.13/+3.37 dB (three views), indicating that more underdetermined inverse problems benefit more from noise coordination.
6. **Cost of TV regularization**: Although TV also improves quantitative metrics, it introduces over-smoothing/"cartoonization" artifacts that erase fine anatomical details — ISCS exhibits no such side effect.

## Highlights & Insights

- **Root-cause treatment vs. symptom treatment**: TV regularization post-hoc smooths the consequences of discontinuity; ISCS controls the cause of discontinuity at the source. The former masks symptoms; the latter eliminates the root cause — a more elegant and thorough solution.
- **Principled use of high-dimensional geometry**: The choice of Slerp is not ad hoc but is rigorously motivated by the Gaussian Annulus Theorem — high-dimensional Gaussian noise concentrates on a hyperspherical shell, so interpolation must follow geodesics to preserve distributional invariance.
- **Zero-cost improvement**: No additional computational overhead, no introduced hyperparameters, no retraining required — highly practical for medical imaging settings with limited computational resources.
- **Cross-domain transfer from video to medical imaging**: Inter-frame discontinuity (video) and inter-slice discontinuity (3D medical) share the same root cause (incoherent sampling stochasticity), but directly applying BCS is insufficient; the correlation structure must be tailored to the characteristics of medical volumes (long sequences + large anatomical variation) — exemplifying the research paradigm of "borrowing ideas while adapting to context."

## Limitations & Future Work

1. **Only two inverse problem solvers evaluated**: Experiments are conducted only on DDNM and DDS, without coverage of more recent diffusion-based inverse solvers (e.g., MCG, DiffPIR), leaving generalizability to be further confirmed.
2. **Fixed noise correlation structure**: The Slerp structure with two endpoint anchors and linear position assignment is fixed — neither learned nor data-adaptive. Regions with rapid anatomical variation (e.g., the cervicothoracic junction) may require spatially adaptive correlation fields.
3. **Only VE diffusion models evaluated**: All experiments are conducted under the VE-SDE framework; performance under VP-SDE or flow matching-based frameworks remains unverified.
4. **Limited evaluation data scale**: CT evaluation uses a single patient volume, and MRI also uses a single volume — limiting statistical significance.
5. **Multi-anchor or piecewise Slerp not explored**: For very long sequences (>300 slices), two endpoint anchors may be insufficient to finely control inter-slice correlation in intermediate regions; piecewise interpolation or multi-anchor strategies warrant exploration.

## Related Work & Insights

### vs. DDS+TV (Chung et al., 2024)
DDS+TV performs an additional TV regularization optimization step after re-noising to smooth the z-axis — requiring tuning of a sensitive regularization weight $\lambda$ and risking over-smoothing that erases detail. ISCS operates directly on the re-noising noise itself, requiring no extra optimization steps or hyperparameters. On SVCT, Sagittal LPIPS is 0.065 vs. TV's 0.088 (↓26%), while avoiding cartoonization artifacts.

### vs. BCS (Kwon & Ye, 2025)
BCS is designed for video restoration and applies identical noise to all frames/slices — viable for short videos (<16 frames) but producing "copy artifacts" in medical volumes (>300 slices). ISCS Slerp noise allows inter-slice correlation to decay with distance, better accommodating the locally continuous yet globally varying nature of medical data. Ablation experiments (Table 2) show ISCS outperforms BCS by 0.54 dB PSNR and 0.008 LPIPS on the sagittal view.

### vs. DiffusionBlend (Song et al., 2024)
DiffusionBlend mixes diffusion scores via 3D patch training to enhance 3D consistency — requiring additional 3D training cost and specialized data processing. ISCS requires no training whatsoever, modifying only the noise sampling at inference time, making it simpler and more general.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Concise root-cause analysis + principled solution leveraging high-dimensional geometry; application of Slerp in diffusion sampling is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Cross-validation across three tasks (SVCT + LACT + MRI SR) × two solvers (DDNM + DDS), with ablation and trajectory analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear and coherent logical chain from problem definition → root-cause analysis → solution derivation → experimental validation.
- **Value**: ⭐⭐⭐⭐⭐ Zero additional computation + plug-and-play + open-source code; directly practical for the 3D medical imaging community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](../../CVPR2026/medical_imaging/marker-based_3d_reconstruction_of_aggregates_with_a_comparative_analysis_of_2d_a.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/medical_imaging/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[ICLR 2026\] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[AAAI 2026\] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](../../AAAI2026/medical_imaging/apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)

<!-- RELATED:END -->
