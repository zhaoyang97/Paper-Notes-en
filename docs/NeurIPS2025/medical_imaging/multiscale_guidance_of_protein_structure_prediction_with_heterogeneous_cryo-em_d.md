---
title: >-
  [Paper Note] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data
description: >-
  [NeurIPS 2025][Medical Imaging][protein structure prediction] CryoBoltz leverages cryo-EM density maps to guide the sampling trajectory of a pretrained diffusion-based structure prediction model (Boltz-1) via a multiscale guidance mechanism (global → local), generating multi-conformational atomic models consistent with experimental data without any retraining.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - protein structure prediction
  - cryo-EM
  - diffusion model guidance
  - conformational diversity
  - Boltz-1
date: 2026-05-08
content_hash: 763d47dde60dd727
---

# Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data

**Conference**: NeurIPS 2025
**arXiv**: [2506.04490](https://arxiv.org/abs/2506.04490)
**Code**: [GitHub](https://github.com/ml-struct-bio/cryoboltz)
**Area**: Medical Imaging
**Keywords**: protein structure prediction, cryo-EM, diffusion model guidance, conformational diversity, Boltz-1

## TL;DR

CryoBoltz leverages cryo-EM density maps to guide the sampling trajectory of a pretrained diffusion-based structure prediction model (Boltz-1) via a multiscale guidance mechanism (global → local), generating multi-conformational atomic models consistent with experimental data without any retraining.

## Background & Motivation

The field of protein structure prediction faces two major challenges:

**Single-conformation bias in structure prediction models**: Diffusion models such as AlphaFold3 and Boltz-1 can generate structures, but their sampling distributions are heavily concentrated on a single conformation—for example, Boltz-1 samples only the outward-facing conformation of STP10, while AlphaFold3 samples only the inward-facing one. Existing MSA subsampling strategies moderately increase diversity but remain limited and cannot cover the conformational continuum.

**Bottleneck from cryo-EM reconstruction to atomic models**: Cryo-EM experiments can capture the conformational landscape of proteins, yet the resulting 3D density maps are not atomic models. Existing model-building methods (e.g., ModelAngelo) require high-resolution maps (<4 Å) and frequently fail on low-resolution (>4 Å) or heterogeneous complexes—for instance, ModelAngelo successfully modeled only 2.3%–40.3% of residues across four conformations of P-glycoprotein.

The core idea of this paper is to **inject experimental information from cryo-EM density maps into the reverse sampling process of a pretrained diffusion model**, thereby exploiting both the sequence and biophysical priors learned by the structure prediction model and the real conformational information captured by experimental data.

## Method

### Overall Architecture

CryoBoltz is built on the Diffusion Posterior Sampling (DPS) framework, treating the pretrained Boltz-1 diffusion model as an implicit prior $p(\mathbf{x}|\mathbf{s})$ and the cryo-EM density map as observation $\mathbf{y}$, guiding the reverse diffusion process to sample the posterior $p(\mathbf{x}|\mathbf{y},\mathbf{s})$. The entire process consists of four stages over 200 diffusion steps:

**Warm-up → Global Guidance → Local Guidance → Relaxation**

### Key Designs

1. **Global Guidance**: The density map $\mathbf{y} \in \mathbb{R}^{w \times h \times d}$ is converted into a 3D point cloud $\mathbf{Y} \in \mathbb{R}^{k \times 3}$ via weighted k-means clustering, where $k = \lfloor N/(4r^3) \rfloor$ ($N$ is the number of atoms, $r$ is the voxel size). The guidance term is based on the Sinkhorn divergence (regularized Wasserstein distance):

$$\tilde{s}_\theta(\mathbf{y}, \mathbf{x}, \mathbf{s}, t) = -\nabla_\mathbf{x} \mathfrak{D}(\hat{\mathbf{x}}_\theta(\mathbf{x}, \mathbf{s}, t), \mathbf{Y})$$

This stage focuses solely on the global shape of the protein (e.g., which helices face inward or outward) without involving high-resolution details, thereby avoiding optimization difficulties arising from nonlinear likelihood functions in the early diffusion steps. Guidance strength is annealed via cosine scheduling from 0.25 to 0.05.

2. **Local Guidance**: Using the raw density map and a cryo-EM physical forward model (each non-hydrogen atom contributes a Gaussian scattering potential), the guidance term directly minimizes the L2 distance between the simulated and experimental density maps:

$$\tilde{s}_\theta(\mathbf{y}, \mathbf{x}, \mathbf{s}, t) = -\nabla_\mathbf{x} \|\mathbf{y} - \mathcal{B}(\Gamma(\hat{\mathbf{x}}_\theta(\mathbf{x}, \mathbf{s}, t), \mathbf{s}))\|^2$$

where $\Gamma$ maps atomic coordinates to a density volume and $\mathcal{B}$ simulates the blurring effect of B-factors. Guidance strength is fixed at $\lambda=0.5$.

3. **Multiscale Guidance Scheduling**: For synthetic data, $T_w=125, T_g=25, T_l=25, T_r=25$; for experimental data, $T_w=100, T_g=50, T_l=25, T_r=25$ (more global guidance steps are used for experimental data to drive larger conformational changes). The warm-up and relaxation stages apply no guidance—warm-up establishes a reasonable initialization, while relaxation corrects steric clashes and other fine-grained details.

### Loss & Training

CryoBoltz **requires no training whatsoever** and operates entirely as an inference-time guidance method. A key advantage is that it automatically benefits from continued improvements to the underlying base model. For each experiment, 25 structures × 3 replicates are sampled, and the structure with the lowest RMSD is selected as the final result. Computations are performed on a single A100 80 GB GPU.

## Key Experimental Results

### Main Results (Synthetic Data)

| System | Metric | CryoBoltz | Boltz-1 | Boltz-1+MSA Subsampling | AlphaFold3 |
|---|---|---|---|---|---|
| STP10 (inward) | All-atom RMSD (Å) | **1.057** | 3.815 | 3.768 | 1.263 |
| STP10 (outward) | All-atom RMSD (Å) | **0.888** | 2.656 | 2.542 | 4.478 |
| CH67 antibody | Local RMSD (Å) | **1.269** | 3.120 | 3.270 | 3.191 |
| CH67 antibody | TM score | **0.994** | 0.972 | 0.969 | 0.971 |

### Main Results (Experimental Data)

| System | Resolution (Å) | CryoBoltz RMSD | Boltz-1 RMSD | AF3 RMSD | ModelAngelo Coverage |
|---|---|---|---|---|---|
| P-gp (apo) | 4.3 | **1.382** | 6.994 | 3.827 | 40.3% |
| P-gp (inward) | 4.4 | **1.348** | 5.630 | 2.692 | 18.3% |
| P-gp (occluded) | 4.1 | **1.727** | 2.929 | 3.440 | 2.3% |
| Pma1 (inhibited) | 3.52 | **1.999** | 6.140 | 8.017 | 72.8% |
| CYP (closed) | 4.4 | **2.004** | 8.784 | 3.585 | 18.9% |
| CYP (open) | 6.5 | **4.167** | 8.532 | 6.490 | 0.0% |

### Ablation Study

| Configuration | STP10 Inward RMSD | STP10 Outward RMSD | Notes |
|---|---|---|---|
| Full CryoBoltz | **1.057** | **0.888** | Global + local guidance |
| Local guidance only | 3.860 | 2.722 | Lacks global guidance; fails to drive large conformational changes |
| Global guidance only | 1.287 | 1.164 | Lacks local guidance; insufficient high-resolution detail |

### Key Findings

- CryoBoltz outperforms both Boltz-1 and AlphaFold3 on **all 10 experimental density maps**.
- Unguided Boltz-1/AF3 typically samples only one conformation; CryoBoltz successfully recovers two to four distinct conformations.
- ModelAngelo fails severely at low-resolution maps (>4 Å), achieving 0% coverage at 6.5 Å.
- Global guidance drives large-scale conformational changes, while local guidance refines high-resolution details; both components are indispensable.

## Highlights & Insights

- **Zero training overhead**: A purely inference-time method that can be plug-and-play integrated into any diffusion-based structure prediction model.
- **Necessity of multiscale guidance**: The forward mapping from density maps to atomic models is highly nonlinear; directly optimizing the likelihood leads to multimodality issues, while the point-cloud intermediate representation elegantly resolves optimization instability in the early guidance stages.
- **High practical value**: cryo-EM model building typically requires hours of manual refinement; CryoBoltz accomplishes this within minutes.

## Limitations & Future Work

- Optimization stability is affected by the multimodality of the likelihood $p(\mathbf{y}|\mathbf{x})$, necessitating sampling of multiple structures and selecting the best, which increases computational cost.
- The warm-up stage relies on reasonable initialization by the base model; initialization fails for some complex systems (e.g., the DSL1/SNARE complex).
- The step allocation across guidance stages is set heuristically, and no automatic selection strategy has been established.
- In multi-density-map scenarios, a shared deformation model as an alternative to independent optimization remains unexplored.

## Related Work & Insights

- **DPS / ScoreALD**: Diffusion model posterior sampling frameworks; CryoBoltz is directly built upon DPS.
- **ROCKET**: Uses AF2 as a regularizer but optimizes in AF2's latent space; CryoBoltz optimizes directly in atomic space.
- **ModelAngelo**: The current state-of-the-art de novo model-building method, but constrained by high-resolution requirements.
- Insight: Inference-time guidance of diffusion models represents a general paradigm for integrating experimental data with learned priors.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of the DPS guidance framework to protein structure prediction integrated with cryo-EM data.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic and real data, six biological systems, multiple baselines, comprehensive ablations, and statistical tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation, rigorous methodological derivation, and high-quality figures.
- Value: ⭐⭐⭐⭐⭐ Addresses an important bottleneck in structural biology, with open-source code and strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Quantifying the Role of OpenFold Components in Protein Structure Prediction](quantifying_the_role_of_openfold_components_in_protein_structure_prediction.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[ICCV 2025\] CryoFastAR: Fast Cryo-EM Ab initio Reconstruction Made Easy](../../ICCV2025/medical_imaging/cryofastar_fast_cryo-em_ab_initio_reconstruction_made_easy.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[CVPR 2026\] cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](../../CVPR2026/medical_imaging/cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)

<!-- RELATED:END -->
