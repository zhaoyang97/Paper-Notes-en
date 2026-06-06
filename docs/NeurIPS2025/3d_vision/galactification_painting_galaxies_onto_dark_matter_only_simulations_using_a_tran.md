---
title: >-
  [Paper Note] Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model
description: >-
  [NeurIPS 2025][3D Vision][cosmological simulation] This paper proposes a multimodal Transformer encoder–decoder framework that takes density and velocity fields from inexpensive dark matter N-body simulations as input an…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "cosmological simulation"
  - "Transformer"
  - "dark matter"
  - "galaxy generation"
  - "conditional generative model"
date: 2026-05-08
content_hash: c1e4e612b6a3b535
---

# Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model

**Conference**: NeurIPS 2025
**arXiv**: [2511.08438](https://arxiv.org/abs/2511.08438)  
**Code**: None  
**Area**: 3D Vision / Cosmological Simulation
**Keywords**: cosmological simulation, Transformer, dark matter, galaxy generation, conditional generative model

## TL;DR
This paper proposes a multimodal Transformer encoder–decoder framework that takes density and velocity fields from inexpensive dark matter N-body simulations as input and autoregressively generates galaxy catalogs (positions + physical properties). The model faithfully reproduces hydrodynamical simulation results across multiple statistical metrics while achieving approximately 100× computational speedup.

## Background & Motivation

**Background**: Modern large-scale cosmological surveys require vast ensembles of simulated universes for statistical inference. Hydrodynamical simulations accurately model galaxy formation, but a single run requires ~$2 \times 10^8$ CPU hours, making it infeasible to cover the parameter space.

**Limitations of Prior Work**: Dark matter-only N-body simulations are more than 100× faster, but yield only the dark matter distribution without galaxy information. Traditional approaches such as the Halo Occupation Distribution (HOD) paint galaxies via simple statistical mappings and fail to capture complex subgrid physical dependencies.

**Key Challenge**: Galaxy formation is inherently a stochastic process, requiring a generative model to capture the conditional probability distribution $P(\text{galaxies} \mid \text{DM fields}, \theta)$ rather than a deterministic mapping. Moreover, the number of galaxies itself depends on cosmological and astrophysical parameters and cannot be fixed a priori.

**Goal**: To construct the first accelerated forward model capable of simultaneously learning the spatial distribution of galaxies, their physical properties (stellar mass, velocity, magnitude), and their dependence on cosmological and astrophysical parameters.

**Key Insight**: The problem is formulated as "given 3D fields → generate a point cloud sequence," which naturally suits a multimodal Transformer encoder–decoder architecture.

**Core Idea**: Dark matter fields are encoded via CBAM + ViT, and a cross-attention decoder autoregressively produces a sequence of discretized galaxy tokens, enabling parameter-conditioned galaxy catalog generation.

## Method

### Overall Architecture
**Inputs**: Dark matter density and velocity fields from N-body simulations (5 redshift snapshots × multi-resolution = 30 3D channels), plus 6 cosmological/astrophysical parameters. **Output**: A token sequence decoded into 6-dimensional attributes per galaxy ($x, y, z, v_x, \log M_\star, M_g$). The overall architecture adopts an encoder–decoder Transformer design.

### Key Designs

1. **Multi-Scale Multi-Epoch Input Representation**:

    - **Function**: The $(25\ \text{Mpc}/h)^3$ volume is divided into 8 sub-volumes, each gridded at $16^3$ resolution, supplemented by low-resolution environmental fields at 1.5× and 3× extent.
    - **Mechanism**: Five redshift snapshots ($z=0, 0.1, 0.3, 0.6, 1.0$) are used to capture the temporal evolution of large-scale structure; 30 channels are concatenated along the channel dimension.
    - **Design Motivation**: Galaxy formation depends on both local density peaks and large-scale environment as well as structure growth history; multi-scale and multi-epoch information are both indispensable.

2. **CBAM + Vision Transformer Encoder**:

    - **Function**: Extracts features from the multi-channel 3D input.
    - **Mechanism**: CBAM (channel attention + spatial attention) first extracts local features; 3 ViT layers then model long-range correlations via self-attention. Cosmological/astrophysical parameters are appended to the ViT output tokens.
    - **Design Motivation**: CBAM performs spatial filtering to focus the model on information-rich regions, while ViT captures long-range cosmic web structural correlations.

3. **Autoregressive Token Sequence Decoder**:

    - **Function**: Generates the galaxy catalog as a discrete token sequence.
    - **Mechanism**: Each of the 6 attributes per galaxy is discretized into 64 bins, yielding 6 tokens per galaxy (one "word"). All galaxies are ordered by stellar mass in descending order to form a "sentence," delimited by START/END tokens. The decoder maps tokens into a 192-dimensional space with attribute-type embeddings and RoPE positional encodings; 4 Transformer layers (8-head attention) receive encoder output via cross-attention and predict the next-token probability distribution.
    - **Design Motivation**: The variable galaxy count is handled naturally by the END token, offering greater flexibility than fixed-count point cloud diffusion methods; RoPE facilitates capturing relative dependencies among tokens.

### Loss & Training
- **Training Loss**: Cross-entropy loss following a standard autoregressive language model training paradigm.
- **Dataset**: CAMELS Illustris-TNG Latin hypercube; 1,000 paired N-body/hydro simulations with an 80/12.5/7.5 train/validation/test split.
- **Inference**: A complete galaxy catalog is generated in ~30 seconds on a single H200 GPU, compared to 6,000 CPU hours for the original hydrodynamical simulation.

## Key Experimental Results

### Main Results
The paper validates results using multi-level summary statistics without a conventional numerical baseline table. Core findings are as follows:

| Validation Dimension | Metric | Result |
|---|---|---|
| 6D joint distribution | PQMass $\chi^2$ test | $\chi^2$ histogram of mock vs. truth perfectly matches the theoretical distribution (20 partitions) |
| One-point statistics | Stellar mass / g-band magnitude / velocity histograms | 16–84 percentile intervals consistent with truth; $\Omega_m$ dependence correctly captured |
| Two-point statistics | Redshift-space power spectrum ($k \sim 10\ h/$Mpc) | Unweighted, g-band weighted, and mass-weighted power spectra all match truth |
| Visual comparison | 3 test simulations with different cosmological parameters | Galaxy spatial distributions visually indistinguishable from truth |

### Computational Efficiency

| Method | Computational Cost | Speedup |
|---|---|---|
| Hydrodynamical simulation | ~6,000 CPU hours/run | 1× |
| Ours (inference) | ~30 s / single H200 GPU | ~100× |

### Key Findings
- PQMass testing indicates that mock and truth originate from the same underlying distribution, demonstrating that the model not only approximates the mean but correctly learns the full distribution.
- Weighted power spectrum comparisons constitute a rigorous joint-distribution test—the model successfully captures the joint dependence of positions and attributes.
- Sixteen independent samples effectively reproduce the stochasticity of galaxy formation; high cross-correlation with any single truth realization is not expected.
- The model correctly captures the dependence of galaxy properties on $\Omega_m$, confirming the effectiveness of the parameter conditioning mechanism.
- Power spectrum consistency is maintained down to small scales of $k \sim 10\ h/$Mpc, substantially extending the scale range of prior work.

## Highlights & Insights
- **Discretized token representation**: Binning continuous physical quantities into a 64-token vocabulary allows the standard Transformer language modeling framework to be applied directly—the approach of "converting a scientific problem into a language modeling problem" is particularly elegant.
- **Variable-length output**: The variable number of galaxies across parameters is handled naturally by the END token, offering greater flexibility than fixed-count point cloud diffusion methods.
- **Multi-scale environment encoding**: Appending low-resolution fields provides large-scale context; this "local + environment" design is transferable to any conditional generation task.
- A 100× computational speedup while preserving full statistical characteristics opens new avenues for simulation-based inference (SBI).
- **Attribute-type embeddings**: Using indices 1–6 to distinguish tokens corresponding to different physical attributes within the same sequence provides a simple and effective solution to the multi-attribute mixed-encoding problem.
- **Redshift snapshot stacking**: Five snapshots at different redshifts encode structure growth history, effectively allowing the model to "observe" the evolution of large-scale structure rather than only its final state.

## Limitations & Future Work
- Validation is limited to the $(25\ \text{Mpc}/h)^3$ small volume; scaling to larger survey volumes requires addressing the sequence length explosion problem (the authors mention sparse/linear attention as a potential solution).
- The current set of output attributes is limited (3D position + $v_x$ + $\log M_\star$ + $M_g$) and does not yet include multi-band photometry or full 3D velocities.
- Validation is performed only on the Illustris-TNG subgrid model; cross-model generalization (e.g., to Astrid) is only briefly mentioned.
- The model operates at a single redshift $z=0$ and does not extrapolate to other redshifts.
- Explicit ablation studies and numerical baseline comparison tables against HOD, diffusion models, and similar methods are absent; validation relies solely on indirect statistical metrics.
- Whether the 64-bin discretization precision is sufficient for high-precision cosmological inference has not been assessed through sensitivity analysis.

## Related Work & Insights
- **vs. Bourdin et al. (2024) diffusion model**: Predicts only galaxy number counts without attributes; the proposed method generates complete point clouds with attributes.
- **vs. Cuesta-Lazaro & Mishra-Sharma (2024) point cloud diffusion**: Assumes a fixed output count, targets only dark matter halos, and does not condition on parameter variation; the proposed method supports variable length and full parameter conditioning.
- **vs. Pandey et al. (2024)**: This work is a direct extension thereof, upgrading from dark matter halo prediction to galaxy prediction with added parameter conditioning and smaller-scale ($k \sim 10\ h/$Mpc) structural fidelity.
- **vs. HOD methods**: Traditional HOD uses analytic functions to describe galaxy occupation statistics within halos; its parameterization capacity is limited and effects such as assembly bias are neglected. The proposed data-driven approach automatically captures these complex dependencies.
- **Implications for ML for Science**: This work demonstrates the feasibility of reformulating physical simulation tasks as token sequence generation, offering transferable insights for domains such as molecular generation and fluid simulation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to achieve parameter-conditioned full-attribute galaxy generation, though the Transformer encoder–decoder framework itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐ Multi-level statistical validation is comprehensive, but explicit ablation studies and numerical baseline comparison tables are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The 4-page main text is concise and clear, with well-articulated problem formulation and method description.
- **Value**: ⭐⭐⭐⭐⭐ A 100× speedup carries significant practical importance for cosmological SBI, enabling direct deployment into inference pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes](rgb-only_supervised_camera_parameter_optimization_in_dynamic_scenes.md)
- [\[ICML 2026\] PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations](../../ICML2026/3d_vision/plaid_a_unified_data_model_for_machine_learning_on_heterogeneous_physics_simulat.md)
- [\[NeurIPS 2025\] Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)
- [\[NeurIPS 2025\] How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?](how_many_tokens_do_3d_point_cloud_transformer_architectures_really_need.md)
- [\[NeurIPS 2025\] URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model](urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model.md)

</div>

<!-- RELATED:END -->
