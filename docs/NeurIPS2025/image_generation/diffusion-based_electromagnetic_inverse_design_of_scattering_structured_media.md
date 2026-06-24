---
title: >-
  [Paper Note] Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media
description: >-
  [NeurIPS 2025][Image Generation][Conditional Diffusion Model] This paper proposes a conditional diffusion model-based framework for electromagnetic inverse design that directly generates dielectric-sphere metasurface geometries from target differential scattering cross sections (DSCS), bypassing costly iterative optimization. The approach naturally handles the non-uniqueness of the inverse problem and outperforms CMA-ES evolutionary optimization while being orders of magnitud…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Conditional Diffusion Model"
  - "Metasurface Inverse Design"
  - "Electromagnetic Scattering"
  - "FiLM Conditioning"
  - "Meta-structure Generation"
date: 2026-05-08
content_hash: b65c9523900b9075
---

# Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media

**Conference**: NeurIPS 2025
**arXiv**: [2511.05357](https://arxiv.org/abs/2511.05357)  
**Code**: Available ([https://github.com/mikzuker/inverse_design_metasurface_generation](https://github.com/mikzuker/inverse_design_metasurface_generation))  
**Area**: Image Generation / Diffusion Models / Electromagnetic Inverse Design
**Keywords**: Conditional Diffusion Model, Metasurface Inverse Design, Electromagnetic Scattering, FiLM Conditioning, Meta-structure Generation

## TL;DR

This paper proposes a conditional diffusion model-based framework for electromagnetic inverse design that directly generates dielectric-sphere metasurface geometries from target differential scattering cross sections (DSCS), bypassing costly iterative optimization. The approach naturally handles the non-uniqueness of the inverse problem and outperforms CMA-ES evolutionary optimization while being orders of magnitude faster.

## Background & Motivation

- **Value of metasurfaces**: Engineered metasurfaces enable precise control of electromagnetic waves, with applications in high-resolution imaging, compact optical devices, and next-generation wireless communications.
- **Challenges in inverse design**:
  1. Nonlinear boundary conditions make the structure–response mapping highly complex.
  2. The design space is high-dimensional.
  3. **One-to-many problem**: A single scattering profile can correspond to multiple distinct geometries.
- **Bottlenecks of traditional methods**: Topology optimization and genetic algorithms rely on iterative simulations, incurring high computational costs and requiring expert tuning (CMA-ES requires 15–20 hours per optimization run).
- **Advantages of generative models**: Generative models can sample from the distribution of feasible designs, naturally handling the one-to-many characteristic; diffusion models offer stable training and diverse outputs.

## Method

### Overall Architecture

An end-to-end conditional generation pipeline:

1. **Forward simulator** (SMUTHI): Computes DSCS for a $2\times2$ array of dielectric spheres → generates 11,000 training samples.
2. **Conditional diffusion model**: Learns the inverse mapping from DSCS profiles to metasurface geometries.
3. **Inference**: Given a target DSCS, samples multiple candidate structures.

#### Metasurface Parameterization

- A virtual square substrate is divided into an $N \times N$ grid (here $N=2$, i.e., 4 cells).
- Each cell contains one dielectric sphere described by 3 parameters: center $(x, y)$ and radius $r$.
- Encoded as a 1D vector $\in \mathbb{R}^{3N^2} = \mathbb{R}^{12}$, with all parameters normalized to $[0,1]$.
- Conditional input: DSCS values at 10 polar angles.

### Key Designs

#### 1. 1D U-Net Denoising Network

- A one-dimensional U-Net architecture processes the 12-dimensional geometry vector.
- Channel configuration: {16, 32, 64, 128, 128, 64, 32, 16}.
- DDPM framework with 1,000 denoising steps.

#### 2. FiLM Conditioning Mechanism

Feature-wise Linear Modulation (FiLM) is used to inject the target DSCS into the network:

$$\text{FiLM}(F_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}$$

- $\gamma$ and $\beta$ are produced by a two-layer network from the 10-dimensional DSCS condition vector.
- FiLM transformations are applied at each layer of the U-Net, enabling the model to adapt to specific electromagnetic response requirements.

#### 3. Noise Schedule

A cosine noise schedule (Nichol & Dhariwal, 2021) is employed:

$$\bar{\alpha}_t = \frac{f(t/T)}{f(0)}, \quad f(\tau) = \cos^2\left(\frac{\tau + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

#### 4. Forward Simulator

- The SMUTHI package (based on the T-matrix method) is used for efficient computation of electromagnetic scattering by spherical objects.
- A fixed refractive index of $n = 2$ is used as a hyperparameter.
- 11,000 unique samples with corresponding 10-angle DSCS values are generated.

### Loss & Training

- **Loss function**: Standard DDPM denoising loss, minimizing the L2 distance between predicted and true noise:

$$\mathcal{L} = \mathbb{E}_{t, y_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(y_t, t)\|^2\right]$$

- Learning rate $4 \times 10^{-6}$, batch size 16, trained for 116 epochs.
- EMA decay coefficient 0.995.
- Evaluation metric: Mean Percentage Error (MPE) between the DSCS of generated structures and the target DSCS.

## Key Experimental Results

### Main Results: Generation Quality and Generalization

| Metric | Value |
|--------|-------|
| Best single-sample MPE | **1.39%** |
| Median MPE over 40 samples | 18.91% |
| Interquartile range | Compact, no significant outliers |

- Tested on **unseen** metasurface configurations (out-of-distribution generalization).
- The best sample nearly perfectly reproduces the target DSCS profile.
- Different samples correspond to distinct geometries yet yield consistent scattering responses—validating the model's natural handling of inverse problem non-uniqueness.

### Comparison with CMA-ES Evolutionary Optimization

| Metric | Diffusion Model | CMA-ES |
|--------|----------------|--------|
| MPE | **~3%** | ~5% |
| One-time training cost | 6 hours | — |
| Per-design inference time | **Seconds** | 15–20 hours |
| Forward simulation calls | 11,000 (one-time) | ~105,000 per run |
| Multi-problem advantage | Amortized cost decreases | Re-optimized each time |

Key distinction: The diffusion model's computational cost is **one-time**; after training, each generation requires no additional simulation calls. CMA-ES requires a full optimization for every new problem instance.

### Ablation Study

- Checkpoints are saved every 1,000 training steps; all three MPE statistics (mean, median, standard deviation) show consistent decrease throughout training.
- Stable convergence within 116 epochs demonstrates that the model successfully learns the physics–geometry mapping.

### Key Findings

1. **Generation quality surpasses optimization**: Diffusion model MPE (~3%) outperforms CMA-ES (~5%).
2. **Orders-of-magnitude speedup**: Post-training inference takes only seconds, versus tens of hours for CMA-ES.
3. **Natural handling of non-uniqueness**: Different samples yield geometrically distinct yet scattering-equivalent designs.
4. **Out-of-distribution generalization**: High-quality designs are generated for unseen target DSCS profiles.

## Highlights & Insights

- **Amortization argument for computational efficiency**: Forward simulation cost is 11K calls (one-time) versus 105K calls per problem for CMA-ES—an overwhelming advantage when solving multiple inverse design problems.
- **Implicit learning of physical constraints**: The model learns the geometry–scattering mapping from data, without explicitly embedding Maxwell's equations.
- **Non-uniqueness as diversity**: The "ill-posed" nature of the inverse problem becomes an advantage in the generative modeling framework—providing multiple candidate design solutions.
- **Practical feasibility**: Generated structures are realizable under RF laboratory conditions (~10 GHz, ~30 cm scale).

## Limitations & Future Work

1. **Scale limitation**: Only a $2\times2$ grid (12-dimensional parameters) is validated; extension to larger grids (e.g., $4\times4$, $8\times8$) remains to be explored.
2. **Limited conditioning information**: Only 10-angle DSCS values are used as conditions; denser angular sampling may improve accuracy.
3. **Fixed refractive index**: The refractive index is fixed as a hyperparameter rather than treated as a designable variable.
4. **High median MPE** (18.91%): While the best samples are excellent, overall average quality has room for improvement.
5. **Small training dataset** (11K): Scaling behavior with larger datasets and more complex structures has not been explored.
6. **Caution in comparison with CMA-ES**: The two approaches represent fundamentally different methodologies; a comprehensive comparison requires additional evaluation dimensions.

## Related Work & Insights

- **An et al. (2019)**: Used GANs to generate multifunctional metasurfaces, but training instability was a concern.
- **Pahlavani et al. (2022)**: Applied VAEs to generate 3D-printed mechanical metamaterials, demonstrating the potential of generative models for inverse design.
- **Bastek et al. (2022)**: Used deep learning to invert structure–property mappings of truss metamaterials.
- **FiLM** (Perez et al., 2017): A general-purpose conditioning layer for visual reasoning, here adapted for physics-based conditioning.
- **Insight**: The application of diffusion models to physical inverse design remains at an early stage, with potential extensions to photonic crystals, antenna arrays, and acoustic metamaterials.

## Rating

| Dimension | Score | Comment |
|-----------|-------|---------|
| Novelty | ★★★☆☆ | The method itself is standard DDPM+FiLM; innovation lies in the application domain. |
| Technical Depth | ★★★☆☆ | Architecture is concise and effective, but theoretical contributions are limited. |
| Experimental Thoroughness | ★★★☆☆ | Feasibility is demonstrated, but scale is small and comparisons are limited. |
| Value | ★★★★☆ | Practical demand for electromagnetic inverse design is clear; speedup is substantial. |
| Writing Quality | ★★★★☆ | Concise and clear; problem formulation is well-defined and figures are well-presented. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](../../ICLR2026/image_generation/rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[NeurIPS 2025\] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)

</div>

<!-- RELATED:END -->
