---
title: >-
  [Paper Note] Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion
description: >-
  [NeurIPS 2025][3D Vision][Discrete diffusion models] This paper proposes Scaffold Diffusion, which treats sparse multi-category 3D voxels as token sequences and employs a Masked Diffusion Language Model (MDLM) with 3D sinusoidal positional encodings to generate spatially coherent multi-category voxel structures conditioned on occupancy maps. The method substantially outperforms autoregressive and conventional discrete diffusion baselines on an extremely sparse (>98% backgroun…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Discrete diffusion models"
  - "voxel generation"
  - "sparse structures"
  - "3D generation"
  - "Minecraft"
date: 2026-05-08
content_hash: 9e3f9772ff33348f
---

# Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2509.00062](https://arxiv.org/abs/2509.00062)  
**Code**: [https://scaffold.deepexploration.org/](https://scaffold.deepexploration.org/) (Online Demo)  
**Area**: 3D Vision
**Keywords**: Discrete diffusion models, voxel generation, sparse structures, 3D generation, Minecraft

## TL;DR

This paper proposes Scaffold Diffusion, which treats sparse multi-category 3D voxels as token sequences and employs a Masked Diffusion Language Model (MDLM) with 3D sinusoidal positional encodings to generate spatially coherent multi-category voxel structures conditioned on occupancy maps. The method substantially outperforms autoregressive and conventional discrete diffusion baselines on an extremely sparse (>98% background) Minecraft house dataset.

## Background & Motivation

Sparse multi-category 3D voxel structures are widely applicable in computer vision, robotics, gaming and entertainment, and environment simulation. However, generating such structures poses two unique challenges:

**Memory issue**: Memory consumption of voxel structures grows cubically with resolution ($O(n^3)$), quickly hitting memory bottlenecks as resolution increases.

**Severe class imbalance**: Due to sparsity, over 98% of voxels are empty background, causing generative models to be dominated by the background class and fail to accurately generate foreground structures.

Prior work has studied generative models for binary voxels and general 3D structures (e.g., PointVoxelDiffusion, XCube), but research on multi-category sparse voxel generation remains very limited. Lee et al. attempted to apply multinomial diffusion directly on full voxel grids, but suffered from class imbalance induced by sparsity. Autoregressive models also perform poorly on such extremely sparse data.

The **Core Idea** of this paper is: **model only the occupied voxels**, extracting them as a token sequence and generating category labels with a discrete diffusion language model (MDLM), while introducing 3D positional encodings to maintain spatial coherence. This naturally avoids both background-class dominance and cubic memory scaling.

## Method

### Overall Architecture

The pipeline of Scaffold Diffusion consists of two stages: given a boolean occupancy map $O \in \mathbb{Z}^{D \times D \times D}$ (indicating which positions contain voxels), the positions $\{(x_i, y_i, z_i)\}$ of all $k$ occupied voxels are extracted, and their category labels are encoded into a token sequence of length $L \geq k$ (potentially with padding tokens), which is then fed into MDLM to generate voxel categories at each position. After generation, the full voxel grid is reconstructed from position information.

### Key Designs

1. **Sparse Formulation (modeling occupied voxels only)**: The core design decision — rather than operating on the full $D^3$ voxel grid, only occupied voxel token sequences are extracted. This simultaneously resolves the memory issue (sequence length $L=1024 \ll D^3=32768$ or $262144$) and the class imbalance problem (no longer dominated by vast amounts of empty background tokens). This stands in sharp contrast to Lee et al., who operate on the full voxel grid.

2. **MDLM as the generative model**: The paper adopts the Masked Diffusion Language Model (absorbing-state discrete diffusion), a special case of the D3PM framework. The forward process converts each token to [MASK] with probability $\beta_t$; the reverse process learns to recover the original category from the mask. Training uses a simplified continuous-time ELBO objective. Compared to multinomial diffusion (Lee et al.), MDLM provides a cleaner training objective and better generation quality.

3. **3D sinusoidal positional encodings**: Each voxel's $(x, y, z)$ position is encoded as a 3D sinusoidal embedding (rather than a learnable positional encoding) and injected into the DiT backbone. This enables the model to perceive the spatial positions of tokens and generate spatially coherent structures. Ablation experiments confirm this is significantly more effective than learnable positional encodings (NLL: 0.58 vs. 3.369).

4. **DiT Backbone**: A Diffusion Transformer architecture with 12 blocks, 12 heads, and sequence length 1024. Log-linear noise scheduling and cached updates are used to accelerate inference. EMA $\beta = 0.9999$.

### Loss & Training

The continuous-time ELBO objective of MDLM is used:

$$\mathcal{L} = \mathbb{E}_q \int \frac{\alpha'_t}{1 - \alpha_t} \cdot \log \langle x_\theta(z_t, t),\, x \rangle \, dt$$

Optimization uses AdamW (lr=$3 \times 10^{-4}$, $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$), with a constant warm-up of 2500 steps and a maximum of 1 million training steps. All experiments are run on a single RTX 5090 GPU in under 12 hours each.

## Key Experimental Results

### Main Results

The paper primarily relies on qualitative evaluation (as 3D generation quality is inherently a qualitative task), while also reporting quantitative metrics:

| Method | NLL ↓ | Perplexity ↓ | Qualitative Performance |
|--------|-------|--------------|------------------------|
| Scaffold Diffusion | **0.58** | **1.787** | Realistic, coherent, and diverse house structures |
| Lee et al. (VQ-VAE + multinomial diffusion) | — | — | Over-representation of background tokens; incomplete structures |
| Autoregressive Baseline | — | — | Limited block types; structural collapse |
| DiT-3D (full voxel grid) | — | — | Fails to generate reasonable occupancy structures even with inverse-frequency weighting |

### Ablation Study

| Configuration | NLL ↓ | Perplexity ↓ | Notes |
|---------------|-------|--------------|-------|
| Learnable positional encodings | 3.369 | 29.05 | Fails to learn correct spatial relationships |
| 3D sinusoidal positional encodings | **0.58** | **1.787** | ~6× performance improvement; spatial awareness is critical |

### Key Findings

- **Sparse tokenization is essential**: Methods operating on the full voxel grid (Lee et al., DiT-3D), regardless of whether inverse-frequency loss weighting is applied, are dominated by >98% background tokens and produce poor generation quality. Modeling only occupied voxels fundamentally resolves this issue.
- **Discrete diffusion substantially outperforms autoregressive models**: The autoregressive baseline uses the same backbone with a next-token prediction objective, but generates structures dominated by a few block types or exhibiting implausible placements. This demonstrates the clear advantage of diffusion models on tasks requiring global spatial coherence.
- **3D sinusoidal vs. learnable positional encodings**: The gap is substantial (NLL drops from 3.369 to 0.58), indicating that with limited data (1,432 samples), prior-based positional encodings vastly outperform encodings that must be learned from scratch.
- **Diverse generation**: A single occupancy map can yield multiple distinct yet plausible block material configurations.

## Highlights & Insights

- The voxel generation problem is elegantly reformulated as discrete sequence generation, leveraging mature tools from the language modeling literature.
- The "model only occupied voxels" approach is simple yet highly effective, simultaneously addressing both memory and class imbalance challenges.
- This work demonstrates that discrete diffusion models can extend beyond their native sequential domain to 3D spatial structure generation.
- High-quality structures are generated even under extreme sparsity (98.3% background).
- An interactive online demo is provided for readers to evaluate results independently — a commendable practice worth broader adoption.

## Limitations & Future Work

- The method requires a pre-specified boolean occupancy map as a condition; fully end-to-end generation (first generating the occupancy map, then filling in categories) is left as future work.
- The sequence length constraint of MDLM ($L=1024$) limits the method to relatively small structures.
- Validation is conducted solely on the Minecraft house dataset; generalization to more general 3D scene generation tasks remains untested.
- The dataset is small (1,432 samples), and generalizability warrants further investigation.
- The paper lacks standardized quantitative evaluation metrics and relies primarily on qualitative assessment.

## Related Work & Insights

- Lee et al. first applied discrete diffusion to 3D categorical data, but operating on full voxel grids is fundamentally limited by sparsity.
- XCube uses hierarchical latent diffusion for sparse binary voxel grids but does not handle multi-category settings.
- VoxelCNN provides the 3D-Craft dataset but only performs autoregressive local extension rather than full structure generation.
- MDLM (Sahoo et al.) provides a simplified discrete diffusion training objective for text generation; this paper successfully extends it to 3D.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](../../CVPR2025/3d_vision/shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)
- [\[NeurIPS 2025\] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)
- [\[CVPR 2026\] PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion](../../CVPR2026/3d_vision/partdiffuser_part-wise_3d_mesh_generation_via_discrete_diffusion.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)
- [\[NeurIPS 2025\] Linearly Constrained Diffusion Implicit Models](linearly_constrained_diffusion_implicit_models.md)

</div>

<!-- RELATED:END -->
