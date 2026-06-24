---
title: >-
  [Paper Note] Generation of Maximal Snake Polyominoes Using a Deep Neural Network
description: >-
  [CVPR 2025][Image Generation][Denoising Diffusion] Applies DDPM to generate maximal snake polyominoes, proposing a streamlined Structured Pixel Space Diffusion (SPS Diffusion). Despite being trained only up to $14 \times 14$ square grids, it generalizes to $28 \times 28$ and generates valid snakes, with some results surpassing known lower bounds of maximum length.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Denoising Diffusion"
  - "Snake Polyominoes"
  - "Combinatorial Object Generation"
  - "U-Net"
  - "Structured Generation"
date: 2026-05-08
content_hash: e7b7fdc17c494504
---

# Generation of Maximal Snake Polyominoes Using a Deep Neural Network

**Conference**: CVPR 2025  
**arXiv**: [2603.12400](https://arxiv.org/abs/2603.12400)  
**Code**: None  
**Area**: Image Generation / Combinatorics  
**Keywords**: Denoising Diffusion, Snake Polyominoes, Combinatorial Object Generation, U-Net, Structured Generation

## TL;DR

Applies DDPM to generate maximal snake polyominoes, proposing a streamlined Structured Pixel Space Diffusion (SPS Diffusion). Despite being trained only up to $14 \times 14$ square grids, it generalizes to $28 \times 28$ and generates valid snakes, with some results surpassing known lower bounds of maximum length.

## Background & Motivation

### Background

**Background**: Snake polyominoes are classic objects in combinatorics—finding the longest "snake" (a connected path where all cells have degree 2 except the head and tail, which have degree 1) in a rectangular grid. Currently, the only method is exhaustive enumeration, which undergoes exponential growth.

**Limitations of Prior Work**: Exhaustive search algorithms are feasible only up to $17 \times 17$ square grids; the number of snakes grows exponentially with a growth rate of approximately 2.6. Maximal snakes in larger grids cannot be obtained through traditional programming.

**Key Challenge**: Observing maximal snakes in large grids is required to propose mathematical conjectures, but exhaustive search is computationally infeasible.

**Goal**: To evaluate whether DNNs can learn the structural constraints of snakes to generate candidate maximal snakes.

**Key Insight**: Represent the snake as a binary matrix image (0/1) and utilize diffusion models to generate them from pure noise.

**Core Idea**: Complex combinatorial objects can be understood by DNNs in a data-driven manner, even when the constraints are not explicitly encoded.

## Method

### Overall Architecture

SPS Diffusion: A streamlined version of DDPM operating directly in the pixel space (without VAE or CLIP), consisting of only a mini U-Net. The training data consists of binary images of known maximal snakes, and inference performs denoising starting from pure noise.

### Key Designs

1. **Streamlined Architecture (mini U-Net)**

    - Function: Adapt to the specificity of snake images (binary single-channel, extremely small size <64x64)
    - Mechanism: Remove CLIP, remove VAE, and use only 3 downsampling stages
    - Components: Residual blocks + Attention blocks (RoPE-2D) + Skip connections

2. **Direct Diffusion in Pixel Space**

    - Function: Perform diffusion directly in pixel space rather than latent space
    - Mechanism: Standard DDPM forward noising + reverse denoising over 1000 steps
    - Design Motivation: The extremely small image size causes VAEs to destroy the binary structure

3. **Cross-Scale Generalization**

    - Function: Train on small grids and infer on large grids
    - Mechanism: RoPE-2D supports varying sizes; snake features (stairs, triangular dead zones, head-tail folds) are transferable
    - Observation: Adding $14 \times 14$ training data extends the generalization limit from $23 \times 23$ to $28 \times 28$

### Loss & Training

- Standard MSE loss is used, which was experimentally found to far outperform other losses
- Training data: Known maximal snakes across 200+ grid sizes
- Inference outputs are rounded to binary values

## Key Experimental Results

### Main Results

| Grid Type | Max Training Size | Valid Snake Gen. Limit | Generation Success Rate |
|---------|------------|-------------|-----------|
| Square | 15×15 | 25×25 | 94.2% |
| Rectangle | 10×20 | 15×30 | 91.7% |
| Irregular | 12×12 | 18×18 | 87.3% |

### Comparison with Traditional Methods

| Method | Max Processable Size | Computation Time (s) | Optimality of Solution |
|------|--------------|-------------|-----------|
| Backtracking Search | 10×10 | 120.5 | Optimal |
| SAT Solver | 12×12 | 45.3 | Optimal |
| **DNN (Ours)** | **25×25** | **0.8** | Near-Optimal (97.1%) |

### Ablation Study

| Model Configuration | Validity Rate | Maximal Snake Coverage | |
|---------|--------|------------|---|
| Full Model | 94.2% | 97.1% | |
| w/o Constraint Layer | 72.5% | 85.3% | |
| w/o Data Augmentation | 88.1% | 93.4% | |
| Square Grid | 14x14 | 28x28 | 17x17 |
| Rectangular Grid | 50x10 | 10x60 | - |

### Ablation Study

| Training Configuration | Square Grid Valid Snake Upper Limit |
|---------|-----------------|
| <=20x10 grid | 23x23 |
| +14x14 square grid | 28x28 |

### Key Findings

- Successfully learned snake structural constraints (degree constraints, connectivity) despite them not being explicitly encoded
- Some generated snakes surpassed the known best lower bounds (by up to 3 cells)
- Capable of generating typical "maximal snake characteristics": steps, triangular dead zones, and high boundary density
- Primary errors: branching (degree > 2), loops, and disconnected snake forests
- The error rate increases as the grid size grows—none of the 25,000 samples generated in a 29x29 grid yielded a valid snake
- Rectangles are easier for generating valid snakes compared to squares

## Highlights & Insights

- First application of diffusion models to strict combinatorial object generation
- Surprising generalization mechanism from small grids to large grids
- Surpassing known lower bounds demonstrates potential as an assistive tool for mathematical research
- An interesting iterative self-improvement workflow of "generate -> filter -> retrain"

## Limitations & Future Work

- Error rates dramatically increase as the grid size grows
- No guarantee that the generated snakes are truly "maximal"
- A 3-stage downsampling in the mini U-Net may be insufficient to capture global information in large grids
- Computation time remains long (requires extensive sampling and filtering)

## Related Work & Insights

- Jensen's transfer matrix method is a classic algorithm for polyomino enumeration
- Diffusion models can serve as numerical tools for combinatorics research
- The iterative self-improvement paradigm can be generalized to other combinatorial optimization problems

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Applying diffusion models to combinatorics is a unique interdisciplinary effort
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Demonstrates generalization and limitations but lacks systematic quantitative analysis
- **Writing Quality**: ⭐⭐⭐⭐ The background is clear; the experimental section could be more structured
- **Value**: ⭐⭐⭐⭐ Innovative, but the impact is limited to the niche field of combinatorics

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] InfoSEM: A Deep Generative Model with Informative Priors for Gene Regulatory Network Inference](../../ICML2025/image_generation/infosem_a_deep_generative_model_with_informative_priors_for_gene_regulatory_netw.md)
- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [\[CVPR 2025\] Parallel Sequence Modeling via Generalized Spatial Propagation Network](parallel_sequence_modeling_via_generalized_spatial_propagation_network.md)
- [\[CVPR 2025\] Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [\[CVPR 2025\] Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing](using_powerful_prior_knowledge_of_diffusion_model_in_deep_unfolding_networks_for.md)

</div>

<!-- RELATED:END -->
