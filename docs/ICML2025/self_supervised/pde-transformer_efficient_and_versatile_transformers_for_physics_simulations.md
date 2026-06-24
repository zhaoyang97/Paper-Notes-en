---
title: >-
  [Paper Note] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations
description: >-
  [ICML 2025][Self-Supervised Learning][PDE solving] Proposes PDE-Transformer, an improved Transformer architecture for physics simulations. By separating channel embedding, shifted-window attention, and a multi-scale U-shaped structure, it outperforms existing SOTA on 16 PDE types and demonstrates strong transfer capability to downstream tasks.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "PDE solving"
  - "Transformer architecture"
  - "physics simulation surrogate model"
  - "foundation model pre-training"
  - "multi-scale attention"
date: 2026-05-08
content_hash: 882f231bc9c28974
---

# PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations

**Conference**: ICML 2025  
**arXiv**: [2505.24717](https://arxiv.org/abs/2505.24717)  
**Code**: [tum-pbs/pde-transformer](https://github.com/tum-pbs/pde-transformer)  
**Area**: Self-Supervised Learning  
**Keywords**: PDE solving, Transformer architecture, physics simulation surrogate model, foundation model pre-training, multi-scale attention

## TL;DR

Proposes PDE-Transformer, an improved Transformer architecture for physics simulations. By separating channel embedding, shifted-window attention, and a multi-scale U-shaped structure, it outperforms existing SOTA on 16 PDE types and demonstrates strong transfer capability to downstream tasks.

## Background & Motivation

Machine learning surrogate models for physics simulation face several core challenges: (1) the inherent multi-scale nature of physical systems; (2) tight coupling between data representation and numerical methods (regular grids/meshes/particles); (3) huge differences in the number of physical channels and dynamics across different PDE types; (4) the need for models to surpass traditional numerical methods in accuracy or speed to have practical value.

Limitations of Prior Work:

- **Diffusion Transformer (DiT)**: Global self-attention causes computational complexity to scale quadratically with the token count, preventing direct handling of high-resolution raw data; its fixed physical channel embedding causes inconsistent information density when sharing tokens across different PDEs.
- **scOT**: Although it introduces hierarchical structures and shifted windows, it is not tailored specifically for physical PDE simulations, leaving room for performance improvement.
- **MPP**: Multi-physical pre-training based on axial ViT, but lacks fine-grained control over channel-wise interactions.

The core motivation of PDE-Transformer is to design a generic Transformer backbone that is both highly efficient and scalable, and capable of unifying the handling of multiple types of PDEs, making it suitable as a foundation model in physical sciences.

## Method

### Overall Architecture

Based on the DiT architecture, PDE-Transformer undergoes five key modifications to form a dedicated architecture for physical simulation:

1. **Direct operation on raw data** (non-latent space) — splitting input into spatio-temporal tokens via patches.
2. **Multi-scale U-shaped structure** — PixelShuffle up/down-sampling + skip connections.
3. **Shifted window attention** — replacing global attention to scale linearly to high resolutions.
4. **Separated Channel (SC) embedding** — embedding each physical channel independently, with channel-wise interactions through axial attention.
5. **Deep conditioning mechanism** — injecting conditional information such as PDE types and channel types via adaLN-Zero blocks.

Overall pipeline: Given an input $\mathbf{u}_{\text{in}}$ representing snapshots from $T_p$ time steps, it is processed via Patch embedding $\rightarrow$ multi-scale Transformer encoding-decoding $\rightarrow$ outputting the next-step prediction $\mathbf{u}_{\text{out}}$.

### Key Designs

#### 1. Patch Embedding and Expansion Rate

Given a patch size $p$, the input $T \times H \times W$ is split into $H/p \cdot W/p$ patches of size $T \times p \times p$, which are linearly mapped into $d$-dimensional token vectors. The **expansion rate** is defined as:

$$E(p) = \frac{d}{p^2 \cdot T}$$

The expansion rate controls the token information density: a low expansion rate benefits scalability (fewer tokens), but small patches (high expansion rate) achieve higher accuracy. The paper systematically explores this accuracy-computation trade-off.

#### 2. Multi-scale U-shaped Structure

Unlike the flat structure of the original DiT, PDE-Transformer introduces a hierarchical design:

- Uses **PixelShuffle** and **PixelUnshuffle** for token up- and down-sampling at the end of each Transformer stage.
- Uses **skip connections** between encoder and decoder stages of the same resolution.
- Forms a UNet-like multi-scale structure, naturally suited for the multi-scale physics of systems.
- Unlike Bao (2023) and Hoogeboom (2023), this work uses adaLN for conditioning rather than cross-attention, which is more efficient.

#### 3. Shifted Window Attention

To avoid the $O(N^2)$ computational bottleneck of global self-attention, a Swin-Transformer-style shifted window mechanism is used:

- Window size $w$: each window contains $w \times w$ spatio-temporal tokens.
- Shift windows by $w/2$ between adjacent layers to prevent discontinuities at window boundaries.
- **Does not use absolute positional encoding**, replacing it with **log-spaced relative positions** within the window, combined with a feed-forward network to compute attention scores (derived from Swin V2).
- Advantages: Strengthens translation invariance (crucial for PDE learning) and improves generalization across different window resolutions.

#### 4. Mixed Channel (MC) vs. Separated Channel (SC) Representation

This is the most core design innovation of this paper, resolving the issue of varying numbers of physical channels across different PDEs:

**Mixed Channel (MC)**: Defines a maximum channel count $C_{\max}$, concatenating all channels into $T \times C_{\max} \times p \times p$ patches, padding with zeros if insufficient. Issues: (a) The expansion rate is compressed by $1/C_{\max}$, causing over-compression of the token representation; (b) It mixes channels with totally different physical meanings.

**Separated Channel (SC) (Ours)**:

- Each physical channel is **independently embedded** into a token sequence.
- Spatial-wise window self-attention is used (tokens from different channels do not interact within this step).
- Introduces an additional **channel-wise axial self-attention (channel-wise axial MHSA)**: tokens from different channels at the same spatial position interact through this mechanism.
- The expansion rate remains consistent for each channel, and the computational cost scales linearly with the number of channels.
- Channel types (velocity, density, vorticity, etc.) are injected as conditional information.

The SC design enables different PDE types to share the same token information density, significantly enhancing joint learning and transfer capability across multiple PDEs.

#### 5. Conditioning Mechanism (adaLN-Zero)

Inheriting the adaptive layer normalization mechanism from DiT, it extends the range of conditional information:

- **PDE type labels**: corresponding to class labels in DiT.
- **Physical channel type labels** (for the SC version): density, vorticity, velocity, etc.
- **Diffusion timesteps** (when trained as a diffusion model).
- All condition embeddings are summed and then integrated to regress scale and shift vectors through a feed-forward network.
- Residual blocks are initialized to identity functions (Zero initialization) to accelerate training convergence.
- All labels are dropped out with a 10% probability, enabling both conditional and unconditional inference.

#### 6. Boundary Condition Handling

Explicitly supports periodic and non-periodic boundary conditions:

- When shifting attention windows, tokens are rolled along the x and y axes (simulating periodic boundary conditions).
- For non-periodic cases, cross-boundary token interactions are forbidden by masking attention scores.

#### 7. Algorithmic Improvements

- Applies **RMSNorm** to Q and K of self-attention to prevent instability from uncontrolled growth of attention entropy.
- Adjusts learning rate from DiT's $1.0 \times 10^{-4}$ to $4.0 \times 10^{-5}$.
- Uses AdamW optimizer with a weight decay coefficient of $10^{-15}$ (recommended for bf16 training).
- Implements **gradient clipping** based on gradient EMA to eliminate spikes in training loss.

### Loss & Training

**Supervised Training**: For deterministic tasks (e.g., surrogate models of deterministic solvers), single-step inference is trained using MSE loss:

$$\mathcal{L}_S = \mathbb{E} \left[ \| \mathcal{M}_\Theta(\mathbf{u}_{\text{in}}, \mathbf{c}) - \mathbf{u}_{\text{out}} \|_2^2 \right]$$

**Diffusion Training**: For tasks with wide posterior distributions, the model is trained as a diffusion model to support generative inference. The diffusion timestep is injected into adaLN-Zero as an additional condition.

**Pre-training Strategy**: Pre-trained on autoregressive next-step prediction on the PDEBench+ dataset (16 PDE types). It relies only on the first few snapshots (without simulation parameters like viscosity or domain range), requiring the model to infer them implicitly from observations.

## Key Experimental Results

### Main Results

Pre-training evaluation setup on PDEBench+ (16 PDEs including Navier-Stokes, diffusion equations, Burgers' equation, shallow water equations, etc.):

| Model | Parameters | VRMSE (20 steps) | Architecture Type |
|------|--------|------|----------|
| **PDE-Transformer-SC** | ~110M | **Best** | Separated Channel + U-shape + Shifted Window |
| PDE-Transformer-MC | ~110M | Second Best | Mixed Channel + U-shape + Shifted Window |
| DiT (Global Attention) | ~110M | Poor | Flat + Global Attention |
| scOT | ~110M | Medium | Swin + U-shape |
| MPP | ~110M | Medium | Axial ViT |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| Patch size p=2 vs. p=4 | p=2 significantly outpaces p=4 | 4x token count but clear accuracy gains |
| Window size w=4 vs. w=8 | w=8 slightly better | Larger window covers broader spatial context |
| SC vs. MC representation | SC comprehensively outperforms MC | Information density consistency is key |
| With/Without U-shaped structure | U-shape significantly better than flat | Multi-scale inductive bias matches physical properties |
| Shifted window vs. Global attention | Comparable accuracy, significantly reduced computation | Critical design for scalability |
| With/Without relative positional encoding | Relative positional coding outperforms absolute | Translation invariance is crucial for PDEs |
| adaLN-Zero vs. Cross-attention | adaLN-Zero is more efficient | Accelerates training with comparable performance |

### Key Findings

1. **SC representation is crucial for joint multi-PDE training**: Maintaining consistent token information density across channels avoids representation compression in the mixed-channel approach, outperforming MC in both pre-training and fine-tuning.
2. **The U-shaped multi-scale structure is critical for physical simulation**: Compared with the flat structure of DiT, the multi-scale structure provides a strong inductive bias that significantly boots performance.
3. **Pre-training effectively boosts downstream task performance**: The pre-trained and fine-tuned model comprehensively outperforms training-from-scratch across multiple challenging downstream tasks (varying PDE parameters, higher resolutions, longer time steps).
4. **An optimal accuracy-efficiency trade-off exists for patch size**: $p=2$ yields best accuracy but is computationally heavy; $p=4$ is a practical alternative.
5. **Error accumulation is controllable in 20-step autoregressive prediction**: Thanks to the architectural designs, the quality of long-horizon prediction remains stable.

## Highlights & Insights

1. **Elegant core concept of the Separated Channel (SC) design**: The idea of 'representing different physical quantities using different tokens + axial attention interactions' elegantly tackles the channel heterogeneity in multi-PDE joint modeling while maintaining consistent information density.
2. **Clear pathway of modification from DiT to PDE-Transformer**: Every single adjustment (U-shaped structure, shifted windows, SC representation, conditioning mechanism) has a clear physical motivation and ablation validation.
3. **Direct operation on raw data rather than latent space**: Bypasses the extra complexity and information loss of pre-training VAEs, resolving computational bottlenecks at high resolutions through architectural design (shifted windows + multi-scale).
4. **Explicit handling of boundary conditions**: Periodicity and non-periodicity are supported elegantly via token rolling and attention score masking, a crucial but often omitted detail in physics simulation.
5. **Dual support for both supervised and diffusion training modes**: Achieved via the flexible conditioning mechanism of adaLN-Zero, expanding application scenarios.

## Limitations & Future Work

1. **Limited to 2D regular grids**: The current architecture is restricted to 2D spatial data and has not been expanded to 3D or unstructured meshes.
2. **Pre-training data contains only 16 PDEs**: As a 'foundation model', the coverage of PDE types remains limited, leaving a gap toward a truly general physical foundation model.
3. **Lack of direct speed comparison with traditional solvers**: The paper primarily compares with other ML approaches, leaving deep discussion on the speedup factor against traditional numerical methods unexplored.
4. **Linear complexity scaling of the SC version with the channel count**: Computational bottlenecks might occur for highly complex systems with massive numbers of channels.
5. **Future directions**: Extension to 3D, unstructured mesh support, larger-scale pre-training datasets, and hybrid methods integrating traditional solvers.

## Related Work & Insights

- **DiT (Peebles & Xie, 2023)**: The foundational backbone of PDE-Transformer; this study introduces five key modifications to it.
- **Swin Transformer (Liu et al., 2021)**: The source of the shifted window attention mechanism.
- **scOT (Herde et al., 2024)**: Hierarchical ViT + shifted windows for physics, which PDE-Transformer significantly outperforms.
- **MPP (McCabe et al., 2023)**: Pioneering work in multi-physical pre-training using axial ViT.
- **FNO (Li et al., 2021)**: Representative neural operator method operating in the frequency domain.
- **Insights**: The approach of channel-independent embedding + axial interaction in SC can be generalized to the architecture design of other multi-modal/multi-channel Transformers.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 8 | Significant novelty in SC representation and PDE-targeted DiT modifications |
| Value | 8 | Open-source code, strong generalization, dual modes of supervised + diffusion |
| Experimental Thoroughness | 9 | Comprehensive ablations, 16 PDE types, multiple downstream tasks |
| Writing Quality | 8 | Clear architectural design motivations, intuitive illustrations |
| **Overall** | **8** | Solid work on Transformer architecture design for physical simulations |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Transformers without Normalization](../../CVPR2025/self_supervised/transformers_without_normalization.md)
- [\[ICML 2025\] Test-Time Training Provably Improves Transformers as In-Context Learners](test-time_training_provably_improves_transformers_as_in-context_learners.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)
- [\[CVPR 2026\] VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair](../../CVPR2026/self_supervised/vt-intrinsic_physics-based_decomposition_of_reflectance_and_shading_using_a_sing.md)

</div>

<!-- RELATED:END -->
