---
title: >-
  [Paper Note] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion
description: >-
  [AAAI 2026][3D Vision][Point Cloud Completion] This paper proposes Simba, a framework that, for the first time, reformulates point cloud completion as diffusion over a geometric transformation field rather than over poin…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Diffusion Models"
  - "Symmetry Prior"
  - "Mamba"
  - "Affine Transformation"
date: 2026-05-08
content_hash: c1783fcc19f91e6e
---

# Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion

**Conference**: AAAI 2026
**arXiv**: [2511.16161](https://arxiv.org/abs/2511.16161)
**Code**: [https://github.com/I2-Multimedia-Lab/Simba](https://github.com/I2-Multimedia-Lab/Simba)
**Area**: 3D Vision
**Keywords**: Point Cloud Completion, Diffusion Models, Symmetry Prior, Mamba, Affine Transformation

## TL;DR

This paper proposes Simba, a framework that, for the first time, reformulates point cloud completion as diffusion over a geometric transformation field rather than over point coordinates. A Sym-Diffuser learns the conditional distribution of per-point affine transformations to generate coarse completions, which are then progressively refined to high-fidelity outputs via a cascaded Mamba architecture (MBA-Refiner). Simba achieves state-of-the-art performance on PCN, ShapeNet, and KITTI benchmarks.

## Background & Motivation

### State of the Field

Point cloud completion is a fundamental task in 3D vision, aiming to recover complete 3D shapes from partial observations. Existing methods span several generations:

**Coarse-to-fine paradigm** (PCN, FoldingNet): global shape prior → refinement

**Transformer-based methods** (PoinTr, SeedFormer, CRA-PCN): capture long-range dependencies; currently dominant

**Symmetry-prior methods** (SymmCompletion): exploit symmetry to learn per-point local affine transformations

**Diffusion-based methods** (PDR, PCDreamer): perform diffusion in point coordinate space

### Limitations of Prior Work

The authors focus on **symmetry-prior-based methods** (e.g., SymmCompletion) and identify two key deficiencies:

**Overfitting**: Regression-based approaches tend to memorize instance-specific transformation patterns from the training set rather than learning generalizable geometric alignment rules, leading to poor cross-domain generalization (e.g., on real-world KITTI data).

**Noise sensitivity**: Independent per-point regression of transformations is highly sensitive to occlusion and noise, causing fragmentation or distortion of global structure.

Methods that perform diffusion directly in point coordinate space also suffer drawbacks: fine-grained details from the partial input may be washed away, and inference is computationally expensive.

### Root Cause & Starting Point

**How can one exploit the strong geometric information encoded in symmetry priors while preventing the network from simply memorizing specific transformation patterns?**

The authors' key observation is that diffusion models possess strong generative capacity and support diverse sampling. Combining diffusion with transformation matrices allows the model to leverage geometric priors while avoiding overfitting to deterministic solutions.

**Core Idea**: Rather than diffusing point coordinates, the method **diffuses a geometric transformation field**. It learns the conditional distribution $p(\mathcal{T}|\mathcal{F}_k)$ of per-point affine transformations, generates the transformation field via iterative denoising, and applies it to keypoints to construct complete shapes—naturally preserving fine-grained details from the input.

## Method

### Overall Architecture

Two-stage design:

- **Stage 1**: Pre-train a SymmGT network to generate target transformation matrices (used as supervision targets for Stage 2 diffusion).
- **Stage 2**:
  - **Sym-Diffuser** (Symmetry Diffusion Module): performs conditional diffusion in transformation field space to generate coarse completions.
  - **MBA-Refiner** (Cascaded Mamba Refiner): three-level cascaded refinement with upsampling.

### Key Designs

#### 1. SymmGT Pre-training (Stage 1)

**Function**: Generate "clean" target transformation fields $\mathcal{T}_{gt}$ required for diffusion model training.

**Core Pipeline**:
- Input: partial point cloud $\mathcal{P}_{in}$ and complete GT $\mathcal{P}_{gt}$
- Sample keypoints $\mathcal{P}_k$ from $\mathcal{P}_{in}$
- A weight-shared feature extractor (SA layers + Point Transformer) extracts keypoint features $\mathcal{F}_k$ and GT global features $\mathcal{F}_{gt}$ respectively
- Cross-attention fusion followed by regression of the transformation field $\mathcal{T}_{gt} \in \mathbb{R}^{K \times 12}$, consisting of per-point affine matrices $\mathbf{A}_i \in \mathbb{R}^{3 \times 3}$ and translation vectors $\mathbf{T}_i \in \mathbb{R}^3$
- Transformation applied to keypoints: $\mathcal{P}_{init} = \mathcal{P}_k \cup \{\mathbf{A}_i \mathbf{p}_i + \mathbf{T}_i\}$
- Trained with Chamfer Distance

**SymmGT is frozen in Stage 2**, serving solely to produce $\mathcal{T}_{gt}$ as the diffusion target $\mathcal{Z}_0$.

#### 2. Sym-Diffuser (Symmetry Diffusion Module)

**Function**: Learn the conditional distribution of the transformation field to generate structurally complete coarse outputs.

**Mechanism**:
- **Forward process**: Standard DDPM with $T=100$ steps, progressively adding noise to $\mathcal{Z}_0$ (the target transformation field).
- **Reverse process**: A noise predictor $\epsilon_\theta$ estimates the noise and recovers the predicted clean transformation field $\hat{\mathcal{T}}_\theta$.
- **Training objective**: Inspired by Consistency Models, a weighted MSE loss is used:

$$\mathcal{L}_{\text{proxy}} = \mathbb{E}_{t, \mathcal{Z}_0, \epsilon}\left[\lambda(t) \|\mathcal{T}_{gt} - \hat{\mathcal{T}}_\theta(\mathcal{Z}_t, t, \mathcal{F}_k)\|^2\right]$$

- **Inference**: Starting from a random Gaussian vector $\mathbf{Z} \in \mathbb{R}^{N_k \times 12}$, conditioned on $\mathcal{F}_k$, iterative denoising produces a transformation field → applied to keypoints → coarse completion $\mathcal{P}_{init} = \mathcal{P}_k \cup \mathcal{P}_s$.

**Advantages over direct regression**:
- Diffusion models learn distributions rather than deterministic mappings, inherently avoiding overfitting.
- The generative process introduces diversity, improving robustness to noise and occlusion.
- Diffusion in a low-dimensional space (12-dimensional transformation vectors) is more efficient than in high-dimensional point coordinate space.

#### 3. MBA-Refiner (Cascaded Mamba Refiner)

**Function**: Progressively refine and upsample coarse completions to high-fidelity outputs.

**Core Architecture**: Three cascaded levels with upsampling ratios $[2\times, 2\times, 4\times]$, totaling $16\times$. Each level consists of feature fusion followed by MambaForward refinement.

**Heterogeneous fusion strategy**—different fusion mechanisms are applied at different density levels:

- **Blocks 1–2** (low-density): **Cross-Attention Fusion**, prioritizing performance.
  - Base features $\mathcal{F}_l$ attend to keypoint features $\mathcal{F}_k$ and symmetric point features $\mathcal{F}_s$ respectively.
  - Concatenated outputs are fused via MLP.

$$\mathbf{f}_{in}^l = \boldsymbol{\psi}\left([\text{MCA}(\mathcal{F}_l, \mathcal{F}_g)]_{g \in \{k,s\}}\right)$$

- **Block 3** (high-density): **Mamba Fusion**, prioritizing efficiency.
  - The $\mathcal{O}(N^2)$ complexity of attention is prohibitive at high point densities.
  - Mamba's linear complexity $\mathcal{O}(N)$ significantly reduces memory and computational overhead.

**MambaForward Module**: A shared refinement-and-upsampling module used across all levels, comprising MLP → Mamba block (with residual connection) → upsampling layer.

**Design Motivation**: The heterogeneous design (attention for the first two levels, Mamba for the last) achieves the optimal trade-off between performance and efficiency. Pure attention incurs excessive memory (16.4 GB), while pure Mamba yields suboptimal performance (CD 6.43 vs. 6.34).

### Loss & Training

**Stage 1 Loss**:

$$\mathcal{L}_{\text{stage1}} = L_{CD}(\mathcal{P}_k \cup \{\mathbf{A}_i \mathbf{p}_i + \mathbf{T}_i\}, \mathcal{P}_{gt})$$

**Stage 2 Loss** (multi-level supervision):

$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{proxy}} + \sum_{l=1}^{3} L_{CD}(\mathcal{P}_{out}^l, \mathcal{P}_{gt})$$

Both Sym-Diffuser and each level of MBA-Refiner are supervised jointly.

Training setup: PyTorch, 4 × NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

**PCN Dataset** (8 categories, L1-CD ×10³ ↓ / F-Score@1% ↑):

| Method | Conference | Mean CD ↓ | F-Score ↑ |
|--------|------------|-----------|-----------|
| PCN | 3DV 2018 | 9.64 | 0.695 |
| PoinTr | ICCV 2021 | 8.38 | - |
| SnowflakeNet | ICCV 2021 | 7.21 | 0.801 |
| SeedFormer | ECCV 2022 | 6.74 | 0.818 |
| AdaPoinTr | TPAMI 2023 | 6.53 | 0.845 |
| CRA-PCN | AAAI 2024 | 6.39 | - |
| SymmCompletion | AAAI 2025 | 6.47 | 0.840 |
| PointCFormer | AAAI 2025 | 6.41 | 0.855 |
| PCDreamer | CVPR 2025 | 6.52 | **0.856** |
| **Simba (Ours)** | **AAAI 2026** | **6.34** | 0.853 |

Simba achieves the best overall CD, outperforming SymmCompletion by 2% (6.34 vs. 6.47), with particularly strong results on Sofa, Table, and Watercraft.

**ShapeNet-55/34/21** (L2-CD ×10³ ↓):

| Method | 55-class Avg | 34 Seen Avg | 21 Unseen Avg |
|--------|-------------|-------------|---------------|
| AdaPoinTr | 0.81 | 0.73 | 1.23 |
| SVDFormer | 0.83 | 0.75 | 1.28 |
| CRA-PCN | 0.85 | 0.76 | 1.24 |
| **Simba** | **0.79** | **0.70** | **1.23** |

Simba achieves best performance across all 55 categories and the 34 seen categories, while matching AdaPoinTr on the 21 unseen categories, demonstrating strong generalization.

**KITTI Real-World Data** (MMD ×10³ ↓):

| Method | MMD ↓ |
|--------|-------|
| CRA-PCN | 1.737 |
| SeedFormer | 0.516 |
| EINet | 0.967 |
| SymmCompletion | 0.970 |
| **Simba** | **0.423** |

Simba achieves a substantial margin on real-world LiDAR data, validating the cross-domain generalization advantage of the transformation diffusion paradigm—trained solely on synthetic data and applied directly to real data.

### Ablation Study

**Prediction Module Ablation** (PCN, CD-L1 ×10³):

| Configuration | CD ↓ | Note |
|---------------|------|------|
| Diffusion Model (Ours) | **6.34** | Diffusion-based transformation field generation |
| Transformer Regression | 6.48 | Direct regression of transformation field |

Diffusion outperforms regression by 2.2%; qualitative results show that regression produces visible structural artifacts.

**Progressive Upsampling Strategy Ablation** (total ratio 16×):

| Configuration | CD ↓ | Note |
|---------------|------|------|
| 3-level [2×, 2×, 4×] (Ours) | **6.34** | Progressive, best |
| 1-level [16×] | 6.70 | Single step, worst |
| 2-level [2×, 8×] | 6.56 | Uneven |
| 2-level [4×, 4×] | 6.52 | Uniform two-level |

Progressive multi-level refinement significantly outperforms aggressive single-step or two-step upsampling.

**MBA-Refiner Architecture Ablation**:

| Configuration | Fusion Strategy | Memory | CD ↓ |
|---------------|----------------|--------|------|
| [CA, CA, MFusion] (Ours) | Heterogeneous | 14.7 GB | **6.34** |
| [MLP, MLP, MFusion] | Simple fusion | 12.1 GB | 6.49 |
| [CA, CA, MLP] | No Mamba | 12.0 GB | 6.41 |
| [CA, CA, CA] | Full attention | 16.4 GB | 6.35 |
| [MFusion×3] | Full Mamba | 13.8 GB | 6.43 |

The heterogeneous design (CA+CA+Mamba) achieves the best balance between performance (6.34) and memory (14.7 GB). Full attention achieves comparable performance (6.35) but consumes 11.6% more memory.

### Key Findings

1. Learning the distribution of transformations via diffusion is more robust than deterministic regression—fundamentally because it avoids overfitting.
2. Progressive upsampling is critical—single-step 16× upsampling increases CD by 5.7%.
3. Strong performance on KITTI validates the superiority of the transformation diffusion paradigm for synthetic-to-real transfer (MMD reduced by 18% relative to SeedFormer).
4. Mamba is an effective substitute for attention in high-density point clouds, achieving significant memory savings at a negligible performance cost.

## Highlights & Insights

1. **Paradigm innovation**: Diffusing a transformation field rather than point coordinates—diffusion in a low-dimensional space (12-dimensional) is more efficient and naturally preserves input details, since transformations are applied to the original keypoints.
2. **Two-stage decoupling**: Stage 1 generates supervision targets; Stage 2 performs diffusion. This avoids the instability of end-to-end diffusion model training.
3. **Heterogeneous cascaded design**: Adaptively selects fusion strategies based on point density (attention at low density, Mamba at high density), representing a principled integration of engineering and theoretical considerations.
4. **Strong cross-domain generalization**: Achieves state-of-the-art on real-world KITTI data despite being trained exclusively on synthetic data, which is of significant practical value.

## Limitations & Future Work

1. **Inference speed**: Diffusion requires multi-step iterative denoising ($T=100$), which may be slower than purely feed-forward methods. Inference time is not reported in the paper.
2. **Two-stage training**: Stage 1 requires separate pre-training of SymmGT, increasing overall training complexity.
3. **Symmetry assumption**: The framework relies on symmetry priors and may be less effective for highly asymmetric objects (e.g., irregular natural shapes).
4. **PCN dataset scope**: Only 8 object categories are covered; although ShapeNet-55 provides broader validation, evaluation on larger-scale or more diverse scenarios is lacking.
5. F-Score is slightly lower than PCDreamer (0.853 vs. 0.856), indicating room for improvement in surface reconstruction fidelity.

## Related Work & Insights

- **SymmCompletion** (AAAI 2025): The direct predecessor of this work, proposing per-point affine transformation regression. Simba upgrades this from regression to diffusion.
- **Consistency Models** (2023): Inspired the design of the diffusion training objective.
- **Mamba** (2023): Linear-complexity sequence modeling, emerging in the point cloud domain (PointMamba, 3DMambaComplete).
- **PCDreamer** (CVPR 2025): Uses 2D priors with diffusion for point cloud completion, but diffuses directly in coordinate space.
- The transformation field diffusion paradigm is generalizable to other deformation/transformation learning tasks (e.g., registration, deformation prediction).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The "diffuse the transformation field" paradigm is novel and theoretically grounded, representing an innovative application of diffusion models to 3D tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks (PCN/ShapeNet/KITTI) with detailed ablations covering prediction modules, upsampling strategies, and architecture design.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with rich figures and tables; some derivations (e.g., the diffusion training objective) could be elaborated further.
- Value: ⭐⭐⭐⭐⭐ — New paradigm, strong generalization, and open-source code make a significant contribution to the point cloud completion community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[AAAI 2026\] 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)

</div>

<!-- RELATED:END -->
