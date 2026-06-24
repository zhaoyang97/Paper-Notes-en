---
title: >-
  [Paper Note] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion
description: >-
  [AAAI 2026][3D Vision][Point Cloud Completion] This paper proposes the Simba framework, which reformulates point cloud completion as "diffusion on geometric transformation fields" rather than "diffusion on point coordinates." By utilizing Sym-Diffuser to learn the conditional distribution of point-wise affine transformations, it generates a coarse completion. Subsequently, a cascaded Mamba architecture (MBA-Refiner) is employed to progressively refine it to high-fidelity outp…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Diffusion Models"
  - "Symmetry Prior"
  - "Mamba"
  - "Affine Transformation"
date: 2026-05-08
content_hash: 3a6f92a18ded7409
---

# Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion

**Conference**: AAAI 2026  
**arXiv**: [2511.16161](https://arxiv.org/abs/2511.16161)  
**Code**: [https://github.com/I2-Multimedia-Lab/Simba](https://github.com/I2-Multimedia-Lab/Simba)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Completion, Diffusion Models, Symmetry Prior, Mamba, Affine Transformation

## TL;DR

This paper proposes the Simba framework, which reformulates point cloud completion as "diffusion on geometric transformation fields" rather than "diffusion on point coordinates." By utilizing Sym-Diffuser to learn the conditional distribution of point-wise affine transformations, it generates a coarse completion. Subsequently, a cascaded Mamba architecture (MBA-Refiner) is employed to progressively refine it to high-fidelity outputs, achieving SOTA performance across multiple benchmarks including PCN, ShapeNet, and KITTI.

## Background & Motivation

### Background

Point cloud completion is a fundamental task in 3D vision, aiming to recover complete 3D shapes from incomplete, partial observations. Existing methods have undergone several generations of development:

**Coarse-to-fine paradigm** (PCN, FoldingNet): Global shape prior $\rightarrow$ refinement

**Transformer-based methods** (PoinTr, SeedFormer, CRA-PCN): Capable of capturing long-range dependencies, currently the mainstream choice.

**Symmetry prior methods** (SymmCompletion): Utilize symmetry to learn point-wise local affine transformations.

**Diffusion-based methods** (PDR, PCDreamer): Perform diffusion in the point coordinate space.

### Limitations of Prior Work

The authors focus on **methods leveraging symmetry priors** (such as SymmCompletion) and highlight two key defects:

**Overfitting**: Regression-based approaches tend to memorize instance-specific transformation patterns from the training set, rather than learning generalizable geometric alignment rules. This leads to poor generalization in cross-domain scenarios (such as real-world KITTI data).

**Noise Sensitivity**: Regressing transformations independently for each point is highly sensitive to occlusions and noise, resulting in fragmented or distorted global structures.

At the same time, methods that perform diffusion directly in the point coordinate space also face issues: they tend to erase fine details from the input, incur high computational costs, and result in slow inference.

### Key Challenge & Key Insight

**How can we exploit the strong geometric information in symmetry priors while preventing the network from merely memorizing specific transformation patterns?**

The authors' key observation: Diffusion models possess strong generative capabilities and can perform diverse sampling. Combining diffusion with transformation matrices leverages geometric priors while avoiding overfitting to rigid, fixed solutions.

**Core Idea**: Instead of diffusing point coordinates, the proposed method **diffuses the geometric transformation field**. It learns the conditional distribution of point-wise affine transformations $p(\mathcal{T}|\mathcal{F}_k)$. By iteratively denoising to generate the transformation field and applying it to keypoints, the complete shape is reconstructed, naturally preserving the fine details of the input.

## Method

### Overall Architecture

Two-stage design:

- **Stage 1**: Pre-trains the SymmGT network to generate target transformation matrices (serving as the supervision target for Stage 2 diffusion).
- **Stage 2**:
    - **Sym-Diffuser** (Symmetry Diffusion Module): Conducts conditional diffusion in the transformation field space to generate coarse completions.
    - **MBA-Refiner** (Cascaded Mamba Refiner): A three-stage cascaded refiner + upsampler.

### Key Designs

#### 1. SymmGT Pre-training (Stage 1)

**Function**: Generates the "clean" target transformation field $\mathcal{T}_{gt}$ required for training the diffusion model.

**Mechanism**:
- Input: Partial point cloud $\mathcal{P}_{in}$ and complete GT $\mathcal{P}_{gt}$.
- Sample keypoints $\mathcal{P}_k$ from $\mathcal{P}_{in}$.
- A weight-sharing feature extractor (SA layer + Point Transformer) extracts keypoint features $\mathcal{F}_k$ and GT global features $\mathcal{F}_{gt}$ respectively.
- After fusion using cross-attention, the network regresses the transformation field $\mathcal{T}_{gt} \in \mathbb{R}^{K \times 12}$, which consists of pointwise affine matrices $\mathbf{A}_i \in \mathbb{R}^{3 \times 3}$ and translation vectors $\mathbf{T}_i \in \mathbb{R}^3$.
- Apply transformations to keypoints: $\mathcal{P}_{init} = \mathcal{P}_k \cup \{\mathbf{A}_i \mathbf{p}_i + \mathbf{T}_i\}$.
- Trained using Chamfer Distance.

**SymmGT is frozen in Stage 2**, serving solely to produce $\mathcal{T}_{gt}$ as the ground-truth target $\mathcal{Z}_0$ for diffusion.

#### 2. Sym-Diffuser (Symmetry Diffusion Module)

**Function**: Learns the conditional distribution of the transformation field to generate structurally complete coarse completions.

**Mechanism**:
- **Forward Process**: Standard DDPM with $T=100$ steps, progressively adding noise to $\mathcal{Z}_0$ (target transformation field).
- **Reverse Process**: The noise predictor $\epsilon_\theta$ estimates the noise to reconstruct the predicted clean transformation field $\hat{\mathcal{T}}_\theta$.
- **Training Objective**: Inspired by Consistency Models, a weighted MSE loss is employed:

$$\mathcal{L}_{\text{proxy}} = \mathbb{E}_{t, \mathcal{Z}_0, \epsilon}\left[\lambda(t) \|\mathcal{T}_{gt} - \hat{\mathcal{T}}_\theta(\mathcal{Z}_t, t, \mathcal{F}_k)\|^2\right]$$

- **Inference**: Starting from a random Gaussian noise vector $\mathbf{Z} \in \mathbb{R}^{N_k \times 12}$, conditioned on $\mathcal{F}_k$, iterative denoising yields the transformation field $\rightarrow$ applied to keypoints $\rightarrow$ yields the coarse completion $\mathcal{P}_{init} = \mathcal{P}_k \cup \mathcal{P}_s$.

**Advantages over direct regression**:
- Diffusion models learn distributions instead of deterministic mappings, naturally preventing overfitting.
- The generation process offers diversity, enhancing robustness against noise and occlusions.
- Diffusing in a low-dimensional space (12D transformation vector) is significantly more efficient than diffusing in high-dimensional point coordinate spaces.

#### 3. MBA-Refiner (Cascaded Mamba Refiner)

**Function**: Progressively refines and upsamples coarse completions to high-fidelity outputs.

**Core Architecture**: A three-layer cascade with upsampling ratios of $[2\times, 2\times, 4\times]$, totaling a $16\times$ upsampling. Each layer consists of feature fusion and MambaForward refinement.

**Heterogeneous Fusion Strategy**—employing different fusion mechanisms at varying density layers:

- **Block 1-2** (low-density layers): **Cross-Attention Fusion**, prioritizing performance:
    - Base features $\mathcal{F}_l$ attend to keypoint features $\mathcal{F}_k$ and symmetric point features $\mathcal{F}_s$, respectively.
    - Concatenated and fused using an MLP:

$$\mathbf{f}_{in}^l = \boldsymbol{\psi}\left([\text{MCA}(\mathcal{F}_l, \mathcal{F}_g)]_{g \in \{k,s\}}\right)$$

- **Block 3** (high-density layer): **Mamba Fusion**, prioritizing efficiency:
    - The $\mathcal{O}(N^2)$ complexity of attention is unaffordable on high-density point clouds.
    - Mamba's linear complexity $\mathcal{O}(N)$ drastically reduces memory and computational overhead.

**MambaForward Module**: A refinement and upsampling module shared across all layers, containing an MLP $\rightarrow$ Mamba block (with residual connections) $\rightarrow$ upsampling layer.

**Design Motivation**: The heterogeneous design (using attention for the first two layers and Mamba for the final layer) strikes the optimal balance between performance and efficiency. Pure attention consumes excessive memory (16.4GB), while pure Mamba suffers from insufficient performance (CD 6.43 vs. 6.34).

### Loss & Training

**Stage 1 Loss**:

$$\mathcal{L}_{\text{stage1}} = L_{CD}(\mathcal{P}_k \cup \{\mathbf{A}_i \mathbf{p}_i + \mathbf{T}_i\}, \mathcal{P}_{gt})$$

**Stage 2 Loss** (multi-level supervision):

$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{proxy}} + \sum_{l=1}^{3} L_{CD}(\mathcal{P}_{out}^l, \mathcal{P}_{gt})$$

Both the Sym-Diffuser and the output of each layer in MBA-Refiner are supervised.

Training Setup: PyTorch, 4 × NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

**PCN Dataset** (8 categories, L1-CD ×10³ ↓ / F-Score@1% ↑):

| Method | Conference | Average CD ↓ | F-Score ↑ |
|------|------|----------|-----------|
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

Simba achieves the best overall CD, outperforming SymmCompletion by 2% (6.34 vs. 6.47), with particularly outstanding results on Sofa, Table, and Watercraft.

**ShapeNet-55/34/21** (L2-CD ×10³ ↓):

| Method | 55-class Avg | 34 Seen Avg | 21 Unseen Avg |
|------|---------|-----------|-------------|
| AdaPoinTr | 0.81 | 0.73 | 1.23 |
| SVDFormer | 0.83 | 0.75 | 1.28 |
| CRA-PCN | 0.85 | 0.76 | 1.24 |
| **Simba** | **0.79** | **0.70** | **1.23** |

Simba achieves the best performance across the entire 55-class and the 34 seen classes, while performing on par with AdaPoinTr on the 21 unseen classes, demonstrating robust generalization ability.

**KITTI Real-World Data** (MMD ×10³ ↓):

| Method | MMD ↓ |
|------|-------|
| CRA-PCN | 1.737 |
| SeedFormer | 0.516 |
| EINet | 0.967 |
| SymmCompletion | 0.970 |
| **Simba** | **0.423** |

Simba leads by a wide margin on real-world LiDAR data, validating the cross-domain generalization advantages of the transformation diffusion paradigm—trained solely on synthetic data and evaluated directly on real-world datasets.

### Ablation Study

**Ablation of the Prediction Module** (PCN, CD-L1 ×10³):

| Configuration | CD ↓ | Description |
|------|------|------|
| Diffusion Model (Ours) | **6.34** | Generates the transformation field via diffusion |
| Transformer Regression | 6.48 | Directly regresses the transformation field |

Diffusion outperforms regression by 2.2%, and visualizations reveal that the regression approach introduces severe structural artifacts.

**Ablation of Progressive Upsampling Strategy** (Total ratio of 16×):

| Configuration | CD ↓ | Description |
|------|------|------|
| 3 layers [2×, 2×, 4×] (Ours) | **6.34** | Progressive, optimal |
| 1 layer [16×] | 6.70 | Single-step, worst |
| 2 layers [2×, 8×] | 6.56 | Non-uniform |
| 2 layers [4×, 4×] | 6.52 | Two uniform layers |

Progressive multi-level refinement significantly outperforms aggressive single-step or two-step upsampling.

**MBA-Refiner Architecture Ablation**:

| Configuration | Fusion Strategy | Memory | CD ↓ |
|------|---------|------|------|
| [CA, CA, MFusion] (Ours) | Heterogeneous | 14.7GB | **6.34** |
| [MLP, MLP, MFusion] | Simple fusion | 12.1GB | 6.49 |
| [CA, CA, MLP] | Without Mamba | 12.0GB | 6.41 |
| [CA, CA, CA] | Full attention | 16.4GB | 6.35 |
| [MFusion×3] | Full Mamba | 13.8GB | 6.43 |

The heterogeneous design (CA+CA+Mamba) strikes the best balance between performance (6.34) and memory usage (14.7GB). Full attention achieves comparable performance (6.35) but consumes 11.6% more memory.

### Key Findings

1. Learning the distribution of transformations via diffusion is more robust than deterministic regression—fundamentally because it avoids overfitting.
2. Progressive upsampling is crucial—a single-step 16× upsampling increases CD by 5.7%.
3. The excellent performance on KITTI demonstrates the superiority of the transformation diffusion paradigm in synthetic-to-real transfer (MMD is reduced by 18% compared to SeedFormer).
4. Mamba serves as an effective alternative to attention on high-density point clouds, offering significant memory savings at a minor performance cost.

## Highlights & Insights

1. **Novel Paradigm**: "Diffusion on transformation fields" instead of "diffusion on point coordinates"—performing diffusion in a low-dimensional space (12D) is highly efficient and naturally preserves input details (since transformations are applied to original keypoints).
2. **Two-Stage Decoupling**: Stage 1 generates the supervision target while Stage 2 performs diffusion, successfully avoiding the instability associated with end-to-end training of diffusion models.
3. **Heterogeneous Cascaded Design**: Dynamically selecting fusion strategies based on point density (attention for lower density, Mamba for higher density) represents a solid blend of engineering and theory.
4. **Strong Cross-Domain Generalization**: Achieving SOTA on real-world KITTI data when trained solely on synthetic datasets is highly significant for actual deployment.

## Limitations & Future Work

1. **Inference Speed**: The diffusion model requires multi-step iterative denoising ($T=100$), which could be slower than pure feedforward methods. The paper does not report inference times.
2. **Two-Stage Training**: Stage 1 requires independent pre-training of SymmGT, increasing total training complexity.
3. **Symmetry Assumption**: The framework is built upon symmetry priors, meaning performance may be limited on highly asymmetric objects (e.g., irregular natural elements).
4. **PCN Dataset Limitations**: Evaluated on only 8 object classes. Although validated on ShapeNet-55 as well, there is still a lack of validation on larger, more diverse scenes.
5. The F-Score is slightly lower than PCDreamer's (0.853 vs. 0.856), indicating remaining room for improvement in surface reconstruction accuracy.

## Related Work & Insights

- **SymmCompletion** (AAAI 2025): Direct predecessor of this paper, which proposed point-wise affine transformation regression. Simba upgrades this from regression to diffusion.
- **Consistency Models** (2023): Inspired the design of the diffusion training objective.
- **Mamba** (2023): Linear-complexity sequence modeling, which has started to gain traction in the point cloud domain (e.g., PointMamba, 3DMambaComplete).
- **PCDreamer** (CVPR 2025): 2D prior + diffusion for point cloud completion, though it performs diffusion directly in the coordinate space.
- The proposed approach of diffusing transformation fields can be generalized to other deformation/transformation learning tasks (e.g., registration, deformation prediction).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The "diffusion on transformation fields" paradigm is novel and well-grounded, representing an innovative application of diffusion models to 3D tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on three benchmarks (PCN/ShapeNet/KITTI) with comprehensive ablation studies (prediction module/upsampling/architecture).
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with rich illustrations, though certain derivations (e.g., the diffusion training objective) could be more detailed.
- Value: ⭐⭐⭐⭐⭐ — Provides a new paradigm, strong generalization, and open-source code, significantly driving progress in point cloud completion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[CVPR 2026\] Hyper-PCN: Hypergraph-Based Point Cloud Completion via High-Order Correlation Modeling](../../CVPR2026/3d_vision/hyper-pcn_hypergraph-based_point_cloud_completion_via_high-order_correlation_mod.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](../../CVPR2025/3d_vision/pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)

</div>

<!-- RELATED:END -->
