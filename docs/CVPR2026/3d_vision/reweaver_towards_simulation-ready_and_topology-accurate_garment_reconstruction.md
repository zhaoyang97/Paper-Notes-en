---
title: >-
  [Paper Note] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction
description: >-
  [CVPR 2026][3D Vision][Garment Reconstruction] This paper proposes ReWeaver, a framework that jointly reconstructs 3D garment geometry and 2D sewing patterns from as few as four multi-view RGB images. A dual-path Transfo…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Garment Reconstruction"
  - "Sewing Pattern"
  - "Topology Reconstruction"
  - "Multi-View Reconstruction"
  - "Physical Simulation"
date: 2026-05-08
content_hash: b0df561d5e4ab35a
---

# ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2601.16672](https://arxiv.org/abs/2601.16672)  
**Authors**: Ming Li, Hui Shan, Kai Zheng, Chentao Shen, Siyu Liu, Yanwei Fu, Zhen Chen, Xiangru Huang
**Institutions**: Zhejiang University, Shanghai Innovation Institute, Westlake University, Fudan University, Adobe, Xidian University
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: Garment Reconstruction, Sewing Pattern, Topology Reconstruction, Multi-View Reconstruction, Physical Simulation

## TL;DR

This paper proposes ReWeaver, a framework that jointly reconstructs 3D garment geometry and 2D sewing patterns from as few as four multi-view RGB images. A dual-path Transformer predicts 3D patches/curves and their topological connectivity, after which an intra-group attention module unfolds the 3D structure into 2D panel edges. ReWeaver is the first method to produce topology-accurate garment assets that are directly usable in physical simulation.

## Background & Motivation

High-quality 3D garment reconstruction is critical for applications such as virtual try-on, digital humans, gaming, and robotic manipulation. Existing approaches suffer from two major limitations:

1. **Limitations of unstructured representations**: Methods based on point clouds, SDFs, 3D Gaussian splatting, and similar representations can approximate garment geometry but lack explicit seam/panel structure, making them incompatible with physical simulation, garment editing, or retargeting. Such representations are inherently misaligned with industry-standard garment design pipelines, which are centered on 2D sewing patterns.

2. **Limitations of existing sewing pattern methods**:
    - Methods relying on predefined topologies (e.g., DiffAvatar) are restricted to simple garments and cannot handle unseen layouts.
    - Vision-language-model-based methods (e.g., ChatGarment, AIpparel) generate 2D patterns via tokenized JSON descriptions, offering stronger topology generalization but insufficient geometric accuracy.
    - Most methods focus exclusively on 2D patterns, neglecting accurate 3D geometric understanding.

**Core Goal**: Simultaneously reconstruct accurate garment **topology** (which panels/seams are connected) and **geometry** (precise 3D shape of each element), yielding outputs suitable for both 3D perception and high-fidelity physical simulation.

## Method

### Overall Architecture

ReWeaver adopts an encoder–decoder architecture consisting of four stages:

1. **Multi-view visual encoding** (Section 3.2): A VGGT-based multi-view encoder extracts unified features.
2. **3D curve and patch prediction** (Section 3.3): A dual-path Transformer predicts 3D geometry and topology.
3. **2D pattern prediction** (Section 3.4): An intra-group attention module unfolds the 3D structure into 2D panel edges.
4. **Topology/geometry refinement**: Post-processing enforces panel closure and topological consistency.

**Terminology** (dual 2D/3D space):

| Space | Surface Region | Boundary Line |
|:---:|:---:|:---:|
| 3D | Patch | Curve / Seam |
| 2D | Panel | Edge |

### Multi-View Visual Encoder

Following the design of VGGT, the processing pipeline proceeds as follows:

- Each input image is divided into non-overlapping $16\times16$ patches and embedded as tokens via a DINOv2 backbone.
- Intra-frame self-attention (refining single-view features) and inter-frame self-attention (aggregating cross-view information) are stacked in alternation.
- The final intra-frame and inter-frame outputs are concatenated, and tokens from all frames are flattened to form a sequence $T_i \in \mathbb{R}^{N_i \times D}$, where $D=768$.

This alternating attention design progressively integrates **local texture cues** and **global geometric information**, adapting to sparse, arbitrarily distributed multi-view inputs.

### 3D Curve and Patch Prediction (Dual-Path Transformer)

This is the core module of ReWeaver, responsible for predicting 3D geometric elements and their topological connectivity.

**Input**: Visual tokens $T_i$, learnable patch queries $Q_p \in \mathbb{R}^{N_p \times D}$ ($N_p=200$), and curve queries $Q_c \in \mathbb{R}^{N_c \times D}$ ($N_c=70$). Query counts are set to approximately twice the maximum count observed in training data.

**Dual-Path Transformer Architecture** (inspired by ComplexGen):

At each layer, the patch path and curve path each perform:

1. **Intra-group self-attention**: elements of the same type exchange information (patch–patch or curve–curve).
2. **Cross-group cross-attention**: context is retrieved from image tokens and elements of the other type.
3. **LayerNorm + FFN + residual connection**.

After multiple layers, refined tokens are obtained: $T_p \in \mathbb{R}^{N_p \times D}$ and $T_c \in \mathbb{R}^{N_c \times D}$.

**Three decoding heads**:

#### (1) Probability Prediction Head

A 3-layer MLP followed by Sigmoid predicts whether each query corresponds to a valid element:

$$\sigma_p^i = \text{sigmoid}(f_p^{\text{prob}}(T_p^i)), \quad \sigma_c^i = \text{sigmoid}(f_c^{\text{prob}}(T_c^i))$$

Low-probability elements are filtered by thresholds $\epsilon_p$ and $\epsilon_c$, and **topology refinement** produces binary validity masks $\boldsymbol{\sigma}_p^{\star}$ and $\boldsymbol{\sigma}_c^{\star}$.

#### (2) Geometry Prediction Head (HyperNetwork)

A key innovation: rather than directly regressing point coordinates, a HyperNetwork generates the weights of a parameterized mapping MLP.

For curves, the hypernetwork $f_c^{\text{geo}}$ generates, conditioned on token $T_c^i$, a 3-layer MLP that maps $[0,1]$ to $\mathbb{R}^3$:

$$g_c^i(u) = f_c^{\text{geo}}(T_c^i)(u) \in \mathbb{R}^3, \quad \forall u \in [0,1]$$

For patches, the hypernetwork $f_p^{\text{geo}}$ generates a MLP mapping $[0,1]^2$ to $\mathbb{R}^3$:

$$g_p^i(u,v) = f_p^{\text{geo}}(T_p^i)(u,v) \in \mathbb{R}^3, \quad \forall u,v \in [0,1]$$

**Advantages of the HyperNetwork**:

- During training, uniform sampling at arbitrary density is possible without compromising geometric smoothness or continuity.
- At inference, **adaptive sampling density** is supported — small patches are sampled sparsely, large patches densely — yielding a near-uniform 3D point distribution.
- Each token parameterizes an independent continuous mapping, encoding rich shape information.

#### (3) Connectivity Prediction Head

The connection probability between patch $i$ and curve $j$ is predicted as:

$$\sigma_{pc}(i,j) = \text{sigmoid}(f_p^{\text{adj}}(T_p^i) \cdot f_c^{\text{adj}}(T_c^j))$$

A dot product of linear projections followed by Sigmoid yields the adjacency matrix $\sigma_{pc}$. After thresholding at $\epsilon_{\text{adj}}$ and topology refinement, the final binary connectivity matrix $\sigma_{pc}^{\star} \in \{0,1\}^{N_p \times N_c}$ is obtained.

### 2D Pattern Prediction (Intra-Group Attention Unfolding)

Given valid patch/curve tokens and the refined topology, this module "unfolds" the 3D structure into 2D sewing patterns.

**Core Idea**: Based on $\sigma_{pc}^{\star}$, each valid patch token and its connected curve tokens are grouped together, and attention is performed **within each group**.

Specifically:

1. Curve tokens within a group first exchange information via self-attention.
2. They then attend to the associated patch token via cross-attention.
3. After LayerNorm + FFN + residual connection, edge tokens $T_e$ are obtained.

For each connected curve $j \in \partial_i$, another hypernetwork generates a MLP mapping 1D parameters to normalized 2D coordinates:

$$g_e^{ij}(u) = f_e^{\text{edge}}(T_e^j)(u) \in [0,1]^2, \quad \forall u \in [0,1]$$

**Scale Recovery**: Since 2D panels are predicted in a normalized $[0,1]^2$ space, an additional MLP $f_p^{\text{scale}}$ predicts a scaling factor $s_i$ from the patch token, which is multiplied by the normalized coordinates to recover physical dimensions.

**Geometry Refinement**: Since the HyperNetwork does not guarantee perfect alignment of adjacent edge endpoints, a post-processing step enforces edge closure so that panels form closed loops suitable for triangulation and simulation.

### Loss & Training

Hungarian matching establishes correspondences between predicted elements and ground truth. The total loss comprises three terms:

**Geometry Loss** (Chamfer Distance):

$$L_{\text{geo}} = \sum_{g \in \mathcal{G}} w_{\text{geo}}^{(g)} \cdot \text{CD}(V(g), V(m(g)))$$

Chamfer distance is computed between point sets sampled from all parameterized mappings (patch, curve, edge) and the corresponding ground truth.

**Classification and Connectivity Loss** (BCE):

$$L_{\text{cls}} = \sum_{\sigma \in \{\boldsymbol{\sigma}_p, \boldsymbol{\sigma}_c, \sigma_{pc}\}} w_{\text{cls}}^{(\sigma)} \cdot \text{BCE}(\sigma, m(\sigma))$$

**Scale Loss** ($\ell_2$):

$$L_{\text{scale}} = \sum_{i=1}^{N_p} w_{\text{scale}} \|s_i - s_{m(i)}^{\text{gt}}\|_2^2$$

## Key Experimental Results

### Dataset: GCD-TS

Extended from GarmentCodeData (GCD), with the following key improvements:

- Default textures in GCD, which contain strong seam cues, are replaced with approximately 50 BEDLAM body textures and a large collection of tileable garment textures.
- Each garment–body pair is rendered from four viewpoints (front/back/left/right) with small-scale camera pose perturbations.
- A total of approximately **100,000** textured multi-view samples covering a wide range of complex geometry and topology.

### Main Results

| Metric | AIpparel-MV | ReWeaver | Note |
|:---|:---:|:---:|:---|
| $\text{Acc}_p$ ↑ Panel count accuracy | 0.4561 | **0.8923** | ReWeaver **+43.6%** |
| $\text{Acc}_e$ ↑ Edge count accuracy | **0.6774** | 0.6570 | Comparable |
| $\text{Acc}_o$ ↑ Overall topology accuracy | 0.3090 | **0.5863** | ReWeaver **+27.7%** |
| $\text{CD}_e$ ↓ 2D edge Chamfer distance | 0.0648 | **0.0395** | More accurate geometry |
| IoU ↑ Panel intersection over union | 0.7084 | **0.8080** | **+10.0%** |

ReWeaver significantly outperforms multi-view-augmented AIpparel (AIpparel-MV) on 5 of 6 metrics. The panel count accuracy jumps from 45.6% to 89.2%, demonstrating substantially improved reliability in identifying garment topology.

### Ablation Study

**Effect of Topology and Geometry Refinement**

| Configuration | $\text{CD}_p^{\text{base}}$ ↓ | $\text{CD}_p^{\text{adapt}}$ ↓ | $\text{CD}_c$ ↓ | $\text{Acc}_p$ ↑ | $\text{Acc}_e$ ↑ | $\text{Acc}_o$ ↑ | $\text{CD}_e$ ↓ | IoU ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| With refinement | 0.0225 | **0.0187** | 0.0264 | 0.8923 | **0.6570** | **0.5863** | **0.0395** | **0.8080** |
| Without refinement | 0.0225 | 0.0188 | **0.0255** | **0.9101** | 0.5361 | 0.4880 | 0.0416 | 0.7775 |

**Key Findings**:

- **Topology refinement** removes redundant/duplicate edges, improving edge count accuracy ($\text{Acc}_e$) from 53.6% to 65.7% (+12.1%) and overall accuracy from 48.8% to 58.6% (+9.8%).
- **Geometry refinement** closes small gaps between edges in 2D space, producing fully closed panel boundaries and improving IoU from 77.8% to 80.8%.
- Refinement has minimal impact on 3D geometric metrics ($\text{CD}_p$ and $\text{CD}_c$ are nearly unchanged), indicating it primarily improves topological consistency and 2D panel quality.
- $\text{Acc}_p$ is marginally higher without refinement (0.91 vs. 0.89), as refinement occasionally removes valid elements deemed redundant; however, overall topological quality still improves substantially.

### Adaptive Sampling

- Training uses a fixed $20\times20$ patch sampling density.
- At inference, a $20\times20$ grid is first pre-sampled; points are then adaptively retained based on **spatial variance**. Dense points are pruned for small patches while large patches retain dense sampling.
- Adaptive sampling reduces $\text{CD}_p^{\text{adapt}}$ from 0.0225 to 0.0187, confirming the effectiveness of the adaptive strategy.

## Highlights & Insights

- ⭐ **First joint reconstruction**: Simultaneously outputs 3D garment geometry and 2D sewing patterns while maintaining explicit 2D–3D correspondences, making outputs directly usable in physical simulation.
- ⭐ **HyperNetwork parameterization**: HyperNetworks generate continuous parameterized mappings that support arbitrary and adaptive sampling at inference, achieving both flexibility and geometric smoothness.
- ⭐ **Dual-path Transformer**: The patch/curve dual-path design with self-attention and cross-attention effectively fuses multi-view image evidence with structural geometric constraints.
- ⭐ **GCD-TS dataset**: 100K-scale dataset that resolves texture leakage of seam information present in the original GCD, improving generalization.
- ⭐ Panel count accuracy of 89.2% far exceeds the baseline of 45.6%, demonstrating strong topology generalization.

## Limitations & Future Work

- High-quality 3D garment data with complex topology and photorealistic textures remains scarce; experiments exhibit a notable **sim-to-real gap**.
- Edge count accuracy ($\text{Acc}_e = 0.657$) lags behind panel count accuracy, indicating substantial room for improvement in **fine-grained topology** prediction.
- Quantitative evaluation on real-world images is absent; validation is conducted exclusively on synthetic data.
- Geometry refinement relies on post-processing heuristics and does not guarantee 100% successful closure.
- The input is fixed at four standard viewpoints (front/back/left/right); performance under more freely distributed capture conditions has not been thoroughly validated.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to jointly model 3D geometry, 2D sewing patterns, and topological connectivity; the HyperNetwork parameterization is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive validation on synthetic data, but real-image evaluation is absent and only one baseline is compared.
- Writing Quality: ⭐⭐⭐⭐ — Terminology is clearly defined; the dual 2D/3D space description is well-articulated; the framework diagram is intuitive.
- Value: ⭐⭐⭐⭐ — Directly outputs simulation-ready assets, with practical value for digital humans, virtual try-on, and robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physicsgrounded_articulation_for_sim.md)
- [\[CVPR 2026\] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction](fluidgaussian_propagating_simulation-based_uncertainty_toward_functionally-intel.md)
- [\[CVPR 2026\] NI-Tex: Non-isometric Image-based Garment Texture Generation](ni-tex_non-isometric_image-based_garment_texture_generation.md)
- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)

</div>

<!-- RELATED:END -->
