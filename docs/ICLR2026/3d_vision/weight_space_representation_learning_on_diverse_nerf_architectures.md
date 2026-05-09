---
title: >-
  [Paper Note] Weight Space Representation Learning on Diverse NeRF Architectures
description: >-
  [ICLR 2026][3D Vision][NeRF] This paper proposes the first representation learning framework capable of processing weights from diverse NeRF architectures (MLP / tri-plane / hash table). By combining a Graph Meta-Network (GMN) encoder with a SigLIP contrastive loss, it constructs an architecture-agnostic latent space, enabling classification, retrieval, and language-grounded tasks across 13 NeRF architectures, with generalization to architectures unseen during training.
tags:
  - ICLR 2026
  - 3D Vision
  - NeRF
  - weight space
  - graph meta-network
  - contrastive learning
  - architecture-agnostic
date: 2026-05-08
content_hash: d300460017c2ac12
---

# Weight Space Representation Learning on Diverse NeRF Architectures

**Conference**: ICLR 2026
**arXiv**: [2502.09623](https://arxiv.org/abs/2502.09623)
**Code**: Available (link provided in paper)
**Area**: 3D Vision / NeRF
**Keywords**: NeRF, weight space, graph meta-network, contrastive learning, architecture-agnostic

## TL;DR
This paper proposes the first representation learning framework capable of processing weights from diverse NeRF architectures (MLP / tri-plane / hash table). By combining a Graph Meta-Network (GMN) encoder with a SigLIP contrastive loss, it constructs an architecture-agnostic latent space, enabling classification, retrieval, and language-grounded tasks across 13 NeRF architectures, with generalization to architectures unseen during training.

## Background & Motivation

**State of the Field**: NeRF encodes 3D information into network weights. Methods such as nf2vec and Cardace et al. perform downstream tasks (classification, retrieval) by processing NeRF weights, but are restricted to a single NeRF architecture family (MLP-only or tri-plane-only).

**Limitations of Prior Work**: The rapid diversification of NeRF architectures (MLP → tri-plane → hash table) requires a new processing framework to be designed for each new architecture, severely limiting practical applicability.

**Root Cause**: The weight structures of different NeRF architectures differ substantially (MLP weight matrices vs. planar features vs. hash tables), making it non-trivial to construct a unified representation space.

**Paper Goals**: Design an architecture-agnostic NeRF weight processing framework that maps representations of the same object across different NeRF architectures to nearby points in latent space.

**Starting Point**: Leverage Graph Meta-Networks (GMN) to convert arbitrary NeRF weights into parameter graphs, which are then processed by a GNN.

**Core Idea**: Apply a SigLIP contrastive loss to align embeddings of NeRFs representing the same object but trained with different architectures, enabling the GMN encoder to produce an architecture-agnostic latent space.

## Method

### Overall Architecture
NeRF weights are converted into parameter graphs, from which a GMN encoder extracts latent vectors. An nf2vec-style decoder reconstructs the radiance field. Training combines a rendering loss $\mathcal{L}_R$ and a SigLIP contrastive loss $\mathcal{L}_C$. At inference, the encoder output is directly used for classification, retrieval, and language-grounded tasks.

### Key Designs

1. **Parameter Graph Construction (NeRF → Graph)**:

    - **Function**: Convert three NeRF architecture families into a unified graph representation.
    - **Mechanism**: MLPs are represented using standard parameter graphs (weights as edge features); tri-planes follow the spatial parameter grid representation of Lim et al.; **hash tables (new contribution of this paper)**—a node is created for each table entry and each feature dimension, with the corresponding entry–feature value as the edge feature, avoiding the cubic complexity that would arise from explicitly modeling the underlying voxel grid.
    - **Design Motivation**: Hash tables constitute the most prevalent contemporary NeRF architecture and must be supported.

2. **GMN Encoder**:

    - **Function**: Extract architecture-agnostic latent vectors from parameter graphs.
    - **Mechanism**: A standard message-passing GNN updates node and edge features through neighborhood aggregation; mean pooling over edge features yields the final embedding.
    - **Design Motivation**: GNNs are naturally equivariant to node permutations and can handle arbitrary graph structures, making them applicable to arbitrary NeRF architectures.

3. **SigLIP Contrastive Loss**:

    - **Function**: Align latent representations of the same object across different NeRF architectures.
    - **Mechanism**: $\mathcal{L}_C = -\frac{1}{|\mathcal{B}|} \sum_{j,k} \ln \frac{1}{1+e^{-\ell_{jk}(t \mathbf{u}_j \cdot \mathbf{v}_k + b)}}$, where $\ell_{jk}=1$ for the same object and $-1$ for different objects.
    - **Design Motivation**: Training with rendering loss alone causes NeRFs of different architectures to form architecture-based clusters in latent space rather than content-based clusters; the contrastive loss breaks this architectural barrier.

### Loss & Training
$\mathcal{L}_{R+C} = \mathcal{L}_R + \lambda \mathcal{L}_C$, with $\lambda = 2 \times 10^{-2}$.

## Key Experimental Results

### Main Results (Multi-Architecture Classification Accuracy)

| Setting | Training Architectures | Test Architectures | Accuracy |
|---------|----------------------|--------------------|----------|
| Single-arch MLP | MLP | MLP | ~82% |
| Single-arch TRI | TRI | TRI | ~84% |
| Single-arch HASH | HASH | HASH | ~83% |
| Multi-arch ALL ($\mathcal{L}_{R+C}$) | MLP+TRI+HASH | MLP+TRI+HASH | ~83% |
| Multi-arch → unseen architectures | MLP+TRI+HASH | 10 unseen variants | ~78% |

### Ablation Study

| Loss | Multi-Arch Classification | Cross-Arch Retrieval | Notes |
|------|--------------------------|----------------------|-------|
| $\mathcal{L}_R$ only | Architecture-based clustering | Very low | Different architectures form separate clusters |
| $\mathcal{L}_C$ only | ~79% | High | Lacks rendering constraint |
| $\mathcal{L}_{R+C}$ | ~83% | Highest | Optimal combination |

### Key Findings
- **Rendering loss alone induces architecture clustering**: t-SNE visualizations clearly show that NeRFs of different architectures representing the same object are assigned to separate clusters.
- **Contrastive loss is essential**: Adding SigLIP organizes the latent space by object category rather than by architecture.
- **Generalization to unseen architectures**: ~78% accuracy is maintained across 10 unseen hyperparameter-variant architectures.
- **First treatment of hash-table NeRFs**: Validates the generality of the parameter graph representation.

## Highlights & Insights
- **The parameter graph design for hash tables is elegant**: It avoids cubic complexity while preserving the memory efficiency intrinsic to hash tables.
- **The insight that contrastive loss breaks architectural barriers is profound**: Rendering loss learns object content but conflates architectural information into the representation; SigLIP explicitly enforces the constraint that the same object should be nearby regardless of architecture.
- **Implications for standardizing NeRF data formats**: If NeRFs of heterogeneous architectures can be jointly retrieved, NeRF may emerge as a universal storage format for 3D data.

## Limitations & Future Work
- Validation is limited to ShapeNet synthetic data; real-scene NeRFs involve substantially greater complexity.
- Cross-family generalization (e.g., trained on MLP, tested on HASH) is not thoroughly evaluated.
- The parameter graph for hash tables does not preserve spatial adjacency relationships.
- 3D Gaussian Splatting, an important emerging representation, is not addressed.

## Related Work & Insights
- **vs. nf2vec**: Restricted to fixed MLP architectures; this work extends processing to arbitrary architectures.
- **vs. Cardace et al.**: Restricted to tri-plane architectures; this work unifies all three families.
- This work is a pioneering application of meta-networks to the 3D domain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First architecture-agnostic NeRF weight processing framework
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of 13 architectures, though limited to ShapeNet data
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and thorough ablation study
- Value: ⭐⭐⭐⭐ Significant contribution toward unified NeRF processing

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](learning_unified_representation_of_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](../../AAAI2026/3d_vision/point-sra_self-representation_alignment_for_3d_representation_learning.md)
- [\[AAAI 2026\] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space](../../AAAI2026/3d_vision/split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](../../CVPR2026/3d_vision/ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](../../CVPR2026/3d_vision/noderf_neural_ode_nerf_continuous_spacetime_dynam.md)

<!-- RELATED:END -->
