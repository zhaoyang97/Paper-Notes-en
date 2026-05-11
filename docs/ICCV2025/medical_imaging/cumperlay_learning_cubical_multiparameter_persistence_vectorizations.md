---
title: >-
  [Paper Note] CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations
description: >-
  [ICCV 2025][Medical Imaging][Multiparameter Persistent Homology] This paper proposes CuMPerLay, a differentiable Cubical Multiparameter Persistence (CMP) vectorization layer that decomposes CMP into multiple learnable single-parameter persistence lines. By jointly learning bifiltration functions for end-to-end training and embedding the layer into Swin Transformer, the method achieves significant improvements on medical image classification and semantic segmentation tasks, particularly in data-scarce settings.
tags:
  - ICCV 2025
  - Medical Imaging
  - Multiparameter Persistent Homology
  - Cubical Complex
  - Differentiable Vectorization
  - Topological Data Analysis
  - Swin Transformer
date: 2026-05-08
content_hash: 1987cae50f73c6e4
---

# CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations

**Conference**: ICCV 2025
**arXiv**: [2510.12795](https://arxiv.org/abs/2510.12795)
**Code**: [circle-group/cumperlay](https://github.com/circle-group/cumperlay)
**Area**: Medical Imaging / Topological Deep Learning
**Keywords**: Multiparameter Persistent Homology, Cubical Complex, Differentiable Vectorization, Topological Data Analysis, Swin Transformer
**Institution**: Imperial College London + UT Dallas

## TL;DR

This paper proposes CuMPerLay, a differentiable Cubical Multiparameter Persistence (CMP) vectorization layer that decomposes CMP into multiple learnable single-parameter persistence lines. By jointly learning bifiltration functions for end-to-end training and embedding the layer into Swin Transformer, the method achieves significant improvements on medical image classification and semantic segmentation tasks, particularly in data-scarce settings.

## Background & Motivation

**Persistent Homology (PH)** is a core tool in Topological Data Analysis (TDA) that captures global structural information by tracking the "birth" and "death" of topological features (connected components, loops, voids, etc.) across scales. Integration of PH into deep learning has made progress (e.g., PersLay), yet existing methods are almost exclusively limited to **Single-Parameter Persistence (SPP)**, which filters data along only one direction.

**Multiparameter Persistence (MP)** filters data along multiple directions simultaneously, capturing richer topological information. However, MP faces two fundamental challenges:

1. **Structural complexity**: MP modules lack a complete discrete invariant analogous to the single-parameter barcode and cannot be compactly represented via persistence diagrams.
2. **Vectorization difficulty**: The absence of a complete MP descriptor makes it extremely difficult to convert MP information into vectors suitable for machine learning.

**Cubical complexes** are the natural choice for image topology — pixels correspond directly to vertices of cubes, and adjacency relations correspond to edges and faces. Compared to Vietoris–Rips or Alpha complexes, cubical complexes perfectly match the grid structure of images and are computationally more efficient.

Nevertheless, **Cubical Multiparameter Persistence (CMP)** has not yet been effectively integrated into deep learning. The core bottleneck is the absence of a CMP-to-vector mapping that is both differentiable and stable. This paper addresses precisely this problem.

## Method

### Core Idea: Decomposing CMP into Learnable Single-Parameter Persistence Lines

The key insight of CuMPerLay is to **decompose the non-directly-vectorizable multiparameter persistent homology into multiple learnable single-parameter persistence lines**. Concretely, given a feature map $x \in \mathbb{R}^{B \times C \times H \times W}$, the method consists of three core steps:

#### 1. Learnable Bifiltration

A **Filtration Decoder** generates bifiltration functions from the feature map:

$$G = f_\theta(x), \quad G \in \mathbb{R}^{B \times C_0 \times H \times W}$$

The Filtration Decoder consists of multiple deconvolution/upsampling layers with GroupNorm, ReLU, and Conv, mapping high-level features to $C_0$ filtration channels, each corresponding to an independent filtration function.

To obtain discrete thresholds for constructing the bifiltration, **Stair Combined Thresholding** is applied: continuous filtration values are sigmoid-normalized to $[0,1]$, and $n_T$ discrete levels (default $n_T=16$) are sampled via learnable thresholds, yielding a compact filtration representation:

$$\hat{G}_{\text{norm}} \in \mathbb{R}^{B \times C_0 \times H' \times W'}$$

#### 2. Cubical Persistence Computation

2D cubical persistent homology is computed independently for each filtration channel, implemented via the C++/CUDA-accelerated `cubical_persistence_v_2d_full` function:

- **Input**: compact filtration representation $\hat{G}_{\text{norm}}$
- **Output**: persistence pairings $P \in \mathbb{R}^{B \times C_0 \times d \times F \times 2}$, where $d$ denotes the homological dimension (dim 0 = connected components, dim 1 = loops) and $F$ is the maximum number of features; pairing lengths $L$ are also output to generate validity masks.

Each pair $(b_i, d_i)$ represents the birth and death values of a topological feature. By computing SPP independently over $C_0$ channels, CuMPerLay reduces the biparameter problem to $C_0$ independent single-parameter problems, while the $C_0$ filtration functions are learned jointly.

#### 3. Differentiable Vectorization: Silhouette v2

An improved **PersLay-style vectorization** is employed, comprising two learnable components:

**Weight function**: computes importance weights for each persistence pair:

$$w_i = c \cdot |d_i - b_i|^p$$

where $c$ (constant) and $p$ (exponent) are both learnable parameters, learned independently per channel and dimension, with optional cross-feature normalization.

**Phi function (Tent Phi)**: maps persistence pairs to an $S$-dimensional sample space (default $S=128$) using learnable sample points $\{s_k\}_{k=1}^S$:

$$\phi_k(b_i, d_i) = \text{ReLU}\left(\frac{d_i - b_i}{2} - \left|s_k - \frac{b_i + d_i}{2}\right|\right)$$

After weighting and summing over the feature dimension, the final vectorized representation is:

$$v_k = \sum_i w_i \cdot \phi_k(b_i, d_i)$$

with output dimension $\mathbb{R}^{B \times C_0 \times S \times d}$.

### Overall Architecture: TopoSwin-MP

CuMPerLay layers are inserted after each stage of **Swin Transformer V2**, forming the TopoSwin-MP architecture:

```
Input → Patch Embed → [Stage1 → Topo1 → Gate1] → [Stage2 → Topo2 → Gate2]
      → [Stage3 → Topo3 → Gate3] → [Stage4 → Topo4 → Gate4] → Norm → Pool → Head
```

Each TopoBlock after a stage consists of: Filtration Decoder → Cubical PH → Silhouette Vectorization → MLP.

**Gated Topology Linear** fuses topological features back into the backbone features:

$$x' = x + x \cdot \sigma(\text{Linear}(t)) + \text{bias}(t)$$

where $t$ is the topological vector and $\sigma$ is the sigmoid gate — analogous to SE attention but conditioned on topological features.

**Input Guidance** additionally computes a CuMPerLay topological feature directly from the raw input image, which is concatenated into the topological vector of each stage to provide low-level topological priors.

An **auxiliary topological classification head** concatenates persistent representations from all stages and passes them through a dedicated MLP for classification, optimized jointly with the main head:

$$\mathcal{L} = \mathcal{L}_{CE} + 0.25 \cdot \mathcal{L}_{topo\_CE} + 0.01 \cdot \mathcal{L}_{multifilt\_reg}$$

where $\mathcal{L}_{multifilt\_reg}$ is a multifiltration regularization loss that encourages different filtration functions to learn distinct patterns.

### Theoretical Guarantee: Stability Theorem

CuMPerLay admits a stability guarantee under the generalized Wasserstein metric: for two cubical complexes $K_1, K_2$, the difference between their vectorized representations is upper bounded by the difference between the input filtration functions. This guarantees that small perturbations to the filtration functions do not cause drastic changes in the vectorization output, providing a theoretical foundation for end-to-end gradient training.

## Key Experimental Results

### Datasets

| Dataset | Task | Scale | Characteristics |
|--------|------|-------|----------------|
| ISIC 2018 | Skin lesion 7-class classification | ~10K images | Severe class imbalance |
| CBIS-DDSM | Mammography binary classification | ~2K cropped images | Small dataset |
| Glaucoma | Fundus image binary classification | ~1.5K images | Small dataset |
| Pascal VOC + SBD | Semantic segmentation, 21 classes | ~11K images | General vision |

### Classification Performance Comparison

**ISIC 2018 (AUC-ROC)**:
- ResNet-50 baseline: ~80%
- Swin-V2-B baseline: ~85%
- TopoSwin-MP (Ours): **significant improvement**, achieving state-of-the-art

**Key conclusion**: CuMPerLay outperforms all baselines and existing TDA methods across all classification datasets:
- Surpasses PersLay (single-parameter persistence layer)
- Surpasses ATOL (adaptive topology layer)
- Surpasses Betti curve-based methods

### Low-Data Regime

CuMPerLay exhibits **particularly pronounced advantages in limited-data settings**:
- When trained with 25% or 50% of the data, the topologically augmented model achieves larger relative gains over the baseline.
- This is attributed to topological features providing data-agnostic global structural priors.

### Segmentation Performance

On Pascal VOC + SBD semantic segmentation, integrating CuMPerLay into the segmentation network also improves mIoU, validating the generality of the approach.

### Ablation Study

- **Multiparameter vs. single-parameter**: Multiparameter persistence (multi-channel filtration) significantly outperforms a single grayscale filtration.
- **Learnable vs. fixed filtration**: Learned filtration functions outperform hand-crafted density/voxel filtrations.
- **Input Guidance**: Adding topological guidance from the raw input further improves performance.
- **Auxiliary topological head**: Joint training with the topological classification head facilitates learning better filtration functions.

## Highlights & Insights

1. **Elegant decomposition strategy**: The mathematically intractable multiparameter persistent homology is decomposed into multiple learnable single-parameter lines, preserving the expressiveness of multiparameter information while leveraging mature single-parameter toolchains — a paradigmatic example of reducing complexity through principled simplification.
2. **End-to-end differentiability**: The entire pipeline (filtration generation → cubical PH → vectorization) is differentiable, enabling gradients from the classification loss to backpropagate all the way to the filtration function parameters, making topological features genuinely task-driven.
3. **Natural fit of cubical complexes**: Choosing cubical rather than simplicial complexes perfectly exploits the grid structure of image data, avoiding the overhead of point cloud sampling and Rips complex construction.
4. **C++/CUDA acceleration**: Persistent homology computation is implemented via custom CUDA kernels, making training practically feasible. Code is publicly available ([circle-group/cmp](https://github.com/circle-group/cmp)).
5. **Gated fusion mechanism**: Topological features are fused back into the backbone via sigmoid-gated multiplicative interaction rather than simple concatenation, allowing the network to adaptively determine the contribution of topological information.

## Limitations & Future Work

1. **2D cubical complexes only**: The current CUDA implementation supports only 2D images; dim=3 is reserved in the framework but not yet implemented. Extension to 3D medical imaging (CT/MRI) is a natural direction.
2. **Computational overhead**: Each stage requires computing full cubical PH, which increases training time by approximately 30–50% despite CUDA acceleration.
3. **Information loss from decomposition**: Decomposing biparameter persistent homology into independent single-parameter lines discards inter-parameter interaction information; topological features visible only in the biparameter space may be missed.
4. **Tasks explored are limited**: Detection, generation, and other tasks remain unexplored.
5. **Relatively heavy Filtration Decoder**: The multi-layer deconvolution-based Filtration Decoder at each stage has non-trivial parameter counts; lightweight design alternatives merit investigation.
6. **Manual selection of $n_T$ and $C_0$**: Currently rely on hand-tuning (defaults of 16 and 8, respectively); adaptive selection mechanisms may be preferable.

## Related Work & Insights

- **PersLay** (NeurIPS 2020): The first general differentiable PH vectorization layer, limited to single-parameter PH. CuMPerLay can be viewed as an extension of PersLay to the multiparameter + cubical complex setting.
- **PLLay** (ICML 2021): Generalizes PersLay to persistent landscapes, also restricted to single-parameter PH.
- **Multiparameter Persistence Image** (NeurIPS 2020, Carrière & Blumberg): Proposes multiparameter persistence images, but is not differentiable.
- **Smart Vectorizations** (Coskunuzer et al., 2021): Prior work by co-authors of this paper exploring intelligent vectorization for single- and multiparameter persistence.
- **Differentiability of MP** (ICML 2024, Scoccola et al.): Proves the differentiability of multiparameter PH, providing the theoretical foundation for this work.
- **Topology + Transformer**: Integrating functional topological features with Swin Transformer represents an emerging trend with potential to generalize to broader visual Transformer architectures.

**Implications for research**: CuMPerLay demonstrates a new paradigm for combining TDA with deep learning — rather than treating TDA as a fixed feature extraction preprocessing step, topological computation itself becomes a learnable network component. This paradigm is readily transferable to graph neural networks, point cloud processing, and beyond.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition](../../AAAI2026/medical_imaging/recon-ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect-coup.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-Supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[ICCV 2025\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)

</div>

<!-- RELATED:END -->
