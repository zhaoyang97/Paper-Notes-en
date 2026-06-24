---
title: >-
  [Paper Note] GIF: Generative Inspiration for Face Recognition at Scale
description: >-
  [CVPR 2025][Image Generation][Large-scale Face Recognition] Proposes replacing scalar labels in face recognition with structured identity codes (integer sequences). Code vectors are generated via CLIP initialization and hypersphere homogenization, followed by hierarchical clustering to build tree-structured codes. This reduces classifier computational complexity from $\mathcal{O}(m)$ to $\mathcal{O}(\log m)$ while addressing the minority collapse problem.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Large-scale Face Recognition"
  - "Efficient Training"
  - "Identity Tokenization"
  - "Hierarchical Clustering"
  - "Logarithmic Computational Complexity"
date: 2026-05-08
content_hash: 1186479ef15bd019
---

# GIF: Generative Inspiration for Face Recognition at Scale

**Conference**: CVPR 2025  
**arXiv**: [2505.03012](https://arxiv.org/abs/2505.03012)  
**Code**: Yes (Code link mentioned in the paper)  
**Area**: Image Generation  
**Keywords**: Large-scale Face Recognition, Efficient Training, Identity Tokenization, Hierarchical Clustering, Logarithmic Computational Complexity

## TL;DR

Proposes replacing scalar labels in face recognition with structured identity codes (integer sequences). Code vectors are generated via CLIP initialization and hypersphere homogenization, followed by hierarchical clustering to build tree-structured codes. This reduces classifier computational complexity from $\mathcal{O}(m)$ to $\mathcal{O}(\log m)$ while addressing the minority collapse problem.

## Background & Motivation

**Background**: The dominant framework for large-scale face recognition (FR) training is Angular Margin Softmax (AMS), such as ArcFace. The number of identities in training datasets has grown from 10k in CASIA-WebFace (2014) to 2 million in WebFace260M (2021). All existing FR methods represent identities using atomic scalar labels (individual integers).

**Limitations of Prior Work**: The computational cost of AMS scales linearly with the number of identities $m$, i.e., $\mathcal{O}(m)$, as Softmax normalization requires computing dot products across all classes. Existing efficient training methods (PFC, DCQ, F2C, Virtual FC) approximate this by randomly sampling subsets, but their complexity remains $\mathcal{O}(\alpha m)$, only reducing the coefficient. Furthermore, sample distributions in large-scale datasets are highly imbalanced; the Softmax centers of minority classes are dominated by "pushing forces," leading them to collapse into a shared common subspace (minority collapse).

**Key Challenge**: The atomic nature of scalar labels dictates that classification must be $m$-way, and subsampling subsets cannot change the fundamental linear complexity. Additionally, scalar labels do not contain inter-class relationship information, leading to suboptimal performance when randomly sampling negative classes.

**Goal**: (1) Reduce computational complexity from linear to logarithmic; (2) solve the minority collapse problem; and (3) maintain or improve recognition performance.

**Key Insight**: Inspired by generative modeling and large-scale entity retrieval fields, which encode entities into compact integer sequences instead of scalar labels. Utilizing codes of length $l$ and vocabulary size $v$ can represent $v^l$ identities, meaning $v \propto \log(m)$, thereby transforming $m$-way classification into $l$ parallel $v$-way classifications.

**Core Idea**: Replace scalar labels with structured identity codes, transforming FR training from a single $m$-way classification into $l$ parallel $v$-way classifications to achieve logarithmic complexity, while ensuring discriminative power via hypersphere homogenization of code vectors and regression losses.

## Method

### Overall Architecture

GIF is divided into two independent phases: (1) Identity Tokenization—mapping the scalar label $y_i$ first to a hyperspherical code vector $\mathbf{h}_{y_i}$, then generating code $\mathbf{c}^{y_i}$ through hierarchical clustering; (2) FR Training—the backbone network $F_\theta$ predicts the identity code of the input face, jointly trained via $l$ parallel $v$-way AMS classifiers and a code vector regression loss.

### Key Designs

1. **Structured Code Vector Generation ($y_i \to \mathbf{h}_{y_i}$)**:

    - **Function**: Assign a semantically structured and maximally separated code vector for each identity on the hypersphere.
    - **Mechanism**: First, extract the average feature of all samples for each identity using a CLIP image encoder as the initial code vector $\mathbf{h}_{y_i} = \frac{1}{|\mathcal{D}_{y_i}|}\sum CLIP(\mathbf{x})$ to ensure semantic consistency. Then, optimize a uniformity loss based on the Gaussian Potential Kernel: $L_{GP} = \log(\frac{1}{\hat{m}}\sum_i\sum_j e^{-t||\mathbf{h}_i - \mathbf{h}_j||^2})$, distributing the code vectors uniformly over $\mathcal{S}^{d-1}$. Optimization randomly selects subsets $\hat{m} < m$ at each iteration, making it independent of the dataset sample distribution and thereby preventing minority collapse.
    - **Design Motivation**: CLIP provides semantic structure (similar identities have closer vectors), while uniformity ensures inter-class separation. Their combination produces codes that are both structured (similar identities share code prefixes) and discriminative.

2. **Hierarchical Clustering Code Construction ($\mathbf{h}_{y_i} \to \mathbf{c}^{y_i}$)**:

    - **Function**: Discretize continuous code vectors into integer sequence codes.
    - **Mechanism**: Recursively apply $k$-means clustering ($k=v$) to the optimized code vectors $\mathbf{H}$ to form a tree structure. The code for each identity, $\mathbf{c}^{y_i} = \{c_1^{y_i}, ..., c_l^{y_i}\}$, is the concatenation of path indices from the root to the corresponding leaf node. Consequently, identities with similar general information share code prefix tokens, forming structured codes. The codes are determined and fixed prior to training and do not participate in gradient updates during main training.
    - **Design Motivation**: The hierarchical code structure gives the $v$-way classification at each level a clear semantic meaning (from coarse to fine), which is more suitable for classification learning than random coding. The settings for $l$ and $v$ ensure that $5 \le v \le 20$.

3. **Dual-Loss Training Framework**:

    - **Function**: Simultaneously ensure code prediction accuracy and intra-class compactness in the feature space.
    - **Mechanism**: The total loss is $L = L_C + \gamma L_{AR}$. The code classification loss $L_C = \sum_{j=1}^l \lambda_j L_{CE}(\bar{c}_j^{y_i}, c_j^{y_i})$ uses $l$ independent $v$-way classifiers with AMS to predict each token, with each classifier having a projection head $H_{\phi_j}$ and $v$ centers. The regression loss $L_{AR} = \frac{1}{2}(\mathbf{z}_i^\top \mathbf{h}_{y_i} - 1)^2$ directly pulls features closer to the code vectors. Its gradient only contains "pulling forces" and no "pushing forces," thereby avoiding minority collapse.
    - **Design Motivation**: A code classification loss alone cannot explicitly encourage intra-class compactness, which is remedied by the regression loss. The gradient structure of the regression loss naturally avoids the issue in AMS where minority classes are dominated by pushing forces.

### Loss & Training

- Tokenization phase: SGD optimizes uniformity loss, lr=0.1, 1000 epochs, batch 2K/GPU.
- Main training: SGD + cosine annealing for ResNet-100, lr=0.1, 20 epochs; AdamW for ViT, lr=1e-4, 40 epochs.
- $\gamma=1$, all $\lambda_j=1$.
- 8× NVIDIA A100.

## Key Experimental Results

### Main Results

| Method | Dataset | Backbone | Complexity | IJB-B TAR@FAR=1e-4 | IJB-C TAR@FAR=1e-4 |
|------|--------|------|--------|---------------------|---------------------|
| PFC | WebFace4M | R100 | 0.3m | 95.64 | 97.22 |
| GIF | WebFace4M | R100 | log m | **96.90** | **97.83** |
| PFC | WebFace12M | R100 | 0.3m | 96.31 | 97.58 |
| GIF | WebFace12M | R100 | log m | **97.08** | **97.82** |
| PFC | WebFace42M | R100 | 0.3m | 96.47 | 97.82 |
| GIF | WebFace42M | R100 | log m | **97.99** | **98.42** |

GIF outperforms PFC on all dataset scales, and reduces computational complexity from linear to logarithmic.

### Ablation Study

| Configuration | Description |
|------|---------|
| Code vector distances (min/mean/max) | Code vector separation of GIF is significantly superior to Softmax centers in ArcFace FC and PFC. |
| Only $L_C$ (without regression loss) | Insufficient intra-class compactness, leading to a drop in recognition performance. |
| Only $L_{AR}$ (without code classification) | Lack of hard negative pushing forces, yielding insufficient discriminative power. |
| Complete $L_C + L_{AR}$ | Best performance. |
| Different ranges of $v$ values (5~20) | $5 \le v \le 20$ yields the best results. |

### Key Findings

- GIF improves over PFC on WebFace42M by 1.52% on IJB-B and 0.6% on IJB-C, while reducing computational complexity from linear to logarithmic.
- The homogenization of code vectors comprehensively improves inter-class separation (minimum/mean/maximum cosine distance) compared to the Softmax centers of PFC and FC.
- The tokenization process is independent of the sample distribution of training data, fundamentally avoiding minority collapse. This is especially crucial for highly imbalanced FR datasets.
- As the dataset scale increases (4M $\to$ 12M $\to$ 42M), the advantage of GIF continuously expands.

## Highlights & Insights

- **Dimension Reduction from Linear to Logarithmic Complexity**: The core insight is to reformulate the $m$-way classification problem into $l$ independent $v$-way problems, which is a fundamental problem reconstruction rather than just an engineering optimization. This approach can be generalized to any ultra-large-scale classification problem (e.g., product retrieval, speech recognition).
- **Code Vectors Independent of Sample Distribution**: The tokenization process relies solely on the number of classes and CLIP features, unaffected by the number of samples per class. This structurally resolves minority collapse, offering a more fundamental solution than heuristic patches on the loss function.
- **Gradient Analysis of Regression Loss**: The gradient of $L_{AR}$ only provides a "pulling force" (drawing features toward their corresponding code vectors) without a pushing force. Therefore, unlike CE, it does not generate a dominant pushing force on minority classes that leads to collapse. This analysis is highly elegant.

## Limitations & Future Work

- The tokenization process requires CLIP forward inference for all samples, incurring extra overhead for ultra-large-scale datasets (e.g., 260M images).
- The $l$ and $v$ parameters in hierarchical clustering must be set manually, lacking an adaptive selection mechanism.
- Codes are fixed prior to training and cannot be dynamically adjusted during training. A potential improvement is online code updating.
- Evaluated solely on face recognition, leaving other large-scale classification tasks (e.g., product retrieval) unverified.

## Related Work & Insights

- **vs PFC (Partial FC)**: PFC randomly samples 30% of classes for computation, where complexity remains $\mathcal{O}(0.3m)$. GIF reconstructs the problem to achieve $\mathcal{O}(\log m)$ with better performance.
- **vs DCQ / Virtual FC / F2C**: These methods perform approximations within the Softmax framework, whereas GIF fundamentally alters the label representation.
- **vs Generative Retrieval Methods (e.g., DSI)**: GIF draws inspiration from NLP/IR fields that encode entities as token sequences, but customizes its design (homogenization + regression loss) specifically for hyperspherical metric learning in FR.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Shifting label representation from scalars to structured codes is a paradigm-shifting innovation, achieving logarithmic complexity in the FR field for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple dataset scales from 4M to 42M, with comprehensive evaluations on standard benchmarks like IJB-B/C.
- Writing Quality: ⭐⭐⭐⭐ Features clear logic and comprehensive mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ Possesses significant practical value for large-scale face recognition and broader large-scale classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](../../ICCV2025/image_generation/vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[CVPR 2025\] OFER: Occluded Face Expression Reconstruction](ofer_occluded_face_expression_reconstruction.md)
- [\[CVPR 2025\] FDeID-Toolbox: Face De-Identification Toolbox](fdeid-toolbox_face_de-identification_toolbox.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)

</div>

<!-- RELATED:END -->
