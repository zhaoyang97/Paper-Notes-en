---
title: >-
  [Paper Note] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images
description: >-
  [ECCV 2024][Model Compression][Joint Image Alignment] SpaceJAM is proposed as an unsupervised joint image alignment method with only approximately 16K trainable parameters. It requires neither regularization terms nor atlas maintenance, matching the alignment capabilities of existing methods on the SPair-71K and CUB datasets while achieving a speedup of over 10x.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Joint Image Alignment"
  - "Congealing"
  - "Regularization-free"
  - "Spatial Transformer Networks"
  - "Lightweight"
date: 2026-05-08
content_hash: 3b3287b095dffdbf
---

# SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images

**Conference**: ECCV 2024  
**arXiv**: [2407.11850](https://arxiv.org/abs/2407.11850)  
**Code**: [Available](https://bgu-cs-vil.github.io/SpaceJAM/)  
**Area**: Model Compression  
**Keywords**: Joint Image Alignment, Congealing, Regularization-free, Spatial Transformer Networks, Lightweight

## TL;DR

SpaceJAM is proposed as an unsupervised joint image alignment method with only approximately 16K trainable parameters. It requires neither regularization terms nor atlas maintenance, matching the alignment capabilities of existing methods on the SPair-71K and CUB datasets while achieving a speedup of over 10x.

## Background & Motivation

Joint Image Alignment (JA), also referred to as "congealing", is the task of aligning a collection of images containing the same category of objects to a common coordinate system under unsupervised conditions. This task faces multiple challenges:

**High Complexity**: Existing methods rely on expensive models with a large number of parameters.

**Geometric Deformation**: Complex spatial transformations must be handled.

**Local/Global Optimum Convergence Issues**: The optimization process easily gets trapped in poor local minima.

**Difficult Hyperparameter Tuning**: A large number of regularization terms leads to an enormous hyperparameter search space.

Although Vision Transformers (ViTs) have recently provided valuable feature representations for JA, they do not fully address the aforementioned issues. Existing approaches such as Neural Congealing and GANgealing typically rely on:
- Expensive generative models (e.g., GANs, diffusion models)
- Multiple regularization terms (e.g., flow field smoothness, atlas consistency)
- Long training times and complex hyperparameter tuning

**Core Motivation**: Can a lightweight, regularization-free method be designed to achieve joint alignment efficiently?

## Method

### Overall Architecture

The core design philosophy of SpaceJAM is "simplicity and efficiency":

| Design Dimension | SpaceJAM | Existing Methods (e.g., Neural Congealing) |
|----------|----------|-------------------------------|
| Trainable Parameters | ~16K | Millions |
| Regularization Terms | None | Multiple (flow field, atlas, etc.) |
| Atlas Maintenance | Not Required | Required |
| Training Speed | Fast (10x+) | Slow |
| Feature Extraction | Frozen ViT | Frozen ViT / Trainable |

SpaceJAM utilizes a compact network architecture based on the following key observations:
1. Pre-trained ViT features (especially DINO/DINOv2) are already strong enough, eliminating the need to learn additional feature representations.
2. It is only necessary to learn a lightweight mapping from ViT features to spatial transformation parameters.
3. Optimizing the alignment objective directly in the feature space avoids the need for regularization.

### Key Designs

**1. Compact Architecture (~16K Parameters)**

SpaceJAM adopts a minimalist architecture:
- Input: Patch-level features extracted by a frozen pre-trained ViT
- Intermediate Layers: A lightweight MLP or convolutional layer mapping ViT features to transformation parameters
- Output: Spatial Transformer Network (STN)-style transformation parameters

The key to this design is to treat ViT features as already containing rich semantic and geometric information, requiring only a simple parameter prediction head to be learned.

**2. Regularization-free Design**

Traditional methods require various regularizations to prevent deformation field degradation (e.g., folding, excessive deformation). SpaceJAM avoids this through:
- Operating in the ViT feature space rather than the pixel space, where features possess natural semantic smoothness
- Using parametric transformation families (such as affine or Thin Plate Splines) instead of free-form deformation fields
- A compact parameter space that inherently restricts the complexity of the transformation

**3. No Atlas Maintenance**

Many JA methods require maintaining and updating a shared template (atlas). SpaceJAM circumvents this completely by directly optimizing the alignment objective pairwise in the feature space, simplifying the optimization workflow.

### Loss & Training

The optimization objective of SpaceJAM is to maximize the feature consistency of a set of images in the ViT feature space after transformation. The core loss function is based on:
- Similarity metrics in the feature space (such as cosine similarity or L2 distance)
- No inclusion of any additional regularization terms

Training Strategy:
- Uses the Adam optimizer
- Requires no pre-training or progressive/staged training
- Converges extremely fast (one of the sources of the 10x+ speedup)

## Key Experimental Results

### Main Results

Semantic correspondence evaluation on the SPair-71K dataset:

| Method | Parameter Count | Training Time | PCK@0.1 | Requires Regularization |
|------|--------|----------|---------|-----------|
| Neural Congealing | ~Millions | Hours | High | Yes |
| GANgealing | ~Millions | Hours | High | Yes |
| ASIC | Large | Long | High | Yes |
| **SpaceJAM** | **~16K** | **Minutes** | **Comparable** | **No** |

### Results on CUB Dataset

| Method | Parameter Count | Speedup | Alignment Quality | Atlas Maintenance |
|------|--------|----------|----------|-----------|
| Traditional JA Methods | Large | 1x | Baseline | Required |
| ViT-based Methods | Large | ~1x | High | Required |
| **SpaceJAM** | **Minimal** | **≥10x** | **Comparable** | **Not Required** |

### Key Findings

- SpaceJAM matches the alignment quality of methods with parameter counts orders of magnitude larger, using only about 16K trainable parameters.
- It achieves at least a 10x training speedup, making the process more accessible and efficient.
- The elimination of regularization terms greatly simplifies the hyperparameter tuning process.
- It demonstrates the strong representation capability of ViT features—complex alignment tasks do not require complex models.
- Its effectiveness is validated on two standard datasets, SPair-71K and CUB.

## Highlights & Insights

1. **Extreme Simplicity**: Achieving state-of-the-art level alignment with only 16K parameters, challenging the "bigger is better" paradigm.
2. **Regularization-free**: Completely discarding regularization terms is a bold design choice, demonstrating that regularization can be redundant when operating in an appropriate feature space.
3. **Practically Friendly**: No atlas maintenance, no complex hyperparameter tuning, and fast training significantly lower the barrier to entry.
4. **Value of ViT Features**: Further validates that self-supervised ViT features such as DINO/DINOv2 provide extremely powerful visual semantic representations.
5. **Inspiring Insights**: When upstream features are sufficiently strong, downstream task modeling can be vastly simplified.

## Limitations & Future Work

- The alignment capability "matches" rather than "surpasses" existing methods, and it may be less flexible than non-parametric methods under extreme deformation scenarios.
- The compact parametric transformations may limit the ability to handle highly non-rigid deformations.
- Reliance on high-quality pre-trained ViT features means that model selection (e.g., DINO vs. DINOv2) can impact performance.
- Not yet validated in domains requiring high-precision registration, such as medical imaging.
- Future work can explore incorporating a small number of learnable regularizations to further improve robustness in extreme scenarios.

## Related Work & Insights

- **Neural Congealing [Ofri-Amar et al., CVPR 2023]**: A joint alignment method based on ViT features that constructs semantic atlases.
- **GANgealing [Peebles et al., CVPR 2022]**: GAN-supervised dense visual alignment.
- **ASIC [Gupta et al., ICCV 2023]**: Aligning sparse in-the-wild image collections.
- **STN [Jaderberg et al., NeurIPS 2015]**: Spatial Transformer Networks, the cornerstone of parametric spatial transformation.
- **DINO/DINOv2**: Provides the high-quality frozen ViT features on which this work relies.
- The authors' team has a series of works on regularization-free methods (e.g., CPAB transformations, Regularization-free DTAN).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 3.5 |
| Practicality | 4.5 |
| Writing Quality | 4 |
| Overall | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](../../CVPR2025/model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](../../ICML2025/model_compression/joker_joint_optimization_framework_for_lightweight_kernel_machines.md)
- [\[ICLR 2026\] Modality-free Graph In-context Alignment](../../ICLR2026/model_compression/modality-free_graph_in-context_alignment.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](../../ACL2026/model_compression/wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)

</div>

<!-- RELATED:END -->
