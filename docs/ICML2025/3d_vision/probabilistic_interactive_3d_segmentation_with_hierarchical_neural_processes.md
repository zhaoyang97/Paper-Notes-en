---
title: >-
  [Paper Note] Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes
description: >-
  [ICML 2025][3D Vision][Interactive 3D Segmentation] NPISeg3D proposes the first probabilistic interactive 3D segmentation framework based on Hierarchical Neural Processes. Through a two-level latent variable structure (scene-level and object-level) and a probabilistic prototype modulator, it achieves segmentation accuracy superior to AGILE3D under a few clicks, while providing reliable uncertainty estimations.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "Interactive 3D Segmentation"
  - "Neural Processes"
  - "Hierarchical Latent Variables"
  - "Uncertainty Estimation"
  - "Few-Shot Generalization"
date: 2026-05-08
content_hash: 78a748d8023ca04d
---

# Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes

**Conference**: ICML 2025  
**arXiv**: [2505.01726](https://arxiv.org/abs/2505.01726)  
**Code**: [https://jliu4ai.github.io/NPISeg3D_projectpage/](https://jliu4ai.github.io/NPISeg3D_projectpage/)  
**Area**: 3D Vision  
**Keywords**: Interactive 3D Segmentation, Neural Processes, Hierarchical Latent Variables, Uncertainty Estimation, Few-Shot Generalization

## TL;DR
NPISeg3D proposes the first probabilistic interactive 3D segmentation framework based on Hierarchical Neural Processes. Through a two-level latent variable structure (scene-level and object-level) and a probabilistic prototype modulator, it achieves segmentation accuracy superior to AGILE3D under a few clicks, while providing reliable uncertainty estimations.

## Background & Motivation

**Background**: Interactive 3D segmentation generates precise object masks in complex 3D scenes based on user-provided clicks (positive/negative clicks). Existing methods, such as InterObject3D and AGILE3D, are mainly based on deterministic models and focus on attention mechanisms and multi-object segmentation capabilities.

**Limitations of Prior Work**: (1) **Inadequate few-shot generalization**: Users expect precise segmentation with a minimum number of clicks, but existing methods exhibit limited generalization under sparse inputs, especially in complex scenes and diverse objects; (2) **Lack of uncertainty estimation**: Existing methods (e.g., AGILE3D) completely ignore prediction uncertainty, failing to inform users which areas of prediction might be unreliable, which is a critical flaw in high-risk scenarios (e.g., medical imaging, autonomous driving).

**Key Challenge**: Deterministic models are inherently unable to quantify prediction uncertainty, while simple probabilistic models (such as NPs with a single latent variable) struggle to simultaneously capture global scene structures and object-level features in complex multi-object scenes.

**Goal**: To construct a probabilistic framework that simultaneously addresses the dual challenges of few-shot generalization and uncertainty estimation.

**Key Insight**: Neural Processes are naturally suited for few-shot generalization and uncertainty estimation by treating user clicks as the context set and unlabeled 3D points as the target set. However, single-layer NPs lack sufficient representation capability in multi-object scenes.

**Core Idea**: To introduce a hierarchical latent variable structure (scene-level + object-level) and inject latent variable info into click prototypes via a probabilistic prototype modulator, establishing a multi-granularity information flow of "Scene $\rightarrow$ Objects $\rightarrow$ Clicks".

## Method

### Overall Architecture
The input consists of a 3D point cloud scene $\mathbf{S} \in \mathbb{R}^{N \times 6}$ (coordinates + colors) and a set of user clicks. A point encoder extracts scene features $\mathbf{X}_T$ (target set) and click prototypes $\mathbf{X}_C$ (context set). After hierarchical latent variable inference and probabilistic prototype modulation, the segmentation mask is generated. During inference, uncertainty estimation is achieved through multiple Monte Carlo rollouts.

### Key Designs

1. **Hierarchical Latent Variable Structure**:

    - **Function**: To capture multi-granularity scene context, enhancing few-shot generalization.
    - **Mechanism**: To introduce a scene-level latent variable $\mathbf{z}_s$ modeling global context and inter-object relationships, alongside object-level latent variables $\mathbf{z}_o^m$ capturing fine-grained features of each object. The scene-level latent variable is inferred via a scene-level aggregator: first averaging the click prototypes of each object to obtain object-level prototypes, then gathering them into a scene-level prototype to produce $\mathcal{N}(\mu_s, \sigma_s)$ through a Transformer and MLP. The object-level latent variables are conditioned on $\mathbf{z}_s$: $[\mu_o^m, \sigma_o^m] = \text{MLP}(\alpha \mathbf{z}_s + (1-\alpha) \sum_i \mathbf{X}_C^{m,i})$, where $\alpha$ balances scene-level and object-level information.
    - **Design Motivation**: A single latent variable struggles to model both global structures and object details simultaneously in complex multi-object scenes. A hierarchical structure allows information to propagate progressively from coarse to fine scales.

2. **Probabilistic Prototype Modulator**:

    - **Function**: To inject object-level latent variables into click prototypes, enhancing their context-awareness and uncertainty modeling capabilities.
    - **Mechanism**: Through FiLM-style feature modulation, $\tilde{\mathbf{X}}_C^{m,i,j} = \gamma(\mathbf{z}_o^{m,j}) \odot \mathbf{X}_C^{m,i} + \beta(\mathbf{z}_o^{m,j})$, where $\gamma, \beta$ are generated from the $j$-th Monte Carlo sample $\mathbf{z}_o^{m,j}$ by an MLP. Each click prototype is "modulated" by the object-level context, thereby acquiring stronger discriminative capacity.
    - **Design Motivation**: Original click prototypes only contain local information. Injecting global and object-level semantics via latent variable modulation enriches them. Repeated sampling yields different modulation results, naturally achieving uncertainty quantification.

3. **ELBO-based Training Objective**:

    - **Function**: To jointly optimize segmentation accuracy and latent variable distributions.
    - **Mechanism**: During training, the posterior distributions $q(\mathbf{z}_s|\mathbf{X}_T)$ and $q(\mathbf{z}_o^m|\mathbf{z}_s, \mathbf{X}_T^m)$ are inferred using the target set. KL divergence constraints are applied to regularize the priors (inferred from the context set) to approximate the posteriors. The total loss is defined as $\mathcal{L} = \mathcal{L}_{seg} + \lambda_{kl}(D_{KL}^{scene} + \sum_m D_{KL}^{object_m})$, where the segmentation loss uses Dice and CE.
    - **Design Motivation**: The variational inference framework ensures that the prior can effectively infer meaningful latent variables from sparse clicks during inference.

### Loss & Training
The total loss consists of segmentation losses (Dice + Cross-Entropy) and two-level KL divergence regularization, where $\lambda_{kl}$ balances segmentation accuracy and the constraints on the latent variable distributions. During inference, $N_{z_o}$ Monte Carlo samples are drawn from the prior, and segmentation logits are calculated via cosine similarity and averaged. Uncertainty is obtained by computing the variance across the multiple sampled outputs.

## Key Experimental Results

### Main Results

| Dataset (ScanNet40$\rightarrow$) | Metric | NPISeg3D | AGILE3D | Gain |
|-------------------|------|----------|---------|------|
| S3DIS-A5 (Multi-object) | Avg IoU↑ | 90.5 | 88.3 | +2.2 |
| S3DIS-A5 (Multi-object) | Avg NoC↓ | 5.0 | 6.2 | -1.2 |
| KITTI-360 (Multi-object) | Avg IoU↑ | 48.5 | 44.3 | +4.2 |
| KITTI-360 (Multi-object) | Avg NoC↓ | 17.0 | 18.2 | -1.2 |
| ScanNet (Single-object) | Avg IoU↑ | 88.2 | 87.1 | +1.1 |
| Replica (Multi-object) | Avg IoU↑ | 88.5 | 86.9 | +1.6 |

### Ablation Study

| Configuration | Avg IoU (S3DIS) | Description |
|------|----------------|------|
| Full NPISeg3D | 90.5 | Complete model |
| w/o Hierarchical Latent Variables | ~88.5 | Degenerates to a single-layer NP after removing the scene-level $\mathbf{z}_s$ |
| w/o Probabilistic Prototype Modulation | ~89.0 | Directly uses raw click prototypes |
| w/o KL Regularization | ~88.0 | Latent variable distributions are unconstrained |

### Key Findings
- **Most significant advantage on out-of-domain data**: NPISeg3D outperforms AGILE3D by 4.2% IoU on KITTI-360 (outdoor LiDAR), indicating that the probabilistic framework enhances generalization.
- **Hierarchical latent variables make the largest contribution**: The scene-level $\mathbf{z}_s$ is especially critical for multi-object segmentation as it encodes spatial relationships among objects.
- **Reliable uncertainty estimation**: High-uncertainty regions are highly correlated with actual segmentation errors, which can effectively guide subsequent user clicks.

## Highlights & Insights
- **First attempt to apply NP to segmentation**: Capturing interactive segmentation elegantly under the NP context/target framework (clicks = context, unlabeled points = target). This formulation can be transferred to other tasks such as 2D interactive segmentation and medical image segmentation.
- **FiLM modulation + probabilistic sampling**: The probabilistic prototype modulator cleverly combines FiLM (feature-wise linear modulation) and Monte Carlo sampling. This single mechanism simultaneously achieves context enhancement and uncertainty estimation, resulting in a physical design that is exceptionally clean.
- **Necessity of hierarchical structures**: Multi-object scenes require both global understanding (where objects are located) and local understanding (specific features of individual objects). This insight is also inspiring for other tasks requiring multi-granularity reasoning.

## Limitations & Future Work
- The experiments are trained on ScanNetV2, leaving the potential of pre-training on large-scale point cloud data unexplored.
- Monte Carlo sampling increases the computational cost of inference, and efficiency in real-time interactive scenarios needs to be optimized.
- Although the uncertainty estimation is reliable, the paper does not explore how to automatically utilize uncertainty to select the next optimal click position (which can be formulated as active learning).
- The hierarchical structure is fixed to two levels; deeper hierarchies (such as part-level) might be beneficial in more complex scenes.

## Related Work & Insights
- **vs AGILE3D (Yue et al. 2023)**: AGILE3D employs attention mechanisms for multi-object segmentation but remains purely deterministic. NPISeg3D achieves better generalization and uncertainty estimation through a probabilistic framework, particularly on out-of-domain data.
- **vs Standard NPs (Garnelo et al. 2018)**: Standard NPs only have a single global latent variable, leading to insufficient representational capacity in multi-object scenes. NPISeg3D addresses this limitation with hierarchical latent variables.
- **vs InterPCSeg (Zhang et al. 2024)**: InterPCSeg integrates semantic segmentation networks for test-time correction but is still deterministic and does not provide uncertainty information.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to introduce NPs to interactive 3D segmentation; the hierarchical design is elegant and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 4 datasets with multiple baselines, complete with ablation studies and uncertainty analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly described framework with mathematically rigorous derivations.
- **Value**: ⭐⭐⭐⭐ Probabilistic interactive segmentation is a significant research direction, and the framework holds broad transferability potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](../../ICCV2025/3d_vision/easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](../../CVPR2026/3d_vision/gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](../../ECCV2024/3d_vision/click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[ICML 2025\] Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks](thickness-aware_e3-equivariant_3d_mesh_neural_networks.md)
- [\[CVPR 2025\] MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks](../../CVPR2025/3d_vision/maskgaussian_adaptive_3d_gaussian_representation_from_probabilistic_masks.md)

</div>

<!-- RELATED:END -->
