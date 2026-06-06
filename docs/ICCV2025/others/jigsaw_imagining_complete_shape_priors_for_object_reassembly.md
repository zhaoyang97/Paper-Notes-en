---
title: >-
  [Paper Note] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly
description: >-
  [ICCV 2025][object reassembly] Jigsaw++ proposes a generative model-based approach for learning complete shape priors, mapping partially assembled fragment point clouds to the shape space of complete objects via a *retar…
tags:
  - "ICCV 2025"
  - "object reassembly"
  - "3D shape completion"
  - "point cloud generation"
  - "Rectified Flow"
  - "shape prior"
date: 2026-05-08
content_hash: 1b84b99ab0449211
---

# Jigsaw++: Imagining Complete Shape Priors for Object Reassembly

**Conference**: ICCV 2025
**arXiv**: [2410.11816](https://arxiv.org/abs/2410.11816)  
**Code**: [GitHub](https://github.com/Jiaxin-Lu/Jigsaw-PlusPlus)  
**Area**: Other
**Keywords**: object reassembly, 3D shape completion, point cloud generation, Rectified Flow, shape prior

## TL;DR

Jigsaw++ proposes a generative model-based approach for learning complete shape priors, mapping partially assembled fragment point clouds to the shape space of complete objects via a *retargeting* strategy, thereby improving reassembly quality in a manner orthogonal to existing assembly algorithms.

## Background & Motivation

Object reassembly finds broad applications in digital archaeology, robotic furniture assembly, and bone restoration, and is broadly categorized into part assembly and fracture assembly. The core deficiency of existing methods lies in their **lack of global understanding of the complete object**. When only partial fragments are available as input, existing methods heavily rely on category-specific templates and fail to generalize to diverse object types.

Specifically, prior methods exhibit three limitations:

**Fragment-wise processing**: Attention is restricted to individual fragments or their fracture surfaces, neglecting global shape constraints imposed by the complete object.

**Template dependency**: Knowledge of object category or availability of a complete template is required, limiting the applicable scope.

**Missing fragment problem**: Real-world scenarios frequently involve missing fragments, yet existing methods assume complete fragment input.

The core motivation of Jigsaw++ is to **learn a category-agnostic complete shape prior** that "imagines" the complete object from partial assembly results, serving as an additional information layer to enhance downstream assembly algorithms.

## Method

### Overall Architecture

Jigsaw++ adopts a two-stage design:
1. **Stage 1 (Shape Prior Learning)**: A point cloud generative model is trained to learn the shape distribution of complete objects.
2. **Stage 2 (Retargeting Reconstruction)**: Starting from partially assembled inputs, a fine-tuned generative model reconstructs the complete shape.

### Key Designs

1. **Bidirectional Point Cloud–Image Mapping**: This is the cornerstone of the entire framework. Point cloud coordinates $\mathbf{o}_i \in [0,1]^3$ are mapped to RGB color space via $f(\mathbf{o}_i) = \lfloor 255 \mathbf{o}_i \rfloor$, then rasterized from specified camera angles. The inverse mapping $f'(\mathbf{c}_i) = \frac{1}{255}\mathbf{c}_i$ recovers 3D coordinates from color images. The elegance of this design is twofold: (1) it leverages the pretrained LEAP (image-to-3D) model to indirectly exploit knowledge from large-scale 2D data; (2) it removes constraints on the number of points, supporting arbitrary-size inputs and outputs.

2. **Rectified Flow-Based Joint Generative Model**: Rectified Flow is adopted as the generative framework to jointly generate global embeddings $\mathbf{g}$ and reconstruction latents $\mathbf{r}$. Rectified Flow learns a transport mapping between two distributions via an ODE:

$$X_t = (1-t)X_0 + tX_1, \quad \frac{d}{dt}X_t = X_1 - X_0$$

Its advantage lies in learning approximately linear trajectories, making both forward and reverse sampling highly efficient. The model uses U-ViT as the backbone, with DINOv2 features extracted during encoding to provide cross-category generalization.

3. **Retargeting Strategy**: This is the core contribution of Jigsaw++. Given a partially assembled object $\hat{O}$ and its latent $\hat{\mathbf{x}}_1$, the reverse ODE is first solved to obtain $\hat{\mathbf{x}}_0$. Since the input is not a complete object, the likelihood of $\hat{\mathbf{x}}_0$ under $\pi_0 = \mathcal{N}(0,I)$ is low. Langevin dynamics is therefore applied to adjust it:

$$\mathbf{x}_0 = \alpha \hat{\mathbf{x}}_0 + \sqrt{1-\alpha^2}\xi, \quad \xi \sim \mathcal{N}(0,I)$$

The model is then fine-tuned on $(\mathbf{x}_0, \mathbf{x}_1)$ pairs to learn the mapping from incomplete inputs to complete shapes. The linear trajectory property of Rectified Flow compresses the number of reverse sampling steps to $1/25$ of the normal requirement, substantially reducing fine-tuning cost.

### Loss & Training

- **Generation stage**: Standard Rectified Flow training objective $\mathbb{E}\|\frac{d}{dt}X_t - v(X_t, t)\|$
- **Retargeting stage**: Fine-tuning objective $\mathbb{E}_{\mathbf{x}_0,\mathbf{x}_1}\|(\mathbf{x}_0 - \mathbf{x}_1) - v(\mathbf{x}_t, t)\|^2$
- Trained on 407 objects (34,075 fracture patterns) from the Breaking Bad training set, without category labels

## Key Experimental Results

### Main Results

**Breaking Bad dataset (fracture assembly)**:

| Method | CD (×10⁻³) ↓ | Precision (%) ↑ | Recall (%) ↑ |
|--------|--------------|----------------|-------------|
| SE(3) | 22.4 | 20.2 | 22.5 |
| SE(3) + Jigsaw++ | 14.3 | 37.8 | 36.6 |
| Jigsaw | 10.5 | 45.6 | 42.7 |
| Jigsaw + Jigsaw++ | **4.5** | **48.7** | **49.5** |

**PartNet dataset (part assembly, DGL baseline)**:

| Category | CD ↓ | Precision ↑ | Recall ↑ |
|----------|------|------------|---------|
| Chair (DGL) | 47.8 | 21.5 | 20.0 |
| Chair + Jigsaw++ | 41.0 | 52.0 | 33.6 |
| Table (DGL) | 53.6 | 16.6 | 15.4 |
| Table + Jigsaw++ | 42.6 | 53.6 | 31.0 |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Reverse sampling ratio $k=1/10$ | Optimal setting; full reverse sampling over-imitates the input |
| $k=1/25$ | Performance remains acceptable, validating the step compression capability of Rectified Flow |
| $\alpha=1$ | Output nearly copies the input |
| Decreasing $\alpha$ | Output progressively shifts toward complete objects, but specific shapes may deviate |
| 20% missing fragments | CD increases only from 1.8 to 2.0; precision/recall remain nearly unchanged |

### Key Findings

- Jigsaw++ is orthogonal to existing assembly algorithms; even with a weak baseline (e.g., SE(3)), incorporating the shape prior yields substantial gains.
- Using GT shape priors for fragment matching reduces Jigsaw's error by 50%; performance remains significantly better than the original method even with 20% noise added.
- Improvements are larger on Breaking Bad (smaller objects suit the color mapping scheme); on PartNet, precision improves by more than 30 percentage points.

## Highlights & Insights

- **Orthogonal design**: Rather than replacing existing methods, Jigsaw++ serves as a complementary layer from which any assembly algorithm can benefit.
- **2D–3D bridging**: The coordinate-to-color mapping combined with LEAP cleverly circumvents the scarcity of 3D training data.
- **Engineering advantage of Rectified Flow**: Linear trajectories make retargeting fine-tuning extremely low-cost, requiring only $1/25$ of the normal reverse sampling steps.

## Limitations & Future Work

- **Scale limitation**: The color mapping fails for large-scale objects (e.g., streetlights), as image resolution is insufficient to capture the necessary detail.
- **Generalization to unseen categories**: The model's reconstruction capability for object types not seen during training is limited.
- **Topological constraints**: Accurately reconstructing topological relations for complex geometries (e.g., cup handles) remains difficult.
- **Downstream utilization**: No existing assembly algorithm is yet designed to fully exploit the generated shape priors.

## Related Work & Insights

- The paradigm of reusing LEAP (image-to-3D) is instructive, demonstrating how large pretrained models can be transferred to data-scarce 3D tasks.
- Rectified Flow holds unique advantages for generative tasks requiring "conditional retargeting."
- Shape priors generated by this approach may benefit additional downstream tasks, including object recognition and shape reasoning in robotic grasping.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The retargeting + color mapping scheme is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple ablation studies, and robustness tests under missing fragments.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, though mathematical notation is dense.
- **Value**: ⭐⭐⭐⭐ Introduces a new paradigm for object reassembly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](../../CVPR2026/others/order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[ICLR 2026\] Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](../../ICLR2026/others/latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[CVPR 2026\] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](../../CVPR2026/others/teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)

</div>

<!-- RELATED:END -->
