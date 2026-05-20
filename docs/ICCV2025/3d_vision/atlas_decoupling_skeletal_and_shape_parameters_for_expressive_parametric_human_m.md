---
title: >-
  [Paper Note] ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling
description: >-
  [ICCV 2025][3D Vision][Parametric Body Model] This paper presents ATLAS, a parametric human body model that explicitly decouples external surface shape from internal skeletal parameters…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Parametric Body Model"
  - "Skeleton Decoupling"
  - "Pose Correctives"
  - "SMPL"
  - "Human Mesh"
date: 2026-05-08
content_hash: be4f77f41a0739b3
---

# ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling

**Conference**: ICCV 2025
**arXiv**: [2508.15767](https://arxiv.org/abs/2508.15767)  
**Code**: [https://jindapark.github.io/projects/atlas](https://jindapark.github.io/projects/atlas)  
**Area**: 3D Vision / Human Body Modeling
**Keywords**: Parametric Body Model, Skeleton Decoupling, Pose Correctives, SMPL, Human Mesh

## TL;DR

This paper presents ATLAS, a parametric human body model that explicitly decouples external surface shape from internal skeletal parameters, incorporates sparse nonlinear pose correctives, and is trained on 600K high-resolution scans, achieving more accurate and controllable human body modeling than SMPL-X.

## Background & Motivation

Parametric human body models (e.g., SMPL/SMPL-X) are fundamental to 3D human understanding. Prevailing methods follow a **vertex-centric** paradigm: surface vertices are first customized via linear basis functions, internal skeletal joint positions are then **regressed** from surface vertices, and LBS is finally applied for pose driving. This introduces three core issues:

**Joint–vertex coupling**: Deriving joints from the surface leads to spurious associations — SMPL-X's skeletal joints are asymmetric (elbows, spine, feet), and the spine shifts laterally with soft-tissue variation.

**Difficulty of control**: Modifying shoulder width requires adjusting multiple shape components and inevitably affects soft tissue.

**Keypoint fitting introduces hallucination**: Fitting to keypoints induces ungrounded soft-tissue deviations.

The core idea of ATLAS is to **explicitly decouple shape (soft tissue) from skeleton**, so that skeletal parameters are independent of surface shape and can be controlled separately.

## Method

### Overall Architecture

ATLAS generates a shaped and posed human mesh in two steps:
1. **Surface customization**: Surface vertices (soft-tissue attributes) are customized in A-pose under a fixed template skeleton.
2. **Skeleton customization + pose driving**: Internal skeleton is modified via 76 controllable skeletal attributes, then LBS simultaneously scales and drives the pose.

### Key Designs

1. **Decoupled Surface and Skeletal Basis Functions**

    - Surface customization: $\tilde{X}(\beta^s, \beta^f, \theta) = \bar{X} + \mathcal{B}^s(\beta^s, \mathcal{S}) + \mathcal{B}^f(\beta^f, \mathcal{F}) + \mathcal{B}^p(\theta, \mathcal{P})$
    - At this stage the mesh is still aligned to the fixed template skeleton and has not been pose-driven.
    - Skeleton customization: $\ell = \sigma \oplus t$, comprising 15 body-part scale factors and 61 bone-length parameters.
    - The skeleton also has independent basis functions $\mathcal{B}^k(\beta^k)$ to capture common variation.
    - **Key property**: Joint positions are determined solely by $\beta^k$ and $\theta$, and are entirely independent of the vertex shape parameters $\beta^s$.

2. **Controllable Skeletal Attributes (76-dimensional)**

    - 15 scale attributes: overall body size, head, hands, feet, individual fingers.
    - 61 bone-length attributes: spine, neck, upper/lower arms, upper/lower legs, fingers, etc.
    - Each attribute can be adjusted independently, supporting fine-grained customization (e.g., widening only the shoulders or lengthening only the arms).

3. **Sparse Nonlinear Pose Correctives**

    - Combines the advantages of both approaches: sparse linear (STAR, avoids spurious correlations) and dense nonlinear (GHUM, high expressiveness).
    - A lightweight MLP first nonlinearly encodes the pose angles of joint $j$ and its kinematic neighbors:
    $\text{Non-Linear}_j(\theta) = \text{MLP}(\{R_{6d}(\theta_a) - R_{6d}(\vec{0}) \mid a \in n(j)\})$
    - A sparse mask then constrains the vertex influence range of each joint:
    $\mathcal{B}^p_j = \phi(A_j) \odot (P_j \times \text{Non-Linear}_j(\theta))$
    - Mask $A_j$ is initialized with normalized geodesic distances and remains sparse after training with L1 regularization.

4. **Single-Image Fitting Pipeline**

    - Decoupled fitting: skeletal parameters $\beta^k$ are optimized using only keypoint and depth terms; surface parameters $\beta^s$ are optimized using only silhouette mask terms.
    - Leverages depth predictions and foreground masks from Sapiens.
    - VAE pose prior (trained on 600K frames) + PCA hand prior.
    - Expression fitting adopts an improved align-then-minimize strategy.

### Loss & Training

Training data is processed in a decoupled manner:
- Step 1: Registration is optimized using only skeletal parameters and pose (triangulated keypoints regularize joint positions).
- Step 2: Surface shape is optimized to model soft tissue.
- Separate autoencoders are then trained for the skeletal and surface spaces.

Fitting objective:
$$E(\beta^s, \beta^f, \beta^k, \theta) = E_{data} + E_{\theta_{body}} + E_{\theta_{hand}} + E_{\beta^s} + E_{\beta^f} + E_{\beta^k}$$

## Key Experimental Results

### Main Results (3DBodyTex Fitting)

| Method | Vertex Error @ 32 components (mm) | Notes |
|---|---|---|
| SMPL | ~5.5 | Trained on 1.8K scans |
| STAR | ~5.2 | Sparse correctives |
| SMPL-X | ~4.8 | Includes hands and face |
| SUPR | ~4.6 | Federated dataset |
| **ATLAS** | **~3.8** (↓21.6%) | Decoupled skeleton + shape |

On Goliath-Test (100 unseen scans): ATLAS 2.34 mm vs. SMPL-X 2.78 mm.

Single-image fitting (Goliath-Test, 200 scans):

| Method | Vertex Error (mm) | Joint Error (mm) |
|---|---|---|
| SMPLify-X | 87.7 | 73.2 |
| **ATLAS** | **55.4** | **53.7** |

### Ablation Study

| Configuration | Fitting Error (mm) | Notes |
|---|---|---|
| Linear pose correctives | 1.82 | Conventional approach |
| **Nonlinear pose correctives** | **1.61** | Notable improvement at joint regions |
| w/o relative depth | 60.7 | Depth term is effective |
| w/o depth + w/o mask | 61.8 | Both terms contribute |

Runtime performance:

| Method | Vertex Count | Inference Time (ms) |
|---|---|---|
| SMPL-X | 10,475 | 3.74 |
| ATLAS (SMPL topology) | 6,890 | 2.39 |
| ATLAS (high-resolution) | 115,834 | 5.37 |

### Key Findings

- Decoupling enables lower error with fewer components: ATLAS outperforms SMPL-X by 21.6% at 32 components.
- Nonlinear correctives yield the most significant improvements at complex joints such as shoulders and elbows.
- Modifying skeletal attributes precisely preserves original surface details, and vice versa.
- ATLAS is implemented with CUDA optimizations and runs 34% faster than SMPL-X at the same resolution.

## Highlights & Insights

- **Decoupling is essential**: Replacing vertex-regressed joints with independent skeletal parameterization fundamentally eliminates spurious coupling.
- **Remarkable data scale**: 600K high-resolution scans (240 cameras), 15K identities, and 157 pose sequences — far exceeding all prior work.
- **Sparse nonlinear correctives** are elegantly designed: geodesic-distance initialization combined with L1 regularization enforces sparsity, while MLP-based neighbor encoding provides nonlinear expressiveness.
- The decoupled fitting pipeline — keypoints for skeleton, silhouette for surface — is well-motivated and avoids the soft-tissue hallucination induced by keypoint-based fitting.

## Limitations & Future Work

- The 15K subjects cannot cover the full diversity of human body shapes.
- Acquisition and processing of high-resolution scans are costly, limiting further scaling.
- Clothing modeling is not addressed.
- Hand and facial expression spaces are transferred from FLAME/MANO rather than learned end-to-end.

## Related Work & Insights

- This represents a **fundamental advancement** over the SMPL family — a paradigm-level change rather than an incremental improvement.
- BLSM and SKEL have also attempted decoupling but each has limitations (BLSM lacks pose correctives; SKEL inherits SMPL's shape space).
- The sparse + nonlinear combination can be generalized to parametric modeling of other body parts such as the face and hands.
- Developed at Meta with significant data advantages and strong engineering completeness.

## Rating

- Novelty: ⭐⭐⭐⭐ Skeleton–shape decoupling is not entirely new, but is well-engineered.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, metrics, ablations, and a single-image fitting pipeline.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Has the potential to become the next-generation standard human body model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)

</div>

<!-- RELATED:END -->
