---
title: >-
  [Paper Note] SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization
description: >-
  [CVPR 2025][Image Generation][non-rigid shape matching] SGMatch proposes a semantic-guided non-rigid 3D shape matching framework that integrates semantic features from a vision foundation model into geometric descriptors via a Semantic-Guided Local Cross-Attention (SGLCA) module to eliminate symmetry ambiguity, and introduces a Conditional Flow Matching (CFM) regularization to promote spatial smoothness of correspondences, achieving consistent improvements under non-isometric…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "non-rigid shape matching"
  - "functional maps"
  - "semantic guidance"
  - "conditional flow matching"
  - "cross-attention"
date: 2026-05-08
content_hash: 640e30e37236c5b6
---

# SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization

**Conference**: CVPR 2025  
**arXiv**: [2603.12937](https://arxiv.org/abs/2603.12937)  
**Code**: None  
**Area**: Image Generation/3D Shape  
**Keywords**: non-rigid shape matching, functional maps, semantic guidance, conditional flow matching, cross-attention

## TL;DR

SGMatch proposes a semantic-guided non-rigid 3D shape matching framework that integrates semantic features from a vision foundation model into geometric descriptors via a Semantic-Guided Local Cross-Attention (SGLCA) module to eliminate symmetry ambiguity, and introduces a Conditional Flow Matching (CFM) regularization to promote spatial smoothness of correspondences, achieving consistent improvements under non-isometric deformations and topological noise (outperforming the previous SOTA by 24% on SMAL).

## Background & Motivation

**Background**: Non-rigid 3D shape matching aims to establish point-to-point correspondences between shapes. Mainstream methods are based on the Functional Maps framework, which represents dense correspondences by estimating low-dimensional linear operators on Laplace-Beltrami eigenfunctions. In the deep learning era, unsupervised functional mapping methods (such as ULRSSM and HybridFMap) replace hand-crafted features with learned descriptors, performing exceptionally well in near-isometric scenarios.

**Limitations of Prior Work**: (1) **Symmetry Ambiguity**: Relying solely on geometric descriptors (e.g., HKS, WKS) cannot distinguish symmetric body parts (e.g., left vs. right hand), leading to ambiguous correspondences in functional maps. (2) **Spatial Inconsistency**: When projecting truncated spectral bases to dense point-to-point correspondences, even if the global spectral alignment seems plausible, local correspondences can suffer from jumps and discontinuities. (3) Under non-isometric deformations (cross-species animal matching) and topological noise (self-intersecting real scans), the discriminative power of geometric descriptors drops sharply.

**Key Challenge**: Correct matching is inherently semantic (e.g., mouth-to-mouth, tail-to-tail), but the functional maps pipeline traditionally relies entirely on intrinsic geometric features, lacking semantic awareness. However, simply injecting global semantic information can destroy local geometric structures. How to introduce semantic guidance while maintaining geometric continuity?

**Goal**: (1) Utilize semantic features to eliminate symmetry ambiguity; (2) Improve spatial smoothness of point-to-point correspondences through continuous feature transmission regularization.

**Key Insight**: DINO-family vision foundation models provide consistent semantic features across instances, which have been proven to enhance 3D matching. However, the authors argue that semantic features should be treated as "structure-aware anchors" rather than directly replacing geometric features. Through gating mechanisms and local attention constraints, semantic clues can resolve ambiguity while respecting manifold locality.

**Core Idea**: Design an SGLCA module that allows semantic features to adaptively modulate geometric features through gating and local neighborhood attention. Then, use the Conditional Flow Matching (CFM) framework to regularize the feature transmission process, encouraging spatially adjacent vertices to move along non-diverging trajectories, thereby simultaneously resolving ambiguity and inconsistency.

## Method

### Overall Architecture

Given a pair of 3D shapes $\mathcal{X}$ and $\mathcal{Y}$ (triangular meshes), the pipeline of SGMatch is: (1) Extract geometric features $\mathbf{F}^{geo}$ (DiffusionNet) and semantic features $\mathbf{F}^{sem}$ (multi-view distillation of DINOv2) respectively; (2) Fuse them into $\mathbf{F}^{fuse}$ through the SGLCA module; (3) Estimate the functional map $\mathbf{C}_{\mathcal{XY}}$ using the fused features and recover the soft correspondence matrix $\boldsymbol{\Pi}_{\mathcal{YX}}$; (4) Parallelly constrain the smoothness of feature transmission through spectral heat diffusion and Conditional Flow Matching (CFM) regularization.

### Key Designs

1. **Semantic-Guided Local Cross-Attention (SGLCA) Module**:

    - **Function**: Inject semantic context into geometric representations while maintaining local structural continuity.
    - **Mechanism**: Divided into two steps. **Semantic-Guided Gating**: First, project the semantic features linearly to the same dimension as the geometric features to obtain $\tilde{\mathbf{F}}^{sem} = \phi(\mathbf{F}^{sem})$. Then, generate channel-wise gating weights using an MLP: $\mathbf{G} = \sigma(\text{MLP}(\tilde{\mathbf{F}}^{sem}))$, modulating the geometric features as $\tilde{\mathbf{F}}^{geo} = \mathbf{F}^{geo} \odot (1 + \alpha\mathbf{G})$. This allows semantic information to adaptively amplify or suppress different channels of the geometric features. **Local Cross-Attention**: For the neighborhood $\mathcal{N}(i)$ of each vertex $i$, compute the local attention $\omega_{ij} = \text{Softmax}(\mathbf{Q}_i\mathbf{K}_j^\top / \sqrt{d})$ using the modulated geometric features as the Query and the projected semantic features as Key/Value, and then aggregate to obtain the fused features $\mathbf{F}_i^{fuse} = \tilde{\mathbf{F}}_i^{geo} + \text{LN}(\sum \omega_{ij}\mathbf{V}_j)$. Attention is strictly restricted within the local neighborhood of the mesh.
    - **Design Motivation**: Global cross-attention introduces irrelevant long-range interactions (leading to performance degradation in ablation studies) because points on the shape surface that are geographically far apart should not affect each other's matching even if they are semantically similar. The gating mechanism allows semantic clues to participate as "modulators" rather than "replacements", avoiding the risk of semantic features overriding geometric features.

2. **Conditional Flow Matching (CFM) Regularization**:

    - **Function**: Encourage the spatial smoothness of the recovered point-to-point correspondences.
    - **Mechanism**: First, perform **spectral heat diffusion** on the fused features $\mathbf{Z} = \boldsymbol{\Phi}\exp(-\tau\boldsymbol{\Lambda})\boldsymbol{\Phi}^\top\mathbf{M}\mathbf{F}^{fuse}$ to smooth out local noise. Then, define the source feature $\mathbf{z}_0 = \mathbf{Z}_\mathcal{X}$, target feature $\mathbf{z}_1 = \boldsymbol{\Pi}_{\mathcal{XY}}\mathbf{Z}_\mathcal{Y}$ (transported via soft correspondence), linear interpolation path $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$, and target velocity field $\mathbf{v}_{target} = \mathbf{z}_1 - \mathbf{z}_0$. Parameterize a learnable velocity field $\mathbf{v}_\theta(\mathbf{z}_t, t)$ using an MLP (where time $t$ is injected via sinusoidal positional encoding + FiLM conditioning). The training target is $\mathcal{L}_{cfm} = \mathbb{E}_{t,i\in\mathcal{S}}[\sqrt{\|\mathbf{v}_\theta(\mathbf{z}_{t,i}, t) - \mathbf{v}_{target,i}\|^2 + \varepsilon^2}]$, using Charbonnier loss instead of MSE to reduce the impact of outliers from early inaccurate correspondences. Additionally, **importance sampling** based on cosine similarity is introduced to prioritize training on reliable correspondence points.
    - **Design Motivation**: Truncation of spectral bases in functional maps inherently only guarantees low-frequency alignment, and the point-to-point correspondences recovered from them lack high-frequency smoothness guarantees. CFM regularization constrains the feature transmission process to follow a continuous trajectory—spatially adjacent vertices are encouraged to move along non-diverging paths—which is equivalent to requiring local correspondences to vary smoothly, without needing explicit pairwise constraints.

3. **Functional Maps and Point Maps Module**:

    - **Function**: Estimate functional maps in the spectral domain and recover dense correspondences.
    - **Mechanism**: The functional map $\mathbf{C}_{\mathcal{XY}}$ is obtained by minimizing the spectral projection discrepancy of fused features along with structural regularization. Training losses include bijectivity loss $\mathcal{L}_{bij}$, orthogonality loss $\mathcal{L}_{orth}$, and coupling loss $\mathcal{L}_{couple} = \|\mathbf{C}_{\mathcal{XY}} - \boldsymbol{\Phi}_\mathcal{Y}^\dagger\boldsymbol{\Pi}_{\mathcal{YX}}\boldsymbol{\Phi}_\mathcal{X}\|_F^2$ (ensuring consistency between functional maps and point maps). The soft correspondence matrix is calculated via temperature softmax: $\boldsymbol{\Pi} = \text{Softmax}(\mathbf{F}^{fuse}\mathbf{F}^{fuse\top}/\tau_T)$.
    - **Design Motivation**: The coupling loss serves as a bridge connecting the spectral domain (functional maps) and the spatial domain (point-to-point correspondences), ensuring that estimates of both mutually reinforce each other.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{spectral} + \lambda_{cfm}\mathcal{L}_{cfm}$, where $\mathcal{L}_{spectral} = \mathcal{L}_{struct} + \lambda_{couple}\mathcal{L}_{couple}$, with $\lambda_{cfm}=100$, and $\lambda_{bij}=\lambda_{orth}=\lambda_{couple}=1.0$. The model is trained end-to-end using the Adam optimizer.

## Key Experimental Results

### Main Results: Non-Isometric Matching

| Method | SMAL | DT4D-H intra | DT4D-H inter|
|------|------|-------------|-------------|
| ZoomOut | 38.4 | 4.0 | 29.0 |
| GeomFMaps (supervised) | 8.4 | 1.9 | 4.2 |
| ULRSSM | 3.9 | 0.9 | 4.1 |
| HybridFMap | 3.3 | 1.0 | 3.5 |
| DeepFAFM | 3.8 | 0.9 | 3.9 |
| **SGMatch (Ours)** | **2.5** | 1.0 | **3.4** |

On the SMAL dataset, SGMatch outperforms HybridFMap by 24% (2.5 vs. 3.3). On the topologically noisy dataset TOPKIDS, it reaches 3.3 (outperforming HybridFMap at 5.0, a 34% improvement).

### Ablation Study

| Configuration | Geo | Sem | SGLCA | Heat Diff | CFM | SMAL Geo.Err |
|------|-----|-----|-------|-----------|-----|-------------|
| I. Semantic Only | ✓ | ✗ | ✗ | ✓ | ✓ | 3.2 |
| II. Geometric Only | ✗ | ✓ | ✗ | ✓ | ✓ | 21.2 |
| III. Global Attention instead of Local | ✓ | ✓ | global | ✓ | ✓ | 2.6 |
| IV. w/o Spectral Heat Diffusion | ✓ | ✓ | ✓ | ✗ | ✓ | 3.0 |
| V. w/o CFM | ✓ | ✓ | ✓ | ✗ | ✗ | 2.7 |
| **Full** | ✓ | ✓ | ✓ | ✓ | ✓ | **2.5** |

### Key Findings

- **Geometric features are foundational**: Using only semantic features (removing geometry) causes the error to spike to 21.2, showing that semantic features cannot complete matching independently and must use geometric structure as the backbone.
- **Local attention outperforms global attention**: Global attention (2.6 vs. 2.5) introduces irrelevant long-range interactions and is computationally heavier.
- **Spectral heat diffusion and CFM are complementary**: The former stabilizes the feature distribution by smoothing local noise, providing more reliable transmission endpoints for CFM; the latter constrains transmission dynamics via a continuous velocity field, suppressing local inconsistencies that diffusion cannot resolve.
- On near-isometric scenarios (FAUST, SCAPE), SGMatch is on par with the SOTA, and it is stronger in cross-dataset generalization (SHREC'19), indicating that semantic priors are most valuable for out-of-distribution generalization.
- Statistical analysis shows that the standard deviation of SGMatch is much lower than that of HybridFMap (0.01 vs. 0.17 on SMAL), making the optimization process more stable.

## Highlights & Insights

- The design philosophy of **"semantics as a modulator rather than a replacement"** is highly valuable: by utilizing a gating mechanism to enhance or suppress geometric features at the channel level, rather than simple concatenation or addition, the model elegantly balances the contributions of the two modalities.
- **CFM regularization** is an elegant complement to the functional maps framework: while traditional methods focus on spectral domain regularization (bijectivity, orthogonality), CFM imposes constraints from the perspective of the "transport process" in the spatial domain, introducing an entirely different inductive bias. This concept can be transferred to other tasks requiring spatially smooth correspondences (e.g., optical flow estimation, point cloud registration).
- The visualizations in the semantic feature analysis (Appendix C) are highly convincing: in challenging cross-species matching, geometric features yield diffuse high-response regions, whereas DINOv2 semantic features precisely locate the corresponding body parts.

## Limitations & Future Work

- Partial matching scenarios (e.g., partially scanned shapes) are not supported; the framework needs to be extended to partial-to-partial settings.
- The quality of semantic features depends on the domain generalization capability of DINOv2; it may fail if the target shape differs significantly from the pre-training distribution.
- The neighborhood size of SGLCA is fixed to 32, which might not be sufficiently adaptive for meshes of varying resolutions.
- Future directions: Extending to partial correspondences, introducing adaptive semantic feature extraction (e.g., shape-specific fine-tuning strategies), and exploring online learning frameworks to adapt to new categories.

## Related Work & Insights

- **vs. HybridFMap**: Shares the same functional maps foundation but relies solely on geometric descriptors; SGMatch incorporates semantic guidance and CFM regularization, offering distinct advantages in non-isometric scenarios.
- **vs. Diff3F**: Also employs DINOv2 semantic features for 3D matching, but directly matches semantic descriptors in a zero-shot manner; SGMatch integrates semantic features into the learning pipeline, achieving complementary fusion with geometric features.
- **vs. EchoMatch**: Similarly utilizes semantic cues but focuses on partial-to-partial matching; SGMatch addresses ambiguity and inconsistency in full-to-full matching.

## Rating

- Novelty: ⭐⭐⭐⭐ Gating + local attention fusion in SGLCA and CFM regularization are original, though individual components build upon existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets, 4 settings (near-isometric/non-isometric/topological noise/smoothness), and comprehensive ablation and parametric analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, clear methodological motivations, and solid appendix content (7 appendix sections).
- Value: ⭐⭐⭐⭐ Achieves systemic improvements in the classic problem of non-rigid matching; the concept of CFM regularization holds broad transfer value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](../../ICCV2025/image_generation/diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)
- [\[CVPR 2025\] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[CVPR 2025\] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)

</div>

<!-- RELATED:END -->
