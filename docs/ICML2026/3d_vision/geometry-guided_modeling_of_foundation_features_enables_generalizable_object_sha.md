---
title: >-
  [Paper Note] Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning
description: >-
  [ICML 2026][3D Vision][Template Deformation] This paper proposes GODeform: it "attaches" 2D foundation model (DINOv3-like) features onto category-level template surfaces for geometry-guided propagation and cross-view fus…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Template Deformation"
  - "Foundation Model Features"
  - "Geometry-Guided Propagation"
  - "View-Adaptive"
  - "Flow Matching"
date: 2026-05-08
content_hash: c999179a5429a0c9
---

# Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning

**Conference**: ICML 2026  
**arXiv**: [2605.29661](https://arxiv.org/abs/2605.29661)  
**Code**: https://GODeform.github.io/ (Project page, code status to be confirmed)  
**Area**: 3D Vision / Monocular Shape Reconstruction / Category-level Deformation  
**Keywords**: Template Deformation, Foundation Model Features, Geometry-Guided Propagation, View-Adaptive, Flow Matching

## TL;DR
This paper proposes GODeform: it "attaches" 2D foundation model (DINOv3-like) features onto category-level template surfaces for geometry-guided propagation and cross-view fusion. It then employs Flow Matching to learn a point-wise deformation field from the template to the target. This enables 3D shape recovery from a single image across large deformations, arbitrary viewpoints, and unseen categories, directly benefiting dexterous grasping transfer.

## Background & Motivation

**Background**: Monocular 3D shape recovery follows two main paradigms. One is generative reconstruction (LRM / Wonder3D / Phidias), which seeks high fidelity but depends heavily on the training distribution. The other is the "deformation paradigm"—given a category template, the model predicts a deformation field from the template to the target (ShapeMatcher, KP-RED, etc.), leveraging the template's topology to stabilize the geometry of occluded regions.

**Limitations of Prior Work**: Generative methods often "hallucinate" unreasonable geometry in self-occluded areas and are sensitive to viewpoint changes. Deformation-based methods typically use visual encoders trained from scratch on small datasets, leading to unstable cross-category semantics. Furthermore, when there is a significant discrepancy between the target and the template (e.g., four-legged chair → sofa, double-layer table → single-layer table), predicted deformation fields suffer from structural degradation, failing to generalize to entirely new categories.

**Key Challenge**: Deformation requires establishing point-wise fine correspondence between the **3D topology of the template** and the **2D observation of the target**. However, 2D foundation models only provide features for visible surfaces without 3D geometric priors, while 3D foundation models are limited by 3D data scale and lack the generalization of 2D models. Simply concatenating them fails due to semantic mismatches caused by viewpoint variations.

**Goal**: Design a unified deformation framework that satisfies three axes of generalization: large discrepancies between template and target, arbitrary target viewpoints, and unseen object categories, while being directly applicable to downstream robotic dexterous grasping.

**Key Insight**: The authors bet on "making 2D foundation features geometry-aware"—diffusing strong 2D semantic correspondences across the entire surface using the template's 3D topology, and explicitly distinguishing "viewpoint artifacts" from "true deformation" using camera poses.

**Core Idea**: Reformulate deformation learning as "Flow Matching conditioned on geometry-guided foundation features." Visible foundation features are diffused across the entire surface via geometric affinity, and multi-view features are fused into viewpoint-invariant template representations through relative poses.

## Method

### Overall Architecture
Input: A single target RGB image $I_{\mathcal{T}}$ + a category-level 3D template point cloud $\mathcal{S} \in \mathbb{R}^{N\times 3}$ + 16 pre-rendered template views $\{I_{\mathcal{S}}^k\}$ with extrinsic parameters $\{\mathbf{E}_{\mathcal{S}}^k\}$. Output: A point-wise deformation field $\mathcal{D} \in \mathbb{R}^{N\times 3}$, yielding the reconstruction $\hat{\mathcal{T}} = \mathcal{S} + \mathcal{D}$, which naturally provides dense template-to-target correspondences.

The pipeline consists of three stages: (1) View-adaptive fusion—selecting the template view most semantically similar to the target (via DINOv3 cosine similarity) as the primary view, fusing other views via relative pose embeddings to obtain viewpoint-invariant visible features $\tilde{\mathbf{F}}_{\text{partial}}$; (2) Geometry-guided feature modeling—diffusing visible features to all $N$ template points via 3D geometric affinity, then aligning them with target image features using cross-attention; (3) Flow Matching deformation—predicting the deformation field in a single step conditioned on the aligned features $\mathbf{c}$.

The deformation is modeled as a continuous ODE: $d\phi_t/dt = \mathbf{v}_t(\phi_t \mid \mathbf{c})$. The velocity field $v_\theta$ is trained along a linear interpolation path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$. Inference is performed at $t=0$ in one step: $\mathcal{D} = v_\theta(\mathcal{S}, 0, \mathbf{c})$, taking approximately 0.67 s.

### Key Designs

1.  **Geometry-Guided Feature Propagation**:
    -   **Function**: Diffuses 2D foundation features $\mathbf{F}_{\text{vis}}$ present only on visible surfaces ($M$ points) to all $N$ template points (including occluded regions), resulting in $\mathbf{F}_{\text{complete}} \in \mathbb{R}^{N\times D}$.
    -   **Mechanism**: A separate 3D encoder computes geometric embeddings $\mathbf{G} \in \mathbb{R}^{N\times d}$ for the full template. Geometric affinity $S_{ji} = \mathbf{g}_j \cdot \mathbf{g}_i / (\|\mathbf{g}_j\|\|\mathbf{g}_i\|)$ between visible points $i$ and all points $j$ is calculated. Features are aggregated using softmax weighting with temperature $\tau$: $\mathbf{f}_j^{\text{complete}} = \sum_i \frac{\exp(S_{ji}/\tau)}{\sum_k \exp(S_{jk}/\tau)} \mathbf{f}_i^{\text{vis}}$. This is essentially "semantics shared by geometrically similar points"—if the back of a chair leg is occluded, it inherits semantics from the visible front of the leg.
    -   **Design Motivation**: Feeding only visible points to a deformation network treats occluded areas as "no-signal zones," making the template topology a liability. Using geometric similarity as a bridge allows 2D semantics to permeate the 3D surface, providing directional deformation guidance even for occluded parts. These template features then act as a query to retrieve target image features $\mathbf{F}_{\mathcal{T}}$ (key/value) via cross-attention, outputting $\mathbf{F}_{\text{aligned}}$ as the condition for Flow Matching.

2.  **View-Adaptive Feature Aggregation**:
    -   **Function**: Resolves "viewpoint drift" between fixed template views and arbitrary target viewpoints—identical 3D structures exhibit different foundation features from different camera angles. Without this, the model might confuse viewpoint differences with deformation.
    -   **Mechanism**: The primary view $I_{\mathcal{S}}^*$ is selected from $K=16$ template views based on the highest cosine similarity to the target $I_{\mathcal{T}}$ in the DINOv3 feature space. Relative camera transformations $\mathbf{P}_{\text{rel}}^k = (\mathbf{E}_{\mathcal{S}}^*)^{-1} \mathbf{E}_{\mathcal{S}}^k$ are flattened into $\mathbb{R}^{12}$ vectors and projected into pose embeddings $\mathbf{e}^k$. Visible features from each view are "geometrically modulated" by adding pose embeddings, followed by cross-attention: primary view features serve as query, while all view features concatenated serve as key/value, producing $\mathbf{F}_{\text{fused}}$, with a residual connection yielding $\tilde{\mathbf{F}}_{\text{partial}} = \mathbf{F}_{\text{fused}} + \tilde{\mathbf{F}}_{\text{primary}}$.
    -   **Design Motivation**: By informing the network of the camera angle for each feature, the model can explicitly decouple pose-induced feature variations from shape deformation. Ablations show that naive multi-view fusion (w/o PrimSel and w/o PoseAware) performs worse than single-view, indicating that pose-aware anchoring is critical.

3.  **Flow-Matching Deformation**:
    -   **Function**: Treats deformation learning as a continuous trajectory from template to target distribution under condition $\mathbf{c}$, rather than a one-shot offset regression.
    -   **Mechanism**: During training, linear interpolation is performed between $\mathbf{x}_0 = \mathcal{S}$ and $\mathbf{x}_1 = \mathcal{T}$. The network velocity field is supervised to align with $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$ (via Flow Matching loss $\mathcal{L}_{\text{FM}}$). During inference, the constant velocity property of the linear trajectory allows for a single-step deformation at $t=0$. This is theoretically equivalent to one-step sampling of a rectified flow.
    -   **Design Motivation**: Direct offset regression is unstable under complex, large deformations. Treating deformation as a continuous flow provides smoother geometric interpolation, making the model more resilient to topological differences.

### Loss & Training
The total loss incorporates multiple geometric regularizers: $\mathcal{L} = \lambda_{\text{FM}}\mathcal{L}_{\text{FM}} + \lambda_{\text{CD}}\mathcal{L}_{\text{CD}} + \lambda_{\text{Lap}}\mathcal{L}_{\text{Lap}} + \lambda_{\text{ARAP}}\mathcal{L}_{\text{ARAP}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}} + \lambda_{\text{sil}}\mathcal{L}_{\text{sil}}$. Chamfer Loss handles global alignment, Laplacian Loss ensures local continuity, ARAP maintains local rigidity, reg restricts deformation magnitude, and silhouette loss ensures multi-view consistency. A unified model is trained across seven ShapeNetv2 categories (chair/table/airplane/car/cabinet/bowl/bottle), sampling 500 shapes per category with 50 used as a template pool. Only one random target viewpoint is used per object to simulate self-occlusion.

## Key Experimental Results

### Main Results
Two evaluation settings: Retrieved Template (choosing the most similar template via DINOv3) vs. Random Template (inducing large deformations). ShapeMatcher and KP-RED are trained per-category; GODeform is a unified model. Our-SV uses a single-view template, and Our-MV uses full multi-view fusion.

| Dataset / Setting | Metric | Ours (Our-MV) | KP-RED | ShapeMatcher | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Seen / Retrieved | CD $(10^{-3})$ ↓ | **2.38** | 3.05 | 5.92 | 22% better than KP-RED with retrieved templates |
| Seen / Retrieved | S-IoU (%) ↑ | **48.79** | 46.73 | 40.47 | |
| Seen / Random | CD $(10^{-3})$ ↓ | **2.46** | 5.10 | 13.02 | KP-RED CD doubles with random templates; Ours is stable |
| Seen / Random | S-IoU (%) ↑ | **47.31** | 42.05 | 34.36 | |
| Unseen / Retrieved | CD $(10^{-3})$ ↓ | **3.69** | N/A | N/A | Baselines do not support cross-category |
| Unseen / Random | S-IoU (%) ↑ | **52.57** | N/A | N/A | Still maintains ~52% S-IoU on unseen categories |

The Random Template column is most significant: baselines collapse when given dissimilar templates (KP-RED CD rises from 3.05 to 5.10), while GODeform remains robust (2.38 → 2.46). This demonstrates that geometry-guided foundation features solve the problem of template sensitivity.

### Ablation Study
| Configuration | CD ($10^{-3}$, Retrieved) | CD ($10^{-3}$, Random) | S-IoU (%, Random) | Description |
| :--- | :--- | :--- | :--- | :--- |
| Our-MV (Full) | **2.38** | **2.46** | **47.31** | Full model |
| w/o FM | 2.66 | 2.74 | 43.57 | Uses direct regression; CD increases by 11% |
| w/o Prop. | 2.74 | 2.95 | 41.10 | Fills occluded points with mean; largest drop |
| w/o Rel. | 2.56 | 2.70 | 44.67 | Replaces cross-attention with FiLM global broadcast |
| w/o PrimSel. | 2.64 | 2.84 | 44.40 | Uses mean query instead of primary view |
| w/o PoseAware. | 2.60 | 2.79 | 44.47 | Naive multi-view average without pose embeddings |
| Our-SV | 2.45 | 2.61 | 46.78 | Single-view template; still outperforms baselines |

### Key Findings
-   **Propagation is Critical**: w/o Prop. shows the largest drop across all metrics (Random S-IoU 47.31 → 41.10), indicating that leaving occluded points without features forces the network to "guess" the deformation blindly.
-   **Methodical Multi-view Fusion is Necessary**: Naive fusion (w/o PrimSel and w/o PoseAware) is worse than Our-SV (single-view). Simply averaging features introduces viewpoint noise; primary view anchoring + pose modulation is required to reap the benefits of multi-view data.
-   **Downstream Utility**: Validated via dexterous grasping transfer in Isaac Gym with a Shadow Hand (0.67s deformation + 15s optimization). A real-world NAVIAI AW-1 robot achieved a 77% success rate on four object categories, demonstrating the engineering value of the resulting dense correspondences.

## Highlights & Insights
-   **Explicit Geometry-Aware Foundation Features**: Rather than implicitly fusing DINO features, the "3D geometric affinity as a propagation bridge" is elegant. It doesn't rely on 3D foundation model pre-training but uses a lightweight 3D encoder to spread 2D semantics. This recipe is applicable to 6D pose estimation, part segmentation, and affordance prediction.
-   **Pose Embeddings + Primary View Selection** avoids the naive assumption that more views are always better. The finding that unanchored multi-view fusion is detrimental is a crucial counter-intuitive insight.
-   **Practical Use of Dense Correspondence**: The authors actually use the warped contact maps from the template to new objects, bypassing the need to train separate grasping models—a fundamental advantage of the deformation paradigm over generative ones.
-   **Single-step Flow Matching**: Utilizing the linear trajectory of rectified flows allows for 0.67s inference, which stands as a highly efficient engineering alternative to iterative ODE solvers.

## Limitations & Future Work
-   **Viewpoint Limitations**: Monocular input is inherently ill-posed. Significant occlusion of crucial parts (e.g., a hidden cup handle) lacks 2D deformation cues, leading to geometric misalignment.
-   **Template Dependency**: Requires a pre-defined template pool (~50 per category), which is not directly applicable to "free-form" objects (e.g., soft bodies) without an available reference topology. Offline rendering of 16 template views adds overhead for open-world deployment.
-   **Generalization Gap**: Evaluations on "unseen categories" remain largely synthetic; real-world robot experiments focused on four common categories with regular geometries. Robustness to transparent, reflective, or extremely deformed objects remains unaddressed.
-   **Future Directions**: Extending to multi-view target inputs and incorporating VLM semantic priors. Replacing the 3D encoder with a pre-trained large model (e.g., Sonata/Point-LLM) for similarity calculation might improve stability across categories.

## Related Work & Insights
-   **vs. ShapeMatcher (CVPR'24)**: Both use template deformation, but ShapeMatcher targets specific categories and relies on retrieval. Ours uses foundation features + geometric propagation, improving CD by 81% (13.02 → 2.46) under random template selection.
-   **vs. KP-RED (2024)**: KP-RED uses sparse keypoints; Ours uses dense point-level Flow Matching, capturing better detail without requiring GT depth.
-   **vs. Phidias / LRM / Wonder3D**: Generative methods hallucinate in occluded areas; Ours uses template topology as a safeguard for category-level geometric plausibility, albeit at the cost of requiring a template pool.
-   **vs. FreeZe / VFM-pose**: While these use 2D foundation features for 3D tasks like pose estimation, Ours extends this to the more difficult dense prediction task of deformation, where geometric propagation is the key differentiator.

## Rating
-   Novelty: ⭐⭐⭐⭐ — The combination of Flow Matching, geometric propagation, and view fusion is new, though individual components are established.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Complete four-quadrant comparison (Seen/Unseen, Retrieved/Random) + 6 ablation variants + real-world robot validation.
-   Writing Quality: ⭐⭐⭐⭐ — Equations and figures are well-organized, though some details on ARAP/silhouette losses are relegated to the appendix.
-   Value: ⭐⭐⭐⭐⭐ — Addresses the major weaknesses of the deformation paradigm (cross-category and template discrepancy) and provides direct utility for robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](../../NeurIPS2025/3d_vision/learning_generalizable_shape_completion_with_sim3_equivariance.md)
- [\[ICML 2026\] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)
- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](../../ICCV2025/3d_vision/neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](../../CVPR2026/3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)

</div>

<!-- RELATED:END -->
