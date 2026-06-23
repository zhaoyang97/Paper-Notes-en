---
title: >-
  [Paper Note] Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation
description: >-
  [ICLR 2026][3D Vision][novel view synthesis] The authors reformulate multi-view novel view synthesis as a dual-branch diffusion inpainting task of "image + geometry." By utilizing **MoAI (cross-Modal Attention Instillation)** to inject attention maps from the image branch into the geometry branch, the method generates aligned novel view images and point clouds di
tags:
  - ICLR 2026
  - 3D Vision
  - novel view synthesis
  - geometry completion
  - diffusion
  - cross-modal attention
  - warping-and-inpainting
date: 2026-05-08
content_hash: 8253cf8a34e4aa6a
---
# Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vjvwYexMQn](https://openreview.net/forum?id=vjvwYexMQn)  
**Code**: [https://cvlab-kaist.github.io/MoAI/](https://cvlab-kaist.github.io/MoAI/)  
**Area**: 3D Vision / Novel View Synthesis / Diffusion Models  
**Keywords**: novel view synthesis, geometry completion, diffusion, cross-modal attention, warping-and-inpainting  

## TL;DR
The authors reformulate multi-view novel view synthesis as a dual-branch diffusion inpainting task of "image + geometry." By utilizing **MoAI (cross-Modal Attention Instillation)** to inject attention maps from the image branch into the geometry branch, the method generates aligned novel view images and point clouds directly from pose-free reference images, achieving SOTA performance in camera extrapolation settings.

## Background & Motivation
- **Background**: Novel view synthesis (NVS) follows two main paradigms. Feed-forward methods (PixelSplat, MVSplat, DUSt3R, NoPoSplat) directly predict 3D from sparse views; they essentially "fill in visible regions" of reference views, providing **high fidelity in interpolation settings**. Generative diffusion methods (Zero123, CAT3D, ViewCrafter) possess strong extrapolation capabilities but rely on known camera poses and target poses as feature embeddings during training.
- **Limitations of Prior Work**: Feed-forward methods **lack extrapolation capabilities** and cannot synthesize occluded or unobserved regions. Generative methods tend to collapse when target poses fall outside the training distribution and **require reference camera poses**, operating only in posed settings. Warping-and-inpainting approaches (LucidDreamer, GenWarp) bypass pose constraints but only perform inpainting at the 2D image level, lacking 3D structural understanding, which leads to **severe degradation under large view changes** and scale-shift misalignment between predicted depth and reference geometry.
- **Key Challenge**: Achieving **extrapolation of unseen regions** like generative methods, **accurate geometric alignment** like feed-forward methods, and independence from known poses simultaneously is difficult.
- **Goal**: Jointly generate the novel view image $I_t$ and point cloud $P_t$ for any target view from one or more **pose-free** reference images, ensuring strict geometric alignment without additional NeRF/3DGS optimization.
- **Core Idea**: **[Geometry as Inpainting]** Use off-the-shelf geometry predictors to project partial geometry from reference views to the target view, then treat both image and geometry completion as diffusion inpainting tasks. **[Cross-Modal Attention Instillation]** Recognizing that geometry completion is more deterministic with stronger structural constraints than image generation, the method "instills" the semantic correspondence attention maps learned by the image branch into the geometry branch to enable mutual regularization and collaborative alignment.

## Method

### Overall Architecture
Given $N$ pose-free sparse reference images $\{I_n\}_{n=1}^N$, off-the-shelf geometry models (VGGT) first predict point maps and camera poses for each view, which are aggregated into a point cloud and projected to the target view $\pi_t$ to obtain partial projected point maps. The framework consists of **two parallel dual-branch U-Nets**: an image branch (reference network for semantic features + denoising network for image inpainting) and a geometry branch (isomorphic structure with a denoising network for point map completion). Both branches share the same correspondence conditions $c_t, c_r$. The key coupling occurs at the attention layers, where attention maps from the image branch are instilled into the geometry branch.

```mermaid
flowchart LR
    R[Pose-free Reference I_1..I_N] --> G[VGGT Predicts Point Map + Pose]
    G --> AGG[Aggregate Point Cloud P]
    AGG --> PROJ[Project to Target View π_t]
    PROJ --> MESH[Proximity-based Mesh Cond.<br/>Depth/Normal/Mask]
    MESH --> IMG[Image Denoising U-Net]
    MESH --> GEO[Geometry Denoising U-Net]
    IMG -- Attention Map Q^I,K^I Instillation --> GEO
    IMG --> OI[Novel View Image I_t]
    GEO --> OG[Aligned Point Map P_t]
    OI --> PC[Aligned Colored Point Cloud / 3D Completion]
    OG --> PC
```

### Key Designs

**1. Geometry-Completion NVS: Point Map Projection as Inpainting Condition to Avoid Scale-Shift**  
Unlike traditional NVS treated as pure 2D image inpainting, this work uses off-the-shelf models to predict point maps $P_n$ (where each pixel corresponds to a 3D coordinate in world space) for each reference image. These are merged into a point cloud and projected to the target view: $P_t^\Pi = \Pi(P, \pi_t), P = \bigcup_{n=1}^N P_n$. When multiple points project to the same pixel, the nearest point is selected via point cloud rasterization. The projected point map, processed through Fourier positional encoding $E(\cdot)$ and a binary mask $M_t$ (marking regions without projected points), forms the target correspondence condition $c_t = [E(P_t^\Pi), M_t]$. Reference views use an all-one mask $c_r^n = [E(P_n), \mathbf{1}]$ as they are densely predicted. These conditions are **added** to the image latent variables in the first layer of the denoising network. Crucially, the authors **do not provide explicit pixel-to-pixel correspondences** (like warped coordinates in GenWarp) but instead feed the embedded point maps, allowing the model to naturally associate spatial positions in the target with potential correspondences across multiple reference images. The geometry branch uses a similar architecture and conditions, fine-tuned from a Marigold normal prediction model, to complete the point map. Since geometry is generated as a "continuation of reference geometry" rather than independent depth prediction, it naturally avoids scale-shift misalignment between predicted depth and known reference geometry.

**2. Aggregated Attention: Simultaneous Cross-Reference and Self-Attention**  
The spatial self-attention layers of the image denoising network produce target view key/value features $K_t^I, V_t^I$, which are concatenated with $N$ reference features: $K^I = [K_t^I, K_1^I, \dots, K_N^I]$, $V^I = [V_t^I, V_1^I, \dots, V_N^I]$. The target query $Q^I_t$ performs attention: $\mathrm{Attention}(Q^I, K^I, V^I) = \mathrm{Softmax}\!\left(\frac{Q^I K^{I\top}}{\sqrt{d_k}}\right) V^I$. This single attention operation performs both cross-attention across all reference views and self-attention within the target latent, enabling unified multi-view novel view synthesis. Due to this aggregation mechanism, the model can process an **arbitrary number** of reference views during inference (even if trained with only 2 views).

**3. Cross-modal Attention Instillation (MoAI): Replacing Geometry Attention Maps with Image Ones**  
The authors observe an asymmetry (Fig. 3): geometry completion is more deterministic and structurally constrained. When completing partially visible structures (e.g., wheels), the geometry branch correctly attends to similar structures, whereas the image branch might fail to establish such correspondences. Conversely, the geometry branch lacks semantic cues, leading to diffused attention and missing fine-grained cross-view correspondences. MoAI addresses this by **replacing** the query/key in the geometry branch's attention layer with those from the image branch, $Q^I, K^I$, while retaining the geometry's own value $V^P$: $\mathrm{Attention}(Q^I, K^I, V^P) = \mathrm{softmax}\!\left(\frac{Q^I K^{I\top}}{\sqrt{d_k}}\right) V^P$. This creates a bidirectional synergy: the image branch receives regularized training signals from the more deterministic geometry completion task, leading to more consistent generation, while the geometry branch gains more accurate completion through the rich semantic cues of the image branch. Since attention maps only serve as structural cues for aggregating values and do not directly mix cross-modal features, harmful feature aliasing seen in previous works is avoided. This instillation is performed during both training and inference.

**4. Proximity-based Mesh Conditioning: Filtering Erroneous Projections**  
Sparse point clouds from off-the-shelf models contain noise, and projection errors increase as the target view deviates from the reference. The authors use the ball-pivoting algorithm to convert sparse point clouds into meshes, obtaining denser projected point maps $X_t^\Pi$ with fewer errors compared to raw points. The mesh depth map $D_t^\Pi$ and normal map $N_t^\Pi$ are concatenated into the condition: $c_t = [E(X_t^\Pi), D_t^\Pi, N_t^\Pi, M_t]$. Additionally, a **normal mask** is applied to remove mesh faces where the normal deviates more than 90° from the target view direction (typically erroneous projections caused by incomplete geometry), preventing noisy correspondences from polluting the generation.

## Key Experimental Results

Implementation: Image branch initialized from Stable Diffusion 2.1; geometry branch fine-tuned from Marigold normal prediction. Trained on RealEstate10K, Co3D, and MVImgNet using VGGT for pseudo-ground truth geometry.

### Main Results: DTU Zero-shot (Comparison with Feed-forward / Warping methods)

| View | Method | Pose-free | Extrap. PSNR↑ | Extrap. SSIM↑ | Extrap. LPIPS↓ | Interp. PSNR↑ | Interp. LPIPS↓ |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 2-view | PixelSplat | ✗ | 14.66 | 0.517 | 0.334 | 12.75 | 0.637 |
| 2-view | MVSplat | ✗ | 12.22 | 0.416 | 0.423 | 13.94 | 0.385 |
| 2-view | NoPoSplat | ✓ | 13.58 | 0.393 | 0.545 | 14.04 | 0.530 |
| 2-view | **Ours** | ✓ | **15.58** | **0.615** | **0.184** | **16.58** | **0.152** |
| 1-view | LucidDreamer | ✓ | 11.14 | 0.423 | 0.440 | 12.09 | 0.419 |
| 1-view | GenWarp | ✓ | 9.85 | 0.315 | 0.527 | 9.54 | 0.538 |
| 1-view | **Ours** | ✓ | **15.56** | **0.609** | **0.184** | **14.58** | **0.202** |

Ours significantly leads in both extrapolation and interpolation settings and works robustly with a single view. In RealEstate10K in-domain tests, extrapolation PSNR reached 17.41 (vs. NoPoSplat 14.36), while remaining competitive in interpolation (PSNR 24.23).

### Ablation Study (RealEstate10K, Extrapolation Setting)

| Component | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| (a) Baseline (No geo. condition) | 16.55 | 0.559 | 0.260 |
| (b) + Point map conditioning | 16.93 | 0.594 | 0.243 |
| (c) + Proximity-based mesh cond. | 17.01 | 0.601 | 0.238 |
| (d) + Cross-modal attention instillation (MoAI) | **17.41** | **0.614** | **0.229** |

Each component provides incremental gains, with MoAI delivering the final performance leap.

### Key Findings
- **Extrapolation is the Killer App**: Compared to LVSM / ZeroNVS / ViewCrafter, this method provides the best extrapolation quality for large unobserved regions with an inference time of only 9.67s (compared to 209s for 25 frames in ViewCrafter, or 2+ hours for SDS distillation in ZeroNVS).
- **More Views are Better**: Although trained with only 2 views, providing 3 views during inference increased image PSNR from 17.41 to 20.02 and improved geometry accuracy, validating the generalization of aggregated attention.
- **Geometry Alignment without Fitting**: Multi-view point maps are naturally aligned without requiring scale-and-shift fitting because depth is reformulated as "continuation and completion of reference geometry."

## Highlights & Insights
- **Elegant Task Reformulation**: By unifying "novel view synthesis" and "novel view geometry synthesis" into a single inpainting framework with shared conditions and architectures, the structure remains clean while coupling only at the attention layers.
- **MoAI Captures Modal Asymmetry**: Geometry has high determinism but lacks semantics; images have rich semantics but weak correspondence. Choosing to "borrow the attention map instead of mixing features" exploits their strengths without introducing cross-modal noise—the core insight of this work.
- **Comprehensive Pose-free + Extrapolation**: The method simultaneously removes dependence on known poses and possesses generative extrapolation capabilities, filling the gap between feed-forward and generative paradigms.

## Limitations & Future Work
- Heavily dependent on the quality of off-the-shelf geometry predictors (VGGT/Marigold); failures in textureless or reflective scenes directly affect generation.
- Proximity-based mesh conditioning relies on heuristics like ball-pivoting and normal thresholds, with limited robustness for highly sparse or incomplete point clouds.
- Evaluation focuses on object-centric or indoor scenes (Co3D, DTU, RealEstate10K); large-scale open scenes and dynamic scenes remain unvalidated.
- While faster than optimization-based methods, dual-branch diffusion is still an order of magnitude slower than pure feed-forward methods.

## Related Work & Insights
- **Warping-and-inpainting** (LucidDreamer, GenWarp): The direct predecessors, but this work extends them from single-image 2D inpainting to multi-view geometry-aware completion, solving scale-shift and structural issues.
- **Pose-free Geometry Prediction** (DUSt3R, MASt3R, NoPoSplat, VGGT): Used as off-the-shelf geometry backends, this work proves that feeding their "partial geometry" into a generative completion model unlocks extrapolation capabilities.
- **Generative NVS** (Zero123, CAT3D, ViewCrafter): Provides the foundation for spatial cross-attention consistency, but this work bypasses training domain pose-embedding limits via aggregated attention and inpainting.
- **Inspiration**: When a task involves two complementary modalities (one deterministic, one semantically rich), "sharing attention maps rather than features" is a high-synergy, low-coupling multi-task alignment paradigm transferable to other generative scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of MoAI and geometry-completion inpainting is a distinct approach targeting the gap between feed-forward and generative methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers DTU zero-shot, RealEstate10K in-domain, comparisons with large models, ablations, and view count analysis across both settings; lacks open/dynamic scenes.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is logically presented, particularly the modal asymmetry analysis in Fig. 3.
- **Value**: ⭐⭐⭐⭐ — Simultaneous output of aligned images and geometry without poses, strong extrapolation, and relatively fast inference offer high practical value for NVS and 3D completion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention](efficient-lvsm_faster_cheaper_and_better_large_view_synthesis_model_via_decouple.md)
- [\[ICLR 2026\] True Self-Supervised Novel View Synthesis is Transferable](true_self-supervised_novel_view_synthesis_is_transferable.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](../../CVPR2026/3d_vision/pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](../../CVPR2026/3d_vision/affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](dynamic_novel_view_synthesis_in_high_dynamic_range.md)

</div>

<!-- RELATED:END -->
