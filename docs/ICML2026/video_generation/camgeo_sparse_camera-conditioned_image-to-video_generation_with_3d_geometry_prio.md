---
title: >-
  [Paper Note] CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior
description: >-
  [ICML 2026][Video Generation][Image-to-Video Generation] CamGeo distills 3D geometric knowledge from a pre-trained 3D video model (VGGT) through **training-only distillation**. By providing supervision signals only durin…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Image-to-Video Generation"
  - "Sparse Camera Conditioning"
  - "3D Geometry Prior"
  - "Training-only Distillation"
date: 2026-05-08
content_hash: d78a80bdf3cd86e1
---

# CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior

**Conference**: ICML 2026  
**arXiv**: [2605.30895](https://arxiv.org/abs/2605.30895)  
**Code**: To be confirmed  
**Area**: Video Generation / 3D Vision / Knowledge Distillation  
**Keywords**: Image-to-Video Generation, Sparse Camera Conditioning, 3D Geometry Prior, Training-only Distillation

## TL;DR
CamGeo distills 3D geometric knowledge from a pre-trained 3D video model (VGGT) through **training-only distillation**. By providing supervision signals only during the training phase, the diffusion model can generate high-quality videos that are geometrically consistent and motion-smooth under **sparse camera input** conditions. During inference, VGGT is completely removed to maintain efficiency.

## Background & Motivation

**Background**: Controllable image-to-video generation under camera conditions has become an important research direction. Existing methods (CameraCtrl, CamI2V, CPA, etc.) have achieved good results in video generation and camera alignment but rely on **dense per-frame camera pose annotations**.

**Limitations of Prior Work**: In practice, obtaining dense camera pose annotations is extremely difficult—traditional 3D reconstruction pipelines (such as COLMAP) are prone to producing temporally inconsistent poses when handling fast motion or complex non-rigid dynamics. Can a model be trained to work directly under **sparse camera conditions**?

**Key Challenge**: Direct simple interpolation from sparse inputs faces two fundamental problems. First, the model is prone to **pose drift** at frames lacking explicit constraints, producing content that violates physical laws. Second, rigid mathematical interpolation (SLERP) cannot capture the non-linear dynamics of real camera motion (e.g., hand shake), leading to stiff and incoherent generated motion. The root cause is that the model is forced to "hallucinate" 3D geometry while **lacking feedback**.

**Goal**: Achieve high-quality, geometrically consistent image-to-video generation under sparse camera conditions.

**Key Insight**: **Distill geometric priors** from an existing powerful 3D understanding model (VGGT) into a diffusion model.

**Core Idea**: **Training-only distillation**—utilize VGGT to provide supervision only during the training phase and remove it completely during inference, gaining the benefits of geometric constraints while maintaining operational efficiency.

## Method

### Overall Architecture
The system is built upon a pre-trained text-guided image-to-video diffusion model. Given a reference image, a text prompt, and **sparse camera poses** (provided only at a few keyframes), the model needs to synthesize a high-fidelity video $V = \{I_f\}_{f=1}^F$, where the sparse set $\mathcal{S} \subset \{1, \ldots, F\}$ satisfies $|\mathcal{S}| \ll F$. During training, a frozen VGGT teacher processes the generated video prediction $\hat{V}$, extracts **dense camera trajectories** $\hat{C}$ and depth maps $\hat{D}$, and provides multi-level geometric supervision to the student through two distillation mechanisms. VGGT is entirely removed during inference.

### Key Designs

1.  **Keyframe Trajectory Distillation**:

    - **Function**: Enforce cycle consistency constraints with ground-truth camera poses at sparse keyframes.
    - **Mechanism**: For each annotated frame $s \in \mathcal{S}$, the difference between the camera parameters $(\hat{R}_s, \hat{T}_s, \hat{K}_s)$ estimated by VGGT and the ground truth values is calculated. Rotation is represented by quaternions (avoiding the singularity of matrix parameterization), using an L1-norm distillation loss $\mathcal{L}_{\text{traj}} = \sum_{s \in \mathcal{S}}(\|\phi(\hat{R}_s) - \phi(R_s)\|_1 + \|\hat{T}_s - T_s\|_1 + \|\hat{K}_s - K_s\|_1)$, where $\phi(\cdot)$ maps rotation matrices to quaternions.
    - **Design Motivation**: Establish a **self-supervised closed-loop**—ensuring the generated video strictly aligns with user input at frames with explicit conditions to prevent catastrophic pose drift; the L1-norm provides a robust optimization landscape to reduce the impact of estimation errors from the teacher model.

2.  **Cross-frame Consistency Distillation**:

    - **Function**: Propagate geometric consistency through intermediate unsupervised frames using camera trajectories and depth constraints.
    - **Mechanism**: A **geometry-aware warping mechanism** is adopted for unannotated frames—projecting the depth of frame $f$ onto reference frame $f + k$, using relative poses for perspective transformation, and applying scale-invariant depth transformation (handling the inherent ambiguity of monocular depth). The loss simultaneously constrains geometric depth consistency and trajectory smoothness: 
    $$\mathcal{L}_{\text{geo}} = \sum_{f, k} \lambda^{(k)} w_{f, f+k}(\|\hat{D}_{f+k} - \mathcal{W}(\hat{D}_f, \Delta\hat{E}_{f, f+k}, \hat{K})\|_1 + \|\Delta(\hat{C}_{f+k}, \hat{C}_f)\|_1)$$ 
    where dynamic weights $w_{f, f+k} = \exp(\gamma \cdot k) \cdot \exp(-\eta \|\nabla \hat{I}_f\|_1)$ balance long-range anchoring and content adaptability.
    - **Design Motivation**: The key innovation lies in the **span selector** $\lambda^{(k)}$ and **dynamic weighting**—long-range constraints prioritize larger time intervals to propagate anchors and prevent trajectory drift; the content-adaptive term reduces penalties in high-gradient/occlusion regions to mitigate warping artifacts.

3.  **Coarse-to-Fine Curriculum Learning**:

    - **Function**: Gradually introduce geometric constraints through a three-stage curriculum to ensure stable convergence from global structure to details.
    - **Mechanism**: **Phase 1 (Warm-up)** disables distillation loss, with the model learning basic visual coherence and temporal continuity using only standard diffusion loss. **Phase 2 (Coarse-grained)** activates trajectory distillation to enforce global structure following camera motion constraints. **Phase 3 (Fine-grained)** gradually introduces depth-based warping consistency loss, using a smooth sigmoid schedule for $\alpha$ and $\beta$ to control the activation timing of geometric loss and the transition progress from trajectory to depth, respectively.
    - **Design Motivation**: Forcing geometric constraints early can cause optimization instability (as the teacher model produces unreliable estimates on low-quality inputs); curriculum learning aligns with the generative nature of diffusion models.

### Loss & Training
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \alpha \cdot [(1 - \beta) \mathcal{L}_{\text{traj}} + \mathcal{L}_{\text{geo}}]$. The key innovation is **training-only distillation**—the VGGT teacher and auxiliary losses are used only during the training phase and are completely removed during inference.

## Key Experimental Results

### Main Results (RealEstate10K)

| Sparse Ratio | Method | Architecture | RotError ↓ | TransError ↓ | CamMC ↓ | FVD-StyleGAN ↓ | FVD-VideoGPT ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1/2 | SVD-Full | U-Net | 1.46 | 6.26 | 6.83 | 122.5 | 131.9 |
| 1/2 | **SVD-CamGeo** | U-Net | **1.34** | **4.89** | **5.49** | **95.9** | **111.0** |
| 1/2 | CogVideoX-Full | DiT | 1.39 | 5.12 | 5.76 | 94.6 | 102.8 |
| 1/2 | **CogVideoX-CamGeo** | DiT | **1.27** | **4.72** | **5.38** | **83.4** | **97.6** |
| 1/4 | SVD-Full | U-Net | 1.55 | 5.82 | 6.47 | 108.8 | 125.9 |
| 1/4 | **SVD-CamGeo** | U-Net | **1.38** | **4.57** | **5.23** | **94.3** | **106.1** |

Linear interpolation methods perform even worse than direct inference from sparse inputs—rigid geometric interpolation conflicts with learned diffusion priors.

### Ablation Study

| Component | Configuration | RotError ↓ | CamMC ↓ | Background |
| :--- | :--- | :--- | :--- | :--- |
| Cross-frame Smoothness | w/o Smoothness | 1.45 | 5.71 | 1/2 Sparse |
| | Ours | **1.34** | **5.49** | |
| Warm-up | w/o Warm-up | 1.48 | 5.83 | 1/3 Sparse |
| | Ours | **1.35** | **5.40** | |
| Curriculum Scheduling | Linear | 1.33 | 5.53 | 1/2 Sparse |
| | Ours (Sigmoid) | **1.27** | **5.38** | |

### Key Findings
- The cross-frame smoothing mechanism is necessary—removing it significantly drops all camera metrics.
- Warm-up plays a stabilizing role—lack of it leads to overall deterioration.
- User study verification (73 participants × 50 comparison groups) shows a 71.2% preference rate for CamGeo.
- Architecture-agnostic—consistent improvement on both U-Net and DiT.

## Highlights & Insights
- **Innovative Training-only Distillation**: Breaks the conventional wisdom that "using a teacher model requires bearing inference costs." By borrowing VGGT only during training for geometric supervision and removing it at inference, it achieves zero overhead—a widely applicable paradigm.
- **Deep Insight into Rigid Interpolation**: Reveals the counter-intuitive phenomenon where linear interpolation of camera trajectories is worse than sparse condition inference because rigid geometric constraints conflict with the natural motion priors learned by the model.
- **Combining Course-to-Fine Curriculum with Diffusion Nature**: Progressive optimization provides an elegant solution to multi-objective problems.
- **Weight Design for Geometry-Aware Warping**: Dynamic weights balance long-range anchoring (preventing drift) and content adaptability (mitigating artifacts), finding a clever balance between constraints and visual quality.

## Limitations & Future Work
- Estimation errors from VGGT as a teacher model propagate to the student, potentially producing inaccurate depth and trajectory estimates, especially in complex scenes.
- Model extrapolation capability still has a limit when the sparse ratio is extremely low.
- Performance depends on initial reference image quality and text prompt clarity.
- Future work: Explore lighter geometric teacher models or hierarchical distillation to accelerate training; study model sensitivity to keyframe positions; extend to more complex geometric transformations (non-rigid motion).

## Related Work & Insights
- **vs CameraCtrl / CamI2V**: These rely on dense camera supervision or simple interpolation, and performance drops significantly in sparse settings. Ours trains directly under sparse conditions via geometric prior distillation.
- **vs SparseCtrl**: Handles sparse structural cues (sketches, depth) but lacks explicit camera control. Ours is the first to systematically solve the I2V problem under sparse camera conditions.
- **vs Other Distillation Methods**: Standard knowledge distillation is often used for model compression or accuracy enhancement. Ours pioneers the "training-only distillation" paradigm—the teacher provides signals only during training and is removed during inference, which can be widely applied to scenarios requiring external knowledge without inference overhead.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of training-only distillation and coarse-to-fine curriculum in 3D conditional generation is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Main dataset + 3 out-of-distribution datasets + two architectures + three sparse ratios + detailed ablation + user study.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, precise problem statement, and detailed methodological explanation.
- Value: ⭐⭐⭐⭐⭐ Solving sparse camera-conditioned I2V is a common practical requirement; the training-only distillation paradigm has broad transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](../../CVPR2026/video_generation/geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[ICML 2026\] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](../../ICCV2025/video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)

</div>

<!-- RELATED:END -->
