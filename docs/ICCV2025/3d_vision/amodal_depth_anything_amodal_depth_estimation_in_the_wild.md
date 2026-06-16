---
title: >-
  [Paper Note] Amodal Depth Anything: Amodal Depth Estimation in the Wild
description: >-
  [ICCV 2025][3D Vision][amodal depth estimation] This paper proposes a new paradigm for amodal relative depth estimation, constructs a large-scale real-world dataset ADIW (564K samples)…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "amodal depth estimation"
  - "relative depth"
  - "occlusion-aware perception"
  - "Depth Anything V2"
  - "conditional flow matching"
date: 2026-05-08
content_hash: 917554f58998d1a3
---

# Amodal Depth Anything: Amodal Depth Estimation in the Wild

**Conference**: ICCV 2025
**arXiv**: [2412.02336](https://arxiv.org/abs/2412.02336)  
**Code**: [https://github.com/zhyever/Amodal-Depth-Anything](https://github.com/zhyever/Amodal-Depth-Anything)  
**Area**: 3D Vision / Depth Estimation / Amodal Perception
**Keywords**: amodal depth estimation, relative depth, occlusion-aware perception, Depth Anything V2, conditional flow matching

## TL;DR
This paper proposes a new paradigm for amodal relative depth estimation, constructs a large-scale real-world dataset ADIW (564K samples), and designs two complementary frameworks (Amodal-DAV2 and Amodal-DepthFM) built upon Depth Anything V2 and DepthFM. By minimally modifying pretrained models, the method achieves depth prediction in occluded regions, improving RMSE by 27.4% over the previous SOTA on ADIW.

## Background & Motivation
- **Amodal depth estimation** aims to predict the depth of invisible, occluded parts of objects in a scene — an emerging and challenging task.
- Prior methods (AmodalSynthDrive, Amodal-3D-FRONT) rely on **synthetic datasets** and focus on **metric depth**, suffering from severe domain shift and poor zero-shot generalization.
- Constructing synthetic datasets is costly (requiring manual placement of occluders one by one) and difficult to scale.
- In real-world scenes, **no sensor can capture ground-truth depth in occluded regions**, making data acquisition the core bottleneck.
- Recent models such as Depth Anything have demonstrated strong generalization for **relative depth** estimation, offering a new solution for amodal depth.

## Core Problem
1. **How to construct large-scale training data without ground-truth annotations for occluded depth?**
2. **How to leverage the strong priors of pretrained depth models to predict depth in occluded regions?**
3. **Metric depth vs. relative depth: which paradigm better supports generalization in amodal depth estimation?**

## Method

### Overall Architecture
Input: observation image $I_o$, observed depth map $D_o$, target amodal mask $M_a$ → Output: amodal depth map including depth for occluded regions. Two complementary frameworks are proposed: the deterministic Amodal-DAV2 and the generative Amodal-DepthFM.

### Key Designs
1. **ADIW Dataset Construction Pipeline**:

    - SAM is applied to SA-1B to automatically generate segmentation masks; a heuristic algorithm [pix2gestalt] is used to filter for complete objects.
    - A synthesis strategy is adopted: foreground objects are composited onto background images to form observation images $I_o$.
    - Depth Anything V2 (ViT-G) is applied separately to $I_o$ and the background image $I_b$ to generate depth maps.
    - **Scale-and-Shift Alignment**: since foreground objects alter the relative depth of the background, least-squares estimation is used to compute scale factor $s$ and shift factor $t$ to align the two depth maps, ensuring label consistency during training.
    - A total of **564K** training samples are generated.

2. **Amodal-DAV2 (Deterministic Model)**:

    - A **Guidance Conv layer** is added in parallel to the RGB Conv layer in DAV2's ViT encoder, accepting $D_o$ and $M_a$ as additional guidance channels.
    - Guidance Conv weights are **zero-initialized**, ensuring the model initially ignores the additional inputs and learns to leverage them gradually.
    - **LayerNorm** is applied to the input features of the DPT Head for training stability.
    - The entire model is fine-tuned end-to-end with minimal architectural modifications.

3. **Amodal-DepthFM (Generative Model)**:

    - Built upon DepthFM's conditional flow matching framework.
    - The first Conv layer of the UNet is modified to accept additional guidance channels ($D_o$ and $M_a$).
    - The first 8 channels use pretrained weights; the extra 2 channels are zero-initialized.
    - During inference, **Scale-and-Shift alignment** is applied to align predicted depth with observed depth over visible shared regions, improving consistency.
    - The generative nature allows the model to produce **multiple plausible occluded depth structures**.

### Loss & Training
- **Amodal-DAV2**: Scale-Invariant Log (SILog) Loss with $\lambda=0.85$, supervised over the **entire object** (visible + invisible regions) rather than only the invisible part, which helps the model understand overall scene structure.
- **Amodal-DepthFM**: Conditional flow matching objective $\min_\theta \mathbb{E}_{t,z,p(x_0)} \|v_\theta(t, \phi_t(x_0)) - (x_1 - x_0)\|$, with Gaussian noise augmentation during training.
- Training hyperparameters:
    - Amodal-DAV2: batch 32, lr 1e-5, 50K iterations
    - Amodal-DepthFM: batch 128, lr 3e-5, 15K iterations
    - Adam optimizer, exponential learning rate decay, gradient clipping 0.01
    - 4× A100 GPUs

## Key Experimental Results

| Dataset | Metric | Ours (Best) | Prev. SOTA | Gain |
|--------|------|----------|----------|------|
| ADIW (Overall) | RMSE↓ | **3.682** (Amodal-DAV2-L) | 5.114 (pix2gestalt‡) | 27.4% |
| ADIW (Overall) | δ(%)↑ | **93.251** | 88.717 (pix2gestalt‡) | +4.5pp |
| ADIW (Easy) | RMSE↓ | Best in row | 5.067 | - |
| ADIW (Hard) | RMSE↓ | Best in row | 5.641 | - |

**Overall comparison of methods on ADIW:**

| Method | RMSE↓ | δ(%)↑ |
|------|-------|-------|
| Jo et al.† | 10.260 | 56.118 |
| Sekkat et al.† | 11.264 | 49.222 |
| Sekkat et al.†‡ | 5.194 | 88.367 |
| pix2gestalt‡ | 5.114 | 88.717 |
| Amodal-DepthFM (Full) | 4.645 | 92.295 |
| **Amodal-DAV2-L (Full)** | **3.682** | **93.251** |

### Ablation Study
**Amodal-DAV2 ablations:**
- Without guidance signals: RMSE 7.549 → 3.682 (guidance is essential)
- Without mask guidance: RMSE 4.369 → 3.682 (mask provides significant benefit)
- Supervising only invisible regions: RMSE 3.845 vs. full-object supervision 3.682 (full-object supervision is superior)
- Applying alignment at inference degrades performance: RMSE 4.015 (the deterministic model already learns consistent depth)

**Amodal-DepthFM ablations:**
- Scale-and-Shift alignment is **critical** for DepthFM: RMSE 5.410 → 4.645 (generative model outputs are insufficiently consistent and require alignment)
- Scene-level supervision (⑥) achieves RMSE 4.608, close to object-level (④) 4.645, but with worse log10 metric

## Highlights & Insights
1. **Paradigm innovation**: shifting amodal depth estimation from metric to relative depth substantially improves generalization.
2. **Scalable data construction**: the pipeline based on SA-1B and a synthesis strategy requires no manual annotation and can automatically generate 564K samples.
3. **Minimal modification of pretrained models**: the zero-initialized Guidance Conv design is elegant — it preserves pretrained knowledge while introducing additional conditioning.
4. **Two complementary paradigms**: Amodal-DAV2 achieves higher accuracy; Amodal-DepthFM captures finer details and supports diverse predictions.
5. **No reliance on RGB priors**: directly predicting occluded depth avoids the cascaded errors inherent in inpainting-based approaches.

## Limitations & Future Work
1. **Dependence on amodal mask quality**: inaccurate or ambiguous masks lead to cascading errors.
2. **Slight degradation in detail capture after fine-tuning**: possibly due to limited diversity in the SA-1B dataset.
3. **Single-frame only**: the method could be extended to temporal amodal depth estimation in video settings.
4. **Single-task formulation**: future work could build a unified framework jointly predicting amodal segmentation, RGB, depth, normals, etc.
5. **Data source limited to the SAM dataset**: incorporating more high-quality datasets with complex objects could further improve performance.

## Related Work & Insights
- **vs. Sekkat et al. / Jo et al.**: Prior methods rely on synthetic data and metric depth, yielding poor generalization; this paper uses real images and relative depth, reducing RMSE by over 50%.
- **vs. Invisible Stitch / pix2gestalt**: Inpainting-based methods depend on RGB reconstruction quality and suffer from cascading errors (inpaint first, then estimate depth); this paper directly regresses from observed depth and mask, which is more robust.
- **vs. Depth Anything V2**: DAV2 only estimates depth for visible pixels; this paper extends it to occluded regions, representing a task-level complement.
- **vs. DepthFM**: This paper retains DepthFM's generative capability (diverse predictions) with improved detail quality, though accuracy remains below the deterministic approach.

## Highlights & Insights
- **Zero-initialized guidance layer design**: the practice of adding zero-initialized side channels to a pretrained model (analogous to the ControlNet paradigm) constitutes a general design pattern transferable to other foundation model adaptation scenarios requiring additional conditional inputs.

## Rating
- Novelty: ⭐⭐⭐⭐ — The problem paradigm (replacing metric depth with relative depth) and data construction pipeline are innovative, though the model-level modifications are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations comprehensively cover guidance signals, supervision strategies, and alignment strategies, but comparisons with more depth foundation models and ablations on data scale are lacking.
- Writing Quality: ⭐⭐⭐⭐ — The paper is well-structured with rich figures and detailed method descriptions, though some table layouts are slightly cluttered.
- Value: ⭐⭐⭐⭐ — Opens a new direction for amodal depth estimation with a reusable data pipeline; has practical applications in 3D reconstruction and scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)
- [\[ICCV 2025\] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [\[ICCV 2025\] RePoseD: Efficient Relative Pose Estimation with Known Depth Information](reposed_efficient_relative_pose_estimation_with_known_depth_information.md)
- [\[ICCV 2025\] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting](contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)

</div>

<!-- RELATED:END -->
