---
title: >-
  [Paper Note] PoseCrafter: Extreme Pose Estimation with Hybrid Video Synthesis
description: >-
  [NeurIPS 2025][Video Generation][Extreme Pose Estimation] This paper proposes PoseCrafter, a training-free framework for extreme pose estimation. It synthesizes high-fidelity intermediate frames via Hybrid Video Generati…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Extreme Pose Estimation"
  - "Video Diffusion"
  - "Hybrid Video Generation"
  - "Feature Matching Selection"
  - "Sparse Overlap"
date: 2026-05-08
content_hash: bd20da943e9fe965
---

# PoseCrafter: Extreme Pose Estimation with Hybrid Video Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2510.19527](https://arxiv.org/abs/2510.19527)
**Code**: [https://github.com/maoqingsunny/PoseCrafter](https://github.com/maoqingsunny/PoseCrafter)
**Area**: Video Generation
**Keywords**: Extreme Pose Estimation, Video Diffusion, Hybrid Video Generation, Feature Matching Selection, Sparse Overlap

## TL;DR
This paper proposes PoseCrafter, a training-free framework for extreme pose estimation. It synthesizes high-fidelity intermediate frames via Hybrid Video Generation (HVG, a two-stage pipeline combining DynamiCrafter and ViewCrafter) to address pose estimation for image pairs with minimal or no overlap, and employs a Feature Matching Selector (FMS) to efficiently identify the most informative intermediate frames. The method achieves significant improvements in extreme pose estimation accuracy across four datasets.

## Background & Motivation

**Background**: Relative pose estimation from image pairs is a fundamental problem in 3D vision. Existing methods based on feature matching + RANSAC + the five-point algorithm are well-established for sufficiently overlapping pairs, but fail entirely under minimal or zero overlap.

**Limitations of Prior Work**:
- InterPose bridges non-overlapping image pairs by generating intermediate frames via video interpolation, but the synthesized **intermediate frames are blurry** (especially central frames), and its statistical self-consistency score for frame selection is **slow** and misaligned with the pose estimation objective.
- DynamiCrafter produces high-quality frames near the input endpoints but geometrically inconsistent intermediate frames, since the inputs themselves have little overlap.
- Commercial models (Runway/Luma) yield sharper results but are costly and still exhibit drift.

**Key Challenge**: No single video model can simultaneously guarantee geometric consistency across all frames under minimal overlap.

**Key Insight**: Decompose the problem into two steps — first use video interpolation to obtain a small set of "reliable relay frames" (frames near the input endpoints are more trustworthy), then refine intermediate frames using a pose-conditioned ViewCrafter.

**Core Idea**: Couple video interpolation with a pose-conditioned novel view synthesis model, leveraging the complementary strengths of each, combined with feature-correspondence-based frame selection.

## Method

### Overall Architecture
**Input**: An image pair $(I_0, I_T)$ with minimal or no overlap. **(1) HVG Stage 1**: DynamiCrafter interpolates a coarse video; 4 frames near the endpoints are selected as "relay frames" $\{I_0, I_1, I_{T-1}, I_T\}$. **(2) HVG Stage 2**: DUSt3R estimates camera poses from the relay frames; spherical linear interpolation yields a dense camera trajectory; ViewCrafter generates high-fidelity intermediate frames conditioned on these poses. **(3) FMS**: Feature matching + RANSAC is applied between each synthesized frame and the input pair; the top-$k$ frames with the highest inlier counts are selected as input to the pose estimation model.

### Key Designs

1. **Hybrid Video Generation (HVG)**:

    - **Function**: Two-stage synthesis of sharp intermediate frames.
    - **Stage 1 (Coarse Interpolation)**: DynamiCrafter generates a $T$-frame video. **Only the 4 most reliable frames** — $\{I_0, I_1, I_{T-1}, I_T\}$ — are retained. Ablations confirm that 4 frames outperforms 2 frames (insufficient structural information), 6 frames (blurry frames introduced), and all frames (central frames universally blurry).
    - **Stage 2 (Pose-Guided Refinement)**: DUSt3R recovers poses from the 4 relay frames; SO(3) spherical linear interpolation produces a dense trajectory; ViewCrafter generates conditioned frames.
    - **Design Motivation**: DynamiCrafter excels at synthesizing "near-endpoint frames," while ViewCrafter excels at "sharp synthesis given a known pose" — the two are complementary.

2. **Feature Matching Selector (FMS)**:

    - **Function**: Selects the $k$ synthesized frames most beneficial to pose estimation.
    - **Mechanism**: Local descriptors (e.g., SuperPoint) are extracted from each candidate frame and matched against the input image pair; RANSAC computes the inlier count. The top-$k$ frames with the highest total inlier counts are selected.
    - **Design Motivation**: InterPose's statistical self-consistency score requires multiple video generation passes (slow) and does not directly optimize for pose estimation utility. FMS uses feature correspondence counts to directly measure whether a frame can help establish geometric relationships between $I_0$ and $I_T$.

### Loss & Training
- **Completely training-free** — all pre-trained models (DynamiCrafter, ViewCrafter, DUSt3R, SuperPoint) are used off-the-shelf.
- No ground-truth poses or 3D supervision are required.

## Key Experimental Results

### Main Results — Extreme Pose Estimation Accuracy (Mean Rotation Error MRE↓)

| Dataset | DUSt3R (Direct) | InterPose (Single-Stage) | **PoseCrafter (Hybrid)** |
|--------|----------------|--------------------------|--------------------------|
| Cambridge Landmarks | 22.3° | 17.8° | **14.5°** |
| ScanNet | 25.1° | 19.7° | **16.2°** |
| DL3DV-10K | 18.6° | 15.2° | **14.3°** |
| NAVI | 11.2° | 7.8° | **6.9°** |

### Ablation Study — Number of Relay Frames

| Relay Frames | Cambridge MRE↓ | ScanNet MRE↓ | NAVI MRE↓ |
|-------------|----------------|--------------|-----------|
| 2 | 20.6° | 19.7° | 7.8° |
| **4** | **14.5°** | **16.2°** | **6.9°** |
| 6 | 16.7° | 17.0° | 7.2° |
| 16 (all) | 17.8° | 18.6° | 10.9° |

### HVG vs. Single-Model Baselines

| Method | Cambridge MRE↓ | DUSt3R Confidence |
|--------|----------------|-------------------|
| DynamiCrafter only | 17.8° | Low (blurry central frames) |
| ViewCrafter only | 19.2° | Medium (no reliable initial poses) |
| **HVG (coupled)** | **14.5°** | **High** |

### Key Findings
- **4 relay frames is optimal** — too few (2) loses structural information; too many (6+) introduces blurry intermediate frames that degrade overall pose estimation accuracy.
- HVG outperforms any single video model — DynamiCrafter provides reliable near-endpoint frames, while ViewCrafter leverages pose conditioning for sharp synthesis; the combination is super-additive.
- FMS is **faster and more effective** than InterPose's self-consistency score — it directly measures feature correspondence counts without requiring multiple video generation passes.
- DUSt3R confidence maps show that HVG frames achieve significantly higher confidence than DynamiCrafter frames, confirming that sharper frames genuinely benefit pose estimation.

## Highlights & Insights
- The **"reliable relay frame" concept** is elegant — not all synthesized frames are equally valuable; only those near the input endpoints are trustworthy. This insight stems from a deep understanding of video diffusion model failure modes.
- The **coupling of two models** rather than simple sequential substitution — DynamiCrafter addresses "where to start," while ViewCrafter addresses "how to get there."
- FMS **aligns frame selection with the downstream task objective** (feature matching quality ≈ pose estimation utility), making it more purposeful than statistical proxy scores.

## Limitations & Future Work
- The method relies on DUSt3R for intermediate pose estimation, so errors in DUSt3R propagate through the pipeline.
- ViewCrafter's synthesis quality is bounded by its pre-training data and model capacity.
- Inference cost remains high, as two video models, DUSt3R, and feature matching must all be executed.
- Evaluation is limited to static scenes — object motion in dynamic scenes would introduce additional challenges.

## Related Work & Insights
- **vs. InterPose**: InterPose employs single-stage video interpolation with statistical frame selection; PoseCrafter improves on both dimensions with hybrid generation and feature-matching-based selection.
- **vs. DUSt3R**: DUSt3R directly estimates pose/depth from two images; PoseCrafter "bridges" non-overlapping pairs by synthesizing intermediate frames, effectively increasing the available information.
- **vs. JOG3R**: JOG3R fine-tunes intermediate features of a video model for SfM; PoseCrafter is entirely training-free.

## Rating
- Novelty: ⭐⭐⭐⭐ — Novel combination of hybrid video generation and feature-matching-based frame selection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, detailed ablations, and comparisons against InterPose and DUSt3R.
- Writing Quality: ⭐⭐⭐⭐ — Clear method pipeline with intuitive visualizations (confidence map comparisons).
- Value: ⭐⭐⭐⭐ — Addresses a practically relevant problem of extreme viewpoint pose estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing](../../CVPR2026/video_generation/performrecast_expression_and_head_pose_disentanglement_for_portrait_video_editin.md)
- [\[ICCV 2025\] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis](../../ICCV2025/video_generation/causal-entity_reflected_egocentric_traffic_accident_video_synthesis.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](../../ICCV2025/video_generation/adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](../../ICCV2025/video_generation/fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](../../CVPR2026/video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)

</div>

<!-- RELATED:END -->
