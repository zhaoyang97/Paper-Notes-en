---
title: >-
  [Paper Note] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras
description: >-
  [NeurIPS 2025][camera intrinsics] This paper introduces InFlux, the first real-world video benchmark with per-frame ground-truth dynamic camera intrinsics (386 videos…
tags:
  - "NeurIPS 2025"
  - "camera intrinsics"
  - "dynamic calibration"
  - "benchmark"
  - "lookup table"
  - "video 3D understanding"
date: 2026-05-08
content_hash: fdce33609e965eb9
---

# InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras

**Conference**: NeurIPS 2025
**arXiv**: [2510.23589](https://arxiv.org/abs/2510.23589)  
**Code**: [Project Page](https://influx.cs.princeton.edu/)  
**Area**: Video Understanding
**Keywords**: camera intrinsics, dynamic calibration, benchmark, lookup table, video 3D understanding

## TL;DR

This paper introduces InFlux, the first real-world video benchmark with per-frame ground-truth dynamic camera intrinsics (386 videos, 143K+ annotated frames). Accurate annotations are achieved via a lookup table (LUT) mapping lens metadata to intrinsic parameters. The benchmark reveals that existing intrinsic prediction methods perform poorly under dynamic intrinsic settings.

## Background & Motivation

- **The constant-intrinsic assumption is invalid**: Mainstream 3D methods such as NeRF, 3DGS, and SLAM assume camera intrinsics remain constant throughout a video. However, DSLR zoom lenses and smartphone autofocus both cause per-frame intrinsic changes, severely limiting the robustness of these methods on in-the-wild videos.
- **No dynamic intrinsic benchmark exists**: Existing datasets (KITTI, EuRoC, ETH3D, etc.) are collected under fixed-lens settings requiring only a single calibration. The only work involving varying intrinsics [Liao et al. 2025] provides only checkerboard videos (lacking scene diversity) and focal length annotations for 300 web images (which do not correspond to true CFL values).
- **Synthetic data cannot replace real benchmarks**: Synthetic datasets suffer from a visual sim-to-real gap and lack either intrinsic annotations or scene diversity.
- **Per-frame ground truth is extremely difficult to obtain**: Full calibration for every frame is prohibitively expensive and disrupts video continuity (requiring frame-by-frame pause during capture), which is why no prior work has achieved this.

## Method

### Mechanism: LFL-FD Lookup Table (LUT)

The key observation is that the optical state of a zoom lens is uniquely determined by two parameters—lens focal length (LFL) and focus distance (FD). Using professional cinema lenses that support /i Technology metadata recording (Canon CINE-SERVO 17-120mm and Fujinon Premista 80-250mm), LFL and FD values can be logged per frame. A LUT mapping $(LFL, FD)$ to full intrinsic parameters is constructed once in advance, reducing per-frame calibration to a one-time table lookup.

### Calibration Experiment Design

Different calibration targets are used depending on the size of the field-of-view spatial footprint (FSF):

- **Small/Medium FSF — Checkerboard Calibration**: Four sizes of AprilGrid calibration boards ($100\times75$ mm to $800\times600$ mm) are used, selecting the largest that fits entirely within the FOV. Jitter-based capture aids keyframe extraction, and ANMS is applied to select frames based on detection count.
- **Large FSF — Drone Calibration**: When the FSF is too large for a manufacturable planar target, a Holybro X500 V2 drone equipped with an RTK positioning chip (Septentrio Mosaic X5, cm-level accuracy) and a red LED is used as the calibration target. Red LED images captured at night provide precise 2D detections, while RTK supplies 3D positions; temporal synchronization establishes 2D-3D correspondences.

### Improved Kalibr

The original Kalibr exhibits convergence issues and principal point drift during LM optimization:

- **CFL Initialization**: The original vanishing-point-based method is replaced by the thin-lens approximation formula, leveraging known LFL and FD information.
- **Fixed-Point Initialization**: During the distortion initialization stage, the principal point is periodically reset to the image center to prevent anomalous drift.
- **Median over Multiple Runs**: Multiple rollouts with random orderings are performed on the final optimization stage; the median result is selected to reduce variance.

### LUT Interpolation

- **Grid Region (checkerboard experiments)**: The LFL-FD space approximates a regular grid; trapezoidal bilinear interpolation is applied.
- **Non-Grid Region (including drone experiments)**: Delaunay triangulation with barycentric interpolation is used.

$$\mathbf{K}(l, d) = \text{Interpolate}\big(\{(\text{LFL}_i, \text{FD}_i, \mathbf{K}_i)\}\big)$$

where $\mathbf{K}$ includes $f_x, f_y, c_x, c_y$ and Brown-Conrady distortion parameters.

## Key Experimental Results

### Dataset Statistics

| Attribute | Count |
|:---|:---:|
| Total videos | 386 |
| Annotated frames | 143K+ |
| Indoor videos | 126 |
| Outdoor videos | 260 |
| Lens types | 2 (Canon 17-120mm, Fujinon 80-250mm) |
| Intrinsic variation types | Monotonic zoom/focus, periodic, non-monotonic, cinematic push-pull |

### Table 1: Evaluation of Baseline Intrinsic Prediction Methods

| Method | %$f_x$ Error↓ | %$f_y$ Error↓ | %$c_x$ Error↓ | %$c_y$ Error↓ | %EPE<300px↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| GeoCalib | **56.5** | **56.5** | **0.099** | **0.204** | **52.9** |
| WildCamera | 45.6 | 46.9 | 5.04 | 6.39 | 47.2 |
| UniDepthV2 | 50.6 | 51.1 | 1.61 | 2.58 | 46.1 |
| DroidCalib | 68.1 | 70.0 | 10.1 | 15.7 | 28.0 |
| Perspective Fields | 64.6 | 64.6 | 18.6 | 19.7 | 17.8 |
| COLMAP | 1270 | 1280 | 0.112 | 0.299 | 7.85 |

**Key Findings**:
- **All methods perform poorly**: Even the best-performing GeoCalib achieves only 52.9% of frame point pairs with EPE <300px at $3424\times2202$ resolution.
- **COLMAP nearly completely fails**: 92% of frames yield no prediction, with CFL error as high as 1270%.
- **DroidCalib relies on optical flow**: 15% of frames in low-motion videos cannot be predicted.
- **Per-frame methods lack temporal smoothness**: Single-frame methods such as GeoCalib and WildCamera produce non-smooth intrinsic sequences.

### Synthetic Experiment Validation of Improved Kalibr

On synthetic calibration scenes rendered in Blender, the improved Kalibr versus the original:
- Eliminates occasional large error spikes present in the original.
- Achieves successful convergence across all experiments (the original fails in some cases).
- Substantially reduces prediction variance for both CFL and principal point.

## Highlights & Insights

- **Fills a critical gap**: InFlux is the first real-world video benchmark providing per-frame dynamic intrinsic ground truth, enabling the research community to systematically evaluate dynamic intrinsic prediction methods for the first time.
- **Elegant annotation scheme**: The LUT + lens metadata approach converts the per-frame calibration challenge into a one-time table lookup, balancing accuracy with natural capture conditions.
- **Novel drone calibration**: The RTK + LED design elegantly resolves the inability of planar calibration targets to cover large-FOV scenes.
- **Evaluation exposes weaknesses**: Quantitative results clearly demonstrate the fragility of existing methods under dynamic intrinsics, pointing the way for future research.

## Limitations & Future Work

- **High hardware dependency**: Professional cinema-grade cameras and lenses (ARRI Alexa Mini + cinema zoom lenses) are required, making the approach difficult to generalize to consumer devices.
- **Limited lens coverage**: Only two lens types are included, with a narrow range of camera models.
- **No train/test split**: The benchmark does not provide a standard training set, limiting direct use for data-driven methods.
- **Restricted to metadata-capable lenses**: Lenses that do not record LFL/FD cannot use this scheme to obtain ground truth.
- **Interpolation accuracy**: Linear/barycentric interpolation may not perfectly model complex real-world lens systems.

## Related Work & Insights

- **Real-world intrinsic datasets** (KITTI, EuRoC, ETH3D): All use fixed intrinsics and do not support dynamic variation.
- **Synthetic intrinsic datasets** ([Ray+ 2024], [Liao+ 2025]): Suffer from sim-to-real gap or insufficient coverage.
- **Calibration methods** (Kalibr [Maye+ 2013], OpenCV [Bradski 2000]): The improved Kalibr introduced in InFlux achieves substantially higher accuracy.
- **Intrinsic estimation methods** (COLMAP, GeoCalib [Jin+ 2023], UniDepthV2 [Piccinelli+ 2025]): All reveal shortcomings in dynamic scenes when evaluated on InFlux.

## Rating

- Novelty: ⭐⭐⭐⭐ — Fills the gap in dynamic intrinsic benchmarks; the LUT annotation scheme is creatively designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six baseline methods + synthetic validation + rich diversity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete technical details, and abundant figures and tables.
- Value: ⭐⭐⭐⭐ — Provides the 3D vision community with a much-needed evaluation infrastructure for dynamic intrinsics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/others/measuring_uncertainty_calibration.md)

</div>

<!-- RELATED:END -->
