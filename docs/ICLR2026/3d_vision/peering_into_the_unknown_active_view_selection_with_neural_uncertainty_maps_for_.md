---
title: >-
  [Paper Note] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction
description: >-
  [ICLR 2026][3D Vision][active view selection] This paper proposes PUN (Peering into the UnkNowN), which employs a lightweight feed-forward network, UPNet…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "active view selection"
  - "neural uncertainty map"
  - "3D reconstruction"
  - "NeRF"
  - "3DGS"
date: 2026-05-08
content_hash: 7bbdadc4d3a0ad00
---

# Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction

**Conference**: ICLR 2026
**arXiv**: [2506.14856](https://arxiv.org/abs/2506.14856)
**Code**: [https://github.com/ZhangLab-DeepNeuroCogLab/PUN](https://github.com/ZhangLab-DeepNeuroCogLab/PUN)
**Area**: 3D Vision
**Keywords**: active view selection, neural uncertainty map, 3D reconstruction, NeRF, 3DGS

## TL;DR

This paper proposes PUN (Peering into the UnkNowN), which employs a lightweight feed-forward network, UPNet, to directly predict the uncertainty distribution over all candidate viewpoints on a sphere from a single image — termed a neural uncertainty map (UMap) — thereby replacing the conventional iterative active view selection pipeline that requires repeated retraining of NeRF or 3DGS models. PUN achieves comparable reconstruction quality using only half the viewpoints of the upper bound, while delivering a 400× speedup and over 50% reduction in computational resource consumption during view selection.

## Background & Motivation

Active View Selection (AVS) aims to identify the minimal yet most informative set of viewpoints for training 3D rendering models, with important practical applications in robotic exploration, cultural heritage digitization, and search-and-rescue operations.

Nearly all existing AVS methods follow an iterative "train → evaluate → select → retrain" paradigm: a NeRF or 3DGS model is first trained on the current view set, then used to estimate per-candidate uncertainty (e.g., entropy of ray weight distributions, color variance, occlusion visibility), and after selecting the most informative next viewpoint, the model must be retrained with the newly added view. This cycle creates a severe computational bottleneck — the NVF method requires approximately 175 minutes to select 20 viewpoints.

An alternative class of reinforcement learning- or supervised learning-based methods avoids retraining but relies on fixed discrete candidate sets, limiting generalizability. 3DGS-based methods (ActiveSplat, ActiveGS) focus primarily on geometric information and neglect pixel-level color cues.

**Core Insight**: The uncertainty patterns of natural object reconstructions exhibit regularity — geometrically complex and texture-rich viewpoints tend to have higher uncertainty, and object symmetry reduces uncertainty for symmetric viewpoints. If such an "appearance → uncertainty" mapping can be learned from data, it becomes unnecessary to train a rendering model from scratch at each iteration to evaluate uncertainty.

## Method

### Overall Architecture

PUN operates in two stages:

**Stage 1: Offline training of UPNet.** UPNet is trained on the large-scale NUM (Neural Uncertainty Map) dataset to learn to predict the full-sphere uncertainty distribution from a single image.

**Stage 2: Online view selection.** Given an initial viewpoint, UPNet performs inference to predict a UMap; aggregated historical UMaps are used to select the next viewpoint, iterated for $T$ steps, after which the selected view set is used to train the target 3D rendering model.

### NUM Dataset Construction

The NUM dataset provides supervision signals for UPNet and is constructed as follows:

1. 13 object categories are selected from ShapeNet, with 100 instances per category.
2. For each instance, 48 viewpoints are uniformly sampled on the sphere using HEALPix (nside=2) as anchor points.
3. For each viewpoint, a pretrained SplatterImage (single-view 3DGS method) is used to synthesize novel views at the remaining 47 anchor positions.
4. Reconstruction error (PSNR/SSIM/LPIPS/MSE) is computed between synthesized and Blender-rendered ground-truth images, yielding a 48-dimensional UMap vector.
5. This produces a total of $13 \times 100 \times 48 = 62{,}400$ (viewpoint image, UMap) training pairs.

SplatterImage is chosen as the synthesis backbone for its computational efficiency — no per-viewpoint NeRF training is required; end-to-end feed-forward inference suffices. The dataset is split 8:1:1 for training/validation/testing, with 2 categories held out entirely for cross-category generalization evaluation.

### UPNet Architecture

UPNet adopts an exceptionally simple design: an ImageNet-pretrained ViT serves as the visual encoder, followed by a single fully connected layer appended to the CLS token output, producing a 48-dimensional vector corresponding to uncertainty values at the 48 anchor points. Training is supervised with MSE loss between predicted and ground-truth UMaps. PSNR is used by default as the uncertainty metric for constructing training targets.

### Uncertainty Interpolation and View Selection

The UMap provides uncertainty values only at 48 anchor points, whereas candidate viewpoints may lie anywhere on the sphere. PUN employs the following interpolation strategy:

- For each candidate point $C_i$, anchor neighbors $\tilde{P}$ within an angular distance of 30° are identified.
- Softmax angular distance-weighted summation is applied: $U^{C_i} = \sum_{\tilde{P_j} \in \tilde{P}} \omega_j U^{\tilde{P_j}}$, where $\omega_j = \frac{e^{-\theta_{ij}}}{\sum e^{-\theta_{ij}}}$.

Two key strategies govern view selection:

**Redundancy Filtering**: A candidate viewpoint is excluded if its uncertainty falls below a threshold of 0.1 at any historical time step, indicating proximity to a previously selected view and thus low information gain.

**Multiplicative Aggregation**: Among remaining candidates, uncertainty values across all time steps are multiplied, and the viewpoint with the highest product is selected: $v_{t+1} = \arg\max_{C_i} \prod_{t} U_t^{C_i}$. Multiplicative aggregation semantically enforces that only directions with consistently high uncertainty across all observations are worth exploring, thereby avoiding redundant revisits.

## Key Experimental Results

### Main Results

The paper evaluates PUN against 4 AVS baselines on 6 datasets. All methods select 20 viewpoints and train with the same NeRF backbone for 2,000 steps.

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | MSE↓ |
|--------|------|-------|-------|--------|------|
| NUM-inst (new instances, same category) | A-NeRF | 32.71 | 0.982 | 0.031 | 8.19 |
| | NVF | 33.08 | 0.984 | 0.028 | 6.98 |
| | **PUN** | **33.19** | **0.984** | **0.025** | **6.96** |
| | Upper-bnd | 36.47 | 0.989 | 0.017 | 4.11 |
| NUM-cat (unseen categories) | A-NeRF | 33.16 | 0.984 | 0.024 | 7.04 |
| | NVF | 33.15 | 0.985 | 0.021 | 6.65 |
| | **PUN** | **34.74** | **0.985** | **0.019** | **5.03** |
| | Upper-bnd | 36.91 | 0.990 | 0.013 | 3.33 |
| NUM-3DGS (3DGS backbone) | NVF | 30.67 | 0.977 | 0.07 | 2.28 |
| | **PUN** | **36.71** | **0.990** | **0.03** | **0.40** |
| NeRFAssets (real scenes) | NVF | 26.31 | 0.928 | 0.115 | 0.005 |
| | **PUN** | **26.73** | **0.944** | **0.093** | **0.003** |
| MIP360 (real scenes) | NVF | 15.41 | 0.203 | 0.653 | 32.03 |
| | **PUN** | **17.49** | **0.294** | **0.545** | **19.98** |

On NUM-cat, PUN outperforms NVF by 1.59 dB PSNR, demonstrating substantially superior cross-category generalization of feed-forward uncertainty prediction over iterative methods. On the real-world MIP360 dataset, PUN surpasses NVF by over 2 dB PSNR.

### Ablation Study

| Ablation Dimension | Configuration | PSNR |
|----------|------|------|
| Uncertainty metric | PSNR (default) | 37.4 |
| | SSIM | 36.4 |
| | LPIPS | 36.1 |
| | MSE | 35.0 |
| Redundancy filtering | small+all (default) | 37.4 |
| | Filtering disabled | 36.9 |
| | top-32 exclusion | 36.8 |
| | single (5° neighborhood exclusion) | 36.9 |
| Aggregation strategy | Multiplicative over all steps (default) | 37.4 |
| | Latest UMap only | 36.9 |
| | Differential strategy | 36.8 |
| | Additive aggregation | 36.9 |
| Data diversity | 80 instances / 12 views | 36.9 |
| | 40 instances / 24 views | 36.3 |
| | 20 instances / 48 views | 35.7 |
| Number of anchors | 12 | 37.0 |
| | 48 (default) | 36.8 |
| | 108 | 36.8 |

Key findings:

- **PSNR is the optimal training metric**, outperforming SSIM/LPIPS/MSE by 1–2.4 dB.
- **Both redundancy filtering and temporal aggregation contribute**, with disabling either causing a ~0.5 dB drop.
- **Instance diversity is far more important than viewpoint density**: given a fixed total sample budget, using more object instances (80×12) substantially outperforms fewer instances with more viewpoints (20×48), as redundant viewpoints of symmetric objects provide diminishing incremental information.
- Increasing anchors from 12 to 48 yields improvement, but 108 anchors provide negligible additional gain — 48 is the optimal cost-performance configuration.

### Computational Efficiency

PUN's primary advantage lies in computational efficiency:

- Total view selection time: **5.5 minutes** vs. NVF's **175 minutes** (400× speedup)
- GPU memory: reduced from **8,098 MB** to **655 MB**
- CPU usage: reduced from **903%** to **74%**
- RAM: reduced from **4,292 MB** to **1,870 MB**
- GPU utilization: reduced from **30.6%** to **0.3%**

This is attributed to PUN entirely avoiding NeRF/3DGS training during the view selection phase, requiring only a single ViT forward pass.

## Highlights & Insights

**Prerequisite for the paradigm shift**: The authors find that uncertainty patterns exhibit learnable regularity across objects — the Pearson correlation between UPNet predictions and ground-truth UMaps reaches 0.82. Further analysis reveals that UPNet implicitly captures both geometric complexity (depth gradient variance, edge density) and texture complexity (color entropy, Laplacian energy), which are domain-agnostic low-level features that account for the observed cross-category generalization.

**Rationale for multiplicative aggregation**: Treating each step's UMap as an independent uncertainty estimate, multiplication is equivalent to computing a joint probability — only regions with persistently high uncertainty across all observations are selected. Compared to additive aggregation or using only the latest UMap, full utilization of temporal information yields consistent performance gains.

**An interesting symmetry phenomenon**: When predicting UMaps for highly symmetric objects such as cabinets, UPNet correctly assigns low uncertainty to symmetric viewpoints, indicating that the network has learned to leverage object symmetry to reduce exploratory redundancy — and this capability generalizes to unseen categories.

## Limitations & Future Work

- The method assumes cameras are distributed on a sphere of fixed radius; free-trajectory viewpoint distributions are not supported.
- The approach targets single-object reconstruction; scene-level AVS involving multiple objects requires more complex uncertainty modeling.
- UMap training targets are based on image reconstruction error (PSNR, etc.) rather than directly optimizing geometric quality (mesh accuracy, completion rate), which may be suboptimal for downstream tasks requiring precise geometry.
- SplatterImage as the synthesis backbone introduces its own biases; while ablations confirm effectiveness with other backbones, stronger synthesis models may yield more accurate UMap supervision signals.

## Related Work & Insights

- **vs. NVF**: NVF's visibility modeling is a meaningful contribution, but requires retraining NeRF at every step; PUN replaces iterative training with feed-forward prediction, achieving comprehensive advantages in both efficiency and generalization.
- **vs. ActiveSplat / ActiveGS**: These methods estimate uncertainty from 3D Gaussian attributes (variance, density), prioritizing spatial coverage while neglecting pixel-level color cues; PUN's UMap simultaneously captures geometric and texture complexity.
- **vs. supervised/RL methods**: Fixed candidate sets limit generalizability; PUN supports arbitrary viewpoint selection through continuous spherical interpolation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Transitioning AVS from iterative optimization to feed-forward prediction constitutes a genuine paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 datasets spanning synthetic/real/cross-category/cross-backbone/cross-illumination/cross-distance settings, with ablations covering all key design choices.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with high-quality figures, though certain paragraphs are somewhat repetitive.
- Value: ⭐⭐⭐⭐⭐ — A 400× speedup and minimal memory requirements make AVS practically viable in resource-constrained settings such as robotics and mobile platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction](../../AAAI2026/3d_vision/surface-based_visibility-guided_uncertainty_for_continuous_active_3d_neural_reco.md)
- [\[ICLR 2026\] COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](coopertrim_adaptive_data_selection_for_uncertainty-aware_cooperative_perception.md)
- [\[ICLR 2026\] CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions](clods_visual-only_unsupervised_cloth_dynamics_learning_in_unknown_conditions.md)
- [\[CVPR 2026\] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction](../../CVPR2026/3d_vision/fluidgaussian_propagating_simulation-based_uncertainty_toward_functionally-intel.md)
- [\[ICLR 2026\] Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](joint_shadow_generation_and_relighting_via_light-geometry_interaction_maps.md)

</div>

<!-- RELATED:END -->
