---
title: >-
  [Paper Note] From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision
description: >-
  [ICCV 2025][Infrared small target detection] This paper proposes a Progressive Active Learning (PAL) framework that trains infrared small target detection networks through a three-stage strategy—model pre-start, model enhancement, and model refinement—driving the network to actively identify and learn from hard samples in an easy-to-hard manner. Under single point supervision, PAL substantially narrows the performance gap with fully supervised methods (IoU improvement of 8.53%–29.1%).
tags:
  - ICCV 2025
  - Infrared small target detection
  - single point supervision
  - progressive active learning
  - curriculum learning
  - pseudo-label evolution
date: 2026-05-08
content_hash: 09a5ced1dd06a1d1
---

# From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision

**Conference**: ICCV 2025
**arXiv**: [2412.11154](https://arxiv.org/abs/2412.11154)
**Code**: [github.com/YuChuang1205/PAL](https://github.com/YuChuang1205/PAL)
**Area**: Other
**Keywords**: Infrared small target detection, single point supervision, progressive active learning, curriculum learning, pseudo-label evolution

## TL;DR

This paper proposes a Progressive Active Learning (PAL) framework that trains infrared small target detection networks through a three-stage strategy—model pre-start, model enhancement, and model refinement—driving the network to actively identify and learn from hard samples in an easy-to-hard manner. Under single point supervision, PAL substantially narrows the performance gap with fully supervised methods (IoU improvement of 8.53%–29.1%).

## Background & Motivation

### Problem Definition

Single-frame infrared small target (SIRST) detection is a critical technology for infrared imaging systems, with broad applications in traffic analysis, environmental monitoring, and maritime assistance. Key challenges include:
- Small target size, lack of intrinsic features, and insufficient annotated data
- High cost of pixel-level dense annotation
- The need for high-performance detection under single point supervision

### Limitations of Prior Work

The existing LESPS (Label Evolution with Single Point Supervision) framework suffers from three key issues:

**Training instability**: Directly training on all point-labeled samples causes low-performance models to generate erroneous label evolution in early stages.

**Label over-evolution**: Once evolved regions expand, they cannot contract, causing the annotated target area to grow continuously and deviate from the true target.

**Limited network performance**: The framework fails to fully exploit the detection capability of the embedded network, leaving a large performance gap between single point supervision and full supervision.

### Root Cause

**Core Idea**: Inspired by biological organisms that progressively adapt to their environment and accumulate knowledge—an effective learning process should proceed from easy to hard, accounting for the current learner's (model's) capacity rather than treating all tasks (samples) uniformly. Specifically for SIRST detection:
- Some target regions can be detected by traditional methods (easy samples) and should be used first to establish foundational capability
- As model capacity grows, harder samples are progressively introduced
- Pseudo-labels should be continuously refined with controlled balance between expansion and contraction

## Method

### Overall Architecture

The PAL framework divides training into three stages (partitioned across total training epochs as 0.0–0.2, 0.2–0.8, and 0.8–1.0):
1. **Model Pre-start** (0–20%): Automatically selects easy samples to establish basic detection capability
2. **Model Enhancement** (20%–80%): Progressively introduces hard samples while refining pseudo-labels
3. **Model Refinement** (80%–100%): Fully trains on all samples with further pseudo-label refinement

### Key Designs

#### 1. **Model Pre-start and EPG Strategy**

- **Function**: Automatically selects a subset of easy samples and generates high-quality pseudo-labels, enabling the model to acquire basic task-specific capability in early training
- **Mechanism**: Proposes an Easy-sample Pseudo-label Generation (EPG) strategy:
  1. Processes local image patches centered on point labels (rather than entire images) to reduce background interference
  2. Applies Gaussian filtering for noise suppression → Canny edge detection for target contour extraction → morphological closing to fill contours
  3. Uses point labels to evaluate connected regions: regions containing the labeled point with area below a threshold are considered correct detections; others are treated as false detections
  4. Samples with target-level recall $\geq 0.8$ are classified as easy samples; the rest are hard samples
  5. Overlays segmentation results of easy samples onto a pure black background to generate pseudo-labels, with point labels added to compensate for missed detections

- **Design Motivation**: An untrained model is like a newborn—full of potential but requiring simple knowledge as a starting point. Training directly on all weakly labeled samples leads to instability and low accuracy; establishing foundational capability with easy samples first prevents hard samples from introducing excessive noise in early stages.

#### 2. **Fine Dual-Update Strategy**

- **Function**: During the model enhancement stage, dynamically introduces new hard samples while continuously refining existing pseudo-labels
- **Mechanism**: Comprises two complementary operations—Coarse Outer Update (COU) and Fine Inner Update (FIU)

**Coarse Outer Update (COU)**: Evaluates hard samples in the candidate pool and transfers qualifying samples to the training pool
- Computes missed detection rate $R_m$ and false detection rate $R_f$ from model predictions
- A sample is deemed "recognizable" and transferred to the training pool when $R_m \leq T_m$ and $R_f \leq T_f$:

$$S = \begin{cases} I \in \text{Easy Sample} & \text{if } R_m \leq T_m \text{ \& } R_f \leq T_f \\ I \in \text{Hard Sample} & \text{otherwise} \end{cases}$$

- Eliminates predicted target regions with no intersection with ground-truth point labels (false positive elimination):

$$\hat{P}_b = P_b \setminus \{A_f \in P_b | \text{Intersection}(A_f, L_{\text{true}}) = \emptyset\}$$

**Fine Inner Update (FIU)**: Iteratively refines pseudo-labels for all samples in the training pool
- Candidate region extraction: uses adaptive thresholding to extract local candidate regions from predictions:

$$T_{\text{adapt}} = \max(P_n^i) \cdot (T_b + k(1 - T_b) \cdot L_n^i / (hwr))$$

- False region elimination: removes candidate regions with no intersection with pseudo-label centers
- Pseudo-label update (with decay factor):

$$L_{n+1} = \lambda L_n \odot (1 - N_n) + \frac{L_n + P_n}{2} \odot N_n$$

where $\lambda$ is the decay factor and $N_n$ is the candidate region mask

- **Design Motivation**: Existing methods perform label evolution without considering sample difficulty. COU enables the model to progressively and "actively" select hard samples within its current capacity; FIU leverages the model's enhanced capability to continuously refine pseudo-labels. The decay factor $\lambda$ addresses label over-evolution, achieving a dynamic balance between expansion and contraction of annotated target regions.

#### 3. **Edge-Enhanced Difficult-sample Mining (EEDM) Loss**

- **Function**: Guides the network to focus on edge pixels and hard-to-detect target regions
- **Mechanism**: The EEDM loss consists of two components—edge pixel enhancement (weighted loss on edge pixels) and hard pixel mining (filtering pixels with larger losses)—to alleviate the positive-negative pixel imbalance
- **Design Motivation**: Infrared small targets lack intrinsic features; edge information and hard examples are critical to improving detection accuracy

### Loss & Training

- Employs EEDM loss (edge enhancement + hard example mining)
- AdamW optimizer with initial learning rate $1 \times 10^{-3}$, batch size 16
- Total training of 400 epochs with three-stage division: pre-start 0–80 epochs, enhancement 80–320 epochs, refinement 320–400 epochs
- COU and FIU update interval set to 5 epochs
- $T_m$ linearly increases from an initial value of 0.2 to 1.0

## Key Experimental Results

### Main Results

**Performance comparison of different networks + PAL vs. LESPS on SIRST3 dataset (Coarse point labels)**:

| Backbone | Method | IoU(%) | nIoU(%) | Pd(%) | Fa(×10⁻⁶) |
|---------|------|--------|---------|-------|-----------|
| ACM | Full Supervision | 64.93 | 64.89 | 94.88 | 20.97 |
| ACM | LESPS | 37.42 | 35.20 | 84.12 | 50.13 |
| ACM | **PAL (Ours)** | **51.51** | **54.07** | **92.89** | **39.18** |
| DNANet | Full Supervision | 81.96 | 85.90 | 97.54 | 9.11 |
| DNANet | LESPS | 57.52 | 55.09 | 91.30 | 19.04 |
| DNANet | **PAL (Ours)** | **67.20** | **70.20** | **96.15** | **10.86** |
| MSDA-Net | Full Supervision | 83.46 | 85.97 | 97.41 | 17.15 |
| MSDA-Net | LESPS | 46.26 | 45.73 | 85.38 | 36.16 |
| MSDA-Net | **PAL (Ours)** | **69.38** | **71.55** | **97.41** | **16.34** |

### Ablation Study

**Component contribution analysis** (based on representative network performance on SIRST3):

| Configuration | Key Change | Description |
|------|---------|------|
| w/o Model Pre-start | Significant IoU drop | Direct training on all samples in early stages causes instability |
| w/o COU | IoU drop | Lacks mechanism for progressive introduction of hard samples |
| w/o FIU | Reduced accuracy | Pseudo-labels cannot be continuously refined |
| w/o decay factor λ | Label over-evolution | Target regions expand continuously without contraction |
| Full PAL | Best performance | All components are complementary and indispensable |

**Gap comparison with full supervision**:

| Network | LESPS-IoU | PAL-IoU | Full-IoU | PAL vs. LESPS Gain | PAL vs. Full Gap |
|------|----------|---------|----------|-----------------|----------------|
| ACM | 37.42 | 51.51 | 64.93 | +14.09 | -13.42 |
| ALCNet | 45.30 | 57.11 | 65.69 | +11.81 | -8.58 |
| DNANet | 57.52 | 67.20 | 81.96 | +9.68 | -14.76 |
| MSDA-Net | 46.26 | 69.38 | 83.46 | +23.12 | -14.08 |

### Key Findings

1. **PAL significantly outperforms LESPS**: IoU improvements of 8.53%–29.1% across all backbone networks, with performance trends closely matching those of full supervision
2. **Effectively narrows the gap with full supervision**: PAL provides an efficient and stable bridge between single point supervision and full supervision
3. **Decay factor is critical**: Without the decay factor, label regions in the LESPS framework expand without contraction; PAL's decay factor effectively resolves this issue
4. **Strong generalizability**: PAL is compatible with all existing SIRST detection networks (ACM, ALCNet, MLCL-Net, DNANet, and 4 others—8 networks total) without modifying network architectures
5. **Consistently effective across three independent datasets**: Achieves state-of-the-art results on NUAA-SIRST, NUDT-SIRST, and IRSTD-1K

## Highlights & Insights

1. **Easy-to-hard curriculum learning**: The first work to introduce automatic curriculum learning into single point supervised SIRST detection, designing a difficulty measurer and training scheduler tailored to point labels
2. **Ingenuity of EPG strategy**: Leverages the domain prior that infrared small targets exhibit high brightness, combining traditional image processing methods (Gaussian filtering + Canny + morphology) to automatically select easy samples without any manual annotation
3. **Complementarity of the dual-update strategy**: COU handles "expansion"—introducing new samples; FIU handles "refinement"—improving existing labels; the two operate in concert
4. **Insight of the decay factor**: Identifies the "expand-only, never contract" problem in label evolution and proposes a solution, representing a key improvement over the LESPS framework
5. **Plug-and-play**: PAL is an end-to-end framework directly compatible with existing SIRST detection networks, offering strong practical utility

## Limitations & Future Work

1. **EPG relies on domain priors**: The Gaussian filtering + Canny combination assumes targets have high brightness, which may fail in complex backgrounds or low-contrast scenarios
2. **Threshold configuration**: Multiple thresholds ($T_m$, $T_f$, $T_b$) require manual specification; while the paper provides reasonable justification, different scenarios may require adjustment
3. **Fixed stage partitioning**: The three-stage ratio (20%–60%–20%) is fixed; adaptive stage division may yield better results
4. **Single-frame detection only**: Temporal information is not exploited; extending PAL to video sequence detection could bring further improvements
5. **Computational overhead**: Periodic execution of COU and FIU increases training time, particularly on large-scale datasets

## Related Work & Insights

- **Essential difference from LESPS**: LESPS applies uniform label evolution across all samples; PAL differentiates treatment based on sample difficulty and model capacity
- **Relation to automatic curriculum learning**: Compared to existing automatic CL methods that focus on fully supervised settings, PAL is the first to explore its application in single point supervised tasks
- **Inspiration from EPG**: Traditional image processing methods, while not robust, can serve as "guides" for deep learning by filtering easy samples

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of curriculum learning with single point supervised SIRST detection is novel; EPG and the dual-update strategy are cleverly designed
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 8 backbone networks × 4 datasets with detailed ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear; the biological learning analogy aids in understanding the motivation
- **Value**: ⭐⭐⭐⭐ — Provides an efficient and stable training framework for weakly supervised small target detection; plug-and-play design enhances practical utility

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering](../../NeurIPS2025/others/mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)
- [\[ICCV 2025\] Multi-view Gaze Target Estimation](multi-view_gaze_target_estimation.md)
- [\[NeurIPS 2025\] On Agnostic PAC Learning in the Small Error Regime](../../NeurIPS2025/others/on_agnostic_pac_learning_in_the_small_error_regime.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)

<!-- RELATED:END -->
