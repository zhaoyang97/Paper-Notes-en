---
title: >-
  [Paper Note] YOLO-IOD: Towards Real Time Incremental Object Detection
description: >-
  [AAAI 2026][Object Detection][Incremental Object Detection] This work is the first to systematically integrate incremental object detection (IOD) into the YOLO real-time framework. It identifies three types of knowledge conflict, proposes a three-module solution (CPR + IKS + CAKD), and introduces the more realistic LoCo COCO benchmark for evaluation.
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Incremental Object Detection"
  - "YOLO"
  - "Knowledge Distillation"
  - "Catastrophic Forgetting"
  - "Pseudo Labels"
date: 2026-05-08
content_hash: 050634c1e6b7c160
---

# YOLO-IOD: Towards Real Time Incremental Object Detection

**Conference**: AAAI 2026
**arXiv**: [2512.22973](https://arxiv.org/abs/2512.22973)  
**Code**: None  
**Area**: Object Detection
**Keywords**: Incremental Object Detection, YOLO, Knowledge Distillation, Catastrophic Forgetting, Pseudo Labels

## TL;DR

This work is the first to systematically integrate incremental object detection (IOD) into the YOLO real-time framework. It identifies three types of knowledge conflict, proposes a three-module solution (CPR + IKS + CAKD), and introduces the more realistic LoCo COCO benchmark for evaluation.

## Background & Motivation

**Incremental Object Detection (IOD)** requires a model to learn new categories while retaining the ability to detect previously learned ones. Existing IOD methods are predominantly built on Faster R-CNN or DETR; however, in practical industrial deployment, **YOLO-series detectors** are widely adopted for their real-time inference capability. Directly applying existing IOD methods to YOLO leads to severe performance degradation.

**Core Contribution**: This paper is the first to systematically identify **three types of knowledge conflict** that cause catastrophic forgetting in YOLO-based incremental detectors:

#### 1. Foreground–Background Confusion

In the incremental setting, unannotated objects from previous and future stages are misclassified as background. YOLO relies on aggressive data augmentation (Mosaic, MixUp) and assumes accurate annotations. Under IOD, noise in pseudo labels is amplified by these augmentations, severely degrading performance.

#### 2. Parameter Interference

Different tasks frequently rely on overlapping subsets of model parameters. Updates for new tasks alter shared parameters, disrupting previously learned representations and causing catastrophic forgetting of old tasks.

#### 3. Knowledge Distillation Misalignment

Teacher and student models are optimized for different class distributions, violating the core assumption of standard knowledge distillation that both models share a consistent learning objective. YOLO's dense prediction nature further exacerbates this issue. Existing methods only select old-task outputs that do not overlap with new labels as distillation targets, thus transferring only partial knowledge.

**Necessity of the LoCo COCO Benchmark**: Existing IOD benchmarks partition categories arbitrarily, ignore category co-occurrence relationships, and allow images to appear across multiple incremental stages (on average, each image appears in 1.84 stages). This violates the fundamental premise of continual learning and artificially inflates the effectiveness of pseudo-label methods, since detectors can generate pseudo labels on reused training images.

## Method

### Overall Architecture

YOLO-IOD is built upon pre-trained YOLO-World and achieves incremental learning through stage-wise parameter-efficient fine-tuning. It comprises three modules:

1. **CPR** (Conflict-aware Pseudo-label Refinement) → addresses foreground–background confusion
2. **IKS** (Important Kernel Selection) → addresses parameter interference
3. **CAKD** (Cross-stage Asymmetric Knowledge Distillation) → addresses distillation misalignment

### Key Designs

#### 1. **Conflict-aware Pseudo-label Refinement (CPR)**

**Augmented Pseudo-label Loss**: The pseudo-label confidence $s$ serves as a soft supervision target, combined with confidence-aware weighting and entropy regularization:

$$\mathcal{L}_{pseudo}^{cls} = -|s - p_t|^\gamma \log(p_t) + \lambda \cdot (1-s)^\delta \cdot H(\hat{y})$$

- First term: focal-style confidence-aligned supervision
- Second term: adaptive entropy regularization inversely scaled by confidence
- Low-confidence pseudo labels provide soft supervision and are regularized to preserve uncertainty; high-confidence labels contribute stable supervision

**Clustering Unknown Pseudo Labels**:
1. Construct a generic vocabulary $V_{gen}$ (500 common objects + 50 abstract super-categories summarized by an LLM)
2. Use YOLO-World + $V_{gen}$ to detect all unannotated foreground objects
3. Apply frequency-weighted K-Means clustering on the text features of detected categories to obtain an unknown super-category set $\mathcal{U}$
4. Reframe knowledge conflicts as a process of discovering and learning unknown super-categories

#### 2. **Important Kernel Selection (IKS)**

Parameter importance is quantified at the granularity of convolutional kernels (rather than individual parameters), avoiding storage costs that grow linearly with the number of tasks.

**Fisher Information-based Parameter Importance**:

$$\mathbf{I}_t(\mathbf{w}^k) = \sum_{j=1}^{d_k} \left( \frac{1}{N_t} \sum_{n=1}^{N_t} \left( \frac{\partial \log p(y_n|x_n;\theta)}{\partial w_j^k} \right)^2 \right)$$

**Differential Importance** (excluding parameters critical to old tasks):

$$\Delta \mathbf{I}_t(\mathbf{w}^k) = \mathbf{I}_t(\mathbf{w}^k) - \rho \sum_{i=1}^{t-1} \mathbf{I}_i(\mathbf{w}^k)$$

Only the top-$\mathcal{K}$ kernels are selected for fine-tuning (20% at the base stage, 12% at incremental stages); all remaining kernels are frozen.

#### 3. **Cross-stage Asymmetric Knowledge Distillation (CAKD)** (Core Innovation)

A **dual-teacher framework** is adopted, with the target detector $\mathcal{M}_t$ as the student:
- **Old Teacher** $\mathcal{M}_{t-1}$: specializes in $\mathcal{C}_{1:t-1}$; its detection head suppresses responses to irrelevant features
- **Current Teacher** $\mathcal{M}_{s_t}$: trained solely on current-stage data $D_t$, focusing on $\mathcal{C}_t$

Distillation procedure: the student's neck features $\mathbf{F}_{student}^{neck}$ are fed into each teacher's detection head to generate cross-stage post-head features for distillation.

**Focal Weight**: $w_{focal}(p) = \max_j \text{logit}_{teacher}(p, j)$, suppressing background and noisy regions.

**Classification Distillation Loss**:
$$\mathcal{L}_{cls\_kd} = \sum_p \|\mathbf{E}_{teacher}(p) - \mathbf{E}_{student\_cross}(p)\|_2^2 \cdot w_{focal}(p)$$

**Regression Distillation Loss**:
$$\mathcal{L}_{reg\_kd} = \sum_p \mathcal{L}_{IoU}(B_{tea}(p), B_{stu\_cross}(p)) \cdot w_{focal}(p)$$

**Total Distillation Objective**: $\mathcal{L}_{CAKD} = \alpha \mathcal{L}_{cls\_kd} + \beta \mathcal{L}_{reg\_kd}$

**Advantage over Existing Methods**: Existing methods only distill old-task outputs that do not overlap with new labels, transferring only partial knowledge. CAKD performs global distillation through dual-teacher detection heads, handling new and old categories separately and avoiding misaligned supervision.

### LoCo COCO Benchmark Construction

1. Construct a category co-occurrence matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$
2. Graph clustering assigns frequently co-occurring categories to the same task
3. Images that still span multiple stages are randomly assigned to a single task
4. Each image appears in exactly one stage, eliminating data leakage

### Loss & Training

- Base model: YOLO-World (X)
- Batch size 16, 4 × RTX 3090
- Learning rate: backbone 2e-5, neck/head 2e-4
- AdamW optimizer, 20 epochs; Mosaic augmentation disabled after epoch 10
- IKS kernel selection ratio: 20% at base stage, 12% at incremental stages

## Key Experimental Results

### Main Results

**Single-step Incremental Setting** (COCO 40+40):

| Method | Detector | AP | AbsGap | RelGap |
|--------|----------|----|--------|--------|
| BPF | Faster R-CNN | 34.4 | 5.8 | 14.4% |
| CL-DETR | Deformable DETR | 42.0 | 5.0 | 10.6% |
| SDDGR | Deformable DETR | 43.0 | 4.0 | 8.5% |
| GCD | Grounding DINO | 45.7 | 11.5 | 20.1% |
| ERD | YOLO-World(X) | 49.9 | 4.6 | 8.4% |
| RGR | YOLO-World(X) | 51.5 | 3.0 | 5.5% |
| **YOLO-IOD** | **YOLO-World(X)** | **53.0** | **1.5** | **2.7%** |

YOLO-IOD achieves only a 2.7% relative performance gap (approaching the joint-training upper bound of 54.5) **without any replay** (RGR relies on generative replay).

**Multi-step Incremental Setting** (Key Results):

| Setting | YOLO-IOD Final AP | RGR Final AP | YOLO-IOD RelGap | RGR RelGap |
|---------|------------------|--------------|-----------------|------------|
| 40-10 (5 steps) | 50.6 | 44.8 | 7.1% | 17.8% |
| 40-20 (3 steps) | 51.9 | 48.6 | 4.8% | 10.8% |
| 20-20 (4 steps) | 51.7 | 48.1 | 5.1% | 11.7% |
| **10-10 (8 steps)** | **49.7** | **43.4** | **8.8%** | **20.3%** |

Under the longest 10-10 setting (8 incremental stages), YOLO-IOD maintains only an 8.8% relative gap, significantly outperforming RGR's 20.3%.

### Ablation Study

**Component Ablation** (COCO 70-10 / 40-10):

| Pseudo Labels | CPR | IKS | CAKD | 70-10 AP | 40-10 AP |
|---------------|-----|-----|------|----------|----------|
| ✓ | - | - | - | 48.4 | 44.3 |
| ✓ | ✓ | - | - | 50.3 | 47.3 |
| ✓ | ✓ | ✓ | - | 51.5 | 49.1 |
| ✓ | - | - | ✓ | 50.8 | 49.2 |
| ✓ | ✓ | ✓ | ✓ | **52.4** | **50.6** |

CPR alone contributes +1.9/+3.0 AP; IKS adds +1.2/+1.8 AP on top of CPR; CAKD alone already surpasses the baseline by +2.4/+4.9 AP. All three modules together yield the best result.

**LoCo COCO Evaluation**:

| Method | COCO 40+40 AP | LoCo 40+40 AP | CoGap |
|--------|---------------|---------------|-------|
| RGR | 35.6 | 35.0 | 0.6% |
| CL-DETR | 42.0 | 40.9 | 1.1% |
| GCD | 45.7 | 44.7 | 1.0% |
| **YOLO-IOD** | **53.0** | **52.2** | **0.8%** |

All methods show AP drops on LoCo COCO, confirming the presence of data leakage in the original COCO partition. YOLO-IOD is least affected.

**IKS Kernel Selection Ratio Ablation**: $\mathcal{K}=12\%$ achieves the optimal balance — too small (5%) limits adaptability, while too large (20%) induces forgetting.

**CAKD Dual-Teacher Ablation**: In early stages, using only the current teacher performs better (promoting rapid adaptation); in later stages, using only the old teacher is superior (maintaining stability). The full CAKD consistently achieves the best performance.

### Key Findings

1. YOLO-World's pre-trained semantic knowledge provides a strong initialization for IOD, with joint-training AP reaching 54.5 (far surpassing Faster R-CNN's 40.2)
2. Decoupled handling of the three knowledge conflicts is more effective than a unified approach
3. YOLO-IOD without replay outperforms RGR, which uses replay
4. Data leakage in existing benchmarks is real, albeit modest (0.6–2.0% AP)
5. A RelGap of only 8.8% under the 8-step setting demonstrates the method's long-term stability

## Highlights & Insights

1. **Systematic Problem Identification**: The characterization of three knowledge conflicts is comprehensive and precise, with each conflict addressed by a dedicated module
2. **Dual-Teacher Design in CAKD**: The approach of feeding student features into different teachers' detection heads to realize asymmetric distillation is elegant — it leverages detection heads to naturally filter out irrelevant features
3. **LoCo COCO Benchmark**: Beyond eliminating data leakage, this benchmark incorporates category co-occurrence relationships, more faithfully reflecting real-world incremental scenarios
4. **Real-Time Inference**: Built on YOLO-World, the method achieves state-of-the-art incremental performance while preserving real-time inference speed

## Limitations & Future Work

1. Performance may degrade when switching to lighter YOLO variants, due to dependence on YOLO-World's pre-training quality
2. Fisher information computation in IKS requires an additional forward pass, increasing training overhead
3. Clustering unknown pseudo labels relies on an LLM-generated generic vocabulary, which may be unsuitable for specialized domains (e.g., medical imaging)
4. Evaluation is conducted solely on COCO; validation on additional domain-specific datasets (e.g., long-tail datasets such as LVIS) remains to be explored
5. Dual-teacher training in CAKD requires maintaining two teacher models simultaneously, incurring significant memory overhead

## Related Work & Insights

- The dual-teacher concept in BPF inspired CAKD; however, this work resolves the distillation misalignment problem through cross-stage feature transfer
- ERD's elastic response distillation adapts poorly to YOLO; this work addresses the issue via selective distillation with focal weighting
- YOLO-World's open-vocabulary capability provides the foundation for clustering unknown pseudo labels
- The graph-clustering approach underlying LoCo COCO can be generalized to the construction of other continual learning benchmarks

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Identification of three knowledge conflicts, three corresponding modules, and a new benchmark constitute comprehensive contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Single-step, multi-step, LoCo COCO, and component ablation studies provide extremely comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and framework diagrams are intuitive
- Value: ⭐⭐⭐⭐⭐ — Bringing IOD into the industrial-grade YOLO framework bridges the gap between academic research and practical application

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AKCMamba-YOLO: Selective State Space Models For Real-Time Object Detection](../../CVPR2026/object_detection/akcmamba-yolo_selective_state_space_models_for_real-time_object_detection.md)
- [\[CVPR 2026\] YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection](../../CVPR2026/object_detection/yolo-master_moe-accelerated_with_specialized_transformers_for_enhanced_real-time.md)
- [\[CVPR 2026\] YOLO-ULM: Ultra-Lightweight Models for Real-Time Object Detection](../../CVPR2026/object_detection/yolo-ulm_ultra-lightweight_models_for_real-time_object_detection.md)
- [\[CVPR 2026\] Incremental Object Detection via Future-Aware Decoupled Cross-Head Distillation](../../CVPR2026/object_detection/incremental_object_detection_via_future-aware_decoupled_cross-head_distillation.md)
- [\[CVPR 2026\] Parameterized Prompt for Incremental Object Detection](../../CVPR2026/object_detection/parameterized_prompt_for_incremental_object_detection.md)

</div>

<!-- RELATED:END -->
