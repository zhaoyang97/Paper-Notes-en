---
title: >-
  [Paper Note] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection
description: >-
  [CVPR 2026][Object Detection][Open World Object Detection] This paper proposes the DEUS framework, which introduces ETF-Subspace Unknown Separation (EUS) to effectively separate known, unknown, and background proposals via energy scores within geometrically orthogonal known/unknown subspaces, and designs an Energy-based Known Distinction (EKD) loss to reduce cross-task interference between old and new classes during incremental learning, achieving substantial improvements in unknown object recall on OWOD benchmarks.
tags:
  - CVPR 2026
  - Object Detection
  - Open World Object Detection
  - Energy Function
  - Unknown Object Detection
  - Incremental Learning
  - Catastrophic Forgetting
date: 2026-05-08
content_hash: 5a94cd937a0a2666
---

# Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.29954](https://arxiv.org/abs/2603.29954)
**Code**: N/A
**Area**: Object Detection
**Keywords**: Open World Object Detection, Energy Function, Unknown Object Detection, Incremental Learning, Catastrophic Forgetting

## TL;DR

This paper proposes the DEUS framework, which introduces ETF-Subspace Unknown Separation (EUS) to effectively separate known, unknown, and background proposals via energy scores within geometrically orthogonal known/unknown subspaces, and designs an Energy-based Known Distinction (EKD) loss to reduce cross-task interference between old and new classes during incremental learning, achieving substantial improvements in unknown object recall on OWOD benchmarks.

## Background & Motivation

Open World Object Detection (OWOD) is a highly challenging setting that requires detectors to:
1. **Incrementally learn known classes**: continuously expand the set of recognizable categories;
2. **Detect unknown objects**: identify unseen objects without annotations;
3. **Avoid catastrophic forgetting**: retain knowledge of old classes when learning new ones.

Existing OWOD methods suffer from two core problems:

**Problem 1: Insufficient representation learning for unknown objects**
- Existing methods (including energy-based ones) rely heavily on the detector's **known-class predictions** to detect unknown objects.
- Energy is modeled only in the known space, pushing non-known objects away from known regions, but without constraints to prevent unknown objects from being confused with background.
- Result: Known, unknown, and background features are entangled in feature space, causing many unknowns to be missed or misclassified.

**Problem 2: Cross-task interference between old and new classes during memory replay**
- Memory replay alleviates forgetting of old classes, but lacks explicit regularization to prevent mutual interference between old and new classes.
- Interference worsens as more tasks and categories are added.
- Result: A trade-off exists between retaining old-class knowledge and learning new classes.

The two designs in DEUS correspond to addressing these two problems respectively.

## Method

### Overall Architecture

Built upon OrthogonalDet as the base detector, with two additional modules:
- **EUS (ETF-Subspace Unknown Separation)**: constructs orthogonal known/unknown subspaces to guide the separation of proposal features;
- **EKD (Energy-based Known Distinction)**: separates energy responses of old and new classifiers during memory replay.

### Key Designs

1. **ETF-Subspace Unknown Separation (EUS)**:
    - **Constructing orthogonal subspaces**: Simplex ETF (Equiangular Tight Frame) is used to generate $K$ equiangular basis vectors, divided into a known subspace $W_\mathcal{K}^E$ (first $K/2$ vectors) and an unknown subspace $W_\mathcal{U}^E$ (last $K/2$ vectors), with $K=128$.
    - ETF basis vectors are fixed and non-learnable, guaranteeing geometric orthogonality between the two subspaces.
    - **Dual-space energy computation**: For each proposal feature $f$, the Helmholtz free energy is computed separately in the two subspaces:
     $$E^\mathcal{K}(f) = -\log \sum_{i=1}^{K/2} \exp(W_{\mathcal{K},i}^E \cdot f)$$
     $$E^\mathcal{U}(f) = -\log \sum_{i=1}^{K/2} \exp(W_{\mathcal{U},i}^E \cdot f)$$
    - **Unknown offset**: $\Delta_u(f) = s_u(f) - s_k(f)$ (unknown score minus known score); a positive value indicates a higher likelihood of being unknown.
    - **Learning objectives**: known proposals → $\Delta_u \leq -m$; unknown proposals → $\Delta_u \geq m$; background → boundary region.
    - **Dual loss**: energy margin loss on $\Delta_u$ + subspace focal loss; the former provides the primary separation mechanism, while the latter stabilizes training.
    - **Inference-time calibration**: subspace information is injected into the detector's existing unknown logit: $z_u' = z_u + \sigma_{z_u} \tilde{\Delta}_u(f)$, where $\tilde{\Delta}_u$ is the normalized offset.

2. **Energy-based Known Distinction (EKD)**:
    - **Split classifier**: the known-class classification head is split into an old-task sub-classifier $H_{prev}$ and a new-task sub-classifier $H_{curr}$.
    - **Energy score**: $S(f;H) = \log \sum_{c=1}^{C_H} \exp(z_c(f;H))$; higher values indicate stronger affinity with the corresponding classifier.
    - **Contrastive loss**: encourages old-class proposals to score high on the old classifier and low on the new classifier, and vice versa:
     $$\mathcal{L}_{prev} = \log(1 + \exp[S(f_{prev};H_{curr}) - S(f_{prev};H_{prev})])$$
     $$\mathcal{L}_{curr} = \log(1 + \exp[S(f_{curr};H_{prev}) - S(f_{curr};H_{curr})])$$
    - Activated only during memory replay (i.e., incremental task training).

### Loss & Training

Total loss:
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{bbox} + \mathcal{L}_{EUS} + \mathcal{L}_{EKD}$$

- $\mathcal{L}_{cls}$: sigmoid focal loss
- $\mathcal{L}_{bbox}$: L1 + GIoU loss
- $\mathcal{L}_{EUS} = \mathcal{L}_{energy} + \mathcal{L}_{subspace}$ (weight 1.0)
- $\mathcal{L}_{EKD}$ (weight 1.0, enabled only during incremental tasks)
- ETF space dimension $K=128$ (64 vectors each for known and unknown subspaces)
- Implemented based on MMDetection
- Improved pseudo-label strategy: dynamically scales pseudo-label count and filters noisy detections

## Key Experimental Results

### Main Results

**M-OWODB Benchmark**:

| Method | T1 U-Rec | T1 H-Score | T2 U-Rec | T2 H-Score | T3 U-Rec | T3 H-Score | T4 Known mAP |
|------|---------|-----------|---------|-----------|---------|-----------|-------------|
| OrthogonalDet | 36.3 | 46.6 | 30.2 | 38.0 | 28.7 | 35.7 | 44.7 |
| O1O | 49.3 | 56.1 | 50.3 | 51.6 | 49.5 | 47.4 | 42.4 |
| **DEUS** | **65.1** | **65.6** | **66.2** | **59.0** | **69.0** | **58.0** | **46.0** |

**S-OWODB Benchmark**:

| Method | T1 U-Rec | T1 H-Score | T2 U-Rec | T3 U-Rec | T4 Known mAP |
|------|---------|-----------|---------|---------|-------------|
| OrthogonalDet | 24.6 | 36.6 | 27.9 | 31.9 | 46.2 |
| O1O | 49.8 | 59.1 | 51.1 | 48.1 | 45.9 |
| **DEUS** | **68.7** | **70.1** | **62.9** | **60.7** | **48.8** |

DEUS nearly doubles U-Rec (e.g., M-OWODB T1: 36.3→65.1) while maintaining competitive known mAP.

**RS-OWODB (Remote Sensing)**:

| Method | T1 H-Score | T2 H-Score | T3 H-Score | T4 mAP |
|------|-----------|-----------|-----------|--------|
| OrthogonalDet | 34.8 | 15.6 | 16.2 | 64.2 |
| **DEUS** | **62.5** | **39.4** | **40.9** | **68.3** |

### Ablation Study

| EUS | EKD | T1 U-Rec | T1 H-Score | T2 Known mAP | T3 H-Score | T4 Known mAP | Note |
|-----|-----|---------|-----------|-------------|-----------|-------------|------|
| ✗ | ✗ | 36.8 | 47.2 | 52.0 | 37.6 | 44.7 | Baseline |
| ✗ | ✓ | 36.8 | 47.2 | 52.6 | 43.9 | 45.9 | EKD improves known |
| ✓ | ✗ | 65.1 | 65.6 | 51.9 | 57.5 | 43.5 | EUS substantially improves U-Rec |
| ✓ | ✓ | **65.1** | **65.6** | **53.3** | **58.0** | **46.0** | Best combined |

### Key Findings

1. **EUS is critical for unknown object detection**: U-Rec jumps from 36.8 to 65.1, nearly doubling, demonstrating that dual-subspace modeling is far superior to using only the known space.
2. **EKD independently improves known-class performance**: regardless of whether EUS is present, EKD consistently improves mAP across tasks.
3. **The two modules are complementary**: EUS boosts unknown detection while EKD protects known performance; their combination achieves the best H-Score across all settings.
4. **Negligible computational overhead**: inference time increases by only 1.9%, FLOPs by +0.5%, and training time by +6.2%.
5. **Generalization to remote sensing**: H-Score on RS-OWODB improves from 34.8 to 62.5, demonstrating that the method is not limited to natural images.
6. **PCA visualization** clearly shows that known/unknown/background features are severely entangled in the baseline, while DEUS achieves clean three-way separation.

## Highlights & Insights

1. **Dual-subspace energy modeling**: DEUS is the first work in OWOD to explicitly model a representation space for unknown objects, rather than merely excluding them from the known space. The orthogonality guaranteed by ETF is key—it prevents the two spaces from overlapping.
2. **Unified use of energy functions**: energy functions serve dual purposes—known/unknown separation (EUS) and old/new class distinction (EKD)—forming a unified energy-based framework.
3. **Geometric advantages of ETF**: the equiangular tight frame provides fixed, uniformly distributed orthogonal basis vectors, ensuring spatial separation without learning.
4. **Calibrated inference**: injecting the subspace offset into existing logits is a simple and efficient approach that does not disrupt the detection pipeline.
5. **Improved pseudo-labels**: dynamic scaling and noise filtering serve as auxiliary contributions that also provide practical improvements to baseline performance.

## Limitations & Future Work

1. **Semantic overlap between known and unknown**: the paper acknowledges that separation remains difficult when known and unknown categories are semantically similar.
2. **ETF dimension selection**: $K=128$ is a hyperparameter that may require tuning for different datasets.
3. **EUS may slightly reduce known mAP**: ablation results show that using EUS alone causes T4 Known mAP to drop from 44.7 to 43.5, as more proposals are labeled as unknown.
4. **Validated only within the Faster R-CNN paradigm**: applicability to DETR-based open-world detectors remains unknown.
5. **Pseudo-label quality has room for improvement**: the current approach relies on a dynamic matcher; self-training or consistency regularization could be considered.

## Related Work & Insights

- **OrthogonalDet**: the base model, which decouples objectness and classification predictions via orthogonalization.
- **PROB (Zohar et al.)**: models class-agnostic objectness using a Gaussian distribution.
- **Du et al. (Unknown-Aware OD)**: energy uncertainty regularization, but only in the known space.
- **Neural Collapse / ETF**: the geometric properties of ETF in classification are cleverly leveraged to construct separated subspaces.
- Inspiration: the dual-subspace + energy paradigm may generalize to other open-world tasks (e.g., open-world segmentation, open-world tracking).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — ETF dual-subspace + energy separation is a novel design; EKD's energy-based distinction between old and new classes is also an innovative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks (M-OWODB/S-OWODB/RS-OWODB), comprehensive ablations, analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation and methodology are clearly articulated, though the notation is dense and requires careful reading.
- Value: ⭐⭐⭐⭐⭐ — Nearly doubling U-Rec for unknown detection in OWOD is a significant contribution with practical impact on open-world learning.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](ewdetr_evolving_world_object_detection.md)
- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)

<!-- RELATED:END -->
