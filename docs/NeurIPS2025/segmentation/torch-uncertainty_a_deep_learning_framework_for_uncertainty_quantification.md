---
title: >-
  [Paper Note] Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification
description: >-
  [NeurIPS 2025][Segmentation][Uncertainty Quantification] Torch-Uncertainty is the first unified, scalable, domain-agnostic, and evaluation-centric PyTorch/Lightning framework for uncertainty quantification (UQ)…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Uncertainty Quantification"
  - "PyTorch Framework"
  - "Deep Ensembles"
  - "Semantic Segmentation"
  - "Calibration"
date: 2026-05-08
content_hash: eb065444064151d4
---

# Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2511.10282](https://arxiv.org/abs/2511.10282)  
**Code**: [GitHub](https://github.com/ENSTA-U2IS-AI/Torch-Uncertainty)  
**Area**: Image Segmentation
**Keywords**: Uncertainty Quantification, PyTorch Framework, Deep Ensembles, Semantic Segmentation, Calibration

## TL;DR

Torch-Uncertainty is the first unified, scalable, domain-agnostic, and evaluation-centric PyTorch/Lightning framework for uncertainty quantification (UQ), integrating 6 major UQ method families, 26 evaluation metrics, and 27 plug-and-play datasets across classification, segmentation, and regression tasks, along with comprehensive benchmark results.

## Background & Motivation

Deep neural networks have demonstrated remarkable performance in computer vision and NLP, yet they remain severely limited in their ability to quantify predictive uncertainty, restricting deployment in high-stakes domains such as healthcare, autonomous driving, and finance. Although a substantial body of research on UQ methods exists, three core pain points persist:

**Fragmented implementations**: Existing UQ libraries (TorchCP, BLiTZ, Bayesian-Torch, etc.) each cover only a narrow subset of UQ method families, lacking a unified tool for seamlessly evaluating and integrating different approaches.

**Non-composable methods**: Most libraries adopt rigid architectures that make it difficult to flexibly combine multiple UQ techniques (e.g., constructing an "ensemble of Laplace approximations" or an "ensemble of MC Dropout models").

**Incomplete evaluation**: Existing frameworks lack systematic, multi-dimensional robustness evaluation covering calibration, OOD detection, selective classification, distribution-shift robustness, and related criteria.

Torch-Uncertainty addresses this gap through three design principles: **domain generality** (supporting modalities ranging from single-modal vision to sequential data), **modular UQ design** (each method implemented independently and freely composable), and **evaluation centricity** (the most comprehensive built-in metric suite, with UQ metrics tracked during training and automatic best-checkpoint saving).

## Method

### Overall Architecture

Torch-Uncertainty is built on PyTorch and Lightning. Its core architecture comprises task-specific Routines (`ClassificationRoutine`, `RegressionRoutine`, `SegmentationRoutine`, etc.), modular UQ method implementations, and a complete collection of evaluation metrics and datasets. The framework follows a unified **train → post-process → evaluate** pipeline.

### Key Designs

1. **TU Routines**: Each task routine encapsulates the training loop, UQ-aware validation metrics, post-processing, and visualization. The central innovation is the simultaneous tracking of multiple UQ metrics during validation (e.g., ECE, NLL, Brier score), with efficient multi-metric best-model saving via `CompoundCheckpoint`. **Design Motivation**: the optimal model under different metrics often appears at different epochs, so saving on a single metric risks missing the best overall state. `SegmentationRoutine` additionally computes segmentation metrics efficiently through pixel subsampling.

2. **Composability of UQ Methods**: All methods are implemented within a unified task routine; users can freely combine UQ techniques by selecting appropriate layers (`torch_uncertainty.layers`), model wrappers (`torch_uncertainty.models.wrappers`), and post-processing methods. For example, one can readily construct an ensemble of Laplace approximations or an ensemble of MC Dropout models. The six supported method families include: ensemble methods (Deep Ensembles, Packed-Ensembles, MIMO, etc.), Bayesian methods (Variational BNN, SWAG, SGLD, etc.), post-hoc calibration (Temperature Scaling, MC Dropout, Laplace approximation, etc.), data augmentation (TTA), deterministic UQ (Evidential Networks), and interval/conformal prediction — totaling 20+ concrete methods.

3. **Evaluation Metric Coverage**: 26 metrics are implemented across 7 task categories — classification (Accuracy, Brier, NLL), OOD detection (AUROC, AUPR, FPR95), selective classification (AURC, AUGRC, Coverage@Risk, Risk@Coverage), calibration (ECE, aECE), diversity, regression/depth, and segmentation (mIoU, mAcc, pixAcc) — along with efficiency metrics (parameter count, FLOPs). This constitutes the most comprehensive metric coverage among comparable libraries.

### Loss & Training

- Standard losses are applied per task (cross-entropy for classification, pixel-wise cross-entropy for segmentation, etc.).
- The key innovation lies in continuously tracking multiple validation metrics throughout training and automatically saving best checkpoints per metric.
- Mixup and other data augmentation strategies are available as built-in options.
- Post-processing methods (e.g., Temperature Scaling) are applied as optional steps after training.

## Key Experimental Results

### Main Results — Classification Benchmark (ViT-B/16, ImageNet-1K)

| Method | Accuracy (%) | ECE (%) | FarOOD AUROC (%) | Risk@80Cov (%) |
|--------|-------------|---------|-----------------|---------------|
| Single Model | 80.67 | 0.01 | 90.75 | 9.81 |
| + Temperature Scaling | 80.67 | 0.01 | 90.44 | 9.79 |
| Deep Ensemble | **82.19** | 0.03 | **92.05** | **8.54** |
| + Temperature Scaling | 82.19 | **0.01** | 91.18 | 8.49 |
| Packed Ensemble | 79.23 | 0.01 | 89.84 | 10.88 |
| MiMo | 80.59 | 0.02 | 89.13 | 9.63 |

### Ablation Study — Semantic Segmentation Benchmark (UNet, MUAD)

| Method | mIoU (%) | mAcc (%) | pixAcc (%) | ECE (%) | NLL |
|--------|---------|---------|-----------|---------|-----|
| Baseline UNet | 71.55 | 87.65 | 93.59 | 0.51 | 0.18 |
| + MC Dropout | 68.80 | 85.99 | — | — | — |
| Deep Ensembles | **Best** | — | — | — | — |
| Packed Ensembles | Near DE | — | — | Better | — |
| BatchEnsemble | — | — | — | — | — |
| MIMO | — | — | — | — | — |

### Key Findings

- **Deep Ensembles dominate overall**: they achieve the best accuracy, calibration, and OOD detection; Temperature Scaling further improves calibration (ECE from 0.03% to 0.01%).
- **Compact ensembles are competitive**: Packed Ensembles and MIMO approach Deep Ensemble performance at substantially lower cost.
- **Segmentation-specific findings**: Deep Ensembles achieve the best segmentation accuracy, whereas Packed Ensembles may offer superior calibration, attributed to their built-in augmentation effect.
- **Optimal checkpoint epoch varies by metric**: the best model under different validation metrics emerges at different training stages, validating the necessity of multi-metric checkpoint saving.

## Highlights & Insights

- **Most comprehensive UQ framework**: unified integration of 6 method families, 26 metrics, and 27 datasets represents the broadest coverage among existing UQ tools.
- **Composable design**: users can readily construct hybrid UQ methods (e.g., ensembles of Laplace approximations) that would require substantial additional implementation in other libraries.
- **High engineering quality**: 98% unit test coverage, ruff code standards, a Discord community, HuggingFace pretrained models, and Zenodo datasets.
- **Practical value**: lowers the barrier to UQ research and deployment, allowing researchers to focus on methodological innovation rather than data and evaluation infrastructure.

## Limitations & Future Work

- The segmentation benchmark is limited to UNet + MUAD, without covering larger-scale models (e.g., Mask2Former) or more mainstream datasets (e.g., Cityscapes, ADE20K).
- Gaussian process methods are currently unsupported due to scalability challenges with large models.
- The framework is oriented toward vision tasks; support for NLP and multimodal tasks warrants further development.
- Integration of UQ for recent foundation models (e.g., SAM, CLIP) is absent.

## Related Work & Insights

- Torch-Uncertainty is most closely related to Lightning-UQ-Box, yet comprehensively surpasses it in method coverage (20+ vs. fewer), number of metrics (26 vs. 9), and dataset support.
- It complements specialized libraries such as TorchCP (conformal prediction only) and BLiTZ/Bayesian-Torch (Bayesian methods only).
- By providing standardized UQ benchmarks, it facilitates fair comparison and reproducible research within the community.

## Rating

- **Novelty**: ⭐⭐⭐ A framework integration effort; individual methods are not novel, with the contribution lying in systematic unification.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Classification benchmarks are thorough; segmentation benchmarks are relatively limited; multi-dimensional evaluation is well covered.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear; library design and comparison tables are easy to interpret.
- **Value**: ⭐⭐⭐⭐⭐ Extremely high practical value for the UQ community, filling the gap left by the absence of a unified toolset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] On the Generalization of Representation Uncertainty in Earth Observation](../../ICCV2025/segmentation/on_the_generalization_of_representation_uncertainty_in_earth_observation.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[ICML 2026\] Segment Anything with Robust Uncertainty-Accuracy Correlation](../../ICML2026/segmentation/segment_anything_with_robust_uncertainty-accuracy_correlation.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
