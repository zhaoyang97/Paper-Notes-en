---
title: >-
  [Paper Note] Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes
description: >-
  [CVPR 2026 Workshop (PHAROS-AIF-MIH)][Medical Imaging][Vision Transformer] This work presents the first robustness extension evaluation of ZACH-ViT, a compact permutation-invariant ViT architecture, in low-data medical imaging settings. Across 7 MedMNIST datasets, ZACH-ViT ranks first under both clean and common corruption conditions (Mean Rank 1.57), ranks first under FGSM (2.00), and second under PGD (2.29).
tags:
  - CVPR 2026 Workshop (PHAROS-AIF-MIH)
  - Medical Imaging
  - Vision Transformer
  - Robustness
  - Medical Image Classification
  - Adversarial Attack
  - Low-Data
  - Permutation Invariance
date: 2026-05-08
content_hash: 86e61a75d2d30d9b
---

# Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes

**Conference**: CVPR 2026 Workshop (PHAROS-AIF-MIH)
**arXiv**: [2604.06099](https://arxiv.org/abs/2604.06099)
**Code**: None
**Area**: Medical Image Classification / Robustness Evaluation
**Keywords**: Vision Transformer, Robustness, Medical Image Classification, Adversarial Attack, Low-Data, Permutation Invariance

## TL;DR

This work presents the first robustness extension evaluation of ZACH-ViT, a compact permutation-invariant ViT architecture, in low-data medical imaging settings. Across 7 MedMNIST datasets, ZACH-ViT ranks first under both clean and common corruption conditions (Mean Rank 1.57), ranks first under FGSM (2.00), and second under PGD (2.29).

## Background & Motivation

**Background**: Standard ViT designs explicitly encode spatial order via positional encoding (PE) and [CLS] tokens. However, in medical imaging, diagnostically relevant structures may be loosely distributed, spatially disordered, or highly variable across acquisition protocols. ZACH-ViT, as a compact permutation-invariant ViT, removes PE and the [CLS] token, replacing token aggregation with global average pooling. Its predecessor validated regime-dependent performance on clean data.

**Limitations of Prior Work**: (1) The original ZACH-ViT work evaluated only clean test set performance without examining robustness; (2) in clinical AI deployment, models encounter real-world perturbations such as acquisition device variation, compression artifacts, brightness/contrast shifts, and sensor noise; (3) for compact edge-deployment models, clean performance alone is insufficient to assess practical utility.

**Key Challenge**: The permutation-invariant design that removes positional encoding proves effective under clean evaluation, yet whether this inductive bias choice retains its advantage under real-world perturbations remains entirely unknown.

**Goal**: To systematically evaluate the robustness of ZACH-ViT under common image corruptions and adversarial attacks, addressing the question of whether the clean-performance advantage of permutation-invariant ViTs extends to robustness.

**Key Insight**: Under a controlled MedMNIST few-shot setting, four compact backbone networks are evaluated with a standardized robustness protocol across four conditions (clean / mean corruption / mean FGSM / mean PGD), separating model behavior along three independent dimensions.

**Core Idea**: Removing positional encoding reduces reliance on fragile spatial correlations; this design advantage persists not only under clean evaluation but also under real-world perturbations.

## Method

### Overall Architecture

This paper does not propose a new architecture; its core contribution is a robustness evaluation. Four compact backbones trained from scratch (all with < 1M parameters) are compared under a unified experimental setup: ZACH-ViT (permutation-invariant ViT), ABMIL (attention-based MIL pooling), Minimal-ViT (standard compact ViT), and TransMIL (Transformer-based MIL). All models share the same few-shot training protocol and evaluation pipeline.

### Key Designs

1. **Datasets and Few-Shot Protocol**
    - 7 MedMNIST datasets: BloodMNIST, PathMNIST, BreastMNIST, PneumoniaMNIST, DermaMNIST, OCTMNIST, and OrganAMNIST
    - Only 50 training samples per class; validation and test sets remain unchanged; batch size 16; 23 training epochs; 5 random seeds {3, 5, 7, 11, 13}; fixed hyperparameters
    - Binary classification evaluated with AUC@0.5; multi-class classification evaluated with Macro-F1
2. **Four Evaluation Conditions**
    - **Clean**: Performance on the original test set
    - **Corruption mean**: Average performance across Gaussian noise, Gaussian blur, brightness/contrast adjustment, JPEG compression, and cutout, each at three severity levels
    - **FGSM mean**: Average performance over four perturbation magnitudes $\epsilon \in \{1/255, 2/255, 4/255, 8/255\}$
    - **PGD mean**: Same $\epsilon$ values, 10-step attack, step size $\epsilon/4$, projected back onto the $L_\infty$ ball
3. **Mean Rank Aggregation**
    - Mean Rank across 7 datasets is used as a dataset-agnostic summary metric (lower is better), avoiding misinterpretation caused by absolute metric differences across heterogeneous tasks
    - Retention (performance preserved relative to each model's own clean baseline) is also reported to reveal "starting-point effects"

### Loss & Training

Each model is trained with its standard classification loss. The key methodological contribution lies in variable control: all models share the same training protocol, seeds, and evaluation pipeline, ensuring that observed differences stem solely from architectural design choices.

## Key Experimental Results

### Main Results

| Condition | ZACH-ViT | ABMIL | TransMIL | Minimal-ViT |
|-----------|----------|-------|----------|-------------|
| Clean Mean Rank | **1.57** | 3.29 | 1.71 | 3.43 |
| Corruption Mean Rank | **1.57** | 3.14 | 2.00 | 3.29 |
| FGSM Mean Rank | **2.00** | 3.00 | 2.43 | 2.57 |
| PGD Mean Rank | 2.29 | **2.00** | 2.86 | 2.86 |

Retention (performance preserved relative to each model's own clean baseline):

| Model | Corruption | FGSM | PGD |
|-------|-----------|------|-----|
| ZACH-ViT | 0.92 | 0.23 | 0.18 |
| ABMIL | **0.96** | **0.31** | **0.30** |
| TransMIL | 0.91 | 0.20 | 0.15 |
| Minimal-ViT | 0.91 | 0.17 | 0.13 |

### Ablation Study

This paper uses cross-dataset performance variation across models as an implicit ablation:

| Dataset | Clean ZACH-ViT Advantage | Corruption ZACH-ViT Advantage |
|---------|--------------------------|-------------------------------|
| DermaMNIST | Best (0.301) | Best (0.272) |
| OCTMNIST | Best (0.304) | Best (0.255) |
| PneumoniaMNIST | Best (0.813) | Tied with TransMIL (0.775) |
| PathMNIST | TransMIL best | TransMIL best |

### Key Findings

- ZACH-ViT ranks first under both clean and common corruption conditions, indicating that the advantage of the permutation-invariant design extends across both clean and robust evaluation dimensions.
- Robustness advantages persist across multiple corruption types (Gaussian noise, blur, brightness/contrast, JPEG compression, and cutout).
- All models suffer substantial performance degradation under adversarial attacks — adversarial robustness remains an open problem for all compact models.
- ABMIL achieves the strongest PGD robustness but has the lowest clean starting point — high retention does not imply high absolute performance.

## Highlights & Insights

- **Core Insight**: Removing positional encoding reduces reliance on fragile positional correlations, leading the model to rely more on local visual evidence and compositional statistical features — a property that naturally benefits performance under real-world perturbations.
- The evaluation philosophy merits broader adoption: "inductive bias alignment" rather than "universal benchmark dominance" should serve as the standard for architecture assessment.
- Evaluating clean performance and robustness separately is a sound paradigm — a model with high retention may simply have a lower clean starting point.
- The findings have practical implications for edge deployment in medical AI: compact, robust, and data-efficient models are better suited to clinical settings.

## Limitations & Future Work

- Only compact models trained from scratch (< 1M parameters) are evaluated; pre-trained large models and transfer learning settings are excluded.
- All datasets are drawn from MedMNIST; external validation and prospective clinical evaluation are absent.
- The adversarial analysis is empirical rather than certified (no adversarial defenses such as PGD-AT are applied); results should be interpreted as stress tests.
- Fairness and subgroup performance differences are not assessed.
- Hardware-level metrics relevant to edge deployment — such as inference latency and memory footprint — are not reported.
- The contribution of permutation invariance itself is not isolated from other design choices in ZACH-ViT.

## Related Work & Insights

- **ZACH-ViT predecessor**: Established regime-dependent clean performance; this work extends the evaluation to the robustness dimension and confirms that the advantage transfers.
- **MedMNIST-C**: Proposed a benchmark for evaluating robustness under realistic image degradation; the evaluation protocol in this work is aligned with that framework.
- **ABMIL vs. TransMIL**: Different aggregation strategies (attention pooling vs. Transformer-based inter-instance modeling) provide a meaningful comparison across the architecture design space.
- The corruption-plus-adversarial evaluation protocol proposed here can serve as a template for the PHAROS community to assess compact medical models.

## Rating

- Novelty: ⭐⭐⭐ No new architecture is proposed; the contribution is a robustness extension evaluation of an existing method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 datasets, 4 evaluation conditions, 5 seeds, and 4 baselines — a relatively systematic study.
- Writing Quality: ⭐⭐⭐⭐ Clear and appropriately cautious; conclusions are carefully worded without overclaiming.
- Value: ⭐⭐⭐ Workshop-level contribution that confirms existing findings extend to the robustness dimension.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes](../../AAAI2026/medical_imaging/distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](multimodalpfn_extending_prior-data_fitted_networks_for_multimodal_tabular_learni.md)
- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](../../AAAI2026/medical_imaging/denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)

<!-- RELATED:END -->
