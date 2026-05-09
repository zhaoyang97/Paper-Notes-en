---
title: >-
  [Paper Note] Revisiting Semi-Supervised Learning in the Era of Foundation Models
description: >-
  [NeurIPS 2025][Model Compression][Semi-Supervised Learning] A systematic study reveals that conventional SSL methods offer limited benefit in the VFM era—PEFT on labeled data alone can match SSL—motivating V-PET: a simple and effective semi-supervised learning approach that ensembles pseudo-labels from multiple PEFT methods and multiple VFMs.
tags:
  - NeurIPS 2025
  - Model Compression
  - Semi-Supervised Learning
  - Visual Foundation Models
  - Parameter-Efficient Fine-Tuning
  - Pseudo-Label Ensembling
  - Self-Training
date: 2026-05-08
content_hash: efeb298ba97d3ea1
---

# Revisiting Semi-Supervised Learning in the Era of Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2503.09707](https://arxiv.org/abs/2503.09707)
**Code**: [https://github.com/OSU-MLB/SSL-Foundation-Models](https://github.com/OSU-MLB/SSL-Foundation-Models)
**Area**: Model Compression
**Keywords**: Semi-Supervised Learning, Visual Foundation Models, Parameter-Efficient Fine-Tuning, Pseudo-Label Ensembling, Self-Training

## TL;DR

A systematic study reveals that conventional SSL methods offer limited benefit in the VFM era—PEFT on labeled data alone can match SSL—motivating V-PET: a simple and effective semi-supervised learning approach that ensembles pseudo-labels from multiple PEFT methods and multiple VFMs.

## Background & Motivation

Semi-supervised learning (SSL) leverages abundant unlabeled data alongside limited labeled data to improve model performance, representing a key paradigm in deep learning (e.g., FixMatch, FlexMatch, SoftMatch). However, these methods are largely designed for **training neural networks from scratch**. As visual foundation models (VFMs) such as CLIP and DINOv2 become central to modern vision applications, the following questions arise:

1. Do existing SSL algorithms remain effective when VFMs are used as backbones?
2. What adaptations are needed to improve performance?
3. Can the power of VFMs be exploited to design simpler and more effective SSL algorithms?

The authors note that conventional SSL benchmarks (CIFAR-10/100, Food101) are **insufficiently challenging** in the VFM era—linear probing with a frozen VFM already achieves high accuracy, and domain coverage is too narrow.

## Method

### Overall Architecture

V-PET follows a four-stage pipeline: (a) fine-tune multiple VFMs using multiple PEFT methods on labeled data; (b) generate pseudo-labels for unlabeled data using the fine-tuned models; (c) ensemble pseudo-labels from all models; (d) self-train a final model using the ensembled pseudo-labels. The core idea is to exploit the **diversity and complementarity** of VFMs and PEFT methods to obtain high-quality pseudo-labels, thereby avoiding complex pseudo-label selection strategies.

### Key Designs

1. **VTAB-based SSL Benchmark**: Two datasets are selected from each of the three VTAB categories—Natural (DTD, SUN397), Specialized (RESISC45, Retinopathy), and Structured (CLEVR-C, KITTI)—focusing on tasks where frozen VFMs perform poorly. The benchmark spans six domains: texture recognition, scene understanding, remote sensing, medical imaging, synthetic reasoning, and autonomous driving, across 12 shot configurations. These tasks pose genuine challenges to frozen VFMs, making SSL necessary to unlock VFM potential.

2. **Unsupervised Hyperparameter Tuning Protocol**: Conventional SSL hyperparameter tuning risks data leakage when using a labeled validation set. The proposed protocol integrates 7 unsupervised criteria—5 feature-space metrics (AMI, ARI, V-Measure, FMI, BNM) and 2 logit-based metrics (RankMe, CHI)—and selects the configuration with the lowest average rank across all 7 metrics on an unlabeled validation set, requiring no label information.

3. **Mean Labels Ensembling Strategy**: Although different VFMs and PEFT methods achieve similar overall accuracy, their per-sample predictions differ substantially (complementarity), and their output distributions vary in scale (causing Mean Logits and Mean Probabilities ensembles to be dominated by certain models). Mean Labels converts each model's predictions into one-hot encodings (normalizing scale) before averaging to obtain soft pseudo-labels. This simple strategy effectively eliminates scale inconsistency, and pseudo-label quality improves as more models are included.

### Loss & Training

- The self-training stage uses **all pseudo-labels** ($\tau = 0$, no confidence threshold filtering) with a single round of self-training.
- Model weights are re-initialized from the original pre-trained checkpoint rather than from the PEFT fine-tuned weights.
- PEFT methods include LoRA and AdaptFormer; VFMs include CLIP ViT-B/16 and DINOv2 ViT-B/14.
- AdamW optimizer is used with batch size 32 and 35 training epochs.
- V-PET incurs only approximately $1.16\times$ the computational cost of other SSL methods.

## Key Experimental Results

### Main Results

| Method | Avg. over 6 datasets × 12 configs | Rank-1 Frequency | Notes |
|--------|----------------------------------|-----------------|-------|
| V-PET (CLIP+DINOv2, LoRA+AdaptFormer) | **60.5–61.0%** | Highest | Cross-VFM + PEFT ensemble |
| PET (single VFM, multi-PEFT) | 59.3–59.7% | Second | Intra-VFM ensemble |
| Labeled-Only PEFT | 55.6–55.7% | — | No SSL |
| FixMatch | 53.7% | — | Conventional SSL |
| FlexMatch | 53.9–56.2% | — | Conventional SSL |
| SoftMatch | 56.3–59.7% | — | Conventional SSL |
| FineSSL | 51.6–53.9% | — | Recent VFM SSL |

### Ablation Study

| Configuration | Description | Observation |
|---------------|-------------|-------------|
| Full Fine-Tuning vs. PEFT | SSL with VFM | PEFT consistently outperforms full fine-tuning |
| PEFT Labeled-Only vs. SSL | With/without unlabeled data | Labeled-only PEFT already matches SSL |
| ST → PET → V-PET | Increasing ensemble scale | Pseudo-label quality and performance increase monotonically |
| Mean Labels vs. Mean Logits vs. Mean Probs | Ensembling strategy | Mean Labels is best (scale-invariant) |

### Key Findings

- **Surprising finding**: Under fair comparison, full fine-tuning on labeled data alone matches or surpasses SSL methods—unlabeled data provides virtually no benefit to VFMs under existing SSL frameworks.
- **PEFT explanation**: Allowing SSL to update all VFM parameters may degrade their built-in generalization ability (due to noisy supervision from unlabeled data), whereas PEFT protects VFM generalization by constraining the update scope.
- **Diversity is key**: Prediction diversity across VFM + PEFT combinations underlies the effectiveness of ensembling—Venn diagrams of top-20% high-confidence predictions reveal limited overlap.
- V-PET does not rank first on every individual setting, but achieves the lowest mean rank and the highest rank-1 frequency, demonstrating superior stability.

## Highlights & Insights

- **Important negative result**: The work systematically validates the failure of conventional SSL in the VFM era, providing important guidance for the community—not a simple refutation, but a thorough causal analysis with a principled alternative.
- **Extreme simplicity**: V-PET requires no complex consistency regularization, data augmentation strategies, or pseudo-label filtering; it relies solely on standard PEFT + ensembling + self-training, with a clear conceptual framework and straightforward implementation.
- **Unsupervised tuning protocol**: The 7-metric rank-fusion scheme addresses the longstanding data leakage problem in SSL hyperparameter tuning and can be used independently.
- The complementarity between PEFT methods and VFMs is revealed, offering a new perspective on model selection and ensembling.

## Limitations & Future Work

- The focus is primarily on classification; dense prediction tasks such as segmentation and detection remain to be validated.
- Training multiple PEFT models prior to ensembling increases model management complexity, despite the small overall computational overhead.
- The benchmark comprises only 6 datasets and 12 configurations; broader coverage is possible.
- Large-scale VFMs (ViT-Large/Huge) and more recent VFMs (SigLIP, InternViT, etc.) are not explored.
- Mean Labels ensembling assumes independence among models; when models are highly correlated, ensemble gains may diminish.

## Related Work & Insights

- **Distinction from FineSSL**: The latter uses only the CLIP visual encoder and evaluates on simple datasets (CIFAR), whereas this work extends to multiple VFMs and more challenging benchmarks.
- The VFM-diversity-based ensembling echoes findings from "Eyes Wide Shut" (Tong et al.) and Cambrian, confirming that different VFMs possess genuinely complementary capabilities.
- The work provides direct practical guidance for SSL workflows in label-scarce scenarios such as medical imaging and remote sensing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematic re-examination of SSL in the VFM era; V-PET is simple yet deeply insightful
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 settings × multiple SSL/PEFT/VFM combinations, fair tuning protocol, and highly detailed analysis
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical progression from observation to insight to solution
- **Value**: ⭐⭐⭐⭐⭐ Dual guidance for the SSL community and VFM practitioners; a baseline-level reference work

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](specialization_after_generalization_towards_understanding_test-time_training_in_.md)

<!-- RELATED:END -->
