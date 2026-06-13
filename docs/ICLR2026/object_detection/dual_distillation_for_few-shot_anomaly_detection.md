---
title: >-
  [Paper Note] Dual Distillation for Few-Shot Anomaly Detection
description: >-
  [ICLR 2026][Object Detection][few-shot anomaly detection] This paper proposes D24FAD, a dual distillation framework that combines teacher-student distillation on query images (TSD) and student self-distillation on suppor…
tags:
  - "ICLR 2026"
  - "Object Detection"
  - "few-shot anomaly detection"
  - "dual distillation"
  - "teacher-student"
  - "self-distillation"
  - "medical imaging"
date: 2026-05-08
content_hash: 773f523b0152632a
---

# Dual Distillation for Few-Shot Anomaly Detection

**Conference**: ICLR 2026
**arXiv**: [2603.01713](https://arxiv.org/abs/2603.01713)  
**Code**: [https://github.com/ttttqz/D24FAD](https://github.com/ttttqz/D24FAD)  
**Area**: Medical Imaging / Anomaly Detection / Few-Shot Learning
**Keywords**: few-shot anomaly detection, dual distillation, teacher-student, self-distillation, medical imaging

## TL;DR
This paper proposes D24FAD, a dual distillation framework that combines teacher-student distillation on query images (TSD) and student self-distillation on support images (SSD), augmented by a learning-to-weight mechanism (L2W) for adaptive support importance estimation. The method achieves 100% AUROC on the APTOS fundus dataset with only 2-shot support.

## Background & Motivation

**Background**: Medical image anomaly detection faces the challenge of annotation scarcity. Few-shot anomaly detection leverages a minimal set of normal samples (2–8 images) to define "normality" and detect deviations from it.

**Limitations of Prior Work**: Existing methods either rely solely on teacher-student distillation (neglecting direct support reference) or solely on support matching (neglecting the transfer of pretrained knowledge), leaving these two complementary information sources unexploited jointly.

**Key Challenge**: Teacher-student distillation provides general normal-versus-anomaly discrimination but lacks knowledge of "what is normal in this domain"; support matching provides domain-specific normality references but lacks general discriminative capacity. The two are inherently complementary.

**Goal**: How to simultaneously leverage pretrained knowledge and a small set of normal samples for anomaly detection?

**Key Insight**: Design a dual-path distillation scheme — TSD learns general discrimination from the teacher, while SSD learns domain-specific normality patterns from support images.

**Core Idea**: Teacher-student distillation encodes "what is anomalous" (general knowledge), while student self-distillation encodes "what is normal" (domain-specific knowledge).

## Method

### Overall Architecture
A frozen pretrained encoder serves as the teacher, paired with a trainable student encoder. Query images are passed through both encoders; the TSD loss trains the student to replicate teacher features. Support images are passed through the student encoder, and the SSD loss aligns student query features with support features. At inference, spatial locations where the student fails to match the teacher or the support are identified as anomalies.

### Key Designs

1. **Teacher-Student Distillation (TSD)**:

    - **Function**: Trains the student to learn general representations from the pretrained teacher.
    - **Mechanism**: Minimizes the position-wise cosine distance between student and teacher query features. Normal regions align readily, while anomalous regions are difficult to align — the alignment residual serves as the anomaly score.
    - **Design Motivation**: The pretrained encoder's feature space encodes "natural image normality"; anomalies prevent the student from reproducing the teacher's output.

2. **Student Self-Distillation (SSD)**:

    - **Function**: Aligns student query features with support (normal reference) features.
    - **Mechanism**: Computes student features for $K$ support images and minimizes the average cosine distance between query and each support feature. Normal queries should be similar to normal supports, while anomalous queries should not.
    - **Design Motivation**: SSD provides a domain-specific signal for "what is normal." SSD alone already achieves strong performance (90%+ AUROC), confirming that support reference is a critical source of information.

3. **Learning-to-Weight Mechanism (L2W)**:

    - **Function**: Adaptively estimates the importance of each support sample.
    - **Mechanism**: Computes query-to-support weights via scaled dot-product attention: $w = \text{softmax}(z_\text{query} \cdot \phi(z_\text{support})^\top / \sqrt{C})$, and applies these weights to the SSD loss.
    - **Design Motivation**: Different support samples have varying reference value for different queries; adaptive weighting is more precise than uniform averaging.

### Loss & Training
$$\mathcal{L} = \lambda \cdot \mathcal{L}_\text{TSD} + \mathcal{L}_\text{SSD-L2W}$$
with $\lambda = 0.1$ (TSD is down-weighted; SSD dominates).

## Key Experimental Results

### Main Results (AUROC %)

| Dataset | K-shot | InCTRL | MVFA | **D24FAD** |
|---------|--------|--------|------|------------|
| HIS | 2 | 71.8 | 76.4 | **94.2** |
| LAG | 4 | 71.1 | 77.2 | **96.2** |
| APTOS | 2 | 89.5 | 86.1 | **100.0** |
| RSNA | 4 | 81.4 | 87.4 | **97.9** |
| Brain Tumor | 4 | 91.8 | 93.7 | **95.3** |

### Ablation Study

| Configuration | HIS (2-shot) | APTOS (2-shot) |
|---------------|-------------|----------------|
| TSD only | 66.1% | 58.2% |
| SSD only | 90.0% | 92.7% |
| SSD + TSD | 94.3% | **100.0%** |
| **SSD + TSD + L2W** | **96.7%** | **100.0%** |

### Key Findings
- SSD is the core component (90%+ alone); TSD provides complementary gains (+4–13%).
- L2W yields an average improvement of 1.91%, with a peak gain of 6.81%.
- Inference speed reaches 29.2 FPS, approximately 2× faster than MVFA.
- WideResNet-50 is the optimal backbone; larger models (e.g., Swin-B) underperform.

## Highlights & Insights
- **Complementarity of dual distillation**: TSD captures general knowledge while SSD captures domain-specific normality — conceptually simple yet highly effective. The small $\lambda = 0.1$ indicates that domain-specific signals are more informative than general ones.
- **Remarkable performance on medical imaging**: 100% AUROC on APTOS with only 2-shot support demonstrates that the framework is particularly well-suited for medical anomaly detection where visual differences are pronounced.

## Limitations & Future Work
- Supports only image-level anomaly detection; pixel-level localization is not addressed.
- Support sample quality directly impacts performance; the presence of anomalous samples in the support set can cause catastrophic failure.
- $\lambda = 0.1$ is fixed; adaptive tuning may yield further improvements.
- Evaluation is limited to medical imaging; applicability to other domains such as industrial defect detection remains untested.

## Related Work & Insights
- **vs. MVFA**: MVFA employs multi-view feature alignment, while D24FAD adopts dual distillation — a different paradigm that achieves superior results.
- **vs. InCTRL**: InCTRL is a meta-learning-based few-shot anomaly detection method; D24FAD is simpler yet consistently outperforms it.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual distillation framework is elegant and concise, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five medical datasets, multiple shot settings, and backbone ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and well-organized.
- **Value**: ⭐⭐⭐⭐⭐ Achieves remarkable performance on few-shot medical anomaly detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/object_detection/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](../../CVPR2026/object_detection/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[ICLR 2026\] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/object_detection/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)

</div>

<!-- RELATED:END -->
