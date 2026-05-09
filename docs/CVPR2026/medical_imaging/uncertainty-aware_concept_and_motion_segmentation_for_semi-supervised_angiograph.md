---
title: >-
  [Paper Note] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos
description: >-
  [CVPR 2026][Medical Imaging][Semi-supervised segmentation] This paper proposes SMART, a Teacher-Student semi-supervised framework built upon SAM3's concept-prompt segmentation, integrating progressive confidence regularization and a dual-stream temporal consistency strategy to achieve state-of-the-art vessel segmentation in X-ray coronary angiography videos with minimal annotation.
tags:
  - CVPR 2026
  - Medical Imaging
  - Semi-supervised segmentation
  - coronary angiography
  - SAM3
  - temporal consistency
  - optical flow
date: 2026-05-08
content_hash: ad880b8a94e3465b
---

# Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos

**Conference**: CVPR 2026
**arXiv**: [2603.00881](https://arxiv.org/abs/2603.00881)
**Code**: [GitHub](https://github.com/qimingfan10/SMART)
**Area**: Medical Imaging
**Keywords**: Semi-supervised segmentation, coronary angiography, SAM3, temporal consistency, optical flow

## TL;DR

This paper proposes SMART, a Teacher-Student semi-supervised framework built upon SAM3's concept-prompt segmentation, integrating progressive confidence regularization and a dual-stream temporal consistency strategy to achieve state-of-the-art vessel segmentation in X-ray coronary angiography videos with minimal annotation.

## Background & Motivation

Coronary artery disease (CAD) is a leading cause of mortality worldwide, and X-ray coronary angiography (XCA) serves as the clinical gold standard for diagnosis. Accurate coronary segmentation is essential for automated diagnosis, yet it faces several challenges:

**Annotation Scarcity**: Acquiring annotated data in clinical settings is extremely costly and time-consuming, leaving large volumes of data unlabeled.

**XCA-Specific Difficulties**: Blurred boundaries, inconsistent radiation contrast, complex motion patterns, and low signal-to-noise ratio.

**Limitations of Existing SSL Methods**:
- SAM-based methods relying on geometric or feature prompts generalize poorly across institutions.
- Directly applying SAM3 to XCA sequences ignores temporal dependencies, leading to temporally inconsistent segmentation.
- Teacher model predictions in low-quality regions are unreliable, exhibiting low accuracy and high variance.

## Method

### Overall Architecture

SMART employs a two-stage training pipeline:
1. **Text-Driven Segmentation Fine-Tuning**: The teacher SAM3 is fine-tuned on annotated data using text concept prompts to adapt to the medical domain.
2. **Motion-Aware Semi-Supervised Learning**: The teacher is frozen and guides the student model to learn from unlabeled data. Only the student model is used during inference.

### Key Designs

1. **SAM3 Text Prompt Tuning (TPT)**: Rather than relying on geometric prompts (points/boxes), this module exploits SAM3's unique text concept prompting capability. The image encoder, text encoder, and detector of SAM3 are fine-tuned while all other components remain frozen. The loss is:
    $\mathcal{L}_{\text{ft}} = \lambda_1 \mathcal{L}_{\text{Dice}} + \lambda_2 \mathcal{L}_{\text{Bce}}$
   Core advantage: Text concept prompts capture the semantic understanding of visual structures, offering substantially better cross-institution generalization than geometric prompts.

2. **Progressive Confidence-Aware Consistency Regularization (CCR)**: The teacher model is perturbed $N=8$ times with noise $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ to obtain $N$ predictions, from which an ensemble mean $\bar{\mathbf{P}}$ and uncertainty weight $\boldsymbol{\mathcal{U}}$ are computed. The consistency loss assigns higher weight to uncertain regions:
    $\mathcal{L}_{\text{conf}} = \frac{\sum_{x,y} \mathcal{D}(x,y) \mathcal{U}(x,y)}{\sum_{x,y} \mathcal{U}(x,y) + N\eta} + \frac{\beta}{N} \sum_{x,y} \mathcal{U}(x,y)$
   where $\mathcal{D}(x,y) = (\sigma(S(x,y)) - \sigma(\bar{P}(x,y)))^2$ measures the consistency distance between the student and the teacher ensemble. **Design Motivation**: Teacher predictions in low-contrast regions are unreliable, necessitating adaptive supervision strength.

3. **Dual-Stream Temporal Consistency (DSTC)**: SEA-RAFT is employed to estimate forward and backward optical flows $\mathbf{F}_{t \to t+1}$ and $\mathbf{F}_{t+1 \to t}$, with bidirectional mask warping enforcing temporal consistency:
    $\mathcal{L}_{\text{opti}} = \frac{1}{2N} \sum_{x,y} \Big[\big(\mathbf{S}_t - \mathcal{W}(\mathbf{S}_{t+1}, \mathbf{F}_{t \to t+1})\big)^2 + \big(\mathbf{S}_{t+1} - \mathcal{W}(\mathbf{S}_t, \mathbf{F}_{t+1 \to t})\big)^2\Big]$
   A Flow Coherence Loss $\mathcal{L}_{\text{coh}}$ is additionally introduced to penalize boundary points that deviate from the dominant vessel motion, distinguishing foreground from background.

### Loss & Training

- Total loss: $\mathcal{L}_{\text{all}} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{Bce}} \mathcal{L}_{\text{Bce}} + \lambda_{\text{conf}} \mathcal{L}_{\text{conf}} + \lambda_{\text{opti}} \mathcal{L}_{\text{opti}} + \lambda_{\text{coh}} \mathcal{L}_{\text{coh}}$
- Weights: $\lambda_{\text{Dice}}=0.5,\ \lambda_{\text{Bce}}=0.5,\ \lambda_{\text{conf}}=0.5,\ \lambda_{\text{opti}}=0.3,\ \lambda_{\text{coh}}=0.2$
- AdamW optimizer, lr=1e-4, weight decay=0.01, batch size=4, 6k iterations
- Asymmetric data augmentation: strong augmentation for the teacher (rotation ±15°, noise σ=0.03) and weak augmentation for the student

## Key Experimental Results

### Main Results

Evaluated on XCAV (111 videos) and CAVSA (1061 videos) using only 16 annotated videos:

| Method | XCAV DSC ↑ | XCAV clDice ↑ | CAVSA DSC ↑ | CAVSA clDice ↑ |
|--------|-----------|--------------|-----------|--------------|
| UNet (Supervised) | 70.80 | 69.24 | 64.19 | 70.27 |
| Denver | 73.30 | 70.40 | 76.53 | 79.17 |
| CPC-SAM | 77.90 | 79.15 | 77.90 | 78.28 |
| **SMART (Ours)** | **84.39** | **83.01** | **91.00** | **97.73** |

Using only 14% of annotated videos, SMART surpasses the previous best method CPC-SAM by 6.49% DSC on XCAV; on CAVSA, it achieves a 13.1% DSC improvement using merely 1.5% of annotated data.

### Ablation Study

| Configuration | XCAV DSC ↑ | CAVSA DSC ↑ | Note |
|---------------|-----------|-----------|------|
| TPT + CCR (w/o DSTC) | 82.38 | 78.87 | Missing temporal consistency |
| TPT + DSTC (w/o CCR) | 76.24 | 47.77 | Unreliable pseudo-labels severely degrade performance |
| CCR + DSTC (w/o TPT) | 76.71 | 25.82 | Text concept prompts are critical for SAM3 adaptation |
| **TPT + CCR + DSTC** | **84.39** | **91.00** | All three components are indispensable |

### Key Findings

- **CCR is the core component**: Removing CCR causes a 43.23% drop in CAVSA DSC, demonstrating the critical role of regularizing teacher outputs.
- **DSTC improves spatial connectivity**: clDice improves by approximately 39%, effectively reducing fragmentation and over-segmentation.
- **Optimal perturbation count N=8**: clDice improves from 81.82% (N=2) to 83.01% (N=8).
- Text concept prompts vs. point prompts: Concept prompts exhibit markedly superior cross-institution generalization, with visible qualitative differences on the CADICA dataset.

## Highlights & Insights

- **Medical adaptation via text concept prompts**: Leveraging SAM3's semantic understanding to replace geometric prompts addresses cross-institution domain shift.
- Progressive confidence regularization simultaneously up-weights high-uncertainty regions and ensembles multiple noisy predictions, offering dual robustness enhancement.
- The dual-stream optical flow design (forward + backward) alleviates the confirmation bias inherent in unidirectional flow estimation.
- Remarkable performance under extreme label scarcity: 16 annotated videos with only 1–2 frames each suffice to substantially outperform fully supervised methods.

## Limitations & Future Work

- XCA videos contain a limited number of frames; temporal modeling capability in long-sequence scenarios remains unexplored.
- SAM3 itself incurs substantial computational overhead, potentially precluding real-time intraoperative use.
- Validation is limited to coronary angiography data; generalization to other vascular angiography scenarios requires further investigation.
- The effect of different SAM3 model scales has not been explored.

## Related Work & Insights

- Unlike geometry-prompt-based methods such as MedSAM2 and KnowSAM, SMART eliminates dependence on specific points or bounding boxes through textual semantics.
- The confidence regularization paradigm can be generalized to other Teacher-Student frameworks to handle unreliable pseudo-labels.
- The combination of SEA-RAFT optical flow and mask warping is demonstrated to be effective in medical video contexts.

## Rating

- Novelty: ⭐⭐⭐⭐ — First successful application of SAM3 concept prompts in semi-supervised medical segmentation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, comprehensive ablations, cross-institution generalization evaluation
- Writing Quality: ⭐⭐⭐⭐ — Clear method description with well-motivated component designs
- Value: ⭐⭐⭐⭐ — Highly annotation-efficient with strong clinical applicability

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[AAAI 2026\] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation](../../AAAI2026/medical_imaging/graph-theoretic_consistency_for_robust_and_topology-aware_semi-supervised_histop.md)
- [\[CVPR 2026\] Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

<!-- RELATED:END -->
