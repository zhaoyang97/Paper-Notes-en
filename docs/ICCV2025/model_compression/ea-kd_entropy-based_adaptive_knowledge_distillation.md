---
title: >-
  [Paper Note] EA-KD: Entropy-based Adaptive Knowledge Distillation
description: >-
  [ICCV 2025][Object Detection][Knowledge Distillation] This paper proposes EA-KD, a plug-and-play knowledge distillation method based on information entropy. It dynamically reweights distillation losses by combining the e…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Knowledge Distillation"
  - "Entropy"
  - "Adaptive Weighting"
  - "Plug-and-Play"
  - "Sample Importance"
date: 2026-05-08
content_hash: f95894634861732d
---

# EA-KD: Entropy-based Adaptive Knowledge Distillation

**Conference**: ICCV 2025
**arXiv**: [2311.13621](https://arxiv.org/abs/2311.13621)  
**Code**: [https://github.com/cpsu00/EA-KD](https://github.com/cpsu00/EA-KD)  
**Area**: Object Detection
**Keywords**: Knowledge Distillation, Entropy, Adaptive Weighting, Plug-and-Play, Sample Importance

## TL;DR

This paper proposes EA-KD, a plug-and-play knowledge distillation method based on information entropy. It dynamically reweights distillation losses by combining the entropy values of teacher and student outputs, prioritizing learning from high-entropy (high-information) samples. EA-KD consistently improves multiple KD frameworks across image classification, object detection, and LLM distillation tasks with negligible computational overhead.

## Background & Motivation

**Background**: Knowledge distillation (KD) achieves model compression by training a small student model to mimic a large teacher model. Mainstream approaches are divided into logit-based methods (aligning softened probability distributions) and feature-based methods (aligning intermediate layer features). Recent advances include DKD, MLD, CTKD, and related variants.

**Limitations of Prior Work**:
   - **Drawback of uniform distillation**: Most KD methods treat all samples equally, ignoring the differences in learning value across samples. Intuitively, this is analogous to a teacher delivering the same depth of instruction to all students rather than focusing on critical knowledge points.
   - **Inherent bias of KLD**: The standard KLD loss naturally assigns larger loss values to low-entropy (simple/high-confidence) samples (when the student is initialized as a uniform distribution, KLD $\approx \log(C)$ for low-entropy teacher outputs, while KLD $\approx 0$ for high-entropy outputs), causing training to be dominated by simple samples and obscuring knowledge transfer from high-value samples.
   - **Value of high-entropy samples**: Experiments show that high-entropy samples reside near class decision boundaries in t-SNE visualizations and exhibit larger teacher–student accuracy gaps — precisely these difficult samples carry the most critical knowledge for learning.

**Key Challenge**: The distillation process should prioritize difficult and information-rich samples, yet the mathematical properties of current loss functions produce the opposite tendency.

**Key Insight**: Information-theoretic entropy is used as a measure of sample learning value, combining both the teacher's stable assessment and the student's dynamic evolution to adaptively adjust the distillation weight for each sample.

**Core Idea**: An adaptive reweighting factor $w_{\text{EA}}$ based on teacher and student entropy that can be plugged into any KD framework in a drop-in fashion.

## Method

### Overall Architecture

EA-KD does not alter the architecture or loss function form of any KD method; it only multiplies a sample-level entropy-based weight $w_{\text{EA},n}$ during loss computation. Any KD framework — logit-based or feature-based — can be directly integrated:
$$L_{\text{EA-KD}} = \sum_{n} w_{\text{EA},n} \cdot L_{\text{KD},n}.$$

### Key Designs

1. **Entropy-based sample value quantification**:

    - **Function**: Quantify the learning value of each sample via entropy.
    - **Mechanism**: Compute the temperature-softened entropy $H_n = -\sum_{i=1}^C p_{n,i}(T') \log(p_{n,i}(T'))$ separately for teacher and student outputs, where $T'$ is a temperature parameter dedicated to entropy computation (distinct from the KD temperature $T$; optimal value $T'=3$).
    - **Design Motivation**: High entropy indicates high uncertainty and high information content, corresponding to difficult samples near class boundaries. The introduction of $T'$ allows entropy to more smoothly reflect the value gradient across different samples.

2. **Dual-perspective reweighting factor $w_{\text{EA}}$**:

    - **Function**: Fuse the teacher's stable assessment with the student's dynamic learning state.
    - **Mechanism**:
        - Base term $w_{\text{base},n} = H_n^{\mathcal{T}}$: the teacher's assessment of sample value.
        - Interaction term $w_{\text{interact},n} = \frac{H_n^{\mathcal{T}} \cdot H_n^{\mathcal{S}}}{H_{\text{ub}}}$: normalized product of teacher and student entropies.
        - Final weight $w_{\text{EA},n} = \frac{w_{\text{base}} + w_{\text{interact}}}{2}$, which can be rewritten as $\frac{1}{2}H_n^{\mathcal{T}}\!\left(1 + \frac{H_n^{\mathcal{S}}}{H_{\text{ub}}}\right)$.
    - **Design Motivation**:
        - Using $H^{\mathcal{T}}$ alone yields weights that remain fixed throughout training, failing to capture the student's learning progress (experiments show that the variance of $H^{\mathcal{S}}$ among high-$H^{\mathcal{T}}$ samples increases with epoch).
        - The interaction term moderately reduces the weight of samples the student has already mastered (low $H^{\mathcal{S}}$), while maintaining high weights for samples still being learned.
        - When $H^{\mathcal{T}}$ is low (teacher regards the sample as simple), the weight remains low regardless of the student's state.

3. **Integration with existing KD frameworks**:

    - **Function**: Serve as a plug-and-play module for any KD method.
    - **Mechanism**: Multiply the original per-sample distillation loss $L_{\text{KD},n}$ directly by $w_{\text{EA},n}$. For frameworks such as MLD+LS, the KD loss weight should be moderately reduced to avoid excessive penalization.
    - **Design Motivation**: EA-KD is complementary to methods like DKD — DKD prevents NCKD from being overshadowed by TCKD at the class level, while EA-KD prevents high-value samples from being overshadowed by simple samples at the sample level.

### Loss & Training

- No additional hyperparameters are introduced; the original KD framework's training settings are inherited.
- Entropy temperature: $T' = 3$ (CIFAR-100 / Tiny-ImageNet), $T' = 2$ (ImageNet / transformer teacher).
- Weight range: $w_{\text{EA}} \in [0, H_{\text{ub}}]$, where $H_{\text{ub}} = \log C$.

## Key Experimental Results

### Main Results

**CIFAR-100 (7 teacher–student pairs, 7 KD methods)**:

| Method | Average Gain Δ |
|--------|---------------|
| EA-KD (vs KD) | +1.48% |
| EA-CTKD (vs CTKD) | +0.63% |
| EA-DKD (vs DKD) | +0.47% |
| EA-MLD (vs MLD) | +0.38% |
| EA-ReviewKD (vs ReviewKD) | +0.38% |
| EA-FCFD (vs FCFD) | +0.42% |

All EA variants consistently improve their corresponding baselines, with an average gain of +0.56%.

**Cross-dataset / cross-task results**:

| Task | Dataset | Baseline | EA Variant | Gain |
|------|---------|----------|-----------|------|
| Image Classification | Tiny-ImageNet | 56.00 (KD) | 59.39 (EA-KD) | +3.39% |
| Image Classification | ImageNet | 71.03 (KD) | 71.79 (EA-KD) | +0.76% |
| Object Detection | MS-COCO (R101→R18) | 33.97 AP (KD) | 34.78 AP (EA-KD) | +0.81 AP |
| Object Detection | MS-COCO (R50→MV2) | 30.13 AP (KD) | 31.81 AP (EA-KD) | +1.68 AP |
| LLM Distillation | 5 datasets avg. | 17.63 (KD) | 18.34 (EA-KD) | +0.71 |

### Ablation Study

**Reweighting factor comparison** (ResNet32×4 → ResNet8×4, CIFAR-100):

| Weight | KD | MLD | MLD+LS | FCFD |
|--------|----|-----|--------|------|
| None (baseline) | 73.33 | 77.08 | 78.28 | 76.62 |
| $w_{\text{base}}$ ($H^{\mathcal{T}}$ only) | 75.14 | 77.47 | 78.30 | 77.50 |
| $w_{\text{interact}}$ ($H^{\mathcal{T}} \cdot H^{\mathcal{S}}$) | 74.76 | 77.45 | 78.20 | 77.42 |
| **$w_{\text{EA}}$** | **75.46** | **77.65** | **78.38** | **77.44** |

- Both $w_{\text{base}}$ and $w_{\text{interact}}$ are individually effective; the combined $w_{\text{EA}}$ achieves the best performance on almost all frameworks.
- Inverse weighting (assigning lower weights to high-entropy samples) leads to a notable performance drop (73.33 → 72.73), validating the core hypothesis.
- EA-DKD vs. DKD: the loss landscape is smoother, and the variance of sensitivity to the $\beta$ hyperparameter decreases from 0.31 to 0.10.
- Training time overhead is negligible (Figure 1 data).

## Personal Reflections

- **Highlights**: An extremely simple and general plug-and-play approach with clear theoretical analysis (the inherent bias of KLD toward low-entropy samples); the consistent improvements across tasks and frameworks are impressive.
- **Limitations**: Gains over already strong baselines are modest (+0.16% ~ +0.38%); the validity of sequence-level reweighting in LLM distillation settings requires further investigation.
- **Insights**: Using information entropy as a measure of sample importance is a concise and effective idea that could generalize to other sample weighting scenarios such as active learning and curriculum learning.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/object_detection/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)
- [\[CVPR 2026\] DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](../../CVPR2026/object_detection/da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)

</div>

<!-- RELATED:END -->
