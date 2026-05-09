---
title: >-
  [Paper Note] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration
description: >-
  [CVPR 2026][Medical Imaging][Vision-language pretraining] This paper proposes MedKCO, a knowledge-driven cognitive orchestration strategy for medical vision-language pretraining. It introduces a hierarchical curriculum (label-level ordering by diagnostic sensitivity + description-level ordering by sample representativeness) and a self-paced asymmetric contrastive loss, enabling the model to progressively learn from simple to complex concepts. MedKCO substantially outperforms baselines on zero-shot and downstream tasks across three medical imaging modalities.
tags:
  - CVPR 2026
  - Medical Imaging
  - Vision-language pretraining
  - curriculum learning
  - contrastive learning
  - cognitive orchestration
date: 2026-05-08
content_hash: d967fc99c5bc68e5
---

# MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration

**Conference**: CVPR 2026
**arXiv**: [2603.09101](https://arxiv.org/abs/2603.09101)
**Code**: [Available](https://github.com/Mr-Talon/MedKCO)
**Area**: Medical Imaging
**Keywords**: Vision-language pretraining, curriculum learning, contrastive learning, cognitive orchestration, medical imaging

## TL;DR

This paper proposes MedKCO, a knowledge-driven cognitive orchestration strategy for medical vision-language pretraining. It introduces a hierarchical curriculum (label-level ordering by diagnostic sensitivity + description-level ordering by sample representativeness) and a self-paced asymmetric contrastive loss, enabling the model to progressively learn from simple to complex concepts. MedKCO substantially outperforms baselines on zero-shot and downstream tasks across three medical imaging modalities.

## Background & Motivation

Medical vision-language pretraining (VLP), exemplified by CLIP-based medical variants such as MedCLIP, FLAIR, and KeepFIT, aims to align medical images with textual descriptions but faces unique challenges: (1) diagnostic difficulty varies significantly across diseases—"hard exudates" are directly observable on fundus photographs, whereas "glaucoma" requires deeper domain expertise; (2) sample representativeness varies considerably within the same disease category—typical samples exhibit clear features, while atypical samples are confounded by individual variation and comorbidities; (3) inter-class visual similarity in medical images is extremely high, whereas textual descriptions can distinguish diseases clearly.

Existing methods randomly mix data of all difficulty levels during training, forcing the model to learn simple and complex concepts simultaneously before foundational knowledge is established—a practice that contradicts the progressive nature of human cognition. Inspired by the "Zone of Proximal Development" theory in cognitive science, this paper designs a curriculum-based pretraining orchestration that proceeds from easy to hard.

## Method

### Overall Architecture

MedKCO is a model-agnostic pretraining strategy applicable to any medical VLP framework; the paper validates it on both CLIP and FILIP. It improves pretraining along two dimensions: (1) **data ordering**—a hierarchical curriculum controls the sequence in which data is presented, progressing from simple to complex concepts; (2) **loss function**—a self-paced asymmetric contrastive loss progressively adjusts the difficulty of contrastive learning. Pretraining data is organized into two curriculum levels based on label granularity: label-level (global diagnostic labels) and description-level (detailed descriptions containing local lesion information).

### Key Designs

1. **Label-Level Curriculum (Based on Diagnostic Sensitivity)**: Data is divided into three stages of increasing difficulty according to the diagnostic sensitivity of a given imaging modality for different diseases. Stage 1: structural features directly observable in the modality (e.g., hard exudates in color fundus photography, CFP); Stage 2: high-confidence diagnoses (>80%) requiring multiple supporting evidences and expert interpretation (e.g., diabetic retinopathy in CFP); Stage 3: conditions for which the current modality provides no definitive evidence and reliable identification requires complementary modalities (e.g., glaucoma in CFP). Classification is performed collaboratively by multiple physicians and an LLM, with final review by senior clinicians.

2. **Description-Level Curriculum (Based on Sample Representativeness)**: After acquiring global diagnostic capability, the model further learns local lesion representations. The core hypothesis is that samples farther from the class center are less affected by individual variation and comorbidities, exhibiting clearer disease characteristics and thus greater representativeness. Specifically, image features $r_i^v$ and text features $r_i^t$ are extracted using the pretrained model; each sample is assigned to a cluster via text-label similarity $c = \arg\max(r_i^t \boldsymbol{l}^T)$; the normalized distance to the cluster center $d_i = \|r_i^v - u_c\|_2 / d_{\max}$ is computed; and samples are arranged into $S$ stages in descending order of distance—representative samples (far from center, clear features) are learned first, followed by atypical samples (near center, ambiguous features).

3. **Self-Paced Asymmetric Contrastive Loss**: The standard symmetric contrastive loss $\mathcal{L}_i = \frac{1}{2}(\mathcal{L}_i^{i2t} + \mathcal{L}_i^{t2i})$ is ill-suited for medical images—at early pretraining stages, the visual encoder maps different diseases to similar representations (high inter-class similarity), introducing substantial noise in the text-to-image direction. The loss is reformulated as $\mathcal{L}_i = \frac{1}{2}(\mathcal{L}_i^{i2t} + \alpha(t,T)\mathcal{L}_i^{t2i})$, where $\alpha(t,T)$ increases linearly from 0 to 1 with training progress. Early training focuses solely on the simpler image-to-text alignment (text embeddings are more dispersed and easier to distinguish), with the harder text-to-image alignment gradually incorporated later.

### Loss & Training

- Vision-language contrastive loss with temperature parameter $\sigma$
- Self-paced asymmetric weight: $\alpha(t,T)$ follows a default linear schedule (cosine and exponential schedules are also evaluated)
- Projection head dimensionality: 512 for CLIP, 256 for FILIP
- Maximum text token length: 256
- Warm-up cosine scheduler (first epoch)
- Description-level curriculum stages: $S=2$
- Trained on a single RTX A6000 GPU

## Key Experimental Results

### Main Results

| Dataset | Metric | MedKCO (CLIP) | CLIP Baseline | Gain |
|--------|------|--------------|---------|------|
| ODIR200×3 (CFP, OOD) | ACC | **0.863** | 0.772 | +9.1% |
| REFUGE (CFP) | ACC | **0.947** | 0.897 | +5.0% |
| FIVES (CFP) | AUC | **0.729** | 0.676 | +5.3% |
| OCTID (OCT) | ACC | **0.778** | 0.709 | +6.9% |
| OCTDL (OCT, OOD) | ACC | **0.388** | 0.306 | +8.2% |
| CheXpert5×200 (CXR) | ACC | **0.526** | 0.384 | +14.2% |
| COVIDx (CXR, OOD) | ACC | **0.564** | 0.463 | +10.1% |
| **Average over 9 datasets** | — | **0.693** | 0.616 | +7.7% |

| Task | Framework | MedKCO | Best CL Baseline | Gain |
|------|---------|--------|-----------|------|
| Zero-shot classification (CLIP) | AVG | 0.693 | 0.600 (CL-log) | +9.3% |
| Zero-shot classification (FILIP) | AVG | 0.640 | 0.552 (CL-log) | +8.8% |
| Report generation (CLIP) | AVG | **0.198** | 0.188 (CLIP) | +5.3% |
| Image-text retrieval (CLIP) | AVG R@10 | **11.9** | 10.2 (CL-log) | +16.7% |

### Ablation Study

| Configuration | Key Metric (AVG ACC) | Note |
|------|-------------------|------|
| Full MedKCO | 0.693 | Best overall |
| w/o label-level curriculum | Decreased | Loss of diagnostic sensitivity ordering |
| w/o description-level curriculum | Decreased | No sample representativeness ordering |
| Symmetric contrastive loss ($\alpha=1$ fixed) | Decreased | Early-stage t2i noise interference |
| Linear vs. cosine vs. exponential schedule | Linear is best | Simple linear schedule suffices |
| Description stages S=1/2/3/4 | S=2 is best | Too few or too many stages both hurt |

### Key Findings

- MedKCO achieves the best results on all OOD datasets, demonstrating that cognitive orchestration substantially improves robustness under distribution shift.
- Existing curriculum learning methods (CL-log, CL-logit) adjust difficulty based on model feedback and are unstable in medical VLP; MedKCO defines difficulty externally via domain knowledge, yielding more reliable results.
- t-SNE visualizations show that as the curriculum progresses, the feature space of MedKCO becomes increasingly structured and separable.
- Report generation experiments indicate that MedKCO not only improves zero-shot capability but also provides better initialization weights for downstream transfer.

## Highlights & Insights

- **Cognitive Science × Medical AI**: A novel application of the "Zone of Proximal Development" theory—learning difficulty is defined by domain knowledge rather than model feedback.
- **Insight Behind the Asymmetric Contrastive Loss**: The method reveals the asymmetry between "visually compact, textually dispersed" medical representations and resolves it with a concise progressive weighting scheme.
- **Model Agnosticism**: As a pretraining strategy, MedKCO integrates seamlessly with different frameworks such as CLIP and FILIP.
- **Sample Representativeness Metric**: The assumption "farther from center = more typical" is counter-intuitive yet well-grounded in the medical domain—canonical cases exhibit salient features and reside at the periphery of the feature space.

## Limitations & Future Work

- The three-stage division of diagnostic sensitivity requires domain expert involvement and cannot be fully automated.
- The description-level curriculum depends on the quality of features from the pretrained model; poor initial features may compromise clustering and distance computation.
- Validation is limited to three modalities (CFP, OCT, CXR) and does not cover CT, MRI, pathology, or others.
- The linear schedule, while simple and effective, may not be optimal in all settings.
- The number of curriculum stages $S$ requires manual tuning based on dataset characteristics.

## Related Work & Insights

- Srinivasan et al. organize curricula by text granularity (object → instance), and Chen et al. organize them by visual task difficulty—both are related efforts that do not account for the difficulty structures specific to medicine.
- Compared to medical VLP methods such as KeepFIT and FLAIR, MedKCO focuses on *how to organize the training process* rather than *how to design model architecture*.
- Takeaway: the presentation order of pretraining data is itself an optimizable hyperparameter, particularly in the medical domain where data heterogeneity is pronounced.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel combination of cognitive orchestration and asymmetric contrastive loss with a distinctive problem framing
- Experimental Thoroughness: ⭐⭐⭐⭐ Three modalities, nine datasets, and multiple tasks (zero-shot, retrieval, generation) with comprehensive comparisons
- Writing Quality: ⭐⭐⭐⭐ Clear motivational figures and complete algorithmic pseudocode
- Value: ⭐⭐⭐⭐ The model-agnostic pretraining strategy has broad applicability to the medical VLP community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](../../ICCV2025/medical_imaging/vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/medical_imaging/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](../../AAAI2026/medical_imaging/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [\[ICCV 2025\] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](../../ICCV2025/medical_imaging/gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)

</div>

<!-- RELATED:END -->
