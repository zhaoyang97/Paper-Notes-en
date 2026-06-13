---
title: >-
  [Paper Note] Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering
description: >-
  [ICCV 2025][AI Safety][continual learning] This paper proposes QUAD, which replays only questions from previous tasks (without storing images)…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "continual learning"
  - "VQA"
  - "question-only replay"
  - "attention distillation"
  - "privacy-preserving"
date: 2026-05-08
content_hash: 5bdf3c6df69b3a92
---

# Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering

**Conference**: ICCV 2025
**arXiv**: N/A  
**Code**: [GitHub](https://github.com/IemProg/QUAD)  
**Area**: Continual Learning / Visual Question Answering
**Keywords**: continual learning, VQA, question-only replay, attention distillation, privacy-preserving

## TL;DR

This paper proposes QUAD, which replays only questions from previous tasks (without storing images), combined with attention consistency distillation to preserve intra- and inter-modal attention patterns across tasks, achieving state-of-the-art performance in continual VQA under a privacy-preserving setting.

## Background & Motivation

Continual VQA requires models to retain prior knowledge (stability) while acquiring new visual-language skills (plasticity). Existing methods are primarily designed for unimodal settings and underperform in multimodal scenarios. Memory replay methods require storing complete image-question pairs, which incurs high privacy risks and storage costs — images in particular may contain sensitive information such as faces and license plates. Memory-free methods preserve privacy but suffer from severe forgetting. A key question arises: **is it truly necessary to store visual data, or is retaining past questions alone sufficient to mitigate forgetting?** Additionally, continual VQA introduces a unique challenge of "out-of-answer-set questions," where the model overfits to the current task's answer space and responds to previous tasks' questions using the current task's answer types.

## Method

### Overall Architecture

QUAD (Question-only replay with Attention Distillation) operates under a questions-only replay setting (VQACL-QR) and consists of two core components: (1) a question-only replay mechanism that selectively reuses questions from previous tasks to regularize the current model; and (2) attention consistency distillation that enforces consistency of intra- and inter-modal attention patterns across tasks.

### Key Designs

1. **Question-only Replay**: Questions from previous tasks are stored instead of image-question pairs. During training on the current task, stored past questions are paired with current images. Even with mismatched images, the task-type information embedded in questions (e.g., counting, color recognition) is sufficient to regularize the model's answer space distribution, preventing forgetting of previous answer types. Past questions are strategically selected to counter out-of-answer-set questions — preventing the model from answering all questions using the current task's answer types.

2. **Attention Consistency Distillation**: Consistency of three attention patterns is maintained across task transitions: (a) text-to-text self-attention — preserving language understanding; (b) image-to-image self-attention — preserving visual perception; (c) text-to-image cross-attention — preserving visual-language associations. The model from the previous task serves as the teacher, and KL divergence or cosine similarity is used to constrain the current model's attention patterns to match those of the teacher.

3. **Privacy-Preserving Design**: Only textual questions are stored — general in nature and non-identifiable — completely eliminating the need to store images. This satisfies data protection regulations such as GDPR and is particularly suited for sensitive domains including medical, financial, and surveillance applications.

### Loss & Training

Total loss = VQA task loss + $\lambda_1$ · question replay regularization loss + $\lambda_2$ · attention consistency distillation loss. After each task, a representative subset of questions (without images) is stored; during the next task's training, stored questions are randomly sampled for replay.

## Key Experimental Results

### Main Results

| Method | Data Storage | VQAv2 Performance | NExT-QA Performance | Privacy |
|--------|-------------|-------------------|---------------------|---------|
| Sequential FT | None | Severe forgetting | Severe forgetting | ✓ |
| VQACL (+image replay) | Images + Questions | Good | Good | ✗ |
| **QUAD (questions only)** | **Questions only** | **SOTA** | **SOTA** | **✓** |

QUAD even surpasses replay methods that store images, demonstrating that storing questions alone is sufficient.

### Ablation Study

- Question-only replay vs. no replay: significantly reduces out-of-answer-set questions.
- Attention distillation: all three attention pattern types matter, with cross-attention contributing the most.
- Number of stored questions: a moderate quantity suffices for effective regularization.
- Question selection strategy: diversity-based selection outperforms random selection.

### Key Findings

- Storing questions alone is sufficient to suppress forgetting — questions encode task-structural information.
- Out-of-answer-set questions represent a core challenge in continual VQA — models tend to answer all questions using the current task's answer types.
- Maintaining attention patterns is critical for preserving visual-language associations.
- Privacy preservation and high performance can be achieved simultaneously.

## Highlights & Insights

- The "question-only replay" setting is a novel formulation — establishing a privacy-friendly middle ground between full replay and no replay.
- The analysis and visualization of out-of-answer-set questions (confusion matrices) are intuitive and compelling.
- The application of attention distillation to multimodal continual learning is effective.
- The method practically surpasses approaches that store substantially more data.

## Limitations & Future Work

- Question selection strategy may affect performance, and optimal selection may require domain knowledge.
- Attention distillation increases training time.
- The approach assumes questions are non-sensitive; however, in certain domains, questions themselves may contain private information.
- Validation is limited to VQA tasks; generalizability to other multimodal continual learning tasks remains to be explored.

## Related Work & Insights

- VQACL establishes the benchmark and problem setting for continual VQA.
- Knowledge distillation is widely used in conventional continual learning; this paper extends it to multimodal attention.
- Privacy-preserving continual learning is an emerging direction; the question-only setting may inspire further modality-decoupled approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The question-only replay setting is a novel formulation with deep insights.
- Technical Depth: ⭐⭐⭐⭐ — The attention distillation design is well-targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, multiple baselines, and confusion matrix visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Excellent motivation figures and clear problem definitions.
- Value: ⭐⭐⭐⭐⭐ — High practical value combining privacy preservation with SOTA performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](fret_feature_redundancy_elimination_for_test_time_adaptation.md)
- [\[ICCV 2025\] IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization](iap_invisible_adversarial_patch_attack_through_perceptibility-aware_localization.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)

</div>

<!-- RELATED:END -->
