---
title: >-
  [Paper Note] TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Temporal VLM] TRoVe proposes an automated method for discovering static feature biases that induce systematic prediction errors in temporal VLMs. Through a dual-scoring mechanism combining an Error Contribution Score (ECS) and a Static Bias Score (SBS), TRoVe outperforms baselines by 28.6% on 101 synthetic models and successfully identifies novel biases in 7 real-world VLMs.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Temporal VLM"
  - "Static Feature Bias"
  - "Model Robustness"
  - "Bias Discovery"
  - "Video Understanding"
date: 2026-05-08
content_hash: 7ad139a4f76d8a47
---

# TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2512.01048](https://arxiv.org/abs/2512.01048)  
**Code**: [GitHub](https://github.com/Stanford-AIMI/TRoVe)  
**Area**: Multimodal VLM
**Keywords**: Temporal VLM, Static Feature Bias, Model Robustness, Bias Discovery, Video Understanding

## TL;DR
TRoVe proposes an automated method for discovering static feature biases that induce systematic prediction errors in temporal VLMs. Through a dual-scoring mechanism combining an Error Contribution Score (ECS) and a Static Bias Score (SBS), TRoVe outperforms baselines by 28.6% on 101 synthetic models and successfully identifies novel biases in 7 real-world VLMs.

## Background & Motivation

**Background** Temporal VLMs — designed for understanding visual changes across multi-image sequences — have demonstrated strong performance on tasks such as action recognition and disease progression classification, yet may exploit static feature shortcuts rather than dynamic changes for decision-making.

**Limitations of Prior Work** (1) Existing automated error discovery methods (Domino, George, etc.) are designed for single-image settings and cannot handle static biases in multi-image sequences; (2) ground-truth biases in real models are unknown, making quantitative evaluation difficult; (3) static features may appear only in a subset of frames within a sequence, further complicating detection.

**Key Challenge** Temporal tasks should inherently rely on dynamic change for classification, yet models may take shortcuts — predicting "tree climbing" based solely on background trees. The central challenge is how to automatically discover such error-inducing static biases.

**Goal** To automatically discover static feature biases learned by trained temporal VLMs that lead to systematic prediction errors.

**Key Insight** The problem is decomposed into two quantifiable sub-problems: (1) Does the presence of a given feature increase errors for a specific class? (2) Has the model learned a bias on that feature?

**Core Idea** Decompose multi-image sequences into individual frames → cluster frames to extract candidate features → apply dual ECS+SBS scoring to localize error-inducing static biases.

## Method

### Overall Architecture
TRoVe operates as follows: (1) All images from validation-set sequences are decomposed into individual frames; visual embeddings are extracted using a VLM visual encoder and clustered, with each cluster representing a candidate static feature; (2) a dual score — ECS and SBS — is computed for each cluster; (3) high-scoring (feature, class) pairs are surfaced as discovered biases.

### Key Designs

1. **Candidate Feature Extraction**:
    - Function: Identify recurring static features across the dataset.
    - Mechanism: For each image, a repeated sequence is constructed (eliminating temporal variation), embeddings are extracted via a VLM visual encoder, and spherical K-means clustering is applied. The number of clusters is selected automatically by sweeping candidate values and optimizing the silhouette coefficient.
    - Design Motivation: Reducing multi-image sequences to single frames isolates static feature information; clustering reveals common patterns shared across sequences.

2. **Error Contribution Score (ECS)**:
    - Function: Assess whether a static feature is associated with errors on a given class.
    - Mechanism: $ECS_C^y = acc_{\neg C}^y - acc_C^y$, i.e., the difference in class-$y$ accuracy between sequences containing feature $C$ and those that do not. A large difference indicates that the presence of the feature substantially increases errors on that class.
    - Design Motivation: Directly measures the causal impact of a feature on classification performance.

3. **Static Bias Score (SBS)**:
    - Function: Assess whether the model has learned a bias on the static feature.
    - Mechanism: For misclassified sequences whose frames belong to cluster $C$, purely static sequences are constructed (a single frame repeated $n_i$ times) and classified by the VLM; the average predicted confidence is taken: $SBS_C^y = \frac{1}{|C_{wrong}|}\sum_{I_i \in C_{wrong}} \text{softmax}(F([I_i,...,I_i]))_{\hat{y}_i}$. High confidence indicates that the model makes (erroneous) decisions based on static features alone.
    - Design Motivation: Temporal tasks require dynamic change for correct resolution — if a model produces high-confidence predictions on purely static inputs, it has learned a static shortcut.

### Final Score
The TRoVe score is defined as $ECS_C^y + SBS_C^y$, jointly measuring whether a feature causes errors and whether the model has learned a bias on that feature.

## Key Experimental Results

### Main Results — Synthetic Model Evaluation (P@10 / P@25 / P@100 / R-Prec)

| Method | Background P@10 | Object P@10 | Attribute P@10 |
|--------|----------------|-------------|----------------|
| Random | 20.7 | 14.3 | 16.2 |
| Domino | 48.6 | 52.2 | 63.1 |
| Dist. Failures | 62.9 | 60.4 | 69.2 |
| Confidence | 61.0 | 97.8 | 58.5 |
| **TRoVe** | **100.0** | **97.8** | **100.0** |

### Performance Improvement on Real VLMs at Test Time (Kinetics400 Acc@5)

| Model | Biased Class Label $\tilde{y}$ | Overall |
|-------|-------------------------------|---------|
| VideoCLIP-XL | 51.7 | 82.2 |
| + TRoVe | **94.4** (+82.6%) | **86.7** |
| ViCLIP-L | 71.4 | 77.1 |
| + TRoVe | **96.9** (+35.7%) | **80.7** |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| ECS only | Degraded | Without bias confirmation, coincidental errors are included |
| SBS only | Degraded | Without error association, non-harmful biases are reported |
| ECS + SBS | Best | The two scores are complementary |

### Key Findings
- TRoVe achieves near-perfect performance across all three bias types (P@10 approaching 100%), with far greater cross-category consistency than baselines.
- Non-temporal VLMs (e.g., CLIP) exhibit more static biases than temporal VLMs (e.g., VideoCLIP-XL) on average (134.5 vs. 84.5).
- Leveraging discovered biases for prompt correction improves performance on biased classes by up to 111%.

## Highlights & Insights
- The evaluation framework comprising 101 synthetic models with ground-truth annotations is itself a significant contribution, enabling rigorous quantitative assessment.
- The dual ECS+SBS scoring mechanism elegantly addresses the dual requirement of bias discovery — causal association and model introspection.
- The discovery that medical imaging VLMs also exhibit static feature biases (BioViL-T is biased toward static X-ray features of severe pneumonia) carries important clinical implications.

## Limitations & Future Work
- The current framework supports classification tasks only; generative tasks such as captioning and VQA require new scoring formulations.
- SBS relies on temperature-calibrated softmax confidence, which may be poorly calibrated for certain models.
- Clustering granularity affects discovery quality; excessively coarse clusters may conflate distinct features.

## Related Work & Insights
- **vs. Domino/George**: These methods target single-image settings and cannot detect static features localized to specific frames within a sequence.
- **vs. Li et al. (Confidence)**: This approach ranks by maximum confidence only, without considering the causal relationship to errors.
- **vs. RAVL**: Region-level information is exploited but only for single images; TRoVe extends frame-level analysis to the temporal setting.

## Rating
- Novelty: ⭐⭐⭐⭐ First automated bias discovery method targeting temporal VLMs; the problem formulation is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 101 synthetic models, 7 real VLMs, and 2 task types — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the method is described systematically.
- Value: ⭐⭐⭐⭐ Strong practical value for safety auditing of temporal VLMs prior to deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](../../CVPR2025/multimodal_vlm/revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](../../CVPR2026/multimodal_vlm/pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)

</div>

<!-- RELATED:END -->
