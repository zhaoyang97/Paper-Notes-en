---
title: >-
  [Paper Note] Rhetorical Questions in LLM Representations: A Linear Probing Study
description: >-
  [ACL 2026][Interpretability][Rhetorical questions] Through linear probing analysis of how LLMs internally represent rhetorical questions…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Rhetorical questions"
  - "linear probing"
  - "LLM representations"
  - "cross-dataset transfer"
  - "rhetorical analysis"
date: 2026-05-08
content_hash: 95eb2a6404d9c84e
---

# Rhetorical Questions in LLM Representations: A Linear Probing Study

**Conference**: ACL 2026  
**arXiv**: [2604.14128](https://arxiv.org/abs/2604.14128)  
**Code**: [GitHub](https://github.com/ruyi101/rq-representation-probing)  
**Area**: Interpretability  
**Keywords**: Rhetorical questions, linear probing, LLM representations, cross-dataset transfer, rhetorical analysis

## TL;DR
Through linear probing analysis of how LLMs internally represent rhetorical questions, it is discovered that rhetorical questions are linearly separable in the representation space and transferable across datasets; however, the probe directions learned from different datasets are inconsistent—rhetorical questions are encoded by multiple heterogeneous linear directions rather than a single unified dimension.

## Background & Motivation

**Background**: Rhetorical questions (RQs) are common rhetorical forms in daily communication used to express stance, challenge, or persuade rather than to seek information. Research in computational linguistics has primarily focused on classification and detection tasks using classifiers trained on explicit labels.

**Limitations of Prior Work**: Although LLMs frequently generate and understand RQs in practice, there is little research on how the model "internally represents rhetorical intent." Existing work focuses on prediction accuracy, neglecting understanding at the representation level.

**Key Challenge**: A natural hypothesis is that if RQs can be detected by linear probes, then a "rhetorical question direction" should exist within the model. However, if RQs in different contexts serve different rhetorical functions (e.g., discourse-level stance expression vs. syntactic questioning), the single-direction hypothesis may be an oversimplification.

**Goal**: To systematically answer three questions: (1) At which layers do RQ signals emerge? (2) Are different probing methods consistent? (3) Are probe directions aligned during cross-dataset transfer?

**Key Insight**: Analyze the internal representations of Qwen3-32B and Llama-3.3-70B using various linear probes (diffMean, Logistic Regression, SVM) across two social media datasets. The study focuses not only on classification accuracy but also on direction and ranking consistency between probes.

**Core Idea**: Rhetorical questions are "heterogeneously encoded" in LLM representations—captured by multiple unaligned linear directions corresponding to different rhetorical phenomena, rather than a single shared dimension.

## Method

### Overall Architecture
Last-token representations are extracted from each layer of the pre-trained LLM and projected into a 64-dimensional PCA space. Three linear probes (diffMean, Logistic Regression, and hinge loss SVM) are used to evaluate the separability between RQs and information-seeking questions. Comparisons are conducted across four dimensions: AUROC, directional cosine similarity, Spearman rank correlation, and Jaccard overlap.

### Key Designs

1.  **Comparison Framework of Three Linear Probes**:
    - **Function**: Evaluate the linear separability of RQs from different perspectives.
    - **Mechanism**: diffMean is a training-free direction based on the difference of class means $w_{\text{DM}} = \mu_+ - \mu_-$; Logistic Regression optimizes cross-entropy; hinge loss (linear SVM) optimizes the margin. All three yield a linear scoring function $w^\top h(x)$, but with different optimization objectives.
    - **Design Motivation**: If the three probes are consistent in AUROC but inconsistent in direction, it demonstrates that separability does not imply direction uniqueness—the core hypothesis of this work.

2.  **Multi-level Evaluation Metric System**:
    - **Function**: Distinguish between "consistency in classification performance" and "consistency in representation direction."
    - **Mechanism**: AUROC measures classification performance; cosine similarity measures directional alignment; Spearman rank correlation measures global ranking consistency; Jaccard overlap measures consistency among extreme samples (top/bottom 20%).
    - **Design Motivation**: Traditional probing studies only consider AUROC. Adding directional and ranking metrics reveals that high AUROC does not necessarily mean identical directions.

3.  **Cross-Dataset Transfer Analysis**:
    - **Function**: Test the universality of the rhetorical question direction.
    - **Mechanism**: Probes are trained on the RQ dataset and transferred to the SRAQ dataset (and vice versa). Since PCA spaces differ between datasets, directions are mapped back to the original embedding space for comparison.
    - **Design Motivation**: If a universal "rhetorical question direction" exists, probe directions should align and rankings should be consistent after transfer; otherwise, the encoding is context-dependent.

### Loss & Training
The diffMean method requires no training. Logistic Regression and hinge loss are optimized on the training set, with models selected via a validation set and results reported on the test set. All representations are projected into a PCA-64 space for noise reduction.

## Key Experimental Results

### Main Results

| Model | Dataset | Probe | AUROC (Deep Layers) | Representation Choice |
|-------|---------|-------|--------------------|-----------------------|
| Llama-3.3-70B | RQ | Hinge/Logistic | ~0.85-0.90 | last-token |
| Llama-3.3-70B | SRAQ | Hinge/Logistic | ~0.80-0.85 | last-token |
| Qwen3-32B | RQ | diffMean | ~0.80 | last-token |
| Qwen3-32B | SRAQ | diffMean | ~0.75 | last-token |
| Both Models | RQ→SRAQ Transfer | All | ~0.70-0.80 | last-token |

### Cross-Dataset Directional Consistency

| Analysis Dimension | Between Probes (RQ) | Between Probes (SRAQ) | Cross-Dataset (RQ↔SRAQ) |
|--------------------|---------------------|-----------------------|-------------------------|
| Cosine Sim (Hinge vs Logistic) | ~1.0 | ~1.0 | ~0.2-0.4 |
| Cosine Sim (diffMean vs Trained) | ~0.5-0.7 | ~0.3-0.5 | ~0.2-0.4 |
| Top-20% Jaccard | ~0.25 | ~0.25 | <0.20 |
| Bottom-20% Jaccard | ~0.50 | ~0.50 | ~0.30-0.40 |

### Key Findings
- **Last-token outperforms mean pooling**: In deep layers, last-token representations consistently outperform mean pooling, indicating that RQ signals are concentrated at the end of the sequence.
- **Trained probes and diffMean directions are inconsistent**: Despite similar AUROCs (on SRAQ) or small gaps (on RQ), the cosine similarity of directions learned by the three probes is only between 0.3 and 0.7.
- **Ranking is extremely inconsistent across datasets**: Jaccard overlap for top-20% samples is often below 0.2, implying that samples considered "most rhetorical" by two probes hardly overlap.
- **Qualitative analysis reveals essential differences**: The SRAQ direction prefers discourse-level rhetoric in long arguments (RQs driving an argument), while the RQ direction prefers short, syntax-driven local interrogative forms.

## Highlights & Insights
- **Insight: "High AUROC $\neq$ Shared Direction"**: This serves as a reminder for probing methodologies—linear separability does not guarantee a single separable direction. This can be generalized to probing other linguistic attributes.
- **Heterogeneity of RQs**: Rhetorical questions are not a single attribute but a spectrum ranging from local syntactic markers to global rhetorical strategies, aligning with linguistic theory.
- **Asymmetry of top vs bottom**: Rankings for information-seeking questions are more consistent, while rankings for RQs are more inconsistent—suggesting that "non-rhetorical" is relatively homogeneous, whereas "rhetorical" is heterogeneous.

## Limitations & Future Work
- Experiments were restricted to two social media datasets, excluding formal styles (e.g., academic papers or news).
- Only linear probes were used, excluding non-linear representation structures.
- No systematic causal intervention experiments were performed—linear separability does not equal linear controllability.
- Future work should utilize Sparse Autoencoders (SAEs) or causal intervention methods to verify the causal efficacy of these rhetorical directions.

## Related Work & Insights
- **vs. Ikumariegbe et al. 2025**: They studied RQs using a QA classification framework focused on accuracy; this paper delves into the representation level to reveal directional heterogeneity.
- **vs. Marks & Tegmark 2024 (diffMean)**: This paper uses their diffMean method as a baseline but finds it inconsistent with trained probes, suggesting that diffMean may miss specific signals despite its simplicity.
- **vs. Sparse Autoencoder (SAE) methods**: SAEs can decompose activations into interpretable feature directions, which could be used to validate the multi-direction hypothesis proposed here.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic analysis of RQ encoding in LLM representations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive analysis with multiple probes, models, and metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain, progressing from phenomena to analysis and qualitative verification.
- Value: ⭐⭐⭐⭐ Stimulating for both probing methodology and rhetorical understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](../../ICLR2026/interpretability/one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](../../ICLR2026/interpretability/dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ICML 2026\] What Linear Probes Miss: Multi-View Probing for Weight-Space Learning](../../ICML2026/interpretability/what_linear_probes_miss_multi-view_probing_for_weight-space_learning.md)
- [\[ACL 2026\] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)

</div>

<!-- RELATED:END -->
