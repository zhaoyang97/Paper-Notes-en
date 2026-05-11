---
title: >-
  [Paper Note] Rhetorical Questions in LLM Representations: A Linear Probing Study
description: >-
  [ACL 2026][rhetorical questions] This work applies linear probing to analyze how LLMs internally represent rhetorical questions (RQs), finding that RQs are linearly separable in representation space and that probes transfer across datasets. However, probes trained on different datasets learn inconsistent directions, indicating that RQs are encoded along multiple heterogeneous linear directions rather than a single unified dimension.
tags:
  - ACL 2026
  - rhetorical questions
  - linear probing
  - LLM representations
  - cross-dataset transfer
  - rhetorical analysis
date: 2026-05-08
content_hash: 1cac2a2a631f8050
---

# Rhetorical Questions in LLM Representations: A Linear Probing Study

**Conference**: ACL 2026
**arXiv**: [2604.14128](https://arxiv.org/abs/2604.14128)
**Code**: [GitHub](https://github.com/ruyi101/rq-representation-probing)
**Area**: Interpretability
**Keywords**: rhetorical questions, linear probing, LLM representations, cross-dataset transfer, rhetorical analysis

## TL;DR
This work applies linear probing to analyze how LLMs internally represent rhetorical questions (RQs), finding that RQs are linearly separable in representation space and that probes transfer across datasets. However, probes trained on different datasets learn inconsistent directions, indicating that RQs are encoded along multiple heterogeneous linear directions rather than a single unified dimension.

## Background & Motivation

**Background**: Rhetorical questions are a prevalent figure of speech in everyday communication, used to express stance, challenge, or persuade rather than to genuinely seek information. Computational linguistics research on RQs has largely focused on classification and detection tasks, training classifiers with explicit labels.

**Limitations of Prior Work**: Although LLMs routinely generate and interpret RQs in practice, virtually no research has examined how models internally represent rhetorical intent. Existing work prioritizes predictive accuracy while overlooking representational understanding.

**Key Challenge**: A natural assumption is that if RQs can be detected by a linear probe, the model must encode a single "rhetorical question direction." However, if RQs serve distinct rhetorical functions across contexts—such as discourse-level stance expression versus syntactic-level interrogative marking—this single-direction assumption may be an oversimplification.

**Goal**: To systematically address three questions: (1) At which layers does the RQ signal emerge? (2) Do different probing methods yield consistent results? (3) Are probe directions aligned when transferred across datasets?

**Key Insight**: The paper analyzes the internal representations of Qwen3-32B and Llama-3.3-70B on two social media datasets using multiple linear probes (diffMean, logistic regression, SVM), examining not only classification accuracy but also directional alignment and rank consistency across probes.

**Core Idea**: RQs are *heterogeneously encoded* in LLM representations—captured by multiple misaligned linear directions reflecting distinct rhetorical phenomena rather than a single shared dimension.

## Method

### Overall Architecture
Last-token representations are extracted from each layer of pre-trained LLMs, projected into a 64-dimensional PCA space, and then evaluated for the separability of rhetorical questions from information-seeking questions using three linear probes (diffMean, logistic regression, hinge-loss SVM). Comparison is conducted along four dimensions: AUROC, cosine similarity of directions, Spearman rank correlation, and Jaccard overlap.

### Key Designs

1. **Comparative Framework of Three Linear Probes**:

    - Function: Evaluate the linear separability of RQs from complementary perspectives.
    - Mechanism: diffMean is a training-free direction defined as $w_{\text{DM}} = \mu_+ - \mu_-$; logistic regression optimizes cross-entropy; hinge-loss SVM optimizes the margin. All three produce a linear scoring function $w^\top h(x)$ but differ in their optimization objectives.
    - Design Motivation: If the three probes agree on AUROC but disagree on direction, this demonstrates that separability does not imply directional uniqueness—precisely the core hypothesis the paper seeks to validate.

2. **Multi-Level Evaluation Metric System**:

    - Function: Distinguish between consistency in classification performance and consistency in representational direction.
    - Mechanism: AUROC measures classification performance; cosine similarity measures directional alignment; Spearman rank correlation measures global ranking consistency; Jaccard overlap measures consistency among extreme samples (top/bottom 20%).
    - Design Motivation: Conventional probing studies only report AUROC. Adding directional and ranking metrics reveals the phenomenon that high AUROC does not imply identical probe directions.

3. **Cross-Dataset Transfer Analysis**:

    - Function: Test the generality of the learned RQ direction.
    - Mechanism: Probes trained on the RQ dataset are transferred to SRAQ for evaluation, and vice versa. Because the two datasets have distinct PCA spaces, directions are first mapped back to the original embedding space before comparison.
    - Design Motivation: If a universal "RQ direction" exists, probes transferred across datasets should be directionally aligned with consistent rankings; otherwise, the encoding of RQs is context-dependent.

### Loss & Training
diffMean requires no training. Logistic regression and hinge-loss SVM are optimized on the training set, with model selection based on the validation set and results reported on the test set. All representations are projected into PCA-64 space for denoising.

## Key Experimental Results

### Main Results

| Model | Dataset | Probe | AUROC (Deep Layers) | Representation |
|-------|---------|-------|---------------------|----------------|
| Llama-3.3-70B | RQ | Hinge/Logistic | ~0.85–0.90 | last-token |
| Llama-3.3-70B | SRAQ | Hinge/Logistic | ~0.80–0.85 | last-token |
| Qwen3-32B | RQ | diffMean | ~0.80 | last-token |
| Qwen3-32B | SRAQ | diffMean | ~0.75 | last-token |
| Both models | RQ→SRAQ transfer | All | ~0.70–0.80 | last-token |

### Cross-Dataset Directional Consistency

| Analysis Dimension | Within-RQ (inter-probe) | Within-SRAQ (inter-probe) | Cross-dataset (RQ↔SRAQ) |
|--------------------|------------------------|--------------------------|-------------------------|
| Cosine similarity (hinge vs. logistic) | ~1.0 | ~1.0 | ~0.2–0.4 |
| Cosine similarity (diffMean vs. trained) | ~0.5–0.7 | ~0.3–0.5 | ~0.2–0.4 |
| Top-20% Jaccard | ~0.25 | ~0.25 | <0.20 |
| Bottom-20% Jaccard | ~0.50 | ~0.50 | ~0.30–0.40 |

### Key Findings
- **Last-token outperforms mean pooling**: In deep layers, last-token representations consistently outperform mean pooling, suggesting that the RQ signal concentrates at the end of the sequence.
- **Trained probes and diffMean learn inconsistent directions**: Despite comparable AUROC scores (on SRAQ) or notable gaps (on RQ), the cosine similarity among directions learned by the three probes is only 0.3–0.7.
- **Severe cross-dataset rank inconsistency**: Top-20% Jaccard overlap frequently falls below 0.2, indicating that the samples deemed "most rhetorical" by probes from different datasets are almost entirely non-overlapping.
- **Qualitative analysis reveals the underlying divergence**: The SRAQ direction favors discourse-level rhetoric in extended argumentative texts (where RQs drive the argument), while the RQ direction favors short, syntactically driven, locally scoped interrogative forms.

## Highlights & Insights
- **"High AUROC ≠ shared direction"**: This is an important methodological caution for probing research in general—linear separability does not imply a unique separating direction. The insight generalizes to probing studies of other linguistic properties.
- **Heterogeneity of rhetorical questions**: RQs do not constitute a monolithic property; rather, they span a spectrum from local syntactic marking to global rhetorical strategy, consistent with theoretical accounts in linguistics.
- **Asymmetry between top and bottom rankings**: Rankings of information-seeking questions are more consistent across probes, while rankings of RQs are more variable—suggesting that "non-rhetorical" is a relatively homogeneous category, whereas "rhetorical" is heterogeneous.

## Limitations & Future Work
- Experiments are conducted on only two social media datasets; formal registers such as academic papers and news articles are not examined.
- Only linear probes are employed; non-linear representational structures are excluded from analysis.
- No systematic causal intervention experiments are conducted—linear separability does not imply linear controllability.
- Future work should integrate sparse autoencoders or causal intervention methods to validate the causal efficacy of the identified RQ directions.

## Related Work & Insights
- **vs. Ikumariegbe et al. 2025**: Their work studies RQs within a QA classification framework, focusing on predictive accuracy. The present paper goes deeper into the representational level, revealing directional heterogeneity underlying classification performance.
- **vs. Marks & Tegmark 2024 (diffMean)**: This paper uses diffMean as one of its baselines but finds that diffMean and trained probe directions are inconsistent, suggesting that while diffMean is elegant, it may miss part of the signal.
- **vs. sparse autoencoder methods**: SAEs can decompose activations into interpretable feature directions; they could be applied in future work to validate the multi-direction hypothesis identified here.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic analysis of how RQs are encoded in LLM representations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive analysis across multiple probes, models, and metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical progression from phenomena to analysis to qualitative validation.
- Value: ⭐⭐⭐⭐ Informative for both probing methodology and rhetorical understanding.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](../../ICLR2026/interpretability/one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](../../ICLR2026/interpretability/dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning](../../ICLR2026/interpretability/dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ACL 2026\] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)

</div>

<!-- RELATED:END -->
