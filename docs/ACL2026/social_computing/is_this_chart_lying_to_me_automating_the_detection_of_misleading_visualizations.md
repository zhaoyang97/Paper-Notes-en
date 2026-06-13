---
title: >-
  [Paper Note] Is this chart lying to me? Automating the detection of misleading visualizations
description: >-
  [ACL 2026][Social Computing][Misleading Visualizations] Ours introduces the Misviz (2,604 real-world misleading visualizations) and Misviz-synth (57,665 synthetic visualizations) benchmarks…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Misleading Visualizations"
  - "Chart Detection"
  - "Multi-modal Large Language Models"
  - "Data Visualization"
  - "Multi-label Classification"
date: 2026-05-08
content_hash: 3b7ac18ae18908ed
---

# Is this chart lying to me? Automating the detection of misleading visualizations

**Conference**: ACL 2026  
**arXiv**: [2508.21675](https://arxiv.org/abs/2508.21675)  
**Code**: [GitHub](https://github.com/UKPLab/acl2026-misviz)  
**Area**: Visualization / Misinformation Detection  
**Keywords**: Misleading Visualizations, Chart Detection, Multi-modal Large Language Models, Data Visualization, Multi-label Classification

## TL;DR

Ours introduces the Misviz (2,604 real-world misleading visualizations) and Misviz-synth (57,665 synthetic visualizations) benchmarks, covering 12 misleading types. It systematically evaluates the performance of MLLMs, rule-checkers, and image classifiers in detecting misleading charts, revealing that the task remains highly challenging.

## Background & Motivation

**Background**: Misleading visualizations are significant vehicles for misinformation on social media. They distort data and mislead readers into drawing incorrect conclusions by violating chart design principles (e.g., truncated axes, 3D effects, inconsistent scale intervals). Existing research demonstrates that both humans and MLLMs are easily deceived by such visualizations.

**Limitations of Prior Work**: The automatic detection of misleading visualizations and the identification of specific violation types are constrained by a lack of large-scale, diverse, and open datasets. Existing datasets are either small in scale (150 images), not publicly accessible, or only cover a few misleading types, which limits comparability between methods and research progress.

**Key Challenge**: Misleading features are often hidden in subtle visual details (e.g., axis scale intervals) and are highly diverse (recent taxonomies identify 70+ types), making automated detection extremely difficult.

**Goal**: To construct the first large-scale open misleading visualization benchmark and systematically evaluate the strengths and weaknesses of different detection methods.

**Key Insight**: Collect real charts from three sources (academic corpora, the WTF Visualizations website, and Reddit communities) and combine them with synthetic generation based on real data tables to build complementary benchmark pairs.

**Core Idea**: Define misleading visualization detection as a multi-label classification problem and systematically compare three detection paths—zero-shot MLLMs, rule-checkers based on axis metadata, and image-axis classifiers.

## Method

### Overall Architecture

Misviz contains 2,604 real visualizations (70% misleading + 30% normal), annotated with 12 misleading types and bounding boxes. Misviz-synth contains 57,665 synthetic visualizations, accompanied by data tables, Python code, and axis metadata. Three detection methods are evaluated: (1) Zero-shot MLLM reasoning; (2) DePlot for axis metadata extraction → Rule-checker; (3) Image (+axis) classifier.

### Key Designs

1.  **Systematic Coverage of 12 Misleading Types**:
    *   Function: Covers the most common chart misleading patterns in the real world.
    *   Mechanism: Filtered from the 74 categories in Lo et al.'s taxonomy based on four criteria: must appear frequently in the real world ($\geq 15$ instances), directly violate design rules (not purely inferential), explicitly distort data (not just reducing readability), and require no domain knowledge. The final 12 categories cover 62.3% of real-world cases.
    *   Design Motivation: To ensure both broad coverage and the feasibility of annotation and automated detection.

2.  **Synthetic Data Generation Pipeline (Misviz-synth)**:
    *   Function: Provides large-scale training data and rich metadata.
    *   Mechanism: A two-step process—first, obtain real data tables from Our World in Data and determine valid column combinations and chart types; then, use hand-written Matplotlib plotting functions to generate a visualization for each (chart type, misleading type) pair. Each instance includes the data table, code, and axis metadata to support axis extraction model training.
    *   Design Motivation: Real data annotation is costly and small-scale; synthetic data allows for large-scale generation with automatically obtained perfect annotations.

3.  **Axis-Metadata-Based Rule-Checker (Linter)**:
    *   Function: Utilizes structured axis information to detect design rule violations.
    *   Mechanism: Fine-tunes DePlot to extract axis metadata (tick labels, positions, axis names) from chart images, then applies hand-crafted rule checks for each misleading type (e.g., checking if the start value is 0 for truncated axes, or checking tick value differences for inconsistent intervals).
    *   Design Motivation: Axis metadata provides key clues for many misleading types; the rule-checker offers strong interpretability and performs excellently on synthetic data.

## Key Experimental Results

### Main Results (Misviz Test Set)

| Method | F1 | EM (Exact Match) | PM (Partial Match) |
| :--- | :--- | :--- | :--- |
| GPT-o3 | **71.3** | **24.0** | **38.2** |
| GPT-4.1 | 67.7 | 22.1 | 36.2 |
| Qwen2.5-VL-72B | 59.0 | 13.2 | 22.3 |
| InternVL3-38B | 58.3 | 6.1 | 19.9 |
| Linter (GT Axis) | — | — | — |
| Image Classifier | ~55 | — | — |

### Misviz-synth Test Set

| Method | F1 | EM |
| :--- | :--- | :--- |
| Image-Axis Classifier (GT Axis) | **~85** | **~75** |
| Linter (GT Axis) | ~80 | ~70 |
| GPT-o3 | ~70 | ~45 |

### Key Findings
*   MLLMs are the strongest on real charts (F1 71.3), but rule-checkers and classifiers are superior on synthetic charts because they can leverage training data.
*   Axis extractors trained on synthetic data do not generalize well to real-world charts, limiting the performance of rule-checkers and classifiers on Misviz.
*   Even the best models achieve only 24% EM, indicating that precise identification of all misleading types is extremely difficult.
*   "Misrepresentation" is the most common misleading type (32%) but also the hardest to detect—it requires comparing visual encoding with labeled values.
*   The majority of visualizations contain 1 misleading type (85%), while 14% contain 2, and 1% contain 3.

## Highlights & Insights
*   **Filling the Data Gap**: The first large-scale open misleading visualization benchmark, over 15 times larger than the previous largest public dataset.
*   **Comprehensive Methodology**: Systematically compares three completely different detection paths, revealing their respective strengths and weaknesses.
*   **In-depth Analysis of the Synthetic-Real Gap**: Synthetic data has training value, but generalizing to real-world charts remains challenging.
*   **Social Value**: Automated detection of misleading visualizations has direct applications in combating the spread of misinformation.

## Limitations & Future Work
*   **Covers Only 12/74 Misleading Types**: Many rare types or those requiring domain knowledge are not yet covered.
*   **Synthetic-to-Real Generalization Gap**: Axis extractors lack sufficient accuracy on real-world charts.
*   **Low EM for MLLMs**: Even the best models achieve only 24% exact match, suggesting significant room for improvement.
*   **Future Directions**: Expanding misleading type coverage, improving synthetic-to-real generalization, and combining LLM reasoning with rule-checkers.

## Related Work & Insights
*   **vs Lo and Qu (2025)**: They evaluated MLLMs using only 150 real visualizations; Misviz is 15x larger and uses a more comprehensive methodological approach.
*   **vs Maciborski et al. (2025)**: They used synthetic data + CNN training but covered only 5 misleading types and lacked evaluation on real charts.
*   **vs Rule-Checkers (Linters)**: Traditional linters require data tables or code; Misviz's linter is more practical as it extracts axis metadata directly from images.

## Rating
*   Novelty: ⭐⭐⭐⭐ First large-scale open misleading visualization benchmark; complete method comparison framework.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9+ MLLMs, multiple methods, two datasets, detailed ablation, and error analysis.
*   Writing Quality: ⭐⭐⭐⭐ Clearly structured; intuitive visual examples for the 12 misleading categories.
*   Value: ⭐⭐⭐⭐ Practical application value for anti-misinformation and data visualization education.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/social_computing/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ACL 2026\] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)

</div>

<!-- RELATED:END -->
