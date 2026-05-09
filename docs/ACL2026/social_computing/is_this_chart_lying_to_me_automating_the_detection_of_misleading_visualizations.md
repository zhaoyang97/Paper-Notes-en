---
title: >-
  [Paper Note] Is this chart lying to me? Automating the detection of misleading visualizations
description: >-
  [ACL 2026][Social Computing][misleading visualization] Proposes Misviz (2,604 real-world misleading visualizations) and Misviz-synth (57,665 synthetic visualizations) benchmarks covering 12 misleading types, systematically evaluating MLLMs, rule-based checkers, and image classifiers for misleading chart detection, revealing the task remains highly challenging.
tags:
  - ACL 2026
  - Social Computing
  - misleading visualization
  - chart detection
  - multimodal LLM
  - data visualization
  - multi-label classification
date: 2026-05-08
content_hash: c841e6031d5f4975
---

# Is this chart lying to me? Automating the detection of misleading visualizations

**Conference**: ACL 2026  
**arXiv**: [2508.21675](https://arxiv.org/abs/2508.21675)  
**Code**: [GitHub](https://github.com/UKPLab/acl2026-misviz)  
**Area**: Social Computing  
**Keywords**: misleading visualization, chart detection, multimodal LLM, data visualization, multi-label classification

## TL;DR

Proposes Misviz (2,604 real-world misleading visualizations) and Misviz-synth (57,665 synthetic visualizations) benchmarks covering 12 misleading types, systematically evaluating MLLMs, rule-based checkers, and image classifiers for misleading chart detection, revealing the task remains highly challenging.

## Background & Motivation

**State of the Field**: Misleading visualizations are an important vehicle for disinformation on social media, distorting data through violations of chart design principles (e.g., truncated axes, 3D effects, inconsistent scale intervals), misleading readers into drawing incorrect conclusions. Prior research has shown both humans and MLLMs are susceptible to such deceptive visualizations.

**Limitations of Prior Work**: Training and evaluation for automatically detecting misleading visualizations and identifying specific violation types is limited by the lack of large-scale, diverse, open datasets. Existing datasets are either small (150 images), not openly accessible, or cover only a few misleading types, limiting method comparability and research progress.

**Root Cause**: Misleading features are often hidden in subtle visual details (e.g., axis scale intervals) and are highly diverse (the latest taxonomy identifies 70+ types), making automated detection extremely difficult.

**Paper Goals**: Construct the first large-scale open benchmark for misleading visualizations and systematically evaluate the strengths and weaknesses of different detection methods.

**Starting Point**: Collect real charts from three sources (academic corpora, WTF Visualizations website, Reddit communities), combined with synthetic generation based on real data tables, to construct complementary benchmark pairs.

**Core Idea**: Define misleading visualization detection as a multi-label classification problem and systematically compare three detection paths—zero-shot MLLMs, axis metadata-based rule checkers, and image-axis classifiers.

## Method

### Overall Architecture

Misviz contains 2,604 real visualizations (70% misleading + 30% normal), annotated with 12 misleading types and bounding boxes. Misviz-synth contains 57,665 synthetic visualizations with accompanying data tables, Python code, and axis metadata. Three detection methods: (1) MLLM zero-shot reasoning; (2) DePlot axis metadata extraction → rule checker; (3) Image (+axis) classifier.

### Key Designs

1. **Systematic Coverage of 12 Misleading Types**:

    - Function: Cover the most common chart misleading patterns in the real world
    - Mechanism: Filter from Lo et al.'s 74-class taxonomy using four criteria—must frequently occur in the real world (≥15 instances), directly violate design rules (not purely inference-based), actually distort data (not merely reduce readability), and not require domain knowledge. 12 types are selected, covering 62.3% of real cases
    - Design Motivation: Balance coverage and annotability while ensuring feasibility of automated detection

2. **Synthetic Data Generation Pipeline (Misviz-synth)**:

    - Function: Provide large-scale training data and rich metadata
    - Mechanism: Two-step process—first obtain real data tables from Our World in Data and determine valid column combinations and chart types; then use handwritten Matplotlib plotting functions to generate visualizations for each (chart type, misleading type) pair. Each instance includes data table, code, and axis metadata, supporting axis extraction model training
    - Design Motivation: Real data annotation is costly and limited; synthetic data can be generated at scale with automatic perfect annotations

3. **Axis Metadata-Based Rule Checker (Linter)**:

    - Function: Detect design rule violations using structured axis information
    - Mechanism: Fine-tune DePlot to extract axis metadata from chart images (tick labels, positions, axis names), then apply handcrafted rule checks for each misleading type (e.g., truncated axis checks whether start value is 0, inconsistent intervals checks tick value differences)
    - Design Motivation: Axis metadata is a key clue for many misleading types; rule checkers offer strong interpretability and perform excellently on synthetic data

## Key Experimental Results

### Main Results (Misviz Test Set)

| Method | F1 | EM (Exact Match) | PM (Partial Match) |
|--------|-----|-----------------|-------------------|
| GPT-o3 | **71.3** | **24.0** | **38.2** |
| GPT-4.1 | 67.7 | 22.1 | 36.2 |
| Qwen2.5-VL-72B | 59.0 | 13.2 | 22.3 |
| InternVL3-38B | 58.3 | 6.1 | 19.9 |
| Image classifier | ~55 | — | — |

### Misviz-synth Test Set

| Method | F1 | EM |
|--------|-----|-----|
| Image-axis classifier (GT axis) | **~85** | **~75** |
| Linter (GT axis) | ~80 | ~70 |
| GPT-o3 | ~70 | ~45 |

### Key Findings

- MLLMs are strongest on real charts (F1 71.3), but rule checkers and classifiers outperform on synthetic charts—because they can leverage training data
- Axis extractors trained on synthetic data do not generalize well to real charts, limiting rule checker and classifier performance on Misviz
- Even the best model achieves only 24% EM, indicating precisely identifying all misleading types is extremely difficult
- Misrepresentation is the most common misleading type (32%) but also the hardest to detect—requiring comparison between visual encoding and annotated values
- Most visualizations contain 1 misleading type (85%), 14% contain 2, and 1% contain 3

## Highlights & Insights

- **Fills a data gap**: The first large-scale open benchmark for misleading visualizations, 15x larger than the previous largest public dataset
- **Methodologically comprehensive**: Systematic comparison of three entirely different detection paths, revealing respective strengths and weaknesses
- **In-depth synthetic-to-real gap analysis**: Synthetic data has training value but generalization to real charts remains challenging
- **Social value**: Automated detection of misleading visualizations has direct applications in combating disinformation

## Limitations & Future Work

- **Covers only 12/74 misleading types**: Many rare or domain-knowledge-requiring types remain uncovered
- **Synthetic→real generalization gap**: Axis extractor accuracy on real charts is insufficient
- **Low MLLM EM**: Even the best model achieves only 24% exact match, indicating vast room for improvement
- Future directions: Expand misleading type coverage, improve synthetic→real generalization, combine LLM reasoning with rule checkers

## Related Work & Insights

- **vs Lo and Qu (2025)**: Only 150 real visualizations for MLLM evaluation; Misviz is 15x larger with more comprehensive methods
- **vs Maciborski et al. (2025)**: Synthetic data + CNN training but only 5 misleading types with no real chart evaluation
- **vs Rule checkers (linters)**: Traditional linters require data tables or code; Misviz's linter extracts axis metadata from images for greater practicality

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale open misleading visualization benchmark with complete method comparison framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9+ MLLMs, multiple methods, two datasets, detailed ablation and error analysis
- Writing Quality: ⭐⭐⭐⭐ Clear structure; visual illustrations of 12 misleading types are intuitive
- Value: ⭐⭐⭐⭐ Practical application value for anti-disinformation and data visualization education

## Related Papers

- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](../../AAAI2026/social_computing/argumentative_debates_for_transparent_bias_detection_technic.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](../../ACL2025/social_computing/implihatevid_video_hate.md)
- [\[ACL 2025\] Culture Matters in Toxic Language Detection in Persian](../../ACL2025/social_computing/culture_matters_in_toxic_language_detection_in_persian.md)

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](../../AAAI2026/social_computing/argumentative_debates_for_transparent_bias_detection_technic.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](../../AAAI2026/social_computing/reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)

<!-- RELATED:END -->
