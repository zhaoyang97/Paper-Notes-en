---
title: >-
  [Paper Note] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations
description: >-
  [ACL 2026][NLP Understanding][Annotation Disagreement] The LiTEx reasoning taxonomy is extended from "variation within the same label" to "label disagreement" scenarios. It is discovered that annotators may have differen…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Annotation Disagreement"
  - "Natural Language Inference"
  - "LiTEx Taxonomy"
  - "Reasoning Strategies"
  - "Human Annotation Variation"
date: 2026-05-08
content_hash: 09ecaf5bfd125963
---

# Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.16458](https://arxiv.org/abs/2510.16458)  
**Code**: None  
**Area**: NLI / Annotation Analysis  
**Keywords**: Annotation Disagreement, Natural Language Inference, LiTEx Taxonomy, Reasoning Strategies, Human Annotation Variation

## TL;DR

The LiTEx reasoning taxonomy is extended from "variation within the same label" to "label disagreement" scenarios. It is discovered that annotators may have different labels but similar reasoning, and the consistency of reasoning categories reflects the semantic similarity of explanations better than label consistency does.

## Background & Motivation

**Background**: Annotation variations are prevalent in NLI datasets; understanding these disagreements is crucial for building reliable NLU systems. Explanation-based methods reveal the nature of disagreement by analyzing the reasoning behind annotator decisions.

**Limitations of Prior Work**: The LiTEx taxonomy classifies free-text explanations into 8 reasoning strategies, but it was previously used only for analyzing "consistent label, different explanation" (within-label variation), ignoring label inconsistency itself.

**Key Challenge**: Label inconsistency may mask reasoning consistency (the same reasoning leading to different labels), while label consistency might mask reasoning disagreement (different reasoning yielding the same label by chance). Looking at labels alone cannot reveal the true cognitive disagreement.

**Goal**: Extend LiTEx to label variation scenarios, and analyze NLI annotation variation from three dimensions: labels, reasoning categories, and explanation text similarity.

**Key Insight**: Annotate LiTEx categories on two NLI datasets with explanations (LiveNLI and VariErr) and track individual annotator label and reasoning strategy preferences.

**Core Idea**: The consistency of reasoning categories reflects the semantic similarity between explanations better than label consistency itself, indicating that more attention should be paid to the reasoning process rather than the final labels.

## Method

### Overall Architecture

The LiTEx taxonomy is applied to annotate explanations across three datasets (e-SNLI, LiveNLI, VariErr). Variations are then analyzed across three dimensions: (1) NLI label consistency; (2) reasoning category consistency (LiTEx); and (3) semantic similarity of explanation text. Behavior patterns are revealed by tracking individual annotators.

### Key Designs

1.  **Cross-dataset Expansion of LiTEx Taxonomy**:
    - **Function**: Applying the reasoning taxonomy originally developed on e-SNLI to LiveNLI and VariErr.
    - **Mechanism**: 8 reasoning categories are divided into textual (coreference, syntax, semantics, pragmatics, absence of mention, logical contradiction) and world knowledge (factual knowledge, reasoning knowledge). Trained annotators classify all explanations.
    - **Design Motivation**: To verify the cross-dataset generalizability of LiTEx while extending its scope to label variation scenarios.

2.  **Multi-dimensional Consistency Analysis**:
    - **Function**: Revealing the asymmetric relationship between label consistency and reasoning consistency.
    - **Mechanism**: Comparing the consistency of annotators on the same NLI instance across three dimensions—cases may arise where "labels differ but reasoning categories are the same" or "labels are the same but reasoning categories differ."
    - **Design Motivation**: A single dimension of consistency measurement is insufficient to reveal the true nature of annotation disagreement.

3.  **Individual Annotator Tracking**:
    - **Function**: Discovering systematic preferences of annotators.
    - **Mechanism**: Tracking label distribution and reasoning category preferences of 4 annotators in LiveNLI and 4 in VariErr to reveal individual consistency patterns.
    - **Design Motivation**: Annotation disagreement may stem not only from text ambiguity but also from the individual reasoning styles of annotators.

### Loss & Training

This work is an empirical analysis study and does not involve model training. Inter-annotator agreement is measured using Cohen's Kappa (LiveNLI $\kappa=0.828$, VariErr $\kappa=0.792$).

## Key Experimental Results

### Main Results

| Dataset | Annotation Count | $\kappa$ value | Main Findings |
| :--- | :--- | :--- | :--- |
| e-SNLI | Original labels | - | Reasoning knowledge and absence of mention are the primary categories |
| LiveNLI | 1404 pairs | 0.828 | Absence of mention biases toward the neutral label |
| VariErr | 1933 pairs | 0.792 | Absence of mention is the most frequent category |

### Key Findings

| Finding | Description |
| :--- | :--- |
| Inconsistent labels but consistent reasoning | Annotators use the same reasoning strategy but arrive at different labels, showing disagreement lies in judgment rather than understanding |
| Stable reasoning category-label co-occurrence | Despite differences in absolute distributions across datasets, the label distribution for each reasoning category is highly consistent |
| Significant individual label preferences | For example, VariErr annotator 2 has a nearly 60% preference for the neutral label, and w7 has a 52% preference |
| Reasoning similarity > Label similarity | Reasoning category consistency predicts the semantic similarity of explanations more effectively |

### Key Findings
- The LiTEx taxonomy generalizes well across datasets; category-label co-occurrence patterns are highly consistent across all three datasets.
- "Inconsistent labels but similar explanations" occurs frequently, suggesting that surface disagreement may mask deep-level understanding consistency.
- The strong correlation between the "Absence of Mention" category and the neutral label is consistent across all datasets.
- Individual annotators exhibit stable label preferences and reasoning strategy preferences.

## Highlights & Insights
- The core finding that "labels do not represent understanding" challenges the traditional practice of treating labels as ground truth.
- Explanations are not merely tools for interpretability, but a window into understanding the nature of annotation disagreement.
- Individual annotator tracking reveals systematic personal preferences, which has direct implications for the design of annotation processes.

## Limitations & Future Work
- The study only covers English NLI datasets; cross-lingual generalizability remains unverified.
- The 8 categories of LiTEx may not encompass all reasoning types.
- The number of annotators is limited (only 4 tracked per dataset), which restricts statistical power.
- Future work could extend the methodology to analyze annotation disagreement in other NLU tasks.

## Related Work & Insights
- **vs Traditional Annotation Consistency Research**: Examines the reasoning process rather than just the labels, providing a more fine-grained analysis.
- **vs ChaosNLI/AmbiEnt**: These datasets focus on quantifying disagreement; this work focuses on the cognitive sources of disagreement.
- **vs Original LiTEx Work**: Extends the analysis scope from within-label variation to label variation.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of extending reasoning analysis to label variation scenarios is novel.
- Experimental Thoroughness: ⭐⭐⭐ The analysis is deep but limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Clear case illustrations and distinct layers of analysis.
- Value: ⭐⭐⭐⭐ Provides significant insights for research on annotation pipelines and data quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](exploring_concreteness_through_a_figurative_lens.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)

</div>

<!-- RELATED:END -->
