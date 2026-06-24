---
title: >-
  [Paper Note] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations
description: >-
  [ACL 2026 Findings][NLP Understanding][Annotation Disagreement] The LiTEx reasoning taxonomy is extended from "explanation variation under label agreement" to "label disagreement" scenarios. It is found that annotators may have different labels but similar reasoning, and the consistency of reasoning categories reflects the semantic similarity of explanations better than label consistency.
tags:
  - "ACL 2026 Findings"
  - "NLP Understanding"
  - "Annotation Disagreement"
  - "Natural Language Inference"
  - "LiTEx Taxonomy"
  - "Reasoning Strategies"
  - "Human Label Variation"
date: 2026-05-08
content_hash: be4cb8b967203b0d
---

# Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.16458](https://arxiv.org/abs/2510.16458)  
**Code**: None  
**Area**: NLI / Annotation Analysis  
**Keywords**: Annotation Disagreement, Natural Language Inference, LiTEx Taxonomy, Reasoning Strategies, Human Label Variation

## TL;DR

The LiTEx reasoning taxonomy is extended from "explanation variation under label agreement" to "label disagreement" scenarios. It is found that annotators may have different labels but similar reasoning, and the consistency of reasoning categories reflects the semantic similarity of explanations better than label consistency.

## Background & Motivation

**Background**: Annotation disagreement is prevalent in NLI datasets, and understanding these disagreements is crucial for building reliable NLU systems. Explanation-based methods reveal the essence of disagreement by analyzing the reasoning behind annotators' decisions.

**Limitations of Prior Work**: The LiTEx taxonomy categorizes free-text explanations into 8 reasoning strategies, but it was previously used only to analyze "within-label" variation where labels are consistent but explanations differ, ignoring the inconsistency of the labels themselves.

**Key Challenge**: Label inconsistency may mask reasoning consistency (the same reasoning leading to different labels), while label consistency may mask reasoning disagreement (different reasoning happening to yield the same label). Looking at labels alone cannot reveal true cognitive divergence.

**Goal**: Extend LiTEx to label variation scenarios and analyze NLI annotation variation across three dimensions: labels, reasoning categories, and explanation text similarity.

**Key Insight**: Annotate LiTEx categories on two NLI datasets with explanations—LiveNLI and VariErr—to track the label preferences and reasoning strategy preferences of individual annotators.

**Core Idea**: The consistency of reasoning categories reflects the semantic similarity between explanations better than label consistency itself, indicating that more attention should be paid to the reasoning process rather than just the final label.

## Method

### Overall Architecture

The LiTEx taxonomy is applied to annotate explanations across three datasets (e-SNLI, LiveNLI, VariErr). Variation is then analyzed from three dimensions: (1) NLI label consistency; (2) reasoning category consistency (LiTEx); and (3) semantic similarity of explanation text. Behavioral patterns are revealed by tracking individual annotators.

### Key Designs

**1. Cross-dataset extension of the LiTEx taxonomy: Moving reasoning classification validated only on e-SNLI to scenarios where labels vary**

LiTEx was originally developed only on e-SNLI and only analyzed "within-label" variation. This work migrates it to LiveNLI and VariErr, two datasets where the labels themselves are in disagreement. Trained annotators assign categories to all free-text explanations. The 8 reasoning categories are divided into two groups: Textual (Coreference, Syntactic, Semantic, Pragmatic, Absence of Mention, Logical Contradiction) and World Knowledge (Factual Knowledge, Reasoning Knowledge). This first tests whether LiTEx generalizes across datasets and second extends its applicability from "label agreement" to "label disagreement," establishing a unified annotation foundation for multi-dimensional comparison.

**2. Multi-dimensional consistency analysis: Comparing label, reasoning category, and explanation text consistency on the same NLI instance**

Looking at NLI labels alone cannot distinguish the source of disagreement—the same reasoning might lead to different labels, and different reasoning might happen to result in the same label. This work compares three dimensions of consistency side-by-side for the annotators of each instance: (1) whether NLI labels are consistent; (2) whether LiTEx reasoning categories are consistent; and (3) the semantic similarity of explanation text. This allows for the explicit decoupling of "different labels but same reasoning category" and "same label but different reasoning category," revealing the asymmetric relationship between label consistency and reasoning consistency.

**3. Individual annotator tracking: Attributing disagreement to specific individuals to find systematic preferences**

Annotation disagreement does not necessarily stem only from textual ambiguity; it may also arise from the fixed reasoning styles of individual annotators. This work tracks the label distributions and reasoning category preferences of 4 annotators each in LiveNLI and VariErr, calculating individual tendencies (e.g., Annotator 2 in VariErr has a nearly 60% preference for Neutral). This brings group-level disagreement down to the individual level, suggesting that some "disagreements" are actually stable personal preferences, which has direct implications for annotation pipeline design.

### Loss & Training

This work is an empirical analysis and does not involve model training. Inter-annotator agreement was measured using Cohen's Kappa (LiveNLI $\kappa=0.828$, VariErr $\kappa=0.792$).

## Key Experimental Results

### Main Results

| Dataset | Annotation Count | $\kappa$-value | Primary Findings |
|---------|------------------|----------------|------------------|
| e-SNLI  | Original labels  | -              | Reasoning Knowledge and Absence of Mention are primary categories |
| LiveNLI | 1404 pairs       | 0.828          | Absence of Mention biases toward Neutral labels |
| VariErr | 1933 pairs       | 0.792          | Absence of Mention is the most frequent category |

### Key Findings

| Finding | Description |
|---------|-------------|
| Inconsistent labels but consistent reasoning | Annotators use the same reasoning strategy but arrive at different labels, indicating disagreement in judgment rather than understanding |
| Stable reasoning category-label co-occurrence | Despite different absolute distributions across datasets, the label distributions corresponding to reasoning categories are highly consistent |
| Significant individual label preferences | For instance, VariErr annotator 2 has a nearly 60% neutral preference, while w7 has a 52% neutral preference |
| Reasoning similarity > Label similarity | Consistency in reasoning categories predicts the semantic similarity of explanations better |

### Key Findings
- The LiTEx taxonomy generalizes well across datasets; the category-label co-occurrence patterns are highly consistent across the three datasets.
- Cases of "label disagreement but similar explanations" occur frequently, suggesting that surface disagreement may mask consensus in deep understanding.
- The strong association between the Absence of Mention category and the Neutral label is consistent across all datasets.
- Individual annotators exhibit stable label preferences and reasoning strategy preferences.

## Highlights & Insights
- The core finding that "labels do not represent understanding" challenges the traditional practice of treating labels as ground truth.
- Explanations are not just tools for interpretability but also windows into understanding annotation disagreement.
- Individual annotator tracking reveals systematic personal preferences, providing direct insights for annotation process design.

## Limitations & Future Work
- Only English NLI datasets were covered; cross-lingual generalizability remains unverified.
- The 8 categories of LiTEx may not cover all types of reasoning.
- The number of annotators is limited (only 4 tracked per dataset), restricting statistical power.
- Future work could extend the method to analyze annotation disagreement in other NLU tasks.

## Related Work & Insights
- **vs. Traditional Annotation Consistency Research**: Focuses not only on labels but also on the reasoning process, providing a more fine-grained analysis.
- **vs. ChaosNLI/AmbiEnt**: These datasets focus on quantifying disagreement, whereas this work focuses on the cognitive origins of disagreement.
- **vs. Original LiTEx Work**: Extends the scope of analysis from within-label to label variation.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of extending reasoning analysis to label variation scenarios is novel.
- Experimental Thoroughness: ⭐⭐⭐ The analysis is deep but the scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Case illustrations are clear and levels of analysis are distinct.
- Value: ⭐⭐⭐⭐ Provides important insights for annotation processes and data quality research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](exploring_concreteness_through_a_figurative_lens.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] TruthSplit: Operationalizing Conditional Validity in Arguments Through Multi-Perspective Reasoning](truthsplit_operationalizing_conditional_validity_in_arguments_through_multi-pers.md)

</div>

<!-- RELATED:END -->
