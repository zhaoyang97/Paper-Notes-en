---
title: >-
  [Paper Note] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations
description: >-
  [ACL 2026][NLP Understanding][Paper Note] The LiTEx reasoning taxonomy is extended from "explanation variation under label agreement" to "label disagreement" scenarios. It is found that annotators may assign different labels while employing similar reasoning, and reasoning category consistency reflects the semantic similarity of explanations more accurately th
tags:
  - ACL 2026
  - NLP Understanding
date: 2026-05-08
content_hash: 149d58ba5007f062
---
# Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.16458](https://arxiv.org/abs/2510.16458)  
**Code**: None  
**Area**: NLI / Annotation Analysis  
**Keywords**: Annotation Disagreement, Natural Language Inference, LiTEx Taxonomy, Reasoning Strategies, Human Label Variation

## TL;DR

The LiTEx reasoning taxonomy is extended from "explanation variation under label agreement" to "label disagreement" scenarios. It is found that annotators may assign different labels while employing similar reasoning, and reasoning category consistency reflects the semantic similarity of explanations more accurately than label consistency.

## Background & Motivation

**Background**: Annotation disagreement is prevalent in NLI datasets. Understanding these disagreements is vital for building reliable NLU systems. Explanation-based methods reveal the essence of disagreement by analyzing the reasoning behind annotator decisions.

**Limitations of Prior Work**: The LiTEx taxonomy categorizes free-text explanations into eight reasoning strategies, but it was previously used only to analyze within-label variation where labels agree but explanations differ, ignoring instances of label disagreement.

**Key Challenge**: Label inconsistency may mask reasoning consistency (the same reasoning leading to different labels), while label consistency may mask reasoning divergence (different reasoning coincidentally resulting in the same label). Relying solely on labels fails to reveal true cognitive divergence.

**Goal**: To extend LiTEx to label variation scenarios and analyze NLI annotation variation across three dimensions: labels, reasoning categories, and explanation text similarity.

**Key Insight**: By annotating LiTEx categories on the LiveNLI and VariErr datasets (which include explanations), individual annotator preferences for labels and reasoning strategies can be tracked.

**Core Idea**: Reasoning category consistency reflects the semantic similarity between explanations better than label consistency itself, indicating that more attention should be paid to the reasoning process rather than the final label.

## Method

### Overall Architecture

The LiTEx taxonomy is applied to annotate explanations across three datasets (e-SNLI, LiveNLI, VariErr). Variation is then analyzed via three dimensions: (1) NLI label consistency; (2) reasoning category consistency (LiTEx); and (3) semantic similarity of explanation text. Behavior patterns are revealed by tracking individual annotators.

### Key Designs

**1. Cross-dataset extension of the LiTEx taxonomy: Moving reasoning classification from label-consistent scenarios to label-varying scenarios**

LiTEx was originally developed on e-SNLI to analyze within-label variation. Ours migrates it to LiveNLI and VariErr, datasets where labels themselves exhibit disagreement. Trained annotators assign categories to all free-text explanations. The eight reasoning categories are grouped into Textual (Coreference, Syntactic, Semantic, Pragmatic, Absence of Mention, Logic/Conflict) and World Knowledge (Factual, Inferential). This tests the cross-dataset generalization of LiTEx and expands its scope from label agreement to label disagreement.

**2. Multi-dimensional consistency analysis: Comparing label, reasoning category, and text consistency on the same NLI instance**

NLI labels alone cannot distinguish the source of disagreement. This study performs side-by-side comparisons of three dimensions for each instance: (1) NLI label agreement; (2) LiTEx reasoning category agreement; and (3) semantic similarity of explanation text. This explicitly decouples cases of "different labels but same reasoning" and "same labels but different reasoning," revealing the asymmetric relationship between label and reasoning consistency.

**3. Individual annotator tracking: Attributing disagreement to specific individuals to find systematic preferences**

Annotation disagreement may stem from individual reasoning styles rather than just textual ambiguity. Ours tracks label distributions and reasoning strategy preferences for four annotators each in LiveNLI and VariErr. For example, Annotator 2 in VariErr labeled nearly 60% of instances as "neutral." Mapping group-level disagreement to individual levels shows that some "disagreements" are actually stable personal preferences.

### Loss & Training

This is an empirical analysis; no model training is involved. Inter-annotator agreement is measured using Cohen's $\kappa$ (LiveNLI $\kappa = 0.828$, VariErr $\kappa = 0.792$).

## Key Experimental Results

### Main Results

| Dataset | Annotation Count | $\kappa$ Value | Key Findings |
| :--- | :--- | :--- | :--- |
| e-SNLI | Original | - | Inferential Knowledge and Absence of Mention are primary categories |
| LiveNLI | 1404 pairs | 0.828 | Absence of Mention correlates with the Neutral label |
| VariErr | 1933 pairs | 0.792 | Absence of Mention is the most frequent category |

### Key Findings

| Finding | Description |
| :--- | :--- |
| Label disagreement but reasoning consistency | Annotators use the same reasoning strategy but arrive at different labels, suggesting disagreement in judgment rather than understanding. |
| Stable reasoning-label co-occurrence | Despite different absolute distributions across datasets, the label distribution for each reasoning category remains highly consistent. |
| Significant individual label preference | For instance, VariErr Annotator 2 and w7 show neutral preferences of nearly 60% and 52%, respectively. |
| Reasoning similarity > Label similarity | Reasoning category consistency is a better predictor of semantic similarity in explanations. |

### Key Findings
- The LiTEx taxonomy generalizes well across datasets, with category-label co-occurrence patterns being highly consistent across all three datasets.
- Scenarios of "label disagreement but similar explanations" occur frequently, suggesting surface disagreements may mask deep conceptual alignment.
- The strong correlation between the "Absence of Mention" category and the "neutral" label is consistent across all datasets.
- Individual annotators exhibit stable label preferences and reasoning strategy preferences.

## Highlights & Insights
- The core finding that "labels do not equal understanding" challenges the traditional practice of treating labels as absolute ground truth.
- Explanations are not just tools for interpretability but serve as a window into understanding annotation disagreement.
- Individual annotator tracking reveals systematic personal biases, which has direct implications for the design of annotation pipelines.

## Limitations & Future Work
- The study only covers English NLI datasets; cross-lingual generalization remains unverified.
- The eight LiTEx categories may not encompass all possible types of reasoning.
- The number of tracked annotators is limited (4 per dataset), which may constrain statistical power.
- Future work could extend this methodology to analyze annotation disagreements in other NLU tasks.

## Related Work & Insights
- **vs. Traditional Annotation Consistency Research**: Moves beyond labels to examine reasoning processes, providing finer-grained analysis.
- **vs. ChaosNLI/AmbiEnt**: While those datasets focus on quantifying disagreement, this work focuses on the cognitive origins of disagreement.
- **vs. Original LiTEx Work**: Expands the scope of analysis from within-label variation to label variation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perspective of extending reasoning analysis to label variation scenarios is novel.
- **Experimental Thoroughness**: ⭐⭐⭐ The analysis is deep but the scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Case illustrations are clear and the analysis is well-structured.
- **Value**: ⭐⭐⭐⭐ Provides important insights for annotation processes and data quality research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](exploring_concreteness_through_a_figurative_lens.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)

</div>

<!-- RELATED:END -->
