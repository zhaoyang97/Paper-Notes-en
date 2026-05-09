---
title: >-
  [Paper Note] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations
description: >-
  [ACL 2026][annotation disagreement] This paper extends the LiTEx reasoning taxonomy from "label-consistent, explanation-variant" settings to label-disagreement scenarios, finding that annotators may share similar reasoning strategies despite assigning different labels, and that reasoning category agreement better reflects the semantic similarity of explanations than label agreement alone.
tags:
  - ACL 2026
  - annotation disagreement
  - natural language inference
  - LiTEx taxonomy
  - reasoning strategies
  - human label variation
date: 2026-05-08
content_hash: 274509ce6c278bae
---

# Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations

**Conference**: ACL 2026
**arXiv**: [2510.16458](https://arxiv.org/abs/2510.16458)
**Code**: None
**Area**: NLI / Annotation Analysis
**Keywords**: annotation disagreement, natural language inference, LiTEx taxonomy, reasoning strategies, human label variation

## TL;DR

This paper extends the LiTEx reasoning taxonomy from "label-consistent, explanation-variant" settings to label-disagreement scenarios, finding that annotators may share similar reasoning strategies despite assigning different labels, and that reasoning category agreement better reflects the semantic similarity of explanations than label agreement alone.

## Background & Motivation

**Background**: Annotator disagreement is pervasive in NLI datasets, and understanding such disagreement is critical for building reliable NLU systems. Explanation-based approaches reveal the nature of disagreement by analyzing the reasoning behind annotator decisions.

**Limitations of Prior Work**: The LiTEx taxonomy categorizes free-text explanations into 8 reasoning strategies, but has previously only been applied to within-label variation — cases where labels agree but explanations differ — leaving label-level inconsistency unaddressed.

**Key Challenge**: Label inconsistency may conceal reasoning consistency (the same reasoning leading to different labels), while label consistency may conceal reasoning disagreement (different reasoning coincidentally yielding the same label). Examining labels alone fails to reveal the true cognitive source of disagreement.

**Goal**: Extend LiTEx to label variation settings and analyze NLI annotation variation along three dimensions: NLI labels, reasoning categories, and explanation text similarity.

**Key Insight**: LiTEx categories are annotated on two NLI datasets with explanations — LiveNLI and VariErr — and individual annotator label and reasoning strategy preferences are tracked.

**Core Idea**: Reasoning category agreement better reflects the semantic similarity between explanations than label agreement does, suggesting that the reasoning process deserves more attention than the final label.

## Method

### Overall Architecture

The LiTEx taxonomy is applied to annotate explanations across three datasets (e-SNLI, LiveNLI, VariErr), followed by analysis along three dimensions: (1) NLI label agreement; (2) reasoning category agreement (LiTEx); and (3) semantic similarity of explanation texts. Individual annotator behavior patterns are revealed through per-annotator tracking.

### Key Designs

1. **Cross-Dataset Extension of the LiTEx Taxonomy**:

    - Function: Apply the reasoning taxonomy — originally developed on e-SNLI — to LiveNLI and VariErr.
    - Mechanism: The 8 reasoning categories are divided into text-based types (coreference, syntactic, semantic, pragmatic, absence of mention, logical contradiction) and world knowledge types (factual knowledge, inferential knowledge). Trained annotators classify all explanations.
    - Design Motivation: Validate cross-dataset generalizability of LiTEx while extending its applicability to label variation settings.

2. **Multi-Dimensional Agreement Analysis**:

    - Function: Reveal the asymmetric relationship between label agreement and reasoning agreement.
    - Mechanism: Compare annotator agreement on the same NLI instance across three dimensions — cases of "different labels but same reasoning category" or "same label but different reasoning category" are both examined.
    - Design Motivation: A single-dimensional agreement measure is insufficient to capture the true nature of annotation disagreement.

3. **Individual Annotator Tracking**:

    - Function: Identify systematic annotator preferences.
    - Mechanism: Label distributions and reasoning category preferences are tracked for 4 annotators in LiveNLI and 4 in VariErr, revealing individual consistency patterns.
    - Design Motivation: Annotation disagreement may stem not only from textual ambiguity but also from individual annotators' personal reasoning styles.

### Loss & Training

This is an empirical analysis study; no model training is involved. Inter-annotator agreement is measured using Cohen's Kappa (LiveNLI $\kappa=0.828$, VariErr $\kappa=0.792$).

## Key Experimental Results

### Main Results

| Dataset | Annotations | κ | Key Finding |
|---------|-------------|---|-------------|
| e-SNLI | Original | — | Inferential knowledge and absence of mention are dominant categories |
| LiveNLI | 1,404 pairs | 0.828 | Absence of mention skews toward the *neutral* label |
| VariErr | 1,933 pairs | 0.792 | Absence of mention is the most frequent category |

### Key Findings

| Finding | Description |
|---------|-------------|
| Label disagreement with reasoning agreement | Annotators employ the same reasoning strategy yet arrive at different labels, indicating disagreement in judgment rather than understanding |
| Stable reasoning–label co-occurrence | Despite differences in absolute distributions across datasets, the label distributions associated with each reasoning category are highly consistent |
| Significant individual label preferences | E.g., VariErr annotator 2 assigns *neutral* in nearly 60% of cases; annotator w7 in 52% |
| Reasoning similarity > label similarity | Reasoning category agreement more accurately predicts the semantic similarity of explanations |

### Key Findings
- The LiTEx taxonomy generalizes well across datasets; category–label co-occurrence patterns are highly consistent across all three datasets.
- Cases of "label disagreement but similar explanations" occur frequently, suggesting that surface-level disagreement may conceal underlying convergence in understanding.
- The strong association between the *Absence of Mention* category and the *neutral* label holds consistently across all datasets.
- Individual annotators exhibit stable label preferences and reasoning strategy preferences.

## Highlights & Insights
- The central finding that "labels do not represent understanding" challenges the conventional practice of treating labels as ground truth.
- Explanations serve not only as interpretability tools but also as windows into the nature of annotation disagreement.
- Per-annotator tracking reveals systematic individual preferences, with direct implications for annotation pipeline design.

## Limitations & Future Work
- Coverage is limited to English NLI datasets; cross-lingual generalizability remains unverified.
- The 8 LiTEx categories may not cover all reasoning types.
- The number of tracked annotators is small (4 per dataset), limiting statistical power.
- Future work could extend the framework to annotation disagreement analysis in other NLU tasks.

## Related Work & Insights
- **vs. traditional annotation agreement studies**: This work examines not only labels but also reasoning processes, enabling finer-grained analysis.
- **vs. ChaosNLI/AmbiEnt**: Those datasets focus on quantifying disagreement; this paper focuses on its cognitive sources.
- **vs. original LiTEx work**: The scope of analysis is extended from within-label variation to label variation.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of extending reasoning analysis to label variation settings is original.
- Experimental Thoroughness: ⭐⭐⭐ Analysis is in-depth but limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Case illustrations are clear; the analysis is well-structured.
- Value: ⭐⭐⭐⭐ Offers important insights for annotation pipeline design and data quality research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[AAAI 2026\] Formal Abductive Latent Explanations for Prototype-Based Networks](../../AAAI2026/others/formal_abductive_latent_explanations_for_prototype-based_networks.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](../../AAAI2026/others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[AAAI 2026\] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](../../AAAI2026/others/align_when_they_want_complement_when_they_need_human-centere.md)
- [\[ICCV 2025\] LayerD: Decomposing Raster Graphic Designs into Layers](../../ICCV2025/others/layerd_decomposing_raster_graphic_designs_into_layers.md)

<!-- RELATED:END -->
