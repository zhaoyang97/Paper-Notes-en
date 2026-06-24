---
title: >-
  [Paper Note] The Harmonic Structure of Information Contours
description: >-
  [ACL 2025][uniform information density] Proposes the Harmonic Surprisal (HS) hypothesis—that surprisal curves in text fluctuate periodically and align with discourse structures (EDUs/sentences/paragraphs). Tested via harmonic regression with time scaling, consistent periodic patterns are found across 6 languages, refining the classical Uniform Information Density hypothesis.
tags:
  - "ACL 2025"
  - "uniform information density"
  - "surprisal"
  - "harmonic regression"
  - "discourse structure"
  - "periodicity"
  - "time scaling"
date: 2026-05-08
content_hash: 41b04b7c9d02db83
---

# The Harmonic Structure of Information Contours

**Conference**: ACL 2025  
**arXiv**: [2506.03902](https://arxiv.org/abs/2506.03902)  
**Code**: [GitHub](https://github.com/rycolab/harmonic-surprisal)  
**Area**: Computational Linguistics / Information Theory  
**Keywords**: uniform information density, surprisal, harmonic regression, discourse structure, periodicity, time scaling

## TL;DR
Proposes the Harmonic Surprisal (HS) hypothesis—that surprisal curves in text fluctuate periodically and align with discourse structures (EDUs/sentences/paragraphs). Tested via harmonic regression with time scaling, consistent periodic patterns are found across 6 languages, refining the classical Uniform Information Density hypothesis.

## Background & Motivation
**Background**: The UID hypothesis suggests that speakers prefer to distribute information uniformly; empirical studies support that information rates regress to the mean at a global scale. However, surprisal is not uniform at the discourse level—it fluctuates in observable patterns.

**Limitations of Prior Work**: These fluctuations are often attributed to ad hoc factors such as syntactic constraints, stylistic choices, or audience design, lacking a unified quantitative framework to explain global fluctuation patterns. Existing studies have independently linked discourse structures (paragraphs, topic segments, EDUs) and information rates, but no overarching theory has been proposed.

**Key Challenge**: Are information rate fluctuations random noise or structured periodicity? If periodic, do the periods correspond to linguistically meaningful units?

**Key Insight**: Signal processing—treating surprisal as a time series, using harmonic regression (frequency domain analysis) to detect periodic components, and innovatively introducing time scaling to embed linguistic hypotheses into statistical tests.

**Core Idea**: HS hypothesis = surprisal can be modeled as a mixture of sinusoids at multiple frequencies, where the period of each frequency corresponds to the length of discourse units at different granularities.

## Method

### Overall Architecture
Uses a Transformer language model to estimate the token-level surprisal curve of the text, then fits it with harmonic regression, testing whether the periodicity of surprisal aligns with EDU/sentence/paragraph boundaries through a time-scaling mechanism. Cross-lingual validation is conducted on the RST Discourse Bank for 6 languages.

### Key Designs

1. **Harmonic Regression**:

    - Function: Models surprisal as a linear combination of multiple sine and cosine components
    - Core formula: $f(t) = \beta_0 + \sum_{k=1}^{K} (\beta_{1,k} \sin(\frac{k 2\pi t}{T}) + \beta_{2,k} \cos(\frac{k 2\pi t}{T}))$
    - The amplitude $A_k = \sqrt{\beta_{1,k}^2 + \beta_{2,k}^2}$ of each harmonic component $k$ captures the strength of that frequency component

2. **Time Scaling (Core Innovation)**:

    - Function: Replaces the global period $T$ in harmonic regression with the length of discourse units
    - Mechanism: Replaces $T$ with $U_t$ (the length of the structural unit containing the current token $w_t$), aligning the period of the sine wave with the actual span of EDUs, sentences, or paragraphs
    - Design Motivation: If surprisal is high at the beginning of each paragraph and low at the end, a sine wave scaled by paragraph length will fit well—this directly tests whether fluctuations align with discourse structures
    - Difference from Prior Methods: Standard harmonic regression can only discover frequencies but cannot associate them with linguistic structures; time scaling embeds structural hypotheses directly into the statistical model

3. **Cross-Lingual Validation**:

    - 6 languages: English, Spanish, German, Dutch, Basque, Brazilian Portuguese
    - Uses the RST Discourse Bank to provide EDU/sentence/paragraph boundary annotations
    - 10-fold cross-validation + L1 regularized feature selection + one-way ANOVA significance tests

### Baseline Features
Includes token character length, previous token surprisal, relative position of the token in the document, and a boolean feature vector representing distances of 1/2/4 tokens from structural boundaries—ensuring that harmonic features capture true periodicity beyond simple boundary effects.

## Key Experimental Results

### Main Results: Cross-Lingual Periodicity Detection (MSE, lower is better)

| Model | English | Spanish | German | Dutch | Basque | Brazilian Portuguese |
|------|------|---------|------|--------|---------|---------|
| Baseline | 9.91 | 14.63 | 12.43 | 9.32 | 9.00 | 9.62 |
| Document-scaling | 9.92 | **13.52** | 12.29 | 9.60 | 9.17 | 9.80 |
| EDU-scaling | **9.46** | 13.83 | **11.31** | — | — | — |

### Ablation Study: Effect of Time Scaling

| Scaling Granularity | Improvement in Fit | Significance |
|---------|---------|--------|
| EDU-scaling | Largest improvement | Significant across all languages |
| Sentence-scaling | Moderate improvement | Significant in most languages |
| Paragraph-scaling | Some improvement | Significant in some languages |
| Document-scaling (ref) | Small improvement | — |

### Key Findings
- **Significant periodicity exists in all 6 languages**: The harmonic model significantly outperforms models containing only baseline features
- **EDU-level scaling has the strongest effect**: The first-order sine wave (exactly corresponding to the EDU span) has the highest amplitude, indicating that the main structural component of information rate fluctuations corresponds to the minimal discourse unit
- **UID is globally violated but can be locally preserved**: Global information rates undergo structured periodic fluctuations rather than being uniform
- **Periodicity of information rate is a cross-lingual universal**: 6 languages from different language families (including Basque, a language isolate) exhibit consistent patterns

## Highlights & Insights
- **Elegant fusion of signal processing and computational linguistics**: Introduces frequency domain analysis of surprisal into discourse analysis, providing a brand-new perspective. Time scaling naturally embeds linguistic hypotheses into statistical tests
- **Refinement of UID by HS**: Translating from "uniform" to "periodic fluctuation"—this does not overturn UID but rather upgrades it to a more precise description. UID can still hold locally, but periodic pressures exist globally
- **EDUs provide the simplest and finest grain of structural information**: This has direct implications for applications like automatic segmentation, topic detection, and reading difficulty assessment—the periodicity of surprisal can conversely infer discourse structure

## Limitations & Future Work
- **Dependency on LM surprisal estimation**: Uses the surprisal of Transformer LMs as a proxy for human psycholinguistic processing, but systematic differences may exist between LMs and humans in perceived information density
- **Written text only**: Information rate fluctuation patterns in spoken dialogue might differ—influenced by prosody, pauses, repairs, etc.
- **Causality not established**: Although co-variation between periodicity and discourse structure is observed, this does not prove causal relationships—do speakers actively modulate the information rate, or does the discourse structure itself produce this side effect?
- **Scalability to more languages and extra granularities**: Such as syntactic structures (clauses, phrases) or even word-level features

## Related Work & Insights
- **vs UID (Levy & Jaeger, 2006)**: UID predicts uniform distribution, while HS predicts periodic fluctuations—HS is a refinement rather than a negation of UID
- **vs SC Hypothesis (Tsipidi et al., 2024)**: The SC hypothesis posits that position affects surprisal but does not specify the precise relationship; HS restricts this relationship to periodic functions
- **vs Genzel & Charniak (2002)**: They found that surprisal increases along with position within a paragraph; HS provides a more general periodic perspective

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The HS hypothesis is completely new, and the time-scaled harmonic regression method is highly novel
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 languages + multi-granularity comparisons + significance testing
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, and Figure 1 intuitively demonstrates the harmonic decomposition
- Value: ⭐⭐⭐⭐ Has a primary theoretical contribution to linguistic information structure, with potential applications in discourse analysis and LM evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](../../ICML2026/others/structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ACL 2025\] ChuLo: Chunk-Level Key Information Representation for Long Document Understanding](chulo_chunk-level_key_information_representation_for_long_document_understanding.md)
- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/others/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)

</div>

<!-- RELATED:END -->
