---
title: >-
  [Paper Note] Explain the Flag: Contextualizing Hate Speech Beyond Censorship
description: >-
  [ACL 2026][Social Computing][Hate speech detection] This paper proposes a hybrid approach combining LLMs with human-curated lexicons in three languages (English/French/Greek) to detect and explain hate speech—the term-based pipeline uses lexicon matching + LLM semantic disambiguation to detect inherently derogatory terms, the term-free pipeline uses LLMs to detect group-targeted content, and both are fused to generate evidence-based explanations.
tags:
  - ACL 2026
  - Social Computing
  - Hate speech detection
  - Explainability
  - Multilingual lexicon
  - Contextualized explanation
  - Hybrid system
date: 2026-05-08
content_hash: b6acd366c09c1e54
---

# Explain the Flag: Contextualizing Hate Speech Beyond Censorship

**Conference**: ACL 2026  
**arXiv**: [2604.14970](https://arxiv.org/abs/2604.14970)  
**Code**: [GitHub](https://github.com/ails-lab/detoex)  
**Area**: Social Computing / Hate Speech  
**Keywords**: Hate speech detection, Explainability, Multilingual lexicon, Contextualized explanation, Hybrid system

## TL;DR
This paper proposes a hybrid approach combining LLMs with human-curated lexicons in three languages (English/French/Greek) to detect and explain hate speech—the term-based pipeline uses lexicon matching + LLM semantic disambiguation to detect inherently derogatory terms, the term-free pipeline uses LLMs to detect group-targeted content, and both are fused to generate evidence-based explanations.

## Background & Motivation

**Background**: Automated hate speech detection systems are widely used for online platform moderation, but most focus on censorship or removal, lacking transparency and explainability—users are flagged but do not know why they were flagged.

**Limitations of Prior Work**: (1) Pure deletion approaches lack transparency, limiting users' understanding of why their language is harmful; (2) Moderation decisions may appear arbitrary or biased; (3) Hate speech has two forms—inherently derogatory terms (e.g., slurs) and group-targeted content (which can be harmful even without slurs)—requiring different detection strategies; (4) Low-resource languages (e.g., Greek) lack relevant resources.

**Key Challenge**: Moderation needs to balance "blocking harmful content" and "explaining why it is harmful"—pure LLM approaches lack stable terminological knowledge, while pure lexicon approaches lack contextual understanding.

**Goal**: Build a hybrid system that can detect and explain hate speech, covering English/French/Greek.

**Key Insight**: Dual-pipeline design—the term-based pipeline uses curated lexicons for precise matching + LLM disambiguation, the term-free pipeline uses LLMs for context-aware group-targeting detection.

**Core Idea**: Curated lexicon (with meaning explanations + identity feature annotations) + LLM contextual reasoning → evidence-based explanations.

## Method

### Overall Architecture
Dual pipelines in parallel: (1) Term-based pipeline: lemmatization + string matching detects potential derogatory terms → LLM disambiguates in context (derogatory/non-derogatory usage) → outputs explanation; (2) Term-free pipeline: LLM directly judges whether text attacks groups/individuals based on identity characteristics → outputs explanation. Pipeline fusion: flags if either flags, if both flag then LLM fuses and deduplicates outputs into unified explanation.

### Key Designs

1. **Multilingual Curated Lexicon**:

    - Function: Provides reliable terminological knowledge base for LLMs
    - Mechanism: Extracts terms labeled "derogatory/offensive/vulgarities" from Wiktionary, constructs through five-step process: initial collection (11,310 English/3,749 French/965 Greek) → filtering (retains inherently derogatory terms targeting groups) → categorization (annotates identity features) → enrichment of descriptions (LLM generates continuous text containing controversial/non-controversial usages) → human verification. Final result: 3,904 English/1,644 French/288 Greek entries
    - Design Motivation: LLMs may not know rare or culturally-specific derogatory terms; curated lexicon provides reliable external knowledge to address LLM knowledge gaps

2. **LLM Semantic Disambiguation**:

    - Function: Judges whether detected term is used derogatorily in current context
    - Mechanism: LLM receives source text and term's meaning description from lexicon (including controversial and non-controversial usages), outputs judgment of whether derogatorily used + explanation. This handles polysemous words (e.g., "bitch" can mean female dog/insult) and reclaimed terms (cases where terms are reclaimed by target groups)
    - Design Motivation: Many derogatory terms have non-derogatory meanings; simple matching produces many false positives—requires LLM's contextual understanding for disambiguation

3. **Dual-Pipeline Fusion and Explanation Generation**:

    - Function: Synthesizes results from two detection strategies, generates evidence-based unified explanation
    - Mechanism: Only judges as safe when both pipelines deem no hate speech. If one pipeline detects, uses that pipeline's explanation. If both detect, LLM fuses the two explanations, removes redundancy, generates coherent unified explanation
    - Design Motivation: Two pipelines complement each other—term-based pipeline detects inherently derogatory terms but may miss group attacks without slurs; term-free pipeline detects contextual attacks but may miss rare terms

### Loss & Training
Hybrid system does not involve training. Uses Claude Sonnet 3.7 as the large model, Llama series as lightweight open-source alternative.

## Key Experimental Results

### Main Results

| Language | Model | Precision | Recall | F1 (Safe) |
|----------|-------|-----------|--------|-----------|
| English | Claude (hybrid) | 0.92 | 0.89 | 0.90 |
| English | Llama (hybrid) | 0.82 | 0.82 | 0.82 |
| French | Claude (hybrid) | 0.96 | 0.91 | 0.93 |
| Greek | Claude (hybrid) | - | - | Above baseline |

### Ablation Study

| Config | Metric | Note |
|--------|--------|------|
| Term-free pipeline only (LLM-only) | Lower | Misses rare/culturally-specific terms |
| Term-based pipeline only | Lower | Misses group attacks without slurs |
| Hybrid system | Best | Two pipelines complement each other |

### Key Findings
- Hybrid system consistently outperforms pure LLM baseline, proving curated lexicon enhances LLMs
- Human evaluation shows high explanation quality—users can understand why content was flagged
- Claude significantly outperforms Llama series, but Llama has practical value in low-resource deployment (single GPU)
- Lexicon gains are especially significant in Greek (low-resource language)

## Highlights & Insights
- The conceptual shift **from censorship to explanation** has important social value—explaining why harmful is more conducive to user understanding and behavior change than simple deletion
- The hybrid model of curated lexicon + LLM is a generalizable paradigm—applicable in any task requiring "precise domain knowledge + contextual understanding"
- The methodology for constructing multilingual lexicon (Wiktionary + LLM filtering + human verification) is a reusable resource construction pipeline

## Limitations & Future Work
- Lexicon requires continuous maintenance to cover newly emerging derogatory terms
- Only evaluated on tweets (short text); long text scenarios may differ
- Handling of reclaimed terms (e.g., terms reclaimed by LGBTQ community) remains challenging—difficult to judge when user identity information is lacking
- Limited automatic evaluation metrics for explanations, mainly relies on human evaluation

## Related Work & Insights
- **vs pure LLM detection**: Lacks stable terminological knowledge, may miss rare slurs
- **vs pure lexicon methods**: Lacks contextual understanding, high false positive rate
- **vs Menis Mastromichalakis et al. (2025)**: They work on explainable hate speech but do not involve multilingual lexicon

## Rating
- Novelty: ⭐⭐⭐ Dual-pipeline hybrid approach not entirely new, but multilingual lexicon is valuable resource contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-language coverage, human evaluation of detection and explanation quality, multi-model comparison
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated socially

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[ACL 2026\] Is this chart lying to me? Automating the detection of misleading visualizations](is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)

</div>

<!-- RELATED:END -->
