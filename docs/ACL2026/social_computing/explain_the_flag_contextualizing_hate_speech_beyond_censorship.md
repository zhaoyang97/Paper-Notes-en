---
title: >-
  [Paper Note] Explain the Flag: Contextualizing Hate Speech Beyond Censorship
description: >-
  [ACL 2026][Social Computing][Hate Speech Detection] This paper proposes a hybrid approach combining LLMs with manually curated vocabularies in three languages (English, French…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Hate Speech Detection"
  - "Explainability"
  - "Multilingual Vocabulary"
  - "Contextualized Explanation"
  - "Hybrid System"
date: 2026-05-08
content_hash: 2c179bb487e1dc15
---

# Explain the Flag: Contextualizing Hate Speech Beyond Censorship

**Conference**: ACL 2026  
**arXiv**: [2604.14970](https://arxiv.org/abs/2604.14970)  
**Code**: [GitHub](https://github.com/ails-lab/detoex)  
**Area**: Social Computing / Hate Speech  
**Keywords**: Hate Speech Detection, Explainability, Multilingual Vocabulary, Contextualized Explanation, Hybrid System

## TL;DR
This paper proposes a hybrid approach combining LLMs with manually curated vocabularies in three languages (English, French, and Greek) to detect and explain hate speech—the term-based pipeline detects inherently derogatory terms via lexical matching and LLM semantic disambiguation, while the term-free pipeline utilizes LLMs to detect group-targeted content; the fusion of both generates evidence-based explanations.

## Background & Motivation

**Background**: Automated hate speech detection systems are widely used for online platform moderation, yet most focus on censorship or removal, lacking transparency and explainability—users are flagged without knowing why.

**Limitations of Prior Work**: (1) Pure removal approaches lack transparency, limiting user understanding of why language is harmful; (2) moderation decisions may appear arbitrary or biased; (3) hate speech takes two forms—inherently derogatory terms (e.g., slurs) and group-targeted content (which may be harmful even without slurs)—requiring different detection strategies; (4) low-resource languages (e.g., Greek) lack relevant resources.

**Key Challenge**: Moderation needs to balance "blocking harmful content" and "explaining why it is harmful"—pure LLM methods lack stable terminological knowledge, while pure lexical methods lack contextual understanding.

**Goal**: To build a hybrid system capable of detecting and explaining hate speech across English, French, and Greek.

**Key Insight**: A dual-pipeline design—a term-based pipeline using curated vocabularies for exact matching plus LLM disambiguation, and a term-free pipeline using LLMs for context-aware group-targeting detection.

**Core Idea**: Curated vocabularies (with meaning explanations and identity trait annotations) + LLM contextual reasoning $\rightarrow$ Evidence-based explanations.

## Method

### Overall Architecture
Parallel dual pipelines: (1) Term-based pipeline: Lemmatization + string matching to detect potential derogatory terms $\rightarrow$ LLM disambiguation within context (derogatory vs. non-derogatory usage) $\rightarrow$ output explanation; (2) Term-free pipeline: LLM directly determines whether the text targets individuals/groups based on identity traits $\rightarrow$ output explanation. Pipeline fusion: Content is flagged if either pipeline identifies it; if both flag it, an LLM merges and deduplicates the results into a unified explanation.

### Key Designs

1.  **Multilingual Curated Vocabularies**:
    - **Function**: Provide a reliable terminological knowledge base for the LLM.
    - **Mechanism**: Terms with "derogatory/offensive/vulgarities" tags are extracted from Wiktionary and built through a five-step process: initial collection (11,310 EN / 3,749 FR / 965 EL) $\rightarrow$ filtering (retaining inherently derogatory terms targeting groups) $\rightarrow$ classification (labeling identity traits) $\rightarrow$ enrichment (LLM generation of descriptive text including controversial/non-controversial usage) $\rightarrow$ manual verification. Final dataset: 3,904 EN / 1,644 FR / 288 EL entries.
    - **Design Motivation**: LLMs may lack knowledge of rare or culture-specific derogatory terms; curated vocabularies provide reliable external knowledge to fill these gaps.

2.  **LLM Semantic Disambiguation**:
    - **Function**: Determine whether a detected term is used derogatorily in the current context.
    - **Mechanism**: The LLM receives the source text and the dictionary definition (including controversial and non-controversial usage examples) to output a judgment and explanation. This handles polysemy (e.g., "bitch" referring to a female dog vs. an insult) and reclaimed language (terms reclaimed by targeted groups).
    - **Design Motivation**: Many derogatory terms have non-derogatory meanings, and simple matching yields high false positives—requiring contextual understanding from an LLM.

3.  **Dual-Pipeline Fusion and Explanation Generation**:
    - **Function**: Synthesize results from both detection strategies to generate a unified, evidence-based explanation.
    - **Mechanism**: Content is judged safe only when both pipelines find no hate speech. If one identifies hate speech, its explanation is used. If both identify it, the LLM merges both explanations, removing redundancies to generate a coherent unified output.
    - **Design Motivation**: The pipelines are complementary—the term-based pipeline finds specific slurs but may miss group attacks without insults, while the term-free pipeline finds contextual attacks but may miss rare terms.

### Loss & Training
The hybrid system does not involve training. Claude Sonnet 3.7 is used as the primary model, with the Llama series as lightweight open-source alternatives.

## Key Experimental Results

### Main Results

| Language | Model | Precision | Recall | F1 (Safe) |
| :--- | :--- | :--- | :--- | :--- |
| English | Claude (Ours) | 0.92 | 0.89 | 0.90 |
| English | Llama (Ours) | 0.82 | 0.82 | 0.82 |
| French | Claude (Ours) | 0.96 | 0.91 | 0.93 |
| Greek | Claude (Ours) | - | - | Above baseline |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Term-free only (LLM-only) | Lower | Misses rare/culture-specific terms |
| Term-based only | Lower | Misses group attacks without explicit slurs |
| Hybrid System | Optimal | Dual pipelines are complementary |

### Key Findings
- The hybrid system consistently outperforms pure LLM baselines, proving that curated vocabularies enhance LLM performance.
- Human evaluation indicates high explanation quality—users can understand why content was flagged.
- Claude significantly outperforms the Llama series, though Llama provides practical value for low-resource deployment (single GPU).
- Gains from the vocabulary are particularly significant for Greek (a low-resource language).

## Highlights & Insights
- The conceptual shift **from censorship to explanation** holds significant social value—explaining why content is harmful promotes user understanding and behavior change better than simple deletion.
- The hybrid paradigm of curated vocabularies + LLMs is a generalizable model applicable to any task requiring "precise domain knowledge + contextual understanding."
- The methodology for constructing multilingual vocabularies (Wiktionary + LLM filtering + manual verification) provides a reusable resource-building workflow.

## Limitations & Future Work
- Vocabularies require continuous maintenance to cover emerging derogatory terms.
- Evaluation was limited to tweets (short text); performance may vary in long-form text scenarios.
- Reclaimed language remains challenging to handle without information about the user's identity.
- Automatic evaluation metrics for explanations are limited, relying primarily on human evaluation.

## Related Work & Insights
- **vs. Pure LLM Detection**: Lacks stable terminological knowledge and may overlook rare insults.
- **vs. Pure Lexical Approaches**: Lacks contextual understanding, resulting in high false positive rates.
- **vs. Menis Mastromichalakis et al. (2025)**: They address explainable hate speech but do not involve multilingual vocabularies.

## Rating
- **Novelty**: ⭐⭐⭐ The dual-pipeline hybrid approach is not entirely new, but the multilingual curated vocabulary is a valuable resource contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three languages, human evaluation of detection and explanation quality, and comparisons across multiple models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with sufficient social motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[ACL 2026\] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)
- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)

</div>

<!-- RELATED:END -->
