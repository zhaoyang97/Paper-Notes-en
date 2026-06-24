---
title: >-
  [Paper Note] Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements
description: >-
  [ACL2025][LLM (Other)][Indexical understanding] First systematic evaluation of LLMs' understanding of English indexicals (I/you/here/tomorrow) by constructing a 1,600-item 2×2 factorial design evaluation set. The study reveals that LLMs heavily rely on irrelevant contextual information rather than grammatical rules for "you/here/tomorrow", and quotation marks have completely opposite effects on different indexicals.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Indexical understanding"
  - "coreference resolution"
  - "LLM evaluation"
  - "pragmatics"
date: 2026-05-08
content_hash: d80e9e114356b400
---

# Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements

**Conference**: ACL2025  
**arXiv**: [2506.01089](https://arxiv.org/abs/2506.01089)  
**Code**: [metehanoguzz/LLMs-Indexicals-English](https://github.com/metehanoguzz/LLMs-Indexicals-English)  
**Area**: LLM/NLP  
**Keywords**: Indexical understanding, coreference resolution, LLM evaluation, pragmatics

## TL;DR

First systematic evaluation of LLMs' understanding of English indexicals (I/you/here/tomorrow) by constructing a 1,600-item 2×2 factorial design evaluation set. The study reveals that LLMs heavily rely on irrelevant contextual information rather than grammatical rules for "you/here/tomorrow", and quotation marks have completely opposite effects on different indexicals.

## Background & Motivation

**Background**: Evaluations of LLMs on coreference resolution tasks have mainly focused on third-person pronouns (he/she/they) and noun phrases. However, indexicals, which hold a highly unique status in linguistics by directly anchoring the coordinates of a speech act (e.g., "I", "you", "here", "tomorrow"), have rarely been systematically evaluated.

**Limitations of Prior Work**: Third-person pronouns are inherently ambiguous (e.g., in "John hit Bill and he ran away", "he" can refer to John or Bill) and require contextual disambiguation. In contrast, the semantics of indexicals are strictly determined by grammatical rules: "I" refers to the speaker, "you" to the addressee, "here" to the place of utterance, and "tomorrow" to the day after the utterance. This means a model that truly "understands" language should be able to **ignore** potentially misleading contextual information. Previous work by the same research group found that LLMs performed poorly on the Turkish indexical "ben" ("I") (Oğuz et al., 2024), but whether this holds true for English remains unclear.

**Key Challenge**: Resolving indexicals requires models to follow syntactic rules rather than statistical associations—posing an "anti-intuitive" challenge for LLMs, which are essentially statistical models. In particular, the shifting of indexicals in direct quotations (e.g., in `Andrew said "I am smart"`, "I" refers to Andrew rather than the actual speaker) introduces an extra layer of reasoning.

**Goal**: Systematically evaluate the understanding of four categories of English indexicals ("I", "you", "here", "tomorrow") in frontier LLMs, distinguishing whether models "truly understand grammatical rules" or "just happen to get it right by relying on contextual guessing".

**Key Insight**: Design a 2×2 factorial controlled experiment (sentence type × contextual prime) and build a precisely controlled evaluation set of 1,600 instances to eliminate surface statistical correlations, forcing models to demonstrate whether they genuinely reason based on grammatical rules.

**Core Idea**: Through an orthogonally controlled indexical evaluation dataset, reveal that LLMs perform close to human level on "I", but heavily rely on irrelevant contextual information instead of grammatical rules on "you", "here", and "tomorrow".

## Method

### Overall Architecture

This paper is a benchmark study rather than proposing a new methodology. The core workflow is: (1) Constructing the English Indexical Dataset, containing 1,600 multiple-choice questions designed under a 2×2 factorial layout; (2) Evaluating four state-of-the-art LLMs as subjects under a forced-choice protocol; (3) Analyzing model behavior at the granularity of each indexical (I/you/here/tomorrow), with a special focus on whether models make choices based on grammatical rules or contextual priming.

### Key Designs

1. **2×2 Factorial Controlled Evaluation Set Construction**:

    - **Function**: To construct an evaluation dataset that can precisely distinguish between "grammatical rule understanding" and "contextual guessing."
    - **Mechanism**: The dataset covers 4 indexical types (I/you/here/tomorrow), with 400 samples each. Each sample is expanded across two orthogonal factors: **Sentence Type** (quoted vs. non-quoted) and **Contextual Prime** (shifted prime favoring a quoted interpretation vs. non-shifted prime favoring a literal interpretation), resulting in 4 conditions. The correct answer is uniquely determined by grammatical rules—non-quoted sentences should choose the non-shifted option, and quoted sentences should choose the shifted option—while the contextual prime acts as a distractor that should be ignored. The data is generated using GPT-4o for scenarios, and 25% (400 items) are manually verified to ensure grammatical correctness and balance gender bias (50% male/female names).
    - **Design Motivation**: Simple accuracy cannot distinguish whether a model truly understands grammar or just guesses correctly. The orthogonal design allows observation of whether the model's choices shift along with the direction of the contextual prime—if the change is significant, it indicates the model relies on context rather than grammatical rules.

2. **Forced-Choice Evaluation Protocol**:

    - **Function**: To ensure the model's outputs can be parsed precisely and eliminate noise from open-ended responses.
    - **Mechanism**: Selecting four state-of-the-art LLMs: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek-V3. The prompt strictly limits the models to selecting between two predefined options (shifted vs. non-shifted interpretation) to prevent them from generating verbose explanations or refusing to answer. Accuracy for each indexical is calculated under the 4 conditions (2 sentence types × 2 priming directions).
    - **Design Motivation**: Parsing open-ended responses can introduce annotator bias. The forced-choice paradigm is a standard experimental setup in psycholinguistics adapted for NLP.

3. **Indexical-Granularity Analysis Framework**:

    - **Function**: To reveal the unique behavioral patterns of different indexicals rather than offering a generic "good/bad" conclusion.
    - **Mechanism**: Analyzing "I", "you", "here", and "tomorrow" individually, focusing on three comparative dimensions: (a) whether the model correctly chooses the non-shifted option under non-quoted conditions; (b) whether the model correctly identifies quotation shifts to select the shifted option under quoted conditions; (c) the magnitude of change in accuracy when the direction of the contextual prime changes (a larger delta indicates higher reliance on context). Combining positive and negative quotation effect analyses separates the distinct impacts of quotation marks on different indexicals.
    - **Design Motivation**: Different indexicals have fundamental linguistic differences—the identity of "I" is the strongest (directly equating to the speaker), while the anchoring of "here" and "tomorrow" relies more heavily on contextual reasoning. Granular analysis reveals exactly at which level LLM pragmatic reasoning fails.

## Key Experimental Results

### Main Results

| Indexical | Condition | GPT-4o | Claude 3.5 | Gemini 1.5 | DeepSeek-V3 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| I | Non-quoted - shifted prime | ~99 | ~99 | ~99 | ~99 |
| I | Non-quoted - non-shifted prime | ~99 | ~99 | ~99 | ~99 |
| I | Quoted - shifted prime | >94 | 89 | >94 | 78 |
| I | Quoted - non-shifted prime | >94 | 89 | >94 | 17 |
| you | Non-quoted - shifted prime | ~70 | ~70 | 92 | ~70 |
| you | Non-quoted - non-shifted prime | ~80 | ~80 | 92 | ~80 |
| you | Quoted (average of both primes) | Sig. Drop | Sig. Drop | Sig. Drop | Sig. Drop |
| here | Non-quoted - shifted prime | >96 | >96 | >96 | >96 |
| here | Non-quoted - non-shifted prime | <2 | <2 | <2 | <2 |
| here | Quoted (average) | 37 | 64 | 94 | >97 |
| tomorrow | Non-quoted | ~94 | 100 | 100 | 83 |
| tomorrow | Quoted | Extremely Low | ~0 | ~0 | Extremely Low |

### Ablation Study

- **I**: All models achieve an average accuracy of 99% in non-quoted conditions, approaching human performance. In quoted conditions, GPT-4o and Gemini maintain >94%, but DeepSeek-V3 drops sharply to 17% under the non-shifted prime, showing that quotation marks make it more susceptible to contextual interference.
- **you**: All models are heavily influenced by the contextual prime—accuracy drops significantly when the context biases towards the wrong option. Quoted conditions consistently lower accuracy, indicating that the models fail to leverage quotation shift rules. Gemini performs best under the non-quoted condition (92%) but also experiences a major fallback in the quoted condition.
- **here**: In non-quoted conditions, the models' choices are **entirely dominated by the contextual prime** (>96% for shifted prime, <2% for non-shifted prime), with grammatical rules exerting almost no effect. Surprisingly, quotation marks help models "escape" context dependency: DeepSeek reaches >97%, and Gemini reaches 94%.
- **tomorrow**: All models exhibit a strong bias towards "non-shifted" interpretations, showing artificially high performance under non-quoted conditions (94-100%) but hitting near-zero accuracy under quoted conditions. Claude and Gemini select the non-shifted option 100% of the time under quoted conditions (i.e., 100% incorrect), showing that the models are completely incapable of performing quotation shifts for "tomorrow."

### Key Findings

| Indexical | Impact of Quotes on Accuracy | Core Cause |
|:---:|:---:|:---|
| I | Minor Drop | Most models have mastered the quotation shift for "I"; only DeepSeek suffers from contextual distraction. |
| you | Negative (Decreased) | Quotes increase task complexity, and models fail to apply the shift rules. |
| here | Positive (Improved) | Quotes help models break free from over-reliance on context. |
| tomorrow | Strongly Negative | Models have a deep-seated non-shifted bias for "tomorrow" that quotes cannot correct. |

## Highlights & Insights

- **First systematic evaluation of LLMs' understanding of English indexicals**: Fills a gap at the intersection of linguistics and NLP. The 2×2 factorial design elegantly distinguishes "following grammatical rules" from "relying on contextual guessing."
- **Reveals diverse behavioral patterns**: Performance varies dramatically across indexicals—"I" is close to human level, whereas "tomorrow" is a complete failure. Quotation marks have a positive impact on "here" but a strongly negative impact on "tomorrow," indicating that LLMs' pragmatic reasoning capabilities are highly uneven.
- **Cross-lingual comparative perspective**: While English "I" is close to human performance, previous work showed Turkish "ben" is extremely poor, suggesting that the language coverage in training data plays a critical role in indexical understanding.

## Limitations & Future Work

- Black-box evaluation: Does not analyze the models' internal representations or attention mechanisms to explain why performance differs so drastically across various indexical words.
- Only evaluates 4 closed-source models, lacking a comparison with open-source models and exploration of potential mitigation strategies.
- Potential data leakage/bias risk since GPT-4o is used to generate the test data while also serving as one of the evaluated models.
- Does not cover other types of indexicals such as "now", "this", or "that".

## Related Work & Insights

- **vs. Oğuz et al. (2024)**: Their evaluation of the Turkish indexical "ben" (I) showed that LLMs performed exceptionally poorly. This work extends the scope to four types of English indexicals, finding that LLMs perform much better on English "I", implying the crucial influence of training data distribution.
- **vs. WinoBias / WinoGrande**: These benchmarks evaluate disambiguation capabilities for third-person pronouns but do not address indexicals. This paper fills this important blank, with an opposite evaluation goal—a good model should ignore context rather than rely on it.

## Rating

- Novelty: ⭐⭐⭐⭐ First evaluation of English indexicals in LLMs, featuring a unique topic with linguistic depth.
- Experimental Thoroughness: ⭐⭐⭐ Good coverage with 4 models × 4 indexicals × 4 conditions, but lacks open-source models and improvement experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear linguistic background, tightly logical 2×2 design, and comprehensive discussion of results.
- Value: ⭐⭐⭐⭐ Unveils deep defects in LLMs' pragmatic reasoning: their reliance on statistical associations over grammatical rules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)
- [\[ACL 2025\] Understanding the Dark Side of LLMs' Intrinsic Self-Correction](understanding_the_dark_side_of_llms_intrinsic_self-correction.md)

</div>

<!-- RELATED:END -->
