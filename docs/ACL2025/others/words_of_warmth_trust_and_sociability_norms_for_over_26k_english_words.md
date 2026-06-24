---
title: >-
  [Paper Note] Words of Warmth: Trust and Sociability Norms for over 26k English Words
description: >-
  [ACL 2025][warmth] The first large-scale warmth ($W$), trust ($T$), and sociability ($S$) association lexicon (covering 26k+ English words) was constructed through a rigorous crowdsourced annotation process. Its extensive value in social cognition research is demonstrated through analyses of child vocabulary acquisition and social media stereotype case studies.
tags:
  - "ACL 2025"
  - "warmth"
  - "trust"
  - "sociability"
  - "lexicon"
  - "stereotypes"
  - "social cognition"
date: 2026-05-08
content_hash: 65487953c66b1615
---

# Words of Warmth: Trust and Sociability Norms for over 26k English Words

**Conference**: ACL 2025  
**arXiv**: [2506.03993](https://arxiv.org/abs/2506.03993)  
**Code**: [http://saifmohammad.com/warmth.html](http://saifmohammad.com/warmth.html)  
**Area**: Other  
**Keywords**: warmth, trust, sociability, lexicon, stereotypes, social cognition

## TL;DR

The first large-scale warmth ($W$), trust ($T$), and sociability ($S$) association lexicon (covering 26k+ English words) was constructed through a rigorous crowdsourced annotation process. Its extensive value in social cognition research is demonstrated through analyses of child vocabulary acquisition and social media stereotype case studies.

## Background & Motivation

Social psychology research demonstrates that warmth ($W$) and competence ($C$) are two core dimensions for evaluating individuals and groups (Stereotype Content Model, Fiske et al. 2002), profoundly influencing interpersonal interaction, emotion regulation, workplace performance, and political perception. Recent studies have further decomposed warmth into two independent sub-dimensions: trust ($T$, incorporating morality, honesty, reliability, etc.) and sociability ($S$, incorporating friendliness, gregariousness, enthusiasm, etc.), laying a theoretical foundation for finer-grained social cognition analysis.

However, a severe asymmetry exists at the resource level: the competence dimension already has large-scale lexicon annotations (e.g., the NRC VAD Lexicon covering dominance annotations for over 20k words), while the warmth dimension only has extremely small-scale annotated lexicons (Nicolas et al. 2021 covers only 341 words). Prior attempts at automatic expansion using WordNet synonyms and word embeddings yielded poor results—near-synonyms can convey distinct warmth implications (e.g., *slip* vs. *fault*, *skinny* vs. *slender*). Experiments by Fraser et al. (2024) confirmed that automatically expanded lexicons are ineffective at capturing warmth-competence ($W-C$) dimensions.

This resource gap severely limits progress in computational social sciences (stereotype tracking, public discourse analysis), developmental psychology ($WCTS$ patterns in child development), and NLP applications (bias detection, sentiment analysis). This study fills this gap through large-scale crowdsourcing annotation.

## Method

### Overall Architecture

Through a carefully designed crowdsourcing annotation process, trust and sociability scores were annotated for more than 26k English words. A union strategy was then used to generate a composite warmth score, ultimately producing three complementary lexicon resources (Words of Warmth / WCTS Lexicons).

### Key Designs

1. **Vocabulary Source Selection and Filtering**:

    - Function: To identify the set of words to be annotated
    - Mechanism: Filtering approximately 44k words from the NRC VAD Lexicon v2 to exclude highly emotionally neutral words (with valence scores between -0.2 and +0.2), ultimately retaining 26,188 words
    - Design Motivation: To cover a sufficient number of common English words while prioritizing emotionally associated words—which are more likely to carry warmth/coldness connotations, thereby increasing the information density of the annotated resource

2. **Dual-Dimension Independent Annotation System**:

    - Function: To obtain human annotations for the two sub-dimensions of Trust and Sociability separately
    - Mechanism: Designing two independent 7-point Likert scales (ranging from -3 to +3), corresponding respectively to "highly trustworthy/highly untrustworthy" and "highly sociable/highly unsociable". Each word received independent annotations from 9 annotators (12 for Trust), and the scores were averaged
    - Design Motivation: Social cognition theory shows that Trust and Sociability are two separate sub-dimensions of warmth. Independent annotation allows for a more precise capture of differences across dimensions (e.g., *homosexual* having a much lower $T$ score than $S$ score)

3. **Dual-Layer Gold Standard Quality Control**:

    - Function: To ensure annotation quality and prevent cheating
    - Mechanism: Approximately 2% of the questions were pre-annotated control questions, half of which were popup gold (providing immediate popup feedback upon incorrect answers), and the other half were no-popup gold (silently monitored to prevent annotators from sharing control question answers). Annotations from annotators with an accuracy below 80% were discarded
    - Design Motivation: In crowdsourced annotation, annotators might share answers or perform carelessly. The dual-layer mechanism guarantees quality from both immediate correction and covert monitoring directions

### Warmth Score Aggregation Strategy

The Warmth lexicon adopts a union/or strategy of "taking the larger absolute value": for each word, the score with the larger absolute value between Trust and Sociability is designated as the Warmth score. For example, the Warmth score for *uplifting* ($S = 3, T = 0.67$) is 3, while that of *birdbrain* ($S = -1.71, T = -2.62$) is $-2.62$.

This design is grounded in social cognition theory: we perceive an individual as warm either due to trust, friendliness, or both—it does not require the simultaneous satisfaction of both sub-dimensions.

## Key Experimental Results

### Main Results: Annotation Quality Validation

| Dimension | Number of Annotated Words | Average Annotations per Word | SHR (Spearman $\rho$) | SHR (Pearson $r$) |
|------|---------|--------------|------------------|-----------------|
| Sociability | 26,123 | 7.9 | 0.965 | 0.969 |
| Trust | 26,185 | 11.4 | 0.943 | 0.957 |
| Warmth | 26,085 | 8.8 | 0.965 | 0.974 |

The split-half reliability (SHR) for all dimensions exceeded 0.94 (averaged over 1000 random splits), significantly higher than the reliability levels found in similar works (e.g., the SHR for word-anxiety association lexicons is in the 0.8s range).

### Vocabulary Distribution Statistics

| Dimension | Positive Proportion | Neutral Proportion | Negative Proportion |
|------|---------|---------|---------|
| Trust | 28.9% (Trustworthy) | 38.6% | 32.4% (Untrustworthy) |
| Sociability | - | Few | A large number of inanimate objects were rated as slightly unsociable |
| Warmth | 42% (Warm) | 10.5% | 47.5% (Cold) |

### Ablation Study: Child Vocabulary Acquisition Analysis

Based on the age-of-acquisition dataset (30k words) from Kuperman et al. (2012), intersected with the WCTS lexicons:

| Configuration | Key Metric | Description |
|------|---------|------|
| Polar $W$ words vs. Polar $C$ words | The proportion of polar $W$ words is consistently higher than polar $C$ words across all age groups | Supports the "primacy of valence hypothesis" |
| Early High Warmth | Toddlers acquire more high $W$ words and low $C$ words | Consistent with child development characteristics of relying on the warmth of caregivers |
| $S$ before $T$ | The proportion of early polar $S$ words is much higher than polar $T$ words | The sociability dimension is more important in early childhood than the trust dimension |
| $T$ increases with age | Low-trust words are extremely rare at age 3 and steadily increase with age | Moral/trust concepts develop progressively |

### Key Findings: Stereotype Case Studies

Using US/Canadian Twitter corpus from 2015-2021:

| Analysis Type | Goal | Key Findings |
|---------|------|---------|
| Direct Query | $W-C$ coordinates of social group words | *god*: high $W$, high $C$; *criminal*: extremely low $W$; *disabled*: extremely low $C$ |
| Co-occurrence Word Analysis | $WCTS$ co-occurrence in the Twitter corpus | *muslim*/*jew*/*immigrant* obtain low $W$ scores (reflecting negative stereotypes) |
| Gender Differences | Gender pronouns/titles | Males are higher in $C$, females are higher in $W$ (manifestation of gender stereotypes in language) |
| In-group vs. Out-group | Canadian vs. American mutual evaluations | Canadians self-evaluate as warmer and more competent (in-group favoritism); Americans also perceive Canadians as warmer |
| Pronoun Analysis | *I*/*me*/*you*/*we* | *I*/*me* has low $C$ (consistent with the low-status marker hypothesis); *we* has high $W$ (associated with positive contexts) |

## Highlights & Insights

- Qualitative leap in resource scale: over 75 times larger than the previous largest warmth lexicon (341 words), transitioning from a "toy dataset" to a genuinely usable research tool.
- Independent annotation of $T$ and $S$ reveals internal heterogeneity within the warmth dimension—for example, *homosexual* has a much lower $T$ score than $S$ score, reflecting how certain groups associate homosexuality with "unmorality" rather than "unfriendliness".
- The dual-layer gold standard quality control (popup + no-popup gold) can be generalized to other crowdsourcing annotation tasks.
- Vocabulary acquisition analysis provides new evidence for computational psycholinguistics: the age distribution of polar warmth words versus competence words clearly supports the primacy of valence hypothesis.
- The co-occurrence $WCTS$ analysis method is simple and efficient—capturing complex social cognitive patterns using only word frequency statistics.

## Limitations & Future Work

- The lexicon only covers English words, with cross-lingual/cross-cultural versions yet to be developed—the weight of $W-C$ dimensions may vary across cultures.
- Annotators are predominantly from the US (69%), followed by India/UK/Canada, presenting potential cultural bias.
- Word-level annotations cannot capture dynamic warmth perceptions in context—the same word can convey different shades of warmth across different contexts.
- Geolocation in the Twitter case studies may contain noise.
- The integration effects with downstream NLP tasks (sentiment analysis, bias detection, hate speech/toxicity filtering) were not directly evaluated.
- Lexicon entries reflect cultural perceptions at the time of annotation and are not static—requiring regular updates.

## Related Work & Insights

- **vs. NRC VAD Lexicon (Mohammad 2018/2025)**: The VAD lexicon covers the Competence (dominance) dimension, whereas this study addresses the gap in the Warmth dimension; their combination forms the complete $WCTS$ framework.
- **vs. Nicolas et al. (2021)**: A warmth lexicon of only 341 words, which shows poor results via WordNet/embedding automatic expansions; this study fundamentally resolves the coverage issue through large-scale manual annotation.
- **vs. Word Embedding Bias Research (Caliskan et al. 2017)**: WEAT detects implicit biases via embedding spaces; the word-level $WCTS$ annotation in this work offers a more direct and interpretable analytical path.
- **vs. Generative AI Bias Research (Kotek et al. 2023)**: Words of Warmth can be utilized to systematically evaluate warmth/competence bias in LLM-generated texts.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale $W-T-S$ lexicon, filling an important resource gap; decomposing warmth into Trust and Sociability for independent annotations is a key innovation
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation of reliability, vocabulary acquisition analysis, multi-dimensional stereotype case studies (social groups, gender, in-group/out-group, pronouns), demonstrating the broad applicability of the lexicon
- Writing Quality: ⭐⭐⭐⭐⭐ Saif Mohammad's consistently high-quality writing, featuring clear organization and responsible, thorough ethical discussion
- Value: ⭐⭐⭐⭐ As a fundamental resource contribution, it possesses long-term influence on social cognition research, bias detection, and computational social sciences

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](neodiff_unified_text_diffusion.md)

</div>

<!-- RELATED:END -->
