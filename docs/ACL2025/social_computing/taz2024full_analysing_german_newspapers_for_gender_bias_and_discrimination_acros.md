---
title: >-
  [Paper Note] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades
description: >-
  [ACL 2025][Social Computing][German Corpus] Constructed the largest publicly available German news corpus to date, taz2024full (1.8M+ articles, 1980–2024), and adapted an actor-level discourse analysis pipeline to German, revealing persistent gender representation imbalances and sentiment biases in news reporting over more than four decades.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "German Corpus"
  - "Gender Bias"
  - "Media Analysis"
  - "Longitudinal Study"
  - "NER"
date: 2026-05-08
content_hash: fcba6b8b0cf81627
---

# taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades

**Conference**: ACL 2025  
**arXiv**: [2506.05388](https://arxiv.org/abs/2506.05388)  
**Code**: [Ognatai/corpus_pipeline](https://github.com/Ognatai/corpus_pipeline)  
**Area**: Social Computing / Bias Detection  
**Keywords**: German Corpus, Gender Bias, Media Analysis, Longitudinal Study, NER

## TL;DR

Constructed the largest publicly available German news corpus to date, taz2024full (1.8M+ articles, 1980–2024), and adapted an actor-level discourse analysis pipeline to German, revealing persistent gender representation imbalances and sentiment biases in news reporting over more than four decades.

## Background & Motivation

**Background**: News media is the core channel for the public to obtain information, and gender bias in the media can subtly shape societal perceptions. However, long-term longitudinal quantitative studies on gender bias are extremely scarce, primarily limited by the lack of large-scale, long-duration corpora.

**Limitations of Prior Work**: The open corpus ecosystem is dominated by English, and large-scale German news resources are severely lacking. Existing German corpora (such as DWDS, DeReKo, Leipzig Wortschatz, etc.) are mostly limited to keyword searches or sentence-level queries, and do not support full-volume downloading and large-scale analysis. The vast majority of German newspapers cannot be made publicly available due to licensing fees and legal restrictions. Additionally, existing bias detection methods are mostly based on word embeddings or statistical associations, lacking discourse analysis at the actor level.

**Key Challenge**: Answering "how gender bias evolves over time" requires two simultaneous conditions: a large-scale public corpus spanning decades and an automated pipeline capable of analyzing gender representation at the actor granularity. Currently, neither exists in the German NLP ecosystem.

**Goal**: (1) Construct and publicly release the first large-scale German news corpus supporting a 44-year longitudinal study; (2) Adapt the actor-level bias detection pipeline of Urchs et al. (2024) from English to German, supporting full-corpus analysis.

**Key Insight**: Utilizing the Berlin left-wing daily newspaper *taz* as the only German newspaper source that allows free academic use, all public articles from 1980 to 2024 were scraped. Traditional interpretable methods (rather than LLMs) were deliberately chosen for bias analysis to avoid the methodological contradiction of "detecting bias with biased tools".

**Core Idea**: The combination of large-scale corpus construction and the German adaptation of an actor-level discourse analysis pipeline enables a longitudinal quantitative study of gender bias in news over a 44-year span.

## Method

### Overall Architecture

The work consists of two core components: (1) Corpus construction—scraping all public articles of the Berlin left-wing daily *taz* from 1980 to 2024 and storing them structurally; (2) Bias analysis pipeline—adapting the actor-level discourse analysis method of Urchs et al. (2024) to German, scaling from single-document processing to full-corpus analysis, and outputting annual discrimination reports.

### Key Designs

1. **Large-Scale German Corpus Construction (taz2024full)**:

    - **Function**: Provides the first publicly available, large-scale German news corpus covering 44 years.
    - **Mechanism**: The data source is the German left-wing daily *taz* (*die Tageszeitung*), which is the only German newspaper that allows free academic use. All public articles were scraped between August and November 2024 and stored in JSON format. Each record contains metadata (publishing date, author, keywords, token count, and whether it contains person entities) and text (headline, lead paragraph, body). The SoMaJo tokenizer was used for processing, and fragments of $\le 3$ tokens were filtered out. The final corpus is publicly released on Zenodo.
    - **Design Motivation**: Existing German corpora do not support full-volume downloading and large-scale analysis, and most cannot be made public due to copyright. The liberal licensing policy of *taz* makes it the only viable data source.

2. **Actor-Level Bias Detection Pipeline**:

    - **Function**: Automatically analyzes gender representation at the actor granularity and outputs annual discrimination reports.
    - **Mechanism**: The pipeline process is: NER person entity extraction (spaCy) $\rightarrow$ generic referent word supplementation (e.g., "mother", "father") $\rightarrow$ merging identical entities $\rightarrow$ coreference resolution (coreferee) linking pronouns to corresponding actors $\rightarrow$ gender classification based on pronoun distribution (classified as female/male if $> 70\%$ feminine/masculine pronouns) $\rightarrow$ extracting discrimination indicators for each actor (count, mention frequency, sentiment polarity, gendered terms, highest PMI adjectives). German adaptation includes: switching to a pronoun-driven analysis, retaining only actors with coreference chains, adding generic masculine and gender-neutral language detection, and replacing sentiment analysis with `german-sentiment-bert`.
    - **Design Motivation**: Adapting the English pipeline of Urchs et al. (2024) to German ensures method comparability while addressing grammatical gender issues specific to German.

3. **Methodological Choice of Deliberately Avoiding LLMs**:

    - **Function**: Ensures that the bias detection toolchain itself does not introduce systematic bias.
    - **Mechanism**: Choosing traditional interpretable methods (spaCy + coreferee + PMI) prevents the gender/political biases inherent in LLMs from contaminating the analysis results. Discrimination analysis is output as reports rather than final judgments, leaving the interpretation to the researcher.
    - **Design Motivation**: Using biased tools to detect bias is methodologically contradictory; traditional methods produce transparent and auditable outputs, ensuring the credibility of the research conclusions.

## Key Experimental Results

### Main Results

| Metric | Value |
|------|------|
| Total Articles | 1,834,370 |
| Time Span | 1980–2024 (44 years) |
| Unique Tokens | 6,944,197 |
| Ratio of Articles with Person Entities | 83% |
| Average Token Length | 5.15 characters |
| Average Sentence Length | 20.07 tokens |
| Average Article Length | 396.89 tokens / 19.77 sentences |
| Median Article Length | 276 tokens / 13 sentences |
| Peak Article Count | Year 2004 (73,002 articles) |
| Trend after 2007 | Ongoing decline in public articles due to increased paywalled content |

### Ablation Study

| Dimension | Result |
|---------|------|
| Actor Gender Ratio | Since the 1990s, the number and mention frequency of male actors have been consistently and significantly higher than female actors. |
| Temporal Evolution | Since the 2010s, the proportion of female actors included has gradually increased, but male mentions still dominate. |
| Media Visibility | Even as the number of actors approached balance in recent years, males still receive more text space. |
| Sentiment Polarity | Overall slightly negative; over the 44 years, the sentiment score for female actors has always been slightly lower than for male actors. |
| Gender-Coded Lexicon | Very little usage of gendered words from the Gender Decoder lexicon. |
| PMI Adjectives | High-PMI adjectives do not differ much between genders and remain stable over time. |
| Gender-Neutral Language | *taz* has barely systematically adopted German gender-neutral language (Gendern). |
| Neo-pronouns | Only 5 articles in the entire corpus contain German neo-pronouns. |

### Key Findings

- **Male dominance throughout forty years**: Whether in terms of actor occurrences or mention frequencies, males consistently dominate; even as the number of actors neared balance after the 2010s, the gap in mentions persisted.
- **Persistent subtle sentiment bias**: The sentiment of descriptions of female actors is consistently slightly lower than that of male actors; although the difference is small, it has never reversed in 44 years.
- **Blind spots in progressive media**: As a left-wing progressive media outlet, *taz* has not been at the forefront of using gender-neutral language.

## Highlights & Insights

- **Longitudinal gender bias research covering 44 years is highly rare**: Provides a unique long-term perspective, revealing that bias elimination is a much slower process than imagined.
- **The design of avoiding LLMs aligns with methodological consistency**: Avoiding the methodological contradiction of "detecting bias with biased tools" is highly exemplary for bias research.
- **Language-agnostic pipeline design**: The core pipeline can be migrated to other languages. The discrimination reports do not make final judgments, leaving them to the users' interpretation.

## Limitations & Future Work

- The data only originates from a single left-wing media outlet (*taz*) and cannot represent the entire German media ecosystem.
- The accuracy of German coreference resolution models is limited, impacting gender inference precision.
- Limited to binary gender analysis (German lacks generalized non-binary pronouns).
- The decline in available articles post-2007 due to increased paywalled content may introduce sampling bias.

## Related Work & Insights

- **vs Urchs et al. (2024)**: They proposed an English actor-level bias detection pipeline. This work adapts it to German, extending it with generic masculine detection and PMI analysis, proving the cross-lingual transferability of the framework.
- **vs Embedding-based bias detection methods**: Methods like WEAT/SEAT detect bias at the word/sentence embedding level, whereas this work analyzes at the actor level, providing finer granularity and directly mapping to real-world news reporting.

## Rating
- **Novelty**: ⭐⭐⭐ Corpus construction is valuable, but technical methods are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐ Longitudinal analysis over 44 years is comprehensive, but comparison with other media outlets is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth and thorough discussion of research motivations and methodological choices.
- **Value**: ⭐⭐⭐⭐ The large-scale public German corpus itself is a significant resource contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[ACL 2026\] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](../../ACL2026/social_computing/gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](../../NeurIPS2025/social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)

</div>

<!-- RELATED:END -->
