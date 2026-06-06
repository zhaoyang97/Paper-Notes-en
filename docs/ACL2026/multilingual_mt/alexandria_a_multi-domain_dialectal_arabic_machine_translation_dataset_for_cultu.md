---
title: >-
  [Paper Note] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs
description: >-
  [ACL 2026][Multilingual & Machine Translation][Dialectal Arabic] Alexandria constructs a multi-turn dialogue parallel dataset for Dialectal Arabic-English covering 13 Arab countries, 11 social impact domains…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Dialectal Arabic"
  - "Machine Translation"
  - "Multi-domain Dataset"
  - "Cultural Inclusion"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 37a13c2bf8d7c7bf
---

# Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs

**Conference**: ACL 2026  
**arXiv**: [2601.13099](https://arxiv.org/abs/2601.13099)  
**Code**: [https://github.com/UBC-NLP/Alexandria](https://github.com/UBC-NLP/Alexandria)  
**Area**: Audio and Speech  
**Keywords**: Dialectal Arabic, Machine Translation, Multi-domain Dataset, Cultural Inclusion, LLM Evaluation

## TL;DR

Alexandria constructs a multi-turn dialogue parallel dataset for Dialectal Arabic-English covering 13 Arab countries, 11 social impact domains, and 107K turns. Through a community-driven human translation and revision process, it provides unprecedented fine-grained training and evaluation resources for Dialectal Arabic machine translation and conducts a systematic benchmark evaluation on 24 LLMs.

## Background & Motivation

**Background**: Neural Machine Translation (NMT) has achieved significant progress in high-resource language pairs. However, Arabic faces a severe "diglossia" challenge—daily communication primarily uses regional dialects, while MT systems are mainly trained on Modern Standard Arabic (MSA). This results in extremely poor generalization to dialectal inputs.

**Limitations of Prior Work**: Existing Dialectal Arabic resources have three major constraints: (1) Insufficient scale, with PADIC covering only ~6,400 sentences/dialect and MADAR only 2,000; (2) Narrow domain coverage, where MADAR focuses on tourism and lacks social impact domains like health, education, and agriculture; (3) Coarse granularity, providing only regional labels like "Levantine" or "North African" while lacking city-level dialectal variations and metadata such as gender configurations or code-switching annotations.

**Key Challenge**: The daily dialectal communication needs of millions of Arabic speakers vs. the systematic neglect of dialects by MT systems and the lack of evaluation resources.

**Goal**: Construct a large-scale, multi-domain, city-level parallel dataset for Dialectal Arabic to serve as both a training resource and an evaluation benchmark, comprehensively revealing the capabilities and deficiencies of current LLMs in dialectal translation.

**Key Insight**: Adopt a community-driven model by recruiting 55 participants from 13 Arab countries (including 29 women), with each participant associated with a specific city to ensure the authenticity and localized features of the dialects.

**Core Idea**: Through city-level annotations, gender configuration metadata, coverage of 11 domains, and a human translation-revision workflow, the project significantly surpasses existing resources in scale and granularity, providing the first comprehensive evaluation framework for Dialectal Arabic MT.

## Method

### Overall Architecture

The construction of Alexandria consists of three stages: (1) Generating multi-turn English dialogue scenarios using Gemini-1.5 Pro, conditioned on target countries and domains; (2) Human translation into Dialectal Arabic by native speakers; (3) Peer review and revision by participants from the same country. The final output is turn-aligned English-Dialectal Arabic parallel multi-turn dialogues, totaling 34,488 dialogues and 107K turns.

### Key Designs

1.  **Two-stage English Source Text Generation Pipeline**:
    *   **Function**: Generates diverse and culturally appropriate multi-turn English dialogues for each country-domain pair.
    *   **Mechanism**: Phase 1 generates 550 topic specifications for each pair (55 sub-domains × 10 topics), including roles and gender attributes. Phase 2 generates 2-4 turn dialogues based on these topics. English paraphrases are used instead of Arabic transliterations (e.g., "God willing" instead of "inshallah") to avoid lexical leakage. Semantic diversity is verified via t-SNE visualization, with a mean cosine similarity of only 0.20.
    *   **Design Motivation**: To avoid the single-domain and short-sentence limitations of datasets like MADAR, while ensuring translation is based on semantic transfer rather than surface transcription by prohibiting transliteration seepage.

2.  **Community-Driven City-Level Dialect Data Collection**:
    *   **Function**: Ensures authenticity and geographical diversity of dialectal data.
    *   **Mechanism**: 55 participants from different cities across 13 countries translated dialogues corresponding to their city dialects. Each country was coordinated by a country lead to ensure annotation consistency. Data is linked to city-of-origin metadata to support sub-dialectal analysis. Metadata also includes speaker→listener gender configurations (F→M 33.19%, M→F 32.78%, M→M 21.43%, F→F 12.60%).
    *   **Design Motivation**: Previous resources used coarse regional labels, failing to capture systematic dialectal differences between cities within the same country (e.g., Ramallah vs. Shuqba in Palestine).

3.  **Peer Review Revision and Quality Assurance**:
    *   **Function**: Ensures translation quality through cross-verification.
    *   **Mechanism**: Each translation is cross-evaluated by a second participant from the same country across six dimensions: dialect authenticity, gender alignment, register appropriateness, semantic faithfulness, punctuation, and code-switching consistency. Results showed 68.4% of turns required no modification, 30.6% needed minor edits, and only 1% had major issues.
    *   **Design Motivation**: LLM-generated English source texts may contain unnatural phrasing or cultural mismatches, and human translations require systematic quality assurance to ensure data reliability.

### Evaluation Setup

Three input settings: (1) Turn-level (translating a single turn); (2) Context-level (translating the current turn given previous dialogue history); (3) Conversation-level (translating the entire dialogue at once). Automatic evaluation uses spBLEU and chrF++, avoiding COMET (which has limited reliability for dialects). Human evaluation covers semantic adequacy (5-point XSTS), gender accuracy (Pass/Fail), and dialectalness/fluency (1-5 scale).

## Key Experimental Results

### Main Results

**English→Dialect Context-Level spBLEU (Representative Models and Dialects)**

| Model | SA | EG | SY | LB | MA | MR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-1.5-Pro | 31.4 | 27.1 | 34.4 | 27.8 | 20.3 | 8.2 |
| Gemini-1.5-Flash | 29.6 | 27.8 | 31.1 | 27.9 | 19.5 | 10.1 |
| Command-R | 29.2 | 25.8 | 29.0 | 19.5 | 18.0 | 8.9 |
| Gemma-2-27b | 30.0 | 25.7 | 26.8 | 21.3 | 17.3 | 7.4 |
| Qwen2-32B | 17.6 | 14.8 | 15.2 | 10.4 | 13.2 | 4.4 |
| ALLaM-7B | 12.5 | 10.4 | 10.3 | 7.1 | 8.9 | 2.5 |

### Ablation Study

**Metadata Ablation (Single-turn English→Dialect spBLEU)**

| Model | Metadata | EG | SA | SY | MA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gemma-2-9b | None | 25.54 | 25.65 | 25.79 | 11.33 |
| gemma-2-9b | Full | 25.11 | 24.39 | 24.90 | 11.34 |
| Command-R | None | 28.78 | 28.88 | 27.74 | 18.60 |
| Command-R | Full | 29.45 | 29.40 | 26.96 | 20.01 |
| NLLB-200-3.3B | N/A | 17.16 | 17.96 | 22.24 | 9.82 |

**Thinking Mode Ablation**: Only Gemini models showed improvement through reasoning (approx. 2.0 spBLEU), while reasoning decreased performance for other models.

### Key Findings

*   Significant directional asymmetry exists: Dialect→English translation quality is consistently better than English→Dialect, indicating that generating dialects is harder than understanding them.
*   Models perform best on Levantine and Egyptian dialects, while Maghrebi dialects (especially Mauritanian) are the most challenging.
*   The Gemini series demonstrates the strongest performance in both directions; a large gap exists for small open-source models (ALLaM-7B, Fanar-9B).
*   Human evaluation reveals that dialect authenticity/fluency (~2-3/5) is significantly lower than semantic adequacy (>3/5) for all models, suggesting a tendency to generate outputs closer to MSA.
*   Code-switching (using Latin characters) significantly degrades translation quality, with Moroccan and Tunisian dialects being the most affected.
*   Lexical overlap with MSA correlates positively with translation quality (Saudi $r=0.48$, Yemen $r=0.44$).

## Highlights & Insights

*   The community-driven dataset construction methodology is highly transferable: city-level annotation + country lead coordination + peer cross-revision balances scale and quality.
*   Gender configuration labeling (F→M, M→F, etc.) is a unique requirement for Arabic MT evaluation and fills a critical gap.
*   The scale of 107K turns far exceeds PADIC (38K) and MADAR (100K) and covers 11 high-social-impact domains.
*   Sub-dialectal analysis reveals systematic differences in translation difficulty within countries, with model rankings remaining highly consistent across sub-dialects.
*   The effect of metadata varies by model—"more information" is not always better; some models see performance drops under Full metadata conditions.

## Limitations & Future Work

*   Gender distribution imbalance: F→F accounts for only 12.60%, stemming from LLM generation biases toward mixed-gender scenarios.
*   Difficulty in translating technical terms leads to MSA seepage in certain domains.
*   Closed-source model evaluation was limited by budget, testing only the Gemini series.
*   Does not cover all Arabic dialects (e.g., Iraqi and Bahraini are not included).

## Related Work & Insights

*   **vs MADAR**: Alexandria surpasses it in scale (107K vs 100K), number of domains (11 vs 1), and annotation granularity (city-level + gender + code-switching).
*   **vs FLORES+**: FLORES+ has been reported to have dialect sections too close to MSA; Alexandria avoids this through native speaker translations.
*   **vs NLLB-200**: All evaluated LLMs consistently outperform NLLB-200-3.3B even without metadata.

## Rating

*   Novelty: ⭐⭐⭐⭐ City-level dialectal granularity, gender configuration labeling, and 11-domain coverage are unprecedented in Dialectal Arabic resources.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24 models, 13 dialects, automatic + human evaluation, and multi-dimensional ablations make it extremely comprehensive.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed data, and rich visualizations.
*   Value: ⭐⭐⭐⭐⭐ Fills a major resource gap in Dialectal Arabic MT and holds high practical value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[ACL 2026\] TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)

</div>

<!-- RELATED:END -->
