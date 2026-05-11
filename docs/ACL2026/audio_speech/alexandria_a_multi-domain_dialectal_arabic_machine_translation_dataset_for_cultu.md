---
title: >-
  [Paper Note] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs
description: >-
  [ACL 2026][Audio & Speech][Dialectal Arabic] Alexandria constructs a parallel English-Dialectal Arabic multi-round dialogue dataset covering 13 Arabic countries, 11 social impact domains…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Dialectal Arabic"
  - "Machine Translation"
  - "Multi-domain Dataset"
  - "Cultural Inclusion"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 581912fec071cedb
---

# Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs

**Conference**: ACL 2026  
**arXiv**: [2601.13099](https://arxiv.org/abs/2601.13099)  
**Code**: [https://github.com/UBC-NLP/Alexandria](https://github.com/UBC-NLP/Alexandria)  
**Area**: Audio & Speech  
**Keywords**: Dialectal Arabic, Machine Translation, Multi-domain Dataset, Cultural Inclusion, LLM Evaluation

## TL;DR

Alexandria constructs a parallel English-Dialectal Arabic multi-round dialogue dataset covering 13 Arabic countries, 11 social impact domains, and 107K turns. Through a community-driven manual translation and revision process, it provides unprecedented fine-grained training and evaluation resources for dialectal Arabic machine translation, accompanied by a systematic benchmark assessment across 24 LLMs.

## Background & Motivation

**Background**: Neural Machine Translation has achieved significant progress in high-resource language pairs. However, Arabic faces severe "diglossia" challenges: daily communication primarily uses regional dialects, while MT systems are mainly trained on Modern Standard Arabic (MSA), leading to poor generalization on dialectal inputs.

**Limitations of Prior Work**: Existing dialectal Arabic resources suffer from three major limitations: (1) scale is severely insufficient (PADIC covers only ~6,400 sentences/dialect, MADAR only 2,000); (2) narrow domain coverage (MADAR focuses on travel, lacking social impact domains like health, education, and agriculture); (3) coarse granularity (labels are restricted to regions like "Levantine" or "Maghrebi," lacking city-level variants and metadata like gender configurations or code-switching).

**Key Challenge**: The daily dialectal communication needs of millions of Arabic speakers vs. the systematic neglect of dialects by MT systems and the scarcity of evaluation resources.

**Goal**: To build a large-scale, multi-domain, city-level dialectal Arabic parallel dataset serving as both a training resource and an evaluation benchmark, fully revealing the capabilities and deficiencies of current LLMs in dialect translation.

**Key Insight**: A community-driven model was adopted, recruiting 55 participants from 13 Arabic countries (including 29 women), each associated with a specific city to ensure the authenticity and localization of dialects.

**Core Idea**: By utilizing city-level labeling, gender configuration metadata, 11-domain coverage, and a manual translation-revision workflow, the dataset significantly exceeds existing resources in scale and granularity, providing the first comprehensive evaluation framework for dialectal Arabic MT.

## Method

### Overall Architecture

The construction of Alexandria consists of three stages: (1) using Gemini-2.5 Pro to generate multi-round English dialogue scenarios conditioned on target countries and domains; (2) manual translation into dialectal Arabic by native speakers; (3) cross-review and revision by peers within the same country. The final output is turn-aligned English-Dialectal Arabic parallel multi-round dialogues, totaling 34,488 conversations and 107K turns.

### Key Designs

1. **Two-Stage English Source Text Generation Pipeline**:
    - **Function**: Generate diverse and culturally appropriate multi-round English dialogues for each country-domain pair.
    - **Mechanism**: Phase 1 generates 550 topic specifications (55 sub-domains × 10 topics) per country-domain pair, including roles and gender attributes. Phase 2 generates 2-4 turn dialogues based on these topics. English paraphrases replace Arabic transliterations (e.g., "God willing" instead of "inshallah") to avoid lexical leakage. Semantic diversity was verified via t-SNE visualization, with a mean cosine similarity of only 0.20.
    - **Design Motivation**: To avoid the single-domain and short-sentence restrictions of datasets like MADAR, while ensuring translations are based on semantic transfer rather than surface transcription by prohibiting transliteration seepage.

2. **Community-Driven City-Level Dialect Data Collection**:
    - **Function**: Ensure authenticity and geographical diversity of dialect data.
    - **Mechanism**: 55 participants from different cities in 13 countries translated dialogues into their respective city dialects. Each country was coordinated by a country lead to ensure tagging consistency. Data is linked to city-origin metadata to support sub-dialectal analysis. Speaker $\rightarrow$ listener gender configurations were also labeled (F$\rightarrow$M 33.19%, M$\rightarrow$F 32.78%, M$\rightarrow$M 21.43%, F$\rightarrow$F 12.60%).
    - **Design Motivation**: Previous resources used coarse regional labels, failing to capture systematic differences between cities within the same country (e.g., Ramallah vs. Shuqba in Palestine).

3. **Peer Review Revision and Quality Assurance**:
    - **Function**: Ensure translation quality through cross-verification.
    - **Mechanism**: Each translation was cross-evaluated by a second participant from the same country across six dimensions: dialect authenticity, gender alignment, register appropriateness, semantic faithfulness, punctuation, and code-switching consistency. Ultimately, 68.4% of turns required no modification, 30.6% required minor edits, and only 1% had major issues.
    - **Design Motivation**: English source texts generated by LLMs may contain unnatural phrasing or cultural mismatches; manual translations also require systematic QA to ensure data reliability.

### Evaluation Setup

Three input settings: (1) Turn-level; (2) Context-level; (3) Conversation-level. Automatic evaluation used spBLEU and chrF++, avoiding COMET due to its limited reliability for dialects. Human evaluation covered semantic adequacy (XSTS 1-5 scale), gender accuracy (Pass/Fail), and dialectness/fluency (1-5 scale).

## Key Experimental Results

### Main Results

**English$\rightarrow$Dialect Context-Level spBLEU (Representative Models and Dialects)**

| Model | SA | EG | SY | LB | MA | MR |
|------|-----|-----|-----|-----|-----|-----|
| Gemini-2.5-Pro | 31.4 | 27.1 | 34.4 | 27.8 | 20.3 | 8.2 |
| Gemini-3-Flash | 29.6 | 27.8 | 31.1 | 27.9 | 19.5 | 10.1 |
| Command-A | 29.2 | 25.8 | 29.0 | 19.5 | 18.0 | 8.9 |
| Gemma-3-27b | 30.0 | 25.7 | 26.8 | 21.3 | 17.3 | 7.4 |
| Qwen3-32B | 17.6 | 14.8 | 15.2 | 10.4 | 13.2 | 4.4 |
| ALLaM-7B | 12.5 | 10.4 | 10.3 | 7.1 | 8.9 | 2.5 |

### Ablation Study

**Metadata Ablation (Single-turn English$\rightarrow$Dialect spBLEU)**

| Model | Metadata | EG | SA | SY | MA |
|------|--------|-----|-----|-----|-----|
| gemma-3-12b | None | 25.54 | 25.65 | 25.79 | 11.33 |
| gemma-3-12b | Full | 25.11 | 24.39 | 24.90 | 11.34 |
| Command-A | None | 28.78 | 28.88 | 27.74 | 18.60 |
| Command-A | Full | 29.45 | 29.40 | 26.96 | 20.01 |
| NLLB-200-3.3B | N/A | 17.16 | 17.96 | 22.24 | 9.82 |

**Thinking Mode Ablation**: Only Gemini-3-Flash improved by ~2.0 spBLEU via reasoning; other models showed performance degredation with reasoning enabled.

### Key Findings

- Significant directional asymmetry: Dialect$\rightarrow$English quality is consistently better than English$\rightarrow$Dialect, suggesting generating dialect is harder than understanding it.
- Models perform best on Levantine and Egyptian dialects; Maghrebi dialects (especially Mauritanian) are the most challenging.
- Gemini series is the strongest in both directions; a large gap exists for small open-source models (ALLaM-7B, Fanar-9B).
- Human evaluation reveals dialectness/fluency (~2-3/5) is significantly lower than semantic adequacy (>3/5) for all models, indicating a tendency to generate MSA-like outputs.
- Code-switching (using Latin characters) significantly degrades translation quality, with Moroccan and Tunisian dialects most affected.
- Lexical overlap with MSA correlates positively with translation quality (Saudi $r=0.48$, Yemen $r=0.44$).

## Highlights & Insights

- The community-driven dataset construction methodology is exemplary: city-level labeling + country lead coordination + peer review balances scale and quality.
- Gender configuration labeling (F$\rightarrow$M, M$\rightarrow$F) represents a unique and essential requirement for Arabic MT evaluation, filling a critical gap.
- The 107K turn scale far exceeds PADIC (38K) and MADAR (100K), while covering 11 high-impact social domains.
- Sub-dialectal analysis reveals systematic translation difficulty variances within countries, with model rankings being highly consistent across sub-dialects.
- Metadata effectiveness varies by model—"more information" is not always better, as some models degraded under the Full metadata condition.

## Limitations & Future Work

- Imbalanced gender distribution: F$\rightarrow$F accounts for only 12.60% due to LLM generation biases toward mixed-gender scenarios.
- Technical terminology translation difficulties led to some MSA seepage in specific domains.
- Budget constraints limited closed-source evaluation primarily to the Gemini series.
- Not all Arabic dialects are covered (e.g., Iraq, Bahrain are missing).

## Related Work & Insights

- **vs MADAR**: Alexandria surpasses it in scale (107K vs 100K), domains (11 vs 1), and granularity (city-level + gender + code-switching).
- **vs FLORES+**: While FLORES+ is reported to have dialect portions too close to MSA, Alexandria avoids this through native speaker translations.
- **vs NLLB-200**: All evaluated LLMs consistently outperform NLLB-200-3.3B, even without metadata.

## Rating

- Novelty: ⭐⭐⭐⭐ City-level granularity, gender configuration labeling, and 11-domain coverage are unprecedented in dialectal Arabic resources.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24 models, 13 dialects, auto + human evaluation, and multi-dimensional ablations make it extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed data, and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Fills a major resource gap in dialectal Arabic MT with high practical value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ACL 2026\] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)
- [\[ICLR 2026\] Statistical Guarantees for Offline Domain Randomization](../../ICLR2026/audio_speech/statistical_guarantees_for_offline_domain_randomization.md)
- [\[NeurIPS 2025\] Merlin L48 Spectrogram Dataset](../../NeurIPS2025/audio_speech/merlin_l48_spectrogram_dataset.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)

</div>

<!-- RELATED:END -->
