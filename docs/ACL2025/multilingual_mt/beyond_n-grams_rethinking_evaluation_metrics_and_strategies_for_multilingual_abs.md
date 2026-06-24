---
title: >-
  [Paper Note] Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual evaluation] This paper systematically evaluates the correlation of n-gram and neural network evaluation metrics with human judgments across 8 languages (representing 4 morphological typology families). The authors find that n-gram metrics negatively correlate with human judgments in highly fusional languages (Arabic, Hebrew), whereas COMET, a specially trained neural metric, consistently outperforms other methods acr…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual evaluation"
  - "ROUGE"
  - "COMET"
  - "morphological typology"
  - "summarization metrics"
date: 2026-05-08
content_hash: 1291f68fe9a899ac
---

# Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization

**Conference**: ACL 2025  
**arXiv**: [2507.08342](https://arxiv.org/abs/2507.08342)  
**Code**: [https://github.com/itaimondshine/Beyond_ngrams](https://github.com/itaimondshine/Beyond_ngrams)  
**Area**: Multilingual / Machine Translation and Evaluation  
**Keywords**: multilingual evaluation, ROUGE, COMET, morphological typology, summarization metrics

## TL;DR
This paper systematically evaluates the correlation of n-gram and neural network evaluation metrics with human judgments across 8 languages (representing 4 morphological typology families). The authors find that n-gram metrics negatively correlate with human judgments in highly fusional languages (Arabic, Hebrew), whereas COMET, a specially trained neural metric, consistently outperforms other methods across all language typologies.

## Background & Motivation

**Background**: n-gram metrics represented by ROUGE are the de facto standard for summarization evaluation, widely accepted as reasonably correlated with human judgments in English. With the popularity of multilingual LLMs (GPT-4o, Gemini, LLaMA3), the evaluation of non-English generation tasks has surged.

**Limitations of Prior Work**: n-gram metrics rely heavily on whitespace tokenization and exact word matching, failing severely on fusional languages (where multiple morphemes merge into one word, with flexible word order) and agglutinative languages (which feature complex internal word structures). Previous studies have found that BLEU correlates poorly with human judgments in Arabic, and ROUGE even exhibits negative correlation in Hebrew. Meanwhile, neural metrics like BERTScore underperform in low-resource languages due to sparse training data.

**Key Challenge**: Existing studies on multilingual evaluation suffer from three major deficiencies: (1) Insufficient linguistic diversity—failing to cover all morphological typological families; (2) Insufficient metric diversity—focusing heavily on n-gram metrics while neglecting specially trained neural metrics; (3) Insufficient statistical evidence—often failing to report p-values or using inadequate sample sizes (approximately 400 samples are required to detect a significant effect at $p \leq 0.05$).

**Goal**: Conduct the first comprehensive and systematic evaluation of the effectiveness of n-gram and neural metrics across diverse language families. This work covers 4 morphological typologies (isolating, agglutinative, weakly fusional, and highly fusional) and investigates the impacts of tokenization strategies and lemmatization on these metrics.

**Key Insight**: Design a controlled experiment based on linguistic typology: select one high-resource and one low-resource language from each of the 4 typological families (8 languages in total), and collect ~20,000 human annotations to ensure at least 400 samples per language for statistical significance.

**Core Idea**: The morphological typology of a language dictates the reliability of its evaluation metrics. Fusional languages require specially trained neural metrics (such as COMET) rather than n-gram methods.

## Method

### Overall Architecture
Construct a large-scale multilingual summarization evaluation resource: Select 8 languages $\rightarrow$ Generate summaries using GPT-3.5-Turbo and Gemini 1.0 Pro $\rightarrow$ Artificially corrupt 1/3 of the data (to increase score variance) $\rightarrow$ Employ 36 annotators to rate summaries on a scale of 1-4 across coherence and completeness dimensions $\rightarrow$ Calculate Pearson/Spearman correlation coefficients between metrics and human judgments $\rightarrow$ Report p-values and statistical significance.

### Key Designs

1. **Typology-aware Language Selection**:

    - **Function**: Ensure the generalizability of evaluation findings across different morphological typologies.
    - **Mechanism**: Select one high-resource and one low-resource language from four typological groups: isolating (Chinese-H / Yoruba-L), agglutinative (Japanese-H / Turkish-L), weakly fusional (Spanish-H / Ukrainian-L), and highly fusional (Arabic-H / Hebrew-L). High resource is defined as having a token proportion of $\geq 0.1\%$ in the GPT-3 pre-training data distribution.
    - **Design Motivation**: Prior evaluations either ignored highly fusional languages (e.g., Koto et al.) or covered only 3 languages (e.g., Forde et al.), preventing universal conclusions across typological families.

2. **Score Diversification via Corruption**:

    - **Function**: Address the issue where human scores are highly clustered and correlation analysis fails due to the generally high quality of LLM-generated summaries.
    - **Mechanism**: Randomly degrade 1/3 of the dataset along specific dimensions—Coherence: replace nouns/verbs with their lemma forms + shuffle non-adjacent sentence orders; Completeness: replace entities in summaries + insert irrelevant sentences.
    - **Design Motivation**: Preliminary experiments without corruption resulted in overly concentrated scores with extremely low variance, making it impossible to calculate correlation coefficients reliably.

3. **Comprehensive Metric Assessment**:

    - **Function**: Systematically compare three categories of metrics: n-gram, general neural, and specially trained neural.
    - **Mechanism**: Test n-gram metrics (ROUGE-1/2/3/L, BLEU, METEOR, chrF with different tokenizer versions), general neural metrics (BERTScore with multilingual/monolingual encoders, LLM-as-judge via Gemini), and specialized metrics (COMET, trained specifically for translation evaluation). Pearson correlation and p-values are reported for each language-metric combination.
    - **Design Motivation**: Previous studies only compared n-grams or a small subset of neural metrics, lacking a systematic coverage of specially trained evaluation models.

### Annotation Quality Control
Each summary is rated independently by 3 annotators, taking the average. Average Krippendorff's $\alpha$: 0.40 for coherence, 0.47 for completeness (moderate agreement). Hebrew achieves the highest agreement ($\alpha=0.71/0.65$), while Arabic has the lowest ($\alpha=0.32/0.35$).

## Key Experimental Results

### Main Results
Pearson correlation coefficients of ROUGE-1 with human judgments (coherence) across typological families:

| Language Typology | ROUGE-1 | COMET | BERTScore (mBERT) |
|---|---|---|---|
| Isolating | 0.20** | 0.30** | 0.22** |
| Agglutinative | 0.27** | 0.35** | 0.25** |
| Weakly Fusional | 0.11* | 0.25** | 0.15** |
| Highly Fusional | **-0.25*** | 0.20** | 0.05 |

Note: ** indicates p<0.01, * indicates p<0.05

### Ablation Study

| Language (Typology) | Original ROUGE-1 | ROUGE-1 + Lemmatization | Gain |
|---|---|---|---|
| Hebrew (Highly Fusional) | -0.25** | 0.05 | +0.30, eliminates negative correlation |
| Arabic (Highly Fusional) | -0.20** | 0.02 | +0.22, approaches zero correlation |
| Spanish (Weakly Fusional) | 0.11* | 0.15** | +0.04, minor gain |
| Chinese (Isolating) | 0.20** | 0.21** | +0.01, virtually unchanged |

### Key Findings
- ROUGE is significantly negatively correlated with human judgments (-0.25) in highly fusional languages, meaning humans prefer summaries with lower ROUGE scores, rendering the metric completely unreliable.
- Lemmatization significantly improves the performance of n-gram metrics on fusional languages but still fails to outperform COMET.
- COMET consistently outperforms n-gram and general neural metrics across all language typological groups, with especially pronounced advantages in low-resource languages.
- BERTScore is heavily affected by pre-training data volume, dropping significantly in correlation for low-resource languages.
- Summary quality of different LLMs varies across typologies: Gemini scores higher in Elo ranking in highly fusional and low-resource languages, while GPT excels in high-resource settings.

## Highlights & Insights
- This work systematically challenges the "default trust" assumption of n-gram metrics in multilingual environments from a linguistic typology perspective, backed by robust statistical evidence of ~20,000 annotations along with p-value reporting. The negative correlation of ROUGE in highly fusional languages is a striking and impactful discovery.
- Ablation experiments with language-specific tokenizers reveal a practical trick: performing lemmatization on highly fusional languages before calculating ROUGE is low-cost yet effectively eliminates the most severe biases.
- The consistent advantage of COMET suggests that evaluation metrics themselves require specialized "training" to generalize across languages, aligning with the multilingual capability requirements of the evaluated models.

## Limitations & Future Work
- Only two languages are selected per typological family, leaving intra-family variance under-explored (e.g., Persian in highly fusional, French in weakly fusional).
- The evaluation is restricted to summarization tasks; whether conclusions generalize to other generation tasks (dialogue, translation, QA) remains to be validated.
- The 1/3 proportion of corrupted data might introduce evaluation bias, as the distribution of artificially degraded samples may not necessarily reflect natural quality differences.
- The study does not incorporate state-of-the-art LLM-as-judge approaches (such as GPT-4o as an evaluator), relying solely on Gemini 1.0.

## Related Work & Insights
- **vs Koto et al. (2021)**: Only evaluated on 150 samples per language and excluded highly fusional languages, lacking statistical power. In contrast, this work collects 400+ samples per language and comprehensively covers four morphological typologies.
- **vs BERTScore**: Theoretically, BERTScore should adapt well to multiple languages; however, experiments demonstrate it is inferior to the specially trained COMET in low-resource environments, indicating that general pre-training does not automatically translate into robust evaluation capabilities.
- **vs Machine Translation Evaluation**: Initially designed for MT evaluation, COMET is proved by this paper to possess advantages in summarization too, suggesting that "evaluation capabilities" can transfer across tasks.

## Rating
- **Novelty**: ⭐⭐⭐ An evaluation methodology paper whose core contribution lies in empirical findings rather than raw algorithmic novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Benchmarked across 8 languages, ~20K annotations, p-value reporting, and multidimensional ablations—setting a gold standard for evaluation papers.
- **Writing Quality**: ⭐⭐⭐⭐ Marginally organized structure with clean writing and ample background context on linguistic typology.
- **Value**: ⭐⭐⭐⭐ Serves as a vital warning to the multilingual NLG research community: stop mindlessly relying on ROUGE to evaluate non-English generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics](../../ACL2026/multilingual_mt/steereval_inference-time_interventions_strengthen_multilingual_generalization_in.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [\[ACL 2026\] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Evaluation Evaluation](../../ACL2026/multilingual_mt/beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)
- [\[ACL 2025\] LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)

</div>

<!-- RELATED:END -->
