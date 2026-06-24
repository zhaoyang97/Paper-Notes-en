---
title: >-
  [Paper Note] AskQE: Question Answering as Automatic Evaluation for Machine Translation
description: >-
  [ACL 2025 (Findings)][Multilingual & Machine Translation][machine translation evaluation] This paper proposes AskQE, a question answering-based quality estimation framework for machine translation. By generating questions from the source text, answering them based on both the source text and the back-translation output, and comparing the answers to detect translation errors, it helps users who do not understand the target language determine the acceptability of translations.…
tags:
  - "ACL 2025 (Findings)"
  - "Multilingual & Machine Translation"
  - "machine translation evaluation"
  - "quality estimation"
  - "question answering"
  - "QA-based evaluation"
  - "error detection"
date: 2026-05-08
content_hash: 617ecf25e22b61b0
---

# AskQE: Question Answering as Automatic Evaluation for Machine Translation

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2504.11582](https://arxiv.org/abs/2504.11582)  
**Code**: [GitHub](https://github.com/dayeonki/askqe)  
**Area**: NLP Understanding/Machine Translation  
**Keywords**: machine translation evaluation, quality estimation, question answering, QA-based evaluation, error detection

## TL;DR
This paper proposes AskQE, a question answering-based quality estimation framework for machine translation. By generating questions from the source text, answering them based on both the source text and the back-translation output, and comparing the answers to detect translation errors, it helps users who do not understand the target language determine the acceptability of translations. On the BioMQM dataset, its Kendall's $\tau$ correlation and decision accuracy outperform existing QE metrics.

## Background & Motivation

**Background**: Machine Translation Quality Estimation (QE) helps users assess translation quality without reference translations. Existing QE methods output a single score (e.g., xCOMET-QE) or error spans (highlighting erroneous words in the target language), which are not suitable for users who do not understand the target language.

**Limitations of Prior Work**: (a) Single scores lack explainability—users do not know *why* the translation has issues; (b) Error spans in the target language are useless for monolingual users; (c) High-risk scenarios (e.g., COVID-19 clinical guideline translation) require actionable feedback to decide whether to "accept or reject".

**Key Challenge**: How can a monolingual English user determine if a French translation is good enough? There is a need for a QE method that provides interpretable feedback *on the source language side*.

**Goal**: Design a QG/QA-based framework to generate questions about the source text and detect whether the translation accurately and completely conveys the source information.

**Key Insight**: If the translation is correct, answers based on the source text and those based on the back-translation should be consistent; inconsistency indicates translation errors. This is analogous to the QA paradigm used in summarization factual consistency evaluation.

**Core Idea**: QG (generating questions from the source text) $\rightarrow$ QA (answering on the source text to obtain $A_{src}$, and on the back-translation to obtain $A_{bt}$) $\rightarrow$ answer discrepancy = translation quality signal.

## Method

### Overall Architecture
Input: Source text $X_{src}$ + machine translation $Y_{tgt}$. Output: AskQE score + explainable QA pairs.

### Key Designs

1. **Question Generation (QG)**

    - **Function**: Generate a set of questions from $X_{src}$ that cover key information.
    - **Mechanism**: A two-step NLI pipeline: (a) extract atomic facts using GPT-4o; (b) filter out non-entailed facts using an NLI classifier; (c) generate questions using LLaMA-3 70B based on $X_{src}$ + filtered facts.
    - **Design Motivation**: Fact-guided generation ensures questions comprehensively cover the source text information, preventing the omission of key content.

2. **Question Answering (QA)—Dual-path Answering**

    - **Source-side Answering $A_{src}$**: Answer each question using the LLM with $X_{src}$ as the context $\rightarrow$ reference answer.
    - **Back-translation-side Answering $A_{bt}$**: Translate $Y_{tgt}$ back to English using Google Translate to obtain $Y_{bt}$, then answer using $Y_{bt}$ as context $\rightarrow$ predicted answer.
    - **Design Motivation**: Use back-translation instead of cross-lingual QA, because English QA is more reliable than cross-lingual QA.

3. **Answer Comparison and Scoring**

    - **Function**: Compute the similarity between $A_{src}$ and $A_{bt}$ as the translation quality score.
    - **Similarity Metrics**: Word-F1, Exact Match, BLEU, chrF, SentenceBERT.
    - **AskQE Score** = average similarity of all question-answer pairs: $$\text{AskQE}(Y_{tgt}) = \frac{1}{N}\sum_{i=1}^N D(A_{src}^i, A_{bt}^i)$$

### Datasets
- **ContraTICO (Controlled Experiments)**: Based on the TICO-19 COVID translation dataset, utilizing GPT-4o to generate 8 types of synthetic perturbations (5 minor + 3 critical) across 5 language pairs.
- **BioMQM (Real-world Errors)**: A biomedical-domain MT dataset annotated with MQM errors by professional translators across 5 language pairs.

## Key Experimental Results

### Error Severity Detection (ContraTICO, LLaMA-3 70B)

| Perturbation Type | Severity | AskQE F1 | AskQE EM |
|---------|--------|----------|----------|
| Spelling Error | Minor | 0.815 | 0.682 |
| Word Swap | Minor | 0.756 | 0.610 |
| Synonym Substitution | Minor | 0.741 | 0.589 |
| Semantic Distortion | **Critical** | **0.496** | **0.384** |
| Information Omission | **Critical** | **0.558** | **0.425** |
| Meaning Expansion | **Critical** | **0.567** | **0.442** |

### Correlation with QE Metrics (Pearson r)

| AskQE Metric | vs xCOMET-QE | vs MetricX-QE | vs BT-Score |
|-----------|-------------|---------------|-------------|
| F1 | 0.871 | -0.923 | 0.877 |
| EM | 0.878 | -0.919 | 0.882 |

### BioMQM Real-world Error Evaluation

| Metric | Kendall's $\tau$ | Decision Accuracy |
|------|------------|-----------|
| xCOMET-QE | 0.42 | 68.5% |
| MetricX-QE | 0.39 | 65.2% |
| BT-Score | 0.35 | 62.8% |
| **AskQE (F1)** | **0.45** | **71.3%** |

### Key Findings
- **AskQE is highly sensitive to critical errors**: The F1 for critical errors is significantly lower than for minor errors (gap >0.2), indicating that semantic alterations and omissions are effectively captured.
- **Analogous to the success of QA evaluation on summarization**: The QA paradigm has been successfully transferred from summarization consistency evaluation to MT evaluation.
- **Outperforms existing QE in decision accuracy**: AskQE is more actionable, providing not only a score but also identifying specific questions for which answers are inconsistent.
- **Back-translation is more reliable than cross-lingual QA**: The accuracy of the English QA system is significantly higher than that of cross-lingual QA.
- **LLaMA-3 70B + NLI fact-guided generation is the optimal configuration**: It yields the best overall performance among 15 configurations.

## Highlights & Insights
- **"Functional Explanations" rather than "Mechanical Explanations"**: Instead of telling users "the third word is mistranslated" (which is meaningless to users who do not know the target language), it shows them "the translation says X instead of Y"—rendering it truly useful for monolingual users.
- **Actionable Decision Support**: Users can review QA pairs one by one to determine which errors are acceptable, offering greater transparency than a single score.
- **Precise Localization in High-Risk Scenarios**: In domains like COVID-19 clinical guidelines, omitting key information can be life-threatening. AskQE can precisely pinpoint such errors.
- **Simple and Elegant Framework**: QG $\rightarrow$ dual-path QA $\rightarrow$ comparison. It requires no specialized training and relies entirely on off-the-shelf LLMs.

## Limitations & Future Work
- **Dependency on Back-translation Quality**: Back-translation introduces additional noise, and back-translations for low-resource language pairs may be unreliable.
- **Sentence-level Evaluation Only**: It has not been extended to paragraph- or document-level translation evaluation.
- **Potential Incompleteness in Question Generation**: If key information is not extracted as an atomic fact, the corresponding error cannot be detected.
- **Computational Cost**: It requires multiple LLM calls per sentence (fact extraction + QG + $2\times\text{QA}$), making it unsuitable for large-scale batch evaluation.
- **English Only as the Source Language**: The effectiveness on non-English source languages has not been validated.

## Related Work & Insights
- **vs QAFactEval (Fabbri et al., 2022)**: QAFactEval utilizes QA to evaluate summarization consistency, whereas AskQE transfers this paradigm to MT evaluation.
- **vs xCOMET-QE (Guerreiro et al., 2024)**: xCOMET provides scores and target-language error spans, whereas AskQE provides source-side QA explanations.
- **vs MTEQA (Krubiński et al., 2021)**: MTEQA requires reference translations, whereas AskQE requires no reference—addressing a true QE scenario.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The transfer of QA evaluation from summarization to MT is creative, and the NLI fact-guided generation is a solid design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Controlled experiments (ContraTICO) + real-world errors (BioMQM) + 5 language pairs + 15 configurations + decision emulation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definition, driven by practical scenarios, and Figure 2's framework diagram is highly lucid.
- **Value**: ⭐⭐⭐⭐⭐ Holds direct practical value for high-risk translation scenarios, with operability far exceeding existing QE methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation](../../ACL2026/multilingual_mt/pear_pairwise_evaluation_for_automatic_relative_scoring_in_machine_translation.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] Has Machine Translation Evaluation Achieved Human Parity?](mt_eval_human_parity.md)
- [\[ACL 2025\] M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)
- [\[ACL 2026\] What Factors Affect LLMs and RLLMs in Financial Question Answering?](../../ACL2026/multilingual_mt/what_factors_affect_llms_and_rllms_in_financial_question_answering.md)

</div>

<!-- RELATED:END -->
