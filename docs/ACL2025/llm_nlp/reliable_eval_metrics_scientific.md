---
title: >-
  [Paper Note] Identifying Reliable Evaluation Metrics for Scientific Text Revision
description: >-
  [ACL 2025][LLM (Other)][text revision] This study systematically analyzes the limitations of traditional similarity metrics (such as ROUGE and BERTScore) in evaluating scientific text revisions, revealing that they strongly correlate with edit distance and penalize deep modifications. To address this, a hybrid evaluation method combining LLM-as-Judge with task-specific, cross-domain metrics is proposed, which significantly outperforms any single metric in aligning with human…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "text revision"
  - "evaluation metrics"
  - "LLM-as-Judge"
  - "scientific writing"
  - "human evaluation"
date: 2026-05-08
content_hash: 7de3c636a1657eaf
---

# Identifying Reliable Evaluation Metrics for Scientific Text Revision

**Conference**: ACL 2025  
**arXiv**: [2506.04772](https://arxiv.org/abs/2506.04772)  
**Area**: Other  
**Keywords**: text revision, evaluation metrics, LLM-as-Judge, scientific writing, human evaluation  

## TL;DR

This study systematically analyzes the limitations of traditional similarity metrics (such as ROUGE and BERTScore) in evaluating scientific text revisions, revealing that they strongly correlate with edit distance and penalize deep modifications. To address this, a hybrid evaluation method combining LLM-as-Judge with task-specific, cross-domain metrics is proposed, which significantly outperforms any single metric in aligning with human judgment.

## Background & Motivation

**Task Definition**: Scientific Text Revision refers to the task of generating a revised version of a paragraph given the original paragraph and a revision instruction. Revisions involve multi-dimensional improvements such as readability, style, and clarity, which make it a crucial step in the academic writing workflow.

**Limitations of Prior Work**: Current automatic evaluation metrics cannot reliably measure revision quality. Mainstream metrics like ROUGE and BERTScore inherently measure the surface similarity between the generated text and the gold-standard reference, rather than whether the revision actually improves the original text. Experiments demonstrate that "making no edits" yields the highest scores under most metrics—a paradox that exposes the fundamental flaws of the traditional evaluation paradigm.

**Limitations of Existing Approaches**:  
1. **Prohibitively high cost of human evaluation**: 10 annotators (3 professors + 7 PhD students) spent a significant amount of time to complete 1,548 pairwise annotations, which is unscalable for large-scale iterative evaluations.  
2. **Incomplete coverage of single metrics**: text revision encompasses various subtasks such as paraphrasing, simplification, grammatical error correction, and content reduction, requiring different evaluation dimensions for different revision types.  
3. **Contradictory findings in existing LLM-as-Judge explorations**: Doostmohammadi et al. (2024) reported that the alignment of GPT-4o drops without gold references, whereas Mita et al. (2024) found that LLM judgments are even worse than fine-tuned BERT classifiers.

**Key Insight**: This study is the first to systematically compare traditional metrics, cross-domain metrics, and LLM-as-Judge methods in the scientific text revision task, and introduces the ParaReval human-annotated dataset, revealing the applicability of different evaluation methods across various revision types and difficulty levels.

## Method

### Overall Architecture

The study constructs a four-stage evaluation and analysis pipeline:
1. **Revision Generation**: Utilizing 6 models (CoEdIT-XL, Llama-3-8B/70B, Mistral-7B, GPT-4o-mini, GPT-4o) to generate revisions on the ParaRev dataset (258 paragraph pairs $\times$ 2 instructions = 516 data points).
2. **Human Annotation**: 10 annotators conduct pairwise comparisons on 1,548 revision pairs, evaluating relevance, correctness, and preference.
3. **Traditional Metric Analysis**: Computing the cross-correlation matrix of BLEU, ROUGE-L, METEOR, GLEU, SARI, and BERTScore, and analyzing their relationship with edit distance.
4. **Exploration of Alternative Methods**: Testing cross-domain metrics (BETS, BLANC, ParaPLUIE) and LLM-as-Judge (Choice/Likert paradigms $\times$ with/without gold references).

### Key Designs

**1. Multi-level Human Annotation Scheme**: A progressive annotation design is introduced, spanning from instruction following to subjective preferences. Q1A/Q1B evaluate relevance (whether the model follows the revision instruction), Q2 evaluates correctness (whether the revision is acceptable), and Q3 evaluates preference (which version is preferred for inclusion in a paper). Category-specific questions are also tailored to the revision types: academic style improvement for light revision, readability and structure for medium revision, readability and clarity for heavy revision, and compression capability under core-information preservation for concision. Additionally, the concept of "extended preference" is introduced—even if Q3 is marked "None", one version is still preferred if it is the only one deemed Correct or Related.

**2. Cross-domain Metric Migration Strategy**: Based on the assumption that "the core of revision evaluation lies in comparing with the original text rather than the reference text," three metrics taking both the original text and the generated text as input are selected from related NLP tasks: BETS (text simplification, evaluating the balance between semantic preservation and simplification using BERT embeddings-based word pair comparison), BLANC (document summarization, measuring how much the summary helps in understanding the original text via BERT), and ParaPLUIE (paraphrase detection, utilizing Mistral-7B perplexity scores to determine semantic equivalence).

**3. Dual-Paradigm LLM-as-Judge Design**: (1) LLM-Choice: pairwise comparison + Yes/No questions, where the model selects the superior revision or outputs a tie; (2) LLM-Likert: independent scoring, rating a single revision on Relatedness and Correctness. Both paradigms are tested with and without gold references, leveraging multiple open-source and closed-source LLMs as judges to reduce self-preference bias.

### Revision Type Classification System

| Revision Type | Description | Evaluation Focus |
|---|---|---|
| Light Revision (Light) | Fine-tuning wording | Academic style and English improvement |
| Medium Revision (Medium) | Complete sentence phrasing | Readability and structural improvement |
| Heavy Revision (Heavy) | Major edits affecting $\ge 50\%$ of the paragraph | Readability and clarity improvement |
| Concision | Removing unnecessary details | Compression capability while retaining core information |
| Deletion | Deleting a specific point | Reasonableness of content modifications |

## Key Experimental Results

### Failure Evidence of Traditional Metrics

**Ranking of Revision Models under Traditional Metrics** (ParaRev Dataset, 516 data points):

| Revision Model | BLEU | ROUGE-L | METEOR | GLEU | SARI | BERTScore |
|---|---|---|---|---|---|---|
| no edits (no modification) | **66.00** | **78.30** | **83.80** | 25.78 | 60.63 | **95.95** |
| CoEdIT-XL | 50.24 | 67.46 | 66.66 | 23.84 | 39.60 | 93.90 |
| Llama-3-70B | 46.78 | 65.61 | 67.20 | 30.31 | 42.74 | 93.90 |
| GPT-4o-mini | 51.68 | 69.54 | 72.70 | **32.67** | **45.06** | 94.80 |
| GPT-4o | 49.34 | 68.20 | 69.88 | 31.35 | 43.54 | 94.45 |

**Core Finding**: Except for GLEU, all traditional metrics rate the "no edits" strategy as the best option. BLEU, ROUGE-L, METEOR, and BERTScore are highly redundant (exhibiting extremely high cross-correlation) and strongly correlate with edit distance—meaning more modifications lead to lower scores, effectively penalizing deep revisions.

### Disagreement: Human Evaluation vs. Automatic Metrics

| Evaluation Dimension | Human Judgment Results | Traditional Metric Results |
|---|---|---|
| Best Model | GPT-4o (58.33% preference rate) | no edits (no modification) |
| Second Best Model | Llama-3-70B (53.68%) | CoEdIT-XL (minimal edit) |
| Worst Model | CoEdIT-XL | GPT-4o / Llama-3-70B |

Inter-annotator agreement: Relatedness $\kappa=0.54$ (moderate), Correctness $\kappa=0.55$ (moderate), Preference $\kappa=0.33$ (fair).

### Alignment of Metrics with Human Judgment

| Evaluation Method | Pairwise Acc. | Cramér's V | Cohen's $\kappa$ |
|---|---|---|---|
| **LLM-Choice (Mean)** | **0.564** | **0.244** | **0.247** |
| **ParaPLUIE** | **0.551** | **0.241** | **0.218** |
| LLM-Likert (Mean) | 0.436 | 0.240 | 0.181 |
| GLEU | 0.504 | 0.193 | 0.138 |
| BETS | 0.492 | 0.152 | 0.127 |
| SARI | 0.465 | 0.183 | 0.071 |
| BERTScore | 0.445 | 0.161 | 0.034 |
| ROUGE-L | 0.414 | 0.179 | -0.013 |
| BLANC | 0.357 | 0.117 | -0.080 |
| Random | 0.334 | 0.027 | 0.003 |

**LLM-Choice achieves the highest overall alignment**, while ParaPLUIE serves as an excellent low-cost alternative (processing the dataset takes only 11 minutes vs. 82 minutes for Mistral-Choice).

### Performance by Difficulty Level

| Difficulty Level | Definition | Best Method | Best Acc. |
|---|---|---|---|
| Easy (530 pairs) | Only one side follows instructions | LLM-Choice | 0.821 |
| Medium (214 pairs) | Both follow instructions, but only one is correct | Traditional similarity metrics | Outperforms LLM |
| Hard (575 pairs) | Both are correct, with differing preferences | ParaPLUIE | Low alignment across all methods |

### Performance by Revision Type

- **Light/Medium Revision + Concision**: ParaPLUIE is a good low-cost alternative to LLM-Choice.
- **Heavy Revision**: BETS performs the best due to its balance between semantic preservation and simplification.
- **Deletion**: n-gram metrics like GLEU and SARI exhibit performance comparable to LLM-Choice.

### Impact of Gold References

Providing a gold reference has almost no impact on LLM-as-Judge: LLM-Choice accuracy slightly shifts from 0.564 to 0.563, and LLM-Likert changes from 0.436 to 0.457. This suggests that LLMs primarily rely on their internal reasoning rather than direct comparison with reference texts, contradicting the findings of Doostmohammadi et al. (2024).

## Highlights & Insights

1. **Revealing the Evaluation Paradox**: Substantiating the absurd metric-level conclusion "no edits > any edits" with empirical data, strongly demonstrating the fundamental flaws of traditional metrics.
2. **Systematic Comparison**: Conducting the first comprehensive, three-dimensional comparison (traditional metrics vs. cross-domain metrics vs. LLM-as-Judge) on the scientific text revision task, with fine-grained analysis across both revision types and difficulty levels.
3. **Practical Recommendations**: Proposing a cost-effective recommended combination of metrics—small LLMs for evaluating instruction following + ParaPLUIE for evaluating semantic preservation + SARI/GLEU for handling hard cases.
4. **Open-Source Contribution**: Releasing the ParaReval human-annotated dataset.

## Limitations & Future Work

1. The scale of the ParaRev dataset is limited (258 paragraph pairs), and the annotators are non-native English-speaking NLP researchers, which may introduce domain and linguistic biases.
2. The cost of LLM-as-Judge remains high (GPT-4o experiments were run only once), and only a single prompt was used without validating prompt robustness.
3. The study does not cover non-English scientific writing or content-addition revision operations.

## Rating

| Dimension | Score (1-10) | Description |
|---|---|---|
| Novelty | 6 | The method itself lacks novelty; the primary contribution lies in the systematic empirical analysis and the exposure of the evaluation paradox. |
| Value | 8 | Provides concrete metric selection guidelines and cost-benefit analysis for scientific writing assistant systems. |
| Experimental Thoroughness | 8 | 6 generation models, 9 metrics, 1,548 pairwise human annotations, and fine-grained analysis by difficulty and category. |
| Writing Quality | 7 | Well-structured with progressive analysis, though some conclusions are repetitive. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers](can_llms_identify_critical_limitations_within_scientific_research_a_systematic_e.md)
- [\[ACL 2025\] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)

</div>

<!-- RELATED:END -->
