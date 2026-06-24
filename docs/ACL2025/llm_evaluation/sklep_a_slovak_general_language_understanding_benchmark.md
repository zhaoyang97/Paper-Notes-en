---
title: >-
  [Paper Note] skLEP: A Slovak General Language Understanding Benchmark
description: >-
  [ACL 2025 (Findings)][LLM Evaluation][Slovak] This paper introduces skLEP, the first comprehensive natural language understanding benchmark for Slovak, consisting of 9 multi-level tasks (word-level, sentence-pair-level, and document-level). Systematic evaluation of Slovak-specific, multilingual, and English models reveals that mDeBERTaV3 outperforms all Slovak-specific models in average performance.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Evaluation"
  - "Slovak"
  - "NLU benchmark"
  - "language understanding evaluation"
  - "multilingual models"
  - "low-resource languages"
date: 2026-05-08
content_hash: 251f2ac8e92316dc
---

# skLEP: A Slovak General Language Understanding Benchmark

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.21508](https://arxiv.org/abs/2506.21508)  
**Code**: [github.com/slovak-nlp/sklep](https://github.com/slovak-nlp/sklep)  
**Area**: NLP Understanding / Benchmark Evaluation  
**Keywords**: Slovak, NLU benchmark, language understanding evaluation, multilingual models, low-resource languages

## TL;DR

This paper introduces skLEP, the first comprehensive natural language understanding benchmark for Slovak, consisting of 9 multi-level tasks (word-level, sentence-pair-level, and document-level). Systematic evaluation of Slovak-specific, multilingual, and English models reveals that mDeBERTaV3 outperforms all Slovak-specific models in average performance.

## Background & Motivation

**Background**: GLUE and SuperGLUE benchmarks have rapidly advanced English NLU models, prompting various languages to establish equivalent benchmarks, such as Polish KLEJ, Russian SuperGLUE, and Slovene SuperGLUE. However, Slovak, as a medium-resource language (approximately 10 million native speakers), has lacked a standardized NLU evaluation benchmark.

**Limitations of Prior Work**: Although several Slovak-specific models (e.g., SlovakBERT, FERNET-CC, HPLT BERT) have been released in recent years, comparing them with each other or with multilingual models has been impossible to conduct systematically due to the absence of a unified evaluation standard. Existing scattered evaluations cover limited tasks (e.g., the SlovakBERT paper evaluated only a few tasks like POS tagging and semantic similarity), and some datasets suffer from poor translation quality.

**Key Challenge**: Slovak-specific models continue to emerge, but they lack a "racetrack" for fair comparison. Meanwhile, certain tasks (such as textual entailment and natural language inference) lack native Slovak datasets altogether, requiring construction from scratch or translation from English.

**Goal**: (1) Build skLEP, a comprehensive NLU benchmark covering 9 tasks; (2) systematically evaluate 14 models fairly; and (3) open-source the evaluation toolkit and public leaderboard.

**Key Insight**: Integrate existing Slovak datasets and translate English datasets to fill missing task slots. Translation quality is safeguarded via specialized evaluation experiments (comparing 5 translation systems and performing post-editing by native speakers).

**Core Idea**: Fill the void in Slovak NLU evaluation by establishing a standardized GLUE-like benchmark, and introduce the Relative Error Reduction (RER) as an aggregate metric to compare model differences across tasks more fairly.

## Method

### Overall Architecture

skLEP consists of 9 tasks categorized into three levels: word-level tasks (POS tagging, two NER datasets), sentence-pair tasks (textual entailment, natural language inference, semantic similarity), and document-level tasks (hate speech detection, sentiment analysis, QA). For tasks lacking native Slovak versions, DeepL translation (assisted by native speaker post-editing) was employed, while the large-scale NLI dataset was translated using MADLAD-400-3B for cost control. Fourteen pre-trained models were fine-tuned and evaluated across all tasks.

### Key Designs

1. **Multi-level Task Coverage and Dataset Construction**:

    - **Function**: Provide comprehensive and diverse NLU evaluation.
    - **Mechanism**: The nine tasks are evenly distributed across three granularity levels. Word-level includes UD (POS tagging based on the Slovak Dependency Treebank), UNER (named entity recognition using the Slovak subset of Universal NER, annotated for PER/ORG/LOC), and WikiGoldSK (another NER dataset from Slovak Wikipedia, additionally including MISC). Sentence-pair-level includes RTE (textual entailment, translated from English and reannotated), NLI (three-class classification based on XNLI), and STS (semantic similarity, 0-5 continuous scale). Document-level includes HS (hate speech, binary classification), SA (sentiment analysis, positive/negative binary classification), and QA (question answering, based on SK-QuAD).
    - **Design Motivation**: A single task cannot comprehensively evaluate model capabilities. The three-level schema covers the full spectrum of language understanding, from fine-grained word structures to paragraph-level semantic reasoning, and each level contains three tasks to enhance the statistical reliability of the evaluation.

2. **Translation Quality Assurance Pipeline**:

    - **Function**: Ensure the reliability of translated datasets.
    - **Mechanism**: A systematic comparison was conducted among five translation systems (DeepL, GPT-4o, Google Translate, MADLAD-400-3B, and NLLB-3.3B), evaluated by 4 native speakers on ranking, fluency, and adequacy. DeepL performed best overall (fluency 3.70/4, adequacy 3.73/4) and was chosen as the primary translator. For cost reasons, MADLAD-400-3B was used for the large-scale NLI corpus. All translated datasets underwent native speaker post-editing; verification experiments showed that post-editing improved translation quality in 28 out of 30 samples.
    - **Design Motivation**: The greatest risk in translation-based benchmarks is that poor translation quality leads to unreliable evaluation conclusions. The dual protection of multi-system comparison and human post-editing minimizes this risk.

3. **Relative Error Reduction (RER) Aggregate Metric**:

    - **Function**: Enable a fairer comparison of model performance across heterogeneous tasks.
    - **Mechanism**: Absolute score ranges vary widely across tasks (e.g., UD F1 is typically >95, whereas QA F1 is around 75). Direct averaging of absolute scores would disproportionately favor high-scoring tasks. Taking SlovakBERT as the baseline, RER calculates the percentage of error reduction achieved by each model relative to the baseline and averages these percentages. Thus, a 1-point improvement on UD (approx. 20% error reduction) is weighted more than a 1-point improvement on QA (approx. 3% error reduction), accurately reflecting the true difficulty of improvement.
    - **Design Motivation**: Inspired by de Vries et al. (2023), RER rationalizes the aggregation of heterogeneous tasks, avoiding biases from simple arithmetic means.

### Loss & Training

All models were fine-tuned on each task using the AdamW optimizer, linear learning rate decay, and a batch size of 12. Extensive hyperparameter grid searching (learning rate, epochs, warmup ratio) was performed on A100 GPUs. A total of approximately 4,024 experimental runs were executed, consuming around 130 GPU-days.

## Key Experimental Results

### Main Results

| Model | Type | Avg Score | Avg RER | UD(F1) | RTE(Acc) | NLI(Acc) | QA(F1) |
|------|------|---------|---------|--------|----------|----------|--------|
| **mDeBERTaV3-Base** | Multilingual | **85.17** | **+6.43** | 98.02 | **70.94** | **84.41** | **75.89** |
| SlovakBERT | Slovak | 83.95 | 0.00 | 98.04 | 65.20 | 82.75 | 74.36 |
| HPLT BERT-sk | Slovak | 82.96 | -1.29 | **98.23** | 56.81 | 80.71 | 75.14 |
| FERNET-CC | Slovak | 84.27 | +0.55 | 97.87 | 68.23 | 81.83 | 73.85 |
| XLM-R-Large | Multilingual | 83.36 | -11.90 | 98.23 | 57.35 | **85.80** | **77.14** |
| DeBERTaV3-Large | English | 83.95 | -10.22 | 97.55 | **74.30** | 85.13 | 74.73 |
| ModernBERT-Base | English | 72.82 | -132.43 | 91.52 | 57.77 | 71.42 | — |

### Translation Quality Assessment

| Translation System | Avg Rank↓ | Fluency/4 | Adequacy/4 |
|----------|----------|---------|---------|
| **DeepL** | **1.81** | **3.70** | **3.73** |
| GPT-4o | 1.85 | 3.62 | 3.73 |
| Google Translate | 2.05 | 3.57 | 3.67 |
| MADLAD-400-3B | 2.54 | 3.48 | 3.53 |
| NLLB-3.3B | 2.68 | 3.40 | 3.54 |

### Key Findings

- **mDeBERTaV3 is the strongest model**: With 276M parameters, it achieves the best performance with an average score of 85.17 and RER of +6.43, demonstrating that the architectural advantages of multilingual DeBERTa (disentangled attention + replaced token detection) are highly effective for medium-resource languages.
- **Slovak-specific models remain competitive**: Although SlovakBERT is the earliest specialized model, it maintains strong performance on multiple tasks, particularly UD, HS, and SA. However, it lags significantly behind mDeBERTaV3 on reasoning-intensive tasks like RTE and NLI.
- **ModernBERT performs unexpectedly poorly**: Despite being the latest large-scale English model, its RER on Slovak is -132.43, far below the baseline. This indicates that models trained purely on English can be severely deficient in non-English languages, regardless of their advanced architectures.
- **UD and SA are close to being "solved"**: Most models achieve an F1/Accuracy >90 on these two tasks, leaving limited room for improvement. Conversely, RTE and QA still have substantial room for improvement (<75) and represent key directions for future research.
- **Translation quality is controllable**: Post-editing verification experiments show that translation-induced labeling errors are restricted to single-digit percentages (RTE ~2%, NLI ~5%), which does not undermine the validity of the benchmark.

## Highlights & Insights

- **Comprehensive Benchmark Ecosystem**: The project provides not only datasets but also an open-source evaluation toolkit, HuggingFace integration, and a public leaderboard, lowering the barrier to entry for subsequent researchers. This "benchmark-as-a-service" philosophy is highly exemplary for other low-resource language communities.
- **Introduction of the RER Metric**: Utilizing relative error reduction instead of simple averages for cross-task comparison uncovers differences masked by absolute scores (e.g., DeBERTaV3-Base and SlovakBERT have identical absolute average scores but very different RER values), providing a more faithful reflection of models' overall capabilities.
- **Systematic Verification of Translation Quality**: Rather than simply claiming that the data was translated, the authors ensured quality through multiple layers of validation—comparing five translation systems, utilizing native speaker post-editing, and calculating reannotation alignment—resulting in a highly rigorous methodology.

## Limitations & Future Work

- The evaluation only covers encoder-only Transformer models, omitting generative LLMs (e.g., GPT-4, Llama), whose performance on NLU tasks remains to be explored.
- Three of the nine tasks are translated. Although post-edited, they may still contain artifacts such as "translationese".
- Slovak-specific models are all base-scale (<150M parameters), making comparisons with large-scale multilingual models (500M+) somewhat unfair.
- A human baseline is missing to serve as an upper bound.
- Some tasks share high similarity (e.g., both NLI and RTE involve reasoning relationships); task diversity could be further expanded.
- Future iterations can incorporate more task types (e.g., coreference resolution, relation extraction) and non-text/multimodal data.

## Related Work & Insights

- **vs KLEJ (Polish)**: KLEJ comprises 8 tasks, while skLEP comprises 9, covering comparable scopes. Both face dataset scarcity in low/medium-resource languages, but skLEP establishes a more comprehensive translation-quality validation pipeline.
- **vs BgGLUE (Bulgarian)**: BgGLUE is also a Slavic language benchmark. However, resource conditions vary between Slovak and Bulgarian, and skLEP contributes more novel datasets.
- **vs XTREME/XGLUE (Multilingual)**: Multilingual benchmarks are broad but exclude Slovak, and they primarily provide English training data (focusing on cross-lingual transfer). skLEP supports Slovak-native training data, facilitating more targeted evaluations.

## Rating

- Novelty: ⭐⭐⭐ Underpinned by benchmark development; technical innovation is limited, but it fills an important vacancy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with 14 models evaluated across 9 tasks, 4,024 runs for hyperparameter tuning, and multi-tier translation quality validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly organized, standard data representation, and extremely detailed appendices.
- Value: ⭐⭐⭐⭐ Highly valuable to the Slovak NLP community, supported by a complete open-source ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)
- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
