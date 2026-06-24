---
title: >-
  [Paper Note] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian
description: >-
  [ACL 2025 (Long Paper, acl-long.25)][LLM Evaluation][Belarusian] This paper introduces BelarusianGLUE, the first NLU benchmark for the Belarusian language (East Slavic branch), containing approximately 15K instances across 5 tasks. It systematically evaluates the performance of BERT models and LLMs, finding that while simple tasks like sentiment analysis approach human-level performance, difficult tasks like the Winograd Schema Challenge still exhibit a significant gap…
tags:
  - "ACL 2025 (Long Paper, acl-long.25)"
  - "LLM Evaluation"
  - "Belarusian"
  - "NLU Benchmark"
  - "Low-Resource Language"
  - "GLUE"
  - "Multilingual Evaluation"
date: 2026-05-08
content_hash: bcafa05378d31c9f
---

# BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian

**Conference**: ACL 2025 (Long Paper, acl-long.25)  
**Code**: [https://github.com/maaxap/BelarusianGLUE](https://github.com/maaxap/BelarusianGLUE)  
**Data**: https://hf.co/datasets/maaxap/BelarusianGLUE
**Area**: NLU Benchmark / Low-Resource Languages / Multilingual Evaluation  
**Keywords**: Belarusian, NLU Benchmark, Low-Resource Language, GLUE, Multilingual Evaluation  

## TL;DR
This paper introduces BelarusianGLUE, the first NLU benchmark for the Belarusian language (East Slavic branch), containing approximately 15K instances across 5 tasks. It systematically evaluates the performance of BERT models and LLMs, finding that while simple tasks like sentiment analysis approach human-level performance, difficult tasks like the Winograd Schema Challenge still exhibit a significant gap, and the optimal model type varies by task.

## Background & Motivation
In the era of multilingual large language models, evaluating model comprehension capabilities for low-resource languages remains a challenge. Although Belarusian is an East Slavic language (related to Russian and Ukrainian), it has long suffered from a lack of dedicated NLU evaluation resources. Existing multilingual benchmarks (such as XGLUE and XTREME) offer extremely limited coverage of Belarusian, while language-specific phenomena (such as the distinction between the reflexive pronoun свой vs. possessive pronouns яго/ягоны, and orthographical variants like narkamaŭka vs. taraškievica) imply that simply translating English benchmarks is insufficient. Therefore, an expert-curated NLU benchmark tailored to the characteristics of the Belarusian language is highly needed.

## Core Problem
How can a high-quality, multi-task NLU benchmark be constructed for a low-resource language like Belarusian? What is the current performance level of existing BERT models and LLMs in comprehending Belarusian? How large is the gap with human performance? Which model architectures are better suited for which tasks?

## Method

### Overall Architecture
BelarusianGLUE comprises 5 binary classification NLU tasks totaling approximately 15K instances. All data were annotated or reviewed by native Belarusian speakers with backgrounds in linguistics:

| Task | Abbreviation | Instances | Train/Val/Test | Data Source |
|------|------|--------|---------------|---------|
| Sentiment Analysis | BeSLS | 2000 | 1500/250/250 | Reviews across 5 domains (movies, books, travel/hotels, shopping, social media) |
| Linguistic Acceptability | BelaCoLA | 3592 | 1992/300/300 (in-domain) + 500/500 (out-of-domain) | RuCoLA translation, prescriptive grammar sources, CommonVoice, LM hallucinations, machine translation |
| Word-in-Context Word Sense Disambiguation | BeWiC | ~5000+ | Large and unlabelled/400/400 | Explanatory Dictionary of the Belarusian Language (1977-1984) |
| Winograd Schema Challenge | BeWSC | 970 | 570/200/200 | WSC-285 translation + original samples |
| Textual Entailment | BeRTE-WD | 1800 | 1080/360/360 | Wikidata knowledge base |

Evaluation focuses on three main subjects:
1. **Human baseline** — Native Belarusian speakers annotating via a custom Streamlit UI.
2. **BERT series** — Fine-tuning of mBERT, XLM-RoBERTa, mDeBERTa-v3, and HPLT BERT (be).
3. **LLMs** — Zero-shot evaluation using the lm-evaluation-harness + Gemma 2 9B fine-tuning.

### Key Designs

1. **BeSLS (Sentiment Analysis)**: Evenly sampled from 5 domains (movies, books, travel/hotels, shopping, social media), with balanced positive/negative classes in each domain. Sources cover professional film reviews (from newspapers like Zviazda and Kultura), Telegram channels, LiveLib book reviews, Booking/TripAdvisor travel reviews, Onliner product reviews, and Mastodon social media posts. Lingua was used to filter out non-Belarusian sentences, and usernames were anonymized. The data reflects the real-world distribution of Belarusian written variants (official modern orthography narkamaŭka, classical orthography taraškievica, and Latin-script łacinka).

2. **BelaCoLA (Linguistic Acceptability)**: Designed with reference to CoLA, RuCoLA, and BLiMP, but featuring a broader range of unacceptable sentences. These include not only morphological, syntactic, and semantic deviations, but also pragmatic anomalies, prescriptive rule violations, language model hallucinations, and machine translation errors. The out-of-domain test set specifically contains trigram model outputs, GPT-2 (117M) generations, and machine translation outputs from NLLB, Google Translate, and Belazar, which are increasingly common in real-world Belarusian texts.

3. **BeWiC (Word-in-Context Word Sense Disambiguation)**: Constructed using example sentences from the 5-volume explanatory dictionary of Belarusian to build context pairs. Unlike the original WiC, the distinction between positive and negative instances is based on stronger homonymy criteria rather than polysemy, making the task easier for humans but better suited for dictionary-based construction. Phrase-level examples were expanded into complete sentences, and multi-sentence examples were shortened to single sentences.

4. **BeWSC (Winograd Schema Challenge)**: Provided in both WSC and WNLI formats. The training set is primarily translated from the English WSC-285, but heavily adapted due to grammatical differences in Belarusian (grammaticalized gender, reflexive pronoun свой vs. possessive pronouns). The test set of 200 items is uniquely designed based on original Belarusian literature, intentionally structured to be difficult to solve using selectional restrictions alone.

5. **BeRTE-WD (Textual Entailment)**: Innovatively constructed using the Wikidata knowledge base. Entity-attribute-value triples with Belarusian labels (200 triples each for timestamps, numerical values, and entities) were extracted. Three experts converted the triples into natural language text and drafted entailment/non-entailment hypotheses. The entailment types are extremely diverse, including temporal comparison, numerical reasoning, constraint satisfaction, unit conversion, domain knowledge, world knowledge, monotonicity, logical deduction, and paraphrasing.

### Evaluation Strategy
- **BERT Evaluation**: Standard fine-tuning + cross-lingual transfer learning (utilizing similar datasets in other languages such as MELA, XLWiC, RUSSE, and WinoGrande for pre-training) + layer-freezing experiments.
- **LLM Evaluation**: Zero-shot log-probability evaluation using lm-evaluation-harness (local models), generative evaluation (commercial API models), and Gemma 2 9B fine-tuning using Belarusian and English prompts.
- **Human Baseline**: Native speaker judgments collected via a custom Streamlit interface.

## Key Experimental Results

Key findings based on the paper (specific values sourced from the paper's tables):

| Task | Metric | Best BERT | Best LLM | Human |
|------|------|---------|---------|-------|
| BeSLS (Sentiment Analysis) | Accuracy | Near Human | Near Human | ~High Level |
| BelaCoLA (Acceptability) | MCC/Accuracy | BERT Competitive | Weaker than BERT | ~High Level |
| BeWiC (Sense Disambiguation) | Accuracy | Moderate | Moderate | High Level |
| BeWSC (Winograd) | Accuracy | Significantly below human | Significantly below human | High Level |
| BeRTE-WD (Entailment) | Accuracy | BERT Weak | LLM Better | High Level |

**Evaluated BERT Models**: mBERT, XLM-RoBERTa-base, mDeBERTa-v3-base, HPLT BERT Belarusian.

**Key Findings**:
- **Sentiment analysis** is the simplest task, with both BERT and LLMs achieving near-human performance.
- **The Winograd Schema Challenge** shows the largest gap, with a significant disparity between machine and human performance.
- **Model preference varies by task**: BERT is highly competitive on linguistic acceptability but struggles with textual entailment; conversely, LLMs perform better on entailment tasks that require world knowledge.
- **Cross-lingual transfer learning** (leveraging datasets of similar tasks from other languages) can improve the performance of BERT models on BelarusianGLUE.

### Ablation Study Highlights
- **Layer-Freezing Experiments**: Differences between freezing all 12 encoder layers of mDeBERTa-v3 (training only the classification head) and performing full fine-tuning were explored to test the quality of pre-trained representations.
- **Cross-Lingual Transfer**: Evaluates pre-training on similar datasets from other languages (e.g., English WiC/XLWiC and Russian RUSSE for BeWiC; multilingual CoLA like MELA, Dutch CoLA, and HuCoLA for BelaCoLA; English WinoGrande for BeWSC) followed by fine-tuning on Belarusian data.
- **Prompt Language**: Fine-tuning of Gemma 2 9B compared the efficacy of Belarusian prompts versus English prompts.

## Highlights & Insights
- **High-Quality Expert Annotation**: All data were annotated or reviewed by native Belarusian speakers with Master's or PhD degrees in linguistics, rather than relying on crowdsourcing or machine translation, thereby ensuring high benchmark quality.
- **Ingenious Design of BeRTE-WD**: By utilizing Wikidata structured knowledge to construct the entailment task, a wide variety of entailment types (temporal reasoning, numerical reasoning, world knowledge, etc.) are covered, creating a challenging task that demands multiple reasoning capacities.
- **Out-of-Domain Test Set of BelaCoLA**: By using LM hallucinations and machine translation errors as out-of-domain unacceptable sentences, the task directly addresses real-world issues of low-resource languages, such as poor translation quality and language model output errors.
- **Linguistic Adaptation of BeWSC**: Instead of direct translation from English Winograd schemas, the task is deeply adapted to Belarusian grammatical features (grammaticalized gender, reflexive pronoun system), and the test set is originally created from Belarusian literature.
- **Open-Sourced Code and Data**: Provides reproducible BERT fine-tuning and LLM evaluation pipelines.

## Limitations & Future Work
- **Small Dataset Size**: A total of 15K instances is relatively small for an NLU benchmark; in particular, BeSLS contains only 2,000 sentences and BeWSC has just 970 samples.
- **Limited Task Types**: Contains only 5 tasks, all of which are binary classification. It lacks complex task types like question answering, reading comprehension, or generative tasks.
- **Insufficient Training Data**: The training set for BeWSC is mostly translated rather than originally authored, potentially introducing translation bias. In the BeWiC training set, sentence and target word repetitions are allowed.
- **Legal Risks**: Since the Belarusian authorities have designated numerous information sources as "extremist materials", the authors cannot guarantee the legal safety of using this dataset inside Belarus.
- **Lack of Broader LLM Comparisons**: State-of-the-art large models (such as GPT-4, Claude, etc.) available at the time of publication might not have been fully evaluated.
- **Simplistic Evaluation Metrics**: Most tasks rely solely on Accuracy; although BelaCoLA employs MCC and F1, finer-grained analyses are lacking.

## Related Work & Insights
- **vs. SuperGLUE/GLUE**: The task design of BelarusianGLUE directly corresponds to GLUE/SuperGLUE (sentiment analysis $\leftrightarrow$ SST, acceptability $\leftrightarrow$ CoLA, word sense disambiguation $\leftrightarrow$ WiC, Winograd $\leftrightarrow$ WSC, entailment $\leftrightarrow$ RTE), but incorporates deep localization for Belarusian linguistic features.
- **vs. RussianSuperGLUE**: Compared to the Russian benchmark in the same East Slavic branch, BelarusianGLUE is smaller in scale but features higher data quality (fully expert-annotated). Additionally, the Wikidata-based construction method of BeRTE-WD is more challenging than the CommonGen-based SCR task.
- **vs. Other Low-Resource Benchmarks** (such as KorNLI, TurkishGLUE, etc.): BelarusianGLUE is distinguished by the knowledge-base-driven construction of BeRTE-WD and the coverage of LM hallucinations/machine translation errors in BelaCoLA, which closer align with the actual challenges faced by low-resource languages.

## Related Work & Insights
- **Methodology for Low-Resource Benchmark Construction**: The Wikidata-driven approach used in BeRTE-WD to build entailment tasks can be transferred to other low-resource languages with Wikidata labels.
- **Cross-Lingual Transfer Strategies**: The paper demonstrates how to leverage datasets of similar tasks from multiple languages to provide richer training signals for target language BERT fine-tuning, representing a practical strategy for low-resource scenarios.
- **LM Hallucination Detection**: The out-of-domain design of BelaCoLA—using LM-generated text to evaluate acceptability—can serve as an indirect method for evaluating language model quality.
- **Multilingual Model Evaluation**: Holds reference value for exploring the capabilities of multilingual LLMs across diverse languages.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The task design follows the classic GLUE paradigm. The core contribution lies in high-quality Belarusian localization, and the Wikidata-based construction method for BeRTE-WD offers notable novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ It covers three categories of evaluation subjects (BERT, LLMs, and Human) and includes cross-lingual transfer and layer-freezing ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The description of the data construction process is highly detailed, clearly explaining the data sources, annotation workflows, and dataset splits for each task.
- **Value**: ⭐⭐⭐ Primarily serves the Belarusian NLP community, while the benchmark construction methodology and cross-lingual transfer policies offer general reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] skLEP: A Slovak General Language Understanding Benchmark](sklep_a_slovak_general_language_understanding_benchmark.md)
- [\[ACL 2025\] MisMatched: A Benchmark for Scientific Natural Language Inference](a_mismatched_benchmark_for_scientific_natural_language_inference.md)
- [\[ACL 2025\] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)

</div>

<!-- RELATED:END -->
