---
title: >-
  [Paper Note] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-Lingual Generalization] This work systematically evaluates the zero-shot cross-lingual generalization capabilities of LLMs on three classical languages (Sanskrit, Ancient Greek, and Latin) across three NLU tasks: NER, machine translation, and question answering. It also contributes a dataset of 1,501 Sanskrit QA pairs and validates the effectiveness of RAG strategies, revealing that model scale is the decisive factor in cro…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Generalization"
  - "Classical Languages"
  - "Sanskrit"
  - "Zero-Shot"
  - "RAG"
  - "NER"
date: 2026-05-08
content_hash: 16f4f503d02b989b
---

# A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs

**Conference**: ACL 2025  
**arXiv**: [2505.13173](https://arxiv.org/abs/2505.13173)  
**Code**: [GitHub](https://github.com/mahesh-ak/SktQA)  
**Institution**: University of Tübingen, University of Lyon 1, Indian Institute of Technology Kanpur  
**Area**: NLP / Multilingual & Machine Translation / Cross-Lingual Generalization  
**Keywords**: Cross-Lingual Generalization, Classical Languages, Sanskrit, Zero-Shot, RAG, NER

## TL;DR

This work systematically evaluates the zero-shot cross-lingual generalization capabilities of LLMs on three classical languages (Sanskrit, Ancient Greek, and Latin) across three NLU tasks: NER, machine translation, and question answering. It also contributes a dataset of 1,501 Sanskrit QA pairs and validates the effectiveness of RAG strategies, revealing that model scale is the decisive factor in cross-lingual generalization.

## Background & Motivation

While LLMs have demonstrated strong cross-lingual generalization capabilities, research on their zero-shot NLU capabilities for classical languages remains extremely limited. Previously, only Volk et al. (2024) explored Latin translation and summarization, lacking systematic multi-task evaluations.

- **Unique Status of Classical Languages**: Although digital downstream task data for Sanskrit, Ancient Greek, and Latin is extremely scarce, these languages possess rich historical literature and have profoundly influenced high-resource languages (e.g., Latin contributes ~28% of the English vocabulary). Furthermore, their highly inflectional morphology poses unique challenges for NLP processing.
- **Severe Deficit in Existing Data Resources**: The Sanskrit NER dataset contains only 139 sentences (1,558 tokens), and Latin has 3,410 sentences. Sanskrit QA datasets are even scarcer, with only 80 kinship-related questions (Terdalkar & Bhattacharya, 2019), failing to support effective evaluation studies.
- **Lack of Verification on Cross-Lingual Transfer Mechanisms**: Although prior work (Cahyawijaya et al., 2024; Han et al., 2024) demonstrated LLM generalization capabilities on low-resource languages, whether classical languages benefit from cross-lingual transfer remains systematically unexamined due to their morphological complexity and limited pre-training data coverage.
- **Core Problem**: Can LLMs comprehend classical languages through cross-lingual transfer (rather than specialized training)? What roles do model scale, prompt language, and writing scripts play?

## Method

### Overall Architecture

The research designs two sets of zero-shot experiments: (1) evaluating two NLU tasks, NER and machine translation (translation to English), across three classical languages using existing public datasets; (2) focusing on Sanskrit to contribute a new factual QA dataset and evaluate deep comprehension capabilities combined with BM25-retrieved RAG methods. The evaluation uses two pairs of large and small models: the closed-source GPT-4o / GPT-4o-mini, and the open-source LLaMA-3.1-405B-Instruct / LLaMA-3.1-8B-Instruct (whose knowledge cutoff is late 2023, while most of the test datasets were released afterward).

### Key Designs

1. **Sanskrit Factual QA Dataset Construction**: To fill the gap in Sanskrit QA resources, this work constructs 1,501 factual QA pairs covering two representative domains: the ancient epic *Rāmāyaṇa* and the classical Ayurvedic medical text *Bhāvaprakāśanighaṇṭu*. All QA pairs are manually annotated, with answers formatted for word-level exact matching, supporting both closed-book and RAG evaluation modes.

2. **BM25 + Lemmatization RAG Enhancement Scheme**: A BM25 retriever is employed to fetch the top-$k$ relevant passages (with $k=4$ being optimal). This is compared against embedding-based retrievers using FastText and GloVe, where BM25 consistently outperforms embedding-based methods across all metrics. To address the highly inflectional nature of Sanskrit, a Transformer-based Seq2Seq Sanskrit lemmatizer (trained on the DCS corpus, F1=0.94) is introduced to reduce inflected forms to lemmas, thereby improving the lexical recall of BM25.

3. **Multi-Dimensional Systematic Contrastive Experiments**: Four orthogonal analysis dimensions are designed: (a) model scale: GPT-4o / LLaMA-405B vs. GPT-4o-mini / LLaMA-8B; (b) prompt language: English prompts vs. target language prompts; (c) script: Devanagari vs. IAST romanization; (d) context validity: retrieved contexts with answers vs. contexts without answers, providing a systematic analysis of how each factor affects cross-lingual generalization.

## Key Experimental Results

### Dataset Overview

| Task | Language | Data Source | Test Set Size |
|------|------|---------|-----------|
| NER | Sanskrit | Terdalkar (2023) | 139 sentences |
| NER | Latin | Erdmann et al. (2019) | 3,410 sentences |
| NER | Ancient Greek | Myerston (2025) | 4,957 sentences |
| MT→en | Sanskrit | Maheshwari et al. (2024) | 6,464 sentences |
| MT→en | Latin | Rosenthal (2023) | 1,014 sentences |
| MT→en | Ancient Greek | Palladino et al. (2023) | 274 sentences |
| QA | Sanskrit | **Ours** | 1,501 pairs |

### Core Zero-Shot Performance Comparison

| Task | Metric | GPT-4o | LLaMA-405B | GPT-4o-mini | LLaMA-8B |
|------|------|--------|-----------|-------------|----------|
| NER (Sanskrit) | Macro F1 | 0.637 | 0.561 | 0.359 | 0.164 |
| MT (Sanskrit) | BLEU | 0.179 | 0.193 | 0.135 | 0.120 |
| QA Closed-book (Sanskrit) | EM | 0.36 | 0.41 | 0.18 | 0.13 |
| QA + RAG (Sanskrit) | EM | **0.46** | 0.42 | 0.25 | 0.09 |

### Impact of Inflection on QA (English Prompt)

| Model | Closed-book (Inflected) | Closed-book (Lemma) | +RAG (Inflected) | +RAG (Lemma) |
|------|----------|----------|----------|----------|
| GPT-4o | 0.36 | 0.37 | 0.46 | 0.48 |
| LLaMA-405B | 0.41 | 0.40 | 0.42 | 0.44 |
| GPT-4o-mini | 0.18 | 0.20 | 0.25 | 0.28 |
| LLaMA-8B | 0.13 | 0.15 | 0.09 | 0.10 |

### Script Comparison (Sanskrit, English Prompt)

| Model | MT-BLEU (Devanagari) | MT-BLEU (IAST) | NER-F1 (Devanagari) | NER-F1 (IAST) |
|------|-----------------|----------------|----------------|---------------|
| GPT-4o | 0.179 | 0.165 | 0.637 | 0.599 |
| LLaMA-405B | 0.193 | 0.148 | 0.561 | 0.556 |
| GPT-4o-mini | 0.135 | 0.099 | 0.359 | 0.318 |
| LLaMA-8B | 0.120 | 0.063 | 0.164 | 0.149 |

### Multi-Dimensional Analysis Summary

| Analysis Dimension | Key Findings |
|---------|---------|
| Model Scale | Large models consistently outperform small ones across all tasks and languages; smaller models degrade significantly, particularly on specific entity types and RAG utilization. |
| Prompt Language | English prompts generally outperform native language prompts (especially for smaller models), indirectly proving that these models have not undergone instruction-tuning in classical languages. |
| Script | Devanagari behaves slightly better than IAST romanization, but the gap is small—both forms have certain coverage in the pre-training data. |
| NER Error Analysis | Small models suffer from severe confusion among semantically close entities (e.g., Deva/Asura/Rakshasa, Kingdom/City/Forest), whereas large models exhibit clear decision boundaries. |
| RAG Context | Having contexts containing the answer yields an EM of 0.46, which is far superior to contexts without the answer, demonstrating the capacity of large models to comprehend Sanskrit passages. |
| Inflection | Lemmatization only yields a marginal EM improvement (+1–3%), indicating that inflectional errors are not the primary bottleneck. |

## Highlights & Limitations

### Highlights & Insights

- The first systematic study to evaluate the zero-shot NLU capabilities of LLMs across multiple tasks (NER/MT/QA) in three classical languages.
- Contributes a factual QA dataset of 1,501 pairs covering Sanskrit epics and medical texts, filling a resource gap.
- A multi-dimensional orthogonal contrastive design (Model Scale $\times$ Prompt Language $\times$ Script $\times$ Retrieval Method) provides a comprehensive analysis.
- Deep NER confusion matrix analysis reveals the differences in behavioral performance among models of different scales regarding classical language-specific entities.
- Validates the effectiveness of BM25 + Lemmatization for RAG in highly inflectional languages, providing a practical solution for classical language NLP.

### Limitations & Future Work

- The size of the Sanskrit dataset is limited (1,501 QA pairs, and only 139 sentences for NER), which may affect statistical confidence.
- Since some datasets were released before the knowledge cutoff dates of the models, data contamination risks exist (e.g., abnormally high performance on Ancient Greek MT).
- Only BM25 is explored as the primary retrieval method, leaving cross-lingual dense retrievers (such as mDPR) unexplored.
- Only 4 models are evaluated (2 closed-source and 2 open-source), leaving other model families like Gemini, Claude, and Mistral unassessed.
- Exploration into KG-QA was unable to yield robust conclusions due to incomplete knowledge graphs; this direction warrants further improvement.

## Related Work & Insights

- **Cross-Lingual Generalization**: Cahyawijaya et al. (2024) demonstrated the few-shot in-context learning capabilities of LLMs in low-resource languages; this work extends the evaluation to the zero-shot setting in classical languages.
- **Classical Language NLP**: Riemenschneider & Frank (2023) and Nehrdich et al. (2024) built dedicated models for classical languages, but those are limited to tasks like morphological parsing. This work demonstrates that cross-lingual transfer in general-purpose LLMs can serve as an alternative solution.
- **RAG Paradigm**: Lewis et al. (2020) proposed RAG. This work is the first to apply it to Sanskrit QA and validate the necessity of a lemmatization step.
- **Scripts and Transfer**: Muller et al. (2021) and Fujinuma et al. (2022) found that orthography is a key factor in cross-lingual transfer; this study further validates this on Sanskrit through Devanagari vs. IAST comparison.
- **Insights**: Although classical languages suffer from scarce downstream data, large LLMs may have acquired a degree of comprehension through indirect pathways (shared roots, syntactic structures) due to their profound historical impact on high-resource languages.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Unique topic choice; zero-shot generalization in classical languages is a valuable and under-explored research direction.
- **Practicality**: ⭐⭐⭐ — The application scenarios are relatively niche (digital humanities, linguistic research), but the QA dataset and the RAG scheme hold practical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Offers detailed analysis through multi-task, multi-lingual, multi-model, and multi-dimensional orthogonal comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Follows an appropriate case-study style, with a clear structure and deep confusion matrix analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2025\] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)

</div>

<!-- RELATED:END -->
