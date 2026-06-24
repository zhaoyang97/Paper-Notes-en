---
title: >-
  [Paper Note] CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages
description: >-
  [ACL 2025][NLP Understanding][Multilingual QA] The first multilingual long-form question answering dataset, CaLMQA (51.7K questions, 23 languages), is constructed. Culturally specific questions are collected using a translation-free approach. The study reveals that the factuality of large language models (LLMs) on culturally specific questions (45-52%) is significantly lower than on culturally neutral questions (64-71%), with low-resource languages showing particularly poor p…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Multilingual QA"
  - "cultural specificity"
  - "long-form question answering"
  - "low-resource languages"
  - "evaluation benchmarks"
date: 2026-05-08
content_hash: 8f7a58fca14899dd
---

# CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages

**Conference**: ACL 2025  
**arXiv**: [2406.17761](https://arxiv.org/abs/2406.17761)  
**Code**: [https://github.com/2015aroras/CaLMQA](https://github.com/2015aroras/CaLMQA)  
**Area**: NLP Understanding  
**Keywords**: Multilingual QA, cultural specificity, long-form question answering, low-resource languages, evaluation benchmarks  

## TL;DR
The first multilingual long-form question answering dataset, CaLMQA (51.7K questions, 23 languages), is constructed. Culturally specific questions are collected using a translation-free approach. The study reveals that the factuality of large language models (LLMs) on culturally specific questions (45-52%) is significantly lower than on culturally neutral questions (64-71%), with low-resource languages showing particularly poor performance.

## Background & Motivation
**Background**: LLMs are used by global users, but research on multilingual QA is primarily based on datasets translated from English (such as translated versions of MMLU), and long-form QA focuses almost exclusively on English scenarios.

**Limitations of Prior Work**: Translated datasets fail to cover unique local cultural concepts (e.g., why the first king of Burundi was named "Lion"), which prevents evaluation results from reflecting the true capability of LLMs in processing cultural knowledge. Furthermore, there is a severe lack of evaluation data for low-resource languages (e.g., Afar, Fijian).

**Key Challenge**: Existing multilingual benchmarks conflate "language" with "culture"—testing with translated English questions measures cross-lingual capability rather than cultural knowledge understanding. Genuine culturally specific questions must be collected from native communities using native languages.

**Goal**: (1) How to collect culturally specific long-form QA data across multiple languages at scale? (2) Whether there are systematic differences in LLM performance between culturally specific and culturally neutral questions.

**Key Insight**: A "translation-free" strategy is adopted—scraping questions from native community forums for high-resource languages and hiring native annotators to write questions manually for low-resource languages, ensuring the questions truly represent local cultures.

**Core Idea**: By comparing culturally specific versus culturally neutral questions, this work decouples the impact of "linguistic capability" and "cultural knowledge" on LLM performance, revealing the true gaps of LLMs in multicultural knowledge.

## Method

### Overall Architecture
CaLMQA consists of three components: (1) 51,150 culturally specific questions in high- and medium-resource languages (11 languages, scraped from community forums), (2) 548 questions in low-resource languages (12 languages, written by native speakers), and (3) 51 culturally neutral questions translated into 22 languages as a baseline comparison. The evaluation framework comprises three dimensions: surface quality, factuality, and relevance.

### Key Designs

1. **Translation-Free Data Collection (High/Medium-Resource Languages)**:

    - **Function**: Collect culturally specific questions from local community QA sites (similar to Quora/Reddit).
    - **Mechanism**: Prolific crowdsourced workers were first asked to provide links to local community forums and write 923 seed questions ($1,427). Web crawlers then automatically gathered about 10K questions per language. Finally, GPT-4o-Mini performed two rounds of filtering (for cultural specificity and quality), retaining 52% of the questions.
    - **Design Motivation**: Translation loses culturally unique concepts. Direct collection from source-language communities yields genuine culturally specific questions.

2. **Manual Collection for Low-Resource Languages**:

    - **Function**: Cover 12 low-resource languages (Afar, Fijian, Kirundi, etc.).
    - **Mechanism**: Employed 29 bilingual native speakers via Upwork, where each participant first completed a $7 qualification test, and then wrote culturally specific questions with English translations at a rate of $0.65-1.00 per question.
    - **Design Motivation**: Low-resource languages lack online community forums and must rely on native speakers for manual writing. The selection of languages intentionally includes those rarely addressed in previous studies.

3. **Three-Dimensional Evaluation Framework**:

    - **Function**: Synthesively evaluate the quality of the generated long-form answers from LLMs.
    - **Mechanism**: (1) $S_{surf}$: Detects whether the answer uses the correct language and checks for repetitions (sequences of more than 20 tokens repeating more than 4 times); (2) $S_{fact}$: Translates the answer into English, extracts verifiable statements using VeriScore, and verifies them via Google Search; (3) $S_{rel}$: Employs GPT-4o as a judge to evaluate relevance. The overall score is calculated as $S = S_{surf} \times S_{fact} \times S_{rel}$.
    - **Design Motivation**: Traditional metrics like BLEU/ROUGE correlate poorly with human judgment on long-form QA, necessitating a multi-dimensional evaluation approach.

### Human Evaluation
5 languages (Kirundi, Fijian, Hindi, German, English) $\times$ 3 models (Claude-3-Opus, GPT-4-Turbo, Mixtral-8x22B), with 20 questions per language. Native speakers rated the answers on a 5-point scale, annotated errors, and ranked the generated outputs.

## Key Experimental Results

### Main Results
Overall scores $S$ of 7 LLMs on culturally neutral and culturally specific questions:

| Model | Culturally Neutral Overall | Culturally Specific Overall | Factuality (Neutral / Specific) |
|------|------------------|------------------|-------------------|
| GPT-4o | 56.9 | 49.2 | 69.6 / 52.2 |
| GPT-4-Turbo | 56.9 | 48.7 | 69.9 / 51.9 |
| Claude-3-Opus | 52.9 | 42.6 | 63.6 / 45.5 |
| Aya-Expanse-32B | 43.4 | 39.5 | 63.8 / 45.6 |
| Gemini-1.5-Pro | 40.9 | 46.6 | 71.1 / 48.7 |
| Mixtral-8x22B | 35.6 | 35.7 | 64.0 / 46.2 |
| Llama-3-70B | 15.3 | 13.5 | 66.6 / 46.7 |

### Human Evaluation Results

| Analysis Dimension | Findings |
|----------|------|
| Model Ranking | GPT-4-Turbo > Claude-3-Opus > Mixtral-8x22B |
| Culturally Neutral vs. Specific | The probability of culturally neutral questions receiving a high rating is twice that of specific ones (p<.001) |
| Rating Predictor | Omission (R²=0.740) > Factual Error (R²=0.560) > Model > Question Type |

### Key Findings
- **Systematically lower factuality on culturally specific questions**: The factual precision of all models on culturally specific questions is 15-20 percentage points lower than on culturally neutral questions, regardless of the specific model.
- **Open-source models collapse on low-resource languages**: Llama-3-70B responded in the wrong language for 76% of prompts in low-resource languages (e.g., answering a Fijian question in English), whereas GPT-4o failed in only 2.7% of cases.
- **Omission is the strongest predictor of answer quality**: In human evaluations, the omission of information is a stronger predictor of user rating than factual errors ($R^2=0.740$ vs $0.560$), showing that users place a higher value on answer completeness.
- No model is capable of reliably generating text in the Afar language.

## Highlights & Insights
- **Elegant experimental design decoupling language capability and cultural knowledge**: Through the comparison between culturally neutral (translated) and culturally specific (native) questions, the contribution of "language" versus "cultural knowledge" to the LLM performance gap can be quantitatively analyzed.
- **The dual-track design of data collection is highly exemplary**: Utilizing automatic scraping + LLM filtering for high-resource languages and manual writing + quality control for low-resource languages balances scale and quality. The construction cost of the entire dataset was only about ~$2,300.
- **Finding that omissions affect user experience more than factual errors**: This offers a crucial insight for LLM evaluation—existing evaluation paradigms that over-index on factuality may underestimate the significance of information completeness.

## Limitations & Future Work
- Factuality evaluation relies on translating responses into English for verification, where translation quality and the availability of English evidence may introduce bias.
- Language identifiers for low-resource languages are inaccurate, leading to 4 languages (such as Balochi) being excluded from statistical analysis.
- The scale of human evaluation is limited (only 20 questions per language), which restricts the generalizability of the statistical conclusions.
- The definition of cultural specificity relies on GPT-4o-Mini filtering, which may introduce systematic bias.

## Related Work & Insights
- **vs. Translation benchmarks like MMLU/XQuAD**: CaLMQA achieves genuine cultural representation through translation-free collection, exposing gaps in cultural knowledge that translation-based benchmarks fail to capture.
- **vs. FactScore/VeriScore**: This work extends the VeriScore pipeline to multilingual scenarios (translation + search), though it remains constrained by the coverage of English search engines.
- The dataset can serve as a standard benchmark for evaluating the multicultural capabilities of LLMs and guidance for the composition of multilingual training data.

## Rating
- Novelty: ⭐⭐⭐⭐ The first multilingual long-form QA dataset focusing on cultural specificity, filling a crucial gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 7 models, combines automated and human evaluations, and includes statistical significance tests, though low-resource evaluation remains constrained.
- Writing Quality: ⭐⭐⭐⭐ Clear and systematic, with detailed descriptions of data collection and evaluation protocols.
- Value: ⭐⭐⭐⭐ Drives forward research in multilingual and culturally sensitive LLM evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] On Synthesizing Data for Context Attribution in Question Answering](on_synthesizing_data_for_context_attribution_in_question_answering.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](../../ACL2026/nlp_understanding/beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)

</div>

<!-- RELATED:END -->
