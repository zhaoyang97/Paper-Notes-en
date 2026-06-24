---
title: >-
  [Paper Note] Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books
description: >-
  [ACL 2025][Multilingual & Machine Translation][extremely low-resource translation] This work decomposes grammar-book-assisted extremely low-resource translation into two steps: **grammar rule retrieval** and **rule application**. It proposes a Rule-by-Rule retrieval strategy and a code-format grammar rule representation, achieving a 13.1% BLEU end-to-end improvement in Zhuang translation.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "extremely low-resource translation"
  - "grammar book"
  - "code representation"
  - "Zhuang language"
  - "rule retrieval"
date: 2026-05-08
content_hash: 658816088456d95f
---

# Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books

**Conference**: ACL 2025  
**arXiv**: [2506.01796](https://arxiv.org/abs/2506.01796)  
**Code**: [GitHub - ZhuangRules](https://github.com/Infinite-set/ZhuangRules)  
**Institution**: Peking University (Wangxuan Institute of Computer Technology)  
**Area**: Multilingual MT / Extremely Low-Resource Translation  
**Keywords**: extremely low-resource translation, grammar book, code representation, Zhuang language, rule retrieval  

## TL;DR

This work decomposes grammar-book-assisted extremely low-resource translation into two steps: **grammar rule retrieval** and **rule application**. It proposes a Rule-by-Rule retrieval strategy and a code-format grammar rule representation, achieving a 13.1% BLEU end-to-end improvement in Zhuang translation.

## Background & Motivation

**Background**: The majority of the world's languages are extremely low-resource (XLR) languages, with only thousands of parallel sentences available, rendering traditional pre-training/fine-tuning methods completely infeasible. LLMs have shown potential in XLR MT through in-context learning (ICL) by leveraging small-scale linguistic resources such as dictionaries and parallel sentence pairs. As the most systematic linguistic description, grammar books are theoretically best suited to guide translation.

**Limitations of Prior Work**: Aycock et al. (2024) pointed out that LLMs may only extract bilingual vocabulary explanations from grammar books as a "shortcut" (lexical leakage), rather than truly understanding grammar rules. Existing studies remain divided on whether grammar books are genuinely effective, largely due to the lack of controlled evaluation methods that can isolate grammar comprehension from vocabulary knowledge.

**Key Challenge**: Grammar books contain a large number of fine-grained rules. When the entire book is presented to the LLM at once, the model struggles to locate relevant rules (with recall at only around 50%), and the accuracy of LLMs in understanding and executing rules is limited when rules are formulated in natural language text. Efficiently retrieving and accurately applying grammar rules remains a key bottleneck.

**Goal**: (1) Construct the ZhuangRules controlled dataset to answer whether LLMs truly understand grammar rules; (2) Decouple grammar-book-assisted translation into retrieval and application steps to pinpoint bottlenecks and solve them targetedly.

**Key Insight**: Leverage the natural similarity between grammatical operations and code structures (e.g., affixation $\rightarrow$ arithmetic operations, conditional selection $\rightarrow$ if-else) to convert grammar rules into Python pseudocode format, simultaneously improving performance in both the retrieval and application steps.

**Core Idea**: Achieve a 13.1% BLEU end-to-end improvement in grammar-book-assisted XLR translation using a Rule-by-Rule retrieval strategy and code-format grammar rule representation.

## Method

### Overall Architecture

The grammar-book-assisted extremely low-resource translation is decomposed into two independent phases:

1. **Grammar Rule Retrieval**: Locate the required grammar rules from the grammar book given the sentence to be translated.
2. **Grammar Rule Application**: Complete the translation based on the retrieved rules.

The core innovations are: (a) proposing a Rule-by-Rule retrieval strategy that simplifies long-context comprehension into binary classification; (b) representing grammar rules in code format to improve both retrieval and application performance.

### Key Designs

1. **ZhuangRules Controlled Dataset**:

    - **Function**: Provides a controlled evaluation benchmark that isolates grammar comprehension from vocabulary knowledge.
    - **Mechanism**: Construct a modular grammar rule dataset for the Zhuang language: 109 atomic grammar rules, with an average of 5.6 Zhuang-Chinese parallel sentence pairs per rule (608 pairs in total). The key design is to provide a Zhuang-Chinese dictionary covering all relevant vocabulary for each test sentence pair, completely decoupling grammatical understanding from vocabulary knowledge. Rules are annotated along three dimensions: action type (e.g., affixation, word order swapping), difficulty (Easy/Medium/Hard, with average operations of 1.2/1.5/2.1), and linguistic domain (morphology, word order, etc., following the WALS classification).
    - **Design Motivation**: Previously, it was impossible to determine whether the LLM was diligently applying grammar rules or exploiting lexical shortcuts. Only after strictly controlling local vocabulary variables can we truly answer whether LLMs can understand grammar rules.

2. **Rule-by-Rule Retrieval Strategy**:

    - **Function**: Transforms the long-context grammar book comprehension problem into an efficient binary classification problem, significantly boosting rule retrieval recall.
    - **Mechanism**: A pilot study revealed that the Full-Book approach (providing the entire grammar book at once) achieves only ~50% recall, and translation performance drops sharply as the number of irrelevant rules increases. The Rule-by-Rule strategy evaluates whether each individual rule is relevant to the sentence to be translated (binary classification), trading more API calls (109 times vs. 1 time) for significantly higher accuracy, raising recall to ~89%.
    - **Design Motivation**: Pinpointing the bottleneck in retrieval rather than application makes trading computational overhead for accuracy a reasonable engineering trade-off. The binary classification problem is much simpler than selecting a relevant subset from 109 rules.

3. **Code-Format Grammar Rule Representation**:

    - **Function**: Leverages the strong code comprehension capability of LLMs to enhance the retrieval and application of grammar rules.
    - **Mechanism**: Utilizing the natural similarity between grammatical operations and code structures (affixation $\rightarrow$ arithmetic operations, conditional selection $\rightarrow$ if-else, word order transformation $\rightarrow$ sequence operations), GPT-4o is used via 5-shot ICL to translate textual rules into Python pseudocode functions. Each code rule consists of brief comments and a pseudocode function body. Quality check: out of 10 randomly sampled rules, all followed Python syntax, with only one missing minor information.
    - **Design Motivation**: LLMs perform far better in code comprehension and execution than in following natural language instructions. Procedural representation of code is particularly effective for complex, multi-step grammatical operations.

### Loss & Training

This work does not train any models; all experiments are based on ICL prompting. Parallel sentences use 2-shot ICL in translation experiments. IGT (Interlinear Glossed Text) is generated by GPT-4o, using 123 manually annotated IGTs as ICL exemplars, achieving ~72% morpheme accuracy.

## Key Experimental Results

### Main Results

Compare the recall of different retrieval strategies on ZhuangRules:

| Retrieval Strategy | Model | za→zh recall | zh→za recall | Avg. Retrieved Rules |
|---------|------|:----------:|:----------:|:----------:|
| BM25 | — | 41.6 (rec@5) | 27.3 (rec@5) | 5 |
| Full-Book | Qwen-72B | 52.8 | 49.4 | 1.8 |
| Rule-by-Rule (text) | Qwen-72B | 89.4 | 84.7 | 4.1 |
| Rule-by-Rule (code) | Qwen-72B | **89.6** | **87.1** | 3.9 |
| Rule-by-Rule (text) | Llama-70B | 69.7 | 75.8 | 2.2 |
| Rule-by-Rule (code) | Llama-70B | **82.2** | **87.5** | 4.2 |
| Rule-by-Rule (text) | Qwen-7B | 55.1 | 67.9 | 2.5 |
| Rule-by-Rule (code) | Qwen-7B | **68.4** | **80.3** | 3.8 |

Code format shows the most significant improvement on small models: Qwen-7B receives a +13.3/+12.4 recall boost, and Llama-70B receives a +12.5/+11.7 recall boost.

### Rule Application Performance

Compare different rule application settings on ZhuangRules (average BLEU/chrF++ across 6 model-direction combinations):

| Setting | Avg. BLEU | Avg. chrF++ |
|------|:--------:|:----------:|
| No Rule (w/o Lexicon) | 0.9 | 3.0 |
| No Rule | 25.5 | 38.0 |
| Parallel Examples | 60.2 | 67.4 |
| Gold Textual Rule | 45.7 | 60.7 |
| Gold Textual Rule + Parallel Examples | 70.2 | 75.4 |
| Gold Code Rule | 57.9 | 69.2 |
| **Gold Code Rule + Parallel Examples** | **72.4** | **77.9** |

Code rules vs. textual rules: +12.2 BLEU (45.7 $\rightarrow$ 57.9); combining with parallel examples yields the optimal 72.4 BLEU.

### Ablation Study

| Rule Format | Easy (za→zh) | Medium | Hard | Easy (zh→za) | Medium | Hard |
|---------|:----------:|:------:|:----:|:----------:|:------:|:----:|
| Text Rule | 65.6 | 51.3 | 34.6 | 85.5 | 82.4 | 69.3 |
| Code Rule | 76.3 | 57.9 | 48.6 | 93.0 | 87.5 | 76.8 |
| **Δ** | +10.7 | +6.6 | **+14.0** | +7.5 | +5.2 | +7.5 |

The code format brings the largest improvement on hard rules (za $\rightarrow$ zh: +14.0 BLEU), indicating that procedural representation of code is especially effective for complex multi-step operations.

### Cross-lingual Generalization (MTOB, Kalamang, Qwen-72B)

| Rule Format | kgv→eng BLEU | eng→kgv BLEU |
|---------|:----------:|:----------:|
| Gold Textual Rule | 14.6 | 43.8 |
| Gold Code Rule | **16.0** | **44.5** |

It remains effective on Kalamang, another XLR language, validating the cross-lingual generalization ability of the code format.

### Key Findings

- The code format brings the largest improvement on hard rules (za $\rightarrow$ zh: +14.0 BLEU), showing that procedural representation of code is particularly effective for complex multi-step operations.
- The Rule-by-Rule retrieval strategy raises recall from ~50% (Full-Book) to ~89%, at the cost of 109 API calls vs. 1.
- The code format achieves the most significant improvement on smaller models: Qwen-7B gets a +13.3/+12.4 recall boost.
- Cross-lingual generalization on Kalamang validates the universality of the proposed method.
- The best end-to-end combination (Code Rule + Rule-by-Rule) delivers a 13.1% BLEU improvement over Full-Book + Textual Rule.

## Highlights & Insights

- **Elegant Problem Decomposition**: Decomposing end-to-end grammar-book translation into retrieval and application phases helps pinpoint the main bottleneck (retrieval) and propose a targeted Rule-by-Rule strategy. This decomposition idea is highly transferable to other NLP tasks requiring retrieval and application of specific rules from a large knowledge repository.
- **First Application of Code-Augmented Reasoning in Linguistics**: Leveraging the strong code comprehension of LLMs to process grammar rules is an innovative application of code-augmented reasoning in the field of linguistics.
- **Meticulous Controlled Evaluation Design**: ZhuangRules establishes a methodological benchmark for XLR translation evaluation by providing dictionaries to eliminate lexical interference and enabling controlled analysis with atomic rules and difficulty levels.

## Limitations & Future Work

- Experiments are only conducted on two languages (Zhuang and Kalamang); generalization to other language families remains to be verified.
- The Rule-by-Rule strategy requires querying the LLM for each rule individually (109 times vs. 1 time), incurring substantial computational overhead.
- Code rule conversion relies on GPT-4o, and its applicability to more complex or irregular grammars is unknown.
- There is still room for improvement in IGT generation quality (~72% morpheme accuracy).

## Related Work & Insights

- **vs. Aycock et al. (2024)**: They pointed out that LLMs extract lexical shortcuts (lexical leakage) from grammar books. This work directly responds to this challenge through the dictionary-controlled design of the ZhuangRules dataset.
- **vs. MTOB (Tanzer et al., 2024)**: MTOB provides a translation benchmark with grammar books and dictionaries but lacks a modular decomposition of grammar rules. The proposed Rule-by-Rule strategy is also validated effectively on their data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of two-step decomposition and code-format grammar rules is highly novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 3 models, 2 datasets, and multi-dimensional ablation studies, covering both retrieval and application phases.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, well-controlled experimental designs, and detailed analysis.
- Value: ⭐⭐⭐⭐ Offers a feasible new paradigm for extremely low-resource translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)

</div>

<!-- RELATED:END -->
