---
title: >-
  [Paper Note] KnowCoder-X: Boosting Multilingual Information Extraction via Code
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-lingual IE] This work proposes KnowCoder-X, which represents multilingual IE schemas through uniform Python classes and introduces an IE cross-lingual alignment instruction tuning stage (including a high-quality ParallelNER dataset), significantly boosting cross-lingual information extraction performance across 64 IE benchmarks.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-lingual IE"
  - "Code Generation"
  - "NER"
  - "Schema Unification"
  - "Parallel Data"
date: 2026-05-08
content_hash: f200b2beefaff919
---

# KnowCoder-X: Boosting Multilingual Information Extraction via Code

**Conference**: ACL 2025  
**arXiv**: [2411.04794](https://arxiv.org/abs/2411.04794)  
**Code**: [ICT-GoKnow/KnowCoder](https://github.com/ICT-GoKnow/KnowCoder)  
**Area**: Multilingual Translation  
**Keywords**: Cross-lingual IE, Code Generation, NER, Schema Unification, Parallel Data  

## TL;DR

This work proposes KnowCoder-X, which represents multilingual IE schemas through uniform Python classes and introduces an IE cross-lingual alignment instruction tuning stage (including a high-quality ParallelNER dataset), significantly boosting cross-lingual information extraction performance across 64 IE benchmarks.

## Background & Motivation

- **Background**: While large language models exhibit spontaneous cross-lingual alignment capabilities after pre-training on multilingual corpora, the performance gap between different languages remains substantial in information extraction (IE) tasks.
- **Limitations of Prior Work**: Experiments show that even after training on English NER data, the F1 score on Chinese parallel datasets is only 52.7 (compared to 95.1 in English), indicating that cross-lingual alignment for IE in LLMs is still weak; existing cross-lingual IE methods lack a unified schema representation.
- **Key Challenge**: The schemas (entity types, relation types, etc.) in IE tasks have different names across various languages but share the same semantics (e.g., Korean "사람" and Chinese "人物" both correspond to PER). The lack of a unified representation hinders cross-lingual knowledge transfer.
- **Goal**: To enhance the cross-lingual alignment capability of LLMs in IE tasks, enabling models trained on Chinese and English data to generalize to 29 unseen languages.
- **Key Insight**: Leveraging code (Python classes) to unify multilingual schema representations and performing cross-lingual alignment training through translated instance prediction tasks.
- **Core Idea**: Unifying multilingual IE schemas with Python classes + cross-lingual alignment instruction tuning = strong zero-shot cross-lingual IE migration.

## Method

### Overall Architecture

KnowCoder-X adopts a two-stage instruction tuning paradigm: (1) an IE cross-lingual alignment stage—training on translation instance prediction tasks to enhance cross-lingual transfer; (2) a Chinese-English UIE instruction tuning stage—training on 46 IE datasets to obtain the final model.

### Key Designs

#### 1. Multilingual Schema Unified Representation Based on Python Classes
- **Function**: Maps IE schemas of all languages to unified Python class definitions.
- **Mechanism**: Defines three base classes: Entity, Relation, and Event, with each specific concept inheriting from its corresponding base class. Non-English schemas are first mapped to the corresponding English classes; for instance, Korean "사람" and Chinese "人物" are both mapped to the `PER(Entity)` class. Class docstrings contain instance exemplars and concept descriptions.
- **Design Motivation**: The object-oriented nature of code is naturally suited for unifying schema representations and cross-lingual knowledge sharing; consistent schemas allow the model to efficiently share the knowledge of the same ontology across different languages.

#### 2. IE Cross-Lingual Alignment Instruction Tuning
- **Function**: Designs translation instance prediction tasks that concatenate the IE input-output of the source language with the IE input of the target language as the instruction, with the IE output of the target language as the completion.
- **Mechanism**: Given a source language sentence $s^{src}$ and annotated spans $I^{src}$, it concatenates the target language sentence $s^{tgt}$ and predicts $I^{tgt}$.
- **Design Motivation**: Unlike directly predicting full IE parallel data (which focuses on sentence alignment), this method prioritizes aligning translated instances, which is the core goal of cross-lingual IE alignment.

#### 3. Three-Stage IE Parallel Data Construction Pipeline
- **Function**: Automatically constructs high-quality IE parallel corpora.
- **Mechanism**:
    - **Stage 1 - Joint Translation**: Translates the sentence and spans simultaneously (rather than translating the sentence first and then aligning the spans) to minimize error accumulation.
    - **Stage 2 - Span Rewriting**: Rewrites spans that are not present in the target sentence after translation (analogous to retrieval-based correction).
    - **Stage 3 - Sentence Rewriting**: Rewrites the target sentence to contain all spans when some spans are still missing (span-then-sentence strategy).
- **Design Motivation**: Traditional label projection methods (translating the sentence first then locating spans) suffer from error propagation; the three-stage pipeline integrates three translation strategies: parallel processing, sentence-then-span, and span-then-sentence.
- **Gain**: Achieves an average of **99% faithfulness** across 10 languages on WikiANN.

#### 4. ParallelNER Dataset
- **Function**: Constructs a high-quality Chinese-English NER parallel dataset, totaling 257,190 samples.
- **Mechanism**: Uses WikiNeural (en→zh) and CLUENER2020 (zh→en) as source data, with GPT-4o-mini as the pipeline's backbone LLM. If failed, it switches to GPT-4o for reprocessing, and the remaining 97 hard cases are manually annotated.
- **Missing Rate**: Only 82/92720 (8.84‱) for WikiNeural, and 15/10000 for CLUENER2020.

### Loss & Training

LoRA fine-tuning (rank=32) is conducted using the standard language modeling cross-entropy loss, with Baichuan2-7B-Base as the base model.

## Key Experimental Results

### Main Results: Multiconer22 Cross-Lingual Zero-Shot Evaluation (9 Unseen Languages)

| Model | English | Chinese | Avg_cross (9 Unseen Languages) | Avg (Total) |
|------|------|------|----------------------|-----------|
| ChatGPT | 37.20 | 18.80 | 30.37 | 29.94 |
| B2NER | 54.80 | 45.40 | - | - |
| IEPILE | 53.19 | 39.26 | 28.48 | 31.71 |
| **KnowCoder-X** | **56.37** | **47.53** | **39.53** | **41.79** |
| Supervised | 62.70 | 53.10 | 54.22 | 54.89 |

- Outperforms ChatGPT by **+30.17%** and SoTA by **+20.03%**.
- Achieves zero-shot performance close to the supervised method on Spanish and Turkish.

### Ablation Study: Supervised NER Evaluation

| Dataset | YAYI-UIE | IEPILE | B2NER | KnowCoder-X |
|--------|---------|--------|-------|-------------|
| CoNLL 2003 | 96.77 | 92.49 | 92.56 | **94.69** |
| MultiNERD | 88.42 | 94.60 | 93.98 | **95.94** |
| Chinese MSRA | 95.97 | 87.99 | 92.22 | **96.01** |
| Chinese Avg | - | 90.96 | 94.06 | **96.03** |

- Chinese IE comprehensively achieves SoTA, with an average improvement of +4.12 in the EAE task.

### Key Findings

1. **Emergence Effect of Cross-Lingual Alignment**: Training only on Chinese and English data enables migration to 29 unseen languages, with an average improvement of +11.43% on 20 low-resource African languages.
2. **NER Alignment Benefits All IE Tasks**: The improvement in cross-lingual NER capabilities drives the overall advancement of RE, ED, and EAE through foundational spotting ability.
3. **Advantages of Code Representation**: Unified Python classes ensure semantic consistency of schemas across different languages, which is key to cross-lingual transfer.

## Highlights & Insights

- The design of **unifying schema representation via code** is simple yet effective—leveraging object-oriented inheritance and typing systems to naturally adapt to IE ontological structures.
- The **three-stage parallel data pipeline** achieves nearly 100% annotation faithfulness, addressing the error propagation issue of traditional label projection.
- This work systematically validates the effectiveness of **code-based IE methods** in cross-lingual scenarios for the first time.
- ParallelNER stands as a valuable community resource for future cross-lingual IE research.

## Limitations & Future Work

- Only Chinese-English parallel data was constructed, which can be extended to more language pairs.
- The base model Baichuan2-7B is relatively small; larger models could potentially yield better results.
- Parallel data for RE and EE tasks was not used for alignment training due to context length limitations caused by excessive schema concepts.
- The three-stage pipeline relies on the capabilities of the GPT-4o series, which may limit its effectiveness for low-resource languages with poor LLM quality.
- Unsupervised cross-lingual alignment methods remain unexplored.

## Related Work & Insights

- **KnowCoder**: The pioneering work representing IE via code; KnowCoder-X is its multilingual extension.
- **GoLLIE**: Embeds annotation guidelines as class docstrings in schemas, sharing a similar concept.
- **IEPILE / B2NER**: Prominent UIE baselines, which nonetheless neglect the unified representation of schemas across different languages.
- **CLaP**: A representative of traditional label projection, which KnowCoder-X's three-stage pipeline significantly outperforms.
- **Insights**: Leveraging the structural properties of code to bridge semantic gaps across different languages is a promising direction.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The combined idea of code unifying schema + cross-lingual alignment fine-tuning is novel and practical.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: The comprehensive evaluation across 64 benchmarks is highly convincing, and the cross-lingual zero-shot results are outstanding.
- **Value** ⭐⭐⭐⭐: The ParallelNER dataset and the three-stage pipeline hold independent value.
- **Writing Quality** ⭐⭐⭐: The paper is somewhat long, and some mathematical formulations are quite redundant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [\[ACL 2026\] A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction](../../ACL2026/multilingual_mt/a_multilingual_dataset_and_empirical_validation_for_the_mutual_reinforcement_eff.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)

</div>

<!-- RELATED:END -->
