---
title: >-
  [Paper Note] CLIX: Cross-Lingual Explanations of Idiomatic Expressions
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-Lingual Explanation] The cross-lingual idiom explanation task (CLIX) is proposed, along with a dataset containing English idioms and their Spanish/German explanations. The performance of seq2seq models and LLMs on this task is systematically evaluated, revealing that a GPT-3.5 Turbo pipeline strategy (generating English explanations followed by translation) combined with few-shot learning achieves the best results…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Explanation"
  - "Idiom Comprehension"
  - "Definition Generation"
  - "Language Learning"
  - "LLM"
date: 2026-05-08
content_hash: 92180524aa976696
---

# CLIX: Cross-Lingual Explanations of Idiomatic Expressions

**Conference**: ACL 2025  
**arXiv**: [2501.03191](https://arxiv.org/abs/2501.03191)  
**Code**: [https://github.com/blast-cu/CLIX](https://github.com/blast-cu/CLIX)  
**Area**: Multilingual Translation  
**Keywords**: Cross-Lingual Explanation, Idiom Comprehension, Definition Generation, Language Learning, LLM

## TL;DR

The cross-lingual idiom explanation task (CLIX) is proposed, along with a dataset containing English idioms and their Spanish/German explanations. The performance of seq2seq models and LLMs on this task is systematically evaluated, revealing that a GPT-3.5 Turbo pipeline strategy (generating English explanations followed by translation) combined with few-shot learning achieves the best results, with human evaluation scores for fluency and accuracy exceeding 4.7+/5.

## Background & Motivation

Language learning technology has become an essential tool for foreign language education, with vocabulary expansion being one of the core focus areas. Automatic definition generation systems have been proposed to assist learners in expanding their vocabulary. However, existing systems face two core limitations:

**Limitations of Prior Work**:
- **Vocabulary barriers in definitions**: Generated definitions themselves may contain vocabulary and grammatical structures unfamiliar to learners, causing a recursive problem of comprehension difficulties.
- **Challenges of non-standard language**: Existing systems mostly ignore the complexity of non-standard language, such as idioms. The meaning of an idiom usually cannot be inferred from the literal sense of its constituent words (e.g., "see eye to eye" has nothing to do with "seeing" or "eyes" and actually expresses agreement).

**Key Challenge**: Idioms are crucial yet difficult elements in language learning, and current definition generation systems neither handle the non-literal meanings of idioms well nor ensure that the generated explanations are understandable to learners.

**Key Insight**: Providing explanations in the learner's native language can resolve both problems simultaneously—(1) native language explanations eliminate vocabulary barriers, and (2) transforming the task from translation to explanation generation allows for more flexible phrasing. Based on this, the CLIX (Cross-Lingual explanations of Idiomatic eXpressions) task is proposed, which generates natural language explanations in a target language (Spanish/German) given an English idiom.

The key novelty lies in using "explanation" instead of "definition", allowing the output to include richer content such as usage examples and etymological information, forming a one-to-many mapping.

## Method

### Overall Architecture

CLIX is formulated as a text-to-text generation task: given an idiom $I$ in the source language and an optional context $C$, the system generates an explanation $E$ in the target language $L_T$. The authors explore two major categories of strategies:

- **Direct**: The model directly generates target-language explanations from English idioms.
- **Pipeline**: The model first generates English explanations and then translates them into the target language.

Systematic experiments are conducted under different model architectures (fine-tuning vs. few-shot LLM) and context enrichment methods.

### Key Designs

1. **Dataset Construction (EPIE-ME and Oxford-ME)**:

    - EPIE-ME: Constructed based on the EPIE corpus, containing 628 English idioms and their English/Spanish/German explanations. Idioms are tagged with thematic category labels (81 categories), annotated using GPT-3.5 and manually corrected.
    - Oxford-ME: Based on the Oxford Dictionary of Idioms (4th edition), containing 6,218 idioms. For idioms lacking contexts (71.4%), Llama 3.1 is used to generate example sentences.
    - The test sets for both datasets are manually verified and corrected by native speakers.
    - Non-English explanations are initially translated using Google Translate, and the test sets are manually corrected by native experts.

2. **Context Enrichment Strategies (Only for LLMs)**:

    - **Sentence-Level Context (SL)**: Appending an example sentence containing the target idiom to the end of the prompt to provide more contextual information.
    - **Categorical Information (Cat)**: Adding the thematic category label of the idiom (e.g., "happiness", "anger") to the prompt as a semantic cue.
    - Experiments show that these enrichments do not benefit GPT models, but have positive effects on Llama under certain configurations.

3. **Few-Shot Exemplar Selection Strategies**:

    - **Random Selection**: Randomly choosing $k$ exemplars from the training set.
    - **Category-Aware Selection**: Retrieving the category label of the target idiom (known or inferred by the LLM), selecting $2k$ candidates from the same or similar categories, and then randomly sampling $k$ exemplars.
    - Surprisingly, the random selection strategy performs the best, indicating that category relevance provides limited help for in-context learning in this task.

### Loss & Training

- **Fine-Tuned Models**: T5 and mT5 are trained using standard seq2seq training. T5 is used for the English explanation generation step in the pipeline (as it is primarily pre-trained on English data), and mT5 is used for direct cross-lingual generation.
- **LLMs**: GPT-3.5 Turbo and Llama 3.1 8B Instruct are used with zero-shot and few-shot prompting without fine-tuning.
- **Translation Step**: Google Translate is used in the fine-tuning setup, while the LLM translates autonomously in the LLM setup.

## Key Experimental Results

### Main Results

| Model | Strategy | EPIE-ME Sentence Similarity | Oxford-ME Sentence Similarity |
|------|------|-------------------|---------------------|
| mT5 Direct | Fine-tuned | 38.06 | 43.21 |
| T5 Pipeline | Fine-tuned | 43.54 | 46.09 |
| Llama Direct | Zero-Shot | 59.39 | 55.32 |
| Llama Pipeline | Zero-Shot | 60.16 | 55.01 |
| GPT Direct | Zero-Shot | 65.06 | 61.03 |
| GPT Pipeline | Zero-Shot | 69.60 | 66.10 |
| GPT Direct | 5-Shot | 71.15 | 66.13 |
| **GPT Pipeline** | **5-Shot** | **71.84** | **68.54** |

LLMs significantly outperform fine-tuned models. The pipeline strategy consistently outperforms the direct strategy, and few-shot learning further improves performance.

### Ablation Study

| Context Enrichment | GPT (EPIE-ME) | Llama (EPIE-ME) |
|-----------|---------------|-----------------|
| Direct (No Enrichment) | 63.91 | 60.45 |
| + Sentence Context | 61.36 | 59.73 |
| + Category | 61.96 | 59.44 |
| + Sentence + Category | 61.61 | 58.98 |
| Pipeline (No Enrichment) | 66.98 | 61.32 |
| + Sentence Context | 66.22 | 64.00 |
| + Category | 65.64 | 60.53 |
| + Sentence + Category | 66.36 | 63.04 |

Context enrichment has a negative impact on GPT, but shows some improvement for Llama under the pipeline + sentence context setup.

### Human Evaluation

| Dimension | Average Score (1-5) |
|------|-------------|
| Fluency | 4.70 |
| Accuracy | 4.78 |
| Krippendorff's $\alpha$ (Fluency) | 0.642 |
| Krippendorff's $\alpha$ (Accuracy) | 0.417 |

### Key Findings

- The pipeline strategy (explanation followed by translation) consistently outperforms the direct strategy, demonstrating that step-by-step processing effectively reduces the difficulty of cross-lingual generation.
- LLMs show a massive improvement over fine-tuned seq2seq models (~30 points), but the gap is smaller on Oxford-ME. This is because Oxford-ME contains shorter gold-standard explanations, which penalize the verbose outputs of LLMs.
- T5 Pipeline outperforms mT5 Direct by 10%+, indicating that T5, specialized in English understanding, is more effective in the first step of the pipeline.
- Pure translation methods are insufficient for educational scenarios, with 42% of Spanish and 48.5% of German translations rated as unnatural.
- Performance analysis by topic shows that the "Anger" category achieves up to 87.98 in Spanish, but only 73.12 in German, indicating an interaction effect between language and topic.

## Highlights & Insights

- **Education-oriented task design**: Emphasizing "explanation" over "definition" allows for more flexible output forms, which aligns better with the practical needs of educational applications.
- **Divergence between automatic and human evaluation**: While automatic metrics (sentence similarity ~72) suggest the task remains challenging, human evaluation (4.7+/5) indicates that the generation quality is actually quite high, revealing the limitations of automatic evaluation metrics.
- **Insights on Pipeline vs. Direct**: The multi-step strategy is more effective because direct generation forces the model to handle both comprehension and cross-lingual generation sub-tasks simultaneously.
- **Translation noise analysis**: Quantifying the quality of Google Translate via edit distance reveals that German EPIE-ME requires the most correction (normalized edit distance of 0.283), underscoring the necessity of strict data quality control.

## Limitations & Future Work

- **Limited dataset size**: EPIE-ME contains only 628 idioms, and Oxford-ME cannot be publicly released due to copyright constraints.
- **Insufficient language coverage**: The study only covers English to Spanish/German, which are high-resource languages. The performance on low-resource target languages remains unknown.
- **Imperfect evaluation metrics**: Current metrics struggle to capture core dimensions of explanation quality, such as whether the core metaphor of an idiom has been successfully conveyed.
- **Metaphorical gradient of idioms**: Different idioms exhibit varying distances from their literal meanings (e.g., "building bridges" is easier to deduce literally than "kick the bucket"), but there is currently no metric to quantify this difference.
- **Incorporation of multimodality**: Combining images or animations to explain idiom meanings could be considered, which might be more effective for educational purposes.

## Related Work & Insights

- This work is closely related to IdiomKB by Li et al. (2024), but the latter focuses on translation views and ontology construction, whereas CLIX is dedicated to explanation generation in educational settings.
- Cross-lingual definition generation by Zhang et al. (2023) utilizes contrastive learning to prevent language confusion, which provides useful insights into language control issues in multilingual generation.
- For idiom explanation in low-resource languages, incorporating retrieval-augmented generation (RAG) to retrieve relevant information from idiom databases could be a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The task definition is novel, but the methodology mainly combines existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, strategies, and enrichment methods, and includes human evaluation along with detailed error analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, with a clear motivation and strong relevance to educational settings.
- Value: ⭐⭐⭐⭐ Provides class-leading baselines and analysis for NLP applications in language education, though practical deployment is still some distance away.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)

</div>

<!-- RELATED:END -->
