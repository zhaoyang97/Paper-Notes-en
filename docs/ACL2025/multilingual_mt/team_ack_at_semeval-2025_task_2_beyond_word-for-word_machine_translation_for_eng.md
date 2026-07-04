---
title: >-
  [Paper Note] Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs
description: >-
  [ACL 2025][Multilingual & Machine Translation][Machine Translation] This paper systematically evaluates the performance of 13 models (LLMs + traditional MT) on English-Korean entity-dense text translation in SemEval-2025 Task 2. Through automatic metrics and bilingual human evaluation, it reveals that while LLMs outperform traditional MT, they still generally fail on entity translations requiring cultural adaptation, and establishes a translation error taxonomy.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Machine Translation"
  - "English-Korean Translation"
  - "Entity Translation"
  - "Cultural Adaptation"
  - "Evaluation Metrics"
date: 2026-05-08
content_hash: 8df5feba23777c9f
---

# Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs

**Conference**: ACL 2025  
**arXiv**: [2504.20451](https://arxiv.org/abs/2504.20451)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Machine Translation, English-Korean Translation, Entity Translation, Cultural Adaptation, Evaluation Metrics

## TL;DR
This paper systematically evaluates the performance of 13 models (LLMs + traditional MT) on English-Korean entity-dense text translation in SemEval-2025 Task 2. Through automatic metrics and bilingual human evaluation, it reveals that while LLMs outperform traditional MT, they still generally fail on entity translations requiring cultural adaptation, and establishes a translation error taxonomy.

## Background & Motivation

Machine translation has made significant progress with the introduction of the Transformer paradigm, but still faces severe challenges when translating knowledge-intensive and entity-rich texts. The core problem lies in the fact that entity translations cannot simply be transliterated; instead, they often require transcreation across cultures.

**Classic Case**: The English phrase "Rotten Tomatoes" (a movie review website) should be translated into Korean with a phonetic transliteration rather than the literal meaning "rotten tomatoes." Such translations, which require cultural background knowledge, represent the weak point of current MT systems.

**Limitations of Prior Work**:

**Traditional MT systems lack cultural understanding**: Models like NLLB-200 and mBART-50 excel at general translation but fail to handle entity names that require cultural adaptation.

**English bias of LLMs**: Although LLMs show promise in zero-shot translation, their training data is English-centric, rendering them deficient in capturing the socio-cultural and historical contexts of Korean.

**Unreliable automatic metrics**: Existing evaluation metrics (BLEU, COMET) cannot accurately reflect the cultural appropriateness of entity translation, potentially yielding misleading scores.

**Understudied English-Korean pair**: Cross-cultural entity translation research primarily focuses on Western language pairs, leaving phonetically and morphologically complex pairs with distinct writing systems like English-Korean under-researched.

**Key Challenge**: Entity translation requires making the correct choice between literal transliteration and contextual transcreation, yet existing systems lack this judgment capability.

**Key Insight**: Quantitatively analyzing the performance differences of diverse models in English-Korean entity translation through a comprehensive automatic and human evaluation of 13 models, constructing an error taxonomy, and revealing the gap between automatic metrics and human evaluation.

## Method

### Overall Architecture
This study is a systematic evaluation rather than proposing a new method. The framework consists of: selecting 13 models $\rightarrow$ translating 5,082 English-Korean sentence pairs from the XC-Translate dataset $\rightarrow$ evaluating using three automatic metrics $\rightarrow$ conducting bilingual human evaluation on $50\times13$ samples $\rightarrow$ constructing an error taxonomy $\rightarrow$ multi-dimensional analysis.

### Key Designs

1. **Model Selection Coverage**:

    - LLMs (11 models): GPT-4/4o/o1/o1-mini (OpenAI), Claude 3.5 Sonnet/Haiku (Anthropic), Gemini 1.5 Pro/Flash (Google), Grok 2 (xAI), DeepSeek R1 (DeepSeek), Llama 3 (Meta)
    - Traditional MT (2 models): NLLB-200 and mBART-50
    - **Design Motivation**: Covering closed-source/open-source, small/large parameter sizes, and different architectures to comprehensively evaluate the state-of-the-art.

2. **Three-Dimensional Automatic Evaluation**:

    - **BLEU**: A classic n-gram overlap metric, resource-efficient but weakly correlated with human judgment.
    - **COMET**: Predicts translation quality utilizing neural networks, aligning better with human judgment but lacking word-level insights.
    - **M-ETA**: Specifically evaluates entity-level translation quality, compensating for the lack of entity evaluation in the former two.
    - Reason for using them complementarily: No single metric is sufficient to comprehensively measure translation quality.

3. **High-Standard Human Evaluation**:

    - Recruited two bilingual annotators who have lived in both South Korea and the United States for over 5 years.
    - Annotated 50 translation samples for each model, assessing translation accuracy, error locations, and error causes.
    - Constructed an error taxonomy extended from Popović (2018).

4. **Error Taxonomy**:

    - **Incorrect Response (308 pairs)**: The model answers the prompt instead of translating, making it the most common error.
    - **Incorrect Entity Name (266 pairs)**: Entity names are translated incorrectly through literal/phonetic/word-for-word translation rather than semantic translation.
    - Other categories: grammatical errors, inappropriate style, etc.
    - **Design Motivation**: To provide actionable error insights to guide the improvement of future translation systems.

## Key Experimental Results

### Main Results — Automatic Metric Evaluation

| Model | BLEU | COMET | M-ETA |
|------|------|-------|-------|
| o1 (OpenAI) | **0.387** | 0.920 | 0.375 |
| o1 Mini | 0.383 | **0.920** | 0.331 |
| Gemini 1.5 Pro | 0.381 | 0.909 | **0.483** |
| GPT-4o | 0.369 | 0.909 | 0.395 |
| Grok 2 | 0.381 | 0.914 | 0.351 |
| NLLB-200 | 0.220 | 0.890 | 0.166 |
| DeepSeek R1 | 0.007 | 0.490 | 0.003 |
| Llama 3 | 0.033 | 0.553 | 0.056 |

### Ablation Study — Impact of Entity Popularity on Translation Quality

| Popularity | BLEU Avg. | COMET Avg. | M-ETA Variation | Description |
|--------|----------|------------|--------------|------|
| Low~High | 0.24~0.27 | 0.83~0.84 | 0.00224 | BLEU/COMET are insensitive to entity popularity |
| - | - | - | - | M-ETA is able to capture the impact of popularity on entity translation |

### Key Human Evaluation Data

| Metric | Value | Description |
|------|------|------|
| Samples containing translation errors | 459/650 (70.6%) | The vast majority of translations contains errors |
| Entity translation errors | 266/459 (57.9%) | Over half of the errors are entity-related |
| Model with lowest error rate | Grok 2 | Best in human evaluation |
| Correlation between BLEU and human judgment | 0.41 | Moderate positive correlation |
| M-ETA entity error detection rate | 88.7% | Most reliable at the entity level |

### Key Findings
- LLMs generally outperform traditional MT, with DeepSeek R1 and Llama 3 being exceptions (likely due to the insufficient Korean capability of their smaller-parameter versions).
- The most common translation error is "answering the question instead of translating" (Incorrect Response), indicating that LLMs still suffer from instruction-following issues in simple translation tasks.
- Entity popularity affects entity translation but not overall sentence translation—standard metrics like BLEU/COMET fail to capture this.
- Translation difficulty varies greatly across different entity types: "Plant" and "Natural Place" types are more challenging (requiring language-specific names), while "Book Series" is relatively easier (allowing literal translation).
- The Claude 3.5 series performs exceptionally poorly on BLEU (0.16-0.20) despite performing reasonably well in COMET and human evaluations, highlighting the inconsistency among evaluation metrics.

## Highlights & Insights
- Provides a valuable benchmark reference in the English-Korean MT field through a systematic evaluation and comprehensive comparison of 13 models.
- The construction of the error taxonomy goes beyond simple correct/incorrect judgments to reveal specific failure modes.
- The discovery of "Incorrect Response" as the primary error type suggests that instruction-following in LLMs for translation tasks still has room for improvement.
- Demonstrates the unique value of M-ETA in entity-level translation evaluation, quantifying the limitations of standard BLEU and COMET on such tasks.

## Limitations & Future Work
- Only evaluates English-to-Korean translation, leaving the Korean-to-English direction unaddressed.
- The scale of human evaluation is limited (650 samples), which may lower statistical significance.
- Only uses a dataset with a fixed Q&A template format, without involving genres such as long documents or narrative texts.
- The error taxonomy does not distinguish the impact of different translation error severities on comprehension.
- Some open-source models (DeepSeek R1, Llama 3) were evaluated using smaller-parameter versions, which may not represent their optimal performance.

## Related Work & Insights
- Continues the direction of cross-cultural translation evaluation established by XC-Translate (Conia et al., 2024), focusing on an in-depth analysis of the English-Korean pair.
- Complements work in KG-MT (utilizing knowledge graphs to improve entity translation)—this paper provides problem diagnosis, while KG-MT offers a solution direction.
- Suggests the future direction of automatic evaluation metrics: requiring more fine-grained, task-specific evaluation metrics like M-ETA.
- Implications for translation system development: requiring more integration of cross-cultural entity knowledge during pre-training and fine-tuning.

## Rating
- Novelty: ⭐⭐⭐ The evaluation methodology itself has limited innovation, but the error taxonomy and dimensions of analysis contribute value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 13 models, 3 automatic metrics, human evaluation, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and in-depth analysis, though some tables could be more concise.
- Value: ⭐⭐⭐ Provides valuable references for the English-Korean translation direction, but the generalizability of results (to other language pairs) remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2026\] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models](../../ACL2026/multilingual_mt/vocabulary_shapes_cross-lingual_variation_of_word-order_learnability_in_language.md)

</div>

<!-- RELATED:END -->
