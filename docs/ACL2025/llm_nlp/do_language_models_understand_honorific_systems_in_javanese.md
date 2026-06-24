---
title: >-
  [Paper Note] Do Language Models Understand Honorific Systems in Javanese?
description: >-
  [ACL 2025][LLM (Other)][honorifics] This work constructs the first Javanese honorific corpus, Unggah-Ungguh (4,024 sentences covering four honorific levels), and systematically evaluates the capability of LLMs to understand the Javanese honorific system across four tasks: classification, style transfer, cross-lingual translation, and dialogue generation. The results reveal that even the strongest closed-source model (GPT-4o) achieves a zero-shot classification accuracy of onl…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "honorifics"
  - "Javanese"
  - "low-resource"
  - "Unggah-Ungguh"
  - "linguistic evaluation"
date: 2026-05-08
content_hash: d8b123e7e4ba0b06
---

# Do Language Models Understand Honorific Systems in Javanese?

**Conference**: ACL 2025  
**arXiv**: [2502.20864](https://arxiv.org/abs/2502.20864)  
**Code**: [https://github.com/JavaneseHonorifics/Unggah-Ungguh](https://github.com/JavaneseHonorifics/Unggah-Ungguh)  
**Area**: LLM/NLP - Low-Resource Language Evaluation  
**Keywords**: honorifics, Javanese, low-resource, Unggah-Ungguh, linguistic evaluation

## TL;DR
This work constructs the first Javanese honorific corpus, Unggah-Ungguh (4,024 sentences covering four honorific levels), and systematically evaluates the capability of LLMs to understand the Javanese honorific system across four tasks: classification, style transfer, cross-lingual translation, and dialogue generation. The results reveal that even the strongest closed-source model (GPT-4o) achieves a zero-shot classification accuracy of only 53.5% and shows a severe bias toward specific honorific levels.

## Background & Motivation

**Background**: Javanese has over 98 million speakers. One of its core features is a complex honorific system known as Unggah-Ungguh Basa, which comprises four levels—Ngoko (most informal), Ngoko Alus (semi-formal), Krama (formal), and Krama Alus (most formal). The choice of honorifics depends on the social relationship between the speaker, the listener, and the referent.

**Limitations of Prior Work**: (1) Existing Javanese corpora suffer from a severe imbalance in honorific levels, with the majority biased toward Ngoko; (2) There is a lack of specially annotated Javanese honorific corpora for NLP tasks; (3) As LLMs are increasingly deployed as personal assistants, their ability to understand and generate appropriate honorifics directly influences cultural sensitivity and user trust.

**Key Challenge**: The honorific system requires models to not only understand semantics but also capture pragmatic information, such as social hierarchy, conversational roles, and situational context—presenting a massive challenge for existing models, particularly in low-resource settings.

**Goal**: To systematically evaluate the capabilities of LLMs in understanding and generating Javanese across the four honorific levels, and to identify their biases and limitations.

**Key Insight**: Build a balanced honorific corpus and design four benchmark tasks covering both comprehension and generation.

**Core Idea**: By constructing the first balanced corpus annotated with four Javanese honorific levels and proposing four evaluation tasks, this work reveals that current LLMs suffer from a severe lack of understanding of complex honorific systems.

## Method

### Overall Architecture
Construct the Unggah-Ungguh corpus → Design four evaluation tasks → Perform comparative evaluation using both fine-tuned models and zero-/few-shot general-purpose models. The fine-tuned models are used for classification tasks and serve as automatic evaluation tools for subsequent tasks. General-purpose models include closed-source (GPT-4o, Gemini 1.5 Pro) and open-source (Llama 3.1 8B, Gemma2 9B, Sailor2 8B, SahabatAI) models.

### Key Designs

1. **Unggah-Ungguh Corpus Construction**:

    - **Function**: Manually curate a labeled corpus of 4,024 sentences from four authoritative reference books, such as *Kamus Unggah-Ungguh Basa Jawa*.
    - **Mechanism**: Since the original sources were not digitized, the pipeline involved scanning → OCR → two-stage native-speaker correction. The second stage of independent auditing identified and corrected 58 errors (1.5%). The final Shannon entropy reached 1.88, which is higher than nine other existing datasets, signifying the most balanced distribution.
    - **Design Motivation**: Existing Javanese corpora have highly imbalanced honorific distributions (mostly concentrated in Ngoko), making it impossible to fairly evaluate model capabilities.

2. **Task 1: Honorific Level Classification**:

    - **Function**: Classify the input text into one of the four honorific levels.
    - **Mechanism**: Fine-tune Javanese BERT/DistilBERT/GPT-2 and LSTM/rule-based baselines. Javanese DistilBERT achieves the highest accuracy of 95.65% and is used as the automatic evaluator for Task 4.
    - **Design Motivation**: To evaluate the models' capability to recognize honorific levels—a fundamental step in understanding Javanese honorific systems.

3. **Task 2: Honorific Style Transfer**:

    - **Function**: Translate a given text from one honorific style to another (e.g., Ngoko → Krama Alus).
    - **Mechanism**: Zero-shot translation to evaluate whether the model can alter the honorific level while preserving the semantic meaning.
    - **Design Motivation**: Honorific transfer requires precise lexical substitution and grammatical adjustments, serving as a direct test of the model's depth of understanding of the honorific system.

4. **Task 3: Cross-Lingual Honorific Translation**:

    - **Function**: Translate between Javanese and Indonesian at specific honorific levels.
    - **Mechanism**: Indonesian lacks an explicit honorific system, whereas Javanese has a rich honorific hierarchy. The KL divergence between the two is as high as 2.26, indicating a large discrepancy in vocabulary distributions.
    - **Design Motivation**: To test whether models can preserve honorific information in cross-lingual scenarios.

5. **Task 4: Dialogue Generation**:

    - **Function**: Generate dialogues that use appropriate honorifics given the social status of two speakers (e.g., student and teacher) and the context.
    - **Mechanism**: Manually design 160 evaluation scenarios and use the fine-tuned DistilBERT to automatically assess whether the generated text uses the correct honorific level.
    - **Design Motivation**: The most challenging task—models must simultaneously understand role relationships, honorific rules, and maintain conversational coherence.

## Key Experimental Results

### Main Results (Task 1: Honorific Classification)

| Model | Accuracy | F1 |
|------|--------|-----|
| Dictionary-Based | 88.37 | 88.64 |
| LSTM | 93.47 | 91.34 |
| Javanese BERT (Fine-tuned) | 93.91 | 93.97 |
| Javanese DistilBERT (Fine-tuned) | **95.65** | **95.66** |
| GPT-4o (Zero-shot) | 53.50 | 40.70 |
| Gemini 1.5 Pro (Zero-shot) | 50.70 | 45.40 |
| Llama 3.1 8B (Zero-shot) | 43.00 | 24.00 |

### Ablation Study (GPT-4o Classification Performance Per Level)

| Honorific Level | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Ngoko | 78.00 | 91.10 | 84.00 |
| Ngoko Alus | 0 | 0 | 0 |
| Krama | 53.50 | 26.00 | 35.00 |
| Krama Alus | 29.90 | 82.40 | 43.80 |

### Key Findings
- Fine-tuned specialized models (DistilBERT at 95.65%) far outperform general LLMs (GPT-4o at 53.5%), illustrating that Javanese honorifics remain a major low-resource challenge.
- GPT-4o completely fails to identify the Ngoko Alus level (F1=0), showing a severe level bias.
- Closed-source models bias towards the two extreme levels (Ngoko and Krama Alus) in classification, while ignoring the intermediate levels.
- The rule-based baseline (88.37%) is already quite strong, because honorifics are heavily realized through lexical substitution.
- In cross-lingual translation, KL divergence and Jensen-Shannon distance indicate a significant vocabulary gap between Javanese and Indonesian.

## Highlights & Insights
- This work is the first to systematically evaluate LLM performance on complex honorific systems, filling a gap in the pragmatic evaluation of low-resource languages.
- The finding that GPT-4o is completely "blind" to Ngoko Alus is a crucial warning—seemingly "multilingual capabilities" are entirely insufficient at a fine-grained cultural level.
- The rigorous corpus construction process (scanning → OCR → two-stage native-speaker validation) provides a blueprint for the digitization of other low-resource languages.

## Limitations & Future Work
- The scale of the corpus is relatively small (4,024 sentences), which might be insufficient to train larger models.
- Only four honorific levels were evaluated, while real-world usage has even more fine-grained distinctions.
- The effect of fine-tuning general LLMs (e.g., fine-tuning Llama with Unggah-Ungguh) was not tested.

## Related Work & Insights
- **vs Japanese Honorific Corpus (Liu & Kobayashi, 2022)**: The Javanese honorific system is more complex (four levels vs the Japanese "Keigo/Kenjougo" binary distinction) and exhibits a lower Yule's K value (105.43 vs 125.54), demonstrating higher lexical diversity.
- **vs Wongso et al. (2021)**: The latter pre-trained Javanese models but did not address the honorific system.
- **vs Marreddy et al. (2022)**: The latter noted that low-resource language models perform poorly due to the lack of labeled data; this work directly addresses this pain point by constructing a specialized corpus.

## Rating
- Novelty: ⭐⭐⭐⭐ First evaluation benchmark for Javanese honorifics, with a clear and unique problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of four tasks with comparisons across multiple model types, but the corpus size remains limited.
- Writing Quality: ⭐⭐⭐⭐ Informative linguistic background and clearly organized experiments.
- Value: ⭐⭐⭐⭐ Holds significant reference value for research in low-resource NLP and culturally sensitive AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)

</div>

<!-- RELATED:END -->
