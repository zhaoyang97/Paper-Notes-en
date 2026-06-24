---
title: >-
  [Paper Note] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension
description: >-
  [ACL 2025][Multimodal VLM][Multilingual Large Models] This paper proposes the CAPTex benchmark. Through culturally procedural text comprehension tasks (step ordering, multiple-choice questions, and conversational reasoning, etc.) across 7 countries/languages, it systematically reveals the blind spots and limitations of multilingual large models in understanding culturally specific procedural texts.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multilingual Large Models"
  - "Procedural Text"
  - "Cultural Understanding"
  - "Benchmark Evaluation"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: 8367f64b3c11c81c
---

# Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension

**Conference**: ACL 2025  
**arXiv**: [2502.14315](https://arxiv.org/abs/2502.14315)  
**Code**: [HuggingFace](https://huggingface.co/datasets/AmirHossein2002/CAPTex)  
**Area**: Multimodal VLM  
**Keywords**: Multilingual Large Models, Procedural Text, Cultural Understanding, Benchmark Evaluation, Low-Resource Languages

## TL;DR

This paper proposes the CAPTex benchmark. Through culturally procedural text comprehension tasks (step ordering, multiple-choice questions, and conversational reasoning, etc.) across 7 countries/languages, it systematically reveals the blind spots and limitations of multilingual large models in understanding culturally specific procedural texts.

## Background & Motivation

Although multilingual large models (mLLMs) perform excellently in various NLP tasks, their ability to comprehend procedural texts (instructional texts with systematic steps), especially those involving specific cultural content, remains under-explored. The following key challenges exist:

1. **Implicit Knowledge Embedded in Culture**: Procedural texts embed a large amount of culturally specific default knowledge, such as the steps of religious rituals and traditional craft processes. Similar procedures across different cultures vary drastically (e.g., funeral procedures in Iran and Indonesia are completely different).
2. **WEIRD Bias**: Existing LLMs tend to be biased toward Western, Educated, Industrialized, Rich, and Democratic social cultural norms, underrepresenting other cultural contexts.
3. **Challenges in Low-Resource Languages**: In low-resource languages, the comprehension capability of models for procedural texts is even weaker.
4. **Lack of Evaluation Benchmarks**: Prior works are mainly restricted to the recipe domain or monolingual English, lacking a comprehensive cross-cultural and multilingual evaluation benchmark.

Core research question: Can mLLMs correctly comprehend and reason about procedural texts embedded within cultural contexts?

## Method

### Overall Architecture

CAPTex (Culturally-Aware Procedural Texts) is a multi-task evaluation benchmark consisting of three basic components: (1) a collection of procedural texts from 7 cultural regions (10 categories), (2) carefully designed multiple-choice questions (MCQs), and (3) a procedure-based conversational corpus. The evaluation covers 4 task formats.

### Key Designs

1. **Data Construction—Ensuring Cultural Authenticity**: For each language/culture, two native speakers manually wrote procedural texts (use of AI generation tools was strictly prohibited), covering 7 countries (China, India, Indonesia, Iran, Japan, Nigeria, Pakistan) and 10 cultural areas (Cooking, Festivals & Celebrations, Social Etiquette, Crafts, Traditional Clothing, Agricultural Practices, Religious Rituals, Life Milestones, Sports & Games, Nature Practices). Ten procedures were created per country per area, totaling 1,400 procedural texts (including native versions and English translations). Quality control included automated validation and manual cross-review.

2. **Four Evaluation Tasks**: 
   * **Task 1 (Step Ordering)**: Shuffling the step order and requiring models to reconstruct the correct order, evaluated using Spearman's correlation, Levenshtein distance, and Kendall's Tau.
   * **Task 2 (PB-MCQ - Procedure-Based Multiple-Choice Questions)**: Covering positive/negative variations of next/previous step questions, totaling 5,600 questions.
   * **Task 3 (CB-MCQ - Conversation-Based MCQ)**: Multiple-choice questions based on a dialogue framework, mimicking procedural knowledge Q&A in a conversation between two people.
   * **Task 4 (CB-QA - Conversation-Based QA)**: Similar to CB-MCQ but requiring generated answers instead of choices, evaluated using ROUGE-L, BERTScore, and semantic similarity.

3. **Language Resource Diversity Design**: The 7 selected languages cover different resource hierarchies from "Winners" (Chinese, Japanese) to "Hopefuls" (Hausa), ensuring a systematic evaluation of mLLM performance under different resource levels.

### Loss & Training

Ours is an evaluative work and does not involve model training. The evaluation adopts a zero-shot setting, and English prompt templates are used throughout (as research shows that English prompts typically yield the best performance). A total of 31 models were evaluated, including DeepSeek-R1, Gemma-2, Llama-3, Qwen2.5, GPT-4o, O3-mini, etc.

## Key Experimental Results

### Main Results

| Model | Ordering $\rho\uparrow$ | PB-MCQ Acc | CB-MCQ Acc | CB-QA (R-L) |
|------|---------|------------|------------|-------------|
| Random | 0.00 | 0.25 | 0.25 | 0.00 |
| GPT-4o | Best | Best | Best | Best |
| Gemma-2-9b-it | 0.75 | 0.43 | 0.46 | 0.50 |
| Qwen2.5-14B-Instruct | - | Best Open-source | - | - |
| Qwen2.5-7B | - | - | - | Best Open-source |
| Llama-3.1-8B-Instruct | 0.43 | 0.37 | 0.48 | 0.59 |
| DeepSeek-R1-Qwen-14B | 0.43 | 0.42 | 0.54 | 0.55 |
| Mamba | Significantly Lagging | Significantly Lagging | Significantly Lagging | Significantly Lagging |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| English vs. Native Language | English performance consistently outperforms native language | Larger gaps in low-resource languages |
| PB-MCQ vs. CB-MCQ | CB-MCQ is generally higher | Conversational framework aids model reasoning |
| Different Cultural Areas | Certain areas (religious rituals) are harder | Deeper cultural embedding leads to higher difficulty |
| Instruct vs. Base | Instruct is generally better | Exceptions: Gemma-2-2B and Qwen2.5-1.5B |

### Key Findings

* mLLMs perform poorly on culturally procedural texts, with a significant performance drop in low-resource languages.
* Model performance varies widely across different cultural areas; some areas (e.g., religious rituals, traditional crafts) are more challenging.
* MCQ performance under the conversational framework is superior to direct questioning, indicating that conversational context aids model reasoning.
* The Mamba architecture is significantly weaker than the Transformer architecture in procedural text comprehension.
* Sorting, PB-MCQ, and CB-MCQ are highly correlated (Kendall's Tau $0.8$–$0.9$), while the generative task CB-QA shows lower correlation ($0.4$–$0.5$), capturing different dimensions of ability.
* Increasing parameter size generally improves performance within the same model series.
* Kendall correlation analysis across the four tasks indicates that sorting and multiple-choice tasks measure similar abilities, while the generation task measures a different aspect.

## Highlights & Insights

* **Genuine Cultural Diversity**: The dataset was manually written by native speakers from 7 countries, ensuring cultural authenticity, which distinguishes it from machine-generated or translated data.
* **Multidimensional Evaluation Design**: Four task formats cover sorting comprehension, knowledge recall, conversational reasoning, and free-form generation, comprehensively depicting the model's capability in procedural text comprehension.
* **Insights from Conversational Framework**: Models perform better on MCQs in conversational contexts than on direct questions, suggesting that an appropriate conversational framework can activate the model's implicit cultural knowledge.
* **Relationship Between Resource Level and Performance**: The work systematically validates the positive correlation between the volume of language resources and the model's cultural understanding capability.

## Limitations & Future Work

* It only covers 7 countries/cultures, with a heavy bias toward Asia (China, India, Japan, Pakistan), and lacks representation from Latin America, European non-English speaking countries, and Sub-Saharan Africa.
* The number of steps in procedural texts is fixed at 5–10 steps, not involving more complex long-sequence procedures.
* The evaluation only utilizes English prompts, whereas using prompts in different native languages could affect the results.
* The performance under few-shot settings or RAG augmentation remains unexplored.
* The conversational portion was generated by GPT-4o (although manually reviewed), which might introduce stylistic biases.

## Related Work & Insights

* Complementary extension of the taxonomy in IndoCulture (Koto et al., 2024).
* Cross-cultural adaptation of recipes (Cao et al., 2023) is limited to the food domain, whereas this work expands to 10 cultural areas.
* Evaluation of multilingual LLMs deepens from general capabilities to culture-specific reasoning abilities.
* Insight: Cultural procedural knowledge can serve as an important dimension for LLM alignment and knowledge enhancement.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 3 | Benchmark construction work; task design is novel but methodological innovation is limited. |
| Practicality | 4 | Provides an important benchmark for multilingual/multicultural LLM evaluation. |
| Experimental Thoroughness | 4 | Comprehensive evaluation with 31 models, 4 tasks, and 7 languages. |
| Writing Quality | 4 | Clear structure and detailed analysis. |
| Overall Score | 3.5 | A valuable evaluation work that reveals key blind spots in mLLMs' cultural understanding. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](punchbench_mllm_punchline.md)
- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[ACL 2025\] MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](multimm_cultural_metaphor.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)

</div>

<!-- RELATED:END -->
