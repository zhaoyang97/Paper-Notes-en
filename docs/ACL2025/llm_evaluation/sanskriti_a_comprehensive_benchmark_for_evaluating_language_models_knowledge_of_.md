---
title: >-
  [Paper Note] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture
description: >-
  [ACL 2025][LLM Evaluation][Cultural Benchmark] Developed SANSKRITI, a large-scale cultural knowledge benchmark covering all 36 administrative regions of India, 16 cultural attributes, and containing 21,853 multiple-choice questions (MCQs). Zero-shot evaluation across 11 LLMs/SLMs/ILMs reveals significant imbalances in the models' cultural knowledge across geographic regions and attributes.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Cultural Benchmark"
  - "Indian Culture"
  - "Language Model Evaluation"
  - "Multicultural NLP"
  - "Knowledge Assessment"
date: 2026-05-08
content_hash: f463ca75db8d753c
---

# SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture

**Conference**: ACL 2025  
**arXiv**: [2506.15355](https://arxiv.org/abs/2506.15355)  
**Code**: Yes (HuggingFace: 13ari/Sanskriti)  
**Area**: LLM Evaluation  
**Keywords**: Cultural Benchmark, Indian Culture, Language Model Evaluation, Multicultural NLP, Knowledge Assessment

## TL;DR

Developed SANSKRITI, a large-scale cultural knowledge benchmark covering all 36 administrative regions of India, 16 cultural attributes, and containing 21,853 multiple-choice questions (MCQs). Zero-shot evaluation across 11 LLMs/SLMs/ILMs reveals significant imbalances in the models' cultural knowledge across geographic regions and attributes.

## Background & Motivation

**Background**: Language models are highly mature in syntactic and semantic layers, but their practical utility in fields such as education, governance, and healthcare depends heavily on their understanding of the socio-cultural context of users. Existing evaluation benchmarks like TyDi QA and XQUAD primarily focus on multilingual capabilities—measuring whether a model can "read and understand different languages"—but rarely verify if models "understand different cultures."

**Limitations of Prior Work**: India has 28 states and 8 union territories, each carrying distinctly different traditions, festivals, cuisines, dances, and historical narratives. Models trained primarily on global datasets often fail to capture region-specific cultural nuances (such as Kathakali performances in Kerala or Madhubani paintings in Bihar) and frequently make errors in region-specific Q&A. While the DOSA dataset attempted to address this gap, it is small and does not cover all states and union territories.

**Key Challenge**: The lack of a systematic, large-scale, comprehensive cultural knowledge benchmark to quantitatively identify where models have knowledge blind spots across different regions and cultural dimensions. Without such a benchmark, it is impossible to improve models' cultural understanding in a targeted manner.

**Goal**: (1) How to construct a high-quality benchmark covering all administrative regions and diverse cultural dimensions of India? (2) What is the performance gap in cultural knowledge among language models of different scales and types? (3) Where exactly are the models' cultural knowledge blind spots distributed across regions and attributes?

**Key Insight**: The authors operationalize "cultural understanding" into 16 quantifiable cultural attribute dimensions (rituals, history, tourism, food, dance & music, etc.) and design four question types to probe the models' cultural knowledge from different cognitive angles—thus transforming the abstract concept of "cultural understanding" into an evaluable MCQ benchmark.

**Core Idea**: To systematically evaluate and expose the cultural knowledge blind spots of language models by collaboratively constructing a 21K+ MCQ benchmark covering 36 Indian administrative regions $\times$ 16 cultural attributes using a large pool of annotators.

## Method

### Overall Architecture

The construction of SANSKRITI follows a five-stage process: "multi-source data collection $\rightarrow$ structured processing $\rightarrow$ categorized annotation $\rightarrow$ cross-verification $\rightarrow$ final review." The input consists of raw cultural content collected from five reliable public platforms, and the output is 21,853 standardized four-option MCQs. Each question is associated with three-dimensional labels: "State/Union Territory $\times$ Cultural Attribute $\times$ Question Type." During evaluation, a zero-shot setting is adopted, where the prediction is determined by the option with the highest output probability from the model.

### Key Designs

1. **Multi-Source Structured Data Collection**:

    - **Function**: Scrapes cultural content covering all administrative regions of India from five complementary public platforms to ensure the authority and diversity of the data.
    - **Mechanism**: Five data sources were selected: Wikipedia (verified cross-domain knowledge), Ritiriwaz (regional customs and traditions), Holidify (cultural and geographic associations), Google Arts & Culture (in-depth multimedia culture), and Times of India (contemporary cultural perspectives). Collected data is organized in a three-tier structure of `"State name": "Attribute": "Relevant content"`, enabling the subsequent question generation to accurately align with regional and attribute dimensions.
    - **Design Motivation**: A single data source is prone to bias (e.g., Wikipedia biases toward history, Holidify toward tourism). The cross-coverage of five complementary sources more comprehensively captures the multi-faceted nature of Indian culture, while structured storage allows annotators to systematically traverse all "region $\times$ attribute" combinations.

2. **Design of Four Question Types**:

    - **Function**: Systematically probes the depth and breadth of models' cultural knowledge from four cognitive perspectives.
    - **Mechanism**: Questions are classified into Association Prediction (identifying a cultural entity given its description), Country Prediction (determining the country based on cultural features), General Knowledge (GK) Prediction (factual cultural Q&A), and State Prediction (locating the specific state based on cultural clues). The difficulty of the four question types progresses, with Country Prediction being the easiest (coarse-grained) and State Prediction being the hardest (requiring fine-grained regional knowledge). Each question contains four options with carefully designed distractors.
    - **Design Motivation**: A single question type cannot comprehensively assess cultural understanding. Country Prediction tests the general knowledge of "knowing it belongs to India," Association Prediction tests "knowing what it relates to," GK Prediction tests "knowing specific facts," and State Prediction tests "knowing which region it belongs to." The combination of these four types forms a multi-grain evaluation pyramid.

3. **40-Annotator Grouping and Cross-Verification**:

    - **Function**: Ensures the quality, consistency, and cultural sensitivity of the 21K+ questions.
    - **Mechanism**: 40 annotators with backgrounds in linguistics or Indian culture were divided into 4 sub-teams (10 people per group), with each group focusing on one question type. Upon completion, the outputs of each sub-team were cross-verified by another sub-team, followed by a unified final review. Annotators received standardized training on question standards, cultural attribute definitions, and cultural sensitivity. 75% of them are native speakers of local Indian languages, and 80% have lived in their primary language region for more than 15 years.
    - **Design Motivation**: Dedicating specific teams to specific question types improves annotation efficiency and question quality, while the cross-verification mechanism effectively identifies ambiguities and errors. The cultural backgrounds of the annotators ensure the authenticity and sensitivity of the questions, preventing stereotypes and cultural misinterpretations.

### Evaluation Setup

All models are evaluated in a zero-shot setting. Open-source models utilize FP16 precision with greedy decoding, while closed-source models are accessed via APIs. Accuracy is used as the evaluation metric, calculated by selecting the option with the highest output probability.

## Key Experimental Results

### Main Results: Overall Performance of 11 Models

| Model | Type | Parameters | Average Accuracy |
|------|------|--------|-----------|
| GPT-4o | LLM | Closed-source | **0.87** |
| Llama-3.1-70B-Instruct | LLM | 70B | 0.86 |
| Qwen2.5-72B-Instruct | LLM | 72B | 0.84 |
| Phi-3-medium-4k-Instruct | LLM | 14B | 0.77 |
| Qwen2-1.5B-Instruct | SLM | 1.5B | 0.74 |
| Mistral-7B-Instruct | LLM | 7B | 0.70 |
| Llama-3.2-3B-Instruct | SLM | 3B | 0.52 |
| Gemma-2-2B-Instruct | SLM | 2B | 0.48 |
| Navarasa-2.0 | ILM | - | 0.40 |
| OpenHathi-7B-Instruct | ILM | 7B | 0.32 |
| SmolLM-1.7B-Instruct | SLM | 1.7B | 0.16 |

### Fine-Grained Performance by Question Type

| Question Type | GPT-4o | Llama-3.1-70B | Qwen2.5-72B | Phi-3-medium | Qwen2-1.5B | SmolLM-1.7B |
|---------|--------|---------------|------------|-------------|------------|-------------|
| Association | 0.82 | **0.85** | 0.80 | 0.71 | 0.75 | 0.18 |
| State Prediction | 0.80 | 0.78 | 0.72 | 0.70 | 0.51 | 0.18 |
| GK Prediction | **0.96** | 0.93 | 0.94 | 0.85 | 0.82 | 0.16 |
| Country Prediction | 0.93 | 0.88 | 0.92 | 0.84 | 0.90 | 0.13 |

### Key Findings

- **LLMs lead across the board, but show significant internal gaps**: GPT-4o (0.87) and Llama-3.1-70B (0.86) run neck-and-neck, while Mistral-7B scores only 0.70. This indicates that parameter size is not the sole determinant of cultural knowledge; the coverage of cultural content in training data is more critical.
- **The surprise of SLMs**: Qwen2-1.5B, despite having only 1.5B parameters, achieves an accuracy of 0.74, surpassing the 7B Mistral. This indicates that smaller models can outperform larger ones on domain-specific knowledge. Particularly in Country Prediction, Qwen2-1.5B (0.90) almost matches GPT-4o (0.93).
- **Indic Language Models (ILMs) perform worst**: Navarasa-2.0 (0.40) and OpenHathi (0.32) score significantly lower than general SLMs, signifying that "indic-specific development" does not equate to "rich cultural knowledge," likely due to insufficient training data quality or scale.
- **Obvious geographical bias**: Models perform poorly on northeastern states (Sikkim, Arunachal Pradesh, Tripura), Bihar, and Jharkhand, but show better performance on internationally recognized regions like Delhi and Maharashtra. States with globally famous cities are easier for models to "remember."
- **Imbalance across attribute dimensions**: Religions, medicine, and general knowledge generally yield high accuracy, but attire, cuisine, and arts present a greater challenge, as these attributes rely heavily on region-specific knowledge.
- **Question type difficulty hierarchy**: GK of highest accuracy (coarse-grained facts) $\rightarrow$ Country-level (national judgment) $\rightarrow$ Association (semantic connection) $\rightarrow$ State-level of lowest accuracy (requiring fine-grained regional knowledge), validating the progressive difficulty design.

## Highlights & Insights

- **Scale and Systematization**: With 21,853 Q&A pairs covering all 36 administrative regions and 16 cultural dimensions, it is the largest and most comprehensive benchmark of its kind. This systematic design allows researchers to perform a three-dimensional "region $\times$ attribute $\times$ question type" analysis to pinpoint cultural knowledge blind spots.
- **Participatory Annotation Design**: Drawing on participatory research methods from HCI, a Taboo game variant was used to motivate annotators to provide deeper cultural knowledge rather than generating only superficial factual questions. This methodology can be transferred to the construction of other cultural benchmarks.
- **Insightful Error Analysis of GPT-4o**: The authors analyzed cases where GPT-4o succeeded and failed, finding that the model relies on strong semantic associations between keywords and cultural entities. When cultural clues are vague (e.g., "the primary livelihood of Ladakh") or options overlap semantically (e.g., "Land of Himalayas" vs "Abode of Gods"), the model fails. This points the way toward improving the cultural reasoning capabilities of models.

## Limitations & Future Work

- **English-only Q&A**: India has 22 official languages, and many cultural concepts find more natural expressions in local languages. English-only Q&A may underestimate the models' cultural understanding in respective native languages and ignores code-mixed scenarios.
- **Inherent limitations of MCQs**: The four-option format only tests recognition rather than generation. Models may guess the correct answer through elimination or keyword matching, making it difficult to truly evaluate deep cultural understanding.
- **Limited data in certain regions**: The number of questions for regions like Jammu and Kashmir is limited (around 300 questions), affecting the statistical confidence of the evaluation.
- **Lack of visual dimensions**: Indian culture is rich in visual elements (dance postures, attire patterns, architectural styles). Text-only MCQs cannot evaluate multimodal cultural understanding.
- Future work can construct multilingual and VQA versions to address these limitations.

## Related Work & Insights

- **vs DOSA**: Both focus on evaluating Indian cultural knowledge, but DOSA is smaller in scale and does not cover all administrative regions. SANSKRITI comprehensively surpasses DOSA in scale (21K vs a few thousands) and geographic coverage (36 regions vs partial). However, DOSA focuses more on the deep semantics of social airfacts, whereas SANSKRITI leans toward factual Q&A.
- **vs CVQA**: CVQA is a cross-cultural multilingual visual question-answering benchmark. It covers multiple countries but has limited depth for each culture. SANSKRITI sacrifices visual and multi-country coverage in exchange for extreme depth in a single culture.
- **vs LoFTI**: LoFTI evaluates the accuracy of models on localized Indian facts, focusing on translation and localization scenarios. SANSKRITI focuses more on systematic coverage across cultural dimensions.

## Rating

| Dimension | Score (1-5) | Reason |
|------|-----------|------|
| Novelty | ⭐⭐⭐ | The scale and coverage of the dataset represent the core contributions, but the paradigm of "constructing an MCQ benchmark + running models" is conventional. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | The multi-dimensional analysis over 11 models $\times$ 4 question types $\times$ 36 regions $\times$ 16 attributes is systematic, and the error analysis is insightful. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure and rich data presentation, though the Related Work section could be more compact. |
| Value | ⭐⭐⭐⭐ | Fills an important gap in evaluating Indian cultural knowledge. The dataset is publicly available, directly facilitating subsequent research. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)

</div>

<!-- RELATED:END -->
