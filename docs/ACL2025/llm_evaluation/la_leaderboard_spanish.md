---
title: >-
  [Paper Note] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America
description: >-
  [ACL 2025][LLM Evaluation][Multilingual Evaluation] This work constructs the first open-source LLM leaderboard for Spanish and Latin American languages, integrating 66 datasets covering Spanish, Catalan, Basque, and Galician, to evaluate 50 models and analyze the relationships between training strategies, compute, and performance.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Multilingual Evaluation"
  - "Spanish"
  - "Leaderboard"
  - "Low-Resource Languages"
  - "Community-Driven"
date: 2026-05-08
content_hash: 4efb648d7ee9402f
---

# La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America

**Conference**: ACL 2025  
**arXiv**: [2507.00999](https://arxiv.org/abs/2507.00999)  
**Code**: [https://hf.co/spaces/la-leaderboard/la-leaderboard](https://hf.co/spaces/la-leaderboard/la-leaderboard)  
**Area**: LLM Evaluation  
**Keywords**: Multilingual Evaluation, Spanish, Leaderboard, Low-Resource Languages, Community-Driven  

## TL;DR
This work constructs the first open-source LLM leaderboard for Spanish and Latin American languages, integrating 66 datasets covering Spanish, Catalan, Basque, and Galician, to evaluate 50 models and analyze the relationships between training strategies, compute, and performance.

## Background & Motivation
**Background**: LLM leaderboards (such as HELM and Open LLM Leaderboard) primarily focus on English or a few high-resource languages. Although Spanish has over 600 million speakers, it is usually only included as a translated version in multilingual leaderboards.

**Limitations of Prior Work**: (a) Machine-translated evaluation sets fail to capture linguistic and cultural nuances; (b) there is a rich linguistic diversity in Spain and Latin America (Spanish varieties + Catalan, Basque, Galician + indigenous languages) but no unified evaluation standard exists; (c) existing Spanish leaderboards either feature closed evaluation sets, evaluate only a fixed set of models, or cover merely a single language.

**Key Challenge**: "What cannot be measured cannot be improved" — the lack of a comprehensive leaderboard restricts the development of LLMs serving Spanish-speaking communities.

**Goal**: To build a community-driven, open-source, extensible, multilingual, and multi-task LLM evaluation platform.

**Key Insight**: Community collaboration (13 research groups contributing datasets) paired with high-quality dataset curation (prioritizing native datasets over translations) and resource-efficient evaluation design (fewer few-shot examples).

**Core Idea**: A community-driven, open-source multilingual leaderboard that provides the first systematic evaluation of LLM performance on Spanish and Latin American languages.

## Method

### Overall Architecture
La Leaderboard comprises 66 datasets covering 4 languages (22 for Spanish, 18 for Catalan, 17 for Basque, and 9 for Galician), spanning task types such as commonsense reasoning, NLI, QA, text classification, summarization, mathematical reasoning, linguistic acceptability, and ethics. Anyone can submit open-source models for evaluation, and the results are publicly reproducible.

### Key Designs

1. **Principles of High-Quality Dataset Curation**:

    - Function: To ensure the linguistic and cultural representation of the evaluation datasets.
    - Mechanism: Prioritization schema — native datasets ($55\%$) > human-translated datasets ($38\%$) > human-verified machine translation ($7\%$). All datasets involved at least one native speaker for annotation or verification. Seven datasets were newly created specifically for La Leaderboard.
    - Design Motivation: Machine-translated evaluation sets fail to capture subtle linguistic nuances and cultural characteristics; translation errors also introduce noise into the evaluation results.

2. **Resource-Efficient Few-Shot Configuration**:

    - Function: To reduce the number of few-shot examples in order to lower computational costs and environmental impact.
    - Mechanism: Demonstrating through experiments that using fewer few-shot examples (e.g., 0-shot or 3-shot instead of 5-shot), which is fewer than common settings in the literature, drastically reduces compute requirements with a negligible impact on performance.
    - Design Motivation: To ensure that researchers with limited computational resources can also reproduce the evaluation results.

3. **Multidimensional Analysis Framework**:

    - Function: To analyze model performance across multiple dimensions.
    - Mechanism: Conducting cross-analyses across six dimensions: language, task type, training data strategy, compute budget, model size, and quantization schemes.
    - Design Motivation: A single ranking fails to reveal the underlying reasons for model performance; multidimensional analysis helps the community comprehend the effectiveness of different training strategies.

### Loss & Training
La Leaderboard is an evaluation platform. Multiple-Choice QA (MCQA) tasks are evaluated using LogProbs, while text generation tasks are evaluated using BLEU, ROUGE, and SAS. It is built on the LM Evaluation Harness, featuring an open-source fork that supports custom metrics.

## Key Experimental Results

### Main Results
Most frequent models in the Top-10 (ordered by the number of tasks in which they reached the Top-10):

| Model | Parameters | No. of Top-10 Tasks | Spanish | Catalan | Basque | Galician |
|------|------|------------|---------|-------------|---------|-----------|
| Gemma-2-9B-IT | 9B | 36 | 59.01 | 57.86 | 50.17 | 47.46 |
| Llama-3.1-8B-IT | 8B | 32 | 59.36 | 57.89 | 47.34 | 47.73 |
| Gemma-2-9B | 9B | 31 | 58.01 | 57.29 | 50.04 | 46.36 |
| Qwen2.5-32B-IT (GPTQ-Int4) | 32B | 30 | 61.30 | 56.50 | 47.35 | 47.87 |
| Qwen2.5-14B-IT (GPTQ-Int8) | 14B | 29 | 61.60 | 57.89 | 47.84 | 48.34 |

### Ablation Study (Training Strategy Analysis)

| Training Strategy | Representative Model | Advantages | Disadvantages |
|---------|---------|------|------|
| Large-scale multilingual pre-training | Qwen-2.5, Llama-3.1 | Consistently high performance across languages | Requires high compute power |
| Language-balanced pre-training | Salamandra, EuroLLM | Good performance on low-resource languages | Inferior to general models on high-resource languages |
| Large-scale English + transfer | Gemma-2 | Excellent performance via knowledge transfer | Insufficient linguistic diversity |
| Continual pre-training | Latxa | Peak performance in target languages | Risk of catastrophic forgetting |
| Instruction-tuning only | RigoChat | Improved fluency | Limited improvement on reasoning/QA tasks |

### Key Findings
- **Large-scale multilingual pre-training is most effective**: The earlier and deeper a model is exposed to the target language, the higher its average score; downstream strategies are beneficial supplements but cannot replace it.
- **Quantized large models outperform full-precision small models**: Under the same VRAM constraints, quantized 14B/32B models outperform full-precision 7-9B models.
- **Galician exhibits the poorest performance**: Having the fewest datasets (9) and being extremely low-resource, its performance on QA and reasoning tasks is significantly lower than other languages.
- **Summarization is a common weakness**: Performance across all four languages is poor in summarization tasks, while performance in NLI tasks is the best.
- **Breadth of knowledge compensates for lack of language-specific data**: Gemma-2's intensive training on English and code enables good performance across all languages via knowledge transfer.
- **Energy consumption strongly correlates with performance**: Instruction-tuned versions generally consume less energy (due to base models being more verbose) while delivering better performance.
- **Total carbon emissions for evaluation: $92.09\text{ kg CO}_2$**: Based on 660.87 hours of compute time, averaging 9.25 kWh per model.

## Highlights & Insights
- **Community-driven methodology is highly generalizable**: 13 research groups contributed datasets, demonstrating how to build evaluation infrastructure for low-resource language communities. The paper also shares this methodology as a reference framework for other language communities.
- **Systematic comparison of training strategies**: Provides a comprehensive perspective from pre-training strategies and compute investment to model size, aiding in the understanding of the determinants of model performance.
- **Extensible design**: The framework is designed to be incrementally extensible, with future plans to incorporate indigenous languages of Latin America (e.g., Guarani, Nahuatl, etc.).

## Limitations & Future Work
- **Indigenous languages are not yet included**: Latin America has hundreds of indigenous languages (e.g., Guarani, Quechua), which are currently only planned for future inclusion.
- **Closed-source models are not evaluated**: Commercial closed-source models like GPT-4 and Claude are not included in the comparison.
- **Insufficient evaluation of cultural appropriateness**: Linguistic capability $\neq$ cultural understanding; there is a lack of datasets specifically assessing cultural appropriateness.
- **Uneven dataset coverage**: Galician has only 9 datasets, which limits the comprehensiveness of its evaluation.

## Related Work & Insights
- **vs Open LLM Leaderboard**: The latter is primarily English-centric, with Spanish included only through translation; La Leaderboard is predominantly based on native datasets.
- **vs ODESIA Leaderboard**: ODESIA targets Spanish but has private evaluation sets and evaluates only 10 models; La Leaderboard is fully open-source and evaluates 50 models.
- **vs CLUB**: CLUB only covers Catalan and BERT-like models; La Leaderboard covers all four languages and generative LLMs.

## Rating
- Novelty: ⭐⭐⭐ The methodology is not highly novel, but it fills an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ $50\text{ models} \times 66\text{ datasets} \times \text{multidimensional analysis}$.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described in detail, providing a reference framework for other language communities.
- Value: ⭐⭐⭐⭐ Represents a significant infrastructural contribution to the Spanish NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ICLR 2026\] Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation](../../ICLR2026/llm_evaluation/holistic_agent_leaderboard_the_missing_infrastructure_for_ai_agent_evaluation.md)
- [\[ACL 2025\] Where Are We? Evaluating LLM Performance on African Languages](where_are_we_evaluating_llm_performance_on_african_languages.md)
- [\[ACL 2025\] BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English](besstie_a_benchmark_for_sentiment_and_sarcasm_classification_for_varieties_of_en.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)

</div>

<!-- RELATED:END -->
