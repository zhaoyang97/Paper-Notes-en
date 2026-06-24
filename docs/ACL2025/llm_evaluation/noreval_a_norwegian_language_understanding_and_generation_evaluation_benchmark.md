---
title: >-
  [Paper Note] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark
description: >-
  [ACL 2025 (Findings)][LLM Evaluation][Norwegian evaluation] This paper proposes NorEval, a comprehensive Norwegian evaluation suite containing 24 manually created datasets across 9 task categories. It systematically evaluates the language understanding and generation capabilities of 19 open-source Norwegian language models, finding that models still lag significantly behind humans in common-sense reasoning, truthfulness, and instruction following.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Evaluation"
  - "Norwegian evaluation"
  - "language model benchmark"
  - "low-resource languages"
  - "multi-task evaluation"
  - "human baseline"
date: 2026-05-08
content_hash: 73d9fbf1dc1779a9
---

# NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2504.07749](https://arxiv.org/abs/2504.07749)  
**Code**: [https://github.com/ltgoslo/noreval](https://github.com/ltgoslo/noreval)  
**Area**: NLP Understanding / Evaluation Benchmarks  
**Keywords**: Norwegian evaluation, language model benchmark, low-resource languages, multi-task evaluation, human baseline

## TL;DR

This paper proposes NorEval, a comprehensive Norwegian evaluation suite containing 24 manually created datasets across 9 task categories. It systematically evaluates the language understanding and generation capabilities of 19 open-source Norwegian language models, finding that models still lag significantly behind humans in common-sense reasoning, truthfulness, and instruction following.

## Background & Motivation

**Background**: The advancement of language models relies heavily on standardized benchmarks, but evaluation resources for low-resource languages (such as Norwegian) are severely lacking. Existing Norwegian benchmarks, such as NorBench, ScandEval, SEB, and NLEBench, have different focuses but limited coverage.

**Limitations of Prior Work**: Existing benchmarks suffer from four main issues: (1) low task coverage and highly overlapping datasets; (2) NLEBench and ScandEval contain unproofread machine-translated datasets, which introduce evaluation biases; (3) Nynorsk, the minority written standard of Norwegian, is severely neglected, with existing benchmarks almost exclusively covering Bokmål; (4) no benchmarks have established human baselines to measure the upper bounds of model performance.

**Key Challenge**: To comprehensively evaluate the capabilities of Norwegian generative language models, a large-scale benchmark is required that covers both understanding and generation, supports both official written standards (Bokmål and Nynorsk), and is centered on high-quality manual data; however, such resources are currently entirely absent.

**Goal**: To build a comprehensive, high-quality Norwegian evaluation suite that covers a wide range of task categories, establishes human baselines, and provides sufficient evaluation support for Nynorsk.

**Key Insight**: Creating 5 new datasets from scratch, integrating 19 existing peer-reviewed datasets, and drafting over 100 manually written prompts, all integrated into the LM Evaluation Harness framework to ensure reproducible evaluations.

**Core Idea**: Building the largest multi-task benchmark for Norwegian — with 24 datasets covering 9 task categories, focusing on both Bokmål and Nynorsk, and featuring human baselines and LLM-as-a-judge evaluations.

## Method

### Overall Architecture

NorEval is an evaluation suite rather than a model method. Its overall design workflow is: (1) collecting and creating 24 datasets covering 9 task categories; (2) writing 4 to 6 Bokmål and Nynorsk prompts for each dataset; (3) fully integrating them into the LM Evaluation Harness; (4) evaluating 19 models in k-shot ($k \in \{0, 1, 16\}$) settings; (5) establishing human baselines for 5 tasks; and (6) using LLM-as-a-judge to evaluate instruction-following capabilities.

### Key Designs

1. **Dataset Architecture Covering Nine Task Categories**:

    - **Function**: Constructing a comprehensive evaluation system covering Norwegian language understanding and generation.
    - **Mechanism**: Categorizing tasks into 9 high-level types: sentiment analysis, Norwegian linguistic knowledge (grammatical error correction, punctuation, idioms), Norwegian/world knowledge (multiple-choice QA), reading comprehension, common-sense reasoning, machine translation, text summarization, instruction following, and truthfulness. Among the 24 datasets, 16 cover Bokmål and 8 cover Nynorsk. The 5 brand-new datasets include NCB (Punctuation Benchmark), NorIdiom (Idiom Completion), NorRewrite-Instruct, NorSummarize-Instruct, and a series of QA datasets.
    - **Design Motivation**: Existing benchmark task types are single-dimensional and heavily overlap; NorEval aims to fill evaluation gaps in Norwegian linguistic knowledge, truthfulness, and instruction following.

2. **Multi-Prompt + Dual-Written-Standard Evaluation Strategy**:

    - **Function**: Mitigating model sensitivity to specific prompt formulations while simultaneously evaluating both Bokmål and Nynorsk.
    - **Mechanism**: Creating over 100 prompts through a two-stage annotation pipeline. In Stage 1, three native speakers manually translate/write Bokmål prompts. In Stage 2, linguistics students adapt these prompts for Nynorsk. During evaluation, the maximum score across all prompts for each model is selected to alleviate prompt sensitivity.
    - **Design Motivation**: Research shows that the phrasing of prompts significantly affects LM performance; utilizing a multi-prompt strategy and selecting the best result yields more robust evaluations.

3. **Hybrid Performance Aggregation Method**:

    - **Function**: Rationalizing the aggregation of heterogeneous multi-task metrics into a consolidated score.
    - **Mechanism**: Employing three complementary approaches: (1) multi-prompt aggregation (selecting the maximum score); (2) normalized average score (averaging performance after normalizing individual task scores between random baseline and ceiling); and (3) Borda count (a rank-based social choice theory method that assigns scores based on model rankings for each task and sums them), replacing simple arithmetic means to handle heterogeneous evaluation metrics.
    - **Design Motivation**: Traditional average aggregation fails to treat metrics with different scales fairly. The Borda count is more robust as it is based on rankings rather than absolute scores, allowing reciprocal validation between the methods.

### Loss & Training

This paper focuses on evaluation and does not involve model training. The evaluation strategies include: the log-likelihood approach (for classification/multiple-choice tasks, selecting the option with the highest probability) and the generation approach (for generative tasks, using greedy decoding or following Hugging Face's recommended hyperparameters). "LLM-as-a-judge" employs Llama-3.3-70B-Instruct as the evaluator, utilizing the HREF framework combined with human reference answers.

## Key Experimental Results

### Main Results

| Model | Borda Total Score | Sentiment Analysis | Linguistic Knowledge | Knowledge QA | Reading Comprehension | Commonsense Reasoning | Translation | Summarization | Instruction Following | Truthfulness |
|------|-----------|---------|---------|--------|---------|---------|------|------|---------|--------|
| NorMistral-11B | 54.4 | 82.2 | 94.0 | 64.7 | 43.0 | 59.5 | 45.4 | 23.4 | 46.3 | 73.4 |
| AI-Sweden/Llama-3-8B | 51.3 | 80.3 | 84.0 | 54.8 | 51.0 | 47.1 | 34.8 | 31.4 | 38.1 | 71.5 |
| Mistral-Nemo-12B-IT | 52.1 | 82.9 | 33.0 | 58.8 | 16.1 | 67.3 | 44.1 | 42.7 | 55.7 | 43.7 |
| NB-GPT-6B | 33.0 | 34.2 | 42.0 | 29.6 | 30.6 | 7.8 | 27.9 | 33.0 | 39.1 | 55.1 |
| Human Baseline | — | — | 92.0 | — | 90.0 | — | — | — | — | 83.3 |

### Ablation Study — Model vs Human Baseline

| Task | Best Model | Best Model Score | Human Baseline | Gap |
|------|---------|------------|---------|------|
| Belebele (Reading Comprehension) | Mistral-Nemo-12B-IT | 80.2 | 90.0 | -9.8% |
| NorOpenBookQA (World Knowledge) | AI-Sweden/Llama-3-8B-IT | 84.8 | 100.0 | -15.2% |
| NorCommonsenseQA (Commonsense Reasoning) | AI-Sweden/Llama-3-8B-IT | 72.2 | 90.0 | -17.8% |
| NorTruthfulQA MC (Truthfulness) | Mistral-7B | 74.6 | 83.3 | -8.7% |
| NCB (Punctuation Knowledge) | NorwAI-Llama2-7B | 90.0 | 88.0 | +2.0% |

### Key Findings

- **No single model consistently leads in all task categories**. The NorMistral-11B pretrained model is the strongest overall, closely followed by AI-Sweden/Llama-3-8B, but the impact of instruction fine-tuning varies across tasks.
- **Bokmål generally outperforms Nynorsk**: In knowledge-based QA and commonsense reasoning tasks, the models' performance on Bokmål is systematically higher than on Nynorsk, yet Nynorsk performs better on NRK-Quiz-QA and NorIdiom.
- **The double-edged sword of instruction fine-tuning**: Instruction-tuned (IT) variants show prominent gains in multiple-choice QA and sequence generation tasks, but degenerate in Norwegian linguistic knowledge (especially idiom completion) and English-to-Norwegian translation — the NorIdiom score of AI-Sweden/Llama-3-8B-IT dropped precipitously from 31.3 to 0.0.
- **Severe language drift**: In the instruction-following evaluation, only NorMistral-7B-warm-IT consistently replies in Norwegian. For other models (especially Mistral-7B-IT and Meta/Llama-3-8B-IT), up to 60-66% of the responses are in English.

## Highlights & Insights

- **First comprehensive Norwegian evaluation suite**: 24 manually created datasets, 100+ manual prompts, and 5 human baselines serve as a template methodology for evaluating low-resource languages. This end-to-end design from "data to prompts to evaluation framework" can be easily migrated to other low-resource language scenarios.
- **Borda count instead of simple averaging**: Leveraging the rank-based voting method from social choice theory to aggregate heterogeneous evaluation metrics is more robust than arithmetic averaging. This concept is highly referenceable for any evaluation scenario requiring the aggregation of diverse, heterogeneous metrics.
- **Nynorsk degradation due to instruction fine-tuning**: The results reveal an important practical issue where English instruction-tuning data erases the model's capabilities in minority language variants. This serves as a cautionary tale for model development in multi-dialect/multi-variant languages.

## Limitations & Future Work

- **Lack of test data decontamination mechanisms**: The authors acknowledge that the models' pre-training corpora might contain NorEval test data, which could overestimate model performance.
- **Limited human baseline coverage**: Baselines were only established for 5 Bokmål tasks, with only 50 samples per baseline, and no human baseline was provided for Nynorsk.
- **Unverified reliability of LLM-as-a-judge for low-resource languages**: Whether large models trained primarily on English can reliably judge Norwegian generation quality requires further investigation.
- The evaluated model sizes are capped at 7-13B, lacking reference points from larger models (such as GPT-4). Future work could incorporate evaluation results of proprietary LLMs as upper-bound references.

## Related Work & Insights

- **vs NorBench**: NorBench only covers 10 traditional NLP tasks (POS, NER, etc.), whereas NorEval expands to 24 datasets and 9 task categories, particularly adding generative evaluations such as truthfulness and instruction following.
- **vs ScandEval**: ScandEval is a cross-Nordic language benchmark with overlapping datasets but includes machine-translated data. NorEval is almost entirely built upon manually created data, ensuring better quality.
- **vs NLEBench**: NLEBench focuses on Norwegian generation capabilities but entirely neglects Nynorsk, and 7 out of 9 datasets contain unproofread machine translations. NorEval holds advantages in both data quality and language diversity.

## Rating

- **Novelty**: ⭐⭐⭐ While the innovation of evaluation benchmark work is somewhat limited, it makes a significant contribution to the Norwegian NLP community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, consisting of 19 models, 24 datasets, diverse evaluation scenarios, human baselines, and bias analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Structured cleanly, rich in empirical data, and equipped with extremely comprehensive appendices.
- **Value**: ⭐⭐⭐⭐ Successfully fills the gap in Norwegian evaluation, and the proposed methodology can be transferred to other low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] skLEP: A Slovak General Language Understanding Benchmark](sklep_a_slovak_general_language_understanding_benchmark.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
