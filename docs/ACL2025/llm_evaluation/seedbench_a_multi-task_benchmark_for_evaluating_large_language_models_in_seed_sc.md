---
title: >-
  [Paper Note] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science
description: >-
  [ACL2025][LLM Evaluation][benchmark] This paper proposes SeedBench—the first multi-task LLM evaluation benchmark for seed science (seed breeding). It contains 2,264 expert-verified questions covering three major breeding components: gene information retrieval, gene function regulation, and variety selection. Systematically evaluating 26 LLMs, it reveals a significant gap between current LLMs and real breeding requirements.
tags:
  - "ACL2025"
  - "LLM Evaluation"
  - "benchmark"
  - "seed science"
  - "agriculture"
  - "domain-specific"
date: 2026-05-08
content_hash: b086f2fddab58c7b
---

# SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science

**Conference**: ACL2025  
**arXiv**: [2505.13220](https://arxiv.org/abs/2505.13220)  
**Code**: [open-sciencelab/SeedBench](https://github.com/open-sciencelab/SeedBench)  
**Area**: LLM Evaluation  
**Keywords**: benchmark, seed science, agriculture, LLM evaluation, domain-specific

## TL;DR

This paper proposes SeedBench—the first multi-task LLM evaluation benchmark for seed science (seed breeding). It contains 2,264 expert-verified questions covering three major breeding components: gene information retrieval, gene function regulation, and variety selection. Systematically evaluating 26 LLMs, it reveals a significant gap between current LLMs and real breeding requirements.

## Background & Motivation

**Domain Importance**: Seed science is the cornerstone of modern agriculture, directly affecting crop yields and global food security. However, its highly interdisciplinary nature and low financial returns lead to a persistent shortage of breeding experts.

**LLM Potential in Scientific Domains**: LLMs have demonstrated powerful capabilities in fields like finance, healthcare, and education, but their application in seed breeding—an emerging direction of AI for Science—remains highly limited.

**Scarcity of Digital Resources**: Online literature and structured data in seed science are far sparser than in mainstream research fields, restricting the training and evaluation of LLMs.

**Lack of Standardized Benchmarks**: Existing agricultural benchmarks (AgEval for plant stress phenotypes, AgXQA for agricultural extension, CROP for crop knowledge) do not cover the multi-step decision-making process of seed breeding.

**Complex Breeding Workflows**: Real-world breeding involves multi-stage decision-making (gene retrieval → functional analysis → variety selection), requiring the integration of genetic, environmental, and agronomic data, which general benchmarks cannot cover.

**Clear Research Gap**: In the field of LLM evaluation, seed breeding has never had a dedicated benchmark as a critical agricultural task, and there is an urgent need to fill this gap.

## Method

### Overall Architecture

Using the actual workflow of breeding experts as a template, SeedBench categorizes evaluation tasks into three major categories (corresponding to the three breeding steps), ten subcategories, and eleven question types, totaling 2,264 questions. The construction pipeline involves: data collection (308,727 papers → 1.1B token corpus) → expert curation of 279 high-quality text paragraphs → automated question generation by GPT-4 → machine review + human expert double verification.

### Key Design I: Three-tier Breeding Task Classification System

- **Function**: Categorizes questions into three major categories and ten subcategories based on the actual breeding workflow: (1) gene information retrieval (basic information/expression patterns/subcellular localization), (2) gene function and regulation (experimental observations/downstream regulation/functional prediction), and (3) variety selection and agronomic traits (breeding process/agronomic traits/cultivation techniques/recommended suitable planting regions).
- **Mechanism**: Simulates the complete decision chain of breeding experts from "querying genes → analyzing functions → selecting varieties," covering all stages from basic knowledge to application.
- **Design Motivation**: Evaluating only a single dimension (such as Q&A) cannot truly reflect the usability of LLMs in breeding scenarios; a multi-tier, multi-stage evaluation system is required to comprehensively measure the model's integrated capabilities in knowledge retrieval, reasoning, and decision-making.

### Key Design II: 11 Diverse Question Types

- **Function**: Designs 11 question types across Q&A (single-choice/multi-choice/fill-in-the-blank/generation), summarization (simple summarization/key information extraction), and reading comprehension (single-choice/multi-choice/fill-in-the-blank/generation/subcategory classification), evaluated using metrics such as Accuracy, Macro-F1, and ROUGE-L, respectively.
- **Mechanism**: Different question types examine different capability dimensions—choice questions measure knowledge recall, generation tasks measure expressiveness, summarization tasks measure information compression, and reading comprehension tasks measure retrieval and reasoning within long texts.
- **Design Motivation**: The diversity of breeding tasks requires LLMs to simultaneously possess capabilities in precise recall, reasoning analysis, and text generation. A single question type cannot comprehensively evaluate these, whereas a multi-type design ensures comprehensiveness and robustness of the evaluation.

### Key Design III: Dual-Stage Quality Verification Mechanism

- **Function**: Implements a two-stage verification for GPT-4 automatically generated questions—the first stage involves automated checking by GPT-4 for coherence, logical consistency, and task compliance (filtering ~0.01%); the second stage involves manual review by domain experts with PhDs in seed science for relevance and accuracy (filtering ~20%), ultimately retaining 2,264 high-quality questions.
- **Mechanism**: Machine review quickly filters out obvious errors, while human review ensures domain professionalism and scientific rigor.
- **Design Motivation**: Purely automatically generated questions vary in quality, especially in highly specialized breeding domains where GPT-4 may generate questions that seem reasonable but are actually inaccurate; double verification balances efficiency and quality.

## Key Experimental Results

### Main Results: Performance of 26 LLMs on SeedBench

| Model | Type | Parameters | Average Score |
|------|------|--------|--------|
| DeepSeek-V3 | Open-source | 671B | **63.30** |
| GPT-4 | Closed-source | - | 62.06 |
| GLM-4-Plus | Closed-source | - | 59.61 |
| GPT-4o mini | Closed-source | - | 58.40 |
| Qwen2-72B | Open-source | 72B | 57.62 |
| Qwen2.5-14B | Open-source | 14B | 54.21 |
| Claude-3.5-Sonnet | Closed-source | - | 55.45 |
| OpenAI o1-mini | Closed-source | - | 53.25 |
| QwQ-32B | Open-source | 32B | 33.55 |
| Aksara-v1 | Domain Finetuned | 7B | 35.04 |
| PLLaMa-13B | Domain Finetuned | 13B | 17.57 |

### Ablation Study

| Analytical Dimension | Key Findings |
|----------|----------|
| Reasoning Models vs. General Models | Both o1-mini (53.25) and QwQ (33.55) performed worse than GPT-4 (62.06); explicit reasoning chains introduced noise in knowledge-retrieval tasks instead. |
| Domain Finetuning vs. General Models | PLLaMa-7B (16.46) and PLLaMa-13B (17.57) were far behind the general-purpose model Qwen2.5-7B (48.45) of the same parameter scale, showing that finetuning degrades instruction-following capabilities. |
| Impact of Model Scale | The 7B–14B range was the most cost-effective; the Qwen series had almost no gain from 14B to 72B (54.21 → 52.63), suggesting that training data distribution is more important than scale. |
| Subcategory Difficulty | C5 (downstream gene regulation analysis) was the most difficult, with the highest score being only 56.34; C3 (subcellular localization query) was the easiest, with multiple models exceeding 70. |

### Key Findings

1. **DeepSeek-V3 leads all models with a score of 63.30**, outperforming GPT-4's 62.06, but the average score of all models did not break 65, indicating that seed science remains a huge challenge.
2. **Reasoning models perform poorly on breeding tasks**—lengthy reasoning chains introduce useless information in knowledge-retrieval-heavy tasks, dragging down metrics like ROUGE.
3. **Domain-finetuned models fall behind across the board**—the core reason is the degradation of instruction-following and conversation capabilities after finetuning, with PLLaMa even mistaking examples as queries during one-shot evaluations.
4. **Diminishing returns exist for model scale**—7B-14B is the optimal range, larger models are not necessarily better, and training data quality is more critical than model size.

## Highlights & Insights

1. **Pioneering Work**: Fills the gap in LLM benchmarks for the seed breeding domain, serving as a critical infrastructure for AI for Science in agriculture.
2. **Expert-Driven Design**: Involves PhD-level experts with interdisciplinary backgrounds throughout the process to ensure task designs align with actual breeding workflows.
3. **Counter-Intuitive Discovery**: Reasoning-enhanced models (o1, QwQ) performed worse than general-purpose models on seed science tasks, challenging the assumption that "stronger reasoning ability is always better."
4. **Deep Reflection on Domain Finetuning**: Highlights the degradation of general capabilities during finetuning processes, providing important references for future domain-adaptation strategies.

## Limitations & Future Work

1. **Single Data Source**: Primarily relies on peer-reviewed papers without incorporating online databases and expert knowledge bases, leading to limited data diversity.
2. **Language Bias**: Although both Chinese and English papers were collected, the evaluation is predominantly in English, failing to fully account for the impact of cross-lingual differences on model performance.
3. **Lack of Multimodal Evaluation**: Real-world breeding highly relies on images (phenotypes) and sensor data, while the current benchmark only covers textual modalities.
4. **Static Evaluation**: Real-world breeding is a multi-round iterative decision-making process, whereas the current benchmark is in a single-turn Q&A format and cannot evaluate the sequential decision-making capabilities of models.

## Related Work & Insights

| Comparison Dimension | SeedBench | AgEval / AgXQA / CROP |
|----------|-----------|----------------------|
| Domain Coverage | Whole process of seed breeding (gene → function → breeding) | Single aspect (plant stress / agricultural extension / crop knowledge) |
| Task Complexity | 11 question types, 10 subcategories, multi-step decision simulation | Single question type, mostly simple Q&A or classification |
| Quality Assurance | Dual verification (machine + PhD-level experts) | Highly reliant on web data and GPT re-labeling |
| Evaluation Scale | 26 models (closed-source + open-source + domain-finetuned) | Typically evaluates only a small number of models |

**vs. General Scientific Benchmarks (GMAI-MMBench, LawBench)**: The core difference of SeedBench is that its task design is strictly aligned with the actual workflows of domain experts, rather than simply collecting domain-related Q&A pairs; this "workflow-driven" design paradigm can be transferred to other AI for Science fields.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first LLM benchmark in the seed breeding field, filling an important gap, with a creatively designed task classification system.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive and systematic experiments evaluating 26 models across zero-shot and one-shot settings with multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with a clear logical progression from breeding workflow to task design, supported by rich tables and figures.
- **Value**: ⭐⭐⭐⭐ — Holds significant reference value for the AI for Science community, with universal insights revealed regarding domain finetuning and reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models](exposing_numeracy_gaps_a_benchmark_to_evaluate_fundamental_numerical_abilities_i.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ICLR 2026\] CatalystBench: A Comprehensive Multi-Task Benchmark for Advancing Language Models in Catalysis Science](../../ICLR2026/llm_evaluation/catalystbench_a_comprehensive_multi-task_benchmark_for_advancing_language_models.md)

</div>

<!-- RELATED:END -->
