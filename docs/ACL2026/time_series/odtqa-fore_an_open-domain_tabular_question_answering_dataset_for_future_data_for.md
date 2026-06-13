---
title: >-
  [Paper Note] ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning
description: >-
  [ACL2026][Time Series][open-domain tabular QA] ODTQA-FoRe proposes an open-domain tabular question answering task focused on future numerical forecasting and post-forecast reasoning. It introduces the TimeFore tri-agent…
tags:
  - "ACL2026"
  - "Time Series"
  - "open-domain tabular QA"
  - "time-series forecasting"
  - "LLM agent"
  - "text-to-SQL"
  - "real estate"
date: 2026-05-08
content_hash: 1e9f9422365d1246
---

# ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning

**Conference**: ACL2026  
**arXiv**: [2606.02433](https://arxiv.org/abs/2606.02433)  
**Code**: https://github.com/jensenw1/ODTQA-FoRe  
**Area**: Time Series / Tabular QA  
**Keywords**: open-domain tabular QA, time-series forecasting, LLM agent, text-to-SQL, real estate

## TL;DR
ODTQA-FoRe proposes an open-domain tabular question answering task focused on future numerical forecasting and post-forecast reasoning. It introduces the TimeFore tri-agent framework to integrate table retrieval, SQL data extraction, specialized time-series forecasting, and answer normalization into an evaluable baseline.

## Background & Motivation

**Background**: LLM + RAG has advanced open-domain QA and tabular QA, with many systems capable of retrieving tables, generating SQL, and performing historical factual or numerical reasoning based on user queries. Datasets like WikiTableQuestions, Spider, Open-WikiTable, NQ-TABLES, and RETQA cover closed-domain/open-domain tabular QA, SQL generation, or multi-table retrieval.

**Limitations of Prior Work**: Most of these tasks answer questions about "existing historical data in the database," rarely addressing future-oriented questions commonly asked by users, such as "What will the price of a certain residential complex be next year?" or "Which projects will have future prices exceeding a threshold?" LLMs themselves are unreliable for time-series forecasting, and in open-domain scenarios, users do not directly provide continuous historical sequences; the system must find tables, extract data, forecast, and سپس reason autonomously.

**Key Challenge**: Traditional ODTQA excels at retrieval and tabular reasoning, while time-series models excel at forecasting, but the two are typically disconnected. Future-oriented QA requires a system to simultaneously possess open-domain historical data acquisition capabilities, external numerical forecasting capabilities, and standardized answering capabilities for diverse question types.

**Goal**: The authors propose the ODTQA-FoRe task and dataset, requiring systems to autonomously locate historical data from a large pool of candidate tables, forecast future prices for 2024, and answer direct forecasting or forecast-based reasoning questions. They also introduce TimeFore as a strong baseline.

**Key Insight**: The paper selects the real estate vertical domain due to its continuous time series and real-world decision-making needs. While the domain is specific, the task format is transferable to finance, retail, climate, or any scenario requiring historical structured data and future forecasting.

**Core Idea**: An LLM agent is utilized for semantic understanding, table retrieval, SQL generation, and final interpretation, while precise numerical forecasting is delegated to specialized time-series models such as TimesNet or TimeXer.

## Method

The paper includes two main threads: the construction of the ODTQA-FoRe dataset and the TimeFore framework. The dataset provides natural language questions, answers, historical data SQL, and future label SQL. TimeFore simulates a real-world system, accessing only the historical database during inference and completing answers step-by-step through three roles: Retriever, Forecaster, and Analyzer.

### Overall Architecture

Data originates from the real estate sales data in RETQA, extended from January 2022 to December 2024, covering 10 Chinese cities. The authors set December 31, 2023, as the reference date: 2022-2023 is historical visible data, and 2024 is the future ground truth used only for evaluation. After filtering, 11,149 projects remain, partitioned 6:2:2 at the project level for training, validation, and testing to prevent project leakage.

The historical database consists of 288 tables aggregated by city, district, and year from 2022-2023 data, averaging 845 rows per table. The future database consists of 2024 data, used only for executing ground-truth SQL. In the QA generation phase, 26 sets of templates were designed, including 7 for direct time-series forecasting and 19 for forecast-based reasoning. The authors manually reviewed 10 samples per set (260 QA pairs total) before large-scale generation.

The TimeFore framework consists of three types of agents. The Retriever summarizes user questions into table caption-style text, matching table captions directly or using BM25 retrieval upon failure, then generates SQL via a few-shot LLM with an execution feedback loop for correction. The Forecaster receives SQL results, converts `[project, year-month, price]` triplets into numerical sequences, uses TimesNet for missing value imputation, and employs TimeXer to forecast 12 months of 2024. The Analyzer first uses a BERT classifier to determine if the question is direct forecasting or forecast-reasoning, selects the corresponding prompt to synthesize historical and predicted data, and finally outputs a standardized answer through a numerical extraction module.

### Key Designs

1. **Future-Oriented Open-Domain Tabular QA Dataset**:
    - **Function**: Extends open-domain tabular QA from historical retrieval to future numerical forecasting and post-forecast reasoning.
    - **Mechanism**: Each QA pair includes a natural language question, answer, historical SQL, and future label SQL. Historical SQL simulates retrievable data, while future SQL evaluates labels.
    - **Design Motivation**: Providing only target tables or historical facts fails to evaluate whether a system can retrieve continuous historical sequences, perform forecasting, and conduct subsequent reasoning.

2. **Two-Stage Retrieval and SQL Self-Correction for Retriever**:
    - **Function**: Locates relevant historical data among 288 candidate tables and generates executable SQL.
    - **Mechanism**: The LLM uses a 5-shot prompt to compress the query into a table caption-style summary; if it fails to hit a caption directly, BM25 finds the most relevant table. SQL generation also uses 5-shot examples, refined through an execution feedback loop of up to 25 iterations to correct syntax or logical errors.
    - **Design Motivation**: Retrieval in open-domain tabular QA determines subsequent forecasting quality. Direct BM25 matching of user questions is unstable; summarizing into caption style reduces the semantic gap.

3. **Division of Labor between Forecaster + Analyzer**:
    - **Function**: Separates task orchestration (LLM strength) from numerical forecasting (specialized model strength).
    - **Mechanism**: The Forecaster calls the `imputationThenPredictionTool`, using TimesNet for imputation and TimeXer for 12-month forecasting. The Analyzer then selects a prompt based on question type and standardizes output using a numerical extraction module.
    - **Design Motivation**: Experiments show general LLMs are significantly weaker than specialized models at time-series forecasting; without standardized output, LLMs tend to provide long explanations instead of evaluable numerical answers.

### Loss & Training

TimeFore itself is not trained end-to-end. A BERT classifier is used for query type classification, and time-series modules are trained using their official optimal hyperparameters. The imputation dataset selects sequences with at least 6 months of history in 2022-2023, containing 8,418 / 2,815 / 2,853 for train / validation / test. The forecasting dataset selects sequences with at least 9 months of history and at least 2 months of 2024 data, containing 5,806 / 1,975 / 1,963 for train / validation / test.

LLM baselines perform forecasting via in-context learning with a temperature of 0.8. LLM inference uses SGLang on a cluster of 20 NVIDIA A800-SXM4-80GB GPUs. BERT and time-series models are trained on a single RTX 4090.

## Key Experimental Results

### Dataset Scale

| Item | Value | Description |
|------|-------|-------------|
| QA pairs | 28,507 | Obtained after removing duplicates, invalid queries, and empty results |
| Train / Val / Test | 16,944 / 5,742 / 5,821 | Split by project to prevent leakage |
| Question Types | 8,042 forecasting + 20,465 reasoning | 7 forecasting templates, 19 reasoning templates |
| Cities and Time | 10 Chinese cities, 2022-2024 | 2022-2023 for history, 2024 for future labels |
| Candidate Historical Tables | 288 tables | Average 845 rows per table |
| Number of Projects | 11,149 refined projects | Filtered from initial 60,183 projects by time coverage |

### Main Results

| Model | Method | Forecast MSE | Forecast MAE | Forecast MRE | Reasoning Acc | Reasoning F1 |
|------|------|--------------|--------------|--------------|---------------|--------------|
| Qwen3 30B | Vanilla | 40,385,720.95 | 3698.36 | 0.1627 | 12.19 | 24.87 |
| Qwen3 30B | TimeFore | 31,572,410.36 | 2788.20 | 0.1326 | 31.59 | 60.25 |
| Qwen3 Next 80B | Vanilla | 30,942,598.21 | 3406.43 | 0.1586 | 24.45 | 46.62 |
| Qwen3 Next 80B | TimeFore | 22,442,845.96 | 2588.87 | 0.1181 | 36.31 | 58.41 |
| GPT OSS 20B | Vanilla | 105,115,117.20 | 4394.56 | 0.1838 | 21.52 | 44.25 |
| GPT OSS 20B | TimeFore | 29,757,828.58 | 2887.44 | 0.1280 | 27.80 | 49.11 |
| GPT OSS 120B | Vanilla | 47,324,931.58 | 3786.48 | 0.1683 | 21.60 | 41.06 |
| GPT OSS 120B | TimeFore | 18,493,634.83 | 2501.18 | 0.1151 | 31.37 | 51.86 |
| GLM4.5 Air | Vanilla | 139,922,250.13 | 3324.41 | 0.1415 | 23.59 | 48.72 |
| GLM4.5 Air | TimeFore | 90,865,440.59 | 2709.25 | 0.1172 | 35.46 | 61.24 |

### Specialized Prediction Model Comparison

| Model | MSE | MAE | MRE | Observation |
|------|-----|-----|-----|------|
| TimesNet | 2.77E+07 | 3103.52 | 0.1254 | Stronger than general LLMs |
| TimeMixer | 2.78E+07 | 3108.29 | 0.1255 | Similar to TimesNet |
| TimeXer | 2.50E+07 | 2989.55 | 0.1209 | Best, used as TimeFore forecaster |
| WPMixer | 2.75E+07 | 3097.81 | 0.1248 | Close to TimesNet |
| AutoTimes | 2.93E+07 | 3204.13 | 0.1288 | Weaker than lightweight specialized models |
| Time-MoE | 2.95E+07 | 3164.47 | 0.1271 | Weaker than TimeXer |
| Qwen3 30B | 6.69E+07 | 4344.02 | 0.1706 | Direct LLM forecasting is significantly worse |
| GLM 4.5 Air | 7.30E+07 | 4824.57 | 0.1869 | LLM forecasting error is the largest |

### Ablation Study

| Module / Setting | Key Numbers | Conclusion |
|-------------|----------|------|
| Table Retrieval Summary+BM25 | GPT OSS 120B F1 99.21, lowest GLM 4.5 Air F1 97.59 | Table retrieval is strong overall and not the primary bottleneck |
| SQL Generation | Qwen3 30B ECR 99.85, Qwen3 Next 80B EA 85.72 | Executability and execution accuracy are high |
| Golden table captions | Max accuracy gain approx. 1.02% | Table location alone is not the main error source |
| Golden history + predicted future | Max accuracy gain approx. 1.93% | SQL / historical data extraction is not the main bottleneck |
| Golden history + golden future | Qwen3 30B +46.80%, Qwen3 Next 80B +49.23%, GLM +44.73% | Future forecasting error is the core bottleneck |
| Analyzer query classification | BERT fine-tuned F1 99.98 | Simple classifier is sufficiently reliable |
| Remove Numerical Extraction | Qwen3 30B Valid Completion Rate drops 33.08% | Standardized output is critical for evaluability |

### Key Findings

- TimeFore outperforms Vanilla across five LLMs, indicating that "direct future numerical forecasting by LLMs" is not a good baseline; forecasting must be delegated to specialized models.
- TimeXer was selected as the forecasting backbone with an MRE of 0.1209, significantly lower than Qwen3 30B's 0.1706 and GLM 4.5 Air's 0.1869.
- The system bottleneck lies not in table retrieval or SQL, but in future data forecasting; this is crucial for the design of subsequent ODTQA-FoRe methods.

## Highlights & Insights

- The task definition is practical: users do not only ask "what is in the database," but also "what will happen in the future." Merging open-domain table QA with forecasting is a natural but previously missing benchmark direction.
- The division of labor in TimeFore aligns with engineering intuition: LLMs handle language understanding and tool orchestration, time-series models handle numerical forecasting, and BERT handles simple type classification. This is more reliable than having a single large model manage all steps.
- The ablation design is diagnostic. Instead of just reporting end-to-end scores, the authors use golden captions, golden history, and golden future to locate error sources, clearly proving that forecasting is the primary bottleneck.
- The dataset construction uses future SQL to generate labels, ensuring answers are objective and unique. This is especially critical for future-oriented questions, as evaluation would otherwise rely on subjective text matching.

## Limitations & Future Work

- **Lack of External Factors**: Current forecasting only uses historical project price sequences, omitting exogenous variables like macroeconomics, policy, environment, and regional supply-demand. Real estate prices are heavily influenced by external factors, limiting the forecasting ceiling.
- **Single Domain**: The dataset covers only real estate. Although TimeFore is claimed to be domain-agnostic, it needs validation in fields with different volatility and periodicity, such as finance, retail, and climate.
- **Limited Template Generation**: Although LLM rewriting is used to improve naturalness and manual verification is performed, the 26 sets of templates may not cover all complex user queries.
- **Unverified Forecast Backbone Generalization**: While TimeXer is optimal for this dataset, its suitability in cross-domain, multi-frequency, or highly non-stationary scenarios requires further experimentation.
- **Unexplored End-to-End Optimization**: The current pipeline is a modular assembly. Future work could attempt to have the Retriever, Forecaster, and Analyzer share uncertainty estimates to output confidence intervals rather than single-point answers.

## Related Work & Insights

- **vs WikiTableQuestions / Spider**: These datasets focus on given tables or SQL generation without open-domain future forecasting; ODTQA-FoRe requires retrieving tables first, then forecasting the future.
- **vs Open-WikiTable / NQ-TABLES / RETQA**: These advance open-domain table retrieval and historical QA but mainly answer archived information; ODTQA-FoRe explicitly introduces 2024 future ground truth.
- **vs LLMTIME / TP-BERTa, etc.**: These methods study time-series forecasting with language models but do not integrate open-domain table retrieval, SQL extraction, and final QA.
- **Insight**: Future agent benchmarks should evaluate the combination of "retrieval-tool call-numerical model-linguistic answer" rather than just single-step text-to-SQL or single-step forecasting.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Novel task combination with clear dataset positioning; TimeFore framework is a logical modular combination rather than a radically new model algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, specialized model comparisons, and ablations (Retrieval/SQL/Analyzer) are comprehensive; lacks cross-domain validation.
- Writing Quality: ⭐⭐⭐⭐☆ Data construction and pipeline are clearly explained with sufficient tabular data; template generation details are mainly in the appendix.
- Value: ⭐⭐⭐⭐⭐ Highly valuable as a benchmark for combining open-domain QA, LLM agents, and time-series forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](../../ICML2026/time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](../../ICLR2026/time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[ACL 2026\] STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)
- [\[AAAI 2026\] Harmonic Dataset Distillation for Time Series Forecasting](../../AAAI2026/time_series/harmonic_dataset_distillation_for_time_series_forecasting.md)
- [\[AAAI 2026\] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](../../AAAI2026/time_series/detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)

</div>

<!-- RELATED:END -->
