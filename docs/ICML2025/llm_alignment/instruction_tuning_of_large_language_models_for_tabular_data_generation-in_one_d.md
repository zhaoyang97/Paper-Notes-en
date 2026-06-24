---
title: >-
  [Paper Note] Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day
description: >-
  [ICML 2025][LLM Alignment][Instruction Tuning] This paper is the first to explore utilizing instruction tuning to enhance the tabular data generation capabilities of LLMs. By constructing a high-quality instruction dataset of only 10K instances and fine-tuning Llama3.1-8B-Instruct on a single A100 for less than 6 hours, the approach achieves tabular data generation performance comparable to GPT-4o.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Instruction Tuning"
  - "Tabular Data Generation"
  - "LLM"
  - "Data-Efficient Fine-Tuning"
  - "Synthetic Data"
date: 2026-05-08
content_hash: ee0370824340a90f
---

# Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day

**Conference**: ICML 2025  
**arXiv**: [2511.23220](https://arxiv.org/abs/2511.23220)  
**Code**: None  
**Area**: LLM Alignment / Tabular Data Generation  
**Keywords**: Instruction Tuning, Tabular Data Generation, LLM, Data-Efficient Fine-Tuning, Synthetic Data

## TL;DR

This paper is the first to explore utilizing instruction tuning to enhance the tabular data generation capabilities of LLMs. By constructing a high-quality instruction dataset of only 10K instances and fine-tuning Llama3.1-8B-Instruct on a single A100 for less than 6 hours, the approach achieves tabular data generation performance comparable to GPT-4o.

## Background & Motivation

**Background**: LLMs exhibit outstanding performance on natural language tasks. However, due to pre-training objectives optimized for text modalities, they perform poorly on tabular data-related tasks. Recently, tabular training through instruction tuning (Tabular Instruction Tuning) has emerged as a promising direction, significantly improving table comprehension by fine-tuning LLMs with table-based natural language instructions.

**Limitations of Prior Work**: Existing tabular instruction tuning works (e.g., TableLLM, TableLlama, TAMA) focus exclusively on question-answering (QA) and reasoning tasks, **completely ignoring the tabular data generation task**. However, generating realistic tabular data is of significant value in data-scarce scenarios (e.g., healthcare, finance) for data augmentation and accelerating ML model training.

**Key Challenge**: An inductive bias mismatch exists between the two-dimensional relational structure of tabular data and the one-dimensional autoregressive training objectives of LLMs. Existing instruction tuning methods require massive datasets and computational resources; for instance, TableLlama utilizes approximately 2 million instructions and 48 A100 GPUs, which is prohibitively expensive for most researchers.

**Goal**: The core question is: Can the tabular data generation capabilities of LLMs be enhanced through instruction tuning using limited data and computational resources? Specifically, the sub-questions include: (a) How to construct a high-quality tabular generation instruction dataset? (b) How to achieve effective fine-tuning under extremely low resource settings?

**Key Insight**: The authors observe that data quality is far more important than quantity. Rather than pursuing a million-scale instruction dataset, focusing on metadata design—attaching detailed general and column-wise descriptions to each table—can better assist the LLM in understanding the tabular context.

**Core Idea**: By leveraging a meticulously designed, metadata-rich, small-scale instruction dataset (10K instances) paired with an efficient fine-tuning strategy, comparable tabular generation performance to GPT-4o is achieved at an extremely low cost.

## Method

### Overall Architecture

The pipeline of ITT-Gen (Instruction Tuning for Tabular data Generation) is divided into two phases:

- **Phase 1: Instruction Dataset Construction** — Samples from 20 public tabular datasets to generate training samples. Each sample contains an instruction $\mathcal{I}$, an input table $\mathcal{T}$ with its metadata $\mathcal{M}$, and the target output table $\mathcal{T}'$.
- **Phase 2: Instruction Tuning** — Conducts supervised fine-tuning on Llama3.1-8B-Instruct using this dataset, enabling the model to generate in-distribution tabular data based on instructions and context.

Formally, the model's objective is to learn the mapping $f_\theta(\mathcal{I}, \mathcal{T}, \mathcal{M}) \rightarrow \mathcal{T}'$, where $\mathcal{T}'$ must maintain the same column structure, marginal distributions, and column-wise relationships as $\mathcal{T}$.

### Key Designs

1. **High-Quality Metadata-Driven Instruction Construction**

    - **Function**: Generates rich metadata for each table, including a general description (topic, structure, application scenario) and detailed column-wise descriptions (column names, data types: numerical/categorical/text).
    - **Mechanism**: GPT-4o is employed to generate tabular metadata, followed by manual review to ensure quality. The authors first manually draft a standard metadata format for a single table, then use it as a template prompt to let GPT-4o generate uniform descriptions for all tables.
    - **Design Motivation**: Preliminary experiments demonstrate that metadata quality is crucial in guiding LLMs to generate correct tabular data. Without metadata, base LLMs fail to comprehend instruction intent and generate completely irrelevant content (e.g., generating irrelevant instruction text instead of tabular data).

2. **Multi-Row Sampled Input-Output Construction**

    - **Function**: In each training instance, $N=20$ rows are randomly sampled as the input table, and another 20 rows are randomly sampled as the target output.
    - **Mechanism**: Unlike previous works that generate row-by-row via next-token prediction, this paper directly supervises the generation with a group of rows as the target output. Experiments prove that this set-of-rows output method outperforms token-by-token generation.
    - **Design Motivation**: Forcing the model to learn the overall distributional characteristics rather than fitting row-by-row helps maintain consistency in column relations.

3. **Multi-Domain Coverage Strategy**

    - **Function**: Samples from 20 public datasets across 10 different topics, covering areas such as consumer analysis, healthcare, finance, employment, real estate, energy, tourism, social media, chemistry, and ML benchmarks.
    - **Mechanism**: 14 datasets are used for training and in-domain evaluation, while 6 are reserved as held-out datasets for out-of-distribution (OoD) evaluation. Each training dataset generates 500 training instructions + 100 evaluation instructions, totaling approximately 7K training + 2K evaluation tasks.
    - **Design Motivation**: Ensuring the diversity of instruction data allows the fine-tuned model to generalize across domains.

### Loss & Training

- **Base Model**: Llama3.1-8B-Instruct, which has already undergone DPO post-tuning to enhance instruction-following capabilities.
- **Fine-Tuning Configuration**: Learning rate of 2e-5, batch size of 3, training for 2 epochs on a single A100 80GB GPU.
- **Acceleration Strategy**: DeepSpeed ZeRO-2 is used for efficient training.
- **Total training time is less than 6 hours**, and 7K instructions are sufficient to significantly boost the model's tabular generation performance.
- The method is independent of the base model; appendix experiments show that fine-tuning on TableLlama is equally effective.

## Key Experimental Results

### Main Results: Fidelity Comparison

Evaluation metrics include Shape (similarity of column-wise marginal distributions) and Trend (correlation capturing capability between columns), with higher values being better.

| Dataset | Base LLM Shape | Base LLM Trend | ITT-Gen Shape | ITT-Gen Trend | GPT-4o Shape | GPT-4o Trend |
|--------|---------------|---------------|--------------|--------------|-------------|-------------|
| adult | 87.48 | 75.13 | 85.73 | 52.54 | 92.34 | 87.96 |
| bank | 75.63 | 65.08 | 85.57 | 86.34 | 93.42 | 91.70 |
| biodeg | 89.59 | 80.04 | 91.68 | 86.61 | 94.12 | 86.54 |
| boston | 88.91 | 87.47 | 92.38 | 88.98 | 90.87 | 93.02 |
| breast_cancer | 55.31 | 37.07 | 84.12 | 69.36 | 78.65 | 64.16 |
| BTC-USD stock | 90.19 | 95.06 | 88.20 | 99.31 | 93.52 | 98.00 |
| california_housing | 88.70 | 90.52 | 73.29 | 80.06 | 96.27 | 97.84 |
| diabetes | 89.45 | 91.02 | 83.41 | 88.77 | 89.93 | 88.11 |
| iris | 82.69 | 55.39 | 88.17 | 77.86 | 89.58 | 87.13 |

Note: Although the metrics for the Base LLM seem high, in reality, ~80% of its outputs consist of irrelevant content; its metrics are calculated based only on the ~20% usable portion, indicating a severe overestimation.

### Utility Comparison: TSTR Framework

Uses the Train-on-Synthetic, Test-on-Real (TSTR) framework to evaluate the downstream task utility of generated data using three ML models (Linear, Random Forest, XGBoost). Classification tasks report AUC, and regression tasks report R².

| Dataset | Real | Base LLM | ITT-Gen | GPT-4o |
|--------|------|----------|---------|--------|
| adult (AUC↑) | 0.8796 | 0.6559 | 0.8265 | 0.8732 |
| bank (AUC↑) | 0.8007 | 0.3534 | 0.6162 | 0.8199 |
| biodeg (AUC↑) | 0.9172 | 0.8161 | 0.8625 | 0.9223 |
| boston (R²↑) | 0.7453 | 0.6774 | 0.6555 | 0.7299 |
| breast_cancer (AUC↑) | 0.9942 | – | 0.9831 | 0.9919 |
| BTC-USD (R²↑) | 0.9955 | 0.9174 | 0.9939 | 0.9909 |
| diabetes (AUC↑) | 0.8204 | 0.8212 | 0.7982 | 0.7973 |
| iris (AUC↑) | 1.0000 | – | 0.9871 | 0.9971 |

### Key Findings

- **Marginal gap between ITT-Gen and GPT-4o**: On most datasets, the gap in Shape/Trend metrics is within 5 percentage points, and on several datasets (e.g., breast_cancer, BTC-USD), the improvement significantly outperforms the Base LLM.
- **Base LLM almost completely fails the task**: When Llama3.1-8B-Instruct is not instruction-tuned, ~80% of the outputs consist of irrelevant text (e.g., generating new instructions rather than tabular data), where "–" indicates completely unusable output.
- **Fine-tuning TableLlama is equally effective**: In the appendix experiments, TableLlama (based on Llama2) produced entirely unusable outputs before fine-tuning (Fidelity was "–" across the board). After fine-tuning, its Shape score reached 60-79, demonstrating the model-agnostic nature of the proposed approach.
- **Highly data-efficient**: Only 7K instructions, 2 epochs, and 6 hours of training are required to elevate the model from "completely unusable" to "close to GPT-4o" performance.

## Highlights & Insights

- **Extreme resource efficiency**: This is likely one of the works with the highest ROI in the tabular AI domain—compared to TableLlama's 2 million instructions + 48×A100, ITT-Gen achieves performance comparable to GPT-4o using only 7K instructions + 1×A100. The key insight is that for task-specific applications, data quality (especially metadata design) is far more important than quantity.
- **Ingenious metadata design**: Generating standardized table descriptions using GPT-4o followed by manual quality review ensures both high quality and cost-effectiveness. This paradigm of "strong model-assisted data labeling $\rightarrow$ small model distillation" is highly transferable.
- **Set-of-rows vs. next-token**: Directly supervising the output of an entire group of rows rather than token-by-token prediction allows the model to better capture structural constraints (inter-column relationships). This idea can be extended to other structured data generation tasks.
- **Cross-domain coverage**: The diverse design across 10 domains ensures model generalization, as demonstrated by its performance on OoD datasets.

## Limitations & Future Work

- **Limited dataset scale**: Only 20 tabular datasets are used, and the ability to generalize to more complex, larger-scale real-world tables (e.g., industrial tables with hundreds of columns) remains unverified.
- **Evaluation limited to conditional generation**: The paper only considers scenarios where in-distribution new rows are generated given an input table, leaving unconditional generation, conditional column generation, and missing value imputation unexplored.
- **GPT-4o remains superior**: In most metrics, ITT-Gen is slightly inferior to GPT-4o, with a noticeable gap on complex datasets (e.g., california_housing, job_posting).
- **Lack of ablation studies**: There is no systematic ablation of key designs such as metadata content, instruction quantity, or the number of sampled rows $N$, making it difficult to quantify the contribution of each component.
- **Limitations in evaluation metrics**: While Fidelity and Utility are classic metrics, there is a lack of privacy protection evaluation (whether generated data memorizes training samples) and diversity metrics.
- **Table serialization methods are undiscussed**: The specific format used for serializing tables into text is not analyzed in detail, though different serialization formats could significantly impact performance.

## Related Work & Insights

- **vs TableLlama**: TableLlama focuses on QA/reasoning and is trained with 2 million instructions + 48 GPUs, but completely ignores tabular generation. ITT-Gen focuses on generation tasks using minimal resources and proves that TableLlama can also benefit by being used as a base model.
- **vs CTAB-GAN/TabDiff**: Traditional generative models (GAN/Diffusion) require separate training for each dataset, whereas ITT-Gen can generate across datasets after a single fine-tuning phase, offering higher flexibility but potentially lower accuracy than specialized models.
- **vs GReaT/Tabula**: These works also use LLMs for tabular generation, but employ a row-by-row auto-completion style without instruction tuning. ITT-Gen introduces instruction formatting + metadata context, leading to more controllable generation quality.
- **Insights**: This method provides a low-cost data augmentation solution for data-scarce domains and can be combined with differential privacy to generate privacy-safe synthetic tabular data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply instruction tuning to tabular data generation, filling an important gap; the idea is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers 20 datasets, two base models, and multiple metrics, but lacks ablation studies and quantitative analyses of key designs.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly structured with well-focused motivations and rich experimental tables, though some details (e.g., serialization formats) could be further elaborated.
- **Value**: ⭐⭐⭐⭐ Highly practical, providing a low-cost tabular synthesis solution for resource-constrained researchers, though application scenarios still require broader validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Federated Data-Efficient Instruction Tuning for Large Language Models](../../ACL2025/llm_alignment/federated_data-efficient_instruction_tuning_for_large_language_models.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](../../ACL2025/llm_alignment/call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](../../NeurIPS2025/llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](../../ACL2025/llm_alignment/measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](../../ACL2025/llm_alignment/tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)

</div>

<!-- RELATED:END -->
