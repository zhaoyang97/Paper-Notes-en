---
title: >-
  [Paper Note] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning
description: >-
  [ACL 2025][LLM Alignment][Table Instruction Tuning] Proposes TableDreamer, a two-stage data synthesis framework: Stage 1 synthesizes highly diverse tables and seed instruction data from scratch; Stage 2 explores the input space through weakness-guided iterative data evolution (evolving data along three orthogonal directions, and utilizing an LLM-as-judge to filter out low-performing samples as seeds for the next round). Using only 27K GPT-4o synthetic data…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Table Instruction Tuning"
  - "Synthetic Data"
  - "Weakness-Guided"
  - "Progressive Data Evolution"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: 428abf4be5cf002d
---

# TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning

**Conference**: ACL 2025  
**arXiv**: [2506.08646](https://arxiv.org/abs/2506.08646)  
**Code**: [https://github.com/SpursGoZmy/TableDreamer](https://github.com/SpursGoZmy/TableDreamer)  
**Area**: LLM Alignment / Table Understanding  
**Keywords**: Table Instruction Tuning, Synthetic Data, Weakness-Guided, Progressive Data Evolution, Curriculum Learning

## TL;DR

Proposes TableDreamer, a two-stage data synthesis framework: Stage 1 synthesizes highly diverse tables and seed instruction data from scratch; Stage 2 explores the input space through weakness-guided iterative data evolution (evolving data along three orthogonal directions, and utilizing an LLM-as-judge to filter out low-performing samples as seeds for the next round). Using only 27K GPT-4o synthetic data, it improves the average accuracy of Llama 3.1-8B by 11.62%, outperforming all baseline methods utilizing 80K–100K data.

## Background & Motivation

**Background**: Table understanding is one of the core capabilities of LLMs. The dominant approach is to perform SFT on general LLMs using table instruction-tuning data to generate dedicated Tabular LLMs (such as TableLLM, TableGPT). Early datasets were derived from human annotations or converted from public datasets, while recent trends have shifted towards LLM-synthesized data.

**Limitations of Prior Work**:
   - **Inadequate Data Diversity**: General-purpose methods like Self-Instruct do not consider the uniqueness of structured tables (various structures and formats), resulting in overly simplistic generated tables; methods like TableLLM can only generate instructions based on tables from existing public datasets, unable to synthesize new tables, thereby limiting table diversity.
   - **Low Data Efficiency**: Current methods blindly pursue data volume, but a large portion of the synthesized data can already be correctly answered by the target LLM, yielding low training value. TableLLM uses 80K data and Self-Instruct uses 100K data, yet the performance gains remain limited.

**Key Challenge**: The input space of table understanding is extremely vast (table structures $\times$ table content $\times$ instruction types). How to efficiently sample from this space to identify the data points with the highest learning value for the target model?

**Goal**: (a) How to synthesize highly diverse tables and instructions from scratch without relying on public datasets? (b) How to focus synthetic data on the model's weaknesses to improve data efficiency?

**Key Insight**: Inspired by curriculum learning and Evol-Instruct, the proposed approach first employs an LLM to evaluate the target model's performance on synthetic data, identifies samples that are difficult to answer correctly (weakness data), and then uses these weakness samples as seeds for multi-directional evolution, iteratively generating increasingly targeted training data.

**Core Idea**: Utilize an iterative loop of "identifying weaknesses $\rightarrow$ evolving using weaknesses as seeds $\rightarrow$ re-identifying weaknesses" to allow a small amount of synthetic data to precisely cover the target model's weak spots, thereby achieving highly efficient tabular instruction tuning.

## Method

### Overall Architecture

TableDreamer consists of two stages. **Stage 1 (Table and Seed Data Generation)**: Synthesizes diverse tables from scratch, then generates seed instruction-response pairs based on the tables and 20 pre-defined task descriptions. **Stage 2 (Weakness-Guided Iterative Evolution)**: Performs data evolution along three directions (instruction complication, instruction generalization, table generalization) based on seed data, utilizes LLM-as-Judge to evaluate target LLM responses, filters out poorly performing weakness data as seeds for the next round of evolution, and accumulates them over 2 iterations to form the final training set. The input consists of task descriptions and LLM prompt templates, and the output is a collection of 27K `(instruction, table, response)` triplets.

### Key Designs

1. **Multi-Attribute Controlled Table Generation**:

    - **Function**: Synthesize diverse tables covering various structures, formats, and sizes from scratch.
    - **Mechanism**: First, direct the LLM to generate table titles across diverse topics (e.g., "Technology $\rightarrow$ AI Application $\rightarrow$ Autonomous Driving AI Integration Analysis 2022"). Then, control five key table attributes in the prompt: (1) **Table Type**—randomly choose from flat, horizontal, or **hierarchical tables**; (2) **Table Size**—random row and column counts; (3) **Header Structure**—hierarchical tables can specify multi-level row/column headers (e.g., 3-level column headers + 2-level row headers); (4) **Cell Relationships**—require the LLM to mark dependencies using markdown formulas (e.g., Net Profit = Revenue - Expense), which are later automatically computed via python scripts; (5) **Table Format**—hierarchical tables use HTML (to accurately reflect merged cells), while flat/horizontal tables use Markdown.
    - **Design Motivation**: Existing methods primarily use tables from public datasets (mostly flat tables) with simple architectures. Real-world tables have highly varied structures. Controlling multiple attributes significantly improves table diversity. Post-processing automatically computes formulas and filters incomplete tables.

2. **Seed Instruction Generation across 20+ Task Types**:

    - **Function**: Generate seed instruction-response pairs covering a broad range of task types based on synthesized tables and pre-defined descriptions.
    - **Mechanism**: Collect 20 table understanding tasks from literature (such as table numerical reasoning, structural understanding, fact verification, query answering, etc.), and input the task descriptions along with synthesized tables into the LLM to generate instruction-response pairs. 3,272 seed data points were generated covering 1,541 tables.
    - **Design Motivation**: To avoid generating instructions based only on a few predefined templates like TableLLM, ensuring task diversity from the starting point by utilizing 20+ task types.

3. **Three-direction Data Evolution (Input Space Exploration)**:

    - **Function**: Starting from each seed data point, generate more diverse data along three orthogonal directions.
    - **Instruction Complication**: Based on the original table and instruction, complicate instructions by increasing the number of sub-tasks (requiring multiple tasks in one instruction) or increasing reasoning steps (generating multi-step questions). This exposes the weaknesses of SOTA LLMs in complex composite tasks.
    - **Instruction Generalization**: Prompt the LLM to synthesize new types of instructions beyond the 20 predefined task types (e.g., creative tasks such as analyzing tables to provide recommendations, translating specific columns), while also generating instructions of the same type but with varying phrasing to improve robustness.
    - **Table Generalization**: Perturb existing tables through format conversion, header modifications, row/column shuffling, etc., to generate table variants. This addresses the known issue where LLMs are non-robust to layout and structural perturbations.
    - **Design Motivation**: The three directions expand instruction difficulty, instruction type, and table layout respectively. Joint exploration is necessary to adequately cover the vast input space.

4. **Weakness Data Identification**:

    - **Function**: Filter out samples where the target LLM performs poorly from the massive pool of evolved data.
    - **Mechanism**: For each evolved data point, let the target LLM (e.g., Llama3.1-8B) generate a response, while a stronger LLM (e.g., GPT-4o) generates a reference answer. An LLM-as-Judge is then employed to grade the target LLM's response on a scale of 1–5. **Samples scoring below 3 are flagged as weakness data**. These weakness data points are utilized as seeds for the next round of evolution $\rightarrow$ ensuring that the subsequent evolutionary paths naturally bias toward the model's weak zones.
    - **Design Motivation**: Utilizing all evolved data (34K) without filtering yields 4.41% worse performance than using only weakness data (27K). This proves that a large amount of "easy" data dilutes the training signals. Weakness filtering is the key to improving data efficiency.

### Loss & Training

- GPT-4o is utilized as the data synthesis LLM, and Llama3.1-8B-Instruct is used as the target LLM.
- Iterating 2 rounds starting from 3,272 seed data points, yielding a final set of 27,083 data points (TableDreamer-27K) covering 7,950 tables.
- For a fair comparison, Llama3.1-70B-Instruct was also evaluated as an alternative generator instead of GPT-4o.
- The accumulated weakness data, along with the high-quality LLM reference responses, are used for SFT.

## Key Experimental Results

### Main Results

Evaluated across 10 tabular benchmarks, covering table question answering (TQA), table fact verification (TFV), table-to-text generation (T2T), and OOD generalization (TableGPT):

| Method | Data Size | TABMWP | WTQ | HiTab | AIT-QA | TabMCQ | TabFact | InfoTabs | TableGPT | Avg. |
|------|--------|--------|-----|-------|--------|--------|---------|----------|----------|------|
| Llama3.1-8B-Instruct | - | 53.39 | 36.53 | 11.35 | 43.63 | 75.31 | 53.87 | 48.94 | 21.68 | 49.07 |
| Self-Instruct | 100K | 46.68 | 28.98 | 13.77 | 48.92 | 80.27 | 52.92 | 45.07 | 43.13 | 49.44 |
| Magpie | 100K | 57.11 | 34.66 | 13.89 | 47.16 | 76.96 | 51.21 | 43.83 | 40.59 | 52.23 |
| TableLLM-syn-data | 80K | 46.10 | 42.24 | 13.92 | 39.72 | 25.46 | 29.24 | 31.31 | 23.74 | 38.68 |
| TableGPT2 (Qwen2.5-7B) | 2.36M | 56.35 | 49.35 | 38.26 | 73.97 | 85.71 | 60.42 | 54.87 | 70.25 | 63.80 |
| **TableDreamer (70B)** | **27K** | 60.57 | 42.47 | 17.25 | 56.75 | 82.99 | 57.32 | 49.98 | 33.03 | **56.02** |
| **TableDreamer (GPT-4o)** | **27K** | **64.61** | **54.66** | **22.88** | 53.22 | **84.29** | **63.09** | **57.65** | 46.20 | **60.69** |

### Ablation Study

| Configuration | Data Size | TQA Avg. | TFV Avg. | T2T Avg. | TableGPT | Overall Avg. | Notes |
|------|--------|----------|----------|----------|----------|--------|------|
| Full TableDreamer | 27K | 55.93 | 60.37 | 80.17 | 46.20 | **60.69** | Full Model |
| w/o Flat Tables | 17K | 51.41 | 52.02 | 74.85 | 40.59 | 55.13 | -5.56% |
| w/o Hierarchical Tables | 17K | 49.24 | 52.54 | 76.37 | 46.79 | 55.08 | -5.61% |
| w/o Horizontal Tables | 18K | 54.58 | 51.40 | 78.07 | 45.38 | 57.72 | -2.97% |
| w/o Data Evolution | 3K | 47.71 | 49.50 | 71.28 | 38.68 | 51.88 | **-8.82%**, only utilizes seed data |
| w/o Instruction Generalization | 18K | 52.32 | 51.77 | 78.26 | 40.89 | 56.26 | -4.44% |
| w/o Instruction Complication | 18K | 50.83 | 51.25 | 73.95 | 39.82 | 54.44 | **-6.26%**, largest drop |
| w/o Table Generalization | 19K | 50.20 | 54.29 | 76.19 | 42.35 | 55.43 | -5.26% |
| w/o Weakness Filtering | 34K | 53.12 | 51.72 | 75.82 | 42.12 | 56.28 | -4.41%, **more data but worse performance** |

### Key Findings

- **27K data outperforms all 80K-132K baselines**: Weakness-guided data efficiency is significantly higher than blindly increasing data volume. TableDreamer-27K (60.69%) even approaches the performance of TableGPT2 trained on 2.36M data (63.80%).
- **Weakness filtering is crucial**: Omitting weakness filtering and using all 34K data results in a 4.41% drop compared to using only the 27K filtered weakness data, demonstrating that "quality > quantity".
- **Instruction complication contributes the most**: Removing instruction complication leads to the worst performance drop (-6.26%) among the three evolution directions, indicating that increasing instruction difficulty is the most effective way to expose model weaknesses.
- **Omitting data evolution leads to an 8.82% drop**: This indicates that merely using LLMs to synthesize predefined instruction types (the standard paradigm of existing Tabular LLMs) is far from sufficient.
- **Table diversity is critical**: Removing any table type degrades performance, with hierarchical tables having the biggest impact (-5.61%).
- **More significant gains in few-shot scenarios**: In settings with only 20-200 annotated training samples, incorporating TableDreamer-27K yields an average improvement of 7.5% - 10.8%.
- **Outperforms R1-distilled models**: Llama3.1-8B fine-tuned via TableDreamer outperforms R1-Distill-Llama-8B by 3.07%.
- **Data scale is highly scalable**: Performance continuously scales up from 3K $\rightarrow$ 10K $\rightarrow$ 27K, validating the efficacy of iteratively accumulating valuable weakness data.

## Highlights & Insights

- **The iterative closed-loop of "Identify Weakness $\rightarrow$ Target Evolution with Weakness as Seed"**: The core mechanism integrates curriculum learning with data synthesis—focusing the data generation process on the weak spots of the model rather than sheer data volume. This concept can easily transfer to any domain requiring synthetic training data (e.g., code, math), active as long as the model's performance on generated samples can be evaluated.
- **From-scratch synthesis completely independent of external sources**: Generates diverse tables by controlling 5 table attributes. The entire pipeline is controllable and reproducible, avoiding copyright issues and distribution biases generic to public datasets.
- **Ingenious three-directional orthogonal evolution**: Instruction complication (enhancing difficulty), instruction generalization (extending task types), and table generalization (boosting robust layout representation) do not overlap, systematically exploring different dimensions of the input space.
- **Counter-intuitive finding—more data is not always better**: The "w/o weakness filtering" ablation using 34K data yielded worse results than the 27K weakness subset. This provides a highly compelling case for data quality-driven pruning.

## Limitations & Future Work

- **Caters only to textual format tables**: Visual table understanding is not explored. Though authors suggest html2image translation, it remains unvalidated.
- **Dependency on strong teacher LLMs**: Follows the Strong-to-Weak distillation paradigm. Weakness identification and reference target generation are heavily reliant on GPT-4o, preventing true self-evolution.
- **Lack of synthesis cost analysis**: Multiple API calls to GPT-4o are needed for data generation and grading, but the exact API cost is not detailed in the paper.
- **SFT-only stage**: Generation of preference data for RLHF is left unexplored, though combining with R1-styled reasoning is mentioned as a promising future direction.
- **Limited iteration rounds**: Evaluated only on 2 iteration rounds. Whether more rounds yield continuous improvement or lead to overfitting on weaknesses remains unstudied.
- **Fixed weakness thresholds**: Uses a fixed score <3 threshold to identify weaknesses. The impact of varying thresholds is not explored.

## Related Work & Insights

- **vs TableLLM**: TableLLM synthesizes 80K instructions using GPT-3.5 on top of public datasets. It is restricted to pre-existing tables and limited task types, leading to poor OOD performance (38.68%). TableDreamer synthesizes tables from scratch and introduces weakness-guided evolution, achieving 60.69% with only 27K data.
- **vs Self-Instruct / Magpie**: While general synthesis methods can extend to table tasks (e.g., Magpie 100K achieves 52.23%), they lack dedicated table structural diversity and weakness-guided mining, rendering them less efficient than TableDreamer.
- **vs TableGPT2**: TableGPT2 reaches 63.80% using 2.36M high-quality curated data (including manual refinement). TableDreamer achieves 60.69% using less than 1.1% of its training volume, highlighting extreme data efficiency.
- **vs Evol-Instruct**: TableDreamer's evolution concept is inspired by Evol-Instruct but integrates a weakness-guiding filter alongside table-specific evolutionary dimensions, significantly outperforming generic Evol-Instruct (100K $\rightarrow$ 50.54%).

## Rating

- Novelty: ⭐⭐⭐⭐ The weakness-guided iterative evolution framework is creative, seamlessly integrating curriculum learning principles with synthetic data generation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive evaluation across 10 benchmarks, detailed ablations, few-shot analyses, comparisons to R1, and scaling analyses.
- Writing Quality: ⭐⭐⭐⭐ The framework description is precise, figures/diagrams are intuitive, and the logical chain from problem definition to methodology design is coherent.
- Value: ⭐⭐⭐⭐ Practical and directly applicable to the Tabular LLM domain. The concept of weakness-guided synthesis is highly transferable to other data synthesis domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection](beyond_similarity_a_gradient-based_graph_method_for_instruction_tuning_data_sele.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)

</div>

<!-- RELATED:END -->
