---
title: >-
  [Paper Note] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation
description: >-
  [ICML 2025][LLM Evaluation][mathematical reasoning] By leveraging community content from the Art of Problem Solving (AoPS) forum, this work constructs AoPS-Instruct, a training set of 652K Olympiad-level mathematical QA pairs, and LiveAoPSBench, a timestamped contamination-resistant evaluation set. It reveals that the high performance of LLMs on older datasets may stem from pre-training data leakage rather than genuine reasoning capabilities.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "mathematical reasoning"
  - "data contamination"
  - "Olympiad mathematics"
  - "instruction tuning"
  - "evaluation benchmark"
date: 2026-05-08
content_hash: e63460c9b9f51f4a
---

# Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation

**Conference**: ICML 2025  
**arXiv**: [2501.14275](https://arxiv.org/abs/2501.14275)  
**Code**: [livemathbench.github.io](https://livemathbench.github.io/leaderboard.html)  
**Area**: LLM Evaluation  
**Keywords**: mathematical reasoning, data contamination, Olympiad mathematics, instruction tuning, evaluation benchmark

## TL;DR

By leveraging community content from the Art of Problem Solving (AoPS) forum, this work constructs AoPS-Instruct, a training set of 652K Olympiad-level mathematical QA pairs, and LiveAoPSBench, a timestamped contamination-resistant evaluation set. It reveals that the high performance of LLMs on older datasets may stem from pre-training data leakage rather than genuine reasoning capabilities.

## Background & Motivation

Olympiad-level mathematical reasoning represents one of the most challenging tasks for LLMs, yet it faces two major bottlenecks:

**Scarcity of training data**: Prior SFT datasets (such as GSM8K, MATH, and Orca-Math) primarily cover primary to high school levels, lacking large-scale competition-grade mathematical data. Manually creating Olympiad-level problems and solutions is extremely costly, requiring substantial time even for experts.

**Unreliable evaluation**: Classic benchmarks like GSM8K and MATH are approaching saturation, with SOTA models achieving over 90% accuracy. More critically, public test sets are highly susceptible to contamination from training data—decontamination methods like n-gram matching fail to capture paraphrased duplicate problems.

The AoPS forum contains over 1 million mathematical discussion threads, focusing mainly on competition-level problems (such as AMC, AIME, and IMO), making it a natural source of high-quality mathematical data. However, the forum data is unstructured, containing irrelevant comments and incomplete solutions, making direct application difficult. The main goal of this paper is to **design an automated pipeline to convert unstructured forum content into high-quality training sets and contamination-resistant evaluation benchmarks**.

## Method

### Overall Architecture

The overall system is divided into two parallel pipelines:

- **Training Pipeline** $\rightarrow$ AoPS-Instruct (652K QA pairs, threads prior to December 2023)
- **Evaluation Pipeline** $\rightarrow$ LiveAoPSBench (3,863 samples, threads from January to August 2024)

Both pipelines share the initial data acquisition and problem detection steps but differ in their downstream quality control strategies.

### Key Designs

#### Five-Step Pipeline for Training Set Construction (AoPS-Instruct)

**Step 0 — Raw Forum Data Acquisition**: Collect 1,076,712 discussion threads (topics) from the AoPS forum, each containing a problem description and subsequent replies.

**Step 1 — Mathematical Problem Detection**: Utilize the Qwen 2.5 14B model to classify the initial message of each thread to determine whether it is a math problem, implemented via hand-designed few-shot prompts. This step filters out 598,375 irrelevant threads, retaining 478,337 mathematical problems.

**Step 2 — QA Pair Extraction**: Prompt the Llama 3.1 70B model to identify and extract problems and their corresponding solutions from the subsequent discussions within each thread. A 70B parameter model is selected because this task requires understanding the entire conversational context to judge which replies contain valid solutions.

**Step 3 — Solution Rewriting**: This is the most crucial step of the pipeline. Forum users' solutions are typically brief, often skipping "obvious" reasoning steps (e.g., directly outputting the result of the AM-GM inequality without naming the theorem). Experiments show that directly fine-tuning on these brief solutions **significantly degrades** model performance on standard benchmarks. Therefore, Qwen 2.5 72B is employed to rewrite all solutions into a detailed, step-by-step reasoning format, supplementing intermediate reasoning steps, unifying the format, and placing the final answer in `\boxed{}`.

**Step 4 — Data Decontamination**: Apply a 10-gram exact matching decontamination method to ensure the training set does not overlap with the test sets of popular mathematical benchmarks (such as MATH, GSM8K, etc.).

#### Evaluation Set Construction (LiveAoPSBench)

The core design philosophy of LiveAoPSBench is to **leverage timestamps to achieve contamination resistance**:

1. **Timestamp Sorting**: Use only recent forum threads (January to August 2024), ensuring the data postdates the training cutoff of most current LLMs.
2. **Heuristic Filtering**: Exclude proof-based problems and preserve only questions with concrete numerical or closed-form answers (boxed answers).
3. **Stricter Decontamination**: Apply 8-gram matching (more stringent than the 10-gram matching used for the training set) to further exclude questions that might overlap with any training corpus.
4. **Dual-Model Cross-Validation**: Independently rewrite solutions for each problem using Llama 3.1 70B and Qwen 2.5 72B to obtain the triplet $(A_{\text{qwen}}, A_{\text{llama}}, A_{\text{original}})$. Only QA pairs where the two model solutions yield matching answers are retained. Consistency checks are performed using string matching, numeric matching, and SymPy symbolic equivalence checking.
5. **Continuous Update**: The fully automated pipeline continuously crawls the latest forum data to update the evaluation set, keeping the benchmark perpetually "unseen."

### Loss & Training

- **Fine-Tuning Strategy**: Standard instruction fine-tuning (SFT), with the problem as the instruction and the rewritten solution as the response.
- **Training Epochs**: 3 epochs (ablation studies showed no additional benefit from more epochs).
- **Data Mixing**: Three configurations were explored—using AoPS-Instruct alone, using Numina alone, and mixing both.
- **Template Format**: Native chat templates of respective models were utilized (e.g., Mathstral uses `<s>[INST] question [/INST] solution`).

## Key Experimental Results

### Main Results

The effects of different training datasets were evaluated across 4 models $\times$ 4 benchmarks:

| Model | Fine-tuning Data | LiveAoPS'24 | MATH | OlympiadBench | Omni-Math |
|------|---------|-------------|------|---------------|-----------|
| DeepSeek-Math-7B | No SFT | 11.7 | 47.1 | 14.5 | 12.3 |
| DeepSeek-Math-7B | Numina | 16.3 | 55.5 | 22.7 | 17.0 |
| DeepSeek-Math-7B | **AoPS-Ins** | **19.0** | **58.8** | **24.3** | **17.8** |
| DeepSeek-Math-7B | Numina+AoPS | **19.7** | 58.8 | **25.6** | **18.0** |
| Mathstral-7B | No SFT | 15.4 | 56.3 | 21.2 | 15.9 |
| Mathstral-7B | Numina | 16.6 | 54.6 | 23.4 | 17.1 |
| Mathstral-7B | **AoPS-Ins** | **23.6** | **60.8** | **27.1** | **19.9** |
| Mathstral-7B | Numina+AoPS | **24.9** | 59.6 | **29.6** | **21.1** |
| Llama-3.2-3B | No SFT | 12.0 | 47.4 | 16.1 | 12.9 |
| Llama-3.2-3B | AoPS-Ins | 16.7 | 54.6 | 19.6 | 16.4 |
| Llama-3.2-3B | Numina+AoPS | **17.4** | **55.6** | **22.8** | **17.2** |
| Llama-3.2-1B | No SFT | 5.3 | 28.8 | 4.7 | 7.0 |
| Llama-3.2-1B | AoPS-Ins | 10.0 | 34.7 | 11.1 | 11.0 |
| Llama-3.2-1B | Numina+AoPS | **11.2** | **36.6** | **12.0** | **11.7** |

**Key Conclusion**: AoPS-Instruct outperforms Numina-only fine-tuning across all models and benchmarks; mixing the two datasets yields the best performance.

### Ablation Study

#### Impact of Solution Rewriting

| Configuration | Effect | Description |
|------|------|------|
| Raw forum solutions (no rewriting) | Performance **drops significantly** | Concise solutions degrade chain-of-thought capabilities |
| Llama 3.1 70B rewriting | Significant improvement | But slightly inferior to Qwen on competition-level benchmarks |
| Qwen 2.5 72B rewriting | **Optimal** | Richer details and more concise (less verbose) |

#### Evaluation Set Quality Verification

| Verification Dimension | Result | Description |
|----------|------|------|
| Human annotation accuracy | 92% | 10 graduate students annotated 386 samples (10%), with 5% errors and 3% no answer |
| Correlation with OlympiadBench | Highly correlated | Automatically constructed benchmark maintains consistent quality with manually constructed ones |
| Inter-annotator agreement | High | Each question independently annotated by two people |

#### Relationship between Timestamp and Contamination Rate

| Time Window | 23/01-04 | 23/05-08 | 23/09-12 | 24/01-04 | 24/05-08 |
|----------|----------|----------|----------|----------|----------|
| 10-gram overlap rate | 13.24% | 11.65% | 12.82% | 9.92% | 6.88% |

Over time, the overlap rate with the Numina training set steadily declines, validating the efficacy of timestamp-based splitting.

### Key Findings

1. **Performance Declines Over Time**: All 17 evaluated LLMs performed worse on 2024 problems compared to 2023 problems, with accuracy decreases ranging from 2.4% to 23.6%, indicating that high scores on older benchmarks may stem from data leakage.
2. **Smaller Models Are More Affected by Contamination**: Llama-3.2-1B experienced the most severe performance drop (23.6%), suggesting that smaller models rely more heavily on memorization rather than reasoning.
3. **Math-Specific Models Are More Robust**: The Qwen2.5-Math series saw performance drops of only 4-5%, significantly lower than general-purpose models.
4. **Dataset Complementarity**: The overlap rate between AoPS-Instruct and Numina is below 14.1%, and hybrid training yields additional gains.

## Highlights & Insights

1. **Timestamps as Natural Decontaminators**: This is a simple yet profound insight—as long as the evaluation data is strictly newer than the training data cutoff, contamination can be effectively avoided. This is more reliable than complex n-gram matching or LLM-based detection.
2. **Solution Rewriting is Crucial**: Merely "having data" is insufficient; the presentation format of the data directly influences model capabilities. Concise expert solutions can be counterproductive, whereas detailed step-by-step formats are necessary to enhance chain-of-thought reasoning.
3. **The Value of Community Data is Underestimated**: The AoPS forum generates over 1,000 new math problems every month. This continuously growing data source offers utility for both training and evaluation, making it far more valuable than static, one-off human-annotated datasets.
4. **Pipeline Portability**: This methodology is not restricted to the mathematical domain and can be generalized to other knowledge-intensive forums such as physics or computer science.

## Limitations & Future Work

1. **Lack of Visual Content**: The current pipeline only processes pure-text problems, resulting in insufficient coverage of domains that rely heavily on diagrams, such as geometry.
2. **Inability to Evaluate Proof Problems**: The evaluation set only includes questions with explicit numerical or closed-form answers, excluding a large volume of Olympiad problems that require logical reasoning and multi-step proofs.
3. **Inconsistent Quality of Community Content**: The quality of forum solutions is highly variable. Despite filtering, noise may still be introduced.
4. **Bottleneck of the Rewriting Model**: The quality of rewritten solutions is capped by the capability of the rewriting model (Qwen 2.5 72B). Future work may leverage stronger models for improvement.
5. **Unexplored RL/RLHF**: Only SFT was utilized, without combining reinforcement learning techniques (such as DPO or PPO), which may have capped the performance gains.

## Related Work & Insights

- **LiveCodeBench** (Jain et al., 2024): A contamination-resistant benchmark in the coding domain that inspired the timestamp-based splitting approach in this work.
- **Numina** (Li et al., 2024): The most relevant baseline, containing a mix of 190K Olympiad-level QA pairs alongside other SFT datasets, also utilizing GPT-4o for solution rewriting.
- **OpenMathInstruct** (Toshniwal et al., 2024): Features 1.8M QA pairs but is entirely generated by Mixtral and is not Olympiad-level.
- **DeepSeek-R1** (2025): A state-of-the-art reasoning model which is also evaluated on LiveAoPSBench.

**Insights for Future Work**: This pipeline can be applied to online communities in other domains (such as Stack Overflow or Physics Forums) to construct continuously updated training and evaluation resources.

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐ | The timestamp-based decontamination concept is not entirely fresh (borrowed from LiveCodeBench), but its systematic implementation in the mathematical domain is commendable. |
| Technical Depth | ⭐⭐⭐ | The pipeline engineering is well-rounded, but the core techniques (LLM filtering + rewriting) are relatively straightforward. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Evaluated on 4 models across multiple benchmarks, featuring ablation studies, human validation, and temporal trend analysis. |
| Value | ⭐⭐⭐⭐⭐ | The dataset and benchmarks are continuously updated, offering direct contributions to the community. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure and rich visualizations. |
| **Overall** | **⭐⭐⭐⭐** | **A solid dataset contribution, with key strengths in contamination-resistant evaluation and open-sourcing high-quality Olympiad data.** |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](../../ACL2026/llm_evaluation/challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ICML 2025\] How Much Can We Forget about Data Contamination?](how_much_can_we_forget_about_data_contamination.md)
- [\[ICML 2025\] PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation.md)
- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](../../ICLR2026/llm_evaluation/detecting_data_contamination_in_llms_via_in-context_learning.md)
- [\[ICML 2025\] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)

</div>

<!-- RELATED:END -->
