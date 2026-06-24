---
title: >-
  [Paper Note] CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
description: >-
  [ACL 2025][LLM Evaluation][Constraint-Following] CFBench is proposed, a large-scale Chinese constraint-following benchmark containing 1,000 finely annotated samples across 200+ real-world scenarios and 50+ NLP tasks. It systematically defines a taxonomy of 10 major categories and over 25 subcategories of constraints. Furthermore, a multi-dimensional evaluation framework is designed, combining Constraint Satisfaction Rate (CSR), Instruction Satisfaction Rate (ISR)…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Constraint-Following"
  - "Instruction-Following"
  - "Chinese Benchmark"
  - "Multi-dimensional Evaluation"
  - "Constraint Taxonomy"
date: 2026-05-08
content_hash: 9ad857ac4f5ed7eb
---

# CFBench: A Comprehensive Constraints-Following Benchmark for LLMs

**Conference**: ACL 2025  
**arXiv**: [2408.01122](https://arxiv.org/abs/2408.01122)  
**Code**: Not publicly available (the paper states it will be released)  
**Authors**: Tao Zhang, Chenglin Zhu, Yanjun Shen, Wenjing Luo, Yan Zhang, Hao Liang, Tao Zhang, Fan Yang, Mingan Lin, Yujing Qiao, Weipeng Chen, Bin Cui, Wentao Zhang, Zenan Zhou
**Institution**: Baichuan Inc., Peking University  
**Area**: LLM Evaluation / Instruction Following  
**Keywords**: Constraint-Following, Instruction-Following, Chinese Benchmark, Multi-dimensional Evaluation, Constraint Taxonomy

## TL;DR

CFBench is proposed, a large-scale Chinese constraint-following benchmark containing 1,000 finely annotated samples across 200+ real-world scenarios and 50+ NLP tasks. It systematically defines a taxonomy of 10 major categories and over 25 subcategories of constraints. Furthermore, a multi-dimensional evaluation framework is designed, combining Constraint Satisfaction Rate (CSR), Instruction Satisfaction Rate (ISR), and Priority Satisfaction Rate (PSR). The benchmark reveals significant room for improvement in constraint-following for current top-tier LLMs.

## Background & Motivation

**Background**: In real-world applications, LLMs must understand and adhere to various constraints (format, word count, style, content, etc.) specified in user instructions. Existing evaluation benchmarks primarily focus on fragmented constraints or limited scenarios.

**Limitations of Prior Work**:
   - IFEval only focuses on 25 types of programmatically verifiable instructions, lacking generalizability.
   - FollowBench increases difficulty by adding more constraints but only covers 5 constraint types and 75 instances, with limited data volume.
   - ComplexBench focuses on constraint combinations but only designs 4 types of constraints.
   - There is a lack of evaluation methods from a multi-dimensional user perspective, leading to inconsistencies between evaluation metrics and user perception.

**Core Problem**:
   - **Q1**: How to construct high-quality, comprehensively covered evaluation data?
   - **Q2**: How to perform fine-grained and accurate evaluations from a user perspective?

## Method

### Overall Construction Process

The construction of CFBench consists of three major stages: constraint system construction $\rightarrow$ dataset assembly $\rightarrow$ multi-dimensional evaluation method design.

### Constraint Taxonomy

By mining, filtering, and clustering millions of real online instructions, over 5,000 atomic constraints were extracted. These were organized by domain experts into **10 major categories and 25+ subcategories** based on taxonomic and statistical principles:

| ID | Constraint Type | Description | Subcategory Examples |
|------|----------|------|----------|
| C1 | Content Constraints | Controls output scope and depth | Vocabulary, elements, semantic constraints |
| C2 | Quantitative Constraints | Length and quantity requirements | Word-level, sentence-level, paragraph-level, passage-level |
| C3 | Style Constraints | Imparts a distinct style and tone | Tone, formality, audience, author style |
| C4 | Format Constraints | Standardized expression | Basic format, custom format, professional scenarios |
| C5 | Language Constraints | Controls internal linguistic features | Pragmatics, syntax, morphology, phonology |
| C6 | Situational Constraints | Guides responses via background parameters | Role, task, complex context |
| C7 | Exemplar Constraints | Leverages patterns from limited samples | In-context constraint learning |
| C8 | Negative Constraints | Narrows the space through indirect exclusion | Exclusionary instructions |
| C9 | Contradictory Constraints | Mutually exclusive conditions | Commonly overlooked in online logs |
| C10 | Rule-based Constraints | Defines logical flows or actions | Conditional logic, flow control |

### Dataset Construction

1. **Data Sources and Filtering**: Initial instructions were collected from real scenarios and NLP tasks. LLMs were utilized to evaluate the types and number of constraints, filter out unreasonable constraints, and balance the distribution of scenarios and types, resulting in 2,000 candidate instructions.
2. **Iterative Refinement**: Professional annotators repeatedly reviewed and revised the data to ensure the rationality of constraints and the quality of gold-standard answers. Each sample consists of high-quality instructions, ideal answers, a checklist for evaluation, constraint types, and priorities.
3. **Final Scale**: 1,000 samples = Easy Set (500) + Hard Set (500)

### Dataset Statistics

| Metric | Easy Set | Hard Set | Full Set |
|------|----------|----------|----------|
| Avg. Instruction Length | 413 | 605 | 509 |
| Avg. Primary Requirements | 1.69 | 1.98 | 1.84 |
| Avg. Constraints | 3.59 | 4.89 | 4.24 |
| Avg. Constraint Types | 2.83 | 3.58 | 3.20 |

### Evaluation Methodology

#### Evaluation Criteria
Complex instructions are decomposed into multiple simple, independent **checkpoints** (checklist), which are annotated with constraint types and priorities. GPT-4o is used to judge each item point-by-point.

#### Three Evaluation Metrics

1. **CSR (Constraint Satisfaction Rate)**: The average proportion of satisfied constraints across all instructions, reflecting performance at the constraint level.
2. **ISR (Instruction Satisfaction Rate)**: The percentage of instructions where all constraints are fully satisfied, reflecting strictness at the instruction level.
3. **PSR (Priority Satisfaction Rate)**: A weighted score incorporating the priorities of primary and secondary requirements.
    - When all primary requirements are satisfied, $\text{score} = 0.5 + 0.5 \times A$ (where $A$ is the average score of secondary requirements).
    - When $\text{score} > 0.8$, $\text{PSR}_i = 1$, otherwise $0$.
    - If any primary requirement is not met, $\text{PSR}_i = 0$.

## Experiments

### Evaluation Setup

More than 50 mainstream models, including both API and open-source models, were evaluated. The maximum inference length was set to 2048, and GPT-4o was employed as the evaluator ($\text{temperature}=0$).

### Main Results

| Model | CSR (Full) | ISR (Full) | PSR (Full) | PSR (Hard) |
|------|-----------|-----------|-----------|-----------|
| DeepSeek-R1 | 0.908 | 0.699 | 0.783 | 0.672 |
| DeepSeek-V3 | 0.890 | 0.648 | 0.740 | 0.616 |
| GPT-4o | 0.886 | 0.653 | 0.735 | 0.582 |
| o1-preview | 0.870 | 0.634 | 0.718 | 0.592 |
| Claude-3.5-Sonnet | 0.871 | 0.626 | 0.723 | 0.564 |
| Qwen2-72B-Instruct | 0.867 | 0.589 | 0.705 | 0.530 |
| GLM-4-0520 | 0.862 | 0.596 | 0.694 | 0.536 |
| Llama-3-8B-Instruct | 0.609 | 0.211 | 0.297 | 0.238 |

**Key Findings**:
- DeepSeek-R1 ranks first across all metrics, achieving a CSR of 0.908.
- Even for the strongest models, the PSR on the Hard Set is only 0.672, indicating that complex constraints remain a major challenge.
- There is a massive gap between CSR and ISR/PSR (e.g., GPT-4o achieves $\text{CSR}=0.886$ vs $\text{ISR}=0.653$), signifying that while models satisfy partial constraints, they struggle to satisfy all of them fully.

### Constraint Type Analysis

- **Contradictory constraints (C9)** pose the greatest challenge for most models.
- Performance is generally poor on fine-grained quantitative constraints such as vocabulary constraints, word counts, and sentence counts.
- Performance is better on document-level quantitative constraints and audience style constraints.
- No single model maintains a lead across all constraint types.

### Domain and Task Analysis

- Models perform poorly in domains like employment and psychology, while technology and recruitment are strengths for most models.
- Among NLP task types, GPT-4o excels in sentence relation tasks, whereas Qwen2-72B is stronger in sequence labeling.

### Influencing Factors Analysis

Four factors exhibit a positive correlation with ISR:
1. Prompt length
2. **Number of constraints** (most significant impact)
3. Number of constraint types
4. Number of primary requirements

For PSR: the number and types of constraints are not perfectly correlated; the number of primary requirements has a greater impact. Users are more sensitive to unsatisfied constraints when there are fewer of them, but more forgiving of secondary constraints when there are many.

### Comparison with Other Benchmarks

| Benchmark | Samples | Types | Systematic | Priority |
|------|--------|--------|--------|--------|
| IFEval | 541 | 4 | ✗ | ✗ |
| FollowBench | 820 | 5 | ✗ | ✗ |
| ComplexBench | 1150 | 4 | ✔ | ✗ |
| **CFBench** | **1000** | **10-25** | **✔** | **✔** |

### Exploration of Improvement Strategies

- **SFT**: Models with instruction tuning show significant improvement (e.g., Qwen series).
- **Model Scale**: Qwen2-72B achieves a 40% relative improvement in PSR compared to Qwen2-7B.
- **Complex Constraint Training**: Replicating the Conifer method and fine-tuning with complex constraint instructions can further boost performance.

## Highlights & Insights

1. **Systematic Constraint Taxonomy**: Proposes the first instruction-constraint framework based on taxonomic and statistical methodologies (10+25), which is far more comprehensive than existing benchmarks.
2. **Requirement Priority Mechanism**: PSR introduces the concept of primary/secondary requirements, which aligns more closely with real-world user tolerance levels for LLM outputs.
3. **The CFBench rankings are not entirely aligned with MMLU/GSM8K rankings**, indicating that constraint-following is an independent ability dimension distinct from general knowledge and mathematical capabilities.
4. **Large-Scale Evaluation**: A comprehensive cross-evaluation of over 50 models provides rich capability profiling.
5. **Identifying "Contradictory Constraints" as a Universal Blind Spot**: Even the most advanced models struggle to gracefully handle mutually exclusive requirements.

## Limitations & Future Work

1. The study mainly focuses on models with strong Chinese capabilities, lacking a broader investigation into English-centric models.
2. The analysis of differences in instruction-following between Chinese and English is not sufficiently deep.
3. The evaluation heavily relies on GPT-4o as the judge, which may introduce evaluation bias.
4. There is a lack of in-depth analysis regarding the constraint-following enhancement mechanism of reasoning models (such as DeepSeek-R1).

## Related Work & Insights

- **Instruction-Following Training**: Alpaca-style SFT $\rightarrow$ complex instructions (Xu et al., 2023) $\rightarrow$ scaling constraint quantity and diversity (Sun et al., 2024a; He et al., 2024a).
- **Constraint-Following Evaluation**: IFEval (verifiable instructions) $\rightarrow$ FollowBench (multi-level constraints) $\rightarrow$ ComplexBench (constraint combinations) $\rightarrow$ CFBench (systematic & prioritized).

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The constraint taxonomy system and PSR priority evaluation are highly novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation of 50+ models, with detailed ablation and impact analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with abundant tables and charts.
- **Value**: ⭐⭐⭐⭐ — Offers clear guidance on optimization paths for constraint-following.
- **Limitations**: The dataset is skewed toward Chinese scenarios, which may limit its direct applicability to multilingual research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ACL 2025\] StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)

</div>

<!-- RELATED:END -->
