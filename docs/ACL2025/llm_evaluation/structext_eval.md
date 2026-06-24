---
title: >-
  [Paper Note] StrucText-Eval: Evaluating LLM's Reasoning on Structure-Rich Text
description: >-
  [ACL 2025 (Long Paper)][LLM Evaluation][Structured Text Reasoning] This paper proposes StrucText-Eval, which automatically generates semantic-free structured text samples covering 8 structured languages and 29 tasks with a total of 5,800 samples. By adjusting difficulty through controllable nesting depth and width, it reveals that the strongest open-source LLM achieves only 45.8% accuracy on the hard set compared to 92.6% for humans, systematically exposing serious shortcomin…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM Evaluation"
  - "Structured Text Reasoning"
  - "Automated Benchmark Generation"
  - "Complexity-Controlled Evaluation"
  - "Semantic-Free Evaluation"
  - "LLM Capability Boundary"
date: 2026-05-08
content_hash: 1178725e8f0ebe16
---

# StrucText-Eval: Evaluating LLM's Reasoning on Structure-Rich Text

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2406.10621](https://arxiv.org/abs/2406.10621)  
**Code**: [MikeGu721/StrucText-Eval](https://github.com/MikeGu721/StrucText-Eval)  
**Area**: LLM Evaluation  
**Keywords**: Structured Text Reasoning, Automated Benchmark Generation, Complexity-Controlled Evaluation, Semantic-Free Evaluation, LLM Capability Boundary  

## TL;DR
This paper proposes StrucText-Eval, which automatically generates semantic-free structured text samples covering 8 structured languages and 29 tasks with a total of 5,800 samples. By adjusting difficulty through controllable nesting depth and width, it reveals that the strongest open-source LLM achieves only 45.8% accuracy on the hard set compared to 92.6% for humans, systematically exposing serious shortcomings of LLMs in pure structural reasoning.

## Background & Motivation

**Background**: Structured data (e.g., JSON, YAML, XML, Markdown, LaTeX) is ubiquitous in enterprise data scenarios. While LLMs have made significant progress in unstructured text understanding, a natural question arises: can LLMs directly understand and reason over these structured data in pure text formats? Existing studies have explored limited categories such as graphs, tables, and JSON, but their coverage is narrow, and the evaluation systems are incomplete.

**Limitations of Prior Work**: Existing structured text understanding benchmarks suffer from three major limitations. First, coverage is restricted to a few formats like graphs, tables, and JSON, while ignoring widely used structured languages such as LaTeX, Markdown, YAML, Org, and custom trees. Second, most benchmarks rely on manual annotation, which restricts the scalability of evaluation frameworks. Third, static datasets are easily contaminated by being included in the training sets of models, preventing continuous and effective evaluation.

**Key Challenge**: Existing evaluations fail to decouple semantic understanding from pure structural reasoning. When tasks contain meaningful textual content, models can leverage semantic prior knowledge to "take shortcuts," masking their actual structural parsing capability. Furthermore, the lack of a complexity-controlled generation mechanism prevents benchmarks from precisely highlighting capability boundaries under different levels of complexity.

**Goal**: (1) How to design a structured text reasoning evaluation that is wide in coverage, automated, and controllable in difficulty? (2) How weak is the pure structural reasoning ability of LLMs, and how large is the gap compared to humans? (3) What factors determine the difficulty variations across different structured languages and task types?

**Key Insight**: The authors observe that filling the semantic fields of structured data with meaningless strings forces models to reason solely based on the structural syntax markers themselves, thereby purely testing their "structural parsing capability." Concurrently, by parameterizing control over nesting depth, width, and column, evaluation data of varying difficulties can be precisely generated.

**Core Idea**: Constructing a controllable difficulty benchmark covering 8 structured languages $\times$ 29 tasks using semantic-free automated generation methods to purely evaluate the structural reasoning of LLMs.

## Method

### Overall Architecture
The core of StrucText-Eval is an automated evaluation data generation pipeline. The input consists of the task type, target language, and complexity parameters (depth/width/column), and the output is a sample containing four fields: Reference (structured text), Question, Requirement, and Answer (ground truth). The entire process is divided into five steps: defining complexity parameters $\to$ building an abstract structure tree $\to$ applying question templates $\to$ computing ground truth via rule-based algorithms $\to$ translating the abstract tree and answer into the target structured language. Ultimately, this builds two test suites: Test (3,712 samples, depth 1-2) and Test-Hard (2,088 samples, depth/width up to $3 \times 3$, average length of 16,535 characters, and maximum reach of 102,531 characters).

### Key Designs

1. **Semantic-Free Structured Text Generation**:

    - **Function**: Generate evaluation samples that purely test structural parsing capabilities, excluding semantic shortcuts.
    - **Mechanism**: When building the abstract structure tree, all leaf node contents are filled with meaningless random strings instead of real semantic text. The model must reason by understanding structural markers (e.g., curly braces in JSON, indentation in YAML, tag nesting in XML) and cannot rely on "common sense" or "thematic correlation" to guess answers. Complexity is precisely controlled via three parameters: depth controls the nesting levels, width controls the number of child nodes of each non-leaf node, and column controls the number of fields in each node.
    - **Design Motivation**: Existing benchmarks use real semantic data, which allows models to potentially "guess" the answers using semantic priors instead of truly understanding the structure. This semantic-free design is the fundamental difference from all prior works, ensuring that evaluation results reflect pure structural understanding capability.

2. **8 Languages $\times$ 29 Tasks Taxonomy Coverage**:

    - **Function**: Produce systematic coverage of structured languages and tasks, revealing difficulty differences across various formats and task types.
    - **Mechanism**: Categorize structured texts into two main types: structured data (custom Tree format, Tabular/CSV) and semi-structured data (Object Notation: JSON/YAML/XML; Markup Language: Markdown/LaTeX/Org). Based on this, 8 major categories comprising 29 tasks are designed, including PathCompose (hierarchical path reasoning), PathWalk (section extraction), TextRetrieval (information retrieval), Syntax (syntax error detection), Statistic (conditional statistics), Join (SQL-style multi-table join), Tree.Height (tree height calculation), and Node.Depth (node depth calculation), covering a complete difficulty spectrum from simple information localization to complex multi-step reasoning.
    - **Design Motivation**: Prior works only involved 1-2 formats and a few tasks, failing to systematically assess the competency differences across various structural languages. The $8 \times 29$ grid-style coverage enables multi-dimensional comparative analysis.

3. **Dual-Suite Difficulty Gradient Design**:

    - **Function**: Differentiate the capabilities of models across different tiers and quantify the performance gap between LLMs and humans.
    - **Mechanism**: The Test suite contains 3,712 relatively simple samples ($\text{depth} \le 2$, $\text{width} \le 2$, averaging 804 characters) to effectively differentiate low-to-medium tier models. The Test-Hard suite contains 2,088 highly complex samples (depth/width up to $3 \times 3$, averaging 16,535 characters), ensuring that even GPT-4o level models encounter sufficient challenges. Combining the two suites forms a complete evaluation chain ranging from competency screening to limit testing.
    - **Design Motivation**: A single-difficulty benchmark cannot differentiate intermediate and high-level models (e.g., Qwen2-72B already reaches 78.4% on the standard set). A hard version is required to expose performance ceilings.

### Evaluation Strategy
Experiments use 6 prompt designs (Naive, Self-CoT, PS-CoT, w/ Hint, Few-Shot, Simple Few-Shot) and RougeL as the primary metric. The RougeL threshold is set to 0.75; scores below this are directly penalized to 0 to prevent artificially inflated scores from long reasoning paths. Experimental validation shows that the correlation between RougeL and human judgment is 0.9932, which is superior to Exact Match (0.9501) and BLEU.

## Key Experimental Results

### Main Results

| Model | Test-Naive | Test-Hard (Base) | Test-Hard (3-Shot) | Remarks |
|------|-----------|-----------------|-------------------|------|
| Qwen2-72B | 78.4% | 42.5% | 61.4% | One of the strongest open-source models |
| Llama-3.1-70B | 75.4% | 45.8% | 58.4% | PS-CoT yields significant improvement |
| Llama-3.1-405B | 74.9% | 34.4% | 48.7% | Large scale but weaker on Hard set |
| GPT-4o | — | 51.1% | **69.5%** | Strongest closed-source model |
| GLM-4-Plus | — | 47.3% | 65.8% | Close to GPT-4o |
| Mistral-7B | ~30% | 7.0% | 21.0% | Small models are severely deficient |
| **Humans** | — | **92.6%** | — | Far outperforming all models |

### Performance Differences Under Different Languages and Prompt Strategies (RougeL, Test Suite)

| Model | Prompt | JSON | CSV | YAML | Tree | XML | Markdown | Overall |
|------|--------|------|-----|------|------|-----|----------|------|
| Qwen2-72B | Naive | 85.8 | 92.6 | 82.7 | 86.4 | 71.2 | 75.1 | 78.4 |
| Qwen2-72B | PS-CoT | 89.5 | 92.0 | 93.4 | 84.8 | 81.1 | 68.9 | 80.8 |
| Llama-3.1-70B | PS-CoT | 94.5 | 93.7 | 98.5 | 83.2 | 93.9 | 72.7 | 84.2 |
| Llama-3.1-405B | PS-CoT | 84.5 | 92.0 | 94.7 | 86.7 | 94.7 | 76.0 | 74.9 |
| Qwen2-7B | Naive | 70.4 | 83.5 | 68.5 | 68.9 | 57.6 | 68.0 | 30.0 |

### Key Findings
- **JSON Outperforms the Rest**: All models perform best on JSON (Llama-3.1-70B reaches 94.5%), due to JSON's extremely high frequency in internet training data — a classic manifestation of training data bias. Custom Tree and XML perform the worst.
- **Sharp Performance Drop with Increasing Depth/Width**: From $\text{depth}=1$ to $\text{depth}=3$, the accuracy of all models drops drastically, and the performance variance among models only truly widens under high complexity — indicating that standard benchmarks cannot distinguish strong models.
- **Self-CoT / PS-CoT is Harmful to Small Models**: After using Self-CoT, Qwen2-7B plummets from 30.0% to 17.2% because small models cannot correctly generate reasoning paths, introducing more errors instead. However, for large models (e.g., Llama-3.1-70B), PS-CoT boosts performance by +9 percentage points.
- **Few-Shot Performance Rises then Falls with Shot Count**: From 1-shot $\to$ 3-shot, performance improves continuously, but 5-shot shows signs of overfitting, where models begin to overly copy specific patterns from the exemplars instead of performing generalized reasoning.
- **Evaluation Metric Consistency**: The correlation between RougeL and Human Judge is 0.9932, and the correlation between GPT-4o Judge and Human Judge is 0.9937, validating the reliability of automated evaluation.
- **Significant Model Scaling Effects**: Across the same model series, larger parameter sizes yield better performance (Qwen2-7B 29.6% $\to$ 72B 42.5%; Llama-3.1-8B 22.3% $\to$ 70B 45.8% $\to$ 405B 34.4%, where 405B performs worse on the Hard set than 70B, which might relate to prompt sensitivity).

## Highlights & Insights
- **Semantic-Free Design as the Core Innovation**: Replacing real text content with meaningless strings forces models to reason purely from structural syntax. While seemingly simple, this fundamentally shifts the nature of evaluation from "semantic-assisted structural understanding" to "pure structural parsing," unmasking the true liabilities of the models.
- **Dual Advantages of Controllable Complexity + Infinite Generation**: The parameterized design of depth/width/column not only makes difficulty precisely adjustable, but more importantly, allows generating brand new datasets at any time, fundamentally solving the "data leakage" problem where benchmarks are contaminated into training sets.
- **Clear Evidence of Training Data Bias**: The huge performance gap between JSON and custom Trees (up to a 30+ percentage point difference for the same model) directly proves that current LLMs' structural understanding relies heavily on training data distribution, rather than genuinely mastering generalized structural parsing rules.
- **Quantitative Comparison of Humans vs Models**: The stark gap of 92.6% vs 45.8% provides subsequent research with clear room for improvement and target anchors.

## Limitations & Future Work
- **Language and Task Coverage Remain Limited**: Covering only 8 structured languages and 29 tasks, it overlooks key formats like SQL, TOML, Protocol Buffers, and HTML DOM querying. Real-world structured languages and reasoning patterns are far more diverse.
- **Semantic-Free Design is a Double-Edged Sword**: Although it purely evaluates structural parsing capability, interactive reasoning between semantics and structure is typical in real-world scenarios. Designing evaluations for "joint semantic-structural reasoning" is needed in the future.
- **Immediacy of Baseline Models is Lacking**: The evaluated models are from mid-2024, lacking newer ones like Claude, Gemini 2.0, or DeepSeek, restricting the contemporary relevance of findings.
- **Lack of Systematic Fine-Tuning Experiments**: Fine-tuning effectiveness was only demonstrated in a case study (successful performance on the Join task after 5,100 steps of training), but key aspects such as fine-tuning data volume and multi-task joint training are not systematically explored.
- **Evaluation Metrics Can Be Further Refined**: RougeL might be too lenient for exact-match tasks (e.g., Syntax error detection), necessitating customized metrics for different task types.

## Related Work & Insights
- **vs GraphQA / Struc-Bench**: These works focus on single-format understanding of graph structures or tables, relying on semantic content and static datasets. StrucText-Eval comprehensively outperforms them in coverage (8 vs. 1-2 languages), semantic independence, and scalability.
- **vs TableLLM / TEMPTABQA**: These are specialized domain-specific evaluations focusing on tabular operations and temporal reasoning, complementing StrucText-Eval's positioning on general structured reasoning.
- **vs General LLM Benchmarks (e.g., NaturalBench)**: StrucText-Eval fills the gap of "structured text reasoning" in general evaluation frameworks and serves as an important complement to existing benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of semantic-free structural reasoning evaluation is novel, but the methodology for benchmark construction itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models $\times$ 6 prompt strategies $\times$ 8 languages $\times$ 8 task categories $\times$ multiple difficulty levels, with extremely comprehensive analytical dimensions.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured; taxonomy and experimental analysis are orderly, though some analytical paragraphs are relatively wordy.
- Value: ⭐⭐⭐⭐ It reveals apparent shortcomings of LLMs in structural reasoning and provides a reliable evaluation tool, holding direct guidance value for subsequent research.
- **CoT's Effectiveness on Structural Reasoning is Questionable**: In complex structural tasks, Self-CoT actually degrades performance, indicating that structural reasoning might require prompting strategies distinct from semantic reasoning.
- **The Design Paradigm of Complexity-Controlled Benchmarks is Worth Imitating**: Implementing controllable difficulty and leakage prevention via parameterized generation represents a methodology that can be generalized to other evaluation scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of semantic-free structural reasoning evaluation is unique and the automatic generation method is elegant, but the paradigm for benchmark-style work itself is relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ A systematic evaluation of 12 models, 6 prompt strategies, and 8 languages $\times$ 29 tasks, though lacking systematic fine-tuning experiments and more recent models (e.g., Claude, Gemini Ultra).
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the taxonomy and experimental setups are described in detail, but the sheer volume of data in some tables reduces readability.
- Value to Me: ⭐⭐⭐ The evaluation methodology (controllable complexity + leakage-proof generation) is valuable for reference; it exposes LLM structural reasoning weaknesses; however, it offers limited direct inspiration for non-benchmark research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ACL 2026\] Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA](../../ACL2026/llm_evaluation/beyond_case_law_evaluating_structure-aware_retrieval_and_safety_in_statute-centr.md)
- [\[ACL 2025\] Where Are We? Evaluating LLM Performance on African Languages](where_are_we_evaluating_llm_performance_on_african_languages.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)

</div>

<!-- RELATED:END -->
