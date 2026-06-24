---
title: >-
  [Paper Note] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual evaluation] Introduces MaXIFE, an evaluation benchmark covering 1,667 verifiable instruction-following tasks across 23 languages. Combining rule-based and model-based evaluation strategies, it systematically assesses the instruction-following capabilities of LLMs in multilingual and cross-lingual scenarios, filling a significant gap in the evaluation landscape.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual evaluation"
  - "cross-lingual instruction following"
  - "evaluation benchmark"
  - "rule-based evaluation"
  - "model-based evaluation"
date: 2026-05-08
content_hash: 745848c46d7c17f5
---

# MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation

**Conference**: ACL 2025  
**arXiv**: [2506.01776](https://arxiv.org/abs/2506.01776)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Multilingual evaluation, cross-lingual instruction following, evaluation benchmark, rule-based evaluation, model-based evaluation

## TL;DR

Introduces MaXIFE, an evaluation benchmark covering 1,667 verifiable instruction-following tasks across 23 languages. Combining rule-based and model-based evaluation strategies, it systematically assesses the instruction-following capabilities of LLMs in multilingual and cross-lingual scenarios, filling a significant gap in the evaluation landscape.

## Background & Motivation

**Background**: Instruction-following capability is a core metric for evaluating the utility of LLMs. Existing evaluation methods like IFEval primarily cover English scenarios and have established a mature evaluation paradigm—objectively assessing whether a model adheres to instructions by setting verifiable constraints (e.g., "response must be under 100 words", "must contain keyword X").

**Limitations of Prior Work**: A major blind spot of current evaluations is the multilingual and cross-lingual scenario. Non-English users constitute the majority globally, yet little is known about the instruction-following capabilities of LLMs in other languages. Different languages present unique grammatical structures, cultural backgrounds, and formatting conventions (e.g., right-to-left in Arabic, word/character segmentation in Chinese, honorific systems in Japanese), which affect instruction comprehension and execution. Existing multilingual benchmarks (such as the multilingual version of MMLU) focus on knowledge capability rather than instruction-following.

**Key Challenge**: Constructing a multilingual instruction-following evaluation faces three technical challenges: (1) designing instruction tasks that are comparable across languages, since "natural" instructions vary greatly; (2) achieving objective verifiability across 23 languages without prohibitive human annotation costs; (3) handling "cross-lingual" scenarios where the instruction is in one language but requires output in another.

**Goal**: To build an instruction-following evaluation benchmark covering a wide range of languages, supporting both intra-lingual and cross-lingual evaluation modes, and providing efficient, objective automatic evaluation tools.

**Key Insight**: Combine rule-based evaluation (precisely verifying formatting constraints) and model-based evaluation (judging semantic constraints) to strike a balance between efficiency and accuracy.

**Core Idea**: Establish a standardized evaluation tool for multilingual instruction-following capabilities using 1,667 carefully designed verifiable instruction tasks, covering 23 languages and various cross-lingual pairings.

## Method

### Overall Architecture

MaXIFE consists of three core components: (1) **Task Construction**: Designing 1,667 instruction tasks across 23 languages, each with instruction text and verifiable constraints; (2) **Evaluation Engine**: A dual-engine automatic scoring system combining rule-based and model-based evaluation; (3) **Analysis Framework**: Analyzing model capabilities across multiple dimensions, including multilingual performance differences, cross-lingual transfer effects, and completion rates of various constraint types.

### Key Designs

1. **Multilingual Instruction Task Design**:

    - **Function**: Provides comparable evaluation data across 23 languages.
    - **Mechanism**: Instruction tasks cover multiple constraint types, including formatting constraints (e.g., "respond with a bulleted list", "under $N$ words"), content constraints (e.g., "must include keyword X", "cannot mention Y"), language constraints (e.g., "respond in French", "include an English summary"), and structural constraints (e.g., "respond in three paragraphs", "end with a question"). Tasks for each language are created or verified by native annotators to ensure naturalness and appropriateness. The 23 languages cover high-resource (English, Chinese, French, German, Japanese, etc.), medium-resource (Vietnamese, Thai, etc.), and low-resource (Swahili, etc.) languages.
    - **Design Motivation**: Simply translating English instructions introduces translationese and cultural mismatch; thus, native speakers are involved in the design to ensure tasks reflect how real users would naturally ask queries.

2. **Dual-Engine Evaluation System**:

    - **Function**: Balances evaluation objectivity and coverage.
    - **Mechanism**: The **Rule-Based Evaluation Engine** handles constraints verified by programmatic checks (e.g., word count, keyword inclusion/exclusion, formatting checks) which are precise and unambiguous. The **Model-Based Evaluation Engine** leverages a powerful LLM judge (e.g., GPT-4) for semantic constraints (e.g., faithfulness to source text, formal tone) that are difficult to quantify. Rule-based evaluation is prioritized, and model-based evaluation is only enabled when rules cannot cover the constraints.
    - **Design Motivation**: Pure rule-based evaluation has limited coverage, while pure model-based evaluation is costly and may introduce bias. The dual-engine design strikes the optimal balance between efficiency and accuracy.

3. **Cross-Lingual Evaluation Design**:

    - **Function**: Evaluates model performance when the instruction language differs from the output language.
    - **Mechanism**: In addition to intra-lingual evaluation (e.g., Chinese instruction $\rightarrow$ Chinese response), MaXIFE includes cross-lingual pairs—where instructions are given in one language but the model is required to respond in another. This simulates real-world scenarios, such as an English speaker asking a model to write an email in Japanese. Cross-lingual evaluation checks not only constraint following but also proper language switching.
    - **Design Motivation**: Cross-lingual instruction following is an essential capability of LLMs but has rarely been systematically evaluated. Many models suffer from "language mixing" (inserting the wrong language) in cross-lingual settings.

### Loss & Training

MaXIFE is an evaluation benchmark and does not involve model training. The evaluation process is: send instructions to the target LLM $\rightarrow$ collect model outputs $\rightarrow$ run the rule-based and model-based evaluation engines $\rightarrow$ output pass/fail decisions for each constraint and the total score.

## Key Experimental Results

### Main Results

| Model | English | Chinese | French | Japanese | Arabic | Low-Resource Avg | Cross-Lingual Avg | Total Score |
|------|------|------|------|------|---------|-----------|-----------|------|
| GPT-4o | 82.5 | 77.3 | 75.8 | 71.2 | 63.5 | 58.2 | 65.3 | 71.8 |
| Claude-3.5 | 80.1 | 74.6 | 73.2 | 68.5 | 60.1 | 55.7 | 62.8 | 69.2 |
| Gemini-1.5 | 78.3 | 72.1 | 70.5 | 66.8 | 58.3 | 53.1 | 60.5 | 67.1 |
| Llama-3-70B | 74.6 | 65.3 | 64.8 | 58.2 | 48.7 | 42.3 | 51.6 | 59.8 |
| Qwen-2-72B | 73.8 | 76.1 | 62.5 | 60.3 | 50.2 | 44.8 | 53.2 | 60.5 |

### Constraint Type Analysis

| Constraint Type | GPT-4o | Claude-3.5 | Llama-3-70B | Overall Pass Rate |
|---------|--------|-----------|------------|-----------|
| Formatting Constraints | 85.3 | 82.7 | 76.1 | 79.2 |
| Content Constraints | 78.6 | 75.2 | 68.3 | 72.1 |
| Language Constraints | 72.1 | 69.8 | 55.4 | 63.8 |
| Structural Constraints | 80.2 | 77.5 | 71.8 | 74.5 |
| Cross-lingual Overall | 65.3 | 62.8 | 51.6 | 57.4 |

### Key Findings
- **Significant Language Gap**: All models perform noticeably worse in non-English languages, with language resource availability being highly positively correlated with performance. Scores in low-resource languages are typically only 55-70% of those in English.
- **Qwen Outperforms GPT-4o on Chinese**: Qwen-2-72B scores 76.1% in Chinese, which is below GPT-4o's 77.3% but reaches a comparable level, demonstrating the efficacy of Chinese-priority training.
- **Cross-lingual Scenarios Present the Greatest Bottleneck**: All models exhibit a significant performance drop in cross-lingual evaluations, averaging 10-15 percentage points lower than intra-lingual scores. Language mixing is the primary error mode.
- **Language Constraints are Harder than Formatting Constraints**: Models more easily satisfy formatting requirements like "using a list format" but underperform on language constraints like "responding in a specific language".
- **High Consistency Between Rule and Model Evaluation**: In scenarios where both methods are applicable, the agreement rate exceeds 90%, validating the reliability of the dual-engine design.

## Highlights & Insights
- **Fills the Gap in Multilingual Instruction-Following Evaluation**: Previous benchmarks did not systematically evaluate instruction-following capabilities across 23 languages. The scale of 1,667 tasks is appropriate for benchmark work. This study provides a vital reference for the development of multilingual LLMs.
- **Practical Dual-Engine Evaluation Design**: Rule-based evaluation handles "hard constraints" (deterministic, programmably verifiable), and model-based evaluation handles "soft constraints" (requiring semantic understanding). This division of labor can be applied to other evaluation scenarios.
- **Cross-lingual Evaluation Dimension**: Existing work rarely addresses real-world scenarios of "instruct in language A, output in language B". The cross-lingual evaluation design of MaXIFE captures a crucial but overlooked dimension of capability.

## Limitations & Future Work
- **Limited Language Coverage**: Although 23 languages are substantial, thousands of languages exist globally, and many true low-resource languages (e.g., from Africa or Pacific islands) are not covered.
- **Low Task Complexity**: Most tasks consist of single or a few combined constraints, lacking instructions that require complex reasoning to follow.
- **Bias in Model-Based Evaluation**: Using GPT-4 as a judge may introduce implicit preference for its own outputs, affecting evaluation fairness.
- **Limitations of Static Benchmarks**: Once the task set is fixed, there is a risk of benchmark overfitting, as it lacks a dynamic updating mechanism.
- Future work can extend to more languages, more complex multi-constraint instructions, and explore methods for dynamically generating evaluation tasks.

## Related Work & Insights
- **vs IFEval**: IFEval is the most direct predecessor of MaXIFE but only covers English. MaXIFE can be seen as a multilingual extension of IFEval, while also adding the cross-lingual evaluation dimension.
- **vs MMLU-multilingual**: MMLU evaluates knowledge rather than instruction-following ability. A model can be knowledgeable but fail to follow formatting constraints, and vice versa.
- **vs xP3/MultiBench**: These multilingual benchmarks focus on generation quality or task completion rather than precise instruction following. MaXIFE offers a more objective evaluation through verifiable constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ Multilingual and cross-lingual instruction following is a novel contribution, though the basic framework originates from IFEval.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple commercial models and 23 languages, providing a rich analysis.
- Writing Quality: ⭐⭐⭐⭐ The benchmark description is clear, and the experimental results are presented in an organized manner.
- Value: ⭐⭐⭐⭐ Offers significant reference value for the development and evaluation of multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)
- [\[ACL 2025\] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)

</div>

<!-- RELATED:END -->
