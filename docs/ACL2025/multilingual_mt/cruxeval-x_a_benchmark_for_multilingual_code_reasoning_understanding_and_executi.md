---
title: >-
  [Paper Note] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution
description: >-
  [ACL 2025][Multilingual & Machine Translation][Code Reasoning] CruxEval-X is proposed, a multilingual code reasoning benchmark covering 19 programming languages. It is extended from the Python version of CruxEval using a fully automated test-guided translation pipeline, containing 12,660 problems and 19K test cases. Evaluation of 24 LLMs reveals correlations among programming languages and the cross-lingual generalization capabilities of monolingually trained models.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Code Reasoning"
  - "Multilingual Benchmark"
  - "Input Reasoning"
  - "Output Reasoning"
  - "Cross-Lingual Generalization"
date: 2026-05-08
content_hash: 35d35f6cd0a6977c
---

# CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution

**Conference**: ACL 2025  
**arXiv**: [2408.13001](https://arxiv.org/abs/2408.13001)  
**Code**: Yes (dataset publicly available)  
**Area**: Code Intelligence / Multilingual Evaluation  
**Keywords**: Code Reasoning, Multilingual Benchmark, Input Reasoning, Output Reasoning, Cross-Lingual Generalization

## TL;DR

CruxEval-X is proposed, a multilingual code reasoning benchmark covering 19 programming languages. It is extended from the Python version of CruxEval using a fully automated test-guided translation pipeline, containing 12,660 problems and 19K test cases. Evaluation of 24 LLMs reveals correlations among programming languages and the cross-lingual generalization capabilities of monolingually trained models.

## Background & Motivation

Existing code benchmarks suffer from two major biases, which severely limit the comprehensive evaluation of LLM coding capabilities:

**Programming Language Bias**: Over 95% of code generation benchmarks are dominated by Python. LLM capabilities on popular languages like Java and C/C++ have rarely been systematically evaluated. Simply changing the programming language from Python to C++ can turn correct reasoning into errors (e.g., due to different runtime behaviors caused by type systems).

**Coding Task Bias**: Most benchmarks focus on code generation (natural language $\rightarrow$ program), while code reasoning (given a program, reasoning about input or output), which is an equally critical capability, has been largely neglected.

**Challenges in Building Multilingual Benchmarks**:
   - High cost of manual annotation (e.g., McEval cost $12,000)
   - Poor automated translation performance (an average success rate of only 64% with ChatGPT)
   - Potential data leakage issues with competition platform data

CruxEval-X is designed to fill these gaps by automatically expanding from the existing Python code reasoning dataset, CruxEval, to 19 programming languages.

## Method

### Overall Architecture

The construction pipeline consists of three main steps:
1. Function signature translation (type mapping)
2. Test case translation (rule enhancement)
3. Iterative generation and repair (multi-LLM collaboration)

### Key Designs

#### 1. Function Signature Translation (Step I)

- **Function**: Translate Python function signatures to equivalent signatures in other languages.
- **Mechanism**: Since Python does not require explicit type annotations, variable types are first inferred from test cases, and then transformed using type mapping rules.
- **Design Motivation**: Type information is crucial for cross-lingual translation; fundamental differences exist between Python's dynamic typing and C++'s static typing.
- For example: `def f(s1:str, s2:str) -> str` $\rightarrow$ `std::string f(std::string s1, std::string s2)`
- Signature translation can be completed for all 800 Python problems.

#### 2. Test Case Translation (Step II)

- **Function**: Translate Python test cases to equivalent formats in other languages.
- **Mechanism**: Adopt and enhance mapping rules from MultiPL-E.
- **Two Improvements**:
    - Enhance the handling of structured types (e.g., adding an equality function for C# Dict).
    - Generalize complex types into more generic types (e.g., `List[Union(str, int)]` $\rightarrow$ `List(str)`, provided functionality is unchanged).
- **Design Motivation**: Existing rules lack sufficient support for mixed types and need targeted enhancements.

#### 3. Iterative Generation and Repair (Step III)

- **Function**: Use multiple LLMs to iteratively translate Python code to target languages.
- **Mechanism**:
    - **Multi-round Generation**: Initial translation is generated using GPT-3.5-Turbo, and then iteratively processed using DeepseekCoder-33B-Instruct.
    - **Automated Repair**: After each LLM output, erroneous code and error messages are fed back to the LLM for correction.
    - **Early Stopping Mechanism**: Stop if the number of newly corrected problems in $k$ consecutive rounds falls below a threshold $\delta$.
    - **Overlap-based Multi-round Repair**: For problems correctly translated into $\ge 15$ languages but failed in others, GPT-4o is used for final repairs.
- **Design Motivation**: The translation success rate of a single LLM is limited. Combining multi-model collaboration with iterative repair significantly improves coverage.

Finally, out of 800 Python problems, 500 high-quality problems aligned across 19 languages are produced, with an additional 38 completed through manual tuning.

### Evaluation Task Design

Two types of reasoning tasks:
- **Output Reasoning**: Predict the output given the code and input.
- **Input Reasoning**: Predict the input given the code and output.
- Metric: Pass@1 (greedy decoding, temperature=0).

## Key Experimental Results

### Main Results: Pass@1 of 24 LLMs across 19 Languages

**Input Reasoning (Partial)**:

| Model | Params | Python | C++ | Java | JavaScript | Rust | Racket | Average |
|------|--------|--------|-----|------|-----------|------|--------|------|
| GPT-4o | - | 70.6 | 64.6 | 69.8 | 73.2 | 73.6 | 67.4 | ~71 |
| DeepseekCoder-V2 | 236B | 64.0 | 57.0 | 64.8 | 67.0 | 63.6 | 58.0 | ~63 |
| GPT-4o-mini | - | 59.6 | 52.2 | 57.2 | 59.6 | 61.2 | 51.2 | ~58 |
| CodeLlama-Instruct | 34B | 51.2 | 48.4 | 44.4 | 52.6 | 48.6 | 42.4 | ~48 |
| phi-1.5 | 1.3B | 25.8 | 16.0 | 26.8 | 9.8 | 25.2 | 6.6 | ~19 |
| phi-1 | 1.3B | 11.8 | 7.0 | 2.8 | 17.0 | 5.4 | 11.2 | ~11 |

**Output Reasoning (Partial)**:

| Model | Python | C++ | Java | JavaScript | Rust | Racket | Average |
|------|--------|-----|------|-----------|------|--------|------|
| GPT-4o | 75.4 | 74.8 | 73.2 | 77.6 | 74.4 | 70.8 | ~74 |
| DeepseekCoder-V2 | 66.8 | 66.2 | 67.6 | 65.4 | 65.8 | 62.2 | ~65 |
| phi-1.5 | 25.6 | 26.0 | 15.8 | 23.0 | 22.0 | 16.8 | ~22 |

### Ablation Study: Cross-Lingual Generalization

Performance of phi-1 (trained only on Python):

| Metric | Python | Average of other 18 languages |
|------|--------|----------------|
| Input Reasoning | 11.8% | ~10.7% |
| Output Reasoning | 22.4% | ~15.1% |
| Syntax Accuracy | 97.0% | 49.1% |

phi-1.5 (Python + Natural Language Enhancement):

| Metric | Python | Average of other 18 languages |
|------|--------|----------------|
| Input Reasoning | 25.8% | ~19.0% |
| Output Reasoning | 25.6% | ~21.7% |
| Syntax Accuracy | 98.7% | 72.0% |

**Key Findings**: Models trained only on Python can still achieve decent reasoning performance on other languages, indicating that LLMs possess significant cross-lingual generalization capability.

### Programming Language Correlation

| Language Pair | Cosine Similarity |
|-------|----------|
| JavaScript - TypeScript | 0.87-0.91 (Highest) |
| Racket - Other languages | Lowest |
| Average (all language pairs) | 0.7+ |

Language correlation of output reasoning is slightly higher than that of input reasoning (0.79 vs 0.75).

### Key Findings

1. **CruxEval-X is challenging for all LLMs**: Even GPT-4o only achieves around 70-75% Pass@1.
2. **Comparable Input and Output Reasoning Performance**: Across various languages, performance on both reasoning tasks is similar.
3. **Cross-Lingual Generalization of Monolingual Models**: phi-1 (trained only on Python) still achieves 16-26% output reasoning success rates in unseen languages.
4. **Natural Language Capabilities Facilitate Code Reasoning**: Improvements from phi-1 to phi-1.5 primarily stem from natural language data augmentation, yet code reasoning also improves significantly.
5. **Syntactic Structure Similarity Determines Generalization Level**: PHP has structural similarity with Python, leading to good generalization; Racket has unique syntax, yielding the worst generalization.
6. **Negative Correlation with Input/Output Length**: Longer input/output strings make reasoning more difficult.

## Highlights & Insights

- **Fully Automated Construction Pipeline**: Test-guided + iterative generation/repair by multiple LLMs, incurring much lower costs than manual annotation while guaranteeing quality.
- **Filling the Multilingual Evaluation Gap in Code Reasoning**: This is the first code reasoning benchmark covering 19 languages.
- **Cross-Lingual Generalization Insights**: The performance of monolingually trained models on unseen languages reveals language-agnostic reasoning capabilities within LLMs.
- **Practical Benchmark Construction Methodology**: Extensible to multilingual expansion of other Python-based datasets.

## Limitations & Future Work

1. **Potential Bias of Model-Generated Data**: Although experiments suggest this does not affect fairness, theoretical risks remain.
2. **Less-than-perfect Translation Quality**: Out of 800 problems, only 500 were completely aligned; some languages faced adaptation difficulties.
3. **Inability to Retain Language-Specific Features**: Code translated from Python cannot reflect the unique paradigms and best practices of target languages.
4. **Scale of 500 Aligned Problems**: Decent for distinguishing LLMs, but still insufficient for deep analysis of specific languages.
5. **Latest Large Language Models Not Evaluated**: Models post-GPT-4o and the Claude series are not evaluated.

## Related Work & Insights

- Extends the monolingual limitations of CruxEval (Gu et al., 2024).
- Complements McEval (manually annotated multilingual benchmark) by offering an automated alternative.
- Cross-lingual generalization findings echo multilingual transfer learning in natural language processing.
- Provides a new angle to measure "code understanding" vs "pattern matching" in LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First code reasoning benchmark spanning 19 languages; construction pipeline is exquisitely designed)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (24 models × 19 languages, with rich analysis dimensions: generalization, correlation, key factors)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, vivid case studies, though some oversized tables slightly affect readability)
- **Value**: ⭐⭐⭐⭐ (Fills an important gap, offers reusable methodology, highly inspiring findings)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral](are_rules_meant_to_be_broken_understanding_multilingual_moral_reasoning_as_a_com.md)
- [\[ACL 2025\] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)
- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)

</div>

<!-- RELATED:END -->
