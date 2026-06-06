---
title: >-
  [Paper Note] MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][Many-Shot ICL] This paper proposes MIR-Bench, the first large-scale and diverse many-shot in-context reasoning benchmark. By automatically generating input-output pairs from pr…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "Many-Shot ICL"
  - "Pattern Recognition"
  - "Inductive Reasoning"
  - "Transductive Reasoning"
  - "Long-Context Evaluation"
date: 2026-05-08
content_hash: dea7e512a75d704f
---

# MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?

**Conference**: NeurIPS 2025
**arXiv**: [2502.09933](https://arxiv.org/abs/2502.09933)  
**Code**: [https://github.com/KaiYan289/MIR-Bench](https://github.com/KaiYan289/MIR-Bench)  
**Area**: Information Retrieval
**Keywords**: Many-Shot ICL, Pattern Recognition, Inductive Reasoning, Transductive Reasoning, Long-Context Evaluation

## TL;DR

This paper proposes MIR-Bench, the first large-scale and diverse many-shot in-context reasoning benchmark. By automatically generating input-output pairs from programming problems, it evaluates LLMs' pattern recognition capabilities, revealing a performance saturation phenomenon caused by attention diffusion in many-shot settings, and demonstrating that transductive reasoning consistently outperforms inductive reasoning across models.

## Background & Motivation

**Background**: Recognizing patterns from examples and generalizing to new instances is a foundational capability for general intelligence, widely studied in psychology and AI. The continuous expansion of LLM context lengths (128K→2M or beyond) has given rise to the many-shot ICL paradigm—task learning at test time using hundreds to thousands of demonstrations, without costly fine-tuning.

**Limitations of Prior Work**: Existing pattern recognition benchmarks (ARC, WILT, KORBench, etc.) focus on few-shot settings (typically <10 demonstrations), lacking evaluation of the ability to integrate large amounts of information from long contexts. Meanwhile, many-shot ICL evaluations are largely confined to classification tasks, and mainstream long-context benchmarks (e.g., NIAH) reduce to retrieval problems that do not require extracting complex patterns from multiple clues. Both communities exhibit a notable gap.

**Key Challenge**: Some real-world problems have underlying rules that are intrinsically too complex or ambiguous to determine from a small number of examples (e.g., three points cannot distinguish a circle from a quadratic curve), whereas 300 examples can make the pattern unambiguous. LLMs should be capable of handling such long-context, multi-example reasoning, yet no suitable benchmark exists to measure this capability.

**Goal**: To construct the first large-scale, diverse many-shot pattern recognition reasoning benchmark, simultaneously filling the gap of missing many-shot evaluation in the pattern recognition community and the gap of missing complex reasoning tasks in the long-context community.

**Key Insight**: The paper leverages solution functions from programming problems as the source of underlying patterns—programming problems naturally offer diverse input-output types and difficulty levels, and solution functions can automatically generate large quantities of input-output pairs while avoiding data leakage.

**Core Idea**: Solution functions from entry-level programming problems are transformed into pattern recognition challenges, requiring LLMs to infer the underlying function solely from large numbers of input-output pairs. This yields a systematic benchmark covering 6,930 problems with support for 4 to 2,048 demonstrations.

## Method

### Overall Architecture

The core task of MIR-Bench: given $n$ input-output pairs $(x_1,y_1),\ldots,(x_n,y_n)$ from an unknown function $y=f(x)$ and a new input $x_{\text{new}}$, the LLM must predict $y_{\text{new}}=f(x_{\text{new}})$. Evaluation uses exact-match accuracy.

Benchmark construction follows a four-stage pipeline:

1. **Function Collection**: Solution functions are collected from three programming benchmarks—HumanEval+ (164 problems), MBPP+ (378 problems), and APPS (2,640 introductory problems).
2. **Input Generation**: GPT-4o-0806 generates a "data generator" script for each function; executing the script produces 20,000 shots and 10 test cases. This approach is more scalable and less error-prone than directly prompting the LLM to produce data.
3. **Output Generation**: Generated inputs are concatenated with the ground-truth function and executed to obtain true outputs. Problems with floating-point outputs, low output diversity, or execution errors are filtered out.
4. **Prompt Construction**: Input-output pairs and task descriptions are automatically assembled; problems on which all models achieve zero accuracy are filtered as unsolvable.

The final benchmark comprises:
- **MIR-Extended**: 693 valid functions × 10 test cases = 6,930 problems
- **MIR-Core**: 300 problems selected from the above that benefit most from many-shot demonstrations, based on the $D$ metric

### Key Designs

#### Data Generator Rather Than Direct Data Generation
- **Function**: GPT-4o writes data generator code rather than directly producing input data.
- **Mechanism**: Generating code first and then executing it scales easily to 20,000 shots with consistent formatting.
- **Design Motivation**: Directly prompting an LLM to generate input data is not scalable and prone to format mismatches; a code generator is written once and run repeatedly.

#### MIR-Core Selection Mechanism
- **Function**: The $D$ metric is defined to quantify the degree to which a problem benefits from many-shot demonstrations.
- **Mechanism**: $D = (D_1+D_2)/2$, where $D_1$ measures the difference between average accuracy at 64/128 shots and at 16/32 shots, and $D_2$ measures the difference between 32/64/128 shots and 4/8/16 shots. Values are computed as averages over five models.
- **Design Motivation**: Not all pattern recognition problems require many shots (e.g., simple addition or absolute value can be inferred from a few examples). Problems that genuinely benefit from more demonstrations must be identified.

#### Analysis of Factors Influencing $D$
- **Finding**: Fitting a quadratic function to the relationship between normalized factors and $D$ reveals:
    - **Function complexity is the dominant factor**: LLM-annotated difficulty is the strongest positive predictor.
    - **Answer diversity and input complexity are relatively unimportant.**
- **Challenge of LLM Difficulty Assessment**: LLMs tend to underestimate the inductive reasoning difficulty of seemingly simple underlying functions. Solution: a multi-turn dialogue with self-reflection—the LLM first attempts to solve the problem, then the ground-truth answer is revealed and the LLM reflects and re-rates the difficulty.

#### Superset Guarantee Design
- **Function**: Ensures that test cases for larger shot counts are strict supersets of those for smaller shot counts.
- **Design Motivation**: Eliminates difficulty fluctuations caused by sampling across different shot counts, guaranteeing strictly monotone information growth.

### Loss & Training

This paper presents an evaluation benchmark rather than a training method. Evaluation strategy:
- Greedy decoding (temperature = 0) is used to ensure reproducibility.
- Exact-match accuracy is the sole metric.
- Rule-based answer extraction is applied to parse final outputs from LLM responses.
- Each function is tested on 10 cases, each evaluated under 10 different shot counts (4 to 2,048).

## Key Experimental Results

### Main Results

**Performance of 15 models on MIR-Extended (693 functions × 10 cases = 6,930 problems)**

| Model | Best Accuracy | Shot Count at Best | Characteristics |
|---|---|---|---|
| o1-mini-0912 | <0.7 | ~256 | Strongest among all models |
| o1-preview-0912 | Below o1-mini | ~256 | Both clearly ahead of other models |
| Claude-3.5-Sonnet | ~0.6 | ~128–256 | Third tier |
| GPT-4o-0806 | <0.4 | ~128–256 | Fourth tier |
| Most other models | <0.4 | ≤256 | Performance saturates after 256 shots |

**Additional frontier models on MIR-Core**

| Model | Notes |
|---|---|
| o1-1217 | Stronger models generally score higher, but saturation persists |
| o3-mini-high | Strong reasoning model |
| DeepSeek-R1 | Long-CoT model; competitive performance |
| GPT-4.5-Preview | Included in evaluation |
| Gemini-2.0 Pro | Included in evaluation |

### Ablation Study

**Repeated-Shot Experiment: Analysis of Saturation Causes**

| Setting | Result | Implication |
|---|---|---|
| Normal many-shot | Saturation after 256–512 shots | Baseline performance |
| Single shot repeated to N | Gap with normal widens before 512 shots | LLMs do benefit from more information |
| All shots repeated K times | Similar to single-shot repetition | Issue is not information retrieval |
| >512 shots | Gap stops widening | Excessive information becomes harmful |

**Insight 1**: Saturation does not stem from insufficient information retrieval capacity, but from attention diffusion when aggregating too much information.

**Inductive Reasoning vs. Transductive Reasoning**

| Model | Acc. with CoT | Acc. without CoT | CoT Usage Rate |
|---|---|---|---|
| Claude-3.5-Sonnet | 0.585 | 0.775 | 98.73% |
| o1-preview-0912 | 0.588 | 0.797 | 56.71% |
| DeepSeek-R1 | 0.298 | 0.757 | 9.69% |
| GPT-4o-0806 | 0.488 | 0.540 | 10.85% |
| o1-mini-0912 | 0.334 | 0.696 | 2.54% |

**Insight 2**: Across all 21 models, transductive reasoning (without CoT) consistently and substantially outperforms inductive reasoning (with CoT).

**Robustness Experiment: Demonstrations Containing Errors**

| Error Ratio | Impact on Performance |
|---|---|
| <1/8 | Only marginal degradation |
| 1/4 | Moderate degradation |
| 3/4 | Some performance retained |
| Whether errors are disclosed | No significant difference |

**Insight 3**: LLMs are highly robust to erroneous demonstrations in many-shot pattern recognition.

**SolverLearner (Write Code First, Then Execute)**

| Model | 16-shot | 64-shot | 256-shot | 1024-shot | Trend |
|---|---|---|---|---|---|
| DeepSeek-R1 | +0.022 | +0.007 | +0.018 | +0.003 | Marginal gain |
| Claude-3.5-Sonnet | -0.009 | -0.015 | -0.017 | +0.04 | Mostly negative |
| GPT-4o-0806 | +0.012 | -0.033 | -0.029 | +0.004 | Inconsistent |
| Gemini-1.5-Pro | -0.029 | -0.055 | -0.067 | -0.04 | Consistently worse |

**Insight 4**: The code-then-execute paradigm is not reliably effective in the many-shot setting, and performance gains do not scale with shot count.

**RAG Experiment**

| Method | Result |
|---|---|
| Embedding-based RAG selecting 64-shot vs. random 64-shot | No significant difference |

**Insight**: Embedding-based RAG is ineffective for many-shot pattern recognition.

### Key Findings

1. **Many-shot saturation is universal**: Most models cease to improve or even degrade after 256–512 shots, including Gemini with its 2M context, indicating an attention diffusion problem rather than a hard context limit.
2. **Transductive reasoning outperforms inductive reasoning**: Across all 21 models, directly predicting the answer (transductive) outperforms reasoning before predicting (inductive). CoT does aid reasoning, but it disrupts the structural consistency between input-output pairs.
3. **Robustness to erroneous demonstrations**: LLMs exhibit strong tolerance for noise in many-shot pattern recognition.
4. **Code-then-execute is not a silver bullet**: The SolverLearner paradigm varies across models and cannot leverage the information gains from many-shot demonstrations.
5. **RAG is ineffective**: Embedding-based retrieval for subset selection provides no significant performance benefit.
6. **Function complexity determines the need for many-shot**: LLM-annotated difficulty is the strongest predictor.

## Highlights & Insights

- **Filling a dual gap**: The paper simultaneously addresses the absence of many-shot evaluation in the pattern recognition community and the absence of complex reasoning tasks in the long-context community, with a precisely targeted scope.
- **Elegant automated pipeline**: Using programming problems as a function source and prompting LLMs to write data generator code achieves large-scale automated data generation without data leakage, and is sustainably extensible.
- **Deep insight: transduction outperforms induction**: The paper proposes the hypothesis that CoT disrupts the structural consistency of input-output pairs, and validates through a "forced meaningless text" experiment that CoT itself is helpful but the cost of structural disruption is greater. This has direct practical implications for prompt engineering.
- **Fine-grained decomposition of the saturation mechanism**: Repeated-shot experiments distinguish between "insufficient information retrieval capacity" and "attention diffusion during information aggregation," confirming the latter as the cause.
- **Comprehensive exploratory experiment matrix**: A single paper systematically addresses six independent research questions (scaling effects, robustness, inductive vs. transductive, RAG, code-execute, cross-domain generalization), achieving extremely high information density.

## Limitations & Future Work

1. **Only introductory programming problems are used**: Although the authors argue that introductory problems are sufficiently challenging, more complex algorithmic patterns (e.g., dynamic programming) are not covered, potentially missing certain reasoning dimensions.
2. **Exact-match evaluation is overly strict**: Numerically approximate or semantically equivalent but differently formatted answers are marked incorrect, potentially underestimating actual model capability.
3. **All underlying functions are deterministic**: Real-world patterns often involve noise or stochasticity; the benchmark does not cover probabilistic function recognition.
4. **Input-output pairs are in text format only**: Multimodal pattern recognition (e.g., visual patterns) is not addressed, limiting applicability to VLM evaluation.
5. **The $D$ metric depends on five specific models**: MIR-Core selection is based on performance differences among a particular model set, which may introduce bias.
6. **Incomplete evaluation of the latest reasoning models**: Due to cost constraints, models such as o1-1217 and o3-mini are evaluated only on MIR-Core and do not participate in all experiments.

## Related Work & Insights

- **Relationship to ARC/ARC-AGI**: ARC is the flagship pattern recognition benchmark but is limited to few-shot settings and grid transformations. MIR-Bench extends pattern recognition to many-shot settings and diverse data types, serving as a complement rather than a replacement.
- **Relationship to NIAH and other long-context tests**: NIAH tests retrieval capability, whereas MIR-Bench tests inductive/transductive reasoning from large volumes of information—a higher-order assessment of long-context intelligence.
- **Relationship to SolverLearner**: The results challenge the overly optimistic conclusions of prior work by showing that code-then-execute is not universally effective in many-shot settings.
- **Implications for ICL theory**: The finding that transduction outperforms induction is highly consistent with theoretical work on ICL that models Transformers as performing implicit gradient descent—maintaining structural consistency between input-output pairs facilitates "implicit regression."
- **Implications for prompt engineering**: In many-shot settings, maintaining structural consistency is more important than adding CoT. This insight can guide prompt design in practical applications.
- **Relationship to the RAG community**: The failure of embedding-based RAG in pattern recognition tasks may be because similar inputs do not necessarily provide complementary pattern information, suggesting that more sophisticated example selection strategies are needed.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First many-shot pattern recognition benchmark filling a clear gap; automated pipeline is cleverly designed, though the core task format (inferring functions from I/O pairs) is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 21 models, six independent research questions, comprehensive ablation and controlled experiments with extremely high information density.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, each experiment distills an Insight box, and figures are abundant; however, high content density leads to somewhat compressed analysis in places.
- **Value**: ⭐⭐⭐⭐ — Direct impact on the many-shot ICL community; the transductive vs. inductive finding has broad reference value; long-term influence depends on community adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[ICML 2026\] How can embedding models bind concepts?](../../ICML2026/information_retrieval/how_can_embedding_models_bind_concepts.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)

</div>

<!-- RELATED:END -->
