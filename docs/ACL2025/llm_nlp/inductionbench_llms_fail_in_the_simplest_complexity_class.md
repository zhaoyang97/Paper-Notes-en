---
title: >-
  [Paper Note] InductionBench: LLMs Fail in the Simplest Complexity Class
description: >-
  [ACL 2025][LLM (Other)][Inductive reasoning] This paper proposes InductionBench, an inductive reasoning benchmark based on the subregular hierarchy, which reveals that even the strongest LLMs (such as o3-mini) struggle to master inductive reasoning tasks in the simplest complexity classes, exposing fundamental limitations of current LLMs in inducing rules from observational data.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Inductive reasoning"
  - "benchmarking"
  - "subregular hierarchy"
  - "function learning"
  - "LLM evaluation"
date: 2026-05-08
content_hash: 080be1901fa746db
---

# InductionBench: LLMs Fail in the Simplest Complexity Class

**Conference**: ACL 2025  
**arXiv**: [2502.15823](https://arxiv.org/abs/2502.15823)  
**Code**: [GitHub](https://github.com/Wenyueh/inductive_reasoning_benchmark)  
**Area**: LLM/NLP  
**Keywords**: Inductive reasoning, benchmarking, subregular hierarchy, function learning, LLM evaluation  

## TL;DR

This paper proposes InductionBench, an inductive reasoning benchmark based on the subregular hierarchy, which reveals that even the strongest LLMs (such as o3-mini) struggle to master inductive reasoning tasks in the simplest complexity classes, exposing fundamental limitations of current LLMs in inducing rules from observational data.

## Background & Motivation

**Background**: Existing LLM reasoning benchmarks mostly focus on deductive reasoning, such as mathematical proofs and code generation. In these tasks, rules (axioms, syntax) are explicitly given, and LLMs only need to perform search and planning based on known rules. Reasoning models like o1 and o3 have already achieved nearly saturated performance on such tasks.

**Limitations of Prior Work**: Inductive reasoning—inferring underlying rules from finite observations—is a core capability for scientific discovery, but existing benchmarks are highly inadequate in evaluating this capability. The few tests that involve induction (e.g., ARC) lack a rigorous formal framework, making it difficult to systematically assess the boundaries of LLM inductive capabilities.

**Key Challenge**: The exact strength of the inductive reasoning capability of LLMs remains unknown. If LLMs cannot master the simplest function classes in formal language theory, more complex scientific induction is out of the question. A systematic benchmark based on computational complexity theory is needed.

**Goal**: (1) Construct an inductive reasoning benchmark based on the subregular hierarchy; (2) systematically evaluate the inductive reasoning capabilities of LLMs across different complexity classes; and (3) quantify the impact of the hypothesis space size on reasoning difficulty.

**Key Insight**: The authors start from the subregular hierarchy in formal language theory—this hierarchy defines a sequence of complexity levels for string-to-string transformation functions, ranging from the simplest Input Strictly Local (ISL) to more complex classes. If LLMs cannot even master the lowest level, it indicates a fundamental deficiency in their inductive capabilities.

**Core Idea**: Use function classes within the subregular hierarchy as a testbed for inductive reasoning. By controlling the function classes (ISL $\rightarrow$ OSL $\rightarrow$ ...) and hypothesis space parameters ($k$, vocabulary size, and the number of rules), the boundary of LLM inductive reasoning capabilities can be precisely measured.

## Method

### Overall Architecture

The design of InductionBench is based on the subregular hierarchy in formal language theory. The overall pipeline is: (1) Select a function class (e.g., ISL-$k$) and hypothesis space parameters; (2) randomly sample a function from that class as the target; (3) generate several input-output pairs as demonstrations; (4) prompt the LLM to infer the underlying rules from these demonstrations and predict the output for new inputs. The benchmark is divided into a Standard Benchmark (function classes with polynomial-time provably correct learning algorithms) and an Exploration Benchmark (more complex classes with no known efficient algorithms).

### Key Designs

1. **Subregular Function Hierarchy**:

    - **Function**: Provide a structural framework that categorizes function complexity from simple to complex.
    - **Mechanism**: The subregular hierarchy stratifies string transformation functions based on locality and directionality. The simplest ISL-$k$ (Input Strictly Local) functions depend only on a local window of length $k$ in the input string to determine the output at each position; OSL-$k$ (Output Strictly Local) functions depend on local windows on the output side. Higher-order categories like Subsequential and Regular functions feature longer-range dependencies. Each level has strict formal definitions and containment relations.
    - **Design Motivation**: This hierarchical structure allows precise control over task difficulty—if an LLM fails at the bottom of the hierarchy, there is no need to test higher levels, thereby quickly identifying the ceiling of capabilities.

2. **Controllable Hypothesis Space Parameterization**:

    - **Function**: Precisely control the difficulty of each test instance via parameter combinations.
    - **Mechanism**: Each function class is controlled by three parameters: $k$ (local window size), vocabulary size $|\Sigma|$, and the number of rules $n$. The size of the hypothesis space can be precisely calculated as $\binom{|\Sigma|^k}{n} \times (|\Sigma|)^n$. By sweeping combinations of these parameters, precise curves representing LLM performance as a function of hypothesis space size can be plotted.
    - **Design Motivation**: Existing inductive reasoning evaluations lack quantitative control over task difficulty. The parameterized design enables distinguishing between "the model is entirely incapable of induction" and "the model can perform induction only in simple cases."

3. **Dual-Tier Benchmark Design (Standard + Exploration)**:

    - **Function**: Distinguish between algorithmically "solvable" and "unsolved" inductive reasoning tasks.
    - **Mechanism**: The Standard Benchmark contains function classes with known polynomial-time learning algorithms (such as ISL and OSL), providing a theoretical upper bound for reference. The Exploration Benchmark contains more complex function classes (such as non-monotone subsequential) with no known efficient and provably correct algorithms. Comparing the two benchmarks reveals whether LLMs can partially solve tasks through "brute-force search" or "pattern matching."
    - **Design Motivation**: By distinguishing between algorithmically solvable and unsolvable classes, it is possible to better understand whether the LLM's reasoning strategy resembles systematic search or merely shallow pattern matching.

### Loss & Training

This paper introduces an evaluation benchmark and does not involve training. The evaluation metric is weighted accuracy, where weights are inversely proportional to the hypothesis space size (more difficult settings receive higher weights). Variants using log-weights are also provided to reduce the impact of extreme settings.

## Key Experimental Results

### Main Results

| Model | ISL (Standard) | OSL (Standard) | ISL (Exploration) |
|------|----------------|-----------------|---------------------|
| Llama-3.1 8B | 0.00 | 0.00 | 0.00 |
| Qwen2.5-Coder-32B | 0.073 | 0.007 | 0.0003 |
| Llama-3.3 70B | 0.066 | 0.058 | 0.001 |
| DeepSeek-R1-Distill-70B | 0.038 | 0.051 | 0.008 |
| o3-mini | 0.289 | 0.431 | 0.057 |

### Ablation Study (Impact of Hypothesis Space Size)

| Parameter Settings (k, vocab, rules) | Hypothesis Space Size | o3-mini Accuracy | Description |
|---------------------------|-------------|---------------|------|
| (2, 2, 1) Simplest | ~6 | ~0.8+ | Almost solved |
| (3, 3, 2) Medium | ~Hundreds | ~0.3 | Sharp drop-off |
| (4, 4, 3) Hard | ~Tens of thousands | <0.1 | Near failure |
| (4, 4, 4) Hardest | ~Hundreds of thousands | ~0.01 | Completely unfeasible |

### Key Findings

- **All models perform poorly on the simplest ISL class**: even o3-mini achieves a weighted accuracy of only 0.29, suggesting that inductive reasoning is indeed a fundamental weakness of current LLMs.
- **Hypothesis space size is the primary determinant of difficulty**: performance drops exponentially with hypothesis space size, showing that LLMs cannot effectively search combinatorial spaces.
- **Reasoning-enhanced models (e.g., o3-mini) exhibit clear advantages but are still far from sufficient**: o3-mini shows some capability in simple settings but fails as the hypothesis space grows slightly larger.
- **Small models (8B) completely lack capability**: Llama-3.1 8B scores 0 under all settings.

## Highlights & Insights

- **Rigorous definition of inductive reasoning difficulty tiers using formal language theory**: this provides a more scientific foundation compared to intuition-based benchmarks like ARC, as the complexity and hypothesis space size of each function class can be precisely calculated, enabling truly controllable evaluation.
- **Revealing a fundamental blind spot in LLM reasoning**: while current discussions around LLM reasoning capabilities focus heavily on deductive reasoning, this work clearly demonstrates that inductive reasoning is a completely different and currently unsolved dimension.
- **The parameterized hypothesis space control method can be transferred to other cognitive evaluations**: for instance, using similar frameworks to evaluate LLMs' capabilities in concept learning, causal inference, and other abilities.

## Limitations & Future Work

- Currently only string-to-string transformation functions are tested, without covering more natural inductive reasoning scenarios (such as inducing rules from image sequences).
- The subregular hierarchy is only a sub-domain of formal language theory; more complex function classes (such as Turing-computable functions) are not covered.
- The evaluation is zero-shot/few-shot prompt-based, leaving long-sequence variants of fine-tuning or in-context learning unexplored for potential performance improvements.
- The concrete reasons for LLM failure have not been analyzed—whether it stems from inappropriate search strategies or a fundamental lack of inductive reduction capabilities.

## Related Work & Insights

- **vs ARC Benchmark**: ARC also tests inductive reasoning but is based on visual patterns and lacks formal complexity stratification. InductionBench has the advantage of a more solid theoretical foundation, allowing precise quantification of difficulty.
- **vs BIG-Bench (Induction tasks)**: Inductive tasks in BIG-Bench lean toward simple sequence extrapolation, lacking a systematic hierarchy of complexity.
- **vs Program Synthesis**: Program synthesis also involves inducing rules from examples, but typically within explicit DSL constraints. InductionBench focuses more on pure inductive capability rather than programming ability.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce the subregular hierarchy to LLM evaluation, providing a unique angle with a deep theoretical foundation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple models and parameter combinations, but lacks exploration of fine-tuning and long contexts.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formal definitions, though the barrier to entry might be high for readers without a background in formal language theory.
- **Value**: ⭐⭐⭐⭐⭐ Points out a fundamental blind spot in LLM capabilities, providing valuable implications for future reasoning-enhancement research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ACL 2026\] Model-Agnostic Meta Learning for Class Imbalance Adaptation](../../ACL2026/llm_nlp/model-agnostic_meta_learning_for_class_imbalance_adaptation.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)

</div>

<!-- RELATED:END -->
