---
title: >-
  [Paper Note] DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination
description: >-
  [ICML 2025][Reasoning][Code LLM benchmarking] Based on the concept of metamorphic testing, this work decomposes programming problems into complexity-related algorithmic abstractions and complexity-independent contextual descriptions. Through the collaboration of four LLM agents, it automatically generates semantically equivalent yet textually distinct variants of programming problems. This effectively mitigates data contamination and evaluates the true reasoning capabilities…
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Code LLM benchmarking"
  - "data contamination"
  - "metamorphic testing"
  - "dynamic evaluation"
  - "DyPass"
date: 2026-05-08
content_hash: 38329eb67caf6080
---

# DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination

**Conference**: ICML 2025  
**arXiv**: [2503.04149](https://arxiv.org/abs/2503.04149)  
**Code**: [Project Page](https://codekaleidoscope.github.io/dycodeeval.html)  
**Area**: Code Intelligence  
**Keywords**: Code LLM benchmarking, data contamination, metamorphic testing, dynamic evaluation, DyPass

## TL;DR

Based on the concept of metamorphic testing, this work decomposes programming problems into complexity-related algorithmic abstractions and complexity-independent contextual descriptions. Through the collaboration of four LLM agents, it automatically generates semantically equivalent yet textually distinct variants of programming problems. This effectively mitigates data contamination and evaluates the true reasoning capabilities of Code LLMs, validating the effectiveness of the framework across 18 models.

## Background & Motivation

**Background**: The evaluation of Code LLMs (such as DeepSeek-Coder, Qwen2.5-Coder, CodeLlama, etc.) primarily relies on static benchmarks like HumanEval and MBPP. Pass@1 scores on these benchmarks are widely used to measure code reasoning capabilities.

**Limitations of Prior Work**: Static benchmarks face severe data contamination issues—training corpora of LLMs inevitably contain these public benchmarks, leading to inflated evaluation scores that fail to reflect actual reasoning capability. Existing mitigation strategies have drawbacks: LiveCodeBench fetches new problems from online platforms but still relies on manual problem creation, and the semantic complexity of problems is uncontrollable; PPM generates variants via manually defined lambda operators, but this incurs high manual cost and yields limited variant diversity (BLEU-4 only drops to 0.69).

**Key Challenge**: When LLM performance drops on a new benchmark, it is impossible to distinguish whether it is due to insufficient model capability or higher benchmark difficulty. There is a need for a method that generates sufficiently diverse test problems while keeping the semantic complexity unchanged.

**Goal**: (1) How to automatically generate programming problems that are semantically equivalent but textually different from the original? (2) How to ensure that the generated variants do not alter the algorithmic complexity of the original problem? (3) How to provide a complete evaluation package containing test cases and canonical solutions?

**Key Insight**: The authors draw inspiration from the concept of metamorphic testing in software engineering. The core observation is that a programming problem can be decomposed into a "complexity-related algorithmic abstraction" and a "complexity-independent contextual description". Modifying the contextual description does not change the algorithmic logic or complexity, but can yield entirely new problem text, thereby breaking memorization.

**Core Idea**: Automatically replace the contextual descriptions of programming problems using LLM agents (e.g., changing "filtering rain hitting windows" to "filtering bank transactions"), keeping the algorithmic abstraction unchanged, thereby generating complexity-equivalent but textually distinct evaluation variants to detect and prevent data contamination.

## Method

### Overall Architecture

DyCodeEval takes a seed programming problem (from HumanEval or MBPP) and processes it through four sequentially executed LLM agents to produce semantically equivalent new problems. The new problems retain the canonical solutions and test cases of the original problems (as the algorithmic logic remains unchanged), but the prompt texts are completely different. The generation process has inherent randomness (scenario sampling $\times$ context generation), theoretically producing different variants with each run. Overall pipeline: Seed problem $\rightarrow$ Scenario Proposer $\rightarrow$ Context Generator $\rightarrow$ Prompt Rewriter $\rightarrow$ Validator $\rightarrow$ Output variants.

### Key Designs

1. **Scenario Proposer Agent**:

    - **Function**: Generates diverse application scenarios for the variant problems to ensure variants do not repeat across different runs.
    - **Mechanism**: Maintains a scenario pool (initially containing predefined scenarios like banking, medical, education). It uses LLMs to iteratively generate new scenarios using existing scenarios as few-shot exemplars and appends them to the pool until a target size is reached (50 scenarios in the experiment). A scenario is randomly sampled from the pool during each variant generation.
    - **Design Motivation**: Scenario diversity is the first line of defense against contamination. Iterative expansion via LLMs rather than manual definition ensures broad and sustainable scenario coverage.

2. **Context Generator Agent**:

    - **Function**: Assigns scenario-related semantic contexts to each input variable of the seed problem.
    - **Mechanism**: First, a recursive type inference algorithm analyzes the concrete values in ASSERT statements to deduce the data types of each input variable (e.g., `List[int]`, `Tuple[int|string]`). Then, it prompts the LLM to assign meaningful contextual names to each variable based on the selected scenario. For instance, in a "recommender systems" scenario, a `List[int]` might become "list of view counts for user blogs". Although the type inference algorithm is not complete, it is sound, meaning the collected types definitely appear in the canonical solution.
    - **Design Motivation**: Python lacks explicit type declarations, requiring automatic type inference to assign reasonable semantic contexts to variables.

3. **Prompt Rewriter Agent**:

    - **Function**: Rewrites the prompt of the original programming problem into a version that fits the new scenario and context.
    - **Mechanism**: Provides detailed scenario descriptions and variable contextual information, requiring the LLM to perform a rewriting task (rather than generating from scratch) while keeping the original problem's core algorithmic requirements unchanged. Rewriting is simpler and more controllable than generation; combined with detailed contextual information, it ensures semantic equivalence.
    - **Design Motivation**: Avoids the risk of introducing additional algorithmic constraints that may occur when generating from scratch.

4. **Validator Agent**:

    - **Function**: Acts as a probabilistic oracle to verify if the rewritten problem remains consistent with the original one.
    - **Mechanism**: Verified from two aspects: (1) Comparing the original and rewritten prompts to ensure core concept and factual accuracy; (2) Checking if the canonical solution can successfully solve the rewritten problem. The rewritten variant is accepted only if both checks pass; otherwise, it is regenerated. The consistency rate reaches 95% when Claude-3.5-Sonnet is used as the base model.
    - **Design Motivation**: The rewriting process of LLMs may unintentionally alter the problem intent. Dual validation ensures high quality.

### Loss & Training

There is no traditional training loss in this paper. The core evaluation metrics are Pass@K and the newly proposed DyPass@K. For a seed problem, DyPass@K generates $n$ semantic variant prompts and verifies whether the model can stably solve all variants, rather than generating $n$ candidate solutions for the same prompt. DyPass expands the input space, making it possible to distinguish whether a model has memorized the problem context or truly understands the algorithmic logic. Theoretical analysis of collision probability shows that 50 scenarios $\times$ 50 contexts = 2500 combinations, making the duplication probability extremely low.

## Key Experimental Results

### Main Results: Detection Performance Under Manual Contamination (HumanEval)

| Model | 0% Contaminated Pass@1 | 100% Contaminated Pass@1 (Static) | 100% Contaminated Pass@1 (DyCodeEval) |
|------|---------------|----------------------|------------------------------|
| Llama-3.2-1B | 4.3 | 28.7 | 4.9 |
| Llama-3.2-3B | 13.4 | 42.1 | 10.4 |
| DeepSeek-Coder-1.3B | 53.0 | 72.0 | 27.4 |

Under 100% leakage, the static Pass@1 score improves significantly (due to memorization), while the DyCodeEval score remains almost unchanged or even drops, proving the framework's effectiveness in resisting contamination.

### Pass@K vs DyPass@K Comparison Table

| Model | Pass@3 | Pass@5 | Pass@10 | DyPass@3 | DyPass@5 | DyPass@10 |
|------|--------|--------|---------|----------|----------|-----------|
| Llama-3.2-1B | 0.22 | 0.27 | 0.34 | 0.17 | 0.21 | 0.26 |
| Llama-3.2-1B (Contaminated) | **0.82** | **0.83** | **0.85** | 0.13 | 0.15 | 0.17 |
| Llama-3.2-3B | 0.35 | 0.40 | 0.48 | 0.31 | 0.36 | 0.43 |
| Llama-3.2-3B (Contaminated) | **0.88** | **0.88** | **0.89** | 0.24 | 0.27 | 0.29 |

The Pass@K of the contaminated models is inflated to 0.82–0.89, whereas DyPass@K drops instead (0.13–0.29), showing a sharp contrast.

### Key Findings

- In in-the-wild model evaluations, Qwen2.5-Coder-7B falls outside the 95% confidence interval on both HumanEval and MBPP, strongly implying the presence of data contamination.
- The external BLEU-4 of problems generated by DyCodeEval is only 0.17 (HumanEval) and 0.02 (MBPP), which is significantly lower than all baseline methods (PPM: 0.69/0.57), demonstrating high diversity.
- The variance of Pass@1 across 10 independent runs is extremely small, proving the stability and reproducibility of the evaluation results.
- When the base model is changed from Claude-3.5-Sonnet to Haiku, the consistency rate drops from 95% to 83%, indicating that highly capable models are crucial for quality.

## Highlights & Insights

- The mapping of the metamorphic testing concept from software engineering to LLM evaluation is natural: the bifurcation of algorithmic abstraction vs. contextual description corresponds precisely to the core concept of metamorphic relations.
- The proposed DyPass metric establishes a new contamination-aware standard for Code LLM evaluation.
- The contamination discovery in Qwen2.5-Coder-7B provides valuable practical warnings.
- The division of labor among the four agents is clear, with simple tasks at each step, reducing the probability of LLM errors.

## Limitations & Future Work

- Only validated on Python programming problems; multi-language scenarios are unexplored.
- The validator agent is a probabilistic oracle, carrying the risk of missed detections (only 83% with Haiku).
- The detection effect of data contamination under Chain-of-Thought (CoT) reasoning modes has not been considered.
- The computational cost of the base model (Claude-3.5-Sonnet) is relatively high, limiting large-scale deployment.

## Related Work & Insights

- **vs LiveCodeBench**: LCB filters old problems based on timestamps, whereas DyCodeEval relies on semantic transformations—the latter does not depend on the continuous generation of new problems.
- **vs PPM**: PPM defines operators manually, resulting in limited diversity (BLEU-4 = 0.69); DyCodeEval is fully automated with 4x higher diversity (BLEU-4 = 0.17).
- **Insights**: The metamorphic testing approach can be extended to other LLM benchmarks, such as mathematical reasoning and NLP understanding, to prevent contamination.

## Rating

⭐⭐⭐⭐⭐ The transfer of metamorphic testing to Code LLM evaluation is highly innovative. The large-scale experiments spanning 18 models across 2 seed datasets are thorough. The DyPass metric possesses theoretical value, and the contamination findings on Qwen2.5-Coder-7B hold practical impact. This work is a significant contribution to the methodology of Code LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](../../NeurIPS2025/llm_reasoning/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](../../ICLR2026/llm_reasoning/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICML 2025\] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](../../NeurIPS2025/llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](../../ACL2025/llm_reasoning/glore_long_cot_representation.md)

</div>

<!-- RELATED:END -->
