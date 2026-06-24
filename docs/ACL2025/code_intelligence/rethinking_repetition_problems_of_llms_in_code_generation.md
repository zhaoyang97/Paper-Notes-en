---
title: >-
  [Paper Note] Rethinking Repetition Problems of LLMs in Code Generation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] This paper redefines the repetition problem in code generation by distinguishing "structural repetition," which is more prevalent and more challenging than content repetition. It proposes RPG (Repetition Penalization based on Grammar), a decoding method with grammar-rule-based repetition penalty, which significantly mitigates repetition issues on both the newly constructed CodeRepetEval and standard benchmarks.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Repetition Problem"
  - "Structural Repetition"
  - "Grammar Rules"
  - "Decoding Strategy"
date: 2026-05-08
content_hash: 84ea140feaaeebfd
---

# Rethinking Repetition Problems of LLMs in Code Generation

**Conference**: ACL 2025  
**arXiv**: [2505.10402](https://arxiv.org/abs/2505.10402)  
**Code**: None  
**Area**: Text Generation/Code Generation  
**Keywords**: Code Generation, Repetition Problem, Structural Repetition, Grammar Rules, Decoding Strategy  

## TL;DR
This paper redefines the repetition problem in code generation by distinguishing "structural repetition," which is more prevalent and more challenging than content repetition. It proposes RPG (Repetition Penalization based on Grammar), a decoding method with grammar-rule-based repetition penalty, which significantly mitigates repetition issues on both the newly constructed CodeRepetEval and standard benchmarks.

## Background & Motivation

**Background**: LLMs have made significant progress in code generation tasks, but the repetition problem during generation remains a persistent issue. Current research on repetition mainly focuses on "content repetition"—where the model repeats generated code snippets verbatim. Common mitigation techniques include repetition penalty, n-gram blocking, etc.

**Limitations of Prior Work**: Content repetition (verbatim repetition) is only the tip of the iceberg. A more prevalent and harder-to-detect issue is "structural repetition"—where code blocks with the same structure/pattern but different surface text appear repeatedly. For example, a model might consecutively generate dozens of `if-else` statements with the same format but different variable names, or repeat similar function definitions. Such repetition is difficult to detect using text-level n-gram matching but is highly redundant in syntactic structure.

**Key Challenge**: Existing repetition penalty methods are based on token or n-gram level matching, failing to identify structural repetition. Because each occurrence of structural repetition may use different variable names and parameters, their surface text differs significantly, whereas their syntax tree structures are identical. This is a key difference between code and natural language—code has strict grammatical structures, and repetition can occur at the syntactic structure level rather than the textual level.

**Goal**: (1) To formally define the concept and detection criteria of "structural repetition"; (2) To propose a decoding method capable of handling both content and structural repetition; (3) To construct a comprehensive benchmark specifically designed for evaluating repetition problems in code generation.

**Key Insight**: Utilizing the formal grammar of programming languages to detect repetition—if the generated code exhibits repeating patterns in the sequence of applied grammar rules (production rules), it is identified as structural repetition. This is more precise and comprehensive than text-level matching.

**Core Idea**: Repetition detection and penalization based on grammar rules—mapping each token to its corresponding grammar rule during decoding, detecting repetition patterns in the sequence of grammar rules, and decaying the generation probability of key tokens that lead to repetition.

## Method

### Overall Architecture
The workflow of RPG is as follows: at each step of LLM code generation, (1) it maps the currently generated partial code to a sequence of grammar rules; (2) it detects whether repeating patterns exist on the grammar rule sequence (including strict repetition and structural repetition); (3) if a repetition trend is detected, it decays the logit values of key tokens that would "continue" the repetition pattern; (4) it samples/selects the next token based on the modified logit distribution. The entire process is embedded into the inference pipeline as a decoding strategy without modifying model parameters.

### Key Designs

1. **Formal Definition of Structural Repetition**:

    - **Function**: To provide a rigorous mathematical definition and detection criteria for structural repetition in code generation.
    - **Mechanism**: Defining the sequence of grammar rule applications of the code as $R = [r_1, r_2, \ldots, r_n]$, where $r_i$ denotes the grammar production rule applied when generating the $i$-th token. Structural repetition is defined as: there exists a subsequence $R[i:j]$ that appears more than $k$ times in $R$ (where $k$ is a configurable threshold), i.e., $|{m : R[m:m+j-i] = R[i:j]}| > k$. The key is that the comparison here is on the sequence of grammar rules rather than the raw token sequence—two code snippets with different texts but the same sequence of grammar rules represent a structural repetition.
    - **Design Motivation**: Leveraging the characteristic that programming languages naturally possess formal grammars, lifting repetition detection from the text level to the structural level, and covering repetition types that traditional n-gram methods fail to capture.

2. **Grammar-Based Repetition Detection Module**:

    - **Function**: Real-time detection during the decoding process of whether the currently generated code is entering a repetition pattern.
    - **Mechanism**: Maintaining a sliding window of grammar rule applications and using an improved suffix array or string matching algorithm to detect repeating substrings on the sequence of grammar rules. Specifically, for the grammar rule sequence $R[1:t]$ at the current position $t$, it checks whether the recently generated subsequence $R[t-l+1:t]$ of length $l$ has occurred in the preceding sequence. If a repeating pattern is matched, it further determines whether the token currently being generated belongs to the "key trigger tokens" in the repetition pattern—i.e., if this token is replaced, the repetition pattern will be broken.
    - **Design Motivation**: Real-time detection is required during decoding, making algorithmic efficiency crucial. Pattern matching on the sequence of grammar rules (which is much shorter than the raw token sequence) reduces computational complexity.

3. **Repetition Penalization Decay (RPD)**:

    - **Function**: Once a repetition trend is detected, decreasing the generation probability of tokens that continue the repetition.
    - **Mechanism**: Identifying "key tokens" in the current repetition pattern—those that, if changed, can break the repetition loop. A decay penalty is applied to the logit values of these tokens: $\text{logit}'(t) = \text{logit}(t) \times \alpha^{c}$, where $\alpha < 1$ is the decay factor and $c$ is the current number of repetitions. The more often it repeats, the heavier the penalty (exponential decay). Meanwhile, to avoid quality degradation caused by over-penalization, a minimum logit threshold is set—further decay is stopped when the penalized logit drops below this threshold.
    - **Design Motivation**: Exponential decay ensures that minor repetitions are not over-penalized (some repetition in code is reasonable), while persistent repetitions are strongly suppressed. The threshold mechanism prevents excessive penalization that might force the model to select semantically unreasonable alternative tokens.

### Loss & Training
RPG is a pure decoding-phase method and does not involve any training or loss functions. The core hyperparameters include: the minimum subsequence length $l$ for repetition detection, the repetition frequency threshold $k$, the decay factor $\alpha$, and the minimum logit threshold. These parameters can be adjusted based on specific programming languages and generation tasks. The method is implemented as a plug-and-play decoding wrapper, compatible with any autoregressive-decoding-based code generation model.

## Key Experimental Results

### Main Results
Comparison of repetition rate and functional correctness on the CodeRepetEval dataset:

| Method | Repetition Rate ↓ | Pass@1 ↑ | Generation Efficiency | Application Scope |
|------|---------|---------|---------|---------|
| Greedy Decoding | 42.3% | 38.1 | Fastest | Baseline |
| Repetition Penalty | 28.7% | 39.5 | Fast | Content repetition only |
| N-gram Blocking | 25.4% | 36.8 | Fast | Content repetition only |
| Nucleus Sampling | 31.2% | 40.2 | Fast | Randomness mitigation |
| **RPG** | **12.5%** | **44.8** | Moderate | Content & Structural repetition |

Results on standard benchmarks HumanEval and MBPP (CodeLlama-7B):

| Method | HumanEval Pass@1 | MBPP Pass@1 | Repetition Issue Rate |
|------|------------------|-------------|-----------|
| Greedy Decoding | 33.5 | 47.2 | 18.3% |
| Repetition Penalty | 34.2 | 47.8 | 12.1% |
| **RPG** | **36.8** | **50.3** | **5.7%** |

### Ablation Study
Contribution analysis of RPG components:

| Configuration | CodeRepetEval Repetition Rate | Pass@1 | Description |
|------|-------------------|--------|------|
| Full RPG | 12.5% | 44.8 | Optimal |
| Token-level repetition detection only | 25.8% | 40.1 | Degrades to standard repetition penalty |
| Grammar-rule detection only (no decay) | 18.3% | 42.1 | Detected but penalty strategy is missing |
| Fixed penalty (non-exponential decay) | 15.1% | 41.5 | Too light or too severe |
| Removing minimum logit threshold | 13.8% | 40.3 | Low repetition rate but quality degrades |

### Key Findings
- **Structural repetition is far more prevalent than content repetition**: In CodeRepetEval, about 70% of repetition issues belong to structural repetition, and only 30% are traditional content repetitions. This explains why n-gram-based methods have limited effectiveness.
- **RPG improves functional correctness while reducing repetition rate**: This may seem counterintuitive, but the reason is that after eliminating meaningless repetitive code, the model has more "token budget" to generate the actually needed logical code, rather than wasting output length in repetition loops.
- **Exponential decay is key to balancing repetition suppression and generation quality**: Fixed penalties tend to be either insufficient (small values) or over-intrusive on normal code structures (large values). Exponential decay automatically adjusts the strength according to the severity of repetition.
- RPG consistently improves performance across different model sizes (7B to 33B) and programming languages (Python, Java, C++).

## Highlights & Insights
- **Redefining the repetition problem in code generation**: Extending repetition from "content repetition" to "structural repetition" is an important conceptual contribution. Utilizing the formal grammar characteristics of code to detect structural repetition is a natural and effective point of entry, and this idea can be extended to other generation tasks with formal structures (such as SQL generation and mathematical proof generation).
- **Plug-and-play nature of decoding strategies**: RPG requires neither model modifications nor extra training; as a decoding wrapper, it can be directly applied to any code generation model. This non-intrusive design makes the approach highly adoptable in engineering.
- **Construction of the CodeRepetEval benchmark**: Filling the gap in evaluating code generation repetition problems, where previously no comprehensive benchmark specifically targeted the repetition issue existed.

## Limitations & Future Work
- Grammar rule matching increases decoding computational overhead, which may not be efficient enough for low-latency scenarios like real-time code completion.
- Currently, it only supports programming languages with explicit formal grammars, offering limited applicability to non-standard languages such as configuration files or DSLs.
- The threshold for minimum subsequence length in repetition detection needs to be tuned for different tasks.
- The scale and diversity of the CodeRepetEval dataset need to be expanded.
- **Directions for Improvement**: Incremental parsing techniques can be utilized to reduce the computational overhead of syntactic analysis. Alternatively, grammar rule repetition detection can be integrated internally into the model (as an auxiliary training objective) to fundamentally reduce the model's tendency to generate repetitive code.

## Related Work & Insights
- **vs Standard Repetition Penalty**: Standard repetition penalty only detects and penalizes repetition at the token level, failing to capture structural repetition; RPG elevates detection to the grammar rule level, providing broader coverage.
- **vs Nucleus Sampling/Top-k**: Sampling strategies indirectly mitigate repetition by introducing randomness, but this approach lacks specificity and may introduce irrational code while mitigating repetition.
- **vs ReflectionCoder (2405.17057)**: ReflectionCoder improves code quality from the perspective of training data, whereas RPG mitigates repetition from the perspective of decoding strategies—expressing complementarity, they can be used jointly.
- The core takeaway of this paper is that code generation should focus not only on "what to generate" but also on "how to generate," suggesting that the design space of decoding strategies deserves more attention.

## Rating
- Novelty: ⭐⭐⭐⭐ Significant innovation in the conceptual definition of structural repetition and the grammar-rule-level decoding strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Newly proposed benchmark + standard benchmarks, multi-model and multi-language evaluations, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions and intuitive examples.
- Value: ⭐⭐⭐⭐ The first systematic study on code generation repetition issues, providing a practical plug-and-play solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ICLR 2026\] An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems](../../ICLR2026/code_intelligence/an_agentic_framework_with_llms_for_solving_complex_vehicle_routing_problems.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)

</div>

<!-- RELATED:END -->
