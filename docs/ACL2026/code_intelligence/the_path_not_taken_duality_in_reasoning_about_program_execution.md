---
title: >-
  [Paper Note] The Path Not Taken: Duality in Reasoning about Program Execution
description: >-
  [ACL 2026][Code Intelligence][Program Execution Reasoning] This paper introduces the concept of duality in program execution reasoning. Through the DexBench benchmark (445 paired instances)…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Program Execution Reasoning"
  - "Counterfactual Reasoning"
  - "Dual-Path Reasoning"
  - "Code Coverage"
  - "LLM Code Understanding"
date: 2026-05-08
content_hash: a2c7c88416e9ec3b
---

# The Path Not Taken: Duality in Reasoning about Program Execution

**Conference**: ACL 2026
**arXiv**: [2604.20917](https://arxiv.org/abs/2604.20917)
**Code**: [github.com/sail-ucf/dexbench](https://github.com/sail-ucf/dexbench)
**Area**: Code Intelligence / Program Reasoning
**Keywords**: Program Execution Reasoning, Counterfactual Reasoning, Dual-Path Reasoning, Code Coverage, LLM Code Understanding

## TL;DR
This paper introduces the concept of duality in program execution reasoning. Through the DexBench benchmark (445 paired instances), it jointly evaluates LLMs on forward execution reasoning (predicting code coverage under a given input) and backward counterfactual reasoning (inferring input mutations that redirect execution to a target branch). The results reveal that strong performance in a single direction does not transfer to success under joint evaluation, exposing a fundamental deficiency in models' causal understanding of program execution.

## Background & Motivation

**Background**: LLMs are widely adopted in software engineering tasks (code generation, testing, bug fixing, etc.), yet their performance remains inconsistent — they can solve complex programming problems while failing on basic loop reasoning. This may stem from imitating surface patterns rather than genuinely understanding programs. Recent work has begun examining fine-grained reasoning capabilities, such as code coverage prediction, input-output mapping, and intermediate state tracking.

**Limitations of Prior Work**: Existing program execution benchmarks evaluate runtime behavior under a single test case — i.e., reasoning along a single execution path — whereas a given program may traverse multiple paths depending on its inputs. Single-test-case evaluation provides only a narrow view of a model's program understanding. Furthermore, benchmarks defined by fixed input-output pairs are susceptible to training data contamination, as models may have memorized the answers.

**Key Challenge**: Two execution paths of a program share the same execution space and diverge only at branch points due to differing program states. Existing evaluations overlook the causal relationships between execution paths — they assess only whether a model can predict *which path was taken*, without asking *why that path was taken* or *what would cause the other path to be taken instead*.

**Goal**: To design a joint evaluation framework that simultaneously tests LLMs' forward execution reasoning and backward counterfactual reasoning abilities, thereby providing a more robust measure of causal understanding of program execution.

**Key Insight**: Two program paths share a common execution space and diverge only at branch conditions. The authors argue that understanding program execution requires simultaneously understanding *how observed behavior arises* and *under what conditions execution would be redirected to an alternative path*.

**Core Idea**: Define duality in program execution reasoning — forward reasoning predicts the observable behavior of an execution path, while backward reasoning infers the input mutations that redirect execution to the counterfactual path — with the two jointly constituting dual-path reasoning: $\mathcal{R}_{dual} = \mathcal{R}_{exec} \oplus \mathcal{R}_{cf}$.

## Method

### Overall Architecture
DexBench comprises 445 evaluation instances, each consisting of a paired forward and backward reasoning task. Given a program $P$ and a test input $I_{exec}$: the forward task requires predicting code coverage (which lines are executed); the backward task requires mutating $I_{exec}$ into $I_{cf}$ such that execution covers a previously uncovered target branch. Both tasks must succeed simultaneously for dual-path reasoning to be considered successful. Data is sourced from CruxEval (298 instances, simple control flow), HumanEval (100 instances, moderate complexity), and PythonSaga (47 instances, deeply nested and recursive).

### Key Designs

1. **Forward Execution Reasoning ($\mathcal{R}_{exec}$)**:

    - **Function**: Predict statement coverage (code coverage prediction) when the program runs under a given input.
    - **Mechanism**: Given program $P$ and input $I_{exec}$, the model must predict the set of executed line numbers $\phi(\tau_{exec})$. Success is measured by exact match — at least one generated candidate must completely and correctly predict the code coverage. Ground-truth coverage is collected using the SlipCover tool.
    - **Design Motivation**: Code coverage prediction requires the model to maintain a representation of program state and update it according to the semantics of each statement, making it a natural task for evaluating execution reasoning. Difficulty scales with the length of the execution trace and the complexity of state transitions (e.g., non-trivial control flow).

2. **Backward Counterfactual Reasoning ($\mathcal{R}_{cf}$)**:

    - **Function**: Infer how the original input must be mutated so that program execution redirects to cover a specific target branch.
    - **Mechanism**: Given program $P$, original input $I_{exec}$, and a counterfactual target (covering a specific uncovered branch $b$), the model must generate a mutated input $I_{cf}$ such that $b \in \phi(\tau_{cf})$. The branch with the maximum coverage increment is selected as the counterfactual target. Correctness is verified by actually executing the generated inputs.
    - **Design Motivation**: Unlike coverage-guided fuzzing, DexBench requires models to identify necessary input changes through reasoning rather than random mutation. Counterfactual reasoning demands analysis of how input changes affect control flow and program state, making it a critical task for probing causal understanding.

3. **Dual-Path Reasoning ($\mathcal{R}_{dual}$)**:

    - **Function**: Jointly evaluate forward and backward reasoning ability to provide a robust assessment of program execution understanding.
    - **Mechanism**: $S_{dual} = S_{exec} \wedge S_{cf}$ — an instance is considered successful under dual-path evaluation if and only if the model correctly predicts the original coverage *and* generates at least one input that covers the target branch. This avoids the narrowness and incompleteness of single-path evaluation.
    - **Design Motivation**: Experiments reveal that many models succeed in only one direction — correctly predicting coverage but failing to mutate the input (or vice versa) — demonstrating that single-path evaluation overestimates models' program understanding ability.

### Loss & Training
DexBench is an evaluation benchmark and does not involve training. Evaluation uses the pass@k metric ($k=1$ and $k=5$) with one-shot prompting.

## Key Experimental Results

### Main Results (Dual-Path Reasoning pass@5, %)

| Model | CruxEval | HumanEval | PythonSaga |
|-------|----------|-----------|------------|
| Llama-3.2-3B-Inst. | 0.0 | 0.0 | 0.0 |
| Qwen2.5-32B | 57.0 | 31.0 | 6.4 |
| QwQ-32B | 33.2 | 25.0 | 2.1 |
| Gemini 2.5 Flash | 73.8 | 62.0 | 8.5 |
| GPT-5 Mini | 91.6 | 76.0 | 44.7 |
| Claude Sonnet 4 | **95.3** | **79.0** | **70.2** |

### Ablation Study (Input Generation vs. Input Mutation, GPT-5 Mini)

| Setting | CruxEval pass@5 | HumanEval pass@5 | PythonSaga pass@5 |
|---------|-----------------|------------------|-------------------|
| Input generation (without original input) | 97.7 | 99.0 | 95.7 |
| Input mutation (Ours) | 97.0 | 88.0 | 95.7 |

### Key Findings
- **Success in a single direction does not transfer to joint success**: Many models perform well on either execution reasoning or counterfactual reasoning, but performance drops substantially under dual-path evaluation, confirming the incompleteness of single-path evaluation.
- **Reasoning post-training is ineffective**: Among open-source models, non-reasoning variants (Qwen2.5-32B, Mistral Small 24B) outperform their reasoning counterparts (QwQ-32B, Magistral Small) on dual-path reasoning by an average of 63.9% and 47.9%, respectively.
- **Non-monotonic scaling effects**: Qwen2.5-32B consistently outperforms Qwen2.5-72B on dual-path reasoning; all models with fewer than 10B parameters achieve near-zero performance on execution reasoning but exhibit some capability on counterfactual reasoning.
- **Complexity has a significant impact**: Dual-path performance drops substantially from CruxEval to PythonSaga (e.g., Gemini 2.5 Flash falls from 73.8% to 8.5%).
- **Input mutation is harder than input generation**: Removing the original input actually improves performance, because input mutation requires reasoning about how input perturbations propagate through intermediate program states.

## Highlights & Insights
- **Duality framework**: Program execution reasoning has a natural forward/backward dual structure; joint evaluation more faithfully reflects true understanding than either direction in isolation. This idea generalizes to other reasoning tasks — analogous to "forward solving" versus "backward problem construction" in mathematical reasoning.
- **Counterfactual reasoning as a probe for causal understanding**: Unlike random fuzzing, requiring models to identify necessary input changes through reasoning rather than trial and error directly tests causal understanding.
- **Counterintuitive finding on reasoning post-training**: Reasoning-specialized post-training not only fails to improve program execution reasoning but actually causes regression, suggesting that current reasoning enhancement strategies may be overfitted to specific reasoning patterns.

## Limitations & Future Work
- Coverage is limited to Python; extending to other languages requires building secure sandboxed execution environments.
- Data is sourced from public benchmarks, posing a potential risk of data contamination (though the counterfactual reasoning design partially mitigates this concern).
- The counterfactual path is selected by default as the one with the maximum coverage increment, which may bias toward more easily reasoned-about branches.
- Code coverage is currently the only observable behavior; future work can extend to program outputs, intermediate states, and other execution attributes.

## Related Work & Insights
- **vs. CruxEval**: CruxEval pairs output prediction with input prediction, but evaluates each direction independently. DexBench jointly evaluates causal reasoning through a shared execution context.
- **vs. R-Eval (IC-Score)**: R-Eval combines an IC-Score from multiple execution tasks but remains a single-path evaluation. DexBench introduces the counterfactual dimension.
- **vs. CES**: CES evaluates reasoning consistency across multiple inputs but treats each path independently, without probing the causal logic that leads to path divergence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The duality framework offers an entirely new perspective on evaluating program reasoning, with an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 13 models, three complexity levels, and comprehensive sensitivity analysis, though the dataset scale is modest (445 instances).
- Writing Quality: ⭐⭐⭐⭐⭐ — Formal definitions are clear and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ — Provides a more robust evaluation framework for code LLMs and surfaces counterintuitive findings regarding reasoning post-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/code_intelligence/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)

</div>

<!-- RELATED:END -->
