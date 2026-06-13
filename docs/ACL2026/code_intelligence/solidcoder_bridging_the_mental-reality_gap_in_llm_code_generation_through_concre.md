---
title: >-
  [Paper Note] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution
description: >-
  [ACL 2026][Code Intelligence][Code Generation] SolidCoder bridges the "Mental-Reality Gap" through the S.O.L.I.D. architecture (Shift-left Planning, Oracle-based Assertions, Live Execution, Intermediate Simulation…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Generation"
  - "Mental Simulation"
  - "Execution Verification"
  - "Multi-Agent"
  - "Property-Based Testing"
date: 2026-05-08
content_hash: 9d855fbefcce237c
---

# SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution

**Conference**: ACL 2026  
**arXiv**: [2604.19825](https://arxiv.org/abs/2604.19825)  
**Code**: [https://github.com/10kH/SolidCoder](https://github.com/10kH/SolidCoder)  
**Area**: Code Generation / LLM Agent  
**Keywords**: Code Generation, Mental Simulation, Execution Verification, Multi-Agent, Property-Based Testing

## TL;DR

SolidCoder bridges the "Mental-Reality Gap" through the S.O.L.I.D. architecture (Shift-left Planning, Oracle-based Assertions, Live Execution, Intermediate Simulation, Defensive Accumulation), shifting code verification from LLM "imagined execution" to "concrete execution." It achieves pass@1 performance of 95.7% on HumanEval, 77.0% on CodeContests, and 26.7% on APPS using GPT-4o.

## Background & Motivation

**Background**: Current state-of-the-art code generation frameworks (e.g., MapCoder, CodeSIM) adopt multi-agent architectures. Among them, CodeSIM utilizes "Mental Simulation" to let the LLM verify correctness by internally tracking code execution, achieving leading results on multiple benchmarks.

**Limitations of Prior Work**: Mental simulation has a fundamental flaw—LLMs suffer from execution hallucinations. In complex algorithmic scenarios, models "imagine" execution trajectories that do not match actual program behavior, confidently validating buggy code. This is akin to playing blindfolded chess and declaring victory. The CodeSIM team attempted to enhance test cases via self-consistency, but performance dropped by 9.3%, leading them to abandon execution verification.

**Key Challenge**: The Mental-Reality Gap unfolds along two orthogonal dimensions: (1) Specification Gap—neglecting edge cases during the planning stage; (2) Verification Gap—hallucinating correct execution trajectories during the verification stage. These issues exist independently, and fixing one does not resolve the other.

**Goal**: Simultaneously bridge the gaps across both dimensions by forcing the model to consider edge cases during planning and replacing imagined execution with concrete execution for verification.

**Key Insight**: The authors observe that the failure of test generation in CodeSIM is not due to the generation itself, but the attempt to predict precise outputs. Verification does not require exact answers—by checking properties (e.g., "output length equals input length," "result is a permutation of the input") rather than precise values, correctness can be judged without an oracle.

**Core Idea**: Replace precise output prediction with property-based assertions and combine this with sandboxed execution—shifting from "imagined" to "executed" (don't imagine, execute).

## Method

### Overall Architecture

SolidCoder is built upon the three-agent architecture of CodeSIM (Planning Agent, Coding Agent, Debugging Agent), adding five S.O.L.I.D. components. The input natural language problem description passes through the Planning Agent to generate edge-case-aware algorithmic plans. The Coding Agent translates plans into code, which undergoes an intermediate simulation check before entering a Live Verification loop: generating property-based test cases, executing in a sandbox, and accumulating failed cases for regression protection, finally outputting code that passes all tests.

### Key Designs

1.  **Shift-left Planning (S)**:
    - **Function**: Forces the identification of edge cases prior to algorithmic planning.
    - **Mechanism**: Prompts the LLM with "What worst-case inputs could break a naive solution?", injecting identified edge cases (empty inputs, boundary values, corner cases) into the planning prompt. Traditional methods handle edge cases reactively during debugging; this method "shifts them left" to the planning stage.
    - **Design Motivation**: Ablation studies show a -23.7%p drop when removing this component, proving edge-case blindness is a primary failure mode in competitive programming.

2.  **Oracle-based Assertions (O) + Live Execution (L)**:
    - **Function**: Replaces mental simulation with property-based verification and concrete execution.
    - **Mechanism**: The Oracle component generates domain-invariant property assertions (e.g., a sorting function should preserve length, maintain order, and produce a permutation), transforming verification from "Is this output correct?" to "Does this output satisfy necessary properties?"—the latter can be answered through execution. Live Execution runs the code in a sandboxed environment (5s timeout, restricted file system), routing to debugging upon assertion failure or runtime error.
    - **Design Motivation**: Solves the "Missing Oracle Problem"—verification without knowing the exact answer. Removing O leads to a -11.6%p drop, while removing L causes a -7.9%p drop.

3.  **Intermediate Simulation (I) + Defensive Accumulation (D)**:
    - **Function**: I serves as a rapid pre-filter after code generation; D prevents regressions during iterative debugging.
    - **Mechanism**: I immediately asks the LLM to trace code on sample inputs after generation. Unlike CodeSIM, I does not provide the final verdict—Live Execution is the authority. D maintains a persistent test suite; every failure found by Live Execution is added to the accumulation set, and every subsequent code modification must pass all accumulated tests to guarantee monotonicity.
    - **Design Motivation**: I acts as a cost-effective pre-filter, and D contributes -6.7%p in regression protection.

### Loss & Training
This is an inference-time framework and does not involve model training. Core hyperparameters include $p=5$ planning iterations, $d=5$ debugging iterations, and $a=3$ assumption-breaking iterations, following CodeSIM settings.

## Key Experimental Results

### Main Results

| Benchmark | Model | CodeSIM | SolidCoder | Gain |
| :--- | :--- | :--- | :--- | :--- |
| HumanEval | GPT-4o | 95.1% | 95.7% | +0.6%p |
| CodeContests | GPT-4o | 72.7% | 77.0% | +4.3%p |
| APPS | GPT-4o | 23.3% | 26.7% | +3.4%p |
| CodeContests | GPT-OSS-120B | 87.9% | 92.1% | +4.2%p |
| CodeContests | Grok-4.1-Fast | 95.2% | 98.2% | +3.0%p |

### Ablation Study (CodeContests, GPT-4o)

| Configuration | Pass@1 | Δ |
| :--- | :--- | :--- |
| Full SolidCoder | 77.0% | – |
| w/o Shift-left Planning [S] | 53.3% | -23.7%p |
| w/o Intermediate Simulation [I] | 64.0% | -13.0%p |
| w/o Oracle-based Assertions [O] | 65.4% | -11.6%p |
| w/o Live Execution [L] | 69.1% | -7.9%p |
| w/o Defensive Accumulation [D] | 70.3% | -6.7%p |
| GPT-4o Direct | 42.4% | -34.6%p |

### Key Findings
- **Shift-left Planning provides the largest contribution** (-23.7%p), proving that edge-case blindness, rather than execution hallucination, is the primary failure mode for algorithmic tasks.
- **Live Execution captures categorically different errors** that mental simulation would incorrectly validate. While its absolute contribution is smaller than [S], these errors cannot be resolved by improving specifications alone.
- Improvement is proportional to difficulty: HumanEval (Easy) only saw +0.6%p, while CodeContests (Medium) saw the largest increase of +4.3%p; APPS (Hard) reaches a bottleneck where the challenge shifts from verification to generation itself.
- Post-RL models (GPT-OSS-120B, Grok-4.1-Fast) also benefit, indicating that even as generation capacity improves, models still rely on mental simulation for self-evaluation at inference time.

## Highlights & Insights
- **Property testing as a replacement for precise output prediction** is the core innovation: transforming an unsolvable oracle problem into an executable property verification problem is ingenious and highly transferable.
- **The two-dimensional analysis framework** (Specification Gap + Verification Gap) clarifies problem analysis, and ablation experiments perfectly validate that the two are independent and complementary.
- **Shift-left thinking originated in software engineering**; moving testing to the planning stage can be transferred to other multi-agent reasoning frameworks, such as mathematical or scientific reasoning tasks.

## Limitations & Future Work
- Live Execution currently only supports Python; extension to other languages requires language-specific sandboxing.
- Evaluation focuses on function-level benchmarks and has not been validated on repository-level tasks (e.g., SWE-bench).
- Systematic bias may propagate when the LLM generates the code, properties, and verification tests simultaneously.
- Significant token overhead: +50% on CodeContests and +97% on HumanEval; difficulty-aware routing could optimize efficiency.
- Ablation studies only cover one combination (CodeContests + GPT-4o).

## Related Work & Insights
- **vs CodeSIM**: CodeSIM uses mental simulation for the final verdict, whereas SolidCoder replaces it with concrete execution. The key difference is that SolidCoder's [I] is only a pre-filter, not the final authority.
- **vs LDB/MGDebugger**: These executable debuggers act as secondary corrections after code generation and require real test cases. SolidCoder integrates execution into the generation loop, replacing real outputs with property assertions.
- **vs Reflexion/LATS**: These use iterative self-correction and tree search, but verification still relies on internal LLM reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The two-dimensional decomposition of the Mental-Reality Gap and the use of property testing to solve the oracle problem are meaningful innovations, though the overall architecture is incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, three models, and a complete ablation; however, the ablation was performed on only one combination.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation; the "blindfolded chess" metaphor is vivid, and the comparison in Figure 2 is intuitive and persuasive.
- Value: ⭐⭐⭐⭐ The property testing approach has high transferability, though token overhead and Python-specific limitations reduce immediate practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)

</div>

<!-- RELATED:END -->
