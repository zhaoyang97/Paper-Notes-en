---
title: >-
  [Paper Note] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution
description: >-
  [ACL 2026][Code Intelligence][code generation] SolidCoder transforms code verification from LLM "imagined execution" to "real execution" via the S.O.L.I.D. architecture (Shift-left Planning, Oracle-based Assertions…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "code generation"
  - "mental simulation"
  - "execution verification"
  - "multi-agent"
  - "property-based testing"
date: 2026-05-08
content_hash: e60b0f96b94db6b7
---

# SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution

**Conference**: ACL 2026
**arXiv**: [2604.19825](https://arxiv.org/abs/2604.19825)  
**Code**: [https://github.com/10kH/SolidCoder](https://github.com/10kH/SolidCoder)  
**Area**: Code Generation / LLM Agent
**Keywords**: code generation, mental simulation, execution verification, multi-agent, property-based testing

## TL;DR

SolidCoder transforms code verification from LLM "imagined execution" to "real execution" via the S.O.L.I.D. architecture (Shift-left Planning, Oracle-based Assertions, Live Execution, Intermediate Simulation, Defensive Accumulation), achieving pass@1 scores of 95.7% on HumanEval, 77.0% on CodeContests, and 26.7% on APPS with GPT-4o.

## Background & Motivation

**Background**: State-of-the-art code generation frameworks (e.g., MapCoder, CodeSIM) adopt multi-agent architectures. In particular, CodeSIM employs "Mental Simulation," in which the LLM internally traces code execution to verify correctness, achieving leading results on multiple benchmarks.

**Limitations of Prior Work**: Mental simulation has a fundamental flaw—LLMs suffer from execution hallucinations. In complex algorithmic scenarios, models "imagine" execution traces that diverge from actual program behavior, confidently validating buggy code. This is akin to playing chess blindfolded and declaring victory. The CodeSIM team previously attempted to augment test cases via self-consistency, only to observe a 9.3% performance drop, and subsequently abandoned execution-based verification.

**Key Challenge**: The Mental-Reality Gap unfolds along two orthogonal dimensions: (1) Specification Gap—edge cases are overlooked during planning; (2) Verification Gap—correct execution traces are hallucinated during validation. These two issues are independent; fixing one does not resolve the other.

**Goal**: To simultaneously close the gap along both dimensions—enabling the model to account for edge cases during planning while replacing imagined execution with real execution for verification.

**Key Insight**: The authors observe that the failure of test generation in CodeSIM stems not from test generation per se, but from attempting to predict exact outputs. Verification does not require precise answers—by checking properties (e.g., "output length equals input length," "result is a permutation of the input") rather than exact values, correctness can be assessed without an oracle.

**Core Idea**: Replace exact output prediction with property-based assertions, combined with sandbox execution, to shift verification from "imagination" to "execution"—don't imagine, execute.

## Method

### Overall Architecture

SolidCoder builds upon CodeSIM's three-agent architecture (Planning Agent, Coding Agent, Debugging Agent) by incorporating five S.O.L.I.D. components. Given a natural language problem description, the Planning Agent produces an algorithm plan with edge-case awareness; the Coding Agent translates the plan into code and applies intermediate simulation checks; the code then enters a Live Verification loop in which property-based test cases are generated, executed in a sandbox, and failing cases are accumulated for regression protection—ultimately outputting code that passes all tests.

### Key Designs

1. **Shift-left Planning (S)**:

    - **Function**: Forces identification of edge cases prior to algorithm planning.
    - **Mechanism**: The LLM is prompted with "what worst-case inputs could break a naïve solution?" The identified edge cases (empty inputs, boundary values, corner cases) are injected into the planning prompt, compelling the model to design robust algorithms from the outset. Whereas conventional approaches reactively address edge cases during debugging, this method "shifts left" their handling to the planning stage.
    - **Design Motivation**: Ablation experiments show that removing this component results in a −23.7%p drop, demonstrating that edge-case blindness is the dominant failure mode in competitive programming problems.

2. **Oracle-based Assertions (O) + Live Execution (L)**:

    - **Function**: Replaces mental simulation with property-based verification and real execution.
    - **Mechanism**: The Oracle component generates domain-invariant property assertions (e.g., a sorting function should preserve length, maintain order, and produce a permutation), shifting verification from "is this output correct?" to "does this output satisfy the necessary properties?"—a question answerable through execution. Live Execution runs the code in a sandboxed environment (5-second timeout, restricted filesystem); assertion failures or runtime errors are routed to the debugger.
    - **Design Motivation**: Addresses the "missing oracle problem"—code can be verified without knowing the correct answer. Removing O results in a −11.6%p drop; removing L results in a −7.9%p drop.

3. **Intermediate Simulation (I) + Defensive Accumulation (D)**:

    - **Function**: I serves as a rapid pre-filter after code generation; D prevents regressions during iterative debugging.
    - **Mechanism**: Immediately after code generation, I prompts the LLM to trace the code on sample inputs. Unlike CodeSIM, I does not render a final verdict—Live Execution serves as the authoritative judge. D maintains a persistent test suite; every failing case discovered by Live Execution is added to the accumulated set, and every code revision must pass all accumulated tests, providing a monotonicity guarantee.
    - **Design Motivation**: I acts as a cost-efficient pre-filter; D contributes −6.7%p in regression protection.

### Loss & Training
This paper presents an inference-time framework and does not involve model training. The core hyperparameters are $p=5$ planning iterations, $d=5$ debugging iterations, and $a=3$ hypothesis-breaking iterations, all inherited from CodeSIM.

## Key Experimental Results

### Main Results

| Benchmark | Model | CodeSIM | SolidCoder | Gain |
|-----------|-------|---------|------------|------|
| HumanEval | GPT-4o | 95.1% | 95.7% | +0.6%p |
| CodeContests | GPT-4o | 72.7% | 77.0% | +4.3%p |
| APPS | GPT-4o | 23.3% | 26.7% | +3.4%p |
| CodeContests | GPT-OSS-120B | 87.9% | 92.1% | +4.2%p |
| CodeContests | Grok-4.1-Fast | 95.2% | 98.2% | +3.0%p |

### Ablation Study (CodeContests, GPT-4o)

| Configuration | Pass@1 | Δ |
|---------------|--------|---|
| Full SolidCoder | 77.0% | – |
| w/o Shift-left Planning [S] | 53.3% | −23.7%p |
| w/o Intermediate Simulation [I] | 64.0% | −13.0%p |
| w/o Oracle-based Assertions [O] | 65.4% | −11.6%p |
| w/o Live Execution [L] | 69.1% | −7.9%p |
| w/o Defensive Accumulation [D] | 70.3% | −6.7%p |
| GPT-4o Direct | 42.4% | −34.6%p |

### Key Findings
- **Shift-left Planning contributes the most** (−23.7%p), demonstrating that edge-case blindness—rather than execution hallucination—is the dominant failure mode in algorithmic tasks.
- **Live Execution captures a categorically distinct class of errors** that mental simulation incorrectly validates. Although its absolute contribution is smaller than [S], such errors cannot be resolved by improving specifications alone.
- Gains are proportional to task difficulty: HumanEval (easy) yields only +0.6%p; CodeContests (medium) shows the largest gain at +4.3%p; APPS (hard) shifts the bottleneck from verification to generation itself.
- RL post-trained models (GPT-OSS-120B, Grok-4.1-Fast) also benefit, indicating that even as generation capability improves, models still rely on mental simulation for self-evaluation at inference time.

## Highlights & Insights
- **Replacing exact output prediction with property-based testing** is the core innovation: it reformulates the intractable oracle problem as an executable property verification problem—an elegant insight with broad transferability.
- **The two-dimensional decomposition** (Specification Gap + Verification Gap) yields a clear analytical framework, and the ablation experiments cleanly validate the independence and complementarity of the two dimensions.
- **The shift-left concept originates in software engineering**; applying it to the planning stage of multi-agent reasoning frameworks is transferable to other domains such as mathematical reasoning or scientific inference tasks.

## Limitations & Future Work
- Live Execution currently supports only Python; extension to other languages requires language-specific sandboxing.
- Evaluation focuses on function-level benchmarks and has not been validated on repository-level tasks (e.g., SWE-bench).
- When the LLM simultaneously generates code, properties, and verification tests, systematic biases may propagate.
- Token overhead is significant: +50% on CodeContests and +97% on HumanEval; difficulty-aware routing could be explored to improve efficiency.
- Ablation experiments cover only one combination (CodeContests + GPT-4o).

## Related Work & Insights
- **vs. CodeSIM**: CodeSIM uses mental simulation as the final arbiter; SolidCoder replaces it with real execution. The key distinction is that SolidCoder's [I] serves only as a pre-filter, not as a final judge.
- **vs. LDB/MGDebugger**: These execution-based debuggers operate as post-hoc correctors after code generation and require ground-truth test cases. SolidCoder integrates execution into the generation loop and substitutes property assertions for exact outputs.
- **vs. Reflexion/LATS**: These methods leverage iterative self-correction and tree search, but verification still relies on LLM internal reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The two-dimensional decomposition of the Mental-Reality Gap and the use of property-based testing to address the oracle problem represent meaningful contributions, though the overall architecture is incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks, three models, and a complete ablation study; however, ablations are conducted on only one combination.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; the "blindfolded chess" analogy is vivid; the comparative example in Figure 2 is intuitive and persuasive.
- **Value**: ⭐⭐⭐⭐ The property-based testing paradigm has broad transferability, though token overhead and Python-only constraints reduce practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)

</div>

<!-- RELATED:END -->
