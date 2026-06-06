---
title: >-
  [Paper Note] Are Large Reasoning Models Interruptible?
description: >-
  [ICML 2026][LLM Reasoning][Interruptible reasoning] This paper shifts the evaluation of large reasoning models from static problems to dynamic environments where models are interrupted by users or receive mid-way updates…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Interruptible reasoning"
  - "dynamic context"
  - "long reasoning models"
  - "reasoning robustness"
  - "evaluation benchmarks"
date: 2026-05-08
content_hash: 9da5e738adbc4b09
---

# Are Large Reasoning Models Interruptible?

**Conference**: ICML 2026  
**arXiv**: [2510.11713](https://arxiv.org/abs/2510.11713)  
**Code**: Authors released code and data, see paper project page  
**Area**: LLM Reasoning  
**Keywords**: Interruptible reasoning, dynamic context, long reasoning models, reasoning robustness, evaluation benchmarks  

## TL;DR
This paper shifts the evaluation of large reasoning models from static problems to dynamic environments where models are interrupted by users or receive mid-way updates. It constructs math and programming evaluation protocols and identifies three stable failure modes: reasoning leakage, panicked answering, and self-doubt.

## Background & Motivation
**Background**: Large reasoning models typically generate long explicit reasoning trajectories before providing a final answer. Existing math and code evaluations mostly assume that the problem, context, and user goals remain constant during generation, requiring the model to deliver the answer only after a complete generation.

**Limitations of Prior Work**: Real-world interactions are not always static. Users may want partial answers immediately, discover errors in the original request and insert new conditions, or change the environment state in collaborative codebases. If every change requires terminating generation, manually modifying the context, and restarting, it wastes computation and discards already formed intermediate reasoning.

**Key Challenge**: Long reasoning improves static accuracy but exposes models to a longer interaction window. Evaluations that only look at answers after a full trajectory mask a critical capability: whether a model can robustly stop, compress, redirect, or absorb new information when reasoning is incomplete.

**Goal**: The authors aim to answer three questions. First, whether models exhibit anytime algorithm-like properties (more reasoning leading to better answers) when hard-interrupted. Second, whether models can maintain correctness while reducing reasoning length when receiving speed-up instructions. Third, whether models can identify and incorporate mid-way updates into subsequent reasoning.

**Key Insight**: Instead of proposing a new training algorithm, the paper defines "interruptibility" as an independent evaluation object. It places interruption points within the model's existing reasoning trajectory and compares accuracy and output length under no-interruption, hard-interruption, soft-interruption, and dynamic update conditions.

**Core Idea**: Replace static one-shot evaluations with a controlled mid-way intervention protocol to directly measure the stability of long reasoning models under time constraints and context changes.

## Method
The methodology is essentially an evaluation framework: models generate a complete reasoning trajectory for standard problems, then interruptions or updates are inserted at different percentage positions of the trajectory. Subsequent answers, lengths, and error types are observed. The concern is not how to make the model think less, but whether the model can continue working correctly when the environment has changed.

### Overall Architecture
Given a query $q$, in static evaluation, model $M$ outputs a reasoning trajectory $r=(r_1, r_2, \dots, r_T)$ and a final answer $a$. Dynamic evaluation splits generation into two phases: the first generates up to a ratio $X$, yielding prefix $r_{:X}$; the second inserts an intervention marker $i$ or update $u$ into the input, allowing the model to generate the remaining trajectory $r'_{X:}$ and answer $a'$.

Two types of metrics are used. Accuracy is denoted as $A_i(X)=Pr[a'=a^* \mid X,i]$, measuring if the final answer is correct under interruption. Length is denoted as $L_i(X)=|r'_{X:}\oplus a'|$, serving as a proxy for the additional computation after interruption. This reveals whether models covertly move reasoning that should have stopped into the answer area.

Experiments cover math and programming tasks. Math includes a 500-item subset of GSM8K, MATH-500, and AIME-24/25; programming uses LiveCodeBench-v6, filtered for problems after October 1, 2024. Main models include Qwen3-8B, GPT-OSS-20B high reasoning effort, and Mistral-Small-1.2. Interruption positions use relative reasoning length $X\in\{0.1,0.3,\dots,0.9\}$.

### Key Designs
1.  **Two types of time-constrained interruptions**:
    - **Function**: Simulates scenarios where users want models to "answer now" or "answer faster."
    - **Mechanism**: Hard-interruption truncates reasoning and inserts end-of-thought or forced-answer markers, pushing the model directly to the answer area; soft-interruption inserts instructions like "Please answer faster," allowing continued thinking but expecting reduced trajectory length.
    - **Design Motivation**: This distinguishes whether partial reasoning has formed usable answers versus the model's ability to actively regulate its reasoning budget under pressure.

2.  **Dynamic context update protocol**:
    - **Function**: Tests if the model can re-align with problem definitions when receiving new facts, constraints, or corrections mid-way.
    - **Mechanism**: In math tasks, initial conditions are modified, and then the original semantics are restored via mid-way updates. In programming, textual descriptions are given first, followed by updates like starter code, variable ranges, or extra constraints. All updates are verified by humans to ensure they are necessary for solving the problem.
    - **Design Motivation**: If a model works reliably in dynamic environments, it should not just follow the old trajectory inertia but judge if new information changes the goal and explicitly absorb it.

3.  **Error mode taxonomy and lightweight mitigation**:
    - **Function**: Breaks down accuracy drops into interpretable behavioral types.
    - **Mechanism**: Three failures are identified: reasoning leakage (writing thinking into the answer area after hard-interruption), panicked answering (ending reasoning immediately with incorrect answers after speed-up instructions), and self-doubt (refusing to trust updates and sticking to old or confused answers). For self-doubt, prompt guidance is appended to confirm the update is verified.
    - **Design Motivation**: These categories correspond to different fix directions: stop-control for leakage, robust budget strategies for panic, and context-trust mechanisms for self-doubt.

### Loss & Training
Ours is not a training method and introduces no new loss functions. All experiments are inference-time evaluations. The main variables are interruption type, position, update form, the presence of prompt guidance, and model scaling. AIME-24/25 runs 16 independent trials per problem due to small sample size; other datasets use single runs with bootstrap 95% confidence intervals.

## Key Experimental Results

### Main Results
The primary conclusion is that models with high static performance deteriorate systematically under dynamic conditions. Under hard-interruption, models generally show anytime behavior (later is better), but early interruptions cause significant reasoning leakage. Soft-interruption triggers panicked answering on hard tasks (AIME/LiveCodeBench). Dynamic updates are the most fragile, with performance dropping up to 60% for late-stage updates.

| Scenario | Primary Evaluation Target | Key Result | Description |
|----------|---------------------------|------------|-------------|
| Hard-interruption | GSM8K / MATH-500 / AIME / LiveCodeBench | Accuracy rises as interruption occurs later | Partial reasoning is valuable, but early stopping is unstable |
| Hard-interruption Length | AIME / LiveCodeBench | Late answers can be 10x longer than full-thinking answers | Models leak reasoning into final answers or code comments |
| Soft-interruption | AIME / LiveCodeBench | Accuracy drops by up to 30% | Speed-up instructions cause premature termination of thought |
| Dynamic Update | Math and Coding Update Tasks | Performance drops by up to 60% for late updates | Static evaluation significantly overestimates dynamic robustness |
| Prompt Guidance | GSM8K / MATH-500 | Eliminates major issues from updates | Brief confirmations mitigate self-doubt in simple math tasks |

### Ablation Study
Ablations show these failures are not caused by single implementation details. Model scale, turn insertion methods, and compact reasoning (Chain of Draft) change the curves, but the three pathological phenomena persist.

| Ablation / Analysis | Setting | Key Metric or Observation | Description |
|---------------------|---------|---------------------------|-------------|
| Leakage Attribution | 30% Hard-interruption | up to 10x answer inflation in failure cases | Stopping the thinking block does not guarantee stopping reasoning |
| Panic Attribution | 30% Soft-interruption | Over 90% of new errors from panic; up to 80% of total loss | "Faster" is interpreted as "End immediately" |
| Self-doubt Attribution | 30% Dynamic Update | ~80% of update-driven errors relate to self-doubt | Models do not always trust mid-way updates |
| Chain of Draft | Qwen3-8B, 30% Hard | Answer length still 1.38x (AIME), 6.27x (LCB) | Compressed reasoning does not solve interruptibility |
| Chain of Draft Soft | AIME, 30% Interruption | Panic rate 13.1% (vs 3.8% in assistant-turn) | Shorter draft-like reasoning might be more prone to sudden stops |
| Dynamic Update Cost | Prompt Guidance Setting | GPT-OSS late update cost < 110% of original | Continuation is cheaper than restarting if updates are absorbed |

### Key Findings
- Static high accuracy does not imply dynamic robustness. Strong performance on fixed problems does not translate to handling mid-way changes.
- Reasoning token curves underestimate actual computation. The answer area may carry significant implicit reasoning after hard-interruption, so counting only thinking tokens misjudges efficiency.
- Prompt guidance suggests room for improvement. Simple confirmations help GSM8K/MATH-500, but AIME and code tasks remain far from solved.
- Model scale is not a panacea. Larger Qwen models perform better on update tasks, but hard and soft interruptions do not show a monotonic scaling solution.

## Highlights & Insights
- The most valuable contribution is Transforming "interruptibility" from an engineering experience issue into a measurable capability. It highlights that models in real deployment are reasoning processes in a changing world.
- The three error modes are highly explanatory. Reasoning leakage, panicked answering, and self-doubt correspond to gaps in stop-control, budget regulation, and context trust, respectively.
- The dynamic update construction is clever. The "modify then restore" approach in math tasks ensures the model is truly absorbing updates rather than relying on memorized answers.
- Insights for agent systems: Multi-agent collaboration and IDE agents naturally involve interruptions and environment changes, making this benchmark more representative of interactive deployment risks than traditional one-shot math problems.

## Limitations & Future Work
- Task scope is narrow. Primarily covers math and coding due to long trajectories and automated verification; multi-turn QA, research assistants, and tool-use are not yet included.
- Interruption forms are idealized. Experiments use single, clean, pre-set interruptions, whereas real users might provide noisy, continuous, or contradictory updates.
- Incomplete closed-source evaluation. Many APIs do not expose internal reasoning or allow mid-trace updates, necessitating approximate proxies.
- Diagnostic focus. The paper does not propose interruption-aware training or new decoding constraints, which remains future work for post-training rewards.
- Scale of manual updates. Although verified, larger-scale dynamic tasks require more systematic generation and verification processes.

## Related Work & Insights
- **vs Budget Forcing / S1**: While budget forcing studies fixed token budgets, Ours emphasizes unpredictable interruptions and evaluates whether reasoning leaks into the answer area.
- **vs NoThinking**: While NoThinking studies removing explicit thought, Ours truncates thoughts mid-way to see if partial trajectories support reliable answers.
- **vs Chain of Draft**: Ours finds that compact reasoning does not automatically eliminate leakage, panic, or self-doubt.
- **vs Efficient Reasoning Training**: Reminds researchers to optimize for "correct even when interrupted" alongside "correct with fewer tokens."

## Rating
- Novelty: ⭐⭐⭐⭐☆ Defines interruptibility as a systematic evaluation problem for long-reasoning models with a clear taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple datasets, models, and ablation settings with solid reliability.
- Writing Quality: ⭐⭐⭐⭐☆ Clear problem consciousness and intuitive naming of failure modes.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for interactive LLMs, IDE agents, and long-task reasoning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)
- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)
- [\[ICML 2026\] Reasoning Structure of Large Language Models](reasoning_structure_of_large_language_models.md)
- [\[ICML 2026\] Modeling Hierarchical Thinking in Large Reasoning Models](modeling_hierarchical_thinking_in_large_reasoning_models.md)
- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](../../AAAI2026/llm_reasoning/text-to-scene_with_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
