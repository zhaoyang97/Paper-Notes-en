---
title: >-
  [Paper Note] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning
description: >-
  [ACL 2025][LLM Agent][table reasoning] This paper proposes the Table-Critic multi-agent framework. Through the collaborative criticism and iterative refinement of four specialized agents—Judge, Critic, Refiner, and Curator—coupled with a self-evolving template tree to accumulate criticism knowledge, it achieves 73.7% and 91.7% accuracy on WikiTQ and TabFact, respectively, significantly outperforming existing methods.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "table reasoning"
  - "multi-agent framework"
  - "self-evolving template tree"
  - "collaborative criticism"
  - "iterative refinement"
date: 2026-05-08
content_hash: adc7d1c391243b17
---

# Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.11799](https://arxiv.org/abs/2502.11799)  
**Code**: [https://github.com/Peiying-Yu/Table-Critic](https://github.com/Peiying-Yu/Table-Critic)  
**Area**: LLM Agent  
**Keywords**: table reasoning, multi-agent framework, self-evolving template tree, collaborative criticism, iterative refinement

## TL;DR
This paper proposes the Table-Critic multi-agent framework. Through the collaborative criticism and iterative refinement of four specialized agents—Judge, Critic, Refiner, and Curator—coupled with a self-evolving template tree to accumulate criticism knowledge, it achieves 73.7% and 91.7% accuracy on WikiTQ and TabFact, respectively, significantly outperforming existing methods.

## Background & Motivation

**Background**:
   - LLMs perform exceptionally in various reasoning tasks but still face challenges in table reasoning, especially concerning consistency in multi-step reasoning.
   - Existing methods have achieved progress through decomposition strategies (SQL sub-programs, table partitioning, dynamic chains of operations).
   - Representative methods: Binder decomposes questions into executable SQL/Python, Dater performs sub-table decomposition, and Chain-of-Table generates intermediate tables.

**Limitations of Prior Work**:
   - Decomposition methods lack effective error detection and correction mechanisms for intermediate steps, resulting in error cascade propagation.
   - The self-reflection capabilities of LLMs are unreliable—either rationalizing and defending prior errors, or over-criticizing correct steps.
   - Although Critic-CoT introduces self-reflection, its error-correction rate (+5.6%) is partially offset by a high degradation rate (-4.9%), yielding a net gain of only +0.7%.

**Key Challenge**:
   - Errors in multi-step reasoning must be detected and corrected in a timely manner, yet the self-reflection capability of LLMs remains unstable.
   - Error types are diverse and unpredictable, making it difficult for the model's internal knowledge alone to cover them comprehensively.

**Goal**:
   - How to maintain error-correction capabilities while minimizing interference to correct reasoning.
   - How to systematically accumulate and organize criticism knowledge to handle diverse error types.

**Key Insight**:
   - Decomposing the complex reasoning-correction task into the cooperation of four specialized agents.
   - Introducing a self-evolving template tree as an experience-driven criticism knowledge base.

**Core Idea**:
   - Simulating the human process of "error detection $\rightarrow$ diagnosis $\rightarrow$ correction $\rightarrow$ experience summarizing" using four specialized agents, thereby continuously evolving the criticism knowledge.

## Method

### Overall Architecture
Table-Critic generates an initial reasoning chain based on Chain-of-Table, then iteratively refines it through four agents:
1. Judge detects errors $\rightarrow$ 2. Critic generates criticism suggestions $\rightarrow$ 3. Refiner corrects the reasoning $\rightarrow$ 4. Curator distills experience templates.
The iteration continues until the Judge determines the reasoning is correct or the maximum number of rounds $K=5$ is reached.

### Key Designs

1. **Judge Agent (Error Detection)**:

    - **Function**: Analyses each step of the reasoning chain to detect potential errors and determine error types; routes to the appropriate criticism template in the template tree based on the error type.
    - **Mechanism**: Outputs three elements—error analysis $E$, overall determination $P \in \{\text{Correct}, \text{Incorrect}\}$, and template tree routing path $R$.
    - **Design Motivation**: Diagnosing before treatment provides precise directional guidance for the subsequent Critic; the template routing mechanism ensures that criticism is well-founded.

2. **Critic Agent (Criticism Generation)**:

    - **Function**: Locates the first erroneous step under template guidance, analyzes error details, and generates specific modification suggestions.
    - **Mechanism**: Focuses only on the first erroneous step (instead of all errors) to avoid cascading modifications introducing new errors.
    - **Design Motivation**: LLMs are most adept at identifying and correcting the "first" error, while corrections for subsequent steps are often unreliable.

3. **Refiner Agent (Reasoning Refinement)**:

    - **Function**: Receives the criticism suggestions and the partial reasoning chain truncated at the erroneous step, and regenerates the remaining steps.
    - **Mechanism**: Exposes only the erroneous step and the preceding correct parts to the Refiner, avoiding bias from the subsequent erroneous reasoning.
    - **Design Motivation**: The truncation strategy forces the Refiner to reason anew from the perspective of the criticism instead of simply patching it up.

4. **Curator Agent + Self-evolving Template Tree (Experiential Learning)**:

    - **Function**: Upon completion of the entire refinement process (when the Judge deems the final chain correct), distills criticism templates from the refinement history to update the template tree.
    - **Mechanism**: The template tree supports two directions of expansion—vertical expansion (subdividing existing error categories) and horizontal expansion (adding new error categories).
    - **Design Motivation**: Since error types are an open set, static templates cannot achieve comprehensive coverage; the self-evolving mechanism allows the system to continuously learn new error patterns.
    - Initially starting with only 2 basic templates, the tree autonomously expands through the self-evolving mechanism.

### Loss & Training
- **No Extra Training**: A pure prompt-engineering-based multi-agent framework.
- Maximum number of iterations $K=5$, using greedy decoding with a temperature of 0.0.
- Validated across three models: Qwen2.5-72B, LLaMA3.3-70B, and GPT-4o-mini.

## Key Experimental Results

### Main Results
- **WikiTQ Average Accuracy**: 73.7% (with a **+6.3** Gain compared to 67.4% of Critic-CoT, and a **+7.7** Gain compared to 66.0% of Chain-of-Table).
- **TabFact Average Accuracy**: 91.7% (with a **+3.5** Gain compared to 88.2% of Critic-CoT, and a **+2.2** Gain compared to 89.5% of Chain-of-Table).
- **Best Performance on Qwen2.5-72B**: WikiTQ 77.2% (+8.2 vs Chain-of-Table), TabFact 92.6% (+2.6).
- Results are consistent across all three LLMs, demonstrating the model-agnostic nature of the framework.

### Ablation Study & Key Findings

**Error Correction Capability Analysis (Core Finding)**:
- **Table-Critic on WikiTQ**: Correction rate $\Delta^{i \to c} = +9.6\%$, degradation rate $\Delta^{c \to i} = -0.7\%$, and net gain **+8.9%**.
- **Critic-CoT on WikiTQ**: Correction rate $+5.6\%$, degradation rate $-4.9\%$, with a net gain of only $+0.7\%$.
- **Similar Pattern on TabFact**: Table-Critic net gain $+2.9\%$ vs. Critic-CoT $+0.1\%$.
- **Key Insight**: The degradation rate of Table-Critic is extremely low ($-0.7\% / -0.5\%$), indicating that the self-evolving template tree effectively protects correct reasoning.

**Analysis of Multi-round Mechanism**:
- On WikiTQ, the accuracy swiftly rises from $67.6\%$ to $76.5\%$ in the first 3 rounds, stabilizing at $\sim 77\%$ after the 6th round.
- A similar trend is observed on TabFact, stabilizing at $\sim 92\%$ after around 5 rounds.
- Setting $K=5$ in practice serves as a reasonable trade-off.

**Analysis of Computational Cost**:
- The computational cost of Table-Critic is approximately $1.8\times - 2.2\times$ that of Chain-of-Table.
- However, even with 15 self-consistency samples (requiring much higher computational cost), Chain-of-Table only reaches $70.0\%$ (vs. $77.2\%$ for Table-Critic) on WikiTQ and $90.1\%$ (vs. $92.6\%$) on TabFact.

**Ablation of Self-evolving Template Tree**:
- Removing the self-evolving mechanism drops the performance on WikiTQ by $1.1\%$ ($77.2\% \to 76.1\%$) and on TabFact by $1.8\%$ ($92.6\% \to 90.8\%$).
- This demonstrates that dynamic template extension is crucial for handling diverse error types.

## Highlights & Insights

- **Control of Degradation Rate**: This is the strongest selling point: correcting errors while barely disrupting correct answers (WikiTQ degradation rate is only $-0.7\%$, substantially superior to Critic-CoT's $-4.9\%$).
- **"Focusing on the First Error" Strategy**: Tactfully leverages the property that LLMs are most accurate at identifying the first error, resolving them one by one through multi-round iterations.
- **Self-evolving Template Tree**: Transforms "prior mistakes" into "correction guidelines", representing a lightweight empirical learning mechanism.
- **Computational Efficiency Advantage**: A $1.8\times$ cost incurs a **+7+ points** Gain in accuracy, which is much more efficient than brute-force majority voting.

## Limitations & Future Work

- Currently only targets textual table reasoning and has not been extended to multimodal scenarios (charts + tables).
- The quality of the template tree depends on the summarization capabilities of the Curator Agent, which might introduce erroneous templates.
- The maximum number of iterations $K=5$ is a hyperparameter; different tasks may require different settings.
- All four agents share the same backbone LLM, with differences in capability differentiated solely through prompts.
- More complex scenarios like cross-table and multi-relation table reasoning are not yet considered.

## Related Work & Insights

- **Binder $\rightarrow$ Dater $\rightarrow$ Chain-of-Table**: Gradual evolution of decomposition strategies, yet all lack error correction mechanisms.
- **Critic-CoT**: Introduces self-reflection but is unstable; Table-Critic resolves the stability issues through multi-agent collaboration and the template tree.
- **Self-refine (Madaan et al., 2023)**: LLM self-reflection has inherent limitations; multi-agent division of labor offers a better solution.
- **Insight**: "Each agent to its own specialty" task division + dynamic knowledge accumulation represents the key pathway to surpassing single-model self-reflection.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)

</div>

<!-- RELATED:END -->
