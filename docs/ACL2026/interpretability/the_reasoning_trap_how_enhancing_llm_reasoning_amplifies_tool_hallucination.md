---
title: >-
  [Paper Note] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination
description: >-
  [ACL 2026][Interpretability][Tool Hallucination] This paper systematically reveals the "reasoning trap" paradox: enhancing LLM reasoning capabilities — whether through RL, distillation…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Tool Hallucination"
  - "Reasoning Enhancement"
  - "Reinforcement Learning"
  - "Reliability-Capability Trade-off"
  - "LLM Agents"
date: 2026-05-08
content_hash: ae20d8abadc5aabc
---

# The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination

**Conference**: ACL 2026  
**arXiv**: [2510.22977](https://arxiv.org/abs/2510.22977)  
**Code**: [GitHub](https://github.com/albert-y1n/Reasoning_Trap)  
**Area**: Interpretability  
**Keywords**: Tool Hallucination, Reasoning Enhancement, Reinforcement Learning, Reliability-Capability Trade-off, LLM Agents

## TL;DR

This paper systematically reveals the "reasoning trap" paradox: enhancing LLM reasoning capabilities — whether through RL, distillation, or switchable reasoning modes — systematically amplifies tool hallucination. This effect is associated with reasoning itself rather than RL training, and existing mitigation strategies (prompt engineering, DPO) face an unavoidable reliability-capability trade-off.

## Background & Motivation

**Background**: LLMs have evolved from text generators into "think-then-act" agents that continuously improve planning and tool-use capabilities through reasoning enhancement (RL, distillation, etc.), which is a central path toward building reliable AI agents.

**Limitations of Prior Work**: More capable reasoning models such as OpenAI o3 exhibit more severe hallucination tendencies, yet no prior work has systematically examined whether reasoning enhancement itself causes tool hallucination — i.e., models fabricating nonexistent tools or misusing irrelevant ones.

**Key Challenge**: Intuitively, stronger reasoning should yield greater reliability, but experimental observations reveal the opposite — stronger reasoning coexists with higher tool hallucination rates. This is not merely an overfitting issue, since training RL on non-tool-related tasks (e.g., mathematics) also amplifies tool hallucination.

**Goal**: Answer three core questions — (RQ1) Does reasoning enhancement increase tool hallucination? (RQ2) What is the underlying mechanism? (RQ3) Can it be effectively mitigated?

**Key Insight**: Construct a lightweight diagnostic benchmark, SimpleToolHalluBench, and use controlled experiments to progressively rule out alternative explanations, ultimately attributing the cause to reasoning itself.

**Core Idea**: Reasoning chain training induces a behavioral pattern of "confidently filling in gaps." When applied to tool-use scenarios, this pattern naturally manifests as tool hallucination — the model tends to generate plausible-sounding but ungrounded tool calls.

## Method

### Overall Architecture

The study proceeds in four steps to progressively eliminate alternative hypotheses: (1) verify that tool-related RL increases hallucination; (2) verify that non-tool RL (mathematics) also increases hallucination (ruling out overfitting); (3) verify that distillation and switchable reasoning modes also increase hallucination (ruling out RL specificity); (4) ablation to disentangle reasoning steps vs. RL training itself. Mechanism analysis (representation collapse + activation probing) follows, and mitigation strategies are then evaluated.

### Key Designs

1. **SimpleToolHalluBench Diagnostic Benchmark**:

    - **Function**: Precisely measure tool hallucination rates.
    - **Mechanism**: Designs two controlled scenarios — NTA (No Tool Available: no tools are provided in the system prompt, yet the user query requires one) and DT (Distractor Tools: irrelevant tools are provided). 296 tools with corresponding queries are included; each query can only be correctly answered via its specific tool, ensuring that any tool call under NTA/DT settings constitutes a hallucination.
    - **Design Motivation**: Existing benchmarks focus on whether models can correctly invoke tools, overlooking the equally critical question of whether models can refrain from invoking tools when they should not.

2. **Four-Step Causal Elimination Experimental Design**:

    - **Function**: Precisely attribute the cause of tool hallucination to reasoning itself.
    - **Mechanism**: (a) Tool RL increases hallucination → possibly overfitting; (b) Math RL also increases hallucination → rules out overfitting; (c) Distillation/switching mode also increases → rules out RL specificity; (d) Ablation of reasoning steps: removing `<think>` blocks only slightly increases hallucination (34.8→41.4), whereas retaining `<think>` blocks causes a dramatic increase (34.8→90.2) → reasoning steps are the core factor.
    - **Design Motivation**: Avoid simple correlational conclusions; establish stronger causal evidence through systematic ablation.

3. **Mechanism Analysis via Representation Collapse and Activation Localization**:

    - **Function**: Reveal the internal mechanism by which reasoning enhancement affects tool behavior.
    - **Mechanism**: (a) CKA is used to measure layer-wise representational similarity before and after RL — in-domain representations remain stable (CKA > 0.9), while tool-related representations drift sharply in early and middle layers (CKA < 0.75); (b) linear probes localize hallucination signals — correct vs. hallucinated responses are most linearly separable in late residual streams (discrimination score > 0.14), while attention and MLP outputs are nearly inseparable.
    - **Design Motivation**: Go beyond "what" to address "why" and "where," providing a foundation for future targeted interventions.

### Mitigation Strategy Evaluation

Prompt engineering (explicitly instructing the model to "not use tools not provided") has negligible effect (NTA: 90.2→87.5). DPO (preference alignment between "honest responses" and "hallucinated responses") is effective but costly (NTA: 90.2→55.8, while SynTool reward drops from 0.45 to 0.34).

## Key Experimental Results

### Main Results

| Model / Configuration | R_NTA (↓) | R_DT (↓) | Notes |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 34.8 | 54.7 | Baseline |
| + ReCall RL (tool) | 90.2 | 100.0 | Large increase from tool RL |
| + GRPO (math) | ↑ | ↑ | Non-tool RL also increases |
| R1-Distill-Qwen-7B | 74.3 | 78.7 | Distillation increases |
| Qwen3-8B Think Off | 4.1 | 36.2 | Reasoning disabled |
| Qwen3-8B Think On | 5.4 | 56.8 | Reasoning enabled increases |

### Ablation Study

| Configuration | R_NTA | R_DT | Reward |
|---|---|---|---|
| Baseline | 34.8 | 54.7 | 0.22 |
| Direct tool RL (no reasoning) | 41.4 | 63.6 | 0.28 |
| Think-then-act RL | 90.2 | 100.0 | 0.45 |
| + Prompt engineering | 87.5 | 98.9 | 0.44 |
| + DPO | 55.8 | 71.4 | 0.34 |

### Key Findings
- Reasoning enhancement consistently increases tool hallucination across all tested methods (RL / distillation / switchable modes).
- Training RL on purely mathematical tasks also increases tool hallucination, ruling out the overfitting hypothesis.
- Ablation results show that reasoning steps (the `<think>` block) — rather than RL training per se — are the core factor.
- Instruction-following capability remains stable (IFEval: −2.6%), and tool-calling capability even improves (BFCL: +9.9%), yet hallucination increases dramatically — demonstrating that tool hallucination is an independent failure mode.
- DPO mitigation is effective but entails an unavoidable capability-reliability trade-off.

## Highlights & Insights
- **Reveals a profound paradox**: Reasoning enhancement makes models "smarter but less honest," issuing a fundamental warning to all current research trajectories that pursue reasoning scaling.
- **Exemplary experimental design**: The four-step elimination method systematically builds causal evidence with rigorous logic.
- **Mechanistic analysis with depth**: CKA representation analysis combined with activation probe localization answers not only "what" but also "why" and "where."
- **Core insight**: Tool hallucination is neither overfitting nor instruction-following degradation — it is an intrinsic side effect of reasoning enhancement.

## Limitations & Future Work
- **Focus on single-step tool calls only**: Real-world agents involve multi-step tool chains, where hallucination effects may compound.
- **Incomplete causality**: The mechanism analysis reveals correlational patterns but does not provide a complete causal explanation.
- **Limited mitigation strategies**: Only prompt engineering and DPO are evaluated; approaches such as process supervision and constitutional AI remain unexplored.
- Future work should develop training objectives that jointly optimize capability and reliability, rather than applying post-hoc fixes.

## Related Work & Insights
- **vs. ToolBeHonest**: Focuses on diagnostic evaluation of tool use but does not investigate the relationship between reasoning enhancement and hallucination.
- **vs. ReCall**: A state-of-the-art agent reasoning RL framework; this paper exposes its "hidden cost."
- **vs. DeepSeek-R1**: Transfers reasoning capabilities through distillation; this paper demonstrates that hallucination tendencies are transferred alongside.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic establishment of the link between reasoning enhancement and tool hallucination; the finding is highly significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four-step elimination + mechanism analysis + mitigation evaluation; the experimental design is exceptionally rigorous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical chain; each experiment answers a specific question in a progressive manner.
- **Value**: ⭐⭐⭐⭐⭐ Issues a fundamental warning to current reasoning scaling trajectories; carries important implications for agent safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](../../ICLR2026/interpretability/the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [\[ACL 2026\] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)
- [\[ICLR 2026\] Position: The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Advanced AI Self-Awareness](../../ICLR2026/interpretability/the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](reasoning_fails_where_step_flow_breaks.md)

</div>

<!-- RELATED:END -->
