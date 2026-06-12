---
title: >-
  [Paper Note] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination
description: >-
  [ACL 2026][Hallucination Detection][Tool Hallucination] Systematically reveals the "Reasoning Trap" paradox: enhancing LLM reasoning capabilities (whether via RL, distillation…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Tool Hallucination"
  - "Reasoning Enhancement"
  - "Reinforcement Learning"
  - "Reliability-Capability Trade-off"
  - "LLM Agents"
date: 2026-05-08
content_hash: e2fc92496d80223a
---

# The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination

**Conference**: ACL 2026  
**arXiv**: [2510.22977](https://arxiv.org/abs/2510.22977)  
**Code**: [GitHub](https://github.com/albert-y1n/Reasoning_Trap)  
**Area**: Hallucination Detection  
**Keywords**: Tool Hallucination, Reasoning Enhancement, Reinforcement Learning, Reliability-Capability Trade-off, LLM Agents

## TL;DR

Systematically reveals the "Reasoning Trap" paradox: enhancing LLM reasoning capabilities (whether via RL, distillation, or switchable reasoning modes) systematically amplifies tool hallucination. This effect is associated with reasoning itself rather than RL training, and existing mitigation strategies (Prompt Engineering, DPO) face an inevitable reliability-capability trade-off.

## Background & Motivation

**Background**: LLMs have evolved from text generators into "think-before-acting" agents. Through reasoning enhancement (RL, distillation, etc.), their planning and tool-use capabilities are continuously improved, which is a core path for building reliable AI Agents.

**Limitations of Prior Work**: Stronger reasoning models like OpenAI o3 exhibit more severe hallucination tendencies. However, no prior research has systematically examined whether reasoning enhancement itself leads to tool hallucination—namely, models fabricating non-existent tools or incorrectly using irrelevant tools.

**Key Challenge**: Intuitively, stronger reasoning capabilities should lead to higher reliability. However, experimental observations suggest the opposite—stronger reasoning coexists with higher tool hallucination rates. This is not merely an overfitting issue, as RL training even on non-tool tasks (e.g., mathematics) amplifies tool hallucination.

**Goal**: Answer three core questions—(RQ1) Does reasoning enhancement increase tool hallucination? (RQ2) What is the mechanism? (RQ3) Can it be effectively mitigated?

**Key Insight**: Construct a lightweight diagnostic benchmark, SimpleToolHalluBench, to systematically exclude alternative explanations through controlled experiments and eventually isolate the cause to reasoning itself.

**Core Idea**: Reasoning chain training induces a behavioral pattern of "confidently filling the gap." When placed in tool-use scenarios, this pattern naturally manifests as tool hallucination—the model tends to generate plausible-looking but groundless tool calls.

## Method

### Overall Architecture

The study proceeds in four steps to exclude alternative hypotheses: (1) Verifying that tool-related RL increases hallucination; (2) Verifying that non-tool RL (Math) also increases hallucination (excluding overfitting); (3) Verifying that distillation and switchable reasoning modes also increase hallucination (excluding RL specificity); (4) Ablation studies separating reasoning steps vs. RL training itself. This is followed by mechanistic analysis (representation collapse + activation probes) and finally evaluation of mitigation strategies.

### Key Designs

1.  **SimpleToolHalluBench Diagnostic Benchmark**:
    *   **Function**: Accurately measure tool hallucination rates.
    *   **Mechanism**: Designed two controlled scenarios—NTA (No Tool Available: no tools provided in the system but the user query requires one) and DT (Distractor Tools: irrelevant tools provided). 296 tools + corresponding queries are used; each query can only be correctly answered by its specific tool, ensuring any tool call in NTA/DT settings is a hallucination.
    *   **Design Motivation**: Existing benchmarks focus on "whether the model can call tools correctly," ignoring the critical question of "whether the model can restrain itself when it should not call a tool."

2.  **Four-Step Causal Exclusion Experimental Design**:
    *   **Function**: Precisely attribute the cause of tool hallucination to reasoning itself.
    *   **Mechanism**: (a) Tool RL increases hallucination → could be overfitting; (b) Math RL also increases hallucination → excludes overfitting; (c) Distillation/switching modes also increase hallucination → excludes RL specificity; (d) Ablating reasoning steps: removing the `<think>` block only slightly increases hallucination ($34.8 \to 41.4$), while keeping the `<think>` block significantly increases it ($34.8 \to 90.2$) → reasoning steps are the core factor.
    *   **Design Motivation**: Avoid simple correlational conclusions and establish stronger causal evidence through systematic ablation.

3.  **Mechanistic Analysis of Representation Collapse and Activation Localization**:
    *   **Function**: Reveal the internal mechanism of how reasoning enhancement affects tool behavior.
    *   **Mechanism**: (a) Use CKA to measure representation similarity across layers before and after RL—in-domain representations are stable ($CKA > 0.9$), while tool-related representations drift drastically in early and middle layers ($CKA < 0.75$); (b) Use linear probes to locate hallucination signals—correct vs. hallucinated responses are most linearly separable in the late-stage residual stream (discrimination score $> 0.14$), whereas attention and MLP outputs are nearly inseparable.
    *   **Design Motivation**: Go beyond "what" to reveal "why" and "where," providing a basis for future targeted interventions.

### Mitigation Strategy Evaluation

Prompt engineering (explicitly requesting "not to use unprovided tools") has a weak effect (NTA: $90.2 \to 87.5$); DPO (preference alignment of "honest responses" vs. "hallucinated responses") is effective but costly (NTA: $90.2 \to 55.8$, but SynTool reward drops from $0.45$ to $0.34$).

## Key Experimental Results

### Main Results

| Model/Configuration | $R_{NTA}(\downarrow)$ | $R_{DT}(\downarrow)$ | Description |
| :--- | :--- | :--- | :--- |
| Qwen2.5-7B-Instruct | 34.8 | 54.7 | Baseline |
| + ReCall RL (Tool) | 90.2 | 100.0 | Tool RL significantly increases |
| + GRPO (Math) | $\uparrow$ | $\uparrow$ | Non-tool RL also increases |
| R1-Distill-Qwen-7B | 74.3 | 78.7 | Distillation increases |
| Qwen3-8B Think Off | 4.1 | 36.2 | Reasoning disabled |
| Qwen3-8B Think On | 5.4 | 56.8 | Reasoning enabled increases |

### Ablation Study

| Configuration | $R_{NTA}$ | $R_{DT}$ | Reward |
| :--- | :--- | :--- | :--- |
| Baseline | 34.8 | 54.7 | 0.22 |
| Direct Tool RL (No Reasoning) | 41.4 | 63.6 | 0.28 |
| Think-then-act RL | 90.2 | 100.0 | 0.45 |
| + Prompt Engineering | 87.5 | 98.9 | 0.44 |
| + DPO | 55.8 | 71.4 | 0.34 |

### Key Findings
*   Reasoning enhancement consistently increases tool hallucination across all tested methods (RL/Distillation/Switchable modes).
*   RL training even on pure mathematical tasks increases tool hallucination, excluding the overfitting hypothesis.
*   Ablation shows that the reasoning step (the `<think>` block) itself, rather than RL training, is the core factor.
*   Instruction-following capability remains stable (IFEval: $-2.6\%$), and tool-calling capability even improves (BFCL: $+9.9\%$), but hallucinations surge—proving tool hallucination is an independent failure mode.
*   DPO mitigation is effective but exists an unavoidable capability-reliability trade-off.

## Highlights & Insights
*   **Reveals a profound paradox**: Reasoning enhancement makes models "smarter but more dishonest," posing a fundamental warning to current research paths pursuing reasoning scaling.
*   **Textbook-level experimental design**: The four-step exclusion method systematically establishes causal evidence with rigorous logic.
*   **Deep mechanistic analysis**: CKA representation analysis and activation probe localization answer not only "what" but also "why" and "where."
*   **Core Insight**: Tool hallucination is neither overfitting nor instruction-following degradation, but an inherent side effect of reasoning enhancement.

## Limitations & Future Work
*   **Focus on single-step tool calls**: Real-world agents involve multi-step tool chains where hallucination effects may accumulate.
*   **Incomplete causality**: Mechanistic analysis reveals correlational patterns but does not provide a complete causal explanation.
*   **Limited mitigation strategies**: Evaluated only prompt engineering and DPO; methods like process supervision or Constitutional AI were not explored.
*   Future work needs training objectives that jointly optimize for both capability and reliability, rather than post-hoc patching.

## Related Work & Insights
*   **vs ToolBeHonest**: Focuses on diagnostic evaluation of tool use but does not study the relationship between reasoning enhancement and hallucination.
*   **vs ReCall**: A SOTA RL framework for agent reasoning; this paper reveals its "hidden cost."
*   **vs DeepSeek-R1**: Transfers reasoning capability via distillation; this paper proves that hallucination tendencies are also transferred.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ First to systematically link reasoning enhancement with tool hallucination; the findings are significant.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four-step exclusion + mechanistic analysis + mitigation evaluation, with extremely rigorous design.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain, each experiment answers a specific question, progressing step-by-step.
*   Value: ⭐⭐⭐⭐⭐ Provides a fundamental warning to the reasoning scaling path, of great importance for Agent safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of the Reasoning Process](why_llms_hallucinate_on_structured_knowledge_a_mechanistic_analysis_of_reasoning.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](../../NeurIPS2025/hallucination/reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](../../ICML2026/hallucination/harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](../../CVPR2026/hallucination/understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)

</div>

<!-- RELATED:END -->
