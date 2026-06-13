---
title: >-
  [Paper Note] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Tool-calling evaluation] The authors translate human-annotated solution steps from the MATH dataset into "reusable Python tools with descriptions and type signatures…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Tool-calling evaluation"
  - "long-horizon multi-tool reasoning"
  - "distractor tools"
  - "missing tool scenarios"
  - "Plan+ReAct"
date: 2026-05-08
content_hash: 264f3d232b22fbbf
---

# ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.21265](https://arxiv.org/abs/2602.21265)  
**Code**: None  
**Area**: LLM Agent / Tool-use Benchmark  
**Keywords**: Tool-calling evaluation, long-horizon multi-tool reasoning, distractor tools, missing tool scenarios, Plan+ReAct

## TL;DR
The authors translate human-annotated solution steps from the MATH dataset into "reusable Python tools with descriptions and type signatures," constructing the ToolMATH benchmark containing 8K problems and 12K tools. It simultaneously covers long-horizon multi-tool combinations (1-8+ hop), controllable distractor tool similarity (5 levels × 4 densities), and "Distractors-only" scenarios where gold tools are removed. Evaluations demonstrate that the dominant factor for model failure is not tool selection but the reasoning itself—thought errors account for over 90%, while distractor tools amplify early minor biases into irreversible execution drift.

## Background & Motivation
**Background**: Tool-augmented LLMs have become a standard agent paradigm. A series of works from Toolformer and Gorilla to BFCL and ToolLLM have standardized function calling. However, most existing benchmarks only capture one or two axes among: (i) standardized schema comparison (BFCL), (ii) robustness under tool unavailability (Treviño et al. 2025), (iii) tool control interfaces (ReAct/DFSDT), and (iv) tool dependency graph construction (TaskBench).

**Limitations of Prior Work**: Realistic deployments present agents with complex joint scenarios involving "huge and semantically overlapping tool catalogs + long-horizon multi-step dependencies + occasionally missing key capabilities." Yet, no benchmark covers these three dimensions within a **single automatically verifiable** task. Existing math reasoning benchmarks (GSM8K, MATH) provide objective correctness but lack the tool dimension, while existing tool benchmarks often rely on manual judgment or lack long-horizon dependencies.

**Key Challenge**: To precisely analyze agent failure modes, a benchmark must simultaneously satisfy four characteristics: (i) **objective automatic verification** (not relying on LLM judges), (ii) **natural long-horizon dependency** (inter-step coupling), (iii) **controllable distractor tool structures**, and (iv) **controllable tool missing scenarios**. MATH’s step-by-step solutions provide exactly this: each step can be extracted as a Python tool, the logic is hard-coupled, and answers are machine-decodable.

**Goal**: (i) Convert MATH solution steps into reusable tools to construct long-horizon compositional tasks; (ii) design distractor tool sampling strategies and "Distractors-only" environments to controllably vary similarity and density; (iii) ensure benchmark reliability through tool-level and question-level verification plus manual review; (iv) decouple long-range difficulty from distractor difficulty using hop counts.

**Key Insight**: Utilize the "logical chain of steps" in mathematics as a natural scaffolding for tool composition. Each step corresponds to a Python implementation, a natural language description, and a type signature. Models see only descriptions and schemas, not the code. This "one-wrong-step-fails-all" characteristic allows both "long-horizon reasoning failures" and "tool selection failures" to be amplified and exposed.

**Core Idea**: Use the MATH step-tool mapping, distractor tools, and missing tools as three independent dimensions to construct a benchmark that can simultaneously analyze "tool selection, long-horizon planning, and missing tool fallback."

## Method

### Overall Architecture
ToolMATH construction is divided into two phases: (1) **Tool extraction & validation**: Human-annotated MATH steps are fed to an LLM to return small Python functions (with name, description, typed input schema, and code). This is followed by tool-level consistency verification (5 test cases per tool judged by an LLM for consistency between description and execution) and question-level trace verification (Plan+ReAct is run using 7 validation models; if at least one model uses the tool and answers correctly, the tool passes). (2) **Tool-grounded evaluation**: The environment for each problem $p$ consists of gold tools $\mathcal G(p)$ plus distractor tools $\mathcal D_{\ell,k}(p)$ sampled from a global pool (5 similarity levels, density $k \in \{5,10,20,50\}$). In Distractors-only mode, $\mathcal G(p)$ is removed. Each problem is annotated with a hop count (parallel-aware logical steps calculated via dynamic programming) as an independent axis for long-horizon difficulty.

### Key Designs

1. **MATH Step → Reusable Tool Two-Round Validation Pipeline**:
    - **Function**: Converts human-annotated steps into a toolset where models only see descriptions and schemas, ensuring tool quality (not benchmark noise).
    - **Mechanism**: Tool-wise verification—5 valid inputs are prepared for each extracted tool to verify description-execution consistency via GPT-4o; tools pass only with a 100% score. Question-wise verification—7 models ({GPT-4o-mini, Llama 3-8B, Mistral-7B, Qwen2-7B, Qwen2.5-7B, Phi-3 Medium, Yi 1.5-9B}) run Plan+ReAct with only descriptions; if at least one model succeeds and calls tool $t$, it passes. Otherwise, it enters a manual repair loop. The final set includes 12,369 tools and 7,699 problems.
    - **Design Motivation**: Single-layer verification is inadequate—consistency checks might pass tools that are correctly described but unusable, while trace checks might misattribute model errors to tool faults.

2. **5 Levels Similarity × 4 Levels Density Distractor Structure**:
    - **Function**: Controllably adjusts the "semantic overlap between gold and non-gold tools" to separate catalog size from confusion difficulty.
    - **Mechanism**: Similarity levels range from Level 1 (different-category random) to Level 5 (keyword overlap + embedding tiebreak). Density $k$ uses **nested guarantees** $\mathcal D_{\ell,k_1}(p) \subseteq \mathcal D_{\ell,k_2}(p)$ to ensure comparisons reflect only increased density.
    - **Design Motivation**: Previous benchmarks used weak or fixed distractors. Level 1–5 allows for "Accuracy vs. Similarity" curves, clearly proving that high-similarity distractors amplify long-horizon failures.

3. **Distractors-only + Logical-hop Annotation for Difficulty Decoupling**:
    - **Function**: Separates "tool availability" from "long-horizon reasoning difficulty" to diagnose specific failure modes.
    - **Mechanism**: Distractors-only mode removes all gold tools, forcing models to fallback or quit. Logical-hop annotation uses LLMs to calculate the hop count (removing parallel steps). Accuracy curves plotted by hop count show a monotonic decrease, indicating that hop count captures inherent difficulty.
    - **Design Motivation**: Previous benchmarks conflated problem difficulty with tool noise. These independent axes ensure clean ablation. Distractors-only also reveals that models like Qwen2.5-7B can combine generic tools for alternative solutions.

### Loss & Training
This is a benchmark paper and does not train models. The primary evaluation protocol is Plan+ReAct (writing a plan first, then alternating reasoning with structured tool calls). Evaluation models include {GPT-4o-mini, Llama 3-8B, Qwen 2.5-7B}. The metric is exact-match accuracy.

## Key Experimental Results

### Main Results
Gold-present average accuracy variations with hop and similarity (represented by GPT-4o-mini):

| Setting | hop 1-2 | hop 5 | hop 7 | hop 8+ |
|---|---|---|---|---|
| No tools | ~High | Mid | Low | Very Low |
| Gold-only | Near ceiling | High | Mid | Low |
| Gold + Level 1-2 Distractor | Near Gold-only | Mid-High | Mid | Low |
| Gold + Level 4-5 Distractor | Still High | Significant Drop | Rapid Drop | Worst |

ToolMATH-Hard framework comparison (gold-only):

| Framework | Low hop | High hop | General Trend |
|---|---|---|---|
| No tools | High | Sharp Drop | Long-horizon failure |
| ReAct | High | Mid | Limited local reasoning |
| DFSDT | Mid-High | Mid-High | Best for mid-range |
| **Plan+ReAct** | High | **Strongest** | No long-horizon drop |

### Ablation Study (Failure Type Manual Annotation, n=100 per model)

| Failure Type | Llama 3-8B | Qwen 2.5-7B | GPT-4o-mini |
|---|---|---|---|
| Thought Error | >90% | >90% | >90% |
| Plan Error | **89** | Mid | Mid |
| Incomplete Execution | 59 | **8** | Mid |
| Observation Omission | Mid | **63** | Mid |
| Repeated Call | Mid | Mid | **67** |
| Tool Hallucination | Low | Low | Low |
| Wrong Parameter Value | Mid | Mid | Mid |

### Key Findings
- Thought Error exceeds 90% across all models, proving that **reasoning capability itself**, rather than tool understanding, is the primary bottleneck for agents.
- Models exhibit distinct behavioral profiles: Llama 3-8B is conservative and fragile (High Plan Error + Incomplete); Qwen 2.5-7B is impulsive (Lowest Incomplete but highest Observation Omission); GPT-4o-mini is overall strongest but suffers from the "repeated call paradox" (trapped in loops without self-correction).
- Plan+ReAct significantly outperforms ReAct/DFSDT at higher hop counts, indicating the value of "explicit global planning" increases with long-horizon execution.
- High-similarity distractors do not cause direct errors but **amplify early deviations**, leading to a steeper failure rate at hop 8+ compared to low-similarity scenarios.
- In Distractors-only mode, Qwen 2.5-7B can use non-gold tools as substitutes to complete tasks, suggesting the multi-path nature of math problems.

## Highlights & Insights
- The observation that **MATH steps = natural tool scaffolding** is ingenious: translating human steps into Python functions preserves logic coupling while making tool schemas a test of model understanding.
- The finding that **"thought error is the primary bottleneck"** is counter-intuitive. While the community focuses on function calling schemas and control flows, this work quantitatively proves reasoning is the main conflict.
- **Behavioral profiling** (Conservative Llama / Impulsive Qwen / Looping GPT) provides a valuable engineering reference for matching model temperaments to specific tasks.

## Limitations & Future Work
- Focus is limited to the math domain, lacking the openness and ambiguity of real-world goals.
- Tool consistency relies on LLM judges; 5 test cases may not cover corner-case behavioral inconsistencies.
- Modern reasoning models (o1/R1 series) were not evaluated; their internal reasoning might reduce thought error ratios.
- The ToolMATH-Hard set is small (329 problems), leading to sparse samples in the hop 8+ bin.

## Related Work & Insights
- **vs ToolLLM / API-Bank**: These focus on large-scale API function calling but lack long-horizon dependencies and objective scoring; ToolMATH uses math logic chains to fill this gap.
- **vs BFCL (Patil et al. 2025)**: BFCL focuses on schema correctness and missing tools without compositional dependencies; ToolMATH explicitly models long-range compositions via hop counts.
- **vs TaskBench (Shen et al. 2024)**: TaskBench uses synthetic graph structures for dependencies; ToolMATH uses real human solution steps.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](../../ACL2026/llm_reasoning/evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICML 2026\] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](diversity_over_frequency_rethinking_tool_use_in_visual_chain-of-thought_agents.md)
- [\[ICML 2026\] DenseSteer: Steering Small Language Models towards Dense Math Reasoning](densesteer_steering_small_language_models_towards_dense_math_reasoning.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](../../ACL2026/llm_reasoning/evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICML 2026\] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](diversity_over_frequency_rethinking_tool_use_in_visual_chain-of-thought_agents.md)
- [\[ICML 2026\] DenseSteer: Steering Small Language Models towards Dense Math Reasoning](densesteer_steering_small_language_models_towards_dense_math_reasoning.md)

</div>

<!-- RELATED:END -->
