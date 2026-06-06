---
title: >-
  [Paper Note] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Tool invocation evaluation] The authors translate the human-annotated solution steps of the MATH dataset into "reusable Python tools with descriptions and type signatures…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Tool invocation evaluation"
  - "long-horizon multi-tool reasoning"
  - "distractor tools"
  - "tool-missing scenarios"
  - "Plan+ReAct"
date: 2026-05-08
content_hash: 756f8df258edaae7
---

# ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.21265](https://arxiv.org/abs/2602.21265)  
**Code**: None  
**Area**: LLM Agent / Tool Use Benchmark  
**Keywords**: Tool invocation evaluation, long-horizon multi-tool reasoning, distractor tools, tool-missing scenarios, Plan+ReAct

## TL;DR
The authors translate the human-annotated solution steps of the MATH dataset into "reusable Python tools with descriptions and type signatures," constructing the ToolMATH benchmark with 8K problems and 12K tools. It covers long-horizon multi-tool composition (hop 1-8+), controllable distractor tool similarity (5 levels × 4 densities), and scenarios where all gold tools are removed. Validation shows that the dominant failure factor is not tool selection but reasoning itself—thought errors account for over 90%, and distractor tools amplify early minor deviations into irreversible execution drift.

## Background & Motivation
**Background**: Tool-augmented LLMs have become the standard agent paradigm. From Toolformer and Gorilla to BFCL and ToolLLM, a series of works have standardized function calling. However, existing benchmarks mostly focus on one or two axes: (i) standardized schema comparison (BFCL), (ii) robustness under tool unavailability (Treviño et al. 2025), (iii) tool control interfaces (ReAct/DFSDT), and (iv) tool dependency graph construction (TaskBench).

**Limitations of Prior Work**: In real-world deployment, agents face complex joint scenarios of "large, semantically overlapping tool catalogs + long-horizon multi-step dependencies + occasional missing key capabilities," but no benchmark simultaneously covers these three dimensions on a single, automatically verifiable task. Existing math reasoning benchmarks (GSM8K, MATH) allow objective correctness verification but lack the tool dimension; existing tool benchmarks often rely on manual correctness judgment or lack long-horizon dependencies.

**Key Challenge**: To simultaneously achieve (i) objective automatic verifiability (not relying on LLM judge), (ii) inherent long-horizon dependencies (step coupling), (iii) controllable distractor tool structure, and (iv) controllable tool-missing scenarios—only with all four can agent failure modes be precisely dissected. The stepwise solutions in MATH naturally provide: each step can be extracted as a Python tool, steps are logically tightly coupled, and answers can be machine-verified.

**Goal**: (i) Transform MATH solution steps into reusable tools to construct long-horizon composition tasks; (ii) Design distractor tool sampling strategies and "distractors-only" environments, allowing controlled variation of distractor similarity and density; (iii) Ensure benchmark reliability through tool-level and problem-level dual validation plus manual review; (iv) Use hop count to decouple long-horizon difficulty from distractor difficulty.

**Key Insight**: Treat the "stepwise logical chain" of mathematics as a natural scaffold for tool composition—each step corresponds to a Python implementation + natural language description + type signature, with models only seeing the description and schema, not the code. This "one wrong step, all wrong" property amplifies both long-horizon reasoning and tool selection failures.

**Core Idea**: Using the MATH step-tool mapping, distractor tools, and missing tools as three independent dimensions, construct a benchmark that can simultaneously dissect "tool selection / long-horizon planning / fallback under missing tools."

## Method

### Overall Architecture
ToolMATH construction has two stages: (1) **Tool extraction & validation**: Feed MATH's annotated solution steps to an LLM, which returns several small Python functions (with name, description, typed input schema, code), then perform tool-level consistency validation (each tool gets 5 test cases, LLM judge checks if description matches execution, allowing floating-point tolerance) + problem-level trace validation (run Plan+ReAct with 7 validation models; if at least one model uses the tool and gets the correct answer, the tool passes); (2) **Tool-grounded evaluation**: For each problem $p$, the environment = gold tool set $\mathcal G(p)$ + distractor tools $\mathcal D_{\ell,k}(p)$ sampled from the global pool (5 similarity levels, density $k \in \{5,10,20,50\}$), or in Distractors-only mode, remove $\mathcal G(p)$ and leave only distractors, forcing model fallback. Each problem is annotated with hop count (dynamically computed parallel post-logic steps) as an independent long-horizon difficulty axis.

### Key Designs

1. **Two-stage validation pipeline for MATH steps → reusable tools**:

    - **Function**: Convert human-annotated solution steps into a tool set where only the description and schema are visible (not code), ensuring tool quality (not benchmark noise).
    - **Mechanism**: Tool-wise validation—prepare 5 schema-valid inputs for each extracted tool, execute to get actual outputs, use GPT-4o as judge to check if description matches output (allowing floating-point tolerance); all 5 must pass. Question-wise validation—give 7 validation models ({GPT-4o-mini, Llama 3-8B, Mistral-7B, Qwen2-7B, Qwen2.5-7B, Phi-3 Medium, Yi 1.5-9B}) only the problem and gold tool set (no code), run Plan+ReAct; if at least one model gets the correct answer and the trace shows successful invocation of tool $t$, then $t$ passes; otherwise, enter manual repair loop (fix description/implementation and re-validate). Final main set: 12,369 tools + 7,699 problems, plus ToolMATH-Hard with 329 problems that cannot be automatically validated.
    - **Design Motivation**: Single-layer validation is insufficient—tool consistency alone may select "well-described but unused" tools; trace-only may attribute tool errors to the model. Dual validation plus manual repair is the engineering best practice for benchmarks, systematized here for large-scale reliability.

2. **5-level similarity × 4-level density distractor tool structure**:

    - **Function**: Adjust the "semantic overlap between gold and non-gold tools" in a controllable way, decoupling catalog size from confusion difficulty.
    - **Mechanism**: Level 1 (different-category random) → Level 2 (pure random) → Level 3 (same-category random) → Level 4 (embedding similarity retrieval) → Level 5 (keyword overlap + embedding tiebreak), with increasing similarity. Density $k \in \{5,10,20,50\}$, and **nesting ensures** $\mathcal D_{\ell,k_1}(p) \subseteq \mathcal D_{\ell,k_2}(p)$, so density changes only reflect increased interference, not sample variation. Each sampling is deterministically reproducible (fixed seed + fixed tool pool serialization order).
    - **Design Motivation**: Previous benchmarks either had weak interference (low overlap) or fixed distractors (no ladder), making it impossible to quantitatively analyze "the effect of tool similarity on failure modes." Levels 1→5 allow plotting "accuracy vs. distractor similarity" curves, cleanly proving that "high-similarity distractors amplify long-horizon failures" (Figure 2).

3. **Difficulty decoupling via Distractors-only + logical-hop annotation**:

    - **Function**: Separate the axes of "tool availability" and "long-horizon reasoning difficulty" to independently diagnose model failure modes.
    - **Mechanism**: Distractors-only mode removes all gold tools, leaving only distractors, forcing the model to either fallback to tool-free reasoning or give up; Logical-hop annotation uses "step extraction + parallelism check" via two LLM prompts to compute each problem's hop count (not simply counting tools, but removing parallelizable steps). Evaluation buckets accuracy by hop, revealing: (i) accuracy monotonically decreases with hop (even for No-tools baseline, proving hop captures inherent difficulty); (ii) high-similarity distractors mainly amplify the drop at high hops.
    - **Design Motivation**: Previous benchmarks mixed "problem difficulty" and "tool confusion," making it unclear which side caused poor model performance. Two independent axes enable clean ablation. Distractors-only also reveals an interesting phenomenon: Qwen2.5-7B can compose alternative solutions using generic distractor tools, indicating multiple solution paths in math problems.

### Loss & Training
This is a pure benchmark/evaluation paper; no models are trained. Main evaluation protocol: Plan+ReAct (write plan first, then alternate reasoning and structured tool calls), evaluated models = {GPT-4o-mini, Llama 3-8B, Qwen 2.5-7B}. Metric = exact-match accuracy (standardized). On ToolMATH-Hard, also compare ReAct, DFSDT, and Plan+ReAct frameworks.

## Key Experimental Results

### Main Results
Average gold-present accuracy by hop and similarity (GPT-4o-mini as representative):

| Setting | hop 1-2 | hop 5 | hop 7 | hop 8+ |
|---|---|---|---|---|
| No tools | ~High | Medium | Low | Very low |
| Gold-only | Near ceiling | High | Medium | Low (still collapses at hop 8+) |
| Gold + Level 1-2 distractors | Near Gold-only | Medium-high | Medium | Low |
| Gold + Level 4-5 distractors | Still high | Noticeable drop | Sharp drop | Worst |

ToolMATH-Hard framework comparison (gold-only):

| Framework | Low hop | High hop | Overall trend |
|---|---|---|---|
| No tools | High | Sharp drop | Long-horizon always fails |
| ReAct | High | Medium | Limited local reasoning |
| DFSDT | Medium-high | Medium-high | Best in mid-range |
| **Plan+ReAct** | High | **Strongest** | No drop in long-horizon |

### Ablation Study (manual failure annotation, 100 problems per model)

| Failure Type | Llama 3-8B | Qwen 2.5-7B | GPT-4o-mini |
|---|---|---|---|
| Thought Error | >90% | >90% | >90% |
| Plan Error | **89** | Medium | Medium |
| Incomplete Execution | 59 | **8** | Medium |
| Observation Omission | Medium | **63** | Medium |
| Repeated Call | Medium | Medium | **67** |
| Tool Hallucination | Low | Low | Low |
| Wrong Parameter Value | Medium | Medium | Medium |

### Key Findings
- Thought Error exceeds 90% for all models, proving that **reasoning ability itself** rather than tool understanding is the main bottleneck for agents.
- Models show distinct behavioral profiles: Llama 3-8B is conservative and brittle (high Plan Error + Incomplete), Qwen 2.5-7B is impulsive (lowest Incomplete but highest Observation Omission, aggressively outputs answers), GPT-4o-mini is overall strongest but suffers from "repeated call paradox" (no self-correction when stuck in loops).
- Plan+ReAct significantly outperforms ReAct/DFSDT at high hops, indicating that "explicit global planning" becomes more valuable as long-horizon execution lengthens; at low hops, the three are similar, and planning overhead is not worthwhile.
- High-similarity distractors do not directly cause errors but **amplify early deviations**, making failure rates at hop 8+ much steeper than in low-similarity scenarios.
- In Distractors-only mode, Qwen 2.5-7B can use non-gold tools to substitute for gold tools and complete tasks (accuracy higher than No-tools baseline), indicating that the multiplicity of solutions in math allows "using the wrong tool but getting the right answer."

## Highlights & Insights
- The observation that **MATH steps = natural tool scaffolding** is very clever: translating human step-by-step solutions into Python functions + descriptions preserves the logical coupling while making "tool description/schema" the object the model must understand. This approach of "using mathematical rigor to build tool benchmarks" can be extended to code, theorem proving, and other domains.
- The finding that **thought error is the main bottleneck** is counterintuitive: the tool-calling community has long focused on function calling schema standardization and ReAct/DFSDT control flow, but this work quantitatively shows these are not the main issues—the real bottleneck is reasoning. This recalibrates research priorities.
- The **behavioral profiles (Llama conservative / Qwen impulsive / GPT looping)** are valuable engineering references: when selecting agents, model temperament can be matched to task characteristics.

## Limitations & Future Work
- Domain is math-specific, lacking the openness, ambiguity, and under-specified goals of real-world tasks; cannot be directly extrapolated to web agents, coding agents, scientific agents, etc.
- Tool consistency relies on LLM judge, and 5 test cases may not cover all corner-case inconsistencies.
- Evaluation frameworks are limited to Plan+ReAct/ReAct/DFSDT; newly emerging reasoning model agents (o1/R1 series with embedded reasoning + tool use) are not evaluated, so the proportion of thought errors may already be decreasing.
- ToolMATH-Hard has only 329 problems, limiting statistical power, with even sparser samples in the hop 8+ bucket.
- Future directions: extend to non-math domains (especially code tasks with objective correctness); include reasoning model agents and self-correction loop baselines; perform factor analysis of hop and distractor to estimate independent contributions.

## Related Work & Insights
- **vs ToolLLM / API-Bank**: These use large-scale APIs to test function calling standardization but lack long-horizon multi-step dependencies and objective automatic scoring; ToolMATH uses MATH's hard logical chains to fill these gaps.
- **vs BFCL (Patil et al. 2025)**: BFCL focuses on function calling format correctness and missing tool behavior, but tools have no compositional dependencies; ToolMATH makes long-horizon composition explicit via hop count.
- **vs TaskBench (Shen et al. 2024)**: TaskBench uses graph structures to generate tool dependency tasks but is synthetic; ToolMATH is based on human solution steps, making dependencies more realistic.
- **vs Treviño et al. 2025 (tool failure benchmark)**: They focus on tool unavailability; ToolMATH treats this as the Distractors-only axis and jointly evaluates with long-horizon difficulty.

## Rating
- Novelty: ⭐⭐⭐⭐ "Using MATH steps as tool scaffolding" is a simple but overlooked idea, and three-dimensional joint evaluation is urgently needed for real deployment.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 5 distractor similarity levels × 4 distractor densities × 8+ hop buckets × 3 frameworks, plus ToolMATH-Hard and 100-problem manual failure analysis, coverage is very high.
- Writing Quality: ⭐⭐⭐⭐ Three challenges and three designs correspond one-to-one, logic is clear; lacks some main table values (relies on figures), readers need to check the appendix.
- Value: ⭐⭐⭐⭐ Provides the agent research community with a much-needed objectively verifiable benchmark, and the finding that "thought error is the main bottleneck" can guide future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](../../ICLR2026/llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ICLR 2026\] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](../../ICLR2026/llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)

</div>

<!-- RELATED:END -->
