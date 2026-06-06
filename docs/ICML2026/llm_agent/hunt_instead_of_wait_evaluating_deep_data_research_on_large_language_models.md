---
title: >-
  [Paper Note] Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models
description: >-
  [ICML 2026][LLM Agent][Deep Data Research] This paper proposes Deep Data Research (DDR), an open-ended agentic task paradigm where an LLM is provided only with a structured database and a minimal toolset (SQL+Python)…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Deep Data Research"
  - "Investigatory Intelligence"
  - "ReAct Agent"
  - "Checklist Evaluation"
  - "Long-horizon Exploration"
date: 2026-05-08
content_hash: b4a94d7cbac4e1fb
---

# Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.02039](https://arxiv.org/abs/2602.02039)  
**Code**: To be confirmed  
**Area**: LLM Agent / Agentic Benchmark / Data Science Agent  
**Keywords**: Deep Data Research, Investigatory Intelligence, ReAct Agent, Checklist Evaluation, Long-horizon Exploration

## TL;DR
This paper proposes Deep Data Research (DDR), an open-ended agentic task paradigm where an LLM is provided only with a structured database and a minimal toolset (SQL+Python), without specific questions or turn limits, requiring the model to autonomously explore, hypothesize, and decide when to stop. The authors construct DDR-Bench (MIMIC-IV / GLOBEM / 10-K, featuring 291 entities and 2058 checklist items) using verifiable fact checklists extracted from unstructured text to objectively evaluate the "investigatory intelligence" of mainstream LLMs; results show that even Claude 4.5 Sonnet achieves only a 47.7% average accuracy.

## Background & Motivation

**Background**: Agentic LLMs are already capable of long-horizon tasks using tools and memory. However, mainstream benchmarks (including LLM4DS and Deep Research) typically provide a default task objective in the prompt, where the model only needs to "finish the job per instructions."

**Limitations of Prior Work**: This setup only evaluates *executional intelligence* (the ability to complete predefined goals) and fails to measure *investigatory intelligence* (the ability to decide "what should be researched" independently). Even in supposedly "open-ended" LLM4DS report generation, prompts often contain numerous implicit "investigate what" instructions, and interactions are usually limited to a few dozen steps. Furthermore, deep research benchmarks primarily focus on search over unstructured web pages with restricted tools, and evaluations often rely on subjective scoring like LLM-as-a-Judge.

**Key Challenge**: To evaluate genuine agentic research capability, three conflicting requirements must be met: the task must be open-ended without preset questions, the interaction must be long enough to elicit long-term behavior, and the evaluation must be objective and reproducible for industrial use. Achieving all three is difficult: complete openness makes verification hard, while strong constraints degrade the task into simple QA.

**Goal**: To solve these three issues simultaneously by providing an agent task definition that is "question-free, step-unlimited, and scaffold-free," constructing a benchmark on real-world massive databases, and designing a scoring mechanism that measures open-ended insights objectively.

**Key Insight**: The authors observe that real-world large databases often contain both structured tables (numerical) and unstructured text (medical records, annual reports). By extracting verifiable facts from unstructured text as a checklist while allowing the model to explore only the structured portion, one can maintain open-ended exploration while obtaining ground-truth anchors.

**Core Idea**: A hybrid database design involving "structured exploration + unstructured-derived checklists" transforms open-ended agentic tasks into a benchmark that is batch-processable, objective, and resistant to data contamination.

## Method

### Overall Architecture
The DDR task is formalized as $I = DDR(LLM, D, T)$: given a database $D$ and toolset $T$, the LLM undergoes a ReAct-style $(r, t, o)$ loop (reasoning, tool call, observation) to query the database repeatedly without a turn limit, deciding itself when to terminate. The final outputs are two types of insights: message-wise insights $I_m$ generated per turn and trajectory-wise insights $I_t$ synthesized globally at the end. The system prompt provides only a minimal trigger (e.g., "Start analyzing user userid=2048") without specifying questions. During evaluation, a checklist $\{f_k\}$ pre-extracted from the database’s unstructured text is matched against the model's insights using GPT-5-mini to check if each fact is supported, yielding sample-averaged and item-averaged accuracy.

### Key Designs

1.  **DDR Task Formalization + Minimal Agent Scaffolding**:
    *   **Function**: Isolates "investigatory intelligence" from executional intelligence into a task paradigm testable by any mainstream LLM.
    *   **Mechanism**: Three constraints are applied: (a) No questions provided, only a task entity; (b) No step limit, with self-determined termination; (c) Only SQL and Python tools exposed via MCP, with explicit planning/memory/workflow modules prohibited. Results include $I_m$ (immediate interpretation of $(r_i, t_i, o_i)$) and $I_t$ (global synthesis via $\text{Summarize}(\{(r_i, t_i, o_i)\}_{i=1}^M)$), testing "process-level interpretation" and "global synthesis" respectively.
    *   **Design Motivation**: Previous agentic benchmarks conflated scaffolding (planners, memory, etc.) with the model's inherent capability, making it hard to discern if performance gains came from the base model or prompt engineering. Stripping it down to ReAct plus two atomic tools ensures scores reflect internalized agentic ability.

2.  **Hybrid Database + Checklist Evaluation Protocol**:
    *   **Function**: Provides objective, scalable, and contamination-resistant scoring while maintaining "complete openness."
    *   **Mechanism**: Three real-world databases containing both structured and unstructured data are used: MIMIC-IV (Electronic Health Records), GLOBEM (Wearable signals + mental health surveys), and 10-K (Financial reports). GPT-5-mini extracts facts from the unstructured side to form checklists, filtered by 50+ domain experts to ensure a "surjective" mapping from the fact-domain to the data-domain (every fact is derivable from structured data). The model only sees structured data during exploration. The evaluation naturally resists training set contamination as no QA-format data exists during the interaction phase. It covers 291 task entities and 2058 checklist items.
    *   **Design Motivation**: Standard routes (QA construction or LLM-as-a-Judge) fail for open exploration. Extracting facts from the unstructured side of the same database provides "expert-level questions" for free, transforming open tasks into verifiable fact recall.

3.  **Four-dimensional Investigation Dynamics Framework**:
    *   **Function**: Goes beyond simple accuracy to dissect model behavior regarding "when, how, and where" to explore and stop.
    *   **Mechanism**: (a) Test-time scaling analyzed across interaction (turns), tokens (cost tokens), and cost (USD). (b) Exploration patterns visualized via $H_{\text{norm}} = \frac{-\sum_{i=1}^n p_i \log_2 p_i}{\log_2 n}$ (normalized exploration entropy) vs. database coverage; strong models balance middle-range entropy and coverage. (c) Self-termination measured by $\frac{1}{N}\sum_{i=1}^N \log P(t_i \mid t_1, \dots, t_{i-1}, T_{\text{partial}})$, tracking the log-probability of generating the end token. (d) Training factor ablation decouples contributions from parameter size, context length, and agentic training.
    *   **Design Motivation**: Accuracy alone hides behavioral differences. This framework uses visualization and metrics to prove why agentic training is more critical than simply scaling parameters.

### Loss & Training
This work presents a benchmark and evaluation protocol and does not train models. The evaluation side utilizes GPT-5-mini for checklist verification and pairwise novelty comparison (aggregated via Bradley-Terry models to mitigate position bias).

## Key Experimental Results

### Main Results
Evaluation of 8 commercial and 13 open-source models across three DDR-Bench scenarios using the average of 4 accuracy types (sample×item × $I_m$×$I_t$) as the Overall score:

| Model | MIMIC ($I_m$) | GLOBEM ($I_m$) | 10-K ($I_m$) | Overall Avg |
| :--- | :--- | :--- | :--- | :--- |
| Claude 4.5 Sonnet | 36.07 | 40.13 | 77.61 | **47.73** |
| GPT-5.2 | 28.85 | 38.81 | 44.89 | 37.09 |
| GPT-5.1 | 28.37 | 38.31 | 37.12 | 36.27 |
| DeepSeek-V3.2 | 28.98 | 38.46 | 60.08 | 38.80 |
| GLM-4.6 | 25.03 | 41.56 | 60.31 | 37.52 |
| Kimi K2 | 33.61 | 37.14 | 51.06 | 36.42 |
| Qwen3-Next-80B-A3B | 18.01 | 35.75 | 44.76 | 30.56 |
| Qwen2.5-72B | 15.65 | 28.83 | 27.13 | 21.08 |
| Llama3.3-70B | 10.59 | 23.99 | 9.91 | 12.30 |

Claude 4.5 Sonnet is the only model to exceed 40%, significantly leading in 10-K (77.61). The gap between top open-source models and next-gen GPT/Gemini is smaller here than in QA benchmarks, suggesting DDR effectively isolates "autonomous research" capability.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Qwen3-Next default reasoning | 10-K 45.58 / GLOBEM 35.40 | Avg 1.20 reasoning tokens per turn (10-K) |
| Qwen3-Next longer reasoning | 10-K 36.40 ↓ / GLOBEM 36.78 ↑ | Reasoning tokens ↑ to 357, but turns ↓ from 27.93 to 11.89 |
| Qwen3-Next w/o memory | 10-K Traj 31.10 / MIMIC Traj 20.80 | Default minimal ReAct |
| Qwen3-Next w/ long-short memory | 10-K Traj 25.21 ↓ / MIMIC Traj 14.34 ↓ | Memory leads to aggressive early stopping |
| Proactive (DDR) | Avg 32.59 / 28.26 | Autonomous questioning |
| Reactive (Checklist as queries) | Avg 43.21 ↑ | Significant gain with explicit goals, except on GLOBEM |

### Key Findings
*   Failure analysis (206 manual labels) shows **58% of errors stem from "insufficient exploration"**—either lack of breadth (missing key fields) or depth (surface-level analysis). The remaining 40% involve "over-reasoning/unfounded hypotheses" in strong models and "context loss/repetitive debugging" in weaker ones.
*   Increasing parameters by 10x yields < 3% gain. Long-context versions (Qwen2.5-1M) show no systemic improvement. Qwen3-Next-4B outperforms Qwen2.5-72B. **Agentic capability stems from training paradigms (reasoning + agency focus), not just scale.**
*   The token scaling curve follows a "flat then steep" pattern—late-stage, precise, and complex tool calls contribute the most performance gains, marking the transition from breadth-first to depth-first exploration.
*   Novelty rankings (via Bradley-Terry) align closely with checklist accuracy, suggesting the checklist captures the "dominant signal" and does not systematically undervalue models focusing on non-checklist insights.
*   Hallucination rates are generally low (Claude 4.08% / Gemini < 1%) and show no statistical correlation with accuracy, ruling out "fabrication from memory" as a strategy.

## Highlights & Insights
*   **The hybrid "non-structured → checklist, structured → exploration" design is ingenious**: Decoupling what the model sees from what the evaluator checks using the same real-world data provides expert ground truth at zero additional cost.
*   **Explicitly quantifying investigatory vs. executional intelligence**: The ~10 point gap between reactive (43.21%) and proactive (32.59%) modes provides the first digital measure of the "difficulty of deciding what to ask."
*   **The Exploration Entropy × Coverage scatter plot** is a highly reusable tool for agentic behavior visualization, indicating whether an agent is spread too thin, too narrow, or balanced.
*   **Self-termination log-probability diagnostics** offer a lightweight, transferable metric to evaluate inherent confidence stability in agent base models, useful for pre-RL diagnostic testing.
*   The empirical conclusion that "agentic ≠ scale" is backed by clean evidence across the Qwen family while controlling for scaffolding, sending a clear signal for future agentic training.

## Limitations & Future Work
*   Evaluation depends on GPT-5-mini, effectively tethering "evaluation reliability" to a commercial LLM's judgment; this remains an implicit ceiling.
*   Scenario coverage is somewhat narrow—focused on "reading databases for insights"—and does not cover the full data science loop (modeling, experimenting, writing code, training ML pipelines).
*   The restricted toolset (SQL + Python) isolates clean variables but means "tool selection capability" is not fully tested compared to a real scientist's toolkit.
*   Checklists are derived from facts already written in text, biasing toward "replicating expert insights" rather than "discovering original insights."
*   No specific training recipe (RLHF/SFT) is provided; the paper identifies *that* training matters, but leaves the *how* to future work.
*   Memory experiments only compare one long-short note design; conclusions that "memory is not necessarily beneficial" might be over-generalized across all architectures.

## Related Work & Insights
*   **vs. LLM4DS Report Generation (Zhang et al., 2025c)**: Those still use "investigate what" instructions and limited turns; DDR removes these constraints for a purer test of investigatory intelligence.
*   **vs. Deep Research Benchmarks (Wong et al., 2025)**: Those focus on unstructured web search with search/browse tools; DDR shifts to structured DBs with SQL/Python for verifiable fact extraction.
*   **vs. Table QA (Lu et al., 2025a)**: Table QA tasks are executional; DDR changes the task from "answer the question" to "discover the question."
*   **vs. Agentic Scaffolding**: This work supports "internalizing capability into the model" rather than stacking complex scaffolds, putting pressure on scaffold-heavy approaches.
*   **Insight for Future RL/SFT**: DDR-Bench provides a long-horizon, question-free, verifiable training-eval loop closer to real research than current short-horizon tool-use SFT data.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ Successfully decouples investigatory intelligence and solves the open-ended scoring problem via hybrid-database checklists.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive coverage: 21 models, 3 databases, 4-way scaling, entropy analysis, and manual error labeling.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear conceptual distinctions and visualizations, though some module analysis sections are extremely dense.
*   **Value**: ⭐⭐⭐⭐⭐ Likely to become a standard benchmark for evaluating true research capabilities in agentic LLMs from 2026 onwards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](../../ACL2026/llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](../../AAAI2026/llm_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](../../NeurIPS2025/llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](../../ACL2026/llm_agent/anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)

</div>

<!-- RELATED:END -->
