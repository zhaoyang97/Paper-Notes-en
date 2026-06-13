---
title: >-
  [Paper Note] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering
description: >-
  [ACL 2026][Multi-Agent][Table Question Answering] The authors propose MATA, a multi-agent TableQA framework that prioritizes reasoning paths (CoT/PoT/text2SQL) via a scheduler, filters answers with a confidence checker…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Table Question Answering"
  - "Multi-Agent Framework"
  - "Multi-Reasoning Paths"
  - "Model-agnostic"
  - "LLM Efficiency"
date: 2026-05-08
content_hash: a62831981605e89b
---

# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

**Conference**: ACL 2026  
**arXiv**: [2602.09642](https://arxiv.org/abs/2602.09642)  
**Code**: [GitHub](https://github.com/AIDASLab/MATA)  
**Area**: LLM Agent  
**Keywords**: Table Question Answering, Multi-Agent Framework, Multi-Reasoning Paths, Model-agnostic, LLM Efficiency

## TL;DR
The authors propose MATA, a multi-agent TableQA framework that prioritizes reasoning paths (CoT/PoT/text2SQL) via a scheduler, filters answers with a confidence checker, and arbitrates via a judge agent. It achieves model-agnostic, efficient, and accurate TableQA, with an average EM improvement of 40.1% across 10 LLMs.

## Background & Motivation

**Background**: LLMs have significantly advanced Table Question Answering (TableQA), enabling interaction between natural language and structured tables. Existing methods typically leverage reasoning strategies such as CoT, PoT (Program-of-Thought), or text2SQL to generate answers.

**Limitations of Prior Work**: (1) Most high-performance methods rely on closed-source LLMs (e.g., GPT-4o), making them unsuitable for privacy-sensitive or cost-constrained scenarios, and their reliability on small open-source models has not been fully verified. (2) To improve reliability, existing frameworks frequently call LLM reasoning (e.g., Self-Consistency), leading to high computational costs or even reduced accuracy due to over-prompting. (3) Most frameworks only utilize CoT+PoT, failing to exploit the diversity of three complementary strategies: CoT, PoT, and text2SQL.

**Key Challenge**: The trade-off between reasoning diversity and efficiency—adding reasoning paths can improve accuracy, but each path incurs LLM overhead. Blindly executing all paths is wasteful and may introduce noise.

**Goal**: Build a model-agnostic TableQA framework that maintains high accuracy across various open/closed-source LLMs while minimizing LLM calls through intelligent scheduling.

**Key Insight**: Reasoning diversity does not require a fixed reasoning budget—a lightweight controller can determine which branches are necessary and when verification can terminate early.

**Core Idea**: Use lightweight tool models (Scheduler, Confidence Checker, Format Matcher) to coordinate reasoning path selection and answer verification across multiple LLM agents, achieving an optimal balance between diversity and efficiency.

## Method

### Overall Architecture
MATA receives table $T$ and question $Q$, producing the final answer through a three-stage process: (1) Agent Selection Stage: The Scheduler determines the execution priority of PoT and text2SQL while the CoT Agent executes in parallel; (2) Code Generation & Debugging Stage: PoT/text2SQL agents generate code, which is iteratively fixed by the Debug Agent; (3) Final Answer Decision Stage: The Confidence Checker evaluates the confidence of candidate answers, calling the Judge Agent for arbitration if necessary.

### Key Designs

1.  **Scheduler**:
    - **Function**: Determines whether to prioritize PoT or text2SQL based on table features and question semantics.
    - **Mechanism**: Built on MobileBERT + two-layer MLP (only 24.65M parameters). It inputs table meta-features (size, schema, data types) and question semantics to output probabilities for PoT and text2SQL. The higher-probability path is executed first; if its answer matches CoT, the other path is skipped.
    - **Design Motivation**: The advantage of different reasoning paths depends on model characteristics and question types. Intelligent scheduling avoids unnecessary LLM calls. Training data is derived from reasoning labels of three LLMs on WikiTQ/TabMWP/TabFact.

2.  **Confidence Checker**:
    - **Function**: Calculates a confidence score for each candidate answer to decide if Judge Agent arbitration is required.
    - **Mechanism**: Fine-tuned based on DeBERTaV3-large (~435M parameters). It inputs the table, question, and candidate answer, outputting confidence scores for each path. If the highest confidence exceeds a threshold $\theta=0.1$, that answer is selected; otherwise, the Judge Agent is called for a comprehensive decision.
    - **Design Motivation**: To avoid calling the expensive LLM Judge every time, as lightweight models can complete high-quality answer filtering in most cases.

3.  **Code Generation & Debugging**:
    - **Function**: Iteratively repairs syntax and logic errors in code generated by PoT/text2SQL.
    - **Mechanism**: PoT/text2SQL agents generate and execute code. If an error occurs, the corresponding Debug Agent (PDA/SDA) repairs it, iterating up to $N=3$ times. An early stop condition is introduced: if new code is highly similar to the previous version and the execution result remains the same, the process stops.
    - **Design Motivation**: Code reasoning is naturally prone to syntax errors, whereas text reasoning (CoT) gains little from iterative repair. Thus, debugging is restricted to code-based paths to balance cost and effect.

### Loss & Training
The Scheduler and Confidence Checker were trained on 173,664 samples. Scheduler labels indicate whether the PoT/text2SQL path was correct, while CC labels reflect the correctness of each of the three paths. All LLM agents share a single backbone model, distinguished only by role prompts.

## Key Experimental Results

### Main Results

| Benchmark | Metric | MATA | MixSC | SynTQA | TabLaP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Penguins (Avg) | EM | 0.881 | 0.626 | 0.810 | 0.524 |
| Penguins (Avg) | F1 | 0.881 | 0.637 | 0.811 | 0.544 |
| TableBench (Avg) | EM | 0.451 | 0.286 | 0.322 | 0.260 |
| TableBench (Avg) | F1 | 0.482 | 0.331 | 0.362 | 0.307 |

### Ablation Study

| Configuration | Penguins EM | TableBench EM | Description |
| :--- | :--- | :--- | :--- |
| MATA (Full) | 0.881 | 0.451 | Full framework |
| w/o Scheduler | ~0.86 | ~0.43 | Executes all paths; increased LLM calls |
| w/o CC (JA only) | ~0.85 | ~0.42 | Calls Judge Agent every time |
| w/o Debug | ~0.82 | ~0.38 | No code debugging |

### Key Findings
- MATA shows the most significant gains on small models (3B-7B): Qwen2.5-3b improved from 0.163 EM (TabLaP) to 0.291; Mistral-7b improved from 0.036 to 0.294.
- On large models, MATA maintains an advantage, though the gap narrows as larger models possess stronger native reasoning.
- The Scheduler effectively reduces LLM calls by approximately 30-40% while maintaining or even improving accuracy.

## Highlights & Insights
- The design combining lightweight tool models (totaling less than 1B parameters) with LLM agents is highly practical—the Scheduler and CC act as "gatekeepers," avoiding unnecessary expensive reasoning calls.
- The complementarity of three reasoning paths is fully verified: CoT excels at ambiguous/intuitive questions, PoT at numerical computation, and text2SQL at precise structured queries.
- The model-agnostic design allows the framework to be directly migrated to any new LLM, which is valuable given the rapid iteration of open-source models.

## Limitations & Future Work
- Training of the Scheduler and CC depends on specific datasets (WikiTQ/TabMWP/TabFact), which may limit generalization to tables with significant domain shifts.
- Currently, only single-table reasoning is supported; multi-table relational QA has not yet been addressed.
- The maximum iteration $N=3$ for the Debug loop is empirical; tasks of varying complexity may require adaptive adjustment.

## Related Work & Insights
- **vs MixSC**: MixSC only integrates CoT and Python paths using self-consistency voting, lacking text2SQL and intelligent scheduling. MATA's average EM is 25.5% higher.
- **vs SynTQA**: SynTQA integrates text2SQL and E2E TQA but does not support model switching. MATA's model-agnostic design gives it a huge advantage on small models.
- **vs TabLaP**: TabLaP relies on the collaboration of multiple different LLMs and only supports specific models. MATA achieves better results by unifying capabilities around a single backbone.

## Rating
- Novelty: ⭐⭐⭐⭐ The architecture combining lightweight tools and multi-agent coordination is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely broad coverage with 10 LLMs, two benchmarks, and three metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed algorithmic descriptions.
- Value: ⭐⭐⭐⭐ The model-agnostic framework provides direct reference value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](../../AAAI2026/multi_agent/hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)

</div>

<!-- RELATED:END -->
