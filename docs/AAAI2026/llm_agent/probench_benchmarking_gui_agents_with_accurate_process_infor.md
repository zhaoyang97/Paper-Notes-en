---
title: >-
  [Paper Note] ProBench: Benchmarking GUI Agents with Accurate Process Information
description: >-
  [AAAI 2026][LLM Agent][GUI Agent evaluation] ProBench is proposed as the first mobile GUI Agent benchmark that evaluates both final state and operational process: 200+ challenging tasks cover 34 mainstream Chinese and English apps. A Process Provider (Structure Description Converter + MLLM Summarizer) automatically captures accurate intermediate process information. Evaluation reveals that even the strongest model, Gemini 2.5 Pro, completes only 40.1% of tasks, exposing three prevalent issues: insufficient grounding, poor awareness of action history, and oversimplified task planning.
tags:
  - AAAI 2026
  - LLM Agent
  - GUI Agent evaluation
  - process information
  - mobile benchmark
  - Process Provider
  - bilingual applications
date: 2026-05-08
content_hash: 82b04427f51aa972
---

# ProBench: Benchmarking GUI Agents with Accurate Process Information

**Conference**: AAAI 2026
**arXiv**: [2511.09157](https://arxiv.org/abs/2511.09157)
**Code**: None (based on adbutils)
**Area**: LLM Agent / GUI Agent / Benchmark
**Keywords**: GUI Agent evaluation, process information, mobile benchmark, Process Provider, bilingual applications

## TL;DR
ProBench is proposed as the first mobile GUI Agent benchmark that evaluates both final state and operational process: 200+ challenging tasks cover 34 mainstream Chinese and English apps. A Process Provider (Structure Description Converter + MLLM Summarizer) automatically captures accurate intermediate process information. Evaluation reveals that even the strongest model, Gemini 2.5 Pro, completes only 40.1% of tasks, exposing three prevalent issues: insufficient grounding, poor awareness of action history, and oversimplified task planning.

## Background & Motivation

**Background**: GUI Agent benchmarks (AndroidWorld, AndroidLab, etc.) can evaluate agents executing GUI tasks on real devices, but nearly all assessments examine only the final screen state to determine task completion.

**Limitations of Prior Work**: Evaluating solely the final state leads to "false successes." For example, in the task "buy the cheapest wireless mouse," if an agent selects an item without sorting by price, the final screen still shows a purchase confirmation and is mistakenly marked as success. Critical intermediate steps (e.g., "sort by price") are simply invisible on the final page.

**Key Challenge**: GUI tasks are inherently multi-step sequential operations, and not all critical information is present on the last few pages. The few works that attempt process-level evaluation (SPA-BENCH, A3) either require manual annotation of intermediate states (not scalable) or rely on LLM-based decomposition (insufficiently accurate).

**Goal**: How to automatically and accurately capture operational process information so that GUI Agent evaluation accounts for both final outcomes and critical intermediate steps?

**Key Insight**: Design a Process Provider to automatically supply process information via two optional approaches — (a) parsing the page hierarchy to obtain action descriptions; (b) using an MLLM to compare pre- and post-action screenshots to identify changes.

**Core Idea**: Distinguish between State-related Tasks (evaluated by final state only) and Process-related Tasks (requiring inspection of critical intermediate actions), with the Process Provider automatically supplying accurate process information.

## Method

### Overall Architecture
ProBench comprises three modules: (1) Task Curation; (2) Dynamic Environment; (3) Evaluation Pipeline (incorporating the Process Provider).

### Key Designs

1. **Two-Category Task Taxonomy**:

    - **State-related Task**: All necessary information is visible in the final screenshot (e.g., "check Alipay balance"); evaluating the final state suffices.
    - **Process-related Task**: Requires specific intermediate actions that are not fully reflected in the final state (e.g., "find the highest-rated sushi restaurant and view the full menu" — requires sorting + filtering + selection, none of which are visible on the final page).
    - **Design Motivation**: A large proportion of real-world tasks require a correct operational process; inspecting results alone is insufficient.

2. **Process Provider — Two Optional Components**:

    - **Structure Description Converter**: After each click, parses the accessibility (a11y) tree to locate the minimal clickable node and extracts its `text`/`content_desc`/`resource_id` attributes to generate a human-readable action description.
    - **MLLM-based Summarizer**: Concatenates pre- and post-action screenshots with annotated click coordinates, prompting an MLLM to compare differences and generate an action summary (e.g., "clicked the search box on the Airbnb home page").
    - **Design Motivation**: The Structure Description Converter is fast and accurate but depends on a11y tree quality; the MLLM Summarizer is more flexible but requires MLLM inference. The two components are complementary and user-selectable.

3. **Evaluation Accuracy Validation**:

    - State-related Task: evaluator accuracy 96.0%
    - Process-related Task + Structure Description Converter: 89.7%
    - Process-related Task + MLLM Summarizer: 94.1%

### Benchmark Scale
34 mainstream applications (14 English + 20 Chinese), 200+ tasks, spanning media, news, social, shopping, and lifestyle scenarios. Each task allows up to 15 interaction steps.

## Key Experimental Results

### Main Results

| Model | State-related | Process-related | Overall |
|-------|-------------|----------------|---------|
| **Gemini 2.5 Pro** | **45.6** | **27.9** | **40.1** |
| Qwen2.5-VL-72B | 40.9 | 27.9 | 36.9 |
| Qwen2.5-VL-32B | 18.8 | 11.8 | 16.6 |
| Qwen2.5-VL-7B | 6.7 | 1.5 | 5.1 |
| UI-TARS-1.5-7B | 11.4 | 2.9 | 8.8 |
| GPT-4o | 0.0 | 0.0 | 0.0 |
| Claude 4 Sonnet | 0.0 | 0.0 | 0.0 |
| UI-R1-E-3B | 0.0 | 0.0 | 0.0 |

### Error Analysis

| Model | Incomplete Task Rate | Early Stop Rate (among incomplete) |
|-------|-------------|----------|
| Gemini 2.5 Pro | 90.0% | 49.6% |
| Qwen2.5-VL-72B | 71.5% | 50.0% |
| Qwen2.5-VL-7B | 93.7% | 63.7% |

### Key Findings
- **Even the strongest model falls below 50%**: Gemini 2.5 Pro achieves only 40.1%, demonstrating that GUI operations in real online environments remain highly challenging.
- **Process-related tasks are substantially harder than State-related tasks**: All models show significant performance drops on process tasks (Gemini: 45.6% → 27.9%), validating the necessity of process-level evaluation.
- **GPT-4o and Claude 4 Sonnet fail severely**: Both exhibit extremely poor grounding ability — they cannot accurately localize GUI element coordinates and fail even at the first step.
- **Social and lifestyle apps are the most difficult**: Frequently refreshed content, complex layouts, and numerous pop-up advertisements coincide precisely with common user needs.
- **GUI-specific models have limited generalization**: UI-TARS-1.5-7B outperforms general-purpose models of the same scale on English tasks but falls far behind on Chinese tasks, revealing the limitations of domain-specific fine-tuning.
- **Repetitive action loops** are a pervasive problem: Agents fail to recognize that an action has been completed (e.g., having already clicked a search box), leading to repeated clicks at the same position.
- **Oversimplified task planning**: Agents tend to paste the entire complex instruction directly into a search box rather than executing it in discrete steps.

## Highlights & Insights
- **The State vs. Process task taxonomy** is the core contribution — it is the first work to systematically distinguish the two categories of GUI tasks and to design an automated process information collection scheme.
- **The three error pattern taxonomy** is highly valuable: (a) insufficient grounding (failure to localize GUI elements); (b) poor awareness of action history (repetition and loops); (c) oversimplified task planning (equating complex tasks with search). These findings clearly indicate future research directions for GUI Agents.
- **The inclusion of Chinese-language apps** fills an important gap in existing benchmarks, most of which cover only English apps.
- **The real online environment** (as opposed to simulated or offline apps) more faithfully reflects challenges encountered in actual usage scenarios, including network latency, dynamically changing content, and anti-automation mechanisms.

## Limitations & Future Work
- Evaluation is binary (success/failure), lacking a measure of partial completion.
- The scale of 200+ tasks is relatively limited, and the benchmark is heavily oriented toward apps dominant in the Chinese market (WeChat, Alipay, etc.).
- The accuracy of the Process Provider (89.7%–94.1%) leaves room for further improvement.
- No experimental validation of proposed improvements is provided (e.g., whether incorporating action history memory is effective).
- Direct comparison of results with analogous benchmarks such as SPA-BENCH and A3 is absent.

## Related Work & Insights
- **vs. AndroidWorld**: AndroidWorld relies solely on terminal-state evaluation and uses open-source apps from F-Droid; ProBench adds process-level evaluation and covers real online mainstream apps.
- **vs. SPA-BENCH**: SPA-BENCH manually decomposes task steps, limiting scalability; ProBench automates this via the Process Provider.
- **vs. A3**: A3 uses LLM-based task decomposition for completion assessment, with limited accuracy; ProBench achieves greater reliability through a11y tree parsing and MLLM screenshot comparison.
- **Implications for GUI Agent development**: (a) Models require better grounding training data; (b) attention mechanisms over action history urgently need improvement to prevent loops; (c) task planning capability is more difficult to improve than grounding capability.

## Rating
- Novelty: ⭐⭐⭐⭐ The State/Process taxonomy and Process Provider design are novel, though the work is overall a benchmark contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 models + three-category error analysis + app-category analysis + evaluation pipeline validation, but lacks direct improvement experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Error case analyses are vivid and intuitive, problem definitions are clear, and the appendix includes the full task list and prompts.
- Value: ⭐⭐⭐⭐⭐ Exposes the reality that "the strongest model achieves less than 50% success rate"; the three error pattern summary provides important guidance to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](../../ACL2026/llm_agent/lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](../../NeurIPS2025/llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](history-aware_reasoning_for_gui_agents.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](../../ACL2026/llm_agent/fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)

</div>

<!-- RELATED:END -->
