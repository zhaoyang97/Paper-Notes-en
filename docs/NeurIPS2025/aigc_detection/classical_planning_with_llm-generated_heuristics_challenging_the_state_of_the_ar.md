---
title: >-
  [Paper Note] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code
description: >-
  [NeurIPS 2025][AIGC Detection][Classical Planning] This paper proposes having LLMs **generate Python code for domain-dependent heuristic functions** (rather than directly generating plans). Candidate heuristics are obtai…
tags:
  - "NeurIPS 2025"
  - "AIGC Detection"
  - "Classical Planning"
  - "LLM Code Generation"
  - "Heuristic Function"
  - "PDDL"
  - "Greedy Best-First Search"
  - "Domain-Dependent Planning"
date: 2026-05-08
content_hash: 8ca2bec8c4ac07ca
---

# Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code

**Conference**: NeurIPS 2025
**arXiv**: [2503.18809](https://arxiv.org/abs/2503.18809)  
**Code**: [Publicly Available](https://doi.org/10.5281/zenodo.14964025)  
**Authors**: Augusto B. Corrêa, André G. Pereira, Jendrik Seipp
**Area**: AI Planning — Classical Planning, Heuristic Search
**Keywords**: Classical Planning, LLM Code Generation, Heuristic Function, PDDL, Greedy Best-First Search, Domain-Dependent Planning

## TL;DR

This paper proposes having LLMs **generate Python code for domain-dependent heuristic functions** (rather than directly generating plans). Candidate heuristics are obtained via $n$ samples and the best is selected on a training set, then injected into the Python planner Pyperplan for use with GBFS. The approach surpasses all C++ Fast Downward traditional heuristics on 8 IPC 2023 benchmark domains using pure Python, matches the SOTA learned planner $h^{\mathrm{WLF}}_{\mathrm{GPR}}$, and guarantees 100% correctness for all plans found.

## Background & Motivation

- **Background**: Classical Planning is a foundational AI problem where tasks are formalized in PDDL. Mainstream methods rely on heuristic search, with heuristics falling into three categories: ① domain-independent heuristics (e.g., $h^{\mathrm{FF}}$, $h^{\mathrm{add}}$), which are general but imprecise; ② manually crafted domain-dependent heuristics requiring substantial expert knowledge; ③ per-domain learned heuristics (e.g., $h^{\mathrm{WLF}}_{\mathrm{GPR}}$), each requiring separate training.
- **Limitations of Prior Work**: Using LLMs directly for end-to-end planning (generating plans from PDDL tasks) is unreliable—even reasoning models (DeepSeek R1, Gemini Thinking) frequently produce incorrect plans and fail to generalize to problems larger than those seen during training. The strongest end-to-end method AoT+ supports only 5 blocks in Blocksworld, while test instances can contain up to 488 blocks.
- **Key Challenge**: LLMs possess domain understanding and code generation capabilities but cannot reliably perform multi-step planning; classical planners offer a sound search framework guaranteeing correctness but lack domain-specific knowledge to guide search efficiently.
- **Goal**: How can the code generation capabilities of LLMs be leveraged to automatically produce high-quality domain-dependent heuristic functions for classical planning problems?
- **Key Insight**: Reframe the problem from "having LLMs solve planning problems" to "having LLMs write tool code for planning problems"—LLMs generate heuristic functions (Python code) while the search algorithm guarantees correctness, with each component fulfilling its respective role.
- **Core Idea**: Rather than planning directly, LLMs generate multiple candidate heuristic function implementations per domain; the best is selected and injected into the planner, replacing the "direct solving" paradigm with a "tool generation" paradigm to enhance planning capability.

## Method

### Overall Architecture

The overall pipeline is remarkably simple (as shown in Figure 1 of the paper):

1. **Prompt Construction**: The PDDL domain description, example tasks, example heuristic code, and Pyperplan interface code are organized into a prompt.
2. **Multiple Sampling**: The same prompt is used to query the LLM $n$ times ($n=25$), yielding $n$ candidate heuristic functions $h_1, \dots, h_n$.
3. **Training Set Evaluation**: Each heuristic is run with GBFS on the IPC training set (5-minute limit per task) and ranked by number of problems solved.
4. **Selection and Injection**: The best-performing heuristic $h_{\text{best}}$ on the training set is injected into Pyperplan for evaluation on the test set.

Key properties: ① **No iterative LLM–planner interaction**—the process is completed in a single pass; ② **Fixed cost per domain**—once a heuristic is selected, it can be applied to unlimited new tasks in that domain; ③ **Correctness guaranteed by the search algorithm**—the heuristic only affects search efficiency.

### Key Designs

1. **Multi-Component Prompt Design**

    - Function: Provides the LLM with sufficient context to generate high-quality, compilable, and executable heuristic function code.
    - Mechanism: The prompt comprises 7 carefully designed components—① the PDDL domain file; ② PDDL files for the smallest and largest training tasks; ③ domain files, tasks, and heuristic Python code for two example domains (Gripper with a perfect heuristic, Logistics with a simple heuristic); ④ an example of Pyperplan's state representation for the target domain; ⑤ an example of static information representation; ⑥ the Task/Action Python class code from Pyperplan; ⑦ a checklist of common pitfalls.
    - Design Motivation: Example heuristics demonstrate Pyperplan's interface and coding style (few-shot), PDDL files convey domain semantics, and the checklist is derived from the authors' observations of typical LLM errors to preempt common bugs. Ablation experiments confirm that removing any component degrades the quality of the best heuristic.

2. **Multiple Sampling + Training Set Selection Strategy**

    - Function: Filters high-quality heuristics from the stochastic LLM outputs.
    - Mechanism: Temperature is set to 1.0 to promote diversity, generating $n=25$ candidate heuristics. Candidates are ranked on the training set using a two-level criterion: first by number of problems solved, then by IPC agile score ($1 - \frac{\log t}{\log 300}$) as a tiebreaker. Experiments show that the largest gain in coverage occurs when $n$ increases from 1 to 5, and by $n=25$ the training set is nearly fully covered across all domains.
    - Design Motivation: LLM output quality exhibits high variance (standard deviation up to ±94 under high-temperature sampling), making single-sample results highly unstable. Sampling multiple candidates and selecting the best yields consistently high-quality heuristics. This process is performed only once per domain; the total API cost is extremely low (DeepSeek V3: $0.25 USD for all 8 domains; DeepSeek R1: $6.12 USD).

3. **Decoupling Search Correctness from Heuristic Quality**

    - Function: Guarantees 100% correctness for all plans found, regardless of heuristic quality.
    - Mechanism: The heuristic function only influences the node expansion order in GBFS (i.e., search efficiency) and does not affect plan correctness—GBFS outputs a plan only upon reaching a goal state, and goal testing is handled internally by the Pyperplan framework. Even if an LLM generates a completely uninformative heuristic, the search will still find a correct plan (albeit less efficiently).
    - Design Motivation: This is the fundamental advantage over end-to-end LLM planning—end-to-end methods generate plans that must be validated with the VAL tool and are frequently incorrect, whereas the proposed method confines the LLM to the role of an "evaluation function," fully delegating correctness to a mature search framework.

### Loss & Training

There is no training or gradient optimization in the conventional sense. The "training set" is used solely to evaluate and select candidate heuristics, with no parameter updates involved. LLM API cost breakdown: DeepSeek V3 generates 200 heuristics (8 domains × 25) for a total of $0.25 USD; DeepSeek R1 totals $6.12 USD. In contrast, end-to-end planning requires 720 LLM calls (one per task), raising the R1 cost from $6.12 to $13.62.

## Key Experimental Results

### Main Results: LLM-Generated Heuristics vs. End-to-End LLM Planning vs. Domain-Independent Heuristics (Table 2)

IPC 2023 Learning Track, 8 domains, 90 test tasks per domain, 720 tasks total:

| Method | Blocks | Child. | Floor. | Miconic | Rovers | Sokoban | Spanner | Trans. | **Total** |
|--------|--------|--------|--------|---------|--------|---------|---------|--------|-----------|
| End-to-end Gemini Think. | 40 | 59 | 0 | 21 | 5 | 14 | 39 | 24 | 202 |
| End-to-end DeepSeek R1 | 17 | 40 | 0 | 24 | 10 | 8 | 47 | 28 | 174 |
| Pyperplan $h^0$ (BFS) | 6 | 9 | 1 | 30 | 12 | 24 | 30 | 8 | 120 |
| Pyperplan $h^{\mathrm{FF}}$ | 24 | 17 | 10 | 74 | 28 | 31 | 30 | 29 | 243 |
| + Gemini 2.0 Flash | 35 | 32 | 4 | **90** | 32 | 31 | 30 | 42 | 296 |
| + Gemini Think. | 37 | 14 | 8 | 88 | 39 | 32 | 30 | 57 | 305 |
| + DeepSeek V3 | 45 | **55** | 3 | 64 | 34 | 31 | **69** | 42 | **343** |
| + DeepSeek R1 | **66** | 22 | 4 | **90** | 32 | 30 | 70 | **59** | **373** |

### Ablation Study: Impact of Prompt Components (Table 4, Gemini 2.0 Flash Thinking, $n=25$)

| Prompt Variant | Best Coverage | Avg. Coverage | Failed Heuristics |
|----------------|--------------|---------------|-------------------|
| Full Prompt (original) | **423** | 267.0 | 64 |
| ⇆ Simplified instructions | 359 | 242.5 | 57 |
| − Remove PDDL domain description | 368 | 237.3 | 52 |
| − Remove PDDL tasks | 401 | 261.5 | 42 |
| ⇆ Replace with domain-independent heuristic | 404 | 263.2 | 64 |
| − Remove state representation example | 402 | 253.7 | 68 |
| − Remove static information representation | 404 | 243.8 | 57 |
| − Remove Pyperplan code | 401 | 270.0 | **97** |
| − Remove Checklist | 382 | 260.0 | 57 |

### Key Findings

1. **Python Surpasses C++**: Pyperplan (Python) + DeepSeek R1 heuristics (373 tasks) outperforms the best traditional heuristic in Fast Downward (C++) ($h^{\mathrm{add}}$ = 324) and matches the SOTA learned method $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ (371). Given that Fast Downward expands nodes up to **669×** faster than Pyperplan in certain domains, this demonstrates that heuristic quality can fully compensate for implementation efficiency gaps.

2. **Dramatic Improvement for Non-Reasoning Models**: Gemini 2.0 Flash solves only 19 tasks end-to-end but solves 296 tasks when used for heuristic generation (**15.6×** improvement); DeepSeek V3 improves from 48 to 343 tasks (**7.1×**). The heuristic generation paradigm eliminates the performance gap between reasoning and non-reasoning models.

3. **More Informative LLM Heuristics**: In domains such as Blocksworld, Spanner, Transport, and Childsnack, the DeepSeek R1 heuristic expands fewer states than $h^{\mathrm{FF}}$, indicating that LLMs do not succeed merely through exhaustive search but generate more accurate heuristics.

4. **Generalization to Unseen Domains**: On the entirely new domain Rod-Rings (never publicly released, absent from LLM training data), heuristics generated by OpenAI o3 solve 58 tasks, approaching Fast Downward $h^{\mathrm{FF}}$'s 59. On an obfuscated Blocksworld (all symbols replaced with random strings), the approach still solves 40/90 tasks, demonstrating that LLMs reason about the logical structure of a domain rather than merely recalling memorized heuristics.

5. **Selection Strategy Has Room for Improvement**: Ablation experiments show that the best heuristic in the candidate pool (423 tasks) substantially outperforms the selected heuristic (305 tasks), indicating that the current training-set-coverage selection strategy is suboptimal and that better heuristics in the pool go unchosen.

## Highlights & Insights

1. **"Generate Tools Rather Than Solve Directly" Paradigm**: This is the paper's most fundamental insight—LLMs are not well-suited for multi-step planning but excel at understanding domain semantics and writing code. Repositioning the LLM from "solver" to "tool generator," with the search algorithm responsible for correctness and the LLM responsible for encoding domain knowledge, allows each component to fulfill its natural role. This paradigm has broad transferability.

2. **Extremely Low Cost Model**: Generating 25 candidate heuristics requires only 25 LLM calls at a total cost of $0.25–$6.12, and the selected heuristic can be reused for an unlimited number of new tasks in the domain. Compared to end-to-end planning (one LLM call per task), the cost advantage grows linearly with the number of tasks.

3. **Surprising Python vs. C++ Conclusion**: A pure Python implementation surpassing a highly optimized C++ system quantifies the trade-off between "heuristic quality" and "implementation efficiency"—in planning, the correctness of the search direction is far more important than node expansion speed.

4. **Interpretability of LLM-Generated Heuristics**: The paper presents generated code for Blocksworld and Spanner—the Blocksworld heuristic identifies "misplaced goal blocks" and computes the number of blocks that must be moved above them, while the Spanner heuristic performs greedy optimal matching and precomputes shortest paths. This code is human-readable and interpretable.

## Limitations & Future Work

1. **Strong Dependence on PDDL Formalization**: The method requires problems to be specified in PDDL and does not support planning tasks described in natural language. Although existing NL→PDDL conversion work can bridge this gap, it adds pipeline complexity.

2. **Suboptimal Heuristic Selection Strategy**: Ablation experiments show that the best heuristic in the candidate pool (423 tasks) far exceeds the performance of the selected heuristic (305 tasks), indicating substantial room for improvement in the training-set-coverage-based selection scheme.

3. **No Iterative Feedback**: The current pipeline is single-pass and does not feed failure cases or previously generated heuristics back to the LLM. The success of FunSearch suggests that iterative refinement could further improve quality.

4. **Pyperplan Implementation Bottleneck**: The Python implementation is significantly less memory-efficient and slower than C++, limiting performance in domains such as Floortile. Porting LLM-generated heuristics to Fast Downward could yield substantial performance gains.

5. **Limited Domain Coverage**: Because Pyperplan does not support the Ferry and Satellite domains, experiments cover only 8 of the 10 IPC 2023 domains.

## Related Work & Insights

| Method | Paradigm | Correctness Guarantee | Domain Adaptation | Reusability | Implementation Cost |
|--------|-----------|-----------------------|-------------------|-------------|---------------------|
| End-to-end LLM planning | LLM directly generates plans | No | Zero-shot | No (one call per task) | High (linear inference cost) |
| AoT+ | LLM + search strategy | No | Zero-shot | No | High |
| $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ | GP regression learned heuristics | Yes (GBFS guarantee) | Per-domain training | Yes (within domain) | Medium (feature extraction + GP training) |
| Katz et al. | LLM generates successor/goal code | Yes (search guarantee) | Requires human feedback | Partial | Medium |
| Silver et al. | LLM generates policy programs | No (no search) | Zero-shot | Yes | Low |
| Tuisov et al. | LLM generates Rust heuristics | Yes | Requires manual PDDL translation | No (task-level) | High (Rust translation) |
| **Ours** | **LLM generates Python heuristics** | **Yes (GBFS guarantee)** | **Zero-shot** | **Yes (domain-level)** | **Extremely low ($0.25–$6.12)** |

**Directions for Transfer**:
- The "generate tool code rather than solve directly" paradigm is transferable to **combinatorial optimization** (e.g., LLMs generating neighborhood search heuristics for TSP/VRP), **theorem proving** (LLMs generating proof strategies rather than proofs directly), and other domains.
- Optimizing the candidate pool selection strategy is a clear direction for improvement—ensemble methods, cross-validation, or diversity-based selection could be explored.
- Combining LLM heuristic generation with FunSearch's iterative refinement loop may enable automated generation of even higher-quality heuristics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Repositioning LLMs from "planner" to "heuristic generator" is a compelling problem formulation with a profound paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ IPC 2023 standard benchmark with 8 domains and 720 tasks, 5 LLM comparisons, 6 C++ heuristic baselines + SOTA learned methods, complete ablation study, and an elegantly designed memorization-vs-reasoning validation (obfuscated + novel domains).
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, the method is minimal (the entire pipeline is comprehensible from a single figure), experiments are rigorously organized, and limitations are candidly discussed.
- Value: ⭐⭐⭐⭐⭐ The work has far-reaching implications for both the AI planning community and LLM application paradigms. Its extremely low cost, simple deployment, and correctness guarantees give it high practical value, and the "generate tools rather than solve directly" insight transfers broadly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](../../ACL2026/aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)
- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)

</div>

<!-- RELATED:END -->
