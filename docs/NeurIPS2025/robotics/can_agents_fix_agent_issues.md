---
title: >-
  [Paper Note] Can Agents Fix Agent Issues?
description: >-
  [NeurIPS 2025][Robotics][Agent system maintenance] This paper presents the first systematic study of automated issue resolution in LLM-based agent systems. Through manual analysis of 201 real-world agent issues, the authors construct a taxonomy comprising 6 categories and 20 subcategories, invest approximately 500 person-hours to build AgentIssue-Bench—a benchmark of 50 reproducible tasks—and find that state-of-the-art software engineering (SE) agents (e.g., SWE-agent, Agentless, AutoCodeRover) achieve correct resolution rates of only 3.33%–12.67% on agent issues, far below their 23%–51% rates on conventional software.
tags:
  - NeurIPS 2025
  - Robotics
  - Agent system maintenance
  - automated issue resolution
  - software engineering agents
  - bug taxonomy
  - benchmark
date: 2026-05-08
content_hash: b886db7142058587
---

# Can Agents Fix Agent Issues?

**Conference**: NeurIPS 2025
**arXiv**: [2505.20749](https://arxiv.org/abs/2505.20749)
**Code**: [https://github.com/alfin06/AgentIssue-Bench](https://github.com/alfin06/AgentIssue-Bench)
**Area**: Robotics
**Keywords**: Agent system maintenance, automated issue resolution, software engineering agents, bug taxonomy, benchmark

## TL;DR

This paper presents the first systematic study of automated issue resolution in LLM-based agent systems. Through manual analysis of 201 real-world agent issues, the authors construct a taxonomy comprising 6 categories and 20 subcategories, invest approximately 500 person-hours to build AgentIssue-Bench—a benchmark of 50 reproducible tasks—and find that state-of-the-art software engineering (SE) agents (e.g., SWE-agent, Agentless, AutoCodeRover) achieve correct resolution rates of only 3.33%–12.67% on agent issues, far below their 23%–51% rates on conventional software.

## Background & Motivation

**Background**: LLM-based agent systems have emerged as a new software paradigm, finding widespread adoption in medicine, programming, robotics, and counseling. Prominent agent frameworks such as MetaGPT, AutoGen, CrewAI, and GPT-Engineer have amassed tens of thousands to hundreds of thousands of GitHub stars, attesting to the field's vitality. Concurrently, a class of SE agents—including SWE-agent, Agentless, and AutoCodeRover—has been developed specifically for automated code issue resolution, demonstrating strong performance on conventional Python software; Agentless, for instance, correctly resolves 50.80% of issues on SWE-bench.

**Limitations of Prior Work**: Like any software product, agent systems inevitably accumulate bugs and feature requests. MetaGPT alone had accrued over 800 GitHub issues by May 2025, imposing substantial maintenance burdens. However, agent systems differ fundamentally from conventional software: they interact with LLM providers, manage agent memory states, invoke external tools, and handle the inherent non-determinism of LLM outputs. These distinctive characteristics imply that agent system issues differ substantially in nature and repair difficulty from conventional software issues. Whether existing SE agents can effectively resolve agent system issues remains entirely unexplored.

**Key Challenge**: Existing SE agents and issue-resolution benchmarks (e.g., the SWE-bench family) target conventional software systems, entirely overlooking the emerging agent software paradigm. The core components of agent systems—LLM-controlled "brains" for task decomposition and planning, perception components for receiving environmental inputs, action components for tool-based environment interaction, and memory mechanisms—introduce problem types absent from conventional software, such as LLM provider API compatibility issues, prompt management errors, and abnormal agent workflow loops. Without a systematic understanding of agent issue characteristics, developing effective automated repair approaches is infeasible.

**Goal**: (1) What issue types commonly arise in agent systems, and how do they differ from conventional software issues? (2) How can a reproducible agent issue resolution benchmark be constructed? (3) How effective are current state-of-the-art SE agents at resolving agent issues?

**Key Insight**: The authors take a real-world-grounded approach, collecting GitHub issues and developer-submitted fix patches from 16 mainstream agent systems, then systematically analyzing and categorizing agent issues via grounded theory. Building on this foundation, substantial human effort is invested to construct reproducible benchmark environments, with each task containerized in Docker and accompanied by failure-triggering test scripts to ensure verifiability. This "understand first, benchmark second, evaluate third" research paradigm ensures the reliability of conclusions.

**Core Idea**: Agent system issues exhibit characteristics fundamentally distinct from conventional software—involving LLM provider compatibility, tool invocation, memory management, LLM operations, workflow control, and other agent-specific components—while current state-of-the-art SE agents demonstrate extremely limited capacity to resolve these issues, underscoring the urgent need for maintenance tooling tailored to agent systems.

## Method

### Overall Architecture

The study proceeds in three core phases: (1) **Agent Issue Taxonomy Construction**—collecting 201 high-quality GitHub issues from 16 mainstream agent systems, applying grounded theory methods with manual annotation to produce a taxonomy of 6 categories and 20 subcategories; (2) **AgentIssue-Bench Construction**—filtering the 201 issues to 50 stably reproducible instances, each equipped with a Docker environment, failure-triggering tests, and both buggy and patched repository versions; (3) **SE Agent Evaluation**—assessing SWE-agent, AutoCodeRover, and Agentless, each paired with GPT-4o and Claude-3.5-Sonnet as backbone LLMs, on AgentIssue-Bench.

### Key Designs

1. **Agent Issue Taxonomy Construction Methodology**:

    - **Function**: Systematically collect and categorize issues from real-world agent systems to establish the first agent issue taxonomy.
    - **Mechanism**: The GitHub Search API is queried with the keyword "AI agents" to retrieve 50 repositories. Manual filtering retains genuine LLM-based agent systems (excluding paper lists, tutorials, and unrelated repositories), further narrowing to active projects with 1k+ stars and 30+ issues, yielding 16 agent systems (including MetaGPT, AutoGen, CrewAI, GPT-Engineer, BabyAGI, CAMEL, and ChatDev). For each system's issues, three selection criteria are applied: (i) the issue is closed and accompanied by a developer-submitted fix patch serving as ground truth for root cause understanding; (ii) the issue description is clear and unambiguous; (iii) each issue contains exactly one problem. This process yields 201 high-quality issues. Of these, 171 (85%) are used for taxonomy construction and 30 (15%) for evaluation. Three annotators with substantial software development and machine learning experience apply open coding, decomposing each issue into segments and assigning descriptive labels, then merging and relating labels into a structured taxonomy through consensus discussion. The evaluation phase achieves a Cohen's Kappa of 0.849 between two independent annotators, with no new categories emerging, validating the taxonomy's generalizability and reliability.
    - **Design Motivation**: Prior work lacked systematic understanding of agent issues. Conventional software taxonomies cannot capture problem types unique to agent systems (e.g., LLM provider compatibility, agent memory errors, workflow anomalies). Establishing a clear taxonomy is a prerequisite for targeted evaluation and improvement of automated repair tools.

2. **Six-Category Agent Issue Taxonomy**:

    - **Function**: Organize agent system issues into 6 categories and 20 subcategories, each with detailed definitions and real-world examples.
    - **Mechanism**: The 6 categories are:
      **(a) LLM Provider Incompatibility (7.46%)**: Includes dependency incompatibility (e.g., breaking API changes in the `anthropic` library), unsupported new models (e.g., missing GPT-4 Turbo support), and incompatible API parameters (e.g., passing a `stop` parameter to `o1-preview`, which does not support it).
      **(b) Tool-Related Issues (18.41%)**: Includes missing tool dependencies (e.g., absent `tenacity` module), tool misconfiguration (e.g., inability to configure the DuckDuckGo retriever independently), tool implementation errors (e.g., crash when decoding `.docx` files in SWE-agent), and tool interface misuse (e.g., Wikipedia API's `auto_suggest` silently correcting search terms).
      **(c) Memory-Related Issues (14.43%)**: Includes memory initialization errors (e.g., failure to locate a crew instance when resetting memory in CrewAI), memory content errors (e.g., message storage logic causing content loss), and memory dependency issues (e.g., import failures after `InnerMessage` was renamed to `AgentMessage`).
      **(d) LLM Operation Issues (31.84%, largest category)**: Includes model access configuration errors (e.g., incompatible API key setup logic), token usage configuration errors (e.g., `max_tokens` defaulting to `NOT_GIVEN`, causing type errors), model output handler errors (e.g., missing exception handling when Gemini blocks sensitive content), model dependency issues (e.g., `transformers` version conflicts), context length issues (e.g., inputs exceeding maximum context length), and prompt-related issues (e.g., Introduction/Conclusion sections still generated in English under multilingual settings).
      **(e) Workflow Issues (6.97%)**: For example, downstream steps failing after multiple parallel prerequisite steps complete.
      **(f) General Utility Issues (20.90%)**: Includes non-LLM implementation errors (e.g., UI, Docker, and logging problems), general dependency issues (e.g., `pytest` version incompatibility), and general configuration issues (e.g., YAML file encoding problems). Only this last category overlaps substantially with conventional software issues; the remaining five categories are closely tied to agent-specific core components.
    - **Design Motivation**: This taxonomy reveals the complexity of agent system maintenance—developers must simultaneously manage the correctness of LLM provider interfaces, LLM operation logic, memory mechanisms, and tool invocations. The dominance of LLM operation issues (31.84%) identifies LLM interaction as the most error-prone component of agent systems.

3. **AgentIssue-Bench Construction**:

    - **Function**: Construct a benchmark of 50 stably reproducible tasks from the 201 issues, each with a complete executable environment.
    - **Mechanism**: A rigorous three-step filtering pipeline is employed. **Step 1 – Fault Reproduction**: The buggy commit for each issue is checked out, the agent system environment is configured, and a hand-written failure-triggering test script is developed to reproduce the buggy behavior described in the issue; issues where the same buggy behavior cannot be observed are discarded. **Step 2 – Patch Validation**: The corresponding patched commit is checked out and the failure-triggering test is run; only issues where the patched version passes the test are retained. **Step 3 – Non-Flakiness Validation**: Steps 1 and 2 are repeated three times to eliminate test instability caused by LLM non-determinism. After this pipeline, 201 issues are reduced to 50 reproducible tasks. Each task instance comprises: (i) the user-reported issue description; (ii) the buggy repository version; (iii) the developer-submitted fix patch as ground truth; (iv) the failure-triggering test script; and (v) a Docker-containerized environment with all dependencies and configurations. All Docker images are hosted on Docker Hub for one-click retrieval and execution. The entire reproduction process required approximately 500 person-hours.
    - **Design Motivation**: Reproducing agent issues is substantially harder than reproducing conventional software issues for four reasons: (i) LLM output non-determinism makes workflow errors difficult to reproduce consistently; (ii) external resources (tools, dependencies, LLM providers) may change after an issue is reported; (iii) issue descriptions often lack sufficient reproduction steps; (iv) unexpected errors unrelated to the described issue may arise during environment setup. These challenges necessitate considerable human investment, explaining why no reproducible agent issue resolution benchmark has previously existed.

### Evaluation Metric Design

Three evaluation levels are employed: **(1) Localization Accuracy**—whether the generated patch modifies the same locations as the developer patch, measured at both file and function granularity; **(2) Plausible Resolution Rate**—whether the generated patch causes the failure-triggering test to pass (without guaranteeing semantic correctness); **(3) Correct Resolution Rate**—whether the generated patch is semantically equivalent to the developer patch, as determined by human annotators beyond merely passing the test. The distinction between plausible and correct resolution is necessary because test coverage is often insufficient: plausible patches may overfit the test case without genuinely addressing the underlying issue (i.e., overfitting patches). To mitigate LLM randomness, all experiments are repeated three times and averaged.

### SE Agent Experimental Setup

Three fully open-source SE agents with strong performance on conventional software issue resolution are selected: **(1) SWE-agent**—interacts with code repository environments via a custom Agent-Computer Interface (ACI), enabling file manipulation and bash command execution; **(2) AutoCodeRover**—equipped with a suite of code search tools, iteratively retrieving relevant code context to navigate repositories and localize issues; **(3) Agentless**—incorporates human expert knowledge into the agent workflow, integrating hierarchical localization and regression testing to improve resolution rates. Each SE agent is evaluated with both GPT-4o and Claude-3.5-Sonnet as backbone LLMs, using the hyperparameter settings from the original releases. All experiments are executed within AgentIssue-Bench's Docker environments to ensure environmental consistency.

## Key Experimental Results

### Main Results

| SE Agent | Backbone LLM | Plausible Rate | Correct Rate | File-level Loc. | Func-level Loc. | Avg. Cost |
|----------|-------------|---------------|-------------|----------------|----------------|-----------|
| Agentless | GPT-4o | 18.67% | 6.00% | 22.97% | 14.40% | $0.65 |
| Agentless | Claude-3.5-S | 12.67% | 8.67% | 21.86% | 11.54% | $0.33 |
| AutoCodeRover | GPT-4o | 12.67% | 4.67% | 17.05% | 11.30% | $0.23 |
| AutoCodeRover | Claude-3.5-S | 17.33% | **12.67%** | **25.61%** | **18.67%** | $0.05 |
| SWE-agent | GPT-4o | 5.33% | 3.33% | 12.65% | 11.77% | $1.15 |
| SWE-agent | Claude-3.5-S | 6.67% | 6.67% | 15.58% | 11.26% | $0.57 |

Compared to conventional software (SWE-bench Lite): these SE agents achieve resolution rates of 23.20%–50.80% on conventional Python software issues, versus only 3.33%–12.67% on agent issues—**a decline of 70%–90%**.

### Per-Category Resolution Breakdown

| Issue Category | Resolved | Representative Subcategory | Subcategory Rate |
|---------------|---------|--------------------------|-----------------|
| Tool-Related Issues | 3/9 (33.33%) | Tool Dependency Issues | 2/3 (66.67%) |
| Memory-Related Issues | 1/8 (12.50%) | Memory Content Errors | 1/5 (20.00%) |
| LLM Operation Issues | 2/11 (18.18%) | Model Access Config / Prompt Issues | 1/2 each (50.00%) |
| Workflow Issues | 1/6 (16.67%) | — | — |
| General Utility Issues | 8/14 (**57.14%**) | General Dependency Issues | 2/2 (100.00%) |
| | | General Configuration Issues | 4/6 (66.67%) |

### Key Findings

- **SE agents primarily resolve "General Utility Issues"** (57.14% resolution rate), as these closely resemble conventional software issues (e.g., logging, file operations, UI configuration). Agent-specific categories (LLM provider compatibility, memory, workflow) exhibit extremely low resolution rates.
- **Claude-3.5-Sonnet generally outperforms GPT-4o**: Claude-paired SE agents achieve higher correct resolution rates, plausible resolution rates, and localization accuracy. AutoCodeRover + Claude-3.5-S achieves the highest correct resolution rate (12.67%) and file-level localization accuracy (25.61%).
- **SE agents exhibit complementary capabilities**: each SE agent uniquely resolves 2–4 issues that no other agent resolves, yet no single issue is resolved by all SE agents simultaneously, suggesting that combining multiple SE agents may yield better aggregate performance.
- **Agent issue patches are substantially larger than conventional software patches**: on average, 66 lines of code, 3.58 files, and 6.79 functions are modified; the largest patch spans 355 lines, 34 files, and 54 functions—far exceeding the typical scale of SWE-bench issue patches.
- **Dependency issues are relatively easier to resolve**: tool dependency and general dependency issues achieve the highest resolution rates (66.67%–100%), as they typically manifest as clear error messages (e.g., missing libraries, version conflicts) that facilitate localization even when agent-specific components are involved.
- **LLM operation issues are nearly intractable**: despite constituting the largest category (31.84%), LLM operation issues have extremely low resolution rates. SE agents lack up-to-date knowledge of LLM API characteristics (e.g., which models support the `stop` parameter, how to handle Gemini content blocking), and the complex non-deterministic LLM interactions in agent workflows impede root cause localization. The appendix presents two illustrative unresolved cases: one involving CrewAI passing a `stop` parameter to `o1-preview`/`o1-mini` models that do not support it—where the SE agent not only fails to fix the issue but exacerbates it—and another involving Aider generating multiple conflicting diffs for a single file, where the SE agent merely prints an error message rather than identifying the root cause (the correct fix is to retain only the first diff).
- **LLM provider compatibility issues are completely unresolvable**: these issues require SE agents to be aware of the latest changes to LLM provider APIs and parameter differences across models—knowledge typically absent from SE agent training data.
- **Statistical significance**: the 2-sigma error range across three experimental repetitions is ±2.31% to ±6.67%, confirming that agent issue resolution rates remain far below conventional software rates even accounting for random variation.
- **Costs are modest but efficiency is poor**: applying SE agents to agent issue resolution costs $0.05–$1.15 per issue on average, comparable to conventional software issue resolution ($0.45–$2.53), yet with substantially lower success rates, yielding very poor cost-effectiveness.

## Highlights & Insights

- **First Agent Issue Taxonomy**: Constructed via rigorous grounded theory methodology with a Cohen's Kappa of 0.849, the 6-category 20-subcategory taxonomy serves as an important reference framework for future research on agent system quality. By decomposing agent system complexity into quantifiable dimensions, it enables researchers to improve SE agents in targeted categories. The appendix provides detailed real-world examples from open-source projects for each subcategory, including issue links, descriptions, and fix strategies—highly valuable as practical reference material.
- **Exceptionally rigorous reproducibility design**: 500 person-hours of investment, three rounds of non-flakiness validation, and Docker containerization make AgentIssue-Bench the only truly reproducible benchmark in the agent issue resolution domain. All Docker images are hosted on Docker Hub for one-click retrieval and evaluation, substantially lowering the barrier for future researchers.
- **The large performance gap reveals a fundamental qualitative difference**: 50.80% on conventional software vs. 12.67% on agent systems is not a quantitative gap but a qualitative one. It demonstrates that the core capabilities of current SE agents—code search, bug localization, patch generation—are designed for deterministic conventional software and are fundamentally insufficient for agent systems involving LLM interaction, tool invocation, and memory management.
- **The finding that "general utility issues are fixable while agent-specific issues are not" provides a clear research direction**: future work needs to inject agent-system domain knowledge into SE agents, including up-to-date LLM provider API documentation, agent framework architectural patterns, and workflow debugging methodologies.
- **The complementarity analysis is insightful**: Venn diagram analysis reveals that the three SE agents' resolution capabilities are highly complementary (each uniquely resolving 2–4 issues), suggesting that agent ensemble or multi-strategy fusion may be a simple and effective approach for improving agent issue resolution rates.

## Limitations & Future Work

- **Limited benchmark scale**: Although the 50 issues are high-quality and reproducible, the quantity is limited and may not cover edge cases across all agent issue types. The authors acknowledge this and plan continued expansion.
- **Agent system sampling bias**: The 16 agent systems are predominantly open-source, Python-based projects and may not be representative of commercial or multilingual agent system issue characteristics.
- **Limited SE agent coverage**: Only SWE-agent, AutoCodeRover, and Agentless are evaluated; newer tools such as Devin, OpenDevin, and Moatless are not included. Given the rapid iteration of SE agents, evaluation conclusions may require updating.
- **No proposed solutions**: The paper primarily identifies the problem—documenting the difficulty of agent issue resolution without proposing technical improvements. Future work could: (i) equip SE agents with real-time retrieval of LLM provider API documentation; (ii) include agent system code and issue resolution examples in SE agent training data; (iii) develop interactive tooling specifically for agent workflow debugging.
- **The fundamental challenge of non-determinism**: Even with a benchmark in place, the inherently non-deterministic nature of LLMs in agent systems means that some issues (e.g., workflow loops and hangs) may be theoretically irresolvable at the patch level, potentially requiring architectural-level solutions rather than code fixes.
- **Limitations of evaluation metrics**: Semantic equivalence judgments rely on human annotation and are inherently subjective. For complex agent system repairs involving prompt adjustments or error-handling strategy changes, multiple reasonable "correct" fixes may exist.
- **Single-shot repair paradigm not examined**: The current evaluation framework requires SE agents to generate complete patches in a single attempt, whereas real-world agent issue debugging typically involves multiple interactive rounds—running the agent, observing outputs, localizing problems, and iterating on fixes. Adopting an interactive repair paradigm could substantially improve resolution rates.
- **Detailed statistics for the 16 agent systems**: The analysis covers diverse agent systems including MetaGPT (55.4k stars, 90.7k LoC), AutoGen (44.2k stars, 198k LoC), and CrewAI (31.3k stars, 171k LoC), with code sizes ranging from BabyAGI's 8.8k LoC to CAMEL's 206k LoC, spanning issues from March 2023 to July 2024, under licenses including MIT, Apache-2.0, and CC-BY-4.0.

## Related Work & Insights

- **vs. SWE-bench family**: SWE-bench, SWE-bench Lite, and SWE-bench Verified all target conventional Python software; SWE-bench Java and SWE-bench Multimodal extend to Java and frontend JavaScript; SWE-Lancer Diamond covers commercial Expensify software—but none addresses agent systems. This paper fills that gap and demonstrates that agent system maintenance warrants independent investigation. Notably, the benchmark quality standards employed here (three-step filtering plus non-flakiness validation) are more stringent than those of many conventional software benchmarks.
- **vs. Cemri et al. (2025)**: That work also examines failure patterns in multi-agent systems but focuses on runtime failure symptom analysis through failed trajectories, whereas this paper targets issue resolution—encompassing not only bug fixes but also feature requests. The two perspectives are complementary; this paper additionally provides a reproducible benchmark and SE agent evaluation.
- **vs. Shao et al. (2025)**: That work studies integration bugs in LLM-integrated systems but covers a broader scope (all systems employing LLMs), whereas this paper focuses specifically on LLM-based agent systems, enabling deeper analysis and providing a reproducible benchmark.
- **Insights**: The central implication of this paper is that agent systems have become a distinct software category whose maintenance and debugging require entirely new tools and methodologies. The poor transfer of conventional SE tools to agent systems is closely tied to their complex component architectures, LLM interaction non-determinism, and rapidly evolving external dependencies. Future work may require "agents for agents"—SE agents designed specifically for maintaining agent systems. Furthermore, the finding that LLM operation issues constitute 31.84% of all issues yet achieve near-zero resolution rates carries important implications for agent framework designers: better LLM interaction abstractions, more comprehensive exception handling mechanisms, and enhanced debuggability should be prioritized at the architectural level.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic study of agent system issue taxonomy and automated resolution, filling an important gap; however, the research paradigm (taxonomy + benchmark + evaluation) is well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 16 agent systems, 201 issues, 50 reproducible tasks, 3 SE agents × 2 LLMs × 3 repetitions, representing an enormous effort; however, only 3 SE agents are included and the benchmark scale is modest.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-motivated, taxonomy richly illustrated with examples; appendix provides complete per-subcategory case studies.
- **Value**: ⭐⭐⭐⭐⭐ Exposes an important and pressing challenge in agent system maintenance, provides the first reproducible benchmark alongside detailed diagnostic analysis, and offers high reference value to both the SE and agent communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](../../ICLR2026/robotics/rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)

</div>

<!-- RELATED:END -->
