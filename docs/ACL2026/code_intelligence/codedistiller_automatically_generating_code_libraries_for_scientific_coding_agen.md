---
title: >-
  [Paper Note] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents
description: >-
  [ACL2026][Code Intelligence][Automatic scientific discovery] CodeDistiller automatically distills scientific GitHub repositories into runnable and debugged code example libraries…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Automatic scientific discovery"
  - "Code agents"
  - "Code-RAG"
  - "Repository distillation"
  - "LLM-as-a-judge"
date: 2026-05-08
content_hash: 89bf8f1fd6102d84
---

# CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents

**Conference**: ACL2026  
**arXiv**: [2512.01089](https://arxiv.org/abs/2512.01089)  
**Code**: https://github.com/cognitiveailab/codedistiller  
**Area**: AI for Science / Code Agents  
**Keywords**: Automatic scientific discovery, Code agents, Code-RAG, Repository distillation, LLM-as-a-judge

## TL;DR
CodeDistiller automatically distills scientific GitHub repositories into runnable and debugged code example libraries, enabling Code-RAG scientific discovery agents to invoke real domain tools. On 250 materials science repositories, the best model achieved a human-verified functional correctness rate of 74.1%, and downstream discovery tasks were more preferred by experts.

## Background & Motivation
**Background**: Automatic scientific discovery systems are evolving from literature discovery and data-driven discovery toward experiment-driven discovery. Many new systems receive research tasks, automatically generate code, run computational experiments, debug errors, and finally write experimental reports. In fields like materials science, computational chemistry, or machine learning systems research, the ability to write correct experimental code directly determines discovery quality.

**Limitations of Prior Work**: Scientific experimental code often relies on highly specialized libraries, data formats, and operational workflows. When agents rely solely on parametric knowledge, they tend to generate unrunnable or scientifically non-compliant code based on "impressions." Relying on manually curated example libraries is costly and slow to scale. Existing agent benchmarks mostly focus on reproducing a few repositories or writing code based on papers, which cannot directly build large-scale reusable examples for Code-RAG scientific discovery systems.

**Key Challenge**: Scientific agents require a large number of real, runnable, domain-specific code examples to enhance their capabilities. However, if these examples depend on manual maintenance by experts, they cannot cover the rapidly growing open-source scientific software ecosystem. If fully automated, it is difficult to ensure that the code is runnable and accurately demonstrates the core functions of the repository.

**Goal**: This work aims to construct an automated pipeline to transform a large number of scientific GitHub repositories into vetted code examples that can be "retrieved and combined by downstream discovery agents," while quantifying the trade-offs of different base models in terms of cost, runtime, and correctness.

**Key Insight**: Instead of having agents write experiments directly from parametric knowledge, CodeDistiller first scans repositories offline, identifies key files, generates and debugs minimal working examples, and then provides these examples as a Code-RAG library for downstream tasks.

**Core Idea**: Using static file filtering combined with dynamic code generation/execution/reflexive debugging to automatically distill open-source scientific repositories into executable example libraries, thereby supplementing the domain tool knowledge of scientific discovery agents.

## Method

### Overall Architecture
The input to CodeDistiller is a batch of domain-related GitHub repositories, and the output is the runnable example code and metadata corresponding to each repository. The process begins with large-scale static information collection: determining the file type, purpose, relevance, and special execution requirements for each file to identify the code, documentation, scripts, or existing examples most likely to assist in building the example. Then, it enters the dynamic example generation phase: high-relevance files and the repository's core purpose are provided to the code generation system to produce Python code, dependencies, runscripts, and resource descriptions. The generated results are executed in Ubuntu cloud containers. If the LLM-as-a-judge determines a failure, the system feeds the execution logs back to the model for reflection and repair until success or the debugging limit is reached.

### Key Designs
1. **Repository File Classification and Relevance Filtering**:
    - **Function**: Select files from repositories that may contain hundreds of nested files which are most helpful for example generation.
    - **Mechanism**: Each file is sent to a prompt and classified as code, documentation, scripts, data, or other; code, documentation, and scripts are further subdivided into existing examples, instructions, entry points, etc. Simultaneously, the system assigns a relevance score of 1-5 to the files and records metadata such as GPU requirements, configuration instructions, and key task information.
    - **Design Motivation**: Code generation models have limited context; directly feeding the entire repository is expensive and noisy. Filtering files first focuses attention on APIs, documentation, and existing examples.

2. **Runnable Example Generation**:
    - **Function**: Transform core repository functions into minimal experimental examples reusable by downstream agents.
    - **Mechanism**: A modified version of CodeScientist receives high-relevance files and repository purposes to generate four types of output: executable Python code, Python dependencies, a bash runscript with Conda environment setup, and metadata (including purpose description, applicable/inapplicable scenarios, CPU/GPU/RAM/disk requirements, and whether user interaction is needed).
    - **Design Motivation**: Downstream Code-RAG requires specific code snippets that can be retrieved, understood, run, and combined, rather than just repository summaries. Metadata helps the agent decide when to invoke the example.

3. **Execution-Judge-Reflect Debugging Loop**:
    - **Function**: Convert "reasonable-looking" code into "actually executed" examples.
    - **Mechanism**: Examples are executed in Ubuntu cloud containers, capturing stdout/stderr, timestamped logs, JSON results, and human-readable outputs like charts. After execution, an LLM-as-a-judge determines if the repository functions are correctly demonstrated. If issues exist, the current code and execution logs are fed back to the model for reflection and modification. The loop typically caps at 8 iterations to control cost.
    - **Design Motivation**: The correctness of scientific code cannot be judged by static text alone. Real execution and log-driven repair are key to distinguishing toy code from usable code libraries.

### Loss & Training
CodeDistiller does not train a model but compares the performance of various base models as agent base models. Experiments use GPT-OSS-120B, GPT-5, and Claude Sonnet 4.5. Cheaper models from the same family, such as GPT-5-mini or Claude Haiku 4.5, can be used for the file classification phase. Evaluation includes automatic LLM-as-a-judge, manual inspection by materials science experts, runtime, debugging rounds, and API costs. Downstream evaluation integrates CodeDistiller-generated examples into CodeScientist and performs an A/B comparison against a baseline using only general materials science code examples.

## Key Experimental Results

### Main Results
Materials science experts listed 30 common Python materials libraries (e.g., PyMatgen, ASE, LAMMPS, PyCalphad). The authors used the GitHub API to find 3,802 repositories importing these libraries with permissive licenses, then randomly sampled 250 for evaluation.

| Agent base model | Auto Success Rate | Manually Error-free | Manually Shows Function | Manually Correct Function | Avg Runtime (Success) | Avg Cost (Success) |
|------------------|-----------:|-------------:|----------------:|-------------:|---------------:|-------------:|
| GPT-OSS-120B | 61.6% | 29.6% | 29.6% | 25.9% | 13.8 min | $0.09 |
| GPT-5 | 70.4% | 69.0% | 69.0% | 60.5% | 20.3 min | $0.70 |
| Claude Sonnet 4.5 | 75.6% | 75.6% | 75.6% | 74.1% | 19.0 min | $1.71 |

### Downstream A/B Study
| Dimension | Key Data | Description |
|------|----------|------|
| Task Construction | 12 Materials Sci repos, 5 questions each, 60 discovery problems total | Analyzed 50 problems where both baseline and enhanced systems produced solutions |
| Running Budget | Max 15 debug rounds, 6h total runtime, 60 min/round, LLM cost cap $5 | Claude Sonnet 4.5 used as the CodeScientist base model |
| Expert Preference | CodeDistiller system generally preferred by over half in accuracy, completeness, soundness | Baseline preferred in only ~18%-24% of cases; ~1/4 were ties |
| Reviewer Consistency | Cohen's $\kappa$: Accuracy 0.77, Soundness 0.70, Completeness 0.62 | Moderate to strong agreement between LLM-as-a-judge and experts on A/B tasks |

### Case Results
| Scenario | Baseline | CodeDistiller Enhanced System | Description |
|------|----------|------------------------|------|
| Tox21 Toxicity Pred. | Synthetic data replicated from 20 hand-picked molecules | Used 6,258 real compounds, 12 toxicity assays | Enhanced system is more scientifically valid |
| Ge/Sb/Te Relaxation | General Lennard-Jones potential (not elemental), 80%-93% collapse | Used CHGNet, volume change -16% to +75% | Enhanced system aligns better with materials physics |
| Alloy Parameter Calc. | Manual DB led to 3.60% atomic size difference in AlTiVNb | Used pymatgen and Parameter-Calculator-for-CCA, got 5.428% | Mature libraries are more reliable than ad-hoc implementations |

### Key Findings
- Automated judges overestimate example quality, especially with GPT-OSS-120B where the auto-success rate (61.6%) deviates significantly from manual functional correctness (25.9%).
- Claude Sonnet 4.5 performs best but at an average success cost of $1.71, approximately 19x that of GPT-OSS-120B; a clear cost-quality trade-off exists.
- Successful cases usually require about 2 debugging rounds; unsuccessful ones iterate until reaching the cap, making failure costs significant.
- Downstream A/B testing indicates these examples are useful not just for repository reproduction but also for improving the accuracy, completeness, and scientific soundness of automated discovery reports.

## Highlights & Insights
- **Transforming GitHub Repositories into Code-RAG Assets**: The focus is not on solving a single repo once, but on building a retrievable, composable code library for scientific agents, which is more aligned with long-term system construction.
- **Human Expert Evaluation is Crucial**: The gap between auto-judges and human results serves as a reminder that "runnable" scientific code is not necessarily "scientifically correct"; domain validation is essential.
- **Offline Distillation Reduces Online Discovery Complexity**: Downstream agents already possess domain examples before executing specific research tasks, avoiding the need for zero-shot learning of every library during online inference.
- **Engineering Value of Cost Data**: Reporting runtime, debugging cycles, and API costs alongside success rates makes the deployability of the method easier to judge.

## Limitations & Future Work
- Human expert evaluation remains a time-limited proxy: Experts check if code, results, and charts are reasonable, but do not write full test suites or replicate original literature for every repo.
- Repository identification methods introduce noise. Using library imports to find repositories led to an estimate that about half were not truly materials science focused but only incidentally imported related libraries.
- Currently only evaluated in materials science; performance may change in biology, chemistry, GIS, or robotics due to data openness, software dependencies, closed-source tools, and experimental standards.
- The comparison between purpose-built agents and general coding agents remains unresolved; varying budgets, models, tools, and output formats make perfectly fair control difficult.
- Future work could introduce stronger automated unit test generation, domain benchmark validation, license/security filtering, and version update mechanisms for generated example libraries.

## Related Work & Insights
- **vs CodeScientist**: CodeScientist depends on an existing vetted code library; CodeDistiller addresses how to automatically expand that library.
- **vs AI Scientist / AgentLab**: These focus on executing research tasks or generating experimental code; CodeDistiller acts as front-end infrastructure, preparing tool examples for subsequent agents.
- **vs SUPER / GISTIFY / RexBench**: These benchmarks test agent setup, reproduction, or single-repo example generation; CodeDistiller evaluates across a larger scale of materials repositories and focuses on downstream discovery gains.
- **Insight**: To build domain research assistants, one can first distill the domain's GitHub ecosystem offline to form a "runnable tool memory bank," then let online agents retrieve and combine these examples rather than writing from scratch using parametric knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The goal of repo-to-runnable-library is practical; the pipeline combination is clear; individual agent components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Features 250 repositories, human expert evaluation, and downstream A/B; however, domain is limited to materials science, and human verification is still a proxy.
- Writing Quality: ⭐⭐⭐⭐☆ Direct narrative, clear on costs and failure modes; some "Table 1/2" naming confusion requires reader attention.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for AI for Science agents and Code-RAG infrastructure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[ACL 2026\] RExBench: Can coding agents autonomously implement AI research extensions?](rexbench_can_coding_agents_autonomously_implement_ai_research_extensions.md)
- [\[ICML 2026\] NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](../../ICML2026/code_intelligence/nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)
- [\[ACL 2026\] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](../../ICLR2026/code_intelligence/paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)

</div>

<!-- RELATED:END -->
