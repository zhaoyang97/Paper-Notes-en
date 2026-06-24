---
title: >-
  [Paper Note] Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery
description: >-
  [ICML 2025 (Workshop on Machine Learning for Astrophysics)][LLM Agent][Multi-Agent Systems] This paper proposes cmbagent, a multi-agent system composed of approximately 30 LLM agents. It adopts a Planning & Control strategy to orchestrate fully autonomous scientific research workflows. Individual agents are responsible for specialized tasks such as literature retrieval, code generation, result interpretation, and output review, with the capability of executing code locally. T…
tags:
  - "ICML 2025 (Workshop on Machine Learning for Astrophysics)"
  - "LLM Agent"
  - "Multi-Agent Systems"
  - "Scientific Discovery Automation"
  - "Planning & Control"
  - "Cosmology"
  - "Autonomous Scientific Discovery"
date: 2026-05-08
content_hash: 6436e2615b418312
---

# Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery

**Conference**: ICML 2025 (Workshop on Machine Learning for Astrophysics)  
**arXiv**: [2507.07257](https://arxiv.org/abs/2507.07257)  
**Code**: [https://github.com/CMBAgents/cmbagent](https://github.com/CMBAgents/cmbagent) (Open Source)  
**Area**: LLM Agent  
**Keywords**: Multi-Agent Systems, Scientific Discovery Automation, Planning & Control, Cosmology, LLM Agent, Autonomous Scientific Discovery

## TL;DR

This paper proposes cmbagent, a multi-agent system composed of approximately 30 LLM agents. It adopts a Planning & Control strategy to orchestrate fully autonomous scientific research workflows. Individual agents are responsible for specialized tasks such as literature retrieval, code generation, result interpretation, and output review, with the capability of executing code locally. The system successfully completes a PhD-level cosmology task (measuring cosmological parameters using supernova data) and outperforms state-of-the-art LLMs on two benchmark datasets.

## Background & Motivation

**Background**: Autonomous scientific discovery (AI for Science) is a frontier of artificial intelligence. The scientific research process encompasses literature review, hypothesis formulation, experimental design, code implementation, data analysis, and result interpretation, which historically relied entirely on human researchers. Recently, Large Language Models (LLMs) have demonstrated strong capabilities in code generation, text understanding, and reasoning, opening up new possibilities for automating scientific workflows.

**Limitations of Prior Work**: Existing LLM-based scientific assistants typically cover only a single stage of the research workflow—such as literature retrieval alone (e.g., Semantic Scholar API wrappers), code generation alone (e.g., Copilot), or data analysis alone (e.g., Code Interpreter). There is a lack of an integrated system capable of end-to-end orchestration of the entire scientific workflow. Furthermore, a single LLM agent faces challenges like distracted attention, insufficient context windows, and limited specialized capabilities when handling complex, multi-step scientific tasks.

**Key Challenge**: Scientific research tasks require diverse and heterogeneous capabilities, such as literature comprehension, mathematical derivation, code implementation, experimental design, and critical evaluation of results. It is highly difficult for a single agent to master all these domains simultaneously. Meanwhile, many stages of the research workflow have complex dependencies (e.g., understanding the theoretical background is required before correctly designing an experiment), which demands meticulous planning and workflow control.

**Key Insight**: Inspired by multi-agent frameworks in the industry (e.g., AutoGen, MetaGPT) and the classical cybernetics concept of Planning & Control, the authors design a system composed of approximately 30 specialized agents. By coordinating their operations through planning and control strategies, the system automates the entire process from literature retrieval to final results, focusing specifically on scientific research automation in the astrophysics and cosmology domains.

**Core Idea**: Construct a fully autonomous, end-to-end scientific research automation system using ~30 specialized LLM agents coordinated by a Planning & Control orchestration strategy.

## Method

### Overall Architecture

cmbagent is a multi-agent system whose core architecture consists of the following layers:

1. **Planning Layer**: Upon receiving a high-level scientific task description, the planning module decomposes it into an ordered sequence of subtasks, determining the dependencies and execution order among them.
2. **Control Layer**: The control module oversees the real-time execution status of each agent, dynamically adjusting subsequent steps based on intermediate results. If an agent's output is unsatisfactory, the control layer triggers retries or switches to alternative strategies.
3. **Agent Pool**: A repository of around 30 LLM agents, each focusing on a specific task type and equipped with specialized capabilities through carefully designed system prompts.
4. **Code Execution Environment**: The system integrates local code execution capabilities, allowing code written by agents to run directly on the local machine to retrieve actual computational results.

The entire system runs fully autonomously, **requiring no human intervention** (no human-in-the-loop), from task reception to final result output.

### Key Designs

#### 1. Specialized Agent Division of Labor

The system features ~30 agents covering various stages of the scientific workflow:

| Agent Type | Responsibility Description | Core Capability |
|-----------|---------|---------|
| Literature Retrieval Agent | Retrieves relevant scientific papers and code repositories | RAG, semantic search, key information extraction |
| Code Formulation Agent | Writes implementation code based on research requirements | Code generation, API invocation, data processing scripts |
| Result Interpretation Agent | Analyzes computational output to extract scientific meaning | Numerical result interpretation, statistical analysis, visualization |
| Output Review Agent | Audits the quality of other agents' outputs | Consistency check, scientific rationality evaluation, error detection |
| Mathematical Derivation Agent | Handles theoretical equations and mathematical derivations | Symbolic calculation, equation verification, theoretical derivation |
| Experimental Design Agent | Plans experimental steps and parameter settings | Experimental protocol design, parameter space definition |

Each agent acquires domain-specific expertise via targeted prompt engineering, shifting the processing bottleneck away from a single-agent architecture.

#### 2. Planning & Control Strategies

- **Planning**: Employs a hierarchical task decomposition strategy to recursively break down complex scientific tasks (e.g., "measuring cosmological parameters using supernova data") into atomic subtasks that can be completed by a single agent. The planning phase establishes the execution topology—identifying which subtasks can be executed in parallel and which have sequential dependencies.
- **Control**: Monitors the output quality and task progress of each agent in real-time during execution. Control strategies include: (a) **State tracking**—recording the completion state of each subtask and its intermediate artifacts; (b) **Quality gating**—utilizing review agents to evaluate key outputs, providing feedback for revision if they do not meet standards; (c) **Dynamic adjustment**—modifying subsequent plans based on intermediate results (e.g., inserting additional data cleaning steps if initial analysis reveals data quality issues).

#### 3. Local Code Execution

Unlike purely conversational LLMs, cmbagent can execute code locally. This means that agents can not only "discuss" how to analyze data but can also actually run Python/Julia scripts and invoke scientific computing packages (such as NumPy, SciPy, Astropy, and other cosmological tools) to retrieve real numerical results. Local execution eliminates the "risk of hallucination" because computational results originate from actual execution rather than LLM generation.

#### 4. Cross-Agent Review Mechanism

The system sets up specialized Critic Agents to audit the output quality of other agents. This "adversarial verification" mechanism effectively reduces the error rate in individual agent outputs, resembling the peer-review process in scientific research.

## Key Experimental Results

### Main Experiment: PhD-Level Cosmology Task

cmbagent was applied to a PhD-level cosmology task: measuring cosmological parameters (such as the Hubble constant $H_0$ and the matter density parameter $\Omega_m$) using Type Ia supernova data. This is a classic problem in cosmological research, which typically requires a graduate student to have a solid theoretical background and strong data analysis skills to complete.

| Evaluation Dimension | cmbagent Performance | Baselines |
|---------|-------------|---------|
| Task Completion | Successfully completed end-to-end cosmological parameter measurement | SOTA LLMs fail to complete the entire pipeline independently |
| Degree of Automation | Fully autonomous throughout the process (no human-in-the-loop) | Existing tools usually require human intervention in multiple stages |
| Code Execution | Executes scientific computing code locally and retrieves actual results | Purely LLM-based methods rely on text generation and cannot verify results |
| Scientific Soundness | Cross-agent review ensures output quality | Single-agent methods lack self-correction mechanisms |

### Benchmark Dataset Evaluation

The system was evaluated against state-of-the-art LLMs on two benchmark datasets:

| Comparison Dimension | cmbagent (Multi-Agent) | SOTA Single LLM | Advantage Analysis |
|---------|-------------------|-------------|---------|
| Overall Performance | Outperforms baseline methods | Baseline level | Systemic advantages of combining Planning & Control with specialized agent division of labor |
| Task Coverage | End-to-end scientific research process | Typically covers only partial stages | ~30 agents cover the entire workflow through division of labor |
| Reliability | Cross-verification by review agents | Single generation without verification | Multi-agent review ≈ peer-review mechanism |
| Interpretability | Each agent's output can be independently audited | End-to-end black box | Modular architecture enhances transparency |
| Deployment Flexibility | GitHub + HuggingFace + Cloud | Usually API calls only | Open source + multi-platform deployment |

### Key Findings

- **Multi-agent systems significantly outperform a single LLM on complex scientific tasks**: The collaboration of ~30 specialized agents enables the system to handle PhD-level research tasks that single LLMs cannot complete independently.
- **Planning & Control is key to success**: Reasonable task decomposition and real-time execution control ensure the smooth advancement of complex, multi-step workflows, preventing the accumulation of errors in intermediate stages.
- **Code execution eliminates hallucination**: Retrieving real computational results via local code execution, rather than relying on LLM-generated values, fundamentally addresses the hallucination issue of LLMs in scientific computations.
- **Review mechanisms enhance reliability**: The auditing mechanism where dedicated Critic Agents review outputs from other agents resembles scientific peer review and effectively improves the scientific quality of the final output.

## Highlights & Insights

- **Scaling agents to ~30 is a meaningful design choice**: Compared to smaller systems with 3–5 agents, cmbagent achieves deeper specialization through granular division of labor. Each agent only needs to excel in a narrow domain, reducing the demands on the general capability of the underlying LLM. This points to a trend: future agent systems may evolve to be "more numerous and more specialized."
- **Scientific agent systems require code execution**: Unlike general dialogue tasks, scientific research is centered on numerical computation and data analysis; pure text interactions are far from sufficient. cmbagent's local code execution capability is a necessary condition for completing PhD-level tasks.
- **Planning & Control draws inspiration from cybernetics**: Distinct from simple chains or graphs, the P&C strategy introduces feedback loops—the control layer dynamically adjusts the planning based on execution results. This closed-loop design is better suited for highly uncertain research tasks.
- **Open-source + multi-platform deployment lowers access barriers**: By providing services simultaneously on GitHub (code), HuggingFace (models/spaces), and the cloud, the system covers different user needs, from developers to end-users.

## Limitations & Future Work

- **High domain-specificity**: Currently, the system primarily focuses on cosmological and astrophysical tasks, and many of its agents and retrieval resources are tailored for these domains. Porting the system to other scientific fields (such as biology or chemistry) requires redesigning specialized agent prompts and knowledge bases.
- **Complexity in agent orchestration**: A system with approximately 30 agents brings significant orchestration complexity. Overhead in agent communication, state synchronization, and error propagation could become bottlenecks in practical deployment. The paper does not thoroughly discuss the scalability and resource consumption of the system.
- **Limited evaluation depth as a workshop paper**: Being an ICML Workshop paper, the evaluation scale and level of detail are not as extensive as main conference papers, lacking systematic ablation experiments on various components and comprehensive comparisons across different LLM backbones.
- **Risks of zero human intervention**: A fully autonomous workflow lacks the critical judgment of human experts (e.g., evaluating the rationality of experimental hypotheses or making physical intuition checks on results). In open-ended research, it may produce scientifically incorrect though plausible-looking conclusions.
- **Cost efficiency**: The inference cost of running ~30 LLM agents is high, which may not be friendly to resource-constrained research teams. The paper lacks a detailed analysis of token consumption and execution time overheads.
- **Reproducibility**: Although the code is open-sourced, the system depends on specific LLM APIs and scientific computing environment configurations. Setting up the environment to fully replicate the PhD-level tasks may pose a high barrier for researchers.

## Related Work & Insights

- **vs AutoGen / MetaGPT**: These are general-purpose multi-agent frameworks, whereas cmbagent builds on them by heavily customizing for scientific automation (with ~30 domain-specific agents, scientific literature/code base retrieval, and local scientific computation execution). cmbagent is more specialized and deeper, though less generalizable.
- **vs ChemCrow / Coscientist**: These are also agent systems aimed at scientific discovery, but they focus on chemistry and experimental science, respectively. The unique feature of cmbagent is its orientation toward computationally intensive cosmology research, emphasizing code execution and numerical computation.
- **vs SciAgent / AI Scientist**: Similar scientific automation systems, but cmbagent utilizes more agents (~30 vs. typically 3–5), features finer division of labor, and employs Planning & Control rather than simple chained execution.
- **Insights**: The architecture of cmbagent, combining Planning & Control with a large pool of specialized agents, provides a reference paradigm for building scientific automation systems in other domains. The key takeaway is: scientific automation requires not only language capabilities (LLM) but also execution capabilities (running code) and quality control mechanisms (cross-agent review).

## Rating

- Novelty: ⭐⭐⭐⭐ A large-scale scientific automation system at the scale of ~30 agents is quite novel, and the introduction of the Planning & Control strategy offers a new perspective on agent orchestration.
- Experimental Thoroughness: ⭐⭐⭐ As a workshop paper, space is limited; succeeding at a PhD-level cosmology task is impressive, but systematic ablation and detailed numerical comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ The system architecture and application scenarios are clearly described, and the open-source deployment information is complete.
- Value: ⭐⭐⭐⭐ Demonstrates the viability of large-scale agent systems on real scientific research tasks, with its open-source code and multi-platform deployment enhancing actual impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)
- [\[ICLR 2026\] Towards Multimodal Data-Driven Scientific Discovery Powered by LLM Agents](../../ICLR2026/llm_agent/towards_multimodal_data-driven_scientific_discovery_powered_by_llm_agents.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ICLR 2026\] ChinaTravel: An Open-Ended Travel Planning Benchmark with Compositional Constraint Validation for Language Agents](../../ICLR2026/llm_agent/chinatravel_an_open-ended_travel_planning_benchmark_with_compositional_constrain.md)
- [\[ICLR 2026\] ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows](../../ICLR2026/llm_agent/scienceboard_evaluating_multimodal_autonomous_agents_in_realistic_scientific_wor.md)

</div>

<!-- RELATED:END -->
