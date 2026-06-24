---
title: >-
  [Paper Note] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML
description: >-
  [ICML 2025][Multi-Agent][AutoML] This paper proposes AutoML-Agent, a multi-agent LLM collaborative framework for full-pipeline AutoML. It expands the search space using a Retrieval-Augmented Planning strategy, decomposes tasks into parallel subtasks handled by specialized agents, and introduces a multi-stage verification mechanism to guarantee code generation quality, achieving higher automation success rates and model performance across 14 datasets in 7 task categories.
tags:
  - "ICML 2025"
  - "Multi-Agent"
  - "AutoML"
  - "Multi-Agent Framework"
  - "LLM Agent"
  - "Retrieval-Augmented Planning"
  - "Full-Pipeline Automation"
date: 2026-05-08
content_hash: 5a02984bf74e8bae
---

# AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML

**Conference**: ICML 2025  
**arXiv**: [2410.02958](https://arxiv.org/abs/2410.02958)  
**Code**: [https://deepauto-ai.github.io/automl-agent](https://deepauto-ai.github.io/automl-agent)  
**Area**: Agent  
**Keywords**: AutoML, Multi-Agent Framework, LLM Agent, Retrieval-Augmented Planning, Full-Pipeline Automation

## TL;DR

This paper proposes AutoML-Agent, a multi-agent LLM collaborative framework for full-pipeline AutoML. It expands the search space using a Retrieval-Augmented Planning strategy, decomposes tasks into parallel subtasks handled by specialized agents, and introduces a multi-stage verification mechanism to guarantee code generation quality, achieving higher automation success rates and model performance across 14 datasets in 7 task categories.

## Background & Motivation

**Background**: Automated Machine Learning (AutoML) aims to automate key steps in the AI development pipeline, such as optimal model search and hyperparameter tuning. Although traditional AutoML tools (e.g., Auto-sklearn, AutoGluon, FLAML) perform exceptionally well in specific modules, they typically require users to possess deep technical expertise to configure complex toolchains and parameter spaces, a process that is time-consuming and requires significant human effort.

**Limitations of Prior Work**: Recently, researchers have begun using natural language interfaces of Large Language Models (LLMs) to lower the barrier to entry for AutoML, enabling non-expert users to build data-driven solutions. However, existing LLM-based AutoML methods suffer from two key limitations: (a) **Narrow coverage**: they typically target only specific stages of the AI development pipeline (e.g., only model selection or only hyperparameter tuning) and cannot cover the entire end-to-end pipeline from data acquisition to model deployment; (b) **Insufficient utilization of LLM capabilities**: a *single* LLM agent struggles to handle heterogeneous tasks simultaneously, such as data preprocessing, model architecture design, and training strategy selection.

**Key Challenge**: Full-pipeline AutoML requires a system to possess diverse and heterogeneous capabilities (data engineering, model design, training optimization, deployment), whereas a single-agent architecture has a limited search space and a rigid planning strategy. It can only generate a single fixed plan, lacking sufficient exploration of the solution space and failing to parallelize mutually independent subtasks.

**Goal**: (a) How can an LLM framework cover the complete AutoML pipeline from data acquisition to model deployment? (b) How can exploration capabilities during the planning phase be enhanced to search for better solutions? (c) How can multi-agent parallelization be leveraged to accelerate subtask solving? (d) How can the correctness and deployability of the generated code be guaranteed?

**Key Insight**: The authors observe that subtasks within the AutoML pipeline possess natural decomposability (e.g., data preprocessing and model design are relatively independent), allowing the design of specialized agents to execute them in parallel. Meanwhile, borrowing the concept of RAG (Retrieval-Augmented Generation), the planning space can be expanded by retrieving relevant historically successful solutions.

**Core Idea**: Employing multi-agent collaboration to divide and conquer full-pipeline AutoML, leveraging retrieval-augmented planning to expand the search space, and using multi-stage verification to guarantee code quality.

## Method

### Overall Architecture

AutoML-Agent takes natural language task descriptions as input from users (e.g., "train a classifier on this CSV data"), processes them through the complete AutoML pipeline, and outputs a deployable model. The entire framework consists of three core phases:

1. **Planning Phase**: The Manager Agent analyzes user requirements and utilizes a retrieval-augmented strategy to generate multiple candidate execution plans, each decomposed into several subtasks.
2. **Execution Phase**: Each subtask is assigned to a corresponding specialized agent (e.g., Data Agent for data preprocessing, Model Agent for neural network design), which executes its respective subtask in parallel.
3. **Verification Phase**: A multi-stage verification mechanism is used to check the execution results of each agent. If errors are detected, the code-generation LLM is guided to correct the implementation, ensuring the executability and correctness of the final solution.

### Key Designs

1. **Retrieval-Augmented Planning**

    - **Function**: Generates multiple candidate solutions during the planning phase instead of a single plan, enhancing exploration of the solution space.
    - **Mechanism**: Unlike traditional AutoML methods where the LLM directly generates a single solution, AutoML-Agent maintains a **solution knowledge base** containing historically successful cases and common AutoML pipeline templates. During planning, the Manager Agent first retrieves successful solutions similar to the current task from the knowledge base as references, and then generates multiple diversified candidate plans based on these references. Each candidate plan contains a different combination of data preprocessing strategies, model architecture choices, and training configurations.
    - **Design Motivation**: Traditional planning relies entirely on the LLM's "intuition", making it prone to getting stuck in local optima (e.g., always selecting common ResNet architectures). By introducing retrieval augmentation, the system can refer to historical successful experiences from similar tasks to inspire more diverse solutions, while parallel evaluation of multiple plans ensures search coverage. This design is inspired by the success of RAG in QA tasks—the introduction of external knowledge significantly improves LLM decision quality.

2. **Specialized Agents & Parallel Subtask Execution**

    - **Function**: Decomposes each execution plan into multiple independent subtasks executed in parallel by specially designed agents.
    - **Mechanism**: The AutoML pipeline can naturally be decomposed into sub-stages such as data acquisition, data preprocessing, feature engineering, model architecture design, hyperparameter configuration, model training, evaluation, and deployment. AutoML-Agent designs specialized agents for each subtask category, where each agent is endowed with domain-expert capabilities via carefully crafted prompts. For example, the Data Agent is prompted as a data engineering expert skilled in handling missing values, feature transformations, and data augmentation; the Model Agent is prompted as a neural network design expert responsible for selecting and configuring network architectures. Since certain subtasks are mutually independent (such as data preprocessing and model architecture selection), they can be **executed in parallel**, greatly reducing end-to-end processing time.
    - **Design Motivation**: A single agent processing the entire pipeline faces two dilemmas: (a) excessively long prompts leading to distracted attention and forgetting; (b) low efficiency from serial execution. Subtask decomposition addresses the first issue (each agent only focuses on its own domain), and parallel execution addresses the second. This divide-and-conquer strategy makes the system's search process much more efficient.

3. **Multi-Stage Verification**

    - **Function**: Checks the correctness of intermediate results and final outputs at multiple key stages of the execution process.
    - **Mechanism**: Verification is not performed as a single check only on the final output, but rather during every stage of subtask execution. Specifically, it includes: (a) **code syntax verification** to check whether agent-generated code runs without syntax errors; (b) **logical consistency verification** to check whether subtask outputs align with the global plan (e.g., whether the data format output by the Data Agent matches the input format expected by the Model Agent); (c) **performance verification** to evaluate model performance on a validation set and judge whether it meets expectations. If validation fails at any stage, the system feeds error messages back to the corresponding agent, guiding it to correct the code implementation.
    - **Design Motivation**: Code generated by LLMs often contains various subtle errors (e.g., API parameter mismatches, tensor dimension inconsistencies, data type conflicts), which are difficult to locate and fix with only a single final validation. Multi-stage verification enables **early detection and early recovery** (fail-fast), preventing errors from propagating through the pipeline and leading to a complete ultimate failure. This is also a critical guarantee for the high success rate of AutoML-Agent.

### Loss & Training

AutoML-Agent itself does not require training—it is a training-free, inference-time framework whose capabilities derive entirely from prompt engineering of pre-trained LLMs and multi-agent collaboration. The framework guides the behavior of various agents through meticulously designed system prompts and few-shot examples. The solution templates in the knowledge base can gradually expand through usage accumulation, achieving an implicit form of "continual learning."

## Key Experimental Results

### Main Results: Comparison of Success Rates and Model Performance across Tasks

AutoML-Agent was comprehensively evaluated on 7 downstream tasks (covering 14 datasets), spanning various AI application scenarios such as image classification, text classification, and tabular data analysis.

| Task Type | Number of Datasets | AutoML-Agent Success Rate | Best Success Rate of Baselines | Notes |
|----------|-----------|-------------------|------------------|------|
| Image Classification | Multiple | High | Low | Full-pipeline automation, no manual configuration required |
| Text Classification | Multiple | High | Medium | Automatically selects appropriate NLP pipeline |
| Tabular Data | Multiple | High | Medium | Automated feature engineering + model selection |
| Regression | Multiple | High | Low | Complete data preprocessing and hyperparameter tuning |
| Time-Series Forecasting | Multiple | High | Low | End-to-end automated processing |
| Multimodal | Multiple | Relatively High | Low | Fully automated processing of cross-modal data |
| Object Detection | Multiple | Relatively High | Low | Full coverage from data to model deployment |

AutoML-Agent achieves the highest automation success rate on the vast majority of tasks, and the generated models consistently exhibit good performance across all domains.

### Ablation Study: Contribution of Core Components

| Configuration | Success Rate Change | Notes |
|------|-----------|------|
| Full AutoML-Agent | Highest | Full model containing all components |
| w/o Retrieval-Augmented Planning | Significant Drop | Uses only a single plan, drastically shrinking the search space |
| w/o Multi-Agent Parallelism | Moderate Drop | Replaced by single-agent serial execution, reducing efficiency and accuracy |
| w/o Multi-Stage Verification | Significant Drop | Without verification, code errors cannot be repaired in time, dramatically reducing success rate |
| Single Plan + Single Agent | Lowest | Degenerates to traditional LLM-based AutoML, yielding the worst success rate |

### Key Findings

- **Multi-stage verification is the strongest guarantee for success**: The success rate drops most significantly when removing the verification mechanism, indicating that LLM-generated code indeed requires multi-round checks and iterative fixes to guarantee executability. This validates the effectiveness of the "early detection, early fix" strategy in code generation scenarios.
- **Retrieval-augmented planning provides diversified solutions**: Compared to a single-plan strategy, retrieval-augmented planning generates more diverse and rational candidate solutions by referencing historical success stories, identifying superior configuration combinations across different tasks.
- **Parallel execution balances efficiency and quality**: Multi-agent parallel execution not only accelerates the overall processing time but also enhances the completion quality of each stage because each agent focuses on a single subtask (compared to the issue of divided attention when a single agent handles all tasks simultaneously).
- **Strong cross-domain generalization capability**: AutoML-Agent maintains robust performance across 7 highly distinct categories of tasks, proving the generalization ability of the multi-agent framework to different task types.

## Highlights & Insights

- **Full-pipeline coverage is the core differentiator**: Compared to LLM-based AutoML methods that only focus on model selection or hyperparameter tuning, AutoML-Agent extends from data acquisition to model deployment, truly achieving end-to-end automation. This means a user with zero ML background can simply describe requirements in natural language and receive a deployable model—the ultimate vision of AutoML.
- **Ingenious application of RAG in planning**: Extending RAG from "retrieving knowledge for QA" to "retrieving historical solutions for planning" is a highly natural and effective transfer. This design concept can be generalized to any agent system requiring multi-solution search—such as code-generation agents retrieving historical code templates, or research agents retrieving related methodologies.
- **Efficiency gains from subtask decomposition and parallel execution**: By analyzing the dependency graph of the AutoML pipeline, parallelizable subtasks (such as data preprocessing and model architecture design) are identified and processed in parallel by specialized agents. This methodology of "dependency analysis $\rightarrow$ decomposition $\rightarrow$ parallelization" can be migrated to any complex, multi-stage task.
- **The "fail-fast" philosophy of multi-stage verification**: Instead of checking the result quality only at the very end, verification and timely debugging occur at every stage. This "quality gating" concept is widely applicable to all LLM code generation scenarios.

## Limitations & Future Work

- **Dependence on backbone LLM capabilities**: The framework's performance heavily depends on the code generation and reasoning capabilities of the underlying LLM. When a task requires highly specific domain knowledge (such as alpha factor mining in quantitative finance), general LLMs may fall short. Exploring the integration of specialized small models or domain-specific knowledge bases for different agents is a potential future direction.
- **Cold-start problem of the knowledge base**: Retrieval-augmented planning relies on the accumulation of historical successful solutions, which might not be fully effective during the early stages of system deployment due to knowledge base sparsity. Effective initialization strategies (e.g., extracting templates from public Kaggle winning solutions) need to be designed.
- **Handling cross-subtask coupling**: The paper assumes that subtasks are relatively independent and can be executed in parallel. However, in practice, data preprocessing (e.g., feature engineering) and model design can be tightly coupled (e.g., certain feature transformations are only effective for specific models). This coupling may prevent parallelized solutions from being optimal once combined.
- **Cost and latency**: Multi-agent + multi-planning + multi-stage verification implies a massive volume of LLM API calls, which can lead to high inference costs and time latency. A cost-benefit analysis is not fully discussed in the paper.
- **Scope of evaluation**: Although the framework was evaluated on 7 task types across 14 datasets, it lacks validation on production-level datasets (millions of samples, high-dimensional features) and systematic performance comparison against traditional non-LLM AutoML tools (such as AutoGluon, H2O).

## Related Work & Insights

- **vs Data-Interpreter / MLCopilot**: These methods typically cover only parts of the AutoML pipeline (e.g., Data-Interpreter only performs data analysis and visualization), whereas the strength of AutoML-Agent lies in full-pipeline coverage. However, the specialized depth of certain stages might not match that of focused tools.
- **vs AutoGen / MetaGPT**: These are general-purpose multi-agent frameworks that require users to manually orchestrate agent collaboration workflows. AutoML-Agent builds upon them with bespoke designs tailored to AutoML scenarios (such as retrieval-augmented planning and domain-specific agent role definitions), offering an out-of-the-box solution at the expense of generalizability.
- **vs CAAFE / AutoML-GPT**: Early LLM-for-AutoML attempts, which directly generated solutions using a single LLM without multi-agent collaboration or verification mechanisms. AutoML-Agent's multi-plan search and divide-and-conquer strategy yield higher success rates.
- **Insight**: The architectural design of AutoML-Agent (Manager Planning $\rightarrow$ Specialized Agent Execution $\rightarrow$ Verification Feedback) serves as a reusable multi-agent collaboration paradigm that can be migrated to similar full-pipeline automation scenarios, such as automated research (AutoResearch) and software development (AutoDev).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of full-pipeline AutoML and multi-agent collaboration is a natural yet meaningful innovation; the retrieval-augmented planning and multi-stage verification are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 14 datasets across 7 task types, but lacks in-depth comparison with traditional AutoML tools and detailed cost analysis.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the motivational derivation is coherent, though some details (such as knowledge base construction) are not thoroughly described.
- Value: ⭐⭐⭐⭐ Demonstrates the practical potential of LLM multi-agent frameworks in full-pipeline AutoML, significantly pushing forward the lowering of ML development barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](../../ICML2026/multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICLR 2026\] Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration](../../ICLR2026/multi_agent/graph-of-agents_a_graph-based_framework_for_multi-agent_llm_collaboration.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/multi_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[ICLR 2026\] MARTI: A Framework for Multi-Agent LLM Systems Reinforced Training and Inference](../../ICLR2026/multi_agent/marti_a_framework_for_multi-agent_llm_systems_reinforced_training_and_inference.md)
- [\[ICML 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)

</div>

<!-- RELATED:END -->
