---
title: >-
  [Paper Note] MasRouter: Learning to Route LLMs for Multi-Agent Systems
description: >-
  [ACL 2025][LLM (Other)][Multi-agent routing] This work defines the Multi-Agent System Routing (MASR) problem for the first time and proposes MasRouter, a cascade controller network. It sequentially determines the collaboration mode, role allocation, and LLM routing. While maintaining high performance, it reduces the inference cost of MAS by up to 52%, achieving an effective balance between performance and efficiency.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multi-agent routing"
  - "LLM routing"
  - "Collaboration mode"
  - "Cascade controller"
  - "Cost efficiency"
date: 2026-05-08
content_hash: f83ffb10f510b734
---

# MasRouter: Learning to Route LLMs for Multi-Agent Systems

**Conference**: ACL 2025  
**arXiv**: [2502.11133](https://arxiv.org/abs/2502.11133)  
**Code**: [https://github.com/yanweiyue/masrouter](https://github.com/yanweiyue/masrouter)  
**Area**: LLM/NLP  
**Keywords**: Multi-agent routing, LLM routing, Collaboration mode, Cascade controller, Cost efficiency

## TL;DR

This work defines the Multi-Agent System Routing (MASR) problem for the first time and proposes MasRouter, a cascade controller network. It sequentially determines the collaboration mode, role allocation, and LLM routing. While maintaining high performance, it reduces the inference cost of MAS by up to 52%, achieving an effective balance between performance and efficiency.

## Background & Motivation

**Background**: LLM-based multi-agent systems (MAS) breach the capability boundaries of single models through the collaboration of multiple LLM agents, demonstrating significant advantages in tasks such as code generation, mathematical reasoning, and question answering. Common MAS frameworks include MapCoder (multi-role code generation) and GPTSwarm (graph-based agent collaboration).

**Limitations of Prior Work**: The core limitation of current MAS is the **exorbitant cost**—each agent calls a powerful LLM (e.g., GPT-4), incurring huge API expenses under multi-turn interactions. Existing LLM routing methods (e.g., RouterBench, RouteLLM) are designed solely for single-agent scenarios, only considering "which query uses which model," while completely ignoring the richer decision-making dimensions in MAS: Which collaboration mode should be used? What model should each role use? Do simple tasks even require multi-agent collaboration?

**Key Challenge**: While the power of MAS stems from multi-agent collaboration, not all tasks require a fully configured MAS. Many simple queries can be solved with a single agent, whereas complex queries require multiple agents but do not necessarily need the most expensive model for every role. The lack of intelligent routing leads to systematic resource waste.

**Goal**: Extend LLM routing from single-agent to multi-agent scenarios, and unifiedly manage decisions across three dimensions: collaboration mode selection, role allocation, and model selection.

**Key Insight**: Model the construction of MAS as a cascade decision problem—first deciding "whether to collaborate," then "who does what," and finally "which model to use." This is achieved by learning a lightweight controller to automate these decisions.

**Core Idea**: Utilize a cascade controller network to dynamically construct the optimal MAS configuration at the query level, achieving adaptive routing that is "simple when possible, complex when necessary."

## Method

### Overall Architecture

MasRouter takes a user query as input and sequentially outputs three levels of decisions through a cascade controller: (1) collaboration mode (e.g., single agent / multi-agent debate / multi-agent workflow); (2) role allocation (if multi-agent, which role is assigned to each position); and (3) LLM selection (which LLM to invoke for each agent). Finally, it constructs the MAS according to the routing result and executes the query. The routing process itself is lightweight and does not require calling large language models.

### Key Designs

1. **Collaboration Mode Determination**:

    - **Function**: Determines whether the current query requires multi-agent collaboration and which collaboration mode to use.
    - **Mechanism**: A lightweight classifier (such as a small BERT or MLP) takes the query embedding as input and outputs the collaboration mode category. Training data is automatically labeled by comparing the performance of historical queries under different modes—for each query, execution is carried out in single-agent and various multi-agent modes, selecting the mode with the optimal balance of high performance and low cost as the label. Predefined collaboration modes include: single-agent direct response, dual-agent debate, multi-agent workflow (with orchestration), etc.
    - **Design Motivation**: Simple queries (such as factual QA) do not require multi-agent overhead. Enabling MAS only for complex tasks that truly need collaboration fundamentally avoids resource waste.

2. **Role Allocation**:

    - **Function**: Allocates appropriate roles to each position in the multi-agent configuration.
    - **Mechanism**: The role allocator is activated if the collaboration mode determination selector chooses a multi-agent mode. Based on query features, it selects an appropriate combination of roles from a predefined role pool (e.g., coder, reviewer, tester, planner). The allocator uses multi-label classification or sequence prediction to generate the role list. Training signals are derived from the performance differences of various role combinations across different queries.
    - **Design Motivation**: Different tasks require different combinations of roles. Code generation may require coder + reviewer, while mathematical reasoning may require solver + verifier. Adaptive role allocation is more flexible and efficient than fixed roles.

3. **LLM Router**:

    - **Function**: Selects the most appropriate LLM for each agent role.
    - **Mechanism**: Once roles are determined, an LLM is independently routed for each role. The routing decision considers three factors: task-LLM matching (different LLMs excel at different types of tasks), cost constraints (selecting the optimal model within a budget), and role requirements (e.g., a reviewer does not need the strongest model). The available LLM pool includes models of varying price and capability levels, such as GPT-4, GPT-3.5, Claude, and Llama. A learned scoring function $s(q, r, m)$ is used to evaluate the tripartite matching between query $q$, role $r$, and model $m$.
    - **Design Motivation**: The core insight is that "not every role requires the most powerful model." For example, a code reviewer might only need comprehension capabilities rather than the strongest generation capability, making GPT-3.5 sufficient. Differentiated routing can dramatically save costs without significantly degrading performance.

### Loss & Training

The training of MasRouter adopts a multi-task learning framework. Each level of the controller has its own classification loss. Additionally, a global performance-cost trade-off objective is introduced: $L = L_{perf} + \lambda \cdot L_{cost}$, where $L_{perf}$ measures the task completion quality of the routing scheme, $L_{cost}$ penalizes excessive inference overhead, and $\lambda$ is the trade-off coefficient. Training data is constructed by running queries under multiple configurations.

## Key Experimental Results

### Main Results

| Dataset | Metric | GPT-4 All | SOTA Router | MasRouter | Cost Savings |
|--------|------|-----------|-------------|-----------|---------|
| HumanEval | Pass@1 | 87.8 | 84.3 | 86.5 | 52.07% |
| MBPP | Pass@1 | 78.2 | 76.4 | 82.7 | 38.2% |
| MATH | Accuracy | 76.5 | 73.1 | 75.8 | 41.5% |
| GSM8K | Accuracy | 94.2 | 91.8 | 93.5 | 35.7% |
| MMLU | Accuracy | 86.7 | 84.5 | 86.1 | 28.3% |

### Ablation Study

| Configuration | MBPP Pass@1 | HumanEval Pass@1 | Cost (Relative) | Description |
|------|------------|------------------|------------|------|
| Full MasRouter | 82.7 | 86.5 | 47.93% | Complete cascade routing |
| w/o Mode Selection | 79.1 | 83.2 | 68.5% | Fixed multi-agent mode |
| w/o Role Allocation | 80.3 | 84.8 | 55.2% | Fixed role configuration |
| w/o LLM Routing | 81.5 | 85.7 | 72.1% | Fixed GPT-4 usage |
| Random Routing | 72.6 | 78.3 | 49.8% | Random routing |

### Key Findings
- **Collaboration mode selection is the largest source of cost savings**: Removing mode selection causes the relative cost to jump from 47.93% to 68.5%, indicating that many queries do indeed not require multi-agent collaboration.
- **Performance improvement is most significant on MBPP** (8.2% higher than SOTA router): This indicates that in code generation tasks, a reasonable combination of roles (coder + reviewer) is more effective than an arbitrary combination of multiple models.
- **Cost savings of up to 52%**: On HumanEval, it slashes costs in half while virtually maintaining original performance, significantly boosting the practicality of MAS.
- **Plug-and-play with mainstream MAS frameworks**: Integrating into MapCoder and GPTSwarm reduces overhead by 17.21% and 28.17% respectively, demonstrating the good generalizability and transferability of the framework.

## Highlights & Insights
- **First to define the MASR problem**: Elevating the construction of multi-agent systems from "manual configuration" to "automatic routing" is a valuable contribution at the problem definition level. The decoupling and cascade design of the three dimensions of the MASR problem (collaboration mode, role, and model) is elegant.
- **Empirical validation that "not every agent needs GPT-4"**: This seemingly intuitive finding is quantitatively validated through systematic experiments. This has direct guiding significance for the engineering practice of MAS—heterogeneous models can be combined to form a cost-effective multi-agent team.
- **Plug-and-play design**: MasRouter does not change the logic of the underlying MAS frameworks. It merely makes intelligent decisions during the MAS "assembly" stage, allowing seamless integration into various existing frameworks.

## Limitations & Future Work
- **High cost of training data collection**: Trying various MAS configurations on a large number of queries is required to construct training data, which itself incurs substantial API calling costs.
- **Generalizability of routing decisions**: Whether a router trained on one dataset can generalize to entirely new task types has not been fully verified.
- **Cascading error propagation**: If the collaboration mode is incorrectly selected, subsequent optimal role allocation and LLM routing cannot salvage the performance.
- **Focus only on API-based models**: For locally deployed open-source models, the cost metric differs (mainly GPU time rather than API fees), which requires adjusting the routing strategy.
- **Future Directions**: Exploring online learning routing strategies (continuously optimizing routing based on actual execution feedback) and latency-constrained rather than just cost-constrained routing can be investigated.

## Related Work & Insights
- **vs RouteLLM**: RouteLLM is an LLM routing method in single-agent scenarios, only deciding which model to use. MasRouter builds upon this by adding two dimensions—collaboration mode and role allocation—acting as a "multi-agent version" of RouteLLM.
- **vs FrugalGPT**: FrugalGPT saves costs through model cascading (trying cheaper models first, then more expensive ones if they fail). MasRouter possesses a more macro-level approach, selecting not only models but also collaboration modes and roles, offering a more comprehensive optimization.
- **vs MapCoder/GPTSwarm**: These are specific MAS frameworks, while MasRouter is the "routing scheduler" sitting on top of them, creating a complementary relationship. Experiments also verify that MasRouter can be integrated directly into these frameworks to reduce overhead.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define the MASR problem, with a novel and practical cascade controller design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively verified across five datasets, complete with ablation and integration experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with a systematic method description.
- Value: ⭐⭐⭐⭐⭐ Highly significant for the deployment and engineering of MAS by directly reducing implementation costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)

</div>

<!-- RELATED:END -->
