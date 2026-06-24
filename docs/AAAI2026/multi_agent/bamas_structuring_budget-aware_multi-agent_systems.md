---
title: >-
  [Paper Note] BAMAS: Structuring Budget-Aware Multi-Agent Systems
description: >-
  [AAAI 2026 Oral][Multi-Agent][Budget-Aware] This paper proposes the BAMAS framework, which employs Integer Linear Programming (ILP) to select the optimal LLM combination under budget constraints, and uses a reinforcement learning policy to choose the best collaboration topology (Linear/Star/Feedback/Planner-Driven). BAMAS achieves accuracy comparable to state-of-the-art multi-agent systems on GSM8K, MBPP, and MATH, while reducing costs by up to 86%.
tags:
  - "AAAI 2026 Oral"
  - "Multi-Agent"
  - "Budget-Aware"
  - "Multi-Agent Collaboration"
  - "Integer Linear Programming"
  - "Topology Selection"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 9ac9afebb1ecc300
---

# BAMAS: Structuring Budget-Aware Multi-Agent Systems

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.21572](https://arxiv.org/abs/2511.21572)  
**Code**: [https://github.com/chunfenri/BAMAS](https://github.com/chunfenri/BAMAS)  
**Area**: Reinforcement Learning
**Keywords**: Budget-Aware, Multi-Agent Collaboration, Integer Linear Programming, Topology Selection, Reinforcement Learning

## TL;DR
This paper proposes the BAMAS framework, which employs Integer Linear Programming (ILP) to select the optimal LLM combination under budget constraints, and uses a reinforcement learning policy to choose the best collaboration topology (Linear/Star/Feedback/Planner-Driven). BAMAS achieves accuracy comparable to state-of-the-art multi-agent systems on GSM8K, MBPP, and MATH, while reducing costs by up to 86%.

## Background & Motivation
**Background**: LLM-based multi-agent systems (AutoGen, MetaGPT, ChatDev) leverage multi-agent collaboration to handle complex tasks, but primarily focus on maximizing performance with little regard for cost control. A single task may require dozens of LLM calls, and costs grow unpredictably with collaboration topology and reasoning depth.

**Limitations of Prior Work**: (1) Existing frameworks treat cost as an afterthought, lacking proactive budget management; (2) users cannot specify a budget ceiling to constrain system behavior; (3) different topologies suit different tasks and budget levels, yet existing systems use fixed topologies without adaptive adjustment.

**Key Challenge**: There exists a fundamental trade-off between performance and cost — using stronger LLMs and more complex collaboration topologies improves performance but dramatically increases cost. Finding the optimal LLM allocation and collaboration strategy under a given budget remains an open challenge.

**Goal**: Given a task, a pool of available LLMs, and a budget ceiling, how can one automatically construct a multi-agent system with optimal performance?

**Key Insight**: The problem is decomposed into two optimizable sub-problems — LLM selection (combinatorial optimization → ILP) and topology selection (policy learning → RL), each solved with the most appropriate optimization method.

**Core Idea**: Use ILP for budget-constrained LLM selection, and RL for task- and budget-adaptive topology selection, achieving tunable cost–performance trade-offs.

## Method

### Overall Architecture
BAMAS operates in three stages: (1) **Budget-Constrained LLM Configuration** — ILP selects the optimal LLM subset $\mathcal{P}$ from the LLM pool within budget $B$; (2) **Collaboration Topology Selection** — an RL policy $\pi_\theta$ chooses the best topology $t$ based on task description and budget; (3) **Agent Instantiation** — LLMs from $\mathcal{P}$ are assigned roles (executor/reviewer/planner) according to topology $t$ and the task is executed.

### Key Designs

1. **Budget-Constrained LLM Configuration (ILP)**:

    - **Function**: Select the performance-optimal LLM combination within budget $B$.
    - **Mechanism**: LLMs are ranked into tiers by performance ($\mathcal{A}_1$ strongest to $\mathcal{A}_L$ weakest), using the LMSys leaderboard as a performance proxy. Recursive weights $W_i = 1 + \sum_{j=i+1}^{L}(W_j \cdot \lfloor B/c_j \rfloor)$ are constructed to ensure that the weight of any higher-tier LLM always exceeds any combination of lower-tier LLMs. The ILP objective is $\max \sum W_i \cdot x_{ij}$, subject to total cost $\leq B$ and at least 2 LLMs selected.
    - **Design Motivation**: Empirical evidence shows that a single strong model often outperforms an ensemble of weaker models, motivating a "performance-first" selection strategy. ILP guarantees an exact global optimum, avoiding the local optima inherent in greedy approaches.

2. **Collaboration Topology Selection (RL)**:

    - **Function**: Select the most suitable collaboration mode from 4 topologies based on task characteristics and budget level.
    - **Mechanism**: The policy network $\pi_\theta(t|T,B)$ takes a task embedding (MiniLM, 384-dimensional) and a budget scalar as input, outputting a probability distribution over 4 topologies. It is trained offline via REINFORCE with a composite reward $R_{\text{final}} = w_{\text{perf}} \cdot R_{\text{perf}} + w_{\text{cost}} \cdot R_{\text{cost}}$, where success rewards, over-budget penalties, and budget-saving bonuses jointly guide policy learning.
    - **Design Motivation**: Different tasks require different collaboration modes (mathematical reasoning favors Feedback iteration; code generation favors Linear pipelines), making fixed topologies inadequate. RL is preferred over rules because the optimal topology also depends on the budget level — under tight budgets, the policy favors simpler topologies to avoid overspending.

3. **Library of 4 Collaboration Topologies**:

    - **Linear**: Sequential reasoning where each agent continues from the previous agent's output. Suitable for multi-step reasoning.
    - **Star**: Parallel hypothesis generation and evaluation via a divide-and-conquer strategy. Suitable for decomposable problems.
    - **Feedback**: Generate-review loop for iterative refinement. Suitable for tasks requiring self-correction.
    - **Planner-Driven**: A central planner dynamically coordinates agents. Most flexible but highest cost and greatest instability.

### Loss & Training
The RL loss consists of policy gradient with entropy regularization: $\mathcal{L}(\theta) = -\hat{\mathbb{E}}[\log \pi_\theta(t|T,B) \cdot R_{\text{final}}(\tau)] - \beta \cdot H(\pi_\theta)$. Offline training avoids the high cost of online data collection. The Adam optimizer is used with a batch size of 20,000 and training for 10 epochs.

## Key Experimental Results

### Main Results
Cost–accuracy comparison on GSM8K and MBPP:

| Method | Setting | GSM8K Acc (%) | GSM8K Cost | MBPP Acc (%) | MBPP Cost |
|--------|---------|--------------|------------|-------------|-----------|
| AutoGen | DeepSeek-V3 | 95.4 | 1425.3 | 80.8 | 2661.3 |
| MetaGPT | DeepSeek-V3 | 93.5 | 3235.4 | 82.2 | 3735.1 |
| ChatDev | DeepSeek-V3 | 95.0 | 2733.1 | 81.2 | 3635.1 |
| BAMAS | Budget 1625 | **95.3** | **542.9** | – | – |
| BAMAS | Budget 1250 | 94.9 | 447.0 | **82.6** | **529.2** |

MATH dataset:

| Method | Acc (%) | Cost |
|--------|---------|------|
| AutoGen (GPT-4.1 nano) | 77.6 | 797.2 |
| BAMAS (Budget 2000) | **81.2** | 646.0 |

### Ablation Study

| Configuration | GSM8K Acc (%) | GSM8K Cost | Note |
|---------------|--------------|------------|------|
| Naive-CostAware L5+DeepSeek | 95.3 | 1650.8 | Greedy tier-5; highest accuracy but expensive |
| BAMAS Budget 1625 | **95.3** | **542.9** | Same accuracy, 67% cost reduction |
| Naive-CostAware L1+GPT-nano | 89.7 | 216.7 | Cheapest but low accuracy |
| BAMAS Budget 500 | 87.9 | 222.4 | Similar cost, adjustable |

### Key Findings
- BAMAS achieves 82.6% on MBPP at a cost of only 529.2, a reduction of 86% compared to MetaGPT (3735.1) — the most significant cost reduction observed.
- The learned policy exhibits meaningful patterns: math tasks favor the Feedback topology (69.8% on MATH), while code tasks favor the Linear topology.
- Under low budgets, the policy biases toward simpler topologies (Linear/Star); under higher budgets, it selects more complex topologies (Feedback) — reflecting risk-aware behavior.
- The Planner-Driven topology is never selected, indicating that RL has learned it is too costly and unstable to be worth the risk.
- Over-budget rates are negligible: 0 violations on GSM8K and at most 5/500 (1%) on MBPP, demonstrating effective budget control.

## Highlights & Insights
- Decomposing the multi-agent system construction problem into "whom to select" (ILP) and "how to collaborate" (RL) is conceptually clean, with each sub-problem solved by the most appropriate optimization method.
- The recursive weight design guarantees lexicographic optimality for ILP — the weight of any higher-tier LLM always exceeds any combination of lower-tier ones, a concise and effective modeling technique.
- The topology selection preferences learned by the RL policy are highly interpretable (math → iterative feedback, code → linear pipeline, low budget → simple topology), rather than opaque black-box decisions.

## Limitations & Future Work
- Only 2 LLMs are used (DeepSeek-V3 and GPT-4.1 nano); this small pool makes it difficult to fully demonstrate ILP's advantages in large-scale LLM selection.
- The 4 topologies are predefined and do not support automatic discovery of new collaboration modes. In practice, hybrid topologies (e.g., different topologies at different stages) may be needed.
- Cost estimation uses a fixed token count (500 input tokens), but actual token consumption varies significantly, potentially leading to inaccurate budget estimates.
- Evaluation is limited to code generation and mathematical reasoning; validation on more diverse tasks (e.g., creative writing, information retrieval) is lacking.

## Related Work & Insights
- **vs. AutoGen**: AutoGen provides a flexible agent architecture but does not account for budgets. BAMAS can be viewed as a budget-aware wrapper around AutoGen, deciding which models and topologies to use before delegating execution to an AutoGen-like engine.
- **vs. FrugalGPT/TREACLE**: These works optimize costs for single-model settings (routing/cascading). BAMAS extends cost optimization to multi-agent collaboration, requiring joint optimization of model selection and collaboration strategy.
- **vs. ADAS**: ADAS automatically designs agent system architectures but ignores cost constraints. BAMAS complements ADAS by incorporating cost-dimension optimization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First work to systematically introduce budget constraints into multi-agent system construction; the ILP + RL combinatorial optimization approach is novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers three datasets but only 2 LLMs; larger-scale and more diverse scenario validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Research goals are clearly articulated; the RQ-driven evaluation structure is well-organized with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Cost-awareness is a critical requirement for real-world deployment of multi-agent systems; BAMAS offers a practical and viable solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Goal-Aware Identification and Rectification of Misinformation in Multi-Agent Systems](../../ICLR2026/multi_agent/goal-aware_identification_and_rectification_of_misinformation_in_multi-agent_sys.md)
- [\[ICML 2026\] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](../../ICML2026/multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)
- [\[ACL 2026\] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](../../ACL2026/multi_agent/bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)
- [\[ICLR 2026\] ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning](../../ICLR2026/multi_agent/atlas_constraints-aware_multi-agent_collaboration_for_real-world_travel_planning.md)

</div>

<!-- RELATED:END -->
