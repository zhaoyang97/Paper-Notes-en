---
title: >-
  [Paper Note] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems
description: >-
  [NeurIPS 2025][Graph Learning][Multi-LLM collaboration] This paper proposes the Heterogeneous Swarms algorithm, which models multi-LLM systems as directed acyclic graphs (DAGs) and employs PSO to alternately execute a role-step (optimizing the topological connectivity structure among LLMs) and a weight-step (quantifying individual contributions via the JFK-score and optimizing model weights), achieving an average improvement of 18.5% over 17 baseline methods across 12 tasks.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Multi-LLM collaboration
  - particle swarm optimization
  - DAG structure learning
  - model role optimization
  - model weight optimization
date: 2026-05-08
content_hash: f9bbeb7a0df971e9
---

# Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems

**Conference**: NeurIPS 2025
**arXiv**: [2502.04510](https://arxiv.org/abs/2502.04510)
**Code**: [https://github.com/BunsenFeng/heterogeneous_swarm](https://github.com/BunsenFeng/heterogeneous_swarm)
**Area**: Graph Learning
**Keywords**: Multi-LLM systems, particle swarm optimization, DAG structure learning, model collaboration, joint role and weight optimization

## TL;DR
This paper proposes the Heterogeneous Swarms algorithm, which models multi-LLM systems as directed acyclic graphs (DAGs) and employs particle swarm optimization (PSO) to jointly optimize model roles (graph topology) and model weights, achieving an average improvement of 18.5% over 17 baselines across 12 tasks.

## Background & Motivation

Multi-LLM collaboration is an important direction for surpassing the capability ceiling of individual models. The central challenge lies in assigning appropriate roles and calibrating suitable weights for each LLM. Existing approaches suffer from two bottlenecks:

**Fixed-weight systems** (e.g., LLM debate and multi-agent frameworks): These systems employ static black-box LLMs and contextualize roles through textual interaction, preventing the models themselves from adapting to specific tasks, with inflexibility arising from immutable weights.

**Fixed-role systems** (e.g., AgentVerse and MetaGPT): These systems assign fixed roles to LLMs via hand-crafted prompts; extending them to new tasks or domains requires substantial prompt engineering, and role assignment relies heavily on prior knowledge.

**Key Challenge**: Role and weight are two orthogonal dimensions of a multi-LLM system, yet existing methods almost exclusively optimize only one of them. Dynamically and jointly optimizing both is the key to achieving task-adaptive systems.

**Key Insight**: Role optimization is reformulated as a graph structure learning problem (learning a DAG that governs the input–output flow among LLMs), while weight optimization is coupled with individual contribution quantification, with PSO unifying the search over both dimensions.

**Core Idea**: A continuous adjacency matrix represents the discrete DAG; the optimal graph structure is searched via G-Decode decoding and PSO optimization; a JFK-score measures each LLM's contribution to the system and guides weight updates.

## Method

### Overall Architecture
The input consists of a pool of LLM experts $\{x_i\}_{i=1}^n$ and a task utility function $f$ (e.g., accuracy). The output is an optimized DAG structure $\mathcal{A}$ together with adapted model weights, forming the final multi-LLM system. The framework alternates between a role-step and a weight-step until the utility function ceases to improve or the maximum number of iterations is reached.

### Key Designs

1. **Role-step: Graph Structure Optimization**

    - Function: Learn the optimal DAG topology to determine input–output relationships among LLMs.
    - Mechanism: A population of continuous adjacency matrices $\{A^i\}_{i=1}^N \sim \mathcal{U}_{n\times n}(0,1)$ is randomly initialized and decoded into discrete DAGs via **G-Decode**: a terminal node $k$ is first selected by inverse out-degree top-$p$ sampling, after which new nodes $u$ are iteratively chosen by out-degree and connected to existing nodes with softmax probability $\frac{\exp(a_{uv})}{\sum_{i\in\mathcal{E}}\exp(a_{ui})}$. The decoded DAG is executed in topological order, its utility $f$ is evaluated, and the adjacency matrices are updated via PSO:
    $$\{A^i\}_{i=1}^N, A_\text{best} \leftarrow \text{PSO}(\{A^i\}_{i=1}^N, \{f(\text{G-Decode}(A^i))\}_{i=1}^N)$$
    - Design Motivation: Relaxing discrete graph optimization to continuous space enables gradient-free search algorithms such as PSO to be applied directly. G-Decode guarantees that the resulting graph is always a DAG (new nodes are only connected to existing nodes), preventing cycles.

2. **Weight-step: JFK-Score-Based Weight Optimization**

    - Function: Evaluate each LLM's individual contribution within the optimal DAG and guide model weight updates accordingly.
    - Mechanism: Given the optimal DAG $A_\text{best}$, the $n$ LLMs are randomly assigned to positions in the DAG $M$ times, yielding $M$ assignments $\{\mathcal{X}^i\}_{i=1}^M$. The JFK-score is defined as:
    $$\text{JFK-score}(x_i) = \frac{\sum_{j=1}^M \text{cnt}_{i,j} \times f(\mathcal{X}^j)}{\sum_{j=1}^M \text{cnt}_{i,j}}$$
    where $\text{cnt}_{i,j}$ denotes the frequency with which model $x_i$ appears in the $j$-th assignment. Model weights are then optimized via PSO.
    - Design Motivation: Directly assessing a single LLM's utility within a multi-LLM system is non-trivial. The JFK-score addresses this credit assignment problem by robustly quantifying each LLM's average contribution across roles and collaboration partners through repeated random assignment and frequency-weighted aggregation. The name is inspired by the quote: "Ask not what the multi-LLM system can do for you, ask what you can do for the multi-LLM system."

3. **PSO Optimizer**

    - Function: Optimize continuous adjacency matrices and model weights in a non-differentiable search space.
    - Mechanism: The velocity of each particle $x_i$ is updated as a weighted mixture of inertia, personal-best direction, global-best direction, and a repulsion term away from the global worst:
    $$v_i \leftarrow \frac{1}{\mathcal{C}}[r_v\phi_v v_i + r_p\phi_p(p_i - x_i) + r_g\phi_g(g - x_i) - r_w\phi_w(g_w - x_i)]$$
    followed by $x_i \leftarrow x_i + \lambda v_i$.
    - Design Motivation: LLM inference and utility evaluation are non-differentiable, making gradient-based methods inapplicable. PSO, as a gradient-free swarm search algorithm, is naturally suited to this black-box optimization setting and exploits the diversity of multiple search particles.

### Loss & Training
- Ten domain-fine-tuned variants of Gemma-7B serve as the expert pool.
- PSO parameters: $N=10$ particles, $M=10$ random assignments, search patience of 6, maximum 20 iterations.
- Role-step and weight-step are executed alternately.

## Key Experimental Results

### Main Results

| Dataset | Metric | H-Swarms | Best Baseline | Gain |
|--------|------|----------|----------|------|
| MMLU-pro | Acc | 0.312 | 0.254 (Model Swarms) | +22.8% |
| NLGraph | Acc | 0.660 | 0.672→0.633 | Near-optimal |
| GAIA-text | Acc | 0.250 | 0.143 (multiple baselines) | +74.8% |
| GSM8k | Acc | 0.481 | 0.459 (Model Swarms) | +4.8% |
| K-Cross | Acc | 0.450 | 0.428 (Model Swarms) | +5.1% |
| AbstainQA | Acc | 0.220 | 0.175 (Model Swarms) | +25.7% |

The proposed method achieves the best performance on 11 of 12 datasets, with an average improvement of 18.5%.

### Ablation Study

| Configuration | MMLU-pro | K-Cross | GSM8k | NLGraph | GAIA-text | Note |
|------|----------|---------|-------|---------|-----------|------|
| Full | 0.312 | 0.450 | 0.481 | 0.660 | 0.250 | Complete method |
| w/o Role | 0.242 | 0.352 | 0.392 | 0.530 | 0.107 | Role optimization removed |
| w/o Weight | 0.237 | 0.342 | 0.363 | 0.588 | 0.143 | Weight optimization removed |
| Role Baselines avg | 0.218 | 0.318 | 0.323 | 0.531 | 0.095 | Role-only baselines |
| Weight Baselines avg | 0.222 | 0.352 | 0.325 | 0.538 | 0.082 | Weight-only baselines |

The relative importance of role and weight varies by task: weight is more critical for knowledge tasks, while role is more critical for agent tasks; this trend holds consistently across 10 of the 12 tasks.

### Key Findings
- **Collaborative Gain**: The average C-Gain is 0.213, demonstrating a superadditive effect in multi-LLM systems. In the $B_0$ bucket (problems unsolvable by any single LLM), the multi-LLM system still resolves 18.1%.
- **Diversity is critical**: Performance improves by an average of 89% when moving from the lowest to the highest model diversity configuration.
- **Sparsification is viable**: Threshold pruning or $\ell_1$ regularization achieves a favorable trade-off between performance and inference speed (e.g., at $\tau=0.2$, 36.1% of connections are removed with only a 21.8% drop on K-Cross).
- **Heterogeneous role distribution**: Branch nodes tend to adopt a "divide-and-conquer" role, while aggregation nodes tend to perform "refinement and feedback."

## Highlights & Insights
1. Formalizing multi-LLM collaboration as a graph optimization problem is an elegant abstraction that enables joint optimization of roles and weights within a unified framework.
2. The JFK-score is a clever mechanism for quantifying individual contributions in multi-agent systems, with strong generalizability.
3. The choice of PSO as the optimizer is well-motivated: the utility function of a multi-LLM system is inherently non-differentiable.
4. The empirical finding that the relative importance of role versus weight varies across task types is a valuable insight.

## Limitations & Future Work
- Experiments primarily use homogeneous models (fine-tuned variants of Gemma-7B); extending to truly heterogeneous architectures and scales is a more challenging and practically relevant scenario.
- PSO search efficiency is limited, as each iteration requires a large number of LLM calls, incurring substantial deployment overhead.
- Once the DAG topology is fixed, adaptability to out-of-distribution inputs is limited.
- Validation is restricted to 7B-scale models; whether collaborative gains remain significant at larger scales has yet to be verified.

## Related Work & Insights
- **Model Swarms** (Feng et al., 2024): The weight optimization component of this work directly builds upon Model Swarms, extending it with the role dimension.
- **GPT-Swarm**: Optimizes LLM communication graphs via gradients; the PSO-based approach in this paper is more general as it does not require differentiability.
- **MoE / Model Merging**: The shift from single merged models to multi-model collaborative systems represents an important paradigm transition.
- Inspiration: PSO could be applied to other non-differentiable LLM optimization scenarios, such as prompt search and RAG pipeline optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying graph learning and PSO to jointly optimize roles and weights in multi-LLM systems is a novel formulation, though individual components (PSO, DAG) are established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Twelve tasks, 17 baselines, and multi-dimensional analyses (collaborative gain, role statistics, diversity, sparsification) constitute a very comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and figures are informative; the JFK analogy is apt, though some notation could be simplified.
- Value: ⭐⭐⭐⭐ The work provides a systematic optimization framework for multi-LLM collaboration, though computational overhead remains a key bottleneck for practical deployment.

---

**Conference**: NeurIPS 2025
**arXiv**: [2502.04510](https://arxiv.org/abs/2502.04510)
**Code**: [https://github.com/BunsenFeng/heterogeneous_swarm](https://github.com/BunsenFeng/heterogeneous_swarm)
**Area**: Multi-LLM Systems / Graph Learning
**Keywords**: Multi-LLM collaboration, particle swarm optimization, DAG structure learning, model role optimization, model weight optimization

## TL;DR
This paper proposes the Heterogeneous Swarms algorithm, which models multi-LLM systems as directed acyclic graphs (DAGs) and employs PSO to alternately execute a role-step (optimizing the topological connectivity structure among LLMs) and a weight-step (quantifying individual contributions via the JFK-score and optimizing model weights), achieving an average improvement of 18.5% over 17 baseline methods across 12 tasks.

## Background & Motivation
Multi-LLM collaborative systems represent an important direction in current LLM research: multiple models with complementary specializations can collectively transcend individual capability boundaries. Existing approaches fall into two categories: **fixed-weight** systems (e.g., debate and multi-agent frameworks) define roles through textual interaction while leaving model parameters unchanged, creating a bottleneck for flexible adaptation; **fixed-role** systems (e.g., chain/star topologies) organize LLMs according to hand-crafted workflows, yet extending them to new tasks demands substantial prompt engineering. The root cause is that both role and weight are task-dependent and should be jointly optimized rather than independently designed.

The paper adopts a distinctly novel angle: it reinterprets "model role" as the input–output relationship among LLMs (i.e., the DAG structure), reformulating role optimization as a graph structure learning problem, and leverages PSO—a gradient-free optimization method—to simultaneously search for the optimal graph topology and model weights. The core idea is to **use swarm intelligence to jointly search the graph space and weight space, discovering heterogeneous multi-LLM collaborative systems**.

## Method

### Overall Architecture
The input is a pool of LLM experts $\{{\bf x}_i\}_{i=1}^n$ and a task utility function $f$ (e.g., accuracy). The system maintains two sets of optimizable variables: a population of continuous adjacency matrices $\{{\bf A}^i\}_{i=1}^N$ (representing candidate DAG structures) and LLM weights $\{{\bf x}_i\}_{i=1}^n$. The algorithm alternates between a role-step and a weight-step until $f$ ceases to improve, ultimately outputting an optimized DAG topology together with adapted model weights.

### Key Designs

1. **Role-step (Role Optimization = Graph Structure Learning)**:
    - Function: Learn the optimal DAG topology governing information flow among LLMs.
    - Mechanism: $N$ continuous adjacency matrices ${\bf A} \in \mathbb{R}^{n \times n}$ are randomly initialized, where $a_{ij}$ represents the likelihood of a directed edge from model $i$ to model $j$. The **G-Decode** algorithm decodes continuous matrices into discrete DAGs: a terminal node $k$ is selected via inverse out-degree top-$p$ sampling, and remaining nodes are iteratively added and connected to existing nodes with softmax probability. The decoded DAG is executed in topological order, its utility $f$ is evaluated, and PSO updates the adjacency matrices:
    $$\{{\bf A}^i\}_{i=1}^N, {\bf A}_{\text{best}} \leftarrow \text{PSO}(\{{\bf A}^i\}_{i=1}^N, \{f(\text{G-decode}({\bf A}^i))\}_{i=1}^N)$$
    - Design Motivation: Hand-crafted chain/star structures cannot generalize across tasks; the continuous adjacency matrix together with G-Decode elegantly transforms discrete graph search into a continuous optimization problem, enabling PSO to operate in matrix space.

2. **Weight-step (Weight Optimization + JFK-Score)**:
    - Function: Quantify each LLM's individual contribution within the multi-LLM system and optimize weights accordingly.
    - Mechanism: Given the optimal DAG ${\bf A}_{\text{best}}$ found by the role-step, LLMs are randomly assigned to positions in the DAG $M$ times, yielding $M$ assignment configurations $\{\mathcal{X}^i\}_{i=1}^M$. Each model's JFK-score is the frequency-weighted average of the utility of the multi-LLM systems in which it participates:
    $$\text{JFK-score}({\bf x}_i) = \frac{\sum_{j=1}^M \text{cnt}_{i,j} \times f(\mathcal{X}^j)}{\sum_{j=1}^M \text{cnt}_{i,j}}$$
    Model weights are then updated via PSO: $\{{\bf x}_i\}, {\bf x}_{\text{best}} \leftarrow \text{PSO}(\{{\bf x}_i\}, \{\text{JFK-score}({\bf x}_i)\})$.
    - Design Motivation: Directly evaluating a single model's utility within a multi-LLM system is difficult; the JFK-score elegantly resolves this credit assignment problem through repeated random assignment and aggregation.

3. **PSO Optimizer (Particle Swarm Optimization)**:
    - Function: Serve as the unified optimization engine for both role and weight optimization.
    - Mechanism: The velocity of each particle ${\bf x}_i$ is updated by combining four signals—inertia ($\phi_v {\bf v}_i$), personal best ($\phi_p({\bf p}_i - {\bf x}_i)$), global best ($\phi_g({\bf g} - {\bf x}_i)$), and repulsion from the global worst ($-\phi_w({\bf g}_w - {\bf x}_i)$)—with normalization and stochastic factors.
    - Design Motivation: PSO is a gradient-free optimization method naturally suited to non-differentiable LLM utility functions; parallel search with multiple particles exploits the diverse specializations of LLM experts.

### Loss & Training
- Ten domain-fine-tuned Gemma-7B experts serve as the initial LLM pool.
- PSO parameters: $N=10$ candidate DAGs, $M=10$ random assignments, top-$p$ $p=0.8$.
- Search patience of 6, maximum 20 iterations, with other hyperparameters determined by grid search.

## Key Experimental Results

### Main Results

| Task Category | Dataset | H-Swarms | Runner-up | Gain |
|----------|--------|----------|----------|------|
| Knowledge | MMLU-pro | 0.312 | 0.254 (Model Swarms) | +22.8% |
| Knowledge | K-Cross | 0.450 | 0.428 (Model Swarms) | +5.1% |
| Reasoning | GSM8k | 0.481 | 0.459 (Model Swarms) | +4.8% |
| Reasoning | NLGraph | 0.660 | 0.672→0.660 | Competitive |
| Agent | GAIA-text | 0.250 | 0.143 (multiple tied) | +74.8% |
| Agent | AB-kg | 0.425 | 0.392 (multiple tied) | +8.4% |
| Misc. | AbstainQA | 0.220 | 0.175 (Model Swarms) | +25.7% |

The proposed method achieves the best performance on 11 of 12 datasets, surpassing the runner-up by an average of 18.5%.

### Ablation Study

| Configuration | MMLU-pro | NLGraph | GAIA-text | Note |
|------|---------|---------|-----------|------|
| Full | 0.312 | 0.660 | 0.250 | Complete method |
| w/o Role | 0.242 | 0.530 | 0.107 | Role optimization removed |
| w/o Weight | 0.237 | 0.588 | 0.143 | Weight optimization removed |
| Role Baselines avg | 0.218 | 0.531 | 0.095 | Static/dynamic role baselines |
| Weight Baselines avg | 0.222 | 0.538 | 0.082 | Static/dynamic weight baselines |

The relative importance of role and weight varies by task: weight is more critical for knowledge tasks, while role is more critical for agent tasks; this trend is consistent across 10 of the 12 datasets.

### Key Findings
- **Significant collaborative gain**: The LLM ensemble resolves 18.1% of problems in the $B_0$ bucket (problems unsolvable by any single model), with an average collaborative gain C-Gain = 0.213.
- **Role heterogeneity**: LLMs at different topological positions naturally develop differentiated roles—branch nodes tend toward divide-and-conquer, while aggregation nodes tend toward refinement and feedback.
- **Diversity is critical**: Performance improves by an average of 89% when moving from the lowest to the highest diversity configuration.
- **Sparsification is viable**: Threshold pruning or $\ell_1$ regularization reduces inference cost by 3–36% at a marginal accuracy cost of less than 5%.

## Highlights & Insights
- Reformulating "role" from an abstract prompt concept into an optimizable graph structure is an elegant and principled design choice.
- The credit assignment mechanism underlying the JFK-score is clever: decoupling individual model contribution estimation through repeated random assignment and aggregation is both simple and effective.
- The paper demonstrates inference-time scaling: average performance improves by 27.1% as the number of collaborating small models grows from 2 to 10.
- The successful application of PSO to LLM system design underscores the value of gradient-free optimization in the LLM era.

## Limitations & Future Work
- High computational cost: repeated LLM calls for search make the approach less practical in high-API-cost settings.
- Validation is limited to same-architecture 7B-scale models; extension to genuinely heterogeneous model mixtures (e.g., 7B + 70B) remains unexplored.
- The fixed DAG topology does not support dynamic, input-conditioned routing.
- The number of random assignments $M$ for the JFK-score may be insufficient for accurate contribution estimation.

## Related Work & Insights
- Compared to GPT-Swarm (graph-based LLM scheduling), this work adds the weight optimization dimension.
- Compared to Model Swarms (weight optimization), this work adds the graph structure optimization dimension.
- Inspiration: The Mixture-of-Experts router paradigm could be incorporated to enable conditional dynamic graph routing.
- Inspiration: The JFK-score design is transferable to other multi-module systems requiring component contribution assessment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reformulating role optimization as graph learning and introducing the JFK-score for contribution quantification are highly original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Twelve datasets across 4 task categories, 17 baselines, and rich ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, but the notation is dense and requires careful reading.
- Value: ⭐⭐⭐⭐ The work provides a unified optimization framework for multi-LLM system design, though computational cost constrains practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastJAM: a Fast Joint Alignment Model for Images](fastjam_a_fast_joint_alignment_model_for_images.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[NeurIPS 2025\] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)

</div>

<!-- RELATED:END -->
