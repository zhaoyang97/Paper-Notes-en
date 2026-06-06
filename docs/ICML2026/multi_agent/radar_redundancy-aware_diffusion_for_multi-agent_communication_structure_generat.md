---
title: >-
  [Paper Note] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation
description: >-
  [ICML 2026][Multi-Agent][Multi-agent collaboration] RADAR reformulates the communication topology design of multi-LLM-agent systems as a "redundancy-aware" discrete graph diffusion process. By using effective size as a g…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-agent collaboration"
  - "Graph diffusion"
  - "Communication topology"
  - "effective size"
  - "redundancy-aware"
date: 2026-05-08
content_hash: d95edd3cc0a254a2
---

# RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation

**Conference**: ICML 2026  
**arXiv**: [2605.09907](https://arxiv.org/abs/2605.09907)  
**Code**: https://github.com/cszhangzhen/RADAR  
**Area**: Multi-Agent Systems / Graph Diffusion Models / LLM Agent  
**Keywords**: Multi-agent collaboration, Graph diffusion, Communication topology, effective size, redundancy-aware

## TL;DR
RADAR reformulates the communication topology design of multi-LLM-agent systems as a "redundancy-aware" discrete graph diffusion process. By using effective size as a guiding signal, it incrementally generates query-adaptive collaboration graphs, achieving higher accuracy, lower token consumption, and stronger robustness across 6 benchmarks.

## Background & Motivation

**Background**: Multi-LLM-Agent systems (LLM-Debate, MetaGPT, AutoGen, etc.) have proven significantly more capable than single agents. However, their critical bottleneck lies in the "communication topology"—who talks to whom and in what order. Early methods utilized fixed manual structures like chain, star, tree, or fully-connected. Works from the past year (GPTSwarm, G-Designer, MaAS, ARG-Designer, GTD) have shifted toward "automated topology design."

**Limitations of Prior Work**: Automated approaches fall into three categories, each with issues. First, agentic profiling (coordinated by a meta-agent) suffers from single-point bottlenecks. Second, search-based methods (heuristic search in topology space) are computationally expensive and not scalable. Third, graph learning (using VAEs to predict the whole graph at once) produces coarse generation that fails to capture detailed dependencies. Moreover, increased structural complexity leads to excessive token consumption—cited data shows complex topologies can consume $2 \sim 11.8\times$ more tokens than chains. AgentPrune and others mitigate this via pruning, but these "post hoc" fixes only modify fixed agent sets and cannot design from scratch under efficiency constraints.

**Key Challenge**: The contradiction between expressiveness (topologies must be complex enough for hard problems) and efficiency (token costs cannot explode). Existing methods either sacrifice one or treat them as independent sub-problems.

**Goal**: Explicitly model "redundancy" during the communication graph generation process, allowing structural formation and redundancy control to proceed jointly while supporting query-adaptivity (sparse for easy tasks, dense for hard tasks).

**Key Insight**: The authors borrow "effective size" from social network analysis (Burt 1992), which measures the proportion of non-redundant information in a node's ego network. If two neighbors are highly interconnected, the information they provide overlaps, resulting in low effective size. Integrating this into the graph diffusion process provides a natural "redundancy metric" as a guiding signal.

**Core Idea**: Reformulate multi-agent communication topology design as an "effective size-guided + query-conditioned" discrete graph diffusion problem, denoising the final topology step-by-step from an empty graph.

## Method

### Overall Architecture

Input: Task query $\mathcal{Q}$, candidate agent set (with Role / State / Plugin).  
Output: A directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $A_{ij} = 1$ indicates agent $v_i$ sends information to $v_j$. Agents are activated according to the topological sort of $\mathcal{G}$, and a final answer is provided by an Aggregate function (majority vote / concatenation / last agent output).

The training pipeline uses various baseline topologies (fully connected / mesh / star / layered / random with 3 or 4 agents) on 50 training queries to obtain "topology $\to$ task performance" samples for training the graph diffusion model. During inference, for a new query, the denoising network iteratively denoises from an empty graph to produce a query-tailored collaboration topology.

### Key Designs

1.  **Effective Size as Redundancy Metric**:
    - **Function**: Calculates a scalar for each agent in the current graph reflecting the non-redundancy of its in/out-neighborhood, serving as a guidance signal for the diffusion process.
    - **Mechanism**: Inward effective size is defined as $\varphi^i(v_k) = |\mathcal{N}_i(v_k)| - \frac{\sum_{j,q \in \mathcal{N}_i(v_k)} A_{jq} \mathbb{I}[r(j) = r(q)]}{|\mathcal{N}_i(v_k)|}$, where the numerator is the number of in-neighbors and the denominator penalizes redundant neighbor pairs with the same role and mutual connections. Outward $\varphi^o(v_k)$ is defined symmetrically, and they are merged via $\varphi(v_k) = (1-\beta) \varphi^i(v_k) + \beta \varphi^o(v_k)$. High $\varphi$ implies the agent receives diverse inputs and distributes information through non-overlapping paths.
    - **Design Motivation**: Previous methods relied on black-box signals like task accuracy, which are sparse and delayed. Effective size is a local, differentiable (structurally), and redundancy-related geometric quantity that provides fine-grained structural guidance at every diffusion step.

2.  **Redundancy-Aware Forward Diffusion (with ordering network)**:
    - **Function**: Progressively masks nodes and their edges from the training graph $\mathcal{G}_0$ in a meaningful order to obtain partially masked intermediate graphs $\mathcal{G}_1, \mathcal{G}_2, \dots$ for denoising supervision.
    - **Mechanism**: Defines an ordering network $q_\psi(\pi | \mathcal{G}_0, \varphi) = \prod_t q_\psi(\pi_t | \mathcal{G}_0, \varphi, \pi_{(<t)})$ to sample which node to mask. Specifically, GNN + positional encodings yield node embeddings $h_t$, and sampling follows $q_\psi(\pi_t | \cdot) \propto \exp(h_t + \varphi(v_t))$—nodes with higher effective size are masked first. The intuition is that graphs with high effective size decompose into weakly overlapping sub-structures, facilitating incremental generation learning.
    - **Design Motivation**: General graph diffusion uses random or fixed ordering, losing structural regularity. The effective size-guided sequence makes the reverse denoising task more "structured"—simple sub-structures are restored first, followed by complex dependencies.

3.  **Conditional Reverse Denoising Network**:
    - **Function**: Starting from an empty graph, it restores nodes and their connections to previously generated nodes given query $\mathcal{Q}$ to obtain a task-adaptive topology.
    - **Mechanism**: The denoising network $p_\theta(\mathcal{G}_t | \mathcal{G}_{t+1}, \mathcal{Q})$ uses GAT-style attention at each layer $\alpha_{i,j} = \frac{\exp(\text{ReLU}(\mathbf{a}^\top [\mathbf{W h}_i^l \| \mathbf{W h}_j^l]))}{\sum_k \exp(\cdot)}$ to obtain node embeddings. The final layer adds an effective size bias $\mathbf{h}_i^L \leftarrow \mathbf{h}_i^L + \varphi(v_i) \mathbf{1}$. An MLP then predicts the new node's role and its connections to all denoised nodes simultaneously. Crucially, connections are inferred jointly using a "mixture of multinomial distributions" (rather than autoregressively), reducing generation steps to $\mathcal{O}(N)$ and significantly improving efficiency.
    - **Design Motivation**: Query conditioning ensures task-adaptivity; joint edge prediction avoids the $\mathcal{O}(N^2)$ steps of ARG-Designer; the effective size bias implicitly favors low redundancy in every generation step.

### Loss & Training

The denoising network is trained using a weighted NLL loss $\nabla_\theta \mathcal{G} = \sum_{m,t} \sum_{k \in \pi(\leq t)} w_k^m \nabla \log p_\theta(\mathcal{G}_{v_k}^{\pi(>t)} | \mathcal{G}_{t+1}^m, \mathcal{Q})$, where $w_k^m$ represent probability weights from the ordering network. Since the ordering network's output is discrete, it is trained via REINFORCE with a reward set to the negative NLL: $R^m = -\sum_t \sum_k w_k^m \log p_\theta(\cdot)$.

Additionally, a task-utility policy gradient term $\nabla_\theta \mathbb{E}[\mathcal{G}] \approx \frac{1}{\mathcal{B}} \sum_k u(\mathcal{G}^{(k)}(\mathcal{Q})) \nabla_\theta \log p_\theta(\mathcal{G}^{(k)} | \mathcal{Q})$ is included, using task accuracy as a black-box reward. In practice, utility is only evaluated periodically on a subset of generated graphs to save API costs.

## Key Experimental Results

### Main Results

Compared 20+ baselines across 6 benchmarks (MMLU, GSM8K, MultiArith, SVAMP, AQuA, HumanEval) using gpt-4o-mini as the base LLM and 5 agents.

| Method | MMLU | GSM8K | HumanEval | Average |
| :--- | :---: | :---: | :---: | :---: |
| Vanilla (Single Agent) | 78.54 | 87.45 | 87.08 | 85.92 |
| LLM-Debate | 80.56 | 89.47 | 88.68 | 87.46 |
| AgentPrune | 82.40 | 91.92 | 87.17 | 88.22 |
| MaAS | 82.32 | 91.13 | 89.57 | 88.50 |
| ARG-Designer | 79.10 | 91.25 | 89.19 | 88.57 |
| **RADAR** | **83.66** | **92.51** | **91.28** | **90.32** |

RADAR outperforms the strongest learning-based baseline (ARG-Designer) by an average of 1.75% and single agents by 1.96%~6.59%.

### Ablation Study

| Configuration | MMLU | GSM8K | MultiArith | Description |
| :--- | :---: | :---: | :---: | :--- |
| Full RADAR | 83.66 | 92.51 | 98.81 | Full model |
| w/o ES | 81.05 | 91.22 | 98.31 | Removed effective size from both ordering and denoising |
| w/o utility | 82.96 | 92.02 | 98.47 | Removed task-utility policy gradient |
| w/o query | 79.08 | 91.82 | 97.81 | Denoising ignores query (largest drop) |
| non-diffusion | 79.10 | 91.25 | 98.55 | Replaced with ARG-Designer style autoregression |

### Key Findings
- **Query conditioning** has the largest impact (MMLU drops 4.58), indicating task-adaptivity is the core gain source.
- **Effective size** systematically impacts performance; its removal drops MMLU by 2.61, validating the effectiveness of redundancy awareness.
- **Token consumption**: On GSM8K, RADAR uses $4.2 \times 10^6$ tokens, half of G-Designer's usage with higher accuracy. Compared to AFlow's $1.4 \times 10^7$ and AgentPrune's $1.1 \times 10^7$, RADAR's $6.5 \times 10^6$ demonstrates significant token economy.
- **Robustness**: Injecting "liar prompt attacks" into 2/5 agents on MMLU caused a 4.47% drop for complete graphs and 1.05% for ARG-Designer, while RADAR barely dropped.
- **Graph Statistics**: RADAR's effective size (0.92) is much higher than G-Designer (0.73) and ARG-Designer (0.68), while its density (0.289) is slightly lower, indicating "fewer but better" connections.
- **Transferability**: Trained on gpt-4o-mini, RADAR transfers well to DeepSeek-R1 / Qwen3-32B. On DeepSeek-R1, single agent score is 90.81 vs. RADAR's 92.16.

## Highlights & Insights
- **Applying social network effective size to LLM multi-agent communication graphs** is a clean interdisciplinary adaptation. Burt's 1992 concept for "structural holes" in human networks is seamlessly migrated to agent networks as a differentiable guidance signal.
- **Iterative graph diffusion instead of one-step generation** marks a critical paradigm shift relative to G-Designer / MaAS / ARG-Designer, allowing the model to "reflect on redundancy during generation."
- **Joint edge prediction (mixture of multinomial)** reduces generation steps from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$, a significant improvement over ARG-Designer's autoregressive scheme, introducing minimal overhead compared to single workflows while remaining query-adaptive.
- Halving token costs while achieving peak accuracy strongly suggests "complex topology $\neq$ good topology"; structural sparsification is a first-class concern in the LLM multi-agent era.

## Limitations & Future Work
- Training requires "baseline topologies + performance" as an initial dataset, entailing a non-trivial startup cost. New tasks require 50 sample queries run on baselines for initialization.
- Every inference query requires full multi-step denoising. Compared to AFlow's static learned workflow, this presents a latency disadvantage (17.55min vs 7.32min on GSM8K).
- Experiments were limited to 5 agents; the scaling behavior for $N \gg 5$ is unexplored, and the $\mathcal{O}(N^2)$ calculation of effective size may become a bottleneck.
- Agent roles are selected from a fixed candidate pool. Dynamic role generation is deferred to future work.
- Effective size assumes discrete role categories, which might not apply to continuous prompt-based roles.

## Related Work & Insights
- **vs ARG-Designer**: Both are generative topology designs, but ARG-Designer uses autoregressive edge generation. RADAR uses diffusion for joint edge generation and explicit redundancy modeling, leading to superior efficiency and quality.
- **vs GTD**: GTD also uses conditional discrete graph diffusion but lacks structural metrics like effective size for guidance, potentially lacking clear goals for redundancy control.
- **vs AgentPrune**: AgentPrune performs post-hoc pruning on fixed topologies. RADAR generates from scratch; the former is capped by the initial topology, while the latter can produce entirely new structures.
- **vs MaAS**: MaAS learns a continuous architecture distribution for sampling but remains essentially one-step. RADAR's step-by-step approach enables fine-grained exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing effective size to graph diffusion is an elegant cross-domain idea, though the underlying framework draws from Kong et al.'s discrete graph diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets + 20+ baselines + token economy + robustness + transferability + ablation; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Reasonable formula density and clear diagrams; however, details of the ordering and denoising networks are fast-paced and require appendix consultation.
- Value: ⭐⭐⭐⭐ Token cost and robustness are genuine pain points in multi-agent LLM research; RADAR provides a practical and reproducible solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](../../ACL2026/multi_agent/bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](../../ACL2026/multi_agent/cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](../../ACL2026/multi_agent/posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](../../ACL2026/multi_agent/roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[ACL 2026\] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](../../ACL2026/multi_agent/a_multi-agent_framework_for_feature-constrained_difficulty_control_in_reading_co.md)

</div>

<!-- RELATED:END -->
