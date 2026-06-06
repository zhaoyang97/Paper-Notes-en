---
title: >-
  [Paper Note] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation
description: >-
  [ICML 2026][Multi-Agent][Multi-agent collaboration] RADAR models the communication topology design of multi-LLM-Agent systems as a "redundancy-aware" discrete graph diffusion process…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-agent collaboration"
  - "graph diffusion"
  - "communication topology"
  - "effective size"
  - "redundancy awareness"
date: 2026-05-08
content_hash: d92a7122f7a7b0b6
---

# RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation

**Conference**: ICML 2026  
**arXiv**: [2605.09907](https://arxiv.org/abs/2605.09907)  
**Code**: https://github.com/cszhangzhen/RADAR  
**Area**: Multi-Agent Systems / Graph Diffusion Models / LLM Agent  
**Keywords**: Multi-agent collaboration, graph diffusion, communication topology, effective size, redundancy awareness

## TL;DR
RADAR models the communication topology design of multi-LLM-Agent systems as a "redundancy-aware" discrete graph diffusion process, using effective size as a guiding signal to incrementally generate query-adaptive collaboration graphs. It achieves higher accuracy, lower token consumption, and stronger robustness across six benchmarks.

## Background & Motivation

**Background**: LLM-Agent multi-agent systems (LLM-Debate, MetaGPT, AutoGen, etc.) have been shown to outperform single agents, but their key bottleneck lies in "communication topology"—who talks to whom, and in what order. Early methods used manually fixed structures such as chain/star/tree/fully-connected; recent works (GPTSwarm, G-Designer, MaAS, ARG-Designer, GTD) have shifted towards "automatic topology design."

**Limitations of Prior Work**: Automated approaches fall into three categories, each with drawbacks. (1) Agentic profiling (meta-agent coordination) suffers from single-point bottlenecks; (2) Search-based (heuristic search over topology space) is computationally expensive and not scalable; (3) Graph learning (e.g., VAE to predict the entire graph at once) generates at too coarse a granularity, missing fine-grained dependencies. Moreover, as structures become more complex, token consumption skyrockets—cited data shows complex topologies can consume $2 \sim 11.8\times$ more tokens than chains. AgentPrune/Wang et al. attempt pruning to mitigate this, but only perform local modifications on fixed agent sets—these are "post hoc" patches and cannot design from scratch under efficiency constraints.

**Key Challenge**: The contradiction between expressiveness (topology must be complex enough for hard tasks) and efficiency (token usage must not explode). Existing methods either sacrifice one or treat the two objectives as independent subproblems.

**Goal**: Explicitly model "redundancy" during communication graph generation, so structure formation and redundancy control proceed jointly; also support query-adaptive generation—sparse structures for simple tasks, dense for hard ones.

**Key Insight**: The authors borrow the concept of effective size from social network analysis (Burt 1992)—the proportion of non-redundant information in a node's ego network. If two neighbors are themselves highly connected, their information overlaps and effective size is low. Incorporating this into the graph diffusion process provides a natural "redundancy metric" as a guiding signal for generation.

**Core Idea**: Reformulate multi-agent communication topology design as a "effective size-guided + query-conditioned" discrete graph diffusion problem, incrementally denoising from an empty graph to the final topology.

## Method

### Overall Architecture

Input: task query $\mathcal{Q}$, candidate agent set (with Role/State/Plugin).  
Output: a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $A_{ij} = 1$ indicates agent $v_i$ sends information to $v_j$; agents are activated in the topological order of $\mathcal{G}$, and the final answer is produced by an Aggregate function (majority vote/concatenation/last agent output).

The training pipeline first runs 50 training queries with various baseline topologies (fully connected/mesh/star/layered/random, with 3 or 4 agents) to obtain "topology→task performance" samples, which are used to train the graph diffusion model. During inference, given a new query, the denoising network starts from an empty graph and iteratively denoises, ultimately producing a query-tailored collaboration topology.

### Key Designs

1. **Effective Size as Redundancy Metric**:

    - **Function**: Computes a scalar for each agent in the current graph, reflecting the non-redundancy of its in/out neighbors, serving as a guiding signal for the subsequent diffusion process.
    - **Mechanism**: The in-effective size is defined as $\varphi^i(v_k) = |\mathcal{N}_i(v_k)| - \frac{\sum_{j,q \in \mathcal{N}_i(v_k)} A_{jq} \mathbb{I}[r(j) = r(q)]}{|\mathcal{N}_i(v_k)|}$, where the numerator is the number of in-neighbors and the denominator penalizes redundant neighbor pairs that are both connected and share the same role. The out-effective size $\varphi^o(v_k)$ is defined symmetrically, and the final effective size is a weighted combination $\varphi(v_k) = (1-\beta) \varphi^i(v_k) + \beta \varphi^o(v_k)$. High $\varphi$ means the agent receives diverse inputs and distributes information along non-overlapping paths.
    - **Design Motivation**: Previous methods could only use black-box signals like task accuracy, which are sparse and delayed. Effective size is a local, structurally differentiable, and directly redundancy-related geometric quantity, providing fine-grained structural guidance at each diffusion step.

2. **Redundancy-Aware Forward Diffusion (with Ordering Network)**:

    - **Function**: Sequentially masks nodes and their edges in the training graph $\mathcal{G}_0$ according to a meaningful order, producing a series of partially masked intermediate graphs $\mathcal{G}_1, \mathcal{G}_2, \dots$, which serve as supervision for reverse denoising training.
    - **Mechanism**: Defines an ordering network $q_\psi(\pi | \mathcal{G}_0, \varphi) = \prod_t q_\psi(\pi_t | \mathcal{G}_0, \varphi, \pi_{(<t)})$, sampling which node to mask at each step. Specifically, node embeddings $h_t$ are obtained via GNN+positional encoding, and $q_\psi(\pi_t | \cdot) \propto \exp(h_t + \varphi(v_t))$ is used for sampling—nodes with higher effective size are masked earlier. The intuition is that graphs with high effective size can be decomposed into weakly overlapping substructures, facilitating incremental generation learning.
    - **Design Motivation**: General graph diffusion (Kong et al., Chen et al.) uses random or fixed masking orders, losing structural regularity. Effective size-guided ordering makes the reverse denoising task more "structured"—simple substructures are restored first, complex dependencies last.

3. **Conditional Reverse Denoising Network**:

    - **Function**: Starting from an empty graph and given query $\mathcal{Q}$, incrementally restores nodes and their connections to already generated nodes, ultimately yielding a task-adaptive topology.
    - **Mechanism**: The denoising network $p_\theta(\mathcal{G}_t | \mathcal{G}_{t+1}, \mathcal{Q})$ uses GAT-style attention at each layer: $\alpha_{i,j} = \frac{\exp(\text{ReLU}(\mathbf{a}^\top [\mathbf{W h}_i^l \| \mathbf{W h}_j^l]))}{\sum_k \exp(\cdot)}$ to obtain node embeddings. The last layer adds an effective size bias $\mathbf{h}_i^L \leftarrow \mathbf{h}_i^L + \varphi(v_i) \mathbf{1}$. An MLP then jointly predicts the new node's role and its connections to all previously denoised nodes—crucially, connections are inferred via a "mixture of multinomial distributions" (not autoregressive), reducing generation steps to $\mathcal{O}(N)$ and greatly improving efficiency.
    - **Design Motivation**: Query conditioning ensures task adaptivity; joint edge prediction (vs. edge-wise autoregression) avoids the $\mathcal{O}(N^2)$ steps of ARG-Designer; effective size bias implicitly favors low redundancy at each generation step.

### Loss & Training

The denoising network is trained with a weighted NLL loss: $\nabla_\theta \mathcal{G} = \sum_{m,t} \sum_{k \in \pi(\leq t)} w_k^m \nabla \log p_\theta(\mathcal{G}_{v_k}^{\pi(>t)} | \mathcal{G}_{t+1}^m, \mathcal{Q})$, where $w_k^m$ are probability weights from the ordering network. The ordering network, due to its discrete output, is trained with REINFORCE, using negative NLL as the reward: $R^m = -\sum_t \sum_k w_k^m \log p_\theta(\cdot)$.

Additionally, there is a task-utility policy gradient term: $\nabla_\theta \mathbb{E}[\mathcal{G}] \approx \frac{1}{\mathcal{B}} \sum_k u(\mathcal{G}^{(k)}(\mathcal{Q})) \nabla_\theta \log p_\theta(\mathcal{G}^{(k)} | \mathcal{Q})$, directly using task accuracy as a black-box reward. In practice, utility is periodically evaluated on a subset of generated graphs to save API costs.

## Key Experimental Results

### Main Results

On six benchmarks (MMLU, GSM8K, MultiArith, SVAMP, AQuA, HumanEval), RADAR is compared with 20+ baselines, all using gpt-4o-mini as the base LLM and 5 agents.

| Method | MMLU | GSM8K | HumanEval | Avg |
|--------|------|-------|-----------|-----|
| Vanilla (single agent) | 78.54 | 87.45 | 87.08 | 85.92 |
| LLM-Debate | 80.56 | 89.47 | 88.68 | 87.46 |
| AgentPrune | 82.40 | 91.92 | 87.17 | 88.22 |
| MaAS | 82.32 | 91.13 | 89.57 | 88.50 |
| ARG-Designer | 79.10 | 91.25 | 89.19 | 88.57 |
| **RADAR** | **83.66** | **92.51** | **91.28** | **90.32** |

RADAR outperforms the strongest learning-based baseline (ARG-Designer) by an average of 1.75%, and surpasses the single agent by 1.96%~6.59%.

### Ablation Study

| Configuration | MMLU | GSM8K | MultiArith | Notes |
|---------------|------|-------|------------|-------|
| Full RADAR | 83.66 | 92.51 | 98.81 | Full model |
| w/o ES | 81.05 | 91.22 | 98.31 | Removes effective size from both ordering and denoising |
| w/o utility | 82.96 | 92.02 | 98.47 | Removes task-utility policy gradient |
| w/o query | 79.08 | 91.82 | 97.81 | Denoising ignores query, largest drop |
| non-diffusion | 79.10 | 91.25 | 98.55 | Replaces with ARG-Designer-style autoregression |

### Key Findings
- Query conditioning has the largest impact (MMLU drops 4.58), indicating task adaptivity is the main source of improvement.
- Effective size consistently improves performance; removing it drops MMLU by 2.61, validating the effectiveness of redundancy awareness.
- Token consumption: On GSM8K, RADAR uses only $4.2 \times 10^6$ tokens, half of G-Designer, with higher accuracy; compared to AFlow's $1.4 \times 10^7$ and AgentPrune's $1.1 \times 10^7$, RADAR uses $6.5 \times 10^6$, demonstrating significant token economy.
- Robustness: On MMLU, injecting "liar prompt attacks" into 2/5 agents causes complete graph to drop 4.47%, ARG-Designer 1.05%, RADAR almost unaffected.
- Generated graph statistics: RADAR's effective size is 0.92, much higher than G-Designer (0.73) and ARG-Designer (0.68), with density 0.289 slightly lower, indicating "fewer but better" connections.
- Strong cross-base LLM transferability: Trained on gpt-4o-mini, deployable to DeepSeek-R1 / Qwen3-32B; on DeepSeek-R1, single agent scores 90.81, RADAR 92.16.

## Highlights & Insights
- **Transplanting effective size from social networks to LLM multi-agent communication graphs**—a very clean interdisciplinary borrowing. Burt 1992's classic concept, originally for analyzing "structural holes" in human social networks, is seamlessly transferred to agent networks as a differentiable guiding signal.
- **Iterative graph diffusion instead of one-step generation** is a key paradigm shift over G-Designer / MaAS / ARG-Designer, enabling the model to "reflect on redundancy while generating," rather than a one-shot process.
- Joint edge prediction (mixture of multinomials) reduces generation steps from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$, a practical improvement over ARG-Designer's autoregressive scheme; inference incurs only slightly more overhead than single workflow methods but remains query-adaptive.
- Token cost is halved while accuracy is highest, strongly demonstrating that "complex topology ≠ good topology"; structural sparsification is a first-class concern for multi-agent LLMs.

## Limitations & Future Work
- Training requires an initial dataset of "baseline topologies + task performance," resulting in nontrivial startup cost; new tasks require resampling 50 queries and running baselines for initialization.
- Inference requires running the full multi-step denoising process for each query, making single-query latency less favorable than methods like AFlow ("one-shot workflow learning") (17.55min vs 7.32min on GSM8K).
- Experiments are conducted with only 5 agents; scaling behavior for $N \gg 5$ is unexplored, and the $\mathcal{O}(N^2)$ computation of effective size may become a bottleneck.
- Agent roles are selected from a fixed candidate pool; dynamic role generation is not considered and is left as future work.
- Effective size assumes discrete role categories, making it less applicable to continuous prompt-based roles.

## Related Work & Insights
- **vs ARG-Designer**: Both are generative topology design methods, but ARG-Designer generates edges autoregressively, while RADAR uses a diffusion process to jointly generate edges and explicitly model redundancy, achieving better efficiency and quality.
- **vs GTD**: GTD also uses conditional discrete graph diffusion, but lacks structural metrics like effective size as guidance, so its generated graphs may lack explicit redundancy control.
- **vs AgentPrune**: AgentPrune is post-hoc pruning (removing low-contribution edges from a fixed topology), while RADAR generates from scratch; the former is limited by the initial topology, the latter can produce entirely new structures.
- **vs MaAS**: MaAS learns a continuous architecture distribution for sampling, but is essentially one-shot generation; RADAR's step-by-step approach enables fine-grained exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing effective size into graph diffusion is an elegant cross-domain idea, though the underlying framework is adapted from Kong et al.'s discrete graph diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets + 20+ baselines + token economy + robustness + cross-LLM transfer + ablation, very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Formula density is reasonable, illustrations are clear; however, details of the ordering and denoising networks are somewhat brief and require consulting the appendix.
- Value: ⭐⭐⭐⭐ In the multi-agent LLM direction, token cost and robustness are real pain points; RADAR's solution is practical and easy to reproduce.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](../../ACL2026/multi_agent/bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](../../ACL2026/multi_agent/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](../../ACL2026/multi_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)

</div>

<!-- RELATED:END -->
