---
title: >-
  [Paper Note] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning
description: >-
  [ACL 2026][Graph Learning][Reinforcement Learning] AgentGL is proposed as the first Reinforcement Learning-based Agentic Graph Learning (AGL) framework. It enables LLM agents to autonomously navigate Text-Attributed Grap…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Reinforcement Learning"
  - "Agentic Navigation"
  - "Text-Attributed Graphs"
  - "Tool Use"
date: 2026-05-08
content_hash: 791e6bacfe670b18
---

# AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2604.05846](https://arxiv.org/abs/2604.05846)  
**Code**: [https://github.com/sunyuanfu/AgentGL](https://github.com/sunyuanfu/AgentGL)  
**Area**: Graph Learning / LLM Agent  
**Keywords**: Graph Learning, Reinforcement Learning, Agentic Navigation, Text-Attributed Graphs, Tool Use

## TL;DR
AgentGL is proposed as the first Reinforcement Learning-based Agentic Graph Learning (AGL) framework. It enables LLM agents to autonomously navigate Text-Attributed Graphs (TAGs) using graph-native search tools, achieving absolute accuracy improvements of up to 17.5% in node classification and 28.4% in link prediction.

## Background & Motivation

**Background**: LLMs increasingly rely on agentic capabilities (iterative retrieval, tool calling, decision reasoning) to overcome the limitations of static parametric knowledge. However, existing agentic frameworks primarily process unstructured text and fail to exploit topological dependencies in real-world data.

**Limitations of Prior Work**: Traditional GNNs model structural signals but struggle with rich textual semantics; GraphLLMs (e.g., GraphGPT, GraphICL) rely on static graph contexts and lack adaptive exploration during inference; GraphRAG-constructed knowledge graphs are costly and do not preserve original topological associations. All three categories lack dynamic evidence acquisition mechanisms on real graph structures.

**Key Challenge**: Evidence on graphs is multi-scale—some clues exist in immediate local neighborhoods, while others emerge only in broader structural patterns. Agents must decide "where to go next" in a combinatorial space while avoiding redundant or uninformative regions. Furthermore, effective graph reasoning requires multi-step exploration, but authentic search trajectory annotations are extremely scarce.

**Goal**: Propose a new Agentic Graph Learning (AGL) paradigm that allows LLM agents to autonomously navigate graph structures, accumulate structural evidence, and iteratively adjust search trajectories based on real-time reasoning.

**Key Insight**: Redefine graph learning as an alternating process of topology-aware navigation and LLM reasoning, rather than static feature encoding or one-time retrieval.

**Core Idea**: Drive LLM agents to learn graph-native search strategies via reinforcement learning, suppressing over-retrieval through Search-Constrained Thinking and stabilizing long-horizon policy optimization through Graph-Conditioned Curriculum Learning.

## Method

### Overall Architecture
AgentGL models graph learning as an agent decision process: given a target node/node pair and a query, the LLM agent iteratively acquires evidence through graph-native search tools and eventually outputs a prediction. Training follows two stages: (1) Graph-native policy bootstrapping—learning basic navigation behaviors; (2) Search efficiency optimization—reducing redundant tool calls via Search-Constrained Thinking. Both stages are conducted under a Graph-Conditioned Curriculum Learning strategy.

### Key Designs

1.  **Graph-Native Search Toolset**:
    - **Function**: Provide multi-scale graph structure exploration capabilities.
    - **Mechanism**: Four complementary tools cover "local vs. global" and "structural vs. semantic" dimensions: $\tau_{1hop}$ (1-hop neighborhood search, integrating common-neighbor priority and exclusive-neighbor balanced allocation), $\tau_{2hop}$ (2-hop neighborhood search), $\tau_{ss}$ (Structural Saliency Search, retrieving global topological hubs based on PPR scores), and $\tau_{dense}$ (Graph Dense Search, bridging semantically related but topologically disconnected nodes using cosine similarity).
    - **Design Motivation**: Ensure LLMs can navigate graph structures as flexibly as they navigate text, covering everything from local structures to global semantics.

2.  **Search-Constrained Thinking**:
    - **Function**: Suppress over-retrieval and promote deep reasoning.
    - **Mechanism**: Three components—Backtracking Termination Trigger (injecting a cognitive interruption after each tool execution to force assessment of evidence sufficiency), Cognitive Density Regularization (penalizing sparse reasoning segments $r_{depth} = \alpha \cdot \mathbb{I}[N_{short}=0] - \lambda_d \cdot N_{short}$), and Adaptive Reward Transition (discarding coverage incentives to focus on accuracy and reasoning density).
    - **Design Motivation**: Resolve the inefficiency of the default exhaustive retrieval in the bootstrapping stage to achieve "more thinking, less searching."

3.  **Graph-Conditioned Curriculum Learning (GCCL)**:
    - **Function**: Stabilize training and accelerate convergence.
    - **Mechanism**: Utilize intrinsic graph properties to quantify sample difficulty at zero cost. Node classification uses Wilson lower bound corrected homophily estimation and degree priors; link prediction uses semantic similarity and label consistency. Training progresses from easy to difficult.
    - **Design Motivation**: Graphs naturally provide quantifiable difficulty priors, avoiding the bottleneck of manual labeling required by traditional curriculum learning.

### Loss & Training
Stage 1: $R(\tau) = r_{fmt} + r_{acc} + r_{cov}$ (format + accuracy + tool coverage), optimized using GRPO or REINFORCE++. Stage 2: $R(\tau) = r_{fmt} + r_{acc} + r_{depth}$, where coverage incentives are replaced with cognitive density rewards.

## Key Experimental Results

### Main Results

| Task | Dataset | AgentGL | Prev. SOTA | Gain |
|------|--------|---------|---------|------|
| Node Classification | OGB-Arxiv | 66.3 | 54.1 (GraphPrompter) | +12.2 |
| Node Classification | PubMed | 74.5 | 67.0 (GraphPrompter) | +7.5 |
| Link Prediction | OGB-Arxiv | 91.5 | 79.8 (LLaGA) | +11.7 |
| Link Prediction | PubMed | 75.8 | 62.5 (GraphICL) | +13.3 |
| Zero-shot Transfer (NC) | Arxiv-23 | 63.6 | 52.2 (GraphICL) | +11.4 |
| Zero-shot Transfer (LP) | Reddit | 83.2 | 62.0 (GraphICL) | +21.2 |

### Ablation Study

| Configuration | Description |
|------|------|
| Full AgentGL | Complete model, optimal performance |
| w/o GCCL | Curriculum learning removed, training unstable, performance drops |
| w/o Search-Constrained Thinking | Over-retrieval occurs but maintains basic capability |
| w/o Global Tools | Only local tools, structural field of view limited, significant drop |

### Key Findings
- Significant outperformance of GNN, GraphLLM, and GraphRAG baselines across all 7 datasets.
- Gains are particularly pronounced in zero-shot transfer scenarios (Reddit LP +21.2%), indicating strong generalization of the learned search strategies.
- Search-Constrained Thinking significantly reduces tool call frequency while maintaining or even improving accuracy.

## Highlights & Insights
- **The AGL paradigm is the core contribution**: Redefining graph learning from "static encoding" to "interactive navigation + reasoning" opens new directions for LLM applications on structured data.
- **Zero-cost Curriculum Learning**: Leveraging intrinsic graph properties to automatically quantify difficulty avoids bottlenecks of manual labeling or pilot runs.
- **Transferability of Search-Constrained Thinking**: The "think more, search less" design can be applied to any tool-augmented LLM scenario.

## Limitations & Future Work
- Validated only on node classification and link prediction; community detection, graph classification, etc., are not yet addressed.
- Graph-native tools are manually designed; future work could allow agents to autonomously discover or combine new tools.
- High training costs (multiple RL rounds); scalability on large-scale graphs remains to be verified.

## Related Work & Insights
- **vs GraphRAG (HippoRAG2)**: GraphRAG requires knowledge graph reconstruction and does not preserve original topology; AgentGL navigates the original graph directly.
- **vs GraphCoT**: Relies on heuristic prompting and focuses only on graph QA; AgentGL optimizes search strategies end-to-end via RL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work combining AGL + RL, pioneering a new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 datasets + multiple backbones, though ablations could be more extensive.
- Writing Quality: ⭐⭐⭐⭐ Clear framework, rigorous formulas.
- Value: ⭐⭐⭐⭐⭐ The AGL paradigm has great potential to drive the deep fusion of graph learning and LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](../../ICML2026/graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ICML 2026\] Aitchison Embeddings for Learning Compositional Graph Representations](../../ICML2026/graph_learning/aitchison_embeddings_for_learning_compositional_graph_representations.md)

</div>

<!-- RELATED:END -->
