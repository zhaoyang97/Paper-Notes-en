---
title: >-
  [Paper Note] ResearchTown: Simulator of Human Research Community
description: >-
  [ICML 2025][Multi-Agent][Multi-Agent Simulation] This paper proposes ResearchTown, a multi-agent framework based on agent-data graphs and TextGNN (text-space message passing), which models human scientific communities as heterogeneous graphs to unify the simulation of three core research activities: literature reading, paper writing, and peer review. A scalable and objective simulation quality evaluation is conducted via a node masking prediction task (ResearchBench).
tags:
  - "ICML 2025"
  - "Multi-Agent"
  - "Multi-Agent Simulation"
  - "Graph Neural Networks"
  - "Scientific Community Simulation"
  - "Text-Space Message Passing"
  - "Autonomous Research"
date: 2026-05-08
content_hash: 6733422e6ff2d372
---

# ResearchTown: Simulator of Human Research Community

**Conference**: ICML 2025  
**arXiv**: [2412.17767](https://arxiv.org/abs/2412.17767)  
**Code**: [ulab-uiuc/research-town](https://github.com/ulab-uiuc/research-town)  
**Area**: LLM Evaluation  
**Keywords**: Multi-Agent Simulation, Graph Neural Networks, Scientific Community Simulation, Text-Space Message Passing, Autonomous Research

## TL;DR

This paper proposes ResearchTown, a multi-agent framework based on agent-data graphs and TextGNN (text-space message passing), which models human scientific communities as heterogeneous graphs to unify the simulation of three core research activities: literature reading, paper writing, and peer review. A scalable and objective simulation quality evaluation is conducted via a node masking prediction task (ResearchBench).

## Background & Motivation

**Core Problem**: Can LLMs simulate human scientific communities? Answering this question carries two key implications: (1) understanding the discovery process behind established research ideas; and (2) democratizing and accelerating the discovery of new research ideas.

**Limitations of Prior Work**:

- Existing multi-agent frameworks (for social simulation, game simulation, etc.) cannot handle the complex collaborative activities in research communities, such as multi-author collaboration in paper writing and peer review.
- Prior works in autonomous research are limited to single tasks (e.g., idea generation, coding experiments) or focus solely on a single-agent workflow.
- They fail to simulate collaboration among researchers with diverse backgrounds—the fundamental mode of modern scientific research.

**Key Insight**: Deeply interconnected research communities can naturally be represented as graph structures (e.g., citation networks, academic social networks). Incorporating LLMs can extend traditional prediction and analysis to dynamic simulation and real-time prediction.

## Method

### Overall Architecture

The core architecture of ResearchTown consists of three layers:

1. **Agent-Data Graph**: A novel heterogeneous graph comprised of two types of nodes—agent nodes (researchers) and data nodes (papers)—and three types of edges—agent-agent ($\mathcal{E}_{aa}$), agent-data ($\mathcal{E}_{ad}$), and data-data ($\mathcal{E}_{dd}$).
2. **TextGNN**: A reasoning framework that performs text-space message passing over the agent-data graph.
3. **ResearchBench**: An evaluation benchmark based on node-masking prediction tasks.

The entire simulation workflow is structured as a 2-layer GNN: the first layer performs paper reading (information aggregation), and the second layer manages paper writing and peer review (generating final outputs).

### Key Designs

#### 1. Definition of Agent-Data Graph

The uniqueness of the agent-data graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ lies in:

- **Data nodes**: Carry textual attributes $\mathbf{x}_v$ (such as full-text papers).
- **Agent nodes**: Carry agent functions $f_u(\cdot)$ (i.e., LLMs configured with specific prompts) rather than embedding vectors.
- Agent nodes are inherently functions operating on data nodes: $\mathbf{x}_{uv} = f_u([\mathbf{x}_u, \mathbf{x}_v])$.

In the scientific community graph, this is instantiated as: researchers as agent nodes and papers as data nodes; edges include citation relations ($\mathcal{E}_{dd}$), authorship, and reviewing relationships ($\mathcal{E}_{ad}$); $\mathcal{E}_{aa}$ is omitted (as it can be inferred via 2-hop paths).

#### 2. TextGNN Message Passing Mechanism

The key difference between TextGNN and a standard GNN is that all hidden states are defined in the **text space** $\Sigma^*$ rather than the **embedding space** $\mathbb{R}^d$.

The $k$-th update layer for agent node $u$:

$$\mathbf{h}_u^{(k)} = f_u\Big([\mathbf{h}_u^{(k-1)}, \{f_a([\mathbf{h}_a^{(k-1)}, \mathbf{h}_u^{(k-1)}, \mathbf{h}_d^{(k-1)}]) \mid (u,a) \in \mathcal{E}_{aa}, (u,d) \in \mathcal{E}_{ad}\}]\Big)$$

The $k$-th update layer for data node $v$ utilizes a global agent function $f_g(\cdot)$ (with no specific profile) to aggregate messages from neighboring agent and data nodes.

#### 3. Three-Stage Simulation Workflow

**Stage 1 — Paper Reading**: Aggregates neighborhood paper information to generate researcher profiles for newly added agent nodes:

$$\mathbf{h}_u = f_u\Big([\{\mathbf{h}_d \mid (u,d) \in \mathcal{E}_{ad}\}]\Big)$$

This serves as a special form of message passing, where agent nodes start empty and generate profiles after reading papers.

**Stage 2 — Paper Writing**: Synthesizes paper content for newly added data nodes through collaboration among multiple agents:

$$\mathbf{h}_v = f_g\Big([\{f_a([\mathbf{h}_a, \mathbf{h}_d]) \mid (v,a) \in \mathcal{E}_{ad}, (v,d) \in \mathcal{E}_{dd}\}]\Big)$$

Each author agent generates message updates based on their profile and reference papers, which are then aggregated by a global function.

**Stage 3 — Peer Review**: Reviewer agents generate review comments based on the paper context, their professional background, and related literature:

$$\mathbf{r}_v = f_g\Big([\mathbf{h}_v, \{f_a([\mathbf{h}_a, \mathbf{h}_v, \mathbf{h}_d]) \mid (v,a) \in \mathcal{E}_{ad}, (v,d) \in \mathcal{E}_{dd}\}]\Big)$$

Unlike the previous stages, reviewers are not the authors, and the paper node already contains content at this step.

#### 4. ResearchBench Evaluation Framework

The core paradigm of the evaluation is **node masking prediction**:

- Mask the content of a certain paper node $\mathbf{h}_v^*$ in the community graph.
- Reconstruct the node using ResearchTown from neighborhood information.
- Calculate the cosine similarity between the reconstructed result and the ground truth using text-embedding-3-large.

The benchmark comprises 1,000 paper writing tasks and 200 review tasks from NeurIPS 2024 and ICLR 2024.

### Loss & Training

Rather than training models, this work leverages existing LLMs (GPT-4o-mini) as backbones for the agent functions, with the temperature set to 0 to ensure reproducibility. Evaluation relies on embedding similarity metrics rather than gradient optimization.

Four aggregation strategies are employed for ablation analysis:

- **AGG-self**: Target node itself only.
- **AGG-agent**: Target node + neighboring agent nodes.
- **AGG-data**: Target node + neighboring data nodes.
- **AGG-global** (i.e., ResearchTown): Target node + all neighbors.

## Key Experimental Results

### Main Results

**Paper Writing Simulation** (text-embedding-3-large similarity × 100):

| Aggregation Strategy | Easy | Medium | Hard | Overall |
|---------|------|--------|------|---------|
| AGG-self | 46.42 | 45.92 | 45.90 | 46.08 |
| AGG-agent | 56.90 | 55.55 | 53.26 | 55.24 |
| AGG-data | 74.36 | 66.42 | 56.02 | 65.30 |
| AGG-global (ResearchTown) | **73.79** | **67.85** | **60.89** | **67.51** |

**Peer Review Simulation** (Strength/Weakness indicate embedding similarity; $\Delta$S denotes rating discrepancy):

| Aggregation Strategy | Strength ↑ | Weakness ↑ | $\Delta$S ↓ | Average Score |
|---------|-----------|-----------|------|--------|
| AGG-self | 51.23 | 47.16 | 1.27 | 5.33 |
| AGG-agent | 51.66 | 46.75 | 1.19 | 5.40 |
| AGG-data | 51.45 | 47.62 | 1.26 | 5.30 |
| AGG-global | 51.51 | 47.17 | 1.55 | 5.00 |

### Ablation Study

**Comparison of Different LLM Backbones**:

| Aggregation Strategy | Paper Writing (Qwen/GPT/DS) | Peer Review $\Delta$S (Qwen/GPT/DS) |
|---------|----------------------|---------------------|
| AGG-self | 46.45 / 46.08 / 48.62 | 1.36 / 1.27 / 1.11 |
| AGG-agent | 53.91 / 55.24 / 56.19 | 1.41 / 1.19 / 1.05 |
| AGG-data | 65.03 / 65.30 / 65.05 | 1.28 / 1.26 / 1.07 |
| AGG-global | **65.30** / **67.51** / **65.33** | **0.79** / 1.51 / **0.81** |

**Novelty and Feasibility Evaluation** (using a scale of 0-10):

| Evaluation Method | Sim-Novelty | Sim-Feasibility | Real-Novelty | Real-Feasibility |
|---------|-----------|-----------|-----------|-----------|
| LLM Eval | 7.39 | 6.82 | 7.85 | 7.13 |
| Human Eval | 5.50 | 7.98 | 5.90 | 7.85 |

### Key Findings

1. **Reference papers are more vital than author profiles**: AGG-data (65.30) significantly outperforms AGG-agent (55.24), demonstrating that bibliography is the core information source in paper writing.
2. **Multi-researcher collaboration improves performance on difficult tasks**: On the Hard subset, AGG-global (60.89) considerably exceeds AGG-data (56.02) by 4.87 points, demonstrating that researchers provide multi-hop bibliography information.
3. **Simulating peer review is more challenging than paper writing**: Review similarities (around 47–51) are far lower than those of paper writing (around 67), as peer review data is inherently noisier and more diverse.
4. **Increasing agent count continuously elevates simulation quality**: The improvement is most pronounced when scaling from 1 to 2 agents (49.0 → 52.7), after which diminishing returns occur.
5. **DeepSeek-v3 exhibits top performance**, followed closely by GPT-4o-mini, while Qwen-2.5-7B performs worst—perfectly aligned with their general capabilities.

## Highlights & Insights

- **Elegant Integration of Graphs and LLMs**: Moving message passing in GNNs from the embedding space to the text space yields a unified theoretical framework (TextGNN) for multi-agent systems, characterized by a clean and highly scalable formulation.
- **Innovative Evaluation Methodology**: Leveraging node masking prediction bypasses the subjectivity and high labor cost of traditional human evaluations, offering an objective and scalable alternative.
- **Interdisciplinary Idea Generation**: ResearchTown successfully synthesizes non-trivial or rare domain combinations, such as NLP + Astronomy and NLP + Criminology, yielding research ideas that do not exist in reality.
- **Clever Design of Hidden States**: Summarizing full papers into a bullet-point format to serve as hidden states preserves key information while managing context length effectively.

## Limitations & Future Work

1. **Limited Peer Review Simulation Quality**: The similarity for Weakness detection is only around 47, suggesting that professional peer review requires deeper reasoning and domain expertise.
2. **Breakdowns in Overly Broad Interdisciplinary Configurations**: Attempting to combine 4+ unrelated research domains results in incoherent, buzzword-heavy, and superficial outputs.
3. **Simplifications in Core Information**: The framework relies strictly on paper abstracts during reading and writing, keeping full-text data only for the review phase, thus losing fine-grained details.
4. **Absence of Iterative Feedback Loops**: Real-world research involves back-and-forth modification based on review feedback, whereas the current pipeline operates in a single direction.
5. **Strictly Unimodal (Text) Processing**: The framework ignores codebase, data, and charts, overlooking non-textual research modalities.
6. **Inherent Limitations of Evaluation**: High embedding similarity does not directly equal high research quality; human and LLM evaluators show low correlation on intrinsic quality dimensions.

## Related Work & Insights

- **Difference from AI Scientist (Lu et al., 2024)**: AI Scientist focuses on a single-agent end-to-end research loop, whereas ResearchTown coordinates multi-agent community simulations.
- **Difference from ResearchAgent (Baek et al., 2024)**: ResearchAgent specializes in iterative idea generation, while ResearchTown simulates comprehensive activities across the scientific community.
- **Takeaways from TextGNN**: Abstracting GNN message passing into text-space formulations can open up architectural choices in other realms requiring structured collaboration, such as corporate knowledge management and education.
- **Implications for Research Tools**: Future paper recommendation and reviewer assignment systems could model academic ecosystems via agent-data graphs.

## Rating

| Dimension | Rating (1-5) | Comment |
|------|----------|------|
| Novelty | ⭐⭐⭐⭐ | High architectural originality in unifying TextGNN with agent-data graphs |
| Technical Depth | ⭐⭐⭐⭐ | Formally defined, seamlessly deriving scientific research tasks from graph mechanics |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Rigorous ablation study using multiple models, embeddings, and dimensions |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with informative diagrams |
| Value | ⭐⭐⭐ | While simulation fidelity has room to improve, the trajectory is highly promising |
| Overall Rating | ⭐⭐⭐⭐ | Top-tier systems work, opening a new paradigm for scientific community simulation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PaperMentor: A Human-Centered Multi-Agent Writing Tutor for AI Research Papers on Overleaf](../../ACL2026/multi_agent/papermentor_a_human-centered_multi-agent_writing_tutor_for_ai_research_papers_on.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](../../ACL2026/multi_agent/llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ICML 2026\] Searching for Synergy in Shared Workspace Human-AI Collaboration](../../ICML2026/multi_agent/searching_for_synergy_in_shared_workspace_human-ai_collaboration.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](../../ACL2026/multi_agent/roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)

</div>

<!-- RELATED:END -->
