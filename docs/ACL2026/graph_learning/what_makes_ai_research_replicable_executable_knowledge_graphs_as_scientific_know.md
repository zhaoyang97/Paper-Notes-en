---
title: >-
  [Paper Note] What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations
description: >-
  [ACL2026][Graph Learning][Executable Knowledge Graphs] The authors propose Executable Knowledge Graphs (xKG), which organize technical concepts and executable code snippets from papers into a tri-layer graph structure (P…
tags:
  - "ACL2026"
  - "Graph Learning"
  - "Executable Knowledge Graphs"
  - "Paper Replication"
  - "Code Retrieval"
  - "Scientific Research Agent"
  - "PaperBench"
date: 2026-05-08
content_hash: 4f31dc0e0fe77a68
---

# What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations

**Conference**: ACL2026  
**arXiv**: [2510.17795](https://arxiv.org/abs/2510.17795)  
**Code**: https://github.com/zjunlp/xKG  
**Area**: graph_learning  
**Keywords**: Executable Knowledge Graphs, Paper Replication, Code Retrieval, Scientific Research Agent, PaperBench

## TL;DR
The authors propose Executable Knowledge Graphs (xKG), which organize technical concepts and executable code snippets from papers into a tri-layer graph structure (Paper-Technique-Code). As a plug-and-play knowledge base for research replication agents, it achieves a replication score improvement of up to 10.90 points on PaperBench Code-Dev.

## Background & Motivation
**Background**: LLM agents are increasingly utilized for automated research tasks, including literature review, code implementation, experiment replication, and methodological extension. Benchmarks such as PaperBench, MLE-Bench, and LMR-Bench evaluate the implementation capabilities of agents based on scientific papers.

**Limitations of Prior Work**: Replicating AI research is challenging due to key knowledge being scattered across main texts, appendices, cited works, official repositories, and configuration files. Standard RAG retrieves text fragments but struggles to map "technical concepts" to executable code. Relying solely on papers omits implementation details, while relying only on repositories obscures the methodological structure.

**Key Challenge**: Scientific replication requires "executable scientific knowledge." Existing representations are often limited to text, abstracts, or coarse-grained concepts. Agents typically fail at low-level implementation: writing loss functions, modularizing components, configuring hyperparameters, and invoking code interfaces. Without a link between concepts and executable code, knowledge bases provide only general background rather than repo-level implementation support.

**Goal**: The authors aim to build a paper-centric, auto-updating, and plug-and-play knowledge base. It provides agents with high-level methodological structures and low-level executable references to enhance the reliability of AI research replication.

**Key Insight**: Scientific knowledge is extended from text-based KGs to Executable Knowledge Graphs. Nodes represent not just concepts but also verified code units; edges represent not just semantic relations but also structural dependencies and implementation mappings between concepts and code.

**Core Idea**: Papers are decomposed into reusable technique nodes, each grounded to a rewritten, debugged, and verified Code Node. This allows agents to consult the methodological structure during the planning phase and retrieve executable code during the implementation phase.

## Method
xKG is a hierarchical KG designed for AI replication. It features a structured graph representation, an automated construction pipeline, and an agent integration mechanism. The system revolves around a target paper: identifying relevant papers and repositories, extracting technical concepts and implementations, and integrating this knowledge as tools for replication agents.

### Overall Architecture
The xKG is formally represented as $xKG=(N,E)$. The node set $N$ includes Paper Nodes, Technique Nodes, and Code Nodes. The edge set $E$ includes Structural Edges and Implementation Edges. A Paper Node represents a paper and its metadata; a Technique Node denotes a self-contained scholarly concept or module; a Code Node contains an executable unit, including implementation code, test scripts, and documentation.

The construction process consists of two stages. First, paper-aware corpus curation: automatically identifying core techniques, selecting highly relevant cited papers, downloading arXiv sources and GitHub repos, and filtering papers lacking official implementations. Second, hierarchical KG construction: extracting technique trees from papers and code snippets from repos, generating and verifying Code Nodes, and pruning technique nodes that cannot be grounded to code.

### Key Designs
1. **Paper-Technique-Code Tri-layer Representation**:
    - **Function**: Simultaneously represents overall paper structure, methodological concepts, and executable implementations to support the full replication workflow.
    - **Mechanism**: Paper Nodes store metadata and associated technique/code node sets. Technique Nodes store definitions and sub-techniques, representing frameworks or modules. Code Nodes store implementation $\sigma$, test scripts $\tau$, and documentation $\delta$. Structural Edges link architectural dependencies, while Implementation Edges link techniques to code.
    - **Design Motivation**: Standard RAG returns fragments, leaving agents to determine architectural roles. This tri-layer graph explicitly aligns "what the paper says" with "how it is implemented," reducing the agent's integration burden.

2. **Automated Grounding Pipeline for Executability**:
    - **Function**: Automatically converts papers and repositories into reusable, executable knowledge resources.
    - **Mechanism**: The authors use o4-mini to extract technique trees, supplemented by Paper-RAG for definitions. Using technique definitions as queries, code snippets are retrieved from official repositories via embeddings and synthesized into Code Nodes. Each Code Node undergoes a self-debugging loop to ensure executability; nodes that cannot be grounded are filtered out.
    - **Design Motivation**: Paper extraction often produces hallucinated or non-implementable concepts. Requiring techniques to be grounded in code serves as an inherent quality filter.

3. **Two-stage Agent Integration**:
    - **Function**: Assists agents in both understanding methodology and writing specific code.
    - **Mechanism**: During high-level planning, agents access Paper Nodes without implementation details to avoid distraction. During low-level implementation, agents query Technique-Code pairs. An LLM verifier finalizes the results to ensure relevance and implementability.
    - **Design Motivation**: Replication poses two distinct challenges: structural understanding and functional coding. Phase-based exposure prevents premature code exposure while providing fine-grained references during implementation.

### Loss & Training
The study does not propose a new neural loss function. Instead, it constructs a KG used as a plug-and-play module. Model calls are used for extraction, modularization, self-debugging, and verification. Similarity is calculated using embeddings like `text-embedding-3-small` and `all-MiniLM-L6-v2`, with thresholds such as `technique_similarity=0.6` and `paper_similarity=0.6`.

## Key Experimental Results

### Main Results
Evaluation was conducted on the PaperBench Code-Dev lite subset. The task involves developing code from paper descriptions, scored by o3-mini based on a hierarchical rubric. xKG was integrated into BasicAgent, IterativeAgent, and PaperCoder, tested with o3-mini and DeepSeek-R1 backbones.

| Agent | Backbone | Vanilla Score | +xKG Score | Gain |
|-------|----------|----------------|-------------|------|
| BasicAgent | o3-mini | 17.89 | 24.57 | +6.68 |
| BasicAgent | DeepSeek-R1 | 27.89 | 31.62 | +3.73 |
| IterativeAgent | o3-mini | 24.60 | 31.91 | +7.31 |
| IterativeAgent | DeepSeek-R1 | 27.02 | 35.22 | +8.20 |
| PaperCoder | o3-mini | 42.31 | 53.21 | +10.90 |
| PaperCoder | DeepSeek-R1 | 52.23 | 60.34 | +8.11 |

xKG benefits both simple ReAct agents and advanced frameworks like PaperCoder. The largest gain occurred with PaperCoder + o3-mini (+10.90), suggesting that powerful agents can better translate structured executable knowledge into complete implementations.

### Ablation Study
Ablations on node types were performed using PaperCoder + o3-mini to identify critical components.

| Configuration | Replication Score | Drop | Description |
|------|-------------------|------|------|
| xKG Full | 53.21 | - | Complete graph |
| w/o Paper Node | 51.08 | 2.13 | Planning quality drops without overall structure |
| w/o Code Node | 48.65 | 4.56 | Largest degradation; executable code is the core gain |
| w/o Technique Node | 52.16 | 1.05 | Minor impact; some info is implicit in Code Nodes |

Quality analysis of the pipeline:
| Metric | Value | Meaning |
|--------|------|------|
| Technique valid rate | 89.44% | Most technique nodes are self-contained concepts |
| Code valid rate | 100.00% | All Code Nodes are executable after self-debugging |
| Tech-Code pair match | 74.51% | Precision of alignment requires further improvement |
| Initial Code Node exec rate | 52.38% | Low executability before self-debugging |
| Avg construction cost | ~$0.7344 / paper | Costs driven by modularization and debugging |

### Key Findings
- **Code Nodes are critical**: Removing Code Nodes results in a 4.56-point drop, confirming that the bottleneck in replication is implementation rather than conceptual understanding.
- **Suitability**: xKG is more effective for analytical or combinatorial papers (e.g., MU-DPO) built on reusable technologies than for entirely novel architectures (e.g., One-SBI).
- **Scalability**: The system is self-evolving. Expanding the knowledge base to 56 relevant papers significantly boosted scores for complex tasks like `bridging-data-gaps`.

## Highlights & Insights
- "Executable Knowledge Graphs" directly address the core pain point of research agents: replication is a coding task, not a Q&A task.
- Grounding-based filtering ensures that only implementable technical nodes are retained, sacrificing theoretical completeness for practical utility.
- The phase-based memory design (planning vs. implementation) prevents agents from being overwhelmed by implementation details during high-level strategy formulation.
- Case studies indicate that xKG moves agents from "scaffold writing" to "substantive module implementation."

## Limitations & Future Work
- **Evaluation Cost**: Due to high costs, evaluations were limited to the PaperBench lite subset without massive cross-domain stress tests.
- **Dependency**: xKG relies on existing relevant papers and official code. It struggles with closed-source methods or papers without reliable repositories.
- **Semantic Mapping**: Code retrieval may still package technically irrelevant but semantically similar code, though the verifier mitigates this risk.
- **Future Directions**: Exploring online knowledge updates, incorporating failure feedback into the graph, and execution-driven graph correction.

## Related Work & Insights
- **vs. Standard RAG**: xKG replaces flat retrieval with structured paper architectures and executable grounding.
- **vs. Research Agents (AutoMind, AI-Researcher)**: While those focus on workflow, xKG serves as an underlying plug-and-play knowledge base.
- **vs. Paper2Code/AutoReproduce**: xKG emphasizes reusing verified knowledge from existing repositories rather than zero-shot implementation.
- **vs. ExeKG**: While sharing the name, this work focuses on AI research replication via thin Paper-Technique-Code structures rather than industrial monitoring.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Conceptually clear and well-aligned with agent needs by extending KG nodes to executable code.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers multiple backbones/agents and includes detailed quality analysis, though limited by PaperBench subset size.
- **Writing Quality**: ⭐⭐⭐⭐☆ Highly readable structure; detailed implementation information is well-organized.
- **Value**: ⭐⭐⭐⭐⭐ High reference value for automated research, code RAG, and agent memory design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA](../../ICML2026/graph_learning/what_structural_inductive_bias_helps_transformers_reason_over_knowledge_graphs_a.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[ACL 2026\] CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs](cog_controllable_graph_reasoning_via_relational_blueprints_and_failure-aware_ref.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)

</div>

<!-- RELATED:END -->
