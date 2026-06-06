---
title: >-
  [Paper Note] What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA
description: >-
  [ICML2026][Graph Learning][Knowledge Graph Question Answering] This paper uses a detachable minimal Graph Transformer variant, RASA, for controlled experiments. It discovers that the most useful structural inductive bias…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "Graph Transformer"
  - "Sparse Attention"
  - "Structural Inductive Bias"
  - "Multi-hop Reasoning"
date: 2026-05-08
content_hash: b482de7839c45cd4
---

# What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA

**Conference**: ICML2026  
**arXiv**: [2602.02834](https://arxiv.org/abs/2602.02834)  
**Code**: No public code  
**Area**: Graph Learning / Knowledge Graph Reasoning  
**Keywords**: Knowledge Graph Question Answering, Graph Transformer, Sparse Attention, Structural Inductive Bias, Multi-hop Reasoning  

## TL;DR
This paper uses a detachable minimal Graph Transformer variant, RASA, for controlled experiments. It discovers that the most useful structural inductive bias in multi-hop Knowledge Graph Question Answering (KGQA) is the topological constraint provided by adjacency masks, rather than learnable relation parameters such as relation type bias, query scaling, or value gating.

## Background & Motivation
**Background**: KGQA requires models to perform multi-hop reasoning along entity and relation chains. Common solutions include Graph Neural Networks (GNNs) like R-GCN/GAT, structured Transformers like Graphormer, and KG+LLM pipelines that delegate subgraph retrieval to LLMs. Graph Transformer research typically bundles multiple structural signals—such as centrality encoding, shortest path encoding, edge type encoding, and sparse attention—reporting only the overall performance.

**Limitations of Prior Work**: Although these methods are effective, it is difficult to determine which structural information actually drives performance. If a model simultaneously employs adjacency masks, edge type embeddings, positional encodings, and relation weights, the performance gain could stem from a smaller search space, relation type semantics, or simply extra parameters and hyperparameter tuning. This attribution problem is critical for multi-hop reasoning in KGs, as models often fail not because they lack entity knowledge, but because they do not know which graph path to propagate information along.

**Key Challenge**: While the global attention of a Transformer theoretically allows direct interaction between any nodes, multi-hop reachability inherently requires composing paths hop-by-hop. Relation parameters can only weight or scale existing attention scores but cannot guarantee that information flows strictly along graph edges. In contrast, adjacency masks explicitly restrict the attention space to actual neighbors. The challenge this paper addresses is whether the useful bias in KGQA is "knowing the edge type" or "knowing the edge constraints first."

**Goal**: The authors do not seek to create a new SOTA system. Instead, they construct an experimental apparatus with detachable components to separately measure the contributions of topological masks and learnable relation parameters. They also examine the stability of these contributions across datasets like MetaQA, WebQSP, and CWQ, as well as in held-out relation settings.

**Key Insight**: The paper starts with a simple logical argument: if each attention layer expands a node's representation to its one-hop neighbors at most, then $k$-hop reasoning requires at least $k$ layers of effective propagation. Adjacency masks explicitly implement this "one-hop per layer" mechanism, whereas relation bias, scale, or gates only re-weight dense attention, which might not learn the correct propagation paths.

**Core Idea**: By using RASA to separate "topological constraints" and "relation re-weighting" into independently toggleable components, the paper proves that the primary gain in multi-hop KGQA comes from the topological inductive bias of adjacency masks.

## Method
The method consists of two layers: a theoretical layer explaining why multi-hop KG reasoning requires hop-by-hop propagation, and an experimental layer, RASA, which exposes four types of structural signals as ablatable components with minimal modifications. RASA is not intended as a flagship architecture but as a "microscope" to observe individual contributions within the same Transformer framework.

### Overall Architecture
The input consists of a Knowledge Graph subgraph, the question text, and candidate answer entities. The question is encoded using a frozen DistilBERT. On the graph side, a 3-layer RASA Transformer with a hidden size of 128 and 4 heads is used. Each attention layer reads edge connections and types: if two nodes are connected or have a self-loop, attention is allowed; otherwise, the score is set to $-\infty$. On allowed edges, the model can use relation-type bias, relation-specific query scaling, and value gating to fine-tune the influence of different relations. Finally, the model scores answers based on entity representations, evaluated using Hits@1/Hits@10.

In terms of experimental design, the authors fix the encoder, answer scoring, and hyperparameter tuning budget, then compare Vanilla Transformer, Graphormer, R-GCN, GAT, and various component-wise ablations of RASA. The key comparison is the performance gap between Full, Mask-only, and Bias-only configurations.

### Key Designs
1. **Adjacency Mask as a Topological Propagation Constraint**:
    - **Function**: Restricts each node's attention scope to its graph neighbors and itself, ensuring one layer of attention corresponds to one hop of message propagation.
    - **Mechanism**: Standard attention scores are $S_{ij}=(XW_Q)_i\cdot(XW_K)_j/\sqrt{d_k}$. RASA filters scores before the softmax: if $A_{ij}=0$ and $i\ne j$, then $S_{ij}=-\infty$. This prevents non-adjacent nodes from direct information exchange, forcing multi-hop answers to accumulate through layers.
    - **Design Motivation**: The search space for global dense attention is $O(n^2)$, but real edges in sparse KGs are $O(m)$ where $m\ll n^2$. Masks do more than save computation; they encode "which paths are potentially legal" into the model, reducing the burden of re-discovering connectivity from data.

2. **Relation Parameters for Local Re-weighting**:
    - **Function**: Distinguishes between relation types on allowed edges by applying relation-specific adjustments to attention scores or value flows.
    - **Mechanism**: RASA learns three parameters per relation and attention head: edge-type bias $b_r$ added to the score, query scaling $s_r$ multiplied by the score, and a value gate $g_r$ that regulates information flow via $\sigma(g_r)$. These add only $3|R|H$ parameters per layer.
    - **Design Motivation**: These parameters capture the relative importance of relations like "director," "place of birth," or "starring," but they do not change the attention graph itself. Without adjacency masks, they merely adjust weights in dense attention without restricting interactions to valid paths.

3. **Causal Ablation with Detachable Components**:
    - **Function**: Allows the same architecture to run in "Full," "Mask-only," or "Bias-only" configurations to avoid confounded attribution.
    - **Mechanism**: "Full" enables mask, bias, scale, and gate. "Mask-only" keeps the binary adjacency mask but removes all learnable relation parameters. "Bias-only" removes the mask, keeping only edge-type bias.
    - **Design Motivation**: If "Mask-only" recovers most of the performance gain while "Bias-only" performs near or below the Vanilla Transformer, it demonstrates that the primary contribution is the topological constraint rather than relational parameters or model capacity.

### Loss & Training
The paper introduces no special loss functions, focusing instead on controlled variables. All self-implemented models use a frozen DistilBERT encoder and a unified answer scoring head. RASA is configured with 3 layers, 128 dimensions, 4 heads, batch size 16, using AdamW with a learning rate of $2\times 10^{-5}$, cosine annealing, and early stopping with a patience of 5. Results are reported as the mean and standard deviation across 3 random seeds.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | RASA / Ours | Strong Baselines | Vanilla Transformer | Conclusion |
|--------|------|------|----------|------|------|
| MetaQA 3-hop | Hits@1 | 92.6±0.1 | Graphormer 93.3±0.2 / R-GCN 91.9±0.2 | 12.9±0.2 | RASA's performance is sufficient for its role as an ablation vehicle. |
| WebQSP | Hits@1 | 72.5±0.2 | Graphormer 74.0±0.4 / R-GCN 65.7±0.6 | 18.7 | Structured attention significantly outperforms unstructured Transformers. |
| CWQ | Hits@1 | 59.9±0.2 | Graphormer 64.7±0.1 / R-GCN 58.2±0.0 | 2.7 | Topology remains the core information for complex compositional queries. |
| MetaQA held-out relation | Perf. Drop | RASA -7.2pp | R-GCN -29.2pp | - | Mask-based topological bias generalizes better to unseen relations than relation-specific weights. |

### Ablation Study
| Configuration | Key Metrics | Description |
|------|---------|------|
| Full RASA | MetaQA 3-hop 92.6±0.1 | Mask + bias + scale + gate all enabled. |
| Mask only | MetaQA 3-hop 85.4±0.1 | Adjacency mask only; recovers ~91% of the gain of the full model over the unmasked one. |
| Bias only | MetaQA 3-hop 12.9±0.2 | Without mask, relation bias alone performs similarly to Vanilla Transformer. |
| Mask only (WebQSP / CWQ) | 64.2 / 56.6 | Compared to Vanilla's 18.7 / 2.7, mask contributes +45.5pp / +53.9pp. |
| Bias only (CWQ) | 2.1 | Lower than Vanilla Transformer (2.7), suggesting relation bias without topology can become noise. |

### Key Findings
- The adjacency mask is the largest contributor: On MetaQA 3-hop, the jump from Bias-only to Mask-only is +72.5pp, whereas adding relation parameters onto the mask only adds +7.2pp.
- This conclusion replicates across datasets: On WebQSP and CWQ, the mask contributes +45.5pp and +53.9pp respectively, accounting for most of the gains.
- Relation parameters are not useless but act as refinements on top of the correct topological path. Without the mask providing the valid propagation space, they can harm performance on long compositional chains like in CWQ.
- In terms of efficiency, the current implementation of RASA is approximately 6x slower than a Vanilla Transformer, primarily because it uses dense adjacency matrix operations rather than specialized sparse kernels.

## Highlights & Insights
- The primary value of this work is not the RASA architecture itself but its positioning as an ablation tool. While many Graph Transformer papers bundle structural signals, this paper decouples "topology" from "relation semantics," making the conclusion a mechanistic analysis rather than a leaderboard report.
- The "Topology over Relation Parameters" insight suggests that for multi-hop KGQA, ensuring information flows hop-by-hop along valid edges is more critical than learning fine-grained weights for each relation type. This explains why relation-heavy models can be brittle with unseen relations or compositional generalization.
- The held-out relation experiment provides independent evidence. R-GCN's relation-specific matrices degrade significantly when encountering unseen edge types, whereas the adjacency mask relies only on connectivity, maintaining usable paths even without learned parameters.
- The paper serves as a reminder for future graph learning research: structural constraints that change the attention pattern and relation re-weightings that change only scores are inductive biases at different mechanistic levels.

## Limitations & Future Work
- The paper relies on an explicit Knowledge Graph structure. If the graph is incomplete, missing edges, or the candidate subgraph recall is poor, strict adjacency masking might block information that should have been inferred across edges.
- RASA’s implementation does not utilize sparse attention kernels, resulting in a latency of 49.0ms compared to 7.9ms for the Vanilla Transformer. Practical KGQA systems would require implementing these sparse structures at the kernel and batching levels.
- The main ablation treats bias, query scaling, and value gating as a collective "relation parameter" group without detailing their individual contributions. The paper answers the large-scale "Topology vs. Relation" question but not which specific relation re-weighting is most robust.
- Absolute performance does not exceed LLM-augmented KGQA systems (e.g., SubgraphRAG+GPT-4o reaches 90.1% on WebQSP), indicating these findings are better suited for guiding structural component design rather than replacing retrieval-augmented LLMs.

## Related Work & Insights
- **vs. Graphormer**: Graphormer injects structure into dense attention via centrality, spatial distance, and edge encodings. RASA changes the attention pattern via adjacency masks. While Graphormer has higher absolute performance, this work shows the "structural channel" itself is the key.
- **vs. R-GCN**: R-GCN uses relation-specific weight matrices for message passing, which is effective for seen relations but degrades in held-out relation tests. RASA's mask is more robust to new relation types because it relies on topological connectivity.
- **vs. KG+LLM / SubgraphRAG**: LLM-augmented methods leverage textual knowledge and large-scale reasoning for end-to-end performance. This paper controls for external knowledge to focus on pure structural bias. The two could be combined: use mask-based attention for interpretable path aggregation after subgraph retrieval.
- **Insight**: For other structural reasoning tasks—such as program dependency graphs, molecular graphs, or causal graphs—prioritizing explicit structural masks to restrict information flow may be more robust than relying solely on edge embeddings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While adjacency masks are not new, the clean ablation used to answer "which structural bias matters most" is highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers MetaQA, WebQSP, CWQ, held-out relations, and attention entropy, though internal ablations of relation parameters could be finer.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is well-positioned, emphasizing RASA as an experimental vehicle rather than a SOTA system, with clear boundary conditions for its conclusions.
- **Value**: ⭐⭐⭐⭐ Directly informs the design of Graph Transformers and KGQA models, particularly for structural reasoning requiring compositional generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations](../../ACL2026/graph_learning/what_makes_ai_research_replicable_executable_knowledge_graphs_as_scientific_know.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](../../ACL2026/graph_learning/stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)

</div>

<!-- RELATED:END -->
