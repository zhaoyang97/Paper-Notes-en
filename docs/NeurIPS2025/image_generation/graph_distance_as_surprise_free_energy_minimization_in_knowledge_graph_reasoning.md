---
title: >-
  [Paper Note] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning
description: >-
  [NeurIPS 2025 (NORA Workshop)][Image Generation][Knowledge Graph] This paper establishes a formal connection between the Free Energy Principle (FEP) from neuroscience and knowledge graph reasoning. It proposes using shor…
tags:
  - "NeurIPS 2025 (NORA Workshop)"
  - "Image Generation"
  - "Knowledge Graph"
  - "Free Energy Principle"
  - "graph distance"
  - "surprise minimization"
  - "entity grounding"
date: 2026-05-08
content_hash: 43e3a02d1ab5f589
---

# Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning

**Conference**: NeurIPS 2025 (NORA Workshop)
**arXiv**: [2512.01878](https://arxiv.org/abs/2512.01878)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Knowledge Graph, Free Energy Principle, graph distance, surprise minimization, entity grounding

## TL;DR

This paper establishes a formal connection between the Free Energy Principle (FEP) from neuroscience and knowledge graph reasoning. It proposes using shortest-path graph distance as a measure of surprise, generalizing the tree-structured surprise theory of Murphy et al. to arbitrary directed graphs, and provides a principled theoretical framework for entity grounding in KG-based agents.

## Background & Motivation

**FEP moving from neuroscience to AI systems.** The Free Energy Principle posits that biological systems minimize surprise by maintaining accurate world models. Murphy et al. previously demonstrated that syntactic operations minimize surprise through shallow tree structures, quantifying surprise via tree depth. However, this framework is limited to tree structures and cannot handle the complex topologies common in knowledge graphs, such as directed graphs, cycles, and multi-path structures.

**KG reasoning lacks a theoretical foundation.** Knowledge graphs are widely used in modern AI agents to enhance reasoning, memory, and planning, yet a fundamental question remains theoretically unaddressed: when an agent performs entity grounding based on a KG, how can it principally determine which entities are plausible in a given context? Existing embedding-based methods (e.g., TransE, RotatE) are practical but lack grounding in cognitive science theory.

**Graph distance as a natural measure of surprise.** This paper proposes an intuitive yet theoretically motivated answer: within a KG, entities closer to the context are less surprising, while those farther away are more surprising. By treating the KG as the agent's generative model, the shortest-path distance directly corresponds to the surprise term in FEP, translating an abstract cognitive principle into a computable graph algorithm.

## Method

### Overall Architecture

Given a knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$ and a query context $C$, the framework computes shortest-path distances from the context to all candidate entities via BFS, combines this with a Kolmogorov complexity estimate of relational paths, and derives a free energy value for each entity. Lower free energy implies higher plausibility.

### Key Designs

1. **Geometric Surprise**:

    - Function: Quantifies the degree to which an entity is "surprising" relative to the context.
    - Mechanism: $S_{\text{geo}}(e|C) = \min_{c \in C} d_{\mathcal{G}}(c, e)$, defined as the shortest directed path length computed via BFS when a path exists, or a penalty constant $\alpha$ (required to exceed the graph diameter) otherwise.
    - Design Motivation: Reduces to Murphy et al.'s tree depth on tree-structured graphs, making it a natural generalization. BFS handles cycles via a visited set, guaranteeing termination with complexity $O(|\mathcal{E}| + |\mathcal{T}|)$.

2. **Free Energy Combination Formula**:

    - Function: Combines geometric distance with path complexity to yield a complete surprise measure.
    - Mechanism: $F(e|C) = S_{\text{geo}}(e|C) + \lambda K(\pi_{C \to e})$, where $K(\pi)$ denotes the Kolmogorov complexity of the relational path, approximated via Lempel-Ziv compression.
    - Design Motivation: Distance alone is insufficient — frequently occurring relational patterns (e.g., *hasLeader*) are less surprising than rare ones (e.g., cross-national inference). The algorithmic complexity term captures the regularity or irregularity of a path.

3. **Formal Mapping to FEP**:

    - Function: Provides a cognitive-scientific theoretical foundation for graph-distance surprise.
    - Mechanism: Under FEP, $F = -\log P(o,s) - H[Q(s)]$. Interpreting the KG as the agent's generative model yields $-\log P(e|C) \propto d_{\mathcal{G}}(C,e)$ (shorter distance → higher probability → lower surprise). $S_{\text{geo}}$ realizes the surprise term, while $K(\pi)$ approximates the entropy term.
    - Design Motivation: The least-action principle of FEP aligns with the minimization objective of shortest paths; the $k$-hop neighborhood aggregation in GNN message passing also resonates with this perspective.

### Loss & Training

This paper presents a theoretical framework and does not involve model training. The inference pipeline is as follows: construct the KG → identify context $C$ → compute distances via BFS → estimate path Kolmogorov complexity via LZ77 compression → combine free energy $F = S_{\text{geo}} + \lambda K(\pi)$ → rank candidate entities in ascending order of free energy.

## Key Experimental Results

### Main Results

This paper presents a theoretical framework with a worked example; no standard benchmark experiments are conducted. A Canadian politics KG is used as an illustrative case:

| Entity | Graph Distance $S_{\text{geo}}$ | $K(\pi)$ (Path Complexity) | Free Energy $F$ | Notes |
|--------|------|------|----------|------|
| Trudeau | 1 | Low | ~1.3 | Correct answer; directly connected in 1 hop |
| Harper | 1 | Low | ~1.3 | Correct answer; equally surprising as Trudeau |
| PrimeMinister (node) | 2 | Low | ~2.3 | Abstract role node; not a direct answer |
| Biden | 5 (=α) | High | ~5.5 | No connecting path; correctly excluded |

### Ablation Study

| Configuration | Key Result | Notes |
|------|---------|------|
| $S_{\text{geo}}$ only (no $K(\pi)$) | Distinguishes connected vs. disconnected entities | Cannot differentiate path patterns |
| $S_{\text{geo}} + K(\pi)$ | Full free energy | Accounts for both distance and path regularity |
| Cycle handling | BFS correctly handles the Trudeau↔Harper cycle | Visited set ensures termination |

### Key Findings

- The framework correctly identifies Trudeau and Harper (two Canadian PMs) as low-surprise entities, while Biden (U.S. President, disconnected from context) is assigned the highest surprise.
- Cycles do not affect the framework — BFS handles them naturally via the visited set.
- Multiple valid answers can coexist with equal surprise values.

## Highlights & Insights

- The interdisciplinary perspective is novel: the paper establishes a rigorous formal mapping from the cognitive-scientific FEP to KG reasoning, and offers an FEP-theoretic explanation for the choice of GNN message passing depth.
- The mathematical formulation is concise: the logical chain from the surprise definition to the free energy combination formula is self-consistent and degenerates correctly to existing results on tree structures.
- Three theoretical pillars — proper generalization, least-action, and computational grounding — mutually reinforce one another.

## Limitations & Future Work

- The work is presented as work-in-progress: quantitative evaluation on standard KG benchmarks such as FB15k-237 and YAGO is absent.
- The worked example is overly simplistic (five entities), making it impossible to assess scalability to large-scale KGs with millions of nodes.
- The core claim that "closer graph distance implies greater plausibility" is quite intuitive; the FEP framing adds theoretical decoration but offers limited substantive contribution.
- The LZ compression approximation of Kolmogorov complexity is questionable for short relational sequences of 1–2 hops.
- All relations are treated with equal weight, whereas semantic importance varies considerably across relations in real KGs.
- No empirical comparison with embedding-based methods (TransE, RotatE) or path-based reasoning approaches is provided.
- No systematic strategy is proposed for selecting the hyperparameters $\alpha$ and $\lambda$.

## Related Work & Insights

- Relation to Murphy et al.: This paper is a direct generalization of their tree-structured surprise theory to arbitrary directed graphs.
- Relation to KG embedding methods: The TransE objective $h + r \approx t$ can be viewed as implicit distance minimization; this paper provides an explicit distance–surprise mapping.
- Insights: The selection of message passing depth in GNN architectures can be considered from a surprise-horizon perspective; graph distance can serve as a prior for entity grounding ranking in LLM-KG hybrid systems.

## Rating

- Novelty: ⭐⭐⭐ The interdisciplinary connection is interesting, but the FEP→graph distance mapping is fairly intuitive and lacks theoretical depth.
- Experimental Thoroughness: ⭐⭐ Workshop paper with only a toy example; no benchmark experiments.
- Writing Quality: ⭐⭐⭐ Logically clear but substantively thin; formalization is rigorous.
- Value: ⭐⭐⭐ Offers an interesting theoretical perspective but remains far from practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](graph_diffusion_that_can_insert_and_delete.md)
- [\[NeurIPS 2025\] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)

</div>

<!-- RELATED:END -->
