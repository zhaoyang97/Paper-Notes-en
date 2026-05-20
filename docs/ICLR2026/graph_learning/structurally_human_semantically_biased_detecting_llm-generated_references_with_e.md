---
title: >-
  [Paper Note] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs
description: >-
  [ICLR 2026][Graph Learning][LLM reference detection] By constructing paired citation graphs (human vs. GPT-4o-generated vs. random baseline) for 10,000 papers…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "LLM reference detection"
  - "citation graph"
  - "graph neural networks"
  - "semantic embeddings"
  - "academic integrity"
date: 2026-05-08
content_hash: f2875fd718fcf258
---

# Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs

**Conference**: ICLR 2026
**arXiv**: [2601.20704](https://arxiv.org/abs/2601.20704)  
**Code**: None  
**Area**: AI Safety / Graph Learning
**Keywords**: LLM reference detection, citation graph, graph neural networks, semantic embeddings, academic integrity

## TL;DR
By constructing paired citation graphs (human vs. GPT-4o-generated vs. random baseline) for 10,000 papers, this work finds that LLM-generated reference lists are nearly indistinguishable from human ones in terms of graph topology (RF accuracy only 60%), yet are effectively detectable via semantic embeddings (RF 83%, GNN 93%). This indicates that LLMs accurately mimic citation topology while leaving detectable semantic fingerprints.

## Background & Motivation

**Background**: LLMs are increasingly used to synthesize scientific knowledge, draft literature reviews, and suggest references. Prior studies have found that LLM-generated references resemble human ones on coarse-grained metrics (title length, team size, citation count), but exhibit systematic biases at finer granularity (amplified Matthew effect, preference for recent papers, reduced self-citations).

**Limitations of Prior Work**: It remains unclear whether LLM-generated and human-generated reference lists can be reliably distinguished. Single-reference auditing approaches (e.g., LLM-Check) are insufficient to capture list-level patterns.

**Key Challenge**: Do LLMs genuinely understand citation structure, or merely imitate it superficially? If topological structure is similar, where do the differences lie?

**Goal**: To systematically evaluate the structural and semantic differences between LLM-generated and human citation graphs, and to develop corresponding detection methods.

**Key Insight**: A progressive modeling strategy—from interpretable graph structural features to semantic embeddings to GNNs—that incrementally isolates the contributions of topology vs. semantics.

**Core Idea**: LLM references are "structurally human, semantically biased"—detection should target content signals rather than graph structure.

## Method

### Overall Architecture

10,000 papers are sampled from SciSciNet → paired citation graphs are constructed (ground-truth, GPT-4o-generated, and domain-matched random baseline) → structural features are extracted (degree/closeness/eigenvector centrality, clustering coefficient, edge count) → semantic embeddings are extracted (OpenAI 3072-D) → RF + GNN three-class classification evaluation.

### Key Designs

1. **Citation Graph Construction**:

    - **Function**: Construct paired ground-truth/generated citation graphs for each paper.
    - **Mechanism**: The focal paper serves as the main node; cited papers are child nodes; citation relations are retrieved from SciSciNet. GPT-4o generates references purely parametrically given title, abstract, author information, etc. The random baseline uniformly reshuffles citations within the same field, preserving the degree distribution.
    - **Design Motivation**: Controlled experiment—three citation graphs for the same focal paper are directly comparable.

2. **Structural Features vs. Semantic Embeddings**:

    - **Structural features**: Degree/closeness/eigenvector centrality, clustering coefficient, edge count → RF classification.
    - **Semantic embeddings**: OpenAI text-embedding-3-large (3072-D) → graph-level aggregation → RF / used as GNN node features.
    - **Design Motivation**: Disentangle the contributions of topological signals and content signals.

3. **GNN Graph Classification**:

    - **Function**: Graph-level binary classification using GCN/GAT/GIN/GraphSAGE.
    - **Mechanism**: Node features are either structural attributes (5-D) or semantic embeddings (3072-D); graph-level readout is followed by binary classification.
    - **Design Motivation**: GNNs can jointly exploit both structural and semantic signals.

### Loss & Training

Adam optimizer, 70/15/15 split, balanced dataset. Robustness validated with dual LLMs (GPT-4o + Claude Sonnet 4.5) and dual embedding models (SPECTER + OpenAI).

## Key Experimental Results

### Main Results

| Method | GT vs GPT | GT vs Random | GPT vs Random |
|------|----------|-------------|--------------|
| RF (structural features) | 0.608 | 0.896 | 0.928 |
| RF (semantic embeddings) | **0.835** | 0.908 | 0.953 |
| GNN (structural features) | ~0.55 | ~0.90 | ~0.93 |
| GNN (semantic embeddings) | **0.93** | ~0.95 | ~0.97 |

### Ablation Study

| Configuration | GT vs GPT Accuracy | Notes |
|------|----------------|------|
| GNN + embeddings | 93% | Best |
| RF + embeddings | 83.5% | Semantic embeddings contribute significantly |
| RF + structure | 60.8% | Near random |
| GNN + structure | ~55% | Structure alone entirely insufficient |
| Random embedding substitution | ~50% | Confirms effect is not due to dimensionality |
| Cross-generator (GPT train → Claude test) | ~72% | Generalizes to other LLMs |

### Key Findings
- **Topology is nearly indistinguishable**: Centrality and clustering coefficients of GPT citation graphs heavily overlap with ground-truth graphs; RF achieves only 60%.
- **Semantic fingerprints are detectable**: Embedding features improve accuracy from 60% to 83% (RF) / 93% (GNN).
- **Random baseline is easily separated**: Ground-truth vs. random achieves 89%+, GPT vs. random achieves 93%+—demonstrating that GPT does generate structurally plausible citations.
- **Cross-LLM generalization**: A classifier trained on GPT-4o achieves 72% accuracy on Claude.
- **Replacing embeddings with random vectors drops accuracy to 50%**, confirming that discriminative power derives from semantic structure rather than dimensionality.

## Highlights & Insights
- The finding of **"structurally human, semantically biased"** has direct implications for auditing and debiasing strategies—detection efforts should focus on content signals rather than graph structure.
- The **domain-matched random baseline** design is methodologically rigorous—reshuffling citations within the same field controls for topic distribution.
- The **progressive analysis** (structure → embeddings → GNN) clearly delineates the contribution of each modeling layer.

## Limitations & Future Work
- Only parametric generation (without RAG) is tested; in practice, LLMs may employ retrieval-augmented generation.
- The specific semantic dimensions underlying the observed differences (e.g., recency bias, prestige bias) are not deeply analyzed.
- It remains unclear which dimensions of the 3072-D embeddings drive discriminative power.
- Only binary classification is explored; multi-class settings (e.g., partially LLM-generated references) are not investigated.

## Related Work & Insights
- **vs. LLM-Check**: LLM-Check audits the existence of individual citations, whereas this paper evaluates graph-level patterns of entire reference lists.
- **vs. Algaba et al.**: Prior work identified coarse-grained consistency; this paper achieves high-accuracy automatic detection via GNNs and embeddings.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of citation graphs and GNNs is novel, though the analytical framework itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10,000 graphs, dual LLMs, dual embedding models, multiple baselines, and random embedding controls—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Excellent visualizations and clear layer-by-layer analysis.
- Value: ⭐⭐⭐⭐ Practically meaningful for academic integrity and AI-assisted writing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)
- [\[ICLR 2026\] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](../../ACL2026/graph_learning/graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[AAAI 2026\] Sentient: Detecting APTs Via Capturing Indirect Dependencies and Behavioral Logic](../../AAAI2026/graph_learning/sentient_detecting_apts_via_capturing_indirect_dependencies_and_behavioral_logic.md)

</div>

<!-- RELATED:END -->
