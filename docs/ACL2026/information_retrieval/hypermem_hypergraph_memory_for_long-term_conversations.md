---
title: >-
  [Paper Note] HyperMem: Hypergraph Memory for Long-Term Conversations
description: >-
  [ACL 2026][Information Retrieval & RAG][Hypergraph Memory] HyperMem replaces the pairwise edges of traditional RAG with "hypergraphs (hyperedges connecting $\ge 3$ nodes)." It organizes long-term conversation memory into…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Hypergraph Memory"
  - "Three-layer Architecture"
  - "High-order Association"
  - "Coarse-to-fine Retrieval"
  - "LoCoMo"
date: 2026-05-08
content_hash: 607c1d3da571d6be
---

# HyperMem: Hypergraph Memory for Long-Term Conversations

**Conference**: ACL 2026  
**arXiv**: [2604.08256](https://arxiv.org/abs/2604.08256)  
**Code**: To be open-sourced (Paper footnote states "source code is about to be released")  
**Area**: Long-term Conversation Memory / Agentic Memory / Retrieval-Augmented Generation (RAG)  
**Keywords**: Hypergraph Memory, Three-layer Architecture, High-order Association, Coarse-to-fine Retrieval, LoCoMo

## TL;DR
HyperMem replaces the pairwise edges of traditional RAG with "hypergraphs (hyperedges connecting $\ge 3$ nodes)." It organizes long-term conversation memory into a "Topic → Episode → Fact" three-layer structure. By utilizing coarse-to-fine retrieval combined with hypergraph embedding propagation, it resolves the retrieval fragmentation issue of multi-episode cross-temporal correlations. On the LoCoMo benchmark, it achieves an LLM-as-judge accuracy of 92.73% (Prev. SOTA 86.49%).

## Background & Motivation

**Background**: The fixed context window of dialogue agents cannot accommodate months of dialogue history, necessitating a long-term memory module. Current mainstream solutions fall into two categories: (1) RAG-based (GraphRAG / LightRAG / HippoRAG2 / HyperGraphRAG), which use chunks or graph structures to store external knowledge; (2) Memory system-based (MemoryBank / A-Mem / Mem0 / Zep / MIRIX / MemOS), specifically designed for hierarchical memory in dialogue scenarios.

**Limitations of Prior Work**: Both types of methods **only use pairwise relations**—chunk RAG involves chunk-chunk retrieval, and graph RAG uses entity-entity edges. However, truly important correlations in dialogues are often **high-order**. For example, a user's "sports" topic might involve seven different dialogue fragments over time (episode 1, 3, 4, ...), with each episode scattering multiple facts (what sport, with whom, when, results). Pairwise edges cannot explicitly express that "this group of episodes jointly belongs to one topic," leading to retrieval fragmentation and significant drops in multi-hop reasoning performance.

**Key Challenge**: The **joint dependency** of dialogue memory is inherently high-order, yet existing data structures (graphs/trees) only support binary relations. Even tree-based indices like RAPTOR / SiReRAG / HiRAG rely on hierarchical edges (parent-child pairwise relations) and cannot explicitly group nodes.

**Goal**: (1) Identify a structure capable of expressing joint associations among $\ge 3$ nodes; (2) Organize memory into three semantic granularities: topic / episode / fact; (3) Design a coarse-to-fine retrieval strategy that locates the topic before expanding to facts.

**Key Insight**: The hyperedge in a hypergraph can connect any number of nodes, making it naturally suited for "binding multiple episodes of the same topic into a group." This aligns with the associative nature of human memory (associative memory, Anderson & Bower).

**Core Idea**: Use hyperedges to explicitly group episodes of the same topic and facts within the same episode, unifying fragmented dialogue content into coherent units. Then, apply hypergraph embedding propagation to allow nodes within the same hyperedge to share semantics. Finally, perform coarse-to-fine retrieval via topic → episode → fact, combined with RRF fusion and a reranker to generate the final context.

## Method

### Overall Architecture
The input is a streaming dialogue $X = \{x_t\}_{t=1}^T$, and the output is the agent's answer to a query $q$. The system involves a three-stage construction and a three-stage retrieval:

1.  **Construction Phase (Offline)**:
    *   Stage 1 Episode Detection: LLM-driven streaming boundary detection cuts the dialogue into semantically complete episode nodes $v^E = (v^E_{\text{dialogue}}, v^E_{\text{title}}, v^E_{\text{episode}})$.
    *   Stage 2 Topic Aggregation: Each new episode retrieves similar historical episodes. Three cases—"initialize topic / create new topic / update existing topic"—are handled to establish $e^E_t \in \mathcal{E}^E$, binding multiple episodes of the same topic within a hyperedge.
    *   Stage 3 Fact Extraction: Atomic facts $v^F = (v^F_{\text{content}}, v^F_{\text{potential}}, v^F_{\text{keywords}})$ are extracted from each episode. The `potential` field predicts what types of queries the fact can answer, and `keywords` are used for BM25 retrieval. A hyperedge $e^F$ is established to bind multiple facts from the same episode.
2.  **Indexing Phase (Offline)**: Each node is dual-indexed with BM25 sparse and Qwen3-Embedding-4B dense indices. Hypergraph embedding propagation $\bm{h}'_v = \bm{h}_v + \lambda \cdot \text{Agg}_{e \in \mathcal{N}(v)}(\bm{h}_e)$ is performed to ensure that distant episodes of the same topic acquire similar embeddings.
3.  **Retrieval Phase (Online)**: A three-level coarse-to-fine retrieval (Topic → Episode → Fact) is executed. Each level uses RRF to fuse BM25 and dense rankings, followed by a Qwen3-Reranker-4B for fine-grained ranking, selecting top-$k^T=10$, $k^E=10$, and $k^F=30$ respectively. Finally, fact content and episode summaries are concatenated into a context fed to GPT-4.1-mini to generate the answer.

### Key Designs

1.  **Three-layer Hypergraph Structure (Topic / Episode / Fact)**:
    *   Function: Expands "pairwise relations" to "many-to-many joint associations" to explicitly group dialogue fragments related across time.
    *   Mechanism: Formalizes memory as $\mathcal{H} = (\mathcal{V}^T \cup \mathcal{V}^E \cup \mathcal{V}^F, \mathcal{E}^E \cup \mathcal{E}^F)$, where hyperedge $\mathcal{E}^E$ connects all episodes of the same topic and $\mathcal{E}^F$ connects all facts of the same episode. Each hyperedge carries importance weights $w_{e,v} \in [0,1]$ assigned by the LLM. The Topic layer acts as semantic anchors (spanning months/weeks), the Episode layer as continuous event segments, and the Fact layer as atomic retrieval units.
    *   Design Motivation: The reason traditional GraphRAG's entity-entity edges fail in multi-hop scenarios is that high-order relations, such as "two facts belonging to the same event," cannot be encoded. You can only indirectly link them via a shared entity, but if the entity is not in the query, the link breaks. Hyperedges group them directly; a single topic can pull out all seven mentioned tournaments (Case study Figure 10).

2.  **Hypergraph Embedding Propagation**:
    *   Function: Brings the dense embeddings of nodes within the same hyperedge closer, ensuring that semantically related but temporally distant episodes cluster in the vector space.
    *   Mechanism: Hyperedge embeddings are first aggregated using softmax importance weights: $\bm{h}_e = \sum_v \alpha_{e,v} \bm{h}_v$, where $\alpha_{e,v} = \exp(w_{e,v}) / \sum_u \exp(w_{e,u})$. This is then written back to nodes as $\bm{h}'_v = \bm{h}_v + \lambda \cdot \text{Agg}_{e \in \mathcal{N}(v)}(\bm{h}_e)$, with a default $\lambda = 0.5$. This is a single-pass propagation without training.
    *   Design Motivation: Inspired by HGNN (Feng et al., 2019), but simplified and requires no fine-tuning. It serves as a soft constraint for "semantic sharing on hyperedges"—standard BM25 or raw dense embeddings cannot see topic-level grouping, but after propagation, hitting one episode can pull other episodes of the same topic closer.

3.  **Coarse-to-fine Three-level Retrieval + RRF Fusion + Reranker Fine-ranking**:
    *   Function: Progressively narrows the search space to avoid noise explosion during direct ranking across tens of thousands of facts.
    *   Mechanism: Each stage follows a "BM25 + dense → RRF fusion → reranker" pipeline. The RRF formula $\text{RRF}(d) = \sum_{m=1}^M 1/(k + \text{rank}_m(d))$ sums the inverse ranks of different rankers, independent of original score scales. After selecting top-$k^T$ topics, episode candidates are pulled from topic hyperedges to rank top-$k^E$. Finally, facts are pulled from episode hyperedges to rank top-$k^F$. The final context equals top facts' `content` + top episodes' `summary`.
    *   Design Motivation: Directly retrieving across all facts loses topic coherence, while single-layer chunk RAG has low information density. Coarse-to-fine plus hyperedge expansion is equivalent to "locating the topic before finding evidence," matching human recall patterns. This is also key to token efficiency—HyperMem achieves 92.73% with 7.5× tokens, being much more efficient than GraphRAG’s 67.6% with 35.3× tokens.

### Loss & Training
This method is **unsupervised**. All node construction, hyperedge weighting, and boundary detection are performed by the LLM zero-shot (GPT-4.1-mini generates answers, while the Qwen3 series handles embedding/reranking). Hypergraph embedding propagation is a closed-form one-step forward pass requiring no trainable parameters. Results are averaged over 3 independent runs with hyperparameters $\lambda = 0.5, k^T = k^E = 10, k^F = 30$.

## Key Experimental Results

### Main Results (LoCoMo benchmark, LLM-as-judge accuracy %, judge = GPT-4o-mini)

| Method | Single-hop | Multi-hop | Temporal | Open Domain | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GraphRAG | 79.55 | 54.96 | 50.16 | 58.33 | 67.60 |
| LightRAG | 86.68 | 84.04 | 60.75 | 71.88 | 79.87 |
| HippoRAG 2 | 86.44 | 75.89 | 78.50 | 66.67 | 81.62 |
| HyperGraphRAG | 90.61 | 80.85 | 85.36 | 70.83 | 86.49 |
| Mem0 / Mem0g | 67.13 / 65.71 | 51.15 / 47.19 | 55.51 / 58.13 | 72.93 / 75.71 | 66.88 / 68.44 |
| MIRIX (GPT-4.1-mini) | 85.11 | 83.70 | 88.39 | 65.62 | 85.38 |
| MemOS | 81.09 | 67.49 | 75.18 | 55.90 | 75.80 |
| **HyperMem (Ours)** | **96.08** | **93.62** | **89.72** | 70.83 | **92.73** |

Overall SOTA, with Single-hop +5.5, Multi-hop +9.6, Temporal +1.3, and Overall +6.2 (vs. HyperGraphRAG). Open Domain remains constrained by issues related to "external dialogue knowledge."

### Ablation Study

| Configuration | Overall | $\Delta$ |
| :--- | :--- | :--- |
| HyperMem (Full) | **92.66** | – |
| w/o FC (Fact Context) | 91.75 | −0.91 |
| w/o EC (Episode Context) | 88.90 | **−3.76** |
| w/o TR (Topic Retrieval, starting from Episode) | 91.94 | −0.72 |
| w/o TR & FC | 91.75 | −0.91 |
| w/o TR & EC | 88.83 | −3.83 |
| w/o TR & ER (Fact-only retrieval, flattened hierarchy) | 90.19 | −2.47 |

### Key Findings
*   **Episode context is critical**: Removing EC results in a 3.76% drop overall and a 5.61% drop for Temporal questions—the temporal continuity of episodes is the backbone of cross-session reasoning, as atoms of facts lack time anchors.
*   **Completely flattening the hierarchy to Fact-only leads to a 5.68% drop in Multi-hop**, proving that coarse-to-fine retrieval effectively performs "early pruning + coherence preservation" rather than being just a hierarchical gimmick.
*   **Topic top-k sensitivity**: Performance is 76.88% at $k=1$ and 92.66% at $k=10$ (+15.78%), indicating that topic recall coverage is a performance bottleneck. Conversely, episode top-k is nearly insensitive (only 0.26% difference between k=10 and k=20), while fact top-k introduces noise after 30.
*   **Token Efficiency**: HyperMem achieves 92.73% with 7.5× tokens (with Mem0 as 1×). The "Fact Only" configuration already hits 89.48% with 2.5× tokens, compared to HyperGraphRAG which reaches 86.49% using 26.3× tokens—the token efficiency advantage of hypergraph organization is orders of magnitude.
*   **Case study**: For multi-hop questions like "How many tournaments did Nate win?" (spanning 7 sessions over 10 months), GraphRAG only answers "at least two" due to fragmented pairwise edges. HyperMem pulls all 7 episodes through a topic hyperedge to accurately answer "seven tournaments."

## Highlights & Insights
*   **"Explicit grouping via hyperedges" is a paradigm-level improvement to RAG**: Previous GraphRAG improvements focused on adding more refined entities/relations/paths, which remained pairwise. HyperMem changes the data structure directly, making "high-order joint dependency" a first-class citizen—a fundamental lift in RAG's first principles, evidenced by the +9.6 Multi-hop gain.
*   **Three-layer granularity (Topic / Episode / Fact) + Dual anchors (Time/Topic)** exactly corresponds to the cognitive levels of human dialogue memory. Topics are long-term semantics, episodes are event units, and facts are details. This categorization is theoretically supported by cognitive science (Anderson & Bower), explaining the 89.72% performance on Temporal questions.
*   **The `potential` field (facts predicting which queries they can answer)** is a practical trick. Performing reverse query alignment during the indexing phase adds a query-side semantic index to facts, improving query hit rates. This can be adapted for document expansion in search engines.
*   **The "Coarse-to-fine + RRF + Reranker" trio** is a highly reusable retrieval pipeline template. RRF resolves score scale issues between sparse and dense models, the reranker addresses precision, and the hierarchy manages candidate explosion.
*   **Hypergraph embedding propagation delivers performance without training**, proving that "topological information on the structure" itself is a strong signal, not necessarily requiring complex message passing like HGNN.

## Limitations & Future Work
*   The authors acknowledge the **single-user assumption**; multi-user/multi-agent scenarios require access control and memory isolation mechanisms. Open Domain performance remains weak (70.83%), necessitating integration with external knowledge bases.
*   **Total reliance on LLMs for boundary detection, topic aggregation, and fact extraction** involves high costs and latency for large-scale deployment. The paper does not provide detailed offline construction times.
*   **Hypergraph propagation is performed only once** ($\lambda = 0.5$, one step), a simplified version compared to multi-layer HGNN propagation. While the high-order structure might not be fully exploited, the authors position "simplicity" as a selling point and did not ablate multi-layer propagation.
*   **Reliance on GPT-4.1-mini for generation and GPT-4o-mini for evaluation** poses a risk of model bias. While comparing with MIRIX (also using GPT-4.1-mini) is relatively fair, other baselines evaluated by GPT-4o-mini might be affected by evaluation preferences.
*   **Incremental online update overhead is not shown**. The three algorithms appear to be batch offline constructions. However, actual conversation is streaming—must topics be rebuilt and embeddings re-propagated for every new episode?

## Related Work & Insights
*   **vs. HyperGraphRAG (Luo et al., 2025)**: HyperGraphRAG also uses hypergraphs but for multi-entity relations in static KBs. HyperMem adapts this for **dynamically evolving dialogue memory**, adding topic/episode/fact layers and coarse-to-fine retrieval tailored for agent memory.
*   **vs. RAPTOR / SiReRAG / HiRAG**: These use tree-structured multi-granularity indices where nodes are linked by pairwise parent-child edges. HyperMem uses hyperedges, allowing a fact to belong to multiple topics (overlapping hyperedges), which is more flexible.
*   **vs. Mem0 / A-Mem / Zep**: These systems primarily focus on graph-based persistence and fact evolution tracking via pairwise relations. HyperMem advances the structure further and significantly outperforms Mem0g (92.73% vs 68.44% on LoCoMo).
*   **vs. MIRIX**: MIRIX handles shared memory for multi-agent systems, while HyperMem focuses on high-order memory for a single agent. The two are orthogonal and could be combined (using HyperMem within agents and MIRIX for cross-agent sharing).
*   **Inspiration**: The hypergraph approach can be migrated to code repository indexing (one commit affecting multiple files = hyperedge), recommendation systems (one shopping list involving multiple items = hyperedge), or medical knowledge graphs (one symptom involving multiple diseases/tests = hyperedge).

## Rating
*   Novelty: ⭐⭐⭐⭐ Systematically introduces hypergraphs to dialogue memory for the first time. The combination of three-layer architecture and coarse-to-fine retrieval is a novel paradigm-level improvement.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison with 14 baselines across 4 types of questions, detailed ablation, hyperparameter sensitivity, efficiency analysis, and 4 types of case studies.
*   Writing Quality: ⭐⭐⭐⭐⭐ Very clear structure. Figure 1 clarifies the contribution at a glance; method/experiment/algorithm pseudocode are complete and easy to reproduce.
*   Value: ⭐⭐⭐⭐⭐ Achieves a significant SOTA (92.73%) on LoCoMo with superior token efficiency (7.5× vs. HyperGraphRAG 26.3×); highly applicable to dialogue agent products.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](../../ICLR2026/information_retrieval/amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)
- [\[ICML 2026\] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](../../ICML2026/information_retrieval/hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)
- [\[AAAI 2026\] Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](../../AAAI2026/information_retrieval/mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[AAAI 2026\] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](../../AAAI2026/information_retrieval/comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)

</div>

<!-- RELATED:END -->
