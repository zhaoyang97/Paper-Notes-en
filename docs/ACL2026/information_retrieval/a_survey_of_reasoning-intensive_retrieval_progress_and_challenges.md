---
title: >-
  [Paper Note] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges
description: >-
  [ACL 2026][Information Retrieval & RAG][RIR] This paper systematically organizes the emerging direction of "Reasoning-Intensive Retrieval (RIR)." It provides the first comprehensive three-part survey—benchmarks, methods, and challenges—following the pipeline of query/index/retriever/reranker/iteration, and points out that current evaluations rely excessively on t
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - RIR
date: 2026-05-08
content_hash: c505d2979511ecb7
---
# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges

**Conference**: ACL 2026  
**arXiv**: [2605.00063](https://arxiv.org/abs/2605.00063)  
**Code**: None  
**Area**: Information Retrieval (Survey)  
**Keywords**: Reasoning-Intensive Retrieval, RIR, Reranking, Iterative Retrieval, LLM Embeddings

## TL;DR
This paper systematically organizes the emerging direction of "Reasoning-Intensive Retrieval (RIR)." It provides the first comprehensive three-part survey—benchmarks, methods, and challenges—following the pipeline of query/index/retriever/reranker/iteration, and points out that current evaluations rely excessively on traditional IR metrics like nDCG.

## Background & Motivation
**Background**: Traditional dense retrieval (DPR, Contriever, BGE, etc.) relies on semantic or lexical similarity between queries and documents, having achieved high performance in scenarios with high semantic overlap such as web search.

**Limitations of Prior Work**: In expert domains (medicine, law, mathematics, code) and deep research scenarios, a query and its correct evidence are often linked only through an implicit multi-hop reasoning chain. For example, a query like "Can water boiled from seawater be drunk?" requires realizing that "salt does not disappear when evaporated" to find the correct document. Simple similarity matching fails completely. This paper formally names such problems RIR.

**Key Challenge**: Current research faces two prominent issues: first, evaluations are highly heterogeneous, with benchmarks spanning code, math, and medicine, in which problem formats and data sources vary, making horizontal comparison impossible; second, methods are scattered across stages of the pipeline (query rewriting, retriever training, reranking, iterative RAG), lacking a unified taxonomic framework, which makes it difficult for researchers to choose a starting point.

**Goal**: Establish a unified roadmap for RIR—categorizing benchmarks by reasoning type, domain, and modality, and categorizing methods by "where and how reasoning is injected" into the pipeline, while identifying unresolved challenges.

**Key Insight**: The authors use "the position where reasoning intervenes in the retrieval pipeline" as the primary organizational axis. This is a more stable perspective than classification by model architecture or dataset, as new models emerge constantly while pipeline stages remain finite and stable.

**Core Idea**: Use a two-dimensional taxonomy (pipeline stage $\times$ reasoning injection method) to incorporate fragmented RIR research into a single framework, exposing genuine research gaps.

## Method

### Overall Architecture
The entire text serves as a structured roadmap for incorporating fragmented RIR studies into a unified framework. It unfolds along the primary axis of "at which step of the retrieval pipeline reasoning intervenes," covering the evaluation landscape (Section 3), method taxonomy (Section 4), and unsolved challenges (Section 5). On the evaluation side, 17 benchmarks are categorized into four buckets by "domain $\times$ modality": Open-Domain (e.g., BESPOKE, ImpliRet), Expert-Domain (MIRB, R2MED, CoIR, Bar Exam QA), Multi-Domain (BRIGHT, Bright-Plus, RAR-b), and Multimodal (MRMR, MR²-Bench, ARK). Each is labeled with five types of reasoning: deductive, analogical, causal, analytical, and numerical. On the methods side, studies are divided into four mutually exclusive buckets based on their pipeline position, followed by an inventory of gaps like obsolete nDCG and multimodal deficiencies.

### Key Designs

**1. Classification by Pipeline Position Rather than Model Architecture: Maintaining Taxonomy Stability for New Models**

The survey categorizes all RIR methods into four mutually exclusive buckets based on "which step of the retrieval pipeline reasoning functions in": Pre-Retrieval Augmentation, Reasoning-Aware Retriever Training, Reasoning-Enhanced Reranking, and Iterative Retrieval. Pre-Retrieval is further divided into query-side (query rewriting/decomposition, e.g., TongSearch-QR, ThinkQE, ReDI) and index-side (document expansion, e.g., SPIKE, EnrichIndex, LATTICE). Retriever Training focuses on backbone selection (LLM-based vs. Diffusion LM), hard negative curation (ReasonIR, DIVER, RaDeR), and training objectives (multi-task SFT + RL with format/embedding dual rewards). Rerankers evolve from Prompt-Tuning to SFT/Distillation to RL (Rank1, Rank-K, Rank-R1, ReasonRank). Iterative methods model the retrieval-reasoning alternation as a state machine (SMR) or RL policy. Choosing pipeline position as the skeleton allows researchers to quickly locate relevant work based on practical needs, such as wanting to work on the query side.

**2. Labeling Benchmarks by Five Reasoning Types: Exposing Domain Differences and Gaps**

The survey applies five reasoning types proposed by BRIGHT—deductive, analogical, causal, analytical, and numerical—as labels to each benchmark. Statistics show clear patterns: deductive reasoning (rule-to-case application) is most common in math/science/medicine/law; analogical reasoning is prominent in cross-language mapping for code/math; numerical reasoning often appears in daily time calculations; and causal/analytical reasoning is concentrated in troubleshooting and problem decomposition. This labeling allows researchers to see which benchmarks suit their methods and exposes gaps in types like multimodal causal reasoning.

**3. Scale-Reliability Tradeoff Perspective: Summarizing Benchmark Construction Tension**

There is a fundamental tension in benchmark construction: LLM synthesis (ScIRGen, ImpliRet) is scalable but prone to hallucinations, while human annotation (BRIGHT, Bar Exam QA) is reliable but costly. The survey positions 17 benchmarks along a 2D scale of "scale $\times$ annotation method," discovering that hybrid construction ("LLM generation followed by human review") is becoming mainstream. It suggests that future directions should follow the "synthesize then expert-verify" route, providing a quantifiable dimension for evaluating design tradeoffs.

### Loss & Training
The survey summarizes three major losses for RIR methods in Appendix E: InfoNCE (standard contrastive loss used by almost all retrievers), Generation Loss (for retrievers with "thought," where models like O1 Embedder use next-token prediction to learn intermediate reasoning), and MSE (distilling LLM-reasoned embeddings into student retrievers, e.g., Dense Reasoner). RL schemes (LREM, UME-R1, ReasonRank) combine generation-side rewards (format compliance, length control) and embedding-side rewards (retrieval precision), making the reasoning trajectory itself an optimizable object.

## Key Experimental Results
As this is a survey, the following summarizes the synthesized RIR benchmark and method comparison data.

### Main Results: Benchmark Landscape Comparison

| Benchmark | Area | Scale | Annotation |
|-----------|------|------|----------|
| BRIGHT | Multi-Domain | 1,384 | Hybrid |
| Bright-Plus | Multi-Domain | 1,384 | Hybrid |
| R2MED | Medical | 876 | Hybrid |
| MIRB | Math | 39,029 | Derived |
| CoIR | Code | ~162,000 | Derived |
| CoQuIR | Code | 42,725 | LLM-Automated |
| ScIRGen | Scientific | 61,376 | LLM-Automated |
| BESPOKE | Open Domain | 150 | Human-Curated |
| MRMR | Multi-Modal | 1,435 | Hybrid |
| MR²-Bench | Multi-Modal | 1,309 | Hybrid |

### Ablation Study: Characteristics of Four Method Categories

| Pipeline Stage | Representative Methods | Main Gain | Main Cost |
|---------------|-------------------------|-----------|-----------|
| Pre-Retrieval (query) | TongSearch-QR / ThinkQE | Strong queries via RL even with small models | Token overhead from multiple iterations |
| Pre-Retrieval (index) | EnrichIndex / LATTICE | Offline reasoning, cheap online inference | Index bloat, requires reconstruction |
| Retriever Training | ReasonIR / DIVER | Stronger end-to-end embeddings | Requires careful hard negative curation |
| Reranking | Rank1 / ReasonRank | Best performance on BRIGHT | High inference latency |
| Iterative | SMR / Vijay et al. | Handles complex multi-hop | Risk of "overthinking" |

### Key Findings
- **Multi-stage stacking is not always better**: Although iterative methods achieve SOTA on BRIGHT, they are prone to "overthinking" and drift, making them less stable than well-designed single-stage methods.
- **Specialized methods regress on general IR**: Retrievers trained on RIR benchmarks typically perform worse on general benchmarks like MTEB compared to general LLM embeddings like Gemini Embedding or Jina-V5.
- **Reasoning types determine method choice**: Deductive tasks benefit most from rerankers; numerical tasks require query decomposition to break problems into computable sub-problems.

## Highlights & Insights
- **Organizing methods by pipeline position rather than model architecture** is the most stable design of this survey. By using "where to inject reasoning" as the skeleton, the taxonomy remains compatible with future emerging models.
- **The five-category taxonomy for reasoning types** (deductive, analogical, causal, analytical, numerical) quantifies the difficulty of RIR tasks for the first time, allowing future benchmark designs to intentionally fill gaps (e.g., multimodal causal).
- **Explicitly stating that nDCG is obsolete**: The authors argue that the RIR era needs to include "efficiency" and "fine-grained relevance" in metrics, such as Peng et al.'s joint efficiency-effectiveness FLOPs evaluation or Weller et al.'s instruction-following metrics—serving as a wake-up call to the IR evaluation community.

## Limitations & Future Work
- **Author acknowledgment**: The survey only covers methods with experiments on public RIR benchmarks; other potential directions like HyDE and graph-based retrieval are not explored in depth; industrial private RIR systems are also excluded.
- **Additional limitations**: The taxonomy strictly segments by pipeline position, offering weak coverage of "cross-stage joint training" methods (e.g., end-to-end retriever-reranker co-training). The five reasoning types from BRIGHT may not cover future types like "counterfactual reasoning" or "program synthesis reasoning."
- **Improvement directions**: Establish dedicated "reasoning-faithful" evaluation metrics for RIR (examining not just top-$k$ hits, but whether retrieved evidence chains actually support the reasoning) and perform end-to-end evaluations in real downstream scenarios like deep research or long-term memory.

## Related Work & Insights
- **vs. RAG-Reasoning Survey** (Li 2025g): They view retrieval as a pre-step to support generation; this paper treats retrieval itself as the terminal task, emphasizing the retriever's own reasoning capabilities.
- **vs. Reasoning Agentic RAG Survey** (Liang 2025): They focus on how to schedule retrieval within an agent framework; this paper focuses on how reasoning integrates inside the retriever/reranker. Both are complementary.
- **vs. Classic IR Surveys** (Robertson & Zaragoza 2009; Yates 2021): Classic surveys center on semantic/lexical relevance; this survey defines "reasoning intervention in relevance modeling" as a new paradigm, marking the third wave of IR (following BM25 and dense retrieval).

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic RIR survey; the classification skeleton (pipeline position $\times$ reasoning type) is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 17 benchmarks and 30+ methods, with empirical analysis (LLM-based vs. LRM-based, computation cost vs. performance).
- Writing Quality: ⭐⭐⭐⭐ The 2D taxonomy is clear, and the tradeoff perspectives are well-articulated, though content density varies between sections, with the appendix containing more information than the main text.
- Value: ⭐⭐⭐⭐⭐ A must-read map for researchers entering the RIR field, clearly pointing out high-value directions such as the obsolescence of nDCG and multimodal gaps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ICLR 2026\] Beyond Sequential Reranking: Reranker-Guided Search Improves Reasoning Intensive Retrieval](../../ICLR2026/information_retrieval/beyond_sequential_reranking_reranker-guided_search_improves_reasoning_intensive_.md)
- [\[ICML 2026\] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](../../ICML2026/information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)

</div>

<!-- RELATED:END -->
