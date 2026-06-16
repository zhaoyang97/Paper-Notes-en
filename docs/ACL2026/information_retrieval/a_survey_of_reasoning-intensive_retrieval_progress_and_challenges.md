---
title: >-
  [Paper Note] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges
description: >-
  [ACL 2026][Information Retrieval & RAG][RIR] This paper systematically reviews the emerging direction of "Reasoning-Intensive Retrieval (RIR)," providing the first comprehensive benchmark-method-challenge three-part survey following the query/index/retriever/reranker/iterative pipeline, while noting that existing evaluations over-rely on traditional IR metrics su
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - RIR
date: 2026-05-08
content_hash: e6d6958f78c9d092
---
# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges

**Conference**: ACL 2026  
**arXiv**: [2605.00063](https://arxiv.org/abs/2605.00063)  
**Code**: None  
**Area**: Information Retrieval (Survey)  
**Keywords**: Reasoning-Intensive Retrieval, RIR, Reranking, Iterative Retrieval, LLM Embedding

## TL;DR
This paper systematically reviews the emerging direction of "Reasoning-Intensive Retrieval (RIR)," providing the first comprehensive benchmark-method-challenge three-part survey following the query/index/retriever/reranker/iterative pipeline, while noting that existing evaluations over-rely on traditional IR metrics such as nDCG.

## Background & Motivation
**Background**: Traditional dense retrieval (DPR, Contriever, BGE, etc.) relies on semantic or lexical similarity between the query and documents, achieving high performance in scenarios with high semantic overlap like web search.

**Limitations of Prior Work**: In expert domains (Medicine, Law, Mathematics, Code) and deep research scenarios, a query and the correct evidence are often connected only via an implicit multi-hop reasoning chain. For example, a query like "Can water boiled from seawater be drunk?" requires reasoning that "salt does not disappear during evaporation" to find the correct document. Simple similarity matching fails completely. This paper formally names this category of problems RIR.

**Key Challenge**: Existing research faces two prominent issues: first, evaluations are highly heterogeneous, with benchmarks spanning code, math, and medicine with varying question formats and data sources, making horizontal comparison impossible; second, methods are scattered across different stages of the pipeline (query rewriting, retriever training, reranking, iterative RAG), lacking a unified taxonomic framework, which makes it difficult for researchers to choose a starting point.

**Goal**: Establish a unified roadmap for RIR—categorizing benchmarks by reasoning type/domain/modality, and grouping methods by "where in the pipeline and how reasoning is injected," while identifying unresolved challenges.

**Key Insight**: The authors use "the position where reasoning intervenes in the retrieval pipeline" as the primary organizational axis—a more stable perspective than classification by model architecture or dataset, as the pipeline stages are finite and stable despite the constant emergence of new models.

**Core Idea**: Use a two-dimensional taxonomy (pipeline stage $\times$ reasoning injection method) to incorporate fragmented RIR research into a single framework, thereby exposing genuine research gaps.

## Method

### Overall Architecture
The entire paper is a structured roadmap incorporating fragmented RIR research into a single framework, organized along the primary axis of "at which step of the retrieval pipeline reasoning interveners." it covers the evaluation landscape (Section 3), methodology taxonomy (Section 4), and unresolved challenges (Section 5). On the evaluation side, 17 benchmarks are categorized into four buckets by "domain $\times$ modality": Open-Domain (e.g., BESPOKE, ImpliRet), Expert-Domain (MIRB, R2MED, CoIR, Bar Exam QA), Multi-Domain (BRIGHT, Bright-Plus, RAR-b), and Multimodal (MRMR, MR²-Bench, ARK). Each benchmark is labeled with five reasoning types: deductive, analogical, causal, analytical, and numerical. On the methodology side, methods are segmented into four mutually exclusive buckets based on their pipeline position.

### Key Designs

**1. Classifying methods by pipeline position rather than model architecture: Keeping the taxonomy stable for new models.**

The survey categorizes all RIR methods into four mutually exclusive buckets based on "where reasoning plays a role in the retrieval pipeline": Pre-Retrieval Augmentation, Reasoning-Aware Retriever Training, Reasoning-Enhanced Reranking, and Iterative Retrieval. Pre-Retrieval is further divided into query-side (query rewriting/decomposition, e.g., TongSearch-QR, ThinkQE, ReDI) and index-side (document expansion, e.g., SPIKE, EnrichIndex, LATTICE). Retriever Training focuses on backbone selection (LLM-based vs. Diffusion LM), hard negative curation (ReasonIR, DIVER, RaDeR), and training objectives (multi-task SFT + RL, format/embedding dual rewards). Rerankers evolve from Prompt-Tuning to SFT/Distillation to RL (Rank1, Rank-K, Rank-R1, ReasonRank). Iterative methods model alternating retrieval-reasoning as state machines (SMR) or RL policies.

**2. Labeling benchmarks by five reasoning types: Exposing reasoning requirement differences and gaps across domains.**

The survey applies five reasoning types proposed by BRIGHT—deductive, analogical, causal, analytical, and numerical—as labels to each benchmark. Statistical analysis reveals clear patterns: deductive reasoning (rule-to-case application) is most common in Math/Science/Medicine/Law; analogical reasoning is prominent in cross-language mapping for Code/Math; numerical reasoning is common in daily time calculations; and causal/analytical reasoning is concentrated in troubleshooting and problem decomposition. This labeling allows researchers to identify which benchmarks suit their methods and exposes gaps in reasoning types like multimodal causality.

**3. Scale-Reliability trade-off perspective: Summarizing the fundamental tension in benchmark construction.**

There is an inherent conflict in benchmark construction: LLM synthesis (ScIRGen, ImpliRet) is scalable but prone to hallucinations, whereas human annotation (BRIGHT, Bar Exam QA) is reliable but costly. The survey positions 17 benchmarks along a 2D "Scale $\times$ Annotation Style" axis, finding that "LLM generation followed by human audit" is becoming the mainstream hybrid approach. It argues that future benchmarks should follow the "synthesize-then-expert-verify" route.

### Loss & Training
The survey summarizes three major losses for RIR methods in Appendix E: InfoNCE (standard contrastive loss used by almost all retrievers), Generation Loss (for retrievers with "thought" capabilities, such as O1 Embedder using next-token prediction to learn intermediate reasoning), and MSE (distilling LLM-reasoned embeddings into student retrievers, e.g., Dense Reasoner). RL schemes (LREM, UME-R1, ReasonRank) combine generation-side rewards (format compliance, length control) with embedding-side rewards (retrieval accuracy), making the reasoning trajectory itself an optimizable object.

## Key Experimental Results

### Main Results: Benchmark Landscape Comparison

| Benchmark | Domain | Scale | Annotation Method |
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

### Ablation Study: Characteristics of the Four Method Categories

| Pipeline Stage | Representative Methods | Main Benefit | Main Cost |
|---------------|----------|----------|----------|
| Pre-Retrieval (query) | TongSearch-QR / ThinkQE | Small models can rewrite strong queries via RL | Multi-round iteration increases token overhead |
| Pre-Retrieval (index) | EnrichIndex / LATTICE | Reasoning is offline; cheap online inference | Index bloat; requires rebuilding |
| Retriever Training | ReasonIR / DIVER | Stronger end-to-end embeddings | Requires carefully curated hard negatives |
| Reranking | Rank1 / ReasonRank | Best performance on BRIGHT | High inference latency |
| Iterative | SMR / Vijay et al. | Handles complex multi-hop | Risk of "overthinking" |

### Key Findings
- **Multi-stage stacking is not always better**: While iterative methods achieve SOTA on BRIGHT, they are prone to "overthinking" and drift, often proving less stable than well-designed single-stage methods.
- **Specialized methods regress on general IR**: Retrievers trained on RIR benchmarks typically perform worse on general benchmarks like MTEB compared to general LLM embeddings like Gemini Embedding or Jina-V5.
- **Reasoning type determines method choice**: Deductive tasks benefit most from rerankers; numerical tasks require query decomposition to break problems into calculable sub-problems.

## Highlights & Insights
- **Organizing methods by pipeline position rather than model architecture** is the most robust design of this survey. Using "where reasoning is injected" as the taxonomic backbone ensures compatibility with future models.
- **The five-fold classification of reasoning types** (Deductive/Analogical/Causal/Analytical/Numerical) quantifies the difficulty of RIR tasks for the first time, allowing future benchmark designs to target specific gaps (e.g., multimodal causality).
- **Explicitly stating that nDCG is obsolete**: The authors argue that the RIR era requires metrics that incorporate "efficiency" and "fine-grained relevance," such as the efficiency-effectiveness FLOPs joint evaluation by Peng et al., or instruction-following metrics by Weller et al.

## Limitations & Future Work
- **Author Acknowledgments**: The survey only covers methods evaluated on public RIR benchmarks; other potential directions like HyDE and graph-based retrieval are not explored in depth; industrial private RIR systems are also excluded.
- **Additional Limitations**: The taxonomy strictly segments by pipeline position, providing weaker coverage for "cross-stage joint training" methods (e.g., end-to-end retriever-reranker co-training). The five reasoning types from BRIGHT may not cover emerging types like "counterfactual reasoning" or "program synthesis reasoning."
- **Future Directions**: Establish RIR-specific "reasoning-faithful" metrics (looking not just at top-k hits, but whether retrieved evidence chains actually support the reasoning) and conduct end-to-end evaluations in real downstream scenarios like deep research or long-term memory.

## Related Work & Insights
- **vs. RAG-Reasoning Survey** (Li 2025g): They treat retrieval as a pre-processing step to support generation; this paper treats retrieval itself as the end task, emphasizing the retriever's own reasoning capabilities.
- **vs. Reasoning Agentic RAG Survey** (Liang 2025): They focus on how to schedule retrieval within an agent framework; this paper focuses on how reasoning integrates into the internal mechanisms of transporters/rerankers.
- **vs. Classic IR Survey** (Robertson & Zaragoza 2009; Yates 2021): Classic surveys center on semantic/lexical relevance; this survey marks the entry of IR into a third wave (following BM25 and dense retrieval) by defining "reasoning-intervened relevance modeling" as a new paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic RIR survey; the classification backbone (pipeline position $\times$ reasoning type) is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 17 benchmarks and 30+ methods, with empirical analysis (LLM-based vs. LRM-based, computation cost vs. performance).
- Writing Quality: ⭐⭐⭐⭐ The 2D taxonomy is clear, and the trade-off perspective is insightful; however, content density varies across sections, with the appendix containing more information than the main text.
- Value: ⭐⭐⭐⭐⭐ An essential introductory map for researchers entering the RIR field, clearly identifying the obsolescence of nDCG and the multimodal gap as high-value directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ICML 2026\] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](../../ICML2026/information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)

</div>

<!-- RELATED:END -->
