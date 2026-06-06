---
title: >-
  [Paper Note] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges
description: >-
  [ACL 2026][Information Retrieval & RAG][Reasoning-Intensive Retrieval] This paper systematically reviews the emerging direction of "Reasoning-Intensive Retrieval (RIR)…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Reasoning-Intensive Retrieval"
  - "RIR"
  - "Reranking"
  - "Iterative Retrieval"
  - "LLM Embeddings"
date: 2026-05-08
content_hash: dbf7621d0d28c75b
---

# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges

**Conference**: ACL 2026  
**arXiv**: [2605.00063](https://arxiv.org/abs/2605.00063)  
**Code**: None  
**Area**: Information Retrieval (Survey)  
**Keywords**: Reasoning-Intensive Retrieval, RIR, Reranking, Iterative Retrieval, LLM Embeddings

## TL;DR
This paper systematically reviews the emerging direction of "Reasoning-Intensive Retrieval (RIR)," providing the first comprehensive three-part survey covering benchmarks, methods, and challenges along the query/index/retriever/reranker/iterative pipeline. It highlights that existing evaluations over-rely on traditional IR metrics like nDCG.

## Background & Motivation
**Background**: Traditional dense retrieval (DPR, Contriever, BGE, etc.) relies on semantic or lexical similarity between queries and documents, achieving high performance in scenarios with significant semantic overlap, such as web search.

**Limitations of Prior Work**: in expert domains (medicine, law, mathematics, code) and deep research scenarios, a query and the correct evidence are often connected only through an implicit multi-hop reasoning chain. For example, a query like "Can water boiled with seawater be drunk?" requires reasoning that "salt does not disappear during evaporation" to find the correct document. Simple similarity matching fails completely. This paper formally names such problems RIR.

**Key Challenge**: existing research faces two prominent issues: first, evaluations are highly heterogeneous, with benchmarks spanning code, math, and medicine, featuring different problem formats and data sources, making horizontal comparison impossible; second, methods are scattered across different stages of the pipeline (query rewriting, retriever training, reranking, iterative RAG), lacking a unified taxonomic framework, making it difficult for researchers to select a starting point.

**Goal**: To establish a unified roadmap for RIR—categorizing benchmarks by reasoning type/domain/modality, categorizing methods by "where they intervene in the pipeline and how they inject reasoning," and identifying unresolved challenges.

**Key Insight**: The authors use the "position of reasoning intervention in the retrieval pipeline" as the primary organizational axis—a more stable perspective than classification by model architecture or dataset, as model iterations are frequent but pipeline stages remain relatively constant.

**Core Idea**: To consolidate fragmented RIR research into a single framework using a two-dimensional taxonomy (pipeline stage × reasoning injection method), thereby exposing genuine research gaps.

## Method

### Survey Structure and Taxonomy
The paper provides a structured RIR roadmap consisting of three major blocks: **Evaluation Landscape** (Section 3), **Method Taxonomy** (Section 4), and **Open Challenges** (Section 5).

**Benchmark Landscape**: The authors organize 17 RIR benchmarks into four categories based on "Domain × Modality":
1. **Open-Domain** (e.g., BESPOKE, ImpliRet), focusing on inferring implicit user intent from chat history;
2. **Expert-Domain** (Science/Law/Medicine/Code), such as MIRB, R2MED, CoIR, CoQuIR, and Bar Exam QA, emphasizing specialized knowledge + deductive reasoning;
3. **Multi-Domain** (BRIGHT, Bright-Plus, RAR-b), a cross-domain mixture;
4. **Multimodal** (MRMR, MR²-Bench, ARK, MM-Bright), introducing joint image-text reasoning. Each benchmark is further labeled with five reasoning types: deductive, analogical, causal, analytical, and numerical.

### Key Designs
1. **Classification by "Pipeline Stage"** (Section 4 Axis):
    - **Function**: Categorizes all RIR methods into four mutually exclusive buckets based on "which step of the retrieval pipeline reasoning occurs": **Pre-Retrieval Augmentation**, **Reasoning-Aware Retriever Training**, **Reasoning-Enhanced Reranking**, and **Iterative Retrieval**.
    - **Mechanism**: Pre-Retrieval is further divided into query-side (query rewriting/decomposition, e.g., TongSearch-QR, ThinkQE, ReDI) and index-side (document expansion, e.g., SPIKE, EnrichIndex, LATTICE). Retriever Training focuses on backbone selection (LLM-based vs. Diffusion LM), data curation (hard negative mining in ReasonIR, DIVER, RaDeR), and training objectives (multi-task SFT + RL with dual rewards for format/embedding). Rerankers evolve from Prompt-Tuning → SFT/Distillation → RL (Rank1, Rank-K, Rank-R1, ReasonRank). Iterative methods treat retrieval-reasoning as a state machine (SMR) or RL policy.
    - **Design Motivation**: Using pipeline stages instead of architectures keeps the taxonomy stable against future model developments and helps researchers quickly locate work relevant to their specific stage of interest.

2. **Benchmarking by "Reasoning Type"** (Section 3.2):
    - **Function**: Applies the five reasoning types proposed by BRIGHT (deductive, analogical, causal, analytical, numerical) as tags to each benchmark to reveal varying domain requirements.
    - **Mechanism**: Statistical observation shows deductive reasoning is most prevalent in math/science/medicine/law (rule-to-case); analogical reasoning is prominent in code/math cross-lingual mapping; numerical reasoning is common in daily scenarios; causal/analytical reasoning is concentrated in troubleshooting and problem decomposition.
    - **Design Motivation**: This labeling allows researchers to match methods to benchmarks and exposes gaps in specific reasoning types (e.g., multimodal causal reasoning).

3. **Scale-Reliability Trade-off Perspective** (Section 3.2):
    - **Function**: Reveals fundamental tensions in benchmark construction—LLM synthesis (e.g., ScIRGen, ImpliRet) is scalable but prone to hallucination; human annotation (e.g., BRIGHT, Bar Exam QA) is reliable but costly.
    - **Mechanism**: The authors map 17 benchmarks on a "Size × Annotation Type" axis, finding that hybrid construction (LLM generation followed by human review) is becoming the dominant trend.
    - **Design Motivation**: Summarizes core trade-offs in benchmark design using quantifiable dimensions rather than subjective judgments.

### Loss & Training
Appendix E summarizes three major loss functions for RIR: **InfoNCE** (standard contrastive loss used by most retrievers), **Generation Loss** (for retrievers with "thought" chains, e.g., O1 Embedder using next-token prediction), and **MSE** (for distilling LLM-reasoned embeddings into student retrievers, e.g., Dense Reasoner). RL schemes (LREM, UME-R1, ReasonRank) employ weighted combinations of generation-side rewards (format compliance, length control) and embedding-side rewards (retrieval accuracy), making the reasoning trajectory itself optimizable.

## Key Experimental Results

### Main Results: Benchmark Landscape Comparison

| Benchmark | Domain | Scale | Annotation Type |
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

### Ablation Study: Characteristics of Method Categories

| Pipeline Stage | Representative Methods | Key Gain | Main Cost |
|---------------|----------|----------|----------|
| Pre-Retrieval (query) | TongSearch-QR / ThinkQE | RL can rewrite strong queries with small models | Increased token cost via multi-turn iterations |
| Pre-Retrieval (index) | EnrichIndex / LATTICE | Offline reasoning; cheap online inference | Index storage bloat; requires rebuilding |
| Retriever Training | ReasonIR / DIVER | Stronger end-to-end embeddings | Requires curated hard negatives |
| Reranking | Rank1 / ReasonRank | Best performance on BRIGHT | High inference latency |
| Iterative | SMR / Vijay et al. | Handles complex multi-hop | Risk of "overthinking" |

### Key Findings
- **Multi-stage stacking is not always better**: While iterative methods achieve SOTA on BRIGHT, they are prone to "overthinking" and drift, often proving less stable than well-designed single-stage methods.
- **Specialized methods regress on general IR**: Retrievers trained on RIR benchmarks usually perform worse on general benchmarks like MTEB compared to general-purpose LLM embeddings like Gemini Embedding or Jina-V5.
- **Reasoning types dictate method choice**: Deductive tasks benefit most from rerankers; numerical tasks require query decomposition to split problems into calculable sub-problems.

## Highlights & Insights
- **Organization by pipeline position rather than architecture** is the most robust design choice, ensuring the taxonomy remains compatible with future models.
- **The five-category reasoning taxonomy** (deductive, analogical, causal, analytical, numerical) quantifies RIR task difficulty for the first time, identifying gaps for future benchmark design (e.g., multimodal causal reasoning).
- **Explicit critique of nDCG**: Ours argues that the RIR era requires metrics incorporating efficiency and fine-grained relevance, such as the efficiency-effectiveness FLOPs joint evaluation by Peng et al., or instruction-following metrics by Weller et al.—serving as a wake-up call for the IR evaluation community.

## Limitations & Future Work
- **Authors acknowledge**: The survey only covers methods with experiments on public RIR benchmarks; other potential directions like HyDE and graph-based retrieval are not explored in depth. Private industrial RIR systems are also excluded.
- **Additional Limitations**: The taxonomy strictly follows pipeline stages, which weakly covers "cross-stage joint training" methods (e.g., end-to-end retriever-reranker co-training). The five reasoning types from BRIGHT might not encompass future types like "counterfactual reasoning" or "program synthesis reasoning."
- **Future Directions**: Establishing "reasoning-faithful" evaluation metrics for RIR (beyond top-k hits to whether the retrieved evidence chain actually supports the reasoning) and conducting end-to-end assessments in real-world scenarios like deep research or long-term memory.

## Related Work & Insights
- **vs. RAG-Reasoning Survey** (Li 2025g): They treat retrieval as a precursor to generation; Ours views retrieval itself as the terminal task, emphasizing the retriever's intrinsic reasoning capabilities.
- **vs. Reasoning Agentic RAG Survey** (Liang 2025): They focus on how agents schedule retrieval; Ours focuses on how reasoning is integrated into the retriever/reranker, making the two complementary.
- **vs. Classical IR Surveys** (Robertson & Zaragoza 2009; Yates 2021): Classical surveys center on semantic/lexical relevance. This survey treats "reasoning-mediated relevance modeling" as a new paradigm, marking the third wave of IR (following BM25 and dense retrieval).

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic RIR survey; taxonomy (pipeline stage × reasoning type) is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 17 benchmarks and 30+ methods with empirical analysis (LLM-based vs. LRM-based, cost vs. performance).
- **Writing Quality**: ⭐⭐⭐⭐ Clear two-dimensional taxonomy and trade-off perspectives; however, density varies between chapters, and the appendix contains more information than the main text.
- **Value**: ⭐⭐⭐⭐⭐ Essential roadmap for researchers entering RIR, highlighting high-value directions such as the obsolescence of nDCG and multimodal gaps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ACL 2026\] Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning](verbal-r3_verbal_reranker_as_the_missing_bridge_between_retrieval_and_reasoning.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)

</div>

<!-- RELATED:END -->
