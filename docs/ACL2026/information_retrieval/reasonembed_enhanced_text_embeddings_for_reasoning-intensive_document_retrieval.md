---
title: >-
  [Paper Note] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Text Embeddings] ReasonEmbed introduces three technical innovations—the ReMixer non-trivial synthetic data method (82K high-quality samples)…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Text Embeddings"
  - "Reasoning-Intensive Retrieval"
  - "Synthetic Data"
  - "Adaptive Training"
  - "BRIGHT Benchmark"
date: 2026-05-08
content_hash: 8b74bc46f1b88bdf
---

# ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval

**Conference**: ACL 2026  
**arXiv**: [2510.08252](https://arxiv.org/abs/2510.08252)  
**Code**: [https://github.com/VectorSpaceLab/agentic-search/tree/main/ReasonEmbed](https://github.com/VectorSpaceLab/agentic-search/tree/main/ReasonEmbed)  
**Area**: Information Retrieval / Reasoning-Intensive Retrieval  
**Keywords**: Text Embeddings, Reasoning-Intensive Retrieval, Synthetic Data, Adaptive Training, BRIGHT Benchmark  

## TL;DR

ReasonEmbed introduces three technical innovations—the ReMixer non-trivial synthetic data method (82K high-quality samples), Redapter adaptive reasoning intensity weighted training, and multi-backbone implementations—significantly outperforming all existing text embedding models by approximately 10 points with an nDCG@10 of 38.1 on the BRIGHT benchmark.

## Background & Motivation

**Background**: With the rise of LLM-driven AI agents, many scenarios require retrieving information from external documents. Traditional retrieval (BM25, general embedding models) relies on keyword matching or shallow semantic matching, performing poorly on reasoning-intensive retrieval benchmarks like BRIGHT.

**Limitations of Prior Work**: (1) Scarcity of training data—existing retrieval datasets originate from traditional search scenarios, differing vastly from reasoning-intensive retrieval in query formats and domain knowledge; (2) Triviality issue in synthetic data—existing synthesis methods generate queries with over-direct relationships to documents (similar words, keyword overlap), allowing models to achieve high scores via surface matching; (3) Limited effectiveness of existing methods—pioneering works like ReasonIR yield only marginal improvements.

**Key Challenge**: Reasoning-intensive retrieval requires models to understand deep semantic relationships between queries and documents (requiring multi-step reasoning to determine relevance), but the triviality of existing synthetic data allows models to take shortcuts—learning surface patterns rather than reasoning capabilities.

**Goal**: To solve the triviality issue in synthetic data, design a reasoning intensity-aware training strategy, and build an efficient embedding model for reasoning-intensive retrieval.

**Key Insight**: The authors identify "triviality" as the core bottleneck—if the positive sample is the source document used to generate the query, both share extensive surface cues. By excluding source documents, mining candidates from independent retrieval, and filtering positive samples with reasoning-enhanced annotation, one can construct training data that truly requires reasoning for discrimination.

**Core Idea**: Eliminate triviality using a three-stage pipeline of "source document exclusion + candidate mining + reasoning annotation," and then adaptively adjust sample weights using reasoning intensity to prioritize learning from difficult samples that require deep reasoning.

## Method

### Overall Architecture

Three-stage data synthesis (ReMixer) $\rightarrow$ Reasoning intensity adaptive training (Redapter) $\rightarrow$ Multi-backbone implementation. Data synthesis starts from the 12 domain corpora of BRIGHT, generates conditional queries using Qwen2.5-72B, mines candidates with off-the-shelf retrievers (excluding source documents), and performs relevance annotation using a distilled Qwen3-8B reasoning annotator. Training continues on MSMARCO pre-trained checkpoints, optimized with the RI-InfoNCE loss.

### Key Designs

1. **ReMixer Data Synthesis (De-trivialization)**:

    - **Function**: Generates 82K high-quality, non-trivial training samples for reasoning-intensive retrieval.
    - **Mechanism**: Three stages—(1) Conditional query generation: Using Qwen2.5-72B to generate long reasoning-required queries from source documents, increasing diversity via sampling query lengths and user education levels; (2) Candidate mining with source document exclusion: Explicitly excluding the source document $d_q^*$, retrieving candidates $\mathcal{C}_q \leftarrow \text{Top-k}\{\phi(q,d) | D/d_q^*\}$ with off-the-shelf retrievers; (3) Reasoning-enhanced relevance annotation: Using a distilled reasoning LLM for three-stage annotation (query analysis $\rightarrow$ document analysis $\rightarrow$ relevance judgment) on a 1-5 point scale.
    - **Design Motivation**: Excluding source documents breaks the trivial query-document connection, forcing positive samples to be documents that are "different in form but essentially relevant," which the model must discover through reasoning.

2. **Redapter Adaptive Training**:

    - **Function**: Dynamically adjusts training weights based on sample reasoning intensity, focusing the model on learning difficult samples.
    - **Mechanism**: Reasoning intensity is defined as $\text{RI}_\theta(s) = \min(\mathcal{L}_{q,D} / \mathcal{L}_{q',D}, \kappa)$, where $q'$ is the reasoning-enhanced query. A larger ratio indicates that reasoning-based rewriting significantly helps retrieval, meaning the sample requires more reasoning for correct retrieval. During training, normalized reasoning intensity is used as the sample weight for the InfoNCE loss.
    - **Design Motivation**: Continuing training after simple samples quickly saturate is wasteful; difficult samples require more learning opportunities. Adaptive weighting tilts computational resources toward the most valuable samples.

3. **Multi-backbone Implementation**:

    - **Function**: Validates the generalizability of the method across different LLM backbones and scales.
    - **Mechanism**: Implemented ReasonEmbed on three backbones: Qwen3-4B, Qwen3-8B, and Llama-3.1-8B, all initialized from MSMARCO pre-trained checkpoints.
    - **Design Motivation**: To prove that performance gains originate from data and training strategies rather than a specific model.

### Loss & Training

RI-InfoNCE loss: $\mathcal{L}_{RI} = \sum_{s \in B} f(\text{RI}_\theta(s), B) \cdot \mathcal{L}_{q,D}$, where $f$ is an intra-batch reasoning intensity normalization function. The base loss is standard InfoNCE, including one positive sample and intra-batch negatives plus hard negatives. The annotator is distilled from the reasoning trajectories of Qwen3-235B into Qwen3-8B.

## Key Experimental Results

### Main Results (BRIGHT nDCG@10)

| Model | Scale | Avg nDCG@10 |
|------|------|-------------|
| BM25 | - | 14.5 |
| OpenAI-3-Large | - | 17.9 |
| gte-Qwen2-7B | 7B | 23.5 |
| ReasonIR-8B | 8B | 24.4 |
| DIVER-Retriever | 4B | 28.9 |
| **Ours-Qwen3-4B** | 4B | **37.1** |
| **Ours-Qwen3-8B** | 8B | **38.1** |

### Ablation Study

| Configuration | Avg nDCG@10 | Description |
|------|-------------|------|
| Qwen3-8B Base InfoNCE | 37.1 | Using ReMixer data only |
| Qwen3-8B + Redapter | **38.1** | +1.0 from adaptive weighting |
| Qwen3-8B-ms (MSMARCO only) | 18.7 | No synthetic data |

### Key Findings

- Ours-Qwen3-4B (37.1) already surpasses all existing models, 8.2 points higher than the Prev. SOTA DIVER (28.9).
- ReMixer data is the primary source of contribution—improving from 18.7 to 37.1 (+18.4 Gain), with Redapter providing an additional +1.0 Gain.
- Consistently leading significantly across all 12 subtasks, with the largest gains in StackExchange-like tasks (requiring domain reasoning) and Coding tasks (requiring code reasoning).
- The Llama-3.1-8B backbone is equally effective (36.2), proving the method does not rely on a specific model.
- De-trivialization is core—the performance of models trained using source documents directly as positive samples is far below that of ReMixer.

## Highlights & Insights

- The proposal and verification of the "triviality" concept are highly valuable—revealing the fundamental flaw in existing synthetic data methods. The simple operation of "excluding source documents and independently mining candidates" brings a huge improvement, demonstrating that data quality is far more important than quantity.
- Clever definition of reasoning intensity—quantifying the "extent to which reasoning aids retrieval" by the ratio of loss changes after reasoning-based query rewriting, without extra annotation, and allowing dynamic calculation during training.
- The approach of distilling a reasoning LLM into a lightweight annotator balances annotation quality and cost.

## Limitations & Future Work

- Evaluation is primarily on the BRIGHT benchmark, which may lead to overfitting on the characteristics of this specific benchmark.
- Synthetic data originates from 12 source corpora in BRIGHT, with limited domain coverage.
- Redapter's contribution (+1.0) is relatively small compared to ReMixer (+18.4), the value of the adaptive strategy requires more verification.
- The selection of the reasoning intensity threshold $\kappa$ relies on empirical experience.

## Related Work & Insights

- **vs ReasonIR**: ReasonIR uses scientific corpora to synthesize long queries and hard negatives but does not solve the triviality issue (24.4). Ours completely resolves triviality through source document exclusion (38.1), a Gain of 13.7 points.
- **vs DIVER**: DIVER employs more complex retrieval-augmented generation (28.9), but still suffers from triviality. Ours demonstrates that fundamental improvements in data quality are more effective than methodological complexity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Identification and solution of the triviality problem are novel; reasoning intensity adaptive training is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 subtasks, multiple backbones, complete ablation, huge improvement margin.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, precise problem definition.
- **Value**: ⭐⭐⭐⭐⭐ Reaches a historic high on BRIGHT (+10 points), providing a significant push for the field of reasoning-intensive retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges](a_survey_of_reasoning-intensive_retrieval_progress_and_challenges.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ACL 2026\] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](../../AAAI2026/information_retrieval/prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)

</div>

<!-- RELATED:END -->
