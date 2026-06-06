---
title: >-
  [Paper Note] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment
description: >-
  [CVPR 2026][Information Retrieval & RAG][Multimodal Retrieval] This paper proposes BRIDGE, a system that distills noisy multimodal queries into retrieval-optimized pure-text queries via FORGE (an RL-trained query alignme…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "Multimodal Retrieval"
  - "Query Alignment"
  - "Reinforcement Learning"
  - "Dense Retrieval"
  - "Query Rewriting"
date: 2026-05-08
content_hash: 08b6ba441d6d2bf5
---

# BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment

**Conference**: CVPR 2026
**arXiv**: [2604.07201](https://arxiv.org/abs/2604.07201)  
**Code**: [GitHub](https://github.com/mm-bright/multimodal-reasoning-retrieval)  
**Area**: Multimodal Retrieval / Reinforcement Learning
**Keywords**: Multimodal Retrieval, Query Alignment, Reinforcement Learning, Dense Retrieval, Query Rewriting

## TL;DR
This paper proposes BRIDGE, a system that distills noisy multimodal queries into retrieval-optimized pure-text queries via FORGE (an RL-trained query alignment model), paired with LENS, a reasoning-enhanced retriever. BRIDGE achieves 29.7 nDCG@10 on MM-BRIGHT, and as a plug-in further improves Nomic-Vision to 33.3, surpassing the best text-only retriever.

## Background & Motivation
**Background**: Dense retrieval is well-established in text-only settings (BEIR 59.0 nDCG@10), and multimodal encoders (CLIP, Nomic-Vision, VLM2Vec) are advancing, yet they perform poorly on reasoning-intensive multimodal retrieval.

**Limitations of Prior Work**: The MM-BRIGHT benchmark reveals a counterintuitive phenomenon — the best multimodal retriever, Nomic-Vision (27.6), underperforms even the best text-only retriever (32.2). Existing approaches focus on improving the retriever side (larger encoders, contrastive training, LLM reranking), yet all accept noisy queries as fixed input.

**Key Challenge**: The bottleneck lies in the query rather than the retriever — raw multimodal queries entangle image descriptions, conversational noise, and retrieval intent, systematically degrading embedding similarity. No visual encoding capability can compensate for poor query quality.

**Goal**: To restructure queries prior to retrieval, transforming them from "noisy multimodal inputs" into "retrieval-optimized pure-text queries."

**Key Insight**: Query-side alignment (rather than retriever-side improvement), with RL directly optimizing downstream retrieval quality.

**Core Idea**: The modality gap in multimodal retrieval is fundamentally a query representation problem rather than a model capability problem. FORGE learns via RL to "bridge" the user's multimodal expression and the input format required by the retriever.

## Method

### Overall Architecture
A three-stage pipeline: (1) GPT-4o converts query images into textual descriptions $\delta(q_v)$ → (2) FORGE distills the noisy query pair $(q_t, \delta(q_v))$ into a compact retrieval string $\hat{q}$ → (3) LENS encodes $\hat{q}$ and retrieves from the text corpus.

$$\hat{\mathcal{D}}_k = \text{LENS}(\text{FORGE}(q_t, \text{GPT-4o}(q_v)), \mathcal{C}, k)$$

### Key Designs

1. **FORGE (Focused Retrieval Query Generator)**:

    - Fine-tuned from Qwen2.5-7B-Instruct
    - Input: concatenated text question + image description; Output: retrieval-optimized search string of at most 200 words
    - Trained with GRPO reinforcement learning; reward function based on downstream retrieval quality:
    $r(\hat{q}, d^+) = \text{nDCG@}k(\text{LENS}(\hat{q}, \mathcal{C}), \{d^+\})$
    - Training loop: sample $G=8$ candidate queries → compute retrieval reward → GRPO gradient update
    - **Design Motivation**: Unlike supervised query rewriting, RL directly optimizes retrieval outcomes rather than imitating reference rewrites, allowing the model to freely explore optimal query strategies.

2. **LENS (Language-Enhanced Neural Search)**:

    - Dual-encoder dense retriever based on Qwen3-Embedding-4B
    - Fine-tuned on reasoning-intensive retrieval data (mathematics, science, medicine, law, software engineering)
    - InfoNCE loss + in-batch negatives + $M=7$ hard negatives
    - Cosine similarity retrieval: $\text{score}(\hat{q}, d_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{d_i}}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_{d_i}\|}$
    - **Design Motivation**: FORGE produces intent-rich structured queries that require a reasoning-capable retriever for effective matching.

3. **Visual Captioning**:

    - GPT-4o generates dense, domain-aware descriptions capturing object types, spatial relationships, and labels
    - Generated offline once and cached
    - **Design Motivation**: Grounds visual content in natural language, making it processable by text-only models.

### Loss & Training
- FORGE: GRPO training, lr=$1\times10^{-6}$, max 256 tokens, 3 epochs
- LENS: Contrastive learning, lr=$1\times10^{-5}$, batch 512, $\tau=0.02$, 3 epochs
- Trained on 4× H100 80GB GPUs

## Key Experimental Results

### Main Results (MM-BRIGHT, 2803 queries, 29 domains)

| Method | nDCG@10 | Type |
|--------|---------|------|
| CLIP | 10.8 | Multimodal Encoder |
| Nomic-Vision | 27.6 | Multimodal Encoder (best) |
| Stella-400M (text) | 32.2 | Text-only Retriever (best) |
| BRIDGE (FORGE+LENS) | **29.7** | Query Alignment System |
| FORGE + Nomic-Vision | **33.3** | Plug-in Mode |

### Ablation Study

| Configuration | nDCG@10 | Notes |
|---------------|---------|-------|
| LENS only (original query) | Lower | Noisy queries limit retriever |
| FORGE + general retriever | Medium | FORGE alignment effective, but retriever also matters |
| FORGE + LENS | **29.7** | Optimal combination |
| FORGE + Nomic-Vision | **33.3** | Demonstrates FORGE as a universal plug-in |
| GPT-4o query rewriting (non-RL) | Lower | RL training outperforms heuristic rewriting |

### Key Findings
- FORGE as a plug-in improves Nomic-Vision from 27.6 to 33.3 (+5.7), making a multimodal system surpass the best text-only retriever for the first time.
- BRIDGE comprehensively outperforms all multimodal encoder baselines across all 29 domains.
- No multimodal encoder is required at inference time — the system operates entirely in text space, making it lightweight, modular, and scalable.
- Validates the core argument: the bottleneck in multimodal retrieval is query representation, not model capability.

## Highlights & Insights
- The core insight is particularly profound — "fix the query rather than enhance the retriever" fundamentally challenges conventional thinking.
- FORGE's RL training directly targets retrieval outcomes during query optimization, avoiding error propagation from intermediate supervision.
- As a plug-and-play module compatible with arbitrary retrievers, it offers strong practical utility.
- Demonstrates that in certain settings, "understanding retrieval intent" matters more than "understanding image content."

## Limitations & Future Work
- Relies on GPT-4o for image captioning, introducing significant API cost and latency.
- FORGE is based on a 7B model, incurring higher inference overhead than directly encoding queries.
- Visual descriptions may lose fine-grained visual information (e.g., precise UI layouts).
- Future work may explore lightweight FORGE variants or end-to-end multimodal query encoders.

## Related Work & Insights
- DeepRetrieval pioneered RL-based query generation; FORGE extends this paradigm to the multimodal setting.
- Query expansion methods such as HyDE and Query2Doc generate pseudo-documents, whereas FORGE is guided by RL rewards.
- The MM-BRIGHT benchmark exposes the fundamental challenges of multimodal retrieval; this paper provides the first effective response.
- Insight: In many AI systems, "input quality" may be a greater bottleneck than "model capability."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The insight that "the query, not the retriever, is the bottleneck" is profound; the RL-trained query alignment approach is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 29 domains; plug-in mode validates generalizability, though more retriever combinations would strengthen the analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clear; system design logic is coherent and well-structured.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for multimodal retrieval with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](../../ICLR2026/information_retrieval/lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](../../ACL2026/information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)
- [\[ACL 2026\] Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning](../../ACL2026/information_retrieval/verbal-r3_verbal_reranker_as_the_missing_bridge_between_retrieval_and_reasoning.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
