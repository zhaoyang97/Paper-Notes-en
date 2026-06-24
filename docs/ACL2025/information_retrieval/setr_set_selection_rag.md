---
title: >-
  [Paper Note] SetR: Shifting from Ranking to Set Selection for Retrieval Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] SetR is proposed to shift the document ranking paradigm in RAG to a set selection paradigm. By using CoT reasoning to identify the information needs of queries and select the optimal document set, SetR significantly improves multi-hop QA performance while utilizing fewer documents (an average of 2.91 vs. 5).
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Set Selection"
  - "Reranking"
  - "Chain-of-Thought"
  - "Multi-hop QA"
  - "Information Need Identification"
date: 2026-05-08
content_hash: fe7f52d0de18df3a
---

# SetR: Shifting from Ranking to Set Selection for Retrieval Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2507.06838](https://arxiv.org/abs/2507.06838)  
**Code**: [LGAI-Research/SetR](https://github.com/LGAI-Research/SetR)  
**Area**: Information Retrieval / RAG  
**Keywords**: RAG, Set Selection, Reranking, Chain-of-Thought, Multi-hop QA, Information Need Identification  

## TL;DR

SetR is proposed to shift the document ranking paradigm in RAG to a set selection paradigm. By using CoT reasoning to identify the information needs of queries and select the optimal document set, SetR significantly improves multi-hop QA performance while utilizing fewer documents (an average of 2.91 vs. 5).

## Background & Motivation

- **Problem Definition**: The retrieval module of a RAG system must ensure that the retrieved passages are not only individually relevant but also collectively constitute a complete information set to correctly answer complex questions.
- **Limitations of Prior Work**: Existing reranking methods rank passages by scoring their individual relevance and then selecting the top-k, which poses three core issues: (1) ignoring information complementarity between passages, potentially retrieving redundant content; (2) failing to guarantee the completeness of information coverage required for multi-hop questions; (3) requiring manual tuning of the top-k parameter.
- **Design Motivation**: The information needs of RAG systems are fundamentally different from those of search engines — search engines rank individual results, whereas RAG requires a set of passages to collectively support generation. The paradigm should shift from "ranking" to "set selection".
- **Core Contributions**: (1) Proposing a set-based passage selection paradigm based on information need identification; (2) training and open-sourcing the SetR model for efficient selection; (3) comprehensively outperforming commercial and open-source rerankers on multi-hop RAG benchmarks.

## Method

### Overall Architecture

SetR workflow: First-stage retriever (BM25/bge) returns top-20 candidate passages $\rightarrow$ SetR analyzes information needs of queries via CoT reasoning $\rightarrow$ SetR selects the optimal subset that covers all information needs from candidates (without ranking or a fixed k value).

### Key Designs

1. **Information Need Identification (IRI)**: A structured CoT reasoning strategy with a three-step process: (a) enumerating key information needs required to answer the question; (b) identifying candidate passages containing relevant information for each need; (c) selecting a subset of passages that collectively provide the most comprehensive coverage. Unlike general CoT reasoning, IRI explicitly models the complete coverage of information needs.
2. **Model Distillation**: Using GPT-4o as a teacher to perform zero-shot set selection annotation on 40K MS MARCO queries, which is then distilled into the Llama-3.1-8B-Instruct student model. Training is conducted for 5 epochs with an effective batch size of 512, a learning rate of $5 \times 10^{-6}$, and the AdamW optimizer.
3. **Adaptive Number of Passages**: The model dynamically decides how many passages to select — with an average of only 2.63-3.41 passages (vs. fixed 5 in baselines), reducing noise interference while improving information precision.

### Three Model Variants

- **SetR-Selection only**: Only outputs selection results without the reasoning process.
- **SetR-CoT**: Conducts CoT reasoning using a general "Let's think step-by-step" prompt.
- **SetR-CoT & IRI**: The full model, featuring CoT reasoning with structured information need identification.

## Experiments

### Main Results (End-to-End QA)

Using bge-large-en-v1.5 as the first-stage retriever and Llama-3.1-8B as the generator:

| Model | No. of Passages | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue EM/F1 | MHRAG Acc |
|------|-------|---------------|------------|--------------|-----------|
| BM25 only | 5.00 | 30.07/30.97 | 31.17/25.22 | 7.44/10.78 | 41.82 |
| bge-reranker-large | 5.00 | 32.48/33.24 | 31.92/25.47 | 8.06/12.50 | 43.50 |
| RankGPT (gpt-4o) | 5.00 | 33.85/34.45 | 34.36/28.06 | 9.43/13.25 | 45.69 |
| **SetR-CoT & IRI** | **2.91** | **36.62/38.11** | **35.44/30.35** | **10.79/15.43** | **47.14** |

### Ablation Study

| Variant | HotpotQA F1 | MHRAG Acc | Description |
|------|------------|-----------|------|
| SetR-Selection only | 37.84 | 46.20 | Outperforms all reranking baselines even without reasoning |
| SetR-CoT | 38.20 | 45.26 | General CoT is inferior to IRI |
| **SetR-CoT & IRI** | **38.11** | **47.14** | Explicit information need analysis from IRI achieves optimal performance |

### Information Coverage and Robustness Analysis (MultiHopRAG)

| Metric | Reranking Baseline | SetR |
|------|-------------|------|
| Hit@k | 48.87% | **69.90%** (+21.03%) |
| Information Coverage Rate | 19.33% | **36.49%** (+17.16%) |
| Precision@5 | 0.1799 (RankGPT Best) | **0.2268** (+26.1%) |

### Key Findings

1. **Less is More**: SetR comprehensively outperforms all baselines using 5 passages while utilizing only an average of 2.91 passages.
2. **Significant Improvement in Information Coverage**: Hit@k and information coverage are improved by 21% and 17% respectively, whereas traditional reranking only yields an approximate 10% improvement.
3. **IRI vs. General CoT**: The improvement from information need identification does not stem from simple CoT reasoning capabilities, but rather from task-specific structured analysis.
4. **Contradiction between Precision and Rank Metrics**: SetR significantly leads in terms of Precision but shows slightly lower MRR/NDCG, exposing the limitations of traditional ranking metrics in multi-hop scenarios.
5. **Increasing the Number of Passages Degrades Baseline Performance**: More passages introduce more noise and conflicting information; the selective strategy of SetR successfully avoids this issue.

## Highlights

- Paradigm Innovation: Shifting RAG retrieval from "ranking" to "set selection" for the first time, offering deep insights with a concise concept.
- High Efficiency and Practicality: Fewer passages + better performance = reduced context window pressure and inference costs.
- Fully Open Source: Model weights, training data construction pipelines, and code are all publicly available.
- Meticulous Ablation Design: Three variants clearly validate the independent contribution of IRI.

## Limitations

- Distillation relies heavily on the annotation quality of the teacher GPT-4o, which may introduce teacher biases.
- The training data is based on MS MARCO (generic English domain), and the generalizability to non-English or specific domains remains to be verified.
- Performs worse than the best reranking methods on rank-based metrics (MRR/NDCG), suggesting that the two paradigms might be complementary.
- Has not been evaluated on single-hop QA or long-context LLMs.
- Variable-sized outputs may introduce uncertainty into downstream system design.

## Related Work

- **RAG Reranking**: LLM-based listwise reranking methods like RankGPT (Sun et al., 2024) and RankZephyr (Pradeep et al., 2023b) focus on individual relevance.
- **Iterative Retrieval**: Multi-turn retrieval refinements like Self-RAG (Asai et al., 2023) and CoRAG (Wang et al., 2024) are effective but carry high computational overhead.
- **Query Decomposition**: Methods like IRCoT (Trivedi et al., 2023) and RAPTOR (Sarthi et al., 2024) improve retrieval by decomposing complex queries.
- **Context Compression**: Approaches like Chirkova et al. (2025) prune retrieved context, which is complementary to set selection.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 9 |
| Technical Depth | 7 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Value | 9 |
| Overall Score | 8.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)
- [\[ICML 2026\] Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](../../ICML2026/information_retrieval/ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](../../NeurIPS2025/information_retrieval/cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
