---
title: >-
  [Paper Note] Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method
description: >-
  [ACL2026][Information Retrieval & RAG][adaptive retrieval] This paper proposes Adaptive Table Retrieval (ATR), which replaces fixed top-k table retrieval with a query-adaptive threshold. Combined with relevance calibrati…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "adaptive retrieval"
  - "table retrieval"
  - "text-to-SQL"
  - "thresholding"
  - "sliding-window reranking"
date: 2026-05-08
content_hash: c687b76490b4e1e9
---

# Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method

**Conference**: ACL2026  
**arXiv**: [2605.18766](https://arxiv.org/abs/2605.18766)  
**Code**: The paper states that code and data are provided (cache does not contain a specific URL)  
**Area**: Information Retrieval / Table Retrieval / Text-to-SQL  
**Keywords**: adaptive retrieval, table retrieval, text-to-SQL, thresholding, sliding-window reranking

## TL;DR
This paper proposes Adaptive Table Retrieval (ATR), which replaces fixed top-k table retrieval with a query-adaptive threshold. Combined with relevance calibration, inter-table semantic grouping, and sliding-window reranking, it simultaneously improves retrieval recall, text-to-SQL execution accuracy, and inference efficiency across Spider, BIRD, and Spider 2.0.

## Background & Motivation
**Background**: In text-to-SQL and structured RAG, systems typically retrieve a set of relevant tables from a large database before passing them to an LLM to generate SQL. Mainstream table retrievers calculate query-table similarity and return a fixed top-k tables.

**Limitations of Prior Work**: Fixed top-k retrieval ignores the variance in the number of tables required for different queries. Simple queries may only need one or two tables, where retrieving too many introduces noise and token costs. Complex enterprise-level queries may require dozens or even hundreds of tables, where retrieving too few misses necessary evidence. The paper notes that in Spider 2.0, the number of ground-truth tables per query ranges from 1 to 366, making a fixed $k$ difficult to balance.

**Key Challenge**: Errors in table retrieval manifest in two opposite forms: under-retrieval misses tables required for the SQL, while over-retrieval places irrelevant schemas into the generator's context, interfering with SQL generation. Fixed top-k leaves this trade-off to a global hyperparameter rather than allowing each query to determine its own needs.

**Goal**: The authors aim to design an adaptive table retriever that does not rely on iterative LLM interactions, allowing it to dynamically determine the number of returned tables while remaining scalable and low-latency for large-scale databases.

**Key Insight**: ATR transforms the decision of "how many tables to return" into a learnable threshold problem. Each table has a relevance logit, and an adaptive threshold token is introduced; all tables with logits higher than the threshold are retrieved.

**Core Idea**: Use a threshold token to learn a query-specific retrieval boundary, ensuring relevant tables exceed the threshold while irrelevant ones fall below it. This is supported by joinability-aware table representation learning and sliding-window reranking for large-scale schemas.

## Method

### Overall Architecture
Given a natural language query $q$ and a set of candidate tables $C$, ATR uses ModernBERT-large as the encoder to concatenate the query, a threshold token, and the schema representations of multiple tables into a single input sequence. Each table is preceded by a table token $T_i$, and the boundary is represented by a threshold token $T_{th}$. The model outputs logits for each table token and the threshold token. During inference, only tables with logits higher than the threshold logit are retained, allowing the number of returned tables $k_q$ to vary per query.

In practice, ATR first uses a bi-encoder (Contriever or UAE) to retrieve the top 50 candidates, then performs reranking on these. To manage encoder length and self-attention costs, a sliding-window reranking strategy is designed: starting from the lower-ranked end, the model processes one window at a time, keeps the highest-scoring tables within the window, merges them with the next batch of candidates, and repeats until the entire set is reranked.

### Key Designs
1.  **Adaptive Thresholding**:
    - **Function**: Enables the retriever to learn a dynamic decision boundary for each query instead of returning a fixed top-k.
    - **Mechanism**: The input follows the format $[T_{th}; q; T_1; t_1; ...; T_n; t_n]$. During training, relevant table tokens must have logits higher than $T_{th}$, while irrelevant ones must be lower. The loss consists of two parts: $L_1$ increases the probability of relevant tables relative to the threshold, and $L_2$ pushes irrelevant tables below it, resulting in $L_{AT}=\alpha L_1+\beta L_2$.
    - **Design Motivation**: Fixed $k$ only offers a global compromise between recall and noise; the threshold token allows the model to automatically decide the boundary based on query difficulty and schema relevance.

2.  **Relevance Calibration and Semantic Grouping**:
    - **Function**: Simultaneously learns query-table relevance and table-table relationships to avoid ranking based solely on independent table similarity.
    - **Mechanism**: Relevance calibration uses BCE to widen the logit gap between relevant and irrelevant tables. Semantic grouping uses a contrastive loss to pull embeddings of joinable tables closer and push non-joinable ones apart. The final training objective is $L_{ATR}=L_{AT}+\lambda L_{RC}+\gamma L_{SG}$.
    - **Design Motivation**: Text-to-SQL often requires a set of connectable tables rather than isolated relevant ones. Joinability signals help the model retrieve structurally coherent sets of tables.

3.  **Sliding-window reranking**:
    - **Function**: Enables reranking of large table sets without encoding all candidates at once, reducing sequence length, latency, and memory pressure.
    - **Mechanism**: Given a window size $W$ and a retention count $R$, ATR encodes only the tables within the window alongside the threshold. It retains the top-$R$ tables and merges them with the next segment. If the threshold's rank falls below the retention boundary, the tables preceding it are finalized. Finally, all tables outranking the threshold are included in the results.
    - **Design Motivation**: Enterprise database schemas are very long; full reranking triggers encoder length and quadratic complexity bottlenecks. Sliding windows break a large reranking task into controllable smaller steps.

### Loss & Training
ATR is trained using the training sets of Spider and BIRD. It does not use Spider 2.0 training data, making Spider 2.0 a critical out-of-domain test. The retrieval task reports precision, recall, complete recall, and F1. Downstream text-to-SQL evaluation uses retrieved tables with generators like Llama-3.1-8B/70B-Instruct, Qwen2.5-Coder-7B/32B-Instruct, and Gemma-3-4B/27B-IT, evaluated via execution accuracy. ATR is compared against fixed top-k or LLM rerankers like Contriever, UAE, JAR, RankZephyr, and Murre.

## Key Experimental Results

### Main Results
| Dataset | Method | Precision | Recall | Complete Recall | F1 |
|--------|------|-----------|--------|-----------------|----|
| Spider | JAR w/ UAE, k=3 | 48.4 | 96.5 | 94.1 | 62.3 |
| Spider | **Ours** w/ Contriever | 69.6 | 99.5 | 99.2 | 78.3 |
| Spider | **Ours** w/ UAE | 69.3 | 99.6 | 99.4 | 78.1 |
| BIRD | JAR w/ Contriever, k=3 | 54.4 | 87.4 | 76.3 | 65.0 |
| BIRD | **Ours** w/ Contriever | 54.0 | 98.2 | 96.0 | 65.8 |
| BIRD | **Ours** w/ UAE | 52.8 | 98.6 | 97.1 | 65.1 |
| Spider 2.0 | Murre, k=10 | 14.8 | 61.9 | 48.5 | 21.5 |
| Spider 2.0 | **Ours** w/ Contriever | 21.9 | 72.4 | 64.4 | 27.8 |
| Spider 2.0 | **Ours** w/ UAE | 19.9 | 75.4 | 68.7 | 26.7 |

### Ablation Study
| Configuration | Spider R / CR | BIRD R / CR | Spider 2.0 R / CR | Description |
|------|---------------|-------------|-------------------|------|
| **Ours** | 99.5 / 99.2 | 98.2 / 96.0 | 72.4 / 64.4 | Full Model |
| w/o BCE relevance calibration | 99.0 / 98.7 | 97.5 / 95.8 | 68.2 / 60.8 | Weaker query-table discrimination |
| w/o contrastive semantic grouping | 99.0 / 98.4 | 97.4 / 95.2 | 69.1 / 60.1 | Missing inter-table joinability signals |
| w/o both | 96.4 / 94.4 | 91.8 / 85.7 | 67.7 / 58.2 | Largest drop with both auxiliary targets removed |

### Key Findings
- **Retrieval**: ATR achieves an F1 of 78.3/78.1 on Spider, significantly higher than fixed top-k baselines. On Spider 2.0, complete recall reaches 64.4/68.7, demonstrating that the adaptive mechanism is more effective in scenarios with high variance in required tables.
- **Text-to-SQL**: Using Qwen2.5-Coder-32B, ATR improves execution accuracy by 3.7 and 1.7 points on Spider and BIRD compared to JAR. With 7B models, the **Gain** increases to 5.2 and 2.5 points.
- **Efficiency**: ATR is faster than adaptive document retrieval methods. On Spider/BIRD, ATR's Acc./Time are 71.5/2.2s and 53.3/3.8s, whereas Adaptive-RAG scores 62.8/6.3s and 50.7/13.2s.
- **Memory**: Sliding-window memory profiling on Spider 2.0 shows average peak memory reduced from 340.57 MB to 66.52 MB (80.5% reduction), with maximum peak memory dropping from 1027.88 MB to 284.44 MB (72.3% reduction).

## Highlights & Insights
- The adaptive threshold is the most direct and effective design. It converts the "number of tables to retrieve" from a manual hyperparameter into a model output, making it particularly suitable for enterprise scenarios like Spider 2.0 where the range of required tables is extremely wide.
- The paper demonstrates that over-retrieval harms SQL execution accuracy by introducing incorrect joins, faulty column grounding, and long-context noise to the generator.
- Semantic grouping incorporates inter-table joinability into representation learning, emphasizing that structured RAG should not just rely on query-document similarity but also learn the structural relationships between pieces of evidence.

## Limitations & Future Work
- The current ATR targets only structured tabular data and has not been extended to adaptive retrieval for text, images, or hybrid data types.
- Training relies on Spider and BIRD; while out-of-domain results on Spider 2.0 are provided, real-world enterprise database schemas may be larger, noisier, and subject to more complex business constraints.
- ATR still depends on the initial bi-encoder top-50 candidates; if a critical table is missed in the first stage, the adaptive threshold cannot recover it.
- Future work could explore cross-modal adaptive retrieval, end-to-end joint training of the first-stage retriever and ATR reranker, and linking threshold decisions with the generator's uncertainty.

## Related Work & Insights
- **vs Contriever / UAE**: These bi-encoder methods use fixed top-k based on query-table similarity. They are efficient but cannot adapt to varying table requirements; ATR performs adaptive reranking on their top-50 candidates.
- **vs JAR**: JAR considers inter-table joinability but remains constrained by a fixed $k$. ATR further learns a threshold boundary to dynamically decide the size of the retained set.
- **vs FLARE / Adaptive-RAG / Iter-RetGen**: These adaptive document retrieval methods often trigger retrieval iterations through the generator, causing high latency. ATR avoids LLM interaction, completing dynamic retrieval directly with the encoder.
- **Insight**: For structured RAG, the "quantity of evidence" should be a variable predicted by the model; fixed top-k is more of an engineering default than a logical task assumption.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The adaptive threshold concept is straightforward but highly practical for table retrieval when combined with joinability and sliding-window techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers retrieval, downstream SQL, adaptive baselines, ablations, hyperparameters, memory profiling, and multiple generators.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and diagrams are clear; the main results table is dense but contains comprehensive information.
- Value: ⭐⭐⭐⭐⭐ Direct engineering value for text-to-SQL, enterprise DBQA, and structured RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)
- [\[CVPR 2026\] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](../../CVPR2026/information_retrieval/nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/information_retrieval/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](../../AAAI2026/information_retrieval/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)

</div>

<!-- RELATED:END -->
