---
title: >-
  [Paper Note] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation
description: >-
  [ACL2026][Recommender Systems][Personalized RAG] ClusterRAG introduces collaborative filtering into personalized RAG: it first constructs user representations using historical documents and clusters them with HDBSCAN…
tags:
  - "ACL2026"
  - "Recommender Systems"
  - "Personalized RAG"
  - "Collaborative Filtering"
  - "User Clustering"
  - "HDBSCAN"
  - "LaMP"
date: 2026-05-08
content_hash: 951522e3c7b32777
---

# ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation

**Conference**: ACL2026  
**arXiv**: [2605.18769](https://arxiv.org/abs/2605.18769)  
**Code**: https://github.com/academicprojects44/anonymous  
**Area**: Personalized Recommendation / Personalized RAG  
**Keywords**: Personalized RAG, Collaborative Filtering, User Clustering, HDBSCAN, LaMP

## TL;DR
ClusterRAG introduces collaborative filtering into personalized RAG: it first constructs user representations using historical documents and clusters them with HDBSCAN, then performs hierarchical retrieval of profile documents from both the target and similar users to form prompts. On the LaMP multi-task benchmark, the hybrid mode consistently outperforms vanillaRAG, LaMP-IPA, ROPG, and CFRAG.

## Background & Motivation
**Background**: RAG has become a mainstream paradigm for reducing hallucinations and enhancing factuality. The standard approach involves retrieving external documents based on the current query and prepending them to the generation model's context. Personalized RAG further incorporates user history to align responses with user preferences, writing styles, or long-term interests.

**Limitations of Prior Work**: Many personalized RAG methods rely solely on the target user's own profile, which is fragile when user history is sparse, noisy, or when the current query does not perfectly match historical records. On the other hand, non-personalized RAG completely ignores long-term user preferences. Existing collaborative approaches attempt to find similar users but face high costs when calculating pairwise similarities in large-scale user sets. There is also a lack of systematic design for selecting and mixing documents from similar users with the target user’s profile.

**Key Challenge**: Personalization requires full utilization of the target user's history, yet individual histories are often incomplete. While collaborative filtering can compensate for sparsity, it introduces retrieval complexity, privacy concerns, and noisy neighbors. A scalable Personalized RAG must find a controllable way to mix "individual signals" with "similar group signals."

**Goal**: ClusterRAG aims to build a model-agnostic RAG pipeline with replaceable retrievers that leverages collaborative signals without strong dependence on model parameter fine-tuning. It addresses three issues: user representation, efficient search for similar users, and the combined integration of target and similar user documents into the generation prompt.

**Key Insight**: The authors transfer cluster-based collaborative filtering from recommendation systems to the RAG retrieval front-end. Users are organized into semantically consistent cohorts, and similar users are ranked only within the same cluster. Document retrieval also employs profile document clustering to select thematic clusters before fine-grained reranking.

**Core Idea**: Use HDBSCAN to build a hierarchical retrieval space for users/documents and employ hybrid profile retrieval to inject evidence from both the target and similar users, enabling LLMs to generate more stable personalized outputs.

## Method

### Overall Architecture
ClusterRAG consists of three stages: user representation and similar user retrieval, profile document retrieval, and personalized generation. First, the system encodes each user's historical documents into dense embeddings and averages them to obtain a user representation. Subsequently, HDBSCAN clusters users into variable-density cohorts. Within each cluster, ColBERTv2 calculates fine-grained user similarity to identify the top-$k$ nearest neighbors for each user. Given a query, the system can retrieve documents from only the target user, only similar users, or a hybrid of both. Finally, candidate documents are clustered into thematic groups; the system selects relevant clusters and reranks the top-$m$ documents within them to concatenate into an IPA prompt for the generator.

### Key Designs
1. **User-level HDBSCAN Clustering and Intra-cluster Ranking**:

	- **Function**: Finding behaviorally similar users without global pairwise comparisons.
	- **Mechanism**: The representation of each user $u$ is the average of their profile document embeddings $\mathbf{z}_u=\frac{1}{n_u}\sum_i f(d_i)$. HDBSCAN automatically discovers variable-density user clusters. For users in the same cluster, a similarity matrix is constructed using ColBERTv2 as $R^C_{u,v}=ColBERTv2(\mathbf{z}_u,\mathbf{z}_v)$ to keep top-$k$ neighbors.
	- **Design Motivation**: Global user similarity computation is expensive in large-scale scenarios and prone to introducing noisy neighbors with inconsistent themes; clustering before ranking restricts comparisons to cohorts with more consistent behavior.

2. **Three Profile Retrieval Modes**:

	- **Function**: Controlling the sources of individual and collaborative signals.
	- **Mechanism**: The User-only mode uses only the target user's profile; the Collaborative mode fetches documents from the top-$k$ similar users' profiles; the Hybrid mode merges both sets into a candidate pool. By default, $k=1$ and $m=2$, testing the effectiveness of collaborative signals even with minimal profile documents.
	- **Design Motivation**: A single user's profile is insufficient for cold-start or sparse users, while pure collaborative signals might dilute individual preferences; the hybrid approach allows the system to retain personal history supplemented by similar users.

3. **Document Thematic Cluster Retrieval and IPA Generation**:

	- **Function**: Selecting the most relevant and suitable context from candidate profile documents for the prompt.
	- **Mechanism**: Candidate documents are encoded by a dense retriever and formed into thematic clusters via HDBSCAN. The query embedding is compared against cluster centroids to select the top-$B$ clusters, followed by intra-cluster reranking to select the top-$m$ documents. The generation stage uses In-Prompt Augmentation (IPA), concatenating the query and retrieved documents according to task templates. The profile length in the prompt is controlled by $|U_p|=\mathcal{G}_t(L_{max}-\min(|q|,\lfloor \gamma L_{max}\rfloor))$, with a default $\gamma=0.55$.
	- **Design Motivation**: Directly feeding all user documents into the prompt wastes context length and introduces irrelevant history; hierarchical retrieval reduces complexity to $\mathcal{O}(K+B\cdot N/K)$ while ensuring the prompt contains only the most relevant evidence.

### Loss & Training
ClusterRAG is a retrieval and prompt organization framework and does not require architectural changes to the generator. The main experiments use fine-tuned FlanT5-base, while extended experiments use FlanT5-XXL and Qwen2-7B-Instruct for zero-shot personalized testing. Training utilizes AdamW with a learning rate of $5\times10^{-5}$, weight decay of $10^{-4}$, warm-up ratio of 0.05, for up to 30 epochs with a batch size of 16. The maximum prompt length is 512, maximum output length is 128, and beam size is 4. Experiments were conducted on a Quadro RTX 8000 48GB, taking approximately 10-24 hours per task.

## Key Experimental Results

### Main Results
The LaMP benchmark includes tasks such as personalized citation, movie tagging, product rating, headline/title generation, and tweet paraphrasing. The table below excerpts representative metrics; higher is better for classification tasks, while lower is better for LaMP-3 (MAE/RMSE).

| Method | LaMP-1 Acc/F1 | LaMP-2 Acc/F1 | LaMP-3 MAE/RMSE | LaMP-7 R-1/R-L |
|------|---------------|---------------|-----------------|----------------|
| vanillaRAG | 0.630 / 0.630 | 0.520 / 0.440 | 0.371 / 0.709 | 0.310 / 0.273 |
| LaMP-IPA | 0.674 / 0.664 | 0.570 / 0.522 | 0.289 / 0.608 | 0.508 / 0.457 |
| CFRAG | 0.633 / 0.327 | 0.534 / 0.036 | 0.354 / 0.707 | 0.375 / 0.306 |
| ClusterRAG-C | 0.674 / 0.673 | 0.644 / 0.607 | 0.284 / 0.624 | 0.507 / 0.454 |
| ClusterRAG-U | 0.645 / 0.645 | 0.649 / 0.612 | 0.271 / 0.599 | 0.514 / 0.464 |
| ClusterRAG-H | 0.690 / 0.690 | 0.661 / 0.620 | 0.270 / 0.594 | 0.521 / 0.470 |

### Ablation Study
| Variant | LaMP-3 MAE | LaMP-3 RMSE | LaMP-7 R-1 | LaMP-7 R-L | Description |
|------|-----------:|------------:|-----------:|-----------:|------|
| w/o user clustering | 0.320 | 0.637 | 0.458 | 0.371 | Random similar users; noisy collaborative signals |
| w/o intra-cluster sim | 0.329 | 0.639 | 0.501 | 0.442 | No intra-cluster reranking; lower neighbor quality |
| w/o doc ranking | 0.331 | 0.642 | 0.462 | 0.413 | Document-level evidence not effectively ranked |
| Centroids only | 0.400 | 0.643 | 0.472 | 0.438 | Uses only centroid representations; loses specific evidence |
| k-means | 0.291 | 0.610 | 0.502 | 0.453 | Clustering is replaceable but weaker than HDBSCAN |
| ClusterRAG | 0.270 | 0.594 | 0.521 | 0.470 | Full method |

### Key Findings
- The Hybrid mode achieves the best performance across all LaMP tasks, indicating strong complementarity between target user profiles and similar user profiles.
- Using only 2 profile documents allows it to surpass baselines requiring more documents, suggesting that "selecting the right documents" is more important than "including more documents."
- ColBERTv2 is the strongest retriever: achieving 0.690/0.690, 0.661/0.620, and 0.521/0.470 on LaMP-1/2/7; Random, Recency, and BM25 lag significantly, while BGE and Contriever perform in between.
- Zero-shot LLMs also benefit from ClusterRAG: pFlan improved from 0.546/0.540 to 0.648/0.647 on LaMP-1, and pQwen2 improved from 0.521/0.521 to 0.610/0.606 on LaMP-2.

## Highlights & Insights
- **Collaborative Filtering as a RAG Front-end**: Instead of modifying LLM architectures, this approach makes it easier to integrate into existing personalized generation systems and avoids per-user model fine-tuning.
- **Hierarchical correspondence of User and Document Clustering**: User clustering addresses "whom to borrow signals from," while document clustering addresses "which evidence to borrow," resulting in a clear structure.
- **Hybrid Retrieval as the Core Gain**: Experiments show that while both user-only and collaborative-only modes are useful, the hybrid mix is most stable, proving that personalization is about supplementing context rather than just replacing it.
- **Practical Significance for Cold-start**: When a target user's history is sparse, similar users' documents can fill the gap; if collaborative signals are untrustworthy, the framework can fall back to user-only mode.

## Limitations & Future Work
- The generation side depends on IPA prompts, and the authors acknowledge that the prompt formulation is not yet optimal; more structured prompts or joint retrieval-generation optimization may further improve results.
- LaMP-1 and LaMP-5 only contain paper abstracts rather than full texts, limiting the information ceiling for citation/title tasks.
- Experiments are restricted to English and text data, with multi-lingual, multi-modal user history, or cross-platform recommendation scenarios yet to be verified.
- Performance still depends on the underlying LLM and retriever; collaborative filtering might amplify group biases if user or document embeddings contain inherent bias.
- Future work could investigate incremental clustering, online user reassignment, privacy-preserving embedding aggregation, and feeding generation feedback back into retrieval ranking.

## Related Work & Insights
- **vs vanillaRAG / Self-RAG**: Traditional RAG retrieves shared knowledge based on the current query; ClusterRAG further incorporates long-term user history and similar users' histories into the context.
- **vs LaMP-IPA / ROPG**: These personalized methods focus more on the target user's own profile; ClusterRAG's incremental value lies in explicitly modeling cross-user collaborative signals.
- **vs CFRAG**: CFRAG uses contrastive learning to find similar users; ClusterRAG utilizes HDBSCAN and intra-cluster ColBERTv2 ranking, emphasizing scalable cohort retrieval and document-level reranking.
- **Insights**: In enterprise knowledge assistants, learning assistants, or long-cycle writing assistants, historical interactions of similar users or projects can serve as "collaborative memory," though retrieval scope must be constrained by clustering and permission controls.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically combines collaborative filtering with Personalized RAG; individual modules are familiar, but the integration is complete.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers many LaMP tasks with analyses of retrievers, LLMs, and ablations; real-world large-scale online latency and privacy evaluations are still needed.
- Writing Quality: ⭐⭐⭐⭐☆ Method decomposition is clear and tables are comprehensive; some notation and prompt details are slightly dense.
- Value: ⭐⭐⭐⭐☆ Highly insightful for engineering personalized RAG, especially suitable for scenarios with sparse user histories but available group behavior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MemRec: Collaborative Memory-Augmented Agentic Recommender System](memrec_collaborative_memory-augmented_agentic_recommender_system.md)
- [\[AAAI 2026\] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers](../../AAAI2026/recommender/slidetailor_personalized_presentation_slide_generation_for_scientific_papers.md)
- [\[ICML 2026\] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](../../ICML2026/recommender/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](../../NeurIPS2025/recommender/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)

</div>

<!-- RELATED:END -->
