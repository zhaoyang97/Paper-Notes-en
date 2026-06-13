---
title: >-
  [Paper Note] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval
description: >-
  [ACL2026][Information Retrieval & RAG][Unsupervised Domain Adaptation] UnIte shifts the bottleneck of unsupervised domain adaptation for neural retrievers from "generating more pseudo-queries" to "selecting documents mor…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Unsupervised Domain Adaptation"
  - "Document Sampling"
  - "Retrieval Augmentation"
  - "Uncertainty Estimation"
  - "BEIR"
date: 2026-05-08
content_hash: 06380aa0cd43b52f
---

# UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval

**Conference**: ACL2026 Findings  
**arXiv**: [2604.25142](https://arxiv.org/abs/2604.25142)  
**Code**: https://github.com/ldilab/UnIte  
**Area**: Information Retrieval  
**Keywords**: Unsupervised Domain Adaptation, Document Sampling, Retrieval Augmentation, Uncertainty Estimation, BEIR

## TL;DR
UnIte shifts the bottleneck of unsupervised domain adaptation for neural retrievers from "generating more pseudo-queries" to "selecting documents more intelligently." It first filters low-density noise documents using aleatoric uncertainty and then iteratively samples high-value documents based on epistemic uncertainty that evolves dynamically with model training. UnIte consistently outperforms DUQGen on large-scale BEIR corpora using fewer pseudo-queries.

## Background & Motivation
**Background**: Neural retrievers are typically pre-trained on source domains such as MS-MARCO, leading to significant generalization drops in new domains. A mainstream approach for unsupervised domain adaptation is pseudo-query generation: generating pseudo-queries from target domain documents and fine-tuning the retriever with these query-document pairs.

**Limitations of Prior Work**: Large-scale corpora often exceed 100k documents, making it infeasible to call LLMs or query generators for all documents. Consequently, "which documents to sample for pseudo-query generation" becomes the core budget bottleneck. Random sampling is inefficient. DUQGen uses clustering to improve coverage, but it primarily optimizes diversity, often sampling from two low-value regions: low-density outlier documents and high-confidence regions where the current retriever is already certain.

**Key Challenge**: Domain adaptation requires documents that are both reliable and possess high learning value. Low-density outliers may contain noise or off-topic content, leading to negative transfer; high-confidence documents represent common themes the model has already mastered, offering limited training gains. An effective sampler should simultaneously avoid noise with high aleatoric uncertainty and prioritize knowledge gaps with high epistemic uncertainty.

**Goal**: Select target domain documents with higher training value under a fixed pseudo-query budget, enabling the retriever to achieve higher nDCG@10 with smaller sample sizes, while dynamically updating the sampling strategy as the model adapts.

**Key Insight**: The authors decompose uncertainty into data-level AU (Aleatoric Uncertainty) and model-level EU (Epistemic Uncertainty). AU is estimated using model-agnostic BM25 lexical kNN distances to avoid misidentifying "unlearned content" as "data anomalies." EU is measured by the mismatch between the current retriever's document representations and the target domain's IDF distribution.

**Core Idea**: First filter out high-AU outliers. Then, estimate EU after each training round and iteratively sample documents using a strategy of "high EU + high diversity + re-sampling penalty" until the average EU reaches a plateau.

## Method

### Overall Architecture
The UnIte pipeline consists of three steps. The first step is AU Filtering: using BM25 kNN distances across the target corpus to identify low-density documents and remove obvious outliers from the candidate pool. The second step is the iterative sampling-training loop: estimating the current retriever's EU on the filtered corpus and selecting high-value documents within each cluster using a DUQGen-style clustering approach. The third step is pseudo-query generation and retriever update: generating pseudo-queries for the sampled documents using Llama3-8B-Instruct, fine-tuning the retriever, and returning to the second step to re-estimate EU.

This cycle continues until a maximum budget of 5k documents is reached or the EMA (Exponential Moving Average) of the average EU hits a local minimum. Intuitively, an initial decrease in average EU indicates the model is filling target domain knowledge gaps; if EU rebounds, it suggests new samples are becoming redundant or leading to overfitting, signaling an early stop.

### Key Designs
1. **AU Filtering: Model-agnostic Low-density Filtering**:

    - Function: Pre-emptively remove noise or off-topic documents that do not represent the main distribution of the target corpus.
    - Mechanism: For each document, calculate the lexical distance to its $k$-th nearest neighbor, defined as $D_k(d)=1/(\epsilon + BM25(d,n_k))$, then normalize using a modified z-score. If $z(d)>z_{thr}$, the document is considered to be in a low-density region and is filtered. In experiments, $k=3$ and $z_{thr}=1.5$.
    - Design Motivation: AU represents inherent data uncertainty and should be model-independent. Using dense embeddings for outlier detection might accidentally remove target domain content that the model has not yet understood; BM25 relies solely on corpus statistics, providing a cleaner separation between AU and EU.

2. **Domain-aware EU Estimation: Measuring Gaps via Target IDF**:

    - Function: Identify target domain documents that the current retriever has not yet mastered.
    - Mechanism: Pre-compute token-level IDF on the target domain. For document embeddings, use the current model's MLM head to project onto the vocabulary probability space, select the top-1000 tokens, and compare the importance of high-IDF words with the model's predicted probabilities. If the model fails to predict high-IDF words for the target domain, it indicates high EU for that document's knowledge area.
    - Design Motivation: Traditional entropy or MC-dropout only considers the model's internal variance without account for word importance in the target domain. UnIte incorporates target domain distribution statistics into EU, making uncertainty more aligned with the learning value for domain adaptation.

3. **Iterative Sampling, Re-sampling Penalty, and Early Stopping**:

    - Function: Avoid redundancy caused by one-time static sampling and dynamically track the model's knowledge gaps within the budget.
    - Mechanism: Sample 500 documents per round for up to 10 rounds. Within clusters, documents are ranked by $score=\lambda \widehat{EU}+(1-\lambda)\widehat{Diversity}$ (where $\lambda=0.5$ in experiments). Cluster budgets are adjusted by $w_i=|C_i|/(P_i+\epsilon)$; clusters sampled more frequently in the past receive lower weights. Average EU is smoothed via EMA with $\alpha=0.4$, stopping at the plateau.
    - Design Motivation: EU changes during training; clusters that were high-value in the first round may become learned later. Re-sampling penalties and early stopping allow the algorithm to shift the budget from dominant clusters to areas with remaining gaps.

### Loss & Training
UnIte is a sampling strategy and does not change the retriever's training objective. The authors generate one pseudo-query for each sampled document using Llama3-8B-Instruct (temperature 0.8, top-p 0.9); subsequently, they fine-tune the retriever using standard objectives. The experiments cover single-vector retrievers such as DPR, coCondenser, COCO-DR, and Qwen3-Embedding-4B. The primary evaluation metric is nDCG@10 on BEIR. All core experiments were conducted on a single NVIDIA RTX 3090. DPR training for 5k samples takes approximately 10 minutes, and UnIte typically stops at 3k-5k samples.

## Key Experimental Results

### Main Results
Evaluation was conducted on five large-scale BEIR datasets: TREC-COVID, Robust04, Quora, TREC-NEWS, and HotpotQA. The table below excerpts the overall average results from Table 1.

| Retriever | DUQGen AVG | UnIte AVG | Gain vs DUQGen | Representative Gain |
|-----------|------------|-----------|------------------|----------------------|
| DPR | 46.61 | 49.06 | +2.45 | TREC-COVID +4.04, TREC-NEWS +5.08 |
| coCondenser | 54.94 | 55.69 | +0.75 | Robust04 +1.53, TREC-NEWS +3.14 |
| COCO-DR | 62.01 | 62.27 | +0.26 | TREC-COVID +0.86, TREC-NEWS +0.47 |
| Qwen3-Embedding-4B | 69.31 | 72.80 | +3.49 | TREC-COVID +3.00, Quora +4.72, TREC-NEWS +4.86 |

The improvements of UnIte are more pronounced on the smaller DPR model and the larger Qwen3 model, suggesting that the "current knowledge gap" signal scales with model capacity. COCO-DR, being inherently strong, shows smaller absolute gains, but the average remains positive.

### Ablation Study
The paper focuses on ablating AU, EU, and the resampling penalty. Results show that removing EU can result in scores ~4 nDCG@10 lower than zero-shot on Robust04. Removing both AU and EU results in a drop of ~5 and ~9 nDCG@10 on TREC-COVID and Robust04 respectively compared to the UnIte peak.

| Configuration | TC nDCG@10 | QR nDCG@10 | TN nDCG@10 | Description |
|---------------|------------|------------|------------|-------------|
| w/ Resampling Penalty | 61.73 | 74.95 | 30.39 | Full dynamic budget allocation |
| w/o Resampling Penalty | 54.39 | 73.42 | 21.83 | Static cluster budget, prone to oversampling dominant clusters |
| Gain | +7.34 | +1.53 | +8.56 | Penalty is critical for TC / TN |

| EU Estimation Method | TC | RB | TN | AVG | Conclusion |
|----------------------|----|----|----|-----|------------|
| UnIte domain-aware EU | 55.54 | 31.38 | 23.33 | 36.75 | Incorporates target IDF distribution |
| MC-Dropout | 52.79 | 28.27 | 21.60 | 34.22 | Only considers model variance; weak domain awareness |
| Entropy | 54.10 | 25.83 | 22.70 | 34.21 | Prone to ignoring domain mismatch |

| Method | DPR ΔnDCG@10 / 1k | Avg Samples | Qwen3 ΔnDCG@10 / 1k | Avg Samples |
|--------|--------------------|-------------|----------------------|-------------|
| DUQGen | 3.56 | 5k | 0.83 | 5k |
| UnIte | 4.50 | 4.5k | 2.45 | 4.5k |

### Key Findings
- Pursuing diversity alone is insufficient. DUQGen samples low-density outliers and high-confidence regions; the former introduces noise, and the latter provides weak learning signals.
- AU and EU should be estimated separately. BM25 kNN handles data-level outliers, while IDF combined with model vocabulary projection addresses model-level knowledge gaps. Mixing these in dense embeddings causes mutual interference.
- Iterative sampling not only saves training samples but also prevents overfitting. The local minimum of average EU aligns with the nDCG@10 peak on TREC-COVID, indicating that the uncertainty plateau is a valid unsupervised stopping signal.
- Computational overhead is acceptable. AU filtering takes ~120s once, and EU takes ~150s per round. Even with sampling costs, it remains more time-efficient than the fixed 5k baseline due to early stopping at ~4k samples.

## Highlights & Insights
- The most valuable contribution is mapping the active learning uncertainty taxonomy clearly to IR domain adaptation: AU is noise to be excluded, and EU is the gap most worth learning.
- EU estimation is not just entropy; it is the mismatch between "model-predicted vocabulary distribution" and "target domain IDF importance." This is a more relevant signal for IR domain adaptation than pure model uncertainty.
- Iterative sampling is more consistent with the fine-tuning process than one-time sampling. The value of documents changes as the model trains; static selection is particularly wasteful when budgets are small.
- This method can be transferred to RAG data construction or reranker hard-negative selection: first filtering low-density noise, then selecting samples based on the model's mastery of target domain terminology.

## Limitations & Future Work
- The method is primarily designed for single-vector retrievers. For late-interaction models like ColBERT or rerankers like MonoT5, the paper uses pooling and shared vocab heads as approximations, which may not perfectly align with native training objectives.
- Target domain distribution is currently modeled only via token-level IDF, which struggles to capture thematic structures, entity relationships, or query intent. Future work could explore topic-conditioned statistics.
- While the resampling penalty reduces oversampling of dominant clusters, it does not guarantee full coverage of rare minority topics in skewed domains.
- Experiments focus on English BEIR. Validation is still needed for cross-lingual retrieval and complex corpora like patent, legal, or medical domains.
- Pseudo-query quality remains dependent on Llama3-8B-Instruct; if the generator is unreliable in a specific domain, the performance will be bottlenecked by query noise regardless of sampling quality.

## Related Work & Insights
- **vs GPL**: GPL relies on random document sampling followed by cross-encoder labeling for pseudo-relevance. UnIte focuses on making document selection more efficient rather than more expensive labeling.
- **vs DUQGen**: DUQGen uses an external Contriever for diversity clustering, providing good coverage but lacking awareness of the current retriever's learning state. UnIte adds dynamic EU and AU filtering to diversity.
- **vs Quality sampling**: Quality estimation models attempt to determine if a document is suitable for query generation but do not explicitly consider target domain distribution or retriever knowledge gaps. UnIte signals are more aligned with the adaptation objective.
- **vs standard active learning uncertainty sampling**: Traditional entropy/MC-dropout sampling is prone to treating outliers as high-value samples. UnIte separates AU first, then uses domain-aware EU to select learnable samples.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically applies AU/EU taxonomy to IR adaptation; design is solid, though components leverage existing IR and active learning concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 5 large corpora and 4 retrievers with extensive ablations; adaptation for multi-vector/rerankers remains preliminary.
- Writing Quality: ⭐⭐⭐⭐☆ Motivations are clear, and methods are well-supported by ablations; some table layouts are slightly cluttered.
- Value: ⭐⭐⭐⭐⭐ Highly practical for budget-constrained IR domain adaptation; provides a reusable framework for RAG corpus selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](s2g-rag_structured_sufficiency_and_gap_judging_for_iterative_retrieval-augmented.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ACL 2026\] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](s2g-rag_structured_sufficiency_and_gap_judging_for_iterative_retrieval-augmented.md)
- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](../../CVPR2025/information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
