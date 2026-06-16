---
title: >-
  [Paper Note] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][PCA] This paper demonstrates that PCA vector compression serves as more than just an efficiency booster; it acts as a zero-training domain adaptation method for dense retrievers. Fitting PCA solely on target domain queries improves NDCG@10 across 75.4% of model-dataset combinations.
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - PCA
date: 2026-05-08
content_hash: 2b464e91981f461e
---
# More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval

**Conference**: ACL 2026  
**arXiv**: [2601.13525](https://arxiv.org/abs/2601.13525)  
**Code**: No public code link provided in the cache  
**Area**: Information Retrieval / Dense Retrieval  
**Keywords**: Dense Retrieval, Domain Adaptation, PCA, Vector Compression, Unsupervised Retrieval  

## TL;DR
This paper demonstrates that PCA vector compression serves as more than just an efficiency booster; it acts as a zero-training domain adaptation method for dense retrievers. Fitting PCA solely on target domain queries improves NDCG@10 across 75.4% of model-dataset combinations.

## Background & Motivation
**Background**: RAG systems rely heavily on dense retrievers. Pre-trained encoders map queries and documents into the same vector space, using cosine similarity for retrieval. Prevailing methods for improving cross-domain retrieval involve collecting target-domain query-document labels, generating pseudo labels, or fine-tuning the retriever.

**Limitations of Prior Work**: Labeling query-document relevance is expensive in professional domains like medicine, finance, code, and law. Synthetic data and fine-tuning also introduce additional computational overhead. For many deployed retrievers, there is a greater need for domain adaptation techniques that do not modify the model, require no labels, and have low deployment costs.

**Key Challenge**: Traditional views regard dimension reduction methods like PCA primarily as efficiency optimizations—shorter vectors, smaller indices, and faster retrieval. However, if the principal components of target domain embeddings happen to correspond to the most critical semantic variations within that domain, the compression itself might filter out source-domain noise and improve retrieval relevance.

**Goal**: The authors re-examine the role of PCA in dense retrieval, investigating whether it can improve out-of-domain retrieval through low-dimensional projection of target domain embeddings without additional encoder training.

**Key Insight**: The paper compares two PCA fitting strategies: using only query embeddings or using a combination of query and document embeddings. Intuitively, the document corpus has higher variance, but this does not necessarily represent user information needs. Query-only PCA may be more closely aligned with the retrieval task itself.

**Core Idea**: Utilize the principal components of target domain query embeddings as a task-relevant subspace. Project both queries and documents into this space to preserve target-domain discriminative semantics and suppress irrelevant dimensions.

## Method
The methodology is concise: a pre-trained dense retriever is kept fixed with no parameter updates. Initially, target domain queries and documents are encoded into original vectors. A PCA projection matrix is then fitted from target domain samples. Finally, queries and documents are projected into the low-dimensional space, where ordinary cosine retrieval is performed. The focus is not on algorithmic complexity but on systematically verifying "what samples should be used to fit PCA" and "whether compression truly improves retrieval quality."

### Overall Architecture
Given a query set $Q=\{q_i\}_{i=1}^{n}$ and a document set $D=\{d_j\}_{j=1}^{m}$, the encoder first produces $d$-dimensional representations. PCA learns a projection $W \in \mathbb{R}^{d \times d'}$ on the target domain embedding matrix, where $d' < d$. During retrieval, $q'_i=(q_i-\mu)W$ and $d'_j=(d_j-\mu)W$ are used to calculate the cosine similarity. Default experiments use a retention ratio $r=0.9$, while analyses scan ratios from 0.1 to 1.0.

### Key Designs

**1. Query Compression: Fitting PCA only on target domain queries and applying the same projection to documents**

Traditional cross-domain retrieval entails labeling or fine-tuning, both of which are costly. The high variance in document corpora often stems from differences in topics, styles, or lengths, which do not necessarily align with axes that distinguish relevance. This paper instead fits PCA solely on target domain query embeddings—since query distributions more directly express user information needs, their principal components are more likely to correspond to directions that differentiate query intent in that domain. Projecting both into this query-driven subspace focuses similarity calculations on task-relevant directions, effectively performing domain adaptation without model changes or labels. This setting improves NDCG@10 in 95/126 (75.4%) of model-dataset combinations.

**2. Query+Document Compression Comparison: Testing if "complete target domain variance" is truly better**

To exclude the trivial explanation that gains only come from dimensionality reduction, the paper adds a control group fitted on the union of queries and documents. If document variance truly represents the domain's semantic structure, this set should be stronger; if it introduces broad thematic noise, it will dilute query intent. Results show improvement in only 56.3% of combinations, with extreme counterexamples like a -52.8% drop for Dis-RoBERTa on Apps, proving that gains stem from the "selection of fitting samples" rather than the act of reduction itself.

**3. Retention Ratio Scanning and Domain Familiarity Analysis: Quantifying "how much to compress and for whom"**

The effectiveness of PCA is intertwined with data structure, the model, and compression intensity. The paper re-scans NDCG@10 over retention ratios from 0.1 to 1.0 and defines domain familiarity through paraphrase robustness (embedding stability between a text and its paraphrase). Scanning shows that moderate compression can simultaneously preserve information and de-noise (e.g., Sent-T5 on CodeSearch rising from 63.0 to 87.3). Structured query domains (medical QA, spatial reasoning) remain robust even at 10%-40% retention, indicating that PCA gains are multi-factorial and must be validated per model and dataset.

### Loss & Training
No training loss functions are used, and the encoder is not updated. The only "learning" is standard PCA: mean-centering the sample matrix $X$ and finding an orthogonal projection $W$ that maximizes projected variance by retaining the top-$d'$ eigenvectors of the covariance matrix. Retrieval scores remain cosine similarities in the projected space.

## Key Experimental Results

### Main Results
The main experiments cover 9 dense retrievers (<2B parameters) and 14 MTEB retrieval datasets, with an additional 11 MTEB datasets with insufficient query counts. The primary metric is NDCG@10, comparing baseline, Query Compression, and Query+Document Compression.

| Setting | Improved Model-Dataset Combinations | Percentage | Main Conclusion |
|---------|------------------------------------|------------|-----------------|
| Query Compression | 95 / 126 | 75.4% | Most stable; GTE and Sent-T5 improved in 12/14 datasets |
| Query+Document Compression | 71 / 126 | 56.3% | Also helpful, but more prone to dilution by document variance |
| 90% Query Compression | 9 models × 14 datasets | Default setting | Most performance drops <4%; improvements can be significant |
| Total Retrieval Runs | 9 models × 25 datasets × 3 settings | 675 runs | Completed in ~36 hours on an RTX 4090 |

### Ablation Study
| Analysis Item | Key Metric | Description |
|---------------|------------|-------------|
| Dataset Success Rate | MedQA, SpartQA, FaithDial, NarrativeQA, ARC, TV2Nord: 9 / 9 | All models benefited from Query Compression on these datasets |
| Query+Document Failure | Dis-RoBERTa on Apps: -52.8% | Document variance can capture broad themes or styles rather than relevance |
| Retention Ratio | CodeSearch: MiniLM 77.4 -> 82.8, Sent-T5 63.0 -> 87.3 | Moderate compression preserves info and de-noises |
| Low-Dim Robustness | ArguAna and MedQA strong at 40% retention; GTE peaks at 10% on MedQA | Structured query domains are particularly PCA-friendly |
| Comparison with IDA | PCA on SciDocs: 12.1, FiQA: 14.3; GPL: 13.4, 14.6 | PCA approaches GPL and outperforms GPL+JPQ/BPR on 5 shared datasets |

### Key Findings
- Query-only PCA is more reliable than query+document PCA, suggesting retrieval domain adaptation should center on user information needs rather than corpus-wide variance.
- Datasets with strong hierarchical or categorical structures (e.g., Medical QA, Spatial Reasoning) benefit more easily, as principal components likely correspond to true semantic axes.
- PCA is not a substitute for full fine-tuning; GPL remains stronger on ArguAna and SciFact. However, PCA's cost is nearly zero, making it an excellent first-step baseline.

## Highlights & Insights
- The primary highlight is the reinterpretation of "compression" as "unsupervised domain adaptation," transforming a common engineering trick into a systematically analyzable adaptation mechanism.
- The discovery that PCA should be fitted only on queries is highly practical. Many industrial scenarios have abundant query logs but no high-quality relevance labels; this paper leverages that exact data profile.
- The paper avoids over-complicating PCA, instead proving the value of simple methods as strong baselines through large-scale evaluations.
- Findings on domain familiarity are mixed: some models benefit more when familiar with a domain, others when unfamiliar. This suggests PCA gains are not driven by a single factor and require per-model verification.

## Limitations & Future Work
- The method assumes the original encoder has already captured sufficient target-domain information; if technical terminology is entirely outside the model's capacity, PCA cannot create knowledge from nothing.
- PCA still requires a sufficient number of unlabeled target domain queries to estimate covariance; principal components may be unstable if queries are too sparse.
- The optimal retention ratio varies by model and dataset. While $r=0.9$ is suggested as a default, deployment should involve a small-scale scan.
- The method is a linear projection; future work could explore combinations with synthetic data, pseudo-label fine-tuning, or non-linear manifold learning.

## Related Work & Insights
- **vs GPL / IDA**: GPL achieves domain adaptation via synthetic labels and fine-tuning, which is more powerful but more expensive. PCA approaches GPL performance on SciDocs and FiQA without training.
- **vs Traditional Vector Compression**: While traditional PCA focuses on speed and storage, this work emphasizes that compression alters retrieval relevance and can improve domain adaptation.
- **vs Prompt/Few-shot Retriever Adaptation**: Methods like Promptagator rely on LLMs for data generation; PCA relies only on unlabeled queries/documents.
- **Insight**: After a RAG system goes live, a lightweight PCA adapter learned from online query logs can serve as a low-cost adaptation and diagnostic tool before deciding if expensive fine-tuning is necessary.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Simple method, but the systematic demonstration of "query-only PCA improves domain adaptation" is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 9 models, 25 datasets, 675 runs, and diverse analyses provide a solid foundation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear arguments with experiments that directly support core conclusions.
- Value: ⭐⭐⭐⭐⭐ Extremely practical, especially for retrieval/RAG systems with query logs but no labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](../../ICML2026/information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)

</div>

<!-- RELATED:END -->
