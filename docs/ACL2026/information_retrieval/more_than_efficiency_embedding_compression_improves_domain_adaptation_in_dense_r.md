---
title: >-
  [Paper Note] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Dense Retrieval] This paper demonstrates that PCA vector compression is more than just a speed optimization…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Dense Retrieval"
  - "Domain Adaptation"
  - "PCA"
  - "Vector Compression"
  - "Unsupervised Retrieval"
date: 2026-05-08
content_hash: c1df7f4216b8a5db
---

# More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval

**Conference**: ACL 2026  
**arXiv**: [2601.13525](https://arxiv.org/abs/2601.13525)  
**Code**: No public code link provided in the cache  
**Area**: Information Retrieval / Dense Retrieval  
**Keywords**: Dense Retrieval, Domain Adaptation, PCA, Vector Compression, Unsupervised Retrieval  

## TL;DR
This paper demonstrates that PCA vector compression is more than just a speed optimization; it serves as a zero-training domain adaptation method for dense retrievers. Fitting PCA using only target domain queries improves NDCG@10 across 75.4% of model-dataset combinations.

## Background & Motivation
**Background**: RAG systems rely heavily on dense retrievers. Pre-trained encoders map queries and documents into the same vector space, using cosine similarity for retrieval. Dominant methods for cross-domain retrieval involve collecting target domain query-document annotations, generating pseudo-labels, or fine-tuning the retriever.

**Limitations of Prior Work**: In specialized domains like medicine, finance, code, and law, labeling query-document relevance is expensive. Synthetic data generation and fine-tuning also introduce additional computational overhead. For already deployed retrievers, many scenarios require a domain adaptation approach that is low-cost, label-free, and requires no model modification.

**Key Challenge**: Traditional views treat dimensional compression methods like PCA primarily as efficiency optimizations: shorter vectors, smaller indices, and faster retrieval. However, if the principal components of target domain embeddings happen to correspond to the most significant semantic variances within the domain, compression itself might filter out source-domain noise and improve retrieval relevance.

**Goal**: The authors re-examine the role of PCA in dense retrieval, investigating whether it can improve out-of-domain retrieval through low-dimensional projections of target domain embeddings without training the encoder.

**Key Insight**: The paper compares two PCA fitting strategies: using only query embeddings or using a combination of query and document embeddings. Intuitively, while the document corpus has higher variance, it does not necessarily represent user information needs; query-only PCA may align more closely with the retrieval task itself.

**Core Idea**: Use the principal components of target domain query embeddings as a task-relevant subspace. Projecting both queries and documents into this subspace preserves discriminative semantics in the target domain while compressing irrelevant dimensions.

## Method
The method is highly concise: a pre-trained dense retriever is fixed with no parameter updates. Target domain queries and documents are first encoded into original vectors. A PCA projection matrix is then fitted from target domain samples. Finally, queries and documents are projected into a low-dimensional space for standard cosine retrieval. The focus of the paper is not on algorithmic complexity, but on systematically verifying "what samples should be used to fit PCA" and "whether compression truly improves retrieval quality."

### Overall Architecture
Given a query set $Q=\{q_i\}_{i=1}^{n}$ and a document set $D=\{d_j\}_{j=1}^{m}$, the encoder first generates $d$-dimensional representations. PCA learns a projection $W \in \mathbb{R}^{d \times d'}$ on the target domain embedding matrix, where $d' < d$. During retrieval, $q'_i=(q_i-\mu)W$ and $d'_j=(d_j-\mu)W$ are used to calculate the cosine similarity for ranking. Default experiments use a retention ratio $r=0.9$, with analysis scanning ratios from 0.1 to 1.0.

### Key Designs
1. **Query Compression**:
	- **Function**: Fits PCA using only target domain query embeddings, then applies the same projection to both queries and documents.
	- **Mechanism**: The query distribution directly expresses user information needs, making principal components more likely to correspond to "axes that distinguish query intent in this domain." After projection, retrieval focuses more on these task-relevant directions.
	- **Design Motivation**: High variance in the document corpus may stem from differences in topic, style, or length, which do not necessarily equate to relevance discrimination. Query-only PCA aligns better with the retrieval objective.

2. **Query+Document Compression Comparison**:
	- **Function**: Fits PCA on the union of queries and documents to test if "more complete target domain variance" is more beneficial.
	- **Mechanism**: If document primary variance represents the domain semantic structure, query+document should be stronger; otherwise, if it introduces broad thematic noise, it will dilute query intent.
	- **Design Motivation**: This comparison helps prove that improvements do not simply come from dimensionality reduction or efficiency, but from the selection of fitting samples.

3. **Retention Ratio Scanning and Domain Familiarity Analysis**:
	- **Function**: Analyzes how the compression ratio and the model's inherent domain familiarity affect gains.
	- **Mechanism**: Re-evaluates NDCG@10 across multiple retention ratios and defines domain familiarity using paraphrase robustness—the embedding stability of a model for the same text and its paraphrases.
	- **Design Motivation**: PCA effectiveness may be jointly related to data structure, the model itself, and compression intensity; reporting a single default ratio is insufficient.

### Loss & Training
This paper employs no training loss functions and does not update the encoder. The only "learning" is standard PCA: performing mean centering on the sample matrix $X$ and finding the orthogonal projection $W$ that maximizes projected variance, i.e., retaining the top-$d'$ eigenvectors of the covariance matrix. Retrieval scoring remains cosine similarity in the projected space.

## Key Experimental Results

### Main Results
The main experiments cover 9 dense retrievers with fewer than 2B parameters and 14 MTEB retrieval datasets, with additional discussion on 11 MTEB datasets with insufficient queries. The primary metric is NDCG@10; each model-dataset combination compares the baseline, Query Compression, and Query+Document Compression.

| Setup | Improved Model-Dataset Combinations | Proportion | Key Finding |
|------|----------------------|------|----------|
| Query Compression | 95 / 126 | 75.4% | Most stable; GTE and Sent-T5 improved on 12 / 14 datasets each |
| Query+Document Compression | 71 / 126 | 56.3% | Also helpful, but more prone to dilution by document variance |
| 90% Query Compression | 9 models × 14 datasets | Default setting reported | Most performance drops are < 4%, while gains can be substantial |
| Total retrieval runs | 9 models × 25 datasets × 3 setups | 675 runs | Completed in ~36 hours on an RTX 4090 |

### Ablation Study
| Analysis Item | Key Number | Description |
|--------|----------|------|
| Dataset Success Rate | MedQA, SpartQA, FaithDial, NarrativeQA, ARC, TV2Nord all 9 / 9 | These datasets benefited from Query Compression across all models |
| Query+Document Counter-example | Dis-RoBERTa on Apps dropped -52.8% | Document variance may capture broad themes or styles rather than relevance |
| Retention Ratio | MiniLM on CodeSearch from 77.4 to 82.8, Sent-T5 from 63.0 to 87.3 | Moderate compression can simultaneously preserve information and denoise |
| Low-dim Robustness | ArguAna and MedQA remain strong at 40% retention; GTE peaks at 10% compression on MedQA | Structured query domains are particularly friendly to PCA |
| Comparison with IDA | PCA at 12.1 on SciDocs, 14.3 on FiQA; GPL at 13.4, 14.6 respectively | PCA approaches GPL and outperforms GPL+JPQ and GPL+BPR on 5 shared datasets |

### Key Findings
- Query-only PCA is more reliable than query+document PCA, suggesting that retrieval domain adaptation should center on user information needs rather than overall corpus variance.
- Datasets with strong hierarchical or categorical structures (e.g., medical QA, spatial reasoning, narrative QA) benefit more easily, as principal components are more likely to correspond to true semantic axes.
- PCA is not a replacement for full fine-tuning; GPL remains stronger on ArguAna, NFCorpus, and SciFact. However, PCA's cost is near zero, making it a suitable first-step baseline.

## Highlights & Insights
- The greatest highlight is the reinterpretation of "compression" as "unsupervised domain adaptation." This transforms a common engineering trick into a systematically analyzable adaptation mechanism.
- The discovery that PCA can be fitted using only queries is highly practical. Many enterprise scenarios have large volumes of real query logs but lack high-quality relevance labels; this paper leverages this exact data form.
- The paper does not overcomplicate PCA, instead proving the value of a simple method as a strong baseline through large-scale model-dataset combinations.
- Domain familiarity results are mixed: some models benefit more when familiar with a domain, while others benefit more when unfamiliar. This reminds us that PCA gains are not determined by a single factor and require validation per model and data.

## Limitations & Future Work
- The method assumes the original encoder has already encoded sufficient target domain information; if technical terminology or task semantics are entirely outside the original model's boundary, PCA cannot create knowledge from nothing.
- PCA still requires a sufficient number of unlabeled target domain queries to estimate covariance; principal components may be unstable when queries are too few.
- The optimal retention ratio varies by model and dataset; while the paper suggests a default $r=0.9$, real deployment benefits from a small-scale scan.
- The method is a linear projection; future work could explore combinations with synthetic data, pseudo-label fine-tuning, or non-linear manifold learning.

## Related Work & Insights
- **vs GPL / IDA**: GPL performs domain adaptation via synthetic labels and fine-tuning, which may be stronger but higher in cost; PCA requires no model training yet approaches GPL on SciDocs and FiQA, outperforming compact IDA variants.
- **vs Traditional Vector Compression**: Traditional PCA compression primarily pursues speed and storage efficiency; this paper emphasizes that compression also alters retrieval relevance and can even improve domain adaptation.
- **vs Prompt/Few-shot retriever adaptation**: Methods like Promptagator and CONVERSER rely on LLM-generated data or few-shot examples; PCA relies only on unlabeled queries/documents.
- **Inspiration**: After a RAG system goes online, a lightweight PCA adapter can be learned from online query logs as a low-cost domain adaptation and diagnostic tool before deciding if expensive fine-tuning is warranted.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The method itself is simple, but the systematic demonstration that "query-only PCA improves domain adaptation" is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 9 models, 14+11 datasets, 675 runs, and various analyses provide a solid foundation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear arguments and the experimental organization supports the core conclusions.
- Value: ⭐⭐⭐⭐⭐ Extremely practical, especially for retrieval/RAG systems with query logs but no annotations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)
- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](../../ICML2026/information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
