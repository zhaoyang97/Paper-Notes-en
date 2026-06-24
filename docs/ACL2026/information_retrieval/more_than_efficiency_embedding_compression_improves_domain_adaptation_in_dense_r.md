---
title: >-
  [Paper Note] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Dense Retrieval] This paper demonstrates that PCA vector compression is not merely for acceleration but also serves as a zero-training domain adaptation method for dense retrievers, where fitting PCA solely with target domain queries improves NDCG@10 across 75.4% of model-dataset combinations.
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Dense Retrieval"
  - "Domain Adaptation"
  - "PCA"
  - "Vector Compression"
  - "Unsupervised Retrieval"
date: 2026-05-08
content_hash: 99f9ca2bea65cd2d
---

# More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval

**Conference**: ACL 2026  
**arXiv**: [2601.13525](https://arxiv.org/abs/2601.13525)  
**Code**: The link to the public code is not currently provided.  
**Area**: Information Retrieval / Dense Retrieval  
**Keywords**: Dense Retrieval, Domain Adaptation, PCA, Vector Compression, Unsupervised Retrieval  

## TL;DR
This paper demonstrates that PCA vector compression is not merely for acceleration but also serves as a zero-training domain adaptation method for dense retrievers, where fitting PCA solely with target domain queries improves NDCG@10 across 75.4% of model-dataset combinations.

## Background & Motivation
**Background**: RAG systems rely heavily on dense retrievers. Pre-trained encoders map queries and documents into the same vector space, using cosine similarity to retrieve relevant documents. Mainstream methods for improving cross-domain retrieval involve collecting target domain query-document annotations, generating pseudo labels, or fine-tuning the retriever.

**Limitations of Prior Work**: In specialized domains such as medicine, finance, code, and law, annotating query-document relevance is expensive. Synthetic data and fine-tuning also incur additional computational costs. For already deployed retrievers, many scenarios require a domain adaptation method that does not modify the model, requires no labels, and has low deployment costs.

**Key Challenge**: Conventional wisdom views dimensionality reduction methods like PCA primarily as efficiency optimizations: shorter vectors, smaller indices, and faster retrieval. However, if the principal components of target domain embeddings happen to correspond to the most important semantic variations within the domain, the compression itself may filter out source-domain noise and improve retrieval relevance.

**Goal**: The authors re-examine the role of PCA in dense retrieval, investigating whether it can improve out-of-domain retrieval through low-dimensional projections of target domain embeddings without training the encoder.

**Key Insight**: The paper compares two PCA fitting methods: using only query embeddings versus using query + document embeddings. Intuitively, the document corpus has higher variance, but this does not necessarily represent user information needs; query-only PCA might be more closely aligned with the retrieval task itself.

**Core Idea**: Use the principal components of target domain query embeddings as a task-relevant subspace, projecting both queries and documents into it. This retains discriminative target domain semantics while suppressing irrelevant dimensions.

## Method
The method is highly concise: fix a pre-trained dense retriever without updating any parameters; encode both target domain queries and documents into original vectors; fit a PCA projection matrix from target domain samples; and finally project queries and documents into a low-dimensional space for standard cosine retrieval. The focus is not on algorithmic complexity but on systematically validating "what samples should fit the PCA" and "whether compression truly improves retrieval quality."

### Overall Architecture
Given a query set $Q=\{q_i\}_{i=1}^{n}$ and a document set $D=\{d_j\}_{j=1}^{m}$, the encoder first yields $d$-dimensional representations. PCA learns a projection matrix $W \in \mathbb{R}^{d \times d'}$ on the target domain embedding matrix, where $d' < d$. During retrieval, $q'_i=(q_i-\mu)W$ and $d'_j=(d_j-\mu)W$ are used to calculate cosine similarity for ranking. Default experiments use a retention ratio $r=0.9$, with analysis scanning ratios from 0.1 to 1.0.

### Key Designs

**1. Query Compression: Fitting PCA only with target domain queries, then applying the same projection to documents**

Traditional cross-domain retrieval methods require expensive annotations or fine-tuning. The high variance of a document corpus often stems from differences in topics, styles, or lengths, which do not necessarily correspond to axes that distinguish relevance. This paper fits PCA solely using target domain query embeddings—query distributions more directly express user information needs, and their principal components are more likely to correspond to directions that "distinguish query intent in that domain." Projecting both queries and documents into this query-driven subspace focuses similarity calculations on task-relevant directions, effectively performing domain adaptation without model changes or labels. This setting improved NDCG@10 on 95/126 (75.4%) of model-dataset combinations.

**2. Query+Document Compression Comparison: Testing whether "more complete target domain variance" is truly better**

To rule out the explanation that "improvements only come from dimensionality reduction," the authors established a control group: fitting PCA on the union of queries and documents. The logic is straightforward: if document variance represents the domain's semantic structure, query+document should be stronger; if it introduces broad thematic noise, it will dilute query intent. Results showed it improved only 56.3% of combinations, with negative examples like a -52.8% drop for Dis-RoBERTa on Apps, proving that gains stem from the "selection of fitting samples" rather than the act of reduction itself.

**3. Retention Ratio Scanning and Domain Familiarity Analysis: Quantifying "how much to compress and for whom it works"**

PCA effectiveness is not reported as a single default. The authors scanned NDCG@10 across retention ratios from 0.1 to 1.0 (default $r=0.9$) and defined domain familiarity via paraphrase robustness (embedding stability between a text and its paraphrase). Scanning showed moderate compression balances information retention and denoising (e.g., Sent-T5 on CodeSearch rose from 63.0 to 87.3). Domains with structured queries (medical QA, spatial reasoning) remained robust even at 10%-40% retention, indicating PCA gains are multi-factorial and must be validated per model and data.

### Loss & Training
There is no training loss function or encoder update in this work. The only "learning" is standard PCA: mean-centering sample matrix $X$ and finding orthogonal projection $W$ to maximize variance, retaining the top-$d'$ eigenvectors of the covariance matrix. Retrieval scoring remains cosine similarity in the projected space.

## Key Experimental Results

### Main Results
The main experiment covers 9 dense retrievers with <2B parameters, 14 MTEB retrieval datasets, and 11 additional MTEB datasets with insufficient queries. The primary metric is NDCG@10, comparing baselines with Query Compression and Query+Document Compression.

| Setting | Improved Model-Dataset Combinations | Proportion | Key Findings |
|------|----------------------|------|----------|
| Query Compression | 95 / 126 | 75.4% | Most stable; GTE and Sent-T5 improved on 12 / 14 datasets. |
| Query+Document Compression | 71 / 126 | 56.3% | Helpful, but more susceptible to dilution by document variance. |
| 90% Query Compression | 9 models × 14 datasets | Default | Most performance drops are <4%, while gains can be substantial. |
| Total retrieval runs | 9 models × 25 datasets × 3 settings | 675 runs | Completed in approx. 36 hours on an RTX 4090. |

### Ablation Study

| Analysis Item | Key Metric | Description |
|--------|----------|------|
| Dataset Success Rate | MedQA, SpartQA, FaithDial, NarrativeQA, ARC, TV2Nord are 9 / 9 | These datasets benefited from Query Compression across all models. |
| Query+Document Failures | Apps (Dis-RoBERTa) dropped -52.8% | document variance may capture broad topics/styles rather than relevance. |
| Retention Ratio | CodeSearch: MiniLM 77.4 $\to$ 82.8, Sent-T5 63.0 $\to$ 87.3 | Moderate compression balances information and denoising. |
| Low-dim Robustness | ArguAna and MedQA strong at 40% retention; GTE peaked at 10% on MedQA | Structured query domains are particularly friendly to PCA. |
| Comparison with IDA | PCA: SciDocs 12.1, FiQA 14.3; GPL: 13.4, 14.6 | PCA approaches GPL and exceeds GPL+JPQ/BPR on 5 shared datasets. |

### Key Findings
- Query-only PCA is more reliable than query+document PCA, indicating retrieval domain adaptation should center on user information needs rather than overall corpus variance.
- Datasets with strong hierarchical or categorical structures benefit most (e.g., medical QA, spatial reasoning, narrative QA), as principal components likely correspond to true semantic axes.
- PCA is not a replacement for full fine-tuning; GPL remains stronger on ArguAna, NFCorpus, and SciFact, but PCA's cost is nearly zero, making it a suitable first-step baseline.

## Highlights & Insights
- The biggest highlight is the reinterpretation of "compression" as "unsupervised domain adaptation." This transforms a common engineering trick into a systematically analyzable adaptation mechanism.
- The discovery of fitting PCA only with queries is highly practical. Enterprise scenarios often have massive query logs but lack high-quality relevance labels; this paper leverages that data state.
- The paper avoids over-complicating PCA, proving the strong baseline value of simple methods through large-scale model-dataset combinations.
- Domain familiarity results are mixed: some models benefit more when familiar with a domain, while others benefit when unfamiliar. This serves as a reminder that PCA gains are not determined by a single factor and require validation per model/data.

## Limitations & Future Work
- The method assumes the original encoder has already encoded sufficient target domain information; if specialized terminology or task semantics are entirely beyond the base model's capacity, PCA cannot create knowledge from nothing.
- PCA still requires a sufficient number of unlabeled target domain queries to estimate covariance; if queries are too few, principal components may be unstable.
- The optimal retention ratio is model- and dataset-dependent. While the paper suggests $r=0.9$ as a default, practical deployments should perform small-scale scans.
- The method is a linear projection; future work could explore combinations with synthetic data, pseudo-label fine-tuning, or non-linear manifold learning.

## Related Work & Insights
- **vs GPL / IDA**: GPL performs domain adaptation via synthetic labels and fine-tuning, which is more powerful but costlier. PCA approaches GPL on SciDocs and FiQA and outperforms compact IDA variants without training.
- **vs Traditional Vector Compression**: Conventional PCA compression prioritizes speed and storage; this work emphasizes that compression also alters retrieval relevance and can improve domain adaptation.
- **vs Prompt/Few-shot retriever adaptation**: Methods like Promptagator and CONVERSER depend on LLM data generation; PCA depends only on unlabeled queries/documents.
- **Insight**: After a RAG system is deployed, a lightweight PCA adapter can be learned from online query logs as a low-cost tool for adaptation and diagnosis before deciding if expensive fine-tuning is necessary.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The method is simple, but the systematic demonstration that "query-only PCA improves domain adaptation" is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 9 models, 14+11 datasets, 675 runs, and diverse analyses provide a solid foundation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear arguments with experiments that support core conclusions.
- Value: ⭐⭐⭐⭐⭐ Extremely practical, especially for retrieval/RAG systems with query logs but no labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation](skmteb_slovak_massive_text_embedding_benchmark_and_model_adaptation.md)
- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](../../ICML2026/information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)

</div>

<!-- RELATED:END -->
