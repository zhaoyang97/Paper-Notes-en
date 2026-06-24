---
title: >-
  [Paper Note] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA
description: >-
  [ACL 2025][LLM (Other)][Financial QA] HiREC proposes a hierarchical retrieval and evidence curation framework that first retrieves relevant documents and then selects passages from them. It filters out irrelevant passages and automatically generates complementary queries to retrieve missing information. On the LOFin benchmark containing 145k SEC documents, it improves answer accuracy by over 13% compared to the strongest RAG baseline.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Financial QA"
  - "Hierarchical Retrieval"
  - "Evidence Curation"
  - "RAG"
  - "Standardized Documents"
date: 2026-05-08
content_hash: 55fc30690f7b657f
---

# Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA

**Conference**: ACL 2025  
**arXiv**: [2505.20368](https://arxiv.org/abs/2505.20368)  
**Code**: [GitHub](https://github.com/deep-over/LOFin-bench-HiREC)  
**Area**: NLP Understanding  
**Keywords**: Financial QA, Hierarchical Retrieval, Evidence Curation, RAG, Standardized Documents

## TL;DR

HiREC proposes a hierarchical retrieval and evidence curation framework that first retrieves relevant documents and then selects passages from them. It filters out irrelevant passages and automatically generates complementary queries to retrieve missing information. On the LOFin benchmark containing 145k SEC documents, it improves answer accuracy by over 13% compared to the strongest RAG baseline.

## Background & Motivation

**Limitations of Prior Work**: Standardized financial documents (such as SEC annual reports on Form 10-K) share consistent templates across different companies and periods. They contain highly similar table structures (e.g., operating income tables for Amazon, Meta, and Walmart differ only in numerical values). Traditional RAG methods struggle to distinguish these near-duplicate texts, leading to retrieval confusion and redundancy.

**Insufficient Datasets**: Existing financial QA benchmarks (such as FinanceBench with only 150 questions/368 documents and SEC-QA without a fixed test set) are small-scale and closed-source, failing to reflect real-world scenarios.

**Key Challenge**: A large number of comparison questions in financial QA (e.g., "What is the difference between the operating incomes of Amazon and Walmart in 2023?") require complete retrieval of all necessary evidence from multiple documents, where single-turn retrieval often misses critical information.

## Method

### Overall Architecture

HiREC consists of two core components and operates via an iterative workflow (up to $i_{\max}=3$ rounds):

**A. Hierarchical Retrieval**
- Step 1: Document-level Retrieval $\rightarrow$ Narrows down the search space
- Step 2: Passage-level Retrieval $\rightarrow$ Selects the most relevant passages from candidate documents

**B. Evidence Curation**
- Step 3: Passage Filtering $\rightarrow$ Removes irrelevant passages
- Step 4: Answerability Judgment $\rightarrow$ Evaluates whether the evidence is sufficient
- Step 5: Complementary Query Generation $\rightarrow$ Generates new queries to trigger the next round of retrieval if evidence is insufficient

**C. Answer Generation**
- Program-of-Thought (PoT) is used for numerical questions, while Chain-of-Thought (CoT) is used for textual questions.

### Key Designs

**1. Document Indexing and Retrieval (Document Retriever)**

Standardized documents are lengthy and structurally uniform, making it difficult for a single vector to capture all key information. HiREC adopts a **cover-page summary indexing** strategy:
- An LLM-based method extracts core distinguishing information (company name, report type, fiscal period) from each document's cover page to generate a summary $d'$.
- A bi-encoder (E5 model) precomputes summary embeddings $\mathbf{v}_d = E^D(d')$ to store in the document library.
- During retrieval: query $q$ $\rightarrow$ LLM converts $q$ into a refined query $q'$ (removing distracting financial jargon) $\rightarrow$ dense retrieval fetches $k'_D$ candidates $\rightarrow$ cross-encoder (DeBERTa-v3) reranks to select the top-$k_D$ ($k_D=5$).

**2. Passage Retriever**

- Within the retrieved document set $\mathcal{D}_r$, a cross-encoder calculates $\text{CrossEncoder}^P(q, p)$ for each passage $p$, selecting the top-$k_P$ ($k_P=5$) passages.
- **Key Improvement**: Standard pretrained rerankers perform poorly on financial tables. This work **fine-tunes the cross-encoder** on the FinQA training set: for each question $q$ and evidence table $p$, $n_{\text{neg}}=8$ negative samples (tables from non-evidence pages) are sampled, and the model is trained utilizing a binary cross-entropy loss:

$$\mathcal{L} = \sum_{(q,p) \in \mathcal{X}} \left[ -\log(\text{CE}^P(q,p)) - \sum_{p' \in \mathcal{P}^-} \log(1 - \text{CE}^P(q,p')) \right]$$

**3. Three-Step Evidence Curation Process**

Three modules are performed in a single LLM call:

- **Passage Filter**: Removes passages irrelevant to the question from $\mathcal{P}_r$ and retains at most $k'_P=10$ passages to form $\mathcal{P}_f$, while considering passages previously confirmed as relevant in earlier iterations.
- **Answerability Checker**: Assesses whether $\mathcal{P}_f$ contains sufficient information to answer the question. If sufficient, the system proceeds to answer generation; otherwise, it triggers complementary retrieval.
- **Complementary Question Generator**: Analyzes information gaps in $\mathcal{P}_f$ and generates a complementary query $q_c$ for the next round of hierarchical retrieval (e.g., if only Amazon's operating income is retrieved $\rightarrow$ it generates a complementary query like "Walmart operating income 2023").

**4. LOFin Benchmark Construction**

- **Corpus**: Collects 10-K/10-Q/8-K filings of S&P 500 companies from 2001 to 2025 via SEC EDGAR, totaling 145,897 documents across 516 companies.
- **QA Pairs**: Handcrafted using FinQA (transformed from closed-domain to open-domain), FinanceBench (directly adopted), and SEC-QA multi-document templates, yielding 1,595 pairs.
- **Evidence Annotation**: Uses a two-step automatic matching approach consisting of BM25 and NLI, followed by human verification.

### Loss & Training

- Passage Retriever Fine-tuning: DeBERTa-v3, $n_{\text{neg}}=8$, batch size 128, 3 epochs, lr $2 \times 10^{-7}$, trained on a single RTX 4090 GPU.
- Other LLM modules (query transformation, summarization, evidence curation) utilize Qwen-2.5-7B-Instruct.
- Answer generation utilizes GPT-4o.

## Key Experimental Results

### Main Results: Comprehensive Comparison on LOFin-1.4k

| Method | Page Recall | Answer Acc | Avg. Passages |
|------|-----------|-----------|-----------|
| GPT-4o (Zero-shot) | - | 13.92 | - |
| Perplexity | - | 10.55 | - |
| Self-RAG | 18.96 | 7.63 | 10.0 |
| RQ-RAG | 18.54 | 8.34 | 36.0 |
| IRCoT | 25.15 | 22.31 | 20.0 |
| Dense | 34.78 | 29.22 | 10.0 |
| HHR | 33.31 | 28.67 | 10.0 |
| **HiREC** | **45.35** | **42.36** | **3.7** |

Compared to Dense, HiREC improves page recall by over 10% and answer accuracy by over 13%, using only 3.7 passages on average.

### Ablation Study

| Configuration | Page Precision | Page Recall | Answer Acc |
|------|---------------|-------------|-----------|
| HiREC (Full) | 21.79 | 45.35 | 42.36 |
| w/o HR (No Hierarchical Retrieval) | 14.75 | 34.16 | 32.76 |
| w/o EC (No Evidence Curation) | 4.70 | 41.41 | 36.70 |
| w/o Fine-tuning | 21.07 | 42.77 | 40.13 |
| w/o Filter (No Filter) | 8.43 | 50.19 | 42.08|

- Removing Hierarchical Retrieval (HR) has the largest impact (Acc decreases by 9.6%), demonstrating the core value of document-level pre-screening.
- Although removing the filter yields the highest recall (50.19), the accuracy does not improve, suggesting that it introduces conflicting or erroneous information.
- Even without fine-tuning the reranker, HiREC still outperforms the Dense baseline by over 10%.

### Cross-Generator Analysis

| Method | Qwen-2.5-7B | DeepSeek-14B | GPT-4o |
|------|------------|-------------|--------|
| Dense | 23.87 | 30.77 | 29.22 |
| HiREC | 32.32 | **38.76** | **42.36** |

HiREC combined with DeepSeek-14B (38.76) outperforms Dense with GPT-4o (29.22) by over 9%, indicating that high-quality retrieval can compensate for the performance gap in generator models.

### Key Findings

- **Hierarchical retrieval is the core of performance**: First locating the correct company/document significantly reduces confusion from near-duplicate texts.
- **Iterative evidence curation drives continuous improvement**: With each iteration, both recall and precision steadily improve, while the number of passages per query decreases.
- **Evident cost-efficiency advantages**: The token consumption of HiREC during the retrieval stage is only ~45% of IRCoT, and only ~30% during the generation stage.
- **Small models can substitute large models for curation**: Qwen-2.5-7B can effectively perform evidence curation, eliminating the need for expensive API fees.

## Highlights & Insights

1. **Cover-page summary indexing** is an elegant design specifically tailored for standardized documents—it relies on the most distinguishing information (company name and period) rather than the entire text for document-level retrieval.
2. **Complementary query generation** addresses the inherent difficulty of comparative questions by automatically identifying and filling information gaps rather than expecting users to rephrase.
3. **LOFin Benchmark**: An open-domain financial QA dataset scaling up to 145k documents, which is 100 times larger than the previous largest benchmark.
4. **Retrieval Quality > Generator Capacity**: The finding that a smaller model paired with high-quality retrieval outperforms a larger model paired with basic retrieval holds strong practical implications.

## Limitations & Future Work

1. **Domain Specificity**: The cover-page summary indexing strategy relies on the structural characteristics of financial documents. Generalizing this method to other standardized documents (such as legal contracts or medical records) requires domain-specific adaptation.
2. **LLM Dependency**: All three modules of evidence curation rely on LLMs, keeping the computational overhead non-negligible in extremely cost-sensitive environments.
3. **Multi-document Reasoning Limits**: Currently, the system handles at most $k_D=5$ documents, which may be insufficient for complex analytical scenarios requiring reasoning across dozens of documents.
4. **Table Understanding**: Despite fine-tuning the passage retriever, complex calculation and reasoning on financial tables remain core bottlenecks (with an accuracy of only 37% on numerical table categories).

## Related Work & Insights

- **Financial RAG**: Graph-structured methods such as GraphRAG and HybridRAG complement the hierarchical approach proposed in this work.
- **Iterative Retrieval**: IRCoT and Self-RAG use previous retrieval contexts as inputs for subsequent queries. In contrast, HiREC is designed to explicitly discover missing information rather than simply reuse existing contexts.
- **Insights**: (1) Retrieval for standardized documents should prioritize "document identification before passage retrieval"; (2) Complementary query generation can be applied to any QA scenario requiring the aggregation of multiple pieces of evidence.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 7 | The combined design of hierarchical retrieval, evidence curation, and complementary queries is highly effective. |
| Technical Depth | 7 | The multi-stage pipeline design is complete, and the fine-tuning strategy is sound. |
| Experimental Thoroughness | 9 | Features a large-scale benchmark, multi-method comparisons, ablation studies, cross-model analysis, and cost analysis. |
| Writing Quality | 8 | Highly clear framework diagrams, complete pseudocode, and deep analysis of error types. |
| Value | 8 | The LOFin benchmark and HiREC framework are immediately applicable to real-world financial scenarios. |
| Overall Score | 7.5 | A practical RAG framework oriented towards real-world financial scenarios, possessing both great engineering and academic value. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)
- [\[ACL 2025\] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA](team_anotheroption_at_semeval-2025_task_8_bridging_the_gap_between_open-source_a.md)
- [\[ACL 2025\] Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)
- [\[ACL 2025\] Hierarchical Attention Generates Better Proofs](hierarchical_attention_generates_better_proofs.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)

</div>

<!-- RELATED:END -->
