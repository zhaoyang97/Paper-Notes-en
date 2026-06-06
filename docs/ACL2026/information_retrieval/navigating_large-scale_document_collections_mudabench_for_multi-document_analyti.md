---
title: >-
  [Paper Note] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA
description: >-
  [ACL2026][Information Retrieval & RAG][Multi-document analytical QA] This paper proposes MuDABench, advancing multi-document QA from "finding a few relevant snippets" to "extraction, aggregation…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Multi-document analytical QA"
  - "RAG evaluation"
  - "financial documents"
  - "metadata planning"
  - "Agent workflow"
date: 2026-05-08
content_hash: 957cdcb6d1669a12
---

# Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA

**Conference**: ACL2026  
**arXiv**: [2604.22239](https://arxiv.org/abs/2604.22239)  
**Code**: https://github.com/Zhanli-Li/MuDABench  
**Area**: Information Retrieval / Multi-Document QA / Document Intelligence  
**Keywords**: Multi-document analytical QA, RAG evaluation, financial documents, metadata planning, Agent workflow

## TL;DR
This paper proposes MuDABench, advancing multi-document QA from "finding a few relevant snippets" to "extraction, aggregation, and quantitative analysis across large-scale semi-structured document collections." It demonstrates that standard RAG struggles with such tasks even with increased recall, whereas metadata-aware multi-Agent workflows significantly improve results but still lag far behind human experts.

## Background & Motivation
**Background**: In current corporate knowledge bases, web QA, and document QA systems, the dominant paradigm is RAG: segmenting documents into chunks, retrieving a few relevant chunks from an approximately flat corpus pool, and having the LLM generate an answer within a single context window. Multi-hop QA datasets such as HotpotQA, 2WikiMultiHopQA, MuSiQue, and FanOutQA continue this setting, while long-context benchmarks focus more on whether models can fit longer inputs into their windows.

**Limitations of Prior Work**: Real-world multi-document analytical tasks often involve "treating document collections as semi-structured databases for analysis" rather than "finding evidence sentences." For instance, if a regulatory body wants to know which companies changed accounting firms in 2024, it must filter reports by company and year, extract accounting firm fields page by page, align 2023 and 2024 records, and finally aggregate the list of changes. Missing a single report, misreading a table, or confusing years leads to incorrect conclusions.

**Key Challenge**: Most existing multi-document QA benchmarks contain only a few webpages or short documents, primarily testing cross-entity multi-hop reasoning. Financial document benchmarks like FinanceBench focus more on single-document QA, while FinAgentBench emphasizes retrieval localization. Systems work like Aryn and DocETL discuss multi-step pipelines but lack public large-scale standard benchmarks. Consequently, current evaluations do not stress-test the complete chain of "large volume + explicit metadata + single-doc extraction + cross-document aggregation + numerical analysis."

**Goal**: The authors aim to define a task closer to real-world institutional document analysis: given a collection of financial documents with metadata and a natural language analytical question, the system must identify relevant documents, extract necessary facts from each, structure these facts, and perform aggregation operations such as sorting, comparison, variance, or growth rate calculations to provide a final answer.

**Key Insight**: MuDABench utilizes distant supervision through the correspondence between public financial disclosures and authoritative financial databases. Structured databases provide verifiable indicator values, while PDF disclosures provide real noise, long documents, tables, and cross-year contexts. Experts then transcribe structured indicators into natural language intermediate facts and question templates.

**Core Idea**: Construct a multi-document analytical QA benchmark using "financial document collections + metadata + intermediate fact annotations," and replace flat RAG with a "plan, document-level extraction, JSON normalization, code-based aggregation" Agent workflow.

## Method
MuDABench serves both as a benchmark and a reference solution. The dataset emphasizes organizing real financial disclosures into evaluable multi-document tasks, while the methodology demonstrates why standard RAG is insufficient and how a more structured multi-Agent pipeline should function.

### Overall Architecture
The authors gathered reports including annual reports, ESG reports, and announcements for Chinese A-share and US-listed companies from sources like cninfo and SEC, totaling 589 documents and over 80,000 pages. On average, each question is associated with 14.8 PDFs and 149.7 pages, a scale significantly larger than most existing benchmarks.

Each document has explicit metadata, including stock code, fiscal year, and document type. These fields are not the answers themselves but serve as critical entry points for system filtering and task planning. For example, to find "the three companies with the highest revenue growth from 2022 to 2023," the system must first identify which documents belong to which year, company, and report type before extracting values and calculating growth.

Question construction combines distant supervision with expert review. Authors compiled structured data (revenue, cost, dividends, ESG metrics, executive info) from authoritative databases and had financial experts design templates covering trend analysis, peer comparison, outlier identification, and year-over-year changes. For each question, they annotated not only the final answer but also a set of natural language intermediate facts required to reach it, used to diagnose whether the model successfully extracted necessary information.

During evaluation, the system takes question $Q_j$, document collection $D_j$, and metadata collection $M_j$ as input to output a final answer. The format is $X = \{(Q_j, D_j, M_j, S_j)\}$, where $S_j$ is the set of required intermediate facts. This allows MuDABench to check both string accuracy and where the extraction chain fails.

The reference system is a metadata-aware multi-agent workflow. Instead of flat RAG, it splits analysis into four stages: a Planning Agent generates per-document sub-queries based on metadata schemas; a Document-Level Extractor performs RAG on each relevant document; a Normalization Agent converts natural language extractions into flat JSON; and a Code Agent writes programs based on the JSON schema to perform aggregation (sorting, filtering, statistics) and generate the final answer.

### Key Designs
1.  **Metadata-based Benchmark Construction**:
    *   **Function**: Organizes large-scale financial disclosures into semi-structured collections, aligning multi-document QA with real analysis scenarios.
    *   **Mechanism**: Binds each document to three metadata types (stock code, fiscal year, document type). Each question is linked to a set of 5 to 38 PDFs. Questions range from simple single-year statistics to complex queries requiring cross-year filtering and computation.
    *   **Design Motivation**: Real-world document libraries naturally possess metadata; ignoring this structure forces RAG into blind retrieval. By including metadata, systems must learn to narrow the analysis space before extraction and aggregation.

2.  **Intermediate Fact-driven Evaluation Protocol**:
    *   **Function**: Measures whether the system extracted the necessary facts to support the answer, beyond just final accuracy.
    *   **Mechanism**: For standard RAG, an LLM-as-judge assesses how many gold supporting facts were covered, using a double-check judge to estimate error/omission rates. For the workflow, correct extraction of "metric cells" from source tables is calculated. Metrics reported include process accuracy, final-answer accuracy, and full accuracy (requiring both to be correct).
    *   **Design Motivation**: In multi-document analysis, a final answer might appear correct due to chance or lenient judging. Intermediate fact annotation allows researchers to distinguish failures in retrieval, extraction, or aggregation.

3.  **Plan-Extract-Normalize-Code-Aggregate Agent Pipeline**:
    *   **Function**: Replaces single-shot RAG with a modular process to handle document collections exceeding context windows.
    *   **Mechanism**: The Plan Agent generates sub-query templates with metadata constraints. The Document-Level Extractor instantiates these for specific documents. The Norm Agent defines a flat JSON schema from few-shot examples and converts dialogue into records. The Code Agent writes executable programs to analyze the structured data.
    *   **Design Motivation**: Complex questions often require extracting similar data from dozens or hundreds of documents. Expecting an LLM to read everything at once leads to entity/year confusion; separating extraction from computation makes numerical analysis controllable and scalable.

### Loss & Training
This work does not propose a new model requiring training but rather a benchmark, evaluation protocol, and inference workflow. The RAG baseline uses OpenAI File Search with GPT-4o. In the Agent workflow, Planning and Code generation use DeepSeek-R1-0528, Normalization uses DeepSeek-Chat-V3-0324, and single-document extraction relies on OpenAI File Search. All temperatures are set to 0.

## Key Experimental Results

### Main Results
Table 1 summarizes MuDABench results, showing process/final/full accuracy for Simple and Complex questions. Simple RAG, even with recall up to $2.5|D|$, yields low accuracy. The workflow shows significantly higher process coverage, but final-answer accuracy for complex questions still lags far behind human experts.

| System Config | Simple Proc | Simple Final | Simple Full | Complex Proc | Complex Final | Complex Full |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o RAG, chunk = $1|D|$ | 0.1572 | 0.0663 | 0.0241 | 0.1459 | 0.0482 | 0.0181 |
| GPT-4o RAG, chunk = $2|D|$ | 0.1793 | 0.1265 | 0.0422 | 0.2212 | 0.0361 | 0.0181 |
| GPT-4o RAG + metadata, chunk = $2.5|D|$ | 0.2514 | 0.1325 | 0.0542 | 0.2522 | 0.0422 | 0.0120 |
| WF + GPT-4o, chunk = 1 | 0.4179 | 0.0667 | 0.0000 | 0.4021 | 0.0667 | 0.0095 |
| WF + GPT-4.1 mini, chunk = 3 | 0.5803 | 0.2430 | 0.0654 | 0.5338 | 0.0865 | 0.0673 |
| WF + GPT-4.1 mini, chunk = 5 | 0.5888 | 0.2243 | 0.0748 | 0.5749 | 0.1619 | 0.1143 |
| Noise WF + GPT-4.1 mini, chunk = 5 | 0.5961 | 0.1636 | 0.0727 | 0.5680 | 0.1238 | 0.0762 |
| Human Performance | 0.8940 | 0.8334 | 0.7334 | 0.8120 | 0.7334 | 0.6667 |

Two conclusions emerge: First, increasing recall primarily improves process accuracy rather than final answer stability; e.g., Complex process accuracy for GPT-4o RAG rose from 0.1459 to 0.2623 at $2.5|D|$, but final accuracy remained at 0.0482. Second, the workflow is better suited for these tasks, with WF + GPT-4.1 mini at chunk = 5 reaching 0.1619 on Complex final accuracy (roughly 3x the RAG baseline), yet still far below the human performance of 0.7334.

### Ablation Study
Rather than traditional component removal, analysis focused on chunk budget, metadata, noise, and step-level diagnostics. Table 2 shows that document category and chunk count affect process accuracy, indicating that long docs, scanned announcements, and language differences impact extraction difficulty.

| Doc Category | Avg Length | Chunk=1 Simple | Chunk=1 Complex | Chunk=3 Simple | Chunk=3 Complex | Chunk=5 Simple | Chunk=5 Complex |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A-share Annual (CN) | 499k tokens | 0.4696 | 0.4555 | 0.6537 | 0.6674 | 0.6447 | 0.6570 |
| A-share ESG (CN) | 72k tokens | 0.3998 | 0.3898 | 0.6067 | 0.4813 | 0.5865 | 0.4992 |
| A-share Announce (CN) | 144k tokens | 0.3903 | 0.3786 | 0.5222 | 0.4976 | 0.5542 | 0.5575 |
| US-stock Annual (EN) | 120k tokens | 0.4472 | 0.3955 | 0.3167 | 0.5374 | 0.4643 | 0.7375 |

Table 3 presents step-level error analysis on 30 random samples. Planning and Code phases appear relatively controlled, and Normalization is highly accurate, but the Extraction phase is fragile, particularly for complex questions.

| Step | Simple Indep. Acc | Complex Indep. Acc | Avg | Avg Acc given Prev. Step Correct | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Planning | 86.7% | 93.3% | 90.0% | 90.0% | Most sub-queries are reasonable; lack of domain knowledge causes directional errors. |
| Extraction | 40.0% | 20.0% | 30.0% | 25.9% | Primary bottleneck; complex questions drop to 14.3% even if planning is correct. |
| Normalization | 100.0% | 100.0% | 100.0% | 100.0% | Aligning results to flat JSON is easy if extraction is correct. |
| Code | 93.3% | 93.3% | 93.3% | 85.7% | Errors in coding or JSON pathing still occur. |

### Key Findings
*   RAG's core issue is not simply "insufficient recall." As budget increases, process coverage improves, but the final answer often fluctuates or decreases, suggesting LLMs cannot reliably transform fragmented evidence into cross-document statistical conclusions.
*   Metadata is helpful but insufficient. Putting metadata in prompts provides coarse structure, but without explicit planning and per-document execution, models still confuse companies, years, and types.
*   The Agent workflow's primary gain comes from reframing the task as "table construction + program analysis." This decouples analysis from one-shot long-context reasoning, making numerical analysis controllable and scalable.
*   Extraction is the bottleneck, not calculation. Extraction accuracy averages only 30.0%; once a key cell is missed, the subsequent JSON and code cannot recover it.
*   Noisy documents drag down the final answer. Noise WF shows that while process accuracy might not suffer, final accuracy on Simple questions drops from 0.2243 to 0.1636, indicating irrelevant documents interfere with aggregation.

## Highlights & Insights
*   MuDABench shifts the multi-document QA paradigm from "multi-hop reasoning" to "collection analysis," which is closer to professional knowledge work (collecting local facts and then filtering/sorting/comparing/statistically analyzing them).
*   Intermediate fact annotation is crucial. Many datasets only provide final answers; this work exposes the fact set, allowing researchers to see if the failure was in retrieval, extraction, normalization, or calculation.
*   Metadata is a structural entry for task planning, not prompt decoration. Systems should use metadata to generate structured query plans rather than treating it as additional text.
*   Code-based aggregation is a transferable idea for tasks like scientific table extraction or auditing, where per-document evidence can be converted into flat records for verifiable program analysis.
*   Long context does not equate to large-scale document analysis capability. Even with long windows, systems must solve entity alignment, year alignment, table parsing, and computational executability.

## Limitations & Future Work
*   Domain limitation: Currently focused on finance. Whether other fields (law, medicine, research) can use distant supervision to construct high-quality intermediate facts needs validation.
*   Scale constrained by cost: While distant supervision allows for expansion, evaluation is expensive, and more samples may not introduce new error patterns.
*   LLM-as-judge uncertainty: While double-checking was used, financial atomic facts have nuances in granularity and numerical tolerance; process accuracy is not an absolute truth.
*   Dependency on closed-source services: The workflow relies on OpenAI File Search and high-tier models, presenting barriers to reproduction and control.
*   Extraction remains a "hard problem": Long PDFs, complex tables, and year confusion (current vs. previous FY) cause errors. Future work requires stronger document parsing and domain validation.
*   Code Agent robustness: Small errors in key names or JSON paths cause program failure. Future iterations could include schema validators and unit-test styled execution feedback.

## Related Work & Insights
*   **vs HotpotQA / 2WikiMultiHopQA / MuSiQue / FanOutQA**: These emphasize multi-hop reasoning over a few webpages; MuDABench focuses on collection analysis across dozens of long PDFs.
*   **vs LongBench / RULER / Loong**: These test context capacity; MuDABench requires extraction and aggregation across a collection exceeding the context window.
*   **vs FinanceBench**: FinanceBench focuses on single-document QA; MuDABench extends finance to multi-company, multi-year analysis.
*   **vs FinAgentBench**: FinAgentBench focuses on identifying types and snippets; MuDABench adds batch extraction, structuring, and quantitative analysis.
*   **Insights**: A practical document analysis system should behave like a database query executor, integrating schema-aware planning, document-grounded extraction, structured output, and executable analysis code with process-level verification.

## Rating
*   Novelty: ⭐⭐⭐⭐☆ Systematically defines analytical QA on semi-structured collections; the benchmark perspective is very clear.
*   Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers chunk budgets, metadata, noise, and step-level failures. Samples are diagnostic, though scale and closed-source dependencies limit wider reproduction.
*   Writing Quality: ⭐⭐⭐⭐☆ Motivation and error analysis are direct and helpful for understanding RAG failures.
*   Value: ⭐⭐⭐⭐⭐ Highly relevant for document intelligence, enterprise knowledge bases, and agentic RAG by emphasizing intermediate fact quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval](a_picture_is_worth_a_thousand_words_an_empirical_study_of_aggregation_strategies.md)

</div>

<!-- RELATED:END -->
