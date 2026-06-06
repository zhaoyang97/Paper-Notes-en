---
title: >-
  [Paper Note] Stress Testing Factual Consistency Metrics for Long-Document Summarization
description: >-
  [ACL2026][LLM Evaluation][Factual Consistency] This paper stress-tests six commonly used reference-free factuality metrics in the context of long-document summarization. It reveals that these metrics are significantly af…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Factual Consistency"
  - "Long-Document Summarization"
  - "Robustness Evaluation"
  - "Retrieval-based Scoring"
  - "Metric Stress Testing"
date: 2026-05-08
content_hash: 68a8b194f5ba355f
---

# Stress Testing Factual Consistency Metrics for Long-Document Summarization

**Conference**: ACL2026  
**arXiv**: [2511.07689](https://arxiv.org/abs/2511.07689)  
**Code**: https://github.com/zainmujahid/metricEval-longSum  
**Area**: Text Generation / Evaluation  
**Keywords**: Factual Consistency, Long-Document Summarization, Robustness Evaluation, Retrieval-based Scoring, Metric Stress Testing

## TL;DR
This paper stress-tests six commonly used reference-free factuality metrics in the context of long-document summarization. It reveals that these metrics are significantly affected by meaning-preserving perturbations, retrieval window sizes, and claims with high information density, suggesting that metrics designed for short summaries cannot be reliably transferred to long-document scenarios without caution.

## Background & Motivation
**Background**: While abstractive summarization systems have become increasingly fluent, factual consistency remains a core risk. Traditional metrics like ROUGE/BLEU only measure surface overlap and cannot determine if a summary is supported by the source document. Consequently, several reference-free factuality metrics have emerged, such as NLI-based SummaC, QA-based metrics, generation probability-based BARTScore, and comprehensive metrics like MiniCheck, AlignScore, and UniEval.

**Limitations of Prior Work**: Most of these metrics were developed and validated on short-document summaries, assuming that the source and summary can be encoded together or that evidence exists within a local context. Long-document summarization differs: evidence may span hundreds or thousands of tokens, and a single sentence in the summary might compress information from multiple paragraphs or documents, often requiring metrics to retrieve evidence snippets before judging consistency.

**Key Challenge**: A factually consistent summary should remain consistent after paraphrasing, simplification, compression, or synonym substitution. However, many factuality metrics may rely on local lexical matching, syntactic forms, or specific retrieved snippets, leading to score fluctuations in response to surface-level, meaning-preserving changes.

**Goal**: The authors aim to answer three questions: whether existing factuality metrics are stable under meaning-preserving perturbations; how the retrieval context window in long documents affects these metrics; and whether the information density of summary claims and the dispersion of evidence cause metrics to fail.

**Key Insight**: Instead of proposing a new metric, the paper designs a stress-testing protocol. It generates seven types of meaning-preserving perturbations for original summaries across three long-document datasets and uses a unified retrieval-based scoring framework to compare factuality scores before and after perturbation.

**Core Idea**: If a factuality metric truly evaluates factual consistency, it should remain stable under semantically equivalent perturbations and provide reasonable scores across different retrieval windows and claim densities in long documents. Conversely, large fluctuations in scores serve as evidence of metric fragility.

## Method
The methodology focuses on an evaluation protocol that combines perturbation generation, retrieval-based scoring, and claim density analysis to expose metric distortions in long-document scenarios.

### Overall Architecture
The input consists of source documents and human-written summaries. First, the authors use GPT-4o to generate seven types of meaning-preserving perturbations for each summary: paraphrased, simplified, synonym replaced, less diverse, logically equivalent negated, summarized, and added source text. Then, for each sentence in the original and perturbed summaries, the Top-K similar sentences are retrieved from the source document and expanded into evidence snippets using surrounding windows. Each factuality metric scores the consistency between the summary sentence and the candidate snippets, taking the maximum value as the sentence-level score. These are averaged for a summary-level score. Finally, the authors analyze the score differences and the impact of retrieval window size and claim similarity.

### Key Designs
1.  **Seven Types of Meaning-Preserving Perturbations**:
    - **Function**: To change the surface form of the summary without altering the factual meaning, testing whether metrics are truly semantically robust.
    - **Mechanism**: **Paraphrased** changes syntax and wording; **Simplified** breaks down complex structures; **Synonym Replaced** swaps words with near-synonyms; **Less Diverse** reduces lexical variety; **Negated** uses logically equivalent negative expressions; **Summarized** further compresses the summary; **Added Source Text** inserts sentences that are true in the source but weakly related to the main summary. An NLI-based faithfulness check is used for a sanity check to ensure low contradiction rates (except for Negated).
    - **Design Motivation**: Evaluation should not be affected by style or minor compression. Sensitivity to these perturbations suggests the metric measures local formal matching rather than factual support.

2.  **Retrieval-based Long-Document Factuality Scoring**:
    - **Function**: To enable short-input factuality metrics to function on long documents and observe the effect of retrieval granularity.
    - **Mechanism**: For each summary sentence $s_j$, SBERT embeddings compute similarity with every source sentence to select the Top-K source sentences. Each sentence is expanded into a context snippet $d_{j,k}^{(w)}$ with a window size $w$. The metric $M$ evaluates consistency between $s_j$ and these snippets. The sentence score is $max_k M(s_j,d_{j,k}^{(w)})$. The experiment varies $w=0, 1, 2$ to see if larger contexts improve scores and stability.
    - **Design Motivation**: Evidence in long documents is often not contained in a single sentence. Expanding the window provides necessary local context, but metrics that cannot utilize extra context will not show improvement.

3.  **Claim Information Density / Similarity Analysis**:
    - **Function**: To measure if a summary sentence is a "compressed and evidence-dispersed" claim and analyze its impact on metrics.
    - **Mechanism**: The average cosine similarity between a summary sentence and all source sentences is calculated: $Sim(s_j,D)=1/n * \sum_i cos(e_j,e_i^D)$. High similarity indicates the claim has semantic overlap with many parts of the document (generalized/compressed), while low similarity indicates specific, localized claims.
    - **Design Motivation**: The hardest claims to evaluate are those integrating multiple paragraphs. this analysis reveals if metrics fail when dealing with distributed evidence.

### Loss & Training
No models are trained. Six public factuality metrics are evaluated: BARTScore, MiniCheck, SummaC-Conv, SummaC-ZS, AlignScore, and UniEval. Public versions are used without task-specific fine-tuning. Experiments use the SQuALITY, LexAbSumm, and ScholarQABench datasets, covering fiction, legal judgments, and multi-document scientific QA.

## Key Experimental Results

### Main Results
The three datasets vary significantly: LexAbSumm has the longest documents and structured legal language, while ScholarQABench has the longest summaries.

| Dataset | Samples | Avg. Sum. Sents | Avg. Sum. Tokens | Avg. Doc Sents | Avg. Doc Tokens | Summary Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SQuALITY | 260 | 12.5 | 273 | 456.6 | 6,131 | Human-written |
| LexAbSumm | 351 | 4.2 | 169 | 385.9 | 10,840 | Human-written |
| ScholarQABench | 100 | 43.2 | 1,158 | 575.4 | 14,652 | Human-written |

### Ablation Study
The core "ablation" here is the retrieval window size. Expanding the window generally improves factuality scores, especially in the legal domain, though NLI-based SummaC is less sensitive.

| Metric | ScholarQA w=0 | ScholarQA w=2 | SQuALITY w=0 | SQuALITY w=2 | LexAbSumm w=0 | LexAbSumm w=2 | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BARTScore | 0.03 | 0.02 | 0.03 | 0.03 | 0.15 | 0.16 | Low overall scores; minimal window gain |
| MiniCheck | 0.17 | 0.15 | 0.11 | 0.19 | 0.47 | 0.60 | Significant gain in SQuALITY/Legal |
| SummaC-Conv | 0.22 | 0.25 | 0.22 | 0.24 | 0.33 | 0.34 | Minimal change |
| SummaC-ZS | 0.14 | 0.20 | 0.11 | 0.14 | 0.36 | 0.39 | Slight improvement |
| AlignScore | 0.15 | 0.27 | 0.10 | 0.24 | 0.36 | 0.64 | One of the most window-sensitive |
| UniEval | 0.72 | 0.74 | 0.67 | 0.70 | 0.81 | 0.84 | High baseline and stable |

### Key Findings
- **MiniCheck and UniEval** are the most stable overall but fail significantly on logically equivalent negations (UniEval scores drop to 0.32-0.39).
- **LexAbSumm** is the most unstable domain. Legal language sensitivity results in higher mean absolute score changes for AlignScore, SummaC-ZS, and UniEval.
- **Retrieval window expansion** generally helps, but it indicates that metrics are highly dependent on retrieval configurations rather than being independent ground truths.
- **Claim similarity analysis** shows that in LexAbSumm and SQuALITY, higher density (higher similarity) claims receive lower scores, indicating that highly compressed sentences with dispersed evidence are harder to evaluate.

## Highlights & Insights
- The paper systematically demonstrates exactly *where* current metrics are unreliable for long documents, which is highly valuable for applied research.
- The perturbation selection is comprehensive, distinguishing between lexical, logical, and compression-related vulnerabilities.
- The **Claim Similarity** analysis is clever, concretizing the difficulty of long-doc evaluation as evidence dispersion and semantic hubness.
- **Added Source Text** is a realistic perturbation that tests whether metrics can distinguish between "factually true" and "contextually appropriate."

## Limitations & Future Work
- Perturbations are auto-generated by GPT-4o with only NLI-based sanity checks; there is no large-scale human verification of global equivalence.
- There is no direct alignment between metric outputs and human factuality judgments for these long documents.
- Retrieval strategies are limited to SBERT and Top-K windows; query-aware or multi-hop retrieval was not explored.
- Coverage is limited to English-language fiction, law, and science.

## Related Work & Insights
- **Compared to LongDocFACTScore**: This work follows retrieval-based sentence-level evaluation but focuses on stability analysis rather than proposing a new scoring method.
- **Compared to Ramprasad and Wallace**: This study transfers robustness testing to the long-document domain and adds retrieval/claim density analysis.
- **Insights for Evaluation**: Reporting a single factuality score is insufficient. It is more robust to report perturbation stability, window sensitivity, and performance on high-density claim subsets.

## Rating
- Novelty: ⭐⭐⭐⭐☆
- Experimental Thoroughness: ⭐⭐⭐⭐☆
- Writing Quality: ⭐⭐⭐⭐☆
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/llm_evaluation/document_summarization_with_conformal_importance_guarantees.md)
- [\[ACL 2026\] Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations](pressure-testing_deception_probes_in_llms_scaling_robustness_and_the_geometry_of.md)
- [\[ACL 2026\] PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](politnuggets_benchmarking_agentic_discovery_of_long-tail_political_facts.md)
- [\[ACL 2026\] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/llm_evaluation/document_summarization_with_conformal_importance_guarantees.md)
- [\[ACL 2026\] Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations](pressure-testing_deception_probes_in_llms_scaling_robustness_and_the_geometry_of.md)
- [\[ACL 2026\] PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](politnuggets_benchmarking_agentic_discovery_of_long-tail_political_facts.md)
- [\[ACL 2026\] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)

</div>

<!-- RELATED:END -->
