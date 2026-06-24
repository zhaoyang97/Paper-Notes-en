---
title: >-
  [Paper Note] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation
description: >-
  [ACL 2025][LLM (Other)][Legal Interpretation] This work proposes the ATRIE framework, which simulates the doctrinal legal research workflow of legal experts. It leverages LLMs to automatically retrieve relevant information from case databases, generate interpretations of legal concepts, and evaluate interpretation quality, thereby eliminating the reliance on human legal experts.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Legal Interpretation"
  - "RAG"
  - "Legal Concept Entailment"
  - "Doctrinal Legal Research"
  - "LLM"
date: 2026-05-08
content_hash: 3aed7dfb6eef4deb
---

# ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation

**Conference**: ACL 2025  
**arXiv**: [2501.01743](https://arxiv.org/abs/2501.01743)  
**Code**: [GitHub](https://github.com/lkc233/ATRIE)  
**Area**: Legal NLP / Retrieval-Augmented Generation  
**Keywords**: Legal Interpretation, RAG, Legal Concept Entailment, Doctrinal Legal Research, LLM  

## TL;DR

This work proposes the ATRIE framework, which simulates the doctrinal legal research workflow of legal experts. It leverages LLMs to automatically retrieve relevant information from case databases, generate interpretations of legal concepts, and evaluate interpretation quality, thereby eliminating the reliance on human legal experts.

## Background & Motivation

**Research Problem:** Legal interpretation—especially of vague legal concepts—is crucial to legal practice. However, existing methods heavily rely on human legal experts, resulting in high time consumption, delayed updates, and issues with subjective incompleteness.

**Limitations of Prior Work:** Prior research (e.g., Savelka et al. 2023) utilizes GPT-4 combined with human-annotated key sentences from cases to interpret legal concepts. However, it still depends on legal experts to (1) manually annotate valuable sentences in cases related to the concepts, and (2) evaluate the quality of the generated interpretations, making automated scaling impossible.

**Core Motivation:** Inspired by doctrinal legal research methodology, this work simulates the workflow of legal experts in reading, extracting, and summarizing from large-scale historical cases to construct a fully automated framework for legal concept interpretation.

## Method

### Overall Architecture

ATRIE consists of two core modules: the **Legal Concept Interpreter** and the **Legal Concept Interpretation Evaluator**. The interpreter uses a RAG framework to retrieve information from a case database and generate interpretations; the evaluator automatically measures the quality of the interpretations based on performance changes in the downstream task, Legal Concept Entailment (LCE).

### Key Designs

1. **Three-Stage Case Retrieval and Extraction:** (1) Exact string matching retrieves case set $\mathcal{D}_0$ mentioning the target concept from China Judgements Online database; (2) An LLM filters out relevant cases $\mathcal{D}_1$ where the court's view discusses the concept in detail, and extracts the reasons for applicability/non-applicability; (3) Balanced sampling of positive and negative samples constructs the final dataset $\mathcal{D}$ and the reason set $\mathcal{R}$.

2. **Structured Interpretation Generation:** Taking legal provisions, vague concepts, the reason set, and example interpretations as input, the LLM is prompted to output a three-part interpretation: Analysis (basic meaning and application conditions), Case Examples (positive and negative cases), and Judicial Discretion (discretionary standards for judges).

3. **Legal Concept Entailment (LCE) Evaluation Task:** Given factual descriptions of cases, the task is to determine whether a vague concept is applicable and provide reasons. Different interpretations are used as reference inputs for the LLM to perform the LCE task, and changes in classification accuracy serve as surrogate indicators of interpretation quality. The evaluation covers both classification tasks (Accuracy, Macro-F1) and reason generation tasks (GPT-4o consistency score of 1-10).

### Loss & Training

The evaluator does not involve training losses; instead, it measures the quality of interpretation by locking the performance changes of the LLM on the downstream LCE task. The interpreter uses Qwen2.5-72B with the temperature set to 0.9 to encourage diverse outputs.

## Experiments

### Main Results

| Method | Acc (72B) | Ma-F (72B) | CS (72B) | Acc (14B) | Ma-F (14B) | CS (14B) |
|------|-----------|------------|----------|-----------|------------|----------|
| Random | 51.66 | 50.32 | / | 51.66 | 50.32 | / |
| Zero-Shot | 71.38 | 61.42 | 5.658 | 70.92 | 59.88 | 5.525 |
| Chain-of-Thought | 71.95 | 63.46 | 5.717 | 71.52 | 61.01 | 5.666 |
| Judicial Interp. | 72.10 | 66.54 | 5.573 | 70.92 | 65.23 | 5.347 |
| Expert Interp. | 72.13 | 65.30 | 5.630 | 71.95 | 66.01 | 5.581 |
| Direct Interp. | 72.35 | 67.18 | 5.642 | 72.72 | 66.90 | 5.677 |
| **ATRIE** | **75.03** | **70.87** | **5.946** | **74.50** | **70.39** | **5.840** |

### Ablation Study

| Retrieval Method | Ma-F (14B) | CS |
|----------|------------|------|
| No Retrieval | 66.90 | 5.677 |
| String Match | 69.04 | 5.772 |
| + Filter | 69.60 | 5.817 |
| + Filter + Balance (ATRIE) | 70.39 | 5.840 |

| Ablation of Interpretation Components | Macro-F1 |
|-------------|----------|
| w/o Example Cases | 67.41 |
| w/o Analysis | 70.43 |
| w/o Judicial Discretion | 70.69 |
| ATRIE (Full) | 70.87 |

### Key Findings

- ATRIE significantly outperforms interpretations written by human legal experts (Expert/Judicial Interpretation) across almost all metrics, demonstrating the advantages of LLMs in large-scale case analysis.
- Case examples (Example Cases) are the most critical component of the interpretation; removing them drops Macro-F1 by 3.46 percentage points.
- General LLMs (Qwen2.5-72B) significantly outperform domain-specific legal LLMs (Farui-plus) on the legal interpretation task, which is attributed to their stronger long-context understanding and generation capabilities.
- Inputting more cases continuously improves the quality of interpretations, matching the actual working experience of legal experts.

## Highlights & Insights

- Completing the automated workflow simulation of doctrinal legal research, establishing a closed loop from retrieval to generation to evaluation.
- Proposing Legal Concept Entailment as an objective and reproducible evaluation method for legal concept interpretation quality, replacing subjective human evaluation.
- Human evaluation demonstrates that generated interpretations outperform expert interpretations in comprehensiveness and readability, with only a minor gap in precision.

## Limitations & Future Work

- This work is validated only on the Chinese legal system and Chinese cases; generalization across different legal systems remains unknown.
- Although the chosen 16 legal concepts are representative, the overall coverage is limited.
- The exact string-matching retrieval strategy may miss relevant cases when legal terms have synonymous expressions.

## Related Work & Insights

- **Legal Interpretation:** Savelka et al. (2023) utilized GPT-4 to interpret legal concepts based on human-annotated sentences; Coan & Surden (2024) directly generated constitutional interpretations using GPT.
- **Doctrinal Legal Research Automation:** Yung-chin Su (2024) suggested that Legal AI can replace human experts in case reading and theory extraction.
- **RAG for Law:** The RAG framework proposed by Lewis et al. (2020) is applied in the legal domain to enhance LLM understanding of specific cases.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 7 |
| Practicality | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Overall Rating| 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)

</div>

<!-- RELATED:END -->
