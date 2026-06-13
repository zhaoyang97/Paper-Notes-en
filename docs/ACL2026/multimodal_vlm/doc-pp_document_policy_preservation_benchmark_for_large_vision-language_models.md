---
title: >-
  [Paper Note] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Document Visual Question Answering] This paper proposes the Doc-PP benchmark, revealing a "reasoning-induced safety gap" in Large Vision-Language Models (LVLMs) during multimodal document QA—mo…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Document Visual Question Answering"
  - "Information Leakage"
  - "Policy Preservation"
  - "Multimodal Reasoning"
  - "Safety Alignment"
date: 2026-05-08
content_hash: 7bc639b0da1b101f
---

# Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.03926](https://arxiv.org/abs/2601.03926)  
**Code**: [Project Page](https://hwanchang00.github.io/docpp_project_page)  
**Area**: Multimodal VLM / Document Security  
**Keywords**: Document Visual Question Answering, Information Leakage, Policy Preservation, Multimodal Reasoning, Safety Alignment

## TL;DR

This paper proposes the Doc-PP benchmark, revealing a "reasoning-induced safety gap" in Large Vision-Language Models (LVLMs) during multimodal document QA—models bypass explicit non-disclosure policies to leak sensitive information when cross-modal reasoning is required. The authors further propose the DVA (Decompose–Verify–Aggregation) structured reasoning framework to significantly reduce leakage rates.

## Background & Motivation

**Background**: LVLMs are widely utilized for QA tasks involving complex multimodal documents. In real-world deployments, documents often come with user-defined dynamic policies specifying which information can or cannot be disclosed (e.g., specific regional revenue data in quarterly reports must remain confidential). These constraints vary by user, organization, and access scenario, making manual masking of sensitive areas infeasible.

**Limitations of Prior Work**: (1) Existing safety research primarily focuses on implicit social norms or pure text scenarios, overlooking the complexity of multimodal documents; (2) Text-domain works like CoPriva only handle text inputs and do not address heterogeneous visual components such as charts and tables; (3) Even advanced models like GPT-5.2, when explicitly instructed "do not disclose revenue for the Middle East," still extract percentages from pie charts and total revenue from text to calculate the protected information via implicit reasoning.

**Key Challenge**: The stronger a model's reasoning capability, the easier it is to bypass safety constraints through cross-modal evidence synthesis—there is a fundamental tension between reasoning ability and policy compliance.

**Goal**: To build the first benchmark for evaluating user-defined policy preservation in multimodal documents and to propose an effective defense framework.

**Key Insight**: The evaluation focuses on queries that require cross-modal reasoning to answer, revealing the safety gap between explicit and implicit queries.

**Core Idea**: Safety checks should be embedded in every step of the reasoning process rather than being filtered only at the final output—DVA decouples reasoning from policy verification, where each sub-step is independently verified before aggregation.

## Method

### Overall Architecture

Doc-PP consists of a three-stage construction pipeline: (1) Policy Construction—generating non-disclosure targets from real documents and filtering via checklists; (2) Query Construction—generating explicit and implicit queries; (3) Evaluation—measuring leakage rates and faithfulness using a checklist framework. An evaluation instance is defined as a triplet $(D, P, Q)$, representing the document, safety policy, and query. The document supports two input conditions: $D^{ocr}$ (OCR-parsed content) and $D^{img}$ (PNG images).

### Key Designs

1.  **Policy Construction**:
    *   **Function**: Automatically generate high-quality non-disclosure policies from real PDF documents.
    *   **Mechanism**: First, GPT-5.2 proposes non-disclosure targets based on a sensitive category taxonomy (strategic decisions, roadmaps, internal debates, legal details, etc.), requiring evidence types (text/table/chart/mixed), page indices, and original quotes. Then, target-aligned clipping is used to extract a relevant page window $[p-2, p+2]$ from long documents (averaging 100 pages), establishing a one-to-one mapping between targets and document segments. Finally, low-quality candidates are filtered through a five-point checklist.
    *   **Design Motivation**: Non-disclosure targets are not simple factual snippets but information requiring deep understanding (e.g., interpreting chart trends, synthesizing cross-modal context) to locate, thereby truly testing the model's policy compliance.

2.  **Explicit vs. Implicit Query**:
    *   **Function**: Distinguish between two safety challenges of varying difficulty.
    *   **Mechanism**: Explicit queries $Q_e$ directly request target information (e.g., "What is the revenue for the Middle East?"); implicit queries $Q_i$ are presented as summary requests where a faithful answer would naturally involve disclosure (e.g., "Please summarize the revenue distribution across regions"). The model must satisfy the information need while selectively withholding sensitive values.
    *   **Design Motivation**: In real-world scenarios, information leakage often results from indirect reasoning rather than direct questioning—implicit queries reflect real-world threats more accurately.

3.  **DVA (Decompose–Verify–Aggregation) Structured Reasoning Framework**:
    *   **Function**: Decouple reasoning from policy verification to structurally prevent policy violations during the reasoning process.
    *   **Mechanism**: (1) Decompose—break down complex queries into independent sub-questions; (2) Verify—independently perform policy compliance checks on each sub-answer to identify and block evidence involving non-disclosure targets; (3) Aggregation—generate the final output by aggregating only the sub-answers that passed verification.
    *   **Design Motivation**: Standard prompting defenses (e.g., CoT, post-hoc revision) fail to intercept intermediate reasoning steps that lead to policy violations—once information is computed within the reasoning chain, subsequent filtering is often too late.

### Loss & Training

Doc-PP is an evaluation benchmark rather than a training method. The dataset comprises 90 long PDF documents collected from MMlongbench-Doc and Sustainable QA, covering business, finance, and industry reports. Evaluation utilizes a checklist framework to measure information leakage rates and answer faithfulness.

## Key Experimental Results

### Main Results

| Finding | Description |
| :--- | :--- |
| Reasoning-Induced Safety Gap | Leakage rates for implicit queries are significantly higher than for explicit queries—models comply with direct requests but fail to block reasoning-based derivations. |
| OCR Paradox | Providing OCR text enhances perception but significantly increases information leakage. |
| Cross-modal Leakage | Policy compliance drops significantly in multimodal settings requiring the integration of text and visual evidence. |
| DVA Advantage | DVA substantially outperforms standard prompting defenses across all document types and query settings. |

### Ablation Study

| Defense Strategy | Effect |
| :--- | :--- |
| Standard CoT prompting | Limited protection; fails to intercept intermediate reasoning steps. |
| Post-hoc output revision | Limited protection; information is already computed during reasoning. |
| DVA (Full) | Significantly reduces leakage rates, providing a practical safety baseline. |

### Key Findings

*   Even state-of-the-art models like GPT-5.2 systematically leak protected information in cross-modal reasoning scenarios.
*   Providing OCR text is a double-edged sword—it improves perception but exacerbates leakage, revealing a "capability-safety" tradeoff.
*   Mixed evidence types (mixed) pose the highest leakage risk as they require integrating information across multiple modalities.
*   The step-by-step verification strategy of DVA effectively blocks information propagation paths within the reasoning chain.

## Highlights & Insights

*   The "reasoning-induced safety gap" is a profound observation—the model's reasoning capability itself becomes a source of safety vulnerability, which differs sharply from the "adversarial input" paradigm in traditional safety research.
*   The core idea of DVA—embedding safety checks into every sub-step of reasoning—can be generalized to any scenario requiring constraint maintenance during information processing.
*   The dataset design anchors non-disclosure targets to information requiring deep understanding (rather than simple facts), significantly enhancing the real-world relevance of the benchmark.

## Limitations & Future Work

*   The dataset size is relatively small (90 documents), which may not cover all document types and policy patterns.
*   DVA increases reasoning latency, which may impact real-time applications.
*   Only non-disclosure policies were evaluated; more complex conditional disclosure rules were not addressed.
*   The impact of model fine-tuning or safety alignment training on policy preservation was not explored.

## Related Work & Insights

*   **vs. CoPriva**: CoPriva is limited to pure text inputs and local text segment queries; Doc-PP extends to multimodal documents and cross-document reasoning.
*   **vs. VLM-GEOPRIVACY**: The latter focuses on implicit privacy norms (geolocation inference), while Doc-PP focuses on explicit user-defined constraints.
*   **vs. Traditional Safety Alignment**: Methods like RLHF are trained for implicit social norms and cannot handle dynamic, user-specified policies.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First multimodal document policy preservation benchmark; the "reasoning-induced safety gap" concept is highly novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated multiple LVLMs and various defense strategies, though the dataset scale is limited.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, intuitive threat model, and rigorous experimental design.
*   Value: ⭐⭐⭐⭐⭐ Reveals a neglected yet critical safety issue in the deployment of LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
