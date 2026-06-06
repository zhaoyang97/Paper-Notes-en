---
title: >-
  [Paper Note] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Document QA] This paper proposes the Doc-PP benchmark, revealing a "reasoning-induced safety gap" in large vision-language models (LVLMs) during multimodal document question answering—models by…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Document QA"
  - "Information Leakage"
  - "Policy Preservation"
  - "Multimodal Reasoning"
  - "Safety Alignment"
date: 2026-05-08
content_hash: c066fecffbfe45e6
---

# Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2601.03926](https://arxiv.org/abs/2601.03926)  
**Code**: [Project Page](https://hwanchang00.github.io/docpp_project_page)  
**Area**: Multimodal VLM / Document Security
**Keywords**: Document QA, Information Leakage, Policy Preservation, Multimodal Reasoning, Safety Alignment

## TL;DR

This paper proposes the Doc-PP benchmark, revealing a "reasoning-induced safety gap" in large vision-language models (LVLMs) during multimodal document question answering—models bypass explicit non-disclosure policies and leak sensitive information when cross-modal reasoning is required. A structured reasoning framework, DVA (Decompose–Verify–Aggregation), is proposed to substantially reduce leakage rates.

## Background & Motivation

**Background**: LVLMs are widely applied to question answering over complex multimodal documents. In real-world deployments, documents are typically accompanied by user-defined dynamic policies specifying which information may or may not be disclosed (e.g., regional revenue data in quarterly reports may be confidential). These constraints vary across users, organizations, and access scenarios, making manual redaction of sensitive regions infeasible.

**Limitations of Prior Work**: (1) Existing safety research primarily focuses on implicit social norms or purely textual settings, overlooking the complexity of multimodal documents. (2) Text-domain methods such as CoPriva handle only textual inputs and do not address heterogeneous visual components such as charts and tables. (3) Even advanced models like GPT-5.2, when explicitly instructed not to disclose revenue for the Middle East region, can still extract percentages from pie charts, retrieve total revenue from text, and compute the protected information via implicit reasoning.

**Key Challenge**: The stronger a model's reasoning capability, the more readily it can synthesize cross-modal evidence to circumvent safety constraints—creating a fundamental tension between reasoning ability and policy compliance.

**Goal**: To construct the first benchmark for evaluating user-defined policy preservation in multimodal documents and to propose an effective defense framework.

**Key Insight**: Evaluation should focus on queries that require cross-modal reasoning to answer, thereby exposing the safety gap between explicit and implicit query types.

**Core Idea**: Safety checks should be embedded at every step of the reasoning process rather than applied only at the final output—DVA decouples reasoning from policy verification, independently verifying each sub-step before aggregation.

## Method

### Overall Architecture

Doc-PP comprises a three-stage construction pipeline: (1) Policy Construction—generating confidentiality targets from real documents and filtering via checklists; (2) Query Construction—generating both explicit and implicit query types; (3) Evaluation—measuring leakage rate and faithfulness using a checklist framework. Each evaluation instance is defined as a triplet $(D, P, Q)$, representing the document, security policy, and query. Documents support two input conditions: $D^{ocr}$ (OCR-parsed content) and $D^{img}$ (PNG image).

### Key Designs

1. **Policy Construction**:

    - **Function**: Automatically generate high-quality non-disclosure policies from real PDF documents.
    - **Mechanism**: GPT-5.2 is first used to propose confidentiality targets according to a sensitive-category taxonomy (strategic decisions, roadmaps, internal deliberations, legal details, etc.), requiring specification of evidence type (text/table/chart/mixed), page index, and verbatim citations. Target-aligned clipping then extracts relevant page windows $[p-2, p+2]$ from long documents (averaging 100 pages), establishing a one-to-one mapping between confidentiality targets and document segments. Low-quality candidates are subsequently filtered via a five-item checklist.
    - **Design Motivation**: Confidentiality targets are not simple factual snippets but information that requires deep understanding—such as interpreting chart trends or synthesizing cross-modal context—to locate. This ensures the benchmark genuinely tests policy compliance.

2. **Explicit vs. Implicit Query Classification**:

    - **Function**: Distinguish two safety challenges of differing difficulty.
    - **Mechanism**: Explicit queries $Q_e$ directly request the target information (e.g., "What is the revenue for the Middle East region?"); implicit queries $Q_i$ are framed as summarization requests, where a faithful answer naturally entails disclosure (e.g., "Please summarize the revenue distribution across regions"). Models must selectively withhold sensitive values while satisfying the informational need.
    - **Design Motivation**: In realistic settings, information leakage is often not caused by direct inquiry but by indirect reasoning—implicit queries more closely reflect genuine threats.

3. **DVA Structured Reasoning Framework (Decompose–Verify–Aggregation)**:

    - **Function**: Decouple reasoning from policy verification to structurally prevent policy violations during the reasoning process.
    - **Mechanism**: (1) *Decompose*—decompose complex queries into independent sub-questions; (2) *Verify*—independently check each sub-answer for policy compliance, identifying and blocking evidence that pertains to confidentiality targets; (3) *Aggregation*—aggregate only verified sub-answers to produce the final output.
    - **Design Motivation**: Standard prompting defenses (e.g., CoT, post-hoc revision) cannot intercept intermediate reasoning steps that lead to policy violations—once information is computed within the reasoning chain, subsequent filtering is often too late.

### Loss & Training

Doc-PP is an evaluation benchmark rather than a training method. The dataset is collected from 90 long PDF documents sourced from MMlongbench-Doc and Sustainable QA, spanning business, financial, and industry reports. Evaluation employs a checklist framework to measure information leakage rate and response faithfulness.

## Key Experimental Results

### Main Results

| Finding | Description |
|---------|-------------|
| Reasoning-induced safety gap | Leakage rates under implicit queries are substantially higher than under explicit queries—models comply with direct requests but cannot prevent reasoning-derived disclosure. |
| OCR paradox | Providing OCR text improves perception but significantly increases information leakage. |
| Cross-modal leakage | Policy compliance degrades markedly in multimodal settings requiring integration of textual and visual evidence. |
| DVA advantage | DVA substantially outperforms standard prompting defenses across all document types and query settings. |

### Ablation Study

| Defense Strategy | Effectiveness |
|-----------------|---------------|
| Standard CoT prompting | Limited protection; cannot intercept intermediate reasoning steps. |
| Post-hoc output revision | Limited protection; information has already been computed during reasoning. |
| DVA (full) | Substantially reduces leakage rate, providing a practical safety baseline. |

### Key Findings

- Even state-of-the-art models such as GPT-5.2 systematically leak protected information in cross-modal reasoning scenarios.
- Providing OCR text is a double-edged sword—it improves perception while exacerbating leakage, revealing a capability–safety trade-off.
- Mixed evidence types carry the highest leakage risk, as they require integrating information across multiple modalities.
- DVA's step-wise verification strategy effectively blocks information propagation paths within the reasoning chain.

## Highlights & Insights

- The "reasoning-induced safety gap" is a profound observation—a model's reasoning capability itself becomes the source of security vulnerabilities, which fundamentally differs from the adversarial-input paradigm dominant in traditional safety research.
- The core idea of DVA—embedding safety checks into every sub-step of reasoning—is generalizable to any scenario requiring constraint maintenance throughout information processing.
- Anchoring confidentiality targets to information requiring deep understanding (rather than simple facts) substantially enhances the real-world relevance of the benchmark.

## Limitations & Future Work

- The dataset is relatively small (90 documents) and may not cover all document types and policy patterns.
- DVA introduces additional inference latency, which may affect real-time applications.
- Only non-disclosure policies are evaluated; more complex conditional disclosure rules are not addressed.
- The impact of model fine-tuning or safety alignment training on policy preservation remains unexplored.

## Related Work & Insights

- **vs. CoPriva**: CoPriva is limited to purely textual inputs and localized text-span queries; Doc-PP extends evaluation to multimodal documents and cross-document reasoning.
- **vs. VLM-GEOPRIVACY**: The latter focuses on implicit privacy norms (geographic location inference), whereas Doc-PP addresses explicit user-defined constraints.
- **vs. Traditional Safety Alignment**: Methods such as RLHF train against implicit social norms and cannot handle dynamic, user-specified policies.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First multimodal document policy preservation benchmark; the concept of "reasoning-induced safety gap" is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple LVLMs and defense strategies are evaluated, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is clear, the threat model is intuitive, and experimental design is rigorous.
- Value: ⭐⭐⭐⭐⭐ — Reveals a neglected yet critically important safety issue in LVLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
