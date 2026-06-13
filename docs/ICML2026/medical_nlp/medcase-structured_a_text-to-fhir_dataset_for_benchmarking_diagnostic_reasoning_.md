---
title: >-
  [Paper Note] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings
description: >-
  [ICML2026][Medical NLP][FHIR] The authors propose a "staged LLM + terminology grounding + repair loop" pipeline to convert free-text medical cases into HL7 FHIR R4 standard-compliant patient bundles. Based on this…
tags:
  - "ICML2026"
  - "Medical NLP"
  - "FHIR"
  - "Clinical Decision Support"
  - "Terminology Grounding"
  - "Synthetic EHR"
  - "Diagnostic Reasoning"
date: 2026-05-08
content_hash: ba8461750e824e1d
---

# MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings

**Conference**: ICML2026  
**arXiv**: [2605.30295](https://arxiv.org/abs/2605.30295)  
**Code**: https://github.com/SystemInternal/MedCase-Structured (Available)  
**Area**: Medical NLP  
**Keywords**: FHIR, Clinical Decision Support, Terminology Grounding, Synthetic EHR, Diagnostic Reasoning

## TL;DR
The authors propose a "staged LLM + terminology grounding + repair loop" pipeline to convert free-text medical cases into HL7 FHIR R4 standard-compliant patient bundles. Based on this, they construct the MedCase-Structured dataset from MedCaseReasoning, consisting of 1,408 structured synthetic cases (82.5% success rate). Experiments show that the diagnostic accuracy of GPT-5.4 / Gemini-3.1-Pro / Claude-Opus-4.6 consistently drops by 4–23 points on structured FHIR inputs compared to pure text inputs.

## Background & Motivation

**Background**: LLM-based Clinical Decision Support Systems (CDSS) are increasingly discussed. Standard evaluations typically use pure text QA like MedQA or restricted real EHRs like MIMIC-IV. Modern hospital systems generally exchange patient data between modules using HL7 FHIR resource objects.

**Limitations of Prior Work**: Existing benchmarks do not match real deployment forms: (1) Pure text cases cannot test model robustness on structured, interoperable formats; (2) MIMIC-IV-FHIR is an offline "reverse mapping" with a single distribution and privacy restrictions; (3) Synthea is based on rule templates with limited clinical diversity and insufficient pressure testing; (4) Text→FHIR methods like FHIR-GPT/Infherno aim for "faithful reconstruction" of existing cases rather than generating controllable, high-volume evaluation samples.

**Key Challenge**: To perform "deployment-aligned" CDSS evaluation, one must have large-scale, structured, controllable, and privacy-free synthetic FHIR cases. However, directly letting LLMs write FHIR results in significant **hallucinated medical codes** (LOINC / RxNorm / SNOMED made up when unknown) and **structurally non-compliant** resource objects, rendering the quality unusable.

**Goal**: To split the problem into two tasks: (a) Build a pipeline to controllably generate clinically realistic FHIR R4 bundles from free text, suppressing hallucinated codes and structural errors; (b) Use this pipeline to convert MedCaseReasoning into a public dataset, MedCase-Structured, and compare the diagnostic accuracy difference of LLMs between "pure text cases" and "structured FHIR cases."

**Key Insight**: It is observed that free-generation failure modes for FHIR cluster into two types: "hallucinated/non-standard terminology codes" and "structural/semantic inconsistency between resources." The former can be grounded by a **deterministic terminology library + embedding retrieval**, while the latter can be constrained by a **multi-stage split + validation-repair loop**.

**Core Idea**: Decompose text→FHIR into four stages: "Information Extraction → Terminology Grounding → FHIR Synthesis and Validation → Diagnosis Hiding." LLMs act at three fixed anchors (Extraction, Synthesis, Semantic Leakage Scanning). Deterministic grounding using a SapBERT+FAISS terminology library performs acceptance/replacement/rejection based on three thresholds. During the synthesis stage, a "validation failure → LLM rewrite" repair loop of up to 3 rounds is introduced.

## Method

### Overall Architecture
Input is an English free-text medical case (from MedCaseReasoning); output is a validated HL7 FHIR R4 patient bundle, with diagnostic conclusions optionally removed according to "diagnosis hiding mode" for downstream evaluation. The pipeline contains 4 serial stages:

1.  **Extraction**: LLM #1 extracts free text into a flat intermediate structure (demographics, symptoms, physical exams, vitals, labs, medications, procedures, history), retaining the **verbatim citation** for each entry for traceability.
2.  **Terminology Grounding**: An internally maintained terminology library (aggregating OMOP + SNOMED CT / LOINC / RxNorm / CVX) grounds all LLM-extracted codes. It performs keyword retrieval followed by semantic similarity matching using SapBERT embeddings in a FAISS index, with "Accept / Replace / Reject" based on three cosine thresholds.
3.  **FHIR Synthesis and Validation**: LLM #2 assembles the grounded intermediate representation into FHIR resources (Patient / Encounter / Condition / Observation / MedicationRequest / Procedure / DiagnosticReport / FamilyMemberHistory / AllergyIntolerance / Immunization) according to HL7 R4 templates. Failed validation items are fed back to LLM for rewriting (up to 3 rounds); post-processing uses rules to complete missing resources and normalize units/dates/status fields.
4.  **Diagnosis Hiding**: Configurable hiding of diagnostic conclusions based on NONE / HIDDEN / EXPLICIT / FULL modes. In NONE/HIDDEN modes, hard filtering by code + substring is performed first, followed by a semantic scan by LLM #3 on all narrative fields to redact abbreviations, suggestive conclusions, and synonyms.

All LLM calls use Claude (claude-sonnet-4-20250514) with temperature 0 for reproducibility.

### Key Designs

1.  **Terminology Grounding via SapBERT + FAISS + Three-Threshold Decision**:
    *   **Function**: Hard-binds SNOMED CT / LOINC / RxNorm / CVX codes extracted by LLM to real codes in the terminology library, rejecting hallucinated codes on the spot.
    *   **Mechanism**: Retrieves candidates by keyword, then uses SapBERT to index embeddings of candidates and all preferred terms in the library into FAISS. Three cosine similarity thresholds determine "Original code acceptable / Replace with nearest neighbor / Reject and pass to repair loop"—transforming "term legality" from a probabilistic matching problem into a retrieval matching problem with adjustable thresholds.
    *   **Design Motivation**: The authors report in Table 2 that major failures involve LOINC hallucinations (183 cases) + RxNorm hallucinations (126 cases) + coarse drug granularity (103 cases), which grounding suppresses. SapBERT is a strong baseline for biomedical entity embedding, and FAISS maintains retrieval latency.

2.  **Three Fixed Anchors + Validation-Repair Closed Loop (vs agent-style)**:
    *   **Function**: Makes the text→FHIR process controllable and debuggable, explicitly injecting "FHIR validator errors" back into LLMs for repair.
    *   **Mechanism**: Unlike agent-styles where LLMs dynamically decide when to use tools, the pipeline fixes LLM calls at three points (extraction, synthesis, semantic leakage scan), interspersed with deterministic grounding, structure/clinical consistency validation, and rule post-processing. If validation fails during synthesis, an error list is returned to LLM for rewriting (up to 3 rounds).
    *   **Design Motivation**: Fixed anchors are more debuggable and reproducible (temperature 0 + fixed order) than agents. It shifts deterministic constraints (syntax/terminology compliance) from LLM back to rules and libraries, letting LLM focus on "natural language/structural mapping."

3.  **Configurable Diagnosis Hiding + LLM Semantic Leakage Scanning**:
    *   **Function**: Prevents models from "reading diagnosis answers" from bundles during evaluation while retaining legitimate context like patient self-reports or family history.
    *   **Mechanism**: Filters the assembled bundle via NONE / HIDDEN / EXPLICIT / FULL modes. For NONE/HIDDEN, hard filtering (code + substring) is followed by LLM #3 semantic scanning on all narrative fields to catch implicit leaks like abbreviations, suggestive conclusions, or synonyms not in the list.
    *   **Design Motivation**: CDSS evaluation credibility depends on input not hiding answers. Narrative fields in FHIR bundles are prone to residual diagnosis info; rule-based redaction is insufficient and requires LLM-based semantic scanning.

### Loss & Training
No models are trained in this work. It uses off-the-shelf Claude APIs, SapBERT embeddings, and FAISS indices. Temperature is fixed at 0. No fine-tuning or RLHF is performed; the system is a synthetic data pipeline combining LLM, retrieval, and rule validation.

## Key Experimental Results

### Main Results

Dataset construction results—14,489 cases from MedCaseReasoning were filtered by rules (removing non-human, multi-patient, and imaging-dependent samples) then processed through the pipeline:

| Dataset Split | Original Total | Imaging Exclusion | Code Error Exclusion | Other Exclusion | Final |
|---|---|---|---|---|---|
| Train | 13,092 | 11,568 | 232 | 28 | 1,263 |
| Val | 500 | 438 | 10 | 2 | 50 |
| Test | 897 | 777 | 14 | 11 | 95 |
| Total | 14,489 | 12,783 | 256 | 41 | 1,408 |

The success rate of generating FHIR bundles for samples entering the pipeline was **82.5%**.

LLM diagnostic accuracy comparison—the same cases were provided as "Pure Text (MCR)" and "Structured FHIR (MCS)" to the same models, comparing zero / 1-shot / 5-shot settings:

| Model | Setting | MCR (%) | MCS (%) | Δ |
|---|---|---|---|---|
| GPT-5.4 | zero-shot | 65.26 | 61.05 | −4.21 |
| GPT-5.4 | 1-shot | 74.74 | 51.58 | −23.16 |
| GPT-5.4 | 5-shot | 74.74 | 53.68 | −21.06 |
| Gemini-3.1-Pro | zero-shot | 58.95 | 52.63 | −6.32 |
| Gemini-3.1-Pro | 1-shot | 65.26 | 53.68 | −11.58 |
| Gemini-3.1-Pro | 5-shot | 63.16 | 57.89 | −5.28 |
| Claude-Opus-4.6 | zero-shot | 68.42 | 53.63 | −14.79 |
| Claude-Opus-4.6 | 1-shot | 69.47 | 54.74 | −14.73 |
| Claude-Opus-4.6 | 5-shot | 66.32 | 58.95 | −7.37 |

Across all settings, the "structured FHIR input" results were significantly worse than pure text, with the largest drop (GPT-5.4 + 1-shot) reaching 23 points.

### Ablation Study

While traditional ablation was not performed, the paper reports fine-grained failure modes on MedCaseReasoning (indicative of pipeline bottlenecks):

| Category | Failure Type | Count | Example |
|---|---|---|---|
| Terminology Error | LOINC Hallucination | 183 | "septic workup", "pharmacological challenge test" |
| Terminology Error | RxNorm Hallucination | 126 | Hallucinated invalid code after repair |
| Terminology Error | Coarse Drug Granularity | 103 | "oral antibiotics", "topical corticosteroid paste" |
| Terminology Error | CVX Synonym Gap | 12 | "Moderna booster", "fully immunized" |
| Semantic Mapping | Detailed Description | 32 | "loosening of lower teeth requiring dental implants" |
| Semantic Mapping | SNOMED Mismatch | 33 | Procedure code assigned to clinical finding |
| Exclusion | Missing Demographics | 4 | No age in text |
| Exclusion | Multi-patient | 9 | Multiple patients per case |
| Exclusion | Non-human | 25 | Veterinary records |

### Key Findings
*   Even with a 3-round validation-repair loop, **terminology hallucination (LOINC + RxNorm + Coarse Granularity) remains the largest bottleneck** (>410 code-level errors), far exceeding structural/semantic errors. This indicates that SapBERT+FAISS grounding effectively prevents "made-up codes" but cannot solve descriptions that are too broad to match specific codes.
*   The performance gap between structured FHIR and pure text **expands in few-shot settings**: Whereas GPT-5.4's MCR performance rises to 74.74% with 1/5-shot, MCS lingers at 51–53%, suggesting examples cannot compensate for the model's unfamiliarity with FHIR resources.
*   Imaging-related cases account for a massive portion of MedCaseReasoning (11,568 out of 13,092 filtered, ~88%) because the current pipeline does not model ImagingStudy resources; this represents the largest area for future work.

## Highlights & Insights
*   **The "LLM anchor + retrieval library + validation loop" is a standard paradigm** for structured generation outside natural language. Passing deterministic constraints (valid codes/structures) back to rules/retrieval while letting LLM handle narrative-to-structure mapping is a template applicable to legal contracts, tax forms, and API schema generation.
*   **The two-stage "hard filter + LLM semantic scan" for diagnosis hiding** is an underrated design. FHIR narrative fields often leak answers via abbreviations or synonyms that rule-based systems miss. This two-stage approach can be ported to other synthetic data scenarios (Law/Finance/Exams).
*   **Aligning evaluation distribution with deployment format yields high empirical ROI**: Simply changing the representation caused GPT-5.4 to drop 20+ points, implying that performance on MedQA / MedCaseReasoning does not reflect real-world utility in EHR systems—providing hard evidence for future clinical LLM evaluation design.

## Limitations & Future Work
*   The pipeline only covers 10 FHIR resources; critical resources like ImagingStudy / Specimen / Service Request / Goal are missing, resulting in the exclusion of 88% of imaging-related cases.
*   The system does not model true longitudinal trajectories (time-series of multiple encounters), using only "date-aware duplicate resources," which limits evaluation of temporal reasoning.
*   Terminology grounding remains a bottleneck: Coarse descriptions ("oral antibiotics") and English synonym gaps ("Moderna booster") cannot be resolved by SapBERT+FAISS alone, requiring stronger context-aware validation or broader terminology expansion.
*   Evaluations were limited to three closed-source models on MedCaseReasoning; no direct comparison with MIMIC-IV-FHIR or Synthea was performed to decouple the "FHIR difficulty" from the "synthetic bundle noise."
*   There is potential for self-assessment bias, as Claude-Sonnet-4 was used for synthesis, masking, and as one of the subjects; future work could use cross-family LLMs for synthesis and evaluation.

## Related Work & Insights
*   **vs Synthea**: Synthea uses rule-based templates which give broad clinical coverage but limited diversity and no text-to-FHIR mapping. Ours is text-driven, allowing for controllable clinical complexity. The two are complementary: Synthea for large-scale training, and Ours for targeted diagnostic evaluation.
*   **vs FHIR-GPT / Infherno**: These projects treat text→FHIR as a "faithful reconstruction" task for system integration; this work treats it as an "eval-sample production pipeline," emphasizing controllability, masking, and eval-alignment.
*   **vs FHIR-AgentBench / EHRStruct**: These benchmarks provide static structured EHR/FHIR evaluations. MedCase-Structured upgrades the "dataset" to a "sample factory" that can be driven by text to allow controllable perturbations of complexity and ambiguity.
*   **Inspirations**: (a) Incorporating ImagingStudy resources could expand the dataset tenfold; (b) The expanding gap in few-shot settings is an in-context learning problem worthy of separate study; (c) This synthesis/masking pipeline can be adapted to other standards like OMOP-CDM or openEHR.

## Rating
*   Novelty: ⭐⭐⭐⭐ Controllable evaluation FHIR generation from text is a new positioning; it solves a real problem by integrating existing components effectively.
*   Experimental Thoroughness: ⭐⭐⭐ Detailed statistics and failure modes, but lacks parallel comparisons with Synthea/MIMIC-IV-FHIR and lacks ablation studies.
*   Writing Quality: ⭐⭐⭐⭐ Logical flow from motivation to failure modes and evaluation is smooth. Tables 1/2/3 are highly informative.
*   Value: ⭐⭐⭐⭐ 1,408 cases + open repository + controllable masking provide an immediately usable utility for clinical LLM evaluation, providing empirical proof for deployment-aligned testing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](../../ACL2026/medical_nlp/dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](../../ACL2026/medical_nlp/reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](../../ICLR2026/medical_nlp/from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)
- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](../../ICLR2026/medical_nlp/biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](../../ACL2026/medical_nlp/multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)

</div>

<!-- RELATED:END -->
