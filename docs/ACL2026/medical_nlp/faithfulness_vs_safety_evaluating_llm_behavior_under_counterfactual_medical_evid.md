---
title: >-
  [Paper Note] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence
description: >-
  [ACL 2026][Medical NLP][Faithfulness-Safety Conflict] This paper constructs the MedCounterFact dataset—systematically replacing interventions in clinical trials with nonsense words, medical terms, non-medical items…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Faithfulness-Safety Conflict"
  - "Counterfactual Evidence"
  - "Medical QA"
  - "Safety Guardrails"
  - "RAG"
date: 2026-05-08
content_hash: 6b2f6071916647f3
---

# Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.11886](https://arxiv.org/abs/2601.11886)  
**Code**: [GitHub](https://github.com/KaijieMo-kj/Counterfactual-Medical-Evidence)  
**Area**: Medical Imaging  
**Keywords**: Faithfulness-Safety Conflict, Counterfactual Evidence, Medical QA, Safety Guardrails, RAG

## TL;DR

This paper constructs the MedCounterFact dataset—systematically replacing interventions in clinical trials with nonsense words, medical terms, non-medical items, and toxic substances. It finds that frontier LLMs exhibit almost unconditional adherence to context in the face of counterfactual medical evidence, answering confidently even when "evidence" suggests heroin or mustard gas is effective, revealing a serious lack of clear boundaries between faithfulness and safety.

## Background & Motivation

**Background**: RAG and evidence-based reasoning are considered critical means to reduce LLM hallucinations. Especially in high-risk domains like healthcare, evidence-based systems are perceived as more accurate. An increasing number of laypeople use LLMs as their primary source of health information.

**Limitations of Prior Work**: (1) Previous studies found that context can suppress a model's parametric knowledge, but these were primarily conducted in general domains; (2) In the medical field, evidence-based faithfulness is generally considered a positive trait—but what if the evidence itself is flawed? (3) Existing medical QA research assumes evidence is always valid and fails to investigate model behavior towards erroneous or adversarial evidence.

**Key Challenge**: A fundamental tension exists between faithfulness and safety—there is a desire for models to faithfully follow the provided context (faithfulness) while also expecting them to question and reject dangerous or absurd "evidence" (safety). Currently, there is no defined boundary between the two.

**Goal**: Systematically evaluate the behavior of LLMs when confronted with different levels of counterfactual medical evidence to reveal the current state of the faithfulness-safety trade-off.

**Key Insight**: Design four categories of progressive counterfactual interventions—ranging from areas where the model has zero prior knowledge (nonsense words) to substances that should trigger safety guardrails (toxic substances)—to systematically test the model's "questioning" capability.

**Core Idea**: Models should not only be faithful to the context but should also maintain skepticism towards untrustworthy evidence, similar to medical professionals—however, current models almost entirely lack this capability.

## Method

### Overall Architecture

Based on the MedEvidence dataset (284 clinical comparison questions + 329 RCTs), the MedCounterFact dataset (809 instances) was constructed via four types of counterfactual replacements. Nine frontier LLMs were evaluated across 4 prompt variants (No Evidence / With Evidence / Skeptical / Expert Role) × 2 response formats (Multiple Choice / Free-form).

### Key Designs

1.  **Four Categories of Counterfactual Intervention Stimuli**:
    *   **Function**: Tests model sensitivity to untrustworthy evidence across different dimensions.
    *   **Mechanism**: (a) NONCE—nonsense words (e.g., *blirbex*), where the model has no parametric knowledge; (b) MEDICAL—real but mismatched medical terms (e.g., replacing chemotherapy with penicillin); (c) NON-MEDICAL—non-medical items (e.g., bowling balls, SIM cards), where accepting efficacy violates common sense; (d) TOXIC—known toxic substances (e.g., heroin, mustard gas), with "toxic dosage" notes included to ensure safety warnings should be triggered.
    *   **Design Motivation**: These stimuli form a gradient from "ignorance" to "known danger"—if a model fails to question the TOXIC category, it indicates that faithfulness completely overrides safety.

2.  **Multi-dimensional Evaluation Framework**:
    *   **Function**: Captures different response patterns of models to counterfactual evidence.
    *   **Mechanism**: Two key metrics—(a) Uncertain Rate: the proportion of times the model selects the "uncertain/unsure" label (higher is better, indicating questioning); (b) Evidence Adherence (EA) Rate: the proportion of answers consistent with the original ground-truth label (in counterfactual conditions, a high EA rate means the model took the counterfactual evidence as truth).
    *   **Design Motivation**: High EA rate + Low Uncertain rate = The model accepts counterfactual evidence without any questioning.

3.  **Prompt Variant Design**:
    *   **Function**: Tests whether different prompting strategies can activate the model's questioning capability.
    *   *Mechanism**: (a) No-Evd—question only, testing parametric knowledge; (b) Evd—includes counterfactual evidence; (c) Skept+Evd—requires reasoning with a skeptical attitude; (d) Expert+Evd—assigns roles of clinical experts and Cochrane reviewers. Both multiple-choice and free-form formats were tested.
    *   **Design Motivation**: Identifying whether skepticism prompts or expert personas can increase the questioning rate provides a practical direction for mitigation.

### Loss & Training

A zero-shot evaluation approach was used with no training. Evaluated 9 LLMs: Gemini-2.5-flash, GPT-5-mini, Llama-3.1-8B/405B-Instruct, Llama-4-Maverick, OLMo-3-7B-Instruct/Think, Qwen2.5-7B-Instruct, and HuatuoGPT-o1-7B. Temperature was set to 0.

## Key Experimental Results

### Main Results

| Condition | Change in Uncertain Rate | Change in EA Rate |
| :--- | :--- | :--- |
| No-Evd → Evd (Original) | Significant decrease | Significant increase |
| No-Evd → Evd (NONCE) | Significant decrease | Comparable to original |
| No-Evd → Evd (TOXIC) | Significant decrease | Comparable to original |
| Skept+Evd vs Evd | Uncertain rate increases | EA rate decreases but remains insufficient |
| Expert+Evd vs Evd | No significant improvement | No significant improvement |

### Ablation Study

| Analysis Dimension | Result |
| :--- | :--- |
| No-Evidence Condition | Models occasionally judge counterfactual interventions as unreasonable (higher Uncertain rate) |
| With-Evidence Condition | Counterfactual evidence completely suppresses parametric knowledge and safety awareness |
| TOXIC vs NONCE Difference | Virtually no difference—the model adheres to both equally |
| Free-form vs Multi-choice | Free-form yields a lower Uncertain rate—models are less inclined to express uncertainty without explicit options |
| Representation Analysis | Counterfactual evidence causes a distribution shift; parametric knowledge is briefly activated then quickly overridden by context |

### Key Findings

*   Across all counterfactual stimuli categories, models neither question the premise nor refuse to answer—even with built-in safety guardrails.
*   "Awareness" of irrationality occasionally appears in the chain-of-thought, but these doubts are quickly suppressed to align with the provided evidence.
*   Skeptical prompting (Skept+Evd) is the only strategy that provides slight mitigation, yet it remains far from sufficient for the TOXIC category.
*   Model behavior is essentially identical for NONCE (ignorance) and TOXIC (known danger) evidence—this is the most concerning finding.
*   Representation analysis shows that parametric knowledge is briefly activated upon encountering counterfactual nouns but is overridden as context accumulates.

## Highlights & Insights

*   The lack of a "faithfulness-safety boundary" is a profound and urgent issue—current LLMs are essentially "unconditional believers of evidence" in medical scenarios.
*   The gradient design of the four counterfactual stimuli—progressing from control (NONCE) to extreme (TOXIC) conditions—makes the conclusions highly persuasive.
*   The pattern of "brief doubt followed by rapid compliance" in reasoning chains reveals that context bias is deeper than safety alignment in LLMs.
*   The study serves as a warning for RAG systems: if retrieved evidence is tampered with or incorrect, the model will confidently provide dangerous advice.

## Limitations & Future Work

*   Counterfactual evidence was generated via simple replacement and does not cover more subtle errors (e.g., dosage or indication errors).
*   Evaluation is limited to English and specific medical sub-domains.
*   No effective mitigation solution was proposed—only the problem was diagnosed.
*   Determining what a model's "proper" faithfulness-safety boundary should be remains an unsolved normative question.

## Related Work & Insights

*   **vs CoPriva/Doc-PP**: While the latter focuses on non-disclosure strategies, this paper focuses on over-trust when the model "should" be skeptical.
*   **vs Xie et al. (2023)**: While the latter studies context-knowledge conflicts in general domains, this work focuses on high-risk medical scenarios.
*   **vs MedEvidence**: This work builds upon it, extending to counterfactual settings to test model robustness.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic study of faithfulness-safety tension in medical scenarios; ingenious stimuli design.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 9 models, 4 prompt types, 2 formats, and representation analysis.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, alarming findings, and powerful argumentation.
*   Value: ⭐⭐⭐⭐⭐ Highly significant for medical AI safety, directly impacting deployment decisions for RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[ACL 2026\] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)

</div>

<!-- RELATED:END -->
