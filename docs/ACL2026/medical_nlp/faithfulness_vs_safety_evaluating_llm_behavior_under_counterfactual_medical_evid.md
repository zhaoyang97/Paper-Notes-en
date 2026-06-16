---
title: >-
  [Paper Note] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence
description: >-
  [ACL 2026][Medical NLP][RAG] This paper constructs the MedCounterFact dataset by systematically replacing interventions in clinical trials with nonsense words, medical terms, non-medical items, and toxic substances. It finds that frontier LLMs exhibit nearly unconditional compliance with the provided context when faced with counterfactual medical
tags:
  - ACL 2026
  - Medical NLP
  - RAG
date: 2026-05-08
content_hash: 07d74d6e9f09ea7f
---
# Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.11886](https://arxiv.org/abs/2601.11886)  
**Code**: [GitHub](https://github.com/KaijieMo-kj/Counterfactual-Medical-Evidence)  
**Area**: Medical NLP  
**Keywords**: Faithfulness-safety conflict, counterfactual evidence, medical QA, safety guardrails, RAG

## TL;DR

This paper constructs the MedCounterFact dataset by systematically replacing interventions in clinical trials with nonsense words, medical terms, non-medical items, and toxic substances. It finds that frontier LLMs exhibit nearly unconditional compliance with the provided context when faced with counterfactual medical evidence, confidently answering questions even when "evidence" suggests heroin or mustard gas is effective, revealing a severe lack of a clear boundary between faithfulness and safety.

## Background & Motivation

**Background**: RAG and evidence-based reasoning are considered key methods for reducing LLM hallucinations, particularly in high-risk areas like medicine where evidence-based systems are deemed more accurate. An increasing number of laypeople use LLMs as their primary information source for health issues.

**Limitations of Prior Work**: (1) Previous studies found context can suppress the parametric knowledge of LLMs, but research has primarily focused on general domains; (2) In medicine, evidence-based faithfulness is considered positive—but what if the evidence itself is flawed? (3) Existing medical QA tasks assume evidence is always valid and have not studied model behavior toward erroneous or adversarial evidence.

**Key Challenge**: There is a fundamental tension between faithfulness and safety—models are expected to faithfully follow provided context (faithfulness), yet they should also question or refuse dangerous or absurd "evidence" (safety). Currently, no clear boundary exists between the two.

**Goal**: To systematically evaluate LLM behavior when facing different levels of counterfactual medical evidence and reveal the current state of the faithfulness-safety tradeoff.

**Key Insight**: Design four types of progressive counterfactual interventions—ranging from cases where the model has zero prior knowledge (nonsense words) to those that should trigger safety guardrails (toxic substances)—to systematically test the model's "questioning" capability.

**Core Idea**: Models should not only be faithful to the context but also maintain skepticism toward untrustworthy evidence, similar to medical professionals—however, current models almost entirely lack this capability.

## Method

### Overall Architecture

Based on the MedEvidence dataset (284 clinical comparison questions + 329 RCTs), MedCounterFact (809 instances) was constructed through four types of counterfactual replacements. Nine frontier LLMs were evaluated under 4 prompt variants (No evidence/Evidence/Skeptical attitude/Expert role) × 2 response formats (Multiple-choice/Free-form). The entire pipeline is a serial construction-evaluation pipeline: "Base data → Four types of counterfactual interventions → Counterfactual dataset → Prompt variant responses → Multi-dimensional evaluation," as illustrated below.

```mermaid
graph TD
    A["MedEvidence Base<br/>284 Clinical Questions + 329 RCTs"]
    subgraph SUB["Four Counterfactual Interventions (Ignorance → Danger Gradient)"]
        direction TB
        B1["NONCE: Nonsense words<br/>No parametric knowledge"]
        B2["MEDICAL: Mismatched medical terms"]
        B3["NON-MEDICAL: Non-medical items<br/>Violates common sense"]
        B4["TOXIC: Toxic substances<br/>Should trigger safety guardrails"]
    end
    A --> SUB
    SUB --> C["MedCounterFact Dataset<br/>809 Counterfactual Instances"]
    C --> D["Prompt Variant Design<br/>No-Evd / Evd / Skept+Evd / Expert+Evd × MC / Free-form"]
    D -->|9 LLMs at Temp 0| F["Multi-dimensional Evaluation Framework<br/>Uncertain Rate (Higher is better) / EA Rate (Higher is worse)"]
    F --> G["Finding: Nearly Unconditional Compliance<br/>TOXIC ≈ NONCE, Safety Guardrails Fail"]
```

### Key Designs

**1. Four types of counterfactual intervention stimuli: Using a gradient from "ignorance" to "known danger" to find the threshold of the model's questioning capability.**

To determine how models respond when evidence is flawed, the "flaw" is designed as a controllable gradient. The authors systematically replaced clinical trial interventions with four categories: NONCE uses nonsense words (e.g., *blirbex*), where the model has no parametric knowledge; MEDICAL uses real but mismatched medical terms (e.g., replacing chemotherapy with penicillin); NON-MEDICAL uses non-medical items like bowling balls or SIM cards, where accepting their efficacy violates common sense; TOXIC uses known toxic substances like heroin or mustard gas, specifically including notes about "toxic dosages" to ensure safety warnings should be triggered. These four categories form a continuous gradient—if a model does not question the TOXIC category, it indicates that faithfulness has completely overridden safety.

**2. Prompt variant design: Using mitigation strategies as controls to see if skeptical prompts or expert roles can restore questioning.**

Beyond diagnosing the problem, this research explores potential low-cost mitigation strategies. Evaluations are conducted across four prompt variants: No-Evd provides only the question (testing parametric knowledge); Evd includes counterfactual evidence; Skept+Evd requires the model to reason with a skeptical attitude; Expert+Evd assigns the model roles of a clinical expert and Cochrane reviewer. Each variant is tested with multi-choice and free-form formats. The logic is straightforward—if skeptical prompts or expert roles can increase the questioning rate, it provides a ready-to-use mitigation path; otherwise, the problem lies deeper than prompt engineering.

**3. Multi-dimensional evaluation framework: Quantifying "blind compliance" using two opposing metrics: Uncertain rate and EA rate.**

To determine if a model takes counterfactual evidence seriously, two complementary metrics are used. The "Uncertain rate" is the proportion of times a model selects an "uncertain" label (higher is better, indicating skepticism of the premise). The "Evidence Adherence" (EA) rate is the proportion of answers consistent with the original true label; in counterfactual conditions, a high EA rate is negative—it means the model accepted tampered evidence entirely. Reading both together is crucial: a high EA rate combined with a low Uncertain rate is a clear signal of "accepting counterfactual evidence without questioning."

### Loss & Training

No training method. Evaluated 9 LLMs: Gemini-2.5-flash, GPT-5-mini, Llama-3.1-8B/405B-Instruct, Llama-4-Maverick, OLMo-3-7B-Instruct/Think, Qwen2.5-7B-Instruct, HuatuoGPT-o1-7B. Temperature set to 0.

## Key Experimental Results

### Main Results

| Condition | Change in Uncertain Rate | Change in EA Rate |
|------|-----------------|----------|
| No-Evd → Evd (Original) | Significantly decreased | Significantly increased |
| No-Evd → Evd (NONCE) | Significantly decreased | Comparable to Original |
| No-Evd → Evd (TOXIC) | Significantly decreased | Comparable to Original |
| Skept+Evd vs Evd | Uncertain rate increased | EA rate decreased but remains insufficient |
| Expert+Evd vs Evd | No significant improvement | No significant improvement |

### Ablation Study

| Analysis Dimension | Result |
|----------|------|
| No-Evd Condition | Models can sometimes judge counterfactual interventions as unreasonable (higher Uncertain rate) |
| Evd Condition | Counterfactual evidence completely suppresses parametric knowledge and safety awareness |
| TOXIC vs NONCE behavior | Almost no difference—models comply with both equally |
| Free-form vs Multi-choice | Free-form yields lower Uncertain rates—models are less prone to expressing uncertainty without explicit options |
| Representation Analysis ("toaster" case) | Counterfactual evidence causes distribution shift; parametric knowledge is briefly activated then quickly overridden by context |

### Key Findings

- Across all counterfactual stimuli categories, models neither questioned the premise nor refused to answer—even with built-in safety guardrails.
- "Awareness" of unreasonableness occasionally appeared in Chain-of-Thought reasoning, but these doubts were rapidly suppressed to conform to the evidence.
- Skeptical prompting (Skept+Evd) was the only slightly effective mitigation strategy, but it remained far from sufficient for the TOXIC category.
- Model behavior toward NONCE (ignorance) and TOXIC (known danger) counterfactual evidence was nearly identical—a deeply concerning finding.
- Representation analysis showed that parametric knowledge is briefly activated when encountering counterfactual nouns but is overridden as context accumulates.

## Highlights & Insights

- The lack of a "faithfulness-safety boundary" is a profound and urgent issue—current LLMs in medical scenarios are essentially "unconditional believers in evidence."
- The gradient design of the four counterfactual stimuli is clever—progressive exposure from control (NONCE) to extreme (TOXIC) makes the conclusions highly persuasive.
- The pattern of "brief doubt followed by rapid compliance" in reasoning chains reveals that context bias in LLMs runs deeper than safety alignment.
- This serves as a wake-up call for RAG systems—if retrieved evidence is tampered with or incorrect, models will confidently provide dangerous advice.

## Limitations & Future Work

- Counterfactual evidence was generated through simple substitution and did not cover subtle errors like dosage or indication mistakes.
- Evaluation was limited to English and specific medical domains.
- No effective mitigation solutions were proposed—the study focuses on diagnosis.
- Defining where the "intended" faithfulness-safety boundary should lie remains an unresolved normative question.

## Related Work & Insights

- **vs CoPriva/Doc-PP**: The latter focuses on information non-disclosure strategies, while Ours focuses on excessive trust when one "should distrust."
- **vs Xie et al. (2023)**: The latter studies context-knowledge conflict in general domains, while Ours focuses on high-risk medical domains.
- **vs MedEvidence**: Ours is built upon it, extending to counterfactual settings to test model robustness.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Systematic study of faithfulness-safety tension in medical scenarios is a first; stimuli design is sophisticated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 9 models, 4 prompts, 2 formats, and representation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, alarming findings, and strong argumentation.
- Value: ⭐⭐⭐⭐⭐ Major warnings for medical AI safety, directly impacting deployment decisions for RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[ACL 2026\] Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)
- [\[ACL 2026\] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)
- [\[AAAI 2026\] ShortageSim: Simulating Drug Shortages under Information Asymmetry](../../AAAI2026/medical_nlp/shortagesim_simulating_drug_shortages_under_information_asymmetry.md)

</div>

<!-- RELATED:END -->
