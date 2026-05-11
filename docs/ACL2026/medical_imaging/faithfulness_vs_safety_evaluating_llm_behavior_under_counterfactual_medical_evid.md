---
title: >-
  [Paper Note] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence
description: >-
  [ACL 2026][Medical Imaging][Faithfulness-safety conflict] This paper introduces the MedCounterFact dataset—constructed by systematically replacing interventions in clinical trials with nonsense words, medical terminology…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Faithfulness-safety conflict"
  - "counterfactual evidence"
  - "medical QA"
  - "safety guardrails"
  - "RAG"
date: 2026-05-08
content_hash: 7cc671cc89dfcc30
---

# Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence

**Conference**: ACL 2026
**arXiv**: [2601.11886](https://arxiv.org/abs/2601.11886)
**Code**: [GitHub](https://github.com/KaijieMo-kj/Counterfactual-Medical-Evidence)
**Area**: Medical Imaging
**Keywords**: Faithfulness-safety conflict, counterfactual evidence, medical QA, safety guardrails, RAG

## TL;DR

This paper introduces the MedCounterFact dataset—constructed by systematically replacing interventions in clinical trials with nonsense words, medical terminology, non-medical objects, and toxic substances—and finds that state-of-the-art LLMs almost unconditionally defer to context when presented with counterfactual medical evidence, confidently providing answers even when the "evidence" attributes therapeutic efficacy to heroin or mustard gas. The findings expose a critical lack of a well-defined boundary between faithfulness and safety.

## Background & Motivation

**Background**: RAG and evidence-grounded reasoning are widely regarded as key approaches to reducing hallucination in LLMs, particularly in high-stakes domains such as medicine, where evidence-based systems are assumed to be more accurate. An increasing number of lay users rely on LLMs as their primary source of health information.

**Limitations of Prior Work**: (1) Prior studies have shown that in-context information can suppress LLMs' parametric knowledge, but this has been examined primarily in general-domain settings. (2) In the medical domain, evidence-grounded faithfulness is considered a desirable property—but what happens when the evidence itself is flawed? (3) Existing medical QA work assumes that provided evidence is always valid and does not investigate model behavior under erroneous or adversarial evidence.

**Key Challenge**: There is a fundamental tension between faithfulness and safety—models are expected both to faithfully follow provided context (faithfulness) and to question or refuse dangerous or absurd "evidence" (safety). Currently, no clear boundary exists between these two objectives.

**Goal**: To systematically evaluate LLM behavior when confronted with counterfactual medical evidence of varying severity, and to characterize the current state of the faithfulness-safety trade-off.

**Key Insight**: Four progressively severe counterfactual interventions are designed—ranging from cases where the model has no prior parametric knowledge (nonsense words) to cases that should trigger safety guardrails (toxic substances)—to systematically probe the model's capacity for skepticism.

**Core Idea**: Models should not only be faithful to context but should also maintain appropriate skepticism toward implausible evidence, as a medical professional would. Current models almost entirely lack this capability.

## Method

### Overall Architecture

Building on the MedEvidence dataset (284 clinical comparative questions + 329 RCTs), MedCounterFact (809 instances) is constructed via four types of counterfactual substitutions. Nine frontier LLMs are evaluated across 4 prompt variants (no evidence / with evidence / skeptical stance / expert role) × 2 response formats (multiple-choice / free-form).

### Key Designs

1. **Four Categories of Counterfactual Interventions**:

    - Function: To probe model sensitivity to implausible evidence along distinct dimensions.
    - Mechanism: (a) NONCE—nonsense words (e.g., *blirbex*), for which the model has no parametric knowledge; (b) MEDICAL—real but contextually mismatched medical terms (e.g., substituting penicillin for a chemotherapy agent); (c) NON-MEDICAL—everyday non-medical objects (e.g., a bowling ball, a SIM card), whose therapeutic efficacy would violate common sense; (d) TOXIC—known toxic substances (e.g., heroin, mustard gas), with notes on "toxic dosage" to ensure safety warnings should be triggered.
    - Design Motivation: The four categories form a gradient from "ignorance" to "known danger." If models fail to express skepticism even in the TOXIC category, this demonstrates that faithfulness completely overrides safety.

2. **Multi-Dimensional Evaluation Framework**:

    - Function: To capture distinct response patterns toward counterfactual evidence.
    - Mechanism: Two key metrics—(a) *Uncertain rate*: the proportion of responses in which the model selects an "uncertain" label (higher is better, indicating skepticism); (b) *Evidence Adherence (EA) rate*: the proportion of responses consistent with the original ground-truth label (under counterfactual conditions, a high EA rate indicates the model has accepted the counterfactual evidence as true).
    - Design Motivation: High EA rate + low Uncertain rate = the model uncritically accepts counterfactual evidence.

3. **Prompt Variant Design**:

    - Function: To test whether different prompting strategies can activate the model's skeptical reasoning.
    - Mechanism: (a) No-Evd—question only, testing parametric knowledge; (b) Evd—question with counterfactual evidence; (c) Skept+Evd—instruction to reason skeptically; (d) Expert+Evd—model assigned the role of a clinical expert and Cochrane reviewer. Both multiple-choice and free-form response formats are tested.
    - Design Motivation: If skeptical prompting or expert role-playing improves the skepticism rate, it provides a practical direction for mitigation.

### Loss & Training

No training is involved. Nine LLMs are evaluated: Gemini-2.5-flash, GPT-5-mini, Llama-3.1-8B/405B-Instruct, Llama-4-Maverick, OLMo-3-7B-Instruct/Think, Qwen2.5-7B-Instruct, and HuatuoGPT-o1-7B. Temperature is set to 0.

## Key Experimental Results

### Main Results

| Condition | Change in Uncertain Rate | Change in EA Rate |
|-----------|--------------------------|-------------------|
| No evidence → With evidence (original) | Significantly decreases | Significantly increases |
| No evidence → With evidence (NONCE) | Significantly decreases | Comparable to original |
| No evidence → With evidence (TOXIC) | Significantly decreases | Comparable to original |
| Skept+Evd vs. Evd | Uncertain rate increases | EA rate decreases but remains insufficient |
| Expert+Evd vs. Evd | No significant improvement | No significant improvement |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| No-evidence condition | Models sometimes correctly identify counterfactual interventions as implausible (relatively higher Uncertain rate) |
| With-evidence condition | Counterfactual evidence fully suppresses models' prior knowledge and safety awareness |
| TOXIC vs. NONCE behavioral difference | Virtually none—models defer equally to both |
| Free-form vs. multiple-choice | Free-form yields lower Uncertain rates—without explicit options, models are less inclined to express uncertainty |
| Representational analysis ("toaster" case) | Counterfactual evidence causes a distributional shift; parametric knowledge is briefly activated but rapidly overridden by context |

### Key Findings

- Across all counterfactual intervention categories, models neither question the premise nor refuse to answer—even when built-in safety guardrails are present.
- Chains of reasoning occasionally exhibit awareness of implausibility, but such skepticism is swiftly suppressed in order to conform to the provided evidence.
- Skeptical prompting (Skept+Evd) is the only mitigation strategy with any measurable effect, yet it remains far from sufficient for the TOXIC category.
- Model behavior is essentially identical for NONCE (ignorance) and TOXIC (known danger) counterfactual evidence—this is the most alarming finding.
- Representational analysis shows that parametric knowledge is briefly activated upon encountering counterfactual intervention terms but is progressively overridden as context accumulates.

## Highlights & Insights

- The absence of a well-defined faithfulness-safety boundary is a profound and urgent problem—current LLMs are, in effect, unconditional believers in whatever evidence they are provided in medical contexts.
- The gradient design of the four counterfactual categories is methodologically elegant: the progression from a controlled condition (NONCE) to an extreme condition (TOXIC) lends strong credibility to the conclusions.
- The recurring pattern of "brief skepticism followed by immediate compliance" in reasoning chains reveals that LLMs' context-following bias operates at a deeper level than safety alignment.
- The findings serve as a clear warning for RAG systems: if retrieved evidence is tampered with or erroneous, models will confidently generate dangerous recommendations.

## Limitations & Future Work

- Counterfactual evidence is generated via simple substitution and does not cover more subtle errors (e.g., incorrect dosage, wrong indication).
- Evaluation is limited to English and specific medical domains.
- No effective mitigation solution is proposed—the work is diagnostic in nature.
- The normative question of what the "appropriate" faithfulness-safety boundary should be remains unresolved.

## Related Work & Insights

- **vs. CoPriva/Doc-PP**: The latter focuses on information non-disclosure strategies, whereas this work addresses over-trust in evidence that ought to be distrusted.
- **vs. Xie et al. (2023)**: The latter examines context-knowledge conflicts in general domains; this work focuses on the high-stakes medical setting.
- **vs. MedEvidence**: This work builds upon MedEvidence and extends it to a counterfactual setting to probe the robustness of model behavior.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The systematic study of the faithfulness-safety tension in medical contexts is pioneering; the four-category intervention design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 9 models, 4 prompt variants, 2 response formats, and representational analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, findings are striking, and argumentation is compelling.
- Value: ⭐⭐⭐⭐⭐ Carries significant implications for medical AI safety and directly informs deployment decisions for RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/medical_imaging/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[NeurIPS 2025\] Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions](../../NeurIPS2025/medical_imaging/robust_or_suggestible_exploring_non-clinical_induction_in_llm_drug-safety_decisi.md)
- [\[ACL 2026\] OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)

</div>

<!-- RELATED:END -->
