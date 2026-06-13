---
title: >-
  [Paper Note] Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA
description: >-
  [ACL 2026][Medical NLP][Calibration bias] This study investigates how social identity markers (sexual orientation and religious belief) distort the accuracy and confidence calibration of LLMs in medical QA. It finds that…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Calibration bias"
  - "social identity markers"
  - "medical QA"
  - "uncertainty estimation"
  - "intersectional identities"
date: 2026-05-08
content_hash: 7911d3f148ac8303
---

# Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA

**Conference**: ACL 2026  
**arXiv**: [2604.17316](https://arxiv.org/abs/2604.17316)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: Calibration bias, social identity markers, medical QA, uncertainty estimation, intersectional identities

## TL;DR

This study investigates how social identity markers (sexual orientation and religious belief) distort the accuracy and confidence calibration of LLMs in medical QA. It finds that the "homosexual" marker consistently leads to performance degradation and a "calibration crisis" across nine LLMs, and that intersectional identities produce non-additive, specific harms.

## Background & Motivation

**Background**: LLMs are being rapidly integrated into clinical workflows (e.g., patient communication, decision support). Clinical systems often rely on model confidence scores to triage cases, trigger escalations, or hand over to clinicians. Therefore, safe deployment requires not only high accuracy but also robust uncertainty calibration.

**Limitations of Prior Work**: Existing research has shown that social descriptors (e.g., race, gender) can alter LLM clinical recommendations, but the impact of identity markers on model uncertainty remains unevaluated. This blind spot is particularly dangerous in clinical scenarios—if identity cues systematically influence confidence signals, it leads to inequitable patient triage.

**Key Challenge**: Social identity information with no diagnostic value should not affect medical reasoning. However, LLMs may learn biased patterns associated with these identities from training data, causing simultaneous impairment of accuracy and calibration.

**Goal**: Systematically quantify the impact of sexual orientation and religious markers on LLM medical QA performance and Semantic Entropy calibration.

**Key Insight**: A counterfactual approach is employed—inserting different identity marker sentences into the same clinical case to compare changes in model performance.

**Core Idea**: Identity markers do more than shift prediction distributions; they undermine the reliability of confidence signals. A "calibration crisis" is more dangerous than a decrease in accuracy.

## Method

### Overall Architecture

The study randomly samples 2,364 medical questions from MedQA-USMLE, generates counterfactual variants via templates (adding sexual orientation and/or religious descriptions), and evaluates QA accuracy and Semantic Entropy calibration across nine LLMs.

### Key Designs

1.  **Counterfactual Variant Construction**:
    *   **Function**: Generates medical question variants that differ only in identity markers.
    *   **Mechanism**: A template sentence, such as "The patient identifies as heterosexual/homosexual" and/or "The patient is Catholic/Muslim/atheist," is inserted before the final sentence of each clinical case. Questions already containing orientation, religion, or psychiatric content are excluded. This results in 8 categories of variants: +hetero, +homo, +Cat, +Mus, +Ath, +homo+Cat, +homo+Mus, +homo+Ath.
    *   **Design Motivation**: To ensure that observed differences originate solely from identity markers while all other conditions remain identical.

2.  **Semantic Entropy Calibration Evaluation**:
    *   **Function**: Quantifies changes in the reliability of model uncertainty across different identity conditions.
    *   **Mechanism**: Semantic Entropy (SE) is used as the measure of uncertainty—quantifying predictive uncertainty over semantically equivalent output classes rather than surface forms. Calibration quality is evaluated via Brier scores to detect whether identity markers cause the model to maintain high confidence during incorrect predictions.
    *   **Design Motivation**: Semantic Entropy reflects the true uncertainty of the model better than simple output probabilities and performs best in clinical scenario validation.

3.  **Intersectional Identity Impact Analysis**:
    *   **Function**: Detects whether combined identity markers produce additional harm beyond the additive effects of single markers.
    *   **Mechanism**: Performance under single markers (e.g., +homo) is compared against combined markers (e.g., +homo+Muslim). If the combined effect exceeds the sum of the two individual effects, it indicates non-additive intersectional harm.
    *   **Design Motivation**: Real-world patients often hold multiple identities, making intersectional effects more reflective of actual risks than single-dimension analysis.

### Loss & Training

This is a pure evaluation study and does not involve training.

## Key Experimental Results

### Main Results

Changes in accuracy (Base represents original accuracy; others represent relative changes):

| Model | Base | +hetero | +homo | +homo+Cat | +homo+Mus | +homo+Ath |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-3.2-3B | 55.58 | +0.72 | -0.33 | **-3.46** | -1.31 | **-2.66** |
| Bio-Medical-Llama-8B | 64.21 | -1.60 | **-2.37** | **-5.58** | **-4.44** | **-4.27** |
| LLaMA-3.1-70B | 84.31 | **-1.74** | **-2.92** | **-3.47** | **-1.95** | **-2.84** |
| OpenBioLLM-70B | 77.44 | **-2.65** | **-7.21** | **-5.10** | **-2.65** | **-3.78** |
| GPT-5.1 | 89.21 | -0.80 | **-1.35** | -0.59 | **-1.44** | **-1.52** |

### Ablation Study

Relative change in Brier scores (higher = worse calibration):

| Model | +homo | +homo+Cat | +homo+Mus |
| :--- | :--- | :--- | :--- |
| Bio-Medical-Llama-8B | +14.1% | +11.2% | +14.3% |
| LLaMA-3.1-8B | +5.1% | +6.8% | +7.2% |
| OpenBioLLM-70B | Significant Deterioration | Significant Deterioration | Significant Deterioration |

### Key Findings

*   The "heterosexual" marker acts as an approximately neutral baseline, while the "homosexual" marker consistently triggers accuracy drops and calibration degradation across all 9 LLMs.
*   Intersectional identities produce non-additive harm: the effect of +homo+Catholic often exceeds the combined individual effects of +homo and +Catholic.
*   Specialized biomedical models (Bio-Medical-Llama, OpenBioLLM) exhibit larger biases than general-purpose models, which is counter-intuitive.
*   The same patterns were confirmed in open-ended generation settings, ruling out potential artifacts from the multiple-choice format.
*   Even the most powerful model, GPT-5.1, is affected, though to a lesser degree.

## Highlights & Insights

*   The introduction of the "calibration crisis" concept is crucial: in clinical settings, a high-confidence incorrect answer is far more dangerous than a low-confidence one. Identity markers undermine the reliability of the confidence signal itself, not just accuracy.
*   The finding that domain-specific biomedical models have larger biases is counter-intuitive—likely because biomedical fine-tuning data contains more identity-related bias patterns.
*   The use of Semantic Entropy instead of simple probabilities to measure uncertainty is a methodological highlight that makes the results more robust.

## Limitations & Future Work

*   Identity markers only cover 3 religions and 2 sexual orientations; broader identity coverage is a future direction.
*   Identity insertion uses template sentences; in real clinical records, identity information is presented in more diverse and implicit styles.
*   Only English USMLE questions were evaluated; bias patterns might differ in other languages and healthcare systems.
*   No mitigation solution was proposed—how to eliminate identity bias while maintaining clinical accuracy remains an open question.

## Related Work & Insights

*   **vs. Ji et al. (2025)**: Studied the impact of sociodemographic attributes on clinical trial matching but did not evaluate uncertainty calibration; this paper is the first to bring calibration analysis into bias research.
*   **vs. Hirsch et al. (2026)**: Researched LGBTQIA+ bias but not in clinical scenarios; this paper focuses on actual safety risks in medical QA.
*   **vs. Schmidgall et al. (2024)**: Examined the impact of cognitive bias on LLMs but did not involve identity markers; this paper focuses on systematic bias caused by social identity.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ First to combine calibration bias with social identity markers.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models, 2364 questions, multiple identity combinations, and open-ended validation.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and rigorous experimental design.
*   Value: ⭐⭐⭐⭐⭐ Provides a major warning regarding the fairness and safety of clinical LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[AAAI 2026\] Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](../../AAAI2026/medical_nlp/measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)

</div>

<!-- RELATED:END -->
