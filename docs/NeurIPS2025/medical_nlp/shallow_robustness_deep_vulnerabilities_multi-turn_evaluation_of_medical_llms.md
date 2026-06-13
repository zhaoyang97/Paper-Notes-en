---
title: >-
  [Paper Note] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs
description: >-
  [NeurIPS 2025][Medical NLP][Medical LLM] This paper proposes the MedQA-Followup framework to systematically evaluate the multi-turn robustness of medical LLMs. It reveals that models exhibit acceptable performance under…
tags:
  - "NeurIPS 2025"
  - "Medical NLP"
  - "Medical LLM"
  - "Multi-Turn Dialogue Robustness"
  - "Cognitive Bias"
  - "MedQA"
  - "Clinical Safety"
date: 2026-05-08
content_hash: 488c9ec9aeb51271
---

# Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.12255](https://arxiv.org/abs/2510.12255)  
**Code**: [GitHub](https://github.com) / [HuggingFace](https://huggingface.co)  
**Area**: Medical Imaging
**Keywords**: Medical LLM, Multi-Turn Dialogue Robustness, Cognitive Bias, MedQA, Clinical Safety

## TL;DR

This paper proposes the MedQA-Followup framework to systematically evaluate the multi-turn robustness of medical LLMs. It reveals that models exhibit acceptable performance under single-turn perturbations (shallow robustness), yet accuracy can catastrophically drop from 91.2% to 13.5% under multi-turn follow-up challenges (deep vulnerability). Notably, indirect contextual manipulation proves more destructive than direct incorrect suggestions.

## Background & Motivation

Medical LLMs are rapidly transitioning from research prototypes to clinical deployment, yet existing evaluations almost exclusively focus on **single-turn question answering** under idealized conditions. In real clinical settings, however:

**Clinical dialogue is inherently multi-turn**: Physicians receive differing opinions from colleagues, patients present web-sourced information, and authoritative figures voice opposing viewpoints.

**Existing robustness research is confined to the "shallow" level**: Works such as BiasMedQA and KGGD only examine the effect of injecting misleading information into the initial prompt, without considering scenarios in which the model's answer is subsequently challenged.

**Critical questions remain unanswered**: Once an LLM has produced a correct diagnosis and is then questioned, can it maintain the correct answer? How do social pressure and authority influence its behavior?

This paper argues that safe clinical deployment requires distinguishing two forms of robustness:
- **Shallow Robustness**: Resistance to misleading information embedded in the initial prompt.
- **Deep Robustness**: Sustained accuracy when challenged across subsequent conversational turns.

## Method

### Overall Architecture

MedQA-Followup constructs a **two-dimensional taxonomy** to organize all evaluation modalities:

- **Temporal axis**: Single-turn ($I(Q) \to A'$) vs. multi-turn follow-up ($Q \to A \to I_1(Q) \to A_1$) vs. cumulative follow-up ($Q \to A \to I_1 \to A_1 \to I_2 \to A_2 \to \ldots$)
- **Intent axis**: Indirect intervention (does not explicitly steer toward a wrong answer) vs. direct intervention (explicitly suggests a wrong answer)

Evaluations are grounded in the MedQA dataset (1,273 USMLE questions), spanning 15 medical domains and 3 examination stages.

### Key Designs

1. **Neutral Re-evaluation (rethink)**: Five techniques serving as a control group — "high-stakes neutral" (emphasizing clinical importance), "temporal neutral" (brief recapitulation), "assumption checking" (identifying unsupported assumptions), "double-checking" (explicitly verifying reasoning), and "option mapping" (systematically eliminating contradictory options). These introduce no misleading information and solely assess whether re-evaluation itself degrades accuracy.

2. **Incorrect Suggestions (inc_letter)**: Five techniques that explicitly recommend an incorrect answer through social or authority-based framing — "authority prior" (senior clinician opinion), "auto-grader prior" (expected system output), "commitment alignment" (multi-source support for the wrong option), "recency prior" (citing analogous recent cases), and "social proof prior" (peer consensus). A key design principle is that incorrect options are always framed as others' perspectives rather than factual assertions (e.g., "A senior physician believes the answer is X").

3. **Contextual Manipulation (context)**: Four techniques introducing additional misleading information — "misleading context" and "RAG-style context" (GPT-4.1-generated text supporting the second most likely option, presented as clinical evidence and retrieved system output, respectively), "alternative context" (supporting evidence for a diagnosis outside the answer set), and "edge case context" (emphasizing atypical presentations and limitations of the correct answer). All contexts comprise 4–10 sentences and are framed as "information to weigh" rather than definitive facts.

### Loss & Training

This paper presents an evaluation framework and does not involve model training. Evaluation configuration:
- Five models: GPT-4.1, GPT-4.1 mini, Claude Sonnet 4, MedGemma 27B, MedGemma 4B
- Deterministic decoding (temperature = 0), fixed random seed 42
- General-purpose models use a minimal system prompt; domain-specific models (MedGemma) use no system prompt

## Key Experimental Results

### Main Results: Single-Turn vs. Multi-Turn Accuracy

| Model | Baseline | Single-Turn inc_letter (avg) | Multi-Turn rethink (avg) | Multi-Turn inc_letter (avg) | Multi-Turn context (avg) |
|---|---|---|---|---|---|
| Claude Sonnet 4 | 91.2% | 89.3% (−2.1%) | 84.6% (−7.2%) | 48.6% (**−46.7%**) | 45.6% (**−50.0%**) |
| GPT-4.1 | 92.5% | 92.1% (−0.4%) | 92.0% (−0.5%) | 89.5% (−3.2%) | 61.7% (**−33.2%**) |
| GPT-4.1 mini | 90.5% | 89.6% (−1.0%) | 89.9% (−0.6%) | 87.0% (−3.8%) | 50.1% (**−44.6%**) |
| MedGemma 4B | 64.2% | 53.1% (−17.3%) | 64.1% (−0.1%) | 54.2% (−15.5%) | 40.0% (−37.6%) |
| MedGemma 27B | 84.7% | 78.8% (−6.9%) | 84.4% (−0.4%) | 69.1% (−18.3%) | 56.9% (−32.8%) |

### Ablation Study: Most Destructive Techniques

| Model | Worst context Technique | Accuracy | Relative Drop |
|---|---|---|---|
| Claude Sonnet 4 | RAG style context | 13.5% | −85.2% |
| GPT-4.1 | RAG style context | 47.8% | −48.3% |
| GPT-4.1 mini | RAG style context | 32.5% | −64.1% |
| MedGemma 4B | Misleading context | 20.7% | −67.8% |
| MedGemma 27B | RAG style context | 27.6% | −67.4% |

### Key Findings

1. **Shallow robustness is largely achieved**: Single-turn direct suggestions have negligible effect on state-of-the-art models (GPT-4.1 drops only 0.4%).
2. **Deep robustness is catastrophically lacking**: All models exhibit accuracy drops exceeding 30% under multi-turn context interventions.
3. **Counterintuitive finding**: Indirect interventions (context) are more destructive than direct interventions (inc_letter).
4. **Pronounced inter-model differences**: Claude is extremely susceptible to authority-based suggestions (−46.7%), while GPT-series models are nearly immune to inc_letter yet equally vulnerable to context.
5. **Clinical application questions (Steps 2 & 3) are more vulnerable than foundational knowledge questions (Step 1)**: Context interventions yield an additional 6–13.5% accuracy drop on clinical questions.
6. **Cumulative interventions exhibit sub-additive effects**: 85% of combinations demonstrate sub-additivity, suggesting a natural ceiling on exploitability through stacked interventions.

## Highlights & Insights

- The first work to systematically define and evaluate the concept of "deep robustness" in medical LLMs.
- Reveals a critical safety vulnerability: context integrated via RAG systems may be more dangerous than direct incorrect suggestions.
- The proposed taxonomy (shallow/deep × direct/indirect) provides a clear conceptual framework for future research.
- The extreme vulnerability of Claude Sonnet 4 is particularly alarming (91.2% → 13.5%).

## Limitations & Future Work

- Evaluation is based on multiple-choice format, which may underestimate vulnerabilities in open-ended clinical dialogue.
- Only five models are assessed; additional model families (e.g., Gemini, Llama 3.1) are not covered.
- Mitigation strategies such as adversarial training or confidence-weighted decoding are not explored.
- Context generation relies on GPT-4.1, introducing methodological bias.
- The combinatorial space of cumulative interventions is vast; only a subset is evaluated.

## Related Work & Insights

- **BiasMedQA**: A pioneering work in single-turn bias evaluation; this paper extends the paradigm to multi-turn settings.
- **AgentClinic**: Evaluates models in simulated multi-turn clinical environments but does not focus on scenarios in which prior answers are challenged.
- Key insight: With the increasing prevalence of RAG systems, ensuring that retrieved context does not mislead models is an urgent and unresolved problem.

## Rating

- Novelty: ⭐⭐⭐⭐ The distinction between deep and shallow robustness is conceptually novel and practically meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five models, 14 intervention techniques, cumulative analysis, length ablations, and domain-level analysis.
- Writing Quality: ⭐⭐⭐⭐ The taxonomy is clear and the experimental organization is systematic.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a core safety concern in clinical LLM deployment; findings carry significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_nlp/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](../../ACL2026/medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[NeurIPS 2025\] H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis](h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)

</div>

<!-- RELATED:END -->
