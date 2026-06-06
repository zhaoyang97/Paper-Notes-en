---
title: >-
  [Paper Note] Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions
description: >-
  [NeurIPS 2025][LLM Safety][LLM fairness] Through a persona-based evaluation framework, this paper finds that ChatGPT-4o and Bio-Medical-Llama-3-8B are systematically influenced by clinically irrelevant sociodemographic a…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "LLM fairness"
  - "drug safety"
  - "adverse event prediction"
  - "sociodemographic bias"
  - "persona bias"
date: 2026-05-08
content_hash: 3b3342ca5411b6aa
---

# Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions

**Conference**: NeurIPS 2025
**arXiv**: [2510.13931](https://arxiv.org/abs/2510.13931)  
**Code**: Unavailable  
**Area**: Medical Imaging
**Keywords**: LLM fairness, drug safety, adverse event prediction, sociodemographic bias, persona bias

## TL;DR

Through a persona-based evaluation framework, this paper finds that ChatGPT-4o and Bio-Medical-Llama-3-8B are systematically influenced by clinically irrelevant sociodemographic attributes (education, insurance, housing, etc.) in adverse drug event prediction, exhibiting both explicit and implicit bias patterns.

## Background & Motivation

LLMs are increasingly deployed in biomedical domains, yet their reliability and fairness in **drug safety prediction** have not been systematically audited. This paper addresses three core questions:

**Gap in sociodemographic bias research for drug safety**: Prior work has documented LLM bias in law, education, and emergency medicine, but whether patient sociodemographic attributes introduce predictive bias in pharmacovigilance tasks remains entirely unexplored.

**Undue influence of clinically irrelevant attributes**: Adverse event (AE) prediction should rely solely on clinical variables such as age, sex, weight, medications, and diagnoses. Attributes such as education level, marital status, insurance type, and religion have no clinical causal relationship to AE occurrence—yet whether LLMs are influenced by them is unknown.

**Role-stratified deployment**: Commercial AI systems typically provide differentiated services to general practitioners (GPs), specialists, and patients. Whether model behavior is consistent across different user role framings is an open question.

The central hypothesis is: if predictions change when only sociodemographic attributes are altered while all clinical information remains identical, the model is biased.

## Method

### Overall Architecture

A **Persona × Role** evaluation matrix is constructed:
- 25 personas spanning 7 sociodemographic dimensions
- 3 user roles (GP, specialist, patient)
- Evaluated on 1,000 FAERS oncology AE reports

Each record contains 6 structured variables: age, sex, weight, medication, diagnosis, and adverse event. Prompt templates first establish a "baseline assumption (answer: Yes)," then inject a role and persona, instructing the model to "reassess based on the updated context."

### Key Designs

1. **Drug-Safety Decisions Dataset (DSD)**: Constructed from FAERS Q4 2024. The DEMO, DRUG, INDI, and REAC tables are merged, retaining only oncology-related indications, patients aged ≥18, and records with no missing values; only the first adverse event Preferred Term is kept per record. The first 1,000 records are selected as a lightweight, reproducible subset.

2. **25 Sociodemographic Personas**: Spanning 7 dimensions—education level (4 tiers: less than high school to postgraduate), marital status (4 categories), employment status (3 categories), insurance type (3 categories), household language (3: Arabic/Spanish/English), housing stability (4 tiers: homeless to homeowner), and religious affiliation (4 categories). These attributes have no clinical causal relationship to AE occurrence; thus any predictive variation constitutes bias.

3. **Dual Bias Analysis Framework**:

    - **Explicit Bias**: The model explicitly references persona attributes during its reasoning (e.g., "college graduates have better cognitive function and lower fall risk").
    - **Implicit Bias**: Predictions vary with persona changes without any mention of persona attributes in the reasoning, indicating bias embedded in model behavior at a deeper level.

   Explicit bias frequency is quantified by analyzing model-generated explanation texts; implicit bias is assessed by measuring accuracy changes after excluding explicit-bias cases.

### Loss & Training

This paper presents an evaluation framework and involves no model training. The two evaluated models are:
- **ChatGPT-4o**: OpenAI's general-purpose model, accessed via API.
- **Bio-Medical-Llama-3-8B**: A Llama-3 8B variant adapted on biomedical corpora, run locally (NVIDIA 3090 Ti).

## Key Experimental Results

### Main Results: Impact of Sociodemographic Attributes on Prediction Accuracy

| Dimension | Persona | ChatGPT-4o GP (%) | ChatGPT-4o Patient (%) | Bio-Med GP (%) | Bio-Med Patient (%) |
|---|---|---|---|---|---|
| Education | Less than high school | 59.8 | 63.5 | **73.7** | 73.8 |
| Education | Postgraduate | 54.8 | 50.8 | 45.9 | **43.4** |
| Housing | Homeless | **76.3** | 72.2 | 73.0 | 72.0 |
| Housing | Homeowner | **51.8** | 57.1 | 50.1 | 47.7 |
| Insurance | Uninsured | 58.1 | 60.6 | **66.9** | 67.9 |
| Insurance | Private insurance | 49.6 | 54.5 | 49.1 | **45.3** |
| Language | Arabic | 68.6 | **80.4** | 60.6 | 59.5 |
| Language | English | 67.0 | 76.8 | **40.8** | 37.0 |

### Ablation Study: Explicit Bias Frequency and Impact

| Dimension | ChatGPT-4o Citation Rate (Specialist) | Bio-Med Citation Rate (Specialist) | Accuracy Change After Exclusion |
|---|---|---|---|
| Housing – Temporary shelter | **51.8%** | 32.0% | Accuracy improves after exclusion |
| Housing – Homeless | 45.8% | 11.6% | Accuracy improves after exclusion |
| Religion – Religious | **53.6%** | 3.9% | Accuracy improves after exclusion |
| Insurance – Private insurance | 42.7% | 10.8% | Accuracy improves after exclusion |
| Employment – Full-time | 43.8% | 8.1% | Accuracy improves after exclusion |
| Education – Postgraduate | 47.1% | 9.5% | Accuracy improves after exclusion |

### Key Findings

1. **Counter-intuitive accuracy inversion**: Disadvantaged groups (low education, homeless, uninsured) receive higher prediction accuracy, while advantaged groups (postgraduate, homeowner, privately insured) receive lower accuracy.
2. **Severe explicit bias in ChatGPT-4o**: The frequency of referencing sociodemographic attributes during reasoning exceeds 50% for several personas (e.g., "temporary shelter" at 51.8%, "religious" at 53.6%), far exceeding Bio-Medical-Llama (typically <15%).
3. **Spurious causal reasoning**: Models generate clinically ungrounded inferences such as "college graduates have stronger cognitive ability and lower fall risk" and "Arabic speakers are more prone to abdominal pain."
4. **Significant user role effects**: The Patient role consistently achieves the highest accuracy in ChatGPT-4o, with inter-role differences reaching statistical significance across multiple dimensions (p<0.001).
5. **Accuracy improves after excluding explicit bias cases**: This confirms that persona-citing reasoning actively degrades prediction quality.
6. **Implicit bias persists**: Even when persona attributes are not mentioned in reasoning, predictions still vary with persona, indicating that bias is embedded deeply in model behavior.

## Highlights & Insights

- The first systematic audit of sociodemographic bias in LLMs for **drug safety prediction**.
- The **explicit vs. implicit bias** conceptual framework is generalizable and applicable to other high-stakes AI applications.
- Reveals a fundamental contradiction: models exhibit systematic bias even in domains where they "know" such bias is inappropriate.
- The persona framework is elegant and concise, enabling low-cost replication and extension.

## Limitations & Future Work

- The dataset contains only 1,000 oncology records, limiting both scale and domain breadth.
- Only 2 models are evaluated (ChatGPT-4o and Bio-Medical-Llama-3-8B).
- FAERS data inherently carries reporting bias, underreporting, and potential duplication.
- Fixing the baseline assumption as "Yes" may introduce a directional bias.
- Records are selected by sequential rather than random sampling, potentially introducing temporal bias.
- Mitigation strategies such as counterfactual prompting or calibration are not explored.

## Related Work & Insights

- **Omar et al. (2025)**: Evaluates sociodemographic bias in LLMs in emergency department settings; this paper extends such investigation to pharmacovigilance.
- **Gupta et al. (2024)**: Studies implicit reasoning bias arising from persona assignment in LLMs.
- **Pfohl et al. (2024)**: A toolkit for detecting health equity bias in LLMs.
- Implication: Fairness auditing should become a mandatory pre-deployment step for all LLM applications involving clinical decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐ First to reveal persona-induced bias in drug safety prediction; the explicit/implicit bias analysis framework is valuable.
- Experimental Thoroughness: ⭐⭐⭐ Limited data scale and number of models, though the experimental design is rigorous (25 personas × 3 roles × 7 dimensions).
- Writing Quality: ⭐⭐⭐⭐ Well-structured, rich in tables, and qualitative examples are persuasive.
- Value: ⭐⭐⭐⭐ Serves as an important warning for the equitable deployment of LLMs in high-stakes medical settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions](evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)
- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](../../ACL2026/llm_safety/robust_multimodal_safety_via_conditional_decoding.md)
- [\[NeurIPS 2025\] ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search](reliabilityrag_effective_and_provably_robust_defense_for_rag-based_web-search.md)
- [\[NeurIPS 2025\] ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)

</div>

<!-- RELATED:END -->
