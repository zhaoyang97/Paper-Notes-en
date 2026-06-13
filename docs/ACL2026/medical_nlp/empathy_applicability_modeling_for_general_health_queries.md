---
title: >-
  [Paper Note] Empathy Applicability Modeling for General Health Queries
description: >-
  [ACL2026][Medical NLP][clinical empathy] This paper proposes the Empathy Applicability Framework (EAF) to determine whether it is "appropriate" to express emotional reactions or interpretative understanding in single-tur…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "clinical empathy"
  - "health queries"
  - "empathy applicability"
  - "annotation framework"
  - "RoBERTa classifier"
date: 2026-05-08
content_hash: daa1d2f0fb540047
---

# Empathy Applicability Modeling for General Health Queries

**Conference**: ACL2026 Findings  
**arXiv**: [2601.09696](https://arxiv.org/abs/2601.09696)  
**Code**: https://github.com/shanmrandhawa/Empathy-Applicability-Framework  
**Area**: Medical NLP / Clinical Empathy Modeling  
**Keywords**: clinical empathy, health queries, empathy applicability, annotation framework, RoBERTa classifier

## TL;DR
This paper proposes the Empathy Applicability Framework (EAF) to determine whether it is "appropriate" to express emotional reactions or interpretative understanding in single-turn patient health queries. By constructing a benchmark with human and GPT-4o annotations and training classifiers, the work provides upstream signals for recognizing empathy needs before medical LLMs generate responses.

## Background & Motivation
**Background**: Empathy in clinical communication typically comprises components such as understanding the patient's situation, responding to emotions, and taking action. Existing NLP frameworks like EmpatheticDialogues, ESConv, and EPITOME mostly focus on "how to generate or evaluate empathetic responses."

**Limitations of Prior Work**: Not every medical query requires an emotional response. For instance, purely factual questions are better suited for direct medical information, whereas queries involving fear, severe symptoms, life burdens, or uncertainty require varying degrees of emotional response or interpretative understanding. Existing frameworks often annotate empathy after response generation, lacking an applicability decision prior to the response.

**Key Challenge**: If LLMs express empathy indiscriminately, they may appear vacuous, offensive, or deviate from facts; if they do not express it at all, they miss the patient's genuine emotional needs. Therefore, the system needs to first determine "when to respond with empathy and which type of empathy to use."

**Goal**: The authors aim to establish a cue-based framework to predict the applicability of two empathy dimensions in single-turn, asynchronous, general health queries: Emotional Reactions Applicability (EA) and Interpretations Applicability (IA).

**Key Insight**: The paper shifts empathy from a response quality problem to a query understanding problem. Instead of generating a response and then checking for empathy, the approach identifies clinical, contextual, and linguistic cues within the patient's query beforehand.

**Core Idea**: Using EAF, the tasks of "appropriateness for expressing emotional warmth" and "appropriateness for understanding/interpreting feelings or situations" are decoupled into two binary classification tasks. Validation through human and GPT labeling, classifier training, and disagreement analysis demonstrates that the framework is learnable and interpretable while retaining necessary subjectivity.

## Method
The focus of EAF is applicability rather than generating specific empathetic sentences. The framework labels patient queries as EA Applicable/Not Applicable and IA Applicable/Not Applicable. EA leans toward emotional responses (e.g., warmth, concern, compassion), while IA leans toward cognitive empathy (e.g., understanding explicit/implicit feelings, experiences, context, or health uncertainty).

### Overall Architecture
The authors sampled 9,500 patient queries from HealthCareMagic and iCliniq. Among these, 1,500 were reserved for dual human and GPT-4o annotation, while the remaining 8,000 were annotated only by GPT-4o. After a three-stage calibration, 1,296 queries were independently labeled by two human annotators. GPT-4o performed five annotation passes on these 1,296 queries to obtain labels via majority vote, and a single pass on the remaining 8,000. Subsequently, two independent RoBERTa-base binary classifiers were trained to predict EA and IA applicability, compared against baselines including random, always applicable, always not applicable, o1 zero-shot, and TF-IDF+LR/SVM.

### Key Designs
1.  **EAF Dual-dimension cue framework**:
    - **Function**: Decomposes "whether empathy is needed" into two upstream judgments: emotional response and interpretative understanding.
    - **Mechanism**: EA Applicable cues include severe negative emotion, inferred negative state, seriousness of symptoms, and concern for relations; EA Not Applicable cues include routine health management, purely factual medical queries, and neutral symptom descriptions. IA focuses on cues like expression of feeling, experiences affecting emotional state, symptoms with emotional impact, and distressing uncertainty.
    - **Design Motivation**: In medical queries, "the patient having emotions" is not identical to "the response needing to express understanding." The dual-dimension design avoids categorizing all non-factual queries crudely as emotional support.

2.  **Layered data construction (Human + GPT)**:
    - **Function**: Obtains a high-confidence human consensus set while scaling with GPT annotations.
    - **Mechanism**: Two trained lay annotators independently labeled 1,296 queries. GPT-4o used contrastive prompts containing definitions, subcategory descriptions, and examples. For the human set, GPT performed 5 passes with majority voting to improve stability.
    - **Design Motivation**: Empathy judgment is highly subjective; the authors emphasized consistency training over crowdsourcing and used GPT to expand the data to 8,000 samples to test if automatic labels could train models approximating human consensus.

3.  **Learnability verification and disagreement analysis**:
    - **Function**: Proves that EAF is a predictable and interpretable framework rather than just a conceptual one.
    - **Mechanism**: Classifiers were evaluated on a human-consensus test set using accuracy, macro-F1, and weighted-F1. Disagreement analysis used UpSet plots to check rationale selection and divergence bars to categorize mismatches as Annotator Spread, LLM-Adds, or LLM-Omits.
    - **Design Motivation**: Clinical empathy cannot pursue a maximum F1 alone; identifying which cues cause disagreement and why determines how the framework is calibrated in real clinical systems.

### Loss & Training
The modeling consists of two independent binary classification tasks: given a patient query $P_i$, predict whether $A_{i,EA}$ and $A_{i,IA}$ are Applicable. The model used is RoBERTa-base, trained for 10 epochs with a learning rate of $2 \times 10^{-5}$ and a batch size of 8. The Human Set was split 75%/5%/20% for training, validation, and testing. The Autonomous Set used the 8,000 GPT-labeled queries for training but was tested on the same human-consensus test set for alignment.

## Key Experimental Results

### Main Results
Reliability was first measured by annotation consistency. Human-human agreement reached a moderate level, while GPT showed higher agreement with the human-consensus subset.

| Dimension | Human-Human $\kappa$ | Human-Human agree/disagree | Human-GPT $\kappa$ | Human-GPT agree/disagree |
|-----------|--------------------:|---------------------------:|------------------:|-------------------------:|
| EA        | 0.521               | 981 / 315                  | 0.614             | 667 / 153                |
| IA        | 0.404               | 898 / 398                  | 0.659             | 681 / 139                |

Classification results show that RoBERTa-base significantly outperforms baselines, indicating that EAF labels possess learnable linguistic patterns.

| Training Set / Model             | EA Acc | EA Macro-F1 | EA Wtd-F1 | IA Acc | IA Macro-F1 | IA Wtd-F1 |
|----------------------------------|-------:|------------:|----------:|-------:|------------:|----------:|
| Random                           | 0.47   | 0.47        | 0.47      | 0.44   | 0.43        | 0.44      |
| Always Applicable                | 0.52   | 0.34        | 0.36      | 0.53   | 0.35        | 0.37      |
| Always Not Applicable            | 0.48   | 0.32        | 0.31      | 0.47   | 0.32        | 0.30      |
| o1 Zero-Shot                     | 0.55   | 0.40        | 0.41      | 0.62   | 0.53        | 0.54      |
| Logistic Regression              | 0.84   | 0.84        | 0.84      | 0.80   | 0.80        | 0.80      |
| Linear SVM                       | 0.83   | 0.83        | 0.83      | 0.77   | 0.77        | 0.77      |
| RoBERTa, Human Set               | 0.92   | 0.92        | 0.92      | 0.87   | 0.87        | 0.87      |
| RoBERTa, GPT-only Autonomous Set | 0.85   | 0.85        | 0.85      | 0.78   | 0.77        | 0.77      |

### Ablation Study
While there was no traditional module ablation, comparisons between data sources and baselines provided interpretability.

| Comparison              | Purpose                                           | Key Result                               | Explanation                                             |
|-------------------------|---------------------------------------------------|-----------------------------------------:|---------------------------------------------------------|
| o1 Zero-Shot vs RoBERTa | Test if labels are more learnable than LLM judgment | EA Macro-F1 0.40 vs 0.92                 | Structured annotation training is superior to zero-shot |
| LR/SVM vs RoBERTa       | Test if surface features are sufficient           | LR EA F1 0.84 vs RoBERTa 0.92            | Surface cues are strong, but context adds gain         |
| Human Set vs GPT Set    | Test GPT label substitutability                   | GPT-only EA F1 0.85                      | Auto-labels are useful but lose to human consensus      |
| EA vs IA                | Test difficulty difference between dimensions     | Human-Human $\kappa$: 0.521 vs 0.404 | IA is more subjective, relying on implicit context      |

### Key Findings
- Human annotation consistency falls within the typical moderate range for empathy tasks, confirming it is not an objective fact classification.
- GPT agreement with human-consensus exceeds 0.6 $\kappa$, indicating EAF effectively guides GPT in clear cases.
- RoBERTa's high Macro-F1 (0.92 for EA) suggests EAF cues represent stable linguistic signals rather than random subjectivity.
- Systematic challenges include subjective inference of implied distress, clinical severity ambiguity, and cultural differences in contextual hardship.

## Highlights & Insights
- The major highlight is shifting empathy modeling forward to the "applicability" stage, serving as a control signal for generation.
- The dual-dimension design avoids the simplification of "emotion = empathy." Many health queries lack explicit emotion but include distressing uncertainty that requires interpretative understanding.
- The authors analyze disagreements rather than treating them as noise, which is crucial for medical NLP where demographic and clinical backgrounds influence empathy.
- Models trained on GPT-only data perform well, suggesting LLMs can scale annotation, though human consensus remains more reliable.

## Limitations & Future Work
- The study relies on only two human annotators without clinical training, which may not represent clinical experts.
- Results are based on GPT-4o and may not generalize to other models or reasoning models.
- The binary classification (Applicable/Not Applicable) lacks intensity levels. Modeling demand strength or uncertainty calibration is a future direction.
- Ethnically, EAF should assist rather than replace clinical judgment; automated empathy may risk an "uncanny valley" effect if it feels insincere.

## Related Work & Insights
- **vs EPITOME**: EPITOME evaluates empathy mechanisms in responses; EAF predicts if those mechanisms are needed at an earlier stage.
- **vs EmpatheticDialogues / ESConv**: These datasets often assume emotional support is needed; EAF explicitly allows for "not applicable."
- **vs cause-aware empathetic generation**: EAF acts as an upstream gate to trigger such generation strategies.
- **vs zero-shot LLM classification**: Structured cue frameworks significantly improve stability over direct zero-shot prompts.

## Rating
- Novelty: ⭐⭐⭐⭐☆
- Experimental Thoroughness: ⭐⭐⭐⭐☆
- Writing Quality: ⭐⭐⭐⭐☆
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)
- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)

</div>

<!-- RELATED:END -->
