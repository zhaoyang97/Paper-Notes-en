---
title: >-
  [Paper Note] Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning
description: >-
  [ACL2026][Video Understanding][idiom decomposability] This paper re-examines the Idiom Decomposability Hypothesis (IDH) using contextualized language models as "controlled distributional learners." It finds that model-de…
tags:
  - "ACL2026"
  - "Video Understanding"
  - "idiom decomposability"
  - "syntactic flexibility"
  - "distributional learning"
  - "contextual representations"
  - "OLMo"
date: 2026-05-08
content_hash: 94bfad378092fb44
---

# Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning

**Conference**: ACL2026  
**arXiv**: [2606.03817](https://arxiv.org/abs/2606.03817)  
**Code**: https://github.com/mi-m1/idiom_decomp  
**Area**: NLP Understanding / Phrasal Semantics / LM Analysis  
**Keywords**: idiom decomposability, syntactic flexibility, distributional learning, contextual representations, OLMo

## TL;DR
This paper re-examines the Idiom Decomposability Hypothesis (IDH) using contextualized language models as "controlled distributional learners." It finds that model-derived decomposability is only weakly correlated with human judgments and exhibits a small but stable negative correlation with syntactic flexibility. This suggests that idiomatic behavior is better explained as being shaped by distributional experience, surprisal, and representation stabilization processes.

## Background & Motivation
**Background**: Idiom research has long focused on decomposability—the extent to which the literal meanings of an idiom's constituent words contribute to its overall figurative meaning. The classic IDH posits that more decomposable idioms are more likely to undergo syntactic transformations such as passivization, modifier insertion, and nominalization.

**Limitations of Prior Work**: This hypothesis largely relies on human decomposability ratings and acceptability judgments. However, psycholinguistic studies show these ratings are task-dependent, speaker-variant, and unstable. Human judgments also conflate world knowledge, semantic intuition, familiarity, and linguistic experience, making it difficult to isolate what can be learned solely from distributional exposure.

**Key Challenge**: If idiomatic syntactic behavior is truly determined by internal semantic structure, decomposability should stably predict syntactic flexibility. If behavior primarily stems from usage experience, then factors like frequency, predictability, and representation stability during training might be more critical than constituent mapping.

**Goal**: The authors aim to construct a decomposability diagnostic using internal representations of language models and link it to human ratings, corpus-based syntactic flexibility, frequency, predictability, and pre-training dynamics to test whether IDH holds for distributional learners.

**Key Insight**: Contextualized models learn solely from text distributions without explicit semantic role labeling or human acceptability judgments, serving as a control system for "distributional experience only." If the IDH is naturally recovered within these models, it suggests a distributional basis for the hypothesis; if not, the role of decomposability requires reinterpretation.

**Core Idea**: The similarity between an idiomatic sentence and its gloss representation is treated as an alignment of overall meaning. A leave-one-out mask approach is then used to estimate the contribution of each constituent word to this overall meaning, yielding a model-internal decomposability score to test whether semantic structure predicts actual usage.

## Method

### Overall Architecture
The pipeline consists of four steps. First, contextualized representations are extracted from bidirectional transformers (e.g., BERT, ModernBERT) for each idiom-bearing sentence $s$ and its corresponding gloss-replaced sentence $s_g$. Second, the similarity between the full idiom sentence and the gloss is calculated, followed by masking individual idiom tokens to observe similarity shifts. These token contributions are aggregated into an expression-level decomposability score. Third, the frequency of idioms in different constructional frames is extracted from the enTenTen corpus, using Shannon entropy to measure syntactic flexibility alongside frequency and predictability. Fourth, the study tracks 100 pre-training checkpoints of OLMo-2 7B and OLMo-3 7B to analyze how the similarity between idiom and gloss representations evolves during training.

### Key Designs
1.  **Model-internal Decomposability Metric**:
    - **Function**: Estimates the contribution of constituent words to the figurative meaning directly from hidden-state geometry without relying on human ratings.
    - **Mechanism**: Calculates the similarity $S_{fig}$ between the original sentence $s$ and the gloss sentence $s_g$. For each token $j$ in the idiom span, a masked version $s^{(-j)}$ is created to compute $S_{mask}^{(j)}$. Token contribution is defined as $\Delta_j=|S_{fig}-S_{mask}^{(j)}|$. Idiom-level decomposability is derived using aggregation functions such as mean, maximum, Gini dispersion, entropy, or sum.
    - **Design Motivation**: If a constituent word carries figurative weight, masking it should significantly perturb the alignment between the sentence and its gloss; this perturbation is more indicative of representational mechanisms than direct prompting.

2.  **Corpus-based Syntactic Flexibility and Usage Factors**:
    - **Function**: Converts "syntactic flexibility" from subjective acceptability judgments to actual usage distributions in corpora.
    - **Mechanism**: Idiom occurrences are categorized into constructional types (base form, adverb insertion, adjective insertion, passivization, action nominalization, etc.). Flexibility is represented by the Shannon entropy of these types: $H(i)=-\sum_c p_{i,c}\log_2 p_{i,c}$. Frequency and predictability (masked final-word probability) are also calculated using enTenTen.
    - **Design Motivation**: Since IDH claims decomposability constrains real-world syntactic behavior, it should be tested against actual usage distributions rather than offline human ratings.

3.  **Pre-training Dynamics Analysis**:
    - **Function**: Observes whether idiom representations stabilize early or late in training and identifies which attributes influence this process.
    - **Mechanism**: Cosine similarity between idiom and gloss sentences is computed across 100 checkpoints of OLMo-2 7B and OLMo-3 7B. Linear regression is used to model the interaction between training steps and log frequency, surprisal, and decomposability.
    - **Design Motivation**: While static correlations show final state representations, dynamics reveal whether a distributional learner prioritizes frequency, predictability, or decomposability when forming idiomatic representations.

### Loss & Training
No new models are trained; the core is diagnostic evaluation. Bidirectional models include BERT-base/large (cased/uncased) and ModernBERT-base/large. Pre-training dynamics utilize 100 checkpoints each from OLMo-2-1124-7B and OLMo-3-1025-7B. Statistical tools include Spearman rank correlation, regression analysis, bootstrap confidence intervals, partial correlation, Pearson correlation, and VIF. Multiple similarity functions (cosine, CKA, Wasserstein distance) are compared.

## Key Experimental Results

### Main Results
| Research Question | Sample/Model | Key Result | Interpretation |
|----------|-----------|----------|------|
| Human decomposability vs syntactic flexibility | 90 idioms (Bulkes & Tanner / IMPLI overlap) | No significant relationship | Human decomposability ratings do not stably predict corpus flexibility. |
| Model decomposability vs Human ratings | BERT-large uncased, final layer, Wasserstein + sum | $r(90)=.24, p=.005$ | Weak positive correlation exists, but overlap is limited. |
| Model decomposability vs syntactic flexibility | IMPLI (527 samples) | Max correlation approx. $r(527)=-.16, p=.0002$ | Relationship is small and often negative, contradicting IDH. |
| PP idioms subset | 127 Prepositional Phrase idioms | $\rho=-0.24, p=0.01$ | Higher decomposability in PP idioms correlates with lower flexibility. |
| VP idioms subset | 284 Verb Phrase idioms | $\rho=-0.02, p=0.68$ | No significant relationship for VP idioms, the primary focus of IDH. |

### Ablation Study
| Configuration | Key Metric | Description |
|----------|---------|------|
| Human ratings: frequency | coef = -0.20, z = -2.26, p = 0.02 | Higher corpus frequency correlates with lower human decomposability ratings. |
| Human ratings: predictability | coef = -0.52, z = -0.33, p = 0.73 | Predictability is not significant for human decomposability ratings. |
| BERT-large cased: frequency | coef = -0.29, z = -4.07, p < .001 | Model-derived decomposability also correlates negatively with frequency. |
| Bootstrap CI | 95% CI = [0.07, 0.40] | Best model-human correlation is likely not due to noise but has high uncertainty. |
| VIF | All values near 1 | No severe multicollinearity between frequency, predictability, and decomposability. |

### Key Findings
- The IMPLI dataset contains 527 samples (382 unique idioms); the Bulkes & Tanner subset includes 90 idioms. Analysis covers 8 models (6 bidirectional encoders and 2 OLMo 7B causal LMs).
- In pre-training dynamics, interaction terms for all three attributes were significantly negative: Steps x Frequency (-0.0008, z = -24.69), Steps x Surprisal (-0.0007, z = -22.301), and Steps x Decomposability (-0.0010, z = -36.367). Decomposability had the strongest effect on training dependence.
- Frequency is not the sole explanation. Findings emphasize that frequency alone cannot explain the formation of idiom representations; surprisal and decomposability both participate in the stabilization process.

## Highlights & Insights
- The strength of this paper lies in transforming a traditional linguistic hypothesis into a computable, reproducible representation diagnostic rather than a simple classification task on LLMs.
- The "leave-one-out mask + gloss similarity" design is ingenious: it translates the theory of "constituent contribution" into representational alignment perturbations, maintaining theoretical fidelity while allowing internal measurement.
- The negative correlation is the most striking finding: if decomposability supported syntactic flexibility, positive correlations should emerge. Their absence suggests that high-frequency holistic storage, constructional constraints, and distributional predictability carry more explanatory weight than traditional semantic decomposability.
- The pre-training dynamics analysis moves beyond static probing by identifying when these correlations strengthen or decay during the model's learning process.

## Limitations & Future Work
- The authors acknowledge that the decomposability metric is one possible operationalization and does not exhaust the complex linguistic concept.
- Using BERT-derived decomposability to predict OLMo learning dynamics introduces architectural bias; ideally, metrics should be computed directly on the target model, though causal LMs are less suited for bidirectional masking diagnostics.
- The experiments are limited to English idioms and may not generalize to languages with richer morphology or different idiomatic structures.
- Syntactic flexibility counts rely on predefined constructional frames; information may be lost if idioms have finer constructional variants or register differences.
- The study focuses on correlation and regression; it has not yet tested whether these metrics improve downstream idiom identification, translation, or paraphrasing.
- Future work could explore architecture-agnostic metrics and stabilization processes across different languages and generative model states.

## Related Work & Insights
- **vs. Idiom Decomposability Hypothesis**: IDH predicts a positive correlation between decomposability and flexibility. This paper finds no robust support and even observes negative correlations in human and model data.
- **vs. Usage-based / Constructionist Accounts**: These theories emphasize frequency, predictability, and constructional distribution. The negative frequency effects and pre-training dynamics in this study align more closely with these accounts.
- **vs. Traditional Human Norming**: While human ratings capture subjective transparency, they are confounded by familiarity. This study uses models as controlled learners to isolate what can be explained by distributional exposure.
- **Insight**: Language model analysis is suitable for testing linguistic theories, provided theoretical variables are converted into interpretable internal representation operations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective of re-evaluating IDH through internal representations is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, layers, metrics, and checkpoints, though cross-lingual and architecture-agnostic coverage could be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Clear linguistic background and cautious interpretation; heavy use of formulas and appendices raises the entry barrier slightly.
- Value: ⭐⭐⭐⭐☆ Insightful for idiom processing, LM interpretability, and using NLP to test linguistic theories.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](../../CVPR2026/video_understanding/rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)
- [\[ACL 2026\] TRACE: Evidence Localization-based Multi-Video Event Understanding and Statement Generation](trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)
- [\[AAAI 2026\] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](../../AAAI2026/video_understanding/rethinking_progression_of_memory_state_in_robotic_manipulation_an_object-centric.md)
- [\[ICML 2026\] Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning](../../ICML2026/video_understanding/foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](../../CVPR2026/video_understanding/herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)

</div>

<!-- RELATED:END -->
