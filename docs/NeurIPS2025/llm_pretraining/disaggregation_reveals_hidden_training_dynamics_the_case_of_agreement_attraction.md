---
title: >-
  [Paper Note] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction
description: >-
  [NeurIPS 2025][LLM Pretraining][Training Dynamics] This paper disaggregates language model performance on subject-verb agreement tasks by experimental condition…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Training Dynamics"
  - "Subject-Verb Agreement"
  - "Grammar Learning"
  - "Fine-Grained Analysis"
  - "Psycholinguistics"
date: 2026-05-08
content_hash: 3cd55d93f85aaf81
---

# Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction

**Conference**: NeurIPS 2025
**arXiv**: [2510.24934](https://arxiv.org/abs/2510.24934)
**Code**: [GitHub](https://github.com/jmichaelov/sv-disaggregation-cognitive-interpretability)
**Area**: LLM Pre-training / Interpretability
**Keywords**: Training Dynamics, Subject-Verb Agreement, Grammar Learning, Fine-Grained Analysis, Psycholinguistics

## TL;DR

This paper disaggregates language model performance on subject-verb agreement tasks by experimental condition, revealing multi-phase training dynamics obscured by aggregate metrics: models first learn frequency biases, then local context sensitivity, and finally develop general grammatical rules — a process involving multiple "hidden breakthroughs" rather than simple monotonic improvement.

## Background & Motivation

**Background**: Large language models generally produce grammatically correct text and perform well on basic grammatical tasks such as subject-verb agreement. The linguistic competence of LLMs is widely acknowledged. Nevertheless, even large models such as Chinchilla frequently fail on more challenging grammatical tasks, suggesting that models may be learning increasingly complex heuristics rather than fully generalizable grammatical rules.

**Limitations of Prior Work**: Existing evaluations of grammatical competence typically report aggregate scores across all conditions, masking substantial variation across individual conditions. For instance, language models perform worse on sentences containing attractor nouns — particularly when the attractor's number mismatches the subject — analogous to the agreement attraction effect observed in humans. These fine-grained patterns are averaged away in aggregate scores. Moreover, little research has examined how grammatical competence develops incrementally throughout training.

**Key Challenge**: Aggregate metrics suggest "slow, gradual" learning, but this may conceal rapid and non-monotonic underlying dynamics.

**Goal**: By disaggregating performance across experimental conditions (subject number, presence/absence of attractors, and attractor match/mismatch) and examining multiple checkpoints throughout training, this paper aims to reveal the true dynamics of grammatical learning in language models.

**Key Insight**: The paper draws on classic psycholinguistic paradigms — analyzing error patterns and developmental trajectories — treating the training process of language models as analogous to human language acquisition, and conducting comparative analyses across experimental conditions.

**Core Idea**: Disaggregating grammatical evaluation datasets by condition and tracking per-condition performance across training checkpoints exposes multi-phase "hidden breakthroughs" that aggregate metrics conceal.

## Method

### Overall Architecture

Experiments employ the PolyPythia model suite (10 random seeds, 14M to 410M parameters), evaluated on subject-verb agreement tasks at multiple training checkpoints. Evaluation uses the BIG-bench simple agreement and PP-attractor subsets, as well as psycholinguistic stimuli from Bock and Cutting (1992). Distinct learning phases are identified by disaggregating performance curves across conditions.

### Key Designs

1. **Condition-level Disaggregation**:

    - **Function**: Track each experimental condition separately rather than reporting only aggregate scores.
    - **Mechanism**: Subject-verb agreement sentences are divided into 8 condition combinations: subject number (singular/plural) × attractor (absent/matching/mismatching) × verb type (be-verb vs. other single-token / multi-token). Accuracy is computed per condition at each training checkpoint. The model selects the verb form with the higher log probability; for multi-token verbs, the sum of token log probabilities is used.
    - **Design Motivation**: In psycholinguistics, differences across conditions are themselves theoretically informative — agreement attraction is a classic effect in human sentence processing, and disaggregation enables direct comparison of human and model behavioral patterns.

2. **Multi-seed Stability Validation (PolyPythia)**:

    - **Function**: Control for variance introduced by random initialization and data shuffling.
    - **Mechanism**: Ten Pythia models per size (14M to 410M) with different random seeds are used; results report means and 95% confidence intervals. Each training step corresponds to the same number of tokens, enabling comparison across model sizes and seeds.
    - **Design Motivation**: Results from a single training run may be coincidental; consistent phase patterns across 10 seeds provide strong evidence that the dynamics are systematic.

3. **Single-token vs. Multi-token Verb Distinction**:

    - **Function**: Reveal the influence of tokenization on grammatical learning dynamics.
    - **Mechanism**: Some verbs have singular and plural forms that each occupy one token (e.g., *know/knows*), while others have a plural form occupying one token and a singular form occupying two tokens (e.g., *admire* vs. *admires*). The latter is harder, as recognizing the second token as part of a singular verb requires longer contextual dependencies.
    - **Design Motivation**: If models learn incrementally via n-gram statistics, multi-token verbs require longer dependencies (at least trigrams) and should be learned later — which is precisely what the experiments observe.

### Loss & Training

This paper involves no model training; all experiments use pre-trained PolyPythia models for inference evaluation only. The evaluation metric is accuracy: whether the model assigns higher log probability to the correct verb form.

## Key Experimental Results

### Main Results

The central finding is a three-phase learning dynamic (illustrated with be-verb *is/are*):

| Training Phase | Steps | Behavioral Pattern | Interpretation |
|---|---|---|---|
| Phase 1 | 0–128 | High accuracy on singular conditions; low accuracy on plural conditions | Model learns frequency bias: *is* is more frequent than *are* |
| Phase 2 | 128–512 | Plural and plural+matching-attractor conditions rise sharply; mismatching-attractor conditions drop sharply | Model begins attending to local context (number of the preceding word) but is disrupted by attractors |
| Phase 3 | 512+ | All conditions gradually improve | Model progressively learns longer-distance dependencies |

For non-be verbs, the pattern is reversed but symmetric: an initial bias toward plural forms (bare forms being more frequent), followed by an attractor effect, and subsequent gradual improvement.

### Ablation Study

| Analysis Dimension | Finding |
|---|---|
| Model size (14M–410M) | Smaller models exhibit the same pattern but with greater instability; larger models show clearer patterns with faster transitions |
| Random seed | 10 seeds are largely consistent; minor fluctuations occur in smaller models for individual seeds |
| Single-token vs. multi-token verbs | Phase 2 onset is delayed and the magnitude of change is smaller for multi-token verbs |
| Per-verb analysis | Most verbs show consistent patterns; *stimulate/stimulates* has a smaller frequency difference, resulting in a weaker Phase 1 bias |
| Aggregate scores | Show only slow, gradual improvement, entirely concealing the underlying non-monotonic dynamics |

### Key Findings

- **Aggregate metrics are misleading**: Overall scores suggest slow, stable improvement, but disaggregation reveals that each condition undergoes rapid, non-monotonic change — some conditions rising sharply while others fall simultaneously.
- **Learning is neither sudden nor gradual, but involves multiple "hidden breakthroughs"**: This supports the hidden breakthroughs hypothesis of Kangaslahti et al. (2025).
- **N-gram interpretation**: Chang et al. (2024) found that transformers sequentially overfit unigram → bigram → trigram probabilities during training. Phase 1 corresponds to unigram frequency bias, Phase 2 to bigram sensitivity, and the delayed transition for multi-token verbs reflects trigram dependency requirements.
- **Temporal characteristics of the attractor effect**: The attractor effect is strongest at mid-training and gradually weakens thereafter but does not disappear — both models and humans exhibit similar agreement attraction.

## Highlights & Insights

- **Methodological contribution over technical contribution**: The primary value of this paper lies not in specific discoveries but in demonstrating the power of "condition-level disaggregation + tracking training dynamics" as a general analytical tool — one transferable to any structured evaluation dataset.
- **Bridging psycholinguistics and LLM analysis**: By adopting classic paradigms from human sentence processing research (minimal pairs, agreement attraction), the paper leverages decades of psycholinguistic foundations to understand AI systems. This interdisciplinary perspective is highly instructive.
- **Warning for benchmark design**: If a grammatical task can be solved via bigram statistics, it may lack sufficient construct validity. Many BLiMP subtasks can substantially exceed chance using 5-grams alone, indicating that models need not acquire genuine grammatical knowledge to "pass the test."

## Limitations & Future Work

- **English subject-verb agreement only**: This is among the simplest grammatical phenomena; whether findings generalize to more complex structures (e.g., long-distance dependencies, nested constructions) warrants further investigation.
- **PP-phrase attractors only**: Other attractor types (e.g., relative clauses) may exhibit different dynamics.
- **Observational rather than confirmatory**: Patterns are identified but no mechanistic validation (e.g., ablations or probing) is provided; the n-gram interpretation remains a hypothesis.
- **PolyPythia only**: Constrained by the availability of publicly released multi-seed, multi-checkpoint model suites; generalizability to other architectures is unclear.
- **Overly simple evaluation metric**: Using only accuracy (comparing log probabilities of two verb forms); finer-grained metrics (e.g., surprisal differences) may yield additional insights.

## Related Work & Insights

- **vs. Evanson et al. (2023)**: They also studied training dynamics of grammatical learning but reported only aggregate scores, concluding that subject-verb agreement is "learned relatively early." This paper's disaggregation reveals considerably richer dynamics.
- **vs. Kangaslahti et al. (2025) Hidden Breakthroughs**: They introduced the concept of "hidden breakthroughs" and a bottom-up subset discovery method. This paper reaches analogous conclusions via theory-driven top-down disaggregation.
- **vs. Schaeffer et al. (2023)**: They argued that emergent abilities are "mirages" (artifacts of measurement), whereas this paper provides evidence that emergence is real but concealed by aggregation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The analytical approach is novel and the findings are surprising, though no new algorithms are proposed.
- Experimental Thoroughness: ⭐⭐⭐ — Multi-seed validation is rigorous, but scope is limited to English and a single grammatical phenomenon.
- Writing Quality: ⭐⭐⭐⭐⭐ — Argumentation is clear, figures are highly informative, and the interdisciplinary perspective is compelling.
- Value: ⭐⭐⭐⭐ — Offers practical guidance for understanding LLM training dynamics and for benchmark design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[NeurIPS 2025\] Conformal Risk Training: End-to-End Optimization of Conformal Risk Control](conformal_risk_training_end-to-end_optimization_of_conformal_risk_control.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)

</div>

<!-- RELATED:END -->
