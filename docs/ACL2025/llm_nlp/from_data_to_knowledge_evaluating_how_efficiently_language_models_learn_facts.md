---
title: >-
  [Paper Note] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts
description: >-
  [ACL 2025][LLM (Other)][sample efficiency] This work is the first to directly investigate the relationship between the frequency of facts in pre-training data and an LLM's ability to recall them. It proposes two sample efficiency metrics and reveals that while models of different architectures and scales perform similarly on high-frequency facts, they differ significantly on low-frequency facts—making the ability to learn low-frequency facts the key differentiator of model sa…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "sample efficiency"
  - "fact learning"
  - "knowledge probing"
  - "pre-training"
  - "power law"
date: 2026-05-08
content_hash: fbb47147368d92e7
---

# From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts

**Conference**: ACL 2025  
**arXiv**: [2506.16912](https://arxiv.org/abs/2506.16912)  
**Code**: [https://github.com/Jabbawukis/sample-efficiency-evaluation](https://github.com/Jabbawukis/sample-efficiency-evaluation)  
**Area**: LLM/NLP  
**Keywords**: sample efficiency, fact learning, knowledge probing, pre-training, power law

## TL;DR
This work is the first to directly investigate the relationship between the frequency of facts in pre-training data and an LLM's ability to recall them. It proposes two sample efficiency metrics and reveals that while models of different architectures and scales perform similarly on high-frequency facts, they differ significantly on low-frequency facts—making the ability to learn low-frequency facts the key differentiator of model sample efficiency.

## Background & Motivation

### Background

**Background**: LLMs store vast amounts of factual knowledge through pre-training, but their sample efficiency (how many times they need to see a fact before learning it) has not been systematically investigated.

**Limitations of Prior Work**: Information in real-world text follows a long-tail distribution, requiring models to learn rare facts from few occurrences. Existing works compare model performance without considering the frequency information in the training data.

**Key Challenge**: Given two models trained on the same data, which one is more capable of learning facts from limited exposure?

**Goal**: Establish a quantitative framework mapping factual frequency to recall capability to measure sample efficiency.

**Key Insight**: Train various models on the same pre-training corpus, label the frequency of each fact in the training data, and evaluate recall capability using the BEAR probe.

**Core Idea**: Sample efficiency should be modeled as a function of the fact recall probability with respect to the number of training exposures—the larger the slope $\alpha_m$, the more efficient the model is.

## Method

### Overall Architecture
Wikipedia corpus fact frequency statistics -> pre-train multiple models on the same corpus -> BEAR knowledge probing -> analysis by frequency bins -> compute weighted accuracy and power-law fitted sample efficiency metrics.

### Key Designs

1. **Fact Frequency Statistics**

    - For each fact triple (s,r,o) in the BEAR probe, search the training corpus for the number of co-occurrences of s and o within the same sentence.
    - Use aliases and lemmatization to increase matching rates.
    - Design Motivation: Estimate the number of times the model "sees" a specific fact during pre-training.

2. **Two Sample Efficiency Metrics**

    - **Weighted Accuracy**: Bin by frequency, where low-frequency bins are assigned higher weights ($w_i = \exp(-0.05 \cdot l_i)$).
    - **Power-Law Fitting $\alpha_m$**: $F(x) = 1 - (L_0 + \frac{x_0}{(1+x)^{\alpha_m}})$, where a larger $\alpha_m$ indicates higher sample efficiency.
    - Design Motivation: Weighted accuracy is intuitive but difficult to compare, whereas $\alpha_m$ provides a single comparable metric.

3. **Model Training**

    - Train on ~5B tokens of Wikipedia.
    - Three architectures $\times$ two scales = 6 models.
    - Save intermediate checkpoints to track learning dynamics.
    - Design Motivation: Control the training data variable and isolate the impact of architecture and scale.

## Key Experimental Results

### Main Results -- Accuracy by Frequency Bins

| Frequency Bins | Model A (Large) | Model A (Small) | Model B (Large) | Model B (Small) |
|--------|----------|----------|----------|----------|
| 1-5 times | ~25% | ~15% | ~20% | ~12% |
| 6-20 times | ~40% | ~30% | ~35% | ~25% |
| 21-100 times | ~60% | ~50% | ~55% | ~45% |
| 100+ times | ~75% | ~70% | ~73% | ~68% |

### Sample Efficiency Metric $\alpha_m$

| Model | $\alpha_m$ | Description |
|------|-----|------|
| Large Model A | **0.35** | Most efficient |
| Large Model B | 0.30 | |
| Small Model A | 0.25 | |
| Small Model B | 0.20 | Least efficient |

### Key Findings
- **Small performance gaps on high-frequency facts** (almost all models can learn them), **but large gaps on low-frequency facts**.
- **Larger models are more sample-efficient**: $\alpha_m$ increases with scale.
- **Architectural variations are most pronounced on low-frequency facts**.
- **Power-law functions fit the frequency-accuracy relationship well**.
- **Sample efficiency gradually improves during the training process** (validated through checkpoint tracking).

## Highlights & Insights
- **First to directly correlate fact frequency with knowledge probing**—bridging the gap in research on "training data characteristics $\to$ model behavior."
- **Power-law fitting provides an elegant, single metric $\alpha_m$** to compare model efficiency.
- **The discovery of low-frequency facts as a key differentiator** has direct implications for training data strategies.

## Limitations & Future Work
- The frequency estimation method (co-occurrence in the same sentence) may be imprecise.
- Models are only trained on Wikipedia, which may not represent diverse pre-training corpora.
- Future Directions: More precise frequency estimation and analysis on larger-scale pre-training data.

## Related Work & Insights
- **vs Kandpal et al. (2023)**: They discovered that LLMs perform poorly on low-frequency entities, while this work quantifies "how much worse" they perform.
- **vs Neural Scaling Laws (Kaplan et al.)**: They study the power-law relationship of loss with respect to data volume, whereas this work studies the power-law relationship of fact recall probability with respect to frequency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to establish a quantitative framework mapping factual frequency to recall capability
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient variable control, covering multiple architectures and scales
- Writing Quality: ⭐⭐⭐⭐ Elegantly designed metrics
- Value: ⭐⭐⭐⭐ Provides important insights for pre-training data strategies and model comparison

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)
- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)

</div>

<!-- RELATED:END -->
