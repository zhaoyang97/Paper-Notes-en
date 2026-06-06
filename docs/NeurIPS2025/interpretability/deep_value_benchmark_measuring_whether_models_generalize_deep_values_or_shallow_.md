---
title: >-
  [Paper Note] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences
description: >-
  [NeurIPS 2025][Interpretability][AI alignment] This paper proposes the Deep Value Benchmark (DVB), which employs a confound-then-deconfound experimental design to measure whether LLMs learn deep human values or merely me…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "AI alignment"
  - "deep values"
  - "shallow preferences"
  - "benchmark"
  - "value generalization"
date: 2026-05-08
content_hash: c127249104916fa2
---

# Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences

**Conference**: NeurIPS 2025
**arXiv**: [2511.02109](https://arxiv.org/abs/2511.02109)  
**Code**: [GitHub](https://github.com/josh-ashkinaze/deep-value-benchmark-neurips)  
**Area**: Interpretability
**Keywords**: AI alignment, deep values, shallow preferences, benchmark, value generalization

## TL;DR
This paper proposes the Deep Value Benchmark (DVB), which employs a confound-then-deconfound experimental design to measure whether LLMs learn deep human values or merely memorize shallow preference patterns. Results show that the Deep Value Generalization Rate (DVGR) of all evaluated models averages only 0.30, far below chance level.

## Background & Motivation
**Background**: LLMs are increasingly deployed as AI agents acting on behalf of users (e.g., financial planning, medical advice), and such systems are trained on human preference data to align with human intent.

**Limitations of Prior Work**: There is no systematic method to measure whether LLMs learn deep values (e.g., moral principles) or merely shallow statistical patterns in preference data. Existing alignment benchmarks do not directly distinguish between the two.

**Key Challenge**: If AI systems learn only surface-level correlations rather than deep values, they may exhibit unpredictable or harmful behavior in novel scenarios — yet no quantitative framework currently exists to assess this risk.

**Goal**: To establish an interpretable, quantitative metric for measuring whether models genuinely understand and generalize deep human values.

**Key Insight**: Drawing on confound-control methodology from experimental psychology — first creating a perfect correlation between deep values and surface features, then breaking that correlation to observe model choices.

**Core Idea**: A confound-then-deconfound experimental paradigm that precisely separates whether models learn "values" or "patterns."

## Method

### Overall Architecture
DVB employs an in-context learning paradigm: during the training phase, user preferences are demonstrated with deep value $v_1$ perfectly correlated with surface feature $s_1$; during the test phase, the correlation is broken (surface features are swapped), and the model's choice — whether to follow values or surface features — is observed.

### Key Designs
1. **Deep Values**: Six prima facie duties from W.D. Ross (beneficence, fidelity, justice, non-maleficence, reparation, self-improvement) and five basic social values from Schwartz (security, conformity, tradition, universalism, benevolence), totaling 11 value types.
2. **Shallow Preferences**: Binary preferences (e.g., formal vs. informal address) are generated via GPT-4o and manually validated to confirm they are perceived as "superficial and non-moral." Human annotation accuracy is 0.9.
3. **Contexts**: Eight domains (business, customer service, finance, productivity, communication, healthcare, legal, education) are extracted from Y Combinator AI assistant startup tags, and specific work scenarios are generated using the O*NET occupational activities database.
4. **Deep Value Generalization Rate (DVGR)**: The core metric. During training, user preference is $(v_1, s_1) > (v_2, s_2)$; at test time, options become $(v_1, s_2)$ vs. $(v_2, s_1)$. DVGR = proportion of cases in which the model selects the value-consistent option $(v_1, s_2)$. DVGR = 1 indicates perfect deep value generalization; DVGR = 0 indicates complete reliance on surface features.
5. **Triple Human Validation**: (1) Verifying that humans can distinguish deep values from shallow preferences (construct validity); (2) verifying that options genuinely embody the specified values and preferences (internal validity); (3) verifying that contexts reflect real-world AI applications (external validity).

### Loss & Training
No model training is involved. Evaluation uses in-context learning; each test prompt contains $N \in \{4, 20, 40\}$ training examples and one test question. The full benchmark comprises 400 experimental tuples × 3 training sample sizes × 10 test questions = 12K test instances.

## Key Experimental Results

### Main Results
DVGR scores for 9 models on DVB (all significantly below the chance level of 0.5):

| Model | DVGR |
|-------|------|
| Gemini 2.0 Flash | 0.40 |
| Llama 3-8B | 0.37 |
| GPT-4.1-nano | 0.35 |
| Gemini 2.0 Flash Lite | 0.34 |
| GPT-4o-mini | 0.27 |
| GPT-4o | 0.25 |
| GPT-4.1 | 0.24 |
| Llama 3-70B | 0.24 |
| GPT-4.1-mini | 0.23 |
| **Average** | **0.30** |

### Ablation Study

| Prompt Strategy | Avg. DVGR | vs. Baseline |
|-----------------|-----------|--------------|
| Baseline | 0.30 | — |
| Chain-of-Thought | 0.25 | ↓ Significant drop |
| Explicit instruction to "generalize deep values" | 0.33 | ↑ Slight improvement, still below chance |

Contributing factors:
- **Model size**: Smaller models achieve higher DVGR in 3 out of 5 comparisons (larger models perform worse).
- **Domain**: Business, healthcare, and finance yield relatively higher DVGR; communication, education, and customer service yield lower DVGR.
- **Value type**: Tradition (0.51) and universalism (0.42) are easiest to generalize; fidelity and self-improvement are hardest.
- **Number of training examples**: 4/20/40 demonstrations have virtually no effect on DVGR (Cramér's $V = 0.01$).

### Key Findings
- **All models achieve DVGR below 0.5**, indicating a universal tendency to generalize shallow preferences rather than deep values.
- **Chain-of-Thought is harmful**: Reasoning traces frequently reference surface features, amplifying shallow preference effects.
- **Scale does not help**: Larger models do not improve deep value generalization and may even regress slightly.
- **Models from the same developer behave more similarly** (76.8% agreement rate vs. 72.2% across developers).

## Highlights & Insights
- **Confound-then-deconfound paradigm**: An elegant experimental design inspired by psychological methodology, providing a general framework extensible to other alignment problems.
- **Interpretability of DVGR**: The intuition is clear — above 0.5 indicates value understanding; below 0.5 indicates reliance on surface patterns.
- **Triple validation ensures quality**: Human validation covers shallow preference definitions, option generation, and context design.
- **Counterintuitive finding on CoT**: Chain-of-thought reasoning is not universally beneficial and is in fact harmful on certain alignment tasks.
- **Model concentration risk**: Similar value generalization patterns across models from the same developer raise concerns about AI monoculture.

## Limitations & Future Work
- The experimental design deliberately creates perfect confounding (a worst-case scenario); in practice, the correlation between deep values and surface preferences would not be so extreme.
- Deep values are inherently ambiguous, and surface preferences are sometimes a legitimate basis for decision-making — absolute interpretations of DVGR should be treated with caution.
- Only in-context learning is evaluated; fine-tuning scenarios are not considered — it remains an open question whether targeted training could improve performance.
- Coverage is limited to 9 models, with no evaluation of reasoning-oriented models (e.g., o1, DeepSeek R1).
- Whether the chosen value systems (Ross's duties + Schwartz's values) sufficiently represent "deep values" warrants further discussion.

## Related Work & Insights
- DVB complements the reward hacking literature ("alignment faking"): DVB directly measures what signal the model has learned, rather than whether the model is manipulating rewards.
- LLMs have been shown to over-rely on surface correlations in ARC and concept induction tasks; this paper confirms the same pattern in the value alignment domain.
- The association errors observed in developmental psychology (e.g., over-relying on spurious associations) mirror the shallow generalization behavior of LLMs.
- The findings carry an important warning for preference-based alignment methods such as RLHF and DPO: learning from preference data alone may be insufficient to capture the underlying values.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The confound-then-deconfound paradigm is highly innovative, introducing experimental psychology methodology into AI alignment evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Nine models, 12K test instances, and multi-dimensional analysis, though reasoning models and fine-tuning experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, rigorous methodology, triple validation, and thorough and candid discussion.
- **Value**: ⭐⭐⭐⭐⭐ The work has far-reaching implications for AI alignment, exposing fundamental limitations of current preference learning paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Modularity Networks with Diversity-Preserving Regularization](deep_modularity_networks_with_diversity-preserving_regularization.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](../../ACL2026/interpretability/style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)

</div>

<!-- RELATED:END -->
