---
title: >-
  [Paper Note] Saying the Unsaid: Revealing the Hidden Language of Multimodal Systems Through Telephone Games
description: >-
  [NeurIPS 2025][Hidden language] By running multi-round "telephone games" (image→text→image loops), this paper exploits the preference biases of multimodal systems to quantify the connection strength between concepts in the system's implicit space (i.e., the "hidden language"). It contributes the Telescope dataset (10,000+ concept pairs) and establishes a scalable test-time "world map" of multimodal systems.
tags:
  - NeurIPS 2025
  - Hidden language
  - telephone game
  - concept association
  - multimodal interpretability
  - test-time probing
date: 2026-05-08
content_hash: ed42e71551caae9a
---

# Saying the Unsaid: Revealing the Hidden Language of Multimodal Systems Through Telephone Games

**Conference**: NeurIPS 2025
**arXiv**: [2511.10690](https://arxiv.org/abs/2511.10690)
**Code**: None
**Area**: Interpretability
**Keywords**: Hidden language, telephone game, concept association, multimodal interpretability, test-time probing

## TL;DR
By running multi-round "telephone games" (image→text→image loops), this paper exploits the preference biases of multimodal systems to quantify the connection strength between concepts in the system's implicit space (i.e., the "hidden language"). It contributes the Telescope dataset (10,000+ concept pairs) and establishes a scalable test-time "world map" of multimodal systems.

## Background & Motivation

Closed-source multimodal systems (e.g., GPT-4o) have achieved remarkable progress in recent years, yet their closed features, data, and architectures preclude the use of training-based methods to study how these systems understand the world. Traditional probing approaches—attention maps, PCA, linear probes—require access to internal model representations, rendering them infeasible in the era of closed-source systems.

The root cause is as follows: multimodal systems learn concept associations by fitting textual and visual representations, but imbalances in training data lead to uneven connection strengths across concept pairs—some combinations are thoroughly trained and thus tightly coupled, while others remain weakly linked. This disparity constitutes the system's "hidden language," yet effective test-time methods for revealing it are lacking.

The paper's key insight is that multimodal systems tend to drop weakly connected concepts during image→text compression and to introduce strongly connected ones during text→image reconstruction. This preference bias can be strategically exploited: by amplifying these shifts over multiple rounds of a telephone game, the connection strength between concepts can be quantified entirely at test time.

## Method

### Overall Architecture

The framework consists of three components: the Telephone Game, a Co-occurrence Frequency metric, and the Telescope dataset. For fully integrated multimodal systems (e.g., GPT-4o), the same system performs both the text→image and image→text processes; for systems composed of separate components, a V-LLM and a text-to-image model from the same organization are combined.

### Key Designs

1. **Telephone Game Mechanism**: The framework involves two core processes. During image→text compression, the system tends to read out the more strongly connected concepts in its understanding (e.g., reading a cow as a pig). During text→image reconstruction, it tends to generate more strongly connected concepts (e.g., replacing cherries with balloons). Changes from a single reconstruction round may be subtle (e.g., the generated image may resemble both a cow and a pig), but multiple rounds of the telephone game amplify these shifts: fragile concept combinations gradually collapse, revealing their weak connection strength within the system's understanding.

2. **Co-occurrence Frequency Metric**: Over $n$ rounds of telephone games, the co-occurrence frequency of a concept pair "A and B" is defined as $F(A,B) = \frac{\sum_{i=1}^{r}\sum_{j=1}^{n}\mathcal{I}_{i,j}(A,B)}{r \times n}$, where $r$ is the number of repetitions and $\mathcal{I}_{i,j}(A,B)$ indicates whether A and B co-occur in round $j$ of repetition $i$ (as judged by an LLM). Higher co-occurrence frequency indicates stronger conceptual connection. This metric simultaneously captures training bias and generalization ability: stronger generalization corresponds to a more uniform distribution of connection strengths.

3. **Telescope Dataset**: The dataset contains 150 common visual concepts forming 11,175 simple-pattern concept pairs (two concepts placed side by side) and 450 complex-pattern concept pairs (three fusion strategies: displayed on a television screen, rendered in Van Gogh's style, and depicted in a wood texture). This dataset serves as the systematic foundation for telephone game probing.

### Loss & Training

This paper does not involve model training. The entire framework is a purely test-time method that requires no access to model parameters. An LLM is used as a "concept verifier" to determine whether concepts co-occur in image descriptions, and reasoning-capable LLMs (GPT-o1, DeepSeek-R1) are used to analyze implicit associations beyond semantic and visual similarity.

## Key Experimental Results

### Main Results (Metric Correlation Analysis)

| Metric Pair | Pearson Correlation | Note |
|-------------|---------------------|------|
| Co-occurrence frequency vs. semantic similarity | 0.046 | Near zero |
| Co-occurrence frequency vs. visual similarity | -0.178 | Weak negative |
| Semantic similarity vs. visual similarity | 0.041 | Near zero |
| OpenAI vs. StepFun | 0.506 | Moderate positive |
| OpenAI vs. QWen | 0.475 | Moderate positive |
| StepFun vs. QWen | 0.503 | Moderate positive |

### Ablation Study (Concept Collapse Rate in Complex Patterns)

| Pattern | Concept Collapse Rate | Note |
|---------|-----------------------|------|
| Van Gogh style | 0.767 | Highest collapse rate, worst generalization |
| Displayed on TV screen | 0.740 | High collapse rate |
| Wood texture | 0.560 | Relatively stable |
| TV screen (after bridge improvement) | 0.427 | Significantly improved by introducing "cartoon style" as intermediate concept |

### Key Findings

- **Semantic and visual similarity cannot explain the hidden language.** Across 246 collapsed concept pairs, the correlation between co-occurrence frequency and semantic/visual similarity is near zero, demonstrating the need for a new metric.
- **The hidden languages of different systems are moderately correlated** (~0.5), supporting the Platonic Representation Hypothesis: as multimodal systems scale up, their internal representations tend to converge toward modeling the joint statistical structure of real-world events.
- In simple patterns, after 5 rounds of telephone games, the collapse rate for original concepts is 26.4% (single concept) and 24.4% (two identical concepts), revealing significant biases in multimodal systems.
- Complex patterns (Van Gogh style, TV display) are more fragile than simple patterns, indicating insufficient generalization by the systems to these scenarios.
- Reasoning-capable LLMs can uncover associations beyond semantic/visual similarity—for example, the stable connection between "cow and cola" stems from cows frequently appearing on milk packaging, while milk and cola often co-occur in beverage aisles.

## Highlights & Insights

- The telephone game metaphor is highly apt: just as information becomes distorted through human whisper chains, multimodal systems introduce preference biases during compression–reconstruction cycles.
- The framework is test-time scalable: each new round of telephone games tends to reveal new conceptual connections, progressively building a more complete "world map" of concepts as computational budget increases.
- The discovery of "concept bridging" has strong practical value—introducing an intermediate concept (e.g., "cartoon style") can significantly strengthen fragile concept connections, offering a pathway to improve the controllability of multimodal system outputs.
- The analogy between reasoning-capable LLMs and "MLP linear probes" for interpreting textual evolution is illuminating, representing a paradigm shift from internal inspection to behavioral analysis.

## Limitations & Future Work

- The number of telephone game rounds is finite; concept pairs with a co-occurrence frequency of 1.0 must be excluded from metric correlation analysis, as infinite rounds cannot be conducted.
- The diversity of outputs from large-scale systems may introduce stochastic noise.
- The current work focuses on combinations of two concepts; more complex multi-concept combinations are left for future exploration.
- The high experimental cost (requiring calls to closed-source APIs) constrains reporting to a subset of the Telescope dataset.
- Concept collapse/survival judgments rely on an LLM, which may introduce judgment biases.

## Related Work & Insights

- This work is related to the "hidden language of diffusion models" proposed by Chefer et al., but targets closed-source systems and operates entirely at test time.
- The Platonic Representation Hypothesis predicts that internal representations of different systems converge; the cross-system correlation analysis (~0.5) reported in this paper provides empirical support.
- Validation experiments on the preference biases of open-source model components such as CLIP (Appendix F) strengthen the credibility of the conclusions.
- Insight: In the absence of access to model internals, carefully designed input–output interaction loops can effectively probe a model's internal preferences and knowledge structure.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The telephone game framework and the concept of hidden language are highly innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Rich multi-system comparisons and bridging experiments; data scale is limited by API costs)
- Writing Quality: ⭐⭐⭐⭐⭐ (Fluent narrative, apt metaphors, excellent visualizations)
- Value: ⭐⭐⭐⭐ (Offers a novel paradigm for interpretability of closed-source systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](evaluating_llms_in_open-source_games.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[NeurIPS 2025\] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
