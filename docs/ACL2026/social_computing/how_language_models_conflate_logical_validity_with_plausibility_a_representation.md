---
title: >-
  [Paper Note] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects
description: >-
  [ACL 2026][Social Computing][Content Effects] Through representational analysis, this work reveals that the concepts of "logical validity" and "plausibility" are highly aligned in LLM hidden layer spaces, causing models to conflate plausibility with validity (content effects). The paper constructs debiasing steering vectors that effectively decouple these two concepts, reducing content effects while improving reasoning accuracy.
tags:
  - ACL 2026
  - Social Computing
  - Content Effects
  - Logical Validity
  - Plausibility
  - Linear Representations
  - Steering Vectors
date: 2026-05-08
content_hash: 9843776fdfb796dc
---

# How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects

**Conference**: ACL 2026  
**arXiv**: [2510.06700](https://arxiv.org/abs/2510.06700)  
**Code**: [https://github.com/leobertolazzi/content-effect-interpretability](https://github.com/leobertolazzi/content-effect-interpretability)  
**Area**: Social Computing  
**Keywords**: Content Effects, Logical Validity, Plausibility, Linear Representations, Steering Vectors

## TL;DR
Through representational analysis, this work reveals that the concepts of "logical validity" and "plausibility" are highly aligned in LLM hidden layer spaces, causing models to conflate plausibility with validity (content effects). The paper constructs debiasing steering vectors that effectively decouple these two concepts, reducing content effects while improving reasoning accuracy.

## Background & Motivation

**Background**: Humans exhibit "content effects" in logical tasks such as syllogistic reasoning—the plausibility of semantic content influences judgments of logical validity (e.g., invalid arguments with plausible conclusions are easily misjudged as valid). This human phenomenon is explained by dual-process theory (fast intuitive system vs. slow analytical system). Recent studies have found that LLMs also exhibit similar content effects.

**Limitations of Prior Work**: While content effect behaviors in LLMs have been well-documented, their underlying mechanisms remain unclear. Existing research has remained at the behavioral level of observation, lacking deep analysis of LLM internal representations.

**Key Challenge**: Logical validity depends on argument structure rather than content, but LLMs may entangle these two conceptually independent notions in their representation space.

**Goal**: (1) Verify whether LLMs exhibit content effects; (2) Analyze how validity and plausibility are encoded in internal representations; (3) Explore whether representational entanglement predicts behavioral content effects; (4) Design interventions to decouple these two concepts.

**Key Insight**: Based on the linear representation hypothesis—high-level concepts are linearly encoded in LLM hidden layer spaces—examine whether the linear directions for validity and plausibility are highly similar.

**Core Idea**: The root cause of content effects in LLMs is the entangled alignment of validity and plausibility directions in representational geometry, which can be decoupled by constructing debiasing steering vectors.

## Method

### Overall Architecture
Evaluate 10 LLMs (Qwen-2.5, Qwen-3, Gemma-3 series) on 1,280 syllogisms, extract linear directions for validity and plausibility using the difference-in-means method, analyze their similarity, conduct cross-task steering experiments to verify causal relationships, and finally construct debiasing vectors to reduce content effects.

### Key Designs

1. **Concept Direction Extraction (Difference-in-Means Method)**:

    - Function: Represent binary concepts as single directions in hidden layer space
    - Mechanism: For each layer $l$, compute the difference between mean activation vectors at the last token position for samples predicted as positive class (e.g., "valid") and negative class (e.g., "invalid"): $v_{\text{concept}}^l = \mu_{\text{positive}}^l - \mu_{\text{negative}}^l$. Uses model's own predicted labels rather than true labels, as the focus is on encoding the model's own "beliefs".
    - Design Motivation: The difference-in-means method is simple and effective, directly corresponding to the linear representation hypothesis

2. **Cross-Task Steering Experiments**:

    - Function: Verify causal interaction between validity and plausibility—whether plausibility vectors can influence validity judgments, and vice versa
    - Mechanism: Apply steering vectors $v_{\text{plausibility}}^l$ extracted from plausibility tasks to logical validity classification tasks (and in reverse), measuring steering strength (label flip proportion). Steering always opposes the model's original prediction: adds the vector when predicting negative class, subtracts when predicting positive class.
    - Design Motivation: If plausibility vectors can effectively change validity judgments, it indicates causal entanglement rather than mere correlation between these concepts in representation space

3. **Debiasing Steering Vector Construction**:

    - Function: Decouple validity and plausibility representations to reduce content effects
    - Mechanism: Construct debiasing vectors to make the model evaluate logical validity without being influenced by plausibility. Apply debiasing vectors at effective layers (steering strength $>0.75$) to decrease content effect metric CE while improving reasoning accuracy.
    - Design Motivation: If entanglement is the root cause of content effects, decoupling should simultaneously reduce bias and improve reasoning capabilities

### Metric Design
Content Effect CE = $\frac{1}{2}(\Delta_{v^+} + \Delta_{v^-})$, where $\Delta_{v^+}$ measures the accuracy advantage of valid arguments with plausible conclusions. CE=0 indicates validity is independent of plausibility, CE=1 indicates complete plausibility-driven behavior.

## Key Experimental Results

### Main Results
Behavioral content effects:

| Model | Config | $D_{v^+,p^+}$ Accuracy | $D_{v^-,p^+}$ Accuracy | $D_{v^+,p^-}$ Accuracy | CE |
|------|------|---------------------|---------------------|---------------------|-----|
| Qwen2.5-32B | 0-shot | 100.00 | 67.50 | 60.92 | 0.348 |
| Qwen2.5-32B | CoT | 98.67 | 86.64 | 93.10 | 0.096 |
| Qwen3-14B | 0-shot | 97.33 | 90.83 | 60.92 | 0.213 |
| Qwen3-14B | CoT | 95.31 | 99.10 | 92.50 | 0.014 |

### Representational Analysis

| Concept Pair | Average Cosine Similarity | Note |
|--------|--------------|------|
| Validity - Plausibility | 0.48-0.64 | High alignment |
| Validity - Harmlessness | 0.10-0.13 | Low similarity (control) |
| Validity - Hypernymy | -0.12 to -0.17 | Low similarity (control) |

### Key Findings
- All tested models exhibit content effects; CoT prompting significantly reduces CE (from 0.213-0.348 to 0.014-0.096)
- Cosine similarity between validity and plausibility vectors (0.48-0.64) is much higher than control concepts (0.10-0.13), confirming specific entanglement
- Cross-task steering succeeds: plausibility vectors can effectively flip validity judgments, and vice versa
- Degree of validity-plausibility alignment positively correlates with behavioral CE strength
- Debiasing vectors simultaneously reduce CE and improve reasoning accuracy, proving decoupling effectiveness
- While CoT reduces behavioral CE, representational alignment does not significantly change (p=0.625)

## Highlights & Insights
- Provides the first representational-level explanation of LLM content effects—not a behavioral "bug" but a structural issue in representational geometry. This finding is far more profound than pure behavioral studies
- The discovery that CoT reduces behavioral CE without changing representational alignment is very interesting—CoT may "bypass" rather than "solve" the entanglement problem during reasoning
- Debiasing steering vectors as an intervention demonstrate a complete closed loop from representational analysis to practical improvement

## Limitations & Future Work
- Only validated on syllogistic reasoning; content effect mechanisms for other reasoning forms (conditional reasoning, probabilistic reasoning) may differ
- Relatively small dataset size (1,280 syllogisms), though covering all 64 types with limited semantic variation
- Debiasing vector effectiveness depends on layer selection, requiring validation sets to determine optimal layers
- Future work could explore whether concepts with non-linear encoding exhibit similar entanglement phenomena

## Related Work & Insights
- **vs Lampinen et al.**: They documented LLM content effect behaviors; this paper reveals the mechanism from a representational level
- **vs Marks & Tegmark (truth directions)**: They found truth is linearly encoded; this paper further discovers entanglement between validity and plausibility directions
- **vs Arditi et al. (refusal directions)**: Similar methodology but applied to different concepts; this paper's innovation lies in analyzing interaction between two concepts rather than a single concept

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to explain LLM content effects from representational geometry perspective, with profound insights
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive with 10 models, control experiments, and causal validation, but limited to single dataset
- Writing Quality: ⭐⭐⭐⭐⭐ Clear RQ-driven structure with layer-by-layer analysis
- Value: ⭐⭐⭐⭐⭐ Important implications for understanding and improving LLM logical reasoning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] Among Us: Language of Conspiracy Theorists on Mainstream Reddit](among_us_language_of_conspiracy_theorists_on_mainstream_reddit.md)
- [\[CVPR 2026\] As Language Models Scale, Low-order Linear Depth Dynamics Emerge](../../CVPR2026/social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)

</div>

<!-- RELATED:END -->
