---
title: >-
  [Paper Note] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis
description: >-
  [NeurIPS 2025][Interpretability][Intrinsic Motivation] This paper proposes a systematic post-hoc interpretability framework to analyze how intrinsic motivation (based on Random Network Distillation) shapes the geometric…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Intrinsic Motivation"
  - "Decision Transformer"
  - "Embedding Analysis"
  - "Representation Learning"
date: 2026-05-08
content_hash: b3e5b06d90cd203b
---

# How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2506.13958](https://arxiv.org/abs/2506.13958)
**Code**: Not available
**Area**: Interpretability
**Keywords**: Intrinsic Motivation, Decision Transformer, Interpretability, Embedding Analysis, Representation Learning

## TL;DR

This paper proposes a systematic post-hoc interpretability framework to analyze how intrinsic motivation (based on Random Network Distillation) shapes the geometric structure of the embedding space in Elastic Decision Transformers. The analysis reveals that different intrinsic motivation variants create fundamentally distinct representational structures—EDT-SIL promotes compact representations while EDT-TIL enhances orthogonality—and that embedding properties exhibit strong environment-specific correlations with task performance.

## Background & Motivation

The Elastic Decision Transformer (EDT) combines the sequence modeling capability of Transformers with decision-making, achieving efficient offline RL through dynamic history length adjustment. Integrating intrinsic motivation (e.g., curiosity-driven RND) into EDT has been empirically shown to improve performance, yet **the representational mechanisms underlying these performance gains remain unclear**.

Core problem: Does intrinsic motivation function solely as an exploration reward, or does it more fundamentally **reshape the model's internal representational structure**? Understanding this mechanism is critical for interpretable RL, as these models learn implicit state representations in high-dimensional spaces that lack the interpretability of traditional hand-crafted features.

The paper's core hypothesis is that **intrinsic motivation is not merely an exploration reward bonus, but rather a representational prior that shapes the geometry of the embedding space in a manner analogous to biological neural systems**.

## Method

### Overall Architecture

Building on the EDT architecture, the paper compares three model variants—baseline EDT, EDT-SIL (State Input Loss), and EDT-TIL (Transformer Input Loss)—and employs statistical analysis of the geometric properties of the embedding space to understand the mechanistic effects of intrinsic motivation.

### Key Designs

1. **Two intrinsic motivation variants**:

    - **EDT-SIL**: The intrinsic loss is applied directly to the state representations at the embedding layer, $L_{\text{int}} = |f_{\text{pred}}(x_{\text{embed}}; \theta_{\text{pred}}) - f_{\text{target}}(x_{\text{embed}}; \theta_{\text{target}})|_2^2$. The RND module receives the embedded states, so the intrinsic signal directly influences the learning of the state embedding layer, tending to promote more compact representations.
    - **EDT-TIL**: The intrinsic loss is applied to the Transformer output representations, causing the intrinsic signal to jointly influence both the embedding and Transformer layers, thereby shaping more coherent sequential representations and tending to enhance orthogonality.
    - Total loss: $L_{\text{overall}} = L_{\text{EDT}} + L_{\text{int}}$

2. **Embedding analysis framework (three key metrics)**:

    - **Covariance Trace**: $\text{cov\_trace} = \text{Tr}(\text{Cov}(E))$, measuring the total variance distribution across embedding dimensions, reflecting the total information captured by the representation space.
    - **L2 Norm**: $\text{l2\_norm} = \frac{1}{N}\sum_{i=1}^{N}|e_i|_2$, quantifying the average magnitude of embedding vectors and reflecting representational compactness.
    - **Cosine Similarity**: Evaluating the average pairwise cosine similarity among embeddings, reflecting representational orthogonality. Lower cosine similarity indicates that state representations are more dispersed and more easily distinguishable.

3. **Quantitative correlation analysis**: Pearson correlation coefficients are computed between embedding metrics and normalized performance scores to identify the most predictive metric for each environment–model combination.

### Loss & Training

- D4RL benchmark Medium and Medium-Replay datasets are used.
- Four continuous control tasks are evaluated: Ant, HalfCheetah, Hopper, and Walker2d.
- Each configuration uses 5 random seeds; embedding analysis is averaged over 3 repetitions.
- The RND module employs a 3-layer prediction network, determined to be the optimal configuration through systematic hyperparameter tuning.

## Key Experimental Results

### Main Results

Human-Normalized Scores (HNS) on the D4RL Medium dataset:

| Model | Ant | HalfCheetah | Hopper | Walker2d |
|-------|-----|-------------|--------|----------|
| EDT (Baseline) | 88.84±3.61 | **42.30**±0.14 | 57.49±3.81 | 68.50±2.03 |
| EDT-SIL | **90.49**±5.01 | 42.46±0.12 | 59.31±6.16 | 69.44±4.46 |
| EDT-TIL | 89.01±5.83 | 42.18±0.34 | **59.63**±2.35 | **73.50**±4.29 |

Medium-Replay dataset:

| Model | Ant | HalfCheetah | Hopper | Walker2d |
|-------|-----|-------------|--------|----------|
| EDT | **85.51**±5.06 | 37.32±2.46 | 81.56±9.96 | 62.25±5.21 |
| EDT-SIL | 84.02±3.72 | 37.64±2.44 | **84.67**±4.80 | 57.21±8.54 |
| EDT-TIL | 83.72±4.13 | **38.60**±1.28 | 81.72±9.27 | **65.06**±3.81 |

### Ablation Study

Environment-specific correlation analysis between embedding properties and performance:

| Environment | Strongest Metric | Correlation | EDT-SIL Effect | EDT-TIL Effect |
|-------------|-----------------|-------------|----------------|----------------|
| Ant | Covariance Trace | r = -0.907 | Reduced trace (526→620 vs. baseline) | Moderate reduction (573) |
| HalfCheetah | Covariance Trace | r = +0.850 | Slight increase (632) | Reduction (563) |
| Hopper | Cosine Similarity | r = +0.658 | Increase (0.082) | **Largest increase (0.117)** |
| Walker2d | Cosine Similarity | r = -0.950 | Increase (0.083) | **Reduction (0.073)** |

### Key Findings

- **EDT-SIL consistently creates more compact representations**: by reducing covariance trace and L2 norm, it compresses information at the input layer.
- **EDT-TIL promotes representational orthogonality**: by varying cosine similarity to optimize state distinguishability, with particularly pronounced effects in Walker2d (cosine similarity decreasing from 0.081 to 0.073, r = -0.950).
- **Environment specificity**: performance in different environments is driven by different embedding properties—Ant relies on variance control, Walker2d on orthogonality—suggesting that intrinsic motivation creates customized representational structures aligned with task demands.
- **3-layer RND is optimal**: too few layers (1 layer) lack capacity, while too many (10 layers) lead to overfitting or representational instability.

## Highlights & Insights

- This is the first work to explain the mechanistic role of intrinsic motivation from a representational geometry perspective, moving beyond the simplistic view of "exploration reward."
- An interesting biological correspondence is revealed: the complementary mechanisms of EDT-SIL and EDT-TIL resemble the hierarchical organizational principle in biological neural systems, whereby different processing stages maintain distinct homeostatic mechanisms.
- The view of intrinsic motivation as a "representational prior" offers a new direction for designing better auxiliary loss functions.

## Limitations & Future Work

- HNS improvements are modest (on the order of a few percentage points), and statistical significance warrants further verification.
- The analysis framework focuses solely on the geometric properties of the embedding space, without employing explicit interpretability methods (e.g., SHAP, attention visualization).
- Only the state embedding layer is analyzed; the framework is not extended to Transformer outputs or action representations.
- The temporal evolution of embedding structure during training is not explored, lacking a dynamic perspective.
- Tasks are limited to D4RL continuous control settings; discrete action spaces and multimodal observation environments are not considered.

## Related Work & Insights

- **Elastic Decision Transformer (EDT)**: achieves flexible offline RL policy learning through dynamic history length adjustment.
- **Random Network Distillation (RND)**: a classical intrinsic motivation method that measures novelty via prediction error.
- **Allostatic Regulation**: the concept of predictive adaptation in biology; the intrinsic motivation loss is analogous to an allostatic regulator.
- **Insights**: Auxiliary loss functions influence learning not only through gradient signals, but also by implicitly constraining the geometric structure of the representation space—this provides a new perspective for designing better self-supervised or auxiliary tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Analyzing intrinsic motivation from a representational geometry perspective is a novel starting point.
- **Experimental Thoroughness**: ⭐⭐⭐ Environments and datasets are limited; performance gains are small in magnitude.
- **Writing Quality**: ⭐⭐⭐⭐ The analytical framework is clear, and the biological analogy is thought-provoking.
- **Value**: ⭐⭐⭐⭐ Provides analytical tools and insights for understanding the representational-level effects of auxiliary loss functions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](empowering_decision_trees_via_shape_function_branching.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)

</div>

<!-- RELATED:END -->
