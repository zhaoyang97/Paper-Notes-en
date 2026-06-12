---
title: >-
  [Paper Note] Action-Guided Attention for Video Action Anticipation
description: >-
  [ICLR 2026][Causal Inference][Action Anticipation] This paper proposes an Action-Guided Attention (AGA) mechanism that uses the model's own action prediction sequences as the Query and Key in attention (rather than pixel…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "Action Anticipation"
  - "Attention Mechanism"
  - "Video Transformer"
  - "Interpretability"
  - "EPIC-Kitchens"
date: 2026-05-08
content_hash: 83ea1a837c3e00c5
---

# Action-Guided Attention for Video Action Anticipation

**Conference**: ICLR 2026
**arXiv**: [2603.01743](https://arxiv.org/abs/2603.01743)  
**Code**: None  
**Area**: Causal Inference
**Keywords**: Action Anticipation, Attention Mechanism, Video Transformer, Interpretability, EPIC-Kitchens

## TL;DR
This paper proposes an Action-Guided Attention (AGA) mechanism that uses the model's own action prediction sequences as the Query and Key in attention (rather than pixel-level features), combined with adaptive gated fusion of historical context and current frame features. AGA achieves strong generalization from validation to test set on EPIC-Kitchens-100 and supports post-hoc interpretability analysis.

## Background & Motivation

**Background**: Video action anticipation—predicting future actions from current frames—is an important task in computer vision, with Transformer architectures having become the dominant paradigm.

**Limitations of Prior Work**: Standard self-attention relies on dot products over pixel-level features, which lack the high-level semantics required for modeling future actions. This causes models to overfit to explicit visual cues in past frames rather than capturing latent intentions, resulting in significant performance degradation from validation to test sets.

**Key Challenge**: Action anticipation is inherently non-deterministic—identical past observations may lead to multiple future outcomes. Pixel-level attention is easily misled by visual noise and fails to model semantic dependencies between actions.

**Goal**: To design an attention mechanism that leverages high-level action semantics rather than low-level pixel features to guide sequence modeling.

**Key Insight**: Using action prediction probabilities (rather than feature vectors) as Q/K, exploiting semantic correlations between actions to select relevant historical moments, followed by gated fusion with the current frame.

**Core Idea**: Using the model's own action prediction sequence to guide attention, focusing it on "semantically relevant past moments" rather than "visually similar past frames."

## Method

### Overall Architecture
After a backbone extracts features from input video frames, the AGA module performs multi-head attention using the EMA of action predictions as the Query, the most recent $S$ action predictions as Keys, and corresponding frame features as Values, producing a historical context $\tilde{h}_t$. This is then fused with the current frame feature $e_t$ via adaptive gating to yield the final action prediction.

### Key Designs

1. **Action-Guided Query/Key**:

    - **Function**: Constructs attention Queries and Keys from predicted action probability distributions rather than visual features.
    - **Mechanism**: $K_t = E_K(\hat{y}_{t-S:t-1})$, $Q_t = E_Q(\bar{y}_t)$, where $\bar{y}_t = \alpha \hat{y}_{t-1} + (1-\alpha)\bar{y}_{t-1}$ is the EMA of action predictions. Values remain frame-level visual features $V_t = E_V(e_{t-S:t-1})$.
    - **Design Motivation**: In dot-product attention, Q/K correlations assign weights to V. When Q/K are action predictions, the attention weights reflect "which past actions are most relevant to the currently anticipated action" rather than "which past frames are most visually similar to the current frame."

2. **Adaptive Gated Fusion**:

    - **Function**: Element-wise gated fusion of historical attention output $\tilde{h}_t$ and current frame feature $e_t$.
    - **Mechanism**: $o_t = g_t \odot \tilde{h}_t + (1-g_t) \odot e_t$, with gate $g_t = \sigma(\text{MLP}(\tilde{h}_t \| e_t))$.
    - **Design Motivation**: The relative importance of historical context versus current visual evidence varies over time; the gating mechanism allows the model to adaptively decide how much to rely on history versus the current frame.

3. **Post-hoc Interpretability Analysis**:

    - **Function**: Reveals action dependencies and counterfactual evidence learned by the model through forward and backward analysis.
    - **Mechanism**: Forward analysis examines the distribution of attention weights given past actions (action dependency); backward analysis examines changes in predictions when past actions are modified (counterfactual reasoning).
    - **Design Motivation**: Because Q/K are action probabilities, attention weights directly reflect semantic relationships between actions, making them more interpretable than pixel-level attention.

### Loss & Training
Standard cross-entropy loss is used for future action prediction. A modular design with a frozen backbone and trainable encoders is adopted. A FIFO queue maintains the temporal window $S$.

## Key Experimental Results

### Main Results

**EPIC-Kitchens-100 (Action Anticipation):**

| Method | Val Verb | Val Noun | Val Action | Test-Val Gap |
|--------|----------|----------|------------|--------------|
| AVT | High | High | Medium | Large |
| MemViT | High | High | Medium | Large |
| **AGA** | **Competitive** | **Competitive** | **Competitive** | **Smallest** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| AGA (full) | Best generalization | Action-guided Q/K + gating |
| Standard self-attention | Overfitting | High Val, large Test drop |
| No gating (history only) | Degraded | Missing current frame information |
| No EMA (direct last-step prediction) | Slightly degraded | EMA provides more stable long-range signal |

### Key Findings
- AGA consistently shows a smaller validation-to-test performance gap than baselines, indicating stronger generalization and less overfitting.
- Robust performance is also observed on EPIC-Kitchens-55 and EGTEA Gaze+.
- Post-hoc analysis reveals meaningful action dependencies (e.g., high attention weights for "pick up → put down"), validating the semantic plausibility of action-guided attention.
- EMA coefficient $\alpha=0.8$ is optimal across most settings, with low sensitivity to this hyperparameter.

## Highlights & Insights
- **Semantic-Level Attention**: Elevating attention from pixel-level to action-probability-level is the key innovation. This abstraction shifts the model's focus from "what did the past look like" to "what was done in the past," which is better suited for action anticipation.
- **Recycling Self-Predictions**: Using the model's own predictions as input (the EMA of action distributions) combines the benefits of autoregressive and non-autoregressive approaches—capturing sequential dependencies without introducing latency.
- **Interpretability as a Byproduct**: Because Q/K are action probabilities, the attention matrix directly quantifies "how much influence action A has on predicting action B," achieving a natural interpretability for Transformers in video anticipation for the first time.

## Limitations & Future Work
- Only RGB video frames are used; multimodal signals (text, audio, optical flow) are not incorporated.
- The EMA of action predictions may be unstable at the beginning of a sequence (cold-start problem).
- The FIFO queue size $S$ is a fixed hyperparameter; an adaptive window may be preferable.
- Experiments are limited to kitchen scenarios (EPIC-Kitchens).

## Related Work & Insights
- **vs. AVT**: AVT applies standard causal attention over visual tokens; AGA uses action predictions as Q/K, avoiding pixel-level overfitting.
- **vs. MemViT**: MemViT stores longer history via token compression; AGA implicitly encodes long-range dependencies through EMA, with a lighter footprint.
- **vs. AFFT**: AFFT fuses multiple modalities but still uses standard attention; AGA achieves generalization gains within a single modality by redesigning the attention mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of using action probabilities as attention Q/K is novel and intuitively natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + ablation study + interpretability analysis.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; interpretability analysis is insightful.
- Value: ⭐⭐⭐⭐ Provides a new direction for attention design in video action anticipation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](../../ICML2026/causal_inference/density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)
- [\[ICLR 2026\] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)

</div>

<!-- RELATED:END -->
