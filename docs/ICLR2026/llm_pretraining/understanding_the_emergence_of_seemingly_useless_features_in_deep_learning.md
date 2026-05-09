---
title: >-
  [Paper Note] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors
description: >-
  [ICLR2026][LLM Pretraining][Feature emergence] This paper explains, from the perspective of gradient signals, why Transformers trained with next-token prediction (NTP) learn features that appear "useless" for predicting the immediate next token. It proposes a decomposition of gradient pathways into three components — direct learning, pre-caching, and circuit sharing — and validates this framework on toy tasks, OthelloGPT, and language models.
tags:
  - ICLR2026
  - LLM Pretraining
  - Feature emergence
  - next-token prediction
  - pre-caching
  - circuit sharing
  - world models
  - mechanistic interpretability
date: 2026-05-08
content_hash: 4c5dc1f6078fd550
---

# Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors

**Conference**: ICLR2026
**arXiv**: [2603.14087](https://arxiv.org/abs/2603.14087)
**Code**: [GitHub](https://github.com/Markfryazino/useless-features-iclr-code)
**Area**: LLM Pre-training
**Keywords**: Feature emergence, next-token prediction, pre-caching, circuit sharing, world models, mechanistic interpretability

## TL;DR

This paper explains, from the perspective of gradient signals, why Transformers trained with next-token prediction (NTP) learn features that appear "useless" for predicting the immediate next token. It proposes a decomposition of gradient pathways into three components — direct learning, pre-caching, and circuit sharing — and validates this framework on toy tasks, OthelloGPT, and language models.

## Background & Motivation

- LLMs are typically trained with the NTP objective: learning $p(x_{t+1}|x_1 \cdots x_t)$
- Intuitively, models need only learn features useful for predicting the next token
- Yet extensive research has shown that Transformers acquire representations far richer than this: abstract features, world models, multi-step lookahead, etc.
- OthelloGPT even learns board-state representations, despite the fact that predicting legal moves does not require knowing all piece positions
- **Core Problem**: What gradient signals arising from the NTP objective drive the learning of these "useless" features?

## Core Problem

How do Transformers trained for NTP learn features that are useless for predicting the current next token? Which components of the gradient signal drive the emergence of such features?

## Method

### Three-Way Gradient Pathway Decomposition

For the residual stream $r_{\theta,i}^k(x)$ at position $i$ and layer $k$, the loss gradient can be decomposed into three independent pathways:

**1. Direct Learning** — green pathway

Gradient from the next-token prediction loss at position $i+1$, passing through $r_{\theta,i}^k(x)$:

$$\nabla_\theta L_{i(\text{direct})}^k = \nabla_\theta L_i - \nabla_\theta L_i^{\text{sg}(k,i)}$$

**2. Pre-caching** — blue pathway

Gradient from prediction losses at positions $j > i+1$, which "look back" to position $i$ via the attention mechanism:

$$\nabla_\theta L_{i(\text{pre-cached})}^k = \nabla_\theta \sum_{j \neq i} \left[L_j - L_j^{\text{sg}(k,i)}\right]$$

**3. Circuit Sharing** — orange pathway

Pathways that do not pass through $r_{\theta,i}^k(x)$. Because Transformers share parameters, gradient signals from other positions also affect parameters involved in computing position $i$:

$$\nabla_\theta L_{i(\text{shared})}^k = \sum_j \nabla_\theta L_j^{\text{sg}(k,i)}$$

**Proposition 3.1**: The three components constitute an exact decomposition of the total gradient:
$$\nabla_\theta L = \nabla_\theta L_{i(\text{direct})}^k + \nabla_\theta L_{i(\text{pre-cached})}^k + \nabla_\theta L_{i(\text{shared})}^k$$

### Ablation Methods

- **Ablating pre-caching**: myopic training, which blocks gradient flow between different positions
- **Ablating circuit sharing**: $m$-untied training, which uses separate parameters before and after position $m$
- **Extreme case**: myopic + untied = "split brain," isolating all three pathways

### Influence Metric

**Feature Mismatch**:
$$R(x|\theta_1, \theta_2, w_i^k) = \frac{1}{2}(\langle w_i^k, r_{\theta_1,i}^k(x)\rangle - \langle w_i^k, r_{\theta_2,i}^k(x)\rangle)^2$$

**Influence**:
$$I_i^k(\theta, x | w_i^k, \theta^*, G) = \frac{d}{d\varepsilon} R(x|\theta + \varepsilon G, \theta^*, w_i^k)\bigg|_{\varepsilon=0}$$

Substituting each of the three gradient components for $G$ yields the integrated influence of each pathway.

### Adaptation to Large Models: Proposition 5.1

For large models where retraining is infeasible, the ratio of influences is approximated via activation interventions:

$$Q(w) = \frac{\sum_{j>i+1} d_j^{/i}}{d_{i+1}^{/i}} \approx \frac{I_{\text{pre-cached}}}{I_{\text{direct}}}$$

A high $Q(w)$ indicates that the feature is primarily driven by pre-caching.

## Key Experimental Results

### Toy Tasks: Majority and Conditioned Majority

| Training Setup | Majority Feature Probe | Conditioned Majority Feature Probe |
|---|---|---|
| Standard training | High | High |
| Myopic (no pre-caching) | Reduced | Cannot be learned |
| Untied (no circuit sharing) | Reduced | High |
| Myopic + Untied | **Lowest** | **Cannot be learned** |

- The Conditioned Majority task requires pre-caching to be learned (requires two-layer attention interaction)
- Ablating both mechanisms largely eliminates the emergence of NTP-useless features

### OthelloGPT

| Gradient Component | NTP-Useful Features 95% CI | NTP-Useless Features 95% CI |
|---|---|---|
| Direct | [2.85, 12.38] | [-4.69, 2.74] |
| Pre-cached | [-1.99, 0.66] | [0.55, 3.05] |
| Shared | [4.80, 12.48] | [2.93, 9.91] |
| Combined | [12.14, 19.05] | [4.42, 10.07] |

- Direct influence is significantly positive only for NTP-useful features; it is not significant for NTP-useless features
- Pre-caching and circuit sharing influences remain positive for NTP-useless features, explaining why they are nonetheless learned
- This provides an explanation for the world model fragility reported by Vafa et al. (2025)

### Language Model (TinyStories)

| Feature Type | Requires Pre-caching | Pre-caching Influence |
|---|---|---|
| POS tags | No | Low |
| Dependency labels | No | Low |
| Positional features (story position) | **Yes** | **High** |

- Myopic model loss: $3.29 \pm 0.02$; standard model loss: $2.53 \pm 0.10$ (large gap)
- Simple syntactic features do not require pre-caching, but coherent text generation does

### Gemma 2 2B: SAE Feature Analysis

- Features with extremely high or low $Q(w)$ are predominantly associated with **programming and formal reasoning**
- $\sigma_{\text{formal}} = 1.63 \pm 0.03$ vs. $\sigma_{\text{not formal}} = 1.23 \pm 0.02$
- Steering via high-$Q(w)$ features generates more code and punctuation
- Pre-caching features are **negatively correlated** with lookahead, supporting the "breadcrumb hypothesis"

## Highlights & Insights

1. **Understanding features from a developmental perspective**: Unlike traditional functional interpretability, this work explains feature emergence from the developmental angle of gradient signals
2. **Elegance of the three-pathway decomposition**: The total gradient is decomposed exactly into three components with clear physical interpretations
3. **A new explanation for OthelloGPT world models**: Representations of NTP-useless board squares arise from pre-caching and circuit sharing, not direct learning
4. **Refuting the pre-caching = lookahead hypothesis**: Empirical results show that pre-caching features are negatively correlated with lookahead ability
5. **Cross-scale validation**: Consistent findings from toy tasks to Gemma 2 2B

## Limitations & Future Work

- The attribution method relies on retraining models, which is infeasible for large models; Proposition 5.1 is only an approximation
- Feature definitions are limited to linear probe directions; nonlinear features are not covered
- Large-model experiments are conducted only on Gemma 2; validation on additional LLMs remains to be done
- Myopic training simultaneously blocks forward information flow and backward gradient flow, making it difficult to fully disentangle causal relationships
- The influence of post-training stages such as RLHF on feature emergence is not explored

## Related Work & Insights

| Direction | Contribution of This Paper |
|---|---|
| Li et al. (2023) OthelloGPT | Explains why NTP-useless features can still be learned |
| Vafa et al. (2025) world model fragility | Uses gradient components to explain the gap between NTP-useful and NTP-useless features |
| Wu et al. (2024) pre-caching hypothesis | Introduces circuit sharing and refutes the pre-caching = lookahead equation |
| Bachmann & Nagarajan (2024) NTP limitations | Complements prior analysis by identifying sources of NTP's positive capabilities |

### Further Connections

- "Developmental interpretability" emerges as a new paradigm for understanding LLMs
- The pre-caching mechanism is connected to chain-of-thought reasoning: models may prepare for later inference at early positions
- Circuit sharing explains why parameter-sharing Transformers are more powerful than position-independent models
- The $Q(w)$ metric serves as a practical tool for identifying "planning-type" features among SAE features

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — An entirely fresh perspective on feature emergence via gradient signal analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ — A complete chain from toy tasks to large models, though large-model experiments remain limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic, rigorous mathematics, and well-crafted figures
- Value: ⭐⭐⭐⭐⭐ — Provides a novel analytical framework for understanding internal representations in LLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)
- [\[ICLR 2026\] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[ICLR 2026\] Block-Sample MAC-Bayes Generalization Bounds](block-sample_mac-bayes_generalization_bounds.md)

</div>

<!-- RELATED:END -->
