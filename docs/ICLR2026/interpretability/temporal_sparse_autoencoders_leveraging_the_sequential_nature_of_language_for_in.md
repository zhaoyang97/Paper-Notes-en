---
title: >-
  [Paper Note] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
description: >-
  [ICLR 2026][Interpretability][Sparse Autoencoders] This paper proposes Temporal SAEs (T-SAEs), which introduce a temporal contrastive loss to encourage high-level features to maintain consistent activations across adjace…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Temporal Consistency"
  - "Semantic Disentanglement"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: bada9dd11c3c8540
---

# Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability

**Conference**: ICLR 2026 Oral  
**arXiv**: [2511.05541](https://arxiv.org/abs/2511.05541)  
**Code**: [github.com/AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)  
**Area**: Model Compression / Interpretability
**Keywords**: Sparse Autoencoders, Temporal Consistency, Semantic Disentanglement, Contrastive Learning, Interpretability

## TL;DR

This paper proposes Temporal SAEs (T-SAEs), which introduce a temporal contrastive loss to encourage high-level features to maintain consistent activations across adjacent tokens. Through self-supervised training without explicit semantic supervision, T-SAEs achieve disentanglement of semantic and syntactic features, recovering smoother and more coherent semantic concepts without sacrificing reconstruction quality.

## Background & Motivation

- Features recovered by existing SAEs on LLMs tend to be **token-level, local, and unstable** syntactic patterns (e.g., "sentence-initial *The*", "sentence-final period")
- Root cause: SAEs treat tokens as independent samples, **ignoring the sequential structure of language**
- Key properties of human language:
    - **Semantic** (high-level) information changes smoothly over time (e.g., a discussion about "plant biology")
    - **Syntactic** (low-level) information changes rapidly at specific tokens (e.g., "capitalized first letter", "plural noun")
- A method is needed to enable SAEs to exploit this temporal structure to discover more meaningful high-level semantic features

## Method

### Data Generation Process Model

Language generation is modeled as: $\tau_t = \phi(\tau^{t-1}, \mathbf{h}_t, \mathbf{l}_t)$

- $\mathbf{h}_t$: high-level variables (semantics, intent) — temporally invariant
- $\mathbf{l}_t$: low-level variables (syntax, lexical choice) — vary across tokens

### Core Assumptions

1. **Temporal Consistency** (Assumption 1): $\mathbf{h}_t \approx \mathbf{h}_{t'}$ within the same sequence
2. **Hierarchical Representation** (Assumption 2): $\mathbf{h}_t$ can reconstruct $\mathbf{x}_t$ to $\epsilon$ precision independently; $\mathbf{l}_t$ supplements the residual

### T-SAE Architecture

The SAE feature space is partitioned into high-level (first $h$) and low-level (remaining $m-h$) features. A Matryoshka loss is employed:

$$\mathcal{L}_{\text{matr}}(\mathbf{x}_t) = \underbrace{\|\mathbf{x}_t - \mathbf{W}_{0:h}^{\text{dec}} \mathbf{f}_{0:h}(\mathbf{x}_t) + \mathbf{b}^{\text{dec}}\|_2^2}_{\mathcal{L}_H} + \underbrace{\|\mathbf{x}_t - \mathbf{W}^{\text{dec}} \mathbf{f}(\mathbf{x}_t) + \mathbf{b}^{\text{dec}}\|_2^2}_{\mathcal{L}_L}$$

### Temporal Contrastive Loss

High-level features are encouraged to be consistent across adjacent tokens within a sequence, and inconsistent across different sequences:

$$\mathcal{L}_{\text{contr}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s(\mathbf{z}_t^{(i)}, \mathbf{z}_{t-1}^{(i)}))}{\sum_j \exp(s(\mathbf{z}_t^{(i)}, \mathbf{z}_{t-1}^{(j)}))} - \frac{1}{N}\sum_{j=1}^N \log \frac{\exp(s(\mathbf{z}_t^{(j)}, \mathbf{z}_{t-1}^{(j)}))}{\sum_i \exp(s(\mathbf{z}_t^{(i)}, \mathbf{z}_{t-1}^{(j)}))}$$

Total loss: $\mathcal{L} = \sum_i \mathcal{L}_{\text{matr}}(\mathbf{x}_t^{(i)}) + \alpha \mathcal{L}_{\text{contr}}$

### Key Design Choices

- The contrastive loss is applied exclusively to **high-level features**
- Low-level features naturally capture fluctuating syntactic signals by fitting the residual
- No explicit semantic labels are required — the method is fully self-supervised

## Key Experimental Results

### Core Performance Metrics

|  | FVE ↑ | Cos Sim ↑ | Frac Alive ↑ | Smoothness (High/Low) | Autointerp ↑ |
|--|-------|----------|-------------|----------------------|-------------|
| **T-SAE** (Pythia-160m) | 0.94 | 0.93 | 0.87 | **0.09** / 0.17 | 0.81 |
| Matryoshka SAE | 0.95 | 0.94 | 0.89 | 0.12 / 0.13 | 0.83 |
| BatchTopK SAE | 0.95 | 0.94 | 0.84 | 0.13 / — | 0.85 |
| **T-SAE** (Gemma2-2b) | 0.75 | 0.88 | 0.78 | **0.10** / 0.15 | 0.83 |
| Matryoshka SAE | 0.75 | 0.89 | 0.76 | 0.15 / 0.12 | 0.83 |

### Semantic / Contextual / Syntactic Probing Accuracy (MMLU)

| Probing Task | T-SAE High | T-SAE Low | Matryoshka | BatchTopK |
|---------|-----------|-----------|-----------|----------|
| Semantic ($k$=5) | **Best** | Low | Mid | Mid |
| Contextual ($k$=5) | **Best** | Low | Mid | Mid |
| Syntactic ($k$=5) | Mid | **Best** | High | High |

### Ablation Study

| Variant | FVE | Frac Alive | Smoothness (High) | Semantic | Contextual | Syntactic |
|------|-----|-----------|-----------------|------|-------|------|
| Random contrast (non $t$-1) | 0.0 | −0.05 | 0.0 | −0.02 | +0.11 | −0.10 |
| 50:50 partition | −0.01 | +0.01 | 0.0 | +0.02 | +0.09 | — |
| Naïve similarity loss | Better reconstruction | — | — | Worse semantic | Worse contextual | — |

### Steering Experiments

T-SAE high-level features **Pareto-dominate** all baseline SAEs on steering tasks:
- Higher steering success rate + higher output coherence
- Baselines exhibit token repetition failures under high-intensity steering; T-SAE does not

### Key Findings

1. T-SAE high-level features are significantly smoother (0.09 vs. 0.12–0.15), exhibiting clear semantic phase transitions across sequences
2. **Explicit disentanglement**: high-level features capture semantics/context; low-level features capture syntax — this separation is absent in Matryoshka SAE
3. Reconstruction quality is largely unaffected (FVE: 0.94 vs. 0.95)
4. Analysis of the HH-RLHF dataset using T-SAE reveals **spurious correlations** in annotations (rejected responses tend to be longer and more formal)
5. High-level feature steering is substantially more effective and stable than that of existing SAEs

## Highlights & Insights

- **Linguistically motivated design**: the distinction between smoothly varying semantics and locally varying syntax is grounded in classical linguistics
- **Semantic structure from pure self-supervision**: clear semantic clusters emerge without any semantic labels
- **Unlocking sequence-level interpretability**: existing SAEs operate at the token level; T-SAEs enable semantic tracking at the sequence level for the first time
- **Practical discovery**: spurious length correlations in the HH-RLHF dataset serve as a warning regarding safety alignment data quality
- **Fundamental steering advantage**: high-level features modify semantic encoding rather than simply increasing the frequency of specific tokens

## Limitations & Future Work

- The high-level/low-level partition ratio (default 20:80) requires manual specification
- Contrastive learning uses only adjacent tokens; capturing longer-range dependencies requires additional design (ablations show that random time-step contrast yields different characteristics)
- Training cost is slightly higher than baseline SAEs
- Validation is limited to Pythia-160m and Gemma2-2b; experiments on larger models are needed

## Related Work & Insights

- Sparse Autoencoders: Bricken et al. 2023, Matryoshka SAE, BatchTopK SAE
- Temporal Representation Learning: CPC (Contrastive Predictive Coding), Slow Feature Analysis
- Semantic–Syntactic Disentanglement: LDA (topic models), Griffiths et al. 2004 (HMM+LDA)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Applying temporal consistency priors within SAEs is original and elegant
- **Technical Depth**: ⭐⭐⭐⭐ — The data generation model and contrastive loss design are clearly formulated
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Probing, visualization, steering, safety case study, and ablations are comprehensive
- **Practical Value**: ⭐⭐⭐⭐⭐ — Unlocks sequence-level interpretability and more effective steering capabilities

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)
- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](../../ICML2026/interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](../../ICML2026/interpretability/on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](../../ACL2026/interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)

</div>

<!-- RELATED:END -->
