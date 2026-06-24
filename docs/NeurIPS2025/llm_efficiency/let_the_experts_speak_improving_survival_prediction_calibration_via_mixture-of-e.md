---
title: >-
  [Paper Note] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads
description: >-
  [NeurIPS 2025][LLM Efficiency][Survival Analysis] Three discrete-time deep Mixture-of-Experts (MoE) survival analysis architectures are proposed, among which Personalized MoE achieves superior clustering, calibration, and predictive accuracy simultaneously by allowing each expert to generate a patient-specific event distribution.
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "Survival Analysis"
  - "Mixture of Experts"
  - "Calibration"
  - "Clustering"
  - "Discrete-Time Models"
date: 2026-05-08
content_hash: c686b8b3b35d736b
---

# Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads

**Conference**: NeurIPS 2025
**arXiv**: [2511.09567](https://arxiv.org/abs/2511.09567)  
**Code**: [https://github.com/ToddMorrill/survival-moe](https://github.com/ToddMorrill/survival-moe)  
**Area**: LLM Efficiency
**Keywords**: Survival Analysis, Mixture of Experts, Calibration, Clustering, Discrete-Time Models

## TL;DR

Three discrete-time deep Mixture-of-Experts (MoE) survival analysis architectures are proposed, among which Personalized MoE achieves superior clustering, calibration, and predictive accuracy simultaneously by allowing each expert to generate a patient-specific event distribution.

## Background & Motivation

Survival analysis — predicting the time to clinical events — is critical in clinical decision support systems. Clinicians care most about three aspects: **predictive accuracy**, **calibration** (probabilities carry intuitive meaning), and **interpretability** (ability to reason by analogy to similar patients).

MoE models are particularly attractive in medical survival analysis due to their capacity to discover latent patient subgroups. However, existing MoE approaches typically exhibit a trade-off between clustering capability and key metrics such as calibration error and predictive accuracy — a consequence of the restrictive inductive bias imposed by MoE: predictions for individual patients must resemble those of the group to which they are assigned.

The core problem addressed in this paper is: **can patient population structure be discovered while simultaneously improving calibration and predictive accuracy?** Through a systematic investigation of the effect of expert expressiveness on performance, the authors find that more expressive experts — those that tailor predictions to each individual patient — outperform those relying on fixed group prototypes.

## Method

### Overall Architecture

All three architectures share the same backbone (a feedforward deep network) and differ only in the design of the final layer (the MoE head). All methods are trained using a discrete-time MTLR-style loss function that predicts monotone label sequences. Patient records (demographics, physiological data, etc.) are passed through embedding and fully connected layers to obtain a hidden state representation $\mathbf{x}$, which is then fed into the MoE head.

### Key Designs

1. **Fixed MoE**: A router $W \in \mathbb{R}^{n \times h}$ and a learnable expert matrix $M \in \mathbb{R}^{n \times m}$. Each expert learns a fixed event distribution shared across all patients. The final PMF is a weighted average of expert distributions: $\mathbf{p} = \boldsymbol{\alpha} M'$. A learnable temperature parameter $\kappa$ regulates routing sharpness. This architecture represents a class of prior work using fixed prototype distributions.

2. **Adjustable MoE**: Built upon Fixed MoE, each expert learns a prototype event distribution that is further adjusted at the patient level via a learnable time-warping function. The warping function uses a normalized mixture of two logistic CDFs, with parameters as linear functions of the patient hidden state. Smooth distribution deformation is achieved through bidirectional mappings (forward $\phi$ and inverse $\psi$) with linear interpolation. This approach enables flexible adjustment of event distributions with a small number of additional parameters.

3. **Personalized MoE**: The most expressive design. The hidden state is projected separately into a routing representation and an expert representation; the expert representation is partitioned into $n$ equal-sized chunks, each passed through an independent linear layer to produce a patient-specific event distribution for the corresponding expert. The design is parameter-efficient — the chunking mechanism potentially encourages the model to utilize distinct information per expert. The dynamic matrix $M(\mathbf{x}_e) \in \mathbb{R}^{n \times m}$ is patient-specific.

### Loss & Training

- Discrete-time MTLR loss is used (predicting monotone label sequences under the "event remains once it occurs" assumption)
- This loss function has been shown to have favorable calibration properties
- All neural models are controlled for parameter count to eliminate capacity differences
- All measurements are averaged over 5 random seeds
- Differences relative to the MTLR baseline are reported for model ranking

## Key Experimental Results

### Main Results

| Dataset | Model | ECE↓ | Concordance↑ | Brier(50th)↓ |
|--------|------|------|-------------|--------------|
| SUPPORT2 | CoxPH | 0.187 | 78.89 | 0.209 |
| SUPPORT2 | RSF | 0.187 | 79.76 | 0.203 |
| SUPPORT2 | MTLR | 0.057 | 79.91 | 0.149 |
| SUPPORT2 | Fixed MoE | 0.054 | 79.78 | 0.147 |
| SUPPORT2 | Adjustable MoE | 0.048 | 79.83 | 0.145 |
| SUPPORT2 | **Personalized MoE** | **0.048** | **80.84** | **0.142** |
| Sepsis | MTLR | 0.017 | 88.36 | 0.033 |
| Sepsis | **Personalized MoE** | **0.005** | **89.77** | **0.030** |

### Ablation Study / Hyperparameter Sensitivity

| Configuration | Key Metric | Notes |
|------|---------|------|
| Fixed MoE (varying expert count) | Loss highly sensitive to expert count | Performance degrades sharply with too few experts |
| Adjustable MoE (varying expert count) | Moderately sensitive | Adjustment can compensate for insufficient expert count |
| Personalized MoE (varying expert count) | **Least sensitive** | Distributions can be customized regardless of expert count |
| Survival MNIST (10 clearly defined groups) | Fixed MoE best | Ideal scenario with perfect group alignment |
| Real data (ambiguous groupings) | Personalized MoE best | Expressiveness matters more in realistic settings |

### Key Findings

- Personalized MoE consistently outperforms all methods on both real-world datasets, including strong baselines such as CoxPH, RSF, and MTLR
- On the Sepsis dataset, Personalized MoE reduces ECE from 0.017 to 0.005 and improves concordance from 88.36 to 89.77
- Expert expressiveness forms a continuum — Fixed → Adjustable → Personalized — with decreasing sensitivity to hyperparameters
- Clustering analysis reveals clinically meaningful patient subgroups (e.g., stratification by risk, age, and diagnosis in SUPPORT2)
- An Adjusted Rand Index of 0.36 indicates moderately stable routing behavior across random seeds

## Highlights & Insights

- **Core Insight**: In MoE-based survival analysis, expert expressiveness is the key differentiating factor for performance. The transition from fixed prototypes to personalized generation enables improvements in prediction and calibration without sacrificing clustering capability.
- Survival MNIST serves as an illuminating counterexample — when data genuinely exhibits well-defined groupings, fixed experts are optimal, highlighting that method selection should be guided by data characteristics.
- The parameter-efficient design of Personalized MoE (via chunked sharing) enables strong performance even on small datasets.
- Standard end-to-end training is used rather than complex variational inference or EM algorithms, lowering the barrier to adoption.

## Limitations & Future Work

- Comparisons with a broader range of model families — such as DeepHit and continuous-time parametric mixtures — are not included.
- Routing stability, reflected by an ARI of 0.36, leaves room for improvement.
- The clinical interpretability of the discovered clusters requires more systematic validation.
- Validation is conducted on only three datasets; generalizability to additional disease domains and larger-scale data remains to be confirmed.
- Combinations with post-hoc calibration methods (e.g., conformal prediction) are not explored.

## Related Work & Insights

- Deep mixture survival models (DSM, SurvivalQuilts) are important predecessors, though they typically require more complex training pipelines.
- The approach of the Conditional Transformation Model (CTM) — preserving structure while enhancing expressiveness — is conceptually aligned with the present work.
- This work offers insights into understanding "when grouping vs. personalization is needed" in other domains, such as recommender systems and personalized treatment.
- The favorable calibration properties of discrete-time MTLR serve as an important foundation for the entire method.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[NeurIPS 2025\] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [\[ACL 2025\] GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](../../ACL2025/llm_efficiency/gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](../../ICML2026/llm_efficiency/hyperparameter_transfer_with_mixture-of-expert_layers.md)

</div>

<!-- RELATED:END -->
