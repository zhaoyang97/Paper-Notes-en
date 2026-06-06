---
title: >-
  [Paper Note] Decomposition of Small Transformer Models
description: >-
  [NeurIPS 2025 (Workshop: Mechanistic Interpretability)][Time Series][Parameter space decomposition] This paper extends Stochastic Parameter Decomposition (SPD) to Transformers by designing a sequence-aware causal importa…
tags:
  - "NeurIPS 2025 (Workshop: Mechanistic Interpretability)"
  - "Time Series"
  - "Parameter space decomposition"
  - "Stochastic Parameter Decomposition"
  - "Induction Head"
  - "GPT-2"
  - "Causal importance"
date: 2026-05-08
content_hash: 9e9afecf5cb35faf
---

# Decomposition of Small Transformer Models

**Conference**: NeurIPS 2025 (Workshop: Mechanistic Interpretability)
**arXiv**: [2511.08854](https://arxiv.org/abs/2511.08854)  
**Code**: None (built on the SPD open-source framework)  
**Area**: Time Series
**Keywords**: Parameter space decomposition, Stochastic Parameter Decomposition, Induction Head, GPT-2, Causal importance

## TL;DR
This paper extends Stochastic Parameter Decomposition (SPD) to Transformers by designing a sequence-aware causal importance function and a novel partial reconstruction loss. On a toy induction head task, the method recovers the expected two-step circuit; on GPT-2-small, it localizes rank-1 parameter subspaces corresponding to interpretable concepts such as "golf" and "basketball."

## Background & Motivation

**Background**: Mechanistic interpretability has proceeded in two waves — the first focused on individual neurons but was limited by polysemanticity; the second shifted to activation space, where sparse autoencoders (SAEs) have uncovered large numbers of interpretable concepts. However, SAEs suffer from feature absorption and splitting.

**Limitations of Prior Work**: Activation-space methods only answer "what is activated given an input" and cannot decompose the model itself into a small set of reusable mechanisms. Parameter-space methods are theoretically more fundamental, as gradient descent directly writes mechanisms into weights.

**Key Challenge**: SPD had previously been validated only on toy models and could not handle sequential data (Transformers), leaving a gap between toy settings and real models unaddressed.

**Goal**: Extend SPD to Transformers and verify whether parameter-space decomposition can recover known circuits and discover interpretable subcomponents.

**Key Insight**: SPD decomposes weights into sparse rank-1 matrices $W_c^l = \vec{U_c^l} \otimes \vec{V_c^l}$ and learns a causal importance function. A new causal importance formulation is designed to account for sequence position dependence.

**Core Idea**: Introduce a position-aware attention-based causal importance function and a partial reconstruction loss, enabling SPD to decompose Transformers and extract interpretable parameter-space mechanisms.

## Method

### Overall Architecture
SPD decomposes $W^l$ into $C$ rank-1 subcomponents. The assembled weight is $W'^l = \sum_{c} \alpha \cdot W_c^l$, where $\alpha \in [0,1]$ is controlled by the causal importance $g_c^l(x)$. The training objective balances faithfulness (subcomponent sum recovers original weights) and minimality (as few subcomponents activated as possible).

### Key Designs

1. **Sequence-Aware Causal Importance Function**:

    - **Function**: Assigns different causal importance to different sequence positions.
    - **Mechanism**: A minimal attention network (1 head, 1 layer) with learned relative positional encodings is prepended to the $\gamma$-MLP, enabling cross-position attention: $g_{c,n}^l = \sigma_H(\gamma_c^l(\bar{x}_n))$, $\bar{x}_n = (\text{softmax}(\frac{q_n K^\top + r_n}{\sqrt{d_k}})V) \oplus x_n$
    - **Design Motivation**: The original SPD computes importance independently per position, but in sequence models the same token type carries different importance at different positions (e.g., "bank" in "river bank" vs. "bank manager"). In OV circuits, identical values may be attended to unequally.

2. **Partial Reconstruction Loss**:

    - **Function**: Prevents the decomposed model from "cheating" through unused subcomponents.
    - **Mechanism**: $\mathcal{L}_{\text{partial}} = D_{KL}(f(x|W^1,...,W^{l\in\mathcal{M}}(x,g^l(x)),...,W^L), f(x|W))$, where only a randomly selected subset of layers has their weights replaced by the decomposed version during each training step.
    - **Design Motivation**: When decomposing a large model with limited data, unused subcomponents may be repurposed as shortcuts. The partial reconstruction loss forces each layer's decomposition to be independently substitutable.

3. **Faithfulness and Minimality Losses**:

    - **Function**: Core training objectives.
    - **Mechanism**: Faithfulness: $\mathcal{L}_{faith} = \frac{1}{N}\sum_{l}\sum_{i,j}(W_{i,j}^l - \sum_c U_{i,c}^l V_{c,j}^l)^2$; Minimality: $\mathcal{L}_{min} = \sum_l\sum_c |g_c^l(x)|^p$; stochastic reconstruction uses $\alpha \sim \mathcal{U}(g_c^l(x), 1)$ to ensure gradient signals reach subcomponents with zero importance.
    - **Design Motivation**: Stochastic sampling serves a dual purpose — it provides gradient pathways to "deactivated" subcomponents while establishing a lower bound on causal importance by requiring the original model output to be reconstructed with as few subcomponents as possible.

## Key Experimental Results

### Induction Head Decomposition

| Component | Unique Subcomponents | Key Position Activation |
|-----------|----------------------|------------------------|
| $Q_0$ | 1 | position $m$ (1.0) |
| $K_0$ | 1 | position $s_1$ (1.0) |
| $V_0$ | 1 | position $s_1$ (1.0) |
| $Q_1$ | 1 | position $s_2$ (1.0) |
| $K_1$ | 1 | position $m$ (1.0) |
| $V_1$ | 11 | position $m$ (5.053) |

$\mathcal{L}_{faithful} = 3 \times 10^{-9}$, $\mathcal{L}_{recon} = 1 \times 10^{-4}$

### GPT-2-small Decomposition

| Metric | Value |
|--------|-------|
| Total active subcomponents | 96 (99% reduction from the original model) |
| Suppressing "obe" + "Bryant" | Basketball probability significantly decreases |
| Suppressing "Woods" | Golf probability significantly decreases |
| Reverse retention | "Most famous golfer" still correctly answered as Tiger Woods |

### Key Findings
- The induction head recovers the expected two-step circuit: Layer 0 causes $m$ to attend to $s_1$ (learning "follows $s$"), and Layer 1 causes $s_2$ to attend to $m$.
- $V_1$ requires 11 subcomponents: representing the identity of $m$ among 128 tokens requires above rank-1 information.
- In GPT-2, the "Kobe Bryant → basketball" knowledge is written into the residual stream as early as Layer 0 MLP, complementing the causal tracing results of Meng et al.
- Knowledge storage is asymmetric: suppressing "athlete → sport" does not affect "sport → athlete."

## Highlights & Insights
- **Causal handles in parameter space**: SPD's rank-1 directions are precise — suppressing a specific direction selectively reduces the target probability without affecting other samples, achieving greater precision than activation-space methods.
- **Partial reconstruction loss** addresses the challenge of decomposing large models with limited data, forcing the decomposition to faithfully reflect the original model; this approach is transferable to pruning and factorization settings.

## Limitations & Future Work
- Validation is limited to small models (2-layer toy model + GPT-2-small); scalability to larger models such as LLaMA/Mistral remains unknown.
- The sequence-aware causal importance parameterization introduces additional computational and memory overhead (one attention network per subcomponent).
- GPT-2 experiments involve only 2 samples, lacking systematic quantitative evaluation and comparative baselines.
- The effects of nonlinear interactions (GELU, LayerNorm, residual connections) are not sufficiently analyzed.
- No head-to-head comparison with existing methods such as SAEs or activation patching on identical tasks.
- The decomposition granularity (number of subcomponents $C$) must be selected manually; automatically determining the optimal $C$ remains an open problem.

## Related Work & Insights
- **vs. SAE**: SAEs operate in activation space to discover interpretable features but suffer from absorption and splitting; SPD decomposes in parameter space into rank-1 mechanisms, offering a complementary perspective.
- **vs. ROME**: ROME's causal tracing identifies mid-layer MLPs as edit sites; SPD reveals that information is present as early as Layer 0 MLP, indicating that edit sites do not necessarily coincide with storage sites.
- **vs. APD**: APD uses batch top-$k$ to hard-code sparsity; SPD uses learned causal importance, which is more flexible.
- **vs. L3D**: L3D learns sparse active parameter directions via gradient reconstruction, allowing higher ranks (Tucker decomposition); SPD maintains the rank-1 constraint for greater interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐ First extension of parameter-space decomposition to Transformers; the sequence-aware causal importance function and partial reconstruction loss are meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐ Workshop scope limits experiments to primarily qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation and method are clearly presented with accurate positioning.
- Value: ⭐⭐⭐⭐ An important step toward parameter-space interpretability in Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transformer Embeddings for Fast Microlensing Inference](transformer_embeddings_for_fast_microlensing_inference.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](../../AAAI2026/time_series/probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](../../ICLR2026/time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
