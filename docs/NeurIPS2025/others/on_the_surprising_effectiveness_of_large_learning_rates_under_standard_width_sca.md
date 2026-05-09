---
title: >-
  [Paper Note] On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling
description: >-
  [NeurIPS 2025][infinite-width limit] This paper reveals that under standard parameterization (SP), the cross-entropy loss causes the previously monolithic "unstable" regime to split into two distinct sub-regimes: catastrophic instability and controlled divergence. In the controlled divergence regime ($\eta_n = \Theta(n^{-1/2})$), logits diverge while gradients and activations remain stable, thereby establishing the first practically useful infinite-width limit for SP that admits feature learning.
tags:
  - NeurIPS 2025
  - infinite-width limit
  - standard parameterization
  - learning rate scaling
  - cross-entropy loss
  - feature learning
  - controlled divergence
date: 2026-05-08
content_hash: 181d34562323e4ed
---

# On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2505.22491](https://arxiv.org/abs/2505.22491)
**Authors**: Moritz Haas, Sebastian Bordt, Ulrike von Luxburg, Leena Chennuru Vankadara (Tübingen, UCL Gatsby)
**Area**: Other
**Keywords**: infinite-width limit, standard parameterization, learning rate scaling, cross-entropy loss, feature learning, controlled divergence

## TL;DR

This paper reveals that under standard parameterization (SP), the cross-entropy loss causes the previously monolithic "unstable" regime to split into two distinct sub-regimes: catastrophic instability and controlled divergence. In the controlled divergence regime ($\eta_n = \Theta(n^{-1/2})$), logits diverge while gradients and activations remain stable, thereby establishing the first practically useful infinite-width limit for SP that admits feature learning.

## Background & Motivation

### Root Cause
Infinite-width theory (Tensor Programs, etc.) predicts that under standard parameterization (He initialization + global learning rate), learning rates larger than $\mathcal{O}(1/n)$ lead to training instability, while $\mathcal{O}(1/n)$ rates cause feature learning to vanish (degenerating to the kernel regime). In practice, however:
- The optimal learning rate decays far more slowly than $\mathcal{O}(1/n)$, typically at $\Omega(1/\sqrt{n})$;
- Networks remain stably trainable at large widths and learn meaningful features;
- This gap is already pronounced in a **single gradient step** of a 2-layer MLP, ruling out depth accumulation and multi-step finite-width effects as primary explanations.

### Limitations of Prior Work

**Finite-width effects**: The gap is more pronounced in a single gradient step of a 2-layer MLP (Figure F.15), ruling out depth/training-time accumulation.

**Catapult mechanism**: Analysis of linear models under SP shows that catapult dynamics require $\eta_n = \mathcal{O}(n^{-1})$ to prevent sharpness divergence, and thus cannot explain stability at larger learning rates.

**Misalignment hypothesis**: Everett et al. (2024) hypothesize that the absence of alignment between weights and activations accounts for the discrepancy; however, the Refined Coordinate Check (RCC) proposed in this work confirms that the infinite-width alignment predictions hold already at moderate widths.

### Core Problem
Why does SP remain stable and effective under large learning rates? Does there exist an infinite-width limit that more faithfully captures the behavior of finite-width networks in practice?

## Method

### Theoretical Framework: Controlled Divergence Regime

Consider an $(L+1)$-layer MLP of width $n$ under SP initialization, trained with SGD at global learning rate $\eta_n = \eta \cdot n^{-\alpha}$.

**Key observation**: The choice of loss function determines the consequences of logit divergence.
- **MSE loss**: The loss-logit gradient $\chi_t = f_t(\xi_t) - y_t$ diverges directly when logits diverge, causing catastrophic instability.
- **CE loss**: The loss-logit gradient $\chi_t = \sigma(f_t(\xi_t)) - y_t$ is bounded by the softmax, which maps logits into $[0,1]$; logit divergence merely drives $\sigma(f)$ toward a one-hot prediction, leaving gradients bounded.

### Proposition 2: Three Asymptotic Regimes under SP (CE Loss)

| Regime | Exponent $\alpha$ | Logits $\|f_t\|_{RMS}$ | Activations $\|x_t^l\|_{RMS}$ | Gradients $\|\chi_t\|_{RMS}$ |
|---|---|---|---|---|
| Stable | $\alpha \geq 1$ | $\mathcal{O}(1)$ | $\Theta(1)$ | $\mathcal{O}(1)$ |
| **Controlled divergence** | $\frac{1}{2} \leq \alpha < 1$ | $\Theta(n^{1-\alpha}) \to \infty$ | $\Theta(1)$ | $\mathcal{O}(1)$ |
| Catastrophic instability | $\alpha < \frac{1}{2}$ | $\to \infty$ | $\to \infty$ | $\to \infty$ |

Under MSE loss, any $\alpha < 1$ leads directly to catastrophic instability; **no controlled divergence regime exists**.

### Proposition 4: Feature Learning in the Controlled Divergence Regime

Under CE loss, SP, and $\eta_n = \eta \cdot n^{-\alpha}$ with $\frac{1}{2} \leq \alpha < 1$:
- Feature learning vanishes in the input layer: $\|\Delta x_t^1\|_{RMS} = \Theta(n^{-1/2 - \alpha})$;
- **Feature learning is non-vanishing in hidden layers**: $\|\Delta x_t^l\|_{RMS} = \Theta(n^{1/2 - \alpha})$ for $l \in [2, L]$;
- In particular, at $\alpha = 1/2$, feature learning in all hidden layers is width-independent: $\|\Delta x_t^l\|_{RMS} = \Theta(1)$.

This constitutes the **first infinite-width limit for SP in a practically useful feature learning regime**.

### Maximum Stable Learning Rate Exponents Across Optimizers and Architectures

The maximum stable learning rate exponent differs by layer type:
- **Output layer** (logits): $\eta_n = \mathcal{O}(n^{-1})$ (though CE loss permits exceeding this);
- **Hidden layers**: $\eta_n = \mathcal{O}(n^{-1/2})$;
- **Input layer / LayerNorm / Embedding**: $\eta_n = \mathcal{O}(1)$.

For Adam, gradient normalization further stabilizes training, and $\eta_n = \Theta(n^{-1})$ suffices for width-independent updates (analogous to $\mu$P). In Transformers, trainable LayerNorm parameters play a key stabilizing role, raising the maximum stable learning rate from $n^{-1}$ to $n^{-1/2}$.

### Refined Coordinate Check (RCC)

The paper proposes a diagnostic tool that separately measures the effective update $(\Delta W_t^l) x_t^{l-1}$ and the propagated update $W_0^l (\Delta x_t^{l-1})$, enabling accurate estimation of width-scaling exponents at moderate widths ($n \leq 512$).

## Key Experimental Results

### Experimental Setup
- **Architectures**: MLPs (2–8 layers), Pythia-GPT (up to 1.4B parameters)
- **Datasets**: CIFAR-10, MNIST, Fashion-MNIST, DCLM-Baseline language data
- **Optimizers**: SGD, Adam, AdamW
- **Width range**: up to 16384 (MLP), 4096 (GPT)
- **Parameterizations**: SP, $\mu$P, SP-full-align

### Table: Predicted vs. Observed Maximum Stable Learning Rate Exponents under CE vs. MSE Loss

| Parameterization | Loss | Predicted max-stable $\alpha$ | Observed max-stable $\alpha$ | Observed optimal $\alpha$ |
|---|---|---|---|---|
| SP | MSE | $-1$ | $\approx -1$ | $\approx -1$ |
| SP | CE | $-0.5$ | $\approx -0.5$ | $\approx -0.5$ |
| SP-full-align | CE | $0$ | $\approx 0$ | $\approx 0$ (language) / decreasing (vision) |

Theoretical predictions are in strong agreement with experiments. In deep nonlinear networks, the maximum stable learning rate typically coincides with the optimal learning rate.

### Table: CE vs. MSE Performance under $\mu$P (8-layer MLP, SGD, Best Training Accuracy)

| Dataset | SP + CE | SP + MSE | $\mu$P + CE | $\mu$P + MSE |
|---|---|---|---|---|
| MNIST | **~98%** | ~85% | **~98%** | ~97% |
| CIFAR-10 | **~50%** | ~28% | **~52%** | ~50% |
| Fashion-MNIST | **~87%** | ~65% | **~88%** | ~86% |

Key findings:
- Under SP, CE substantially outperforms MSE because MSE loses feature learning at large widths;
- Under $\mu$P, both losses perform comparably since $\mu$P ensures balanced feature learning across layers;
- This provides a width-scaling theoretical explanation for the practical dominance of CE loss in deep learning.

### New Findings on SP-full-align
- SP-full-align (SP + $\mu$P learning rates) recommended by Everett et al. (2024) **fails to transfer learning rates on vision datasets**;
- The failure stems from width-independent alignment between $W_0^{L+1}$ and $\Delta x_t^L$, causing logits to diverge with width;
- Successful transfer on language data is attributed to the output dimension satisfying $d_{\text{out}} \gg n$, which makes the initial operator norm approximately width-independent.

## Highlights & Insights

- **First practically useful infinite-width limit for SP with feature learning**: Under CE loss and $\eta_n = \Theta(n^{-1/2})$, all hidden layers exhibit width-independent feature learning, resolving a long-standing theory–practice gap.
- **Fine-grained characterization of the instability regime**: The previously undifferentiated "unstable" regime is decomposed into two qualitatively distinct sub-regimes; the controlled divergence induced by CE loss is identified as the key mechanism.
- **Scaling-theoretic explanation for the dominance of CE loss**: This work is the first to explain from a width-scaling perspective why CE loss substantially outperforms MSE in deep learning—under SP, CE permits larger learning rates while preserving feature learning, whereas the gap vanishes under $\mu$P.
- **Practical diagnostic tool RCC**: The Refined Coordinate Check separates effective and propagated updates, enabling accurate estimation of scaling exponents at moderate widths; code is publicly available.
- **Importance of trainable LayerNorm**: The work provides a scaling-theoretic explanation for why modern architectures almost universally employ trainable LayerNorm parameters—they raise the maximum stable learning rate in Transformers from $n^{-1}$ to $n^{-1/2}$.

## Limitations & Future Work

- **Analysis restricted to single-epoch training**: Dynamics in multi-epoch settings (e.g., interactions with overfitting) are left for future work.
- **Precise prediction of optimal learning rate exponents remains difficult**: The theory predicts only the maximum stable exponent; the optimal exponent further depends on strong assumptions about architecture and data distribution.
- **Numerical precision constraints**: Under standard floating-point precision, logit divergence may exceed the numerical range at moderate widths, requiring specialized implementations to mitigate the accumulation of width-dependent factors.
- **Input-layer feature learning remains absent**: SP at $\alpha = 1/2$ still cannot recover feature learning in the input layer; fully width-independent training continues to require $\mu$P.
- **Logit divergence under CE + SP may cause overconfidence**: Rapid growth of logits can degrade model calibration; models trained under $\mu$P may be better calibrated.

## Related Work & Insights

- **Infinite-width limits**: NTK (Jacot et al., 2018), Mean-field (Mei et al., 2018; Chizat & Bach, 2018), Tensor Programs (Yang & Hu, 2021) → This work extends Tensor Program theory to the controlled divergence regime of SP.
- **$\mu$P and hyperparameter transfer**: Yang et al. (2022) propose $\mu$P for width-independent hyperparameters → This work explains why SP also approximately works in practice (attributing this to CE loss).
- **Edge of stability**: Cohen et al. (2021) observe that training converges toward sharpness $2/\eta$ → This work analyzes the limitations of the catapult mechanism under SP.
- **Benefits of large learning rates**: Andriushchenko et al. (2023b) show that large steps learn sparse features; Cai et al. (2024) demonstrate margin improvements → This work provides complementary theory from a width-scaling perspective.
- **SP-full-align**: Everett et al. (2024) recommend SP + $\mu$P learning rates as best practice → This work identifies its failure on vision data and proposes a corrected initialization scheme.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to identify the critical role of CE loss in width scaling; the discovery of the controlled divergence regime is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive cross-validation across MLP+GPT, SGD+Adam, CE+MSE, and SP+$\mu$P; theoretical predictions are in strong agreement with experiments.
- Writing Quality: ⭐⭐⭐⭐ — Logically rigorous with clearly stated main conclusions; dense mathematical notation poses a barrier for readers without a theoretical background.
- Value: ⭐⭐⭐⭐⭐ — Resolves a long-standing contradiction between infinite-width theory and practice, with direct implications for hyperparameter selection in large-scale model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[AAAI 2026\] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](../../AAAI2026/others/intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)
- [\[NeurIPS 2025\] EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)
- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](../../AAAI2026/others/structure-aware_encodings_of_argumentation_properties_for_clique-width.md)

</div>

<!-- RELATED:END -->
