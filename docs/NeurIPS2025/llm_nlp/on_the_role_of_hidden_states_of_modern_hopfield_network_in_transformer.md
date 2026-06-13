---
title: >-
  [Paper Note] On the Role of Hidden States of Modern Hopfield Network in Transformer
description: >-
  [NeurIPS 2025][LLM/NLP][Modern Hopfield Network] This paper moves beyond the adiabatic approximation underlying the established correspondence between Modern Hopfield Networks (MHN) and Transformers. By retaining the hid…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "Modern Hopfield Network"
  - "self-attention"
  - "rank collapse"
  - "token uniformity"
  - "hidden state"
  - "Vision Transformer"
  - "GPT-2"
date: 2026-05-08
content_hash: 3c6e5e2e0c1944d3
---

# On the Role of Hidden States of Modern Hopfield Network in Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2511.20698](https://arxiv.org/abs/2511.20698)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: Modern Hopfield Network, self-attention, rank collapse, token uniformity, hidden state, Vision Transformer, GPT-2

## TL;DR

This paper moves beyond the adiabatic approximation underlying the established correspondence between Modern Hopfield Networks (MHN) and Transformers. By retaining the hidden-state dynamics of MHN, it derives a novel attention mechanism—Modern Hopfield Attention (MHA)—that introduces a cross-layer propagation mechanism for attention scores within self-attention layers. MHA improves the performance of ViT and GPT-2 systematically without adding any trainable parameters, and both theoretically and empirically demonstrates that it effectively alleviates the rank collapse problem in deep Transformers.

## Background & Motivation

The correspondence between Hopfield networks and Transformers has been an active research direction in recent years.

**Established correspondence**: Ramsauer et al. and Krotov & Hopfield showed that the state update rule of the Modern Continuous Hopfield Network (MCHN) under the **adiabatic limit** ($\tau_h \approx 0$) is exactly equivalent to the self-attention mechanism in Transformers.

**Limitations of Prior Work**: The adiabatic limit eliminates the dynamics of the hidden state $\bm{h}$ by directly setting $\bm{h}_n = \bm{x}_n \bm{W}_2$. However, a general MCHN contains two dynamical variables—the visible state $\bm{x}$ and the hidden state $\bm{h}$—and the physical significance of the latter and its implications for Transformer design remain unexplored.

**Rank collapse in Transformers**: As Transformer depth increases, token representations converge (cosine similarity approaches 1), leading to a loss of representational diversity. Dong et al. proved that the rank of purely attentive networks decays at a doubly exponential rate.

**Core Problem**: Without the adiabatic approximation, what structure in the Transformer does the hidden state of MHN correspond to? Can this structure improve Transformer performance?

## Method

### Overall Architecture

Starting from the continuous-time dynamics of the MCHN, the paper derives MHA—a novel attention mechanism that incorporates hidden-state dynamics—through exact discretization (without neglecting the effect of the discretization step size), and uses it as a drop-in replacement for the self-attention layer in Transformers.

**MCHN dynamics**:

$$\tau_v \frac{d\bm{x}}{dt} = \bm{f}(\bm{h})\bm{W}_1^\top - \bm{x}, \quad \tau_h \frac{d\bm{h}}{dt} = \bm{g}(\bm{x})\bm{W}_2 - \bm{h}$$

where $\bm{x}$ denotes the visible state (feature neurons), $\bm{h}$ denotes the hidden state (memory neurons), and $\tau_{v,h}$ are time constants.

**Exact discretization**: Introducing parameters $\alpha = 1 - \Delta t / \tau_v$ and $\alpha' = 1 - \Delta t / \tau_h$:

$$\bm{x}_{n+1} = \alpha \bm{x}_n + (1-\alpha) \bm{f}(\bm{h}_n) \bm{W}_1^\top$$
$$\bm{h}_{n+1} = \alpha' \bm{h}_n + (1-\alpha') \bm{g}(\bm{x}_n) \bm{W}_2$$

### Key Designs

**From the adiabatic limit to full dynamics**:

- **Adiabatic limit** ($\alpha' = 0$): Setting $\bm{h}_n = \bm{x}_n \bm{W}_2$ recovers standard self-attention: $\bm{x}_{n+1} = \text{softmax}(\bm{q}_n \bm{K}_n^\top) \bm{V}_n$
- **Retaining hidden state** ($\alpha' \neq 0$): Yields **Modern Hopfield Attention (MHA)**:

$$\bm{x}_{n+1} = \alpha \bm{x}_n + (1-\alpha) \text{softmax}(\bm{h}_n) \bm{V}_n$$
$$\bm{h}_n = \alpha' \bm{h}_{n-1} + (1-\alpha') \bm{q}_n \bm{K}_n^\top$$

**Physical interpretation of MHA**:

- The hidden state $\bm{h}_n$ accumulates attention scores $\bm{Q}_\ell \bm{K}_\ell^\top$ across layers via an **exponential moving average**.
- The softmax no longer operates on the current-layer attention scores but on the **accumulated hidden state**, enabling the propagation of attention information from shallow to deep layers.
- $\alpha$ controls the residual connection strength; $\alpha'$ controls the degree of memory over historical attention scores.
- **No additional trainable parameters are introduced**; the computational overhead is only $O(T^2)$ compared to the $O(dT^2)$ complexity of standard self-attention, where $d \gg 1$.

### Loss & Training

**Theoretical analysis of rank collapse mitigation**: For purely attentive networks (without skip connections), the upper bound on rank decay under standard self-attention is (Dong et al.):

$$\|\text{Res}(\text{AttnNet}(\bm{X}))\|_{1,\infty} \leq (rC)^{\frac{3^L-1}{2}} \|\text{Res}(\bm{X})\|_{1,\infty}^{3^L}$$

i.e., rank decays doubly exponentially as $3^L$. MHA improves this bound to:

$$\|\text{Res}(\text{AttnNet}(\bm{X}))\|_{1,\infty} \leq \max_{m=0}^{L} (r(1-\alpha')C_1)^{\frac{3^m-1}{2}} (r\alpha' C_2)^{3^m(L-m)} \|\text{Res}(\bm{X})\|_{1,\infty}^{3^m}$$

When the $m=0$ term dominates, the decay degrades to a linear rate $(r\alpha' C_2)^L$, **reducing doubly exponential decay to linear decay**. The fundamental reason is that the linear propagation term from the hidden state breaks the nonlinear compression chain induced by successive softmax operations.

## Key Experimental Results

### Main Results: GPT-2 Text Generation (WikiText-103 Perplexity)

| Model | Standard Attention | MHA (α=0.5) |
|-------|--------------------|-------------|
| GPT-2 Small (124M) | 22.87 | **20.70** |
| GPT-2 Medium (350M) | 20.85 | **19.61** |

### Main Results: LLaMA Architecture Text Generation (Perplexity)

| Dataset | Standard Attention | MHA (α=0.5) |
|---------|--------------------|-------------|
| WikiText-103 | 14.49 | **14.29** |
| CNN DailyMail | 19.36 | **18.97** |
| BookCorpus | 23.76 | **23.50** |

### Main Results: ViT Image Classification (CIFAR-100 Accuracy)

| Model | Standard Attention | MHA (α=0.5) | MHA (α=0.7) |
|-------|--------------------|-------------|-------------|
| ViT-Tiny (5.5M) | **73.08** | 72.03 | 72.57 |
| ViT-Small (22M) | 74.49 | 75.42 | **75.59** |
| ViT-Base (86M) | 75.36 | **76.22** | 75.59 |
| ViT-Large (303M) | 72.91 | **75.78** | 75.37 |

### Main Results: ViT-Base on ImageNet-1k

| Method | Top-1 Accuracy |
|--------|----------------|
| Standard Attention | 76.07 |
| MHA (α=0.5) | 76.43 |
| MHA (α=0.7) | **77.06** |

### Ablation Study: Independent Effects of α and α' (ViT-Tiny, CIFAR-100)

| Setting | Key Observation |
|---------|----------------|
| Fixed α=0.5, α' swept 0→1 | α'=0.2 achieves peak accuracy of 72.29; α'=1.0 collapses to 66.10 |
| Fixed α'=0.5, α swept 0→1 | α=0.6 achieves peak accuracy of 72.66; α=1.0 collapses to 1.00 (pure skip) |
| α=0 (no residual connection) | Performance drops to 69.89, indicating both parameters are necessary |
| α'=0 (no hidden state) | Performance drops to 71.16, reverting to near-standard-attention behavior |

### Ablation Study: Networks without Skip Connections across Depths

| Depth | Standard Attention (CIFAR-10 / CIFAR-100) | MHA (CIFAR-10 / CIFAR-100) |
|-------|-------------------------------------------|---------------------------|
| 1 | 55.08 / 30.90 | 65.41 / 40.08 |
| 2 | 63.72↑ / 40.06↑ | 79.75↑ / 56.94↑ |
| 4 | 57.38↓ / 32.25↓ | **85.74↑** / **64.39↑** |
| 8 | 48.59↓ / 17.19↓ | 80.34↓ / 49.90↓ |
| 12 | 10.00↓ / 1.00↓ | 10.00↓ / 1.00↓ |

Standard attention collapses at depth 4, whereas MHA continues to improve at depth 4, demonstrating significant suppression of rank collapse.

### Key Findings

1. **MHA yields systematic improvements**: Performance gains are observed across GPT-2, LLaMA, and ViT on five datasets, with no additional parameters.
2. **Gains scale with model size**: ViT-Large on CIFAR-100 improves by 2.87% (72.91→75.78), while ViT-Tiny shows almost no change.
3. **α and α' are jointly necessary**: The two parameters independently control residual connection strength and attention memory; removing either degrades performance.
4. **Theory and experiment are consistent**: MHA's improvement of rank decay from doubly exponential to linear is directly validated by the skip-connection-free ablation.
5. **Strong transfer learning performance**: MHA-ViT pretrained on ImageNet substantially outperforms standard ViT on 3 out of 4 downstream datasets.

## Highlights & Insights

- **Elegant physical intuition**: Cross-layer propagation of attention scores is naturally derived from the hidden-state dynamics of associative memory, rather than engineered ad hoc.
- **Zero-parameter improvement**: MHA adds only $O(T^2)$ computation (negligible relative to $O(dT^2)$) while delivering consistent performance gains.
- **Deep integration of theory and experiment**: The theoretical analysis of rank collapse—from doubly exponential to linear decay—is quantitatively validated in the ablation study.
- **New value of Hopfield networks**: The work demonstrates that Hopfield networks can directly guide architectural improvements, rather than serving merely as post-hoc theoretical interpretations.
- **Minimal implementation effort**: The modification reduces to replacing attention score computation with an exponential moving average and adjusting skip connection weights, requiring minimal engineering changes.

## Limitations & Future Work

1. **Hyperparameter tuning required**: Although α=0.5 is effective in most cases, the optimal α varies across tasks (GPT-2 uses 0.5; some ViT tasks benefit from 0.7).
2. **Insufficient large-scale validation**: Experiments are limited to GPT-2 (124M–350M) and ViT (5.5M–303M); validation on modern large-scale models (e.g., 7B+ LLMs) is absent.
3. **Encoder vs. decoder differences not deeply analyzed**: GPT-2 uses causal attention while ViT uses bidirectional attention; the behavioral differences of MHA in these two settings may be fundamental.
4. **No energy function defined**: Breaking the symmetry assumption between $\bm{W}_1$ and $\bm{W}_2$ means MHA lacks a monotonically decreasing energy function, leaving theoretical convergence guarantees open.
5. **Combination with other attention improvements unexplored**: Compatibility with engineering optimizations such as FlashAttention and Group Query Attention is not investigated.
6. **Some tasks near saturation**: CIFAR-10 accuracy is already close to the ceiling, limiting the ability to demonstrate MHA's full effect.

## Related Work & Insights

- **Ramsauer et al. (2021)**: Established the adiabatic-limit correspondence between MHN and Transformers; the direct theoretical foundation of this work.
- **Krotov & Hopfield (2021)**: Proposed a general theoretical framework for Dense Associative Memory and formalized the adiabatic-limit derivation.
- **RealFormer (He et al., 2021)**: Proposed cross-layer reuse of attention scores from an engineering perspective, but is restricted to encoders and lacks theoretical grounding; MHA derives a more general mechanism naturally from Hopfield network dynamics.
- **Dong et al. (2021)**: Theoretical analysis of rank collapse; this paper directly extends their framework to derive MHA's improvement.
- **Insights**: Physical models of associative memory can provide systematic guidance for deep learning architecture design, rather than serving only as post-hoc explanations. Retaining, rather than simplifying, the complete dynamics of a dynamical system may reveal unexploited architectural improvements.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Deriving a practical attention improvement from Hopfield hidden-state dynamics is theoretically original and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers NLP (GPT-2, LLaMA) and CV (ViT) with detailed ablations, but lacks large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous derivations with tight theory-experiment integration, though notation is dense in places.
- Value: ⭐⭐⭐⭐ — Offers new depth to the Hopfield–Transformer connection; rank collapse mitigation has practical significance, but large-model validation is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[NeurIPS 2025\] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)
- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](../../ICML2026/llm_nlp/anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)

</div>

<!-- RELATED:END -->
