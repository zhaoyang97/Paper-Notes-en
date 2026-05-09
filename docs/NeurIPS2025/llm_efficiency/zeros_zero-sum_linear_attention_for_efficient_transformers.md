---
title: >-
  [Paper Note] ZeroS: Zero-Sum Linear Attention for Efficient Transformers
description: >-
  [NeurIPS 2025][LLM Efficiency][zero-sum attention] By removing the zeroth-order uniform term $1/t$ from softmax, ZeroS constructs a linear attention mechanism with zero-sum weights, breaking the limitation of convex combinations to purely additive mixing. This enables differential/contrastive operations within a single layer while maintaining $O(Nd^2)$ linear complexity, matching or surpassing standard softmax attention across multiple sequence modeling benchmarks.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - zero-sum attention
  - linear attention
  - softmax decomposition
  - radial-angular decoupling
  - O(N) complexity
date: 2026-05-08
content_hash: 8dac4527f7442a59
---

# ZeroS: Zero-Sum Linear Attention for Efficient Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2602.05230](https://arxiv.org/abs/2602.05230)
**Code**: Available
**Area**: LLM Efficiency
**Keywords**: zero-sum attention, linear attention, softmax decomposition, radial-angular decoupling, O(N) complexity

## TL;DR

By removing the zeroth-order uniform term $1/t$ from softmax, ZeroS constructs a linear attention mechanism with zero-sum weights, breaking the limitation of convex combinations to purely additive mixing. This enables differential/contrastive operations within a single layer while maintaining $O(Nd^2)$ linear complexity, matching or surpassing standard softmax attention across multiple sequence modeling benchmarks.

## Background & Motivation

**Background**: Linear attention approximates softmax attention via kernel feature maps $\phi(q)\phi(k)^\top$, reducing complexity from $O(N^2)$ to $O(N)$. Representative methods include Performer, GLA, and Mamba. However, these methods generally underperform standard softmax attention.

**Limitations of Prior Work**:
   - **Convex combination bottleneck**: Softmax attention produces non-negative weights, enabling only additive mixing of value vectors. Linear attention methods also restrict weights to be positive for numerical stability. Consequently, a single attention layer cannot express differential or contrastive operations — even with just two tokens, computing $v_1 - v_2$ in one layer is impossible.
   - **Uniform weight bias**: In the Taylor expansion of softmax, the zeroth-order term $1/t$ introduces a persistent averaging effect. In long sequences, this uniform component dilutes focused attention, leading to attention degradation.

**Key Challenge**: Achieving linear complexity requires decomposable kernel forms, yet existing kernel methods attempt to simulate the non-negativity of softmax — which is precisely the source of their limited expressivity.

**Goal**: (a) Can attention weights be allowed to take negative values while maintaining numerical stability? (b) Can linear attention support contrastive operations within a single layer, closing the performance gap with softmax?

**Key Insight**: The Taylor expansion of softmax — decomposing $\text{softmax}(s_i)$ into a zeroth-order term ($1/t$), a first-order term ($\delta_i/t$), and higher-order residuals ($\varepsilon_i$). The zeroth-order term contributes only uniform averaging with no discriminative value across token interactions. Removing it yields naturally zero-sum weights $\sum_i w_i = 0$, supporting both positive and negative values with norm stability independent of sequence length.

**Core Idea**: Rather than designing complex kernels to approximate softmax while preserving non-negativity, the paper takes the opposite approach — removing the zeroth-order term responsible for the non-negativity constraint and constructing more expressive linear attention using zero-sum residual weights.

## Method

### Overall Architecture

ZeroS directly replaces the multi-head attention module in Transformers, leaving all other components (MLP, LN, residual connections) unchanged. Each ZeroS layer consists of three core operations: (1) computing bias logits $s_i$ (dependent only on step $i$, not the current step $t$); (2) computing reweighted zero-sum softmax weights $w_{t,i}$; and (3) multiplying the radial weights by the angular cosine component to obtain the final attention output. The overall complexity is $O(Nd^2)$ time and $O(d^2)$ memory.

### Key Designs

1. **Softmax Zeroth-Order Removal and Zero-Sum Weight Construction**

    - **Function**: Decomposes softmax into a zeroth-order term ($1/t$), a first-order offset ($\delta_{t,i}/t$), and higher-order residuals ($\varepsilon_{t,i}$); removes the zeroth-order term and mixes the remaining components via learnable gating.
    - **Mechanism**: $w_{t,i} = \sigma_t^1 \frac{\delta_{t,i}}{t} + \sigma_t^h \varepsilon_{t,i}$, where $\sigma_t^1, \sigma_t^h$ are sigmoid gates conditioned on the current step $t$. All weights satisfy $\sum_i w_{t,i} = 0$, naturally admitting both positive and negative values.
    - **Theoretical Support**: Proposition 3.1 proves that the reachable set of zero-sum weights strictly contains that of convex combinations (provided value vectors are not all identical), i.e., zero-sum attention is strictly more expressive than standard attention.
    - **Design Motivation**: The zeroth-order term $1/t$ contributes only uniform averaging (indiscriminate across all $(t,i)$ pairs) and is the "lowest-cost basis" of expressivity. Its removal incurs no loss of generality (the zeroth-order term may optionally be retained in the first layer to span the affine hull); subsequent layers recover information in the uniform direction via residual connections.

2. **Radial-Angular Decoupling**

    - **Function**: Decomposes the final attention weights into a radial component $r_{t,i}$ (from zero-sum softmax) and an angular component $\cos\theta$ (query-key directional similarity).
    - **Mechanism**: $o_t = \sum_{i=1}^{t} r_{t,i} \cos\theta \cdot v_i$, where $\cos\theta = \hat{q}_t \hat{k}_i^\top$ (inner product of normalized query and key). Since the zero-sum weights are already numerically stable, $\cos\theta$ can be directly multiplied without requiring positivity constraints.
    - **Design Motivation**: In standard softmax, $\exp(\|q\|\|k\|\cos\theta)$ couples magnitude and direction — when $\cos\theta$ changes sign, large positive values collapse to near-zero, a "flip effect" central to softmax's expressivity. Linear attention replaces this with $\phi(q)\phi(k)$, but positive mappings $\phi$ restrict the angular range (to $<90°$), losing the flip effect. ZeroS explicitly recovers this capability through decoupling.
    - **Compatibility with RoPE**: $\cos\theta' = \hat{q}_t R_{t-i} \hat{k}_i^\top$; rotary position encodings integrate naturally into the angular component.

3. **Bias Logit Design**

    - **Function**: Computes a scalar logit $s_i$ for each step $i$ (independent of the current step $t$).
    - **Mechanism**: $s_i = -\frac{1}{\sqrt{d}} u_i \bar{u}_i^\top$, where $u_i = x_i W_u$ and $\bar{u}_i = \frac{e^\tau \mu + \sum_{j=1}^i u_j}{e^\tau + i}$ is a cumulative mean with a learnable prior. $s_i$ measures the deviation of step $i$ from its history.
    - **Design Motivation**: Logits must depend only on $i$ rather than $(t,i)$ to enable linear-time computation via prefix sums. The influence of step $t$ is injected through gates $\sigma_t^1, \sigma_t^h$, enabling dynamic control over different-order zero-sum components.

4. **Linear-Time Prefix Scan**

    - **Function**: Achieves $O(d^2)$ per-step computation by maintaining five prefix-sum states.
    - **Mechanism**: $E_t = \sum e^{s_i}$, $P_t = \sum s_i$, $F_t = \sum e^{s_i} \hat{k}_i^\top v_i$, $G_t = \sum s_i \hat{k}_i^\top v_i$, $H_t = \sum \hat{k}_i^\top v_i$. The final output is $o_t = \hat{q}_t(\alpha_t F_t + \beta_t G_t + \gamma_t H_t)$, where $\alpha_t, \beta_t, \gamma_t$ are coefficients derived from the gates and prefix-sum scalars.
    - **Design Motivation**: All state matrices are of size $d \times d$, with $O(d^2)$ update cost per step, yielding $O(Nd^2)$ total time and $O(d^2)$ memory — identical to other linear attention methods.

### Loss & Training

- Directly replaces multi-head attention in standard Transformers while retaining the original MLP, embeddings, hyperparameters, and training configuration.
- By default, no zeroth-order term is added to the first layer (experiments show negligible impact).
- Both causal (decoder) and non-causal (encoder) modes are supported.

## Key Experimental Results

### Main Results: MAD Benchmark (In-Context Learning)

| Model | Compress | Fuzzy | In-Context | Memorize | Noisy | Sel.Copy | Avg. |
|-------|----------|-------|------------|----------|-------|----------|------|
| Transformer | 51.6 | 29.8 | 94.1 | 85.2 | 86.8 | 99.6 | 74.5 |
| Mamba | 52.7 | 6.70 | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| DeltaNet | 42.2 | 35.7 | 100 | 52.8 | 100 | 100 | 71.8 |
| LinAttn | 31.1 | 8.15 | 91.0 | 74.9 | 75.6 | 93.1 | 62.3 |
| **ZeroS** | 44.0 | 14.9 | **99.9** | 88.1 | 96.1 | 97.8 | **73.5** |
| **ZeroS-SM** | 45.2 | 28.0 | **100** | 84.3 | 96.6 | 98.5 | **75.4** |

### Ablation Study (WikiText-103 + MAD)

| Configuration | WikiText PPL↓ | MAD Avg. |
|---------------|--------------|---------|
| ZeroS (full) | 24.61 | 73.5 |
| + retain zeroth-order term | 24.74 (+0.13) | 69.4 (−4.1) |
| replace with standard softmax (no RWSM) | 24.97 (+0.36) | 67.6 (−5.9) |
| remove gating | — | 70.8 (−2.7) |
| remove LayerNorm | — | 69.4 (−4.1) |

### Key Findings

- **Zero-sum weights are critical for in-context learning**: In-Context Recall improves from 91.4 (with zeroth-order term) to 99.9 (zero-sum), validating the hypothesis that convex combinations limit algorithmic reasoning.
- **Surpasses Transformer on WikiText**: ZeroS achieves PPL 24.61 vs. Transformer 24.78, marking the first time a linear attention method outperforms standard attention on this benchmark.
- **Effective on ImageNet classification**: ZeroS achieves 75.51% vs. DeiT's 72.20% on DeiT-Tiny, demonstrating applicability to vision tasks.
- **Consistently superior on time series**: Outperforms GLA, AFT, and domain-specific methods such as iTransformer on Weather, Solar, and ETT datasets.
- **Memory capacity is preserved**: Despite using negative weights, the Memorize task score (88.1) is comparable to Transformer (85.2) and Mamba (89.5), indicating that zeroth-order removal does not impair sequence memorization.

## Highlights & Insights

- **The elegance of subtraction**: Rather than designing increasingly complex kernels to approximate softmax while preserving non-negativity, the paper boldly removes the zeroth-order term that causes the constraint. The simplicity is striking — it is surprising that no prior work explored this direction.
- **Theoretical leap from convex combination to zero-sum**: Proposition 3.1 rigorously proves that the zero-sum reachable set strictly contains the convex-combination reachable set, providing a precise theoretical diagnosis of the expressivity bottleneck in linear attention.
- **Elegant guarantee of numerical stability**: Lemma 3.4 proves $\|\sum_i w_{t,i} v_i\| = O(B)$ (independent of $t$), showing that even with negative weights, the output norm remains bounded — without relying on the norm guarantees of convex combinations.
- **Radial-angular decoupling recovers the "flip effect"**: This reflects a deep understanding of the core source of expressivity in softmax attention — what matters is not $\exp$ itself, but the coupled interaction between magnitude and direction.

## Limitations & Future Work

- No GPU-accelerated implementation (e.g., CUDA kernels) is provided; linear complexity is demonstrated only at the algorithmic level, and practical speed may fall short of engineering-optimized methods such as Mamba/GLA.
- Due to resource constraints, large-scale LLM pretraining (7B+) is not validated; all experiments are conducted on small models (<50M parameters).
- Evaluation focuses primarily on autoregressive tasks; non-causal settings (e.g., BERT-style) are insufficiently explored.
- The bias logit design (negative inner product + cumulative mean) is reasonable but not unique; superior logit function designs may exist.

## Related Work & Insights

- **vs. Differential Transformer**: Diff Transformer obtains negative weights by taking the difference of two attention matrices, but remains $O(N^2)$ in complexity. ZeroS achieves equivalent positive-negative weight capability in $O(N)$ with a more principled theoretical foundation.
- **vs. DeltaNet**: DeltaNet implements "subtraction" by actively deleting state matrix entries, performing well on certain tasks but dropping to 52.8% on Memorize. ZeroS achieves "subtraction" via zero-sum weights without compromising memorization, retaining a Memorize score of 88.1%.
- **vs. GLA/Mamba**: Traditional linear attention and state-space models generally lag Transformers by 5–12 points on the MAD benchmark. ZeroS is the first linear-complexity method to match or exceed Transformer performance on this suite.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The theoretical perspective of zero-sum weights is entirely novel and insightful; the "remove the zeroth-order term" approach is both concise and powerful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task validation across MAD/WikiText/ImageNet/time series with thorough ablations, but large-scale LLM experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The derivation chain for softmax decomposition is clear, with theoretical propositions and experimental conclusions in precise correspondence.
- **Value**: ⭐⭐⭐⭐⭐ Represents a significant theoretical and practical advance in linear attention, with the potential to reshape design paradigms for future linear attention methods.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)
- [\[NeurIPS 2025\] Constant Bit-Size Transformers Are Turing Complete](constant_bit-size_transformers_are_turing_complete.md)
- [\[NeurIPS 2025\] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)

<!-- RELATED:END -->
