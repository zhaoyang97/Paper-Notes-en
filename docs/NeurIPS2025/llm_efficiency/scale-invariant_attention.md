---
title: >-
  [Paper Note] Scale-invariant Attention
description: >-
  [NeurIPS 2025][LLM Efficiency][scale invariance] Drawing inspiration from the scale invariance of natural images, this paper proposes a position-dependent affine transformation on attention logits—comprising a multiplicative scaling and an additive shift—such that the total attention weight and sparsity over any token range satisfy scale invariance. This enables zero-shot generalization from short-context training to long-context inference (4k→64k) with a single hyperparameter $\tau$.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - scale invariance
  - long context
  - attention logits
  - zero-shot length generalization
  - pp-RoPE
  - entropy control
date: 2026-05-08
content_hash: c9185aa5b05e91a7
---

# Scale-invariant Attention

**Conference**: NeurIPS 2025
**arXiv**: [2505.17083](https://arxiv.org/abs/2505.17083)
**Code**: None
**Area**: LLM Efficiency / Attention Mechanism / Long Context
**Keywords**: scale invariance, long context, attention logits, zero-shot length generalization, pp-RoPE, entropy control

## TL;DR
Drawing inspiration from the scale invariance of natural images, this paper proposes a position-dependent affine transformation on attention logits—comprising a multiplicative scaling and an additive shift—such that the total attention weight and sparsity over any token range satisfy scale invariance. This enables zero-shot generalization from short-context training to long-context inference (4k→64k) with a single hyperparameter $\tau$.

## Background & Motivation
**Long context is a core challenge**: Modern LLMs must process sequences at inference time that far exceed their training lengths. Standard attention becomes increasingly "diffuse" (high entropy) as context grows, causing local information to be diluted.

**Limitations of existing positional encodings**: RoPE does not generalize out-of-the-box to longer sequences; YaRN, NTK, and similar methods require continued pretraining or fine-tuning at the target length.

**Problems with LogN/SSMax**: LogN sharpens attention by multiplying by $s\log N$ to ensure sparsity, but it is position-agnostic—applying the same scaling to local tokens (e.g., the first 100) and distant tokens alike. This causes attention weights over local context to decay rapidly toward zero as sequence length increases.

**Inspiration from natural images**: Natural images exhibit structure at all spatial scales—features at both large and small scales are equally important. Attention in text should behave analogously: attention over the first 10–100 tokens, 100–1000 tokens, and 1000–10000 tokens should all be preserved.

**Core insight**: Two properties are needed: (a) *scale-invariant total attention*—the total attention weight over any range should be roughly constant; (b) *scale-invariant attention sparsity*—as the range grows, attention should become increasingly concentrated on a few key tokens rather than spreading uniformly.

**Goal**: Derive a simple logits transformation that satisfies both properties theoretically, thereby enabling zero-shot long-context generalization.

## Method

### Overall Architecture
Within the FlexAttention framework, a position-dependent affine transformation $L_t = a_t S_t + m_t$ is applied to the raw attention score $S_t = \frac{1}{\sqrt{d}} \sum_\lambda q_\lambda K_{t\lambda}$ (where $t$ denotes the distance from the current query to the key). Attention weights are computed from the transformed logits. The transformation parameters $a_t$ (multiplicative) and $m_t$ (additive) are derived analytically from the scale invariance conditions.

### Key Designs

1. **Scale-Invariant Total Attention (Def 3.1)**
   - **Function**: Requires that the total unnormalized attention $\mathbb{E}[\sum_{t'=t}^{t\Delta-1} \exp(L_{t'})]$ over any range $[t, t\Delta)$ is $\Theta(1)$.
   - **Mechanism**: If $\mathbb{E}[\tilde{A}_t] = \alpha / (t/\tau + 1)$, then the total attention over $[t, t\Delta)$ is $\sum \alpha/(t'/\tau+1) \approx \alpha\tau \log\Delta$, which is independent of $t$.
   - **Design Motivation**: Ensures the model attends to both local and global context simultaneously, preventing long-sequence growth from causing attention to completely favor distant tokens.

2. **Scale-Invariant Attention Sparsity (Def 3.3/3.4)**
   - **Function**: The weak form requires the within-range entropy $\mathbb{E}[H_t^{t\Delta}] = o(\log t)$; the strong form requires $\Theta(1)$.
   - **Mechanism**: Achieved by controlling the unnormalized negative entropy $\mathbb{E}[\tilde{A}_t \log\tilde{A}_t] = \beta/(t/\tau+1)$.
   - **Design Motivation**: Ensures that attention over distant ranges concentrates on a few key tokens rather than distributing uniformly.

3. **Closed-Form Solution under Gaussian Assumption**
   - **Function**: Assuming the base logits $\bar{L}_t \sim \mathcal{N}(0,1)$, the transformation is $L_t = a_t \bar{L}_t + m_t$.
   - **Mechanism**: Jointly solving the two conditions yields $a_t = \sqrt{2[\log(t/\tau+1) - \log\alpha + \beta/\alpha]}$ and $m_t = -a_t^2 + \beta/\alpha$.
   - **Design Motivation**: $a_t^2$ grows as $\log t$ (increasing variance to sharpen distant attention), while $m_t$ decreases as $\log t$ (lowering the mean to suppress total distant weight).

4. **Single Hyperparameter Simplification: Only $\tau$**
   - **Function**: Applying boundary conditions $a_0^2 = 1,\, m_0 = 0$ (no modification for local tokens) reduces the three parameters $\alpha, \beta, \tau$ to one.
   - **Mechanism**: Solving yields $\alpha = \beta = e^{0.5}$, leaving only $\tau$ to be tuned. $\tau$ defines the size of the "local region"—for $t \ll \tau$, attention is nearly unmodified.
   - **Design Motivation**: Reduces tuning burden; $\tau = 10$ achieves the best performance in experiments.

5. **Integration with pp-RoPE**
   - **Function**: Applies the scale-invariant transformation to pp-RoPE, a RoPE variant that removes low-frequency (long-wavelength) components.
   - **Mechanism**: Low-frequency components of standard RoPE may interfere with the position-dependent logits transformation; removing them yields better results.
   - **Design Motivation**: Experiments show that scale-invariant RoPE performs poorly, while scale-invariant pp-RoPE performs excellently.

### Loss & Training
- The logits transformation is implemented via FlexAttention without modifying the model architecture.
- GPT-2-style models (with QK-norm and ReLU² activation) are trained from scratch.
- The Muon optimizer is used for linear layers; Adam is used for all other parameters.
- FineWeb dataset with a fixed training data order.

## Key Experimental Results

### Main Results: Validation Loss (162M parameters, trained @4k)

| Method | Val@4k | Val@16k | Val@64k |
|--------|--------|---------|---------|
| RoPE | 3.261 | 3.936 | 5.260 |
| pp-RoPE | 3.260 | 3.984 | 5.735 |
| RoPE+NTK | 3.261 | 3.703 | 5.430 |
| LogN+RoPE | 3.260 | 3.273 | 3.378 |
| LogN+pp-RoPE | 3.256 | 3.262 | 3.317 |
| ALiBi | 3.281 | 3.272 | 3.270 |
| **Scale-inv pp-RoPE** | **3.244** | **3.235** | **3.247** |

### Needle-in-a-Haystack Retrieval (trained @4k, fine-tuned for 300 steps)

| Method | Acc@4k | Acc@16k | Acc@64k |
|--------|--------|---------|---------|
| RoPE | 0.962 | 0.000 | 0.000 |
| LogN+pp-RoPE | 0.969 | 0.962 | 0.939 |
| ALiBi | 0.957 | 0.020 | 0.003 |
| **Scale-inv pp-RoPE** | **0.965** | **0.969** | **0.969** |

### Ablation Study
- $\tau \in \{10^{-2}, 10^{-1}, 1, 10, 100\}$: $\tau=10$ is optimal.
- The advantage is maintained on a 304M model and becomes more pronounced after longer training (10B tokens).

### Key Findings
1. Scale-invariant pp-RoPE achieves lower in-distribution validation loss than all baselines at every training length (4k/16k/64k)—improving not only generalization but also standard performance.
2. In the Train@4k → Val@64k 16× zero-shot generalization setting, the validation loss is only 3.247 (vs. 5.260 for RoPE), with almost no degradation.
3. In needle-in-a-haystack retrieval, the @64k accuracy of 0.969 matches or exceeds LogN+pp-RoPE (0.939), demonstrating that strengthening local attention does not sacrifice distant retrieval.
4. ALiBi generalizes well in terms of validation loss but nearly completely fails on needle-in-a-haystack@64k (0.003), revealing a fundamental trade-off in existing long-context strategies.

## Highlights & Insights
- **Theoretical elegance**: Starting from the scale invariance of natural images, two clear mathematical principles are proposed, yielding a closed-form solution under the Gaussian assumption with only a single hyperparameter.
- **High practical value**: Nearly all analytical effort goes into deriving $a_t$ and $m_t$; implementation requires only one additional line within FlexAttention—an extremely low engineering overhead.
- **Differentiated design**: LogN/SSMax apply a position-agnostic global scaling, whereas the proposed method is position-dependent—local tokens are left largely unchanged, while distant tokens receive increased variance and decreased mean, perfectly balancing local and global attention.
- **Strong zero-shot generalization**: 4k→64k transfer with almost no performance loss is rare among approaches that do not employ long-context continued pretraining.

## Limitations & Future Work
1. **Limited experimental scale**: Validation is conducted only on models trained from scratch at 162M and 304M parameters; the 7B-scale experiment appears only in the appendix as continued pretraining and has not been verified at large-scale pretraining.
2. **Gaussian assumption bias**: The theoretical derivation assumes a Gaussian marginal distribution for the logits; in practice, attention logits may be skewed or heavy-tailed.
3. **Dense attention only**: Integration with sparse attention mechanisms (e.g., sliding window + global attention) is not explored.
4. **Dependency on pp-RoPE**: The method performs poorly under standard RoPE and requires removal of low-frequency components, limiting its direct applicability to existing RoPE-based models.
5. **Combination with other long-context techniques**: Whether the method can be stacked with KV cache compression, ring attention, or similar approaches remains to be investigated.

## Related Work & Insights
- **LogN/SSMax** (Nakanishi et al.): Position-agnostic multiplicative scaling; the direct predecessor of this work—demonstrating the importance of position-dependent treatment.
- **pp-RoPE** (Barbero et al.): A positional encoding that removes low-frequency RoPE components; the best-performing companion to the proposed method.
- **ALiBi** (Press et al.): Static causal bias; generalizes well in perplexity but fails at retrieval—highlighting the trade-off among different long-context strategies.
- **Insight**: In attention mechanism design, the behavior of local and global context should be considered separately rather than handled with a uniform strategy across all positions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The scale-invariance perspective is novel; deriving attention transformations from natural image priors is a distinctive approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive baseline comparisons, ablation studies, and needle-in-a-haystack evaluation; deducted for limited model scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and self-consistent, with a complete logical chain from definitions to theorems to implementation.
- **Value**: ⭐⭐⭐⭐ — A minimal-implementation (single line of code) yet highly effective long-context solution with an extremely low barrier to deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)
- [\[NeurIPS 2025\] Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] UMoE: Unifying Attention and FFN with Shared Experts](umoe_unifying_attention_and_ffn_with_shared_experts.md)
- [\[NeurIPS 2025\] Tensor Product Attention Is All You Need](tensor_product_attention_is_all_you_need.md)

<!-- RELATED:END -->
