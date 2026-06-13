---
title: >-
  [Paper Note] Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Schemes
description: >-
  [ICML 2026][LLM Efficiency][QKV Sharing] The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value)…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "QKV Sharing"
  - "KV cache"
  - "GQA/MQA"
  - "Attention Sharing"
  - "Weight Tying"
date: 2026-05-08
content_hash: 9dfc12a704bfb622
---

# Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Schemes

**Conference**: ICML 2026  
**arXiv**: [2606.04032](https://arxiv.org/abs/2606.04032)  
**Code**: https://github.com/anushamadan02/Do-Transformers-Need-3-Projections  
**Area**: LLM Efficient Inference / Attention Architecture / KV Cache Optimization  
**Keywords**: QKV Sharing, KV cache, GQA/MQA, Attention Sharing, Weight Tying

## TL;DR
The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value), and Q=K=V (all three shared). It finds that Q-K=V in Language Modeling (LM) increases Perplexity (PPL) by only 3.1% while reducing KV cache by 50%. This approach is orthogonal to GQA/MQA, allowing for a combined 87.5%-96.9% cache reduction, providing quantifiable memory benefits for edge inference.

## Background & Motivation

**Background**: Transformer self-attention is a staple of modern AI, but the expansion of context windows and requirements for real-time inference have made architectural efficiency a focal point. Existing work follows two paths: sub-quadratic complexity (Performer/Linformer) and head sharing (GQA/MQA to reduce KV heads). However, the fundamental question of whether the three QKV projections themselves are necessary has been largely ignored.

**Limitations of Prior Work**: CNNs and State Space Models use more unified internal representations, while Transformers maintain three independent Q/K/V projections, creating a persistent redundancy. This redundancy consumes significant memory, especially during PEFT and inference when cache is limited. Multi-head Latent Attention (MLA) compresses K/V but remains functionally independent; other works (linear attention, attention-free) replace the entire mechanism rather than maintaining attention's flexibility.

**Key Challenge**: Maintaining attention flexibility requires three independent Q/K/V projections, while saving memory requires pruning projections. A fine-grained systematic comparison is needed to determine which sharing schemes are lossy or lossless.

**Goal**: (1) Systematically evaluate the performance of Q=K-V, Q-K=V, and Q=K=V on synthetic, vision, and language tasks; (2) Provide KV cache savings figures and demonstrate orthogonality with GQA/MQA; (3) Offer architectural insights into which sharing schemes are viable and why.

**Key Insight**: Generalize the concept of weight tying (common in LM between input and output embeddings) to attention projections. Use 2D positional encoding to resolve the symmetric attention problem introduced by Q=K.

**Core Idea**: Q-K=V (sharing K and V) maintains attention asymmetry while halving the KV cache, serving as a quality-efficiency sweet spot; it is orthogonal to GQA/MQA. Q=K-V (sharing Q and K) makes attention symmetric, requiring 2D positional encoding to mitigate issues; Q=K=V is the most aggressive but suffers significant quality loss in language tasks.

## Method

### Overall Architecture

Three variants are explored: (1) **Q=K-V**: $A = \mathrm{Softmax}(\alpha K K^\top) V$, where K=Q but V is independent, resulting in a symmetric attention matrix $K K^\top$; (2) **Q-K=V**: $A = \mathrm{Softmax}(\alpha Q K^\top) K$, where K=V but Q is independent, maintaining asymmetric attention while only caching K; (3) **Q=K=V**: $A = \mathrm{Softmax}(\alpha K K^\top) K$, utilizing a single projection. The later variants are paired with 2D positional encoding in (X)+ versions to re-introduce asymmetry (used only for non-causal tasks).

### Key Designs

1.  **Q-K=V is the sweet spot for LM**:
    - **Function**: Halves KV cache, maintains attention asymmetry, and results in negligible quality loss.
    - **Mechanism**: A single projection matrix implements weight tying for $V = K$. During inference, the cache only stores K (V is reused from K), leading to a $50\%$ cache reduction. Attention remains $Q K^\top$ asymmetric because Q is independent. Theoretical insight: Quality does not drop drastically because K and V can occupy similar representational spaces and attention operates in a low-rank regime. A 300M model on SlimPajama 10B tokens showed a PPL increase of only $3.1\%$, while a 1.2B model with MQA showed a $+1.06\%$ increase.
    - **Design Motivation**: Cache is the primary memory bottleneck for LLM serving (especially for long context). Q-K=V provides an elegant solution with $50\%$ reduction and almost no quality loss, being simpler than MLA (hard equality instead of compressed latent expansion).

2.  **Q=K-V with 2D Positional Encoding for Symmetric Attention**:
    - **Function**: Replaces directional bias when attention $K K^\top$ becomes symmetric (due to Q=K), which is unfavorable for sequential tasks.
    - **Mechanism**: A fixed 2D sinusoidal positional encoding $P \in \mathbb{R}^{n \times n \times m}$ is constructed. The attention map is broadcasted to channels and added to $P$, then projected back to a 2D attention matrix via $1 \times 1$ conv. This aligns with relative positional encoding and vision Transformer 2D pos embedding. Causal LMs enforce asymmetry via causal masks; (X)+ is only used for non-causal tasks (vision, synthetic).
    - **Design Motivation**: Previously, symmetric attention appeared mostly in graph NNs and relational reasoning, and was avoided in sequential tasks where directionality is crucial. 2D positional encoding is an elegant way to "break symmetry while maintaining efficiency."

3.  **Orthogonality with GQA/MQA**:
    - **Function**: Projection sharing and head sharing represent different dimensions of efficiency; their combination yields multiplicative cache reduction.
    - **Mechanism**: GQA-$g$ shares $g < H$ KV heads for $H$ query heads, reducing cache by $1 - g/H$. Q-K=V further enforces $K = V$ within each GQA group, halving the cache again. Q-GQA-4 results in a total cache reduction of $1 - g/(2H) = 87.5\%$ (for $H=16, g=4$). Q-MQA reaches a cache reduction of $96.9\%$, approaching the theoretical limit for cache-based Transformers.
    - **Design Motivation**: MQA/GQA are widely deployed in PaLM/Llama/Mistral. This paper proves Q-K=V is an orthogonal complement, making it directly actionable for industrial deployment.

### Complexity Contrast Table

| Variant | Computation | Parameters | KV Cache |
|---|---|---|---|
| QKV | $3nd^2$ | $3d^2$ | K + V |
| Q=K-V / Q-K=V | $2nd^2$ | $2d^2$ | K only (Q-K=V) |
| (Q=K-V)+ | $2nd^2 + n^2 m$ | $2d^2 + m$ | K + V |
| Q=K=V | $nd^2$ | $d^2$ | K only |

Q-K=V results in a 33% reduction in computation/parameters and a 50% reduction in cache.

## Key Experimental Results

### Synthetic tasks (Average of 5 tasks)

| Method | Reverse | Sort | Sub | Swap | Copy | Avg. |
|------|---------|------|-----|------|------|------|
| QKV | 0.698 | 0.971 | 1.0 | 0.588 | 1.0 | 0.851 |
| Q=K-V | 0.705 | 0.967 | 1.0 | 0.597 | 1.0 | 0.854 |
| (Q=K-V)+ | **0.718** | 0.963 | 1.0 | **0.671** | 1.0 | **0.870** |
| Q-K=V | 0.701 | 0.958 | 1.0 | 0.590 | 1.0 | 0.850 |
| Q=K=V | 0.514 | 0.939 | 1.0 | 0.446 | 1.0 | 0.780 |
| (Q=K=V)+ | 0.581 | 0.957 | 1.0 | 0.576 | 1.0 | 0.823 |

(Q=K-V)+ outperforms QKV using 2D positional encoding (0.870 vs 0.851). Q=K=V shows a drop but recovers to 0.823 with 2D pos.

### Vision tasks (Average of 5 tasks)

| Method | MNIST | FMNIST | CIFAR-10 | CIFAR-100 | TinyImgNet | Anomaly | Avg. |
|------|-------|--------|----------|-----------|------------|---------|------|
| QKV | 0.981 | 0.887 | 0.663 | 0.363 | 0.229 | 0.942 | 0.767 |
| Q=K-V | 0.981 | 0.885 | 0.666 | 0.369 | 0.236 | 0.954 | 0.771 |
| (Q=K-V)+ | 0.982 | 0.884 | 0.662 | 0.366 | - | **0.966** | 0.772 |
| Q-K=V | 0.976 | 0.883 | 0.659 | 0.358 | - | 0.949 | 0.767 |
| Q=K=V | 0.978 | 0.877 | **0.672** | **0.376** | **0.266** | 0.933 | 0.767 |

In vision, Q=K=V outperforms QKV on CIFAR/TinyImageNet, proving symmetric attention is perfectly fine for non-causal tasks.

### Language Modeling (300M parameters, SlimPajama 10B tokens)

| Model | Train Loss | Train PPL | Val Loss | Val PPL | Speed (tok/s) |
|---|---|---|---|---|---|
| QKV (Baseline) | 1.73 | 5.64 | 1.63 | 5.11 | 423k |
| **Q-K=V** | (~1.74) | (~5.70) | (~1.64) | (~5.27) | (~) |

The 300M Q-K=V model increases PPL by only 3.1% compared to QKV while halving cache. The 1.2B model with MQA shows a 1.06% PPL increase with 97% cache reduction.

### Key Findings

- **Q-K=V is the quality-efficiency sweet spot for LM**: A 3.1% PPL increase for a 50% cache reduction is a favorable trade-off and simpler than MLA.
- **Symmetric attention is fine for vision/sets**: Q=K=V outperforms QKV on CIFAR/TinyImageNet, validating that sequential tasks require asymmetry while set/image tasks do not.
- **2D Pos Encoding saves symmetry in non-causal**: (Q=K-V)+ outperforms QKV on synthetic Reverse/Swap tasks, proving lost asymmetry can be recovered via spatial encoding.
- **Q-K=V is orthogonal and multiplicative with GQA/MQA**: Q-GQA-4 reaches 87.5% cache reduction; Q-MQA reaches 96.9%, enabling on-device LLMs.
- **Theoretical insight**: Q-K=V works because K and V share representation space and attention is in a low-rank regime. Q=K-V fails in causal LMs because it breaks attention directionality.
- **Ranking is stable at 1.2B scale**: Relative quality rankings remain constant as models scale, suggesting conclusions generalize to industrial scales.

## Highlights & Insights

- **Systematic comparison fills a fundamental gap**: The necessity of three QKV projections has been overlooked for years; this paper provides a cross-domain answer across 12 tasks, identifying Q-K=V as a "free lunch" for LMs.
- **Quantifiable cache benefit**: A 50% cache reduction translates directly to a 2× context window or 2× concurrent users, offering real economic value for LLM serving.
- **Orthogonality**: Projection sharing and head sharing do not conflict; combined Q-MQA provides a new path for edge LLM deployment with 97% cache savings.
- **Task-dependent Asymmetry vs Symmetry**: Sequential tasks should use Q-K=V (preserving asymmetry), while non-causal tasks can use Q=K=V + 2D pos, providing a clear decision tree.
- **Non-hollow theoretical explanation**: Insights from K/V sharing and the low-rank regime explain why Q-K=V works while Q=K-V does not, providing mechanism-level clarity.

## Limitations & Future Work

- **Scale limit at 1.2B**: The largest model tested is 1.2B; whether Q-K=V remains a sweet spot for 70B+ models is unverified, as K-V coupling effects might be amplified.
- **Insufficient validation on long context**: Experiments used ~2K context; the value of Q-K=V is greater for 100K context, but attention patterns were not fully tested for such lengths.
- **Incomplete comparison with MLA**: MLA uses compressed latents, which is not equivalent to Q-K=V's hard equality; a direct Pareto comparison was missing.
- **(X)+ variants only for non-causal**: While causal LMs enforce asymmetry, the potential for (X)+ in prefix or non-causal blocks (e.g., system prompt prefixes) was not explored.
- **Modern architecture differences**: Experiments used standard Transformers; the effectiveness of Q-K=V with SwiGLU, RMSNorm, or RoPE (standard in Llama/Mistral) remains unverified.

## Related Work & Insights

- **vs MLA (DeepSeek-V2)**: MLA compresses K/V into a latent vector and expands it during inference; Q-K=V uses hard equality to cache K directly without expansion, making it simpler but slightly less expressive.
- **vs GQA / MQA (Ainslie 2023, Shazeer 2019)**: Head sharing is an orthogonal axis; this paper proves Q-K=V and GQA/MQA are stackable.
- **vs Linear Attention / Performer / Linformer**: Sub-quadratic complexity reduces overhead but sacrifices attention flexibility; Q-K=V maintains the attention mechanism while reducing cache.
- **vs Weight Tying (Press & Wolf 2017)**: Extends the classic input/output embedding sharing concept to attention projections.
- **Insight**: (1) There are unexplored dimensions of weight tying in Transformer architectures that warrant systematic study; (2) Projection sharing and head sharing are orthogonal and should be used in combination.

## Rating

- Novelty: ⭐⭐⭐⭐ Addresses a systematic question with mechanism-level explanations; high value in combined application.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 tasks across 3 domains + dual scale (300M/1.2B) + orthogonal analysis; lacks 7B+ scale and long-context validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear Pareto frontiers, organized taxonomy, and provides a "when-to-use" decision tree.
- Value: ⭐⭐⭐⭐⭐ Directly addresses KV cache bottlenecks for LLM serving (50%-97% reduction), compatible with industry standards, and provides open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)
- [\[NeurIPS 2025\] Tensor Product Attention Is All You Need](../../NeurIPS2025/llm_efficiency/tensor_product_attention_is_all_you_need.md)
- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](../../ICLR2026/llm_efficiency/universe_routing_why_self-evolving_agents_need_epistemic_control.md)
- [\[ICLR 2026\] Efficient Resource-Constrained Training of Transformers via Subspace Optimization](../../ICLR2026/llm_efficiency/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)
- [\[NeurIPS 2025\] Constant Bit-Size Transformers Are Turing Complete](../../NeurIPS2025/llm_efficiency/constant_bit-size_transformers_are_turing_complete.md)

</div>

<!-- RELATED:END -->
