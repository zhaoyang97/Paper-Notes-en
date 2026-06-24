---
title: >-
  [Paper Note] Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Systems
description: >-
  [ICML 2026][LLM Efficiency][QKV Sharing] The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value), and Q=K=V (all three shared). It finds that for Language Modeling (LM), Q-K=V increases Perplexity (PPL) by only 3.1% while reducing the KV cache by 50%. This approach is orthogonal to GQA/MQA, enabling a total cache reduction of 87.5%–96.9%, providing quantifiable memory benefits for edge inference.
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "QKV Sharing"
  - "KV cache"
  - "GQA/MQA"
  - "Attention Sharing"
  - "Weight Tying"
date: 2026-05-08
content_hash: d6ce327698b68e85
---

# Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Systems

**Conference**: ICML 2026  
**arXiv**: [2606.04032](https://arxiv.org/abs/2606.04032)  
**Code**: https://github.com/anushamadan02/Do-Transformers-Need-3-Projections  
**Area**: Efficient LLM Inference / Attention Architecture / KV Cache Optimization  
**Keywords**: QKV Sharing, KV cache, GQA/MQA, Attention Sharing, Weight Tying

## TL;DR
The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value), and Q=K=V (all three shared). It finds that for Language Modeling (LM), Q-K=V increases Perplexity (PPL) by only 3.1% while reducing the KV cache by 50%. This approach is orthogonal to GQA/MQA, enabling a total cache reduction of 87.5%–96.9%, providing quantifiable memory benefits for edge inference.

## Background & Motivation

**Background**: Transformer self-attention is the standard for modern AI, but the expansion of context windows and the demand for real-time inference have made architectural efficiency a primary focus. Existing work generally follows two paths: sub-quadratic complexity (e.g., Performer/Linformer) and head sharing (e.g., GQA/MQA to reduce the number of KV heads). However, the fundamental question of whether the three QKV projections themselves are necessary has largely been ignored.

**Limitations of Prior Work**: CNNs and State Space Models use more unified internal representations, but Transformers have maintained independent Q, K, and V projections, representing a persistent redundancy. This redundancy significantly consumes memory, especially in Parameter-Efficient Fine-Tuning (PEFT) and during inference when cache resources are constrained. While Multi-head Latent Attention (MLA) compresses K/V, they remain functionally independent. Other works (linear attention, attention-free) replace the entire mechanism rather than maintaining the flexibility of attention.

**Key Challenge**: Maintaining the flexibility of attention usually requires independent Q, K, and V. Reducing projections to save memory typically incurs quality loss. A fine-grained systematic comparison is needed to determine which sharing schemes are lossless or acceptable.

**Goal**: (1) Systematically evaluate the performance of Q=K-V, Q-K=V, and Q=K=V on synthetic, vision, and language tasks; (2) Quantify KV cache savings and their orthogonal combination with GQA/MQA; (3) Provide architectural insights into which sharing schemes work and why.

**Key Insight**: This work extends the idea of weight tying (common in LM between input and output embeddings) to attention projections. It utilizes 2D positional encoding to address the symmetric attention problem introduced by Q=K.

**Core Idea**: Q-K=V (sharing K and V) maintains attention asymmetry while halving the KV cache, representing a quality-efficiency sweet spot; it is orthogonal to GQA/MQA. Q=K-V (sharing Q and K) makes attention symmetric and requires 2D positional encoding to recover performance; Q=K=V is the most aggressive but causes significant quality degradation in language tasks.

## Method

### Overall Architecture

Rather than proposing a single new model, the paper deconstructs the eight-year-old assumption that standard Transformers must have independent Q, K, and V projections. By fixing the attention backbone and varying only the projection sharing schemes, the authors quantify the quality costs and KV cache benefits across synthetic tasks, vision, and language modeling. Three schemes are enumerated: **Q=K-V** ($A = \mathrm{Softmax}(\alpha K K^\top) V$, where K is reused for Q), **Q-K=V** ($A = \mathrm{Softmax}(\alpha Q K^\top) K$, where K is reused for V), and **Q=K=V** ($A = \mathrm{Softmax}(\alpha K K^\top) K$, single projection). For schemes resulting in symmetric attention (Q=K-V and Q=K=V), an (X)+ variant with 2D positional encoding is introduced to re-inject directional information (used for non-causal tasks).

### Key Designs

**1. Q-K=V: Reusing K for V to Halve KV Cache with Minimal Quality Loss**

The primary memory bottleneck in LLM inference is the KV cache, which grows linearly with context length. Q-K=V employs weight tying between the Key and Value projection matrices ($V = K$). During inference, only K is stored; V is reused from K, cutting the KV cache by $50\%$. Since Q remains independent, the attention scores $Q K^\top$ stay asymmetric, preserving the directionality required for sequential tasks. The mechanism for its success is that keys and values can occupy similar representational spaces, and attention operates in a low-rank regime where using K as V does not significantly compress expressive power. Experiments show a 300M model trained on 10B SlimPajama tokens increases PPL by only $3.1\%$, and a 1.2B model combined with MQA increases PPL by only $1.06\%$. Compared to MLA, which compresses K/V and requires expansion during inference, Q-K=V achieves similar gains with a simpler equality constraint.

**2. (X)+: Restoring Directionality with 2D Positional Encoding**

Q=K-V causes the attention matrix to degenerate into a symmetric $K K^\top$, which is detrimental to sequence tasks that rely on directional dependencies. Symmetric attention is traditionally limited to Graph Neural Networks and relational reasoning. The (X)+ variant remedies this by constructing a fixed 2D sinusoidal positional encoding $P \in \mathbb{R}^{n \times n \times m}$. The symmetric attention map is broadcasted across $m$ channels, added to $P$, and then projected back via a $1 \times 1$ convolution to a 2D attention matrix. This reintroduces positional bias without abandoning projection sharing, similar to relative positional encoding used in Vision Transformers. Note that causal language models do not need this "patch" as they are already forced into asymmetry by causal masking.

**3. Orthogonality with GQA/MQA: Multiplicative Benefits**

GQA/MQA reduces cache through head sharing (GQA-$g$ shares $H$ query heads into $g < H$ KV heads), yielding a reduction ratio of $1 - g/H$. Q-K=V reduces cache via the projection dimension, which is orthogonal to head sharing. Both can be combined: in a Q-GQA-4 ($H=16, g=4$) setup, the cache of each group is halved again, resulting in a total reduction of $1 - g/(2H) = 87.5\%$. Q-MQA reaches $96.9\%$, approaching the theoretical limit for cache-based Transformers. Given that GQA/MQA are standards in models like Llama and Mistral, Q-K=V is a practical enhancement for industrial deployment.

The table below summarizes the computational, parameter, and cache costs relative to standard QKV:

| Variant | Computation | Parameters | KV Cache |
|---|---|---|---|
| QKV | $3nd^2$ | $3d^2$ | K + V |
| Q=K-V / Q-K=V | $2nd^2$ | $2d^2$ | K only (Q-K=V) |
| (Q=K-V)+ | $2nd^2 + n^2 m$ | $2d^2 + m$ | K + V |
| Q=K=V | $nd^2$ | $d^2$ | K only |

## Key Experimental Results

### Main Results: Synthetic tasks (Average of 5 tasks)

| Method | Reverse | Sort | Sub | Swap | Copy | Avg. |
|------|---------|------|-----|------|------|------|
| QKV | 0.698 | 0.971 | 1.0 | 0.588 | 1.0 | 0.851 |
| Q=K-V | 0.705 | 0.967 | 1.0 | 0.597 | 1.0 | 0.854 |
| (Q=K-V)+ | **0.718** | 0.963 | 1.0 | **0.671** | 1.0 | **0.870** |
| Q-K=V | 0.701 | 0.958 | 1.0 | 0.590 | 1.0 | 0.850 |
| Q=K=V | 0.514 | 0.939 | 1.0 | 0.446 | 1.0 | 0.780 |
| (Q=K=V)+ | 0.581 | 0.957 | 1.0 | 0.576 | 1.0 | 0.823 |

(Q=K-V)+ outperforms QKV using 2D pos encoding (0.870 vs 0.851). Q=K=V drops significantly but recovers to 0.823 with 2D pos.

### Main Results: Vision tasks (Average of 5 tasks)

| Method | MNIST | FMNIST | CIFAR-10 | CIFAR-100 | TinyImgNet | Anomaly | Avg. |
|------|-------|--------|----------|-----------|------------|---------|------|
| QKV | 0.981 | 0.887 | 0.663 | 0.363 | 0.229 | 0.942 | 0.767 |
| Q=K-V | 0.981 | 0.885 | 0.666 | 0.369 | 0.236 | 0.954 | 0.771 |
| (Q=K-V)+ | 0.982 | 0.884 | 0.662 | 0.366 | - | **0.966** | 0.772 |
| Q-K=V | 0.976 | 0.883 | 0.659 | 0.358 | - | 0.949 | 0.767 |
| Q=K=V | 0.978 | 0.877 | **0.672** | **0.376** | **0.266** | 0.933 | 0.767 |

In Vision, Q=K=V outperforms QKV on CIFAR/TinyImageNet, proving that symmetric attention is perfectly fine for non-causal tasks.

### Main Results: Language Modeling (300M parameters, SlimPajama 10B tokens)

| Model | Train Loss | Train PPL | Val Loss | Val PPL | Speed (tok/s) |
|---|---|---|---|---|---|
| QKV (Baseline) | 1.73 | 5.64 | 1.63 | 5.11 | 423k |
| **Q-K=V** | (~1.74) | (~5.70) | (~1.64) | (~5.27) | (~) |

For the 300M model, Q-K=V PPL increases by only 3.1% compared to QKV while halving cache. For the 1.2B model, MQA pairing yields a 1.06% PPL increase with a 97% cache reduction.

### Key Findings

- **Q-K=V is the quality-efficiency sweet spot for LM**: A 3.1% PPL increase in exchange for a 50% cache reduction is a favorable trade-off and simpler than MLA.
- **Symmetric attention is effective for vision/sets**: Q=K=V outperforms QKV on CIFAR/TinyImageNet, validating that sequential tasks require asymmetry while set/image tasks do not.
- **2D Pos Encoding recovers symmetry loss in non-causal tasks**: (Q=K-V)+ exceeds QKV on synthetic Reverse/Swap tasks, proving lost asymmetry can be compensated by spatial encoding.
- **Q-K=V is orthogonal and multiplicative to GQA/MQA**: Q-GQA-4 achieves 87.5% cache reduction; Q-MQA achieves 96.9%, enabling on-device LLMs.
- **Theoretical insight**: Q-K=V works because K and V occupy similar representational spaces and attention operates in a low-rank regime. Q=K-V fails for causal LM because symmetric $K K^\top$ breaks attention directionality.
- **Stable ranking at 1.2B scale**: Relative quality ranking remains stable as scale increases, suggesting conclusions generalize to industrial scales.

## Highlights & Insights

- **Filling a fundamental gap**: The necessity of three projections has been an overlooked question for eight years; the paper provides a systematic answer across 12 tasks—Q-K=V is a "free lunch" for LM.
- **Quantifiable cache benefit**: A 50% cache reduction translates directly to 2x context window or 2x concurrent users, offering real economic value for LLM serving.
- **Orthogonality**: Projection sharing and head sharing are distinct axes; Q-MQA’s 97% reduction provides a critical path for edge LLM deployment.
- **Asymmetric vs Symmetric Task Dependence**: Provides a clear decision tree: use Q-K=V (preserving asymmetry) for sequential tasks and Q=K=V + 2D pos for non-causal tasks.
- **Mechanistic Theoretical Explanation**: Success/failure is explained via "representational space sharing" and "low-rank regimes," providing deeper insights beyond just black-box data.
- **Elegant 2D Pos Encoding Trick**: Converting symmetric attention to asymmetric via spatial encoding is a reusable design pattern.

## Limitations & Future Work

- **Scale limit at 1.2B**: The largest model tested was 1.2B; it remains unverified if Q-K=V is still the sweet spot for 70B+ models where K-V coupling effects might be amplified.
- **Insufficient validation on long context**: Most experiments used ~2K context. Although Q-K=V is more valuable at 100K context, whether the attention patterns remain well-behaved was not fully tested.
- **Incomplete comparison with MLA**: While MLA is noted to use compressed latents, which is not equivalent to the hard equality of Q-K=V, a direct quality-efficiency Pareto comparison was not provided.
- **(X)+ variants limited to non-causal**: Causal LMs already enforce asymmetry, so (X)+ is not applicable. However, its potential for acceleration in prefix/non-causal blocks (like system prompt prefixes) remains unexplored.
- **Modern architectural differences**: Standard Transformers were used. The performance of Q-K=V in modern settings with SwiGLU, RMSNorm, or RoPE (as seen in Llama/Mistral) requires further validation.
- **Large-scale training stability**: While stable at 300M/1.2B, there is no data on whether K-V coupling causes training instability at 7B/30B scales.

## Related Work & Insights

- **vs MLA (DeepSeek-V2)**: MLA uses compressed latent vectors for small cache but expands at inference; Q-K=V uses hard equality for a simpler approach without expansion, though with potentially less expressivity.
- **vs GQA / MQA (Ainslie 2023, Shazeer 2019)**: Head sharing is an orthogonal axis; this paper proves Q-K=V + GQA/MQA are additive.
- **vs Linear Attention / Performer / Linformer**: Sub-quadratic complexity reductions often sacrifice attention flexibility; Q-K=V maintains the attention mechanism while reducing cache.
- **vs Weight Tying (Press & Wolf 2017)**: Extends the classic input/output embedding weight tying to attention projections.
- **vs Borji 2023**: This paper act as a follow-up and scale-up of the lead author's previous work.
- **Insights**: (1) There are unexplored weight tying dimensions in Transformer architectures; (2) Projection sharing and head sharing should be combined; (3) Decisions on symmetric vs asymmetric attention should be based on the task modality (sequential vs non-causal).

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic inquiry + mechanistic explanation are strong methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 tasks across 3 domains + dual scale (300M/1.2B) + GQA/MQA analysis; lacks 7B+ scale and long-context validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Pareto frontiers are clear, taxonomy is organized, and the decision tree is highly readable for engineers.
- Value: ⭐⭐⭐⭐⭐ Directly addresses KV cache bottlenecks for LLM deployment (50%-97% reduction) and is compatible with industry standards like GQA/MQA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Attention Is All You Need for KV Cache in Diffusion LLMs](../../ICLR2026/llm_efficiency/attention_is_all_you_need_for_kv_cache_in_diffusion_llms.md)
- [\[NeurIPS 2025\] Tensor Product Attention Is All You Need](../../NeurIPS2025/llm_efficiency/tensor_product_attention_is_all_you_need.md)
- [\[ICML 2026\] RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inference](rksc_reasoning-aware_kv_cache_sharing_and_confident_early_exit_for_multi-step_ll.md)
- [\[ICLR 2026\] Reasoning Language Model Inference Serving Unveiled: An Empirical Study](../../ICLR2026/llm_efficiency/reasoning_language_model_inference_serving_unveiled_an_empirical_study.md)
- [\[ICML 2026\] Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)

</div>

<!-- RELATED:END -->
