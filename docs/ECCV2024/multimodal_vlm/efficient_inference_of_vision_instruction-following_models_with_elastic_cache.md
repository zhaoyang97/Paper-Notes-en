---
title: >-
  [Paper Note] Efficient Inference of Vision Instruction-Following Models with Elastic Cache
description: >-
  [ECCV 2024][Multimodal VLM][KV Cache Management] Elastic Cache proposes a KV Cache management method tailored for multimodal instruction-following models. It adopts an importance-based cache merging strategy (rather than eviction) in the instruction encoding stage and a fixed-point eviction strategy in the output generation stage. With the "one sequence, two policies" design, it achieves high-efficiency inference at any acceleration ratio, delivering a 78% actual speedup with…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "KV Cache Management"
  - "Multimodal Large Language Models"
  - "Inference Acceleration"
  - "Cache Merging"
  - "Instruction Following"
date: 2026-05-08
content_hash: 707950a221b9a7cb
---

# Efficient Inference of Vision Instruction-Following Models with Elastic Cache

**Conference**: ECCV 2024  
**arXiv**: [2407.18121](https://arxiv.org/abs/2407.18121)  
**Code**: [GitHub](https://github.com/liuzuyan/ElasticCache)  
**Area**: Multimodal VLM  
**Keywords**: KV Cache Management, Multimodal Large Language Models, Inference Acceleration, Cache Merging, Instruction Following

## TL;DR

Elastic Cache proposes a KV Cache management method tailored for multimodal instruction-following models. It adopts an importance-based cache merging strategy (rather than eviction) in the instruction encoding stage and a fixed-point eviction strategy in the output generation stage. With the "one sequence, two policies" design, it achieves high-efficiency inference at any acceleration ratio, delivering a 78% actual speedup with only 20% (0.2) KV Cache budget while maintaining generation quality.

## Background & Motivation

**Background**: Multimodal instruction-following large models (such as LLaVA, Qwen-VL) require real-time responses in dialogue systems. However, during inference, KV Cache consumes a massive amount of GPU memory, sometimes even exceeding the memory of the model weights themselves, which limits the batch size and inference throughput.

**Limitations of Prior Work**:
   - **H2O**: Evicts low-frequency cache based on attention frequency, but it only works when the sequence length exceeds the cache capacity, and the frequency strategy is unfavorable to new tokens in the generation stage (new tokens have low frequencies and are easily evicted).
   - **StreamingLLM**: Uses a sliding window to retain recent cache, which can only accelerate extremely long sequences.
   - **Common issues of both**: (1) The acceleration ratio is bound to the cache capacity, making it impossible to accelerate sequences of arbitrary length; (2) directly discarding cache causes loss of contextual information, which severely hurts multimodal instruction-following capabilities.

**Key Insight**: The characteristics of the instruction encoding stage and the output generation stage are different, and distinct cache management strategies should be adopted for each.

## Method

### Overall Architecture

The core idea of Elastic Cache is **"One Sequence, Two Policies"** (one sequence, two policies):
- **Instruction encoding stage**: Global cache merging—merging unimportant KV vectors into important anchor points.
- **Output generation stage**: Fixed-point eviction—removing old cache at a fixed truncation position.

The entire approach is fully training-free, plug-and-play, and applicable to any multimodal instruction-following model.

### Key Designs

**1. Importance Metric**

- Measures KV cache importance based on attention score statistics.
- For the $j$-th attention head in the $i$-th layer, the importance of the $n$-th token is the sum of the columns of the attention matrix: $I_n^{i,j} = \sum_m A_{m,n}^{i,j}$
- Taking the average over all heads in the same layer: $I_n^i = \frac{1}{K}\sum_j \sum_m A_{m,n}^{i,j}$
- **Key Ablation Finding**: Layer-wise sum is significantly superior to moving average and mean, and per-head granularity is inferior to per-layer.

**2. Cache Merging Strategy (Instruction Encoding Stage)**

Traditional methods directly discard low-importance cache (eviction), which leads to irreversible loss of contextual information. Elastic Cache proposes **cache merging**:

- Given a retention ratio $\gamma$, Top-$N_I = \gamma \times T$ important tokens are selected as anchors.
- The sequence is partitioned into $N_I$ buckets centered around the anchors, with each bucket corresponding to the neighborhood of an anchor.
- All KV vectors within each bucket are averaged and stored in the cache: $\text{KV}_k = \frac{1}{|B_k|}\sum_{t \in B_k} kv_t$
- Bucket partitioning is independent for each layer (layer-wise strategy), but all heads in the same layer share the anchor locations.
- By controlling $\gamma$, an **arbitrary acceleration ratio** can be achieved, independent of the cache capacity upper limit.

**3. Fixed-Point Eviction Strategy (Output Generation Stage)**

- At each step in the generation stage, the KV of the new token is added, and the old cache at a fixed truncation position $N_{tl}$ is removed.
- **Design Motivation**: Initial instruction guidance and recently generated content are highly important, while intermediate earlier-generated content is less important.
- Experiments show that placing the truncation point near the length of the instruction sequence yields the best performance—retaining almost all instruction cache + recently generated cache.
- A fixed 25 caches are kept as the local distance parameter.

### Loss & Training

Elastic Cache is a **pure inference-phase method that requires no training**. Key designs include:
- No extra parameters, no fine-tuning required.
- The cache update strategy introduces only negligible computational overhead.
- Can be orthogonally combined with other compression methods such as quantization and distillation.

## Key Experimental Results

### Main Results

**PPL and ROUGE Evaluation (LLaVA-Description + MM-Vet)**:

| Method | KV Budget=0.5 PPL↓ | ROUGE↑ |
|------|:---:|:---:|
| Local (StreamingLLM) | 32.32 | ~0.50 |
| H2O | 7.94 | ~0.58 |
| **Elastic Cache** | **3.60** | **~0.67** |

At Budget=0.5, Elastic Cache improves PPL by 4.34 and ROUGE by 0.089 compared to H2O.

**GPT-4V Win Rate Evaluation**:

| Method | Budget=0.1 | Budget=0.2 | Budget=0.3 |
|------|:---:|:---:|:---:|
| **Elastic Cache** | **47.54%** | **46.63%** | **37.56%** |
| H2O | 38.55% | 35.26% | 30.26% |
| Local | 46.37% | 35.29% | 10.10% |

**Inference Speed Evaluation (Budget=0.2)**:

| Configuration | Latency Speedup | Throughput Gain |
|------|:---:|:---:|
| BS=8, 13B, 1024+512 | 33.8% | 52.6% |
| BS=16, 7B, 1024+512 | 43.8% | **77.9%** |
| BS=48, 7B, 624+256 | Full OOM | N/A |

### Ablation Study

| Ablation Dimension | Best Choice | PPL (Budget=0.5) |
|----------|----------|:---:|
| Eviction Location: Recent / Frequency / **Fixed-point** | Fixed-point | **3.60** vs 3.93/3.75 |
| Merging Strategy: Clustering / Discarding / **Merging** | Merging | **3.60** vs 3.61/3.68 |
| Attention Granularity: Shared / Head-wise / **Layer-wise** | Layer-wise | **3.60** vs 3.73/3.75 |
| Importance Metric: Moving Average / Mean / **Sum** | Sum | **3.60** vs 8.43/8.70 |

### Key Findings

- **Cache Merging is far superior to Cache Discarding**: The merging strategy that preserves contextual information exhibits clear advantages over direct discarding.
- **Different strategies for the two stages are crucial**: The combination of frequency-metric merging for instruction encoding and fixed-point eviction for generation is significantly superior to a unified strategy.
- **Sensitivity of Importance Metrics**: Layer-wise sum is the only effective metric; moving average and mean lead to a catastrophic decline in performance.
- **Competitors collapse at low budgets**: Local starts generating repetitive text at budget=0.5, while H2O generates extremely short responses.

## Highlights & Insights

- **Cache Merging vs. Cache Discarding** is the most central insight of this paper—unimportant cache still contains valuable contextual information.
- **One sequence, two policies** precisely matches the distinct characteristics of the two stages: instruction encoding and output generation.
- **Fully training-free**: No extra fine-tuning is required, making it applicable to any autoregressive multimodal model.
- **Arbitrary acceleration ratio**: Unconstrained by the upper limit of cache capacity, performing compression starting from the instruction encoding stage.
- Retains reasonable generation quality even under extreme compression (budget=0.2) while avoiding OOM.

## Limitations & Future Work

- The attention score-based importance metric may not always be the optimal cache optimization strategy.
- The joint effects with KV cache quantization and model distillation have not been explored.
- Evaluation is primarily based on PPL and ROUGE, lacking benchmarks on other downstream tasks.
- The location of the fixed truncation point still require hyperparameter tuning, lacking an adaptive mechanism.

## Related Work & Insights

- **H2O**: A frequency-driven KV cache eviction strategy, serving as a direct baseline comparison for Elastic Cache.
- **StreamingLLM**: Retains the most recent cache using a sliding window, suitable only for exceptionally long sequences.
- **Gist Tokens**: Learns to compress instruction tokens, which requires additional training.
- **Insights**: Cache management in multimodal models needs to distinguish the instruction and generation stages, with merging being superior to discarding.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ECCV 2024\] FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_high-resolution_vision-language_models.md)
- [\[ICLR 2026\] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks](../../ICLR2026/multimodal_vlm/mcif_multimodal_crosslingual_instruction-following_benchmark_from_scientific_tal.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](../../ACL2026/multimodal_vlm/efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)

</div>

<!-- RELATED:END -->
