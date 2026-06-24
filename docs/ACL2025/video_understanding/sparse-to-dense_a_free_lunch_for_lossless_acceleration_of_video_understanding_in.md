---
title: >-
  [Paper Note] Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs
description: >-
  [ACL 2025][Video Understanding][Video Large Language Models] Based on the observation of attention score sparsity in Video-LLMs, this paper proposes the Sparse-to-Dense (StD) decoding strategy. It uses a top-K sparse attention model as a draft model to rapidly generate candidate tokens, which are then verified in parallel by a dense model, achieving up to a 1.94× lossless acceleration without requiring additional training or architectural modifications.
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Video Large Language Models"
  - "Speculative Decoding"
  - "Sparse Attention"
  - "KV Cache"
  - "Lossless Acceleration"
date: 2026-05-08
content_hash: e763f3d41412c470
---

# Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs

**Conference**: ACL 2025  
**arXiv**: [2505.19155](https://arxiv.org/abs/2505.19155)  
**Code**: None  
**Area**: Video Understanding / LLM Inference Acceleration  
**Keywords**: Video Large Language Models, Speculative Decoding, Sparse Attention, KV Cache, Lossless Acceleration

## TL;DR

Based on the observation of attention score sparsity in Video-LLMs, this paper proposes the Sparse-to-Dense (StD) decoding strategy. It uses a top-K sparse attention model as a draft model to rapidly generate candidate tokens, which are then verified in parallel by a dense model, achieving up to a 1.94× lossless acceleration without requiring additional training or architectural modifications.

## Background & Motivation

**Background**: Video Large Language Models (Video-LLMs) handle video understanding tasks by representing videos as sequences of image frames, achieving excellent performance on tasks such as video QA and captioning. However, a one-hour video sampled at a 5-second interval generates 720 frames, which can translate into 141,120 visual tokens in VILA, leading to extremely high inference latency.

**Limitations of Prior Work**: The autoregressive decoding mechanism of Video-LLMs requires each new token to attend to all preceding tokens. As the KV cache continuously grows, frequent memory accesses put huge pressure on bandwidth. Existing solutions include KV cache compression and quantization; however, these methods introduce distribution shifts between training and inference, degrading model performance.

**Key Challenge**: To achieve lossless acceleration, the output distribution of the model cannot be modified. Speculative decoding can satisfy this requirement, but it typically requires an additional draft model, which is excessively costly for Video-LLMs.

**Goal**: Design a plug-and-play, training-free, and lossless acceleration method for Video-LLMs.

**Key Insight**: The authors observe that attention scores in Video-LLMs display significant sparsity during decoding—retaining only top-K KV caches maintains about 95% next-token prediction accuracy. This implies that a sparse attention model can serve as a natural draft model, sharing parameters with the original model without requiring extra GPU memory.

**Core Idea**: Use the inherent attention sparsity of the Video-LLM to construct a training-free draft model, achieving lossless speculative decoding acceleration through a "sparse proposal + dense verification" mechanism.

## Method

### Overall Architecture

StD consists of two modules: (1) a Sparse Model, which uses top-K attention to rapidly generate $\gamma$ candidate tokens autoregressively, and (2) a Dense Model (the original Video-LLM), which utilizes the complete KV cache to verify these candidates in parallel. Both models share the exact same architecture and parameters, differing only in how attention is computed—the sparse model loads a selected subset of the KV cache, whereas the dense model uses the complete cache. Once verified, the matched tokens along with a bonus token are appended to the sequence for the next proposal-verification cycle.

### Key Designs

1. **Text-Guided Visual KV Cache Selection**:

    - **Function**: Select the $K$ most critical KV cache pairs from a large volume of visual tokens.
    - **Mechanism**: Since the number of visual tokens ($m_v$) is much larger than the number of text tokens ($m_t$), the visual KV cache is prioritized for compression. During the prefill stage, the average attention scores of text tokens $X_t$ toward visual tokens $X_v$ are analyzed to select the top-K visual KV pairs with the highest attention scores for each layer $l$: $\text{Cache}_s[l] = \text{argTopK}_{x \in X_v}\big(\frac{1}{m_t}\sum_{\hat{x} \in X_t} A_l(\hat{x}, x)\big)$. For the GQA architecture, selection is performed by directly summing up attention scores within each group.
    - **Design Motivation**: Top-K selection is executed only once during the prefill stage, avoiding the additional computational overhead of dynamic selection during the decoding stage. It utilizes text token attention as a signal because text tokens represent the intent of the user query.

2. **Sparse-Dense Collaborative Decoding Workflow**:

    - **Function**: Core inference loop to achieve lossless acceleration.
    - **Mechanism**: The sparse model autoregressively generates $\gamma$ candidate tokens (accessing only $K+m_t$ KV caches), and the dense model verifies all $\gamma$ candidates in parallel in a single step (reading the complete $m_v + m_t$ KV cache but requiring only one I/O). Once the verification identifies the first $n$ matching tokens, these matched tokens, together with an additional bonus token $\hat{x}_{n+m}$, are appended to the sequence to form the context for the next round.
    - **Design Motivation**: The sparse model is fast but prone to errors, whereas the dense model is slower but accurate. Combining both through the speculative decoding framework accepts at least 1 token (the bonus token) and at most $\gamma+1$ tokens per round.

3. **I/O Complexity Optimization Analysis**:

    - **Function**: Theoretically prove under what conditions StD yields speedup.
    - **Mechanism**: The average I/O per token for StD in each round is $\frac{\gamma \times (K + m_t) + m_v + m_t}{\alpha \times \gamma}$, where $\alpha$ is the acceptance rate. Compared to $m_v + m_t$ in original decoding, StD is superior when $\alpha > (K + m_t)/(m_v + m_t) + \gamma^{-1}$. Because of the concentrated nature of attention, $K$ can be much smaller than $m_v$, making this condition easy to satisfy.
    - **Design Motivation**: Provides theoretical guarantees showing that StD is almost guaranteed to accelerate in typical Video-LLM scenarios where visual tokens far outnumber text tokens ($m_v > 10000$, $m_t \approx 100$).

### Loss & Training

StD is a completely training-free method. Hyperparameter settings: total KV cache amount $K + m_t = 1024$, candidate token count $\gamma = 9$, batch size = 8. Transforming the original Video-LLM into the sparse version requires only about 20 lines of code.

## Key Experimental Results

### Main Results

| Model/Method | MLVU Acc/Speedup | VideoMME-s Acc/Speedup | VideoMME-m Acc/Speedup | VideoMME-l Acc/Speedup |
|----------|-----------------|----------------------|----------------------|----------------------|
| LLaVA-OV-7B + LayerSkip | 10.0/0.47× | 5.6/0.33× | 8.1/0.46× | 4.8/0.44× |
| LLaVA-OV-7B + Streaming | 34.7/1.34× | 36.4/1.38× | 41.0/1.51× | 36.2/1.45× |
| LLaVA-OV-7B + **StD** | **47.8/1.72×** | **51.8/1.82×** | **52.1/1.83×** | **52.9/1.59×** |
| Qwen2-VL-7B + LayerSkip | 5.2/0.63× | 3.7/0.59× | 4.9/0.55× | 5.7/0.55× |
| Qwen2-VL-7B + Streaming | 53.9/1.61× | 52.9/1.32× | 59.2/1.36× | 59.6/1.36× |
| Qwen2-VL-7B + **StD** | **66.1/1.94×** | **71.8/1.71×** | **73.4/1.62×** | **81.8/1.70×** |

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| Top-K ratio variation | Acceptance rate ~92% at K=512, ~88% at K=256 | Retaining a small portion of the KV cache is sufficient to maintain a high acceptance rate |
| $\gamma=5$ vs $\gamma=9$ vs $\gamma=13$ | $\gamma=9$ is the optimal trade-off point | An excessively large $\gamma$ leads to low acceptance rates, while a too-small $\gamma$ wastes the parallel verification advantage |
| LayerSkip as draft | Extremely low acceptance rate (5-10%) | Layer skipping causes excessive distribution shifts |
| Streaming as draft | Moderate acceptance rate (35-60%) | Streaming attention retention is less effective than top-K selection |

### Key Findings

- **The sparse attention draft model significantly outperforms other draft strategies**: LayerSkip causes extremely low acceptance rates or even slows down inference due to severe distribution shifts; Streaming relies on a fixed window selection, which is less effective than the text-guided top-K selection in StD.
- **StD achieves more pronounced acceleration on long videos**: The greater the number of visual tokens, the larger the speed advantage of the sparse model relative to the dense model.
- **The lossless guarantee is the core selling point**: Since the verification phase utilizes the full KV cache, the output distribution is completely identical to the original model, ensuring zero performance degradation.
- **Average acceptance rate of 62.2%**: Significantly higher than LayerSkip and offers a substantial advantage over Streaming.

## Highlights & Insights

- **Elegant design of "self-as-draft"**: Leveraging the inherent attention sparsity of the model eliminates the need to train a separate draft model. Sharing parameters avoids increasing GPU memory, representing the best practice for speculative decoding in multimodal LLMs.
- **Plug-and-play with 20 lines of code**: Extremely low engineering barrier allows immediate deployment to any Transformers-based Video-LLM.
- **Text-guided visual cache selection**: Utilizing the attention patterns of text queries to select key visual tokens is much more effective than random or position-based selection, an idea that can be transferred to other multimodal scenarios.

## Limitations & Future Work

- **The complete KV cache still needs to be stored in GPU memory**: Although it reduces I/O access, it does not reduce memory footprint, which remains a bottleneck when processing ultra-long videos.
- **Single-video granularity**: Sparsification is only applied to frame-level visual tokens, without considering temporal redundancy across frames.
- **Only support for greedy/sampling decoding**: Support for complex decoding strategies like beam search is not discussed.
- **Future directions**: Partially offloading the KV cache to CPU memory to bypass HBM bottlenecks using the CPU's larger capacity; extending to long Chain-of-Thought Video-LLMs such as QvQ.

## Related Work & Insights

- **vs MagicDec (Chen et al., 2024a)**: MagicDec employs streaming attention as a draft model for textual LLMs. This paper extends the sparse attention scheme to Video-LLMs, leveraging the sparsity of visual tokens to achieve a higher speedup ratio.
- **vs EAGLE / Medusa**: These methods require training additional draft heads, which is costly and may not seamlessly apply to multimodal models; StD is entirely training-free.
- **vs FastV / VidCompress**: These methods reduce the number of visual tokens via token pruning/compression but introduce information loss; StD remains completely lossless.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using attention sparsity as a "free" draft model is clean and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete experiments covering two mainstream Video-LLMs, two benchmarks, and three baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression from observation → theory → method → experiments.
- Value: ⭐⭐⭐⭐ Plug-and-play, lossless acceleration offers high practical value, with highly generalizable research ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](../../ICCV2025/video_understanding/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](../../CVPR2025/video_understanding/ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](../../CVPR2025/video_understanding/dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)

</div>

<!-- RELATED:END -->
