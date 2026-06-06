---
title: >-
  [Paper Note] APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention
description: >-
  [ACL2026][Video Understanding][Long-Video Understanding] APB-V accelerates long-video LMM inference using sequence-parallelism-aware approximate attention and system-level load balancing. While retaining full visual embe…
tags:
  - "ACL2026"
  - "Video Understanding"
  - "Long-Video Understanding"
  - "Sequence Parallelism"
  - "Approximate Attention"
  - "Multi-GPU Inference"
  - "KV Compression"
date: 2026-05-08
content_hash: a06dcba7107f949c
---

# APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention

**Conference**: ACL2026  
**arXiv**: [2601.21444](https://arxiv.org/abs/2601.21444)  
**Code**: https://github.com/thunlp/APB  
**Area**: Video Understanding / Multimodal Reasoning Acceleration  
**Keywords**: Long-Video Understanding, Sequence Parallelism, Approximate Attention, Multi-GPU Inference, KV Compression  

## TL;DR
APB-V accelerates long-video LMM inference using sequence-parallelism-aware approximate attention and system-level load balancing. While retaining full visual embeddings, it achieves 12.72×, 1.70×, and 1.18× speedups relative to FlashAttn, ZigZagRing, and APB, respectively, under a 64-frame 1440p setting without significant performance loss.

## Background & Motivation
**Background**: Long-video understanding relies on Large Multimodal Models (LMMs) to encode massive frames into visual tokens, which are then fed into long-context LLM backbones. As video duration, resolution, and frame counts increase, the costs of visual encoding, attention, and Feed-Forward Networks (FFN) during the prefill stage grow rapidly.

**Limitations of Prior Work**: Existing methods primarily fall into two categories. One category optimizes attention or KV cache, but usually only alleviates a portion of the LLM backbone burden and cannot handle visual encoding and FFN costs. The other category explicitly compresses input tokens (e.g., token pruning or pooling), which reduces computation but risks losing fine-grained video evidence. The paper notes that SlowFast, for instance, incurs an approximately 25% accuracy drop despite achieving less than 3× speedup in experiments.

**Key Challenge**: Long-video inference requires more computational resources for longer inputs while prohibiting crude compression that sacrifices key frames and detailed evidence. Single-GPU optimizations face an efficiency-performance trade-off, while multi-GPU sequence parallelism encounters communication and load imbalance bottlenecks.

**Goal**: APB-V aims to resolve both efficiency and performance issues by "increasing parallel computation + suppressing quadratic attention costs." Instead of compressing visual embeddings, it performs approximate attention, communication compression, and system-level scheduling across multiple GPUs or hosts.

**Key Insight**: The authors observe that long-video scenarios are naturally suited for parallelism: frame-level visual encoding is independent, and LLM input sequences can be split into blocks. However, exact sequence parallelism is communication-heavy. Thus, each host should only exchange critical KV pairs that subsequent queries truly require.

**Core Idea**: Use local KV compression and passing blocks to approximate global attention. Only key context blocks are transferred between hosts. Simultaneously, multi-GPU performance is unleashed through frame parallelism, ZigZag load balancing, fused forward passes, and communication-computation overlapping.

## Method

### Overall Architecture
APB-V assumes each host maintains a complete LMM replica. Input videos are distributed by frame to different hosts for parallel encoding, followed by an AllGather operation to obtain full video embeddings. Next, the sequence is partitioned into an anchor block, query blocks, and multiple context blocks. Context blocks are allocated to $2H$ virtual hosts, then mapped to $H$ physical hosts to balance computation. In each attention layer, each host retains only the crucial KV pairs of its local context as passing blocks, used alongside the anchor, local context, and query to perform approximate attention.

Ultimately, APB-V does not reduce the number of input frames or visual tokens but minimizes the volume of remote KV pairs each query must access while overlapping communication and computation to improve throughput in Time To First Token (TTFT) scenarios.

### Key Designs
1.  **Frame Parallelism and Context Splitting**:
    - **Function**: Distributes visual encoding and long-sequence prefill across multiple hosts.
    - **Mechanism**: Each host encodes a subset of video frames, then aggregates visual embeddings via AllGather. The sequence is split into an initial anchor block $B_a$, a final query block $B_{qr}$, and remaining context blocks $B^{(h)}$. The anchor provides a global prefix, the query represents the question to be answered, and the context contains the primary video evidence.
    - **Design Motivation**: Visual encoding for long videos is naturally frame-parallel; combining it with sequence parallelism avoids overloading a single GPU with all frames and tokens.

2.  **Sequence-Parallelism-aware Approximate Attention**:
    - **Function**: Reduces inter-host attention computation and communication while preserving long-range dependencies.
    - **Mechanism**: Each virtual host uses query-to-context attention scores to select the $l_p$ most important KV pairs from the local context, forming compressed essential KV pairs. These are shared as passing blocks via AllGather. Attention for context blocks considers the anchor, local context, and compressed passing blocks. Query block results are merged across hosts using the log-sum-exp (lse) from FlashAttn.
    - **Design Motivation**: Not all historical KV pairs are relevant to the final question. Transmitting only query-related key KV pairs preserves long-range dependencies better than StarAttn while being more efficient than exact sequence parallelism.

3.  **System-level Load Balancing and Communication Hiding**:
    - **Function**: Implements the approximate attention design as a high-throughput multi-GPU system.
    - **Mechanism**: Visual load balancing allocates frames via $F^{(h)}=\lfloor F/H\rfloor+\mathbb{I}[h<F\bmod H]$. Fused context-query forward passes avoid memory-bound issues with short queries. ZigZag mapping places the $h$-th and $(2H-1-h)$-th virtual hosts on the same physical host to balance passing block lengths. Overlapped communication allows passing block transmission and attention computation to run in parallel.
    - **Design Motivation**: Approximate attention algorithms can be slowed by query forward overhead, communication latency, and load imbalance if system details are neglected. A significant contribution of APB-V is the codesign of algorithmic approximation and system optimization.

### Loss & Training
APB-V is an inference acceleration framework; it does not train new LMMs or introduce task losses. In experiments, the retaining heads for the APB baseline were trained on NextQA. APB-V itself mainly uses hyperparameters to control anchor length ($l_a=n/64$) and passing length ($l_p=n/128$), maintaining a computational load similar to APB across 8 physical hosts and $2H$ virtual hosts.

## Key Experimental Results

### Main Results
APB-V was evaluated on VNBench and LongVideoBench using InternVL3-2B, Qwen2.5VL-3B, and Qwen2.5VL-7B. Key findings indicate that performance does not drop significantly.

| Dataset / Model | FullAttn | APB | APB-V | Conclusion |
|-----------------|----------|-----|-------|------------|
| VNBench / InternVL3-2B Overall | 44.89 | 41.11 | 43.26 | APB-V is close to FullAttn and superior to APB |
| VNBench / Qwen2.5VL-3B Overall | 52.81 | 43.93 | 50.67 | Retains most accuracy; far better than token pruning |
| VNBench / Qwen2.5VL-7B Overall | 58.44 | 49.93 | 56.22 | Small performance loss on synthetic long-video tasks |
| LongVideoBench / InternVL3-2B | 55.35 | 55.20 | 55.42 | APB-V is slightly higher than FullAttn |
| LongVideoBench / Qwen2.5VL-7B | 58.38 | 59.16 | 59.76 | APB-V outperforms APB and FullAttn on real videos |

Regarding speed, when Qwen2.5-VL-3B processes a 64-frame 1440p video, APB-V achieves 12.72×, 1.70×, and 1.18× speedups compared to FlashAttn, ZigZagRing, and APB, respectively. System ablations confirm that each optimization contributes significantly.

### Ablation Study
| Configuration | 16-frame req/s | 32-frame req/s | 56-frame req/s | Note |
|---------------|----------------|----------------|----------------|------|
| APB-V | 1.846 | 0.916 | 0.471 | Full system (fastest) |
| -O | 1.827 | 0.911 | 0.470 | Without comm-compute overlap |
| -O-F | 1.646 | 0.854 | 0.450 | Without fused forward |
| -O-F-Z | 1.618 | 0.813 | 0.415 | Without ZigZag (unbalanced) |
| -O-F-Z-V | 0.381 | 0.189 | 0.107 | Without system optimizations (~4× slower) |
| FlashAttn | 0.226 | 0.092 | 0.042 | Single-GPU exact attention (slowest) |

### Key Findings
- For synthetic long-video tasks, token pruning methods (e.g., SlowFast) significantly damage fine-grained capabilities like retrieval/counting. Ours avoids this by retaining full visual embeddings.
- On real-world long videos, APB-V achieves an Overall score of 59.76 on Qwen2.5VL-7B, higher than FullAttn (58.38) and APB (59.16), suggesting that approximate attention can achieve a better trade-off via embedding retention.
- Passing blocks are more critical than anchor blocks: in counting sub-tasks, removing passing blocks drops accuracy from 30.00 to 17.33, while removing anchors drops it to 25.33.
- Superior scalability: At $H=8$, APB-V reaches 6.171/2.013 req/s on 720p 16/56-frame videos, outperforming ZigZagRing (5.595/1.666) and APB (4.891/1.766).

## Highlights & Insights
- Instead of "watching less video," the paper retains visual tokens and approximates at the attention layer. This is vital for long-video QA, where answers may hide in brief subtitles, actions, or objects.
- APB-V co-solves algorithmic and system issues: it addresses the memory-bound nature of short query forwards, passing block imbalance, and cross-host communication latency.
- Case studies show that segments containing the target information (e.g., "secret word is Nick") are more frequently selected for passing blocks, proving that query-aware KV selection effectively propagates relevant evidence.

## Limitations & Future Work
- APB-V is primarily designed for decoder-only Transformer-based LMMs; it is incompatible with convolutional or non-standard architectures.
- The method depends on multi-GPU inference and degrades to FlashAttn on a single GPU, making it less suitable for resource-constrained deployments.
- The focus is on TTFT/prefill time for long-video applications (e.g., surveillance, autonomous driving); benefits for multi-turn interaction or decoding phases require further analysis.
- Passing and anchor lengths remain hyperparameters. Performance drops when passing length is too small, indicating a frontier between compression and evidence retention.

## Related Work & Insights
- **vs FlashAttn / ZigZagRing**: FlashAttn is single-GPU exact; ZigZagRing is exact sequence parallelism. APB-V trades exactness for better scalability.
- **vs SlowFast / Token Pruning**: SlowFast speeds up by reducing visual tokens but loses fine-grained evidence. Ours is better for precise retrieval in long-video QA.
- **vs StarAttn**: StarAttn avoids inter-host communication but lacks long-range dependencies; APB-V preserves compressed communication to allow key KV transmission.
- **Insight**: The core of long-context multimodal reasoning is not just reducing tokens, but allowing "remote evidence relevant to the query" to flow across devices at low cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining approximate attention with sequence parallelism is not brand new, but the systematic design for long-video LMMs is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two benchmarks, three LMMs, multiple resolutions/frames/hosts, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear architectural diagrams and system breakdowns, though notation is dense in places.
- Value: ⭐⭐⭐⭐⭐ highly practical for multi-GPU long-video LMM deployment, particularly for TTFT acceleration of high-resolution, long-frame sequences.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](../../CVPR2026/video_understanding/vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](../../CVPR2026/video_understanding/autogaze_attend_before_attention_efficient_video.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)

</div>

<!-- RELATED:END -->
