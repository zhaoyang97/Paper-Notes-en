---
title: >-
  [Paper Note] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding
description: >-
  [ICLR 2026][Video Understanding][Long video understanding] This paper proposes FLoC, a visual token compression framework based on the facility location function. Through submodular optimization…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "Long video understanding"
  - "token compression"
  - "facility location"
  - "submodular function optimization"
  - "training-free"
date: 2026-05-08
content_hash: aac35dc69779aace
---

# FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding

**Conference**: ICLR 2026
**arXiv**: [2511.00141](https://arxiv.org/abs/2511.00141)  
**Code**: N/A  
**Area**: Video Understanding / Visual Token Compression
**Keywords**: Long video understanding, token compression, facility location, submodular function optimization, training-free

## TL;DR

This paper proposes FLoC, a visual token compression framework based on the facility location function. Through submodular optimization, FLoC efficiently selects a token subset that is both representative and diverse under a given budget, enabling training-free, model-agnostic, and query-agnostic token compression for long video understanding.

## Background & Motivation

**Token explosion in long videos**: As video length increases, the number of visual tokens grows explosively, far exceeding the context window limits of large multimodal models (LMMs) (typically 4K–32K tokens), severely constraining long video understanding capabilities.

**Limitations of existing compression methods**:
- Sampling/pooling methods ignore semantic importance and discard information indiscriminately.
- Clustering methods (e.g., k-means) primarily select representatives from dense regions, tending to miss sparse but important tokens (e.g., small objects or fine-grained text in a scene).
- Query-aware methods require advance knowledge of the query, lack flexibility for general scenarios, and require re-compression for each new query.
- Trainable methods depend on specific model architectures and large amounts of annotated data.

**Tension between representativeness and diversity**: Simple scenes contain abundant redundant tokens, yet sparse but critical visual cues (e.g., key tokens in a scene of searching for keys) appear infrequently yet are crucial. Both representativeness and diversity must be simultaneously ensured.

**Practical deployment requirements**: Applications such as CCTV surveillance, smart glasses, and mobile robots demand efficient, general-purpose, plug-and-play compression solutions.

## Method

### Overall Architecture

FLoC operates in two steps: (1) the input video is divided into temporal blocks, within each of which the facility location function is applied to select a token subset; (2) the selected token subset is concatenated with the text prompt and fed into a video LMM. The entire process requires no training, no prior knowledge of the query, and integrates seamlessly with any video LMM.

### Key Designs

**1. Facility Location Function**

- **Function**: Models token selection as a submodular optimization problem $S^* = \arg\max_{S \subseteq V, |S| \leq K} f(S)$.
- **Mechanism**: The objective function is $f(S) = \sum_{v \in V} \max_{u \in S} \text{sim}(v, u)$, where sim denotes cosine similarity. Intuitively, for each original token, the most similar token in the subset is identified; a higher total similarity indicates better coverage of the full token set by the subset.
- **Design Motivation**: The facility location function naturally balances representativeness (selected tokens should cover all original tokens) and diversity (redundant selections are discouraged), and its submodularity guarantees an approximation ratio of $(1-1/e) \approx 0.632$.

**2. Lazy Greedy Algorithm**

- **Function**: Exploits the diminishing returns property of submodular functions to avoid unnecessary recomputation of marginal gains via a priority queue.
- **Mechanism**: A priority queue of marginal gain upper bounds is maintained for all candidate tokens. At each step, the candidate with the highest upper bound is popped and its exact marginal gain is recomputed; if this value still exceeds the upper bounds of all remaining candidates, it is immediately selected without evaluating others. Otherwise, the upper bound is updated and the candidate is reinserted into the queue.
- **Design Motivation**: The naïve greedy algorithm has complexity $O(nK)$; in practice, lazy greedy achieves speedups of an order of magnitude, enabling real-time processing of long videos.

**3. Temporal Blocking**

- **Function**: Divides the video into small temporal blocks, within each of which token selection is performed independently.
- **Mechanism**: Block-wise selection preserves temporal locality while reducing the search space for each optimization step.
- **Design Motivation**: Improves computational efficiency and naturally supports future streaming processing scenarios where tokens are accumulated in a buffer before processing.

**4. Training-Free, Model-Agnostic, and Query-Agnostic Design**

- **Function**: The entire compression process requires no training, is not tied to any specific model, and does not require prior knowledge of user queries.
- **Mechanism**: Selection is based solely on cosine similarity between token embeddings, fully decoupled from both the upstream visual encoder and the downstream LMM.
- **Design Motivation**: Maximizes generality and deployment flexibility. After a single compression pass, only the compressed tokens need to be stored; multiple queries can be answered without re-compression, saving both computation and memory.

### Loss & Training

- FLoC is a **training-free** method; no loss functions or backpropagation are involved.
- The lazy greedy algorithm is run at inference time to select the token subset.
- The compression ratio can be set flexibly (experiments cover $2^{-3}$ to $2^{-5}$).

## Key Experimental Results

### Main Results

Compression comparison on Qwen2.5-VL-7B (compression ratio $2^{-3}$):

| Method | Video-MME | MLVU | LVB | EgoSchema | Avg. |
|--------|-----------|------|-----|-----------|------|
| No compression (ratio=1) | 66.33 | 70.31 | 60.51 | 61.40 | 64.64 |
| TS-LLaVA | 61.15 | 67.57 | 55.20 | 59.60 | 60.88 |
| LongVU | 62.19 | 66.61 | 55.42 | 59.40 | 60.91 |
| DyCoke | 62.11 | 67.53 | 55.12 | 59.60 | 61.09 |
| **FLoC (Ours)** | **63.33** | **68.81** | **58.12** | **60.00** | **62.57** |

Results on InternVL3-8B (compression ratio $2^{-3}$):

| Method | Video-MME | MLVU | LVB | EgoSchema | Avg. |
|--------|-----------|------|-----|-----------|------|
| No compression | 66.63 | 72.68 | 59.39 | 70.00 | 67.18 |
| LongVU | 64.70 | 69.50 | 55.35 | 69.20 | 64.69 |
| **FLoC (Ours)** | **64.93** | **71.57** | **56.69** | **69.40** | **65.65** |

### Ablation Study

Extended temporal input setting (1 FPS, up to 7200 frames, Qwen2.5-VL-7B):

| Max Frames | Method | Video-MME | MLVU | LVB | Avg. |
|------------|--------|-----------|------|-----|------|
| 768 | No compression | 66.33 | 70.31 | 60.51 | 65.82 |
| 7200 | TS-LLaVA | 65.07 | 72.40 | 62.08 | 66.52 |
| 7200 | DyCoke | 65.78 | 71.30 | 62.98 | 66.69 |

At a higher compression ratio ($2^{-4}$), FLoC still achieves the best performance, with an average accuracy of 60.09% on Qwen2.5-VL, surpassing all competing methods.

### Key Findings

1. **FLoC consistently and significantly outperforms clustering-based and other compression methods across all compression ratios**, with the most pronounced advantage on LongVideoBench (58.12% vs. 55.42% for the second-best at 8× compression).
2. **Processing speed substantially exceeds traditional clustering methods**: as shown in Figure 1, FLoC outperforms k-means, spectral clustering, and similar approaches in both accuracy and speed, with processing time often an order of magnitude lower.
3. **Cross-model generality**: FLoC consistently leads on Qwen2.5-VL-7B, InternVL3-8B, Qwen2-VL, and LLaVA-Next-Video.
4. **Complementary to extended temporal input**: FLoC remains effective in the 7200-frame setting, maintaining performance comparable to or better than the uncompressed baseline.
5. **Importance of diversity guarantees**: In sharp contrast to clustering methods (which tend to favor dense regions), the facility location objective in FLoC ensures that sparse but important tokens are retained.

## Highlights & Insights

- **Introducing classical combinatorial optimization into visual token compression**: The facility location function has been successfully applied in document summarization and video summarization; this paper is the first to apply it to visual token selection in LMMs, offering both theoretical guarantees and strong empirical performance.
- The triple "zero-requirement" design—**training-free, model-agnostic, and query-agnostic**—makes FLoC the most deployment-friendly compression solution available.
- The **compress-once, query-many-times** property is of great practical importance: unlike query-aware methods that require re-compression for every new query, FLoC compresses only once and supports arbitrary subsequent queries.
- The practical acceleration of lazy greedy makes real-time long video processing feasible.

## Limitations & Future Work

1. **Purely similarity-based selection may miss semantically important tokens with inconspicuous embeddings**: the facility location objective only considers coverage in embedding space and does not account for semantic importance.
2. **Boundary effects from temporal blocking**: long-range temporal dependencies across block boundaries may be severed.
3. **Cosine similarity may not be the optimal similarity measure**: different token types (objects, actions, text, etc.) may require different distance functions.
4. **Fixed budget $K$ selection**: the optimal compression ratio may vary with video content complexity; adaptive budget selection could yield further improvements.
5. **Evaluation limited to QA-type tasks**: the effectiveness of FLoC on generative tasks (video captioning, video summarization) remains to be validated.

## Related Work & Insights

- **TS-LLaVA / LongVU**: Compression methods based on temporal redundancy filtering — FLoC's diversity mechanism better preserves sparse yet critical information.
- **DyCoke**: Dynamic clustering-based compression — FLoC surpasses simple clustering via the global coverage objective of facility location.
- **PruneVid / Scissor**: Trainable compression methods — their performance is inferior to FLoC, suggesting that model biases introduced by training may negatively affect generalization.
- The submodular optimization approach is generalizable to image token compression (high-resolution image understanding) and multimodal retrieval (selecting representative subsets from large candidate pools).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying facility location to token compression is an elegant cross-domain transfer, though lazy greedy itself is a classical algorithm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks, four models, multiple compression ratios, and detailed speed comparisons constitute a comprehensive experimental design.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, the comparison with clustering methods is persuasive, and the algorithm description is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ The training-free, plug-and-play nature confers extremely high practical value, representing a significant contribution to the long video understanding community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)

</div>

<!-- RELATED:END -->
