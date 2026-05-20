---
title: >-
  [Paper Note] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding
description: >-
  [CVPR 2026][Video Understanding][streaming video understanding] This paper proposes FluxMem, a training-free streaming video understanding framework that employs a three-tier hierarchical memory design (short-term / medi…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "streaming video understanding"
  - "hierarchical memory"
  - "visual token compression"
  - "adaptive thresholding"
  - "training-free"
date: 2026-05-08
content_hash: 0f184acaf6c54d4a
---

# FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.02096](https://arxiv.org/abs/2603.02096)  
**Code**: [https://github.com/YiwengXie/FluxMem](https://github.com/YiwengXie/FluxMem)  
**Area**: Video Understanding
**Keywords**: streaming video understanding, hierarchical memory, visual token compression, adaptive thresholding, training-free

## TL;DR
This paper proposes FluxMem, a training-free streaming video understanding framework that employs a three-tier hierarchical memory design (short-term / medium-term / long-term) and two adaptive token compression modules — TAS for temporal redundancy removal and SDC for spatial redundancy reduction. FluxMem achieves new state-of-the-art results on StreamingBench and OVO-Bench while discarding 60–70% of visual tokens.

## Background & Motivation
Multimodal large language models have demonstrated strong performance in offline video understanding. However, real-world applications such as robotic manipulation, autonomous driving, and smart glasses require real-time processing of continuous video streams. The core challenge in streaming video understanding is how to effectively retain long-term temporal context within limited computational and memory budgets, and to causally generate responses upon query arrival.

**Limitations of Prior Work**:

**KV cache management** (ReKV, LiveVLM): Deduplication is performed only during the LLM prefill stage, by which point the visual encoder has already incurred substantial computational cost.

**Query-guided filtering** (TimeChat-Online): Visual content selection depends on textual queries; however, in streaming scenarios queries may arrive after the video frames, precluding early filtering.

**Fixed compression strategies**: Existing token compression methods apply uniform pruning or merging strategies across all frames, ignoring temporal dependency in memory — recent frames require high-resolution detail for current inference, whereas distant frames can be compressed more aggressively.

**Core Idea**: Inspired by the decay characteristics of human memory, the paper designs a hierarchical memory system (short / medium / long term) in which recent memory is preserved intact while distant memory is progressively compressed. Compression thresholds are determined adaptively via Otsu's method rather than manual tuning.

## Method

### Overall Architecture
FluxMem partitions visual context into three memory tiers and processes streaming frames in a cascaded manner:
- **Short-term memory $\mathcal{M}^s$** (capacity $c_s$ = 8 frames): Retains all visual tokens intact for immediate perceptual inference.
- **Medium-term memory $\mathcal{M}^m$** (capacity $c_m$ = 64 frames): Stores tokens after temporal redundancy removal via TAS.
- **Long-term memory $\mathcal{M}^l$**: Stores compact representations obtained by further merging spatial redundancy via SDC.

When new frames arrive, frames overflowing from short-term memory are compressed by TAS before entering medium-term memory; frames overflowing from medium-term memory are compressed by SDC before entering long-term memory. Upon query arrival, all three memory tiers are concatenated and fed into the LLM.

### Key Designs
1. **Temporal Adjacency Selection (TAS)**:

    - Function: Removes temporally redundant tokens at the short-term → medium-term boundary.
    - Mechanism: For each spatial position $(h,w)$, the minimum cosine distance between the token at time $t$ and neighboring tokens within a $3\times 3$ window in adjacent frames is computed:
    $s_{t,h,w}^{-} = \min_{(i,j) \in \mathcal{N}_{3\times 3}(h,w)} d(v_{t,h,w}, v_{t-1,i,j})$
    $s_{t,h,w}^{+} = \min_{(i,j) \in \mathcal{N}_{3\times 3}(h,w)} d(v_{t,h,w}, v_{t+1,i,j})$
      Forward and backward thresholds $\Theta_t^{-}$ and $\Theta_t^{+}$ are computed separately via Otsu's method. A token is retained if it exhibits significant change relative to either the preceding or the following frame: $(s_{t,h,w}^{-} > \Theta_t^{-}) \lor (s_{t,h,w}^{+} > \Theta_t^{+})$
    - Design Motivation: (1) The $3\times 3$ neighborhood search tolerates slight motion and jitter, preventing important tokens from being discarded due to minor displacement. (2) The bidirectional union operation ensures that both newly appearing content from preceding frames and content about to disappear in subsequent frames are preserved. (3) The method is strictly causal, operates in a single pass, and has $\mathcal{O}(HW)$ complexity.

2. **Spatial Domain Consolidation (SDC)**:

    - Function: Merges spatially redundant regions at the medium-term → long-term boundary.
    - Mechanism: A sparse graph is constructed over the TAS-retained token set within the original $3\times 3$ spatial neighborhood — edges are drawn between tokens whose distance is $\leq \Theta_t$ (Otsu threshold). Connected components $\{C_{t,k}\}_k$ are identified via union-find, and each component is replaced by its mean anchor:
    $a_{t,k} = \frac{1}{|C_{t,k}|} \sum_{(i,j) \in C_{t,k}} v_{t,i,j}$
    - Design Motivation: (1) Operating only on the sparse token set already filtered by TAS yields a naturally sparse graph, and union-find runs in near-linear time. (2) Locally similar regions are merged into centroid anchors, substantially reducing token count while retaining essential information.

3. **Adaptive Otsu Thresholding**:

    - Function: Automatically determines per-frame compression thresholds for both TAS and SDC.
    - Mechanism: Otsu's method identifies the binarization threshold that maximizes inter-class variance:
    $\Theta_t = \arg\max_{\theta} [\omega_1(\theta)\omega_2(\theta)(\mu_1(\theta) - \mu_2(\theta))^2]$
      TAS analyzes the distribution of temporal similarity scores, while SDC analyzes the distribution of spatial distances.
    - Design Motivation: Fixed thresholds lead to over-retention in static scenes (wasted capacity) and over-discarding in dynamic scenes (information loss). Otsu's method is a classic nonparametric approach that introduces no additional parameters; it automatically raises thresholds under high motion (retaining more tokens) and lowers them in static conditions (enabling more aggressive compression).

4. **Proactive Response Triggering**:

    - Function: Detects scene transitions and triggers proactive LLM output with zero additional overhead.
    - Mechanism: The backward score statistics from TAS are reused — the proportion of tokens exceeding the threshold is computed as $r_t^{-} = \frac{1}{HW}\sum_{h,w} \mathbf{1}[s_{t,h,w}^{-} > \Theta_t^{-}]$, and triggering occurs when $r_t^{-} > \gamma$.
    - Design Motivation: During scene transitions, a large fraction of tokens change simultaneously, causing $r_t^{-}$ to rise naturally, with zero additional computation.

### Loss & Training
- **Entirely training-free**: FluxMem is a plug-and-play inference-time module.
- Implemented on top of Qwen2.5-VL-7B.
- Online setting: 1 fps sampling, 256 tokens per frame, up to 256 frames.
- Offline setting: 1 fps, 64 tokens per frame, up to 1024 frames.

## Key Experimental Results

### Main Results

| Method | Type | StreamingBench (real-time) | OVO-Bench (real-time) | OVO-Bench (overall) | VideoMME | MLVU |
|--------|------|--------------------------|----------------------|---------------------|----------|------|
| Qwen2.5-VL (baseline) | Offline | 73.9 | 63.3 | 49.8 | 63.3 | 67.9 |
| TimeChat-Online | Training-based | 75.3 | 61.4 | 47.6 | 63.3 | 65.4 |
| StreamForest | Training-based | 77.3 | 61.2 | 55.6 | 61.9 | 69.6 |
| ViSpeak | Training-based | 74.4 | 66.3 | — | — | — |
| **FluxMem** | **Training-free** | **76.4** | **67.2** | **53.3** | **65.3** | **73.1** |

FluxMem surpasses all training-based methods on online tasks (StreamingBench: 76.4 vs. StreamForest 77.3; OVO-Bench: 67.2 vs. ViSpeak 66.3) and achieves 73.1 on the offline benchmark MLVU (+5.2 over baseline).

### Ablation Study

| Memory Configuration | Token Discard Rate | MLVU | VideoMME | StreamingBench | Average |
|---------------------|--------------------|------|---------|---------------|---------|
| S only | 0% | 67.8 | 63.3 | 73.9 | 68.3 |
| M only | 43.2% | 69.9 | 65.5 | 74.7 | 70.0 |
| L only | 85.1% | 70.9 | 62.0 | 75.9 | 69.6 |
| S+M+L (full) | **64.3%** | **73.1** | **65.3** | **76.4** | **71.6** |

| Efficiency Metric | Dataset | Baseline | FluxMem | Improvement |
|------------------|---------|----------|---------|------------|
| Latency (ms) | OVO-Bench | 2701 | 812 | ↓69.9% |
| Peak GPU Memory (GB) | OVO-Bench | 35.8 | 23.5 | ↓34.5% |
| Latency (ms) | MLVU | 3614 | 2014 | ↓44.3% |
| Accuracy | OVO-Bench | 49.8 | 53.3 | +3.5 |

### Key Findings
- **Complementarity of tiers**: The M+L combination achieves an MLVU score of 73.1, substantially outperforming M alone (69.9) or L alone (70.9); TAS and SDC capture temporal variation and spatial structure respectively and are mutually complementary.
- **Short-term memory is critical for online tasks**: S+L achieves 77.0 on StreamingBench, exceeding both S alone (73.9) and L alone (75.9), confirming that fine-grained recent-frame details are indispensable for instant perception.
- **Adaptive thresholding significantly outperforms fixed thresholds**: In the medium-term memory, the adaptive threshold attains equivalent accuracy at a 42.8% discard rate compared to a fixed threshold requiring only a 29.4% discard rate — a 45% improvement in compression efficiency.
- **FluxMem consistently outperforms all competing methods** (FIFO, Uniform, Random, DTD) across the 50–70% token discard range.

## Highlights & Insights
- The **fully training-free** design is the most prominent contribution — it is plug-and-play, compatible with any MLLM, and eliminates the data collection and training costs associated with supervised fine-tuning.
- The application of Otsu's method to token compression is an elegant analogy: the question of which tokens to retain or discard is reframed as a classical binary segmentation problem.
- The hierarchical memory design elegantly maps onto the temporal decay characteristics of video information.
- TAS's bidirectional layout combined with SDC's union-find merging introduces only 4.1 ms of additional overhead per frame.

## Limitations & Future Work
- The capacity allocation of the hierarchical memory (short-term: 8, medium-term: 64) is set manually; different tasks or video types may require different optimal configurations.
- Otsu's method assumes a bimodal score distribution; it may not yield optimal segmentation for unimodal or multimodal distributions.
- Validation is limited to Qwen2.5-VL-7B; compatibility with other MLLMs (LLaVA, InternVL) has not been tested.
- Performance at high frame rates (e.g., 30 fps) is unknown — all current experiments are conducted at 1 fps.
- The mean-anchor approach in spatial merging may discard fine-grained texture information, potentially disadvantaging tasks requiring spatial precision (e.g., OCR).

## Related Work & Insights
- **LiveVLM / ReKV (KV cache management)**: These methods perform deduplication only during the LLM prefill stage, by which point the visual encoder has already processed all tokens, incurring wasted computation. FluxMem completes compression before tokens enter the LLM, directly reducing the input sequence length and addressing latency at its source.
- **TimeChat-Online (query-guided filtering)**: Visual token selection relies on textual queries, which may arrive long after the corresponding frames in streaming scenarios. FluxMem's TAS and SDC operate entirely based on the intrinsic information density of visual content, requiring no text conditioning.
- **StreamForest (training-based)**: StreamForest achieves 77.3 on StreamingBench, marginally higher than FluxMem's 76.4, but requires dedicated SFT data and training. FluxMem is zero-training plug-and-play, and surpasses StreamForest on OVO-Bench and MLVU.
- **FastV / DTD (static token pruning)**: These methods apply uniform pruning strategies across all frames without distinguishing recent from distant frames. FluxMem's hierarchical design allows recent frames to retain full detail while distant frames are compressed aggressively, better reflecting the temporal decay of video information.
- **StreamMem**: Also adopts a hierarchical memory design but requires training a memory controller module. FluxMem replaces the learnable gating with classical Otsu thresholding, yielding a simpler and more generalizable solution.

**Broader Implications**:
- **Generality of the hierarchical memory paradigm**: The short / medium / long-term memory design is not limited to video understanding and can be transferred to streaming document comprehension (recent paragraphs preserved intact → older paragraphs compressed into summaries) and real-time dialogue (recent turns complete → older turns compressed), among other sequential settings.
- **Otsu as an information density partitioner**: Modeling "retain vs. discard" as binary classification is an elegant abstraction. Similarly, adaptive quantization in image coding (adaptive bit allocation per region) could leverage Otsu to determine per-region quantization precision.
- **Relationship to Token Merging (ToMe)**: SDC's union-find merging shares conceptual similarities with ToMe's bipartite graph matching, but SDC operates on a sparse token set with higher efficiency. Combining the two approaches could further improve compression rates.
- **Extension of the proactive triggering mechanism**: The scene change detection that emerges as a byproduct of TAS can be applied to downstream tasks such as video segmentation and keyframe extraction, enabling multi-task computational sharing.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical memory and Otsu-based adaptive thresholding is novel, though the individual components (TAS/SDC) are not technically complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks (2 online + 3 offline), efficiency analysis, tier-wise ablation, threshold analysis, and multi-method comparison — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, complete algorithmic pseudocode, and information-rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Training-free design, substantial efficiency gains, and state-of-the-art performance confer strong practical value for the streaming video understanding community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](../../ACL2026/video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)

</div>

<!-- RELATED:END -->
