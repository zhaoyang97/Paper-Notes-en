---
title: >-
  [Paper Note] FlowNar: Scalable Streaming Narration for Long-Form Videos
description: >-
  [ICML 2026][Multimodal VLM][Streaming Video Narration] FlowNar employs a combination of "clearing visual KV caches at the end of segments + compressing historical visual information into fixed-length memory tokens via ga…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Streaming Video Narration"
  - "KV Cache Pruning"
  - "Linear Attention"
  - "Long-form Video Understanding"
  - "Self-conditioned Evaluation"
date: 2026-05-08
content_hash: 198a00ee9f94a26a
---

# FlowNar: Scalable Streaming Narration for Long-Form Videos

**Conference**: ICML 2026  
**arXiv**: [2606.00620](https://arxiv.org/abs/2606.00620)  
**Code**: https://github.com/zeyun-zhong/FlowNar (Available)  
**Area**: Video Understanding / Multimodal VLM / Streaming Video  
**Keywords**: Streaming Video Narration, KV Cache Pruning, Linear Attention, Long-form Video Understanding, Self-conditioned Evaluation

## TL;DR
FlowNar employs a combination of "clearing visual KV caches at the end of segments + compressing historical visual information into fixed-length memory tokens via gated linear attention." This allows the streaming video narration model to maintain constant GPU memory and computational overhead, handling $10\times$ longer videos with $3\times$ the throughput. Simultaneously, the introduction of a self-conditioned evaluation protocol reveals that baseline methods are significantly overestimated in real-world deployment scenarios.

## Background & Motivation

**Background**: Online streaming narration requires Large Multimodal Models (LMMs) to continuously receive frame streams, autonomously determine when to output a narration, and generate the content. Representative works like Videollm-online and Videollm-mod are already capable of frame-aligned narration.

**Limitations of Prior Work**: These methods continuously feed the KV pairs of all historical visual frames into the LLM context. Consequently, both GPU memory and computational costs grow at least linearly with video length, leading to OOM (Out of Memory) on 24GB GPUs and a significant drop in FPS over time. Moreover, existing evaluations use *teacher-forcing* where Ground Truth (GT) narrations are used as history, masking the error accumulation in real deployment where "one mistake leads to another."

**Key Challenge**: The dilemma of long context—retaining all visual history provides information but leads to complexity explosion and amplifies noise/errors; pruning history saves memory but causes the loss of long-term visual information and narrative incoherence.

**Goal**: (1) To keep the GPU memory and per-step computational complexity constant relative to video length $T$; (2) To retain long-term visual summaries to prevent performance collapse; (3) To provide evaluation protocols that reflect real-world deployment.

**Key Insight**: It is observed that what is truly needed between segments is not "all raw KV pairs," but a "visual summary sufficient for narrative coherence." Detailed KV pairs can be aggressively pruned after each narration segment, passing forward only a fixed-size memory token block.

**Core Idea**: Combining "Dynamic Context Management (DCM) + Cross-segment Linear Attention Memory (CLAM)" to compress the complexity of visual history from $O(T)$ to $O(1)$, while "realizing" the evaluation through a self-conditioned protocol.

## Method

### Overall Architecture
The input is a continuous frame stream $\mathbf{V}=\{\mathbf{v}_t\}_{t=1}^{T}$, and the output is a narration sequence with timestamps $\Psi=\{(t_n, y_n)\}_{n=1}^{N}$. The pipeline performs the following at each frame $t$: (1) SigLIP encoding + MLP projection into language space to obtain $\mathbf{E}_t$; (2) CLAM incrementally updates a $D\times D$ recurrent state $\mathbf{S}_t$ using current frame tokens, and reads out fixed-length memory $\mathbf{M}_t \in \mathbb{R}^{M\times D}$ using $M$ learnable queries; (3) The LLM calculates the `[SKIP]` probability based on the visual cache $\mathcal{C}_t^{\text{vid}}$ and previous narration cache $\mathcal{C}_{n-1}^{\text{nar}}$ to decide whether to trigger narration; (4) Upon triggering, it auto-regressively generates $y_n$, then clears all detailed visual KV pairs of the current segment, prepending the segment-end memory $\mathbf{M}_{t_n}$ to the next segment as a long-term summary. The FlowNar-C variant further retains only the text KV pairs of the most recent $k$ narrations to achieve constant complexity across all dimensions.

### Key Designs

1.  **Dynamic Context Management (DCM) + Dual-threshold Triggering**:
    - **Function**: Ensures the visual context does not grow with $T$ while controlling the narration pace to avoid bursts or long silences.
    - **Mechanism**: After generating each narration, the current segment's visual KV is **explicitly cleared** $\mathcal{C}_t^{\text{vid}} \leftarrow \emptyset$, and the previous $\mathbf{M}_{t_{n-1}}$ is discarded, forcing the model to rely only on the newly computed $\mathbf{M}_{t_n}$ as history. The triggering logic uses two thresholds: a primary threshold $\theta$ to check $p(\text{[SKIP]} \mid \mathbf{E}_t, \mathcal{C}_{t-1}^{\text{vid}}, \mathcal{C}_{n-1}^{\text{nar}}) \le \theta$. Once triggered, it switches briefly to a lower $\theta_{\text{low}}=0.5$ to suppress immediate re-triggering and avoid explosive bursts.
    - **Design Motivation**: In self-conditioned deployment, long context not only consumes memory but also feeds "errors from the previous sentence" back into the LLM, creating a snowball effect. Aggressive pruning is more robust than retaining the full history.

2.  **CLAM: Cross-segment Linear Attention Memory**:
    - **Function**: Compresses visual history into $M$ memory tokens with constant memory and per-step computation.
    - **Mechanism**: Maintains a recurrent state $\mathbf{S}_t \in \mathbb{R}^{D\times D}$. For each intra-frame token $\mathbf{x}_{t,j}$, it computes key/value pairs and a gating matrix $\mathbf{G}_{t,j}$ (range $(0,1)$), updated via $\mathbf{S}_{t,j} = \mathbf{G}_{t,j} \odot \mathbf{S}_{t,j-1} + \mathbf{k}_{t,j}^\top \mathbf{v}_{t,j}$. $M$ learnable queries $\mathbf{Z}\in\mathbb{R}^{M\times D}$ are used via linear projection to get $\mathbf{Q}$, finally reading out $\mathbf{M}_t = \mathbf{Q}\mathbf{S}_t$. The state can store approximately $O(D)$ independent key-value pairs, sufficient for long segments.
    - **Design Motivation**: Naive KV caches expand linearly. The recurrent perspective of linear attention naturally provides constant memory, constant per-step computation, and parallelizable training. It decouples "compression" (token-by-token within frames) from "reading" (via fixed queries), avoiding the pitfalls of MovieChat-style aggregation or simple sliding windows that lose long-range information.

3.  **Self-conditioned Evaluation Protocol + Alignment-then-Scoring**:
    - **Function**: Evaluates the narration model under real deployment conditions to avoid teacher-forcing overestimation.
    - **Mechanism**: During evaluation, each $y_n$ is based only on the model's own previously generated $\{y_j^{\text{pred}}\}$ without GT history. Since predicted and GT segments may differ in count and boundaries, segment-level matching is first performed using IoU $\tau=0.5$ to calculate Precision/Recall/F1 (temporal alignment). Then, Generalized IoU is used to retrieve the best-matched predicted segment for each GT segment to calculate CIDEr/METEOR/ROUGE-L (narration quality).
    - **Design Motivation**: Teacher-forcing hides error propagation, making "barely usable" models appear superior. Self-conditioning + post-alignment retains temporal evaluation capability while exposing the error accumulation inherent in real deployment.

### Loss & Training
The model end-to-end minimizes standard next-token cross-entropy, with joint supervision on narration tokens $y_n$ and `[SKIP]` trigger tokens. During training, a segment-level attention mask is used (which, in addition to the standard causal mask, blocks attention from the current segment to "raw frame tokens of distant segments" and "memory tokens of distant segments"). This forces the model to rely only on "$\mathbf{M}_{t_{n-1}}$ from the end of the previous segment + current segment frames + generated narration," consistent with the cache-clearing behavior at inference. To bridge the positional encoding discrepancy between training (continuous sequences) and inference (cache clearing), an independent position counter is used during inference to simulate training-style position IDs. Cost: Training FlowNar-1B on 4×H100 takes 67 GPU-hours, approximately $1.9\times$ that of Videollm-online (one-time overhead).

## Key Experimental Results

### Main Results
Comparison with Videollm-online and Videollm-mod (using Llama-3-1B backbone) under the self-conditioned protocol on three long-form egocentric datasets:

| Dataset | Method | F1↑ | CIDEr↑ | Cache (M)↓ |
|---------|--------|-----|--------|-----------|
| Ego4D | Videollm-online | 16.29 | 28.04 | 737.6 |
| Ego4D | FlowNar-C | 17.90 | 34.48 | **20.2** |
| Ego4D | **FlowNar** | **24.85** | **35.64** | 59.2 |
| EgoExo4D | Videollm-online | 31.77 | 69.88 | 878.5 |
| EgoExo4D | **FlowNar** | **32.99** | **75.33** | 125.9 |
| EK100 | Videollm-online | 12.98 | 29.00 | 1096.0 |
| EK100 | FlowNar-C | 25.20 | 37.28 | **22.7** |
| EK100 | **FlowNar** | **29.12** | **46.63** | 65.3 |

FlowNar-C reduces the cache on EK100 from 1096M to 22.7M (approx. $48\times$ reduction) while increasing CIDEr from 29.00 to 37.28.

### Ablation Study
Ablation of visual history strategies under Ego4D self-conditioned protocol:

| Visual History Strategy | DCM | CIDEr↑ | METEOR↑ | ROUGE↑ |
|-------------------------|-----|--------|---------|--------|
| No History Frames | ✓ | 30.40 | 11.36 | 30.54 |
| Only Recent Frames | ✓ | 30.16 | 11.42 | 30.59 |
| Retain All Frames | ✗ | 28.04 | 11.33 | 29.86 |
| **CLAM** | ✓ | **35.64** | **12.14** | **31.64** |

### Key Findings
- **Retaining all history is actually the worst** (CIDEr 28.04)—confirming the judgment that "long context = long error chain" under self-conditioning; DCM is a necessity, not an option.
- **CLAM significantly outperforms alternatives** such as last-$k$, K-Means, MovieChat-style token merging, TokenMLP, and restructured RetNet (Table 5). This suggests that gated linear attention is better suited for streaming narration than similarity-based merging or fixed windows.
- Dual-threshold triggering improves F1 from a static 16.78 to 24.85 (Table 4), indicating that pacing control is as vital as context management.
- Under the teacher-forcing protocol (Table 2), the gap between FlowNar and baselines narrows, proving that previous SOTA figures partially relied on "cheating with GT history."

## Highlights & Insights
- **"Less is More" empirically proven in long video**: Under self-conditioning, aggressive pruning is more accurate than retaining full history because the cost of error propagation outweighs the cost of information loss. This mirrors NLP observations that "garbage context drags down generation" and serves as a highly transferable design principle for streaming generation.
- **The recurrent view of Linear Attention fits streaming compression naturally**: Interpreting $\mathbf{S}_t$ as "content-addressable associative memory" and using learnable queries to read fixed-length summaries effectively turns the Transformer-RNN hybrid into a "compressor + reader," avoiding the fragility of similarity heuristics in token merging. This architecture can be transferred to any continuous generation task requiring constant memory, such as streaming audio narration or real-time driving scene understanding.
- **The evaluation protocol is a core contribution**: Replacing teacher-forcing with self-conditioning + post-alignment reshapes the standards of the "long video narration" track, representing a methodology-level correction.

## Limitations & Future Work
- The training cost is approximately $1.9\times$ that of Videollm-online, primarily due to unoptimized attention kernels under segment-level masks and the additional memory tokens.
- Memory capacity arguments are based on the theoretical $D\times D$ state storing $O(D)$ key-value pairs; whether this holds for extremely long (hour-plus) videos remains to be seen beyond egocentric datasets.
- Hyperparameters like $\theta_{\text{low}}$ and refresh periods depend on the average segment duration of each dataset, requiring recalibration for different video sources.
- Current evaluations are limited to English; error propagation patterns in multilingual streaming generation might differ.

## Related Work & Insights
- **vs Videollm-online**: Both perform frame-aligned narration, but this work explicitly prunes visual KV and adds linear attention summaries, reducing visual context complexity from $O(T)$ to $O(1)$ and improving self-conditioned F1 on Ego4D from 16.29 to 24.85.
- **vs Videollm-mod**: Videollm-mod uses routing to reduce intermediate visual computation, but the cache still grows linearly. This work excels in both cache size and narration quality, while routing is proven more fragile to error history under self-conditioning.
- **vs MovieChat / Online K-Means**: Those methods rely on similarity-based merging and manual heuristics, with memory still growing over time. CLAM uses learnable gating for parametric compression that is fixed-length and end-to-end trainable.
- **vs Streaming QA (e.g., StreamForest)**: Those works trigger answers based on external queries; this work autonomously decides when to narrate. The tasks differ, but memory management strategies are mutually beneficial.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of DCM + Linear Attention compression + Self-conditioned evaluation is effective and comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted on three datasets with two protocols and multi-dimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear pseudocode, motivation, and diagrams.
- Value: ⭐⭐⭐⭐⭐ Directly addresses engineering bottlenecks in long-video streaming and provides a transformative evaluation protocol.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[ACL 2026\] AdaTooler-V: Adaptive Tool-Use for Images and Videos](../../ACL2026/multimodal_vlm/adatooler-v_adaptive_tool-use_for_images_and_videos.md)
- [\[AAAI 2026\] Towards Scalable Web Accessibility Audit with MLLMs as Copilots](../../AAAI2026/multimodal_vlm/towards_scalable_web_accessibility_audit_with_mllms_as_copilots.md)
- [\[CVPR 2026\] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](../../CVPR2026/multimodal_vlm/vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)

</div>

<!-- RELATED:END -->
