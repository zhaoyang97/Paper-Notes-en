---
title: >-
  [Paper Note] StreamForest: Efficient Online Video Understanding with Persistent Event Memory
description: >-
  [NeurIPS 2025][Autonomous Driving][streaming video understanding] This paper proposes StreamForest, an architecture that adaptively organizes streaming video frames into multiple event-level tree structures via a "Persis…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "streaming video understanding"
  - "persistent event memory"
  - "memory tree structure"
  - "visual token compression"
  - "multimodal large language models"
date: 2026-05-08
content_hash: ab8987003305b778
---

# StreamForest: Efficient Online Video Understanding with Persistent Event Memory

**Conference**: NeurIPS 2025
**arXiv**: [2509.24871](https://arxiv.org/abs/2509.24871)
**Code**: [GitHub](https://github.com/MCG-NJU/StreamForest)
**Area**: Autonomous Driving / Online Video Understanding
**Keywords**: streaming video understanding, persistent event memory, memory tree structure, visual token compression, multimodal large language models

## TL;DR

This paper proposes StreamForest, an architecture that adaptively organizes streaming video frames into multiple event-level tree structures via a "Persistent Event Memory Forest," combined with a "Fine-grained Spatiotemporal Window" to capture short-term visual cues. The method achieves 77.3% accuracy on StreamingBench and retains 96.8% of performance under extreme compression (only 1024 visual tokens).

## Background & Motivation

- **Background**: Multimodal large language models (MLLMs) have achieved remarkable progress in offline video understanding, yet face two major challenges in real-time streaming scenarios: (1) the memory burden of storing historical features from continuously arriving frames; and (2) insufficient capacity for real-time spatiotemporal reasoning.
- **Limitations of Prior Work**: Existing streaming video processing strategies exhibit clear deficiencies. Sampling-stage compression (e.g., aggressively discarding frames) sacrifices fine-grained spatiotemporal reasoning; storage-stage compression (merging frames based on inter-frame similarity) tends to miss critical foreground actions due to background noise, and excessive local merging introduces spatiotemporal irregularities.
- **Key Insight**: The paper addresses memory management at the semantic event level — decomposing video into event segments and constructing hierarchical event tree structures. A multi-dimensional penalty function guides adaptive merging to preserve semantic richness while controlling memory overhead. A fine-grained spatiotemporal window is further introduced to focus on detailed visual features at the current timestep.

## Method

### Overall Architecture

StreamForest processes streaming video frames at 1 FPS and comprises two core components: (1) the Fine-grained Spatiotemporal Window (FSTW) for high-resolution perception and short-term memory at the current timestep; and (2) the Persistent Event Memory Forest (PEMF) for hierarchical storage and adaptive compression of long-term history. The visual encoder is SigLiP-so400M and the LLM is Qwen2-7B. Upon receiving a user query, all root node features from the PEMF and the full visual features from the FSTW are fed into the LLM.

### Key Designs

1. **Fine-grained Spatiotemporal Window (FSTW)**:

    - Real-time perception: directly samples high-resolution visual features (729 tokens) from the current frame with encoded spatiotemporal positional information.
    - Short-term spatiotemporal memory: maintains a frame buffer spanning $t_s$ seconds (18 frames, 128 tokens per frame); incoming frames cause older frames to be compressed along the spatial dimension.
    - Computes inter-frame similarity for subsequent event-level segmentation.
    - Upon buffer overflow, a "meta-event" is segmented at the position of the local minimum inter-frame similarity and transferred to the PEMF.
    - A meta-event is a collection of visual tokens from a group of similar consecutive frames, serving as an independent node in the PEMF.

2. **Persistent Event Memory Forest (PEMF)**:

    - Unlike conventional frame-level compression, PEMF organizes memory hierarchically at the semantic event level.
    - Events are managed in a tree structure: when the number of long-term memory tokens exceeds the upper limit $L_q$, the adjacent node pair with the lowest penalty score is selected for merging.
    - Merging employs Token Merging (ToMe), compressing the visual tokens of the selected node pair to half their combined count.
    - A triple penalty function jointly guides merging decisions to ensure adaptivity.

3. **Triple Penalty Function Design**:

    - **Similarity penalty $P_s$**: computes the cosine similarity of tokens between two event nodes via bipartite graph matching, taking the mean of the top-$k$ highest similarity scores; $P_s = 1 - \text{avg}$. Encourages merging of highly similar, redundant events.
    - **Merge-count penalty $P_m$**: $P_m = (c_i + c_{i+1}) / (2c_{max})$. Penalizes nodes that have been repeatedly merged to prevent spatiotemporal inconsistency from accumulated information loss.
    - **Temporal distance penalty $P_t$**: $P_t = 1 - (d_i + d_{i+1})/2$. Preserves more detail for recent events while permitting more aggressive compression for distant ones.
    - Total penalty: $P = w_s P_s + w_m P_m + w_t P_t$ (default weights: 0.4, 0.4, 0.2).
    - Degeneration analysis: using only $P_s$ degenerates to similarity-based compression; only $P_m$ degenerates to uniform downsampling; only $P_t$ degenerates to FIFO.

4. **OnlineIT Training Dataset**:

    - OnlineIT-general (32K): integrates multiple streaming video understanding datasets to address hallucinations caused by spatiotemporal distribution shift.
    - OnlineIT-drive (89K): streaming QA data for autonomous driving scenarios, covering real-time localization, static/dynamic traffic entity understanding, and risk assessment.

5. **ODV-Bench**:

    - A streaming video understanding benchmark for autonomous driving, comprising three task categories: static objects, dynamic objects, and multi-agent interaction events.
    - Constructed via a semi-automatic pipeline: YOLO detection + VLLM annotation + human verification.

### Loss & Training

A five-stage training strategy is adopted: the first three stages follow the offline long-video MLLM training paradigm (VideoChat-Flash); the fourth stage fine-tunes on OnlineIT to yield the base StreamForest; an optional fifth stage fine-tunes on OnlineIT-Drive to yield StreamForest(FT-drive). Training uses 32 A100 GPUs.

## Key Experimental Results

### Main Results

**Online video understanding benchmarks**:

| Method | Scale | StreamingBench | OVBench | OVO-Bench |
|--------|-------|---------------|---------|-----------|
| VideoChat-Online | 4B | - | 62.9 | - |
| Dispider | 7B | - | 52.7 | - |
| Flash-VStream | 7B | - | 40.2 | - |
| StreamForest | 7B | **77.3** | **62.3** | **55.6** |

**ODV-Bench (autonomous driving)**:

| Method | Static Obj. Avg | Dynamic Obj. Avg | Event-level Avg | Overall |
|--------|----------------|-----------------|----------------|---------|
| Qwen2.5-VL-7B | 48.3 | 57.5 | 59.4 | 55.6 |
| StreamForest | 51.5 | 62.3 | 63.8 | 59.9 |
| StreamForest(FT-drive) | **62.6** | **64.0** | **67.5** | **65.0** |
| Human | 95.9 | 88.2 | 92.5 | 91.4 |

### Ablation Study

| Configuration | Avg. Accuracy | Notes |
|--------------|--------------|-------|
| Default 8192 tokens | 100% (baseline) | Full setting |
| 4096 tokens | ~99% | Mild compression, nearly lossless |
| 2048 tokens | ~98% | Moderate compression, well maintained |
| 1024 tokens | **96.8%** | Extreme compression retains most performance |
| $P_s$ only | Degraded | Similarity-based compression |
| $P_m$ only | Degraded | Similar to uniform downsampling |
| $P_t$ only | Degraded | Similar to FIFO |

### Key Findings

- StreamForest loses only 3.2% of performance under extreme compression (1024 tokens), demonstrating the effectiveness of event-level memory management.
- On offline video benchmarks, the method matches or surpasses state-of-the-art offline models, indicating that streaming processing does not sacrifice understanding quality.
- Among the three penalty terms, the merge-count penalty and similarity penalty carry the highest weights (0.4 each), suggesting that preventing excessive merging and eliminating redundancy are equally important.
- Fine-tuning for autonomous driving (FT-drive) substantially improves driving-scene performance (+5.1 overall accuracy), yet a gap of 26 percentage points with human performance remains.

## Highlights & Insights

- The **event-level memory tree** design is elegant and closely mirrors the human cognitive approach to video (episodic memory in event units).
- The **triple penalty function** elegantly balances content redundancy, information fidelity, and temporal importance, with tunable weights that recover multiple known strategies as special cases.
- The **96.8% retention rate** under extreme compression is a highly compelling result that directly demonstrates the robustness of the method.
- **ODV-Bench** fills an important gap in evaluation for streaming video understanding in autonomous driving.

## Limitations & Future Work

- The fixed 1 FPS processing rate may not satisfy applications requiring higher frame rates (e.g., fast-motion scenes).
- Event boundary detection relies on local minima of inter-frame similarity, which may be insufficiently robust to gradual scene transitions.
- While ToMe merging is efficient, it continuously discards details; information from early events may degrade significantly over long operating periods.
- Only 7B-scale models are evaluated; the behavior of larger models remains unknown.
- The substantial gap between ODV-Bench performance and human-level performance (59.9 vs. 91.4) indicates that understanding of autonomous driving scenes requires considerable further improvement.

## Related Work & Insights

- **vs. VideoChat-Online**: StreamForest comprehensively outperforms VideoChat-Online on all online benchmarks, owing to event-level memory management (vs. VideoChat-Online's static hierarchical memory).
- **vs. Flash-VStream**: Flash-VStream employs a similarity-based compression strategy, which is equivalent to a degenerate version of PEMF using only the $P_s$ factor.
- **Implications for streaming AI agents**: Event-level memory management can be directly applied to AI agents requiring long-term memory (e.g., robots, live-streaming assistants).

## Rating

- Novelty: ⭐⭐⭐⭐ The event memory forest constitutes a novel memory management paradigm, and the triple penalty function is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 4 online benchmarks and offline benchmarks; the extreme compression experiments are particularly impressive.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and method descriptions are thorough, though some mathematical notation could be made more consistent.
- Value: ⭐⭐⭐⭐⭐ Makes an important contribution to the streaming video understanding field; ODV-Bench and OnlineIT are valuable community resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](../../ICLR2026/autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)
- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[NeurIPS 2025\] Aha: Predicting What Matters Next — Online Highlight Detection Without Looking Ahead](aha_predicting_what_matters_next_online_highlight_detection.md)
- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)

</div>

<!-- RELATED:END -->
