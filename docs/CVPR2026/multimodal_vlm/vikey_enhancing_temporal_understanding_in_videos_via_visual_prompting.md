---
title: >-
  [Paper Note] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting
description: >-
  [CVPR 2026][Multimodal VLM][Visual Prompting] ViKey overlays frame-index visual prompts (VPs) onto video frames and incorporates a lightweight Keyword-Frame Mapping (KFM) module to significantly improve temporal reasonin…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Visual Prompting"
  - "Video Large Language Models"
  - "Temporal Understanding"
  - "Frame Index"
  - "Training-Free"
date: 2026-05-08
content_hash: c8717d2d005673f3
---

# ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting

**Conference**: CVPR 2026
**arXiv**: [2603.23186](https://arxiv.org/abs/2603.23186)  
**Code**: [https://github.com/MICV-yonsei/ViKey](https://github.com/MICV-yonsei/ViKey)  
**Area**: Multimodal VLM
**Keywords**: Visual Prompting, Video Large Language Models, Temporal Understanding, Frame Index, Training-Free

## TL;DR

ViKey overlays frame-index visual prompts (VPs) onto video frames and incorporates a lightweight Keyword-Frame Mapping (KFM) module to significantly improve temporal reasoning in VideoLLMs without any training, achieving near-dense-frame performance with as few as 20% of frames.

## Background & Motivation

VideoLLMs demonstrate strong performance on multimodal video tasks, yet processing dense video frames incurs prohibitive computational costs, making frame selection a standard practice. However, frame selection introduces a critical side effect: **disruption of temporal continuity**.

**Limitations of Prior Work**: Removing intermediate frames causes VideoLLMs to lose the ability to infer event ordering. For instance, given a sparse sequence showing a player crossing a line followed by a referee showing a red card, humans can infer causality, whereas VideoLLMs may incorrectly conclude that the referee committed the foul.

**Key Challenge**: Frame selection reduces the video to discrete temporal snapshots, making it inherently difficult to reconstruct a coherent event sequence. Existing solutions—such as enhanced temporal encodings and extended context modules—are complex and require substantial training.

**Key Insight**: Visual prompting (VP) has been shown to effectively guide spatial attention, yet its potential for cross-frame temporal reasoning remains largely unexplored. The authors observe that simply annotating each frame with its index number enables the model to perceive temporal continuity.

## Method

### Overall Architecture

ViKey is a training-free, plug-and-play framework: input video frames → overlay frame-index visual prompts → extract key textual concepts from the query → map keywords to the most relevant frames via KFM → rewrite the query with explicit frame indices → feed into the VideoLLM for inference.

### Key Designs

1. **Sequential Visual Prompting**:

    - **Function**: Embeds frame index information (e.g., "frame #01") into the pixel space of each frame.
    - **Mechanism**: A text-format frame index is overlaid at the bottom-left corner of each frame. Font size adapts to frame resolution: $fontsize = \min(width, height) / s$. The effectiveness of VP is validated through three carefully designed experiments: (1) a positional encoding degradation experiment demonstrates that VP can independently recover frame-order information; (2) a frame-level reference experiment shows that VP enables the model to retrieve frame content by index, analogous to a dictionary lookup; (3) attention analysis reveals that VP amplifies image token attention weights in the middle-to-upper layers.
    - **Design Motivation**: Placement at the bottom-left is motivated by an observed positional bias—bottom placements yield substantially higher accuracy than top placements (reverse lookup: bottom 100% vs. top 60–79%), likely because subtitles and watermarks frequently appear at the bottom in training data.

2. **Keyword-Frame Mapping (KFM)**:

    - **Function**: Anchors key concepts from the text query to the most relevant video frames.
    - **Mechanism**: Salient keywords are extracted from the user query; cosine similarity is computed between keyword and per-frame embeddings in a shared embedding space to identify the best-matching frames. The query is then rewritten as an augmented version containing explicit frame indices, e.g., "In frame #03, what does the player do?" This provides explicit temporal anchors for reasoning.
    - **Design Motivation**: VP supplies frame-level indexing capability, while KFM establishes an explicit mapping between textual queries and visual frames; their combination enables precise temporal localization.

3. **Positional Bias Analysis and Optimization**:

    - **Function**: Characterizes and exploits VideoLLM preferences for VP placement positions.
    - **Mechanism**: VP effectiveness is systematically evaluated at four corner positions (TL/TR/BL/BR). BL and BR achieve 100% accuracy on the reverse lookup task, whereas TL reaches only ~60%. The dominant error pattern for TL is an off-by-one association—the model links the current frame's index to the content of the subsequent frame.
    - **Design Motivation**: Frame tokens are concatenated into a single sequence with no explicit boundaries. Top-positioned indices are prone to confusion with tokens of the following frame, whereas bottom-positioned indices align more naturally with the terminal tokens of the current frame.

### Loss & Training

ViKey is entirely training-free; it requires neither modification of model parameters nor any additional training.

## Key Experimental Results

### Main Results

| Model + Setting | TempCompass | MVBench | VideoMME | LongVideoBench |
|---|---|---|---|---|
| LLaVA-Video-7B (64 frames) | 74.68 | 82.50 | — | 56.42 |
| + ViKey (64 frames) | 77.83 | 87.00 | Improved | 58.66 |
| + ViKey (13 frames = 20%) | ~75 | ~83 | ≈ 64-frame | ~56 |

Consistent improvements are observed across temporal reasoning subsets of TempCompass, MVBench, VideoMME, and LongVideoBench.

### Ablation Study

| Configuration | Lookup Accuracy | Reverse Lookup Accuracy | Notes |
|---|---|---|---|
| No VP | 12.43% | 18.57% | Extremely low baseline |
| VP (bottom-left) | 64.62% | 100.00% | Substantial gain in frame-level reference |
| VP (top-left) | 55.56% | 60.19% | Evident positional bias |
| VP + KFM | Best | Best | Complementary combination |

### Key Findings

- VP recovers 2.9–9.9 percentage points of temporal understanding performance even under extreme conditions where positional encodings are corrupted.
- VP increases the average attention weight allocated to image tokens by 11.65%, concentrated in the middle-to-upper layers (layers 4–6, 11–14, and beyond layer 21).
- Using only 20% of frames with ViKey approaches the dense 100%-frame baseline on several benchmarks, demonstrating high efficiency.

## Highlights & Insights

- **Minimal yet effective**: Annotating frames with simple index numbers substantially improves temporal reasoning. This "modify the input, not the model" paradigm is both elegant and practical, enabling zero-cost integration into any VideoLLM.
- **Positional bias discovery**: The marked superiority of bottom-positioned VP over top-positioned VP reveals a training-induced bias in VideoLLMs—stronger attention toward bottom regions—a finding with broader implications for all VP-based methods.
- **Frames as a dictionary**: Treating frame indices as keys and frame contents as values introduces a novel paradigm for fine-grained temporal control in VideoLLMs.

## Limitations & Future Work

- The keyword extraction component of KFM relies on an external embedding model, which may become a bottleneck for very long videos.
- VP inherently occupies pixel space within frames; interference may arise for videos that already contain subtitles or watermarks.
- The positional bias suggests that the model may be "memorizing" text at specific locations rather than genuinely comprehending temporal relationships.
- Future directions include: adaptive VP size/position selection and joint optimization with frame selection strategies.

## Related Work & Insights

- **vs. conventional frame selection methods**: Frame selection focuses on *which frames to retain*; ViKey focuses on *how to make retained frames more informative*. The two approaches are complementary.
- **vs. temporal encoding enhancement methods**: Training-required methods such as extended context modules achieve comparable performance, whereas ViKey requires no training whatsoever.
- **vs. spatial VP methods**: Prior VP work guides spatial attention within frames (e.g., circle annotations); ViKey is the first to systematically explore VP for cross-frame temporal reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ A simple yet insightful observation; the first systematic exploration of VP for temporal reasoning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three analytical experiments, four benchmarks, and multiple models—highly rigorous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, experimental design is precise, and analysis is thorough.
- **Value**: ⭐⭐⭐⭐ Training-free and plug-and-play; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](../../NeurIPS2025/multimodal_vlm/learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[AAAI 2026\] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](../../AAAI2026/multimodal_vlm/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)

</div>

<!-- RELATED:END -->
