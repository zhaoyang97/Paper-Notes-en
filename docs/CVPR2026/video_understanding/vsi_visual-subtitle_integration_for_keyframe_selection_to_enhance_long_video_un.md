---
title: >-
  [Paper Note] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding
description: >-
  [CVPR 2026][Video Understanding][Long video understanding] VSI proposes a dual-branch collaborative retrieval framework (Video Search + Subtitle Match) that fuses visual and textual information for precise keyframe local…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Long video understanding"
  - "keyframe retrieval"
  - "multimodal fusion"
  - "video question answering"
  - "subtitle matching"
date: 2026-05-08
content_hash: 7abe477d99a42490
---

# VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2508.06869](https://arxiv.org/abs/2508.06869)  
**Code**: [https://github.com/Jacksonha7/Visual-Subtitle-Integration.git](https://github.com/Jacksonha7/Visual-Subtitle-Integration.git)  
**Area**: Video Understanding
**Keywords**: Long video understanding, keyframe retrieval, multimodal fusion, video question answering, subtitle matching

## TL;DR
VSI proposes a dual-branch collaborative retrieval framework (Video Search + Subtitle Match) that fuses visual and textual information for precise keyframe localization. On text-dominant subtasks, it improves search accuracy from 29.48 to 45.00, representing the first cross-modal keyframe retrieval method.

## Background & Motivation
1. **Background**: Multimodal large language models achieve strong performance on vision-language tasks, but processing long videos is constrained by input context length and high computational cost, making sparse frame sampling a necessary preprocessing step.
2. **Limitations of Prior Work**: (i) Existing keyframe search algorithms effectively improve performance only on visually dominant subtasks, with marginal gains on text-dominant subtasks; (ii) existing methods rely solely on visual-modal retrieval and lack targeted textual guidance, causing keyframes to over-focus on visually dense regions while deviating from core semantics.
3. **Key Challenge**: VideoQA is inherently multimodal (visual + textual), yet existing keyframe retrieval exploits only the visual modality, resulting in insufficient utilization of modal information.
4. **Goal**: Design a multimodal keyframe retrieval framework that works effectively on text-related tasks while preserving performance on visual tasks.
5. **Key Insight**: Leverage video subtitles as complementary textual cues, integrating visual and textual information through a dual-branch design.
6. **Core Idea**: The Video Search branch processes visual features and object detection; the Subtitle Match branch performs semantic similarity computation; both branches update frame-level sampling probabilities through a dynamic fusion strategy.

## Method

### Overall Architecture
Given an input video and query, the framework initializes frame weights and then, over multiple iterations: (1) the Video Search branch performs preliminary keyframe sampling via object detection and visual feature extraction; (2) the Subtitle Match branch obtains complementary textual information by computing semantic similarity between the query and subtitles; (3) confidence scores from both branches are fused via spline interpolation to update frame-level relevance probabilities. After all iterations, the top-$K$ frames are selected and passed to the downstream QA model.

### Key Designs

1. **Video Search Branch**:
    - **Function**: Query-relevant keyframe filtering based on visual features and object detection.
    - **Mechanism**: A VLM first analyzes the query to identify two categories of objects — Target Objects directly related to the query and Cue Objects providing indirect contextual clues. YOLO-World is then applied to efficiently detect objects in sampled frames. Per-frame scores are computed based on overlap between detected objects and the predefined target set: $S_{\text{obj}}(t) = \max_{o \in \mathcal{O}_t \cap \mathcal{T}}(s_o \cdot w_o)$.
    - **Design Motivation**: Pure visual feature matching may overlook the semantic intent of the query; object detection enables more precise localization of query-relevant visual content. The inclusion of Cue Objects improves robustness.

2. **Subtitle Match Branch**:
    - **Function**: Retrieve subtitle segments semantically aligned with the query.
    - **Mechanism**: A contrastive text embedding model computes semantic similarity between the query and video subtitles, mapping time intervals of high-similarity subtitle segments to frame weights. This provides information for text-related tasks that visual search cannot capture.
    - **Design Motivation**: Answers to many VideoQA tasks are embedded in dialogue or narration, which pure visual search cannot access. Subtitles, as an explicit textual modality, serve as a natural complementary cue.

3. **Dynamic Score Fusion**:
    - **Function**: Fuse confidence scores from the visual and textual branches.
    - **Mechanism**: Frame-level scores from both branches are normalized, smoothed along the temporal dimension via spline interpolation, and then weighted-summed to update the frame sampling probability distribution. Multiple iterations allow the distribution to progressively concentrate on semantically dense regions.
    - **Design Motivation**: Naive score addition may yield temporally discontinuous distributions; spline interpolation ensures temporal continuity.

### Loss & Training
VSI is a training-free, plug-and-play method requiring no additional training. The search model and encoding model can be flexibly replaced as needed.

## Key Experimental Results

### Main Results

| Dataset / Setting | Metric | VSI | Uniform Sampling | VSLS | Gain |
|---|---|---|---|---|---|
| LongVideoBench (8 frames) | Search Accuracy | 73.89% | baseline | 2nd best | Significant |
| LongVideoBench text tasks (8 frames) | Acc | 45.00 | — | 29.48 | +15.52 |
| GPT-4o Long split | Score | 69.57 | 53.76 | — | +15.81 |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| Full VSI | Best | Complete dual-branch model |
| Video Search only | Good on visual, poor on text | Missing textual information |
| Subtitle Match only | Good on text, poor on visual | Missing visual information |
| w/o Dynamic Fusion | Degraded | Simple concatenation underperforms spline fusion |

### Key Findings
- The most significant improvement occurs on text-related tasks (29.48→45.00), validating the value of the Subtitle Match branch.
- VSI yields consistent gains across different downstream models (GPT-4o, LLaVA-Video-7B, Qwen2.5-VL-7B).
- Even on visually dominant tasks, the dual-branch fusion outperforms visual search alone.

## Highlights & Insights
- **First extension of keyframe retrieval from unimodal to multimodal**, opening a new research direction.
- **Plug-and-play design** allows flexible replacement of the search and encoding models, ensuring strong generalizability.
- The concept of Cue Objects is elegant — providing contextual clues rather than direct answers, which enhances retrieval robustness.

## Limitations & Future Work
- Subtitle matching depends on the availability and quality of subtitles, making it inapplicable to videos without subtitles.
- The computational overhead of multi-round iterations grows with video length.
- Future work may explore more fine-grained visual-textual interaction mechanisms.

## Related Work & Insights
- **vs. TStar**: TStar relies solely on visual object detection; the proposed method adds a Subtitle Match branch, yielding substantially better performance on text-dominant tasks.
- **vs. VSLS**: VSLS models inter-frame relationships but remains a purely visual approach; this work introduces the textual modality to achieve cross-modal advantages.

## Rating
- Novelty: ⭐⭐⭐⭐ Multimodal keyframe retrieval is a novel direction, though the technical means are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and models.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ High practical value; plug-and-play design facilitates deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)

</div>

<!-- RELATED:END -->
