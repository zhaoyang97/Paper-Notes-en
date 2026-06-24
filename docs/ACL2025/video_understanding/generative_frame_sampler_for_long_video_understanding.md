---
title: >-
  [Paper Note] Generative Frame Sampler for Long Video Understanding
description: >-
  [ACL 2025][Video Understanding][Long Video Understanding] GenS is proposed, a generative frame sampling module based on VideoLLM. It outputs question-aware relevant frame intervals and confidence scores in natural language format. As a plug-and-play module, it consistently improves multiple VideoLLMs by 2-4 points on LongVideoBench, MLVU, and HourVideo.
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Frame Sampling"
  - "VideoLLM"
  - "Question-Aware Sampling"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: fb9ccb813591bdfc
---

# Generative Frame Sampler for Long Video Understanding

**Conference**: ACL 2025  
**arXiv**: [2503.09146](https://arxiv.org/abs/2503.09146)  
**Code**: [https://generative-sampler.github.io](https://generative-sampler.github.io)  
**Area**: Video Understanding / Multimodal VLM  
**Keywords**: Long Video Understanding, Frame Sampling, VideoLLM, Question-Aware Sampling, Plug-and-Play

## TL;DR
GenS is proposed, a generative frame sampling module based on VideoLLM. It outputs question-aware relevant frame intervals and confidence scores in natural language format. As a plug-and-play module, it consistently improves multiple VideoLLMs by 2-4 points on LongVideoBench, MLVU, and HourVideo.

## Background & Motivation

**Background**: Long video understanding is a core challenge for current VideoLLMs. Due to context window limitations, VideoLLMs must sample a limited number of frames from long videos. Mainstream methods use uniform sampling or fixed FPS, ignoring the relevance between the question and the video content.

**Limitations of Prior Work**: (1) Uniform sampling wastes a large number of tokens on irrelevant frames in long videos; (2) Semantic matching sampling based on CLIP or SigLIP can only score single frames independently, failing to understand temporal relationships between frames; (3) There is a lack of sampling methods that consider multi-hop reasoning and temporal logic.

**Key Challenge**: Efficient frame sampling is the bottleneck of long video understanding—the sampling strategy directly determines whether the VideoLLM can "see" the key frames required to answer the question.

**Goal**: To design a question-aware frame sampler that can understand temporal relationships between frames and select the most relevant frames.

**Key Insight**: Modeling frame sampling as a generative task, using a VideoLLM to directly output relevant frame intervals and confidence scores in natural language format.

**Core Idea**: Using a VideoLLM (Aria with a 256-frame window) to process sparsely sampled video frames to generate continuous intervals and confidence scores (0-5). Dense sampling is then performed from the high-confidence intervals.

## Method

### Overall Architecture
Four-stage dataset construction (GenS-Video-150K) $\rightarrow$ Training the GenS sampler (based on Aria) $\rightarrow$ At inference, sampling first with GenS and then feeding the sampled frames into any downstream VideoLLM for answering.

### Key Designs

1. **GenS-Video-150K Dataset Construction**:

    - **Function**: Creating large-scale frame sampling training data.
    - **Mechanism**: (1) Densely sampling frames from video and using a VLM to generate descriptions for each frame; (2) Using an LLM to generate QA pairs based on the frame descriptions while labeling grounded frames; (3) Using CLIP to expand relevant frame windows around the labeled frames; (4) Labeling the relevance of each frame to the question with fine-grained scores (0-5).
    - **Design Motivation**: Approximately 20% of the frames are labeled (sparse yet precise). Continuous scores from 0 to 5 are more flexible than binary labels and support top-K retrieval.

2. **Generative Frame Sampling (GenS)**:

    - **Function**: Modeling frame sampling as a text-generation task.
    - **Mechanism**: Inputting sparsely sampled frames and a question, and outputting continuous frame intervals (e.g., "frames 10-25") along with corresponding confidence scores. Frames are densely sampled from high-scoring intervals after sorting by confidence.
    - **Design Motivation**: Generating continuous intervals (instead of discrete frame indices) captures temporal continuity; sorting by confidence enables adaptive top-K sampling.

3. **Plug-and-Play Design**:

    - **Function**: GenS operates independently of downstream VideoLLMs.
    - **Mechanism**: GenS first samples key frames $\rightarrow$ Key frames are fed into any downstream VideoLLM $\rightarrow$ Downstream VideoLLM answers the question. GenS is based on Aria (256-frame context), while downstream VLMs can be any model.
    - **Design Motivation**: Decoupling sampling and understanding, allowing a single GenS to serve all downstream VideoLLMs.

### Loss & Training
- Fine-tuning based on the Aria model with standard next-token prediction loss.
- Task-specific prompts outperform unified prompts.
- Text-only indices outperform mixed vision-text indices.

## Key Experimental Results

### Main Results

| Baseline VideoLLM | LongVideoBench | MLVU | HourVideo |
|-------------|---------------|------|----------|
| LLaVA-Video-72B | 62.5$\rightarrow$**66.8** (+4.3) | 74.3$\rightarrow$**77.0** (+2.7) | - |
| Aria | 58.7$\rightarrow$**66.1** (+7.4) | 69.5$\rightarrow$**72.6** (+3.1) | 37.3$\rightarrow$**39.2** (+1.9) |
| Qwen2-VL-7B | 58.7$\rightarrow$**60.3** (+1.6) | 64.7$\rightarrow$**66.9** (+2.2) | - |
| GPT-4o | 66.7$\rightarrow$**67.6** (+0.9) | - | - |
| Gemini-1.5-pro | - | - | 37.3$\rightarrow$**40.7** (+3.4) |

### Ablation Study

| Sampling Strategy | LongVideoBench (Aria) |
|---------|---------------------|
| Uniform | 54.4 |
| CLIP sampler | ~55 |
| GenS (GenS-Video-150K data only) | 57.7 (+3.3) |
| GenS full | **66.1** (+11.7) |

### Key Findings
- **Consistent Improvement Across All Models**: Both open-source (Qwen, LLaVA, Aria, VILA) and closed-source (GPT-4o, Gemini) models show improvements of 1-7 points.
- **Continuous Intervals > Discrete Frame Indices**: Outputting continuous frame intervals + confidence (56.1) outperforms using discrete frame indices.
- **Largest Improvement on LongVideoBench**: This benchmark requires multi-hop reasoning across time, showcasing the advantage of question-aware sampling most clearly.
- **GenS Can Generalize from Small to Large Models**: The GenS trained on Aria effectively enhances the 72B LLaVA-Video model.

## Highlights & Insights
- **Paradigm Shift of "Frame Sampling as Generation"**: Redefines sampling from a retrieval/matching problem to a generation problem, allowing the VideoLLM to directly "speak out" which frames are important. This enables the model to leverage its temporal reasoning capabilities for sampling.
- **Soft Ranking with Confidence Scores**: More flexible than a hard binary selection, supporting different downstream models in retrieving top-K frames as needed.
- **Small Model for Sampling, Large Model for Understanding**: A cost-effective division of labor strategy.

## Limitations & Future Work
- GenS needs to process a window of 256 frames, leading to non-negligible computational overhead (which can be mitigated by parallel windows).
- Currently a single-round sampling method; multi-round iterative retrieval and Video Agent integration have not yet been explored.
- Training data relies on the quality of frame descriptions from the VLM; errors in description propagate to the sampling labels.
- Evaluated only on question-answering tasks; other tasks like video summarization and video grounding have not been tested.

## Related Work & Insights
- **vs. CLIP/SigLIP Sampling**: These methods score single frames independently and fail to understand temporal relationships. GenS outputs intervals, inherently taking temporal continuity into account.
- **vs. TimeChat**: TimeChat uses time-aware encoding but does not perform explicit sampling. GenS serves as an explicit sampling front-end.
- **vs. FPS/Uniform Sampling**: Wastes a large number of tokens on long videos. GenS achieves smart, question-aware sampling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Modeling frame sampling as a generative task is a fresh perspective, and the dataset construction pipeline is highly complete.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 6+ models and 3 benchmarks, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and clear.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play, general-purpose solution with direct practical value for long video understanding.

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Frame Selection for Long Video Understanding via Reinforcement Learning](../../CVPR2026/video_understanding/efficient_frame_selection_for_long_video_understanding_via_reinforcement_learnin.md)
- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](../../CVPR2026/video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](../../CVPR2025/video_understanding/m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](../../CVPR2026/video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](../../CVPR2025/video_understanding/drvideo_document_retrieval_based_long_video_understanding.md)

</div>

<!-- RELATED:END -->
