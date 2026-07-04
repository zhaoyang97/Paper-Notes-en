---
title: >-
  [Paper Note] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding
description: >-
  [CVPR 2025][Multimodal VLM][Long video understanding] Leveraging the internal KV sparsification capability of LLMs for long video token compression by introducing Visual Summarization Tokens (VST) to compress the visual information of each video segment into its KV and offloading the original visual KV. Combined with dynamic compression and curriculum learning, it processes 2048 frames on a single A100 and outperforms GPT-4o on MLVU Dev.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Long video understanding"
  - "KV cache compression"
  - "visual summarization token"
  - "curriculum learning"
  - "hour-scale video"
date: 2026-05-08
content_hash: 7aadc143156d5936
---

# Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2409.14485](https://arxiv.org/abs/2409.14485)  
**Code**: [https://github.com/VectorSpaceLab/Video-XL](https://github.com/VectorSpaceLab/Video-XL)  
**Area**: Multimodal VLM  
**Keywords**: Long video understanding, KV cache compression, visual summarization token, curriculum learning, hour-scale video

## TL;DR
Leveraging the internal KV sparsification capability of LLMs for long video token compression by introducing Visual Summarization Tokens (VST) to compress the visual information of each video segment into its KV and offloading the original visual KV. Combined with dynamic compression and curriculum learning, it processes 2048 frames on a single A100 and outperforms GPT-4o on MLVU Dev.

## Background & Motivation

**Background**: Long-video understanding (hour-scale) is a critical application direction for VLMs, but thousands of frames imply hundreds of thousands of visual tokens, far exceeding the LLM's context window.

**Limitations of Prior Work**: Existing compression methods are performed prior to LLM processing (e.g., pooling, Q-Former, C-Abstractor), resulting in severe performance degradation at high compression ratios ($16\times$). This is because the compression and utilization of visual information are decoupled, leaving the compressor unaware of which information is crucial for subsequent inference.

**Key Challenge**: Long videos require extremely high compression ratios ($16\times$ or higher), but pre-compression methods lose too much critical information at high ratios.

**Goal**: To leverage the internal attention mechanism of LLMs for semantic-aware KV compression, rather than relying on blind token compression outside the LLM.

**Key Insight**: To insert learnable Visual Summarization Tokens (VST) into video segments, allowing the LLM's attention mechanism to naturally compress visual information into the KV of VSTs, followed by offloading the original visual token KV, leaving only the VST KV.

**Core Idea**: To shift visual token compression from outside the LLM to inside the LLM—replacing the original visual KV with VST KV caches to utilize the LLM's own attention for semantic-aware compression.

## Method

### Overall Architecture
Video frames are processed by a visual encoder to extract tokens $\to$ a set of VST tokens is inserted every $n$ frames $\to$ sent to the LLM $\to$ self-attention during LLM processing compresses key visual information into the VST's KV $\to$ original visual token KV is offloaded, keeping only VST KV + text KV $\to$ subsequent inference uses VST KV to represent the visual context.

### Key Designs

1. **Visual Summarization Token（VST）**:

    - **Function**: Serves as the carrier for visual information compression inside the LLM.
    - **Mechanism**: Insert learnable VST tokens after each segment of video frames. The causal attention of the LLM allows VSTs to attend to all preceding visual tokens, so their KV naturally encodes previous visual information. After processing, the original visual KV is offloaded, retaining only the VST KV.
    - **Design Motivation**: Ablation studies show that internal LLM compression (VST) achieves an MLVU score of 41.4 at a $16\times$ compression ratio, significantly outperforming pooling (33.7), Q-Former (35.1), and C-Abstractor (37.1).

2. **Dynamic Compression**:

    - **Function**: Adaptively adjusts the compression granularity based on the information density of the video content.
    - **Mechanism**: Uses CLIP depth scores to measure the magnitude of semantic changes between adjacent frames—regions with high changes (e.g., scene cuts) receive finer granularity (more VSTs), while regions with low changes (e.g., static shots) receive coarser granularity.
    - **Design Motivation**: Uniform compression loses key details in information-dense areas and wastes capacity in information-sparse areas.

3. **Curriculum Learning**:

    - **Function**: Progressively guides the model to learn increasingly higher compression ratios.
    - **Mechanism**: Employs low compression ratios ($2\times$, $4\times$) in the early stages of training, gradually increasing to higher ratios ($8\times$, $12\times$, $16\times$). This prevents training collapse caused by facing extremely high compression ratios from the start.
    - **Design Motivation**: Direct training at $16\times$ yields suboptimal results (MLVU 37.2 vs. 41.4 with curriculum learning). Progressive learning allows the model to step-by-step acquire the compression capability.

### Loss & Training
Standard next-token prediction loss. The training data mixes image, multi-image, and long-video data. The VICO synthetic dataset (a visual clue ordering task generated from CinePile videos) is used to enhance long-range understanding.

## Key Experimental Results

### Main Results

| Model | Size | MLVU Dev | VideoMME | VNBench | LongVidBench |
|------|------|---------|----------|---------|-------------|
| GPT-4o | - | 64.6 | 71.9 | 64.4 | 66.7 |
| LongVA | 7B | 56.3 | 52.6 | 41.5 | 47.8 |
| **Video-XL** | **7B** | **64.9** | **55.5** | **61.6** | **50.7** |

### Ablation Study

| Compression Method ($16\times$) | MLVU | VideoMME | MME | MMB |
|---------------|------|---------|-----|-----|
| Pooling | 33.7 | 41.0 | 1405 | 62.3 |
| Q-Former | 35.1 | 42.1 | 1410 | 61.9 |
| C-Abstractor | 37.1 | 46.3 | 1440 | 65.1 |
| **Video-XL (VST)** | **41.4** | **52.0** | **1510** | **70.9** |

### Key Findings
- **7B Model Outperforms GPT-4o**: Video-XL 7B (64.9) outperforms GPT-4o (64.6) on MLVU Dev, demonstrating the effectiveness of the internal compression strategy.
- **VST >> External Compression at $16\times$ Compression**: VST outperforms C-Abstractor, the strongest external method, by 4.3 points on MLVU, because the internal attention of the LLM knows which information is important for inference.
- **2048 Frames on a Single A100**: Maintains 95% accuracy in Needle-in-Haystack tests, enabling hour-scale video understanding.

## Highlights & Insights
- **"In-LLM Compression" disrupts the external compression paradigm**—allowing the LLM itself to decide what to preserve is far more accurate than an external compressor making blind predictions.
- **Curriculum learning is key to high compression ratios**: Direct $16\times$ training vs. progressive training shows a 4.2-point performance gap.
- **The VST concept is generalizable** to any LLM applications requiring long sequence compression (e.g., long documents, multi-image understanding).

## Limitations & Future Work
- The KV cache of VST still consumes memory; extremely long videos (10+ hours) might still be constrained.
- The calculation of CLIP depth scores for dynamic compression introduces additional overhead.
- Validation is only conducted on 7B models; the performance on larger models remains unknown.

## Related Work & Insights
- **vs. LongVA / LLaMA-VID**: These methods utilize external token compression, resulting in steep performance drops at high compression ratios. In contrast, Video-XL's internal compression maintains high accuracy at $16\times$.
- **vs. MovieChat / StreamingLLM**: These methods process streaming video but do not guarantee long-range understanding. Video-XL's VSTs explicitly encode long-range dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Moving from external token compression to internal LLM KV compression is a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple long-video benchmarks + Needle-in-Haystack + compression method comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of the method; the compression comparison experiments are compelling.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to long-video understanding; outperforming GPT-4o with a 7B model is a highlight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[CVPR 2025\] SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation](svlta_benchmarking_vision-language_temporal_alignment_via_synthetic_video_situat.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)

</div>

<!-- RELATED:END -->
