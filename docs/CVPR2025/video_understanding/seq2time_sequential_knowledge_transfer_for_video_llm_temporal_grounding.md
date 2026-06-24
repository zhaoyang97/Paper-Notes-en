---
title: >-
  [Paper Note] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding
description: >-
  [CVPR 2025][Video Understanding][Video Temporal Understanding] Seq2Time proposes a data-driven training paradigm that converts large-scale image sequences and short video clips into training data simulating the temporal structure of long videos, and introduces unified relative position tokens. Without relying on abundant timestamp annotations, this approach significantly enhances the temporal understanding capability of video LLMs (achieving a 27.6% F1 improvement on YouCook2…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Temporal Understanding"
  - "Large Language Models"
  - "Temporal Grounding"
  - "Dense Video Captioning"
  - "Knowledge Transfer"
date: 2026-05-08
content_hash: 73b9232a21195c2e
---

# Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding

**Conference**: CVPR 2025  
**arXiv**: [2411.16932](https://arxiv.org/abs/2411.16932)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Temporal Understanding, Large Language Models, Temporal Grounding, Dense Video Captioning, Knowledge Transfer

## TL;DR

Seq2Time proposes a data-driven training paradigm that converts large-scale image sequences and short video clips into training data simulating the temporal structure of long videos, and introduces unified relative position tokens. Without relying on abundant timestamp annotations, this approach significantly enhances the temporal understanding capability of video LLMs (achieving a 27.6% F1 improvement on YouCook2 and a 14.7% R@1 improvement on Charades-STA).

## Background & Motivation

**Background**: Video Large Language Models (Video LLMs) have made significant progress in general video understanding, but their temporal perception capabilities remain insufficient. Temporal-sensitive Video LLMs (such as TimeChat, VTimeLLM) improve temporal understanding through architectural innovations (e.g., dual Q-Former, multi-stage training), but they rely heavily on large amounts of long video data annotated with precise timestamps.

**Limitations of Prior Work**: (1) Timestamp annotations are extremely scarce: TimeIT contains only 125K videos and VTG-IT contains only 120K, which is far fewer than general video datasets (e.g., VideoChat2 has over 800K); (2) reducing training data by 12% results in a 13.4% performance drop, and removing task-relevant data causes a 65.5% plunge; (3) captions in existing datasets are of lower quality, and their vision-language alignment accuracy is inferior to high-quality image datasets (such as LLaVA-150K).

**Key Challenge**: The scarcity of training data (especially long videos with timestamp annotations) severely limits the temporal understanding capabilities of Video LLMs, yet the annotation cost is prohibitively high for large-scale acquisition.

**Goal**: Leveraging abundant image and short-form video data to enhance the temporal understanding capabilities of Video LLMs, bypassing the reliance on scarce timestamp annotations for long videos.

**Key Insight**: Video LLMs do not truly "perceive time"; instead, they recognize the correspondence between visual content and its position in a sequence. Therefore, the index-content correspondence in image sequences can be used to simulate the timestamp-event correspondence in videos. Large-scale image and short video data naturally possess rich position-content correspondence information.

**Core Idea**: Converting the position indices in image sequences into temporal annotations, designing three pretext tasks (image index grounding, index image captioning, and adjacent location reasoning) to train Video LLMs to learn sequence-content correspondences, and then transferring this positional knowledge to long-video temporal understanding through unified relative position tokens.

## Method

### Overall Architecture

Seq2Time consists of three data components: (1) **Image Sequence Data**—sampling 96 images from the LLaVA-ReCap dataset to form sequences and designing three pretext tasks to learn index-content correspondences; (2) **Clip Sequence Data**—sampling 2–10 short video clips from Kinetics-700 to stitch into long sequences, training the model on dense video captioning and temporal grounding; (3) **Unified Relative Position Tokens**—encoding image indices and video frame positions uniformly into 4-decimal relative positions, requiring only 10 new tokens (`<0>`–`<9>`).

### Key Designs

1. **Image Sequence Pretext Tasks**:

    - Function: Enabling Video LLMs to acquire the capabilities of "grounding content location from description" and "generating content description from position."
    - Mechanism: Randomly sampling 96 images from high-quality image-caption datasets (such as COCO118K, BLIP558K, and CC3M) to construct image sequences, and designing three complementary tasks: (a) **Image Index Grounding (IIG)**—finding the corresponding image index given a caption, simulating temporal grounding; (b) **Index Image Captioning (IIC)**—generating the corresponding image description given an index, simulating dense video captioning; (c) **Adjacent Location Reasoning (ALR)**—identifying and describing the preceding/succeeding image given a target image's description. There are 100K instances per task, totaling 300K.
    - Design Motivation: The quality of image captions is significantly higher than that of video captions (frame-by-frame description vs. video-level description). These three tasks respectively strengthen grounding, captioning, and sequence reasoning capabilities. Ablation studies demonstrate that IIG contributes the most to overall temporal understanding, IIC primarily improves text generation quality, and ALR enhances the richness of descriptions.

2. **Clip Sequence Data**:

    - Function: Enhancing temporal perception using data that closely resembles real-world long videos.
    - Mechanism: Sampling 2–10 short clips of different action categories from Kinetics-700, utilizing LongVA to generate detailed captions for each clip (conditioned on action labels), and then concatenating them into simulated long videos. Frame rates are intentionally sampled unevenly to avoid uniform temporal spacing, mimicking the multi-event structure of real-world videos. Dense captioning and temporal grounding training data (100K instances) are generated based on clip positions.
    - Design Motivation: Although image sequences pose harder tasks and have higher caption quality, a modality gap remains compared to real videos. Clip sequences are closer to real long videos in data characteristics and training objectives. The two types of data are complementary: image sequences reinforce fine-grained localization, while clip sequences enhance video-level understanding.

3. **Unified Relative Position Token**:

    - Function: Bridging image indices and video timestamps in the LLM embedding space.
    - Mechanism: Normalizing all positions (either image indices or video frame indices) to 4-decimal fractions: $I_{\text{norm}} = \text{round}(i/L, 4)$, where $i$ is the index and $L$ is the sequence length. For example, the 7th image in a 96-image sequence is encoded as 0.0729 → `<0><7><2><9>`. Only 10 new tokens (`<0>` to `<9>`) are added to the LLM vocabulary, with each digit acting as a learnable embedding. During inference, relative positions can be mapped back to absolute timestamps using the video frame rate.
    - Design Motivation: (1) Absolute time is incomparable across videos with different frame rates, making relative position more generalizable; (2) a 4-decimal precision results in an average error of only 0.13% when sampling 96 frames from a 1-minute 30fps video; (3) hierarchical structure—the first digit represents the coarse location while subsequent digits provide fine-grained localization, suitable for temporal understanding at different scales; (4) using only 10 tokens is highly efficient, avoiding vocabulary expansion overhead.

### Loss & Training

Two-stage training: (1) training for 1 epoch on the complete dataset (Seq2Time 400K + TimeIT 110K + Valley 40K + ShareGPT4Video 93K); (2) fine-tuning for 3 epochs solely on TimeIT + Valley data. LoRA allocation is used with rank=32, batch size of 8, and 96 sampled frames per video. Standard autoregressive cross-entropy loss is applied.

## Key Experimental Results

### Main Results

| Method | YouCook2 SODA_c↑ | CIDEr↑ | F1↑ | Charades R@1(0.5)↑ | R@1(0.7)↑ |
|------|-----------------|--------|------|-------------------|-----------|
| VTimeLLM | - | - | - | 27.5 | 11.4 |
| TimeChat | 1.0 | 2.9 | 12.7 | 27.2 | 11.7 |
| TimeChat+Seq2Time w/o RPT | 1.2 | 3.7 | 15.7 | 29.3 | 12.8 |
| **TimeChat+Seq2Time** | **1.3** (+30%) | **4.2** (+44.8%) | **16.2** (+27.6%) | **31.2** (+14.7%) | **13.7** (+17.1%) |

### Ablation Study

| Data Configuration | YouCook2 CIDEr | F1 | Charades R@0.5 |
|---------|---------------|-----|---------------|
| Baseline (TimeChat) | 2.9 | 12.7 | 27.2 |
| +IS only | 4.3 | 13.3 | 28.8 |
| +IS+MC | 4.0 | 15.9 | 30.9 |
| +IS+MC+CS (Seq2Time) | **4.2** | **16.2** | **31.2** |

| Pretext Task Ablation | CIDEr | F1 |
|------------|-------|-----|
| IIC+IIG+ALR (All) | **4.3** | 13.3 |
| w/o IIC | 3.4 | 12.9 |
| w/o IIG | 2.6 | 11.9 |
| w/o ALR | 2.7 | **14.4** |

### Key Findings

- **The performance of image sequence data is surprisingly strong**: Merely adding image sequence data improves CIDEr from 2.9 to 4.3 (+48.3%), demonstrating that static image sequences can effectively transfer temporal understanding capabilities.
- **IIG (grounding task) contributes the most**: Removing IIG causes a significant drop across both benchmarks, showing that "finding positions from descriptions" is the core capability of temporal understanding.
- **Unified relative position tokens are crucial**: Performance gains are weakened when removing RPT, demonstrating that the unified position representation serves as a bridge for sequence-to-time knowledge transfer.
- Experiments with Video-LLaMA further validate generalizability: training based on image sequences improves F1 from 0.2 to 3.3, showing that temporal awareness can be injected even into general Video LLMs without any prior temporal training.
- Additional video captions (MC) improve temporal grounding but slightly degrade text quality, indicating that the precision of image-level captions uniquely contributes to generation quality.

## Highlights & Insights

- **The insight that "Video LLMs do not perceive time but only position"** is highly inspiring: deconstructing temporal understanding into position-content correspondences opens the door to enhancing expensive capabilities using cheap data. This idea can be generalized to other location-aware tasks (such as layout positioning in document understanding).
- **The position encoding with only 10 new tokens** is extremely efficient: 4-decimal precision is sufficient to express location accuracy with only 0.13% error, and the hierarchical structure naturally supports multi-granularity temporal reasoning.
- The complementary design of the three image sequence pretext tasks (grounding/captioning/reasoning) provides comprehensive capability coverage for sequence learning, which is more effective than simple data scaling.

## Limitations & Future Work

- The reproduced performance of the baseline model TimeChat is lower than reported in the original paper (due to the unavailability of some training data), which affects the reference value of absolute numbers.
- Clip sequence captions rely on LongVA generation, and their quality is constrained by the performance of that model.
- Evaluations are limited to TimeChat and Video-LLaMA, leaving more powerful Video LLMs untested.
- Future work can explore the scaling effects on larger image/clip datasets, as well as extending the method to multimodal (audio + video) temporal understanding.

## Related Work & Insights

- **vs. TimeChat**: TimeChat improves temporal perception via a dual Q-Former architecture, representing an architecture-driven approach. Seq2Time is data-driven, modifying no architecture while achieving significant improvements over TimeChat solely through data augmentation. The two approaches are orthogonal and complementary.
- **vs. VTG-LLM**: VTG-LLM introduces absolute temporal tokens to reduce quantization errors, but absolute time is not generalizable across different frame rates. Seq2Time’s relative position tokens are more flexible and require only 10 new tokens.
- **vs. Grounded-VideoLLM**: Also utilizes relative temporal tokens, but Seq2Time is the first to systematically leverage image and short video sequence data to enhance temporal reasoning capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ The approach of "simulating video temporality with image sequences" is novel, and the position token design is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough data ablations, task ablations, scaling experiments, and cross-model validation.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear, and the experimental design is highly reasonable.
- Value: ⭐⭐⭐⭐ Provides an effective, low-cost path to enhancing the temporal capabilities of Video LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[ECCV 2024\] R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](../../ECCV2024/video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)

</div>

<!-- RELATED:END -->
