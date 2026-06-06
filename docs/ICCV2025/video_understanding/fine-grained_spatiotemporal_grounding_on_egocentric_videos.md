---
title: >-
  [Paper Note] Fine-grained Spatiotemporal Grounding on Egocentric Videos
description: >-
  [ICCV 2025][Video Understanding][egocentric video] This paper presents EgoMask, the first pixel-level spatiotemporal grounding benchmark for egocentric videos…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "egocentric video"
  - "spatiotemporal grounding"
  - "pixel-level segmentation"
  - "benchmark"
date: 2026-05-08
content_hash: 98ff53ae70116f8a
---

# Fine-grained Spatiotemporal Grounding on Egocentric Videos

**Conference**: ICCV 2025
**arXiv**: [2508.00518](https://arxiv.org/abs/2508.00518)  
**Code**: [https://github.com/LaVi-Lab/EgoMask](https://github.com/LaVi-Lab/EgoMask)  
**Area**: Video Understanding
**Keywords**: egocentric video, spatiotemporal grounding, pixel-level segmentation, benchmark, video understanding

## TL;DR

This paper presents EgoMask, the first pixel-level spatiotemporal grounding benchmark for egocentric videos, comprising short/medium/long evaluation splits and a large-scale training set EgoMask-Train. Through systematic analysis, it reveals key differences between egocentric and exocentric videos, and demonstrates that fine-tuned models can achieve substantial performance gains.

## Background & Motivation

Spatiotemporal Video Grounding (STVG) aims to localize target entities in video given textual queries. Existing work has focused predominantly on exocentric videos. Although egocentric videos are increasingly important for AR and robotics applications, pixel-level spatiotemporal grounding in this domain remains largely unexplored.

Quantitative analysis of key differences — entity appearance in egocentric videos exhibits:
- **Shorter total presence**: only 21.56% of video duration (vs. 77–94% in exocentric)
- **Sparser continuous trajectories**: a single trajectory spans only 1.33% of the video (vs. 65–90% in exocentric); absence duration is 6× presence duration
- **Smaller targets**: mask area is only 1.20% of the frame (vs. 5%+ in exocentric)
- **Larger positional displacement**: inter-frame mask IoU is only 14.96% (vs. 50%+ in exocentric)

Existing datasets (EgoTracks provides only bounding boxes; RefEgo covers only short videos) cannot support pixel-level evaluation, motivating the need for a new benchmark.

## Method

### Overall Architecture

The primary contribution of this work is dataset construction rather than model design. An automated annotation pipeline is designed in two stages: (1) pixel-level mask generation; and (2) referring expression generation. These stages are used to build both the evaluation benchmark EgoMask and the training dataset EgoMask-Train.

### Key Designs

1. **Pixel-level Mask Generation**: Leveraging bounding box annotations from EgoTracks, SAM2 is applied for video-level object segmentation. Only segments containing the target object are annotated; the bounding box in the first frame serves as the box prompt for SAM2. Post-processing retains only regions overlapping with the provided bounding box annotations, minimizing hallucination errors.

2. **Referring Expression Generation**: Two strategies are employed to ensure diversity — (1) GPT-4o is prompted directly to generate short and long descriptions (using three frames where the target is most visible, with bounding boxes overlaid); (2) GPT-4o first generates metadata (visual attributes, world knowledge, object functions, etc.), which is then combined via predefined templates into referring expressions. All annotations undergo human verification.

3. **Multi-duration Tiered Evaluation Design**:

    - EgoMask-Short (<1 min, 200 videos, 400 expressions): sampled from RefEgo, with manually annotated masks and refined expressions
    - EgoMask-Medium (1–3 min, 100 videos, 200 expressions): randomly clipped from annotated long videos
    - EgoMask-Long (>3 min, 15 videos, 100 expressions): based on the EgoTracks validation set, generated via the pipeline and manually refined
    - EgoMask-Train: 2,624 videos, 9,592 objects, 47,968 expressions

### Loss & Training

Fine-tuning is applied to two state-of-the-art models:
- **Sa2VA-4B (+FT)**: fine-tuned on EgoMask-Train plus 3 exocentric video segmentation datasets; 8× A100 GPUs, ~10 hours; AdamW, lr=4e-6, batch size=16
- **VideoLISA-3.8B (+FT)**: 80% EgoMask-Train + 20% original training data; 4× A100 GPUs, 20 epochs, AdamW, lr=3e-5, batch size=16, 500 steps per epoch, ~12 hours

### Evaluation Metric Design

Four metrics are proposed:
- **T_recall**: ratio of predicted frames to ground-truth frames (temporal localization ability)
- **IoU_all**: mean IoU over all frames (equivalent to the conventional $\mathcal{J}$ metric)
- **IoU_gold**: mean IoU computed only over ground-truth frames
- **IoU_gold_pred**: mean IoU computed over all ground-truth and predicted frames (penalizes hallucinated predictions on background frames)

## Key Experimental Results

### Main Results

Results on the EgoMask benchmark (IoU_gold_pred):

| Method | Short | Medium | Long |
|--------|-------|--------|------|
| Grounded-SAM2 | 49.95 | 25.73 | 24.80 |
| Sa2VA-26B | 37.30 | 25.83 | 12.96 |
| Sa2VA-4B | 29.00 | 17.02 | 8.11 |
| Sa2VA-4B (+FT) | 30.97 (+1.97) | 18.52 (+1.50) | 8.24 (+0.13) |
| VideoLISA-3.8B | 17.85 | 6.48 | 5.15 |
| VideoLISA-3.8B (+FT) | **23.36 (+5.51)** | **9.98 (+3.50)** | **7.16 (+2.01)** |

Comparison on exocentric benchmarks before and after fine-tuning (verifying that original capability is preserved):

| Method | Ref-Davis | Mevis | ReasonVOS |
|--------|-----------|-------|-----------|
| VideoLISA-3.8B | 65.82 | 49.20 | 42.41 |
| VideoLISA-3.8B (+FT) | 65.60 (-0.22) | 49.20 (+0.00) | 44.18 (+1.77) |
| Sa2VA-4B | 69.75 | 50.01 | 42.35 |
| Sa2VA-4B (+FT) | 69.97 (+0.22) | 55.55 (+5.54) | 45.54 (+3.19) |

### Ablation Study

Effect of SAM2 initialization strategy (Grounded-SAM2):

| EgoMask Split | Highest-confidence Init IoU_gold_pred | Naive Init IoU_gold_pred | Gap |
|---------------|--------------------------------------|--------------------------|-----|
| Short | 49.95 | 40.42 | -9.53 |
| Medium | 25.73 | 15.11 | -10.62 |
| Long | 24.80 | 11.65 | -13.15 |

Effect of key frame validity in Sa2VA: when the target object does not appear in the first 5 frames, performance drops sharply to near 0%.

### Key Findings

- All state-of-the-art models perform poorly on egocentric videos: the best result on the Short split is only ~50% IoU_gold_pred, and less than 30% on Medium/Long splits
- Fine-tuning yields substantial improvements: VideoLISA achieves an average relative gain of 41.30% while retaining performance on exocentric benchmarks
- EgoMask-Train is complementary to exocentric data — fine-tuning even improves ReasonVOS by 1.77%
- SAM2 initialization is critical: incorrect initialization causes a performance drop of 10–18%
- The IoU_all metric is misleading for long videos due to the high proportion of background frames (Sa2VA's IoU_all paradoxically increases as video length grows); IoU_gold_pred more faithfully reflects true performance
- Inference speed: VideoLISA (frame-by-frame segmentation) achieves only 0.42 FPS, whereas SAM2-based methods achieve at least 3.17 FPS

## Highlights & Insights

- **Systematic difference analysis**: the first quantitative characterization of four key dimensional differences between egocentric and exocentric videos, providing guidance for future model design
- **Fully automated annotation pipeline**: the combination of SAM2 and GPT-4o substantially reduces annotation cost, requiring only human refinement rather than annotation from scratch
- **Multi-duration tiered design**: Short/Medium/Long splits cover practical application scenarios ranging from seconds to minutes
- **Metric innovation**: IoU_gold_pred penalizes hallucinated predictions and is better suited to sparse target scenarios than conventional IoU_all
- The dataset provides genuine complementary value — fine-tuning on EgoMask-Train improves rather than degrades exocentric performance

## Limitations & Future Work

- Current model designs are not specifically optimized for the characteristics of egocentric videos
- Sa2VA is constrained by input token limits, making key frame selection inflexible — target objects may not appear in the first 5 frames of long videos
- Training set localization annotations are at 1 FPS (vs. the original 5 FPS in EgoTracks), potentially missing fine-grained details of fast-moving objects
- The Long split contains only 15 videos and 100 expressions, which is relatively small in scale
- Promising future directions include: enhancing long-video understanding capability and optimizing frame selection strategies to capture target entities

## Related Work & Insights

- Compared to existing RVOS datasets (Ref-DAVIS, MeViS, ReasonVOS), EgoMask is the first to cover egocentric videos with multi-duration evaluation
- Sa2VA outperforms Grounded-SAM2 on exocentric benchmarks, yet underperforms it on egocentric benchmarks — suggesting that end-to-end models have not fully exploited the potential of pretrained grounding models in egocentric settings
- Frame selection strategy is a key direction for improving SAM2-based methods

## Rating

- **Novelty**: ⭐⭐⭐⭐ First pixel-level egocentric spatiotemporal grounding benchmark, filling an important gap
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparison + fine-tuning validation + detailed analysis + multi-metric evaluation
- **Writing Quality**: ⭐⭐⭐⭐ Thorough analysis, detailed statistics, and rich visualizations
- **Value**: ⭐⭐⭐⭐⭐ The dataset and analysis make a significant contribution to the egocentric video understanding community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[ICCV 2025\] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos](beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)
- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](an_empirical_study_of_autoregressive_pre-training_from_videos.md)

</div>

<!-- RELATED:END -->
