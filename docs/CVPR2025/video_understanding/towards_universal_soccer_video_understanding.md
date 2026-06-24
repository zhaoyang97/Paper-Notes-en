---
title: >-
  [Paper Note] Towards Universal Soccer Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Soccer video understanding] This paper constructs SoccerReplay-1988, the largest multimodal soccer dataset to date (1,988 full matches), and proposes a soccer-specific vision encoder, MatchVision. By utilizing a spatiotemporal attention mechanism, it unifiedly handles multi-task requirements including event classification, commentary generation, and foul recognition, achieving SOTA performance on several benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Soccer video understanding"
  - "Multimodal dataset"
  - "Vision encoder"
  - "Spatiotemporal attention"
  - "Commentary generation"
date: 2026-05-08
content_hash: 0c1f0d63a8879838
---

# Towards Universal Soccer Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2412.01820](https://arxiv.org/abs/2412.01820)  
**Code**: [https://jyrao.github.io/UniSoccer/](https://jyrao.github.io/UniSoccer/)  
**Area**: Video Understanding  
**Keywords**: Soccer video understanding, Multimodal dataset, Vision encoder, Spatiotemporal attention, Commentary generation

## TL;DR

This paper constructs SoccerReplay-1988, the largest multimodal soccer dataset to date (1,988 full matches), and proposes a soccer-specific vision encoder, MatchVision. By utilizing a spatiotemporal attention mechanism, it unifiedly handles multi-task requirements including event classification, commentary generation, and foul recognition, achieving SOTA performance on several benchmarks.

## Background & Motivation

**Background**: Soccer video analysis heavily relies on the SoccerNet dataset series (500 matches). Prior research typically designs specialized models for distinct tasks (such as action detection, commentary generation), resulting in fragmented solutions.

**Limitations of Prior Work**: (1) Limited data scale—SoccerNet contains only 500 match videos, lacking adequate data diversity; (2) Severe model fragmentation—different tasks use different models, lacking a unified framework; (3) General vision models (e.g., CLIP, InternVideo) are not optimized for fast-paced, multi-interactive sports scenarios like soccer, leading to sub-optimal performance.

**Key Challenge**: Soccer video understanding requires simultaneously capturing spatial information (player positions, ball trajectories) and temporal information (action evolution, match tempo), whereas general models fail to effectively model the spatiotemporal correlations in soccer scenes.

**Goal**: (1) Construct a large-scale, high-quality soccer video dataset; (2) Develop a unified soccer vision encoder that generalizes across various downstream tasks.

**Key Insight**: The authors observe that text commentaries in soccer matches are naturally aligned with video time, which can serve as a foundation for automated annotation, enabling the large-scale construction of multimodal datasets.

**Core Idea**: Build an ultra-large-scale dataset utilizing an automated annotation pipeline, and pre-train a soccer-specific vision encoder based on spatiotemporal attention on top of it, serving as a unified framework for multiple tasks.

## Method

### Overall Architecture

The proposed approach consists of two components: dataset construction and model design. Regarding data, 1,988 full soccer match videos are collected from the internet, and event labels and text commentaries are generated using an automated annotation pipeline. Regarding the model, SigLIP serves as the backbone, onto which spatiotemporal attention modules are added to construct the MatchVision encoder. Video features output by the encoder are adapted to event classification, commentary generation, and foul recognition via different task heads.

### Key Designs

1. **SoccerReplay-1988 Dataset**:

    - Function: Provide large-scale multimodal soccer training data
    - Mechanism: Collect 1,988 match videos (totaling 3,323 hours) from the 2014-2024 seasons across six major European leagues. Leverage the MatchTime model for temporal alignment, synchronizing the timestamps of text commentaries with video frames. Employ LLaMA-3-70B to automatically extract event categories from commentary texts (expanding from 17 to 24 categories, covering modern rules like VAR). Finally, apply anonymization to replace entities like players and coaches with placeholders.
    - Design Motivation: Existing SoccerNet has only 500 matches, lacking sufficient scale and diversity, which limits model training effectiveness. The automated pipeline makes data construction scalable, and manual verification on a random 2% sample shows an accuracy of 98%.

2. **MatchVision Spatiotemporal Encoder**:

    - Function: Extract spatiotemporal features from soccer video clips
    - Mechanism: Input a video frame sequence $\mathcal{V} \in \mathbb{R}^{T \times 3 \times H \times W}$, where each frame is projected via ViT-style Token Embedding followed by spatial and temporal position encodings. The core is the interleaved stacking of temporal self-attention layers and spatial self-attention layers—temporal attention enables tokens at the same spatial position to interact across frames, while spatial attention enables tokens within the same frame to interact. After $K$ spatiotemporal attention blocks, frame-level [cls] tokens are concatenated via an aggregation layer to form the video-level feature $\mathcal{F}_{\mathcal{V}} \in \mathbb{R}^{T \times D}$.
    - Design Motivation: Divided spatiotemporal attention (similar to TimeSformer) requires significantly less computation than joint attention, while effectively capturing inter-frame motion dynamics and intra-frame spatial relations, making it highly suitable for fast-moving soccer scenes.

3. **Multi-Task Head Design**:

    - Function: Adapt general visual features to different downstream tasks
    - Mechanism: (1) Event classification head: aggregate frame-level features into [cls] tokens using temporal self-attention, followed by a linear classifier trained with cross-entropy loss; (2) Commentary generation head: integrate visual features via a Perceiver aggregator and project them into prefix embeddings for the LLM using an MLP, allowing the LLM to auto-regressively decode and generate text commentaries; (3) Foul recognition head: apply pooling to multi-view video features, then predict foul types (8 categories) and severity levels (4 grades) using a shared MLP and bilinear classifiers, respectively.
    - Design Motivation: The unified encoder + multi-task head design allows the encoder, once pre-trained, to flexibly adapt to various tasks without the need to retrain the visual feature extractor for each individual task.

### Loss & Training

During the pre-training phase, two strategies are explored: (1) Supervised classification—directly training with event labels using cross-entropy loss; (2) Video-language contrastive learning—employing a SigLIP-like sigmoid loss to contrast video features with text commentary encodings. During training, highly similar commentaries (e.g., "match kickoff") within the same batch are treated as positive pairs. In the downstream task phase, the encoder is frozen, and only the task heads are trained.

## Key Experimental Results

### Main Results

| Vision Encoder | Pre-training Data | Classification Acc@1 | Classification Acc@3 | Commentary CIDEr |
|---|---|---|---|---|
| SigLIP (off-the-shelf) | - | 50.2 | 86.7 | 31.38 |
| MatchVision (sup) | SN | 82.5 | 96.6 | 36.15 |
| MatchVision (sup) | SN+MT+SR | 84.0 | 97.3 | 42.20 |
| MatchVision (contra) | All | **85.7** | **97.7** | **44.12** |

MatchVision improves on event classification by approximately 35 percentage points (Acc@1) compared to the strongest off-the-shelf model (SigLIP), and is further enhanced after pre-training on the larger self-constructed dataset.

### Ablation Study

| Configuration | Acc@1 | CIDEr |
|---|---|---|
| Pre-trained on SN data only | 82.5 | 36.15 |
| Add MT + SR data | 84.0 | 42.20 |
| SigLIP backbone (w/o spatiotemporal attention) | 57.9 | 38.24 |
| MatchVision (w/ spatiotemporal attention) | 84.0 | 42.20 |

Ablation studies demonstrate that: (1) Increasing data scale yields consistent gains; (2) The spatiotemporal attention module is crucial—compared to using SigLIP directly, adding spatiotemporal attention increases classification accuracy from 57.9% to 84.0%.

### Key Findings

- MatchVision also significantly outperforms existing methods on the foul recognition task (SoccerNet-MVFoul), demonstrating its universality.
- Contrastive learning pre-training achieves better results than supervised classification pre-training because text commentaries provide richer semantic signals than discrete labels.
- The 98% accuracy of the automated annotation pipeline validates the feasibility of large-scale annotation.
- On the more challenging SoccerReplay-test benchmark, MatchVision maintains its leading position.

## Highlights & Insights

1. **Data-driven paradigm shift**: Moving from auxiliary annotated small datasets to automated annotation of large datasets, aligning with current AI trends.
2. **Value of a unified framework**: A single encoder unifiedly handles three types of tasks (classification, generation, recognition), reducing deployment costs.
3. **Significant impact of spatiotemporal attention**: Divided spatiotemporal attention yields huge improvements in scenarios with strong spatiotemporal dynamics like soccer.
4. **Scalability**: The annotation pipeline can be directly applied to more match video footage, making the data scale virtually boundless.

## Limitations & Future Work

- The dataset currently covers only European leagues; generalization to other tournaments (e.g., South American or Asian leagues) remains unexplored.
- Only video and text modalities are currently processed, leaving audio information (such as crowd noise, commentator speech) unutilized.
- Commentary generation requires anonymization processing, which reduces the practical utility of the generated text.
- Future work can extend the framework to other sports events (e.g., basketball, tennis) to verify its generalizability.

## Related Work & Insights

- Relationship with SoccerNet series: This work expands the data scale by a factor of 4 on top of SoccerNet and unifies fragmented tasks.
- The divided spatiotemporal attention concept from TimeSformer is successfully transferred to the sports video domain.
- The temporal alignment model of MatchTime plays a key role in dataset construction.
- Insight: Similar "automated annotation + specialized encoder" paradigms can be generalized to other vertical domains.

## Rating

- **Novelty**: 7/10 — The technical components (spatiotemporal attention, contrastive learning) are not completely novel, but the combination of dataset construction and a unified framework is valuable.
- **Experimental Thoroughness**: 9/10 — Extremely thorough experiments spanning three tasks, multiple datasets, and detailed ablation studies.
- **Writing Quality**: 8/10 — Clearly structured with dataset and method described separately, maintaining a smooth logical flow.
- **Value**: 8/10 — The large-scale open-source dataset + unified framework significantly drives progress for the sports AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SoccerMaster: A Vision Foundation Model for Soccer Understanding](../../CVPR2026/video_understanding/soccermaster_a_vision_foundation_model_for_soccer_understanding.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)

</div>

<!-- RELATED:END -->
