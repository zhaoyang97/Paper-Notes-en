---
title: >-
  [Paper Note] OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding
description: >-
  [AAAI 2026][Multimodal VLM][Pedestrian tracking] This paper proposes OmniPT, a unified pedestrian tracking framework built upon large vision-language models (LVLMs). Through a four-stage RL→Mid Training→SFT→RL training strategy, OmniPT simultaneously supports conventional MOT, language-referred tracking (RMOT/CRMOT), and semantic understanding (SMOT), achieving state-of-the-art results on multiple benchmarks—most notably a HOTA of 75.04 on BenSMOT, surpassing the previous SOTA by 3.06.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Pedestrian tracking
  - large vision-language models
  - multi-object tracking
  - semantic understanding
  - reinforcement learning
date: 2026-05-08
content_hash: e2a4cce56f69b9dd
---

# OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding

**Conference**: AAAI 2026
**arXiv**: [2511.17053](https://arxiv.org/abs/2511.17053)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: Pedestrian tracking, large vision-language models, multi-object tracking, semantic understanding, reinforcement learning

## TL;DR
This paper proposes OmniPT, a unified pedestrian tracking framework built upon large vision-language models (LVLMs). Through a four-stage RL→Mid Training→SFT→RL training strategy, OmniPT simultaneously supports conventional MOT, language-referred tracking (RMOT/CRMOT), and semantic understanding (SMOT), achieving state-of-the-art results on multiple benchmarks—most notably a HOTA of 75.04 on BenSMOT, surpassing the previous SOTA by 3.06.

## Background & Motivation

Pedestrian tracking is a classical computer vision task with broad applications in autonomous driving, intelligent surveillance, and motion analysis. The field faces two core challenges:

**Challenge 1: Tracking stability in complex scenes.** Maintaining stable tracking when pedestrians are frequently occluded, blurred, or temporarily absent remains difficult. Traditional trackers rely primarily on appearance features and motion prediction, yet perform poorly under high inter-instance similarity (e.g., dancers in DanceTrack) or prolonged occlusion.

**Challenge 2: Emergence of multimodal tracking tasks.** Recent years have seen a proliferation of new multimodal MOT tasks: RMOT (Referring MOT), which tracks specific targets given natural language descriptions; CRMOT (Cross-view RMOT), which extends RMOT to multiple camera views; and SMOT (Semantic MOT), which requires semantic understanding of tracked targets. These tasks demand that models comprehend high-level semantics while tracking.

**Key Insight**: Humans can effortlessly re-identify targets after prolonged disappearance because they abstract targets into semantic descriptions for "subconscious retrieval." While LVLMs excel at image-level understanding tasks (e.g., VQA, captioning), a gap remains between LVLMs and specialist models on instance-level tasks (e.g., object detection, visual grounding).

**Key Insight**: The paper introduces the strong semantic understanding capabilities of LVLMs into pedestrian tracking, unifying MOT, RMOT, CRMOT, and SMOT under a single "one-for-all" framework. The core idea is to decompose tracking into natural language sub-tasks executable by an LVLM, with a four-stage training strategy enforcing structured output.

## Method

### Overall Architecture

OmniPT is built upon the pretrained LVLM Qwen2.5-VL and adopts a four-stage **RL → Mid Training → SFT → RL** training strategy. The overall pipeline casts tracking as a VQA problem: given a sequence of video frames, tracking results and semantic understanding are obtained through conversational interaction. At inference time, iterative multi-turn dialogue enables long-term tracking.

### Key Designs

1. **Stage 1 RL: Output Format Standardization**

    - **Function**: Lightweight reinforcement learning using GRPO to regularize the model's bounding box output format.
    - **Mechanism**: The model is required to produce standardized coordinates strictly adhering to the `<bbox>x,y,w,h</bbox>` format. A hierarchical reward function is designed:
        - Fully correct format: $R = 2$
        - Alternative format present (e.g., x,y,x,y): $R = 0.6$
        - No bbox tag present: $R = 0.4$
    - Final reward: $R_{s1} = R(bbox_p) \times \left(\frac{IOU(bbox_p, bbox_{gt})}{2} + 0.5\right)$
    - **Design Motivation**: Native LVLM output is uncontrolled; format unification is a prerequisite for subsequent supervised training. Mapping IoU to the range [0.5, 1] in this stage directs the model to prioritize format over localization precision.

2. **Stage 2 Mid Training: Pedestrian Perception Enhancement**

    - **Function**: Large-scale pedestrian-related data and proxy tasks are used to enhance the model's pedestrian perception capability.
    - **Core Modules**:
        - **CLIP Alignment Training**: The visual encoder is trained on the SYNTH-PEDES dataset; a `<CLS>` token is inserted to compute a similarity matrix supervised with cross-entropy loss.
        - **Object Detection Proxy Task**: Images are randomly sampled and GT bounding boxes serve as supervision.
        - **Position Prediction Proxy Task**: Given the target coordinates in the first frame, the model predicts positions in subsequent frames. Special samples containing target disappearance are included to improve robustness. DanceTrack is primarily used due to its unpredictable motion patterns.
        - **Person Re-identification Proxy Task**: Same-identity and negative samples are drawn from different frames, guiding the model to recognize the same individual.
    - **Design Motivation**: Mid Training bridges generic pretraining and task-specific post-training. The proxy tasks jointly and effectively enhance three core tracking capabilities: object detection, position prediction, and ReID.

3. **Stage 3 SFT: Task-Specific Fine-Tuning**

    - **Function**: Supervised fine-tuning across multiple pedestrian tracking datasets.
    - **Training samples are divided into four categories**:
        - **MOT**: Consecutive frame sequences with three key queries (target position in the first frame / sequence-level tracking / newly appearing targets mid-sequence).
        - **RMOT**: Target position in the first frame is replaced by a language description; the model tracks all targets matching the description.
        - **CRMOT**: Extended to multiple views; the original sequence is evenly distributed across views and the target view is specified in natural language.
        - **Video/Instance Description**: Semantic understanding is performed after tracking, given the full image and target position.
    - **Design Motivation**: Decoupling tracking into VQA format enables the LVLM to naturally execute different types of tracking tasks.

4. **Stage 4 RL: Performance Enhancement**

    - **Function**: Additional RL training further improves tracking performance and instruction-following capability.
    - **Difference from Stage 1**: The IoU-to-[0.5, 1] mapping is removed, directing the model to focus on localization precision.

### Inference Strategy

At inference time, iterative multi-turn dialogue enables long-term tracking: the tracking result of the last frame in one dialogue turn serves as the initial prior for the next turn, achieving continuous tracking across dialogue sessions.

## Key Experimental Results

### Main Results

| Dataset | Metric | OmniPT | Prev. SOTA | Gain |
|---------|--------|--------|------------|------|
| BenSMOT (Tracking) | HOTA↑ | **75.04** | 71.98 (SMOTer) | +3.06 |
| BenSMOT (Video Caption) | CIDEr↑ | **1.826** | 0.343 (SMOTer) | +432% |
| BenSMOT (Instance Caption) | CIDEr↑ | **0.482** | 0.087 (SMOTer) | +454% |
| DanceTrack | HOTA↑ | **56.4** | 55.1 (OC-SORT) | +1.3 |
| Refer-KITTI-V2 | HOTA↑ | **36.15** | 35.04 (TempRMOT) | +1.11 |
| CRTrack (In-domain) | CVRIDF1↑ | **62.13** | 54.88 (CRTracker) | +7.25 |
| CRTrack (Cross-domain) | CVRIDF1↑ | **46.54** | 12.52 (CRTracker) | +34.02 |

### Ablation Study

| Configuration | MOT (HOTA) | RMOT (HOTA) | SMOT (CIDEr) | Notes |
|---------------|-----------|-------------|--------------|-------|
| SFT only | 47.38 | 30.37 | 0.40 | Baseline |
| MT + SFT | 51.89 | 33.26 | 0.44 | Mid Training substantially improves tracking |
| SFT + RL | 48.63 | 35.46 | 0.41 | RL gains are limited without sufficient training |
| MT + SFT + RL | **56.40** | **36.15** | **0.48** | Full pipeline achieves best performance |

**Comparison of LVLMs at the 7B scale**: Qwen2.5-VL > LLaVA-NeXT > InternVL2.5. The authors attribute this to Qwen2.5-VL's dynamic resolution design, which enables clearer capture of target semantic features.

**Effect of image count**: Increasing the number of images per training sample from 2 to 32 consistently improves the description task (CIDEr: 0.773→1.826), while the tracking task peaks at 8 frames (HOTA: 73.04) and then degrades, as an excessive number of images challenges the model's ability to process multiple images simultaneously.

### Key Findings

- The most dramatic gain occurs on cross-domain CRMOT (+34.02 CVRIDF1), demonstrating the unified framework's clear advantage in domain transfer.
- Semantic understanding (CIDEr) improves by more than 5×, fully exploiting the LVLM's natural strength in captioning tasks.
- The Mid Training stage contributes most to tracking performance, validating the effectiveness of the proxy task design.

## Highlights & Insights

1. **A new paradigm for unified frameworks**: OmniPT is the first to unify MOT, RMOT, CRMOT, and SMOT under a single LVLM, embodying the "one-for-all" research trend.
2. **Elegant four-stage training design**: Each stage has a well-defined objective—RL for format standardization, Mid Training for pedestrian perception, SFT for task alignment, and RL for performance enhancement.
3. **Insightful Mid Training proxy task design**: The three core tracking capabilities—detection, position prediction, and ReID—are decomposed into independent proxy tasks; notably, the position prediction task incorporates samples with target disappearance.
4. **VQA-based task decoupling**: Casting the tracking task as multi-turn dialogue is both conceptually simple and empirically effective.

## Limitations & Future Work

1. **Crowded scene limitations**: In dense scenes such as MOT20 (200+ simultaneous pedestrians), the LVLM struggles to accurately localize and track all targets, constrained by maximum output length.
2. **Category scope**: The current work focuses solely on the "pedestrian" category and has not been extended to open-vocabulary object tracking.
3. **High computational cost**: Training requires 24 A100 GPUs, and inference involves multi-turn dialogue, which may become a throughput bottleneck.
4. **Inference speed not discussed**: The paper does not report inference latency, which is critical for real-world deployment scenarios.

## Related Work & Insights

- The capability boundary of LVLMs on instance-level tasks is continually being pushed, from visual grounding to object tracking.
- The concept of Mid Training originates from the LLM domain (e.g., OctoThinker, OLMo) and proves equally effective for visual tasks.
- Using GRPO for format standardization is a practical technique that addresses the general problem of uncontrolled LVLM output.
- **Inspiration**: A similar approach may be applicable to unify other vision tasks under a single LVLM (e.g., segmentation + captioning + tracking).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](../../ACL2026/multimodal_vlm/unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[CVPR 2026\] Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models](../../CVPR2026/multimodal_vlm/devil_is_in_narrow_policy_unleashing_exploration_in_driving_vla_models.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](../../CVPR2026/multimodal_vlm/understanding_task_transfer_in_vision-language_models.md)
- [\[AAAI 2026\] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](bridging_the_copyright_gap_do_large_vision-language_models_r.md)
- [\[AAAI 2026\] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](global_compression_commander_plug-and-play_inference_acceler.md)

</div>

<!-- RELATED:END -->
