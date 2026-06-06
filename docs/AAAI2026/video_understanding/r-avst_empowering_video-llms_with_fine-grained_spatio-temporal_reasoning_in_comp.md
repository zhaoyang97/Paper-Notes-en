---
title: >-
  [Paper Note] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios
description: >-
  [AAAI 2026][Video Understanding][Audio-visual reasoning] This paper introduces R-AVST, the first fine-grained spatio-temporal reasoning dataset for complex audio-visual scenarios (5K+ untrimmed videos, 27K objects…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Audio-visual reasoning"
  - "spatio-temporal grounding"
  - "reinforcement learning"
  - "GRPO"
  - "Video-LLM"
date: 2026-05-08
content_hash: 47239a7cb9e49ffa
---

# R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios

**Conference**: AAAI 2026
**arXiv**: [2511.16901](https://arxiv.org/abs/2511.16901)  
**Code**: N/A  
**Area**: Video Understanding
**Keywords**: Audio-visual reasoning, spatio-temporal grounding, reinforcement learning, GRPO, Video-LLM

## TL;DR

This paper introduces R-AVST, the first fine-grained spatio-temporal reasoning dataset for complex audio-visual scenarios (5K+ untrimmed videos, 27K objects, 100 audio-visual event categories), defines three core reasoning tasks, and trains the AVST-Zero model via GRPO with a multi-dimensional reward function to directly optimize audio-visual spatio-temporal reasoning.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have advanced rapidly in video understanding, with models such as InternVL-2.5, Qwen2.5-VL, and VideoLLaMA3 demonstrating strong capabilities. However, existing research focuses predominantly on **simple video scenarios**, failing to reflect the complexity and diversity of real-world audio-visual events.

**Limitations of Prior Work**:

**Dataset level**: Existing audio-visual datasets (e.g., AVE, UnAV-100, PU-VALOR) focus mainly on **temporal** understanding and neglect the **spatial attributes** of sounding objects. Spatio-temporal grounding datasets (e.g., VidSTG, HC-STVG, V-STaR) provide spatio-temporal annotations but do not adequately capture real audio-visual dynamics and cover a limited range of object categories.

**Model level**: Models such as LLaVA-ST and GroundingGPT have progressively extended spatio-temporal modeling but rely on large-scale high-quality annotated data and lack sufficient exploratory capacity. RL-based models including VideoChat-R1 and Video-R1 have begun leveraging GRPO to enhance reasoning, yet their reward designs offer limited support for audio-visual spatio-temporal reasoning and lack task formulations targeting complex audio-visual scenarios.

**Key Challenge**: The absence of a dataset that simultaneously provides **fine-grained spatio-temporal annotations** and **rich audio-visual event coverage** hampers the development of video understanding models capable of spatio-temporal reasoning in realistic, complex scenarios.

**Key Insight**: The paper addresses this gap from an audio-visual perspective by constructing the first audio-visual reasoning dataset with fine-grained spatio-temporal annotations, and adopts a rule-based reward GRPO training paradigm to directly optimize behavioral policy without intermediate supervision signals.

## Method

### Overall Architecture

The R-AVST project comprises two components: (1) construction of the R-AVST dataset; and (2) training of the AVST-Zero model. Dataset construction follows a five-step pipeline: collection and filtering → caption analysis → bounding box annotation → QA generation → quality control. The model is trained on R-AVST using GRPO with a multi-dimensional reward function to guide policy updates.

### Key Designs

#### 1. **R-AVST Dataset Construction**

**Data Collection and Filtering**: Untrimmed YouTube videos are collected from UnAV-100 and filtered in three steps: grouping by duration (short/medium/long), limiting each video to at most three audio-visual events (to balance event distribution), and removing videos where the event proportion is below 0.08. This yields 5,237 high-quality videos.

**Caption Analysis**: GPT-4o-mini is used as an analyzer LLM to extract noun objects from event captions and annotate their **auditory and visual attributes** ("visible and audible" or "visible only"). Carefully designed prompts emphasize the definition of "audibility" in the object–sound relationship to improve analysis accuracy. In total, 27,253 objects are annotated, of which 50.88% are classified as "visible and audible."

**Spatial Annotation**: Grounded-SAM2 is employed for automated frame-level object annotation to reduce the cost of large-scale video labeling, with BOX_THRESHOLD=0.4 and TEXT_THRESHOLD=0.3.

**Automated QA Generation**: Three question types are defined corresponding to three tasks:
- Temporal reasoning: "When is the moment [objects] make sound and are visible?"
- Spatial reasoning: "What objects make sound between [start] and [end], and where are they?"
- Spatio-temporal reasoning: "When is the moment [objects] make sound and are visible, and where are they?"

The training set contains 2,663 temporal + 2,666 spatial + 1,204 spatio-temporal QA pairs; the test set contains 663 + 664 + 306 = 1,633 QA pairs.

#### 2. **Three Core Reasoning Tasks**

- **Audio-Visual Temporal Reasoning (AVTR)**: Given visible and audible objects, infer the temporal interval during which each object is both sounding and visible.
- **Audio-Visual Spatial Reasoning (AVSR)**: Given a temporal interval, localize the spatial positions of sounding and silent objects in the scene.
- **Audio-Visual Spatio-Temporal Reasoning (AVSTR)**: Simultaneously infer both the temporal interval and the spatial position of objects, most closely approximating human perception.

#### 3. **AVST-Zero Model and Multi-Dimensional Rewards**

**Training Paradigm**: The model is trained on Qwen2.5-VL-7B (or Qwen2.5-Omni-7B) using **GRPO (Group Relative Policy Optimization)** with full RL fine-tuning, eliminating the need for an SFT warm-up stage.

**Four-Dimensional Reward Design**:

- **Format Reward** $R_{\text{format}}$: Checks whether the response contains the correct tag pairs (`<answer>`, `<object>`, `<when>`, `<where>`).
- **Object Reward** $R_{\text{object}}$: Computes the semantic similarity between predicted and ground-truth object names using Word2Vec; a reward of 1 is assigned if the similarity exceeds threshold $\tau$.
$$R_{\text{object}} = \begin{cases} 1, & \text{if } \text{sim}(V_{\text{pred}}, V_{\text{gt}}) \geq \tau \\ 0, & \text{otherwise} \end{cases}$$
- **Temporal Reward** $R_{\text{temporal}}$: IoU between the predicted temporal interval and the ground-truth interval.
$$R_{\text{temporal}} = \frac{|I_{\text{pred}} \cap I_{\text{gt}}|}{|I_{\text{pred}} \cup I_{\text{gt}}|}$$
- **Spatial Reward** $R_{\text{spatial}}$: Mean 2D IoU between predicted and ground-truth bounding boxes over the overlapping temporal interval.
$$R_{\text{spatial}} = \frac{1}{N} \sum_{t=T_{\text{start}}}^{T_{\text{end}}} \text{IoU}(t)$$

**Final Reward**: $R = \lambda_f R_{\text{format}} + \lambda_t R_{\text{temporal}} + \lambda_o R_{\text{object}} + \lambda_s R_{\text{spatial}}$, where $\lambda_f = 1$ and the remaining coefficients are set according to task type.

### Loss & Training

The standard GRPO objective is used: G=6 output samples are drawn per question, within-group relative advantages $A_i$ are computed, and the policy is updated via a clipped surrogate objective with KL regularization. Training is conducted on 4 NVIDIA RTX A6000 GPUs for a single epoch, with batch size = 1 per GPU, max_prompt_length = 512, and max_completion_length = 1024.

## Key Experimental Results

### Main Results

**Temporal Reasoning Task (AVTR)**:

| Method | m_tIoU | R1@0.3 | R1@0.5 | R1@0.7 |
|------|--------|--------|--------|--------|
| Qwen2.5-VL (7B) | 36.05 | 46.40 | 34.38 | 16.22 |
| Video-LLaMA3 (7B) | 37.17 | 50.30 | 35.29 | 22.67 |
| VideoChat-R1 (7B) | 43.17 | 60.81 | 46.70 | 25.68 |
| **AVST-Zero (7B)** | **47.96** | **71.13** | **51.43** | 23.91 |

**Spatial Reasoning Task (AVSR)**:

| Method | Obj Acc | m_vIoU | AP@0.3 | AP@0.5 |
|------|---------|--------|--------|--------|
| Qwen2.5-VL (7B) | 1.91 | 2.31 | 0.90 | 0.15 |
| VideoChat-R1 (7B) | 15.54 | 1.99 | 3.11 | 0.36 |
| **AVST-Zero (7B)** | 14.34 | 2.27 | **3.12** | **0.87** |
| **AVST-Zero-Omni (7B)** | **19.48** | **3.87** | **4.47** | **2.17** |

**Spatio-Temporal Reasoning Task (AVSTR)**:

| Method | m_tIoU | m_vIoU | AP@0.3 | AP@0.5 |
|------|--------|--------|--------|--------|
| VideoChat-R1 (7B) | 41.81 | 2.15 | 3.21 | 0.60 |
| **AVST-Zero (7B)** | **46.04** | 8.59 | 10.38 | 3.83 |
| **AVST-Zero-Omni (7B)** | 35.97 | **17.74** | **22.90** | **12.26** |

### Ablation Study

| Configuration | AVTR m_tIoU | AVSR Obj Acc | AVSR m_vIoU | AVSTR m_tIoU | AVSTR m_vIoU |
|------|------------|-------------|-------------|-------------|-------------|
| SFT | 42.84 | 9.52 | 3.42 | 38.40 | 4.26 |
| AVST-Zero | **48.17** | 20.72 | **4.62** | **46.93** | **10.87** |
| w/o temporal reward | 46.67 | **23.95** | 4.54 | 45.82 | 8.31 |
| w/o spatial reward | 47.03 | 23.17 | 3.28 | 44.29 | 9.23 |

### Key Findings

1. **RL outperforms SFT**: Direct GRPO training yields more substantial improvements than SFT across all three tasks, indicating that RL is better suited for fine-grained spatio-temporal reasoning.
2. **AVST-Zero-Omni outperforms AVST-Zero on spatial metrics** (benefiting from the base model's joint audio-visual perception) but underperforms on temporal metrics.
3. **Cross-dimensional reward interaction**: Removing the temporal reward degrades spatial metrics and vice versa, demonstrating the mutual dependency between temporal and spatial reasoning.
4. **Cross-dataset generalization**: The model achieves competitive performance on AVE and AVSBench-V1, attaining the best spatial m_vIoU of 6.84%.

## Highlights & Insights

1. **First audio-visual spatio-temporal reasoning dataset**: R-AVST fills the gap in existing datasets by providing fine-grained spatio-temporal annotations for audio-visual scenarios, covering 100 event categories with an average of 5.2 objects per video.
2. **Full RL training without SFT**: Exploiting the rule-based nature of the tasks, multi-dimensional rewards enable direct GRPO training, eliminating dependence on large-scale high-quality annotated data.
3. **Novel paradigm for object attribute analysis**: Using an LLM to analyze the "audible/visible" attributes of objects mentioned in captions introduces a new annotation paradigm for audio-visual scene understanding.
4. **Automated annotation pipeline**: The combination of LLM-based analysis, Grounded-SAM2 automatic annotation, and programmatic QA generation substantially reduces annotation costs.

## Limitations & Future Work

1. Absolute spatial reasoning performance remains low (AP@0.5 peaks at only 2.17% on spatial tasks), indicating that precise spatial grounding in audio-visual scenes remains highly challenging.
2. The dataset is constructed from UnAV-100, making the video sources relatively homogeneous; extension to broader domains (e.g., autonomous driving, industrial inspection) is desirable.
3. Only 7B-scale models are explored; larger models may yield significant performance gains.
4. Direct encoding of audio features is not explored (the Omni variant leverages audio only indirectly via the multimodal base model); more explicit audio modeling warrants investigation.

## Related Work & Insights

- The rule-based RL training paradigm from DeepSeek-R1/GRPO is successfully transferred to video spatio-temporal reasoning tasks, suggesting that **decomposing task objectives into computable reward dimensions** is key to RL success in visual tasks.
- The automatic annotation capability of Grounded-SAM2 provides a viable pathway for large-scale video dataset construction.
- Compared with VideoChat-R1, the gains of AVST-Zero on temporal and spatio-temporal reasoning demonstrate that **task-specific multi-dimensional reward design** is more effective than generic reward formulations.

## Rating

- Novelty: ⭐⭐⭐⭐ (First audio-visual spatio-temporal reasoning dataset combined with full RL training; strong originality)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-task comparisons, ablations, cross-dataset validation, and qualitative analysis are all comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with a detailed account of the dataset construction pipeline)
- Value: ⭐⭐⭐⭐ (Opens a new direction for audio-visual spatio-temporal reasoning; the dataset has long-term research value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](../../ACL2026/video_understanding/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](../../NeurIPS2025/video_understanding/tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)
- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](../../CVPR2026/video_understanding/frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)

</div>

<!-- RELATED:END -->
