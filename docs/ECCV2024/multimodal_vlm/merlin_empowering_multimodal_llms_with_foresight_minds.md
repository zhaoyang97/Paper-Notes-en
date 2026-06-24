---
title: >-
  [Paper Note] Merlin: Empowering Multimodal LLMs with Foresight Minds
description: >-
  [ECCV 2024][Multimodal VLM][Multimodal Large Language Models] Proposes a two-stage training paradigm consisting of Foresight Pre-Training (FPT) and Foresight Instruction-Tuning (FIT). By incorporating trajectory modeling, it empowers Multimodal Large Language Models (MLLMs) with "foresight thinking" capabilities, enabling them to predict future events and reason based on current observations.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Future Reasoning"
  - "Trajectory Prediction"
  - "Foresight Thinking"
  - "Visual Understanding"
date: 2026-05-08
content_hash: 000cf5f5d65f3c84
---

# Merlin: Empowering Multimodal LLMs with Foresight Minds

**Conference**: ECCV 2024  
**arXiv**: [2312.00589](https://arxiv.org/abs/2312.00589)  
**Code**: [GitHub](https://ahnsun.github.io/merlin)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Future Reasoning, Trajectory Prediction, Foresight Thinking, Visual Understanding

## TL;DR

Proposes a two-stage training paradigm consisting of Foresight Pre-Training (FPT) and Foresight Instruction-Tuning (FIT). By incorporating trajectory modeling, it empowers Multimodal Large Language Models (MLLMs) with "foresight thinking" capabilities, enabling them to predict future events and reason based on current observations.

## Background & Motivation

Humans can predict future events based on current observations, a phenomenon referred to in neuroscience as "predictive processing." However, although existing Multimodal Large Language Models (MLLMs) such as GPT-4V and Bard excel in image understanding and logical reasoning, they lack the ability to predict future events from current image observations. Even when provided with multi-frame image sequences, existing MLLMs still struggle to analyze and infer the specific behaviors of targets (such as predicting object movement or interaction).

The authors decompose the human process of anticipating the future into two stages:

**Observing Dynamic Cues**: Observing the motion and state changes of the targets.

**Analyzing Behavioral Patterns and Reasoning**: Analyzing behavioral patterns based on observations to infer potential events.

The LLM component of existing MLLMs already possesses robust logical reasoning capabilities (the second stage). The key challenge lies in the first stage—how to enable the MLLM to correctly acquire spatiotemporal dynamic information from multi-image observations. Directly modeling the next frame (e.g., reconstructing the next image frame) suffers from visual information redundancy, making it difficult to directly extract dynamic cues. To address this, the authors propose using **trajectories** as the learning objective. As highly structured representations, trajectories can connect past and future temporal contexts.

## Method

### Overall Architecture

Merlin consists of three core components:

1. **Image Encoder**: Uses a pre-trained CLIP ViT-L/14 with an input image resolution of 448×448, generating 1024 encoded tokens.
2. **Large Language Model Decoder**: Uses the open-source Vicuna-7B v1.5.
3. **Modality Alignment Projector**: Uses a 3×3 2D convolutional layer (stride=2, padding=1) to achieve dimension projection and token aggregation.

Reasons for choosing 2D convolution over 1D linear layers or cross-attention layers as the connector:
- 2D convolution can aggregate local visual tokens across spatial scales, efficiently converting spatial information to channel information.
- Compared to cross-attention, 2D convolution exhibits better convergence properties, laying a solid foundation for two-stage training.
- It effectively compresses the number of tokens, supporting high-resolution and multi-frame inputs.

### Key Designs

#### Foresight Pre-Training (FPT)

The core idea of FPT is causal modeling of raw temporal trajectories interleaved with multi-frame images, enabling the MLLM to perceive dynamic cues across frames.

**Modeling formulation**: Given multi-frame images in a video clip, the model takes the observation (description or location) of the target in the first frame as the query, and needs to predict the complete trajectory of the target throughout the video:

$$P(Y|X) \sim P(Y|\{X_1, X_2, ...\}, O_{first})$$

where $X_i$ is the $i$-th frame, $O_{first}$ is the first-frame observation, and $Y$ is the trajectory of the target.

**Data construction** follows three principles:

1. **Precisely defined task prompts and answer formats**: Task prompts are used to inform the MLLM of the specific task (detection or tracking), and the answer format is specified in the question, allowing different types of tasks to be flexibly organized without compromising general language capabilities.
2. **Clear indicators for multimodal information**: Special frame indicators (e.g., `frame1:<image>`, `frame2:<image>`) are added for each set of image tokens to help the MLLM better attend to the corresponding images.
3. **Interleaved organization of frames and observations**: For the same target identity, its appearing frames and location observations are arranged interleavingly, wrapped with ID tokens (`<Idi>` and `</Idi>`) to construct the trajectory.

**Observation types** are categorized into three: location description, appearance description, and action description. One is randomly selected as the query.

**Training strategy**: Unlike previous separated practices that first perform modality alignment and then multi-task pre-training, Merlin merges them into a single stage and **unfreezes all modules** for pre-training. It performs multi-task learning using a hybrid dataset of 10M image-text pairs and approximately 5M QA data from various sources.

#### Foresight Instruction Tuning (FIT)

FPT equips the model with the ability to observe multi-frame dynamic cues, but this alone is insufficient to achieve true "foresight thinking." On top of FPT, FIT introduces **Trajectory Chain-of-Thought (T-CoT)**, utilizing trajectory modeling as a bridge in the logical reasoning chain.

**Core formulation**:

$$P(Z|X,Y) \sim P(Z|\{X_1, X_2, ...\}, O_{first}, Y)$$

where $Z$ is the future observation (which can be actions, events, trends, or possibilities), conditioned on the multi-frame images, the first-frame observation, and the trajectory altogether, enabling the MLLM to causally predict the future.

**How T-CoT works**: When a user queries the future of a target, Merlin first outputs the observed trajectory of this target, then outputs the trajectories of other relevant targets, and finally reasons about potential future events based on these trajectories. For example, in a soccer scene, Merlin first outputs the trajectory of a player in red, then outputs the trajectory of a player in white, and infers that the player in white might slide tackle, causing both players to fall.

**FIT Data construction**: Uses GPT-4 to generate approximately 30K T-CoT dialogue pairs based on the trajectory and action information from three scene datasets: MultiSports, TITAN, and STAR.

### Loss & Training

**Two-stage training configuration**:

| Configuration | Pre-training (FPT) | Instruction Tuning (FIT) |
|--------|-------------|---------------|
| Vision Encoder | Unfrozen | Frozen |
| Projector | Unfrozen | Unfrozen |
| LLM | Unfrozen | Unfrozen |
| Learning Rate | 5e-5 | 5e-5 |
| Global Batch Size | 2048 | 256 |
| Training Steps | 7k | 3k |
| Optimizer | AdamW ($\beta_2=0.95$) | AdamW ($\beta_2=0.95$) |
| LR Scheduler | cosine decay | cosine decay |
| Precision | bfloat16 | bfloat16 |

**Data composition**:
- FPT phase: 10M image-text pairs (LAION) + 5M multi-task QA data
- FIT phase: 665K LLaVA instruction data + 30K T-CoT dialogues + 40K FPT sampled data

Training is conducted on 64 NVIDIA A800 GPUs, taking approximately 12 hours for pre-training and 3 hours for instruction tuning.

## Key Experimental Results

### Main Results

**Table 1: Future Reasoning Capabilities (MMBench Sub-tasks)**

| Method | LLM | Dev Avg | OL | PPR | FR | IR | FP | Test Avg |
|------|-----|---------|---|----|----|----|----|----|
| mPLUG-Owl | 7B | 41.0 | 18.5 | 18.7 | 66.7 | 86.7 | 14.3 | 45.9 |
| Shikra | 7B | 51.5 | 32.1 | 30.7 | 63.0 | 88.9 | 42.9 | 60.0 |
| Kosmos-2 | 1.6B | 54.4 | 38.3 | 33.3 | 56.8 | 91.1 | 52.4 | 58.2 |
| LLaVA-1.5 | 7B | 59.6 | 43.2 | 52.0 | 71.6 | 93.3 | 38.1 | - |
| **Merlin** | **7B** | **64.4** | **42.0** | **54.7** | **72.8** | **97.8** | **54.8** | **66.5** |

Merlin achieves the best results in 8 out of 10 metrics, leading significantly in overall score.

**Table 2: Object tracking evaluation**

| Method | LaSOT Success | GOT10k AO | GOT10k SR₀.₅ | GOT10k SR₀.₇₅ |
|------|--------------|-----------|-------------|--------------|
| SiamFC | 33.6 | 34.8 | 35.3 | 9.8 |
| SiamRPN++ | 49.6 | 51.8 | 61.8 | 32.5 |
| LLaVA-1.5 (Tracking Fine-tuned) | 19.4 | 23.5 | 20.2 | 9.7 |
| **Merlin** | **39.8** | **51.4** | **55.9** | **42.8** |

Merlin is the first MLLM capable of performing tracking tasks, and achieves performance comparable to expert models using only a small amount of tracking data.

**Table 3: Hallucination evaluation (POPE)**

| Method | Random Acc | Popular Acc | Adversarial Acc |
|------|-----------|------------|----------------|
| LLaVA (7B) | 72.16 | 61.37 | 58.67 |
| LLaVA-1.5 (7B) | 83.29 | 81.88 | 78.96 |
| Qwen-VL (7B) | 84.73 | 84.13 | 82.26 |
| **Merlin (7B)** | **91.58** | **89.53** | **84.10** |

Merlin significantly outperforms existing methods across all settings, with the "yes" ratio close to 50%, demonstrating excellent visual perception capabilities.

### Ablation Study

**Table 4: Ablation on FPT and FIT strategies**

| Pre-training (ITP+FPT) | Fine-tuning (ITD+FIT) | GOT10K AO | Inference Avg |
|:---:|:---:|:---:|:---:|
| ITP only | ITD only | - | 59.5 |
| ITP only | ITD+FIT | - | 60.7 |
| FPT only | ITD+FIT | 15.5 | 52.8 |
| ITP+FPT | ITD only | 51.4 | 61.2 |
| ITP+FPT | ITD+FIT | **51.4** | **64.4** |

**Table 5: Ablation on model configurations**

| Resolution | Projector | Encoder | Token Count | Inference | GOT10K |
|--------|--------|--------|---------|------|--------|
| 448x | Conv2d | Unfrozen | 256 | **64.4** | **51.4** |
| 336x | Conv2d | Unfrozen | 256 | 59.8 | 47.3 |
| 336x | MLP | Unfrozen | 576 | 58.1 | 23.5 |
| 448x | Conv2d | Frozen | 256 | 60.8 | 28.4 |

### Key Findings

1. **Complementary nature of FPT and FIT**: FPT provides dynamic cue perception, while FIT activates foresight reasoning capabilities through T-CoT; both are indispensable.
2. **Image-text pair data is essential**: The absence of image-text pairs in pre-training severely harms the model's general capability.
3. **High resolution facilitates precise localization**: 448x resolution yields a significant performance boost over 336x in localization and tracking tasks.
4. **Conv2d projector outperforms MLP**: It effectively compresses the number of tokens, supporting multi-image input without performance degradation.
5. **Foresight learning unexpectedly reduces hallucinations**: By learning trajectory correspondences, the model gains more precise target attention capabilities, preventing misidentification.
6. **Precise task descriptions are critical**: The absence of precise task descriptions causes tracking performance to plunge from 51.4% to 28.4%.

## Highlights & Insights

1. **Insight of trajectories as a structured learning target**: Compared to directly predicting the next image frame, trajectories provide highly abstract and structured spatiotemporal representations, effectively avoiding visual redundancy.
2. **Innovation of Trajectory Chain-of-Thought (T-CoT)**: Extends the CoT concept to the domain of visual trajectories, utilizing trajectories as a "bridge" in the reasoning chain that connects observations with future predictions.
3. **Unified multi-task conversation format design**: Through precise task definitions and format specifications, multiple tasks such as detection, tracking, referring expression comprehension, and future reasoning are handled within a single model.
4. **Spillover effect of foresight learning**: Training the model to predict trajectories not only improves future reasoning capabilities, but also unexpectedly enhances general visual understanding and reduces hallucinations, providing a new perspective for MLLM training.

## Limitations & Future Work

1. **Constraints in handling long videos**: Current support is limited to $\le 8$ frames due to the reliance on image encoders rather than video encoders, making it unable to handle long-term video sequences.
2. **Incomplete evaluation benchmarks**: Existing future reasoning evaluation benchmarks are not comprehensive; the benchmark constructed based on MMBench sub-tasks is only a preliminary attempt.
3. **Video encoding efficiency**: More efficient long-term video tokenizers need to be developed.
4. **Limitations of trajectory representation**: Current trajectories only use bounding boxes, without getting into finer-grained spatial information (such as poses or fine-grained actions).
5. **Dialogue data scale**: The T-CoT dataset is limited to only 30K; expanding the scale could further improve the reasoning quality.

## Related Work & Insights

- **Relationship with LLaVA-1.5**: Merlin is based on the LLaVA architecture, replacing the MLP projector with Conv2d and introducing multi-frame and trajectory modeling.
- **Relationship with Shikra**: Shikra pioneered introducing spatial coordinate dialogue capabilities in MLLMs; Merlin extends this to the temporal dimension.
- **Takeaways**:
  1. Trajectories can serve as a cross-modal "language" connecting visual and textual modalities.
  2. Foresight learning can be a general visual representation enhancement strategy.
  3. Multi-task pre-training and instruction tuning can be combined into a single stage.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4.5 | Systematically introduces "foresight thinking" to MLLMs for the first time, with Trajectory CoT being an innovative contribution |
| Technical Depth | 4.0 | The two-stage training paradigm is well-designed, and the data construction details are comprehensive |
| Experimental Thoroughness | 4.0 | Multi-task evaluation is comprehensive, though the future reasoning benchmark remains relatively simple |
| Writing Quality | 4.0 | The writing is fluent with rich illustrations and clear methodological explanations |
| **Overall** | **4.0** | A pioneering work in the direction of MLLM future reasoning, featuring an innovative idea and thorough verification |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals](../../CVPR2026/multimodal_vlm/merlin_building_low-snr_robust_multimodal_llms_for_electromagnetic_signals.md)
- [\[ECCV 2024\] Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](eyes_closed_safety_on_protecting_multimodal_llms_via_image-to-text_transformatio.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)
- [\[ECCV 2024\] Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)

</div>

<!-- RELATED:END -->
