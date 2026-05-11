---
title: >-
  [Paper Note] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding
description: >-
  [CVPR 2026][Multimodal VLM][Long video understanding] This paper proposes MSJoE, a framework that jointly evolves an MLLM and a lightweight keyframe sampler via reinforcement learning. The MLLM generates visual queries t…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Long video understanding"
  - "keyframe sampling"
  - "reinforcement learning"
  - "GRPO"
  - "joint optimization"
date: 2026-05-08
content_hash: 5386340d4a783f63
---

# MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2602.22932](https://arxiv.org/abs/2602.22932)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Long video understanding, keyframe sampling, reinforcement learning, GRPO, joint optimization

## TL;DR
This paper proposes MSJoE, a framework that jointly evolves an MLLM and a lightweight keyframe sampler via reinforcement learning. The MLLM generates visual queries to guide frame retrieval, a 1D U-Net sampler learns selection weights from a CLIP similarity matrix, and both components are optimized end-to-end, achieving +8% accuracy improvement on long-form video QA.

## Background & Motivation

**Background**: MLLMs perform well on short video understanding, but in long-video scenarios, visual context grows linearly while attention computation scales quadratically. Uniform sampling is both inefficient and prone to missing key events. CLIP-similarity-based sampling methods (e.g., Q-Frame, AKS) have emerged as alternatives.

**Limitations of Prior Work**: Three critical issues arise — (Q1) Is the question alone sufficient to retrieve all relevant frames? (Insufficient information: questions are often interrogative and lack visual cues.) (Q2) How should similarity scores be converted into sampling weights? (Naïve top-$k$ selection results in redundant frames.) (Q3) Can the MLLM and sampler truly collaborate without joint evolution? (Existing methods freeze the MLLM while training the sampler, lacking bidirectional adaptation.)

**Key Challenge**: The sampler and MLLM are optimized independently — the sampler is unaware of what visual evidence the MLLM requires, and the MLLM does not adapt to the sparse frame distribution produced by the sampler.

**Goal**: Achieve joint evolution of the sampler and MLLM, enabling the MLLM to learn to generate sampling-guiding queries while adapting to sparse keyframe inputs.

**Key Insight**: RL (GRPO + REINFORCE) provides zero-shot feedback signals to optimize both components simultaneously.

**Core Idea**: The MLLM first reasons over multiple visual queries → CLIP constructs a query-frame similarity matrix → a lightweight 1D U-Net learns sampling weights → selected keyframes are fed back to the MLLM for answer generation → end-to-end RL joint training.

## Method

### Overall Architecture
A four-step inference pipeline: (1) MLLM-guided query generation — the MLLM infers $N_q$ visual queries from sparse preview frames and the question; (2) CLIP similarity matrix construction; (3) learnable keyframe sampling — a 1D U-Net produces sampling weights from the similarity matrix; (4) answer generation — the MLLM answers based on the selected high-resolution keyframes.

### Key Designs

1. **MLLM-Guided Query Generation (addressing Q1)**:

    - **Function**: The MLLM generates multiple visual queries describing visual events or cues relevant to the question.
    - **Mechanism**: $N_{init}$ frames are uniformly sampled as low-resolution previews (only 32 tokens per frame); the MLLM combines these with the question to generate $N_q$ queries.
    - **Design Motivation**: Questions alone lack visual descriptive power, but the MLLM can reason about "what visual content in the video would be needed to answer this question."

2. **Learnable Keyframe Sampler (addressing Q2)**:

    - **Function**: Learns sampling weights from the query-frame similarity matrix.
    - **Mechanism**: CLIP encodes queries and densely sampled frames to produce a similarity matrix $\mathbf{S} \in \mathbb{R}^{N_q \times N_f}$; a 1D U-Net (~2M parameters) maps this to per-frame sampling probabilities $\mathbf{p} \in \mathbb{R}^{N_f}$.
    - **Design Motivation**: The dense prediction and local perception properties of U-Net are naturally suited for predicting frame importance from multi-query similarity scores.

3. **Joint RL Training (addressing Q3)**:

    - The MLLM is optimized with GRPO: $G$ outputs are sampled per question and updated via group-relative advantage.
    - The sampler is optimized with REINFORCE, sharing the accuracy reward $r_{acc}$.
    - Three reward components: $r_{acc}$ (0.8, awarded for correct answers) + $r_{format}$ (0.1, awarded for correct format) + $r_{info}$ (0.1, query informativeness — requires a peaked rather than flat similarity distribution).
    - Sampler pretraining uses difficulty-aware rewards: for $c=0$ (questions the model has never answered correctly), a correct answer yields high reward $A=10$; for $c \neq 0$, $A = 1/c$ (correct) or $A = -1/(1-c)$ (incorrect).

### Dataset
- A newly constructed long-video QA dataset: 2.8K videos, 7.1K QA pairs.
- Multi-stage filtering ensures long duration, multi-event reasoning, and controllable difficulty.

## Key Experimental Results

### Main Results (Based on Qwen2.5-VL-7B)

| Method | Frames | MLVU | LVB | VideoMME-Long | LVBench |
|--------|--------|------|-----|---------------|---------|
| Uniform Sampling | 32 | 61.5 | 55.0 | 49.9 | 36.5 |
| Q-Frame | 32 | 66.8 | 58.7 | 53.1 | - |
| **MSJoE** | **32** | **69.3** | **60.1** | **54.1** | **46.4** |
| Uniform Sampling | 64 | 65.3 | 57.3 | 52.2 | 39.2 |
| TSPO | 64 | 74.3 | 64.2 | 56.4 | 46.4 |
| **MSJoE** | **64** | **75.1** | **62.2** | **57.4** | - |

### Performance Gains

| Metric | Description |
|--------|-------------|
| vs. base MLLM | +8.0% average accuracy |
| vs. strongest baseline | +1.1% average accuracy |
| LVBench (32 frames) | +9.9% improvement (36.5→46.4) |

### Ablation Study
- Query generation is critical — removing it and using the question directly for retrieval causes significant performance degradation.
- Joint training vs. separate training: joint training outperforms on all benchmarks.
- 1D U-Net vs. simple MLP: U-Net performs better, benefiting from its multi-scale local perception.

## Highlights & Insights
- The first framework to jointly evolve an MLLM and a sampler via RL, genuinely addressing the bidirectional adaptation problem.
- The design of having the MLLM reason and generate queries is elegant, resolving the fundamental issue of questions lacking visual cues.
- A lightweight sampler with only 2M parameters achieves strong frame selection capability.
- The difficulty-aware reward design effectively prevents the misleading effect of simple binary rewards on sampler training.

## Limitations & Future Work
- Inference requires two forward passes (preview for query generation + keyframe-based answering), increasing latency.
- The CLIP encoding overhead for densely sampled frames (1 FPS) is substantial for very long videos (hour-scale).
- The query count $N_q$ is currently fixed; adaptive adjustment based on question complexity could be explored.
- The stability of joint RL training is sensitive to sampler initialization, making pretraining a necessity.
- The dataset scale (2.8K videos) is relatively small, potentially limiting sufficient RL exploration.
- The design choice of the 1D U-Net sampler (vs. MLP or Transformer) warrants more thorough comparison.

### Dataset Details
- Videos in the LongVideoQA dataset average over 10 minutes in length, with some reaching several hours.
- Multi-stage filtering ensures question quality; low-difficulty and low-quality QA pairs are removed.
- Difficulty labels are automatically computed from baseline MLLM pass rates and used for difficulty-aware rewards during RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[AAAI 2026\] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](../../AAAI2026/multimodal_vlm/exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)
- [\[NeurIPS 2025\] in the eye of mllm benchmarking egocentric video intent understanding with gaze-](../../NeurIPS2025/multimodal_vlm/in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)

</div>

<!-- RELATED:END -->
