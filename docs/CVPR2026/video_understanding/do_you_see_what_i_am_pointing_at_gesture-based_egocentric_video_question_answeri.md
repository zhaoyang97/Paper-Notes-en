---
title: >-
  [Paper Note] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering
description: >-
  [CVPR 2026][Video Understanding][Egocentric Video QA] This paper proposes the EgoPointVQA dataset and the HINT (Hand Intent Tokens) method, which encodes 3D hand keypoints into hand intent tokens interleaved with visual tokens as input to an MLLM, addressing deictic gesture-based question answering in egocentric video. HINT-14B achieves 68.1% accuracy, surpassing InternVL3-14B by 5.4 pp.
tags:
  - CVPR 2026
  - Video Understanding
  - Egocentric Video QA
  - Gesture Understanding
  - Deictic Reference
  - 3D Hand Keypoints
  - Multimodal Large Language Models
date: 2026-05-08
content_hash: 9733166984705a9e
---

# Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering

**Conference**: CVPR 2026
**arXiv**: [2603.12533](https://arxiv.org/abs/2603.12533)
**Code**: [https://yuuraa.github.io/papers/choi2026egovqa](https://yuuraa.github.io/papers/choi2026egovqa) (coming soon)
**Area**: Video Understanding
**Keywords**: Egocentric Video QA, Gesture Understanding, Deictic Reference, 3D Hand Keypoints, Multimodal Large Language Models

## TL;DR
This paper proposes the EgoPointVQA dataset and the HINT (Hand Intent Tokens) method, which encodes 3D hand keypoints into hand intent tokens interleaved with visual tokens as input to an MLLM, addressing deictic gesture-based question answering in egocentric video. HINT-14B achieves 68.1% accuracy, surpassing InternVL3-14B by 5.4 pp.

## Background & Motivation
With the growing prevalence of AR/VR devices (Apple Vision Pro, Meta Orion) and smart glasses, AI assistants must understand where users are directing their attention in the environment. In natural communication, people frequently use deictic expressions (e.g., "What is this?" or "Should I use this one?"), which can only be answered by interpreting the user's pointing gesture.

**Limitations of Prior Work**: Current MLLMs (including GPT-4o and Qwen3-VL-32B) perform poorly on such tasks for two reasons: (1) training data lacks gesture-rich egocentric video; and (2) at the architectural level, there is no explicit mechanism to encode gesture information—models perform only global vision-language fusion and cannot map "this" to the specific object indicated by a pointing finger.

**Core Idea**: A lightweight adapter encodes 3D hand keypoints into hand intent tokens aligned with visual tokens, explicitly supplying the model with pointing gesture information.

## Method

### Overall Architecture
HINT adds a parallel hand intent stream to the standard MLLM architecture. For each video frame: (1) a visual encoder extracts visual tokens $V_t$; (2) WiLoR extracts 21 3D hand keypoints $K_t \in \mathbb{R}^{21 \times 3}$; (3) a Keypoint Adapter maps the keypoints to hand intent tokens $H_t$. Visual tokens and hand intent tokens are then interleaved frame-by-frame before being fed into the LLM.

### Key Designs
1. **EgoPointVQA Dataset**:

    - **Function**: Constructs the first egocentric video dataset for deictic gesture-based question answering.
    - **Mechanism**: 4,000 synthetic videos (AI2-THOR simulator, 184 indoor scenes + MIXAMO inverse kinematics animations for pointing gestures) + 400 real videos (Meta Ray-Ban smart glasses, 20 participants, 360 indoor + 40 outdoor), totaling 18,745 QA pairs.
    - Six task categories: Reference (pointing target identification), Counting (same-category counting), Spatial (spatial relations), Temporal (sequential multi-pointing order), Attribute, and Feedback (functional feedback).
    - Test set: 300 real videos, 672 QA pairs, manually verified for correctness and referential ambiguity.
    - **Design Motivation**: No existing egocentric VQA dataset focuses on gesture-pointing scenarios; the absence of such training data is the root cause of model performance bottlenecks.

2. **Keypoint Adapter (Hand Intent Token Encoder)**:

    - **Function**: Compresses 21 3D hand keypoints into a single hand intent token.
    - **Mechanism**: The keypoints are first flattened into a 63-dimensional vector $\tilde{k}_t = \text{flatten}(K_t) \in \mathbb{R}^{63}$, then mapped to the LLM hidden dimension via LayerNorm + two-layer MLP + GeLU:
    $H_t = W_2 \sigma(W_1 \text{LN}(\tilde{k}_t)), \quad W_1 \in \mathbb{R}^{d_h \times 63}, W_2 \in \mathbb{R}^{d \times d_h}$
      When hand detection confidence $c_t < \tau = 0.5$, no token is inserted (frames without hands are skipped).
    - **Design Motivation**: Overlaying keypoints or arrows directly onto frames performs poorly (as shown in ablation studies); learned encoding allows the model to discover how to exploit geometric information. The adapter is extremely lightweight (< 1% token overhead, adding only 0.26 s at inference).

3. **Frame-Keypoint Interleaving**:

    - **Function**: Interleaves hand intent tokens with the visual tokens of the corresponding frame.
    - **Mechanism**: The sequence format is `Frame-1: <vis> Keypoint-1: <key> Frame-2: <vis> ...`, and the model conditions on hand intent signals during autoregressive answer generation:
    $p(X_a | V, X_q, H) = \prod_{i=1}^{L} p(x_i | V, X_{q,<i}, X_{a,<i}, H_{<i})$
    - **Design Motivation**: Interleaving rather than concatenation enables the LLM to naturally associate gestures with their corresponding frames along the temporal dimension, achieving spatiotemporal alignment.

### Loss & Training
- LoRA fine-tuning applied to both the vision encoder and the LLM; the Keypoint Adapter is trained from scratch.
- AdamW + cosine schedule, batch size 32, 1 epoch.
- Training data combines synthetic videos with 100 real videos.

## Key Experimental Results

### Main Results

| Model | Params | Reference | Temporal | Spatial | Count | Attr. | Feed. | Avg. |
|-------|--------|-----------|----------|---------|-------|-------|-------|------|
| GPT-5 | - | 75.6 | 53.6 | 62.3 | 50.0 | 56.1 | 77.8 | 62.6 |
| Qwen3-VL | 32B | 63.7 | 67.9 | 65.8 | 66.7 | 63.4 | 77.2 | 67.5 |
| InternVL3 | 14B | 63.1 | 66.1 | 61.4 | 50.0 | 58.5 | 77.2 | 62.7 |
| **HINT-InternVL3** | **14B** | **73.8** | **69.6** | **64.9** | **54.2** | **63.4** | **82.5** | **68.1** |
| InternVL3 | 8B | 66.1 | 57.5 | 63.2 | 33.3 | 51.3 | 76.8 | 58.0 |
| **HINT-InternVL3** | **8B** | **75.0** | **66.1** | **64.9** | **35.4** | **61.0** | **79.8** | **63.7** |

HINT-14B achieves an average accuracy of 68.1%, surpassing the InternVL3-14B baseline by 5.4 pp and even exceeding GPT-5 (62.6%).

### Ablation Study

| Configuration | Reference | Temporal | Spatial | Attribute | Notes |
|---------------|-----------|----------|---------|-----------|-------|
| InternVL3-8B (zero-shot) | 66.1 | 57.5 | 63.2 | 51.3 | No fine-tuning |
| + SFT only | 68.5 | 60.7 | 59.6 | 56.7 | Fine-tuning only, no hand intent tokens |
| + SFT + HINT | **75.0** | **66.1** | **64.9** | **61.0** | Full method |

| Hand Intent Modeling | Reference | Temporal | Spatial |
|----------------------|-----------|----------|---------|
| None (SFT only) | 68.5 | 60.7 | 59.6 |
| Visual Keypoints (overlaid on frame) | 57.1 | 60.7 | 61.4 |
| Visual Arrow (overlaid on frame) | 70.2 | 60.7 | 62.3 |
| 3D Keypoints in Text | 68.5 | 55.4 | 58.8 |
| **HINT (learned encoding)** | **75.0** | **66.1** | **64.9** |

### Key Findings
- SFT alone yields only +2.4 pp improvement; adding HINT delivers +8.9 pp on Reference, demonstrating that explicit gesture encoding is far more important than data scaling alone.
- Overlaying keypoints on frames actually hurts performance (Reference drops to 57.1%), as visual overlays interfere with the MLLM's visual comprehension.
- Even InternVL3 at 78B achieves only 66.6% average accuracy, confirming that scaling up model size does not resolve the problem.
- HINT tokens account for less than 1% of total tokens, and inference latency increases by only 0.26 s (2.58 s → 2.84 s).
- Mixed synthetic + real training outperforms either data source used alone.

## Highlights & Insights
- The paper precisely identifies a critical yet overlooked problem: MLLMs fail to understand what "this" refers to.
- The dataset design is comprehensive: synthetic + real data, six task categories, and rigorous human quality control.
- The HINT method is extremely lightweight (< 1% token overhead) while yielding substantial performance gains.
- The approach generalizes consistently across multiple backbones (LLaVA-OV, InternVL3-8B, InternVL3-14B).
- The work has strong implications for embodied AI and AR assistant research directions.

## Limitations & Future Work
- Only pointing gestures are supported; other gesture types (e.g., grasping, waving, size indication) are not covered.
- The method relies on the accuracy of WiLoR hand reconstruction and may fail in challenging scenarios (heavy occlusion, incomplete hand visibility).
- A domain gap remains between synthetic and real-world data, and the real training set is small (only 100 of 400 real videos used for training).
- The test set comprises only 672 QA pairs, which is relatively small in scale.
- Gesture disambiguation in multi-person scenarios has not been explored.

## Related Work & Insights
- Unlike region-specific VQA methods (Ferret, Osprey, Artemis), this work does not rely on provided bounding boxes but instead infers the referent region from gestures.
- Visual prompting approaches (SoM, alphanumeric tags) guide MLLMs with manually designed markers; this paper uses natural gesture signals instead.
- EgoGPT and Ego-R1 focus on long-term memory and habit analysis, whereas this work targets fine-grained gesture-pointing comprehension—the two directions are complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First deictic gesture egocentric VQA task and dataset; HINT is a novel and effective design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 15 baseline comparisons + extensive ablation studies, though dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear, structure is complete, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ Identifies a key direction for AR/VR assistants; both the dataset and method are broadly impactful.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EgoPointVQA: Gesture-Based Egocentric Video Question Answering](egopointvqa_gesture_based_egocentric_video_qa.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[CVPR 2026\] StreamReady: Learning What to Answer and When in Long Streaming Videos](streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)

<!-- RELATED:END -->
