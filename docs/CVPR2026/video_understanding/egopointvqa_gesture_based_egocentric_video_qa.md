---
title: >-
  [Paper Note] EgoPointVQA: Gesture-Based Egocentric Video Question Answering
description: >-
  [CVPR 2026][Video Understanding][egocentric VQA] This paper proposes the EgoPointVQA dataset (4,000 synthetic + 400 real egocentric videos) and the HINT method, which encodes 3D hand keypoints into hand intent tokens interleaved with visual tokens as input to an MLLM, enabling the model to interpret pointing gestures and answer deictic questions. HINT-14B achieves 68.1% accuracy, outperforming InternVL3-14B by 6.6 percentage points.
tags:
  - CVPR 2026
  - Video Understanding
  - egocentric VQA
  - pointing gesture
  - deictic reasoning
  - hand intent tokens
  - MLLM
date: 2026-05-08
content_hash: bac1554cf7ec6009
---

# EgoPointVQA: Gesture-Based Egocentric Video Question Answering

**Conference**: CVPR 2026
**arXiv**: [2603.12533](https://arxiv.org/abs/2603.12533)
**Code**: To be released (authors have committed to releasing code/model/dataset)
**Area**: Egocentric Video Understanding / Multimodal QA / Gesture Understanding
**Keywords**: egocentric VQA, pointing gesture, deictic reasoning, hand intent tokens, MLLM

## TL;DR
This paper proposes the EgoPointVQA dataset (4,000 synthetic + 400 real egocentric videos) and the HINT method, which encodes 3D hand keypoints into hand intent tokens interleaved with visual tokens as input to an MLLM, enabling the model to interpret pointing gestures and answer deictic questions. HINT-14B achieves 68.1% accuracy, outperforming InternVL3-14B by 6.6 percentage points.

## Background & Motivation
With the proliferation of AR/VR devices and smart glasses (Apple Vision Pro, Meta Ray-Ban), egocentric AI assistants must understand spatial references expressed through pointing gestures and deictic pronouns such as "this" and "that." Existing MLLMs are severely deficient in this regard: (1) training data lacks gesture-rich egocentric videos; and (2) architecturally, no mechanism exists to explicitly encode gesture information—visual and textual inputs are globally fused, making it impossible to associate deictic expressions with the object the finger points to. Even GPT-4o achieves only 46.8% average accuracy on this task, and GPT-5 reaches only 62.6%.

## Core Problem
How can MLLMs interpret a user's pointing gestures from egocentric video and correctly answer questions containing deictic pronouns?

## Method

### Overall Architecture
The EgoPointVQA system comprises two components: (1) a dataset and evaluation benchmark defining six categories of deictic reasoning tasks, with synthetic and real videos paired with multiple-choice QA; and (2) the HINT method, which augments the standard MLLM visual stream with a gesture intent stream by encoding 3D hand keypoints into tokens interleaved with visual tokens.

### Key Designs

1. **EgoPointVQA Dataset**:

    - **Synthetic videos**: 4,000 videos generated using the AI2-THOR simulator across 184 indoor scenes and 12,000 viewpoints; MIXAMO animations with inverse kinematics align fingertips to target objects; resolution 448×448 at 30 FPS.
    - **Real videos**: 400 videos recorded by 20 participants (from 12 nationalities) using Meta Ray-Ban glasses; resolution 1536×2048 at 30 FPS, 3–8 seconds per clip.
    - **Six task categories**: Reference (object identification), Counting (counting objects of the same category), Spatial (relative position/depth), Temporal (multi-gesture temporal ordering), Attribute (color/shape/material), and Feedback (functionality/suitability).
    - **QA generation pipeline**: Three stages—Stage 1 uses InternVL3-78B to extract dense scene information; Stage 2 generates structured multiple-choice QA; Stage 3 uses GPT-4o to rewrite questions in deictic pronoun form.
    - Training set: 18,073 QA pairs (all synthetic + 640 QA from 100 real videos); test set: 672 QA pairs (300 real videos).

2. **HINT (Hand Intent Tokens)**:

    - **3D hand pose extraction**: WiLoR (a robust in-the-wild hand reconstruction model) extracts 21 3D keypoints $K_t \in \mathbb{R}^{21\times3}$ per frame.
    - **Keypoint Adapter**: The 63-dimensional flattened feature is mapped to a single Hand Intent Token $H_t$ matching the LLM embedding dimension via LayerNorm → $W_1(63\rightarrow d_h)$ → GeLU → $W_2(d_h\rightarrow d)$. Tokens are omitted when detection confidence falls below $\tau=0.5$.
    - **Frame-keypoint interleaving**: In the input sequence, the visual tokens for each frame are immediately followed by the corresponding $H_t$, enabling the LLM to attend jointly to visual and gesture information during autoregressive generation.

3. **Training strategy**: Only the Keypoint Adapter and LoRA modules (applied to the visual encoder and LLM) are trained; backbone parameters are frozen. AdamW optimizer with cosine schedule, warmup ratio 0.03, batch size 32, 1 epoch, trained on mixed synthetic and real data.

### Loss & Training
Standard autoregressive language modeling loss: $p(X_a | V, X_q, H) = \prod p(x_i | V, X_{q,<i}, X_{a,<i}, H_{<i})$, where $H$ provides explicit gesture conditioning signals.

## Key Experimental Results

| Method | Scale | Refer. | Temporal | Spatial | Count | Attr. | Feed. | Avg |
|--------|-------|--------|----------|---------|-------|-------|-------|-----|
| GPT-5 | - | 75.6 | 53.6 | 62.3 | 50.0 | 56.1 | 77.8 | 62.6 |
| GPT-4o | - | 56.1 | 29.5 | 43.1 | 44.8 | 41.5 | 65.7 | 46.8 |
| InternVL3 | 14B | 63.1 | 66.1 | 61.4 | 50.0 | 58.5 | 77.2 | 62.7 |
| **HINT** | **14B** | **73.8** | **69.6** | **64.9** | **54.2** | **63.4** | **82.5** | **68.1** |
| InternVL3 | 8B | 66.1 | 57.5 | 63.2 | 33.3 | 51.3 | 76.8 | 58.0 |
| **HINT** | **8B** | **75.0** | **66.1** | **64.9** | **35.4** | **61.0** | **79.8** | **63.7** |

- HINT-14B achieves an average accuracy of 68.1%, surpassing InternVL3-14B by 5.4 percentage points.
- Human performance is 95.9%, leaving a gap of approximately 28%.
- HINT tokens account for less than 1% of total tokens; inference time increases from 2.58 s to 2.84 s (+10%).
- Performance on standard video understanding benchmarks (Video-MME/MVBench/EgoSchema) is on par with the baseline, with no catastrophic forgetting.

### Ablation Study
- SFT alone (without HINT): Reference accuracy increases from 66.1% to 68.5%; adding HINT further raises it to 75.0%, demonstrating that data and architecture are both indispensable.
- Combined synthetic + real data yields the best result (75.0%); synthetic only achieves 69.0%; real only achieves 67.3%.
- Gesture modeling comparison: visualized keypoints 57.1%, visualized arrows 70.2%, textual keypoints 68.5%, **HINT 75.0%**—indicating that allowing the model to learn hand geometry representations automatically is superior to manual encoding.
- Confidence threshold $\tau=0.5$ is optimal; $\tau=0.7$ is too restrictive (64.9%), $\tau=0.1$ is too permissive (66.7%).
- Removing gesture information causes a dramatic drop to 41.7% on Reference, confirming that gestures are the primary cue.

## Highlights & Insights
- Addresses an important research gap: pointing gesture–driven egocentric VQA has received virtually no prior attention.
- HINT is elegantly simple: a two-layer MLP adapter with an interleaving strategy introduces negligible token overhead.
- The dataset construction pipeline is complete and reproducible, combining AI2-THOR, inverse kinematics, InternVL3-78B, and GPT-4o.
- Ablation experiments are comprehensive, comparing five gesture modeling approaches, threshold values, frame sampling strategies, and data compositions.
- Bias analysis is rigorous: text-only and choices-only baselines both approach chance performance, confirming the absence of shortcut solutions.

## Limitations & Future Work
- Gestures are limited to pointing; other interaction gestures such as grasping or waving are not addressed.
- WiLoR keypoint estimation degrades under motion blur and occlusion, which constitutes the primary failure mode.
- The dataset scale is limited (672 test QA pairs), potentially reducing statistical confidence.
- A domain gap persists between synthetic and real-world scenarios.
- Evaluation is restricted to multiple-choice format; open-ended response settings have not been explored.

## Related Work & Insights
- vs. EgoGPT/Ego-R1: These methods focus on long-term memory and habit analysis in egocentric VQA and do not address gestural reference.
- vs. Ferret/Osprey/DAM: These region-level VQA methods require explicitly provided bounding boxes or masks, whereas HINT infers intent from natural gestures.
- vs. Set-of-Mark/ViP-LLaVA: These approaches rely on manual visual annotations (labels, scribbles), whereas HINT uses natural gesture signals.
- vs. VGLLM-QA: This method leverages 3D geometric priors but does not handle gestures, achieving only 48.9% on this task.

The approach of lightly encoding outputs from off-the-shelf hand reconstruction models into tokens is generalizable to other body language understanding tasks. The gesture token–visual token interleaving strategy parallels the multimodal token mixing paradigm. The method has direct applicability to AR/VR interaction and assistive technologies (e.g., assistance for the visually impaired).

## Rating
- **Novelty**: ⭐⭐⭐⭐ First dataset and method for gesture-driven egocentric VQA; novel problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 15 baseline comparisons, extensive ablations, bias analysis, and human performance comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Task definitions are clear, figures are informative, and dataset construction details are thorough.
- **Value**: ⭐⭐⭐⭐ Significant contribution to AR/VR interaction and embodied AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[CVPR 2026\] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)

</div>

<!-- RELATED:END -->
