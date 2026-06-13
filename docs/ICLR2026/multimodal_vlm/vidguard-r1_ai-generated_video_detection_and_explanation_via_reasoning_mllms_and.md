---
title: >-
  [Paper Note] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL
description: >-
  [ICLR 2026][Multimodal VLM][AI-generated video detection] VidGuard-R1 is the first video authenticity detector that fine-tunes an MLLM with GRPO (Group Relative Policy Optimization). By constructing a 140K shortcut-free…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "AI-generated video detection"
  - "MLLM reasoning"
  - "GRPO"
  - "temporal artifacts"
  - "explainable forensics"
date: 2026-05-08
content_hash: 45889b8cda61e122
---

# VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL

**Conference**: ICLR 2026
**arXiv**: [2510.02282](https://arxiv.org/abs/2510.02282)  
**Code**: [Project Page](https://vidguard-r1.github.io/)  
**Area**: Multimodal VLM
**Keywords**: AI-generated video detection, MLLM reasoning, GRPO, temporal artifacts, explainable forensics

## TL;DR

VidGuard-R1 is the first video authenticity detector that fine-tunes an MLLM with GRPO (Group Relative Policy Optimization). By constructing a 140K shortcut-free real/fake video dataset and designing two specialized reward mechanisms—temporal artifact reward and diffusion-step quality reward—it achieves 86.17% accuracy on its in-house dataset and 95%+ zero-shot SOTA performance on GenVidBench and GenVideo benchmarks, while generating interpretable chain-of-thought reasoning.

## Background & Motivation

**Background**: AI video generation models (Sora, HunyuanVideo, Wan, etc.) have rapidly improved output quality, blurring the boundary between generated and real videos and introducing serious societal risks such as misinformation, privacy violations, and fraud. Accurate and explainable detection tools are urgently needed.

**Limitations of Prior Work**:

1. **Limited generalization of traditional detectors**: Early deepfake detectors target facial forgery only and fail to generalize to open-domain multi-scene videos; spatiotemporal consistency methods are easily bypassed by post-processing.
2. **Poor direct application of MLLMs**: Powerful MLLMs such as GPT-4o achieve only ~57% accuracy when directly applied to video authenticity judgment—barely above random guessing.
3. **Weak reasoning under SFT fine-tuning**: SFT improves detection accuracy but fails to elicit meaningful explanations such as "why this video is fake"—reasoning ability remains insufficient.
4. **Shortcuts in existing datasets**: Benchmarks such as GenVideo and GenVidBench contain systematic differences in resolution, frame rate, bitrate, and duration between real and fake videos, causing models to exploit metadata rather than visual authenticity.

**Key Challenge**: Models must both accurately detect and deeply reason about "where the fake is," yet SFT can only teach output format without stimulating exploratory reasoning.

**Goal**: The paper introduces a GRPO reinforcement learning framework that encourages models to autonomously discover physical inconsistencies through multi-path reasoning sampling and group ranking, and designs two specialized reward signals to guide temporal reasoning and quality awareness.

## Method

### Overall Architecture

VidGuard-R1 adopts a two-stage training pipeline: **Stage 1** SFT initialization (learning CoT reasoning format on 30K videos) → **Stage 2** RL-enhanced reasoning (further improving reasoning and detection with GRPO/DPO on 100K videos). The backbone model is Qwen2.5-VL-7B.

### Key Design 1: Shortcut-Free Training Data Construction

A critical flaw in existing benchmarks is the systematic difference in low-level features between real and fake videos (e.g., real videos >10s, fake videos <4s), causing models to take shortcuts. This paper constructs a 140K video dataset to eliminate such biases:

- **Real video sources**: InternVid (55K) + ActivityNet (15K)
- **Generated videos**: HunyuanVideo-I2V (50K) + CogVideoX-5B (20K) generate corresponding fake videos from the first frame and text description of real videos
- **Standardization**: All videos are unified to 49 frames, 8 FPS, 720×480 resolution, YUV420p format
- **CoT annotation**: Qwen-2.5-VL-72B generates reasoning annotations covering dimensions such as action consistency, lighting consistency, texture artifacts, and physical plausibility

### Key Design 2: GRPO-TA (Temporal Artifact-Enhanced Reward)

Standard GRPO tends to exploit local visual cues (pixel distortion, lighting anomalies) while neglecting temporal inconsistencies. GRPO-TA reinforces temporal reasoning by injecting temporal artifacts:

- **Operation**: Videos are randomly subjected to segment repetition or frame-order reversal (operation regions are selected based on a Gaussian distribution)
- **Asymmetric reward design**: Detecting temporal anomalies in real videos after tampering yields a higher reward $\alpha_1 = 0.5$ (harder to detect), while detecting tampered generated videos yields a lower reward $\alpha_2 = 0.3$ (easier to detect)
- **Conditional activation**: Additional rewards are added only when the original video prediction is correct and the group accuracy of tampered videos satisfies $\tilde{p} > \mu = 0.8$

Reward function:

$$r_i^{\text{GRPO-TA}} = \begin{cases} r_i^{\text{GRPO}} + w_i, & \text{if } o_i \text{ correct and } \tilde{p} > \mu \\ r_i^{\text{GRPO}}, & \text{otherwise} \end{cases}$$

### Key Design 3: GRPO-Q (Quality Evolution Reward)

This design exploits an intrinsic property of diffusion models—different numbers of reverse diffusion steps produce videos of different quality—to train the model for fine-grained quality awareness:

- **Data augmentation**: For 12K real videos, 5 quality levels (20%, 40%, 60%, 80%, 95%) are generated using 10–50 diffusion steps, yielding 72K samples per model
- **Graded reward**: An exact match of the quality level receives full reward $\delta = 1$; an inexact match receives partial reward proportional to the distance between predicted and ground-truth steps: $g(o_i, y_i) = \delta \cdot (1 - |s(o_i) - s(y_i)|)$
- **Beyond binary judgment**: The model not only classifies real vs. fake, but also estimates the degree of quality degradation in generated videos

## Key Experimental Results

### Main Results: Detection Performance on In-House Dataset

| Method | Type | CogVideoX Acc. (%) | HunyuanVideo Acc. (%) |
|--------|------|:---:|:---:|
| I3D | CNN | 64.78 | 62.13 |
| SlowFast | CNN | 77.87 | 77.03 |
| TimeSformer | Transformer | 78.53 | 74.55 |
| VideoSwin | Transformer | 76.81 | 79.71 |
| GPT-4o | MLLM | 56.81 | 57.42 |
| Qwen2.5-VL-7B | MLLM | 50.95 | 52.83 |
| VidGuard-R1 (CoT/SFT) | MLLM | 66.18 | 63.19 |
| VidGuard-R1 (DPO) | MLLM | 79.13 | 80.88 |
| VidGuard-R1 (GRPO) | MLLM | 81.30 | 81.90 |
| VidGuard-R1 (GRPO-TA) | MLLM | 82.17 | 83.72 |
| VidGuard-R1 (GRPO-Q) | MLLM | **84.32** | **86.17** |

Key observations: (1) Qwen2.5-VL-7B/GPT-4o applied directly are near random (~50–57%); (2) SFT raises accuracy to 66%, still below traditional video models; (3) GRPO improves ~2% over DPO; (4) GRPO-TA and GRPO-Q each contribute ~2% and ~5% further gains, validating the effectiveness of specialized rewards.

### Zero-Shot Cross-Benchmark Generalization

| Method | GenVidBench Mean (%) | GenVideo Best Metric |
|--------|:---:|:---:|
| MViT V2 | 79.90 | - |
| GPT-4.1 mini | 59.62 | - |
| VidGuard-R1 (GRPO, zero-shot) | 96.37 | F1: 0.97 |
| VidGuard-R1 (GRPO, fine-tuned) | **97.53** | F1: **0.98** |

VidGuard-R1 achieves 96.37% zero-shot on GenVidBench, surpassing the previous SOTA (MViT V2, 79.90%) by approximately 17 percentage points; F1 on GenVideo also leads by a large margin. Fine-tuning further improves performance to 97.53%.

### Ablation Study: Contribution of Each Training Stage

| Training Configuration | CogVideoX | HunyuanVideo | Gain Source |
|------------------------|:---:|:---:|-------------|
| SFT (CoT) | 66.18 | 63.19 | Basic reasoning format |
| + DPO | 79.13 | 80.88 | Preference alignment +15% |
| + GRPO | 81.30 | 81.90 | Group-ranking exploration +2% |
| + GRPO-TA | 82.17 | 83.72 | Temporal reasoning +1.8% |
| + GRPO-Q | 84.32 | 86.17 | Quality awareness +2.5% |

Each stage yields clear and consistent improvements. The largest jump occurs from SFT to DPO (~15%), indicating that preference learning is critical; the graded quality reward in GRPO-Q delivers the strongest incremental gain.

## Highlights & Insights

### Strengths

1. **Pioneering contribution**: The first work to apply GRPO reinforcement learning to AI-generated video detection, establishing a "detection + explanation" paradigm.
2. **Clever reward design**: The asymmetric temporal artifact reward in GRPO-TA and the diffusion-step quality reward in GRPO-Q both leverage intrinsic properties of generative models in a targeted manner.
3. **Rigorous dataset construction**: Systematic shortcut elimination via standardization ensures models learn visual authenticity rather than metadata differences.
4. **Strong generalization**: Zero-shot performance exceeds 95% on GenVidBench/GenVideo, far surpassing all prior methods.

### Limitations & Future Work

1. The backbone is fixed to Qwen2.5-VL-7B; generalizability to other MLLMs is not verified.
2. GRPO-Q requires generating videos at multiple diffusion steps, incurring high data construction costs.
3. Given the rapid iteration of generative models, the lasting effectiveness of the detection approach remains uncertain.

### Rating

⭐⭐⭐⭐ — A pioneering work introducing reasoning-based RL into video forensics. The method design is elegant, experiments are thorough, and the work provides a compelling paradigm for explainable AI safety detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)

</div>

<!-- RELATED:END -->
