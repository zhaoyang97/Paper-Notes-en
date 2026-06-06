---
title: >-
  [Paper Note] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking
description: >-
  [CVPR 2026][Human Understanding][conversational avatars] This paper proposes UniLS, the first end-to-end framework for unified speaking and listening facial expression generation. Through a two-stage training paradigm—fi…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "conversational avatars"
  - "unified speaking-listening generation"
  - "audio-driven"
  - "facial animation"
  - "two-stage training"
date: 2026-05-08
content_hash: 28cd8495073a0067
---

# UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking

**Conference**: CVPR 2026
**arXiv**: [2512.09327](https://arxiv.org/abs/2512.09327)  
**Code**: None  
**Area**: Human Understanding
**Keywords**: conversational avatars, unified speaking-listening generation, audio-driven, facial animation, two-stage training

## TL;DR
This paper proposes UniLS, the first end-to-end framework for unified speaking and listening facial expression generation. Through a two-stage training paradigm—first learning intrinsic motion priors without audio, then fine-tuning with dual-track audio—UniLS generates natural speaking and listening facial motions simultaneously from dual-track audio input alone, achieving up to 44.1% improvement on listening metrics.

## Background & Motivation

1. **Background**: Most systems in the conversational avatar domain remain unidirectional—either speech-driven talking head generation or listener motion generation. Genuine interaction requires avatars capable of both speaking and listening modes simultaneously.

2. **Limitations of Prior Work**:
    - **Speaking-only methods** (e.g., FaceFormer, CodeTalker, ARTalk): Although capable of generating high-quality speaking facial motions, they entirely neglect listening behavior and cannot be deployed in conversational scenarios.
    - **The only speaking-listening method, DualTalk**: Relies on pre-computed facial sequences of speaker A to generate speaker B's motions. It is not end-to-end, requires an additional motion acquisition/generation pipeline, and cannot be deployed in real time.
    - **Direct end-to-end joint training fails**: Results in stiff and unnatural listening expressions (the "poker face" phenomenon).

3. **Key Challenge**: Speaking motions are strongly correlated with audio (phoneme–lip alignment), whereas listening motions exhibit weak correlation with the interlocutor's audio—eye blinks, head nods, and micro-expressions during listening arise primarily from intrinsic motion priors rather than external speech signals. This imbalance in audio–motion correlation causes the listening branch to collapse toward a low-variance, safe static prior during joint training.

4. **Goal**: Design an end-to-end framework that uses only dual-track audio to simultaneously generate the speaking and listening facial motions of both speakers, with the key challenge being the elimination of stiff listening expressions.

5. **Key Insight**: Listening behavior is reframed as a combination of "intrinsic motion priors" and "external audio modulation"—a person's listening expressions first follow their own motion patterns (e.g., blink frequency, subtle nods) and are only subsequently modulated by external speech.

6. **Core Idea**: Through two-stage training—first learning motion priors without audio, then fine-tuning with dual-track audio—the intrinsic dynamics of listening and external audio modulation are disentangled, fundamentally resolving the end-to-end listening stiffness problem.

## Method

### Overall Architecture
UniLS is built on a chunk-based autoregressive model for facial motion generation. The inputs are dual-track audio from speakers A and B; the outputs are 3D facial motions represented as FLAME parameters (expression parameters $\psi$ and pose parameters $\theta$). Training proceeds in two stages: Stage 1 learns motion priors without audio, and Stage 2 performs audio-driven fine-tuning. A multi-scale VQ codec discretizes facial motions at the base level.

### Key Designs

1. **Multi-Scale VQ Codec**:

    - **Function**: Discretizes continuous 3D facial motions into compact yet expressive discrete representations, serving as the supervision target for the autoregressive generator.
    - **Mechanism**: A Transformer encoder extracts temporal features $\mathbf{f}$ from an input motion chunk $M$, which are then quantized layer by layer using a multi-scale codebook $[k_1, k_2, ..., k_L]$ (scales $[1, 5, 25, 50, 100]$). Each layer quantizes then interpolates: $\mathbf{c}^{(l+1)} = \text{Interp}(\text{Quant}(\mathbf{f}^{(l)}), k_l)$, with residual $\mathbf{f}^{(l+1)} = \mathbf{f}^{(l)} - \mathbf{c}^{(l+1)}$. The codebook contains 256 entries, each of dimension 64.
    - **Design Motivation**: Multi-scale quantization progressively refines motion representations at different temporal resolutions, ensuring both high-fidelity synthesis and temporal coherence.

2. **Stage 1: Audio-Free Generator Training (Motion Prior Learning)**:

    - **Function**: Trains an autoregressive generator without any audio input, enabling the model to learn intrinsic dynamics priors of natural facial motion.
    - **Mechanism**: The input consists only of a motion chunk $M$ and a style embedding $\mathbf{s}$ (encoding speaker-specific motion characteristics); the generator $\mathcal{G}$ predicts the next chunk: $\hat{M}_{t:2t} = \mathcal{G}(M_{1:t}, \mathbf{s})$. An autoregressive reconstruction loss $\mathcal{L} = \sum_{t=1}^{T} ||\hat{M}_{t:2t} - M_{t:2t}||$ is used. Training data comprises unpaired multi-scenario footage including news broadcasts, interviews, live streams, and daily conversations (546.5 hours).
    - **Design Motivation**: By learning without audio conditioning, the model establishes intrinsic priors over natural facial behavior—eye blinks, micro-expressions, subtle head movements—which form the foundation of listening expressions and should not be audio-driven.

3. **Stage 2: Audio-Driven Fine-Tuning (Dual-Track Audio Modulation)**:

    - **Function**: Fine-tunes the generator on paired conversational data so that it generates speaking and listening motions conditioned on dual-track audio.
    - **Mechanism**: Two cross-attention layers are added to each Transformer block—one attending to speaker A's audio (controlling speaking behavior) and the other attending to speaker B's audio (controlling listening behavior). The newly added cross-attention layers are trained from scratch, while the Stage 1 backbone is fine-tuned with LoRA to protect the learned motion priors from being overwritten. The generation process becomes $\hat{M}_{t:2t} = \mathcal{G}(M_{1:t}, \mathbf{a}^A_{1:t}, \mathbf{a}^B_{1:t}, \mathbf{s})$. Audio features are extracted by a frozen wav2vec encoder.
    - **Design Motivation**: The dual cross-attention design allows the model to distinguish between self-speech and interlocutor speech (using a single cross-attention with mixed audio causes severe degradation in speaking quality, with LVE rising from 5.83 to 11.48). LoRA fine-tuning preserves motion priors, preventing Stage 2 from overfitting to audio signals and losing listening diversity.

### Loss & Training
Both stages use chunk-wise autoregressive reconstruction loss. Stage 1 is trained for 10 GPU hours on 4× H200 GPUs; Stage 2 requires 30 GPU hours. The AdamW optimizer is used with a learning rate of 1e-4, batch size of 128, and 200K training iterations.

## Key Experimental Results

### Main Results
Evaluation on the Seamless Interaction dataset test set for both speaking and listening performance.

| Method | LVE↓ | MHD↓ | Speaking FDD↓ | Speaking PDD↓ | Speaking JDD↓ | Listening FDD↓ | Listening PDD↓ | Listening JDD↓ | F-FID↓ | P-FID↓ |
|--------|------|------|--------------|--------------|--------------|---------------|---------------|---------------|--------|--------|
| DiffPoseTalk | 9.48 | 2.96 | 32.66 | 7.89 | 1.40 | - | - | - | - | - |
| ARTalk | 7.46 | 2.12 | 31.64 | 7.66 | 1.19 | - | - | - | - | - |
| ARTalk* | 6.79 | 2.02 | 27.41 | 8.55 | 0.81 | 30.62 | 9.52 | 1.53 | 10.78 | 0.072 |
| DualTalk | 6.35 | 1.95 | 37.46 | 9.70 | 1.02 | 43.58 | 10.71 | 2.02 | 13.14 | 0.079 |
| **UniLS** | **5.83** | **1.89** | **18.41** | **4.67** | **0.71** | **17.12** | **4.75** | **0.98** | **4.30** | **0.038** |

### Ablation Study

| Configuration | LVE↓ | MHD↓ | Listening FDD↓ | F-FID↓ | Notes |
|---------------|------|------|---------------|--------|-------|
| Single cross-attention | 11.48 | 3.00 | 27.46 | 5.97 | Mixed audio causes severe speaking degradation |
| Without Stage 1 | 6.32 | 2.02 | 25.64 | 5.97 | Absence of motion priors leads to stiff listening |
| Without multi-scenario data | 6.26 | 1.99 | 17.81 | 4.62 | Data diversity benefits listening |
| Full UniLS | **5.83** | **1.89** | **17.12** | **4.30** | All components synergize optimally |

### Key Findings
- **Substantial improvement on listening metrics**: UniLS achieves F-FID of 4.30, a 67.3% reduction compared to DualTalk (13.14), and P-FID decreases by 51.9%, indicating that generated listening motions are substantially closer to the real distribution.
- **Speaking accuracy is also state of the art**: LVE of 5.83 is the lowest across all methods, indicating the best lip synchronization accuracy.
- **Dual cross-attention is critical**: Single cross-attention nearly doubles LVE (5.83→11.48) because the model cannot distinguish speaking/listening signals in the mixed audio stream.
- **Stage 1 is indispensable**: Removing Stage 1 increases listening FDD from 17.12 to 25.64 (a 50% increase), validating the central role of motion prior learning in resolving listening stiffness.
- In a user study with 25 participants, the listening preference for UniLS over DualTalk reached 91.35%.

## Highlights & Insights
- **The analysis of "audio–motion correlation imbalance" is highly precise**: t-SNE visualizations clearly demonstrate the strong alignment between speaking audio and motion features versus the weak alignment for listening audio, precisely localizing the root cause. This insight is more valuable than the method design itself.
- **The two-stage decoupled design is elegant and effective**: Learning motion priors first, then fine-tuning with LoRA, simultaneously preserves diversity and introduces audio control—a highly elegant solution. This "prior-first, fine-tune-later" paradigm is transferable to other tasks with multimodal imbalance.
- **The first end-to-end speaking-listening framework**: Driven directly by dual-track audio without requiring pre-generated facial sequences of the interlocutor, it has clear practical value for real-time conversational systems.

## Limitations & Future Work
- The current work addresses only facial motions (FLAME parameters) and does not cover body pose or gestures; a complete conversational avatar requires full-body motion.
- Evaluation metrics for listening behavior are primarily distributional (FID-based), lacking assessment of the "timing" and "semantic relevance" of listening responses—i.e., whether appropriate reactions occur at the right moments.
- Training requires large-scale paired conversational data (657.5 hours), incurring substantial data acquisition costs.
- Extension to multi-party conversations (>2 speakers) is not explored.

## Related Work & Insights
- **vs. DualTalk**: DualTalk requires pre-computed facial sequences of speaker A and is not end-to-end; UniLS requires only dual-track audio. Listening F-FID decreases from 13.14 to 4.30.
- **vs. ARTalk***: ARTalk* is an author-adapted baseline (with additional audio input), but listening remains stiff, with F-FID of 10.78—far above UniLS's 4.30.
- **vs. FaceFormer/CodeTalker**: These speaking-only methods do not handle listening at all; UniLS also surpasses them in speaking accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First end-to-end speaking-listening framework; the two-stage training paradigm fundamentally resolves the listening stiffness problem
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative evaluation, user study, and ablation, but lacks temporal semantic alignment assessment
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough motivation analysis with intuitive t-SNE visualizations
- Value: ⭐⭐⭐⭐⭐ Significant contribution toward interactive avatar systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)
- [\[ICML 2026\] DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing](../../ICML2026/human_understanding/discoforcing_a_unified_framework_for_real-time_audio-driven_character_control_wi.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[CVPR 2026\] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)
- [\[CVPR 2026\] Talking Together: Synthesizing Co-Located 3D Conversations from Audio](talking_together_synthesizing_co-located_3d_conversations_from_audio.md)

</div>

<!-- RELATED:END -->
