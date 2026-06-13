---
title: >-
  [Paper Note] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation
description: >-
  [CVPR 2026][Video Generation][Talking portrait generation] UniTalking is proposed as an end-to-end talking portrait generation framework built upon MM-DiT. Through a joint attention mechanism within a dual-stream symmetr…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Talking portrait generation"
  - "joint audio-video generation"
  - "diffusion transformer"
  - "lip-audio synchronization"
  - "voice cloning"
date: 2026-05-08
content_hash: ae3f8b50ba367aef
---

# UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation

**Conference**: CVPR 2026 Findings  
**arXiv**: [2603.01418](https://arxiv.org/abs/2603.01418)  
**Code**: Unavailable  
**Area**: Video Generation
**Keywords**: Talking portrait generation, joint audio-video generation, diffusion transformer, lip-audio synchronization, voice cloning

## TL;DR
UniTalking is proposed as an end-to-end talking portrait generation framework built upon MM-DiT. Through a joint attention mechanism within a dual-stream symmetric architecture, it explicitly models fine-grained temporal correspondences between audio and video tokens, achieving state-of-the-art lip-audio synchronization accuracy while supporting personalized voice cloning.

## Background & Motivation
Talking portrait generation requires simultaneously producing visually realistic video with precisely synchronized lip movements and natural speech — in the real world, audio and video constitute a synchronous, inseparable perceptual whole.

**Key Challenge**:

**Closed-source vs. Open-source**: Closed-source models such as Veo3 and Sora2 have demonstrated remarkable audio-video consistency, yet their architectures and training procedures remain entirely inaccessible, precluding academic reproduction.

**Cascaded vs. End-to-end**: Open-source approaches fall into two categories — cascaded methods (generating audio first, then driving video) suffer from temporal misalignment and error accumulation; end-to-end methods currently address primarily Foley sound synchronization (e.g., wave sounds), falling far short of the precision required for speech-level lip-audio synchronization.

**Limitations of Prior End-to-End Methods**:
- JavisDiT: dual-branch DiT with bidirectional cross-attention, but insufficient interaction depth between audio and video branches.
- Universe-1: concatenates pretrained unimodal models, but inter-model alignment remains insufficiently fine-grained.
- OVI/OmniTalker: dual-stream with dedicated fusion blocks, but not optimized for the talking portrait scenario.

**Key Insight**: The paper designs a symmetric dual-stream architecture (video stream inheriting Wan2.2 pretrained weights; audio stream as an "identical twin"), with the core innovation being joint attention within the Multi-Modal Transformer Block to directly model temporal correspondences between audio and video tokens. A multi-task training strategy (T2AV + TV2A + TI2AV + TR2AV) further constrains audio-video alignment from multiple perspectives.

## Method

### Overall Architecture
Built on the MM-DiT architecture with 10B total parameters. Training employs continuous normalizing flow (Flow Matching) with CFG-guided inference.
- Video stream: inherits Wan2.2-5B architecture and weights.
- Audio stream: architecturally symmetric to the video stream ("identical twin"), randomly initialized.
- $N=30$ MM-DiT Blocks, $\text{dim}=3072$, 24 heads.

### Key Designs

1. **Latent Representation**:

    - Video: encoded by Wan2.2's 3D causal VAE with spatiotemporal compression of $16\times16\times4$.
    - Audio: MMAudio's 1D VAE encodes mel spectrograms into latent tokens; at inference, these are decoded back to mel spectrograms and synthesized into 44.1 kHz waveforms via the BigVGAN vocoder.
    - Conditioning: UMT5 encodes text (fixed 512 tokens); MMAudio VAE encodes reference audio (fixed 257 tokens).
    - All encoders are frozen, providing stable and semantically rich latent spaces.

2. **Multi-Modal Transformer Block (Core Innovation)**:

    - Function: explicitly models inter-modal dependencies between audio and video tokens within each DiT block.
    - Three key modifications:
    - **(a) Joint Attention**: replaces standard self-attention — video and audio latent tokens are concatenated and processed through a **single unified attention operation**, compelling the model to learn both intra-modal and inter-modal dependencies.
        - Key distinction from cross-attention: joint attention enables audio and video tokens to interact directly within the same attention matrix, rather than being indirectly associated through separate KV projections.
    - **(b) Dual-Condition Cross-Attention**: adds KV projection layers for reference audio; the main token stream attends separately to text conditioning and reference audio conditioning, with outputs summed element-wise.
        - Ensures generation remains faithful to both textual semantics and reference audio style.
    - **(c) Anisotropic RoPE**: standard RoPE is applied along the temporal axis; fixed-position RoPE is applied to the spatial dimensions of audio tokens.
        - Design Motivation: compels the model to prioritize temporal dynamics, reduces interference from spatial dimensions, and enhances temporal alignment between audio and video.

3. **Progressive Two-Stage Training Strategy**:

    - Rationale: the video branch carries strong pretrained weights while the audio branch is randomly initialized — an inherent initialization imbalance.
    - **Stage 1 — Audio Branch Pretraining (TTS)**:
        - Trains only audio-related parameters (audio input projection, audio branch FFN and attention projections).
        - All video and text branch parameters are frozen.
        - Batch size 256, LR $1 \times 10^{-5}$, 100K steps.
        - Jointly trains TTS tasks with and without reference audio.
        - Finding: fine-tuning only these parameters is sufficient to generate high-quality speech.
    - **Stage 2 — Multi-Task Joint Audio-Video Training**:
        - End-to-end full-model training.
        - Batch size 64, LR $1 \times 10^{-5}$, 100K steps.
        - Four tasks trained in rotation:
       - **T2AV (Text → Audio-Video)**: the core joint generation task, establishing coarse-grained alignment.
       - **TV2A (Video → Audio)**: attention masking prevents audio from influencing the video branch, providing strict unidirectional temporal alignment supervision — compelling the audio branch to learn frame-level viseme-to-phoneme mappings; this constitutes **the most critical training signal for lip-audio synchronization**.
       - **TI2AV (Image + Text → Audio-Video)**: supports identity-preserving personalized generation.
       - **TR2AV (Reference Audio + Text → Audio-Video)**: supports voice style cloning.
        - Multi-task rotation creates multi-faceted constraints that prevent the model from exploiting trivial shortcuts.

### Loss & Training
- Flow Matching conditional flow objective: $\mathcal{L}_{CFM} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$
- Audio-only tasks: $L_{total} = L_{CFM}^a$
- Joint tasks: $L_{total} = L_{CFM}^a + L_{CFM}^v$
- CFG guidance scale $\omega > 1$ controls conditioning strength; conditions include combinations of $c_{text}$, $c_{image}$, and $c_{audio}$.
- AdamW optimizer, $\beta_1=0.9$, $\beta_2=0.999$, bf16 precision, FSDP parallelism.

### Data Preparation
- Starting from OpenHumanVid and internal data, processed through three-stage filtering (video → audio → audio-video cross-modal).
- Three-level annotation: detailed video + audio descriptions, brief video + audio descriptions, and fused audio-video descriptions (generated by Qwen3-Omni).
- Reference audio generation: IndexTTS2 synthesizes 3 reference audio clips per video.
- Final dataset: 2.3 million aligned audio-video samples.

## Key Experimental Results

### T2AV Joint Generation — Blind Preference Study

| Dimension | UniTalking vs. OVI | UniTalking vs. Universe-1 |
|------|-------------------|-------------------------|
| Video Quality | ~100% (on par) | Advantage |
| Audio Quality | 116% | Advantage |
| Audio-Video Sync | 107% | Advantage |

### Lip-Audio Synchronization — Objective Evaluation

| Method | Sync-C↑ | Sync-D↓ |
|------|---------|---------|
| Universe-1 | 1.85 | 11.97 |
| OVI | 6.56 | 8.60 |
| Sora2 | 5.35 | 7.78 |
| **UniTalking** | **4.87** | **8.05** |

### Speaker Similarity (TR2AV)

| Method | English↑ | Chinese↑ |
|------|-------|-------|
| ElevenLabs | 0.613 | 0.677 |
| MiniMax | 0.756 | 0.780 |
| Qwen3-Omni | 0.773 | 0.772 |
| **UniTalking** | **0.703** | **0.662** |

### TTS Evaluation

| Method | WER↓ |
|------|------|
| Fish Speech | 0.008 |
| F5-TTS | 0.018 |
| OVI-Aud | 0.035 |
| **UniTalking** | **0.038** |

### Key Findings
- Audio quality and audio-video synchronization surpass OVI (116% and 107%, respectively); video quality is on par (both leverage Wan2.2 pretraining).
- Sync-D of 8.05 approaches the closed-source Sora2 (7.78), substantially outperforming Universe-1 (11.97).
- OVI's anomalously high Sync-C (6.56) is attributed by the authors to a potential bias in the metric toward exaggerated mouth motion.
- Voice cloning performance is comparable to ElevenLabs, though it falls short of dedicated large models (MiniMax, Qwen3-Omni).
- Stage 1 TTS pretraining is shown to be critical for audio quality in the final audio-video generation — omitting it leads to significant degradation.
- Attention visualizations confirm learned cross-modal associations: audio-to-video attention focuses on the face and body, while video-to-audio attention **focuses exclusively on the lips**.

## Highlights & Insights
- The **symmetric dual-stream design** is intuitively elegant: the audio stream serves as an "identical twin" of the video stream, leveraging architectural symmetry to facilitate latent space fusion.
- The contribution of the **TV2A task** to lip-audio synchronization is a central insight — unidirectional supervision compels the audio branch to learn precise viseme-to-phoneme mappings.
- The design choice of **joint attention** (unified attention over concatenated tokens) versus cross-attention merits attention — the former enables more direct inter-modal information flow.
- Attention visualizations (audio attending to the face, video's audio attention attending exclusively to the lips) provide strong evidence that the model has learned meaningful alignment.

## Limitations & Future Work
- Gaps with closed-source models remain due to constraints in training resources and data scale.
- Multi-identity reference generation (e.g., Sora2's "Cameo" feature) is not supported.
- Voice cloning capability is weaker than dedicated speech models (MiniMax, Qwen3-Omni), possibly because joint training disperses the audio branch's capacity.
- The inference cost of 10B parameters is substantial; optimization is required for real-time deployment.
- Partially mismatched highlight regions are present in attention maps, tentatively attributed to training strategy or data noise.

## Related Work & Insights
- Hallo3, HunyuanVideo-Avatar: audio-driven video generation (unidirectional); UniTalking performs joint generation.
- MMAudio: MM-DiT approach for V2A; the present work extends it to the talking portrait scenario.
- OVI: the most direct competitor, also employing a dual-stream architecture, but using cross-attention rather than joint attention.
- Insight: in joint generation, the TV2A (video-to-audio) task as an auxiliary training objective is an underappreciated design choice.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of joint attention, symmetric dual-stream architecture, and multi-task constraint strategy represents clear technical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage via blind evaluation, objective metrics, ablation studies, and attention visualizations; comparisons against Sora2 strengthen persuasiveness.
- Writing Quality: ⭐⭐⭐⭐ Technical descriptions are clear; documentation of the data pipeline and training strategy is thorough.
- Value: ⭐⭐⭐⭐⭐ An important milestone for open-source unified audio-video generation, substantially closing the gap with closed-source models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)
- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](../../ICML2026/video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)

</div>

<!-- RELATED:END -->
