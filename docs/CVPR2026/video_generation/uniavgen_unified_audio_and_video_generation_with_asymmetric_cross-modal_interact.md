---
title: >-
  [Paper Note] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions
description: >-
  [CVPR 2026][Video Generation][joint audio-video generation] UniAVGen proposes a joint audio-video generation framework built on a symmetric dual-branch DiT…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "joint audio-video generation"
  - "cross-modal interaction"
  - "diffusion models"
  - "lip-audio synchronization"
  - "face-aware modulation"
date: 2026-05-08
content_hash: 5bf4263ad0c21406
---

# UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions

**Conference**: CVPR 2026
**arXiv**: [2511.03334](https://arxiv.org/abs/2511.03334)
**Code**: [https://mcg-nju.github.io/UniAVGen/](https://mcg-nju.github.io/UniAVGen/) (project page)
**Area**: Video Generation
**Keywords**: joint audio-video generation, cross-modal interaction, diffusion models, lip-audio synchronization, face-aware modulation

## TL;DR
UniAVGen proposes a joint audio-video generation framework built on a symmetric dual-branch DiT, achieving precise spatiotemporal synchronization through an **asymmetric cross-modal interaction mechanism** and a **Face-Aware Modulation module**. With only 1.3M training samples, it outperforms competitors trained on 30M data across lip-audio synchronization, timbre consistency, and emotional consistency.

## Background & Motivation

1. **Background**: Joint audio-video generation is an important direction in generative AI. Commercial systems (Veo3, Sora2, Wan2.5) have demonstrated impressive results, but open-source methods still predominantly rely on decoupled two-stage pipelines—generating silent video first and then dubbing, or generating audio first and then driving video synthesis.

2. **Limitations of Prior Work**: The fundamental problem with two-stage methods lies in **modal decoupling**—audio and video cannot interact during generation, resulting in poor semantic consistency, weak emotional alignment, and imprecise lip-audio synchronization. Existing end-to-end joint generation methods (JavisDiT, UniVerse-1, Ovi) attempt to address this issue but either support only ambient sound without human speech, or achieve limited cross-modal alignment.

3. **Key Challenge**: Audio and video exhibit a natural **asymmetry** in temporal granularity and semantic space—each video latent frame corresponds to multiple audio tokens, and vice versa. Existing methods ignore this asymmetry, resorting either to global interaction (slow convergence) or symmetric temporal alignment interaction (insufficient context utilization).

4. **Goal**: (a) How to design cross-modal interactions that converge quickly while maintaining strong performance; (b) how to focus interactions on key regions such as the face; (c) how to enhance cross-modal correlation signals at inference time.

5. **Key Insight**: Lip movements in video are influenced by surrounding phonemes, while audio needs to perceive more precise temporal position information from video—the requirements in each direction differ fundamentally, motivating an asymmetric design.

6. **Core Idea**: Employ modality-aware asymmetric cross-modal attention, face-aware soft masking, and modality-aware CFG to achieve state-of-the-art audio-video synchronized generation with far less training data than competing methods.

## Method

### Overall Architecture
UniAVGen adopts a **symmetric dual-branch joint synthesis architecture**: the video branch uses the Wan 2.2-5B DiT backbone, and the audio branch uses the architectural template of Wan 2.1-1.3B (same structure, differing only in channel dimensions). Inputs include a reference speaker image, a video description text, and speech text content, with optional reference audio and conditional audio-video. Both branches are trained under the Flow Matching paradigm, each predicting its own velocity field.

Video branch: Video is processed at 16 fps and encoded by a VAE into latent $z^v$; latents of the reference image and conditional video are concatenated as input; text is encoded by umT5 and injected via cross-attention.

Audio branch: Audio sampled at 24 kHz is converted to a Mel spectrogram as latent $z^a$; reference and conditional audio are similarly concatenated as input; speech text features are extracted by ConvNeXt blocks and injected accordingly.

### Key Designs

1. **Asymmetric Cross-Modal Interaction**:

    - **Function**: Enables bidirectional, temporally aligned cross-modal attention, balancing convergence speed and performance.
    - **Mechanism**: Comprises two modality-specific aligners. The **A2V aligner** constructs an audio context window $C_i^a$ for each video frame (containing audio tokens from $w$ preceding and following frames) and performs per-frame cross-attention, allowing video to perceive the semantic information of neighboring audio. The **V2A aligner** adopts a **temporal neighborhood interpolation strategy**: for each audio token $j$, it weights the two adjacent video frames according to the relative position $\alpha = (j \bmod k)/k$ to obtain a smoothly interpolated video context $C_j^v$, followed by cross-attention. All output matrices are zero-initialized to prevent disrupting each modality's generative capacity early in training.
    - **Design Motivation**: Lip movements in video are affected by surrounding phonemes (requiring windowed context), while audio needs to perceive precise continuous temporal positions (requiring interpolation)—the two directions have inherently asymmetric requirements. Compared to global interaction (slow convergence) and symmetric temporal alignment interaction (limited context), the asymmetric design achieves the best balance between the two.

2. **Face-Aware Modulation (FAM)**:

    - **Function**: Dynamically guides cross-modal interactions to focus on salient regions such as the face.
    - **Mechanism**: A lightweight mask prediction head is introduced at each interaction layer, applying LayerNorm, an affine transformation, a linear projection, and Sigmoid to the video features $H^{v_l}$ to produce a soft mask $M^l \in (0,1)^{T \times N_v}$. In the A2V direction, the mask performs selective updating: $H^{v_l} = H^{v_l} + M^l \odot \bar{H}^{v_l}$; in the V2A direction, the mask amplifies the information transfer from salient video regions: $\hat{H}^{v_l} = M^l \odot \hat{H}^{v_l}$. The mask is supervised by ground-truth face masks, with loss weight $\lambda^m$ linearly decayed from 0.1 to 0.
    - **Design Motivation**: The key semantic coupling between audio and video in human-centric content is concentrated in the face. Constraining the interaction range in early training accelerates convergence and avoids background interference; gradually relaxing the constraint in later stages allows the model to learn more flexible interaction patterns (via $\lambda^m$ decay). Experiments confirm that the decay strategy outperforms a fixed weight on timbre and emotional consistency.

3. **Modality-Aware CFG (MA-CFG)**:

    - **Function**: Explicitly enhances cross-modal correlation signals at inference time.
    - **Mechanism**: Conventional CFG operates within a single modality and cannot enhance cross-modal dependencies. The core insight of MA-CFG is to obtain unconditional estimates $u_{\theta_v}$ and $u_{\theta_a}$ via a single forward pass with cross-modal interaction signals removed (equivalent to single-modality inference), then compute guidance using the estimate with cross-modal interaction $u_{\theta_{a,v}}$: $\hat{u}_v = u_{\theta_v} + s_v(u_{\theta_{a,v}} - u_{\theta_v})$.
    - **Design Motivation**: Conventional CFG only guides text conditioning, ignoring signals from audio driving video or video influencing audio. MA-CFG significantly enhances emotional intensity and motion dynamics.

### Loss & Training

Training proceeds in three stages: Stage 1 trains only the audio branch ($\mathcal{L}^a$, 160k steps, batch=256); Stage 2 jointly trains both branches ($\mathcal{L}^{joint} = \mathcal{L}^v + \mathcal{L}^a + \lambda_m \mathcal{L}^m$, 30k steps, batch=32, lr=5e-6); Stage 3 applies multi-task learning (5 task types at ratio 4:1:1:2:2, 10k steps). $\lambda_m$ linearly decays from 0.1 to 0.

## Key Experimental Results

### Main Results

| Method | Joint Training | Training Samples | PQ↑ | CU↑ | WER↓ | SC↑ | DD↑ | LS↑ | TC↑ | EC↑ |
|--------|---------------|-----------------|-----|-----|------|-----|-----|-----|-----|-----|
| OmniAvatar (two-stage) | ✗ | 21.1B | 8.15 | 7.41 | 0.152 | 0.987 | 0.000 | 6.34 | 0.454 | 0.349 |
| Ovi (joint) | ✓ | 30.7M | 6.03 | 6.01 | 0.216 | 0.972 | 0.360 | 6.48 | 0.828 | 0.558 |
| **UniAVGen** | ✓ | **1.3M** | **7.00** | **6.62** | **0.151** | 0.973 | **0.410** | 5.95 | **0.832** | **0.573** |

UniAVGen surpasses Ovi using 23× less data (30.7M vs. 1.3M), achieving comprehensive improvements in audio quality and audio-video consistency.

### Ablation Study

| Interaction Design (A2V / V2A) | LS↑ | TC↑ | EC↑ |
|-------------------------------|-----|-----|-----|
| SGI / SGI (global) | 3.46 | 0.667 | 0.459 |
| STI / STI (symmetric temporal) | 3.73 | 0.685 | 0.472 |
| **ATI / ATI (asymmetric)** | **4.09** | **0.725** | **0.504** |

| FAM Configuration | LS↑ | TC↑ | EC↑ |
|------------------|-----|-----|-----|
| No FAM | 3.89 | 0.705 | 0.489 |
| Unsupervised FAM | 3.92 | 0.701 | 0.492 |
| Fixed $\lambda_m$ | 4.11 | 0.719 | 0.497 |
| **Decayed $\lambda_m$** | **4.09** | **0.725** | **0.504** |

### Key Findings
- **Asymmetric interaction contributes most**: ATI significantly outperforms SGI and STI across all metrics, validating the necessity of modality-specific design.
- **Supervision signal for FAM is critical**: Supervised FAM substantially improves consistency over unsupervised FAM, demonstrating that constraining the mask to facial regions effectively accelerates training convergence.
- **Decay strategy outperforms fixed weight**: Progressively relaxing the constraint allows the model to learn more flexible interactions, further improving TC and EC.
- **Multi-task training enhances joint generation**: Joint training followed by multi-task learning (JFML) achieves the best results; starting multi-task training from the beginning (MTO) leads to slower convergence.
- On out-of-distribution animated images, UniAVGen demonstrates strong generalization, whereas Ovi produces lip movement failures and UniVerse-1 generates nearly static outputs.

## Highlights & Insights
- **Elegantly precise asymmetric design**: The A2V aligner uses windowed context to account for the influence of surrounding phonemes, while the V2A aligner uses temporal interpolation to perceive continuous video positions—perfectly matching the distinct requirements of each direction.
- **Progressive relaxation strategy in FAM**: Using a decaying supervision signal to constrain early training and release constraints later is an elegant approach that balances training efficiency with model flexibility, and is transferable to other multimodal tasks requiring region-focused interactions.
- **MA-CFG generalizes CFG to the cross-modal setting**: The idea is concise (using single-modality inference as the unconditional baseline) yet yields significant gains, and can be directly applied to any dual-modality generation system.

## Limitations & Future Work
- The method focuses exclusively on **human-centric** audio-video generation and does not cover general scenes (ambient sounds, music, etc.).
- The audio branch supports **English speech only**; multilingual capability has not been validated.
- Video duration is limited (training data presumably consists of short clips); maintaining consistency in long videos has not been explored.
- TC and EC evaluations rely on Gemini-2.5-Pro scoring, lacking standardized open-source evaluation protocols.

## Related Work & Insights
- **vs. Ovi**: Both adopt symmetric dual-tower architectures, but Ovi uses symmetric global interaction without modality-specific design, resulting in poor OOD generalization; UniAVGen surpasses it with 23× less data through asymmetric interaction and FAM.
- **vs. UniVerse-1**: Concatenates two pretrained models; architectural asymmetry leads to integration complexity and limited performance; UniAVGen unifies the architecture from the ground up.
- **vs. two-stage methods**: Two-stage methods achieve good lip-audio synchronization but near-zero dynamics (DD≈0), indicating that video generation proceeds entirely without awareness of audio.

## Rating
- Novelty: ⭐⭐⭐⭐ The asymmetric interaction and FAM decay strategy are novel; MA-CFG is a natural extension of CFG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Main experiments + 5 ablation groups + multi-task analysis + OOD qualitative comparison, though some evaluation metrics rely on closed-source models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, rich figures and tables, and strongly motivated derivations.
- Value: ⭐⭐⭐⭐ State-of-the-art open-source joint audio-video generation with exceptional data efficiency, though limited to human-centric scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[ICLR 2026\] BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](../../ICLR2026/video_generation/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)
- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)

</div>

<!-- RELATED:END -->
