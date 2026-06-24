---
title: >-
  [Paper Note] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation
description: >-
  [CVPR 2025][Audio & Speech][Joint Video-Text Audio Generation] Proposes VinTAGe, the first audio generation model joint-conditioned on video and text. It balances visual and textual guidance via learnable layer weights and mitigates modality bias through a teacher-student framework, achieving comprehensive state-of-the-art performance on both on-screen and off-screen audio generation (FAD 3.05, MOS 3.36).
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Joint Video-Text Audio Generation"
  - "Flow Transformer"
  - "Modality Bias"
  - "Teacher Guidance"
  - "Off-screen Sound"
date: 2026-05-08
content_hash: 803648a21f475cf7
---

# VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.10768](https://arxiv.org/abs/2412.10768)  
**Code**: To be released  
**Area**: Audio & Speech / Generative Models  
**Keywords**: Joint Video-Text Audio Generation, Flow Transformer, Modality Bias, Teacher Guidance, Off-screen Sound

## TL;DR

Proposes VinTAGe, the first audio generation model joint-conditioned on video and text. It balances visual and textual guidance via learnable layer weights and mitigates modality bias through a teacher-student framework, achieving comprehensive state-of-the-art performance on both on-screen and off-screen audio generation (FAD 3.05, MOS 3.36).

## Background & Motivation

**Background**: Existing audio generation is split into V2A (video-to-audio) and T2A (text-to-audio), each utilizing only a single modality condition. Real-world video audio contains both on-screen sounds corresponding to the visual frames (e.g., footsteps) and off-screen sounds described by text (e.g., background music).

**Limitations of Prior Work**: V2A models can only generate sounds corresponding to visual frames (ignoring off-screen audio), while T2A models lack visual temporal alignment. Simple concatenation of both conditions leads to modality bias—the model heavily relies on one modality while ignoring the other (e.g., FoleyCrafter scores 64.9% on-screen vs. 21.7% off-screen, exhibiting severe visual bias).

**Key Challenge**: Visual and textual conditions contribute to audio in different ways—visuals provide temporal synchronization cues, while text provides semantic category signals. Simple fusion fails to balance the two.

**Key Insight**: Utilize pre-trained V2A and T2A teacher models to provide single-modality guidance respectively, and learn adjustable layer weights to control the contribution of each modality within every layer.

**Core Idea**: Flow Transformer + Learnable Layer Weights + Teacher-Guided De-biasing = Balanced On-/Off-screen Joint Audio Generation.

## Method

### Key Designs

1. **VT-Encoder (Joint Video-Text Encoder)**:

    - Function: Fuses video and text conditions into a unified conditioning representation.
    - Mechanism: CLIP encodes video frames + optical flow/positional embeddings, FLAN-T5 encodes text. The two are fused via gated cross-attention.
    - Design Motivation: Optical flow embeddings capture motion signals, while positional embeddings provide temporal cues.

2. **Joint VT-SiT (Flow Transformer with Learnable Layer Weights)**:

    - Function: Balances the contributions of visual and text conditions layer by layer.
    - Mechanism: Extends SiT (Scalable Interpolant Transformers), introducing a learnable weight $\omega_l$ at each layer to control the blending ratio of text/visual conditions.
    - Design Motivation: Different layers may require different modality dependencies—shallower layers might rely more on visuals (timing), while deeper layers rely more on text (semantics).

3. **Teacher-Student De-biasing Framework**:

    - Function: Prevents the joint model from degenerating into relying on only a single modality.
    - Mechanism: Pre-trains independent V2A and T2A teacher models. During training, the joint model's predictions are aligned with predictions from both teachers: $\mathcal{L} = \mathcal{L}_{main} + \lambda_v \mathcal{L}_v + \lambda_t \mathcal{L}_t$.
    - Design Motivation: VinTAGe achieves 57.7% on-screen / 43.6% off-screen, which is far more balanced than FoleyCrafter's 64.9% / 21.7%.

### Loss & Training

$\mathcal{L} = \|v_\theta - (\dot\alpha_t x - \dot\beta_t \epsilon)\|^2 + \lambda_v \mathcal{L}_v + \lambda_t \mathcal{L}_t$, classifier-free guidance $s_{vis} = s_{txt} = 2.5$.

## Key Experimental Results

### Main Results

VinTAGe-Bench (636 pairs):

| Metric | VinTAGe | FoleyCrafter | Tango2 |
|------|---------|-------------|--------|
| FAD↓ | **3.05** | 3.81 | 8.01 |
| AV Align. | **22.29** | 21.45 | - |
| MOS Quality | **3.36** | 3.02 | 2.71 |
| On-screen/Off-screen | 57.7/43.6 | 64.9/21.7 | - |

### Key Findings
- **Modality balance is a key differentiator**: The on-screen/off-screen ratio of VinTAGe (57.7/43.6) is much more balanced than FoleyCrafter's (64.9/21.7).
- **Teacher guidance is effective**: Without teacher guidance, the model tends to ignore textual conditions.

## Highlights & Insights
- **First to define the joint video-text audio generation task**—accompanied by VinTAGe-Bench and on-screen/off-screen evaluation protocols.
- **Generalizability of teacher de-biasing**—this framework of using single-modal teachers to guide a multi-modal student can be applied to any multi-condition generation task.

## Limitations & Future Work
- VinTAGe-Bench is small (636 pairs), leading to limited evaluation.
- The quality of off-screen sound depends on the text descriptions generated by the LLM.
- Requires pre-training two teacher models, which incurs high training costs.

## Rating
- Novelty: ⭐⭐⭐⭐ First joint video-text audio generation
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark + subjective evaluation + ablation
- Writing Quality: ⭐⭐⭐⭐ Clear
- Value: ⭐⭐⭐⭐ Opens up a new direction of multi-modal conditioning for the audio generation community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)
- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)

</div>

<!-- RELATED:END -->
