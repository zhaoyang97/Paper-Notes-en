---
title: >-
  [Paper Note] R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching
description: >-
  [ACL 2025][Image Generation][voice conversion] R-VC is the first zero-shot voice conversion system to achieve rhythm control. It models the target speaker's rhythm style using a Mask Transformer duration model, combined with a Shortcut Flow Matching DiT decoder to achieve efficient and high-quality speech generation in only 2 sampling steps, achieving a WER of 3.51 and speaker similarity of 0.930 on LibriSpeech.
tags:
  - "ACL 2025"
  - "Image Generation"
  - "voice conversion"
  - "flow matching"
  - "rhythm control"
  - "DiT"
  - "duration modeling"
date: 2026-05-08
content_hash: 8b4c4a7d7ef24115
---

# R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching

**Conference**: ACL 2025  
**arXiv**: [2506.01014](https://arxiv.org/abs/2506.01014)  
**Code**: [https://r-vc929.github.io/r-vc/](https://r-vc929.github.io/r-vc/)  
**Area**: Image Generation  
**Keywords**: voice conversion, flow matching, rhythm control, DiT, duration modeling

## TL;DR
R-VC is the first zero-shot voice conversion system to achieve rhythm control. It models the target speaker's rhythm style using a Mask Transformer duration model, combined with a Shortcut Flow Matching DiT decoder to achieve efficient and high-quality speech generation in only 2 sampling steps, achieving a WER of 3.51 and speaker similarity of 0.930 on LibriSpeech.

## Background & Motivation
**Background**: Zero-shot voice conversion (VC) aims to convert the speaker's timbre while preserving the linguistic content. Mainstream methods (HierSpeech++, CosyVoice, etc.) focus primarily on preserving the prosody of the source speech.

**Limitations of Prior Work**:
   - Preserving the source prosody can cause speaker timbre information to leak through prosodic coupling.
   - It is impossible to transfer the target speaker's rhythm/speaking-rate style to the synthesized speech.
   - Diffusion/Flow Matching methods require $\ge 10$ sampling steps, leading to high inference latency.

**Key Challenge**: High-quality speech generation requires multi-step sampling, but real-time applications demand low latency.

**Core Idea**: Model token-level duration to achieve rhythm control using a Mask Transformer, and reduce the sampling steps to 2 via Shortcut Flow Matching.

## Method

### Overall Architecture
R-VC consists of three modules: (1) **Linguistic Content Representation**: HuBERT + K-Means discretization + deduplication to extract token-level durations; (2) **Mask Transformer Duration Model**: predicts token durations style-matched to the target speaker using non-autoregressive iterative decoding; (3) **Shortcut Flow Matching DiT Decoder**: a 22-layer DiT (300M params) conditioned on step size $d$ to enable 2-step generation.

### Key Designs

1. **Content Representation and Duration Extraction**:

    - Function: Eliminate speaker identity from the content representation and extract token-level duration.
    - Mechanism: Apply data perturbations (formant shifting, pitch randomization, parametric EQ) to the input speech, extract discrete tokens using HuBERT + K-Means, and then deduplicate them to obtain the content sequence and corresponding durations. For example: $[u_1, u_1, u_1, u_2, u_3, u_3] \rightarrow [u_1, u_2, u_3]$ with durations $[3, 1, 2]$.
    - Design Motivation: Deduplication eliminates the prosodic patterns and retains only the pure linguistic content, laying the foundation for independent rhythm modeling.

2. **Mask Transformer Duration Model**:

    - Function: Non-autoregressively predict the duration of each content token, conditioned on the target speaker.
    - Mechanism: Perform iterative decoding using mask-predict (with a sinusoidal schedule $p = \sin(u)$) on input composed of deduplicated content tokens + partially unmasked durations + global speaker embeddings, trained using a cross-entropy loss.
    - Design Motivation: Token-level duration modeling is more fine-grained than sentence-level modeling, allowing it to capture target-speaker rhythm characteristics (such as speaking rate and pause patterns).

3. **Shortcut Flow Matching DiT Decoder**:

    - Function: Generate mel-spectrograms from noise in only 2 steps (NFE=2).
    - Mechanism: $x_{t+d}' = x_t + s(x_t, t, d) \cdot d$, while optimization simultaneously targets OT-CFM and self-consistency: $$L_{S-CFM} = \mathbb{E}[\|s_\theta(x_t,t,0) - (x_1-x_0)\|^2 + \|s_\theta(x_t,t,2d) - s_{target}\|^2]$$. It employs hybrid training with 30% self-consistency and 70% flow matching.
    - Design Motivation: While standard CFM requires $\ge 10$ steps, the shortcut strategy encourages the model to learn larger step jumps, allowing 2-step inference to achieve quality close to 10-step generation.

## Key Experimental Results

### Main Results (LibriSpeech test-clean, 2620 samples)

| Method | WER↓ | SECS↑ | UTMOS↑ | RTF↓ |
|------|------|-------|--------|------|
| FACodec | 4.68 | 0.908 | 3.94 | - |
| CosyVoice | 5.95 | 0.933 | 4.09 | - |
| HierSpeech++ | 1.46(CER) | 0.907 | 4.09 | - |
| **R-VC (NFE=2)** | **3.51** | **0.930** | **4.10** | **0.12** |

### Efficiency and Quality

| NFE | WER | SECS | UTMOS | RTF |
|-----|-----|------|-------|-----|
| 2 (R-VC) | 3.51 | 0.930 | 4.10 | 0.12 |
| 10 (vanilla CFM) | ~similar | ~similar | ~similar | 0.34 |
| Speed Improvement | - | - | - | **2.83×** |

### Ablation Study

| Configuration | WER | SECS | EMO score |
|------|-----|------|-----------|
| Full R-VC | 6.95 | 0.880 | 0.590 |
| w/o duration model | 7.03 | 0.878 | **0.425** (-0.165) |
| w/o spk embedding (decoder) | 8.24 | 0.873 | 0.477 |
| w/o perturbation | 7.28 | 0.869 | 0.580 |
| w/ sentence-level duration | 9.86 | 0.872 | 0.528 |

### Key Findings
- **The duration model is crucial for emotion transfer**: Removing it degrades the EMO score from 0.590 to 0.425.
- **Token-level > Sentence-level duration**: Sentence-level duration causes the WER to surge to 9.86.
- **NFE=2 achieves similar quality to NFE=10**: This validates the effectiveness of Shortcut Flow Matching.
- **Rhythm classification accuracy of 90.2%**: Precise control across three rhythm levels (slow, normal, fast) is achieved.

## Highlights & Insights
- **First to achieve rhythm control in zero-shot VC**, filling a gap in the voice conversion domain. This ability is highly valuable for applications such as personalized TTS and dubbing.
- **Shortcut Flow Matching** elegantly integrates the concept of consistency distillation into flow matching. Simultaneously learning continuous and jumping joint targets during training makes it simpler than standalone distillation methods.
- **Data perturbation + deduplication** pipeline is a valuable content extraction strategy: perturbation removes speaker characteristics, while deduplication removes prosodic patterns. This two-step process decouples content, timbre, and prosody.

## Limitations & Future Work
- Fine-grained duration prediction is unstable and can sometimes cause over-prolonged, abnormal pronunciations.
- Trained exclusively on English data; cross-lingual capabilities remain unexplored.
- Single-step generation (NFE=1) has not been investigated.
- Trained on only 20K hours of data (vs. CosyVoice's 171K). While it achieves comparable performance, there is likely still room for improvement.

## Related Work & Insights
- **vs CosyVoice**: CosyVoice uses 171K hours of data, whereas R-VC uses only 20K but achieves comparable performance; R-VC additionally supports rhythm control.
- **vs HierSpeech++**: HierSpeech++ performs better on CER, but R-VC outperforms it in MOS and emotion transfer.
- **vs Diff-HierVC**: R-VC outperforms Diff-HierVC in both WER and emotion scores.

## Rating
- Novelty: ⭐⭐⭐⭐ The first integration of rhythm control and efficient inference, filling a gap in the VC domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three dimensions (VC, emotion-transfer, and rhythm control), including MOS evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and well-designed ablation study.
- Value: ⭐⭐⭐⭐ Practical value for speech synthesis and personalization scenarios.
- Value: ⭐⭐⭐⭐ Practical voice conversion improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)
- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[CVPR 2026\] ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control](../../CVPR2026/image_generation/projflow_projection_sampling_with_flow_matching_for_zero-shot_exact_spatial_moti.md)
- [\[NeurIPS 2025\] Scaling Offline RL via Efficient and Expressive Shortcut Models](../../NeurIPS2025/image_generation/scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
