---
title: >-
  [Paper Note] Resounding Acoustic Fields with Reciprocity
description: >-
  [NeurIPS 2025][Audio & Speech][acoustic field learning] Leveraging the reciprocity principle of acoustic wave propagation, this paper proposes Versa (ELE data augmentation + SSL self-supervised learning)…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "acoustic field learning"
  - "reciprocity"
  - "impulse response"
  - "data augmentation"
  - "self-supervised learning"
date: 2026-05-08
content_hash: a7cab1684523f836
---

# Resounding Acoustic Fields with Reciprocity

**Conference**: NeurIPS 2025
**arXiv**: [2510.20602](https://arxiv.org/abs/2510.20602)
**Code**: [Available](https://waves.seas.upenn.edu/projects/versa)
**Area**: Audio/Speech / Acoustic Modeling
**Keywords**: acoustic field learning, reciprocity, impulse response, data augmentation, self-supervised learning

## TL;DR
Leveraging the reciprocity principle of acoustic wave propagation, this paper proposes Versa (ELE data augmentation + SSL self-supervised learning), which generates physically valid virtual training samples by swapping emitter and receiver roles, substantially improving acoustic field estimation performance under sparse emitter configurations.

## Background & Motivation

- **Background**: Immersive AR/VR experiences require modeling acoustic fields at arbitrary emitter positions, yet data collection faces a fundamental asymmetry: microphones (receivers) can be deployed densely at low cost, while loudspeakers (emitters) are difficult to install in large numbers due to their size and power consumption. Existing neural acoustic field methods either require dense deployment across hundreds of emitter positions, or rely on differentiable ray tracing with simplified geometry.
- **Limitations of Prior Work**: The scarcity of emitter positions severely limits acoustic field estimation accuracy.
- **Goal**: This paper introduces the *resounding* task — analogous to relighting in vision — estimating the acoustic field at arbitrary emitter positions from sparse observations at fewer than 10 emitter locations.

## Method

### Overall Architecture
Building on the acoustic reciprocity principle — reversing the source and receiver positions reverses the wave propagation path but leaves the accumulated propagation effect unchanged — the paper proposes two complementary strategies: Versa-ELE (data augmentation) and Versa-SSL (self-supervised learning).

### Key Designs

**Acoustic Reciprocity Theory**: For a single-path impulse response $h(t;\mathcal{P},\omega_e,\omega_l) = G_e(\omega_0;\omega_e)\Gamma(t;\mathcal{P})G_l(\omega_K;\omega_l)$, the path influence function $\Gamma(t;\mathcal{P})$ remains invariant under emitter/receiver exchange. When the emitter and receiver gain patterns are identical (omnidirectional or co-directional), the impulse response is fully preserved after the swap.

**Versa-ELE (Emitter–Listener Exchange)**: For each training sample $(p_e, p_l, \omega_e, \omega_l, h(t))$, a swapped sample $(p_l, p_e, \omega_l, \omega_e, h(t))$ is created. Dense microphone positions are thus converted into virtual emitter positions, effectively alleviating emitter sparsity. This is implemented as a simple, model-agnostic data augmentation.

**Versa-SSL (Self-Supervised Learning)**: When emitter and receiver gain patterns differ, direct exchange is invalid. The solution proceeds as follows: (1) exploit the AVR model's ability to separately control receiver gain patterns; (2) query the acoustic field model for the emitter gain pattern $G_e$; (3) replace the receiver gain with the emitter gain to make the two consistent; (4) enforce consistency between predictions before and after the swap as a self-supervised loss $\mathcal{L} = \mathcal{L}_a(h, h^*) + \lambda \mathcal{L}_{a\text{-ssl}}(h_1, h_2)$.

### Loss & Training
- **ELE**: Applied directly as data augmentation; no modification to the loss function is required.
- **SSL two-stage training**: Stage 1 fits the impulse response using supervised audio loss $\mathcal{L}_a$; Stage 2 extracts the emitter pattern, encodes it with spherical harmonic parameters, and adds the consistency self-supervised loss.
- Progressive noise injection is used to prevent the model from learning shortcuts.

## Key Experimental Results

### Main Results (Simulated Dataset, Same Gain Pattern — AcoustiX-Same)

| Method | Scene1-STFT | Scene1-C50 | Scene2-STFT | Scene2-C50 | Scene3-STFT | Scene3-C50 |
|------|------------|------------|------------|------------|------------|------------|
| NN | 2.87 | 2.84 | 3.54 | 10.71 | 3.29 | 7.42 |
| INRAS | 1.96 | 2.71 | 1.96 | 2.71 | 4.22 | 7.14 |
| NAF | 4.69 | 2.73 | - | - | - | - |
| INRAS+ELE | 1.36 | 1.72 | 1.81 | 1.98 | 1.67 | 2.79 |

### Reciprocity Verification (Real-World Data)

| Environment | Paired-Amp | Unpaired-Amp | Paired-C50 | Unpaired-C50 |
|------|---------|----------|---------|----------|
| Kitchen | 0.24 | 1.74 | 0.29 | 3.69 |
| Conference | 0.22 | 1.09 | 0.23 | 3.35 |
| Office | 0.23 | 1.54 | 0.18 | 2.49 |

### Key Findings
- Versa-ELE is model-agnostic and achieves an average improvement of 34% on C50 and 31% on STFT across existing neural acoustic field models.
- Versa-SSL further improves C50 by 24% and STFT by 48% on top of AVR.
- Real-world data confirms that reciprocity holds under paired conditions (errors are far smaller than in unpaired cases).
- In simulation, reciprocity improves with increasing ray count (paired error becomes negligible at 1,000k rays).

## Highlights & Insights
1. **Physics-driven ML methodology**: Acoustic reciprocity, a fundamental physical principle, is integrated into ML training rather than treated as black-box augmentation.
2. **Strong generality**: ELE serves as a plug-and-play, model-agnostic data augmentation compatible with any acoustic field model.
3. **Elegant handling of gain asymmetry**: SSL decouples and exchanges gain patterns, extending reciprocity to asymmetric scenarios.
4. **Perceptual user study confirmation**: Versa significantly improves the realism and directional consistency of spatial audio.

## Limitations & Future Work
- Reciprocity holds under idealized conditions; nonlinear media and complex materials in real environments may introduce deviations.
- The current formulation treats reciprocity as a structural regularizer rather than assuming perfect reciprocity.
- Generalization to unseen scenes (cross-room) is beyond the current scope.
- SSL requires two-stage training, increasing overall complexity.

## Related Work & Insights
- The approach is analogous to bidirectional path tracing in computer graphics, which also exploits reciprocity.
- The idea of using reciprocity as a physical constraint or regularizer is extensible to other wave propagation problems (light, radio frequency, elastic waves).
- The definition of the resounding task opens a new research direction for acoustic field modeling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first systematic application of reciprocity in acoustic ML)
- Technical Depth: ⭐⭐⭐⭐⭐ (rigorous theoretical derivation, elegantly designed methodology)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive coverage via simulation, real data, and user studies)
- Writing Quality: ⭐⭐⭐⭐⭐ (direct application to immersive VR/AR audio)
- Value: ⭐⭐⭐⭐⭐ (exemplary integration of physical principles into ML training)

## Supplementary Analysis

The reciprocity verification experiment (Table 1) demonstrates the degree to which reciprocity holds in both real and simulated environments. Paired impulse responses exhibit errors far smaller than unpaired cases across all metrics (e.g., C50 in the kitchen scene drops from 3.69 to 0.29; in the office from 2.49 to 0.18), confirming the practical reliability of the reciprocity principle. In simulation, increasing the ray count (10k → 1,000k) further reduces paired error.

Table 2 shows that Versa-ELE, as a plug-and-play data augmentation, consistently improves multiple baselines including NN, Linear, DiffRIR, INRAS, and NAF. For example, INRAS's STFT error in Scene 3 drops from 4.22 to 1.67 after ELE, validating the method's model-agnostic nature.

The Versa-SSL two-stage pipeline: Stage 1 fits the acoustic field to obtain the emitter directional gain pattern $G_e$ (encoded via spherical harmonics); Stage 2 replaces the receiver gain pattern with $G_e$ to enforce the consistency constraint. At inference time, any HRTF can be substituted to enable personalized auditory rendering. Overall metrics: Versa-ELE achieves an average improvement of 34% on C50 and 31% on STFT; Versa-SSL further improves C50 by 24% and STFT by 48% on top of AVR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](../../ICCV2025/audio_speech/how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[NeurIPS 2025\] SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation](simulmega_moe_routers_are_advanced_policy_makers_for_simultaneous_speech_transla.md)

</div>

<!-- RELATED:END -->
