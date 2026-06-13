---
title: >-
  [Paper Note] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions
description: >-
  [ACL2026][Audio & Speech][Unified Audio Generation] UniSonate integrates Text-to-Speech (TTS), Text-to-Music (TTM), and Text-to-Audio (TTA) into a single flow-matching MM-DiT using a unified Instruction-Content represent…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Unified Audio Generation"
  - "Text Instruction Control"
  - "Flow Matching"
  - "Dynamic SFX token"
  - "MM-DiT"
date: 2026-05-08
content_hash: 5126ca898c3401a0
---

# UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions

**Conference**: ACL2026 Oral  
**arXiv**: [2604.22209](https://arxiv.org/abs/2604.22209)  
**Code**: No public code; Demo: https://qiangchunyu.github.io/UniSonate/  
**Area**: Audio and Speech  
**Keywords**: Unified Audio Generation, Text Instruction Control, Flow Matching, Dynamic SFX token, MM-DiT

## TL;DR
UniSonate integrates Text-to-Speech (TTS), Text-to-Music (TTM), and Text-to-Audio (TTA) into a single flow-matching MM-DiT using a unified Instruction-Content representation, dynamic SFX token injection, and multi-stage curriculum learning. It achieves performance comparable to or exceeding specialized models in TTS and TTM while maintaining viable sound effect generation capabilities in TTA.

## Background & Motivation
**Background**: Audio generation has long been divided into specialized tasks. TTS focuses on speech naturalness, timbre, and phoneme alignment; TTM emphasizes lyrics, rhythm, instrumentation, and musical structure; TTA deals with open-ended environmental sounds, events, and soundscapes. Strong models exist for each, but their input interfaces, control signals, and training data formats remain fragmented.

**Limitations of Prior Work**: Existing unified models often cover only speech and singing or rely on reference audio for timbre control. Multi-task systems frequently require different input formats, task labels, or subsequent fine-tuning. For a truly general audio generation model, the most natural interface is pure natural language instructions describing the speaker, emotion, instrument, atmosphere, or sound event for direct generation.

**Key Challenge**: Speech and music are highly structured, typically possessing discrete temporal skeletons like phonemes, lyrics, or beats. Environmental sound effects (SFX) are more akin to continuous textures without natural token boundaries. Directly mixing TTS, TTM, and TTA data during training risks interference, where the high variance of unstructured SFX disrupts speech articulation and musical structure, leading to negative transfer.

**Goal**: Ours aims to satisfy three objectives in one model: unified generation of speech, music, and sound effects; a unified reference-free natural language instruction interface; and fine-grained control across all three domains.

**Key Insight**: The paper decomposes conditions into two semantic lines: Instruction and Content. Instruction manages high-level attributes (e.g., male voice, sad tone, jazz piano, street footsteps), while Content manages temporal structure. Phonemes or lyrics serve as content for speech and music, whereas SFX utilize learnable `[SFX]` tokens of dynamic length to provide a "pseudo-phoneme" skeleton.

**Core Idea**: Use dynamic `[SFX]` tokens to symbolize unstructured audio, allowing SFX to be treated as a sequence modeling problem by the same phoneme-driven Transformer. Cross-modal optimization conflicts are mitigated through curriculum learning progressing from speech to music and then to sound effects.

## Method

### Overall Architecture
UniSonate is a dual-stream Multimodal Diffusion Transformer based on conditional flow matching. The input side constructs a unified text condition sequence: natural language instructions are encoded by a frozen Qwen2.5-7B; the content part consists of phoneme sequences transformed from text/lyrics for TTS/TTM, or a series of learnable `[SFX]` special tokens for SFX. The audio side uses a pre-trained Mel-VAE to compress 44.1kHz waveforms into continuous latents with a downsampling rate of 1024, forming the acoustic latents for diffusion/flow matching.

The model contains two streams: the Text Stream processes instruction-content conditions, and the Audio Stream processes noisy audio latents. The initial Joint Diffusion Transformer layers align the two streams via joint attention, allowing audio latents to focus on both global style descriptions and local structural tokens. The subsequent Single Diffusion Transformer layers refine acoustic details solely on the audio stream. During training, the model learns the vector field from a noise distribution to real audio latents; during inference, an ODE solver integrates this to obtain the target latent, which is then decoded into a waveform.

### Key Designs
1.  **Instruction-Content Alignment Representation**:
    - **Function**: Converts conditional inputs of different audio tasks into a unified "high-level instruction + temporal content" format.
    - **Mechanism**: Instruction describes acoustic attributes to be generated, while content provides temporal anchors. In TTS, content is the phoneme sequence for the text; in TTM, it is the phonemes for lyrics/singing; in SFX, where no text exists, special tokens are used.
    - **Design Motivation**: Previous unified models often concatenated task labels, leaving interfaces inconsistent. This approach ensures users only write natural language instructions, while the model projects different tasks into the same conditional space internally.

2.  **Dynamic Token Injection for SFX**:
    - **Function**: Grants unstructured environmental sounds a temporal skeleton that can be aligned via Transformer attention.
    - **Mechanism**: The average phoneme density $\lambda$ is estimated from speech data. For an SFX clip of target duration $T_{target}$, $L_{sfx}=\lfloor \lambda \cdot T_{target} \rfloor$ `[SFX]` tokens are generated. These are not a single duration embedding but a sequence of temporal anchors traversed by cross-attention.
    - **Design Motivation**: Providing only a global duration prompt makes it difficult to control the internal progression of sound effects. Introducing a dedicated SFX branch would break the unified architecture. Dynamic token injection converts SFX into a "pseudo-linguistic sequence," balancing unification with duration control.

3.  **Dual-stream MM-DiT + Multi-stage Curriculum Learning**:
    - **Function**: Absorbs speech, music, and SFX data within a single generative model while reducing negative transfer.
    - **Mechanism**: Architecturally, joint attention enables deep interaction between text conditions and audio latents, followed by audio-only layers for acoustic refinement. Training starts with speech anchoring, expands to speech + music, and finally incorporates sound effects.
    - **Design Motivation**: Speech is the most highly structured task; mastering its pronunciation and prosody stabilizes the alignment mechanism. Music provides medium-to-long-term rhythmic structure. Incorporating high-variance SFX last expands timbre and environmental coverage without disrupting speech modeling initially.

### Loss & Training
The objective is conditional flow matching. Given a clean latent $x_0$, noise $x_1$, time $t$, and text condition $C_{text}$, the model predicts the vector field $v_\theta(t, C_{text}, x_t)$ to approximate the velocity from $x_0$ to $x_1$. The 1.34B parameter MM-DiT consists of 14 Joint Diffusion Transformer layers and 6 Single Diffusion Transformer layers, with RoPE for temporal position awareness. Training data includes 50K hours of speech, 20K hours of music, and 1.5M SFX clips. Audio is unified at 44.1kHz in 2-20 second clips. Training was conducted on 32 A800 80GB GPUs using Adam with an initial learning rate of $1e-4$.

## Key Experimental Results

### Main Results
The authors evaluated TTS, TTM, and TTA. For TTS, they used Seed-TTS WER, instruction control accuracy, similarity, and MOS. For TTM, SongEval, control accuracy, and musicality MOS were used. For TTA, metrics included FAD, FD, IS, and CLAP on AudioCaps.

| Task | Metric | Strong Baseline | UniSonate | Conclusion |
|------|------|--------|-----------|------|
| TTS (EN) | WER↓ | InstructAudio 1.52, ZipVoice 1.70 | 1.47 | Unified training did not dilute intelligibility; achieved best result. |
| TTS (ZH) | WER↓ | InstructAudio 1.35, F5-TTS 1.53 | 1.25 | Parity with Ground Truth (1.25). |
| TTS Control | Dialogue accuracy↑ | InstructAudio 90.00, CosyVoice2 N/A | 93.33 | Stronger multi-speaker dialogue control. |
| TTM | SongEval Coherence↑ | ACE-Step 2.89, InstructAudio 3.08 | 3.18 | Highest musical structure consistency. |
| TTM | Musicality MOS↑ | ACE-Step 2.88, InstructAudio 2.91 | 3.01 | Best subjective musicality. |
| TTA | FAD↓ | AudioLDM-L 4.32, Stable Audio 4.19, GenAU-L 2.07 | 4.21 | Comparable to some TTA baselines, behind specialized SOTA. |
| TTA | CLAP score | AudioLDM-L 0.208, GenAU-L 0.300 | 0.156 | Text-audio alignment is not the strongest area. |

### Ablation Study
The critical ablation compared single-task training vs. joint data training under the same architecture to verify positive transfer.

| Configuration | EN WER↓ | ZH WER↓ | Speaker Sim↑ | LSD↓ | MCD↓ | MSEP↓ | MR↓ |
|------|---------|---------|--------------|------|------|-------|-----|
| UniSonate (TTS-only data) | 2.24 | 1.40 | 0.63 | 2.63 | 8.70 | 574.67 | 0.426 |
| UniSonate (Joint data) | 1.47 | 1.25 | 0.77 | 1.79 | 5.46 | 422.36 | 0.31 |

| Configuration | Coh↑ | Mus↑ | Mem↑ | Cla↑ | Nat↑ | Note |
|------|------|------|------|------|------|------|
| UniSonate (TTM-only data) | 3.11 | 3.00 | 3.04 | 2.92 | 2.84 | Music data only |
| UniSonate (Joint data) | 3.18 | 3.07 | 3.10 | 2.99 | 2.90 | Improvement across all metrics with joint data |

### Key Findings
- Joint training facilitates positive transfer for structured tasks: speech WER, spectral distortion, pitch error, and musical SongEval all outperformed single-task counterparts.
- The value of dynamic SFX tokens lies in "accessibility" rather than directly defeating specialized TTA models: UniSonate enables unified sound effect generation, though its FAD still lags behind GenAU-L.
- Curriculum learning is vital: mixing high-variance environmental sounds from the start disrupts speech pronunciation and musical structure. Establishing speech alignment before task expansion is more effective.

## Highlights & Insights
- The most elegant design is treating SFX as "pseudo-phoneme sequences without text." This is more unified than a separate SFX branch and provides better temporal anchoring than simple duration tokens.
- The paper emphasizes positive transfer: sound effects do not hinder speech; instead, they increase acoustic diversity, making speech reconstruction more robust. Strong alignment training in speech also transfers to musical structure.
- Instruction-Content decoupling has strong transferability. Similar logic could be applied to video or multimodal generation: instructions control high-level attributes, while content tokens provide temporal/spatial skeletons.
- A unified interface is more important than a unified model alone. If users still must remember different input formats for different tasks, the practical value of unification is limited; UniSonate unifies the interface via natural language instructions.

## Limitations & Future Work
- SFX quality remains below specialized TTA models (FAD 4.21 vs. GenAU-L 2.07), suggesting unified representations may not yet capture extremely diverse environmental textures.
- Training and evaluation focused on 2-20 second clips; long-form songs, dialogues, audiobooks with plot progression, or complex soundscapes require hierarchical planning mechanisms.
- Pure text instructions are naturally one-to-many; "sad song" or "deep male voice" can have many acoustic realizations. Meeting specific user expectations without a reference cue remains difficult.
- The 1.3B diffusion/flow matching model has high inference costs. Real-time TTS or interactive audio design scenarios would require distillation, caching, or few-step sampling.
- High-fidelity generation poses risks regarding deepfakes, copyright, and style imitation; model releases require watermarking, detectors, and usage restrictions.

## Related Work & Insights
- **vs InstructAudio**: InstructAudio places speech and music within an instruction framework but excludes SFX. UniSonate's extension is the use of dynamic `[SFX]` tokens for unstructured audio.
- **vs CosyVoice / Vevo2**: These excel in speech/singing quality but depend on reference audio or cover only speech-singing. UniSonate emphasizes reference-free text instructions and multimodal unification.
- **vs AudioLDM / GenAU-L**: Specialized TTA models are stronger in environmental fidelity (e.g., GenAU-L's superior FAD). UniSonate's advantage is providing a single model and interface for speech/music/SFX.
- **vs AudioBox / UniAudio**: These models proved multi-task audio generation possible, but input formats and task coverage were less consistent. UniSonate aligns phoneme-driven structures with unstructured SFX.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Dynamic SFX token injection is a clear and effective unification strategy; overall architecture builds on established MM-DiT/flow matching.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers TTS, TTM, TTA, and joint training ablations, though SFX fine-grained control, human evaluation, and long-audio experiments could be more robust.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and methodology are logical, and tables are comprehensive, though some tables are crowded and TTA metric explanations are slightly brief.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for general audio generation, particularly as a reference framework for unifying audio tasks via "text instructions + structural tokens."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)

</div>

<!-- RELATED:END -->
