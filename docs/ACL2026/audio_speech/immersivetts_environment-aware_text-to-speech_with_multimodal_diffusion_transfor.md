---
title: >-
  [Paper Note] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment
description: >-
  [ACL2026][Audio & Speech][Environment-aware speech synthesis] ImmersiveTTS utilizes a dual-stream MM-DiT to simultaneously model transcription content and environmental descriptions. It stabilizes training through dual-t…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Environment-aware speech synthesis"
  - "Multimodal Diffusion Transformer"
  - "flow matching"
  - "representation alignment"
  - "audio generation"
date: 2026-05-08
content_hash: 71491d66c355824a
---

# ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment

**Conference**: ACL2026  
**arXiv**: [2605.30965](https://arxiv.org/abs/2605.30965)  
**Code**: https://jjunak-yun.github.io/ImmersiveTTS  
**Area**: Speech Synthesis / Environment-Aware TTS  
**Keywords**: Environment-aware speech synthesis, Multimodal Diffusion Transformer, flow matching, representation alignment, audio generation  

## TL;DR
ImmersiveTTS utilizes a dual-stream MM-DiT to simultaneously model transcription content and environmental descriptions. It stabilizes training through dual-teacher representation alignment using WavLM and ATST-Frame, improving speech naturalness, intelligibility, and speech-environment fusion quality in TTS with background sound.

## Background & Motivation
**Background**: Text-guided audio generation is generally divided into TTA and TTS. TTA excels at generating environmental sounds, music, and sound effects but lacks precision in expressing speech content; TTS excels at generating clear speech from text but typically treats background sounds, room acoustics, or soundscapes as external conditions rather than generating them alongside the speech.

**Limitations of Prior Work**: Environment-aware TTS must simultaneously satisfy two objectives: speech must be intelligible, natural, and preserve speaker characteristics, while the background sound must match natural language descriptions and blend with the speech like a single real recording. Existing methods such as VoiceLDM and VoiceDiT can control the environment via text descriptions, but the interaction between the speech stream and environment stream remains insufficient. This often results in correct speech content with mismatched backgrounds, or matching backgrounds where the speech is corrupted by noise.

**Key Challenge**: Speech and environmental sounds differ significantly in temporal structure, spectral patterns, and semantic granularity. The speech side emphasizes phonemes, prosody, and content intelligibility, while the environment side emphasizes global soundscapes and local acoustic events. If a single condition or a single SSL teacher is used for constraint, the model often biases toward one end, creating a trade-off between clarity and environmental consistency.

**Goal**: The authors aim to construct a unified model that directly generates mixed audio from content prompts and environment prompts while maintaining low sampling steps, high speech intelligibility, environmental semantic consistency, and natural fusion.

**Key Insight**: The paper migrates the multimodal diffusion transformer approach from the SD3/Flux series to joint speech-environment generation. Transcript-aligned speech latents and text-conditioned environment contexts are placed into two streams and exchange information via joint attention. Simultaneously, domain-specific REPA is introduced, directing different intermediate layers to align with speech SSL and environmental sound SSL representations respectively.

**Core Idea**: Replace simple prompt conditioning with "dual-stream MM-DiT + dual-teacher REPA," allowing speech content and environmental soundscapes to interact explicitly during the generation process rather than being generated separately and mixed post-hoc.

## Method
The input for ImmersiveTTS includes three types of information: a content prompt (the text to be spoken), an environment prompt (background sound or scene description), and a speaker prompt (used to extract speaker embeddings). The output is a 16 kHz waveform containing both speech and environmental sounds. The model follows a latent flow matching framework: target audio is compressed into AudioLDM2 VAE latents, a velocity field from Gaussian noise to the target audio is learned in the latent space, and the waveform is reconstructed by a VAE decoder and HiFi-GAN vocoder.

### Overall Architecture
During training, LibriTTS clean speech is mixed with WavCaps non-speech environmental sounds at SNRs ranging from 2 to 10 dB to form environment-aware TTS training samples; a 0.15 probability of retaining clean speech is included to prevent the model from losing pure speech capabilities. Environmental descriptions are processed via CLAP to obtain global acoustic semantics for modulating AdaLN, and via Flan-T5-Large to obtain token-level environment text sequences for the environment stream.

On the speech side, a frame-level linguistic prior is obtained via a text encoder and MAS duration alignment, mapped to a representation compatible with VAE latents via a convolutional network, and concatenated with the noisy latent. The double-stream blocks of the MM-DiT allow environment tokens and speech latents to read each other through joint attention; subsequent single-stream blocks retain only the speech stream for high-fidelity refinement. Finally, the model predicts the velocity field using flow matching and employs dual classifier-free guidance during inference to independently adjust the intensity of environmental and content conditions.

### Key Designs
1.  **Dual-stream MM-DiT for Speech-Environment Interaction**:
    - **Function**: Places transcript-aligned speech features and text-conditioned environment contexts into two parallel streams to explicitly learn their coupling during generation.
    - **Mechanism**: The environment stream receives Flan-T5 token embeddings, while the speech stream receives noisy audio latents and content-aligned features. Information is exchanged via joint attention in double-stream DiT blocks, followed by refinement of the environment-adapted speech representation in single-stream blocks.
    - **Design Motivation**: Environment-aware TTS is not simply a matter of "generating speech then overlaying a background." Background sound affects intelligibility, soundstage perception, and overall naturalness. The dual-stream structure preserves modal differences while allowing for cross-modal alignment.

2.  **Dual-Granularity Environmental Conditioning via CLAP + T5**:
    - **Function**: Captures both global semantics and local details of environmental descriptions.
    - **Mechanism**: CLAP embeddings are combined with timestep embeddings after an MLP to modulate scale/shift in AdaLN. Flan-T5 token embeddings serve as the environmental context sequence, allowing the speech stream to select specific acoustic cues during attention.
    - **Design Motivation**: Using CLAP alone tends to result in coarse-grained scene labels, while using only text tokens may lack stable global acoustic constraints. The combination of both is better suited for generating results where "speech is embedded within a specific soundscape."

3.  **Domain-specific REPA (Representation Alignment)**:
    - **Function**: Uses different SSL teachers to separately constrain speech intelligibility and environmental acoustic consistency.
    - **Mechanism**: Hidden features are extracted from the intermediate layers of the speech stream and mapped via an MLP projector to calculate cosine alignment loss with frozen WavLM and ATST-Frame representations. WavLM is applied to the clean speech before mixing, while ATST-Frame is applied to the mixed audio. The key objective can be summarized as: $\mathcal{L}=\mathcal{L}_{Prior}+\mathcal{L}_{Dur}+\mathcal{L}_{Flow}+\mathcal{L}_{REPA}$.
    - **Design Motivation**: It is difficult for a single teacher to simultaneously explain both speech and environment domains. WavLM focuses more on speech content, while ATST-Frame focuses more on environmental events. Complementary teachers mitigate the trade-off between domains.

### Loss & Training
The training objective consists of four parts: MAS prior loss and duration loss for training the text encoder and duration predictor, flow matching loss for training the latent velocity field, and REPA loss for intermediate representation alignment. All loss weights are set to 1 in experiments. The model is trained for 400k steps using 2 NVIDIA RTX A6000 GPUs, an AdamW optimizer with a learning rate of $1\times 10^{-4}$, and a batch size of 8 per GPU. The model contains 12 double-stream blocks, 18 single-stream blocks, 6 attention heads, and a hidden size of 1024, totaling approximately 450M trainable parameters.

Inference samples from $Z_0\sim\mathcal{N}(0,I)$ and solves the flow ODE via an Euler solver. Dual CFG independently controls environment guidance and content guidance; the main experiments use $\omega_{env}=3, \omega_{cont}=3$ with 25 NFEs.

## Key Experimental Results

### Main Results
| Test Set | Model | NFEs | SN-MOS↑ | EC-MOS↑ | ON-MOS↑ | WER↓ | FAD↓ | CLAP↑ |
|--------|------|------|---------|---------|---------|------|------|-------|
| AudioCaps | VoiceLDM | 200 | 3.41 ± 0.06 | 3.33 ± 0.07 | 2.55 ± 0.05 | 16.45 | 8.75 | 0.229 |
| AudioCaps | VoiceDiT | 200 | 3.47 ± 0.05 | 3.44 ± 0.07 | 2.63 ± 0.05 | 11.68 | 9.07 | 0.263 |
| AudioCaps | ImmersiveTTS | 25 | 4.20 ± 0.07 | 3.48 ± 0.07 | 3.47 ± 0.05 | 8.06 | 5.80 | 0.308 |
| Seed-TTS + AudioCaps | VoiceLDM | 200 | 3.32 ± 0.06 | 3.24 ± 0.07 | 2.91 ± 0.08 | 11.20 | 6.98 | 0.118 |
| Seed-TTS + AudioCaps | VoiceDiT | 200 | 3.45 ± 0.06 | 3.38 ± 0.06 | 3.12 ± 0.08 | 7.08 | 5.37 | 0.134 |
| Seed-TTS + AudioCaps | ImmersiveTTS | 25 | 4.18 ± 0.07 | 3.32 ± 0.06 | 3.23 ± 0.08 | 4.48 | 3.92 | 0.207 |

Main results indicate that ImmersiveTTS achieves the lowest WER, lowest FAD, and highest CLAP on AudioCaps simultaneously, outperforming 200-step diffusion baselines with only 25 sampling steps. On the augmented test set, VoiceDiT has a slightly higher EC-MOS, but ImmersiveTTS performs better in SN-MOS, ON-MOS, WER, FAD, and CLAP, showing a preference for overall naturalness and intelligibility.

### Ablation Study
| Alignment Strategy | Teacher | Speech Domain | Env Domain | WER↓ | FAD↓ | CLAP↑ |
|----------|---------|--------|--------|------|------|-------|
| Base | None | - | - | 11.21 | 9.64 | 0.236 |
| Single | WavLM | ✓ | - | 10.97 | 8.02 | 0.231 |
| Single | ATST | - | ✓ | 13.77 | 8.78 | 0.271 |
| Single | USAD | ✓ | ✓ | 9.04 | 7.93 | 0.239 |
| Dual | WavLM + USAD | ✓ | ✓ | 8.95 | 7.33 | 0.248 |
| Dual | USAD + ATST | ✓ | ✓ | 8.94 | 8.20 | 0.266 |
| Dual | WavLM + ATST | ✓ | ✓ | 8.06 | 5.80 | 0.308 |

### Key Findings
- WavLM as a single teacher primarily improves speech content, while ATST as a single teacher primarily improves environmental semantics, but using either alone sacrifices the other; the WavLM + ATST dual-teacher combination is optimal across WER, FAD, and CLAP.
- Analysis of sampling steps shows that the gain is greatest when moving from very few steps to a moderate number; the paper notes that 9 steps already outperform VoiceLDM and VoiceDiT using 200 NFEs in WER, FAD, and CLAP.
- In the speaker similarity appendix, ImmersiveTTS achieves an S-MOS of 3.15 ± 0.06, matching VoiceDiT and approaching the 3.18 ± 0.05 of reconstructed samples.

## Highlights & Insights
- This paper explicitly models "environment-aware TTS" as a cross-modal joint generation problem rather than a post-hoc mix of TTS and TTA. This definition is closer to real-world scenarios where speech clarity, background intensity, and overall immersion are inherently interdependent.
- Dual-teacher REPA is a highly practical design: WavLM and ATST-Frame are not simply stacked, but have их supervision signals separated by speech and environment domains. This "domain-specific teacher" concept can be migrated to other multi-source audio generation tasks like video dubbing, speech enhancement, and music-vocal mixing.
- The most persuasive aspect of the experiments is the simultaneous improvement in quality and efficiency. Reaching or exceeding 200 NFE baselines with 25 NFEs demonstrates that flow matching + MM-DiT is deployment-friendly, beyond just having good offline metrics.

## Limitations & Future Work
- The authors acknowledge that training relies primarily on synthetic mixed data, so real-world speech-environment interactions (such as reverberation, occlusion, spatial positioning, and dynamic source changes) may still be insufficient.
- Current exploration of robustness to different SNRs, scene difficulties, and background complexity is not yet exhaustive; the main table proves average performance but does not show stability in extreme noise or high reverberation environments.
- While the model preserves speaker identity and speech content, it lacks explicit control over paralinguistic attributes such as emotion, speaking style, prosody, and expressive intensity. Future work could incorporate prosody/style/emotion prompts into a third control stream or design finer-grained CFG.
- Regarding risks, environment-aware TTS faces similar potential for misuse as standard speech synthesis for unauthorized voice cloning or deceptive audio, necessitating watermarking, detection, and usage regulations upon release.

## Related Work & Insights
- **vs VoiceLDM**: VoiceLDM is based on AudioLDM and conditions a U-Net with content and environment prompts; this work switches to dual-stream MM-DiT and adds domain-specific REPA, reducing WER from 16.45 to 8.06 and increasing CLAP from 0.229 to 0.308 in main experiments.
- **vs VoiceDiT**: VoiceDiT uses DiT with AdaLN for environmental control, but cross-modal interaction remains weak; ImmersiveTTS uses joint attention for interaction between environment tokens and speech latents in intermediate layers, achieving higher ON-MOS with fewer NFEs.
- **vs Single-task TTS/TTA pipelines**: The appendix shows that pipelines like CosyVoice2 + TangoFlux, which generate components separately and then mix them, can be strong in some objective metrics but fail to directly model speech-background interaction. The insight here is that the value of a unified model lies not just in metrics, but in learning the mutual influence within the real mixing process.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combining MM-DiT, flow matching, and domain-specific REPA for environment-aware TTS gives the problem definition and dual-teacher alignment high distinguishability.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers main experiments, single tasks, alignment strategies, sampling steps, CFG scale, speaker similarity, and broad baselines, though real recording scenarios and difficult subsets remain lacking.
- Writing Quality: ⭐⭐⭐⭐☆ The methodological chain is clear, and the tables directly support the conclusions; some formulas converted from PDF/HTML to text have moderate readability.
- Value: ⭐⭐⭐⭐☆ Direct value for immersive speech, gaming/NPCs, accessibility content, and multimedia generation, with the 25-step inference offering practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)
- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)
- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)

</div>

<!-- RELATED:END -->
