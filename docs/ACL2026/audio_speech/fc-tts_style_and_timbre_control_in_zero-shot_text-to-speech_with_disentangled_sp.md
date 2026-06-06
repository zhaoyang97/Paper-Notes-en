---
title: >-
  [Paper Note] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations
description: >-
  [ACL2026][Audio & Speech][Zero-Shot TTS] FC-TTS utilizes the disentangled speech representations of FACodec as conditioning sources. Through two-stage spectrogram generation, VQ-VAE style encoding…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Zero-Shot TTS"
  - "Timbre Control"
  - "Style Control"
  - "FACodec"
  - "flow matching"
date: 2026-05-08
content_hash: 86a157be441bc91b
---

# FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations

**Conference**: ACL2026  
**arXiv**: [2605.24618](https://arxiv.org/abs/2605.24618)  
**Code**: Audio demo: https://qualcomm-ai-research.github.io/fc-tts  
**Area**: Speech Synthesis / Controllable TTS  
**Keywords**: Zero-Shot TTS, Timbre Control, Style Control, FACodec, flow matching  

## TL;DR
FC-TTS utilizes the disentangled speech representations of FACodec as conditioning sources. Through two-stage spectrogram generation, VQ-VAE style encoding, and condition consistency loss, it separates timbre and speaking style—originally entangled within a single reference in zero-shot TTS—into two independently controllable inputs.

## Background & Motivation
**Background**: Zero-shot text-to-speech has advanced significantly in mimicking speaker timbre and expression from a reference audio. Systems like F5-TTS, NaturalSpeech 3, DiTTo-TTS, and CLaM-TTS continue to push boundaries in naturalness, intelligibility, and speaker similarity. Simultaneously, practical applications increasingly require fine-grained control, such as maintaining a specific speaker's timbre while adopting the emotion, rhythm, or intonation from a different reference audio.

**Limitations of Prior Work**: Most reference-based TTS systems bundle style and timbre within the same reference audio. When a user intends to "speak with the voice of A but the emotion of B," these models often fail to distinguish which information belongs to speaker identity and which belongs to prosodic style. Even when representation learning methods like FACodec or NANSY++ attempt to decompose speech into prosody, content, detail, and speaker embeddings, directly reusing their decoders does not guarantee the ability to handle unseen style-timbre combinations during training.

**Key Challenge**: High generation quality typically results from joint modeling of all attributes, but joint modeling leads to attribute leakage. Strong disentanglement usually relies on information bottlenecks, which may sacrifice naturalness and detail. The challenge addressed here is how to synthesize natural, clear, and controllable speech without letting content/details leak into the control paths.

**Goal**: FC-TTS aims to support two references: one providing speaker timbre and the other providing style/prosody. The model must not only maintain competitive zero-shot TTS performance on LibriSpeech but also demonstrate independent control over timbre and style on emotion-rich data like RAVDESS.

**Key Insight**: Rather than retraining a completely new speech representation model, the authors acknowledge that FACodec disentanglement is imperfect and introduce structural and training constraints on the TTS side. The core mechanism involves allowing timbre to define a rough acoustic space first, followed by a second stage where style refines the spectrogram, thereby reducing cross-contamination between the two conditions.

**Core Idea**: Decompose generation into "timbre anchoring" and "style refinement" phases, further constrained by quantized style encoding and condition consistency loss to force the generated speech to match two independent references simultaneously.

## Method
FC-TTS is built upon FACodec and conditional flow matching (CFM). FACodec provides prosody tokens $c_p$, content tokens $c_c$, detail tokens $c_d$, and speaker embedding $z_{spk}$. FC-TTS utilizes only $z_{spk}$ and $c_p$, deliberately discarding content/detail tokens to reduce the leakage of textual content and low-level acoustic details into the control path.

### Overall Architecture
During training, the target speech provides both timbre and style conditions; during inference, these conditions can originate from different utterances. Input text is converted into a phoneme sequence, which is aligned to the frame level via a text encoder and duration predictor. The first stage uses a timbre adapter to inject the speaker embedding into layer normalization, generating a blurry log-mel spectrogram anchored by timbre conditions. The second stage employs a style adapter and a flow-matching decoder to refine the blurry spectrogram into the final log-mel spectrogram under the style embedding condition. Finally, HiFi-GAN converts the log-mel spectrogram into a 22 kHz waveform.

The style embedding is derived from a TCF module: prosody tokens pass through a Transformer encoder and are compressed into a fixed number of latent tokens via a Q-Former style cross-attention mechanism, followed by discretization using finite scalar quantization (FSQ). The authors place TCF modules at both the phoneme and frame levels to capture intra-utterance style variations.

### Key Designs
1.  **Two-Stage Hierarchical Spectrogram Generation**:
    - **Function**: Separates timbre and style control into different generation stages to avoid entanglement within a single decoder.
    - **Mechanism**: The first stage uses only $z_{spk}$ to generate an over-smoothed blurry spectrogram $h$, trained with an MAE loss $L_{blur}=E[\|h-x_0\|]$. The second stage uses conditional flow matching to generate the final spectrogram under the $c_p$ style condition.
    - **Design Motivation**: Directly using a FACodec decoder is unstable for unseen style-timbre combinations. Generating a blurry spectrogram first anchors the timbre and recording conditions in a reasonable acoustic space, allowing the style module to focus on fine-grained prosody.

2.  **VQ-VAE / TCF Style Encoder**:
    - **Function**: Extracts high-level style from prosody tokens instead of copying low-level details from the reference audio.
    - **Mechanism**: TCF consists of a Transformer encoder, a cross-attention query bottleneck, and FSQ quantization. Q-Former style queries compress variable-length prosody into fixed latents, and FSQ forms discrete style codes. An auxiliary ResNet reconstruction loss prevents FSQ collapse.
    - **Design Motivation**: Speaking style can vary within an utterance; assuming consistent reference style in traditional in-context learning is unreliable. Quantization bottlenecks suppress acoustic residuals, biasing representations toward transferable styles like rhythm, intonation, and emotion.

3.  **Condition Consistency Loss (CCL)**:
    - **Function**: Constrains the generated spectrogram to retain both target prosody tokens and speaker embeddings in multi-condition generation.
    - **Mechanism**: Two attribute predictors are trained: one predicts prosody tokens from the generated spectrogram and $z_{spk}$, while the other predicts the speaker embedding from the generated spectrogram and $c_p$. The loss is a weighted sum of prosody cross-entropy and speaker negative cosine similarity: $L_{CCL}=\lambda_{pro}E[CE(c_p,f(\hat{x},z_{spk}))]-\lambda_{spk}E[cos(z_{spk},g(\hat{x},c_p))]$.
    - **Design Motivation**: Ordinary consistency losses focusing on a single attribute can provide ambiguous gradients in multi-condition scenarios. Feeding non-target attributes to the predictor sharpens the posterior, particularly stabilizing the early denoising stages.

### Loss & Training
The total training objective includes CFM loss, blurry spectrogram MAE, prosody CE, speaker cosine consistency, mel reconstruction, aligner forward-sum, binary alignment, and duration CFM. Coefficients include $\lambda_{CFM}=5.0$, $\lambda_{blur}=1.0$, $\lambda_{ccl-pro}=0.2$, $\lambda_{ccl-spk}=0.5$, $\lambda_{mel-recon}=1.0$, $\lambda_{dur}=1.0$, $\lambda_{forwardsum}=0.1$, and $\lambda_{bin}=0.1$.

Training was conducted on LibriHeavy for 200k iterations using AdamW, with a batch size of 64 and a learning rate of 0.0002. Training took 116 hours on 8 V100 GPUs. During inference, duration prediction uses 8 NFEs without classifier-free guidance; log-mel synthesis uses 32 NFEs with a CFG scale of 4.0. Conditioning was randomly dropped during training with a 15% probability.

## Key Experimental Results

### Main Results
| Task / Dataset | Metric | FC-TTS | Comparison | Conclusion |
|----------------|--------|--------|------------|------------|
| LibriSpeech test-clean Zero-Shot TTS | UTMOS / WER / SPK / Params | 4.22 / 1.88 / 0.60 / 204M | NaturalSpeech 3: 4.30 / 1.81 / 0.67 / 500M; F5-TTS†: 4.03 / 3.30 / 0.67 / 205M | Competitive naturalness and WER, but lower SPK than some SOTA. |
| RAVDESS Timbre Control | UTMOS / SPK / WER / Win | 4.03 / 0.48 / 0.18 / 66.1% | FACodec-VC: 3.19 / 0.27 / 8.40 / 10.7% | Significantly more stable under prosody-rich mismatch conditions. |
| RAVDESS Style Control | UTMOS / SPK / WER / MCD / Win | 3.95 / 0.47 / 0.30 / 3.21 / 65.5% | F5-TTS: 3.40 / 0.57 / 4.39 / 3.43 / 8.9% | Stronger style matching and intelligibility, though speaker similarity is sacrificed. |
| AudioLLM-as-a-Judge Style Evaluation | Win Ratio / Style-MOS | 91.7% / 3.92 | F5-TTS: 8.3% / 1.50 | Gemini 2.5 Pro shows a strong preference for FC-TTS. |

### Ablation Study
| Config | LibriSpeech UTMOS / WER / SPK / MCD | RAVDESS Style UTMOS / WER / SPK / MCD | Description |
|--------|-------------------------------------|---------------------------------------|-------------|
| FC-TTS | 4.22 / 1.88 / 0.60 / 5.60 | 3.91 / 0.30 / 0.37 / 3.33 | Full model |
| w/o two-stage generation | 4.15 / 1.93 / 0.60 / 5.83 | 3.57 / 0.30 / 0.37 / 3.26 | Acoustic stability decreases; spectrograms are more prone to prosody interference. |
| w/o VQ-VAE style encoding | 4.25 / 2.00 / 0.57 / 5.62 | 3.99 / 0.25 / 0.34 / 3.47 | Naturalness increases slightly, but style control and F0 following weaken. |
| w/o conditioning in consistency loss | 4.21 / 1.92 / 0.59 / 5.67 | 3.79 / 0.35 / 0.36 / 3.36 | Alignment and intelligibility decrease slightly after removing cross-conditioning. |
| w/o entire consistency loss | 3.95 / 5.88 / 0.48 / 6.34 | 3.70 / 9.36 / 0.21 / 3.75 | Most significant degradation, proving CCL is a critical component. |

### Key Findings
- FC-TTS does not aim for the top spot in every zero-shot TTS metric; it trades some performance for independent controllability while maintaining audio quality.
- The two-stage generation constraint limits the upper bound of naturalness but significantly improves stability for unseen style-timbre combinations.
- Removing the consistency loss led to the most drastic deterioration in WER, validating it as the strongest component evidence.
- In style control experiments, SPK is lower than F5-TTS, indicating that style disentanglement still involves a trade-off with timbre preservation.

## Highlights & Insights
- **Beyond "using FACodec"**: The real contribution lies in acknowledging the imperfections of FACodec disentanglement and constraining attribute flow through the generation pipeline and loss functions on the TTS side.
- **Effective Engineering in Two-Stage Design**: Generating a blurry spectrogram first might seem like a detour, but it sets a timbre anchor for the second stage, which is more controllable than feeding all conditions into a single decoder.
- **Handling Intra-utterance Style Variations**: Many TTS methods assume uniform style within a reference. This work points out that a single long utterance can contain multiple expressions, and the two-level TCF encoding is more faithful to real expression.
- **AudioLLM-as-a-Judge Supplement**: While not a replacement for human evaluation, it provides a scalable automatic assessment for style similarity, which is helpful for future large-scale controllability benchmarks.

## Limitations & Future Work
- The authors acknowledge that training and evaluation are currently limited to English, leaving the model's generalization across multilingual, dialectal, and cross-accent scenarios unproven.
- The model remains dependent on FACodec representations. Imperfect disentanglement in FACodec may still lead to control leakage of timbre or acoustic details.
- The boundary between "timbre" and "style" is inherently ambiguous (e.g., whether a husky voice is a vocal characteristic or a style). The lack of reliable quantitative metrics limits scientific comparison in controllable TTS.
- Zero-shot TTS carries risks of deepfakes and identity theft. FC-TTS's ability to fix timbre while changing emotional style necessitates considerations for authorized speakers, synthesis detection, watermarking, or access control.
- There remains a trade-off between naturalness and disentanglement. Future work could explore codec-free disentanglement, explicit accent/style taxonomies, and stronger speaker preservation objectives.

## Related Work & Insights
- **vs NaturalSpeech 3 / FACodec-based TTS**: NaturalSpeech 3 leverages the strong reconstruction of FACodec but does not demonstrate stability under mismatched style-timbre references; FC-TTS sacrifices some performance ceiling for clearer independent control.
- **vs F5-TTS**: F5-TTS has strong in-context learning, but its single-reference approach makes it difficult to isolate timbre and style. FC-TTS is clearly superior in RAVDESS style control across WER, MCD, ABX preference, and Style-MOS.
- **vs EmoSphere++ / IndexTTS 2**: These methods also pursue style and timbre control but may rely on empirically disentangled representations or specific emotion encoders. FC-TTS more systematically combines factorized codecs, structural staging, and condition consistency.
- **Insight**: Multi-attribute generation does not necessarily require larger unified models; physically separating attribute injection paths and constraining outputs with attribute-specific validators can be more reliable.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The combination of two-stage generation, TCF, and CCL is highly targeted for controllable TTS, though the foundation still relies on existing FACodec/CFM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ evaluations on LibriSpeech and RAVDESS, objective metrics, human evaluations, AudioLLM, and ablations are comprehensive; the lack of multilingual data is the main drawback.
- **Writing Quality**: ⭐⭐⭐⭐☆ Methodological diagrams and component explanations are clear, and ablation discussions are thorough.
- **Value**: ⭐⭐⭐⭐⭐ High practical potential for scenarios requiring controllable voice (e.g., games, audiobooks, assistive communication) and reveals core trade-offs in controllable TTS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[ICML 2026\] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](../../ICML2026/audio_speech/polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

</div>

<!-- RELATED:END -->
