---
title: >-
  [Paper Note] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis
description: >-
  [ACL2026][Audio & Speech][zero-shot speech synthesis] ReStyle-TTS enables relative adjustments to pitch, energy, and emotion in zero-shot TTS—rather than being locked to the reference audio's style—by utilizing decoupled…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "zero-shot speech synthesis"
  - "style control"
  - "LoRA fusion"
  - "relative control"
  - "timbre consistency"
date: 2026-05-08
content_hash: a1f1225d5a84555a
---

# ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis

**Conference**: ACL2026  
**arXiv**: [2601.03632](https://arxiv.org/abs/2601.03632)  
**Code**: https://cucl-2.github.io/Restyle-TTS  
**Area**: audio_speech  
**Keywords**: zero-shot speech synthesis, style control, LoRA fusion, relative control, timbre consistency

## TL;DR
ReStyle-TTS enables relative adjustments to pitch, energy, and emotion in zero-shot TTS—rather than being locked to the reference audio's style—by utilizing decoupled text/reference guidance, continuously scalable style LoRAs, orthogonal LoRA fusion, and timbre consistency optimization while maintaining text intelligibility and speaker timbre.

## Background & Motivation
**Background**: Zero-shot TTS has enabled cloning the timbre of unseen speakers using short reference audio. Given a reference audio and text, models generate speech in the same voice, making voice assistants and personalized dubbing more flexible.

**Limitations of Prior Work**: Reference audio contains not only timbre but also the speaking rate, pitch, energy, and emotion at that moment. To clone the voice, models often inherit the reference's style. If a user only has a happy reference but wants to synthesize angry speech, they must find another matching target sample, which is inconvenient in practice.

**Key Challenge**: Controlling style requires weakening the reference audio's constraint on the generation; however, the reference is also the source of the speaker's timbre. Over-weakening leads to speaker timbre drift. Thus, a trade-off exists between style controllability and timbre consistency.

**Goal**: The authors aim to achieve a user-friendly control method for zero-shot TTS: preserving the speaker identity provided by the short reference while allowing continuous, relative, and composable control over pitch, energy, and emotion.

**Key Insight**: The paper observes that existing controllable TTS methods mostly rely on absolute target styles or discrete text prompts (e.g., "speak in a happy tone"). This does not align with common user needs; a more natural operation is "higher than the reference" or "angrier." Therefore, control should be relative to the reference rather than pushing all samples toward a fixed style.

**Core Idea**: Use Decoupled CFG to reduce the model's implicit dependence on the reference style, apply Style LoRAs to provide explicit style directions, and use Timbre Consistency Optimization to restore the weakened timbre consistency.

## Method
ReStyle-TTS is built on flow-matching zero-shot TTS models like F5-TTS. Instead of retraining a large model, it introduces three layers of modification: separating text and reference guidance during generation, training multiple style-specific LoRAs, and using orthogonal projection to reduce interference when multiple LoRAs are active, combined with a speaker similarity reward during training.

### Overall Architecture
Inputs include target text, reference audio, and one or more style control intensities. The system first generates with lower reference guidance via DCFG to avoid duplicating the original style. Depending on the user-specified intensity, the corresponding Style LoRAs are added to the base model; if multiple styles are active, Orthogonal LoRA Fusion is performed. During training, TCO re-weights the flow-matching loss, giving higher weights to samples with high speaker similarity.

### Key Designs
1. **Decoupled Classifier-Free Guidance**:
    - **Function**: Separates "following the text" and "following the reference audio" into two independent knobs.
    - **Mechanism**: Standard CFG combines $f_{a,t}$ and $f_{\emptyset,\emptyset}$, mixing text and reference audio into one guidance weight. DCFG additionally calculates a text-only prediction $f_{\emptyset,t}$ and uses $\hat{f}_{DCFG}=f_{\emptyset,t}+\lambda_t(f_{\emptyset,t}-f_{\emptyset,\emptyset})+\lambda_a(f_{a,t}-f_{\emptyset,t})$ to separately control text strength $\lambda_t$ and reference strength $\lambda_a$.
    - **Design Motivation**: Keeping $\lambda_t$ high maintains intelligibility, while lowering $\lambda_a$ releases style space, allowing LoRAs to effectively change pitch, energy, and emotion.

2. **Style LoRA and Orthogonal LoRA Fusion**:
    - **Function**: Provides continuous, composable, and interpretable style control directions.
    - **Mechanism**: LoRAs are trained for high/low pitch, high/low energy, and various emotions. During inference, the scaling coefficient $\alpha_i$ for each LoRA serves as the style intensity knob. When multiple LoRAs are superimposed, OLoRA projects each LoRA update vector onto the orthogonal complement of other LoRA subspaces before weighted fusion $\Delta W_{fuse}=\sum_i \alpha_i \tilde{\Delta W_i}$, preventing attribute entanglement.
    - **Design Motivation**: In image generation, LoRAs act as style sliders, but TTS style is embedded in the reference. Only after reducing reference dependencies can LoRAs inject stable, continuous control directions.

3. **Timbre Consistency Optimization**:
    - **Function**: Compensates for timbre drift caused by lowering reference guidance via DCFG.
    - **Mechanism**: While primarily using flow-matching loss, the model calculates a speaker similarity reward with the reference audio using a speaker verification model. The system maintains an EMA baseline to get advantage $A_t=r_t-b_t$, and re-weights the flow-matching loss $\mathcal{L}_{total}=w_t\mathcal{L}_{FM}$ using bounded weights $w_t=1+\lambda \tanh(\beta A_t)$.
    - **Design Motivation**: Compared to policy gradient, this advantage-weighted regression does not require backpropagation through the generation process or reward, making it stable and efficient. Samples with higher similarity are emphasized to pull the speaker identity back.

### Loss & Training
The base model is F5-TTS. The authors train style LoRAs on various subsets of VccmDataset, covering high/low pitch, high/low energy, and emotions including angry, disgusted, fear, happy, sad, surprised, and neutral. LoRAs are injected into all linear layers with rank 32 and alpha 64. AdamW is used with a learning rate of $1\times 10^{-5}$ and a batch size of 30,000 audio frames for 250 hours per subset. DCFG training uses masked speech dropout of 0.3 and joint dropout of 0.2. For inference, $\lambda_a$ is set to 0.5 to reduce reference dependence. TCO parameters are $\lambda=0.2, \beta=5.0, \mu=0.9$.

## Key Experimental Results

### Main Results
The paper compares the control modalities of controllable zero-shot TTS. ReStyle-TTS is positioned for continuous relative control via LoRAs rather than external audio or text descriptions.

| Method | Timbre Source | Style Source | Continuous Control | Control Type |
|------|----------|----------|----------|----------|
| IndexTTS2 / Vevo | Reference Audio | Style Audio | No | Absolute |
| ControlSpeech / EmoVoice / CosyVoice | Reference Audio | Text Description | No | Absolute |
| StyleFusion TTS | Reference Audio | Audio or Text | No | Absolute |
| ReStyle-TTS | Reference Audio | Style LoRA | Yes | Relative |

In scenarios where reference and target emotions conflict, ReStyle-TTS effectively overrides the reference emotion. The table below shows accuracy (ACC) for emotion migration cases.

| Reference → Target | CosyVoice ACC↑ | EmoVoice ACC↑ | IndexTTS2 ACC↑ | ReStyle-TTS ACC↑ |
|--------------------|----------------|---------------|----------------|------------------|
| Happy → Angry | 65.2 | 73.5 | 88.5 | 100.0 |
| Fear → Happy | 82.9 | 85.7 | 90.4 | 100.0 |
| Surprised → Angry | 72.0 | 78.5 | 83.6 | 100.0 |
| Angry → Neutral | 58.4 | 74.2 | 78.9 | 84.6 |
| Disgusted → Happy | 83.5 | 86.5 | 89.3 | 96.8 |

Pitch and energy control are also stable, with ReStyle-TTS significantly outperforming CosyVoice and EmoVoice across all four directions.

| Attribute | Reference → Target | CosyVoice ACC↑ | EmoVoice ACC↑ | ReStyle-TTS ACC↑ |
|----------|--------------------|----------------|---------------|------------------|
| Pitch | Low → High | 74.9 | 72.4 | 90.2 |
| Pitch | High → Low | 76.9 | 73.1 | 92.8 |
| Energy | Low → High | 87.5 | 76.1 | 92.4 |
| Energy | High → Low | 88.6 | 75.9 | 93.0 |

### Ablation Study
DCFG is the key to style controllability, while TCO is critical for timbre preservation. Standard CFG fails to change attributes even with good WER and Spk-sv. Without TCO, control remains but speaker similarity drops.

| Config | Attr Δ(rel.)↑ | WER(%)↓ | Spk-sv↑ | Conclusion |
|------|---------------|---------|---------|------|
| default ($\lambda_t=2,\lambda_a=0.5$) | 51.2% | 2.31 | 0.79 | Balanced control, intelligibility, and timbre |
| w/o DCFG ($\lambda_{cfg}=2$) | 2.1% | 1.83 | 0.90 | Good timbre but nearly uncontrollable |
| w/o DCFG ($\lambda_{cfg}=0.5$) | 7.6% | 2.67 | 0.85 | Slightly controllable but restricted by reference |
| w/o TCO | 51.0% | 2.32 | 0.71 | Control maintained, timbre consistency drops |

### Key Findings
- Attribute control curves change smoothly with LoRA strength, while WER and Spk-sv remain stable, suggesting Style LoRAs act as continuous sliders.
- Negative scaling of high-attribute LoRAs produces opposite effects (e.g., using negative coefficients on a high-pitch LoRA to lower pitch), reducing data requirements.
- 2D and 3D control surfaces show that adjusting one LoRA primarily affects the target attribute with minimal impact on others, validating OLoRA's disentanglement capability.
- In relative control experiments, the regression slope between reference and generated energy ranges from 0.77 to 1.22 with an intercept near 0, proving the model maintains relative rankings rather than shifting to a fixed target.

## Highlights & Insights
- DCFG is a highly practical design: it doesn't remove reference influence entirely but splits text fidelity and reference dependency, directly addressing the core zero-shot TTS trade-off.
- Using LoRAs as sliders for speech style has high transfer value. Compared to text prompts, intensity adjustment is better suited for product interaction and UI controls.
- TCO is a restrained reinforcement learning design: it avoids high-variance policy gradients by using reward re-weighting for supervised loss, leveraging speaker verification signals without instability.
- The paper clearly defines "relative control." While many controllable generation methods are absolute target-based, real users often need fine-tuning based on the current sample.

## Limitations & Future Work
- A primary limitation is that extending to new attributes requires collecting data and training additional LoRAs; the control space is not an open-ended natural language space.
- Experiments currently focus on pitch, energy, and certain emotions, not yet covering speaking rate, pauses, accents, or complex character styles.
- Even with TCO, timbre drift may still occur in extreme emotional or long-form generations, suggesting timbre and style are not perfectly separable.
- Evaluation depends on automated tools like Emotion2Vec; while MOS-SA is provided, larger-scale human preference tests are needed.
- Voice cloning and emotional manipulation pose misuse risks. The authors suggest watermarking and synthesis detection as necessary deployment prerequisites.

## Related Work & Insights
- **vs IndexTTS2 / Vevo**: These rely on additional style audio; ReStyle-TTS utilizes LoRA intensity for relative adjustment without needing a second audio file.
- **vs ControlSpeech / EmoVoice / CosyVoice**: These use text descriptions, which are user-friendly but often discrete and unstable; ReStyle-TTS provides predictable slider-like control.
- **vs StyleFusion TTS**: StyleFusion supports multi-modal style inputs but remains absolute; ReStyle-TTS differentiates itself through reference-relative control.
- **vs Image LoRA composition**: In images, LoRAs can directly change styles, but in TTS, the reference carries both timbre and style. The use of DCFG to decouple before LoRA fusion is a necessary adaptation for the audio domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of DCFG, Style LoRA, OLoRA, and TCO effectively addresses relative style control in zero-shot TTS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers single/multi-attribute control, relative control, conflicting styles, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and understandable diagrams; some continuous control results could be more tabular.
- Value: ⭐⭐⭐⭐⭐ Highly practical for controllable speech synthesis, especially for products requiring fine-grained editing while preserving identity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] UniVocal: Unified Speech-Singing Code-mixed Synthesis](univocal_unified_speech-singing_code-switching_synthesis.md)
- [\[ACL 2026\] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)
- [\[ICML 2026\] MusicDET: Zero-Shot AI-Generated Music Detection](../../ICML2026/audio_speech/musicdet_zero-shot_ai-generated_music_detection.md)
- [\[ACL 2026\] Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations](affectron_emotional_speech_synthesis_with_affective_and_contextually_aligned_non.md)

</div>

<!-- RELATED:END -->
