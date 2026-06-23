---
title: >-
  [Paper Note] FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions
description: >-
  [ICLR 2026][Audio & Speech][DPO] FlexiVoice utilizes an LLM backbone to simultaneously process text, style instructions, and timbre reference speech. Through a three-stage progressive post-training process comprising "DPO → Decoupling GRPO → Instruction GRPO," it specifically addresses the challenge of style-timbre-content entanglement, enabling zero-
tags:
  - ICLR 2026
  - Audio & Speech
  - DPO
  - GRPO
date: 2026-05-08
content_hash: 34d87f08df066516
---
# FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=F7GmbfyVg9](https://openreview.net/forum?id=F7GmbfyVg9)  
**Code**: [https://flexi-voice.github.io/](https://flexi-voice.github.io/) (Audio samples page)  
**Area**: Speech Synthesis / Controllable TTS  
**Keywords**: Zero-shot TTS, Natural Language Instruction Control, Style-Timbre Decoupling, DPO, GRPO, Progressive Post-Training  

## TL;DR
FlexiVoice utilizes an LLM backbone to simultaneously process text, style instructions, and timbre reference speech. Through a three-stage progressive post-training process comprising "DPO → Decoupling GRPO → Instruction GRPO," it specifically addresses the challenge of style-timbre-content entanglement, enabling zero-shot TTS to accurately follow natural language style instructions while maintaining stable timbre cloning.

## Background & Motivation
**Background**: Zero-shot TTS can already clone a speaker's timbre using a short reference audio. To further control the "speaking style" (emotion, speed, etc.), there are two main approaches: using another style reference audio (e.g., Vevo, IndexTTS2) or using instruction-based TTS (PromptTTS, VoxInstruct, CosyVoice2, etc.) with natural language descriptions.

**Limitations of Prior Work**: Instruction-based models often fail to balance competing goals—either they do not faithfully follow the instruction, or they lose timbre consistency when doing so. The root cause is that during supervised training, models over-rely on the strong acoustic priors of the reference speech (**timbre leakage**) or infer prosody from the text content (**content leakage**), resulting in the explicit style instructions being ignored.

**Key Challenge**: The authors identify this as the **Style-Timbre-Content Conflict**. When an instruction specifies "Happy," the reference audio is "Sad," and the text content implies "Sadness," the three modalities conflict. Simply adding an instruction condition cannot suppress the entangled acoustic cues.

**Goal**: To build a unified framework that does more than just "condition" the model; it **actively decouples style from timbre and content** and enforces instruction adherence under conflicting acoustic cues.

**Core Idea**: **Upgrade "instruction following" from simple conditional input to a rigorous decoupling process**—employing a curriculum-based Progressive Post-Training (PPT) that first aligns, then decouples, and finally generalizes to open instructions, supported by a large-scale instruction-speech dataset, FlexiVoice-Instruct, annotated by LLMs.

## Method

### Overall Architecture
FlexiVoice is built upon a pre-trained LLM: a speech tokenizer converts speech into discrete tokens; the LLM core consumes text, natural language instructions, and reference speech tokens to autoregressively generate speech tokens. These tokens are then transformed into Mel-spectrograms via flow matching and finally into waveforms by a vocoder. Training consists of two phases: first, pre-training FlexiVoice-Base using Emilia + FlexiVoice-Instruct (training only the LLM core while freezing others, without reference speech), then refining the Base model into the final FlexiVoice using three-stage **Progressive Post-Training (PPT)**. PPT is inspired by curriculum learning—starting from simple goals and progressing to difficult ones.

```mermaid
flowchart LR
    A[Pre-train FlexiVoice-Base<br/>Emilia + FlexiVoice-Instruct] --> B[S1 Multi-modal DPO<br/>Align Instructions + Timbre Ref]
    B --> C[S2 Decoupling GRPO<br/>Multi-objective Reward to Split Style/Timbre/Content]
    C --> D[S3 Instruction GRPO<br/>ALM Reward to Generalize to Open Instructions]
    D --> E[FlexiVoice]
```

### Key Designs

**1. FlexiVoice-Instruct Dataset: Reverse-Inferring Natural Instructions with LLMs**: To endow the model with basic instruction-following capabilities during pre-training, a large-scale, style-diverse instruction-speech paired dataset is required. The authors constructed a 4,316-hour dataset by cleverly leveraging text metadata instead of direct audio analysis. Since Emilia's audio comes from video platforms and podcasts with titles and tags, Deepseek-V3 was used to infer speaking styles and scenarios from this metadata combined with transcriptions, generating open-ended instructions (e.g., "A documentary narration about a botanical garden, with a calm and educational tone"). Additionally, dubbing data from two popular games were introduced; since speaking styles are strongly tied to character personas, the LLM could identify well-known characters and capture their signature styles. The process included an "information value check"—letting the LLM assess whether metadata was sufficient to infer style, filtering out noisy samples like URLs or metadata-transcript conflicts.

**2. S1 Multi-modal DPO: Establishing Initial Alignment for Instructions and Timbre**: The first stage focuses on emotional instructions to reduce task complexity, with instructions limited to `Use {label} emotion to read it` (Neutral/Happy/Angry/Sad/Surprised). Preference data for emotional tasks are obtained from the ESD dataset—using different emotional versions of the same sentence by the same speaker. Given a target emotion label, the target emotion sentence is the preferred sample $y_w$, a different emotion version of the same sentence is the non-preferred sample $y_l$, and a neutral sample from the same speaker serves as the timbre reference. DPO aligns the model output to the instruction and reference speech without an explicit reward model:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
where $x$ contains the instruction, text, and reference, and both $\pi_\theta$ and $\pi_{\text{ref}}$ are initialized from FlexiVoice-Base.

**3. S2 Decoupling GRPO: Enforcing Separation via Conflict Scenarios**: After DPO, the model can follow emotional instructions with neutral references. However, it still suffers from interference when the reference speech or text carries an emotion that conflicts with the instruction. S2 actively constructs conflict scenarios (e.g., "Happy Instruction vs. Sad Reference") and uses multi-objective GRPO to balance two rewards: a style reward $r_{\text{ser}}\in(0,1)$ provided by Emotion2vec-Large (probability of the target emotion) which punishes style leakage from the reference/text, and a timbre reward $r_{\text{sv}}\in\{0,1\}$ provided by CAM++ speaker verification to ensure speaker consistency. The joint advantage is the sum of the two z-score normalized rewards:
$$A^i_{\text{emo}} = \frac{r^i_{\text{ser}} - \text{mean}(r_{\text{ser}})}{\text{std}(r_{\text{ser}})} + \frac{r^i_{\text{sv}} - \text{mean}(r_{\text{sv}})}{\text{std}(r_{\text{sv}})}$$
To maximize total reward, the model is forced to decouple style, timbre, and content.

**4. S3 Instruction GRPO: Generalizing to Open Instructions via Audio Language Model Rewards**: The final stage extends capabilities from simple emotions to complex real-world instructions. Since preference pairs cannot be constructed at this scale, GRPO is applied directly. To judge the consistency between open instructions and speech, the authors use the open-source Kimi-Audio-7B-Instruct as a reward model, prompting it to output yes/no on "whether the generated speech matches the instruction," mapped to $r_{\text{llm}}\in\{1,0\}$. This stage uses only instruction + text (dropping the reference to avoid training instability caused by conflicts like gender constraints). To prevent catastrophic forgetting, a small portion of S2 data is mixed in, resulting in multi-task multi-objective optimization:
$$A^i_{\text{ins}} = \frac{r^i_{\text{llm}} - \text{mean}(r_{\text{llm}})}{\text{std}(r_{\text{llm}})}, \quad A^i = \begin{cases} A^i_{\text{emo}} & \text{S2 input} \\ A^i_{\text{ins}} & \text{S3 input} \end{cases}$$

## Key Experimental Results

### Main Results Table

Multi-modal controllability and decoupling evaluation (Internal set, EN, TR=Text+Reference Voice, Hard denotes conflicting emotions in reference):

| Model | TO-Easy ACC-I↑ | TO-Hard ACC-I↑/ACC-T↓ | TR-Hard ACC-I↑/ACC-R↓/SV↑ |
|------|------|------|------|
| VoxInstruct | 70.6 | 17.8 / 41.2 | 23.9 / 0.80 / 90.6 |
| CosyVoice2 | - | - | 14.4 / 0.84 / 99.8 |
| FlexiVoice-Base | 72.4 | 39.4 / 30.6 | 32.2 / 0.78 / 99.4 |
| **FlexiVoice** | **97.4** | **89.4 / 6.6** | **78.2 / 10.6 / 95.8** |

Complex instruction following (InstructTTSEval, Avg. Macro Accuracy):

| Model | EN Avg. | ZH Avg. |
|------|------|------|
| Gemini-pro (Closed) | 80.3 | 84.8 |
| GPT-4o-mini-TTS (Closed) | 68.5 | 51.1 |
| MiMo-Audio-7B-Instruct | 72.6 | 70.5 |
| VoxInstruct | 50.4 | 47.5 |
| FlexiVoice-Base | 66.4 | 58.4 |
| **FlexiVoice** | **79.3** | **70.8** |

### Ablation Study Table

Different training sequences and strategies (EN):

| Training Strategy | Decoupling Avg. | InstructTTSEval Avg. |
|------|------|------|
| FlexiVoice-Base | 54.9 | 66.4 |
| + S3 (Direct open instructions) | 54.7 | 72.3 |
| + S3→S1→S2 (Wrong order) | 84.4 | 74.8 |
| + S1→S2+S3 (Joint training) | 84.1 | 75.5 |
| + S1 | 83.3 | 69.0 |
| + S1→S2 | 88.5 | 71.7 |
| **+ S1→S2→S3 (Full PPT)** | **88.7** | **79.3** |

### Key Findings
- **Comprehensive Three-Modality Decoupling**: FlexiVoice increases ACC-I from 39.4 (Base) to 89.4 and decreases ACC-T from 30.6 to 6.6 in the TO-Hard setting. The gap between easy/hard in Chinese is only 1.4%, indicating that style is almost entirely determined by the instruction rather than the text content or reference emotion.
- **Training Order is Essential**: Applying S3 directly yields only 72.3, while the incorrect S3→S1→S2 sequence causes catastrophic forgetting (74.8). S1 DPO serves as a necessary "cold start" foundation to establish robust multi-modal responses before the model can handle subsequent decoupling and generalization.
- **Progressive Superior to Joint**: The gradient directions of S2 (fixed classifier to suppress style leakage) and S3 (ALM reward to encourage open style generalization) conflict. Joint training (75.5) leads to mutual interference, whereas phased training (79.3) allows both to be learned effectively.
- **Expected Style-Timbre Trade-off**: The SV of Base is occasionally slightly higher than FlexiVoice because Base directly clones the reference prosody (high similarity but low ACC-I). To satisfy style instructions, FlexiVoice must modify acoustic features like pitch and energy, causing a slight drop in speaker embedding cosine similarity while maintaining high SV and significantly better instruction following.
- **Intelligibility and Audio Quality**: WER/CER increased slightly compared to Base (due to ASR degradation on highly expressive speech), but Q-MOS was higher (4.08 vs 3.72 in EN-TO-Easy), and CMOS achieved +0.9, showing better human auditory preference.

## Highlights & Insights
- **Redefining "instruction following" as a "decoupling" problem** is the core perspective shift of this work. It points out that simple conditioning cannot overcome modal conflicts; rewards must be used to actively separate style/timbre/content, and a z-score normalized multi-objective advantage function must balance instruction following and timbre preservation.
- **Solid curriculum design for RL post-training**: DPO for cold-start alignment, Decoupling GRPO for hard constraints in conflict scenarios, and Instruction GRPO with ALM rewards for generalization. The three-stage sequence is strictly validated by ablation rather than being arbitrary.
- **Pragmatic data construction**: Instead of analyzing audio, the authors use "cheap" text metadata (titles/tags/character names) to let LLMs reverse-infer styles. Combined with information value filtering, this obtained natural and diverse instruction labels at low cost.
- **Leveraging open-source reward models** (Emotion2vec, CAM++, Kimi-Audio) instead of expensive Gemini evaluations makes GRPO affordable for large-scale instruction sets.

## Limitations & Future Work
- **Reward model quality is the ceiling**: S3 relies on Kimi-Audio's binary yes/no judgment as a reward. Its discriminative power determines the generalization limit for open instructions, and binary rewards provide less information than continuous scores.
- **Style remains dominated by emotion**: Both S1 and S2 revolve around five emotion labels and ESD-like paired data. Decoupling non-emotional attributes like speed or pitch is achieved indirectly through S3, leaving room for improvement in fine-grained continuous attribute control.
- **Gap with closed-source models persists**: The score of 70.8 on Chinese InstructTTSEval still lags behind the Gemini series (84+). Cross-lingual consistency and complex role-playing still have gaps.
- **Small sacrifice in timbre similarity**: While explained as "necessary acoustic modification," its acceptability in scenarios requiring extremely high fidelity (e.g., voice doubles) still needs evaluation.

## Related Work & Insights
- **Two paths for controllable TTS**: Zero-shot TTS (CosyVoice2, IndexTTS2, etc.) clones timbre but has weak style control; instruction-based TTS (PromptTTS, PromptStyle, Parler-TTS, VoxInstruct, AudioBox, ControlSpeech) uses natural language for style but lacks robust decoupling. FlexiVoice attempts a unified framework for both.
- **Instruction-speech datasets**: TextrolSpeech, Audiobox, Parler-TTS, SpeechCraft, ParaSpeechCaps, etc., often use templated descriptions from single sources. FlexiVoice-Instruct emphasizes more natural and diverse annotations.
- **RL in Speech Synthesis**: INTP (DPO for intelligibility), Emo-DPO (emotional preference alignment), Vevo2 (multi-objective post-training for intelligibility and prosody). FlexiVoice differs by using a progressive curriculum to explicitly tackle modal decoupling and complex instruction following.
- **Insight**: When multiple control factors are acoustically entangled, "active construction of conflict scenarios + multi-objective reward constraints" is a more effective decoupling method than simple conditioning. This approach can be transferred to attribute decoupling in image/video generation and general alignment paradigms for multi-modal controllable generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reframing instruction following in TTS as a style-timbre-content decoupling problem and designing a three-stage RL curriculum is original in both perspective and methodology.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes both an internal decoupling evaluation and InstructTTSEval, covers Chinese and English, and features comprehensive objective and subjective metrics. The ablation studies on training order and joint vs. progressive training are very complete.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem naming (Style-Timbre-Content Conflict), coherent logic between motivation and method, with appropriate formulas and diagrams.
- **Value**: ⭐⭐⭐⭐ Provides a reproducible decoupling paradigm for the hot topic of zero-shot instruction-based TTS. Using open-source reward components lowers the barrier to entry, offering direct reference value for industrial-grade controllable speech synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](../../ACL2025/audio_speech/controlspeech_zero_shot.md)
- [\[ICLR 2026\] From Natural Alignment to Conditional Controllability in Multimodal Dialogue](from_natural_alignment_to_conditional_controllability_in_multimodal_dialogue.md)
- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)

</div>

<!-- RELATED:END -->
