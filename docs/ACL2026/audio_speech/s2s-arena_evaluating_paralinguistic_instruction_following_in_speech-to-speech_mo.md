---
title: >-
  [Paper Note] S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models
description: >-
  [ACL2026][Audio & Speech][Speech-to-Speech] S2S-Arena proposes a benchmark to evaluate S2S models directly in the speech modality. Using a four-level paralinguistic interaction protocol, 1,243 speech samples, and 1…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Speech-to-Speech"
  - "Paralinguistic Information"
  - "Arena Evaluation"
  - "Elo"
  - "Instruction Following"
date: 2026-05-08
content_hash: 55f24fe6938ae934
---

# S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models

**Conference**: ACL2026  
**arXiv**: [2503.05085](https://arxiv.org/abs/2503.05085)  
**Code**: https://github.com/FreedomIntelligence/S2S-Arena  
**Area**: Speech Interaction / Speech-to-Speech Models / Evaluation Benchmark  
**Keywords**: Speech-to-Speech, Paralinguistic Information, Arena Evaluation, Elo, Instruction Following

## TL;DR
S2S-Arena proposes a benchmark to evaluate S2S models directly in the speech modality. Using a four-level paralinguistic interaction protocol, 1,243 speech samples, and 1,001 pairwise comparisons, it reveals significant gaps in current systems regarding complex tone, emotion, speaking style, and expressive control.

## Background & Motivation
**Background**: LLMs have driven speech-to-speech systems from ASR→LLM→TTS cascades toward more unified speech interaction models. Existing models typically include a speech encoder, LLM backbone, and speech decoder; representative systems include GPT-4o-realtime, Qwen2.5-Omni, GLM-4-Voice, Kimi-Audio, LLaMA-Omni, and Mini-Omni.

**Limitations of Prior Work**: Many speech benchmarks still transcribe model outputs into text for evaluation or focus solely on speech understanding tasks. This process discards paralinguistic information such as prosody, emotion, speaker traits, and speaking style, which are critical for natural, empathetic, and context-aware S2S interaction.

**Key Challenge**: Real-world speech interaction requires both semantic correctness and the ability for the model to perceive input tone while expressing appropriate speech attributes in the output. Text-based evaluation can measure semantics but struggles to assess whether the model "sounds human, uses the correct tone, and expresses according to instructions."

**Goal**: The authors aim to establish a speech-native benchmark for pairwise comparison of S2S models at both the speech input and speech output levels, systematically evaluating semantic understanding and paralinguistic expressive capabilities.

**Key Insight**: S2S-Arena designs a four-level interaction protocol that increases in difficulty from pure semantic instructions to full paralinguistic interaction. It expands data using human seeds and speech-native self-instruction while utilizing Gemini 2.5-Pro as an automatic speech judge consistent with human preferences.

**Core Idea**: The evaluation of S2S is upgraded from "whether the transcribed text is correct" to "whether the speech interaction itself satisfies semantic and paralinguistic instructions," using arena Elo rankings for continuous model comparison.

## Method
The contribution of S2S-Arena is an evaluation framework rather than a new model. It defines task hierarchies, constructs speech samples, validates automatic judges, and performs speech-native pairwise evaluations of several S2S systems.

### Overall Architecture
On the data side, the authors first design a four-level S2S interaction protocol, then organize 19 representative tasks across four domains: education, entertainment, social, and medical consultation. Human seed data consists of scripts, dubbing, recordings, and high-quality corpora, quality-checked by four native Chinese speaking annotators. Augmented data is generated via GPT-4o for scripts and synthesized by controllable TTS systems like Doubao-TTS, AudioX, and Parler-TTS, resulting in 1,243 total speech samples.

On the evaluation side, the system does not transcribe outputs into text. Instead, it concatenates the spoken instruction with the spoken responses of two candidate models into an audio input for the judge to compare which one better satisfies the instructions. Criteria include instruction alignment, paralinguistic expressiveness, and output audio quality. Elo scores are updated based on the results.

### Key Designs
1. **Four-level Paralinguistic Interaction Protocol**:

    - **Function**: Decomposes S2S capabilities into four categories of scenarios with increasing difficulty to locate model bottlenecks.
    - **Mechanism**: L1 focuses on semantic instruction execution. L2 requires perceiving paralinguistic cues (age, emotion, style) from input speech to adjust semantic responses. L3 allows neutral input but requires the output to express specific speed, emotion, or style per instruction. L4 requires both perceiving input paralinguistic cues and generating matching expressions.
    - **Design Motivation**: Many models perform well at L1 but lag significantly in L3/L4 expressive generation and full interaction. The hierarchical protocol reveals these capability gaps.

2. **Two-stage Data Construction**:

    - **Function**: Balances quality and scale by combining high-quality human-controlled samples with automatically augmented diverse tasks.
    - **Mechanism**: The Seed part contains 293 samples covering 19 tasks. The Augment part uses few-shot self-instruction to generate 950 samples, expanding to over 100 tasks. Random human validation showed 90% difficulty level consistency and 93% paralinguistic consistency.
    - **Design Motivation**: Purely manual collection is costly, while fully automatic generation may drift. Seed + self-instruction maintains task structure while creating sufficiently diverse speech inputs.

3. **Speech-native Arena Evaluation**:

    - **Function**: Compares S2S models without relying on reference answers or text transcriptions.
    - **Mechanism**: All models start with an initial Elo of 1000, updated via the standard Elo formula after each pairwise comparison. Model pairings are not uniformly sampled; they bias toward pairs with moderate rating gaps to avoid trivial or overly subtle comparisons.
    - **Design Motivation**: Speech generation quality often lacks a unique reference answer; pairwise preference more closely resembles real user choices. Elo also facilitates the continuous addition of new models.

### Loss & Training
This work is an evaluation benchmark and does not train the evaluated models. During automatic judge validation, the authors compared 19 human annotators with Gemini 2.5-Pro and Qwen2.5-Omni on the Seed set. Gemini 2.5-Pro demonstrated higher human consistency and was used for large-scale Augment evaluation. Elo updates use $K=32$, and match results are strictly win/loss without ties.

## Key Experimental Results

### Main Results
The consistency between automatic judges and humans was validated first. Gemini 2.5-Pro significantly outperformed Qwen2.5-Omni and was used for subsequent large-scale rankings.

| Automatic Judge | Cohen's kappa | Agreement | Note |
|----------|---------------|-----------|------|
| Gemini 2.5-Pro | 0.6553 | 82.87% | High human consistency |
| Qwen2.5-Omni | 0.4667 | 73.15% | Lower consistency |

The authors then conducted 1,001 pairwise comparisons across 10 S2S systems. Industrial models lead overall, while academic models show larger gaps in complex paralinguistic tasks.

| Model | Elo | Win Rate | W/L | Matches | Observation |
|------|-----|----------|-----|---------|------|
| Qwen 2.5-Omni | 1246.1 | 59.0% | 134/93 | 227 | Rank 1 in total Elo |
| GPT-4o-realtime | 1239.2 | 65.7% | 140/73 | 213 | Most wins, reliable semantics |
| Doubao | 1231.9 | 67.9% | 133/63 | 196 | Highest win rate, strong expressiveness |
| GLM-4-Voice | 1148.2 | 58.3% | 119/85 | 204 | Upper-middle tier |
| FunAudioLLM | 1088.3 | 51.0% | 128/123 | 251 | Strong in entertainment/social scenarios |
| Kimi-Audio | 1056.7 | 49.3% | 142/146 | 288 | Middle tier |
| LLaMA-Omni | 908.7 | 44.4% | 68/85 | 153 | Best academic system relative to industrial ones |
| Mini-Omni2 | 727.4 | 33.1% | 59/119 | 178 | Insufficient complex expressiveness |
| SpeechGPT | 677.1 | 27.3% | 42/112 | 154 | Lower rank |
| Mini-Omni | 676.4 | 26.1% | 36/102 | 138 | Lower rank |

### Ablation Study
Rather than traditional model ablation, this study analyzes differences in system capabilities through task categories and difficulty levels.

| Model | Education | Entertainment | Medical | Social | Average | Conclusion |
|------|-----------|---------------|---------|--------|------|------|
| GPT-4o-realtime | 1230.2 | 1166.8 | 1124.4 | 1056.6 | 1144.5 | Strong knowledge tasks |
| Doubao | 1214.5 | 1144.6 | 1055.7 | 1133.0 | 1136.9 | Strong expression/naturalness |
| Qwen 2.5-Omni | 1096.7 | 1097.0 | 1056.0 | 1155.9 | 1101.4 | Highest in social scenarios |
| FunAudioLLM | 999.3 | 1105.9 | 876.2 | 1123.3 | 1026.2 | Ent./Social better than medical |
| LLaMA-Omni | 922.3 | 1004.6 | 948.3 | 913.6 | 947.2 | Strongest academic model |

| Model | L1 | L2 | L3 | L4 | Average | Structural Observation |
|------|----|----|----|----|------|----------|
| GPT-4o-realtime | 1064.4 | 1199.2 | 1241.7 | 1071.3 | 1144.2 | Strong high-difficulty expression |
| Doubao | 1029.5 | 1163.7 | 1148.2 | 1205.8 | 1136.8 | Strongest in L4 full interaction |
| Qwen 2.5-Omni | 1072.2 | 1109.1 | 1136.2 | 1123.0 | 1110.1 | Stable via Whisper-large + flow matching |
| LLaMA-Omni | 977.7 | 965.2 | 920.2 | 942.4 | 951.4 | Stable L1, falls behind in L3/L4 |
| Mini-Omni | 985.8 | 803.0 | 769.8 | 835.7 | 848.6 | Small backbone/encoder limits paralinguistics |

### Key Findings
- Industrial systems lead overall but in different ways: Qwen 2.5-Omni has the highest total Elo, GPT-4o-realtime has the most wins, and Doubao has the highest win rate and excels at L4.
- The gap between academic and industrial systems widens as task difficulty increases. L1 instruction following gaps are not extreme, but in L3/L4 paralinguistic expression and full interaction, the gap can exceed 300 Elo.
- Architectural factors are crucial: strong backbones benefit semantic instructions, stronger encoders like Whisper-large benefit paralinguistic perception, and flow-matching speech decoders are vital for expressive generation.

## Highlights & Insights
- This paper addresses a blind spot in S2S evaluation: a speech model must not only be correct after transcription but must also "speak in the appropriate manner." This shift is critical for next-generation assistants.
- The four-level protocol is highly diagnostic. it distinguishes whether a model understands semantics, understands emotion, controls output style, or completes full paralinguistic interaction.
- The Arena format is suitable for open-ended speech output. Since many expressive qualities lack a unique reference answer, pairwise preference is closer to user experience than BLEU, WER, or text-LLM judges.
- The model analysis identifies technical route differences: semantic capability, acoustic perception, and generation decoders each affect different levels, providing more info than a simple leaderboard.

## Limitations & Future Work
- The scale of 1,243 samples is relatively small compared to real speech interaction spaces, and augmented data relies on high-quality synthetic speech, which might bias toward models adapted to that distribution.
- Current coverage focuses on utterance-level and short-range interactions, lacking long-term persona consistency, emotional shifts, and multi-turn discourse coherence.
- While automatic judges show high human consistency, they may still exhibit model preferences, voice quality biases, or linguistic/accent biases, requiring continuous calibration.
- Speech evaluation involves potential misuse and privacy risks; the paper uses anonymization and controlled settings, but future open benchmarks will need clear data licensing and safety boundaries.

## Related Work & Insights
- **vs Dynamic-SUPERB / AudioBench / MMAU**: These focus on speech or audio understanding; S2S-Arena evaluates both semantic understanding and paralinguistic expression in speech output.
- **vs VoiceBench / SD-Eval / Voxdialogue**: These are closer to dialogue evaluation but rely heavily on text-based metrics; S2S-Arena compares directly in the speech modality.
- **vs Vstyle / AIR-Bench / Multivox**: The latter focus on style or speech generation, whereas S2S-Arena systematically designs L1-L4 difficulty levels and supports continuous ranking via Arena/Elo.
- **Insights for Model Development**: Improving the LLM backbone is insufficient; S2S systems require stronger speech encoders to capture paralinguistic signals and more controllable speech decoders to express emotion, rhythm, and style.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Innovation lies in evaluation design; the four-level protocol and speech-native arena are highly targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 10 models, 1,001 comparisons, and multi-dimensional analysis are substantial, though sample scale and long-range interaction are limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with dense tabular information; qualitative case analysis helps interpret the leaderboard.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for S2S model evaluation, driving the community from text correctness toward speech interaction quality and human alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction](../../ICLR2026/audio_speech/paras2s_benchmarking_and_aligning_spoken_language_models_for_paralinguistic-awar.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)

</div>

<!-- RELATED:END -->
