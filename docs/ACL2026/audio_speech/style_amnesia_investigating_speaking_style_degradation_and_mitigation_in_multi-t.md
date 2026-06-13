---
title: >-
  [Paper Note] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
description: >-
  [ACL 2026][Audio & Speech][Spoken Language Model] Discovers that Spoken Language Models (SLMs) fail to maintain initially specified speaking styles (emotion, accent, volume, speaking rate) in multi-turn dialogues…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Spoken Language Model"
  - "Style Amnesia"
  - "Multi-turn Conversation"
  - "Speaking Style"
  - "Instruction Following"
date: 2026-05-08
content_hash: 02efa781d39f8942
---

# Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models

**Conference**: ACL 2026  
**arXiv**: [2512.23578](https://arxiv.org/abs/2512.23578)  
**Code**: [GitHub](https://github.com/YuXiangLin1234/SLM-Style-Amnesia)  
**Area**: Spoken Language Models  
**Keywords**: Spoken Language Model, Style Amnesia, Multi-turn Conversation, Speaking Style, Instruction Following

## TL;DR

Discovers that Spoken Language Models (SLMs) fail to maintain initially specified speaking styles (emotion, accent, volume, speaking rate) in multi-turn dialogues, a phenomenon termed "Style Amnesia." Through attention analysis, this work reveals the cause (attention decay) and proposes an explicit recall process as a mitigation method.

## Background & Motivation

**Background**: Spoken Language Models (e.g., GPT-4o, Gemini Live, Qwen2.5-Omni) have demonstrated impressive expressive capabilities by following user-specified speaking styles (emotion, accent, speaking rate, etc.) in single-turn interactions.

**Limitations of Prior Work**: Existing research focuses almost entirely on single-turn evaluation, leaving the maintenance of style consistency in multi-turn dialogues unexplored. In practical applications, users expect the SLM to maintain a style set at the beginning of a conversation without repeating instructions every turn.

**Key Challenge**: SLMs follow style instructions well in the first turn, but the instruction following (IF) rate drops sharply as the conversation progresses. The model does not "forget" the instruction (as shown by recall tests where models accurately repeat instructions) but rather "fails to execute" what it has remembered.

**Goal**: Systematically evaluate and analyze the style maintenance capabilities of SLMs in multi-turn dialogues, identify the underlying causes, and explore mitigation strategies.

**Key Insight**: An end-to-end evaluation framework is constructed using a user simulator for realistic interactive multi-turn dialogues to measure turn-level IF rates.

**Core Idea**: The root cause of style amnesia is attention dilution rather than true memory loss. As dialogue turns increase, the model's attention weight on style instruction tokens decays from ~8% to <0.6%.

## Method

### Overall Architecture

The evaluation framework consists of three core components: (1) Style Instructions—10 types encompassing emotion (sad, happy, angry, neutral), accent (North American, Indian English), volume (high, low), and speaking rate (fast, slow); (2) Dialogue Topics—100 diverse starters selected from the Soda dataset; (3) Multi-turn Interaction—a cascaded SLM (ASR + GPT-5 mini + TTS) serves as a user simulator to conduct 4 turns of interactive dialogue with the evaluated SLM.

### Key Designs

1.  **Turn-Level IF Rate Measurement**:
    - **Function**: Quantify the trends of style following across multi-turn dialogues.
    - **Mechanism**: Define the first-turn following rate $IF_1$ and the degradation rate $D = \sum_{j=2}^{K} \frac{\max(IF_1(s) - IF_j(s), 0)}{K-1}$ to capture baseline capability and the degree of degradation, respectively. Four specialized auto-judges are used: Emotion2vec-Large for emotion, Voxlect for accent, LUFS for volume, and WPM for speaking rate.
    - **Design Motivation**: Unlike aggregated global scores, turn-level analysis precisely reveals when and how degradation develops.

2.  **Attention Dynamics Analysis**:
    - **Function**: Reveal the internal mechanism of style amnesia.
    - **Mechanism**: Extract the average attention weights for style instruction tokens from an open-source model (Step-Audio 2 mini) during response generation. Results show weights of ~8.3% in turn 1, ~1.6% in turn 2, ~0.87% in turn 3, and ~0.58% in turn 4. This severe attention dilution aligns closely with the IF rate degradation.
    - **Design Motivation**: Distinguish between "forgetting instructions" and "failing to execute"—memory issues could be solved via prompt engineering, whereas attention dilution requires architectural improvements.

3.  **Recall Process**:
    - **Function**: Explore methods to mitigate style amnesia.
    - **Mechanism**: Before each turn starting from turn 2, the SLM is prompted to recall the initial style instruction before processing the user input. Experiments show most models recall accurately (closed-source models reach nearly 100% recall), and this process significantly reduces degradation (e.g., GPT-4o mini's sad style degradation dropped from 65.3% to 30.3%).
    - **Design Motivation**: Test whether the model still "remembers" the instruction and if explicit recall improves execution.

### Text-Acoustic Joint Analysis

For emotional styles, both semantic and acoustic features suffer from style amnesia—textual content and vocal expression degrade synchronously. For speaking rate, different models adopt different strategies: Gemini Live reduces word count to "speak faster," while GPT-4o uses acoustic acceleration rather than content compression. However, as turns progress, the WPM gap between fast and slow conditions consistently narrows.

## Key Experimental Results

### Main Results

| Model | Style | $IF_1$ (First Turn) | Degradation $D$ |
| :--- | :--- | :--- | :--- |
| GPT-4o mini | Sad | ~85% | 65.3% |
| GPT-4o mini | Indian Accent | ~75% | 49.7% |
| GPT-4o | Sad | ~95% | 26.7% |
| Gemini Live | Sad | ~85% | 21.3% |
| Step-Audio 2 mini | Sad | ~70% | 14.0% |
| Cascaded Baseline (TTS) | All Emotions | ~95% | <3.0% |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Instruction in System Message | $IF_1$ drops 30-80% | System messages are harder to follow. |
| Instruction in User Message | Higher $IF_1$ | Default setting performs better. |
| + Recall Process | $D$ reduced by 3-35% | GPT-4o mini benefits the most. |
| Attention Weight | 8.3% → 0.58% | 14x decay within 4 turns. |

### Key Findings
- All five evaluated models (3 closed-source + 2 open-source) exhibit style amnesia without exception.
- Models "remember" instructions but "cannot execute" them—recall rates are near 100% while IF rates continue to decline.
- **System Message Paradox**: Though designed for global persistence, SLMs follow style instructions in system messages significantly worse.
- **Default Style Bias**: Models tend to revert to default styles such as "happy/neutral" emotions and "North American" accents.
- The cascaded baseline (providing style instructions to TTS every turn) shows almost no degradation, proving the issue lies in the end-to-end SLM architecture.

## Highlights & Insights
- **Identified a critical, previously unnoticed issue**: Style amnesia is a major hurdle for the practical deployment of SLMs.
- **Distinguished "Memory" vs. "Execution"**: Through the recall test, the problem was pinpointed to attention allocation rather than memory, providing a clear direction for solutions.
- **Robust Evaluation Framework**: High reliability is achieved through realistic interactions with simulators, four specialized judges, and human verification.
- **Value of the System Message Paradox**: Reveals deep-seated issues in the architectural design of current SLMs regarding instruction persistence.

## Limitations & Future Work
- **Limited Style Categories**: Only covers four paralinguistic attributes; complex styles like intonation variance or role-playing are not addressed.
- **No Multi-style Combinations**: Current models fail to maintain even a single style, leaving multi-style combinations for future work.
- **Limited Open-source Attention Analysis**: Attention analysis was only performed on one open-source model (Step-Audio 2 mini).
- Future directions: Style-anchored attention mechanisms, persistent representations of style embeddings, and following combinations of multiple styles.

## Related Work & Insights
- **vs. Multi-Bench**: Also evaluates multi-turn SLMs but only aggregates global scores; this work provides turn-level analysis.
- **vs. VocalBench/VoxDialogue**: Uses predefined dialogues instead of real interactions, preventing turn-by-turn analysis.
- **vs. Text LLM Multi-turn Degradation**: Similar performance degradation has been found in the text domain; this work extends the phenomenon to paralinguistic features in the audio domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically reveal SLM style amnesia; discovers the "remember but cannot do" insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 models, 10 styles, and 1000 dialogue sets, including attention analysis and mitigation experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definition, progressive experimentation, and intuitive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Highlights a key bottleneck for SLM utility with clear implications for model design and training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](../../ICLR2026/audio_speech/when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)

</div>

<!-- RELATED:END -->
