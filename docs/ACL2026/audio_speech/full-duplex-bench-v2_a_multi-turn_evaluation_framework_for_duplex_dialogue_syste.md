---
title: >-
  [Paper Note] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner
description: >-
  [ACL 2026][Audio & Speech][Full-duplex dialogue] The authors propose Full-Duplex-Bench-v2, where an Automated Examiner powered by GPT-Realtime conducts real-time dialogues with full-duplex models via WebRTC. Performance…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Full-duplex dialogue"
  - "multi-turn evaluation"
  - "LLM-as-judge"
  - "WebRTC streaming orchestration"
  - "turn-taking"
date: 2026-05-08
content_hash: 0bc5dd52264d270a
---

# Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner

**Conference**: ACL 2026  
**arXiv**: [2510.07838](https://arxiv.org/abs/2510.07838)  
**Code**: <https://github.com/DanielLin94144/Full-Duplex-Bench>  
**Area**: Dialogue / Full-Duplex Speech / Evaluation Benchmark  
**Keywords**: Full-duplex dialogue, multi-turn evaluation, LLM-as-judge, WebRTC streaming orchestration, turn-taking

## TL;DR
The authors propose Full-Duplex-Bench-v2, where an Automated Examiner powered by GPT-Realtime conducts real-time dialogues with full-duplex models via WebRTC. Performance is evaluated across four task categories (Daily/Correction/Entity/Safety) and two pacing modes (Fast/Slow) based on turn-taking, instruction-following, and task-specific metrics. Findings reveal that systems like GPT-Realtime, Moshi, and Freeze-Omni exhibit performance degradation as dialogues progress, with open-source models performing poorly in correction and entity tracking.

## Background & Motivation
**Background**: Traditional spoken dialogue systems are half-duplex—one party speaks only after the other finishes. While simple, this results in high latency and unnatural interactions. Recently, several full-duplex solutions have emerged: cascaded approaches (ASR+LLM+TTS with FSM, e.g., MiniCPM-Duplex) and end-to-end models (dGSLM, SyncLLM, Moshi, NTPP, SCoT). These models theoretically approximate human conversational speed by "listening while speaking."

**Limitations of Prior Work**: ① Human Evaluation: Natural but expensive and non-reproducible; ② Corpus-level Statistics (pause, floor-transfer offset): Scalable but lacks semantic context; ③ Classifiers (Talking Turns): Automated but restricted by training data generalization; ④ Existing Full-Duplex-Bench v1/v1.5: The first streaming benchmarks, but limited to single-turn, scripted scenarios covering "instantaneous" behaviors like pause, interrupt, and backchannel.

**Key Challenge**: Real human conversation is multi-turn. Success depends not just on a single turn-taking event but on maintaining contextual consistency, task progression, and information retrieval across multiple exchanges. Most existing benchmarks are confined to single turns; the ability of full-duplex models to handle multi-turn interactions has not been systematically quantified.

**Goal**: ① Advance evaluation from "scripted single-turn" to "realistic multi-turn streaming"; ② Maintain naturalism without relying on human evaluation (the Examiner improvises interruptions, follows up, and adjusts based on the examinee's responses); ③ Propose metrics to distinguish turn-taking, instruction-following, and task-specific competence.

**Key Insight**: The authors found that GPT-Realtime itself is a stable, low-latency speech model capable of strict role-play, making it suitable as an "Automated Examiner." This bypasses the bottleneck of utilizing humans as dialogue partners and scales multi-turn evaluation from "slow and expensive" to "batch reproducible."

**Core Idea**: Utilize a spoken-LM Examiner + a WebRTC orchestrator + staged multi-turn task scripts + an LLM-as-judge scorer to form the first streaming-native multi-turn full-duplex evaluation framework.

## Method

### Overall Architecture
The FDB-v2 pipeline consists of three components: ① **Examiner** (a gpt-realtime driven speech model that advances dialogue based on staged goals and interrupts when necessary); ② **Orchestrator** (manages two WebRTC peer connections, enforcing bidirectional transmission in 48 kHz, 16-bit, mono PCM, and strict 10 ms frames as the canonical wire format); ③ **Evaluatee** (the model under test, integrated via an adapter that converts internal audio streams to the canonical format). Each session begins with the Examiner, progresses through pre-orchestrated sub-goals, and ends with a fixed closing statement. Dual-track recordings are saved, transcribed via Parakeet-TDT ASR, and scored by Gemini-2.5-flash acting as the judge.

### Key Designs

1. **Stepwise Semantic Goals + Dual Examiner Pacing**:
    - Function: Transforms multi-turn dialogue from arbitrary chat into a structured process with verifiable sub-goals, while exposing failure modes through pacing shifts.
    - Mechanism: Each scenario is divided into steps with explicit semantic goals; the Examiner advances only when the goal is met, otherwise rephrasing or probing. Pacing is set to **Fast** (Examiner interrupts, adds backchannels, and switches stages immediately upon completion) or **Slow** (Examiner intervenes only after the examinee stops or pauses for a long duration).
    - Design Motivation: Fast pacing tests coordination in turn-taking (handling interruptions), while Slow pacing tests endurance in memory and entity tracking (more time but prone to drift). Dual pacing isolates "scheduling issues" from "memory issues."

2. **Standardized Streaming Interface (Adapter–Orchestrator–Adapter)**:
    - Function: Enables any full-duplex system (closed-source APIs, open-source checkpoints, future models) to connect via an adapter without framework modifications.
    - Mechanism: The Orchestrator enforces a canonical wire format (48 kHz, 16-bit, mono PCM, 10 ms frames = 960 bytes) pushed at a stable cadence; the adapter normalizes model-specific outputs, slices/packs into 10 ms frames, and pads silence for buffer under-runs.
    - Design Motivation: Standardizing the audio interface overcomes the engineering hurdle of disparate protocols (WebSocket, RTSP, SDK callbacks), allowing the framework to evolve independently of specific model implementations.

3. **Four Task Families + Three-Dimensional LLM-as-judge Scoring**:
    - Function: Covers daily dialogue, self-correction, cross-turn reference, and safety refusal, providing automated scores aligned with human judgment.
    - Mechanism: Tasks include **Daily** (reservations, planning), **Correction** (cross-turn self-correction, e.g., "I want a cold coffee" → "Oh, make it hot"), **Entity Tracking** (ordinal/attribute/landmark references), and **Safety** (11 policy categories). The Gemini judge outputs three scores: Turn-Taking Fluency (1-5/event), Instruction Following (1-5/event), and Task-Specific Metric (1-5/dialogue).
    - Design Motivation: Separating metrics distinguishes "fluent but irrelevant" from "accurate but stuttering" failure modes. Task-specific metrics (e.g., reference consistency for Entity Tracking) ensure comparable total scores across different families.

### Loss & Training
FDB-v2 is an evaluation framework and does not include training. The scoring side uses Gemini-2.5-flash-preview-09-2025, following the findings of Chang et al. 2025 that Gemini correlates highly with humans in turn-taking evaluation. ASR uses Parakeet-TDT-0.6B-v2 for time-aligned transcription. The Examiner consistently uses gpt-realtime to ensure zero variance at the Examiner end during cross-model testing.

## Key Experimental Results

### Main Results

| Pacing | System | Correction | Entity | Safety |
|------|------|-----------|--------|--------|
| Fast | Freeze-Omni | 2.74 | 2.62 | 3.94 |
| Fast | Moshi | 2.88 | 2.76 | 3.67 |
| Fast | GPT-Realtime | **4.02** | **4.51** | **4.44** |
| Slow | Freeze-Omni | 3.50 | 2.86 | 4.27 |
| Slow | Moshi | 3.46 | 3.84 | 3.51 |
| Slow | GPT-Realtime | 3.94 | 4.12 | **4.53** |

GPT-Realtime maintained ≥4.0 in all Fast tasks, while open-source models scored <3.0 in Fast Correction and Entity tasks. Slow pacing provided significant relief for open-source models (Moshi Entity +1.08, Freeze-Omni Correction +0.76).

### Ablation Study

| Metric | Krippendorff $\alpha$ | Pearson $r$ |
|------|---------------------|-------------|
| Turn-Taking Fluency | 0.6143 | 0.6137 |
| Instruction Following | 0.6833 | 0.6807 |
| Correction Handling | 0.5879 | 0.5877 |
| Entity Tracking | 0.6383 | 0.6330 |
| Safety | 0.6931 | 0.6914 |

On 120 sessions, the LLM judge correlation with humans reached $r\in[0.59, 0.69]$, with the strongest correlation in Safety and IF, and the weakest in Turn-Taking (where human evaluators also diverged on "natural timing").

### Key Findings
- **All systems degrade over time**: Tracking TT/IF in 15-second bins showed TT drifting slowly while IF often dropped sharply, indicating that long-term robustness is a common weakness in current full-duplex models.
- **Pacing is a diagnostic signal**: Slow pacing allowed GPT-Realtime and Moshi to "recover," improving Entity IF by 0.5-1.0; Fast pacing exposed Freeze-Omni's lack of recovery capability.
- **Task difficulty ranking**: Entity was easiest (explicit references grounded the model), while Daily and Correction were hardest (dependent on cumulative memory, where small errors snowball).
- **Closed vs. Open-source gap**: GPT-Realtime averaged 4.32 in Fast mode, while Moshi and Freeze-Omni both averaged 3.10, showing that open-source models are not yet ready for commercial multi-turn deployments.

## Highlights & Insights
- **Spoken-LM as Examiner is a paradigm shift**: Moving from rigid scripts or expensive humans to a stable speech model preserves dialogue dynamics (interruptions, probes) while ensuring reproducibility.
- **Canonical wire format + adapter mode**: Treating the audio protocol as a public interface (10 ms frames / 48 kHz) makes the framework implementation-neutral—similar to how OpenAI Gym standardized RL environments.
- **Dual pacing to isolate Turn-Taking vs. Memory defects**: The Fast/Slow design provides a diagnostic breakdown (e.g., "is it a timing issue or a memory loss?"), offering high value for industrial deployment.
- **Three-dimensional scoring (TT/IF/Task-specific)**: Disentangles fluency from accuracy, preventing a single total score from masking specific failure modes.

## Limitations & Future Work
- Task coverage (4 types) and pacing (2 levels) remain limited; it does not cover negotiation, education, or complex safety sub-domains.
- No reward for audio-expressive behaviors (emotive prosody, active-listening cues), which may result in under-expressed micro-timing.
- English only; excludes multi-lingual code-switching and cultural variations in overlap patterns.
- Automated Examiner + LLM judge introduces prompt sensitivity and model bias; Turn-Taking correlation at 0.61 suggests this dimension is not yet "fully solved."
- Small sample size of evaluated systems (GPT-Realtime / Moshi / Freeze-Omni); hasn't yet tested newer end-to-end models like NTPP or SCoT.

## Related Work & Insights
- **vs Full-Duplex-Bench v1/v1.5 (Lin et al. 2025a/b)**: Previous versions focused on single-turn instantaneous behaviors; v2 scales to staged goals and conversational endurance.
- **vs Talking Turns (Arora et al. 2025b)**: They use trained classifiers for turn-change detection; Ours uses a spoken-LM partner to generalize to any task while assessing IF and competence.
- **vs Chang et al. 2025 (Game-Time)**: This work inherits the conclusion that Gemini is a valid turn-taking judge and extends it to multi-task scoring.
- **vs MultiWOZ / SLURP**: These are text-based multi-turn benchmarks; FDB-v2 is the first to merge streaming, full-duplex, and multi-turn into one framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First streaming multi-turn full-duplex framework; spoken-LM Examiner paradigm is robust.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 systems × 2 pacings × 4 tasks with human alignment, though more models are needed.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and transparent limitations.
- Value: ⭐⭐⭐⭐⭐ Provides a reproducible testbed for the community and standardizes streaming engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ICML 2026\] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](../../ICML2026/audio_speech/the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)
- [\[ICML 2026\] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](../../ICML2026/audio_speech/moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)
- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[ACL 2026\] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)

</div>

<!-- RELATED:END -->
