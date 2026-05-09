---
title: >-
  [Paper Note] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models
description: >-
  [ACL 2026][LLM Evaluation][Full-duplex speech models] This paper proposes MTR-DuplexBench, a comprehensive multi-round evaluation benchmark for full-duplex speech language models (FD-SLMs). By introducing a novel turn segmentation method, it addresses the challenges of ambiguous turn boundaries and context inconsistency inherent in full-duplex dialogue. The benchmark covers four dimensions: conversational characteristics, dialogue quality, instruction following, and safety. Experiments reveal a consistent performance degradation of existing FD-SLMs across multi-round interactions.
tags:
  - ACL 2026
  - LLM Evaluation
  - Full-duplex speech models
  - multi-round dialogue evaluation
  - turn segmentation
  - conversation quality
  - safety evaluation
date: 2026-05-08
content_hash: bd1e82b538c0d130
---

# MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models

**Conference**: ACL 2026
**arXiv**: [2511.10262](https://arxiv.org/abs/2511.10262)
**Code**: [https://github.com/ZhangHe0918/MTR-DuplexBench](https://github.com/ZhangHe0918/MTR-DuplexBench)
**Area**: Speech Language Models / Evaluation Benchmarks
**Keywords**: Full-duplex speech models, multi-round dialogue evaluation, turn segmentation, conversation quality, safety evaluation

## TL;DR

This paper proposes MTR-DuplexBench, a comprehensive multi-round evaluation benchmark for full-duplex speech language models (FD-SLMs). By introducing a novel turn segmentation method, it addresses the challenges of ambiguous turn boundaries and context inconsistency inherent in full-duplex dialogue. The benchmark covers four dimensions: conversational characteristics, dialogue quality, instruction following, and safety. Experiments reveal a consistent performance degradation of existing FD-SLMs across multi-round interactions.

## Background & Motivation

**Background**: Full-duplex speech language models (FD-SLMs) enable real-time "simultaneous listening and speaking" interactions, supporting complex conversational behaviors such as interruptions and backchannels, and represent the future direction of spoken dialogue systems. Moshi and Freeze-Omni are currently the only two open-source FD-SLMs.

**Limitations of Prior Work**: Existing benchmarks (e.g., Full-Duplex-Bench, Full-Duplex-Bench v1.5) focus primarily on single-turn interaction evaluation, whereas real-world conversations typically unfold over multiple turns. Moreover, most existing benchmarks evaluate only conversational characteristics (e.g., interruptions, backchannels), neglecting critical capabilities such as instruction following and safety. FD-Bench supports multi-round evaluation but is limited to interruption scenarios, and Talking Turns requires costly human data collection.

**Key Challenge**: Full-duplex dialogue evaluation faces two core technical challenges: (1) *Ambiguous turn boundaries*: unlike half-duplex systems, full-duplex communication proceeds spontaneously without explicit turn delimiters; (2) *Context inconsistency*: in multi-round evaluation, model responses in earlier turns may deviate substantially from ground-truth responses, causing subsequent user inputs to diverge from realistic scenarios and undermining evaluation reliability.

**Goal**: To construct a comprehensive FD-SLM evaluation benchmark that supports round-by-round multi-turn assessment across four dimensions: conversational characteristics, dialogue quality, instruction following, and safety.

**Key Insight**: A full-duplex turn segmentation algorithm is designed to decompose continuous full-duplex dialogue into discrete turns. During evaluation of each turn, the assistant channel of all preceding turns is filled with ground-truth responses, simultaneously resolving both the turn boundary ambiguity and context inconsistency problems.

**Core Idea**: Turn boundaries are determined via repeated GPT-4o segmentation with majority voting and clustering-based filtering. Context drift is eliminated through a strategy of "ground-truth responses for prior turns + model inference for the current turn," forming a four-dimensional comprehensive evaluation framework.

## Method

### Overall Architecture

The MTR-DuplexBench pipeline consists of two components: (1) a full-duplex turn segmentation method that decomposes continuous dual-channel audio into discrete user turns and assigns corresponding assistant response intervals; and (2) a four-dimensional evaluation framework designed with dedicated data, procedures, and metrics for conversational characteristics (200 samples × 10 turns), dialogue quality (200 natural conversation segments), instruction following (300 samples × 10 turns), and safety (520 samples × 10 turns).

### Key Designs

1. **Full-Duplex Turn Segmentation Algorithm**:

    - *Function*: Identify the start and end timestamps of user turns from continuous full-duplex dialogue.
    - *Mechanism*: A four-step pipeline — (a) extract transcripts and timestamps from dual-channel audio using Whisper and Silero VAD; (b) sort user and assistant VAD segments chronologically and feed them into GPT-4o for turn segmentation; (c) repeat GPT-4o segmentation six times and aggregate results via majority voting — a new turn is merged with an existing candidate if temporal overlap is ≥30%, with start/end times set as the median; (d) resolve remaining overlaps to finalize turn boundaries. The assistant response interval is defined as the period from the start of the current user turn to the end of the next user turn.
    - *Design Motivation*: Single-pass GPT segmentation yields unstable results; majority voting combined with clustering ensures robust turn boundary estimation. The definition of the response interval guarantees sufficient time for the assistant to complete its response.

2. **Context Consistency Preservation Strategy**:

    - *Function*: Ensure that the input context for each turn during multi-round evaluation remains consistent with realistic scenarios.
    - *Mechanism*: When evaluating turn $k$, the assistant channel for all preceding $k{-}1$ turns is populated with ground-truth speech; only the current turn is generated by the model. This ensures the model faces a "correct" context at every turn, preventing unreliable evaluation caused by error accumulation.
    - *Design Motivation*: If model responses from earlier turns are propagated to subsequent turns, deviations compound progressively, resulting in evaluation scenarios that would never occur in real conversations.

3. **Four-Dimensional Evaluation Framework**:

    - *Function*: Comprehensively assess multiple capabilities of FD-SLMs.
    - *Mechanism*: (a) *Conversational characteristics* — 200 synthetic 10-turn dialogues generated by GPT-4o, evaluating success rate and latency across five behaviors: smooth turn-taking, interruption, pause handling, background speech, and backchannels; (b) *Dialogue quality* — 200 natural 120-second conversation segments from the Candor dataset, evaluated turn-by-turn using GPT-score (0–5) after turn segmentation; (c) *Instruction following* — 300 spoken queries from the Llama Question dataset, evaluated by success rate; (d) *Safety* — 520 harmful queries from AdvBench, evaluated by multi-round rejection rate.
    - *Design Motivation*: Practical deployment of FD-SLMs requires reliable instruction following and safe outputs, particularly under multi-round interruption scenarios.

### Loss & Training

This paper presents an evaluation benchmark and does not involve model training. Evaluation metrics include: binary success rate for conversational characteristics, GPT-score (0–5) for dialogue quality, success rate for instruction following, and rejection rate for safety — all automatically assessed by GPT-4o.

## Key Experimental Results

### Main Results

Success rates for conversational characteristics decline as the number of turns increases (turn 1 vs. average over turns 1–10):

| Model | Smooth Turn-Taking | Interruption | Pause Handling | Background Speech |
|---|---|---|---|---|
| Moshi | 73.0→57.4% | 72.5→54.2% | 93.5→84.8% | 53.0→25.7% |
| Freeze-Omni | 69.0→36.4% | 76.0→56.6% | 89.0→68.5% | 0.5→1.1% |
| VocalNet (HD) | 100→100% | 100→100% | 100→100% | 0→0% |
| Cascaded | 98.5→99.0% | 99.5→96.3% | 100→100% | 0→0% |

### Ablation Study

Performance comparison of multi-characteristic combinations vs. single characteristics (Moshi, average success rate over turns 1–10):

| Configuration | Success Rate | Notes |
|---|---|---|
| Smooth turn-taking only (S) | 57.4% | Single-characteristic baseline |
| S + Interruption (I) | 54.5% | Two alternating characteristics |
| S + I + Pause (P) | 54.3% | Three alternating characteristics |
| S + I + P + Background (B) | 37.6% | Four alternating characteristics; significant drop |

### Key Findings

- FD-SLMs exhibit consistent multi-round degradation: Freeze-Omni's smooth turn-taking drops from 69% to 36%, the most severe decline observed.
- The HD model achieves perfect scores on conversational characteristics (turn-taking/interruption/pause all at 100%) but completely fails to handle background speech.
- Moshi is the only model capable of handling background speech, though its success rate declines from 53% to 25.7%.
- Moshi achieves the lowest latency (approximately 0.6–0.9 s), while the Cascaded system exhibits the highest (approximately 9–12 s).
- Performance further deteriorates under multi-characteristic combinations, indicating that simultaneously handling multiple conversational behaviors poses a greater challenge for FD-SLMs.
- Multi-round interruptions may degrade the safety rejection capability of FD-SLMs, presenting a potential safety risk.

## Highlights & Insights

- **Engineering design of turn segmentation**: The approach of repeated GPT segmentation with majority voting and clustering is highly practical, converting uncertain full-duplex audio streams into deterministic discrete turns. This methodology is transferable to any scenario requiring the extraction of structured events from continuous interactions.
- **Elegance of the context consistency strategy**: Filling historical turns with ground-truth speech to eliminate error accumulation essentially evaluates "single-turn capability under ideal conditions" rather than true multi-round accumulated performance. This is a reasonable trade-off given the current state of model development.
- **Comprehensiveness of the four-dimensional framework**: This work is the first to incorporate instruction following and safety into full-duplex evaluation. The research question of "safety under multi-round interruption" is particularly forward-looking.

## Limitations & Future Work

- Only two open-source FD-SLMs (Moshi and Freeze-Omni) are evaluated; the limited sample size makes it difficult to draw generalizable conclusions.
- While the context consistency strategy ensures evaluation reliability, it does not capture model behavior under realistic multi-round error accumulation.
- Synthetic data for conversational characteristics are generated by GPT-4o and may not fully reflect authentic conversational patterns.
- Safety evaluation relies solely on known harmful queries from AdvBench, providing insufficient coverage of more sophisticated jailbreak attacks.
- Future work could incorporate additional FD-SLMs and explore how benchmark signals can be leveraged during training to improve multi-round consistency.

## Related Work & Insights

- **vs. Full-Duplex-Bench**: Supports only single-turn evaluation without round-by-round analysis; MTR-DuplexBench comprehensively surpasses it in multi-round support and evaluation dimensions.
- **vs. FD-Bench**: Supports up to five turns but focuses exclusively on interruption scenarios and does not enable round-by-round evaluation.
- **vs. Talking Turns**: Requires real human–model interaction for data collection, limiting scalability; MTR-DuplexBench employs an automated pipeline with better reproducibility.

## Rating

- Novelty: ⭐⭐⭐⭐ — First full-duplex benchmark supporting round-by-round multi-turn evaluation and four-dimensional comprehensive assessment; the turn segmentation method is innovative.
- Experimental Thoroughness: ⭐⭐⭐ — Only two genuine FD-SLMs are available for evaluation; experimental scale is constrained by model availability.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and method description is thorough.
- Value: ⭐⭐⭐⭐ — Fills a gap in multi-round evaluation for FD-SLMs and contributes significantly to advancing full-duplex spoken dialogue research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

<!-- RELATED:END -->
