---
title: >-
  [Paper Note] Music Arena: Live Evaluation for Text-to-Music
description: >-
  [NeurIPS 2025][LLM Safety][text-to-music] Music Arena is the first online live evaluation platform for text-to-music (TTM) generation. It addresses the heterogeneous signature problem of TTM systems via an LLM-driven moderation and routing system, collects multi-level preference data including fine-grained listening behavior and natural language feedback, and provides the community with a sustainable open preference data source through monthly rolling data releases.
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "text-to-music"
  - "human preference evaluation"
  - "live evaluation"
  - "leaderboard"
  - "preference data"
date: 2026-05-08
content_hash: 4d775643b9f03a9b
---

# Music Arena: Live Evaluation for Text-to-Music

**Conference**: NeurIPS 2025
**arXiv**: [2507.20900](https://arxiv.org/abs/2507.20900)  
**Code**: [Available](https://github.com/gclef-cmu/music-arena)  
**Area**: AI Safety
**Keywords**: text-to-music, human preference evaluation, live evaluation, leaderboard, preference data

## TL;DR

Music Arena is the first online live evaluation platform for text-to-music (TTM) generation. It addresses the heterogeneous signature problem of TTM systems via an LLM-driven moderation and routing system, collects multi-level preference data including fine-grained listening behavior and natural language feedback, and provides the community with a sustainable open preference data source through monthly rolling data releases.

## Background & Motivation

Text-to-music (TTM) generation has advanced rapidly in recent years (MusicGen, Stable Audio, Riffusion, etc.), yet faces two intertwined core challenges:

**Lack of evaluation standardization.** Current TTM evaluation relies on ad hoc human listening tests, but testing protocols vary enormously across studies—interface design, choice of comparison models, and annotator demographics all differ, making metrics reported across papers (win rates, MOS scores, etc.) directly incomparable. Automatic evaluation metrics (e.g., FAD, FD) have been shown to correlate poorly with human preference and cannot substitute for human evaluation.

**Unsustainable preference data.** Existing one-time preference datasets (e.g., MusicEval) are fixed upon release and cannot reflect the emergence of new models or drift in user preferences. Commercial platforms can collect usage data continuously, but such data are not publicly available. The research community urgently needs a renewable, open preference data source for model alignment and evaluation metric development.

The "live evaluation" paradigm pioneered by Chatbot Arena in the LLM domain has demonstrated the feasibility of scaling preference collection by aligning the incentives of users and researchers. TTS Arena, GenAI Arena, and others subsequently extended this paradigm to speech and image domains. However, the music domain presents unique challenges: (1) TTM systems exhibit highly heterogeneous input–output type signatures—some support lyrics while others do not, some allow duration specification while others do not; (2) music is a temporal medium that must be experienced in real time, unlike images that can be perceived instantaneously; (3) music involves more complex copyright and cultural sensitivity issues. These characteristics require that a live evaluation framework be specifically adapted for the music domain.

## Method

### Overall Architecture

Music Arena adopts a three-component architecture: frontend (Gradio web interface) → backend (core orchestrator) → model endpoints (Dockerized TTM systems). Users submit text prompts via the frontend → the backend moderates and routes the prompt via an LLM, then dispatches two models in parallel for generation → audio is returned synchronously to the user → the user listens and votes → preference data are stored and periodically released.

### Key Designs

1. **LLM-Driven Moderation and Routing System**:
    - Function: Adapts users' natural language prompts to the differing type signatures of heterogeneous TTM systems through a unified single-text-input interface.
    - Mechanism: GPT-4o is used to process each user prompt in two steps—first **moderation**, rejecting prompts containing copyrighted music references, culturally sensitive themes, or inappropriate content (while permitting reasonable profanity in contexts such as heavy metal styles); then **structured information extraction**, determining whether the prompt implicitly requires lyrics/vocals (e.g., "a folk song about a cat named Chamomile" implies a need for lyrics) and whether a duration is specified (e.g., "30-second lo-fi beat"), routing the prompt to a compatible subset of models based on the extracted information.
    - Design Motivation: The heterogeneity of TTM systems far exceeds that in other AI domains—some produce only instrumental music (MusicGen, Stable Audio Open), some support lyrics (SongGen, ACE-Step), some jointly generate lyrics and audio (FUZZ), and some support duration specification (Stable Audio series). A routing system is essential to enable fair comparison within a unified interface.

2. **Multi-Level Preference Collection**:
    - Function: Goes beyond simple binary preference to collect rich user behavior data.
    - Mechanism: Three layers of data collection—(a) explicit four-choice preference (A better / B better / tie / both bad); (b) fine-grained listening behavior logs recording play/pause timestamps and total listening duration for each audio clip; (c) natural language feedback allowing users to freely describe their reasons for preference after voting.
    - Technical Details: Users must listen to each audio clip for at least 4 seconds before the vote button is unlocked. The actual duration of audio clips is hidden to prevent length bias from influencing votes. The backend waits for both models to complete generation before synchronously returning results, avoiding bias from speed differences; generation time is recorded in backend logs.
    - Design Motivation: Music is a temporal medium, and users' listening behavior itself carries rich information—whether they listened to completion, where they paused, whether they replayed certain segments—all of which can help illuminate the mechanisms underlying preference formation.

3. **Privacy Protection and Transparent Data Release**:
    - Function: Maximizes data openness while protecting user privacy.
    - Mechanism: User identifiers (e.g., IP addresses) are pseudonymized via salted hashing; only the hashed anonymous IDs are stored, and raw identifiers are never retained. This prevents de-anonymization attacks (e.g., rainbow table attacks) while preserving the ability to link records (multiple battles from the same user can be associated).
    - Data Release Policy: Monthly rolling releases are committed, including anonymized user IDs, generated audio, and complete preference data. The entire platform codebase is open-sourced (excluding keys). The leaderboard displays not only Arena Scores but also each model's training data provenance and generation speed (median RTF), reflecting responsible evaluation practice.
    - Design Motivation: One-time preference datasets cannot keep pace with the rapid development of the TTM field; rolling releases address the problem of data becoming outdated over time.

### Supported Models

The platform covers three categories of TTM systems:

| Type | Model | Organization | Lyrics Support | Notes |
|------|-------|--------------|---------------|-------|
| Open-source | MusicGen | Meta | No | Instrumental only |
| Open-source | Stable Audio Open/Small | Stability AI | No | Supports duration specification |
| Open-source | SongGen | — | Yes (lyrics via GPT-4o) | Autoregressive model |
| Open-source | ACE-Step | ACE Studio | Yes (lyrics via GPT-4o) | Diffusion model |
| Open-source | Magenta RealTime | Google DeepMind | No | Real-time generation |
| Commercial | FUZZ 1.0/1.1 | Producer.ai (Riffusion) | Yes (joint generation) | Diffusion transformer |
| Commercial | Stable Audio 2.0 | Stability AI | No | Supports duration specification |
| Commercial | Lyria RealTime | Google DeepMind | No | Real-time instrumental |

Each model is encapsulated in an independent Docker container exposing a unified API, facilitating modular development and extensibility.

### Loss & Training

The platform itself does not train models. Leaderboard rankings are based on the Bradley-Terry model, which estimates a global ranking score (Arena Score) for each TTM system from pairwise preferences.

## Key Experimental Results

### Main Results

Initial data collection period (July 28 – August 31, 2025):

| Metric | Value |
|--------|-------|
| Total battles | 1,420 |
| Unique users | 373 |
| Valid votes | 1,051 |
| Average participation per user | ~3.8 battles |
| Platform operation days | ~35 days |

### Ablation Study

| Analysis Dimension | Key Finding | Notes |
|-------------------|-------------|-------|
| Commercial vs. open-source | Commercial models receive higher overall preference scores | FUZZ and other commercial systems rank near the top |
| Lyrics support | Outputs with lyrics/vocals are more preferred | Users show a clear preference for vocalized content |
| Open-source competitiveness | Open-source lyrics models such as ACE-Step perform well | The gap between open-source and commercial models is narrowing |
| User engagement pattern | Average 3.8 battles per user | Room for improvement in user retention |

### Key Findings

- **The heterogeneity of type signatures in the music domain** far exceeds that in NLP and image domains: within the same "generate music" task, different systems differ greatly in support for lyrics, duration, and vocals, making the design of a unified evaluation framework a non-trivial engineering challenge.
- **Listening behavior data is a unique asset**: unlike the instantaneous perception of images or text, the temporal consumption pattern of music provides a window into the process of preference formation—at what second users make a decision, and whether repeated listening is needed before a judgment can be formed.
- **Natural language feedback complements the limitations of binary preference**: users can articulate specific reasons for their preference (e.g., "A has a better sense of rhythm but B has a more appealing melody"), which is crucial for understanding the multidimensional nature of music preference.
- **Leaderboards need to present information beyond preference**: training data provenance (relevant to copyright compliance) and generation speed (affecting creative workflows) are particularly important supplementary dimensions in the music domain.

## Highlights & Insights

- **Domain adaptation matters more than direct replication**: live evaluation cannot simply copy the Chatbot Arena approach—the LLM routing system, listening behavior tracking, and copyright moderation are all necessary adaptations for the music domain. This provides a reference for designing live evaluation frameworks for other AI tasks with unique characteristics (3D generation, code assistance, etc.).
- **The LLM-as-middleware routing and moderation solution** elegantly resolves the engineering challenge of unified evaluation across heterogeneous systems, and can be naturally upgraded as LLM capabilities improve.
- **Monthly rolling data releases** create a sustainable research data ecosystem—the community can continuously access up-to-date preference data to support TTM alignment and evaluation metric research.
- **Embedding ethical considerations into the platform design** (IRB approval, informed consent, copyright moderation, privacy protection) rather than addressing them as an afterthought reflects a responsible approach to AI evaluation.
- **The unified Docker containerization scheme** serves not only the platform itself but also provides infrastructure for other research requiring multi-system comparisons.

## Limitations & Future Work

- **Insufficient representativeness of the user population**: participants are primarily US-based AI enthusiasts, potentially failing to reflect the diverse musical preferences and cultural backgrounds found globally.
- **Limited task scope**: the platform currently supports only text-to-music, excluding other important music AI tasks such as style transfer, symbolic music generation, and audio editing.
- **Simple model pairing strategy**: models are currently selected at random with uniform probability; no optimized pairing algorithm is employed to balance leaderboard accuracy against user experience.
- **Absence of in-audio position tracking**: while total listening duration and play/pause actions can be tracked, it is not possible to determine which specific segment of the audio the user spent more time on (seek behavior).
- **Limited initial data volume**: a sample size of 1,051 votes is relatively small for precisely estimating Bradley-Terry coefficients, resulting in wide confidence intervals on the leaderboard.
- **Long-term sustainability challenges**: self-hosting open-source models requires sustained GPU resource investment, and the availability of commercial APIs is also subject to uncertainty.

## Related Work & Insights

- **Chatbot Arena** pioneered the live evaluation paradigm; Music Arena builds on this foundation with deep domain adaptation for music.
- **TTS Arena 2.0** and **GenAI Arena** extended live evaluation to speech and image/video respectively; Music Arena further advances this into the more subjective domain of music.
- **MusicRL** uses preference data to align TTM systems; the sustainable preference data produced by Music Arena can directly support such work.
- The domain adaptation strategies employed in the platform design (LLM routing, behavioral tracking for temporal media, copyright moderation) can inspire live evaluation design for other AI domains that similarly require customization.
- Meta-evaluation of automatic metrics (using collected human preferences to validate the effectiveness of metrics such as FAD) is an important future direction.

## Rating

- Novelty: ⭐⭐⭐ Transfers live evaluation to the music domain with thoughtful domain-specific design, though the overall paradigm is not novel in itself.
- Experimental Thoroughness: ⭐⭐⭐ Initial data volume is limited; the contribution is primarily in platform construction and methodology rather than large-scale experimental validation.
- Writing Quality: ⭐⭐⭐⭐ Well-organized and comprehensive, with thorough and candid discussion of ethical considerations.
- Value: ⭐⭐⭐⭐ Addresses the gap in standardized evaluation and open preference data for the TTM field; long-term value depends on sustained platform operation and community adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](../../ACL2026/llm_safety/subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](evaluation_of_vision-llms_in_surveillance_video.md)
- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)

</div>

<!-- RELATED:END -->
