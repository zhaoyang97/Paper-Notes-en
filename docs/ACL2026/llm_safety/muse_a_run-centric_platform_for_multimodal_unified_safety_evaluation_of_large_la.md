---
title: >-
  [Paper Note] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models
description: >-
  [ACL 2026][LLM Safety][Multimodal Safety Evaluation] MUSE integrates cross-modal payload generation, multi-turn red teaming, unified model routing, and a five-level safety judge into a run-centric…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multimodal Safety Evaluation"
  - "Automated Red Teaming"
  - "Multi-turn Jailbreaking"
  - "Cross-modal Attacks"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: 0014b909173b1e8a
---

# MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2603.02482](https://arxiv.org/abs/2603.02482)  
**Code**: Repository not public; paper provides Demo video: https://youtu.be/xHTUJlXJSmc  
**Area**: Multimodal Safety Evaluation / Audio & Speech / Red Teaming Platform  
**Keywords**: Multimodal Safety Evaluation, Automated Red Teaming, Multi-turn Jailbreaking, Cross-modal Attacks, LLM-as-Judge

## TL;DR
MUSE integrates cross-modal payload generation, multi-turn red teaming, unified model routing, and a five-level safety judge into a run-centric, reproducible experimental platform. Through approximately 3,700 experiments, it reveals that multi-turn strategies can penetrate multimodal LLMs that exhibit near-total refusal in single-turn settings, while inter-turn modality switching acts more as a mechanism to accelerate the loosening of defenses rather than a universal silver bullet for increasing final ASR.

## Background & Motivation
**Background**: Large model safety evaluation is shifting from pure text to multimodality. Models like GPT-4o, Gemini, Claude Sonnet 4, and Qwen-Omni can process text, images, audio, and video within the same turn or conversation. Consequently, safety alignment must expand from "whether a model refuses a single harmful text request" to "whether a model maintains refusal across any input modality and conversation trajectory."

**Limitations of Prior Work**: Existing research is largely divided into two streams. One focuses on multi-turn attacks (e.g., Crescendo, PAIR, Violent Durian), bypassing single-turn refusal through iterative rewriting and conversational pressure. The other focuses on multimodal safety, such as hiding harmful requests in images, audio, or visual text prompts. The issue is that these streams are often disjointed: multi-turn attack frameworks mostly handle text, while multimodal benchmarks mostly test single-turn, single-modality scenarios, lacking a unified system to manage attackers, target models, modality transitions, automated judges, and experimental records.

**Key Challenge**: Whether safety alignment can generalize across modalities is not a question that can be answered by looking at the final refusal rate alone. A model might be robust in text but become lenient when switching to images or audio. Alternatively, a model might not be weak in a specific non-text modality but becomes unstable in its safety policy when historical context and current inputs are fused across different modalities in a multi-turn dialogue. Existing binary ASR (Attack Success Rate) also conflates "full capability disclosure" with "providing partial actionable information," obscuring gray-zone risks.

**Goal**: The authors aim to solve three specific problems: 1) Build a practical multimodal red teaming platform rather than just releasing offline prompts. 2) Compare multi-turn attack strategies, target models, input modalities, and automated judge results within the same framework. 3) Specifically test whether Inter-Turn Modality Switching (ITMS) itself affects safety boundaries.

**Key Insight**: Instead of proposing a new training method for target models, **Ours** approaches from the perspective of evaluation infrastructure. The authors observed that the most chaotic part of multimodal red teaming is not the attack algorithm itself, but the experimental state being scattered across different scripts and API calls: which media files were generated, which modality was used in which turn, how model responses were labeled, and where to resume after a batch task interruption. These all require persistent tracking.

**Core Idea**: Use a "run" as the minimum reproducible experimental unit, fully linking the configuration of each attack, conversation states, media assets, model outputs, and judge labels. Cross-model, cross-strategy, and cross-modal safety evaluations are then conducted on this unified object.

## Method
MUSE is essentially a red teaming operating system for safety researchers. It does not treat attacks, model calls, media generation, and scoring as isolated scripts but integrates them into a run-centric pipeline that is browser-operable, backend-persistent, and observable in real-time. The input consists of harmful capability targets, attack strategies, target models, and available modalities; the output is a set of runs with full trajectories, including attacker prompts per turn, target model responses, modalities, generated assets, and judge labels.

### Overall Architecture
The system adopts a frontend-backend architecture. The browser frontend handles experimental configuration, launching automated red teaming, and viewing multimodal tests and real-time progress. The backend manages attack strategy execution, model API routing, text-to-audio/image/video conversion, result persistence, and Server-Sent Events (SSE) streaming updates. **Ours** emphasizes that this architecture transforms experiments from "running batches of scripts" into a "pauseable, resumable, auditable, and aggregatable" research workflow.

An automated red teeming experiment consists of five steps: 1) User selects the target model, strategy, harmful category, and modality configuration. 2) The strategy generates the current turn's text attack content. 3) If non-text input is required, the system converts text to audio, renders images, or synthesizes video. 4) The model routing layer converts a unified message format into the specific provider's API format. 5) An LLM judge evaluates the response based on a five-level safety taxonomy and writes the result back to the run; if unsuccessful and within the turn budget, the process continues.

### Key Designs
1.  **Run-centric Data Model**:
    - **Function**: Encapsulates all information from configuration to final judgment into a persistent "run" as the basic unit for tracking and statistics.
    - **Mechanism**: Each run records the target model, strategy, task, turn budget, per-turn modality, attacker content, target response, judge label, media paths, and final outcome. Batch campaigns consist of multiple runs, with the system updating aggregate statistics upon completion of each goal. If interrupted, tasks resume from the last completed goal.
    - **Design Motivation**: The challenge in replicating multi-turn multimodal red teaming lies in the "process," not just the "final score." Saving only the ASR prevents determining which turn or modality led to success. The run-centric design treats these variables as first-class citizens for analysis.

2.  **Strategy Engine + ITMS**:
    - **Function**: Implements three types of multi-turn attacks (Crescendo, PAIR, Violent Durian) via a unified interface and adds cross-turn modality rotation versions for context-retaining strategies.
    - **Mechanism**: Crescendo escalates from benign questions and backtracks/pivots upon refusal; PAIR independently generates candidates guided by judge scores; Violent Durian uses high-pressure rhetoric from turn one. ITMS does not change the attack goal but cyclically selects the next modality from the available set for each turn, converting attack text into the corresponding media input.
    - **Design Motivation**: This separates "strategy strength" from "modality switching effects." Strategies like Crescendo and Violent Durian are particularly suited to observe if refusal behavior loosens when switching between text, image, or audio throughout the conversation history.

3.  **Unified Modality Conversion, Model Routing, and Five-level Judge**:
    - **Function**: Wraps multimodal APIs from different providers and various payload formats into a unified interface, replacing binary success judgment with more granular labels.
    - **Mechanism**: The conversion layer transforms attacker text into TTS audio, text-in-image renders, or synthesized video. Assets are cached for cross-model reuse. The routing layer requires only a thin client implementation for new providers. The evaluation layer uses a GPT-4o judge to classify responses: Compliance, Partial Compliance, Indirect Refusal, Direct Refusal, and Non-Responsive. Hard ASR counts only full Compliance; Soft ASR includes Partial Compliance. The difference is defined as the Gray Zone Width (GZW).
    - **Design Motivation**: Provider API variance and insufficient judge granularity are common bottlenecks. Unified routing lowers expansion costs, while the five-level taxonomy avoids conflating "providing actionable information with a disclaimer" with "genuine refusal."

### Loss & Training
This paper is not focused on model training; thus, no new training losses are introduced. For experimental strategy, GPT-4o is fixed as the attacker and judge (temperature 0). The five attack strategies share a 10-turn budget. Crescendo and Violent Durian use up to 3 backtracks (attacker temperature 0.9). PAIR's success threshold is 9/10. The emphasis is on controlling the evaluation pipeline rather than optimizing target model parameters.

## Key Experimental Results

### Main Results
The experiments comprise approximately 3,700 red teaming runs. The dataset utilizes 50 harmful goals from AdvBench across five categories: weapons, controlled substances, malware, biological threats, and fraud/social engineering. These were rewritten as direct capability requests transferable across text, audio, image, and video. Target models include Qwen3-Omni, Qwen2.5-Omni, Gemini 2.5 Flash, Gemini 3 Flash Preview, GPT-4o, and Claude Sonnet 4.

Single-turn baselines confirm that models are not "easily bypassed by direct requests." Results show refusal rates between 90%-100% for most model-modality combinations, establishing a strong baseline: if multi-turn ASR is high, it is due to interaction and strategy pressure rather than a lack of basic alignment.

| Model | Text Refusal Rate | Image Refusal Rate | Audio Refusal Rate | Video Refusal Rate | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Claude Sonnet 4 | 96 | 100 | - | - | Standard API only |
| GPT-4o | 98 | 100 | - | - | Realtime API not used |
| Gemini 2.5 Flash | 98 | 100 | 100 | 100 | Near-total refusal |
| Gemini 3 Flash | 90 | 98 | 96 | 92 | Lowest text baseline |
| Qwen2.5-Omni | 94 | 98 | 98 | 92 | Solid single-turn |
| Qwen3-Omni | 98 | 100 | 100 | 100 | Highly robust |

The main red teaming experiments compare five strategies. Important results show that Crescendo and PAIR can drive almost entirely resistant models to high Hard ASR.

| Target Model | Crescendo | PAIR | Violent Durian | ITMS-Crescendo | ITMS-VD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Claude Sonnet 4 | 90 | 60 | 2 | 92 | 6 |
| GPT-4o | 96 | 98 | 42 | 92 | 40 |
| Gemini 2.5 Flash | 94 | 100 | 56 | 98 | 62 |
| Gemini 3 Flash | 98 | 96 | 26 | 94 | 34 |
| Qwen2.5-Omni | 96 | 98 | 86 | 88 | 100 |
| Qwen3-Omni | 98 | 96 | 30 | 94 | 22 |

**Key Findings from Main Results**:
1. Crescendo achieves 90%-98% Hard ASR across 6 models, showing that automated multi-turn red teaming systematically breaks single-turn refusal.
2. Violent Durian variance is high (e.g., 2% on Claude vs. 86% on Qwen2.5-Omni), suggesting high-pressure templates are sensitive to model alignment styles.
3. ITMS does not always increase final ASR but its value significantly depends on whether the baseline is already saturated.

### Ablation Study
ITMS ablation focuses on four Omni models, comparing text-only, audio-only, image-only, text+audio, text+image, and 3-way rotation. This addresses whether ASR changes stem from the non-text input itself or the cross-turn switching.

| Config | Gemini 2.5 Flash | Gemini 3 Flash | Qwen2.5-Omni | Qwen3-Omni |
| :--- | :--- | :--- | :--- | :--- |
| Text baseline | 94 | 98 | 96 | 98 |
| Audio-only | 100 (+6) | 100 (+2) | 90 (-6) | 96 (-2) |
| Image-only | 100 (+6) | 100 (+2) | 82 (-14) | 92 (-6) |
| Text+Audio | 98 (+4) | 100 (+2) | 92 (-4) | 94 (-4) |
| Text+Image | 96 (+2) | 98 (+0) | 84 (-12) | 94 (-4) |
| 3-Way | 98 (+4) | 98 (+0) | 90 (-6) | 96 (-2) |

Ablation reveals that Gemini models are more easily breached with non-text inputs (**Gain** of 2-6 points), while Qwen models become *more* resistant with non-text inputs (ASR drops, especially for Qwen2.5-Omni images).

| Metric / Setting | Key Value | Description |
| :--- | :--- | :--- |
| ITMS-Crescendo Avg. Turns to Success | Claude 3.0 -> 2.6; Gemini 3 2.8 -> 2.2 | 4 out of 6 models converge faster |
| ITMS-VD Avg. Turns to Success | Claude 10.0 -> 5.3; Gemini 2.5 3.5 -> 2.8 | Significant reduction in turns to reached Compliance |
| Early Turn Behavior | Turn 1 ITMS-Crescendo refusal: 86.0% | Higher than text-only (81.0%) initially, but drops below text-only by turn 2 |
| Partial Compliance | Turn 2 ITMS-Crescendo: 32.7% | Higher than text-only (27.1%), suggesting modality switching pushes models into gray zones |
| Judge Human Validation | 93% Agreement | Primary disagreements involve Compliance vs. Partial Compliance; no systematic bias found |

### Key Findings
- **Multi-turn interaction is the primary risk source**: Models with 90-100% single-turn refusal still exhibit 90-100% Hard ASR under Crescendo/PAIR.
- **ITMS accelerates alignment erosion**: While final ASR is often saturated, ITMS results in fewer turns to success and higher early-stage Partial Compliance.
- **Modality effects are provider-dependent**: Non-text inputs weaken Gemini but strengthen Qwen's refusal behavior.
- **Fraud/Social Engineering is the most vulnerable category**, while Drugs and Weapons are more resistant.
- **Soft ASR/GZW is valuable**: PAIR's Hard ASR on Claude is only 60%, but a 26% GZW indicates substantial leakage of actionable information.

## Highlights & Insights
- **Elevating experimental state management to a methodology**: MUSE's innovation is treating the "run" as the core object. Since safety failures depend on context and turn-specific triggers, retaining trajectories is vital for analysis.
- **ITMS as a causal probe**: Instead of broadly claiming multimodality is "dangerous," **Ours** uses ablation to show that effects are not uniform across providers.
- **Hard/Soft ASR avoids binary illusions**: Partial Compliance is critical in real-world scenarios, as attackers can chain multiple partial leaks into full capability transfer.
- **High platform portability**: The run-centric design and provider abstractions can easily migrate to other safety domains like privacy, copyright, or agent tool-use.

## Limitations & Future Work
- **Open-source commitment is unclear**: While the paper claims to be open-source, only a demo video is currently visible. Platform reproducibility is contingent on code and prompt availability.
- **Automated judge boundaries**: The 93% agreement is positive, but disagreements occur at the Compliance/Partial Compliance boundary, which affects Hard/Soft ASR interpretation.
- **Commercial API focus**: **Ours** lacks testing on locally deployed open-source models, which is necessary as commercial API policies shift frequently.
- **Video dimension is less explored**: Video was excluded from ITMS ablation due to synthesis latency. Future work should investigate native video rotation.
- **Scale of attack goals**: 50 AdvBench goals are sufficient for a platform validation but lack coverage of long-tail, multilingual, or domain-specific safety policies.

## Related Work & Insights
- **vs. Crescendo / PAIR / Violent Durian**: These are attack algorithms; MUSE integrates them into a unified, reproducible system with cross-modal extensions.
- **vs. FigStep / MM-SafetyBench**: These benchmarks prove non-text carriers weaken alignment but focus on single-turn evaluation. MUSE explores "modality shifts within continuous dialogue," a setting closer to multimodal agents.
- **vs. PyRIT / Garak**: While PyRIT/Garak are programmatic red teaming frameworks, MUSE provides deeper native support for multimodal payloads and run management.
- **Insight for Future Research**: Safety evaluation must evolve from "prompt sets" to "interaction sets." Future benchmarks should release full trajectories (per-turn modality, judge disagreements, context length) to facilitate a deeper understanding of alignment failure mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ System integration is not a new algorithm per se, but defining ITMS as a controlled variable is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ 3,700 runs provide a solid foundation. Future inclusion of open-source models and expanded video ablation would improve this.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure; however, technical access to the platform (repo link) should be more prominent.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical for multimodal LLM safety evaluation, highlighting the importance of run-level trajectories and gray-zone (GZW) analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] ACIArena: Toward Unified Evaluation for Agent Cascading Injection](aciarena_toward_unified_evaluation_for_agent_cascading_injection.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)

</div>

<!-- RELATED:END -->
