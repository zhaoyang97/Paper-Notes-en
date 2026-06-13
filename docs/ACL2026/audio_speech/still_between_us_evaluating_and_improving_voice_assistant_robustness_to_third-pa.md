---
title: >-
  [Paper Note] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions
description: >-
  [ACL 2026][Audio & Speech][Voice Assistants] Addressing the inability of voice assistants to distinguish between Third-Party Interruptions (TPI) and primary user speech…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Voice Assistants"
  - "Third-Party Interruption"
  - "Speaker Awareness"
  - "Hard Negative Mining"
  - "Semantic Shortcut Learning"
date: 2026-05-08
content_hash: 3dd84a23e8ac69bb
---

# Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions

**Conference**: ACL 2026  
**arXiv**: [2604.17358](https://arxiv.org/abs/2604.17358)  
**Code**: [GitHub](https://github.com/pleasedpenguin/tpi-va)  
**Area**: Audio and Speech  
**Keywords**: Voice Assistants, Third-Party Interruption, Speaker Awareness, Hard Negative Mining, Semantic Shortcut Learning

## TL;DR

Addressing the inability of voice assistants to distinguish between Third-Party Interruptions (TPI) and primary user speech, this paper proposes the TPI-Train dataset with 88K instances and the TPI-Bench evaluation framework. Through a speaker-aware hard negative mining strategy, semantic shortcut learning is eliminated, forcing models to rely on acoustic cues for interruption detection.

## Background & Motivation

**Background**: Spoken Language Models (SLMs) are widely deployed in real-world voice assistant scenarios, enabling human-like natural conversations, but are primarily designed for one-on-one interactions.

**Limitations of Prior Work**: In real life, users often encounter third-party interruptions (e.g., bystander comments, background dialogue) while talking to voice assistants. Current SLMs fail to distinguish these interruptions and blindly concatenate multi-person speech into a single continuous utterance, leading to incorrect or nonsensical responses.

**Key Challenge**: Multimodal speech training suffers from "semantic shortcut learning"—models tend to exploit semantic patterns in text (e.g., contradictions, topic shifts) to detect interruptions while ignoring acoustic signals (e.g., changes in speaker voice), making them extremely vulnerable in semantically ambiguous scenarios.

**Goal**: Construct a comprehensive TPI perception framework, including training data, evaluation benchmarks, and training strategies, to enable voice assistants to correctly identify and handle third-party interruptions.

**Key Insight**: Starting from a linguistic taxonomy of interruptions, 26 real-world interruption scenarios are defined to systematically construct training and evaluation data.

**Core Idea**: Utilize speaker-aware hard negative mining (re-synthesizing dual-speaker interruption text with a single speaker's voice) to force models to abandon semantic shortcuts and truly learn acoustic cues.

## Method

### Overall Architecture

The framework consists of three core components: (1) TPI-Train—an 88K training dataset covering 26 interruption scenarios, categorized into "actionable" (to be included in response) and "ignorable" (to be discarded); (2) TPI-Bench—an evaluation framework comprising TPI-Test (2K samples) and Janus-Test (2K adversarial samples); (3) A speaker-aware hard negative training strategy.

### Key Designs

1.  **TPI-Train Dataset Construction**:
    *   **Function**: Provides large-scale, diverse third-party interruption training data.
    *   **Mechanism**: Based on a linguistic interruption taxonomy, 26 real-world scenarios (e.g., agreement/disagreement, topic shift, emotional expression) are designed to generate 88K training instances from voice assistant data. Each interruption is labeled as "actionable" or "ignorable" with corresponding response strategies.
    *   **Design Motivation**: Existing voice dialogue data lacks systematic coverage of third-party interruption scenarios and lacks explicit response strategy guidance.

2.  **TPI-Bench Evaluation Framework (including Janus-Test)**:
    *   **Function**: Strictly evaluates a model's TPI perception capability, specifically the ability to distinguish between acoustic and semantic cues.
    *   **Mechanism**: TPI-Test contains 2K real dual-speaker interruption samples to test situational response capabilities. Janus-Test contains 2K adversarial samples where content that semantically resembles an interruption is re-synthesized using the primary speaker's voice to test if the model truly relies on acoustic cues.
    *   **Design Motivation**: The key insight of Janus-Test is that if the textual content is identical but the voice belongs to the same person, the model should not identify it as an interruption—this serves as a litmus test for acoustic vs. semantic dependence.

3.  **Speaker-aware Hard Negative Mining**:
    *   **Function**: Eliminates semantic shortcut learning and forces model reliance on acoustic signals.
    *   **Mechanism**: Creates training samples that are textually identical to real dual-speaker interruptions but where the audio is re-synthesized by a single speaker. In these samples, models cannot use semantic cues (as the text is identical) and must rely on voice changes to determine if an interruption exists.
    *   **Design Motivation**: t-SNE visualizations show that without hard negatives, embeddings for different speaker configurations overlap heavily; with hard negatives, the embedding space forms clearly separated clusters.

## Key Experimental Results

### Main Results

| Test Set | Metric | Baseline SLM | TPI-Full | Gain |
| :--- | :--- | :--- | :--- | :--- |
| TPI-Test | Detection Accuracy | Low (Concatenation) | High | Significant |
| Janus-Test | Adversarial Robustness | Near Failure | Robust | Significant |
| Human Eval | Naturalness Preference | Low | Highly Preferred | - |

### Ablation Study

| Configuration | Key Metric | Remarks |
| :--- | :--- | :--- |
| w/o Hard Negatives | t-SNE Overlap | Model relies on semantic shortcuts |
| w/ Hard Negatives (TPI-Full) | t-SNE Separation | Model relies on acoustic cues |
| Semantic-only Training | Janus-Test Failure | Misclassifies self-correction as interruption |
| Full Training | Robust on both sets | Balanced acoustic and semantic signals |

### Key Findings

*   Semantic shortcut learning is a critical trap in multimodal speech model training: models exploit patterns like contradictions and topic shifts in text for detection rather than truly "listening" to voice changes.
*   After hard negative training, the model's embedding space shifts from a chaotic mix to clearly separated clusters, proving the model has learned to distinguish based on acoustic identity.
*   Human evaluation confirms that the response strategies embedded in the framework are highly preferred by users in terms of effectiveness and naturalness.
*   The classification of Actionable vs. Ignorable is crucial for response strategies—the model needs to know when to incorporate interruption content and when to ignore it.

## Highlights & Insights

*   The concept of **semantic shortcut learning** has broad significance: it applies beyond TPI; in any multimodal training, models might take "textual shortcuts" and ignore other modality signals.
*   **The design of Janus-Test is ingenious**: By controlling variables (same text, different voice), it strictly tests whether the model truly understands acoustic signals.
*   Constructing the dataset from a **linguistic taxonomy** ensures the systematicity and comprehensiveness of scenarios (26 interruption types).
*   **High practicality**: Directly targets real pain points of voice assistants, and the response strategies are ready for deployment.

## Limitations & Future Work

*   Primarily focused on English; generalization across languages and accents remains to be verified.
*   While systematic, the 26 scenarios may not exhaust all real-world possibilities.
*   The current framework relies on TTS synthesis for hard negatives; the quality of synthesis may affect training outcomes.
*   Complex multi-party dialogue scenarios with more than two speakers have not yet been addressed.
*   Performance and latency in real-time streaming scenarios require further evaluation.

## Related Work & Insights

*   **vs. Traditional Speaker Diarization**: TPI requires not only detecting speaker changes but also judging whether the interruption should influence the response strategy, representing higher-level semantic understanding.
*   **vs. Multi-turn Dialogue Models**: Existing research focuses on continuous dialogue with a single user, overlooking third-party interventions.
*   **vs. Hard Negative Mining**: Borrows ideas from contrastive learning but innovatively applies them to cross-modal (text vs. acoustic) shortcut elimination.

## Rating

*   Novelty: ⭐⭐⭐⭐ First to systematically define and solve the TPI problem for voice assistants; findings on semantic shortcuts are insightful.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Includes large-scale datasets, adversarial sets, ablation studies, and human evaluation.
*   Writing Quality: ⭐⭐⭐⭐ Clear problem definition and intuitive project presentation.
*   Value: ⭐⭐⭐⭐ Addresses real-world voice assistant pain points with direct engineering application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)
- [\[ACL 2026\] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation](drinq_evaluating_conversational_implicature_with_controlled_context_variation.md)
- [\[ACL 2026\] DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition](duivrs-2_an_llm-based_interactive_voice_response_system_for_large-scale_poi_attr.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](../../ICLR2026/audio_speech/avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)
- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](../../AAAI2026/audio_speech/listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)

</div>

<!-- RELATED:END -->
