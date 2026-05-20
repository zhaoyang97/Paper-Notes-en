---
title: >-
  [Paper Note] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions
description: >-
  [ACL 2026][Audio & Speech][voice assistant] To address the inability of voice assistants to distinguish third-party interruptions (TPI) from primary-user speech, this work proposes TPI-Train…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "voice assistant"
  - "third-party interruption"
  - "speaker awareness"
  - "hard negative mining"
  - "semantic shortcut learning"
date: 2026-05-08
content_hash: 11e78acde5582198
---

# Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions

**Conference**: ACL 2026
**arXiv**: [2604.17358](https://arxiv.org/abs/2604.17358)  
**Code**: [GitHub](https://github.com/pleasedpenguin/tpi-va)  
**Area**: Audio & Speech
**Keywords**: voice assistant, third-party interruption, speaker awareness, hard negative mining, semantic shortcut learning

## TL;DR

To address the inability of voice assistants to distinguish third-party interruptions (TPI) from primary-user speech, this work proposes TPI-Train, a dataset of 88K training instances, along with the TPI-Bench evaluation framework. A speaker-aware hard negative mining strategy is introduced to eliminate semantic shortcut learning, enabling models to rely genuinely on acoustic cues for interruption detection.

## Background & Motivation

**Background**: Spoken Language Models (SLMs) have been widely deployed in real-world voice assistant scenarios, supporting human-like natural conversation, but are primarily designed for one-on-one interactions.

**Limitations of Prior Work**: In everyday settings, third parties frequently interject during a user's conversation with a voice assistant (e.g., bystander comments or background dialogue). Current SLMs cannot distinguish such third-party interruptions and blindly concatenate multi-speaker utterances as a single continuous stream, resulting in incorrect or nonsensical responses.

**Key Challenge**: "Semantic shortcut learning" emerges during multimodal speech training—models tend to exploit textual semantic patterns (e.g., contradictions, topic shifts) to detect interruptions while ignoring acoustic signals (e.g., speaker voice changes), making them extremely fragile in semantically ambiguous scenarios.

**Goal**: Construct a comprehensive TPI-aware framework encompassing training data, evaluation benchmarks, and training strategies that enable voice assistants to correctly identify and handle third-party interruptions.

**Key Insight**: Drawing on a linguistically grounded interruption taxonomy, the work defines 26 real-world interruption scenarios and systematically constructs training and evaluation data.

**Core Idea**: By applying speaker-aware hard negative mining—re-synthesizing two-speaker interruption transcripts with a single speaker's voice—the model is forced to abandon semantic shortcuts and genuinely learn acoustic cues.

## Method

### Overall Architecture

The framework comprises three core components: (1) **TPI-Train**—an 88K training dataset covering 26 interruption scenarios, where each interruption is categorized as either "actionable" (to be incorporated into the response) or "ignorable" (to be disregarded); (2) **TPI-Bench**—an evaluation framework consisting of TPI-Test (2K samples) and Janus-Test (2K adversarial samples); and (3) a speaker-aware hard negative training strategy.

### Key Designs

1. **TPI-Train Dataset Construction**:
    - Function: Provides large-scale, diverse training data for third-party interruptions.
    - Mechanism: Based on a linguistically grounded interruption taxonomy, 26 real-world scenarios are designed (e.g., agreement/disagreement, topic deviation, emotional expression), and 88K training instances are generated from voice assistant data. Each interruption is labeled as "actionable" or "ignorable" and paired with a corresponding response strategy.
    - Design Motivation: Existing spoken dialogue data lacks systematic coverage of third-party interruption scenarios and provides no explicit guidance on response strategies.

2. **TPI-Bench Evaluation Framework (including Janus-Test)**:
    - Function: Rigorously evaluates a model's TPI-awareness, particularly its ability to distinguish acoustic cues from semantic cues.
    - Mechanism: TPI-Test contains 2K real two-speaker interruption samples to assess context-sensitive response capability; Janus-Test contains 2K adversarial samples in which content that textually resembles an interruption is re-synthesized using the primary speaker's voice, testing whether the model genuinely relies on acoustic cues.
    - Design Motivation: The key insight behind Janus-Test is that if the textual content is identical but the voice originates from a single speaker, the model should not classify it as an interruption—this serves as a litmus test for acoustic dependence versus semantic dependence.

3. **Speaker-Aware Hard Negative Mining**:
    - Function: Eliminates semantic shortcut learning and compels the model to rely on acoustic signals.
    - Mechanism: Training samples are created whose transcripts are identical to genuine two-speaker interruptions but whose audio is re-synthesized with a single speaker. Since the textual content is identical, the model cannot exploit semantic cues and must rely on voice changes to determine whether an interruption is present.
    - Design Motivation: t-SNE visualizations show that without hard negatives, embeddings for different speaker configurations heavily overlap; after introducing hard negatives, the embedding space forms clearly separated clusters.

## Key Experimental Results

### Main Results

| Test Set | Metric | Baseline SLM | TPI-Full | Gain |
|----------|--------|-------------|----------|------|
| TPI-Test | Interruption detection accuracy | Low (blind concatenation) | High | Significant |
| Janus-Test | Adversarial robustness | Near-complete failure | Robust | Significant |
| Human Evaluation | Response naturalness preference | Low | Strongly preferred | — |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Without hard negatives | t-SNE clusters overlap | Model relies on semantic shortcuts |
| With hard negatives (TPI-Full) | t-SNE clusters clearly separated | Model relies on acoustic cues |
| Semantic-only training | Janus-Test fails | Single-speaker self-correction misclassified as interruption |
| Full training | Robust on both test sets | Balanced acoustic and semantic signals |

### Key Findings

- Semantic shortcut learning is a critical pitfall in multimodal speech model training: models exploit patterns such as contradictions and topic shifts in text to detect interruptions rather than genuinely "listening" for voice changes.
- After hard negative training, the model's embedding space transitions from a disordered mixture to clearly separated clusters, confirming that the model has learned to discriminate based on acoustic identity.
- Human evaluation confirms that the response strategies embedded in the framework are strongly preferred by users in both effectiveness and naturalness.
- The actionable vs. ignorable distinction is crucial for response strategy—the model must know when to incorporate interruption content and when to disregard it.

## Highlights & Insights

- The concept of **semantic shortcut learning** has broad implications: beyond the TPI scenario, models in any multimodal training setting may take "text shortcuts" and ignore signals from other modalities.
- **The design of Janus-Test is elegant**: by controlling variables (identical text, different voices), it strictly tests whether the model genuinely understands acoustic signals.
- **Grounding the dataset construction in a linguistic taxonomy** ensures systematic and comprehensive scenario coverage (26 interruption types).
- **High practical utility**: the framework directly addresses a real pain point in voice assistants, and the response strategies are immediately deployable.

## Limitations & Future Work

- The work primarily targets English; generalization across languages and accents remains to be validated.
- Although systematic, the 26 interruption scenarios may not exhaust all real-world cases.
- The current framework relies on TTS re-synthesis to construct hard negatives; synthesis quality may affect training outcomes.
- Complex multi-party conversation scenarios involving more than two speakers have not been addressed.
- Performance and latency in real-time streaming settings remain to be evaluated.

## Related Work & Insights

- **vs. Traditional Speaker Diarization**: TPI requires not only detecting speaker changes but also determining whether an interruption should influence the response strategy—a higher-level semantic understanding task.
- **vs. Multi-turn Dialogue Models**: Existing multi-turn dialogue research primarily focuses on continuous conversation with a single user and does not account for third-party intrusion.
- **vs. Hard Negative Mining**: The work draws inspiration from hard negatives in contrastive learning but innovatively applies the concept to cross-modal shortcut elimination (text vs. acoustics).

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic definition and solution for third-party interruptions in voice assistants; the discovery of semantic shortcut learning is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes a large-scale dataset, adversarial test set, ablation studies, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; the project page provides intuitive demonstrations.
- Value: ⭐⭐⭐⭐ Targets a genuine voice assistant pain point with direct engineering applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SiNGER: A Clearer Voice Distills Vision Transformers Further](../../ICLR2026/audio_speech/singer_a_clearer_voice_distills_vision_transformers_further.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ICLR 2026\] Improving Black-Box Generative Attacks via Generator Semantic Consistency](../../ICLR2026/audio_speech/improving_black-box_generative_attacks_via_generator_semantic_consistency.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)

</div>

<!-- RELATED:END -->
