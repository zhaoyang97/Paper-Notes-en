---
title: >-
  [Paper Note] Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding
description: >-
  [CVPR 2026][Audio & Speech][Social Interaction Understanding] This paper introduces the Omni-MMSI task—understanding multi-person social interactions from raw audio-visual inputs (rather than pre-processed oracle social cues)—and proposes Omni-MMSI-R, a reference-guided pipeline that achieves accurate social interaction understanding via tool-generated identity-attributed social cues combined with chain-of-thought reasoning.
tags:
  - CVPR 2026
  - "Audio & Speech"
  - Social Interaction Understanding
  - Identity Attribution
  - Multimodal Reasoning
  - Chain-of-Thought Reasoning
  - Reference Guidance
date: 2026-05-08
content_hash: f087cc03abfd4631
---

# Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding

**Conference**: CVPR 2026
**arXiv**: [2604.00267](https://arxiv.org/abs/2604.00267)
**Code**: [Project Page](https://sampson-lee.github.io/omni-mmsi-project-page)
**Area**: Audio & Speech / Social Understanding
**Keywords**: Social Interaction Understanding, Identity Attribution, Multimodal Reasoning, Chain-of-Thought Reasoning, Reference Guidance

## TL;DR

This paper introduces the Omni-MMSI task—understanding multi-person social interactions from raw audio-visual inputs (rather than pre-processed oracle social cues)—and proposes Omni-MMSI-R, a reference-guided pipeline that achieves accurate social interaction understanding via tool-generated identity-attributed social cues combined with chain-of-thought reasoning.

## Background & Motivation

Multimodal multi-person social interaction (MMSI) understanding aims to interpret human behaviors in social scenes and is foundational for building socially intelligent AI systems. Prior work (e.g., speaking target identification STI, pronoun coreference resolution PCR) has made notable progress, but rests on a fundamental assumption: **identity-attributed social cues are provided in perfect oracle form** (e.g., "who says what" and "where each person is" are given as known inputs).

In real-world deployment, however, AI assistants must perceive and reason from raw audio-visual data. Switching from oracle to raw inputs yields:
- An average accuracy drop of **28.1%** for prior pipelines (Lee et al., Li et al.)
- A drop of **9.52%** for human annotators and state-of-the-art Omni-LLMs (Qwen2.5 Omni, Gemini 2.5 Pro)

**The core bottleneck is identity attribution**:
- **Visual attribution**: Existing detectors are prone to identity swaps in multi-person scenes under occlusion or overlap (e.g., Gemini assigns identities by left-to-right spatial order and fails when detection breaks down)
- **Speech attribution**: Transcribed speech cannot be reliably matched to the correct speaker (recognized content is frequently attributed to the wrong person)

## Method

### Overall Architecture

Omni-MMSI-R is a reference-guided LLM pipeline:

$$f: (P, I_{AV}, \mathcal{R}) \rightarrow X_{answer}$$

The input consists of a system prompt $P$, raw audio-visual content $I_{AV}$, and a reference set $\mathcal{R} = \{(a_i, v_i)\}_{i=1}^N$ comprising representative voice and appearance samples for each participant.

Pipeline: Reference Loading → Tool-Based Identity-Attributed Cue Generation → Omni-LLM Chain-of-Thought Reasoning → Answer Output

### Key Designs

1. **Reference Guidance**:

    - Core insight: Humans recognize acquaintances by their appearance and voice, leveraging such memory for identity association when interpreting social interactions
    - Upper-body images and speech clips are manually cropped/extracted per participant to construct reference pairs
    - A total of 69 audio-visual reference profiles are constructed across participants
    - In practice, such references can be collected via device registration or verification workflows

2. **Tool-Based Social Cue Extraction**:

    - **Audio tools**:
        - Whisper transcribes audio into timestamped utterance sequences
        - SpeechBrain performs speaker verification for each utterance: utterances and reference voice clips are encoded into embeddings, and cosine similarity is computed; the reference with the highest similarity is assigned as the predicted speaker identity
    - **Visual tools**:
        - YOLO detects bounding boxes of all visible participants in the last video frame
        - OSNet performs person re-identification on each detection: cropped detections and reference images are encoded into visual embeddings, and the reference with the highest similarity is assigned as the predicted visual identity
    - Output: identity-attributed verbal cues (who said what) + identity-attributed non-verbal cues (where each person is)

3. **CoT Social Reasoning**:

    - A structured two-step chain-of-thought reasoning process is designed:
        - **Speaker confirmation**: The last speaker is jointly confirmed via voice matching and visible lip motion
        - **Reference reasoning**: The referent of the speaker's utterance is inferred by combining verbal cues (dialogue context, utterance matching) with non-verbal interaction signals (mutual gaze, pointing direction)
    - CoT data construction: generated by Gemini 2.5 Pro → rejection sampling (retaining only correctly answered instances) → lightweight human review

4. **Model Training**:

    - Fine-tuned from Qwen2.5-Omni-7B using LoRA (rank=8)
    - LLaMA-Factory framework, cross-entropy loss, cosine learning rate schedule
    - Learning rate $1\times10^{-4}$, 3 epochs, 16384-token context length

### Loss & Training

Standard cross-entropy loss is used to train the model to jointly generate the reasoning process $X_{think}$ and the final answer $X_{answer}$:

$$X_{answer}, X_{think} = f_\theta^{\text{Omni-LLM}}(P, I_{AV}, \mathcal{R}, \mathcal{S})$$

## Key Experimental Results

### Main Results

**Social Interaction Understanding (Ego4D + YouTube)**:

| Method | Ego4D STI | Ego4D PCR | Ego4D Avg. | YouTube STI | YouTube PCR | YouTube Avg. |
|--------|-----------|-----------|------------|-------------|-------------|--------------|
| Qwen2.5 Omni 7B | 26.29 | 28.57 | 27.43 | 14.00 | 26.18 | 20.09 |
| Gemini 2.5 Pro | 36.12 | 39.28 | 37.70 | 36.13 | 53.47 | 44.80 |
| Lee et al. | 28.98 | 32.14 | 30.56 | 29.01 | 34.80 | 31.91 |
| Li et al. | 29.73 | 32.27 | 31.00 | 26.30 | 30.14 | 28.22 |
| **Omni-MMSI-R** | **40.57** | **45.54** | **43.06** | **37.46** | **56.62** | **47.04** |

Omni-MMSI-R surpasses prior pipelines by 12%+ on Ego4D and 15%+ on YouTube.

**Identity Attribution Accuracy**:

| Method | Ego4D Verbal Attr. | Ego4D Non-verbal Attr. | Ego4D Avg. | YouTube Avg. |
|--------|-------------------|----------------------|-----------|-------------|
| Gemini 2.5 Pro | 44.75 | 26.52 | 35.64 | 58.04 |
| Qwen3 Omni 30B | 52.61 | 57.61 | 55.11 | 55.14 |
| **Omni-MMSI-R** | **71.09** | **86.48** | **78.79** | **76.95** |

Omni-MMSI-R outperforms Omni-LLMs by approximately 23.7% on Ego4D and 18.9% on YouTube in identity attribution.

### Ablation Study

**Reference Guidance Input Configurations (Ego4D)**:

| Reference Audio | Reference Visual | Verbal Cues | Non-verbal Cues | Avg. Accuracy |
|----------------|-----------------|-------------|----------------|---------------|
| ✗ | ✗ | ✗ | ✗ | 33.97% |
| ✓ | ✓ | ✗ | ✗ | 35.98% |
| ✗ | ✗ | ✓ | ✓ | 39.44% |
| ✓ | ✓ | ✓ | ✓ | **43.06%** |

Raw references and tool-extracted cues are complementary; their combination yields the best performance.

**CoT Reasoning Granularity**:

| Configuration | Reasoning Steps | Avg. Accuracy |
|---------------|----------------|---------------|
| With reference | None | 39.41% |
| With reference | 1-step (reference reasoning) | 39.70% |
| With reference | 2-step (speaker confirmation + reference reasoning) | **43.06%** |
| With reference | 3-step (+ cue extraction) | 34.43% |

**2-step CoT is optimal**; the 3-step variant yields a substantial drop—overly long reasoning chains distract the model and exceed its capacity given the available training data.

### Key Findings

1. **Identity attribution is the core bottleneck** when transitioning from oracle to raw inputs: state-of-the-art Omni-LLMs perform reasonably at cue extraction but fail significantly at correctly associating cues with individuals
2. Reference guidance is more critical for smaller models (7B) than larger ones: smaller Omni-LLMs show performance degradation when provided with references (possibly failing to leverage reference information effectively), underscoring the necessity of tool assistance
3. LLMs do not blindly trust tool-extracted cues: by jointly leveraging raw audio-visual evidence and extracted cues, the model can self-correct inaccurate cues during CoT reasoning
4. Moderate reasoning granularity (2-step) is optimal; over-decomposing reasoning steps is harmful
5. Audio and visual modalities contribute complementarily: audio attribution alone yields +5.87%, visual attribution alone +4.59%, and their combination +9.09%

## Highlights & Insights

- **The problem formulation itself is highly valuable**: advancing MMSI from oracle inputs to raw inputs represents a critical step from academic research toward real-world deployment
- The **reference guidance** design is practically motivated: analogous to face unlock or voiceprint registration workflows, collecting references in deployment scenarios is feasible
- The combination of **tools + LLM reasoning** is more reliable than purely end-to-end LLMs—lightweight task-specific tools (Whisper/YOLO/SpeechBrain/OSNet) substantially outperform general-purpose LLMs for identity attribution
- The CoT granularity experiment (3-step performing worse) carries meaningful practical implications: reasoning chain design must be calibrated to match model capacity and training data volume

## Limitations & Future Work

1. Reference pairs require manual construction (upper-body images + speech clips per participant); automated reference acquisition is an important future direction
2. Overall accuracy remains modest (43% on Ego4D, 47% on YouTube), indicating a gap before practical deployment
3. CoT data is generated by Gemini 2.5 Pro, potentially introducing biases from automated generation
4. Validation is limited to a Werewolf game dataset, restricting scenario diversity
5. Errors in the current tool chain propagate to downstream reasoning; more robust error-handling mechanisms are needed

## Related Work & Insights

- Directly extends prior MMSI work by Lee et al. and Li et al.—representing a paradigm shift from oracle to raw inputs
- Represents the first systematic exploration of LLM tool use in social understanding, bridging MMSI and LLM agents
- The application of CoT reasoning in social scenarios demonstrates the value of structured reasoning for fine-grained multimodal understanding
- Implications for practical AI assistants: in multi-person interaction settings, identity tracking and attribution capabilities are a prerequisite for understanding social dynamics

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel task formulation and practical reference guidance design; methodologically an engineering integration rather than a fundamentally new architecture)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple baselines, multi-LLM comparisons, extensive ablations, and qualitative analysis)
- Writing Quality: ⭐⭐⭐⭐ (Problem motivation is clear; quantitative evidence effectively demonstrates the oracle-to-raw performance gap)
- Value: ⭐⭐⭐⭐ (Advances the MMSI field toward more realistic deployment scenarios)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](../../ACL2026/audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[AAAI 2026\] Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning](../../AAAI2026/audio_speech/towards_authentic_movie_dubbing_with_retrieve-augmented_director-actor_interacti.md)

<!-- RELATED:END -->
