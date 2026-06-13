---
title: >-
  [Paper Note] Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations
description: >-
  [ACL 2026][Audio & Speech][non-verbal vocalization] Ours proposes the Affectron framework, which implements two training-time augmentation strategies—emotion-driven Top-K NV matching and emotion-aware Top-K routing—to ac…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "non-verbal vocalization"
  - "emotional speech synthesis"
  - "NV-augmented training"
  - "affective routing"
  - "neural codec language model"
date: 2026-05-08
content_hash: 219f001ba6342b2f
---

# Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations

**Conference**: ACL 2026  
**arXiv**: [2603.14432](https://arxiv.org/abs/2603.14432)  
**Code**: [https://github.com/choddeok/Affectron](https://github.com/choddeok/Affectron)  
**Area**: Audio & Speech / Speech Synthesis  
**Keywords**: non-verbal vocalization, emotional speech synthesis, NV-augmented training, affective routing, neural codec language model

## TL;DR
Ours proposes the Affectron framework, which implements two training-time augmentation strategies—emotion-driven Top-K NV matching and emotion-aware Top-K routing—to achieve diverse and emotion-aligned synthesis of non-verbal vocalizations (e.g., laughter, sighs) using small-scale open-source decoupled corpora, significantly outperforming the VoiceCraft baseline.

## Background & Motivation

**Background**: Non-verbal vocalizations (NVs), such as laughter, sighs, and crying, are key means of expressing emotion in emotional speech synthesis. Existing expressive TTS systems primarily rely on two types of methods: label-controlled TTS (manual insertion of NV labels to control type and position) and spontaneous style TTS (implicit prediction of NVs from contextual cues).

**Limitations of Prior Work**: Label-controlled methods rely on alignment annotations or NV detection models; biases and error propagation in detection models lead to temporal inconsistencies in NV placement. Spontaneous methods are limited by the non-reproducibility of proprietary datasets. Publicly available NV corpora are generally biased towards basic types (e.g., breathing and laughter) and contain acoustic artifacts, failing to model fine-grained NV variants (e.g., the difference between a chuckle, giggle, and snicker).

**Key Challenge**: The lack of large-scale, diverse, and high-quality public NV corpora is the fundamental bottleneck. Although existing neural codec language models (NCLMs) can generate natural speech on low-quality corpora, they are primarily oriented toward voice cloning and lack the ability to control prosodic variations of fine-grained NVs.

**Goal**: To achieve emotion-consistent and contextually aligned diverse NV generation using small-scale open-source decoupled corpora (where linguistic speech and NVs are recorded separately).

**Key Insight**: The authors observe that emotional attributes typically evolve gradually rather than abruptly between adjacent segments; segments with shorter time intervals exhibit smaller emotional angular distances. Therefore, positions with minimal emotional change can serve as natural anchors for NV insertion.

**Core Idea**: Design training-time NV augmentation strategies to select appropriate NV types via emotion embedding matching and determine suitable insertion positions via emotional angular distance routing, then fine-tune a pre-trained VoiceCraft model using the augmented samples.

## Method

### Overall Architecture
Affectron uses VoiceCraft (330M parameters), pre-trained on pure linguistic speech, as the backbone. During training, it constructs NV-containing samples through NV augmentation to fine-tune the backbone, granting it NV generation capabilities. During inference, the model generates output directly from NV-tagged text and emotional reference speech, without requiring the matching and routing processes.

### Key Designs

1. **Emotion-Driven Top-K NV Matching (EDNM)**:
    - **Function**: Selects emotion-consistent and diverse NV candidates for each linguistic speech sample.
    - **Mechanism**: Given a linguistic speech $u$ and speaker $s$, all NV candidates for that speaker are retrieved. Emotion2Vec is used to calculate the cosine similarity of emotional embeddings between each NV candidate and the speech. Top-K candidates are selected and normalized into a probability distribution via temperature-scaled softmax, with a maximum of 2 NVs sampled. The temperature parameter $\tau=0.7$ and Top-K is set to 10.
    - **Design Motivation**: Randomly pairing NVs increases diversity but lacks emotional consistency. Matching based on emotional embeddings ensures the selected NVs align with the emotional state of the speech, while probabilistic sampling preserves diversity.

2. **Emotion-Aware Top-K Routing (EAR)**:
    - **Function**: Determines the optimal insertion positions for NVs within the speech.
    - **Mechanism**: Word-level segments are extracted using the Montreal Forced Aligner, and emotional pseudo-labels are generated for each segment using an emotional attribute predictor. Attributes are converted to spherical coordinates to calculate angular distances. For each NV candidate, the emotional distance $\Delta$ (based on arccosine distance on the sphere) to all potential insertion locations is calculated. The Top-K positions with the smallest distances are selected, and the final insertion position is sampled via a softmax distribution of negative distances.
    - **Design Motivation**: NVs should be inserted at positions with minimal changes in emotional attributes (i.e., emotional steady points) to maintain emotional coherence while enhancing expressiveness. Using spherical coordinates rather than direct Euclidean distance better captures directional shifts in emotional attributes.

3. **NV Structural Masking (NSM)**:
    - **Function**: Allows the model to generate NVs based on the emotional context of the surrounding linguistic speech.
    - **Mechanism**: The causal masking strategy of VoiceCraft is extended—NV codec token sequences are rearranged according to the routed positions. An NV segment and its surrounding linguistic tokens are randomly selected to form a mask span, which is moved to the end of the sequence. Delay stacking is then applied for efficient multi-codebook autoregressive modeling.
    - **Design Motivation**: Through masking and rearrangement, the model can utilize bidirectional emotional context (both preceding and succeeding) when generating NVs, which is critical for NV naturalness and emotional expression.

### Loss & Training
The AdamW optimizer is used with a learning rate of $1\times10^{-5}$ and a batch size of 100 (via gradient accumulation), training for 50,000 steps. Training took approximately 5 days on 4 NVIDIA RTX A6000 GPUs. Training data is from the EARS dataset (~100 hours of clean speech + 4 hours of NV, 107 speakers).

## Key Experimental Results

### Main Results (Seen Speakers)

| Method | NV-Acc↑ | NV-Sim↑ | NV-EECS↑ | NV-SECS↑ | WER↓ | V-EECS↑ |
|------|---------|---------|----------|----------|------|---------|
| VoiceCraft (Baseline) | 10.49 | 0.5898 | 0.6149 | 0.8950 | 9.05 | 0.6212 |
| Affectron (Full) | 37.75 | 0.6118 | 0.5748 | 0.8906 | 6.59 | 0.6216 |

### Ablation Study

| Configuration | NV-Acc↑ | NV-EECS↑ | Description |
|------|---------|----------|------|
| w/ DA only | 58.78 | 0.5455 | Data augmentation only; high Acc but emotional misalignment |
| w/ DA + EDNM | 35.83 | 0.5648 | EECS improves after adding emotional matching |
| w/ DA + EDNM + EAR | 32.93 | 0.5707 | EECS further improves after adding routing |
| Full (+ NSM) | 37.75 | 0.5748 | Full model, optimal NV quality |

### NV Type and Location Prediction vs. LLMs

| Method | Type JSD↓ | Type Acc@1↑ | Location JSD↓ |
|------|-----------|-------------|--------------|
| GPT-oss-20B | 0.1130 | 16.98 | 0.1278 |
| Affectron-330M | **0.0051** | **75.77** | **0.0523** |

### Key Findings
- The alignment of Affectron's NV type distribution far exceeds all LLM baselines (JSD of only 0.0051 vs. 0.1130 for the best LLM).
- Removing EDNM actually increases NV-Acc (random matching increases diversity), but EECS significantly drops, confirming the importance of emotional alignment.
- NSM leverages bidirectional emotional context and is better suited for NV generation than standard causal masking.
- Trends remain consistent on unseen speakers, validating zero-shot generalization capabilities.

## Highlights & Insights
- **Training-time augmentation, zero inference cost**: The matching and routing modules are used only during training. During inference, the model generates directly from annotated text without additional overhead. This "train-time augmentation → inference-time simplification" paradigm is highly referenceable.
- **Modeling emotional dynamics via spherical coordinates**: Mapping multi-dimensional emotional attributes to a sphere and measuring changes with angular distance captures emotional directionality better than Euclidean distance, a method transferable to other affective computing tasks.
- **330M small model outperforming 7B-20B LLMs**: Domain-specific explicit emotional modeling in small models is significantly more effective than general text reasoning in large models for NV type prediction.

## Limitations & Future Work
- Validation was limited to the EARS dataset (~100 hours).
- Linguistic speech and NVs are recorded separately, precluding the modeling of overlap phenomena seen in real-world scenarios.
- NV types only cover 15 categories, omitting richer non-verbal expressions.
- No direct comparison with the latest large-scale NV-capable TTS systems like CosyVoice.

## Related Work & Insights
- **vs. VoiceCraft**: The original only supports voice cloning with very weak NV capabilities. Affectron grants NV generation through augmented training.
- **vs. Label-controlled TTS (ELaTE, EmoCtrl-TTS)**: These depend on NV detection models for data annotation, which suffer from error propagation. Affectron's emotional routing is calculated based on emotional attributes.
- **vs. CosyVoice**: Requires large-scale, high-quality annotated corpora. Affectron works on small-scale open-source decoupled corpora.

## Rating
- Novelty: ⭐⭐⭐⭐ Emotion-driven NV matching and routing are novel augmentation strategies.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies and convincing LLM comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from background to methodology and experiments.
- Value: ⭐⭐⭐ The niche domain is specific, but the augmentation strategies are generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UniVocal: Unified Speech-Singing Code-mixed Synthesis](univocal_unified_speech-singing_code-switching_synthesis.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](../../ICLR2026/audio_speech/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](../../NeurIPS2025/audio_speech/adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[ACL 2026\] LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight](llm-mc-affect_llm-based_monte_carlo_modeling_of_affective_trajectories_and_laten.md)

</div>

<!-- RELATED:END -->
