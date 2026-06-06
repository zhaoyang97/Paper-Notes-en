---
title: >-
  [Paper Note] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Speaker Diarization and Recognition] SpeakerLM is the first multimodal large language model designed specifically for end-to-end Speaker Diarization and Recognition (SDR). Through an audio enc…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Speaker Diarization and Recognition"
  - "Multimodal Large Language Models"
  - "End-to-End SDR"
  - "Speaker Enrollment"
  - "Speech Understanding"
date: 2026-05-08
content_hash: 15fdd13822466a27
---

# SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models

**Conference**: AAAI 2026  
**arXiv**: [2508.06372](https://arxiv.org/abs/2508.06372)  
**Code**: [Project Page](https://sites.google.com/view/speakerlm/overview)  
**Area**: Multimodal VLM  
**Keywords**: Speaker Diarization and Recognition, Multimodal Large Language Models, End-to-End SDR, Speaker Enrollment, Speech Understanding

## TL;DR

SpeakerLM is the first multimodal large language model designed specifically for end-to-end Speaker Diarization and Recognition (SDR). Through an audio encoder–projector–LLM architecture and a flexible speaker enrollment mechanism, it significantly outperforms cascaded baseline systems on multiple public benchmarks (absolute cpCER reduction up to 13.82%) and demonstrates strong robustness on out-of-domain test sets.

## Background & Motivation

### State of the Field
The SDR task aims to predict "who spoke what and when" in an audio recording, serving as a core technology in multi-speaker scenarios such as meeting transcription and dialogue systems. SDR requires jointly completing Speaker Diarization (SD, answering "who spoke when") and Automatic Speech Recognition (ASR, answering "what was said").

### Limitations of Prior Work

**Error Propagation in Cascaded Systems**: Traditional SDR systems adopt an SD + ASR cascaded framework, where errors in the SD module (e.g., inaccurate speaker boundaries, incorrect label assignment) propagate directly to the ASR module, degrading transcription quality.

**Difficulty Handling Overlapping Speech**: Conventional SD systems are based on Voice Activity Detection (VAD), which assumes a single active speaker per time segment and thus cannot effectively handle the commonly occurring scenario of simultaneous speech.

**Lack of Joint Optimization**: SD and ASR modules are typically trained independently on different datasets and frameworks, failing to exploit the synergy between the two tasks.

**Limitations of LLM Post-Processing**: Using an LLM to correct cascaded system outputs provides some benefit but is constrained by the quality of the front-end system output; moreover, hallucination in the LLM can cause unintended modifications to the original transcribed content.

### Root Cause
The central challenge is how to leverage an LLM not merely as a post-processing tool, but as the core component of an end-to-end SDR system that enables unified modeling and joint optimization of SD and ASR.

### Core Idea
This paper constructs the first end-to-end multimodal LLM—SpeakerLM—which injects speaker embedding as an additional modality into the LLM's input space and designs a flexible speaker enrollment mechanism to accommodate diverse real-world scenarios.

## Method

### Overall Architecture
SpeakerLM adopts an **encoder–projector–LLM** architecture comprising five components: an audio encoder, an audio projector, a speaker embedding extractor, a speaker projector, and a text LLM. Multi-speaker audio is processed by the encoder and injected into the feature space of a pretrained text LLM via a projector; speaker enrollment information is incorporated through a separate embedding extraction and projection pathway.

### Key Designs

1. **Audio Encoder and Projector**:

    - **Audio Encoder**: Initialized from the pretrained SenseVoice-large encoder, which supports multilingual speech recognition and audio event detection.
    - **Audio Projector**: A randomly initialized two-layer Transformer combined with a CNN layer for dimensionality alignment.
    - **Design Motivation**: SenseVoice-large exhibits strong performance across diverse audio understanding tasks, providing a robust starting point for audio representation.

2. **Speaker Embedding Extractor and Projector**:

    - **Embedding Extractor**: The open-source ERes2NetV2 model, which achieves state-of-the-art performance on multiple speaker verification benchmarks.
    - **Projector**: A single linear layer for dimensionality alignment.
    - **Workflow**: Speech from enrolled speakers is segmented into 2–10 second clips → embeddings are extracted → multiple clip embeddings are averaged into a representative embedding → linearly projected into the LLM space.
    - **Design Motivation**: A frozen pretrained embedding model provides stable and discriminative speaker representations.

3. **Flexible Speaker Enrollment Mechanism (Three Modes)**:

    - **No-Regist**: No speaker prior information is provided; output uses anonymous IDs (e.g., spk 0, spk 1), corresponding to the conventional cascaded SD system setting.
    - **Match-Regist**: All speakers present in the audio are pre-enrolled ($N_{rg} = N_{gt}$); the model must associate each speaker with the correct name.
    - **Over-Regist**: More speakers are enrolled than actually appear ($N_{rg} = N_{gt} + N_{ov}$); the model must determine which enrolled speakers are absent from the current audio.
    - **Design Motivation**: The three modes cover the full spectrum from anonymous transcription to personalized speaker transcription; Over-Regist more closely reflects real-world conditions where only a small subset of a large user pool participates.

4. **Four-Stage Progressive Training Strategy**:

    - **Stage 1 (ASR Pretraining)**: SpeakerLM-ASR is trained on 600K hours of public ASR data, with LoRA fine-tuning applied to the LLM.
    - **Stage 2 (Simulated Data Alignment)**: The randomly initialized projector is trained on simulated SDR data, with the LLM and audio encoder frozen, enabling rapid audio–text alignment.
    - **Stage 3 (Real Data Encoder Fine-Tuning)**: The audio encoder and projector are jointly fine-tuned on real SDR data, with the LLM frozen.
    - **Stage 4 (Full-Module Joint Fine-Tuning)**: All modules are jointly fine-tuned, with the LLM updated via LoRA, enabling deep integration of linguistic and acoustic information.

### Loss & Training
- LLM backbone: Qwen2.5-7B-Instruct, leveraging its strong instruction-following and general language understanding capabilities.
- Optimizer: AdamW, learning rate schedule: $1\text{e-}5 \to 5\text{e-}5$ (warmup) → cosine decay.
- Dynamic batching with a maximum token limit of 6K.
- 4 × NVIDIA A800 GPUs; 1M training steps per stage.

## Key Experimental Results

### Main Results (No-Regist Condition)

| System | Parameters | AliMeeting cpCER↓ | AISHELL4 cpCER↓ | AISHELL5 cpCER↓ (OOD) |
|--------|------------|-------------------|-----------------|----------------------|
| 3D-Speaker+Para | 70M (4 models) | 24.94 | 26.01 | 64.12 |
| Pyannote+Para | 70M (4 models) | 24.45 | 28.22 | 68.37 |
| DiariZen-base+Para | 95M (4 models) | 23.97 | 27.27 | 66.89 |
| DiariZen-large+Para | 140M (4 models) | 23.20 | 25.78 | 61.81 |
| ChatGPT4.5 post-proc. (zero-shot) | - (5 models) | 38.64 | 39.21 | 79.05 |
| Qwen2.5-7B post-proc. (fine-tuned) | 7B (5 models) | 22.65 | 24.93 | 61.63 |
| **SpeakerLM (7639h)** | **7B (1 model)** | **16.05** | **18.37** | **47.81** |

### Ablation Study (Speaker Enrollment & Embedding Model)

| Configuration | AliMeeting CER | AliMeeting saCER | Note |
|---------------|---------------|------------------|------|
| Match-Regist + ERes2NetV2 | 13.98 | 15.57 | Best speaker association |
| Over-Regist + ERes2NetV2 | 13.96 | 15.71 | Minimal impact from redundant speakers |
| Match-Regist + CAM++ | 14.74 | 17.23 | Embedding model quality is significant |
| Over-Regist + CAM++ | 14.71 | 16.92 | CAM++ underperforms ERes2NetV2 |
| SA-Transformer (Match-Regist) | - | 41.55 | SpeakerLM improves by 25.98% |

### Key Findings

1. **Strong Data Scaling Capability**: Scaling training data from 212h to 7639h reduces cpCER on AliMeeting from 32.22 to 16.05, with $\Delta$cp decreasing from 13.59 to 2.08.
2. **Excellent Out-of-Domain Generalization**: In a noisy in-car environment (AISHELL5-Eval), SpeakerLM achieves a $\Delta$cp of only 0.57, far below all cascaded baselines.
3. **LLM Post-Processing Is Inferior to End-to-End Modeling**: Zero-shot post-processing with ChatGPT-4.5 degrades performance, as LLM hallucinations modify speaker utterance content.
4. **Four-Stage Training Yields Incremental Gains**: Each stage contributes to performance improvement; Stages 3 and 4 are critical for out-of-domain generalization.
5. **Embedding Model Quality Directly Affects Performance**: ERes2NetV2 yields a 1–2% saCER improvement over CAM++.
6. **Robustness to Over-Enrolled Speakers**: Increasing the number of redundant enrolled speakers in Over-Regist (from 1 to 50) does not significantly degrade performance.

## Highlights & Insights

1. **First End-to-End MLLM for SDR**: Breaks the conventional paradigm of independent SD and ASR modeling, enabling genuine joint optimization.
2. **Elegant Flexible Enrollment Design**: The three enrollment modes cover the complete spectrum from anonymous to exact and over-enrolled settings, offering strong practical utility.
3. **Single Model vs. Multi-Model Cascade**: SpeakerLM with a single 7B model surpasses cascaded systems requiring 4–5 independent models.
4. **Thorough Data Scaling Analysis**: The scaling curve from 212h to 7639h is documented in detail, providing practical guidance on data requirements for deployment.
5. **Systematic Experimental Design**: Covers in-domain/out-of-domain conditions, with/without enrollment, different embedding models, and varying data scales.

## Limitations & Future Work

1. **Mandarin Chinese Only**: Applicability to multilingual scenarios has not been validated.
2. **High Computational Requirements**: Training requires 4 × A800 GPUs for 1M steps × 4 stages, resulting in non-trivial training costs.
3. **Dependence on Pretrained Speaker Embeddings**: The embedding extractor is frozen; end-to-end joint training of the embedding extraction module has not been explored.
4. **Audio Length Constraint**: Training and evaluation are both limited to 40–50 second segments; the ability to handle long-form audio (e.g., complete meetings) has not been verified.
5. **Upper Bound of Over-Regist Unexplored**: Training uses at most $N_{ov} = 50$; real-world deployment may involve substantially larger speaker pools.

## Related Work & Insights

- **3D-Speaker / Pyannote / DiariZen**: Representative SOTA cascaded SD systems; DiariZen improves SD performance by incorporating WavLM pretrained features.
- **DiarizationLM**: A pioneering work using LLM post-processing for SDR output, revealing the zero-shot hallucination problem of LLMs.
- **SA-Transformer**: Representative end-to-end SA-ASR system, but requires precise pre-enrollment of speaker embeddings.
- **MinMo / Qwen2-Audio / Kimi-Audio**: Audio–text MLLMs primarily targeting single-speaker scenarios.
- **Insights**: Extending multimodal LLM capabilities from single-speaker to multi-speaker settings is an important and practically valuable direction; flexible conditional injection mechanisms (e.g., speaker embedding projection) are a key enabling technique for such extension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures](../../CVPR2026/multimodal_vlm/markushgrapher-2_end-to-end_multimodal_recognition_of_chemical_structures.md)
- [\[ACL 2026\] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](../../ACL2026/multimodal_vlm/e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
