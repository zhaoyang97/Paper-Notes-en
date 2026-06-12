---
title: >-
  [Paper Note] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox
description: >-
  [ICML 2026][Audio & Speech][Audio LLM] The authors construct VoxParadox, a 2000-question MCQ benchmark where "linguistic content" and "acoustic signals" are intentionally in conflict. This benchmark demonstrates that cur…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Audio LLM"
  - "Paralinguistics"
  - "Adversarial Benchmark"
  - "Inter-layer Mixing"
  - "DPO"
date: 2026-05-08
content_hash: 6f0f7986fbb230b5
---

# Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox

**Conference**: ICML 2026  
**arXiv**: [2605.27772](https://arxiv.org/abs/2605.27772)  
**Code**: https://voxparadox.github.io/ (Project Page)  
**Area**: Audio & Speech / Audio LLM / Paralinguistic Understanding  
**Keywords**: Audio LLM, Paralinguistics, Adversarial Benchmark, Inter-layer Mixing, DPO

## TL;DR
The authors construct VoxParadox, a 2000-question MCQ benchmark where "linguistic content" and "acoustic signals" are intentionally in conflict. This benchmark demonstrates that current Audio LLMs primarily "read" rather than "listen" for paralinguistic tasks. By employing PCLM, a lightweight module that adaptively mixes intermediate audio encoder features based on prompts, combined with DPO preference optimization, the performance of Audio Flamingo 3 on VoxParadox is improved from 17.40% to 65.20%.

## Background & Motivation

**Background**: Audio LLMs such as Qwen2-Audio, SALMONN, Audio Flamingo 3, and Kimi-Audio connect audio encoders (mostly from the Whisper family) to powerful LLMs, achieving competent instruction following and conversational speech understanding. However, their "paralinguistic" capabilities—extracting information like emotion, age, gender, pitch, speaking rate, tone, and speaker identity from "how things are spoken" and "who is speaking"—have not been rigorously evaluated.

**Limitations of Prior Work**: General audio benchmarks like MMAU, MMAU-Pro, MMSU, and MMAR either emphasize broad audio understanding or conflate linguistic and acoustic cues. Although MMSU includes a paralinguistic subset, it does not explicitly decouple "linguistic-implied answers" from "acoustic-grounded answers." Consequently, models can achieve high scores by simply guessing based on ASR literal meanings, without truly "listening."

**Key Challenge**: The training paradigm of Audio LLMs (ASR-centric + text alignment) naturally biases them toward literal semantics. Paralinguistic signals require the model to actively ignore semantic shortcuts and focus on acoustic textures. This represents a modality imbalance, yet tools to directly quantify this imbalance are lacking.

**Goal**: (1) Construct a benchmark capable of exposing the "listening vs. reading" gap; (2) determine whether paralinguistic information is lost in the deep layers of the encoder, the projection layer, or due to the LLM's own neglect; (3) provide a remediation method that works without retraining the entire Audio LLM.

**Key Insight**: Drawing inspiration from the use of contradictory captions to expose modality shortcuts in Vision-Language tasks (Shekhar 2017, GVQA), the authors construct samples where "what the audio says" systematically opposes "what the text says"—e.g., an elderly person saying "I am a child," or multiple people saying "only one person is speaking." If the model selects the text-implied incorrect option, it proves it is not listening.

**Core Idea**: Use the adversarial benchmark to localize bottlenecks to "feature loss" and "insufficient utilization," then employ Prompt-Conditioned Layer Mixing (PCLM) to recover features and DPO to improve utilization.

## Method

### Overall Architecture

The work proceeds in two stages. The first stage is **Diagnosis**: constructing VoxParadox, evaluating existing Audio LLMs, and performing layer-wise probing to locate bottlenecks. The second stage is **Treatment**: inserting the PCLM module into models like Audio Flamingo 3 to replace the standard "last-layer output" interface, followed by a round of DPO to align preferences by using "acoustically grounded answers" as chosen and "linguistically implied answers" as rejected. The approach is lightweight as it does not retrain the audio encoder or the main LLM weights.

### Key Designs

1.  **VoxParadox: Linguistic–Acoustic Contradiction Adversarial Benchmark**:

    - **Function**: Comprises 10 paralinguistic tasks with 200 items each, totaling 2000 verified MCQs. Each item sets the linguistically stated attribute $y_{\text{adv}}$ and the true acoustic attribute $y_{\text{true}}$ as opposites, with both appearing as options.
    - **Mechanism**: GPT-4o generates scripts that "explicitly assert $y_{\text{adv}}$ and deliberately exclude $y_{\text{true}}$." On the acoustic side, deterministic mechanisms ensure $y_{\text{true}}$: ElevenLabs for age/gender metadata, signal processing for low-level features, Microsoft Azure SSML pitch-contour for tone, and turn-based splicing for speaker count/identity. Whisper large-v3 ensures transcription fidelity (WER = 0), SpeechBrain Wav2Vec2 SER filters emotions, and 10% are manually reviewed. Evaluation uses two complementary metrics: $\mathrm{Acc}_{\mathrm{GT}} = \frac{1}{N}\sum_i \mathbb{I}[\hat{y}_i = y_{\text{true}}^{(i)}]$ and $\mathrm{ALA} = \frac{1}{N}\sum_i \mathbb{I}[\hat{y}_i = y_{\text{adv}}^{(i)}]$. Since $y_{\text{adv}} \neq y_{\text{true}}$, a high ALA indicates a linguistic bias.
    - **Design Motivation**: By making correct answers dependent on non-linguistic acoustic evidence, the benchmark isolates "whether the model is truly listening" from confounding correlations, which prior benchmarks like MMSU and CP-Bench fail to do.

2.  **PCLM (Prompt-Conditioned Layer Mixer): Adaptive Layer Selection**:

    - **Function**: Replaces the standard interface that only feeds the encoder's last layer to the LLM with an adaptive weighted mixture of multiple intermediate representations based on the user prompt.
    - **Mechanism**: Layer-wise probing revealed that paralinguistic cues are strongest in the middle layers of ASR-pretrained encoders and are suppressed in deeper layers (consistent with Pasad 2022). Standard architectures lose this information by using only the deepest layer. PCLM computes prompt-embedding-conditioned weights $\alpha_\ell(\text{prompt})$ for each encoder output $h^{(\ell)}$, producing $\tilde{h} = \sum_\ell \alpha_\ell(\text{prompt}) \cdot h^{(\ell)}$ as the audio tokens for the LLM. 
    - **Design Motivation**: Unlike prompt-aware mixtures across multiple encoders (PaM) or input-dependent aggregation (VARAN), PCLM emphasizes "selecting layers within a single encoder based on the current question." Prompt-conditioning allows task-specific routing (e.g., different layers for emotion vs. age).

3.  **DPO for Acoustic Preference Alignment: Closing the "Utilization Gap"**:

    - **Function**: Conducts a round of Direct Preference Optimization after PCLM fine-tuning to align model preferences toward "acoustically grounded correct answers" (chosen) over "linguistically implied incorrect answers" (rejected).
    - **Mechanism**: Probing identified a second bottleneck where the LLM systematically ignores acoustic cues even when present in the audio tokens (utilization gap). Standard SFT token-level loss struggles to penalize this shortcut. DPO directly optimizes on paired preferences $\log \pi(y_w | x) - \log \pi(y_l | x)$, rewarding the model for following acoustic evidence when it conflicts with text.
    - **Design Motivation**: Reverses the adversarial structure of the dataset into a training signal. The same contradiction samples used for diagnosis are used for preference correction, creating a closed loop.

### Loss & Training

Two stages: First, SFT on standard paralinguistic data to update the PCLM module. Second, DPO on paired "acoustically grounded vs. language-implied" data with minimal parameter updates.

## Key Experimental Results

### Main Results

VoxParadox covers 10 tasks including age, gender, emotion, pitch, volume, speed, tone, speaker count, ID, and signal comparison. Baselines include state-of-the-art Audio LLMs.

| Model | VoxParadox Avg | MMSU Paralinguistic Subset | Remarks |
| :--- | :--- | :--- | :--- |
| Audio Flamingo 3 (Original) | 17.40% | 37.74% | Strong baseline, yet near random |
| Representative Audio LLMs | Generally low | Moderate | ALA significantly higher than $\mathrm{Acc}_{\mathrm{GT}}$ |
| Audio Flamingo 3 + PCLM + DPO (Ours) | **65.20%** | **54.78%** | Absolute gain of +47.80 on VoxParadox |

### Ablation Study

| Configuration | VoxParadox Avg | Description |
| :--- | :--- | :--- |
| Base Audio Flamingo 3 | 17.40% | Last layer only + no preference alignment |
| + PCLM only | Moderate improvement | Addresses "feature loss," benefits sensitive tasks (emotion, tone) |
| + DPO only | Partial improvement | Addresses "utilization gap," limited by feature quality |
| Full: PCLM + DPO | 65.20% | Highest performance achieved by addressing both bottlenecks |

### Key Findings

- **Audio LLMs "read" instead of "listen"**: Modern Audio LLMs show low GT accuracy and high ALA on VoxParadox, confirming a systematic bias toward following transcriptions over acoustic evidence.
- **Two complementary bottlenecks**: Layer-wise probing reveals: (i) paralinguistic information degrades in deep layers and projection interfaces (an ASR pre-training side effect); (ii) LLMs often ignore these cues even when present (utilization gap).
- **Synergy between PCLM and DPO**: PCLM addresses information presence ("is it provided?"), while DPO addresses information utilization ("is it used?"). Both are required to reach 65.20%.

## Highlights & Insights

- **Dual Use of Adversarial Benchmarks**: VoxParadox serves as both an evaluation tool and a source for DPO preference data. Using the same resource for diagnosis and treatment creates a clean methodology.
- **Quantifiable Definition of "Reading vs. Listening"**: Quantifying modality shortcuts using $\mathrm{ALA}$ allows linguistic prior dependence to be measured, compared, and optimized. This is applicable to any multi-modal system dominated by a single modality.
- **Lightweight Intervention**: PCLM and DPO require minimal parameter updates, proving that paralinguistic weaknesses can be mitigated significantly without retraining new encoders from scratch.
- **Transferable Diagnostic Template**: The pipeline of layer-wise probing to distinguish "information loss" from "insufficient utilization" can be applied to any "encoder + LLM" multimodal architecture.

## Limitations & Future Work

- **TTS Authenticity**: Audio samples are TTS-synthesized. While Whisper verification ensures correctness, a gap remains compared to real human speech, noisy environments, and diverse accents.
- **Task Imbalance**: Continuous attributes like emotion and tone are harder to control than discrete ones. Emotion tasks still require additional SER referee filtering.
- **"Listening Correctness" vs. "Understanding"**: Success is defined as selecting $y_{\text{true}}$, but the ultimate value of paralinguistics lies in downstream dialogue decisions (empathy, intent). Evaluating dialogue-level understanding is the next step.
- **Prompt Distribution Overfitting**: Since layer weights depend on prompt embeddings, the robustness of mixture weights against prompt style drift (e.g., language changes) requires further study.

## Related Work & Insights

- **vs. LISTEN (Chen 2025)**: LISTEN uses decorrelated pairs for emotion recognition; VoxParadox expands this "contradiction by design" to 10 tasks and provides an integrated treatment.
- **vs. MMSU / MMAU / MMAR**: These are general purpose benchmarks; VoxParadox is a focused stress test isolating paralinguistics.
- **vs. PaM (Shan 2025)**: PaM mixes multiple encoders; PCLM mixes intermediate layers of a single encoder, which is more cost-effective and aligns with empirical findings on paralinguistic information layers.
- **vs. VARAN (Diatlova 2025)**: VARAN uses input-dependent aggregation for general tasks; PCLM adds prompt-conditioning to make the strategy task-sensitive.
- **vs. DPO in TTS**: While others use DPO for TTS expressiveness (generation), this work uses it for understanding (aligning model belief with acoustic evidence).

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically migrates "contradiction" tests to audio paralinguistics and reuses the benchmark for DPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple Audio LLMs, layer-wise probing, and manual verification.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative arc from diagnosis to treatment.
- Value: ⭐⭐⭐⭐⭐ Provides both a metric for modality bias and a lightweight, effective mitigation strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](../../AAAI2026/audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[ACL 2026\] Protecting Bystander Privacy via Selective Hearing in Audio LLMs](../../ACL2026/audio_speech/protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](../../ACL2026/audio_speech/omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[ACL 2026\] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](../../ACL2026/audio_speech/analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
