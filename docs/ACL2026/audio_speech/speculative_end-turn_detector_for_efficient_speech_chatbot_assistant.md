---
title: >-
  [Paper Note] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant
description: >-
  [ACL2026][Audio & Speech][End-point detection] The paper constructs OpenETD, the first publicly available end-turn detection dataset, and proposes SpeculativeETD. This framework utilizes an edge-side GRU to continuously…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "End-point detection"
  - "speech chat"
  - "OpenETD"
  - "collaborative inference"
  - "low latency"
date: 2026-05-08
content_hash: b5d43a4b11b6afaa
---

# Speculative End-Turn Detector for Efficient Speech Chatbot Assistant

**Conference**: ACL2026  
**arXiv**: [2503.23439](https://arxiv.org/abs/2503.23439)  
**Code**: The paper states that processing code and OpenETD data scripts will be released; the main text does not provide a complete repository URL.  
**Area**: Speech Dialogue / End-Turn Detection / Efficient Inference  
**Keywords**: End-point detection, speech chat, OpenETD, collaborative inference, low latency

## TL;DR
The paper constructs OpenETD, the first publicly available end-turn detection dataset, and proposes SpeculativeETD. This framework utilizes an edge-side GRU to continuously detect speaking/non-speaking units, triggering a server-side Wav2Vec2 only upon encountering a $200$ ms silence to distinguish between Gaps and Pauses. On real-world speech, this approach achieves real-time turn-taking performance close to large models with $38\times$ lower FLOPs and sub-millisecond edge-side latency.

## Background & Motivation
**Background**: LLM-based voice assistants increasingly emphasize natural dialogue. Systems must determine whether a user has finished speaking or is merely pausing to think. This task, known as end-turn detection (ETD), directly impacts whether the assistant will interrupt prematurely, misinterpret breaks, or suffer from response delays.

**Limitations of Prior Work**: Existing turn-taking data is often proprietary or costly to use (e.g., Fisher corpus), making ETD research difficult to replicate. Regarding models, transformer-based audio models like Wav2Vec2 offer high accuracy but are computationally heavy, making them unsuitable for continuous execution every $100$ ms on edge devices. Conversely, small GRUs can be deployed in real-time but exhibit significantly lower accuracy, particularly in distinguishing between true turn-ends (Gaps) and hesitant silences (Pauses).

**Key Challenge**: ETD requires high-frequency, low-latency, and low-power operation, yet the most difficult task—distinguishing Gaps from Pauses—demands strong speech understanding capabilities. Running large models continuously is too costly, while relying solely on small models compromises interaction quality.

**Goal**: The authors aim to solve both data and inference bottlenecks by constructing the public OpenETD dataset, covering both synthetic and real conversational audio, and designing an edge-cloud collaborative framework that triggers the large model only during necessary silence segments.

**Key Insight**: The three-state classification in ETD can be decomposed into two problems of varying difficulty. Distinguishing Speaking Units (SU) from non-SUs is relatively easy and manageable for small models. Distinguishing Gaps from Pauses is harder and only requires a large model's judgment after a silence segment appears.

**Core Idea**: The architecture of speculative decoding—where "a small model screens quickly and a large model confirms a few instances"—is migrated to speech endpoint detection. However, the small and large models are assigned different categorical granularities rather than predicting the same distribution.

## Method
SpeculativeETD is a two-stage real-time audio segmentation system. An edge-side model runs continuously on $100$ ms chunks, while a server-side model is invoked only when the edge model detects at least $200$ ms of non-SU. The output states include SU, Pause, and Gap; a Gap represents the end of a user turn (triggering an LLM response), while a Pause indicates the user may continue speaking.

### Overall Architecture
The system input is streaming speech. Every $100$ ms, the edge-side GRU reads log-mel chunks to determine if the user is in a speaking unit. If two consecutive chunks are classified as non-SU (reaching the $200$ ms threshold common in turn-taking literature), the audio segment accumulated since the start of the silence is sent to the server-side Wav2Vec2. The server performs binary classification: is this silence a Gap or a Pause? If it is a Gap, the assistant begins generating a response; if a Pause, it continues to wait.

### Key Designs
1.  **OpenETD Dataset Construction**:
    *   **Function**: Provides public, trainable, and evaluatable data for end-turn detection.
    *   **Mechanism**: The synthetic portion is based on MultiWOZ text dialogues, using TTS to generate three variants: V1 (no explicit pauses), V2 (injected pause silences), and V3 (filler words added before pauses). The real portion comes from YouTube and Buckeye; two-person dialogues are segmented via speaker diarization, and silences exceeding $200$ ms are labeled as Pause or Gap based on whether the speaker changes.
    *   **Design Motivation**: Synthetic data is controllable and can cover specific pause/gap patterns; real data provides noise, accents, emotions, and speech rate variations, preventing the model from learning only clean TTS signals.

2.  **Edge-side GRU Coarse Screening**:
    *   **Function**: Continuously identifies SU vs. non-SU with extremely low latency to reduce the frequency of large model invocations.
    *   **Mechanism**: Each $100$ ms audio chunk is sampled at $16$ kHz to extract $40$-dimensional log-mel spectrograms. A two-layer Conv2D frontend generates a $960$-dimensional chunk feature, processed autoregressively by a single-layer GRU with a hidden size of $64$. A linear head outputs SU/non-SU logits, with total parameters totaling approximately $202$ K.
    *   **Design Motivation**: Continuous real-time detection is the most resource-intensive part and must be handled by a small edge model. By simplifying the task to SU/non-SU, the small model is relieved of fine-grained semantic judgments regarding Gaps vs. Pauses.

3.  **Server-side Wav2Vec2 Fine Judgment and Trigger Protocol**:
    *   **Function**: Executes difficult classification only when silence segments occur to determine if a turn has truly ended.
    *   **Mechanism**: When the edge-side GRU predicts non-SU for $200$ ms consecutively, the system sends the audio segment starting from the silence onset to the server-side Wav2Vec2. Wav2Vec2 then classifies it as either Gap or Pause. Since triggering only occurs during silences, Wav2Vec2 does not need to run for every frame.
    *   **Design Motivation**: This mirrors speculative decoding, but instead of the large model verifying the small model's identical output, the large model handles a fine-grained sub-task triggered conditionally. This concentrates computation on the moments where it is most needed.

### Loss & Training
The paper does not propose a specific loss function; training utilizes AdamW for $10$ epochs. The learning rate is searched within $[3\times10^{-6}, 3\times10^{-4}]$, and weight decay within $[0.01, 2.00]$; batch sizes are tuned according to model size. Training data consists of a mix of synthetic and real training splits; evaluation is conducted on synthetic and real held-out test splits. Precision, Recall, F1, and Accuracy are evaluated for binary tasks; macro F1 and IoU for the three classes are evaluated every $100$ ms for real-time segmentation.

## Key Experimental Results

### Main Results

| Model / Data | Synthetic F1 | Synthetic Acc. / IoU | Real F1 | Real Acc. / IoU | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| VAP Binary | 92.1 | Acc. 92.3 | 59.1 | Acc. 69.6 | Open-source turn-taking baseline |
| GRU Binary | 78.1 | Acc. 79.7 | 49.8 | Acc. 69.0 | Lightweight but low precision |
| Wav2Vec2 Binary | 99.2 | Acc. 99.3 | 75.2 | Acc. 81.2 | Highest precision but heavy |
| VAP Real-time 3-class | 90.6 | IoU 84.8 | 33.2 | IoU 25.9 | Weak generalization on real data |
| GRU Real-time 3-class | 58.0 | IoU 52.2 | 34.2 | IoU 31.7 | Edge-capable but inaccurate |
| Wav2Vec2 Real-time 3-class | 94.7 | IoU 90.2 | 58.4 | IoU 46.2 | Accurate but heavy computation |
| SpeculativeETD | 94.0 | IoU 88.9 | 45.6 | IoU 37.8 | Synthetic near Wav2Vec2; real significantly beats VAP/GRU |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| OpenETD synthetic | 122,481 samples, 148.26 h | V1/V2/V3 cover basic, pause, and filler word patterns |
| OpenETD real | 166 h | From YouTube and Buckeye, two-person dialogues |
| Mixed training | Real F1 45.6, Real IoU 37.8 | Synthetic + real achieves best performance |
| Real only | Real F1 43.1, Real IoU 36.3 | $2.5$ F1 drop compared to mixed |
| Synthetic only | Real F1 44.0, Real IoU 36.7 | $1.6$ F1 drop compared to mixed |
| SpeculativeETD FLOPs | 919.64 MFLOPs / 100 samples | $34,971.68$ MFLOPs for Wav2Vec2, ~38x lower |
| SpeculativeETD W2V calls | 26.7x fewer W2V calls | Large model triggers only during necessary silences |
| GRU Edge Latency | execution 0.26 ms | Wav2Vec2 execution 1500.32 ms |

### Key Findings
- OpenETD's synthetic data totals $148.26$ hours ($96,773$ training samples / $116.83$ h; $12,868$ test samples / $15.68$ h). Real data from natural dyadic conversations bridges the synthetic domain gap.
- Gap/Pause duration distributions are similar between synthetic and real data (gap duration KS=$0.083$, Cohen's d=$0.12$), indicating Erlang fitting simulates silence lengths well; however, the positional distribution of pauses/gaps differs, suggesting synthetic data is better for augmentation than as a full replacement for real dialogue.
- Human verification shows a total human-auto agreement of $85.4\%$ for automatic labels ($94.0\%$ for Pause, $76.1\%$ for Gap). Diarization quality averaged $4.17/5$, indicating that while Gap boundaries are harder to label, the overall data is usable.
- End-to-end audio RTT is approximately $106$-$116$ ms on 5G and $98$-$140$ ms on Wi-Fi, both below the $200$ ms turn-taking threshold. Increasing the payload from $3.1$ KB to $312.5$ KB only adds approximately $10$ ms on 5G.

## Highlights & Insights
- Decomposing ETD into coarse on-device and fine server-side stages is highly intuitive. it aligns with the uneven difficulty of the task and the computational constraints of mobile deployment.
- The value of OpenETD is as significant as the method itself. Previously, much ETD work was limited by private datasets; this paper provides a public benchmark mixing synthetic and real data.
- The "speculative" nature of SpeculativeETD is not a simple copy of LLM decoding but a redefinition of model labor: the small model handles trigger conditions, while the large model handles the difficult sub-problem. This structure is transferable to other streaming perception tasks.
- The experiments report accuracy, FLOPs, edge-side latency, and network RTT simultaneously, which is far more relevant to real-world voice assistant deployment than reporting F1 scores alone.

## Limitations & Future Work
- The data primarily covers English conversations; turn-taking patterns, pause lengths, and filler words may vary across different languages and cultures.
- Gap/Pause classification depends on the server-side Wav2Vec2. Although measured RTT is below $200$ ms, production systems face model queuing, network fluctuations, privacy concerns, and offline issues.
- Positional distributions of pauses/gaps in synthetic data still differ from real data, and TTS offers limited accents and voices, lacking the diversity to cover all real users.
- Real data labels rely on diarization and the $200$ ms rule; human-auto agreement for Gaps is only $76.1\%$, indicating boundary noise in the training targets.
- SpeculativeETD's F1 on real three-class tasks ($45.6$) is significantly lower than Wav2Vec2's ($58.4$). It is a compromise for efficiency; applications highly sensitive to interruptions may still require stronger server verifiers or contextual linguistic understanding.

## Related Work & Insights
- **vs. VAP**: VAP is a classic turn-taking model that performs well on synthetic data but has an F1 of only $33.2$ on real data. SpeculativeETD improves real-world segmentation through data and two-stage inference.
- **vs. Full Wav2Vec2**: Wav2Vec2 has the highest accuracy but consumes ~$34,971.68$ MFLOPs per $100$ samples with an edge execution time of ~$1500$ ms. SpeculativeETD reduces computation to $919.64$ MFLOPs via conditional triggering.
- **vs. Pure GRU Edge Models**: GRU latency is extremely low but accuracy is insufficient. SpeculativeETD maintains edge-side real-time performance while using the server to handle Gap/Pause difficulties.
- **Insight**: For streaming multimodal agents, tasks can be split into "cheap edge-side sentinels + expensive cloud-side discriminators." This can be applied to visual wake-words, anomaly detection, speech emotion shifts, or on-device privacy filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ The two-stage ETD design is simple and effective; primary innovation lies in task decomposition and the public dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers binary classification, real-time segmentation, FLOPs, edge latency, RTT, and data quality analysis comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, strong deployment motivation, and straightforward explanations of data and methods.
- Value: ⭐⭐⭐⭐ Highly practical for real-time voice assistants; OpenETD is particularly valuable for subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)
- [\[ACL 2026\] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)

</div>

<!-- RELATED:END -->
