---
title: >-
  [Paper Note] Beyond Transcripts: A Renewed Perspective on Audio Chaptering
description: >-
  [ACL2026][Audio & Speech][audio chaptering] This paper systematically reconstructs the long-audio chaptering task: advancing evaluation from transcript-dependent text space to transcript-invariant time space…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "audio chaptering"
  - "AudioSeg"
  - "timeline evaluation"
  - "acoustic features"
  - "multimodal large models"
date: 2026-05-08
content_hash: 498b5762d271881b
---

# Beyond Transcripts: A Renewed Perspective on Audio Chaptering

**Conference**: ACL2026  
**arXiv**: [2602.08979](https://arxiv.org/abs/2602.08979)  
**Code**: Yes, the paper announces the release of the chunkseg evaluation package, AudioSeg model, and YTSeg additional annotations; specific links were not retained in the cache.  
**Area**: Audio Understanding / Speech Segmentation  
**Keywords**: audio chaptering, AudioSeg, timeline evaluation, acoustic features, multimodal large models

## TL;DR
This paper systematically reconstructs the long-audio chaptering task: advancing evaluation from transcript-dependent text space to transcript-invariant time space, and demonstrating that AudioSeg, which directly utilizes audio representations, significantly outperforms text segmentation and existing MLLM solutions on YTSeg.

## Background & Motivation
**Background**: Long audio and video are increasingly common, such as podcasts, lectures, interviews, and YouTube videos. Users typically do not listen linearly but jump, browse, and re-watch specific segments; thus, automatic chapter marking serves as a crucial interface for navigation and information retrieval. Most existing audio chaptering research simplifies the problem to text segmentation on transcripts: first transcribing, then predicting chapter boundaries within the sentence sequence.

**Limitations of Prior Work**: This transcript-centric perspective leaves three issues. First, the role of the audio itself has not been seriously studied; cues such as pauses, speech rate, speaker changes, music, and sound effects may all signal chapter transitions. Second, ASR errors change sentence counts and boundaries, making text-based segmentation metrics calculated across different transcripts directly incomparable. Third, real chapter boundaries are inherently continuous timestamps; forcing them onto sentence boundaries incurs unavoidable discretization loss.

**Key Challenge**: The object of chaptering is the audio timeline, but traditional models and metrics operate on text sentence sequences. As long as evaluation depends on a specific transcript, it remains difficult to fairly compare text models, audio models, and multimodal models, or to determine whether score changes stem from model capability or variations in ASR granularity.

**Goal**: The authors aim to establish a more robust methodological foundation: comparing pure text, text + acoustic features, pure audio, and various MLLM paradigms; analyzing the impact of ASR quality, acoustic features, audio duration, and speaker structure on segmentation performance; and formalizing evaluation protocols for text and time spaces so that systems with different input modalities can be compared fairly.

**Key Insight**: Instead of merely proposing a new model, the paper first clarifies the evaluation space and then compares models under the same timeline protocol. This approach is critical because many "improvements" in audio chaptering might be illusions caused by transcript granularity or boundary projection methods.

**Core Idea**: Evaluate chapter boundaries as events on a timeline and use AudioSeg to predict boundaries directly from long-audio representations, thereby bypassing transcript dependency and leveraging non-semantic acoustic structural cues.

## Method
The methodology consists of two layers: the first is the evaluation protocol, which unifies existing text segmentation protocols with a new temporal segmentation protocol; the second is model comparison, covering MiniSeg text baselines, manual acoustic feature fusion, AudioSeg audio-only models, and Qwen Omni multimodal large models.

### Overall Architecture
At the evaluation level, the paper defines R1, H1, H2, H3, T1, and T2. R1 evaluates on the reference transcript; H1 evaluates on the ASR transcript; H2/H3 map predictions from ASR back to the reference transcript using token alignment and temporal overlap, respectively. T1 segments the entire audio into fixed-length time blocks for discrete time-grid evaluation; T2 calculates boundary F1 directly on continuous timestamps using a tolerance window. The main experiments adopt T1 with a 6-second chunk size.

At the model level, the text baseline follows MiniSeg: encoding sentences using MiniLM-like vectors, then performing boundary sequence labeling with a RoFormer document encoder. The text+audio feature model concatenates sentence text vectors with manual acoustic features before linear projection. AudioSeg is entirely transcript-independent: it first extracts frame-level representations using a frozen audio encoder, aggregates them into segment embeddings via 6-second windows, and finally models long-range dependencies with RoFormer to predict whether each time block is a chapter boundary. MLLM experiments evaluate Qwen2.5-Omni and Qwen3-Omni in zero-shot, ICL, chunking, self-cascading, and LoRA versions.

### Key Designs
1.  **Transcript-invariant Time-space Evaluation**:
    - **Function**: Allows text models, audio models, and multimodal models to be mapped to the same timeline for comparison.
    - **Mechanism**: T1 discretizes the audio duration $D$ into $K=\lceil D/\Delta t\rceil$ time blocks; gold and predicted boundaries are projected onto these blocks to calculate F1, Boundary Similarity, and $P_k$. T2 directly compares predicted and gold timestamps, calculating boundary F1 under a $\pm3s$ or $\pm6s$ tolerance window.
    - **Design Motivation**: ASR alters sentence segmentation, making text-space metrics transcript-dependent. Time-space evaluation restores the task object to audio boundaries, avoiding artificial score inflation or deflation due to ASR granularity differences.

2.  **Manual Acoustic Feature Fusion**:
    - **Function**: Examines whether acoustic cues can supplement transcript semantics.
    - **Mechanism**: The paper extracts pause duration, speech rate, pitch, loudness, and speaker-related features for each sentence. The sentence vector $e_i$ and feature vector $f_i$ are concatenated and passed through a linear layer to obtain $h_i=Linear([e_i||f_i])$, which is then input to the MiniSeg document encoder.
    - **Design Motivation**: Chapter transitions are often accompanied by pauses, intonation changes, speaker shifts, or sound effects—information typically lost in text. Feature fusion isolates the questions of "is audio useful" versus "is the end-to-end audio model strong enough."

3.  **AudioSeg Audio-only Architecture**:
    - **Function**: Predicts chapter boundaries directly without a transcript.
    - **Mechanism**: Long audio is input to a frozen audio encoder in 30-second chunks to obtain continuous frame-level representations; these are then cut into non-overlapping 6-second windows. Each window passes through a Local Segment Transformer and is pooled into a segment embedding using a learnable [SEG] token. Finally, a RoFormer document encoder outputs the boundary probability for each segment.
    - **Design Motivation**: If audio encoders implicitly contain semantic, prosodic, and non-speech cues, modeling the audio time series directly may be more robust than text segmentation after transcription, especially in capturing music, sound effects, and long pauses invisible to transcripts.

### Loss & Training
MiniSeg uses weighted binary cross-entropy to train sentence boundary labels, mitigating class imbalance from sparse chapter boundaries. AudioSeg also uses binary cross-entropy: continuous-time gold chapter boundaries are discretized into the 6-second segment grid, and the model outputs the probability of each segment being a boundary. LoRA experiments for MLLMs target Qwen2.5-Omni, with hyperparameters provided in the appendix; the main text emphasizes that Qwen3-Omni did not undergo strong fine-tuning due to compute constraints.

The dataset primarily uses YTSeg, containing 19,299 English YouTube videos with transcripts and chapters. The authors added annotations for duration categories, speaker categories, and two types of ASR transcripts: Whisper Tiny and Whisper Large. Cross-domain generalization uses the AMI meeting corpus. Primary metrics are F1@6s, B@6s, and $P_k$@6s under the T1 protocol.

## Key Experimental Results

### Main Results
Text model experiments indicate only a weak correlation between ASR quality and segmentation quality; training with both reference and ASR transcripts is more robust.

| Model / Training Transcript | Ref F1 | ASR Tiny F1 | ASR Large F1 | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA 3.1 8B constrained decoding | 25.92 | 24.71 | 26.26 | Zero-shot text segmentation is weak but stable across transcripts |
| WtP canine-s-12l | 28.92 | 28.99 | 28.79 | Zero-shot is stable but has a low ceiling |
| MiniSeg Ref | 39.54 | 35.87 | 35.58 | Migration from Ref transcript to ASR causes performance drops |
| MiniSeg ASRT | 38.40 | 37.30 | 36.13 | ASR training is more stable for ASR testing |
| MiniSeg Ref+ASRT | 40.01 | 37.76 | 36.38 | Best on Ref and robust on ASR |

Audio modeling results are more critical: AudioSeg achieves the highest F1 when using the Whisper Large encoder.

| Model / Configuration | F1@6s | B@6s | $P_k$@6s | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| MiniSeg ASRT text only | 37.30 | 30.72 | 31.84 | Text baseline |
| MiniSeg + pauses | 40.17 | 33.59 | 30.25 | Largest gain among single features |
| MiniSeg + all audio features | 40.30 | 33.48 | 30.35 | Multi-feature combo driven primarily by pauses |
| AudioSeg + HuBERT Large | 35.58 | 27.95 | 32.23 | Moderate audio representation |
| AudioSeg + AF3-Whisper | 39.02 | 30.75 | 31.23 | Lower than Whisper Large |
| AudioSeg + Whisper Large | 45.52 | 36.17 | 28.89 | Strongest audio-only result in the paper |
| Qwen3-Omni ICL | 41.30 | 35.22 | 33.00 | Limited to <30 min videos |
| Qwen3-Omni + transcription + FA timestamps | 43.84 | 37.83 | 34.83 | Can find topic boundaries, but predicted timestamps are inaccurate |

### Ablation Study
Detailed ablation of manual acoustic features shows that pauses are more important than other acoustic cues.

| MiniSeg ASRT Configuration | F1 | B | $P_k$ | Description |
| :--- | :--- | :--- | :--- | :--- |
| Random baseline | 8.57 | 7.90 | 48.43 | Random boundaries |
| Audio features only | 19.39 | 14.56 | 37.85 | Signal exists even without semantics |
| Text only | 37.30 | 30.72 | 31.84 | Semantics remain critical |
| Text + speaking rate | 37.32 | 30.85 | 31.75 | Almost no improvement |
| Text + loudness | 37.82 | 31.02 | 31.50 | Slight improvement |
| Text + speakers | 37.97 | 31.11 | 31.48 | More useful in multi-speaker scenarios |
| Text + pauses | 40.17 | 33.59 | 30.25 | Largest gain, +2.87 F1 |
| Text + all features | 40.30 | 33.48 | 30.35 | Best overall but mainly from pauses |

### Key Findings
- AudioSeg + Whisper Large's 45.52 F1 significantly exceeds text models and most MLLM settings, proving that transcript-free segmentation is not only feasible but stronger on YTSeg.
- ASR WER is not a sufficient explanation for segmentation performance: Whisper Large has lower WER, but MiniSeg on ASR Large is not necessarily better than on ASR Tiny.
- MLLMs can identify some topic boundaries but have weak temporal localization capabilities. Predicted timestamps for Qwen3-Omni yield an F1 of only ~12, whereas using forced alignment timestamps for the same output reaches 43.84.
- Long audio remains difficult. Performance drops for all models beyond 20-30 minutes; for videos over 60 minutes, text+feature models slightly outperform AudioSeg.
- Multi-speaker content degrades all models, but AudioSeg is more robust; speaker features improve F1 from 26.10 to 29.05 in multi-speaker videos.

## Highlights & Insights
- The greatest highlight is clarifying the evaluation protocol rather than a single model. Many papers default to scoring on transcripts; this paper points out that the same temporal boundary corresponds to different sentence sequences across ASR granularities, making metric comparability questionable.
- The success of AudioSeg suggests that audio encoders contain richer structural cues than text. Boundary signals such as music, sound effects, and pauses are often completely erased in transcripts.
- The strong contribution of pause features is intuitive but important: it reminds us that simple acoustic events remain very strong inductive biases for long-audio structuring, and one need not always rely on larger MLLMs.
- The MLLM experiments have high practical value. Qwen3-Omni's ICL performance is close to text+acoustic features, but context length, instruction following, and timestamp grounding are still bottlenecks.

## Limitations & Future Work
- Experiments rely heavily on YTSeg; although supplemented by the small AMI meeting corpus, conclusions may still be influenced by English YouTube data distribution.
- The dataset is English-only; it is not yet clear if multilingual audio chaptering benefits equally from AudioSeg or time-space evaluation.
- Stronger multimodal foundation models like Qwen3-Omni were not fine-tuned due to compute constraints, meaning the ceiling for MLLMs has not been fully explored.
- YTSeg inherently contains visual modalities, but this paper only studies text and audio. Video chapters are often influenced by scene cuts, slides, and on-screen text; adding visual cues might further improve results.
- AudioSeg performance drops significantly on very long videos; future work needs stronger long-context modeling, sparse boundary learning, or hierarchical temporal structures.

## Related Work & Insights
- **vs MiniSeg**: MiniSeg is a strong text segmentation baseline but lacks transcript independence. This paper retains it as a text baseline while showing audio-only models can exceed it.
- **vs transcript-based podcast/video chaptering**: Traditional methods usually treat chaptering as text segmentation. This paper emphasizes that chapter boundaries are essentially temporal events and should be evaluated via transcript-invariant protocols.
- **vs MLLM end-to-end chaptering**: MLLMs can combine transcription, segmentation, and title generation in one prompt but are prone to issues with context length and format following. AudioSeg is narrower but more stable for boundary detection.
- **Insights for future systems**: Practical systems could adopt a hybrid route: AudioSeg for candidate boundaries, and ASR/LLM for chapter titles and content summaries, unified by time-space protocol evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The evaluation protocol reconstruction and AudioSeg combine well beyond simple model stacking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic analysis of text, audio, MLLM, duration, speaker, cross-domain, and protocol sensitivity.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, solid methodological contribution, and rich appendix information.
- Value: ⭐⭐⭐⭐⭐ Direct reference value for long-audio/video structuring, podcast navigation, and multimodal evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](../../NeurIPS2025/audio_speech/a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)

</div>

<!-- RELATED:END -->
