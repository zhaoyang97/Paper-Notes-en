---
title: >-
  [Paper Note] Beyond Transcripts: A Renewed Perspective on Audio Chaptering
description: >-
  [ACL 2026][Audio & Speech][audio chaptering] This paper systematically reconstructs the long-form audio chaptering task by advancing evaluation from a transcript-dependent text space to a transcript-invariant temporal space. It demonstrates that AudioSeg, which directly utilizes audio representations, significantly outperforms text-based segmentation and existing
tags:
  - ACL 2026
  - Audio & Speech
  - audio chaptering
  - AudioSeg
date: 2026-05-08
content_hash: 10dcd5d25ee8eab9
---
# Beyond Transcripts: A Renewed Perspective on Audio Chaptering

**Conference**: ACL2026  
**arXiv**: [2602.08979](https://arxiv.org/abs/2602.08979)  
**Code**: Yes, the paper declares the release of the chunkseg evaluation package, AudioSeg models, and YTSeg supplementary annotations; specific links were not retained in the cache  
**Area**: Audio Understanding / Audio Segmentation  
**Keywords**: audio chaptering, AudioSeg, timeline evaluation, acoustic features, multimodal large models

## TL;DR
This paper systematically reconstructs the long-form audio chaptering task by advancing evaluation from a transcript-dependent text space to a transcript-invariant temporal space. It demonstrates that AudioSeg, which directly utilizes audio representations, significantly outperforms text-based segmentation and existing MLLM solutions on the YTSeg dataset.

## Background & Motivation
**Background**: Long-form audio and video are increasingly common, such as podcasts, lectures, interviews, and YouTube videos. Users typically navigate non-linearly, jumping through or revisiting specific segments, making automatic chaptering an essential interface for navigation and information retrieval. Most existing audio chaptering research simplifies the problem to text segmentation on transcripts: perform ASR first, then predict chapter boundaries within the sentence sequence.

**Limitations of Prior Work**: This transcript-centric perspective leaves three issues. First, the role of the audio itself has not been seriously studied; cues such as pauses, speech rate, speaker changes, music, and sound effects can all signal chapter transitions. Second, ASR errors change the number and boundaries of sentences, making text-level metrics calculated on different transcripts non-comparable. Third, real chapter boundaries are continuous timestamps, and forcing them onto sentence boundaries causes unavoidable discretization loss.

**Key Challenge**: The object of chaptering is the audio timeline, but traditional models and metrics operate on text sentence sequences. As long as evaluation depends on a specific transcript, it remains difficult to fairly compare text models, audio models, and multimodal models, or to determine whether performance changes stem from model capability or ASR granularity.

**Goal**: The authors aim to establish a more solid methodological foundation: comparing pure text, text + acoustic features, pure audio, and MLLM paradigms; analyzing the impact of ASR quality, acoustic features, audio duration, and speaker structure on segmentation performance; and formalizing evaluation protocols for both text and temporal spaces to enable fair cross-modal comparisons.

**Key Insight**: Instead of merely proposing a new model, the paper first clarifies the evaluation space and سپس compares models under a unified timeline protocol. This is crucial because many "improvements" in audio chaptering may simply be artifacts of transcript granularity or boundary projection methods.

**Core Idea**: Treat chapter boundaries as events on a timeline for evaluation, and use AudioSeg to predict boundaries directly from long-form audio representations, thereby bypassing transcript dependency and leveraging non-semantic acoustic structural cues.

## Method
The methodology consists of two layers: the first is an evaluation protocol that unifies existing text segmentation protocols with a new temporal segmentation protocol; the second is a model comparison covering MiniSeg text baselines, manual acoustic feature fusion, the AudioSeg audio-only model, and Qwen Omni multimodal large models.

### Overall Architecture
At the evaluation level, the paper defines R1, H1, H2, H3, T1, and T2. R1 evaluates on reference transcripts; H1 on ASR transcripts; H2/H3 map predictions from ASR back to reference transcripts via token alignment or temporal overlap; T1 segments the entire audio into fixed-length time chunks for discrete grid evaluation; T2 calculates boundary F1 directly on continuous timestamps using a tolerance window. The main experiments utilize T1 with a chunk size of 6 seconds.

At the model level, the text baseline follows MiniSeg: encoding sentences with MiniLM-style vectors and then using a RoFormer document encoder for boundary sequence labeling. The text+audio model concatenates text vectors with manual acoustic features followed by a linear projection. AudioSeg is entirely transcript-independent: it extracts frame-level representations using a frozen audio encoder, aggregates them into segment embeddings via 6-second windows, and uses a RoFormer to model long-range dependencies and predict whether each chunk is a boundary. MLLM experiments evaluate zero-shot, ICL, chunking, self-cascade, and LoRA versions of Qwen2.5-Omni and Qwen3-Omni.

### Key Designs

**1. Transcript-invariant Time-space Evaluation: Bringing all models to the same timeline to eliminate biases from ASR granularity.**

If evaluation remains tied to sentence sequences in a transcript, text, audio, and multimodal models cannot be compared fairly. Changes in ASR or sentence segmentation would cause fluctuations in scores, making it impossible to distinguish model capability from transcription granularity. The authors' solution is to evaluate chapter boundaries as events on a timeline. T1 discretizes the audio duration $D$ into $K=\lceil D/\Delta t\rceil$ time chunks (with $\Delta t=6$ seconds in main experiments), projecting both gold and predicted boundaries onto these chunks to calculate F1, Boundary Similarity, and $P_k$. T2 bypasses the discrete grid, comparing predicted and gold timestamps directly with a tolerance window of $\pm3s$ or $\pm6s$. Once the task returns to audio boundaries themselves, ASR sentence changes no longer pollute the metrics.

**2. Manual Acoustic Feature Fusion: Answering "are acoustic cues useful" before changing the architecture.**

Chapter transitions often involve pauses, intonation shifts, speaker changes, or sound effects—signals largely smoothed over in text. The authors isolate the "utility of audio" from "audio model strength" by augmenting the MiniSeg text baseline. For each sentence, they extract pause duration, speaking rate, pitch, loudness, and speaker-related features. The sentence vector $e_i$ and feature vector $f_i$ are concatenated and passed through a linear projection: $h_i=\mathrm{Linear}([e_i\|f_i])$, which is then fed into the RoFormer. If performance improves while keeping the backbone constant, it proves acoustic cues complement transcript semantics.

**3. AudioSeg Audio-only Architecture: Discarding transcripts to predict boundaries directly from long-form audio.**

If audio encoders already imply semantics, prosody, and non-speech cues, then "transcribe-then-segment" may be redundant and lose signals like music or long pauses. AudioSeg avoids text entirely: long audio is input into a frozen audio encoder in 30-second chunks to get frame-level representations; these are sliced into 6-second windows, each processed by a Local Segment Transformer using a learnable `[SEG]` token to pool into a segment embedding. A RoFormer then models long-range dependencies to output the probability of a boundary for each chunk. By modeling the audio time series directly, AudioSeg avoids ASR dependency and leverages structural cues invisible in text. It achieved 45.52 F1 with a Whisper Large encoder, significantly outperforming text models.

### Loss & Training
MiniSeg is trained using weighted binary cross-entropy on sentence boundary labels to mitigate class imbalance from boundary sparsity. AudioSeg also uses binary cross-entropy: continuous gold boundaries are discretized into the 6-second segment grid, and the model outputs the probability of a boundary for each segment. LoRA experiments for MLLMs were conducted specifically for Qwen2.5-Omni, with hyperparameters provided in the appendix; the main text notes that Qwen3-Omni was evaluated without heavy fine-tuning due to compute constraints.

The primary dataset is YTSeg, containing 19,299 English YouTube videos. The authors added annotations for duration categories, speaker types, and two types of ASR transcripts: Whisper Tiny and Whisper Large. The AMI meeting corpus was used for cross-domain generalization. Primary metrics are F1@6s, B@6s, and $P_k$@6s under the T1 protocol.

## Key Experimental Results

### Main Results
Text model experiments indicate only a weak correlation between ASR quality and segmentation quality; training jointly on Ref and ASR transcripts proved most stable.

| Model / Training Transcript | Ref F1 | ASR Tiny F1 | ASR Large F1 | Key Conclusion |
|-----------------------------|--------|-------------|--------------|----------------|
| LLaMA 3.1 8B constrained    | 25.92  | 24.71       | 26.26        | Weak zero-shot but stable across transcripts |
| WtP canine-s-12l           | 28.92  | 28.99       | 28.79        | Stable zero-shot but limited ceiling |
| MiniSeg Ref                 | 39.54  | 35.87       | 35.58        | Performance drops when migrating to ASR |
| MiniSeg ASRT                | 38.40  | 37.30       | 36.13        | More stable when trained on ASR |
| MiniSeg Ref+ASRT            | 40.01  | 37.76       | 36.38        | Best on Ref and robust on ASR |

Audio modeling results showed AudioSeg with a Whisper Large encoder achieved the highest F1.

| Model / Configuration | F1@6s | B@6s | $P_k$@6s | Notes |
|-----------------------|-------|------|-----------|-------|
| MiniSeg ASRT text only| 37.30 | 30.72| 31.84     | Text baseline |
| MiniSeg + pauses      | 40.17 | 33.59| 30.25     | Largest gain among single features |
| MiniSeg + all features| 40.30 | 33.48| 30.35     | Combined gain driven by pauses |
| AudioSeg + HuBERT L   | 35.58 | 27.95| 32.23     | Moderate audio representation |
| AudioSeg + AF3-Whisper| 39.02 | 30.75| 31.23     | Lower than Whisper Large |
| AudioSeg + Whisper L  | 45.52 | 36.17| 28.89     | Strongest audio-only result |
| Qwen3-Omni ICL        | 41.30 | 35.22| 33.00     | Limited to <30 min videos |
| Qwen3-Omni + FA       | 43.84 | 37.83| 34.83     | Good at topics, but raw timestamps are inaccurate |

### Ablation Study
Ablation of manual acoustic features shows that pauses are significantly more important than other features.

| MiniSeg ASRT Config | F1 | B | $P_k$ | Description |
|---------------------|----|---|-------|-------------|
| Random baseline     | 8.57 | 7.90 | 48.43 | Random boundaries |
| Audio features only | 19.39| 14.56| 37.85 | Some signal without semantics |
| Text only           | 37.30| 30.72| 31.84 | Semantics remain critical |
| Text + speaking rate| 37.32| 30.85| 31.75 | Almost no improvement |
| Text + loudness     | 37.82| 31.02| 31.50 | Minor improvement |
| Text + speakers     | 37.97| 31.11| 31.48 | Useful in multi-speaker scenarios |
| Text + pauses       | 40.17| 33.59| 30.25 | Largest gain, +2.87 F1 |
| Text + all features | 40.30| 33.48| 30.35 | Best overall, driven by pauses |

### Key Findings
- AudioSeg + Whisper Large (45.52 F1) significantly outperforms text models and most MLLM settings, proving transcript-free segmentation is not only feasible but superior on YTSeg.
- ASR WER does not fully explain segmentation performance: despite Whisper Large having a lower WER, MiniSeg on ASR Large was not necessarily better than on ASR Tiny.
- MLLMs can identify thematic boundaries but have weak temporal grounding. Qwen3-Omni's raw timestamp F1 was around 12, but reached 43.84 when outputs were aligned via forced alignment.
- Long-form audio remains challenging. Performance drops for all models beyond 20-30 minutes; for audio over 60 minutes, text+feature models slightly outperform AudioSeg.
- Multi-speaker content degrades all models, but AudioSeg is more robust; speaker features improved performance from 26.10 to 29.05 F1 in multi-speaker videos.

## Highlights & Insights
- The primary highlight is the clarification of evaluation protocols. While many papers calculate scores on transcripts by default, this paper demonstrates that the same temporal boundary mapping to different sentence sequences makes metrics non-comparable.
- AudioSeg's success indicates that audio encoders contain structural cues richer than text. Specifically, boundary signals like music and sound effects are often erased in transcripts.
- The strong contribution of pause features is intuitive but important: it suggests that for long-audio structuring, simple acoustic events remain powerful inductive biases that do not always require massive MLLMs.
- MLLM experiments offer realistic value. Qwen3-Omni's ICL performance is close to text+audio models, but context length, instruction following, and timestamp grounding remain bottlenecks.

## Limitations & Future Work
- Experiments heavily rely on YTSeg. While supplemented by the AMI corpus, conclusions may still be influenced by English YouTube data distributions.
- The dataset is English-only; it remains to be seen if multilingual audio chaptering benefits similarly from AudioSeg or temporal evaluation.
- The paper did not fine-tune stronger multimodal base models like Qwen3-Omni due to compute limits, leaving the upper bound of MLLMs partially unexplored.
- YTSeg inherently contains visual modalities, but this study focuses on text and audio. Visual cues like scene cuts or slide changes could further improve performance.
- AudioSeg performance drops on very long videos, suggesting a need for better long-context modeling or hierarchical temporal structures.

## Related Work & Insights
- **vs MiniSeg**: MiniSeg is a strong text baseline but relies on transcripts. This paper keeps it as a baseline while showing audio-only models can exceed its performance.
- **vs transcript-based chaptering**: Traditional methods treat chaptering as text segmentation. This paper emphasizes that chapter boundaries are temporal events and should be evaluated using transcript-invariant protocols.
- **vs MLLM end-to-end chaptering**: MLLMs can combine transcription, segmentation, and titling in one prompt but struggle with context length and formatting. AudioSeg is narrower but more stable for boundary detection.
- **Insights for future systems**: Practical systems could adopt a hybrid approach: AudioSeg for candidate boundaries, and ASR/LLM for chapter titles and summaries, unified under a temporal evaluation protocol.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Excellent integration of evaluation restructuring and AudioSeg.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic analysis of text, audio, MLLM, duration, speakers, and protocol sensitivity.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with solid methodological contributions.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to long-form audio/video structuring and multimodal evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](../../NeurIPS2025/audio_speech/a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [\[ICML 2025\] One Wave To Explain Them All: A Unifying Perspective On Feature Attribution](../../ICML2025/audio_speech/one_wave_to_explain_them_all_a_unifying_perspective_on_feature_attribution.md)
- [\[ACL 2026\] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care](exploration_of_perceptual_speech_features_for_clinical_decision-support_in_menta.md)
- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)

</div>

<!-- RELATED:END -->
