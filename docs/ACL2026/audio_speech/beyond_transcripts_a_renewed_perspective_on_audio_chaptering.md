---
title: >-
  [Paper Note] Beyond Transcripts: A Renewed Perspective on Audio Chaptering
description: >-
  [ACL 2026][Audio & Speech][audio chaptering] This paper systematically reconstructs the long-form audio chaptering task: advancing evaluation from transcript-dependent text space to transcript-invariant temporal space, and demonstrating that AudioSeg, utilizing direct audio representations, significantly outperforms text-based segmentation and existing MLLM solut
tags:
  - ACL 2026
  - Audio & Speech
  - audio chaptering
  - AudioSeg
date: 2026-05-08
content_hash: ce97894c7a5c09c7
---
# Beyond Transcripts: A Renewed Perspective on Audio Chaptering

**Conference**: ACL2026  
**arXiv**: [2602.08979](https://arxiv.org/abs/2602.08979)  
**Code**: Yes, the paper declares the release of the `chunkseg` evaluation package, AudioSeg model, and YTSeg supplementary annotations; specific links were not retained in the cache  
**Area**: Audio Understanding / Speech Segmentation  
**Keywords**: audio chaptering, AudioSeg, timeline evaluation, acoustic features, multimodal large language models

## TL;DR
This paper systematically reconstructs the long-form audio chaptering task: advancing evaluation from transcript-dependent text space to transcript-invariant temporal space, and demonstrating that AudioSeg, utilizing direct audio representations, significantly outperforms text-based segmentation and existing MLLM solutions on YTSeg.

## Background & Motivation
**Background**: Long-form audio and video are increasingly common, such as podcasts, lectures, interviews, and YouTube videos. Users typically do not listen linearly but instead jump, browse, and re-watch specific segments; thus, automatic chapter marking serves as a crucial interface for navigation and information retrieval. Most existing audio chaptering research simplifies the problem to text segmentation on transcripts: first transcribing, then predicting chapter boundaries within the sentence sequence.

**Limitations of Prior Work**: This transcript-centric perspective leaves three issues. First, the role of the audio itself has not been seriously investigated; cues such as pauses, speech rate, speaker changes, music, and sound effects may signal chapter transitions. Second, ASR errors change sentence counts and boundaries, making text segmentation metrics calculated on different transcripts not directly comparable. Third, ground-truth chapter boundaries are inherently continuous timestamps; forcing them onto sentence boundaries incurs inevitable discretization loss.

**Key Challenge**: The object of chaptering is the audio timeline, but traditional models and metrics operate on text sentence sequences. As long as evaluation depends on a specific transcript, it remains difficult to fairly compare text models, audio models, and multimodal models, and hard to determine whether score changes stem from model capability or changes in ASR granularity.

**Goal**: The authors aim to establish a more robust methodological foundation: comparing pure text, text + acoustic features, pure audio, and MLLM paradigms; analyzing the impact of ASR quality, acoustic features, audio duration, and speaker structure on segmentation performance; and formalizing evaluation protocols for both text and temporal spaces to enable fair comparison across systems with different input modalities.

**Key Insight**: Instead of merely proposing a new model, the paper first clarifies the evaluation space and then compares models under the same timeline protocol. This approach is critical because many "improvements" in audio chaptering might be illusions caused by transcript granularity or boundary projection methods.

**Core Idea**: Evaluate chapter boundaries as events on a timeline and use AudioSeg to predict boundaries directly from long-form audio representations, thereby bypassing transcript dependency and leveraging acoustic structural cues beyond semantics.

## Method
The methodology consists of two layers: an evaluation protocol that unifies existing text segmentation protocols with a new temporal segmentation protocol, and a model comparison covering MiniSeg text baselines, manual acoustic feature fusion, AudioSeg audio models, and Qwen Omni multimodal large language models.

### Overall Architecture
Regarding evaluation, the paper defines R1, H1, H2, H3, T1, and T2. R1 evaluates on the reference transcript; H1 evaluates on the ASR transcript; H2/H3 map predictions from ASR back to the reference transcript using token alignment or temporal overlap, respectively. T1 segments the entire audio into fixed-length time chunks for discrete temporal grid evaluation; T2 directly calculates boundary F1 on continuous timestamps using a tolerance window. The main experiments adopt T1 with a chunk size of 6 seconds.

Regarding models, the text baseline follows MiniSeg: encoding sentences with MiniLM-style embeddings, followed by a RoFormer document encoder for boundary sequence labeling. The text+audio feature model concatenates text vectors with manual acoustic features per sentence before linear projection. AudioSeg is entirely independent of transcripts, first extracting frame-level representations with a frozen audio encoder, aggregating them into segment embeddings via 6-second windows, and finally using RoFormer to model long-range dependencies and predict whether each time chunk is a chapter boundary. MLLM experiments evaluate zero-shot, ICL, chunking, self-cascade, and LoRA versions of Qwen2.5-Omni and Qwen3-Omni.

### Key Designs

**1. Transcript-invariant temporal space evaluation: Bringing all models onto the same timeline to eliminate biases from ASR granularity.**

As long as evaluation resides in the sentence sequence of a transcript, fair comparison between text, audio, and multimodal models is impossible—changing the ASR or sentence segmentation changes the scores, making it impossible to distinguish if gains come from model capability or transcription granularity. The authors' solution is to restore chapter boundaries as events on the timeline. T1 discretizes the audio duration $D$ into $K=\lceil D/\Delta t\rceil$ time chunks (main experiments use $\Delta t=6$ seconds), projecting both ground-truth and predicted boundaries onto these chunks to calculate F1, Boundary Similarity, and $P_k$. T2 is more direct, bypassing discrete grids to compare predicted timestamps with ground-truth timestamps using a tolerance window of $\pm3s$ or $\pm6s$. Once the task object returns to the audio boundary itself, ASR sentence segmentation no longer contaminates the metrics, allowing systems with different input modalities to be measured with the same yardstick.

**2. Manual acoustic feature fusion: Addressing the prerequisite of whether acoustic cues are useful without changing the backbone.**

Chapter transitions are often accompanied by pauses, intonation changes, speaker transitions, or sound effects. While these signals are mostly flattened during transcription, whether end-to-end audio models are strong enough is a separate issue. The authors decouple them: on the MiniSeg text baseline, they extract additional features per sentence, including pause duration, speaking rate, pitch, loudness, and speaker-related features. The sentence vector $e_i$ and feature vector $f_i$ are concatenated and passed through a linear projection $h_i=\mathrm{Linear}([e_i\|f_i])$ before being fed back into the RoFormer document encoder. By keeping the backbone constant and only adding acoustic features, any performance gain confirms that acoustic cues indeed supplement transcript semantics, cleanly separating the "utility of audio" from the "strength of audio models."

**3. AudioSeg audio-only architecture: Completely discarding transcripts to predict chapter boundaries directly from long-form audio representations.**

If the audio encoder already implicitly contains semantic, prosodic, and non-verbal cues, then "transcribe-then-segment" is a roundabout path that loses signals invisible to transcripts, such as music, sound effects, and long pauses. AudioSeg thus avoids text entirely: long audio is input into a frozen audio encoder in 30-second chunks to obtain continuous frame-level representations; these are sliced into non-overlapping 6-second windows, each processed by a Local Segment Transformer using a learnable `[SEG]` token to pool into a segment embedding; finally, a RoFormer document encoder models long-range dependencies, outputting the probability of a chapter boundary for each time chunk. This pipeline directly models audio time series, bypassing ASR dependency and utilizing structural cues like pauses and music—achieving 45.52 F1 with a Whisper Large encoder, significantly outperforming text models and most MLLM configurations.

### Loss & Training
MiniSeg is trained using weighted binary cross-entropy on sentence boundary labels to mitigate class imbalance from sparse chapter boundaries. AudioSeg also uses binary cross-entropy: continuous ground-truth chapter boundaries are discretized into the 6-second segment grid, and the model outputs the probability of a boundary for each segment. LoRA experiments for MLLM only targeted Qwen2.5-Omni, with hyperparameters provided in the appendix; the main text emphasizes that Qwen3-Omni was evaluated without heavy fine-tuning, primarily due to compute constraints.

The dataset is primarily YTSeg, containing 19,299 English YouTube videos with transcripts and chapters. The authors additionally annotated duration categories, speaker categories, and two types of ASR transcripts: Whisper Tiny and Whisper Large. Cross-domain generalization was tested using the AMI meeting corpus. Primary metrics are F1@6s, B@6s, and $P_k$@6s under the T1 protocol.

## Key Experimental Results

### Main Results
Text model experiments indicate only a weak correlation between ASR quality and segmentation quality; training jointly on reference and ASR transcripts is more stable.

| Model / Training Transcript | Ref F1 | ASR Tiny F1 | ASR Large F1 | Key Conclusion |
|------------------------|--------|-------------|--------------|----------|
| LLaMA 3.1 8B constrained decoding | 25.92 | 24.71 | 26.26 | Zero-shot text segmentation is weak but stable across transcripts |
| WtP canine-s-12l | 28.92 | 28.99 | 28.79 | Zero-shot is stable but has a low ceiling |
| MiniSeg Ref | 39.54 | 35.87 | 35.58 | Transferring from Ref transcript to ASR causes performance drops |
| MiniSeg ASRT | 38.40 | 37.30 | 36.13 | ASR training is more stable for ASR testing |
| MiniSeg Ref+ASRT | 40.01 | 37.76 | 36.38 | Best on Ref and robust on ASR |

Audio modeling results are more critical: AudioSeg achieves the highest F1 when using the Whisper Large encoder.

| Model / Configuration | F1@6s | B@6s | $P_k$@6s | Remarks |
|-------------|-------|------|-----------|------|
| MiniSeg ASRT text only | 37.30 | 30.72 | 31.84 | Text baseline |
| MiniSeg + pauses | 40.17 | 33.59 | 30.25 | Largest gain among single-feature types |
| MiniSeg + all audio features | 40.30 | 33.48 | 30.35 | Multi-feature combo is primarily driven by pauses |
| AudioSeg + HuBERT Large | 35.58 | 27.95 | 32.23 | Moderate audio representation |
| AudioSeg + AF3-Whisper | 39.02 | 30.75 | 31.23 | Lower than Whisper Large |
| AudioSeg + Whisper Large | 45.52 | 36.17 | 28.89 | Overall strongest audio-only results |
| Qwen3-Omni ICL | 41.30 | 35.22 | 33.00 | Limited to video <30 mins |
| Qwen3-Omni + transcription + FA timestamps | 43.84 | 37.83 | 34.83 | Finds topic boundaries, but predicted timestamps are inaccurate |

### Ablation Study
Ablation of specific manual acoustic features shows that pauses are significantly more important than other acoustic cues.

| MiniSeg ASRT Configuration | F1 | B | $P_k$ | Description |
|-------------------|----|---|-------|------|
| Random baseline | 8.57 | 7.90 | 48.43 | Random boundaries |
| Audio features only | 19.39 | 14.56 | 37.85 | Some signal exists even without semantics |
| Text only | 37.30 | 30.72 | 31.84 | Semantics remain critical |
| Text + speaking rate | 37.32 | 30.85 | 31.75 | Nearly no improvement |
| Text + loudness | 37.82 | 31.02 | 31.50 | Slight improvement |
| Text + speakers | 37.97 | 31.11 | 31.48 | More useful in multi-speaker scenarios |
| Text + pauses | 40.17 | 33.59 | 30.25 | Largest improvement, +2.87 F1 |
| Text + all features | 40.30 | 33.48 | 30.35 | Overall best but mainly from pauses |

### Key Findings
- AudioSeg + Whisper Large (45.52 F1) significantly outperforms text models and most MLLM settings, proving that transcript-free segmentation is not only feasible but stronger on YTSeg.
- ASR WER is not a sufficient explanation for segmentation performance: although Whisper Large has lower WER, MiniSeg on ASR Large is not necessarily better than on ASR Tiny.
- MLLMs can identify some topic boundaries but have weak temporal localization. Qwen3-Omni's F1 is only about 12 when predicting timestamps, but reaches 43.84 after applying forced alignment to its output.
- Long-form audio remains difficult. Performance drops for all models beyond 20-30 minutes; for videos over 60 minutes, the text+feature model slightly outperforms AudioSeg.
- Multi-speaker content degrades performance for all models, but AudioSeg is more robust; speaker features improve F1 from 26.10 to 29.05 on multi-speaker videos.

## Highlights & Insights
- The greatest highlight is not any single model but the rigorous clarification of the evaluation protocol. Many audio chaptering papers default to calculating scores on transcripts; this paper points out that the same temporal boundary corresponds to different sentence sequences under different ASR granularities, making metric comparability problematic.
- The success of AudioSeg suggests that audio encoders contain richer structural cues than text. Boundary signals such as music, sound effects, and pauses are often completely stripped in transcripts.
- The strong contribution of pause features is intuitive yet important: it suggests that for long-form audio structuring tasks, simple acoustic events remain very strong inductive biases and that one does not always need to rely on larger MLLMs.
- The MLLM experiments have high practical value. Qwen3-Omni's ICL performance is already close to text+acoustic features, but context length, instruction following, and timestamp grounding remain bottlenecks.

## Limitations & Future Work
- Experiments primarily rely on YTSeg; although supplemented by the small-scale AMI meeting corpus, conclusions may still be influenced by the English YouTube data distribution.
- The dataset is English-only, and it remains to be seen if multilingual audio chaptering benefits similarly from AudioSeg or temporal-space evaluation.
- The paper did not fine-tune stronger multimodal foundation models like Qwen3-Omni due to compute limits. Thus, the upper bound for MLLMs has not been fully explored.
- YTSeg inherently includes visual modalities, but this study focuses only on text and audio. Video chapters are often influenced by scene cuts, slide changes, and on-screen text; incorporating visual cues could further improve performance.
- AudioSeg performance degrades significantly on very long videos; future work requires stronger long-context modeling, sparse boundary learning, or hierarchical temporal structures.

## Related Work & Insights
- **vs MiniSeg**: MiniSeg is a strong text segmentation baseline but relies on transcripts. This paper retains it as a baseline while showing that an audio-only model can exceed it.
- **vs transcript-based podcast/video chaptering**: Traditional methods usually treat chaptering as text segmentation. This paper emphasizes that chapter boundaries are temporal events and should be evaluated using transcript-invariant protocols.
- **vs MLLM end-to-end chaptering**: MLLMs can combine transcription, segmentation, and title generation in one prompt but are prone to context length and format-following issues. AudioSeg is narrower but more stable for boundary detection.
- **Insights for future systems**: Practical systems could adopt a hybrid path: AudioSeg for candidate boundaries, while ASR/LLM handles chapter titles and content summaries, unified by temporal-space evaluation protocols.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The evaluation protocol reconstruction and AudioSeg are well-integrated, not just a stack of models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic analysis of text, audio, MLLM, duration, speaker, cross-domain, and protocol sensitivity.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, solid methodological contributions, and rich appendix information.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for long-form audio/video structuring, podcast navigation, and multimodal evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[ICLR 2026\] Beyond Instance-Level Alignment: Dual-Level Optimal Transport for Audio-Text Retrieval](../../ICLR2026/audio_speech/beyond_instance-level_alignment_dual-level_optimal_transport_for_audio-text_retr.md)
- [\[ICML 2025\] One Wave To Explain Them All: A Unifying Perspective On Feature Attribution](../../ICML2025/audio_speech/one_wave_to_explain_them_all_a_unifying_perspective_on_feature_attribution.md)
- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](../../NeurIPS2025/audio_speech/a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [\[ICML 2026\] Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models](../../ICML2026/audio_speech/beyond_classification_a_cough_regression_benchmark_for_respiratory_acoustic_foun.md)

</div>

<!-- RELATED:END -->
