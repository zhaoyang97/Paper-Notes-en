---
title: >-
  [Paper Note] UniVocal: Unified Speech-Singing Code-mixed Synthesis
description: >-
  [ACL 2026][Audio & Speech][Speech-Singing Code-mixing] UniVocal trains a model to automatically infer speech/singing switching points from pure text semantics through refined pitch tokens and two-stage curriculum learnin…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Speech-Singing Code-mixing"
  - "Chain-of-Thought Generation"
  - "Refined Pitch Representation"
date: 2026-05-08
content_hash: a6a58a557244b306
---

# UniVocal: Unified Speech-Singing Code-mixed Synthesis

**Conference**: ACL 2026  
**arXiv**: [2606.01677](https://arxiv.org/abs/2606.01677)  
**Code**: https://github.com/FunAudioLLM/FunResearch/tree/main/UniVocal  
**Area**: Speech-Singing Synthesis / Multimodal Audio Generation  
**Keywords**: Speech-Singing Code-mixing, Chain-of-Thought Generation, Refined Pitch Representation

## TL;DR
UniVocal trains a model to automatically infer speech/singing switching points from pure text semantics through refined pitch tokens and two-stage curriculum learning. Without explicit labels, it achieves SOTA performance on the newly constructed SCSBench benchmark.

## Background & Motivation

**Background**: Speech Synthesis (TTS), Singing Voice Synthesis (SVS), and music generation specialize in individual domains but struggle to work together. Traditional solutions either generate only a single mode or require manual control through explicit labels (e.g., `<sing>`/`<speech>`).

**Limitations of Prior Work**: Real humans naturally mix speech and singing in daily communication—such as humming a melody during a conversation or using songs to assist memory. Existing systems cannot capture this *automatic switching based on text semantics*. Although systems like Bark attempt mixed-mode generation, they rely on explicit labels and lack semantic awareness, leading to unstable transitions.

**Key Challenge**: TTS lacks melodic expressiveness, while SVS is constrained by musical rules and scores. The distribution of their implicit representation spaces differs significantly, causing learning failure when directly mixed. Furthermore, semantic tokenizers discard fine-grained pitch information, making it impossible to accurately model intonation and melody.

**Goal**: Define the "Speech-Singing Code-mixed (SCS) Synthesis" task to enable the model to: (1) infer when to speak and when to sing from plain text; (2) transition smoothly between the two modes; (3) maintain consistent speaker style.

**Key Insight**: Adopt curriculum learning to break through in stages—first aligning the implicit representation spaces of both modes, then learning semantic trigger mechanisms on synthetic data. Simultaneously, introduce a high-resolution pitch representation (refined cent tokens) as a structural constraint for "plan-then-generate."

**Core Idea**: Use 1200-division refined cent tokens to explicitly supplement the pitch information lost by semantic tokenizers. Combined with interleaved Chain-of-Thought (CoT) generation, the model is forced to pre-plan a musical/tonal framework before generating specific content. This design both enhances singing melody accuracy and inadvertently stimulates the model's text empathy capabilities.

## Method

### Overall Architecture

UniVocal uses CosyVoice 2 as the backbone to build a unified text-to-voice generation framework. System inputs are: (1) task-level natural language instructions (e.g., "Generate a podcast" or "Generate a song"); (2) text to be synthesized. The model outputs: (1) a sequence of refined cent tokens; (2) a sequence of semantic tokens; (3) the final waveform. The key is the interleaved CoT generation strategy: the model alternately predicts pitch and then content for each 25Hz frame, implementing a "planning → generation" two-step framework.

### Key Designs

1.  **Refined Cent Token**:
    -   **Function**: Replaces coarse semitones or raw F0 with a 1200-division log-frequency representation (reference frequency $440$Hz) to accurately capture intonation and melody.
    -   **Mechanism**: The mapping from cents to tokens is defined as $I(f_{cent})=\lceil f_{cent} \bmod 1200 \rceil$, where $f_{cent}=1200 \log_2(f_{Hz}/440)$. Silent regions are marked as token -1. This covers the entire human vocal range (spanning multiple octaves) while mapping it into a single octave through the modulo 1200 operation, forming a unified token space. The maximum quantization error is $\approx 1$ cent (0.08% frequency deviation), which is perceptually lossless.
    -   **Design Motivation**: Traditional semantic tokenizers discard fine-grained pitch, resulting in flat intonation. 1200 divisions are more refined than semitones (12) and more controllable than raw F0. This design bridges the gap between semantic tokens and acoustics.

2.  **Interleaved Chain-of-Thought (CoT)**:
    -   **Function**: Forces the model to generate a pitch token before a semantic token at each timestep, achieving "structural planning priority."
    -   **Mechanism**: The vocabulary of CosyVoice 2 is expanded to include 1201 cent tokens (1200 cent values + 1 silence token), with independently initialized embedding vectors. The joint probability is decomposed as:
        $$P(\mathbf{Y}|\mathbf{X})=\prod_t P(c_t|\mathbf{X},\mathbf{Y}_{<t}) \cdot P(s_t|\mathbf{X},\mathbf{Y}_{<t},c_t)$$
        where $c_t$ and $s_t$ represent the cent token and semantic token, respectively. This sequence is enforced during inference via logit masking.
    -   **Design Motivation**: The model must first "think through" the overall pitch trajectory (e.g., a tone rising from low to high) before generating specific words and details, preventing conflicts between semantic and musical goals.

3.  **Two-stage Curriculum Learning**:
    -   **Function**: Eliminates distribution differences between the two modes in steps to gradually acquire automatic switching capabilities.
    -   **Mechanism**: **Stage 1**: Continued pre-training on aligned data (4:1 singing:speech ratio) using task-level instructions to learn both modes independently. **Stage 2**: Supervised fine-tuning using synthetic code-mixed data, with a 1:1:1 mix of code-mixed/speech/singing data to prevent catastrophic forgetting.
    -   **Design Motivation**: Direct training on mixed data leads to convergence difficulties (the F1 of a single-stage variant drops to 0.496). Initial alignment is a prerequisite for subsequent semantic triggering.

### Data Synthesis Pipeline

To address the scarcity of code-mixed data, the authors introduce a three-step synthesis pipeline:
-   **Semantic Text Generation**: Use Gemini 2.5 Pro to generate scripts with "fuzzy boundaries" (monologues, podcasts, audiobooks), integrating implicit (prose for speech, lyrics for singing) and explicit (e.g., "reminds me of a song") switching triggers.
-   **Unified Audio Synthesis**: Use the Stage 1 model to synthesize all segments. Speech is conditioned on emotion reference audio (9 emotions from the Expresso dataset) to ensure emotional consistency; singing is only conditioned on speaker embeddings to maintain timbre stability.
-   **Quality Filtering**: Use Whisper-v3 to calculate WER. Samples with WER $\geq 20\%$ are discarded, and 10-20% are retained to increase diversity, resulting in 11,769 final samples (262 hours).

## Key Experimental Results

### Main Results

| Model | SCSBench-Implicit F1(O) | F1(S) | SCSBench-Explicit F1(O) | F1(S) | SCSBench-Mixed F1(O) | F1(S) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini + Bark | 0.414 | 0.142 | 0.533 | 0.250 | 0.465 | 0.199 |
| Gemini + Cosy2 + LeVo | 0.752 | 0.685 | 0.572 | 0.489 | 0.607 | 0.566 |
| **Ours (UniVocal)** | **0.626** | **0.595** | **0.714** | **0.635** | **0.871** | **0.810** |

In mixed scenarios (containing both implicit and explicit triggers), UniVocal achieves SOTA on both objective F1 and Whisper F1. It also maintains the lowest WER (5.83-10.90%) and the highest UTMOS (4.36).

### Ablation Study

| Model Variant | Text Empathy E-MOS | P-MOS | Singing Naturalness N-MOS | M-MOS | Switch Accuracy SCS F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ours (Full)** | 2.26 | 2.22 | 2.23 | 2.18 | 0.716 |
| w/o CoT | 2.03 | 1.84 | 2.20 | 1.86 | 0.810 |
| w/o CL | 2.24 | 2.23 | 2.29 | 2.17 | 0.496 |

Removing CoT improves switching stability (F1 $\rightarrow$ 0.810) but at the cost of a significant drop in expressiveness. Removing curriculum learning results in a collapse of switching ability (F1 $\rightarrow$ 0.496).

### Key Findings

-   **Critical Role of Explicit Triggers**: Samples containing trigger phrases like "let me sing" have significantly higher switching accuracy.
-   **Humming Anomaly**: Humming without semantic content but with unique text forms (e.g., "mm-mm") performs exceptionally well as a "strong implicit trigger."
-   **Superior Speaker Consistency**: Although global speaker similarity is slightly lower (0.65), intra-sentence consistency for UniVocal is significantly higher than cascade baselines when viewed across temporal segments.
-   **Real-time Performance**: Training took a total of 6 days (5 days for Stage 1, 1 day for Stage 2) on 4 A800 GPUs.

## Highlights & Insights

-   **"Planning → Generation" Causal Chain**: Interleaved CoT generation allows the model to explicitly model the pitch trajectory before filling in words, breaking the "black box" of traditional end-to-end generation. Correlation analysis ($\rho=0.679$) proves that cent tokens indeed act as structural planners.
-   **Data Synthesis Paradigm**: A closed-loop pipeline using LLMs to generate semantically coherent code-mixed scripts, followed by two-stage model synthesis and quality filtering, ensures data naturalness while avoiding manual annotation.
-   **Invisible Gain—Text Empathy**: The CoT mechanism, originally introduced to enhance musical modeling, unexpectedly stimulated the model's emotional expression capability (E-MOS +0.48 vs. baseline).
-   **Cascade vs. Unified Trade-off**: Cascade systems have higher global similarity but obvious intra-sentence timbre drift; UniVocal, though slightly lower on global metrics, shows stronger overall consistency.

## Limitations & Future Work

**Limitations of Prior Work noted by authors**:
1. Synthetic singing data is limited by source separation and ASR tools, resulting in robotic artifacts and lyric alignment errors.
2. Distribution gap between synthetic training data and complex real-world scenes; purely implicit switching is still not robust enough.
3. F1 evaluation often falls into binary results on single short samples, with limited sample-level correlation.

**Self-identified Limitations**:
- Model generalization to "real-world" scenarios is limited (Raw Real SCS F1=0.201), requiring explicit triggers to recover.
- The 4:1 singing:speech ratio in Stage 1 of curriculum learning was manually tuned; a systematic ratio search is lacking.

**Future Directions**: Collect high-quality real singing data; explore more granular semantic understanding as a supplement to implicit switching triggers; extend the CoT mechanism to other multi-constraint generation tasks.

## Related Work & Insights

-   **vs. Bark (Mixed-mode Generation)**: Bark relies on explicit labels and has unstable transitions. UniVocal achieves automatic switching via text semantic inference with much higher stability (F1 0.871 vs. 0.465).
-   **vs. UniSyn/UniAudio (Unified Audio Generation)**: While supporting multi-tasking, these can only generate a single mode at a time and do not support intra-sequence switching. UniVocal is optimized specifically for mixed generation.
-   **vs. Vevo2 (Unified Acoustic Modeling)**: Vevo2 uses chromagram tokens for acoustics (12 semitone resolution) and requires reference audio to determine the mode. UniVocal uses 1200-division cent tokens and infers from plain text, offering finer granularity and fewer input requirements.
-   **Insight**: Explicit pitch modeling is crucial for multimodal acoustic generation; the "plan-then-generate" CoT paradigm can be generalized to other generation tasks governed by multiple constraints.

## Rating

-   Novelty: ⭐⭐⭐⭐⭐ First to define and solve automatic speech-singing code-mixing, introducing an innovative combination of refined cent tokens and CoT generation.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Constructed the SCSBench benchmark covering implicit/explicit/mixed scenarios. Ablations are comprehensive, though sample-level correlation is limited, system-level verification is sufficient.
-   Writing Quality: ⭐⭐⭐⭐⭐ Logical, well-motivated, complete methodological details, and rich appendices.
-   Value: ⭐⭐⭐⭐ First implementation of automatic speech-singing transitions in an end-to-end framework, offering both academic novelty and application potential; all code and data are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](../../NeurIPS2025/audio_speech/adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations](affectron_emotional_speech_synthesis_with_affective_and_contextually_aligned_non.md)
- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)

</div>

<!-- RELATED:END -->
